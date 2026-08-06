"""Routes for narrator voice profiles and their conflicts."""
from flask import request, jsonify


def register(bp, deps, shared):
    """Register the narrator routes on bp."""
    state_manager = deps.state_manager
    _context_revision = shared.context_revision
    _revision_conflict = shared.revision_conflict

    @bp.route('/api/translation/<translation_id>/narrator-voice', methods=['GET', 'POST'])
    @bp.route('/<translation_id>/narrator-voice', methods=['GET', 'POST'])
    def narrator_voice_profiles_route(translation_id):
        """List the narrator timeline or create/update a manual profile."""

        db = state_manager.checkpoint_manager.db
        if request.method == 'GET':
            from src.utils.narrator_voice import (
                eligible_bootstrap_boundary,
                narrator_policy_payload,
            )

            timeline = db.get_narrator_voice_timeline(translation_id)
            chunks = db.get_chunks(translation_id)
            refinement_overlays = {
                int(item.get("base_chunk_index", -1)): item
                for item in db.get_active_refinement_results(translation_id)
                if item.get("base_chunk_index") is not None
            }
            effective_chunks = []
            for item in chunks:
                overlay = refinement_overlays.get(int(item.get("chunk_index", -1)))
                effective_chunks.append({
                    **item,
                    "translated_text": (
                        overlay.get("refined_text") if overlay else item.get("translated_text")
                    ),
                    "chunk_data": {
                        **(item.get("chunk_data") or {}),
                        **((overlay or {}).get("chunk_data") or {}),
                    },
                })
            chunks = effective_chunks
            stale_chunks = [
                int(item.get("chunk_index", 0))
                for item in chunks
                if (item.get("chunk_data") or {}).get("narrator_voice_stale")
            ]
            conformance_units = [{
                "chunk_index": int(item.get("chunk_index", 0)),
                "conformance": (item.get("chunk_data") or {}).get(
                    "narrator_conformance"
                ) or {"status": "not_checked"},
                "backfill": (item.get("chunk_data") or {}).get(
                    "narrator_backfill"
                ) or {"status": "idle"},
            } for item in chunks if (
                (item.get("chunk_data") or {}).get("narrator_conformance")
                or (item.get("chunk_data") or {}).get("narrator_backfill")
            )]
            completed_count = sum(
                1 for item in chunks
                if item.get("status") in {"completed", "partial"}
                and str(item.get("translated_text") or "").strip()
            )
            job = db.get_job(translation_id) or {}
            config = job.get("config") or {}
            target_language = str(config.get("target_language") or "")
            boundary = eligible_bootstrap_boundary(completed_count)
            attempts = db.get_narrator_bootstrap_attempts(translation_id)
            attempts.extend(db.get_narrator_bootstrap_attempts(
                translation_id, attempt_kind="preflight",
            ))
            return jsonify({
                "translation_id": translation_id,
                "context_revision": _context_revision(translation_id),
                "profiles": db.get_narrator_voice_profiles(
                    translation_id, include_inactive=True,
                ),
                "observations": timeline["observations"],
                "transitions": timeline["transitions"],
                "conflicts": db.get_narrator_voice_conflicts(translation_id),
                "stale_chunks": stale_chunks,
                "policy": narrator_policy_payload(target_language),
                "bootstrap": {
                    "completed_units": completed_count,
                    "current_boundary": boundary,
                    "next_boundary": (
                        2 if boundary is None else boundary + 4
                    ),
                    "attempts": sorted(
                        attempts, key=lambda item: str(item.get("created_at") or "")
                    ),
                },
                "backfill": {
                    "status": (
                        "blocked" if any(
                            unit["backfill"].get("status") == "blocked"
                            for unit in conformance_units
                        ) else "pending" if stale_chunks else "idle"
                    ),
                    "pending_units": stale_chunks,
                    "pending_count": len(stale_chunks),
                    "units": conformance_units,
                },
            }), 200

        data = request.get_json() or {}
        conflict, _ = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        narrator_key = str(data.get("narrator_key") or "default").strip()
        if not narrator_key:
            return jsonify({"error": "narrator_key is required"}), 400
        payload = dict(data)
        payload.update({
            "narrator_key": narrator_key,
            "provenance": "user_manual",
            "confidence": 1.0,
            "status": "active",
            "is_locked": bool(data.get("is_locked", True)),
        })
        profile = db.upsert_narrator_voice_profile(
            translation_id,
            payload,
            expected_revision=data.get("expected_profile_revision"),
        )
        if not profile:
            return jsonify({
                "error": "Narrator profile changed; reload before saving."
            }), 409
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        from src.core.editor_retry import audit_completed_narrator_conformance
        audit = audit_completed_narrator_conformance(
            translation_id=translation_id,
            checkpoint_manager=state_manager.checkpoint_manager,
        )
        stale_chunks = audit.get("queued") or []
        return jsonify({
            "message": "Narrator voice profile saved successfully",
            "translation_id": translation_id,
            "context_revision": revision,
            "profile": profile,
            "stale_chunks": stale_chunks,
        }), 200
    @bp.route('/api/translation/<translation_id>/narrator-voice/<int:profile_id>', methods=['DELETE'])
    @bp.route('/<translation_id>/narrator-voice/<int:profile_id>', methods=['DELETE'])
    def delete_narrator_voice_profile_route(translation_id, profile_id):
        data = request.get_json(silent=True) or {}
        conflict, _ = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        db = state_manager.checkpoint_manager.db
        existing = db.get_narrator_voice_profile(translation_id, profile_id)
        if not db.delete_narrator_voice_profile(
            translation_id, profile_id,
            expected_revision=data.get("expected_profile_revision"),
        ):
            return jsonify({
                "error": "Narrator profile was not found, changed, or is locked."
            }), 409
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        stale_chunks = db.mark_narrator_voice_chunks_stale(
            translation_id, int((existing or {}).get("start_chunk_index", 0) or 0),
        )
        return jsonify({
            "message": "Narrator voice profile deleted successfully",
            "context_revision": revision,
            "stale_chunks": stale_chunks,
        }), 200
    @bp.route('/api/translation/<translation_id>/narrator-voice/<int:profile_id>/lock', methods=['POST'])
    @bp.route('/<translation_id>/narrator-voice/<int:profile_id>/lock', methods=['POST'])
    def lock_narrator_voice_profile_route(translation_id, profile_id):
        data = request.get_json() or {}
        conflict, _ = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        db = state_manager.checkpoint_manager.db
        profile = db.set_narrator_voice_profile_lock(
            translation_id, profile_id, bool(data.get("is_locked", True)),
            expected_revision=data.get("expected_profile_revision"),
        )
        if not profile:
            return jsonify({"error": "Narrator profile changed or was not found."}), 409
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        stale_chunks = db.mark_narrator_voice_chunks_stale(
            translation_id, int(profile.get("start_chunk_index", 0) or 0),
        )
        return jsonify({
            "message": "Narrator voice profile lock updated successfully",
            "context_revision": revision,
            "profile": profile,
            "stale_chunks": stale_chunks,
        }), 200
    @bp.route('/api/translation/<translation_id>/narrator-conflicts', methods=['GET'])
    @bp.route('/<translation_id>/narrator-conflicts', methods=['GET'])
    def narrator_voice_conflicts_route(translation_id):
        status = str(request.args.get("status") or "open")
        return jsonify({
            "translation_id": translation_id,
            "conflicts": state_manager.checkpoint_manager.db.get_narrator_voice_conflicts(
                translation_id, status=status,
            ),
        }), 200
    @bp.route('/api/translation/<translation_id>/narrator-conflicts/<int:conflict_id>/resolve', methods=['POST'])
    @bp.route('/<translation_id>/narrator-conflicts/<int:conflict_id>/resolve', methods=['POST'])
    def resolve_narrator_voice_conflict_route(translation_id, conflict_id):
        data = request.get_json() or {}
        resolution = str(data.get("resolution") or "")
        db = state_manager.checkpoint_manager.db
        conflicts = db.get_narrator_voice_conflicts(translation_id)
        existing = next((item for item in conflicts if item["id"] == conflict_id), None)
        if not existing:
            return jsonify({"error": "Narrator conflict not found"}), 404
        if resolution == "accepted_transition":
            candidate = dict(existing.get("candidate") or {})
            candidate.update({"status": "active", "provenance": "user_manual"})
            if not db.upsert_narrator_voice_profile(translation_id, candidate):
                return jsonify({"error": "Narrator transition could not be accepted"}), 409
        if not db.resolve_narrator_voice_conflict(
            translation_id, conflict_id, resolution,
        ):
            return jsonify({"error": "Invalid narrator conflict resolution"}), 400
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        stale_chunks = db.mark_narrator_voice_chunks_stale(
            translation_id, int(existing.get("chunk_index", 0) or 0),
        )
        return jsonify({
            "message": "Narrator conflict resolved successfully",
            "context_revision": revision,
            "resolution": resolution,
            "stale_chunks": stale_chunks,
        }), 200
