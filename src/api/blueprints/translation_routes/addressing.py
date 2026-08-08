"""Routes for the directed-addressing rule table."""
from flask import jsonify, request


def register(bp, deps, shared):
    """Register the addressing routes on bp."""
    state_manager = deps.state_manager
    _context_revision = shared.context_revision
    _revision_conflict = shared.revision_conflict
    _export_structured_context = shared.export_structured_context

    @bp.route('/api/translation/<translation_id>/addressing-rules', methods=['POST', 'PUT'])
    @bp.route('/<translation_id>/addressing-rules', methods=['POST', 'PUT'])
    def upsert_addressing_rule_route(translation_id):
        """Create or update one user-owned directed addressing rule."""

        data = request.get_json() or {}
        conflict, _current_revision = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        from src.utils.addressing_schema import AddressingCandidateV2
        from src.utils.context_merge_engine import ContextMergeEngine

        candidate = AddressingCandidateV2.from_dict(
            data,
            source_language=str(data.get("source_language") or ""),
            provenance="user_manual",
        )
        if not candidate or candidate.action != "upsert":
            return jsonify({"error": "A complete addressing rule is required"}), 400
        candidate.confidence = max(candidate.confidence, 0.99)
        db = state_manager.checkpoint_manager.db
        known_names = [
            str(node.get("canonical_name") or "")
            for node in db.get_relationship_nodes(translation_id)
        ]
        engine = ContextMergeEngine(db=db)
        with db.context_state_transaction():
            applied = engine.apply_delta(
                translation_id=translation_id,
                chunk_index=-1,
                delta=candidate.to_delta(),
                trigger_source="user_manual",
                target_language=str(data.get("target_language") or ""),
                known_character_names=known_names,
                active_character_names=[candidate.speaker, candidate.addressee],
                source_text="",
                source_language=str(data.get("source_language") or ""),
            )
            if not applied:
                return jsonify({
                    "error": "Addressing rule was rejected by deterministic validation"
                }), 422
            db.upsert_addressing_rule(
                translation_id,
                candidate.speaker,
                candidate.addressee,
                candidate.self_reference,
                candidate.second_person,
                vocative=candidate.vocative,
                register=candidate.register,
                social_basis=candidate.social_basis,
                scope=candidate.scope,
                contract_version=5,
                confidence=max(candidate.confidence, 0.99),
                is_locked=1 if data.get("is_locked", True) else 0,
                chunk_index=-1,
                notes=candidate.notes,
            )
            db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=-1,
                speaker_name=candidate.speaker,
                addressee_name=candidate.addressee,
                old_state=None,
                new_state={"status": "accepted", **candidate.to_dict()},
                trigger_source="user_manual",
                evidence_quote=candidate.evidence_quote,
                confidence=max(candidate.confidence, 0.99),
            )
        _export_structured_context(translation_id)
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        return jsonify({
            "message": "Addressing rule saved",
            "translation_id": translation_id,
            "context_revision": revision,
        }), 200
    @bp.route('/api/translation/<translation_id>/addressing-rules', methods=['GET'])
    @bp.route('/<translation_id>/addressing-rules', methods=['GET'])
    def get_addressing_rules_route(translation_id):
        """Get directed character addressing rules for a translation job."""
        db = state_manager.checkpoint_manager.db
        requested_status = str(request.args.get('status') or '').strip().lower()
        if requested_status not in {'active', 'provisional', 'quarantined', 'all'}:
            requested_status = 'all'
        active_rules = db.get_addressing_rules(translation_id, 'active')
        provisional_rules = db.get_addressing_rules(translation_id, 'provisional')
        quarantined_rules = db.get_addressing_rules(translation_id, 'quarantined')
        rules = (
            active_rules if requested_status == 'active'
            else provisional_rules if requested_status == 'provisional'
            else quarantined_rules if requested_status == 'quarantined'
            else active_rules
        )
        from src.utils.relationship_reasoning_engine import (
            relationship_support_for_addressing,
        )

        audits = db.get_context_audit_logs(translation_id, limit=500)
        evidence = db.get_addressing_evidence(translation_id, limit=1000)
        rejection_by_pair = {}
        rejections = []
        for item in audits:
            state = item.get("new_state") or {}
            if state.get("status") != "rejected":
                continue
            pair = (
                str(item.get("speaker_name") or "").casefold(),
                str(item.get("addressee_name") or "").casefold(),
            )
            reason = str(state.get("reason") or "")
            rejection_by_pair.setdefault(pair, reason)
            rejections.append({
                "speaker_name": item.get("speaker_name"),
                "addressee_name": item.get("addressee_name"),
                "reason": reason,
                "chunk_index": item.get("chunk_index"),
                "confidence": item.get("confidence"),
                "timestamp": item.get("timestamp"),
            })
        for rule in rules:
            resolution = relationship_support_for_addressing(
                db,
                translation_id,
                str(rule.get("speaker_name") or ""),
                str(rule.get("addressee_name") or ""),
            )
            rule["derivation_path"] = resolution.get("path", [])
            rule["derived_hierarchy"] = resolution.get("hierarchy", "unknown")
            rule["relationship_confidence"] = resolution.get("confidence", 0.0)
            rule["rejection_reason"] = rejection_by_pair.get((
                str(rule.get("speaker_name") or "").casefold(),
                str(rule.get("addressee_name") or "").casefold(),
            ))
        return jsonify({
            "translation_id": translation_id,
            "context_revision": _context_revision(translation_id),
            "rules": rules,
            "count": len(rules),
            "active_count": len(active_rules),
            "provisional_count": len(provisional_rules),
            "quarantined_count": len(quarantined_rules),
            "provisional_rules": provisional_rules,
            "quarantined_rules": quarantined_rules,
            "rejections": rejections,
            "evidence": evidence,
            "evidence_count": len(evidence),
        }), 200
    @bp.route('/<translation_id>/addressing-audit-log', methods=['GET'])
    def get_addressing_audit_log_route(translation_id):
        """Get audit log of character addressing updates for a translation job."""
        limit = request.args.get('limit', 100, type=int)
        logs = state_manager.checkpoint_manager.db.get_context_audit_logs(translation_id, limit=limit)
        return jsonify({
            "translation_id": translation_id,
            "audit_logs": logs,
            "count": len(logs)
        }), 200
    @bp.route('/api/translation/<translation_id>/addressing-resolution', methods=['GET'])
    @bp.route('/<translation_id>/addressing-resolution', methods=['GET'])
    @bp.route('/api/translation/<translation_id>/derived-seniority', methods=['GET'])
    @bp.route('/<translation_id>/derived-seniority', methods=['GET'])
    def get_addressing_resolution_route(translation_id):
        """Explain current addressing state and relationship-derived seniority."""

        speaker = str(request.args.get('speaker') or request.args.get('source') or '').strip()
        addressee = str(request.args.get('addressee') or request.args.get('target') or '').strip()
        if not speaker or not addressee:
            return jsonify({"error": "speaker/source and addressee/target are required"}), 400
        from src.utils.relationship_reasoning_engine import (
            relationship_support_for_addressing,
        )
        from src.utils.relationship_schema import normalize_relationship_name

        db = state_manager.checkpoint_manager.db
        current_rule = next((
            rule for rule in db.get_addressing_rules(translation_id)
            if str(rule.get("speaker_name") or "").casefold() == speaker.casefold()
            and str(rule.get("addressee_name") or "").casefold() == addressee.casefold()
        ), None)
        support = relationship_support_for_addressing(
            db,
            translation_id,
            speaker,
            addressee,
        )
        audit = db.get_relationship_pair_audit(
            translation_id,
            normalize_relationship_name(speaker),
            normalize_relationship_name(addressee),
        )
        return jsonify({
            "translation_id": translation_id,
            "speaker": speaker,
            "addressee": addressee,
            "addressing_rule": current_rule,
            "resolution": support,
            "evidence": audit.get("evidence", []),
            "conflicts": audit.get("conflicts", []),
            "derivations": audit.get("derivations", []),
        }), 200
    @bp.route('/<translation_id>/addressing-rules/lock', methods=['POST'])
    def set_addressing_rule_lock_route(translation_id):
        """Lock or unlock an addressing rule from being overwritten by LLM deltas."""
        data = request.get_json() or {}
        speaker_name = data.get('speaker_name')
        addressee_name = data.get('addressee_name')
        is_locked = bool(data.get('is_locked', True))

        if not speaker_name or not addressee_name:
            return jsonify({"error": "speaker_name and addressee_name are required"}), 400

        success = state_manager.checkpoint_manager.db.set_addressing_rule_lock(
            translation_id, speaker_name, addressee_name, is_locked
        )
        if success:
            return jsonify({
                "message": "Addressing rule lock updated successfully",
                "translation_id": translation_id,
                "speaker_name": speaker_name,
                "addressee_name": addressee_name,
                "is_locked": is_locked
            }), 200
        return jsonify({"error": "Failed to update addressing rule lock"}), 500
    @bp.route('/<translation_id>/addressing-rules', methods=['DELETE'])
    def delete_addressing_rule_route(translation_id):
        """Delete a directed character addressing rule for a translation job."""
        data = request.get_json() or {}
        speaker_name = data.get('speaker_name')
        addressee_name = data.get('addressee_name')

        if not speaker_name or not addressee_name:
            return jsonify({"error": "speaker_name and addressee_name are required"}), 400

        db = state_manager.checkpoint_manager.db
        existing = next(
            (
                rule for rule in db.get_addressing_rules(translation_id)
                if rule.get("speaker_name") == speaker_name
                and rule.get("addressee_name") == addressee_name
            ),
            None,
        )
        success = db.delete_addressing_rule(
            translation_id, speaker_name, addressee_name
        )
        if success:
            db.add_context_audit_log(
                translation_id=translation_id,
                chunk_index=-1,
                speaker_name=speaker_name,
                addressee_name=addressee_name,
                old_state=existing,
                new_state={"status": "deleted", "reason": "manual API delete"},
                trigger_source="rest_api:delete",
                evidence_quote="",
                confidence=1.0,
            )
            return jsonify({
                "message": "Addressing rule deleted successfully",
                "translation_id": translation_id,
                "speaker_name": speaker_name,
                "addressee_name": addressee_name,
            }), 200
        return jsonify({"error": "Addressing rule not found or could not be deleted"}), 404
