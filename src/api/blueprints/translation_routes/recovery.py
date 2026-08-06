"""Routes for recovering context state from on-disk backups."""
import time
import shutil
from pathlib import Path
from flask import request, jsonify


def register(bp, deps, shared):
    """Register the recovery routes on bp."""
    state_manager = deps.state_manager
    _export_structured_context = shared.export_structured_context

    @bp.route('/api/translation/<translation_id>/recover-editor-context', methods=['POST'])
    @bp.route('/<translation_id>/recover-editor-context', methods=['POST'])
    def recover_editor_context_route(translation_id):
        """Back up and repair the explicitly selected interrupted job."""

        if translation_id != "trans_1783673914101":
            return jsonify({"error": "No dedicated recovery recipe exists for this job"}), 404
        data = request.get_json() or {}
        if data.get("confirm") is not True:
            return jsonify({"error": "Explicit recovery confirmation is required"}), 400
        db = state_manager.checkpoint_manager.db
        job = db.get_job(translation_id)
        if not job:
            return jsonify({"error": "Translation job not found"}), 404
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        backup_dir = Path(db.db_path).resolve().parent / "recovery_backups" / f"{translation_id}-{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        db.backup_to(str(backup_dir / "jobs.db"))
        config = job.get("config") or {}
        copied = []
        output_path = config.get("output_filepath")
        if output_path and Path(output_path).is_file():
            target = backup_dir / Path(output_path).name
            shutil.copy2(output_path, target)
            copied.append(str(target))
        context_name = (config.get("prompt_options") or {}).get("novel_context_file")
        if context_name:
            from src.config import NOVEL_CONTEXTS_DIR
            from src.utils.novel_context import resolve_novel_context_path
            context_path = resolve_novel_context_path(context_name, NOVEL_CONTEXTS_DIR)
            if context_path.is_file():
                target = backup_dir / context_path.name
                shutil.copy2(context_path, target)
                copied.append(str(target))

        from src.utils.relationship_reasoning_engine import RelationshipReasoningEngine
        from src.utils.relationship_schema import RelationshipCandidate
        with db.context_state_transaction():
            for edge in db.get_relationship_edges(translation_id):
                if {
                    str(edge.get("source_name") or "").casefold(),
                    str(edge.get("target_name") or "").casefold(),
                } == {"frondier de roach", "enfer de roach"} and not edge.get("is_locked"):
                    db.set_relationship_edge_status(translation_id, edge["id"], "quarantined")
            decision = RelationshipReasoningEngine(db=db).merge_candidate(
                translation_id,
                -1,
                RelationshipCandidate(
                    source="Enfer De Roach",
                    target="Frondier De Roach",
                    relationship_type="parent",
                    direction="directed",
                    hierarchy="source_senior",
                    relative_age="source_older",
                    rank_relation="source_higher",
                    confidence=1.0,
                    provenance="user_manual",
                    details="Enfer De Roach is Frondier De Roach's father.",
                ),
                known_character_names=["Enfer De Roach", "Frondier De Roach"],
            )
            if decision.edge_id:
                db.set_relationship_edge_lock(translation_id, decision.edge_id, True)
            for chunk in db.get_chunks(translation_id):
                if chunk.get("chunk_index") not in {2, 3, 4, 5} or chunk.get("status") != "failed":
                    continue
                chunk_data = dict(chunk.get("chunk_data") or {})
                chunk_data.pop("editor_validation", None)
                db.save_chunk(
                    translation_id=translation_id,
                    chunk_index=chunk["chunk_index"],
                    original_text=chunk.get("original_text"),
                    translated_text=None,
                    chunk_data=chunk_data,
                    status="failed",
                )
        _export_structured_context(translation_id)
        state_manager.checkpoint_manager.mark_partial(translation_id)
        revision = state_manager.checkpoint_manager.mark_refinement_stale(translation_id)
        return jsonify({
            "message": "Recovery prepared; resume the job to retry failed chunks.",
            "translation_id": translation_id,
            "backup_directory": str(backup_dir),
            "backed_up_files": copied,
            "context_revision": revision,
            "resume_from_chunk": 2,
        }), 200
    @bp.route('/api/translation/<translation_id>/recovery-preview', methods=['GET'])
    @bp.route('/<translation_id>/recovery-preview', methods=['GET'])
    def recovery_preview_route(translation_id):
        """Preview a non-destructive editor/context recovery operation."""

        if translation_id != "trans_1783673914101":
            return jsonify({"error": "No dedicated recovery recipe exists for this job"}), 404
        db = state_manager.checkpoint_manager.db
        job = db.get_job(translation_id)
        if not job:
            return jsonify({"error": "Translation job not found"}), 404
        chunks = db.get_chunks(translation_id)
        failed = [
            item.get("chunk_index") for item in chunks
            if item.get("status") == "failed"
        ]
        completed = [
            item.get("chunk_index") for item in chunks
            if item.get("status") == "completed"
        ]
        pair_edges = [
            edge for edge in db.get_relationship_edges(translation_id)
            if {
                str(edge.get("source_name") or "").casefold(),
                str(edge.get("target_name") or "").casefold(),
            } == {"frondier de roach", "enfer de roach"}
        ]
        config = job.get("config") or {}
        return jsonify({
            "translation_id": translation_id,
            "backup_required": True,
            "database_path": str(db.db_path),
            "context_file": (config.get("prompt_options") or {}).get("novel_context_file"),
            "output_path": config.get("output_filepath"),
            "completed_chunks_preserved": completed,
            "failed_chunks_reset": [index for index in failed if index in {2, 3, 4, 5}],
            "relationship_edges_quarantined": [edge.get("id") for edge in pair_edges],
            "relationship_correction": {
                "source": "Enfer De Roach",
                "target": "Frondier De Roach",
                "relationship_type": "parent",
                "hierarchy": "source_senior",
                "relative_age": "source_older",
                "rank_relation": "source_higher",
                "is_locked": True,
            },
        }), 200
