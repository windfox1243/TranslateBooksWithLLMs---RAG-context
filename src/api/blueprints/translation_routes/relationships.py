"""Routes for the character relationship graph."""
from flask import jsonify, request


def register(bp, deps, shared):
    """Register the relationships routes on bp."""
    state_manager = deps.state_manager
    _context_revision = shared.context_revision
    _revision_conflict = shared.revision_conflict
    _export_structured_context = shared.export_structured_context

    @bp.route('/api/translation/<translation_id>/relationship-edges', methods=['POST', 'PUT'])
    @bp.route('/<translation_id>/relationship-edges', methods=['POST', 'PUT'])
    def upsert_relationship_edge_route(translation_id):
        """Create or update one locked-by-default manual relationship fact."""

        data = request.get_json() or {}
        conflict, _current_revision = _revision_conflict(translation_id, data)
        if conflict:
            return conflict
        from src.utils.relationship_reasoning_engine import RelationshipReasoningEngine
        from src.utils.relationship_schema import RelationshipCandidate

        candidate = RelationshipCandidate.from_dict(
            {**data, "provenance": "user_manual", "confidence": data.get("confidence", 1.0)},
            default_provenance="user_manual",
            parser_status="rest_api",
        )
        if not candidate:
            return jsonify({"error": "A valid relationship candidate is required"}), 400
        db = state_manager.checkpoint_manager.db
        engine = RelationshipReasoningEngine(db=db)
        with db.context_state_transaction():
            decision = engine.merge_candidate(
                translation_id,
                -1,
                candidate,
                known_character_names=[candidate.source, candidate.target],
                language=str(data.get("language") or ""),
            )
            if decision.status not in {"accepted", "unchanged"}:
                return jsonify({
                    "error": decision.reason,
                    "status": decision.status,
                    "validator": decision.validator,
                }), 422
            if decision.edge_id and data.get("is_locked", True):
                db.set_relationship_edge_lock(
                    translation_id,
                    decision.edge_id,
                    True,
                )
        _export_structured_context(translation_id)
        revision = state_manager.checkpoint_manager.mark_refinement_stale(
            translation_id
        )
        return jsonify({
            "message": "Relationship edge saved",
            "translation_id": translation_id,
            "edge_id": decision.edge_id,
            "status": decision.status,
            "context_revision": revision,
        }), 200
    @bp.route('/api/translation/<translation_id>/relationship-graph', methods=['GET'])
    @bp.route('/<translation_id>/relationship-graph', methods=['GET'])
    def get_relationship_graph_route(translation_id):
        """Get relationship graph nodes and edges for a translation job."""

        db = state_manager.checkpoint_manager.db
        from src.utils.relationship_reasoning_engine import (
            migrate_relationship_reasoning_v2,
        )

        migration = migrate_relationship_reasoning_v2(db, translation_id)
        status = request.args.get('status')
        statuses = [status] if status else None
        nodes = db.get_relationship_nodes(translation_id)
        edges = db.get_relationship_edges(translation_id, statuses=statuses)
        evidence = db.get_relationship_evidence(translation_id, limit=1000)
        return jsonify({
            "translation_id": translation_id,
            "context_revision": _context_revision(translation_id),
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "evidence": evidence,
            "evidence_count": len(evidence),
            "validator_version": 2,
            "migration": migration,
        }), 200
    @bp.route('/api/translation/<translation_id>/relationship-conflicts', methods=['GET'])
    @bp.route('/<translation_id>/relationship-conflicts', methods=['GET'])
    def get_relationship_conflicts_route(translation_id):
        """Get open or historical relationship validation conflicts."""

        status = request.args.get('status')
        limit = request.args.get('limit', 200, type=int)
        conflicts = (
            state_manager.checkpoint_manager.db.get_relationship_conflicts(
                translation_id,
                status=status,
                limit=max(1, min(limit, 1000)),
            )
        )
        return jsonify({
            "translation_id": translation_id,
            "conflicts": conflicts,
            "count": len(conflicts),
        }), 200
    @bp.route('/api/translation/<translation_id>/relationship-audit', methods=['GET'])
    @bp.route('/<translation_id>/relationship-audit', methods=['GET'])
    def get_relationship_pair_audit_route(translation_id):
        """Get accepted state, evidence, and conflicts for one character pair."""

        source_name = str(request.args.get('source') or '').strip()
        target_name = str(request.args.get('target') or '').strip()
        if not source_name or not target_name:
            return jsonify({"error": "source and target are required"}), 400
        from src.utils.relationship_schema import normalize_relationship_name

        audit = state_manager.checkpoint_manager.db.get_relationship_pair_audit(
            translation_id,
            normalize_relationship_name(source_name),
            normalize_relationship_name(target_name),
        )
        return jsonify({
            "translation_id": translation_id,
            "source": source_name,
            "target": target_name,
            **audit,
        }), 200
    @bp.route('/api/translation/<translation_id>/relationship-edges/<int:edge_id>/lock', methods=['POST'])
    @bp.route('/<translation_id>/relationship-edges/<int:edge_id>/lock', methods=['POST'])
    def set_relationship_edge_lock_route(translation_id, edge_id):
        """Lock or unlock a relationship edge."""

        data = request.get_json() or {}
        is_locked = bool(data.get('is_locked', True))
        success = state_manager.checkpoint_manager.db.set_relationship_edge_lock(
            translation_id,
            edge_id,
            is_locked,
        )
        if not success:
            return jsonify({"error": "Relationship edge not found"}), 404
        return jsonify({
            "message": "Relationship edge lock updated successfully",
            "translation_id": translation_id,
            "edge_id": edge_id,
            "is_locked": is_locked,
        }), 200
    @bp.route('/api/translation/<translation_id>/relationship-edges/<int:edge_id>/quarantine', methods=['POST'])
    @bp.route('/<translation_id>/relationship-edges/<int:edge_id>/quarantine', methods=['POST'])
    def quarantine_relationship_edge_route(translation_id, edge_id):
        """Quarantine an unlocked relationship edge and retain its audit state."""

        db = state_manager.checkpoint_manager.db
        existing = next((
            edge for edge in db.get_relationship_edges(translation_id)
            if edge.get("id") == edge_id
        ), None)
        if not existing:
            return jsonify({"error": "Relationship edge not found"}), 404
        if existing.get("is_locked"):
            return jsonify({"error": "Locked relationship edges cannot be quarantined"}), 409
        if not db.set_relationship_edge_status(translation_id, edge_id, "quarantined"):
            return jsonify({"error": "Relationship edge could not be quarantined"}), 409
        db.add_relationship_conflict(
            translation_id=translation_id,
            source_name=existing.get("source_name") or "",
            target_name=existing.get("target_name") or "",
            severity="warning",
            validator="manual_quarantine",
            reason="Relationship edge quarantined through the REST API.",
            remediation_hint="Review the source evidence before restoring this edge.",
            candidate=existing,
            chunk_index=-1,
            edge_id=edge_id,
        )
        return jsonify({
            "message": "Relationship edge quarantined successfully",
            "translation_id": translation_id,
            "edge_id": edge_id,
        }), 200
    @bp.route('/api/translation/<translation_id>/relationship-edges/<int:edge_id>', methods=['DELETE'])
    @bp.route('/<translation_id>/relationship-edges/<int:edge_id>', methods=['DELETE'])
    def delete_relationship_edge_route(translation_id, edge_id):
        """Delete an unlocked relationship edge and retain an audit marker."""

        db = state_manager.checkpoint_manager.db
        existing = next((
            edge for edge in db.get_relationship_edges(translation_id)
            if edge.get("id") == edge_id
        ), None)
        if not existing:
            return jsonify({"error": "Relationship edge not found"}), 404
        if existing.get("is_locked"):
            return jsonify({"error": "Locked relationship edges cannot be deleted"}), 409
        conflict_id = db.add_relationship_conflict(
            translation_id=translation_id,
            source_name=existing.get("source_name") or "",
            target_name=existing.get("target_name") or "",
            severity="info",
            validator="manual_delete",
            reason="Relationship edge deleted through the REST API.",
            remediation_hint="No action required.",
            candidate=existing,
            chunk_index=-1,
            edge_id=edge_id,
        )
        success = db.delete_relationship_edge(translation_id, edge_id)
        if not success:
            return jsonify({"error": "Relationship edge could not be deleted"}), 409
        if conflict_id:
            db.resolve_relationship_conflict(translation_id, conflict_id)
        return jsonify({
            "message": "Relationship edge deleted successfully",
            "translation_id": translation_id,
            "edge_id": edge_id,
        }), 200
