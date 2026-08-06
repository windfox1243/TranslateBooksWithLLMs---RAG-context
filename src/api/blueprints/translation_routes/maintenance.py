"""Routes for database maintenance operations."""
from flask import request, jsonify

from .helpers import (
    _ACTIVE_CONTEXT_RESYNCS,
    _CONTEXT_RESYNC_LOCK,
    logger,
)


def register(bp, deps, shared):
    """Register the maintenance routes on bp."""
    state_manager = deps.state_manager

    @bp.route('/api/maintenance/jobs-db/compact', methods=['POST'])
    def compact_jobs_database_route():
        """Back up and compact the checkpoint database while the app is idle."""

        active_statuses = {'queued', 'running', 'rate_limited', 'resyncing', 'refining'}
        active_jobs = [
            translation_id
            for translation_id, job in state_manager.get_all_translations().items()
            if str(job.get('status') or '').casefold() in active_statuses
        ]
        with _CONTEXT_RESYNC_LOCK:
            active_resyncs = sorted(_ACTIVE_CONTEXT_RESYNCS)
        if active_jobs or active_resyncs:
            return jsonify({
                "error": "Checkpoint maintenance requires an idle application",
                "active_jobs": active_jobs,
                "active_resyncs": active_resyncs,
            }), 409
        try:
            result = state_manager.checkpoint_manager.db.optimize_database()
            return jsonify({"success": True, **result})
        except Exception as exc:
            logger.error(f"Checkpoint database optimization failed: {exc}")
            return jsonify({"error": str(exc)}), 500
