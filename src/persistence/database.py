"""
SQLite database manager for translation job persistence.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.persistence.schema import _evidence_fingerprint, apply_schema

logger = logging.getLogger("persistence.database")


# _evidence_fingerprint now lives in schema.py, because the migration ALTERs
# there backfill it. It is imported above and stays reachable from this module
# for the query methods below.


def _tighten_name(value: str) -> str:
    """Drop the spacing and joining punctuation a name is written with."""

    return re.sub(r"[\s\-_.'’]+", "", str(value or ""))


_TRANSLATION_CHILD_TABLES = (
    "refinement_chunk_results",
    "refinement_passes",
    "editor_repair_batch_items",
    "editor_repair_batches",
    "editor_runs",
    "context_narrator_bootstrap_attempts",
    "context_narrator_conflicts",
    "context_narrator_transitions",
    "context_narrator_observations",
    "context_narrator_profiles",
    "context_resync_chunk_stage",
    "context_resync_runs",
    "context_relationship_derivations",
    "context_relationship_conflicts",
    "context_relationship_evidence",
    "context_relationship_edge_history",
    "context_relationship_edges",
    "context_relationship_nodes",
    "context_reasoning_migrations",
    "context_audit_logs",
    "context_addressing_evidence",
    "context_addressing_rules",
    "context_entities",
    "checkpoint_chunks",
)


def sanitize_config_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of a job config with every API key removed.

    The jobs database must never be a secret-bearing artifact (issue #213):
    it lives on disk unencrypted and is bind-mounted by docker-compose. Keys
    are therefore stripped before persistence and re-resolved at resume time
    from the environment (the LLM factory falls back to <PROVIDER>_API_KEY
    when the config value is empty) or from the resume request body.
    """
    return {
        k: v for k, v in config.items()
        if not (k == 'api_key' or k.endswith('_api_key'))
    }


class Database:
    """
    Manages SQLite database for translation job checkpoints.
    Thread-safe for concurrent access.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to database. If None, uses default path from config.
        """
        if db_path is None:
            from src.config import DATA_DIR
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            db_path = str(DATA_DIR / "jobs.db")

        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.RLock()
        # _local.connection is per-thread and unreachable from other threads;
        # this list lets close_all() drop every connection on shutdown. Without
        # it, every translation worker thread leaks a connection and its file
        # handle for the lifetime of the process.
        self._all_connections: List[sqlite3.Connection] = []
        self._connections_lock = threading.RLock()
        self._repositories = {}

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._initialize_schema()

    def _repository(self, name: str):
        """Return this database's repository facade, building it once.

        The repositories are stateless views over this object, so allocating a
        fresh one per attribute access was pure overhead on hot paths that
        touch db.jobs or db.editor inside a loop.
        """
        cached = self._repositories.get(name)
        if cached is None:
            from src.persistence import repositories

            cached = getattr(repositories, name)(self)
            self._repositories[name] = cached
        return cached

    @property
    def jobs(self):
        """Narrow job repository; legacy methods remain available on Database."""
        return self._repository('JobRepository')

    @property
    def editor(self):
        """Narrow editor repository; legacy methods remain available on Database."""
        return self._repository('EditorRepository')

    @property
    def context(self):
        """Narrow structured-context repository compatibility boundary."""
        return self._repository('ContextRepository')

    @property
    def narrator(self):
        """Narrow narrator repository; legacy methods remain available."""
        return self._repository('NarratorRepository')

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.connection = conn
            with self._connections_lock:
                self._all_connections.append(conn)
        return self._local.connection

    def _commit_connection(self, conn: sqlite3.Connection) -> None:
        """Commit unless an outer context-state transaction owns the connection."""

        if not getattr(self._local, "context_transaction_depth", 0):
            conn.commit()

    @contextmanager
    def context_state_transaction(self):
        """Atomically commit addressing and relationship state for one chunk."""

        with self._lock:
            conn = self._get_connection()
            depth = int(getattr(self._local, "context_transaction_depth", 0))
            if depth == 0:
                conn.execute("BEGIN IMMEDIATE")
            self._local.context_transaction_depth = depth + 1
            try:
                yield
                self._local.context_transaction_depth -= 1
                if depth == 0:
                    conn.commit()
            except Exception:
                self._local.context_transaction_depth = depth
                if depth == 0:
                    conn.rollback()
                raise

    def _initialize_schema(self):
        """Create database tables if they don't exist."""
        apply_schema(self._lock, self._get_connection)

    # The job and editor SQL now lives in JobRepository / EditorRepository.
    # These methods stay as delegations because the call sites -- handlers,
    # checkpoint manager, routes, tests -- all reach the database through this
    # facade. Removing them would be a separate, much wider change.

    def create_job(
        self,
        translation_id: str,
        file_type: str,
        config: Dict[str, Any],
        server_session_id: Optional[str] = None
    ) -> bool:
        """Create a new translation job record."""
        return self.jobs.create_job(
            translation_id,
            file_type,
            config,
            server_session_id,
        )

    def update_job_progress(
        self,
        translation_id: str,
        current_chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None,
        failed_chunks: Optional[int] = None,
        review_required_chunks: Optional[int] = None,
        status: Optional[str] = None,
        epub_accumulated_stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update job progress information."""
        return self.jobs.update_job_progress(
            translation_id,
            current_chunk_index,
            total_chunks,
            completed_chunks,
            failed_chunks,
            review_required_chunks,
            status,
            epub_accumulated_stats,
        )

    def save_chunk(
        self,
        translation_id: str,
        chunk_index: int,
        original_text: str,
        translated_text: Optional[str] = None,
        chunk_data: Optional[Dict[str, Any]] = None,
        status: str = 'completed',
        quality_status: Optional[str] = None,
        execution_failure_class: Optional[str] = None,
    ) -> bool:
        """Save a translated chunk to database."""
        return self.jobs.save_chunk(
            translation_id,
            chunk_index,
            original_text,
            translated_text,
            chunk_data,
            status,
            quality_status,
            execution_failure_class,
        )

    def get_job(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve job information."""
        return self.jobs.get_job(translation_id)

    def update_job_config(self, translation_id: str, config: Dict[str, Any]) -> bool:
        """Update the configuration of an existing job."""
        return self.jobs.update_job_config(translation_id, config)

    def get_chunks(self, translation_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a job.

        Args:
            translation_id: Job identifier

        Returns:
            List of chunk dictionaries ordered by chunk_index
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    SELECT * FROM checkpoint_chunks
                    WHERE translation_id = ?
                    ORDER BY chunk_index
                """, (translation_id,))

                chunks = []
                for row in cursor.fetchall():
                    chunks.append({
                        'chunk_index': row['chunk_index'],
                        'original_text': row['original_text'],
                        'translated_text': row['translated_text'],
                        'chunk_data': json.loads(row['chunk_data']) if row['chunk_data'] else None,
                        'status': row['status'],
                        'quality_status': row['quality_status'],
                        'execution_failure_class': row['execution_failure_class'],
                        'completed_at': row['completed_at']
                    })

                return chunks
            except Exception as e:
                print(f"Error getting chunks: {e}")
                return []

    def get_failed_chunk_indices(self, translation_id: str) -> List[int]:
        """
        Return the chunk_index of every chunk currently marked 'failed'.

        Used by the resume path to retry chunks that previously errored,
        instead of leaving them as untranslated holes in the output.
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT chunk_index FROM checkpoint_chunks
                    WHERE translation_id = ? AND status = 'failed'
                    ORDER BY chunk_index
                """, (translation_id,))
                return [row['chunk_index'] for row in cursor.fetchall()]
            except Exception as e:
                print(f"Error getting failed chunks: {e}")
                return []

    def get_chunk_status_counts(self, translation_id: str) -> Dict[str, int]:
        """Return checkpoint chunk counts grouped by current row status."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT status, COUNT(*) AS count
                    FROM checkpoint_chunks
                    WHERE translation_id = ?
                    GROUP BY status
                """, (translation_id,))
                return {
                    row['status']: row['count']
                    for row in cursor.fetchall()
                    if row['status']
                }
            except Exception as e:
                print(f"Error getting chunk status counts: {e}")
                return {}

    def get_chunk_quality_counts(self, translation_id: str) -> Dict[str, int]:
        """Return checkpoint chunk counts grouped by quality status."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT quality_status, COUNT(*) AS count
                    FROM checkpoint_chunks
                    WHERE translation_id = ?
                    GROUP BY quality_status
                """, (translation_id,))
                return {
                    row['quality_status']: row['count']
                    for row in cursor.fetchall()
                    if row['quality_status']
                }
            except Exception as e:
                print(f"Error getting chunk quality counts: {e}")
                return {}

    def get_resumable_jobs(self, max_age_days: int = 30) -> List[Dict[str, Any]]:
        """Get all jobs that can resume or seed an Add New Content job."""
        return self.jobs.get_resumable_jobs(max_age_days)

    def delete_old_jobs(self, max_age_days: int = 30) -> List[str]:
        """
        Delete old jobs that are no longer relevant and report which ones went.

        Removes jobs older than max_age_days that are in resumable states
        (paused, interrupted, error) to prevent database bloat.

        Callers that also need to clean up on-disk artifacts must drive that
        cleanup from the returned IDs. Re-deriving the cutoff in Python is a
        bug: created_at is stored in UTC by SQLite's CURRENT_TIMESTAMP, so a
        cutoff built from datetime.now() disagrees with this query by the local
        UTC offset and deletes files for jobs the database kept.

        Args:
            max_age_days: Maximum age in days for jobs to keep (default 30)

        Returns:
            The translation IDs that were deleted
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get IDs of jobs to delete (for logging and file cleanup)
                cursor.execute("""
                    SELECT translation_id FROM translation_jobs
                    WHERE status IN ('paused', 'interrupted', 'error', 'partial', 'completed')
                    AND created_at <= datetime('now', ? || ' days')
                """, (f'-{max_age_days}',))

                job_ids = [row['translation_id'] for row in cursor.fetchall()]

                if not job_ids:
                    return []

                for translation_id in job_ids:
                    self._delete_job_rows(conn, translation_id)
                conn.commit()
                return job_ids
            except Exception as e:
                print(f"Error cleaning up old jobs: {e}")
                return []

    def cleanup_old_jobs(self, max_age_days: int = 30) -> int:
        """
        Delete old jobs and return how many were removed.

        Thin wrapper over delete_old_jobs for callers that only need the count.
        """
        return len(self.delete_old_jobs(max_age_days))

    def reset_running_jobs(self, current_session_id: str) -> int:
        """
        Reset jobs with 'running' status from previous server sessions to 'interrupted'.

        Only resets jobs that have a different server_session_id than the current one,
        preserving jobs that are actually running in the current session.

        This should be called on server startup to handle jobs that were
        interrupted by a server crash or restart.

        Args:
            current_session_id: The current server's session ID

        Returns:
            Number of jobs reset
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Reset jobs that are 'running' but from a different session
                # (or have no session_id, meaning they're from before this feature)
                cursor.execute("""
                    UPDATE translation_jobs
                    SET status = 'interrupted',
                        updated_at = CURRENT_TIMESTAMP,
                        paused_at = CURRENT_TIMESTAMP
                    WHERE status = 'running'
                    AND (server_session_id IS NULL OR server_session_id != ?)
                """, (current_session_id,))

                affected_rows = cursor.rowcount
                conn.commit()
                return affected_rows
            except Exception as e:
                print(f"Error resetting running jobs: {e}")
                return 0

    def update_translation_context(
        self,
        translation_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Update translation context for continuity.

        Args:
            translation_id: Job identifier
            context: Context data (last_llm_context, context_accumulator, etc.)

        Returns:
            True if updated successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE translation_jobs
                    SET translation_context = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ?
                """, (json.dumps(context), translation_id))

                conn.commit()
                return True
            except Exception as e:
                print(f"Error updating translation context: {e}")
                return False

    def delete_job(self, translation_id: str) -> bool:
        """Delete a job and all its chunks (CASCADE)."""
        return self.jobs.delete_job(translation_id)

    @staticmethod
    def _delete_job_rows(conn: sqlite3.Connection, translation_id: str) -> None:
        """Delete one job explicitly, including legacy databases without FKs."""

        conn.execute(
            "DELETE FROM editor_attempts WHERE run_id IN "
            "(SELECT id FROM editor_runs WHERE translation_id = ?)",
            (translation_id,),
        )
        for table in _TRANSLATION_CHILD_TABLES:
            conn.execute(
                f"DELETE FROM {table} WHERE translation_id = ?",
                (translation_id,),
            )
        conn.execute(
            "DELETE FROM translation_jobs WHERE translation_id = ?",
            (translation_id,),
        )

    def find_previous_job_for_context_file(
        self,
        novel_context_file: str,
        exclude_translation_id: str = "",
        scan_limit: int = 200,
    ) -> Optional[str]:
        """Return the newest other job that used the same novel context file.

        Read from Python rather than json_extract so the query does not depend
        on the JSON1 extension being compiled into whatever SQLite the packaged
        build ships with.
        """
        wanted = str(novel_context_file or "").strip()
        if not wanted:
            return None
        with self._lock:
            try:
                conn = self._get_connection()
                rows = conn.execute(
                    "SELECT translation_id, config FROM translation_jobs "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (int(scan_limit),),
                ).fetchall()
            except Exception as exc:
                logger.warning("Could not scan jobs for %r: %s", wanted, exc)
                return None
        for row in rows:
            translation_id = row["translation_id"]
            if not translation_id or translation_id == exclude_translation_id:
                continue
            try:
                config = json.loads(row["config"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            options = (config or {}).get("prompt_options") or {}
            if str(options.get("novel_context_file") or "").strip() == wanted:
                return translation_id
        return None

    def _copy_context_rows(
        self,
        conn: sqlite3.Connection,
        table: str,
        source_translation_id: str,
        target_translation_id: str,
        overrides: Optional[Dict[str, Dict[Any, Any]]] = None,
    ) -> List[Tuple[Any, Any]]:
        """Copy one job's rows of `table` to another job, returning id pairs.

        Columns are read from the row itself so a later schema migration is
        carried over without having to be listed here twice.
        """
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE translation_id = ?",
            (source_translation_id,),
        ).fetchall()
        copied: List[Tuple[Any, Any]] = []
        for row in rows:
            values = dict(row)
            old_id = values.pop("id", None)
            values["translation_id"] = target_translation_id
            for column, mapping in (overrides or {}).items():
                if column in values:
                    remapped = mapping.get(values[column])
                    if remapped is None:
                        values = {}
                        break
                    values[column] = remapped
            if not values:
                # An edge whose node did not come across would dangle.
                continue
            columns = ", ".join(values)
            placeholders = ", ".join("?" for _ in values)
            cursor = conn.execute(
                f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
                tuple(values.values()),
            )
            if cursor.rowcount:
                copied.append((old_id, cursor.lastrowid))
        return copied

    def carry_over_structured_context(
        self,
        target_translation_id: str,
        source_translation_id: str,
    ) -> Dict[str, int]:
        """Seed a job's structured context from an earlier job on the same book.

        The markdown context file is per novel, but every structured table is
        keyed by job, and exporting those tables to markdown keeps only the
        rendered line: locks, confidence, provenance and evidence do not
        survive the round trip. So volume 2 used to start with an addressing
        rule the user had locked in volume 1 unlocked, at default confidence,
        and free to be overwritten by the model's first guess.

        Copying is skipped when the target already has structured state, so a
        resumed or re-run job is never overwritten.
        """
        counts: Dict[str, int] = {}
        if not target_translation_id or not source_translation_id:
            return counts
        if target_translation_id == source_translation_id:
            return counts
        with self._lock:
            try:
                conn = self._get_connection()
                existing = conn.execute(
                    "SELECT (SELECT COUNT(*) FROM context_addressing_rules "
                    "        WHERE translation_id = ?) "
                    "     + (SELECT COUNT(*) FROM context_relationship_nodes "
                    "        WHERE translation_id = ?)",
                    (target_translation_id, target_translation_id),
                ).fetchone()[0]
                if existing:
                    return counts

                node_ids = dict(
                    self._copy_context_rows(
                        conn,
                        "context_relationship_nodes",
                        source_translation_id,
                        target_translation_id,
                    )
                )
                counts["relationship_nodes"] = len(node_ids)
                edge_ids = dict(
                    self._copy_context_rows(
                        conn,
                        "context_relationship_edges",
                        source_translation_id,
                        target_translation_id,
                        overrides={
                            "source_node_id": node_ids,
                            "target_node_id": node_ids,
                        },
                    )
                )
                counts["relationship_edges"] = len(edge_ids)
                counts["relationship_history"] = len(
                    self._copy_context_rows(
                        conn,
                        "context_relationship_edge_history",
                        source_translation_id,
                        target_translation_id,
                        overrides={"edge_id": edge_ids},
                    )
                )
                for table, key in (
                    ("context_addressing_rules", "addressing_rules"),
                    ("context_addressing_evidence", "addressing_evidence"),
                    ("context_entities", "entities"),
                ):
                    counts[key] = len(
                        self._copy_context_rows(
                            conn,
                            table,
                            source_translation_id,
                            target_translation_id,
                        )
                    )
                conn.commit()
            except Exception as exc:
                logger.warning(
                    "Could not carry structured context from %s to %s: %s",
                    source_translation_id,
                    target_translation_id,
                    exc,
                )
                return {}
        return {key: value for key, value in counts.items() if value}

    def purge_orphan_rows(self) -> Dict[str, int]:
        """Remove rows whose translation job no longer exists.

        Each orphan translation ID is committed independently so large legacy
        databases do not create one enormous WAL transaction at startup.
        """

        started = time.monotonic()
        deleted_rows = 0
        logical_bytes = 0
        orphan_ids = set()
        with self._lock:
            conn = self._get_connection()
            for table in _TRANSLATION_CHILD_TABLES:
                rows = conn.execute(
                    f"SELECT DISTINCT t.translation_id FROM {table} t "
                    "LEFT JOIN translation_jobs j "
                    "ON j.translation_id = t.translation_id "
                    "WHERE j.translation_id IS NULL"
                ).fetchall()
                orphan_ids.update(row[0] for row in rows if row[0])

            for translation_id in sorted(orphan_ids):
                size_row = conn.execute(
                    "SELECT COALESCE(SUM(length(original_text) + "
                    "COALESCE(length(translated_text), 0) + "
                    "COALESCE(length(chunk_data), 0)), 0) "
                    "FROM checkpoint_chunks WHERE translation_id = ?",
                    (translation_id,),
                ).fetchone()
                logical_bytes += int(size_row[0] or 0)
                conn.execute(
                    "DELETE FROM editor_attempts WHERE run_id IN "
                    "(SELECT id FROM editor_runs WHERE translation_id = ?)",
                    (translation_id,),
                )
                for table in _TRANSLATION_CHILD_TABLES:
                    cursor = conn.execute(
                        f"DELETE FROM {table} WHERE translation_id = ?",
                        (translation_id,),
                    )
                    deleted_rows += max(0, int(cursor.rowcount or 0))
                conn.commit()

            if orphan_ids:
                conn.execute("PRAGMA wal_checkpoint(PASSIVE)")

        return {
            "translation_ids": len(orphan_ids),
            "deleted_rows": deleted_rows,
            "logical_bytes": logical_bytes,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    def optimize_database(self, backup_path: Optional[str] = None) -> Dict[str, Any]:
        """Back up, compact, and integrity-check an idle jobs database."""

        with self._lock:
            conn = self._get_connection()
            quick_check = conn.execute("PRAGMA quick_check").fetchone()[0]
            if quick_check != "ok":
                raise RuntimeError(f"Database quick check failed: {quick_check}")

            db_file = Path(self.db_path).resolve()
            def physical_bytes() -> int:
                return sum(
                    path.stat().st_size
                    for path in (
                        db_file,
                        Path(str(db_file) + "-wal"),
                        Path(str(db_file) + "-shm"),
                    )
                    if path.exists()
                )

            before_bytes = physical_bytes()
            free_bytes = shutil.disk_usage(db_file.parent).free
            required_bytes = max(64 * 1024 * 1024, before_bytes * 2)
            if free_bytes < required_bytes:
                raise RuntimeError(
                    "Not enough free space to back up and compact the database"
                )
            if not backup_path:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backup_path = str(db_file.with_name(f"{db_file.name}.{stamp}.bak"))

            backup = sqlite3.connect(backup_path)
            try:
                conn.backup(backup)
            finally:
                backup.close()

            orphan_stats = self.purge_orphan_rows()
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
            # VACUUM in WAL mode can leave the compact image in a large WAL;
            # checkpoint once more so the physical on-disk size is reclaimed.
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            final_check = conn.execute("PRAGMA quick_check").fetchone()[0]
            if final_check != "ok":
                raise RuntimeError(f"Database integrity check failed: {final_check}")
            after_bytes = physical_bytes()
            return {
                "before_bytes": before_bytes,
                "after_bytes": after_bytes,
                "backup_path": str(Path(backup_path).resolve()),
                "orphan_cleanup": orphan_stats,
                "integrity": final_check,
            }

    def create_editor_run(self, payload: Dict[str, Any]) -> Optional[int]:
        """Create one locally persisted Senior Editor run."""
        return self.editor.create_editor_run(payload)

    def add_editor_attempt(self, run_id: int, payload: Dict[str, Any]) -> bool:
        """Append a bounded diagnostic record for one editor request."""
        return self.editor.add_editor_attempt(run_id, payload)

    def finish_editor_run(self, run_id: int, payload: Dict[str, Any]) -> bool:
        """Finalize one editor run with a classified outcome."""
        return self.editor.finish_editor_run(run_id, payload)

    def get_editor_diagnostics(self, translation_id: str) -> Dict[str, Any]:
        """Return aggregate and per-run editor diagnostics for a job."""
        return self.editor.get_editor_diagnostics(translation_id)

    def create_editor_repair_batch(
        self, batch_id: str, translation_id: str, scope: str, phase: str,
        chunk_indices: List[int], *, stay_paused: bool = True,
    ) -> bool:
        """Persist a repair batch and its immutable initial work list."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO editor_repair_batches "
                    "(batch_id, translation_id, scope, phase, stay_paused, total_items) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (batch_id, translation_id, scope, phase,
                     int(bool(stay_paused)), len(chunk_indices)),
                )
                conn.executemany(
                    "INSERT INTO editor_repair_batch_items "
                    "(batch_id, translation_id, chunk_index, phase) VALUES (?, ?, ?, ?)",
                    [(batch_id, translation_id, int(index), phase)
                     for index in chunk_indices],
                )
                self._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error creating editor repair batch: {exc}")
                return False

    def reconcile_interrupted_operations(self) -> None:
        """Mark work left active by a previous application process as interrupted."""
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "UPDATE editor_repair_batches SET status='interrupted', "
                "error=COALESCE(error, 'application_restarted'), "
                "completed_at=CURRENT_TIMESTAMP "
                "WHERE status IN ('queued', 'pausing', 'running')"
            )
            conn.execute(
                "UPDATE refinement_passes SET status='interrupted', "
                "error=COALESCE(error, 'application_restarted'), "
                "completed_at=CURRENT_TIMESTAMP WHERE status='running'"
            )
            self._commit_connection(conn)

    def update_editor_repair_batch(
        self, batch_id: str, **changes: Any,
    ) -> bool:
        allowed = {
            "status", "cancel_requested", "completed_items",
            "succeeded_items", "failed_items", "error", "started_at",
            "completed_at",
        }
        payload = {key: value for key, value in changes.items() if key in allowed}
        if not payload:
            return True
        with self._lock:
            conn = self._get_connection()
            assignments = ", ".join(f"{key} = ?" for key in payload)
            values = list(payload.values()) + [batch_id]
            cursor = conn.execute(
                f"UPDATE editor_repair_batches SET {assignments} WHERE batch_id = ?",
                values,
            )
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def update_editor_repair_batch_item(
        self, batch_id: str, chunk_index: int, phase: str, **changes: Any,
    ) -> bool:
        allowed = {"status", "outcome", "message", "started_at", "completed_at"}
        payload = {key: value for key, value in changes.items() if key in allowed}
        if not payload:
            return True
        with self._lock:
            conn = self._get_connection()
            assignments = ", ".join(f"{key} = ?" for key in payload)
            values = list(payload.values()) + [batch_id, int(chunk_index), phase]
            cursor = conn.execute(
                f"UPDATE editor_repair_batch_items SET {assignments} "
                "WHERE batch_id = ? AND chunk_index = ? AND phase = ?",
                values,
            )
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def get_editor_repair_batch(self, batch_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM editor_repair_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["stay_paused"] = bool(result.get("stay_paused"))
            result["cancel_requested"] = bool(result.get("cancel_requested"))
            result["items"] = [dict(item) for item in conn.execute(
                "SELECT * FROM editor_repair_batch_items WHERE batch_id = ? "
                "ORDER BY chunk_index",
                (batch_id,),
            ).fetchall()]
            return result

    def find_active_editor_repair_batch(
        self, translation_id: str,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT batch_id FROM editor_repair_batches "
                "WHERE translation_id = ? AND status IN ('queued', 'pausing', 'running') "
                "ORDER BY created_at DESC LIMIT 1",
                (translation_id,),
            ).fetchone()
        return self.get_editor_repair_batch(row["batch_id"]) if row else None

    def get_latest_editor_repair_batch(
        self, translation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest repair batch, including its persisted item state."""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT batch_id FROM editor_repair_batches "
                "WHERE translation_id = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (translation_id,),
            ).fetchone()
        return self.get_editor_repair_batch(row["batch_id"]) if row else None

    def create_refinement_pass(
        self, pass_id: str, translation_id: str, *, context_revision: int = 0,
        source_mode: str = "checkpoint", alignment_mode: str = "exact",
        expected_units: int = 0,
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO refinement_passes "
                    "(pass_id, translation_id, context_revision, source_mode, "
                    "alignment_mode, expected_units) VALUES (?, ?, ?, ?, ?, ?)",
                    (pass_id, translation_id, int(context_revision), source_mode,
                     alignment_mode, int(expected_units)),
                )
                self._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error creating refinement pass: {exc}")
                return False

    def save_refinement_chunk_result(
        self, pass_id: str, translation_id: str, chunk_index: int,
        *, base_chunk_index: Optional[int], source_text: str, refined_text: str,
        chunk_data: Optional[Dict[str, Any]] = None, status: str = "completed",
        quality_status: str = "passed",
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            conn.execute(
                "INSERT INTO refinement_chunk_results "
                "(pass_id, translation_id, chunk_index, base_chunk_index, source_text, "
                "refined_text, chunk_data, status, quality_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(pass_id, chunk_index) DO UPDATE SET "
                "base_chunk_index=excluded.base_chunk_index, source_text=excluded.source_text, "
                "refined_text=excluded.refined_text, chunk_data=excluded.chunk_data, "
                "status=excluded.status, quality_status=excluded.quality_status, "
                "updated_at=CURRENT_TIMESTAMP",
                (pass_id, translation_id, int(chunk_index), base_chunk_index,
                 source_text, refined_text,
                 json.dumps(chunk_data or {}, ensure_ascii=False), status, quality_status),
            )
            self._commit_connection(conn)
            return True

    def finish_refinement_pass(
        self, pass_id: str, *, successful: bool, error: str = "",
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT translation_id, expected_units, alignment_mode "
                "FROM refinement_passes WHERE pass_id = ?", (pass_id,),
            ).fetchone()
            if not row:
                return False
            counts = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS usable "
                "FROM refinement_chunk_results WHERE pass_id = ?", (pass_id,),
            ).fetchone()
            exact = (
                row["alignment_mode"] == "exact"
                and int(counts["total"] or 0) == int(row["expected_units"] or 0)
                and int(counts["usable"] or 0) == int(row["expected_units"] or 0)
            )
            promoted = bool(successful and exact)
            status = "completed" if promoted else "failed"
            failure = error or ("refinement_unit_alignment_mismatch" if successful else "refinement_failed")
            conn.execute(
                "UPDATE refinement_passes SET status=?, promoted=?, error=?, "
                "completed_at=CURRENT_TIMESTAMP WHERE pass_id=?",
                (status, int(promoted), None if promoted else failure, pass_id),
            )
            if promoted:
                conn.execute(
                    "UPDATE refinement_passes SET promoted=0 WHERE translation_id=? "
                    "AND pass_id<>?", (row["translation_id"], pass_id),
                )
            self._commit_connection(conn)
            return promoted

    def get_active_refinement_results(self, translation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT pass_id FROM refinement_passes WHERE translation_id=? "
                "AND promoted=1 ORDER BY completed_at DESC LIMIT 1",
                (translation_id,),
            ).fetchone()
            if not row:
                return []
            results = [dict(item) for item in conn.execute(
                "SELECT * FROM refinement_chunk_results WHERE pass_id=? "
                "ORDER BY chunk_index", (row["pass_id"],),
            ).fetchall()]
            for item in results:
                try:
                    item["chunk_data"] = json.loads(item.get("chunk_data") or "{}")
                except (TypeError, ValueError):
                    item["chunk_data"] = {}
            return results

    def get_running_refinement_pass(
        self, translation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the newest in-progress refinement pass for threshold repair."""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM refinement_passes WHERE translation_id=? "
                "AND status='running' ORDER BY created_at DESC, rowid DESC LIMIT 1",
                (translation_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_refinement_results(self, pass_id: str) -> List[Dict[str, Any]]:
        """Return persisted units for a specific promoted or running pass."""
        with self._lock:
            conn = self._get_connection()
            results = [dict(item) for item in conn.execute(
                "SELECT * FROM refinement_chunk_results WHERE pass_id=? "
                "ORDER BY chunk_index", (pass_id,),
            ).fetchall()]
        for item in results:
            try:
                item["chunk_data"] = json.loads(item.get("chunk_data") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                item["chunk_data"] = {}
        return results

    def upsert_addressing_rule(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
        self_pronoun: str,
        target_pronoun: str,
        vocative: Optional[str] = None,
        register: Optional[str] = None,
        social_basis: Optional[List[str]] = None,
        scope: str = "durable",
        contract_version: int = 1,
        confidence: float = 1.0,
        is_locked: int = 0,
        chunk_index: int = 0,
        notes: str = "",
        validation_status: str = "active",
        validation_reason: str = "",
        provenance: str = "unknown",
    ) -> bool:
        """Upsert a directed addressing rule for a speaker-addressee pair."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO context_addressing_rules (
                        translation_id, speaker_name, addressee_name,
                        self_pronoun, target_pronoun, vocative, register,
                        social_basis, notes, scope, contract_version, confidence,
                        is_locked, validation_status, validation_reason,
                        provenance, validated_at, last_chunk_index, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(translation_id, speaker_name, addressee_name) DO UPDATE SET
                        self_pronoun = excluded.self_pronoun,
                        target_pronoun = excluded.target_pronoun,
                        vocative = excluded.vocative,
                        register = excluded.register,
                        social_basis = excluded.social_basis,
                        notes = excluded.notes,
                        scope = excluded.scope,
                        contract_version = MAX(
                            context_addressing_rules.contract_version,
                            excluded.contract_version
                        ),
                        confidence = excluded.confidence,
                        validation_status = excluded.validation_status,
                        validation_reason = excluded.validation_reason,
                        provenance = excluded.provenance,
                        validated_at = CURRENT_TIMESTAMP,
                        is_locked = CASE WHEN context_addressing_rules.is_locked = 1 THEN 1 ELSE excluded.is_locked END,
                        last_chunk_index = excluded.last_chunk_index,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    translation_id, speaker_name, addressee_name,
                    self_pronoun, target_pronoun, vocative or "", register or "polite",
                    json.dumps(social_basis or [], ensure_ascii=False),
                    notes or "",
                    scope or "durable", int(contract_version or 1),
                    confidence, is_locked, validation_status or "active",
                    validation_reason or "", provenance or "unknown", chunk_index
                ))
                self._commit_connection(conn)
                return True
            except Exception as e:
                print(f"Error upserting addressing rule: {e}")
                return False

    def get_addressing_rules(
        self,
        translation_id: str,
        validation_status: Optional[str] = "active",
    ) -> List[Dict[str, Any]]:
        """Fetch all addressing rules for a given translation job."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, speaker_name, addressee_name, self_pronoun,
                           target_pronoun, vocative, register, social_basis, notes,
                           scope, contract_version, confidence, is_locked,
                           validation_status, validation_reason, provenance,
                           validated_at, last_chunk_index, updated_at
                    FROM context_addressing_rules
                    WHERE translation_id = ?
                      AND (? IS NULL OR validation_status = ?)
                    ORDER BY speaker_name, addressee_name
                """, (translation_id, validation_status, validation_status))
                rules = []
                for row in cursor.fetchall():
                    rule = dict(row)
                    try:
                        rule["social_basis"] = json.loads(
                            rule.get("social_basis") or "[]"
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        rule["social_basis"] = []
                    evidence = self.get_addressing_evidence(
                        translation_id,
                        rule["speaker_name"],
                        rule["addressee_name"],
                    )
                    rule["source_forms"] = []
                    seen_forms = set()
                    for item in evidence:
                        form = str(item.get("source_form") or "").strip()
                        key = form.casefold()
                        if form and key not in seen_forms:
                            seen_forms.add(key)
                            rule["source_forms"].append(form)
                    rule["evidence"] = evidence
                    rules.append(rule)
                return rules
            except Exception as e:
                print(f"Error getting addressing rules: {e}")
                return []

    def set_addressing_rule_validation(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
        status: str,
        reason: str = "",
    ) -> bool:
        """Activate or quarantine one directed addressing rule."""
        normalized = (
            status
            if status in {"active", "provisional", "quarantined"}
            else "quarantined"
        )
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    UPDATE context_addressing_rules
                    SET validation_status = ?, validation_reason = ?,
                        validated_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ? AND speaker_name = ?
                      AND addressee_name = ?
                """, (
                    normalized, reason or "", translation_id,
                    speaker_name, addressee_name,
                ))
                self._commit_connection(conn)
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error setting addressing rule validation: {e}")
                return False

    def set_addressing_rule_lock(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
        is_locked: bool
    ) -> bool:
        """Lock or unlock an addressing rule from being overwritten by LLM deltas."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE context_addressing_rules
                    SET is_locked = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ? AND speaker_name = ? AND addressee_name = ?
                """, (1 if is_locked else 0, translation_id, speaker_name, addressee_name))
                self._commit_connection(conn)
                return True
            except Exception as e:
                print(f"Error setting addressing rule lock: {e}")
                return False

    def add_addressing_evidence(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
        source_form: str,
        *,
        usage: str = "direct_address",
        source_language: str = "",
        evidence_quote: str = "",
        scope: str = "durable",
        confidence: float = 0.5,
        provenance: str = "unknown",
        dialogue_turn_id: str = "",
        chunk_index: int = 0,
    ) -> bool:
        """Store one unique source-form observation for an addressing pair."""

        if not str(source_form or "").strip():
            return False
        fingerprint = _evidence_fingerprint(
            translation_id, speaker_name, addressee_name, source_form,
            usage, scope, evidence_quote,
        )
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    INSERT INTO context_addressing_evidence (
                        translation_id, speaker_name, addressee_name,
                        source_form, usage, source_language, evidence_quote,
                        scope, confidence, provenance, dialogue_turn_id,
                        chunk_index, fingerprint, observation_count,
                        resolution_status, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'open', CURRENT_TIMESTAMP)
                    ON CONFLICT(
                        translation_id, speaker_name, addressee_name,
                        source_form, usage, scope, evidence_quote
                    ) DO UPDATE SET
                        observation_count = observation_count + 1,
                        last_seen_at = CURRENT_TIMESTAMP,
                        confidence = MAX(confidence, excluded.confidence),
                        fingerprint = excluded.fingerprint
                """, (
                    translation_id, speaker_name, addressee_name,
                    str(source_form).strip(), usage or "direct_address",
                    source_language or "", evidence_quote or "",
                    scope or "durable", confidence, provenance or "unknown",
                    dialogue_turn_id or "", chunk_index, fingerprint,
                ))
                self._commit_connection(conn)
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error adding addressing evidence: {e}")
                return False

    def get_addressing_evidence(
        self,
        translation_id: str,
        speaker_name: Optional[str] = None,
        addressee_name: Optional[str] = None,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Return source-form evidence, optionally limited to one pair."""

        with self._lock:
            try:
                conn = self._get_connection()
                if speaker_name is not None and addressee_name is not None:
                    rows = conn.execute("""
                        SELECT * FROM context_addressing_evidence
                        WHERE translation_id = ? AND speaker_name = ?
                          AND addressee_name = ?
                        ORDER BY confidence DESC, id ASC LIMIT ?
                    """, (
                        translation_id, speaker_name, addressee_name, limit,
                    )).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM context_addressing_evidence
                        WHERE translation_id = ? ORDER BY id DESC LIMIT ?
                    """, (translation_id, limit)).fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                print(f"Error getting addressing evidence: {e}")
                return []

    def resolve_addressing_evidence(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
        resolution_status: str,
    ) -> int:
        """Mark retained observations after pair-level reconciliation."""

        status = str(resolution_status or "open").casefold().strip()
        if status not in {"open", "promoted", "rejected", "superseded"}:
            status = "open"
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                UPDATE context_addressing_evidence
                SET resolution_status = ?,
                    resolved_at = CASE WHEN ? = 'open' THEN NULL ELSE CURRENT_TIMESTAMP END,
                    last_seen_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND speaker_name = ? AND addressee_name = ?
            """, (
                status, status, translation_id, speaker_name, addressee_name,
            ))
            self._commit_connection(conn)
            return int(cursor.rowcount or 0)

    def delete_addressing_rule(
        self,
        translation_id: str,
        speaker_name: str,
        addressee_name: str,
    ) -> bool:
        """Delete a directed addressing rule for a speaker-addressee pair."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM context_addressing_rules
                    WHERE translation_id = ? AND speaker_name = ? AND addressee_name = ?
                """, (translation_id, speaker_name, addressee_name))
                deleted = cursor.rowcount > 0
                if deleted:
                    cursor.execute("""
                        DELETE FROM context_addressing_evidence
                        WHERE translation_id = ? AND speaker_name = ?
                          AND addressee_name = ?
                    """, (translation_id, speaker_name, addressee_name))
                self._commit_connection(conn)
                return deleted
            except Exception as e:
                print(f"Error deleting addressing rule: {e}")
                return False

    def add_context_audit_log(
        self,
        translation_id: str,
        chunk_index: int,
        speaker_name: str,
        addressee_name: str,
        old_state: Optional[Dict[str, Any]],
        new_state: Dict[str, Any],
        trigger_source: str,
        evidence_quote: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """Add an audit log entry for context addressing updates."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO context_audit_logs (
                        translation_id, chunk_index, speaker_name, addressee_name,
                        old_state_json, new_state_json, trigger_source, evidence_quote, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    translation_id, chunk_index, speaker_name, addressee_name,
                    json.dumps(old_state) if old_state else None,
                    json.dumps(new_state),
                    trigger_source, evidence_quote or "", confidence or 1.0
                ))
                self._commit_connection(conn)
                return True
            except Exception as e:
                print(f"Error adding context audit log: {e}")
                return False

    def get_context_audit_logs(self, translation_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch audit log entries for a given translation job."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, chunk_index, speaker_name, addressee_name,
                           old_state_json, new_state_json, trigger_source,
                           evidence_quote, confidence, timestamp
                    FROM context_audit_logs
                    WHERE translation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (translation_id, limit))
                rows = cursor.fetchall()
                result = []
                for r in rows:
                    item = dict(r)
                    if item.get('old_state_json'):
                        try:
                            item['old_state'] = json.loads(item['old_state_json'])
                        except Exception:
                            item['old_state'] = None
                    else:
                        item['old_state'] = None
                    if item.get('new_state_json'):
                        try:
                            item['new_state'] = json.loads(item['new_state_json'])
                        except Exception:
                            item['new_state'] = None
                    else:
                        item['new_state'] = None
                    result.append(item)
                return result
            except Exception as e:
                print(f"Error fetching context audit logs: {e}")
                return []

    def upsert_relationship_node(
        self,
        translation_id: str,
        canonical_name: str,
        normalized_name: str,
        aliases: Optional[List[str]] = None,
        entity_type: str = "character",
        gender: str = "unknown",
        is_locked: int = 0,
    ) -> Optional[int]:
        """Create or update a relationship graph node and return its id."""

        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, canonical_name, aliases, entity_type, gender,
                           is_locked
                    FROM context_relationship_nodes
                    WHERE translation_id = ? AND normalized_name = ?
                """, (translation_id, normalized_name))
                existing = cursor.fetchone()
                merged_aliases = []
                if existing and existing["aliases"]:
                    try:
                        merged_aliases.extend(json.loads(existing["aliases"]) or [])
                    except (TypeError, ValueError, json.JSONDecodeError):
                        # The stored aliases are unreadable, so this upsert is
                        # about to overwrite them with only the incoming set.
                        # Losing established aliases silently would make a
                        # character stop being recognized mid-book.
                        logger.warning(
                            "Discarding unreadable stored aliases for "
                            "relationship node '%s' in job %s; keeping only the "
                            "%d incoming alias(es).",
                            normalized_name,
                            translation_id,
                            len(aliases or []),
                        )
                merged_aliases.extend(aliases or [])
                seen = set()
                merged_aliases = [
                    alias for alias in merged_aliases
                    if alias and not (
                        str(alias).casefold() in seen
                        or seen.add(str(alias).casefold())
                    )
                ]

                if existing:
                    locked = bool(existing["is_locked"])
                    cursor.execute("""
                        UPDATE context_relationship_nodes
                        SET canonical_name = ?, aliases = ?, entity_type = ?,
                            gender = ?, is_locked = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (
                        existing["canonical_name"] if locked else canonical_name,
                        json.dumps(merged_aliases, ensure_ascii=False),
                        existing["entity_type"] if locked else entity_type,
                        existing["gender"] if locked else (gender or "unknown"),
                        1 if locked else int(bool(is_locked)),
                        existing["id"],
                    ))
                    node_id = int(existing["id"])
                else:
                    cursor.execute("""
                        INSERT INTO context_relationship_nodes (
                            translation_id, canonical_name, normalized_name,
                            aliases, entity_type, gender, is_locked
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        translation_id,
                        canonical_name,
                        normalized_name,
                        json.dumps(merged_aliases, ensure_ascii=False),
                        entity_type,
                        gender or "unknown",
                        int(bool(is_locked)),
                    ))
                    node_id = int(cursor.lastrowid)
                self._commit_connection(conn)
                return node_id
            except Exception as e:
                print(f"Error upserting relationship node: {e}")
                return None

    def get_relationship_nodes(self, translation_id: str) -> List[Dict[str, Any]]:
        """Return all relationship graph nodes for a translation job."""

        with self._lock:
            try:
                conn = self._get_connection()
                rows = conn.execute("""
                    SELECT id, canonical_name, normalized_name, aliases,
                           entity_type, gender, is_locked, created_at,
                           updated_at
                    FROM context_relationship_nodes
                    WHERE translation_id = ?
                    ORDER BY canonical_name
                """, (translation_id,)).fetchall()
                result = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["aliases"] = json.loads(item.get("aliases") or "[]")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        item["aliases"] = []
                    result.append(item)
                return result
            except Exception as e:
                print(f"Error getting relationship nodes: {e}")
                return []

    def get_relationship_node_by_name(
        self,
        translation_id: str,
        normalized_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a node by an exact normalized canonical name or alias."""

        wanted = str(normalized_name or "").casefold().strip()
        if not wanted:
            return None
        nodes = self.get_relationship_nodes(translation_id)
        for node in nodes:
            if str(node.get("normalized_name") or "").casefold() == wanted:
                return node
            if any(str(alias).casefold().strip() == wanted for alias in node.get("aliases") or []):
                return node

        # Same name written another way is still the same character. A book
        # whose cast is named surname-first is quoted both ways, and a nickname
        # is hyphenated as often as not, so exact matching alone registered
        # "Momozawa Tomio" and "Guri-ko" as strangers to the nodes already
        # holding "Tomio Momozawa" and "Guriko" -- splitting one character's
        # relationships, gender, and addressing rules across two identities.
        # Both fallbacks need every token of the name, so a partial reference
        # such as "Momozawa" still resolves to nobody.
        wanted_tokens = sorted(wanted.split())
        wanted_tight = _tighten_name(wanted)
        for node in nodes:
            labels = [node.get("normalized_name"), *(node.get("aliases") or [])]
            for label in labels:
                key = str(label or "").casefold().strip()
                if not key:
                    continue
                if len(wanted_tokens) > 1 and sorted(key.split()) == wanted_tokens:
                    return node
                if _tighten_name(key) == wanted_tight:
                    return node
        return None

    def add_relationship_node_alias(
        self,
        translation_id: str,
        normalized_name: str,
        alias: str,
    ) -> bool:
        """Attach an exact alias to an existing unlocked character node."""

        node = self.get_relationship_node_by_name(translation_id, normalized_name)
        if not node or node.get("is_locked"):
            return False
        aliases = list(node.get("aliases") or [])
        if str(alias).casefold() not in {str(item).casefold() for item in aliases}:
            aliases.append(alias)
        return self.upsert_relationship_node(
            translation_id=translation_id,
            canonical_name=node["canonical_name"],
            normalized_name=node["normalized_name"],
            aliases=aliases,
            entity_type=node.get("entity_type") or "character",
            gender=node.get("gender") or "unknown",
            is_locked=node.get("is_locked", 0),
        ) is not None

    def upsert_relationship_edge(
        self,
        translation_id: str,
        source_node_id: int,
        target_node_id: int,
        relationship_type: str,
        direction: str,
        scope: str,
        hierarchy: str,
        intimacy: str,
        register: str,
        confidence: float,
        status: str,
        is_locked: int,
        chunk_index: int,
        provenance: str,
        details: str = "",
        relative_age: str = "unknown",
        rank_relation: str = "unknown",
        evidence_tier: str = "unknown",
        reason_code: str = "",
        supporting_units: int = 0,
        match_kind: str = "",
        validator_version: int = 2,
    ) -> Optional[int]:
        """Create or update a relationship graph edge and return its id."""

        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO context_relationship_edges (
                        translation_id, source_node_id, target_node_id,
                        relationship_type, direction, scope, hierarchy,
                        relative_age, rank_relation, intimacy, register,
                        confidence, status, is_locked,
                        last_chunk_index, provenance, details, evidence_tier,
                        reason_code, supporting_units, match_kind, validator_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(
                        translation_id, source_node_id, target_node_id,
                        relationship_type, scope
                    ) DO UPDATE SET
                        direction = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.direction ELSE excluded.direction END,
                        hierarchy = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.hierarchy ELSE excluded.hierarchy END,
                        relative_age = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.relative_age ELSE excluded.relative_age END,
                        rank_relation = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.rank_relation ELSE excluded.rank_relation END,
                        intimacy = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.intimacy ELSE excluded.intimacy END,
                        register = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.register ELSE excluded.register END,
                        confidence = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.confidence ELSE excluded.confidence END,
                        status = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.status ELSE excluded.status END,
                        is_locked = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN 1 ELSE excluded.is_locked END,
                        last_chunk_index = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.last_chunk_index ELSE excluded.last_chunk_index END,
                        provenance = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.provenance ELSE excluded.provenance END,
                        details = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.details ELSE excluded.details END,
                        evidence_tier = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.evidence_tier ELSE excluded.evidence_tier END,
                        reason_code = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.reason_code ELSE excluded.reason_code END,
                        supporting_units = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.supporting_units ELSE excluded.supporting_units END,
                        match_kind = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.match_kind ELSE excluded.match_kind END,
                        validator_version = CASE WHEN context_relationship_edges.is_locked = 1
                            THEN context_relationship_edges.validator_version ELSE excluded.validator_version END,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    translation_id, source_node_id, target_node_id,
                    relationship_type, direction, scope, hierarchy,
                    relative_age, rank_relation, intimacy, register, confidence,
                    status, int(bool(is_locked)), chunk_index, provenance, details,
                    evidence_tier, reason_code, max(0, int(supporting_units or 0)),
                    match_kind, int(validator_version or 2),
                ))
                cursor.execute("""
                    SELECT id FROM context_relationship_edges
                    WHERE translation_id = ? AND source_node_id = ?
                      AND target_node_id = ? AND relationship_type = ? AND scope = ?
                """, (
                    translation_id, source_node_id, target_node_id,
                    relationship_type, scope,
                ))
                row = cursor.fetchone()
                edge_id = int(row["id"]) if row else None
                if edge_id is not None:
                    self._record_relationship_edge_state(
                        conn, translation_id, edge_id, chunk_index
                    )
                self._commit_connection(conn)
                return edge_id
            except Exception as e:
                print(f"Error upserting relationship edge: {e}")
                return None

    _HISTORY_FIELDS = (
        "relationship_type", "direction", "scope", "hierarchy",
        "intimacy", "register", "status", "details",
    )

    def _record_relationship_edge_state(
        self,
        conn: sqlite3.Connection,
        translation_id: str,
        edge_id: int,
        chunk_index: int,
    ) -> None:
        """Append this edge's state to its history when the state has changed.

        Only a change is recorded: a relationship reasserted unchanged for two
        hundred chunks is one row, so the history stays the shape of the story
        rather than the shape of the run.
        """
        current = conn.execute(
            "SELECT {} FROM context_relationship_edges WHERE id = ?".format(
                ", ".join(self._HISTORY_FIELDS)
            ),
            (edge_id,),
        ).fetchone()
        if current is None:
            return
        state = {field: current[field] for field in self._HISTORY_FIELDS}
        chunk = max(0, int(chunk_index or 0))
        latest = conn.execute(
            "SELECT {} FROM context_relationship_edge_history "
            "WHERE translation_id = ? AND edge_id = ? AND from_chunk_index <= ? "
            "ORDER BY from_chunk_index DESC LIMIT 1".format(
                ", ".join(self._HISTORY_FIELDS)
            ),
            (translation_id, edge_id, chunk),
        ).fetchone()
        if latest is not None and all(
            latest[field] == state[field] for field in self._HISTORY_FIELDS
        ):
            return
        columns = ["translation_id", "edge_id", "from_chunk_index", *self._HISTORY_FIELDS]
        conn.execute(
            "INSERT OR REPLACE INTO context_relationship_edge_history ({}) "
            "VALUES ({})".format(
                ", ".join(columns), ", ".join("?" for _ in columns)
            ),
            (translation_id, edge_id, chunk, *state.values()),
        )

    def get_relationship_edge_history(
        self,
        translation_id: str,
        edge_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return recorded edge states, oldest first."""

        with self._lock:
            try:
                conn = self._get_connection()
                query = (
                    "SELECT * FROM context_relationship_edge_history "
                    "WHERE translation_id = ?"
                )
                params: List[Any] = [translation_id]
                if edge_id is not None:
                    query += " AND edge_id = ?"
                    params.append(edge_id)
                query += " ORDER BY edge_id, from_chunk_index"
                return [
                    dict(row) for row in conn.execute(query, tuple(params)).fetchall()
                ]
            except Exception as exc:
                logger.warning("Could not read relationship history: %s", exc)
                return []

    def get_relationship_edges_as_of(
        self,
        translation_id: str,
        chunk_index: int,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return each edge as it stood at `chunk_index`, with node names.

        An edge with no recorded state at or before that chunk did not exist
        yet, so it is left out rather than back-dated.
        """
        edges = {
            int(edge["id"]): edge
            for edge in self.get_relationship_edges(translation_id)
            if edge.get("id") is not None
        }
        if not edges:
            return []
        wanted = {status for status in statuses} if statuses else None
        chunk = max(0, int(chunk_index or 0))
        result: List[Dict[str, Any]] = []
        with self._lock:
            try:
                conn = self._get_connection()
                rows = conn.execute(
                    "SELECT h.* FROM context_relationship_edge_history h "
                    "JOIN (SELECT edge_id, MAX(from_chunk_index) AS latest "
                    "      FROM context_relationship_edge_history "
                    "      WHERE translation_id = ? AND from_chunk_index <= ? "
                    "      GROUP BY edge_id) newest "
                    "  ON newest.edge_id = h.edge_id "
                    " AND newest.latest = h.from_chunk_index "
                    "WHERE h.translation_id = ?",
                    (translation_id, chunk, translation_id),
                ).fetchall()
            except Exception as exc:
                logger.warning("Could not read relationships as of a chunk: %s", exc)
                return []
        for row in rows:
            edge = edges.get(int(row["edge_id"]))
            if edge is None:
                continue
            historical = dict(edge)
            for field in self._HISTORY_FIELDS:
                historical[field] = row[field]
            historical["as_of_chunk_index"] = int(row["from_chunk_index"])
            if wanted is not None and historical.get("status") not in wanted:
                continue
            result.append(historical)
        result.sort(key=lambda item: (item.get("source_name") or "", item.get("target_name") or ""))
        return result

    def claim_reasoning_migration(
        self, translation_id: str, migration_key: str,
    ) -> bool:
        """Claim one durable per-job reasoning migration, retrying failures."""

        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                INSERT OR IGNORE INTO context_reasoning_migrations (
                    translation_id, migration_key, status
                ) VALUES (?, ?, 'running')
            """, (translation_id, migration_key))
            claimed = cursor.rowcount > 0
            if not claimed:
                cursor = conn.execute("""
                    UPDATE context_reasoning_migrations
                    SET status = 'running', details = NULL,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ? AND migration_key = ?
                      AND status = 'failed'
                """, (translation_id, migration_key))
                claimed = cursor.rowcount > 0
            self._commit_connection(conn)
            return claimed

    def finish_reasoning_migration(
        self, translation_id: str, migration_key: str, status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                UPDATE context_reasoning_migrations
                SET status = ?, details = ?, updated_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND migration_key = ?
            """, (
                status, json.dumps(details or {}, ensure_ascii=False),
                translation_id, migration_key,
            ))
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def get_relationship_edges(
        self,
        translation_id: str,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return relationship edges joined to canonical node names."""

        with self._lock:
            try:
                conn = self._get_connection()
                query = """
                    SELECT e.*, source.canonical_name AS source_name,
                           source.normalized_name AS source_normalized_name,
                           source.entity_type AS source_entity_type,
                           source.gender AS source_gender,
                           target.canonical_name AS target_name,
                           target.normalized_name AS target_normalized_name,
                           target.entity_type AS target_entity_type,
                           target.gender AS target_gender
                    FROM context_relationship_edges e
                    JOIN context_relationship_nodes source ON source.id = e.source_node_id
                    JOIN context_relationship_nodes target ON target.id = e.target_node_id
                    WHERE e.translation_id = ?
                """
                params: List[Any] = [translation_id]
                if statuses:
                    query += " AND e.status IN ({})".format(
                        ",".join("?" for _ in statuses)
                    )
                    params.extend(statuses)
                query += " ORDER BY e.source_node_id, e.target_node_id, e.relationship_type"
                return [dict(row) for row in conn.execute(query, params).fetchall()]
            except Exception as e:
                print(f"Error getting relationship edges: {e}")
                return []

    def get_relationship_edges_for_pair(
        self,
        translation_id: str,
        source_normalized_name: str,
        target_normalized_name: str,
        statuses: Optional[List[str]] = None,
        include_reverse: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return exact node-pair edges, optionally including the reverse pair."""

        source = self.get_relationship_node_by_name(translation_id, source_normalized_name)
        target = self.get_relationship_node_by_name(translation_id, target_normalized_name)
        if not source or not target:
            return []
        source_id = source["id"]
        target_id = target["id"]
        result = []
        for edge in self.get_relationship_edges(translation_id, statuses=statuses):
            direct = edge["source_node_id"] == source_id and edge["target_node_id"] == target_id
            reverse = edge["source_node_id"] == target_id and edge["target_node_id"] == source_id
            if direct or (include_reverse and reverse):
                result.append(edge)
        return result

    def set_relationship_edge_lock(
        self,
        translation_id: str,
        edge_id: int,
        is_locked: bool,
    ) -> bool:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    UPDATE context_relationship_edges
                    SET is_locked = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ? AND id = ?
                """, (int(bool(is_locked)), translation_id, edge_id))
                self._commit_connection(conn)
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error setting relationship edge lock: {e}")
                return False

    def set_relationship_edge_status(
        self,
        translation_id: str,
        edge_id: int,
        status: str,
    ) -> bool:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    UPDATE context_relationship_edges
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ? AND id = ? AND is_locked = 0
                """, (status, translation_id, edge_id))
                self._commit_connection(conn)
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error setting relationship edge status: {e}")
                return False

    def delete_relationship_edge(self, translation_id: str, edge_id: int) -> bool:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    DELETE FROM context_relationship_edges
                    WHERE translation_id = ? AND id = ? AND is_locked = 0
                """, (translation_id, edge_id))
                self._commit_connection(conn)
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error deleting relationship edge: {e}")
                return False

    def add_relationship_evidence(
        self,
        translation_id: str,
        edge_id: Optional[int],
        chunk_index: int,
        evidence_quote: str,
        provenance: str,
        parser_status: str,
        confidence: float,
        file_id: str = "",
        dialogue_turn_id: str = "",
        match_kind: str = "",
        source_start: Optional[int] = None,
        source_end: Optional[int] = None,
    ) -> Optional[int]:
        fingerprint = _evidence_fingerprint(
            translation_id, edge_id, chunk_index, dialogue_turn_id,
            evidence_quote,
        )
        with self._lock:
            try:
                conn = self._get_connection()
                existing = conn.execute("""
                    SELECT id FROM context_relationship_evidence
                    WHERE translation_id = ? AND fingerprint = ?
                    ORDER BY id ASC LIMIT 1
                """, (translation_id, fingerprint)).fetchone()
                if existing:
                    conn.execute("""
                        UPDATE context_relationship_evidence
                        SET observation_count = observation_count + 1,
                            last_seen_at = CURRENT_TIMESTAMP,
                            confidence = MAX(confidence, ?)
                        WHERE id = ?
                    """, (confidence, existing["id"]))
                    self._commit_connection(conn)
                    return int(existing["id"])
                cursor = conn.execute("""
                    INSERT INTO context_relationship_evidence (
                        translation_id, edge_id, chunk_index, file_id,
                        dialogue_turn_id, evidence_quote, provenance,
                        parser_status, confidence, match_kind, source_start,
                        source_end, fingerprint, observation_count,
                        resolution_status, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'open', CURRENT_TIMESTAMP)
                """, (
                    translation_id, edge_id, chunk_index, file_id,
                    dialogue_turn_id, evidence_quote, provenance,
                    parser_status, confidence, match_kind, source_start, source_end,
                    fingerprint,
                ))
                self._commit_connection(conn)
                return int(cursor.lastrowid)
            except Exception as e:
                print(f"Error adding relationship evidence: {e}")
                return None

    def resolve_relationship_evidence(
        self,
        translation_id: str,
        edge_id: int,
        resolution_status: str,
    ) -> int:
        status = str(resolution_status or "open").casefold().strip()
        if status not in {"open", "promoted", "rejected", "superseded"}:
            status = "open"
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                UPDATE context_relationship_evidence
                SET resolution_status = ?,
                    resolved_at = CASE WHEN ? = 'open' THEN NULL ELSE CURRENT_TIMESTAMP END,
                    last_seen_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND edge_id = ?
            """, (status, status, translation_id, edge_id))
            self._commit_connection(conn)
            return int(cursor.rowcount or 0)

    def get_relationship_evidence(
        self,
        translation_id: str,
        edge_id: Optional[int] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                conn = self._get_connection()
                if edge_id is None:
                    rows = conn.execute("""
                        SELECT * FROM context_relationship_evidence
                        WHERE translation_id = ? ORDER BY id DESC LIMIT ?
                    """, (translation_id, limit)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM context_relationship_evidence
                        WHERE translation_id = ? AND edge_id = ?
                        ORDER BY id DESC LIMIT ?
                    """, (translation_id, edge_id, limit)).fetchall()
                return [dict(row) for row in rows]
            except Exception as e:
                print(f"Error getting relationship evidence: {e}")
                return []

    def add_relationship_conflict(
        self,
        translation_id: str,
        source_name: str,
        target_name: str,
        severity: str,
        validator: str,
        reason: str,
        remediation_hint: str,
        candidate: Optional[Dict[str, Any]],
        chunk_index: int,
        edge_id: Optional[int] = None,
    ) -> Optional[int]:
        with self._lock:
            try:
                conn = self._get_connection()
                candidate_data = candidate or {}
                fingerprint = "|".join((
                    str(source_name or "").strip().casefold(),
                    str(target_name or "").strip().casefold(),
                    str(validator or "").strip().casefold(),
                    str(candidate_data.get("relationship_type") or "").strip().casefold(),
                    str(candidate_data.get("scope") or "durable").strip().casefold(),
                ))
                existing = conn.execute("""
                    SELECT id FROM context_relationship_conflicts
                    WHERE translation_id = ? AND status = 'open' AND (
                        fingerprint = ? OR (
                            fingerprint = ''
                            AND lower(source_name) = lower(?)
                            AND lower(target_name) = lower(?)
                            AND validator = ?
                        )
                    )
                    ORDER BY id DESC LIMIT 1
                """, (
                    translation_id, fingerprint, source_name, target_name, validator,
                )).fetchone()
                if existing:
                    conn.execute("""
                        UPDATE context_relationship_conflicts
                        SET edge_id = COALESCE(?, edge_id), severity = ?, reason = ?,
                            remediation_hint = ?, candidate_json = ?, chunk_index = ?,
                            fingerprint = ?
                        WHERE id = ?
                    """, (
                        edge_id, severity, reason, remediation_hint,
                        json.dumps(candidate_data, ensure_ascii=False), chunk_index,
                        fingerprint, int(existing["id"]),
                    ))
                    self._commit_connection(conn)
                    return int(existing["id"])
                cursor = conn.execute("""
                    INSERT INTO context_relationship_conflicts (
                        translation_id, edge_id, source_name, target_name,
                        severity, validator, status, reason,
                        remediation_hint, candidate_json, chunk_index, fingerprint
                    ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)
                """, (
                    translation_id, edge_id, source_name, target_name,
                    severity, validator, reason, remediation_hint,
                    json.dumps(candidate_data, ensure_ascii=False), chunk_index,
                    fingerprint,
                ))
                self._commit_connection(conn)
                return int(cursor.lastrowid)
            except Exception as e:
                print(f"Error adding relationship conflict: {e}")
                return None

    def get_relationship_conflicts(
        self,
        translation_id: str,
        status: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            try:
                conn = self._get_connection()
                if status:
                    rows = conn.execute("""
                        SELECT * FROM context_relationship_conflicts
                        WHERE translation_id = ? AND status = ?
                        ORDER BY id DESC LIMIT ?
                    """, (translation_id, status, limit)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM context_relationship_conflicts
                        WHERE translation_id = ? ORDER BY id DESC LIMIT ?
                    """, (translation_id, limit)).fetchall()
                result = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["candidate"] = json.loads(item.get("candidate_json") or "{}")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        item["candidate"] = {}
                    result.append(item)
                return result
            except Exception as e:
                print(f"Error getting relationship conflicts: {e}")
                return []

    def resolve_relationship_conflict(
        self,
        translation_id: str,
        conflict_id: int,
    ) -> bool:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    UPDATE context_relationship_conflicts
                    SET status = 'resolved', resolved_at = CURRENT_TIMESTAMP
                    WHERE translation_id = ? AND id = ?
                """, (translation_id, conflict_id))
                self._commit_connection(conn)
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error resolving relationship conflict: {e}")
                return False

    def get_relationship_pair_audit(
        self,
        translation_id: str,
        source_normalized_name: str,
        target_normalized_name: str,
    ) -> Dict[str, Any]:
        """Return graph state, evidence, and conflicts for an exact pair."""

        edges = self.get_relationship_edges_for_pair(
            translation_id,
            source_normalized_name,
            target_normalized_name,
            include_reverse=True,
        )
        edge_ids = {edge["id"] for edge in edges}
        evidence = [
            item for item in self.get_relationship_evidence(translation_id)
            if item.get("edge_id") in edge_ids
        ]
        source_key = str(source_normalized_name or "").casefold()
        target_key = str(target_normalized_name or "").casefold()
        conflicts = [
            item for item in self.get_relationship_conflicts(translation_id)
            if {
                str(item.get("source_name") or "").casefold(),
                str(item.get("target_name") or "").casefold(),
            } == {source_key, target_key}
        ]
        source_node = self.get_relationship_node_by_name(
            translation_id, source_normalized_name
        )
        target_node = self.get_relationship_node_by_name(
            translation_id, target_normalized_name
        )
        derivations = (
            self.get_relationship_derivations(
                translation_id,
                source_name=source_node.get("canonical_name"),
                target_name=target_node.get("canonical_name"),
            )
            if source_node and target_node
            else []
        )
        return {
            "edges": edges,
            "evidence": evidence,
            "conflicts": conflicts,
            "derivations": derivations,
        }

    def upsert_relationship_derivation(
        self,
        translation_id: str,
        source_name: str,
        target_name: str,
        hierarchy: str,
        confidence: float,
        path: List[Dict[str, Any]],
        basis: str,
        *,
        status: str = "accepted",
        chunk_index: int = 0,
    ) -> Optional[int]:
        """Persist one explainable materialized seniority derivation."""

        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute("""
                    INSERT INTO context_relationship_derivations (
                        translation_id, source_name, target_name, hierarchy,
                        confidence, path_json, basis, status,
                        last_chunk_index, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(translation_id, source_name, target_name)
                    DO UPDATE SET hierarchy = excluded.hierarchy,
                        confidence = excluded.confidence,
                        path_json = excluded.path_json,
                        basis = excluded.basis,
                        status = excluded.status,
                        last_chunk_index = excluded.last_chunk_index,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    translation_id, source_name, target_name, hierarchy,
                    confidence, json.dumps(path, ensure_ascii=False), basis,
                    status, chunk_index,
                ))
                row = conn.execute("""
                    SELECT id FROM context_relationship_derivations
                    WHERE translation_id = ? AND source_name = ?
                      AND target_name = ?
                """, (translation_id, source_name, target_name)).fetchone()
                self._commit_connection(conn)
                return int(row["id"]) if row else None
            except Exception as e:
                print(f"Error upserting relationship derivation: {e}")
                return None

    def get_relationship_derivations(
        self,
        translation_id: str,
        source_name: Optional[str] = None,
        target_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return materialized seniority derivations with decoded paths."""

        with self._lock:
            try:
                conn = self._get_connection()
                if source_name is not None and target_name is not None:
                    rows = conn.execute("""
                        SELECT * FROM context_relationship_derivations
                        WHERE translation_id = ? AND source_name = ?
                          AND target_name = ? ORDER BY id
                    """, (translation_id, source_name, target_name)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT * FROM context_relationship_derivations
                        WHERE translation_id = ? ORDER BY source_name, target_name
                    """, (translation_id,)).fetchall()
                result = []
                for row in rows:
                    item = dict(row)
                    try:
                        item["path"] = json.loads(item.get("path_json") or "[]")
                    except (TypeError, ValueError, json.JSONDecodeError):
                        item["path"] = []
                    result.append(item)
                return result
            except Exception as e:
                print(f"Error getting relationship derivations: {e}")
                return []

    def create_context_resync_run(
        self,
        run_id: str,
        translation_id: str,
        start_chunk_index: int,
        initial_snapshot: str,
    ) -> bool:
        """Create or reset a persisted context-resync staging run."""

        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute("DELETE FROM context_resync_chunk_stage WHERE run_id = ?", (run_id,))
                conn.execute("""
                    INSERT INTO context_resync_runs (
                        run_id, translation_id, start_chunk_index, status,
                        initial_snapshot, last_processed_chunk, updated_at
                    ) VALUES (?, ?, ?, 'staging', ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status = 'staging', initial_snapshot = excluded.initial_snapshot,
                        final_context = NULL, error = NULL,
                        last_processed_chunk = excluded.last_processed_chunk,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    run_id, translation_id, start_chunk_index,
                    initial_snapshot, start_chunk_index,
                ))
                self._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error creating context resync run: {exc}")
                return False

    def stage_context_resync_chunk(
        self,
        run_id: str,
        translation_id: str,
        chunk: Dict[str, Any],
        *,
        relationship_candidates: Optional[List[Dict[str, Any]]] = None,
        addressing_candidates: Optional[List[Dict[str, Any]]] = None,
        parser_status: str = "absent",
    ) -> bool:
        """Persist one replayed chunk without changing the live timeline."""

        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute("""
                    INSERT INTO context_resync_chunk_stage (
                        run_id, translation_id, chunk_index, original_text,
                        translated_text, chunk_data, status,
                        relationship_candidates, addressing_candidates,
                        parser_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, chunk_index) DO UPDATE SET
                        original_text = excluded.original_text,
                        translated_text = excluded.translated_text,
                        chunk_data = excluded.chunk_data,
                        status = excluded.status,
                        relationship_candidates = excluded.relationship_candidates,
                        addressing_candidates = excluded.addressing_candidates,
                        parser_status = excluded.parser_status
                """, (
                    run_id, translation_id, int(chunk.get("chunk_index", -1)),
                    chunk.get("original_text"), chunk.get("translated_text"),
                    json.dumps(chunk.get("chunk_data") or {}, ensure_ascii=False),
                    chunk.get("status") or "completed",
                    json.dumps(relationship_candidates or [], ensure_ascii=False),
                    json.dumps(addressing_candidates or [], ensure_ascii=False),
                    parser_status or "absent",
                ))
                conn.execute("""
                    UPDATE context_resync_runs SET last_processed_chunk = ?,
                        updated_at = CURRENT_TIMESTAMP WHERE run_id = ?
                """, (int(chunk.get("chunk_index", -1)), run_id))
                self._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error staging context resync chunk: {exc}")
                return False

    def get_context_resync_stage(self, run_id: str) -> List[Dict[str, Any]]:
        """Return decoded staged chunks for one resync run."""

        with self._lock:
            conn = self._get_connection()
            rows = conn.execute("""
                SELECT * FROM context_resync_chunk_stage
                WHERE run_id = ? ORDER BY chunk_index
            """, (run_id,)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for source_key, target_key, fallback in (
                    ("chunk_data", "chunk_data", {}),
                    ("relationship_candidates", "relationship_candidates", []),
                    ("addressing_candidates", "addressing_candidates", []),
                ):
                    try:
                        item[target_key] = json.loads(item.get(source_key) or "null") or fallback
                    except (TypeError, ValueError, json.JSONDecodeError):
                        item[target_key] = fallback
                result.append(item)
            return result

    def finish_context_resync_run(
        self,
        run_id: str,
        status: str,
        *,
        final_context: str = "",
        error: str = "",
    ) -> bool:
        """Record the terminal or resumable state of a resync run."""

        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                UPDATE context_resync_runs SET status = ?, final_context = ?,
                    error = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?
            """, (status, final_context or None, error or None, run_id))
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def get_latest_context_resync_run(
        self,
        translation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the latest persisted staging/activation record for a job."""

        with self._lock:
            row = self._get_connection().execute("""
                SELECT run_id, translation_id, start_chunk_index, status,
                       last_processed_chunk, error, created_at, updated_at
                FROM context_resync_runs WHERE translation_id = ?
                ORDER BY created_at DESC, rowid DESC LIMIT 1
            """, (translation_id,)).fetchone()
            return dict(row) if row else None

    def backup_to(self, destination: str) -> str:
        """Create a consistent SQLite backup while the live database is open."""

        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            source_conn = self._get_connection()
            backup_conn = sqlite3.connect(str(target))
            try:
                source_conn.backup(backup_conn)
            finally:
                backup_conn.close()
        return str(target)

    def upsert_narrator_voice_profile(
        self, translation_id: str, profile: Dict[str, Any],
        *, expected_revision: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create or update one timeline profile using optimistic locking."""

        narrator_key = str(profile.get("narrator_key") or "default").strip()
        start_chunk = int(profile.get("start_chunk_index", 0) or 0)
        with self._lock:
            conn = self._get_connection()
            existing = conn.execute("""
                SELECT id, revision, is_locked FROM context_narrator_profiles
                WHERE translation_id = ? AND narrator_key = ?
                  AND start_chunk_index = ?
            """, (translation_id, narrator_key, start_chunk)).fetchone()
            if existing and expected_revision is not None:
                if int(existing["revision"]) != int(expected_revision):
                    return None
            # Model observations cannot overwrite a user lock.
            provenance = str(profile.get("provenance") or "unknown")
            if existing and existing["is_locked"] and provenance != "user_manual":
                return dict(existing)
            values = (
                translation_id, narrator_key,
                str(profile.get("narrator_identity") or "unknown"),
                str(profile.get("point_of_view") or "unknown"),
                str(profile.get("self_reference") or ""),
                str(profile.get("formality") or "neutral"),
                str(profile.get("speech_level") or ""),
                str(profile.get("gender") or "unknown"),
                str(profile.get("number") or "singular"),
                str(profile.get("dialect") or ""),
                str(profile.get("tense") or ""),
                json.dumps(profile.get("stylistic_markers") or [], ensure_ascii=False),
                json.dumps(profile.get("dimensions") or {}, ensure_ascii=False),
                max(0.0, min(1.0, float(profile.get("confidence", 0.0) or 0.0))),
                provenance, str(profile.get("scope") or "durable"), start_chunk,
                profile.get("end_chunk_index"),
                1 if profile.get("is_locked") else 0,
                str(profile.get("status") or "provisional"),
            )
            conn.execute("""
                INSERT INTO context_narrator_profiles (
                    translation_id, narrator_key, narrator_identity,
                    point_of_view, self_reference, formality, speech_level,
                    gender, number, dialect, tense, stylistic_markers,
                    dimensions, confidence, provenance, scope,
                    start_chunk_index, end_chunk_index, is_locked, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(translation_id, narrator_key, start_chunk_index)
                DO UPDATE SET
                    narrator_identity = excluded.narrator_identity,
                    point_of_view = excluded.point_of_view,
                    self_reference = excluded.self_reference,
                    formality = excluded.formality,
                    speech_level = excluded.speech_level,
                    gender = excluded.gender,
                    number = excluded.number,
                    dialect = excluded.dialect,
                    tense = excluded.tense,
                    stylistic_markers = excluded.stylistic_markers,
                    dimensions = excluded.dimensions,
                    confidence = excluded.confidence,
                    provenance = excluded.provenance,
                    scope = excluded.scope,
                    end_chunk_index = excluded.end_chunk_index,
                    is_locked = excluded.is_locked,
                    status = excluded.status,
                    revision = context_narrator_profiles.revision + 1,
                    updated_at = CURRENT_TIMESTAMP
            """, values)
            self._commit_connection(conn)
            row = conn.execute("""
                SELECT * FROM context_narrator_profiles
                WHERE translation_id = ? AND narrator_key = ?
                  AND start_chunk_index = ?
            """, (translation_id, narrator_key, start_chunk)).fetchone()
            return self._decode_narrator_profile(dict(row)) if row else None

    @staticmethod
    def _decode_narrator_profile(row: Dict[str, Any]) -> Dict[str, Any]:
        for key, fallback in (("stylistic_markers", []), ("dimensions", {})):
            try:
                row[key] = json.loads(row.get(key) or json.dumps(fallback))
            except (TypeError, ValueError, json.JSONDecodeError):
                row[key] = fallback
        row["is_locked"] = bool(row.get("is_locked"))
        return row

    def get_narrator_voice_profiles(
        self, translation_id: str, *, effective_chunk_index: Optional[int] = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return profiles, optionally as-of a historical chunk boundary."""

        clauses = ["translation_id = ?"]
        params: List[Any] = [translation_id]
        if not include_inactive:
            clauses.append("status = 'active'")
        if effective_chunk_index is not None:
            clauses.extend([
                "start_chunk_index <= ?",
                "(end_chunk_index IS NULL OR end_chunk_index >= ?)",
            ])
            params.extend([int(effective_chunk_index), int(effective_chunk_index)])
        with self._lock:
            rows = self._get_connection().execute(
                "SELECT * FROM context_narrator_profiles WHERE "
                + " AND ".join(clauses)
                + " ORDER BY is_locked DESC, start_chunk_index, narrator_key",
                tuple(params),
            ).fetchall()
            return [self._decode_narrator_profile(dict(row)) for row in rows]

    def get_narrator_voice_profile(
        self, translation_id: str, profile_id: int,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._get_connection().execute("""
                SELECT * FROM context_narrator_profiles
                WHERE translation_id = ? AND id = ?
            """, (translation_id, int(profile_id))).fetchone()
            return self._decode_narrator_profile(dict(row)) if row else None

    def delete_narrator_voice_profile(
        self, translation_id: str, profile_id: int,
        *, expected_revision: Optional[int] = None,
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute("""
                SELECT revision, is_locked FROM context_narrator_profiles
                WHERE translation_id = ? AND id = ?
            """, (translation_id, int(profile_id))).fetchone()
            if not row or row["is_locked"]:
                return False
            if expected_revision is not None and int(row["revision"]) != int(expected_revision):
                return False
            cursor = conn.execute("""
                DELETE FROM context_narrator_profiles
                WHERE translation_id = ? AND id = ?
            """, (translation_id, int(profile_id)))
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def set_narrator_voice_profile_lock(
        self, translation_id: str, profile_id: int, is_locked: bool,
        *, expected_revision: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            row = conn.execute("""
                SELECT revision FROM context_narrator_profiles
                WHERE translation_id = ? AND id = ?
            """, (translation_id, int(profile_id))).fetchone()
            if not row or (
                expected_revision is not None
                and int(row["revision"]) != int(expected_revision)
            ):
                return None
            conn.execute("""
                UPDATE context_narrator_profiles SET is_locked = ?,
                    provenance = CASE WHEN ? = 1 THEN 'user_manual' ELSE provenance END,
                    revision = revision + 1, updated_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND id = ?
            """, (1 if is_locked else 0, 1 if is_locked else 0, translation_id, int(profile_id)))
            self._commit_connection(conn)
            return self.get_narrator_voice_profile(translation_id, profile_id)

    def add_narrator_voice_observation(
        self, translation_id: str, chunk_index: int, observation: Dict[str, Any],
        *, chapter_index: Optional[int] = None, scene_key: str = "",
        profile_id: Optional[int] = None, provenance: str = "senior_editor",
        status: str = "accepted", rejection_reason: str = "",
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                INSERT INTO context_narrator_observations (
                    translation_id, profile_id, chunk_index, chapter_index,
                    scene_key, segment_id, discourse_mode, narrator_key,
                    narrator_identity, point_of_view, dimensions, source_quote,
                    target_quote, transition_type, transition_evidence,
                    confidence, provenance, status, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(translation_id, chunk_index, segment_id, narrator_key)
                DO UPDATE SET profile_id = excluded.profile_id,
                    dimensions = excluded.dimensions,
                    source_quote = excluded.source_quote,
                    target_quote = excluded.target_quote,
                    transition_type = excluded.transition_type,
                    transition_evidence = excluded.transition_evidence,
                    confidence = excluded.confidence,
                    provenance = excluded.provenance,
                    status = excluded.status,
                    rejection_reason = excluded.rejection_reason
            """, (
                translation_id, profile_id, int(chunk_index), chapter_index,
                scene_key or "", observation.get("segment_id") or "",
                observation.get("discourse_mode") or "narration",
                observation.get("narrator_key") or "default",
                observation.get("narrator_identity") or "unknown",
                observation.get("point_of_view") or "unknown",
                json.dumps(observation.get("dimensions") or {}, ensure_ascii=False),
                observation.get("source_quote") or "",
                observation.get("target_quote") or "",
                observation.get("transition_type") or "none",
                observation.get("transition_evidence") or "",
                float(observation.get("confidence", 0.0) or 0.0), provenance,
                status, rejection_reason,
            ))
            self._commit_connection(conn)
            return True

    def get_narrator_voice_timeline(self, translation_id: str) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            observations = [dict(row) for row in conn.execute("""
                SELECT * FROM context_narrator_observations
                WHERE translation_id = ? ORDER BY chunk_index, id
            """, (translation_id,)).fetchall()]
            transitions = [dict(row) for row in conn.execute("""
                SELECT * FROM context_narrator_transitions
                WHERE translation_id = ? ORDER BY chunk_index, id
            """, (translation_id,)).fetchall()]
            for item in observations:
                try:
                    item["dimensions"] = json.loads(item.get("dimensions") or "{}")
                except (TypeError, ValueError):
                    item["dimensions"] = {}
            return {"observations": observations, "transitions": transitions}

    def add_narrator_voice_conflict(
        self, translation_id: str, *, narrator_key: str, chunk_index: int,
        reason: str, candidate: Dict[str, Any], chapter_index: Optional[int] = None,
        scene_key: str = "",
    ) -> int:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                INSERT INTO context_narrator_conflicts (
                    translation_id, narrator_key, chunk_index, chapter_index,
                    scene_key, reason, candidate_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (translation_id, narrator_key or "default", int(chunk_index),
                  chapter_index, scene_key or "", reason,
                  json.dumps(candidate or {}, ensure_ascii=False)))
            self._commit_connection(conn)
            return int(cursor.lastrowid)

    def get_narrator_voice_conflicts(
        self, translation_id: str, *, status: str = "open",
    ) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._get_connection().execute("""
                SELECT * FROM context_narrator_conflicts
                WHERE translation_id = ? AND (? = 'all' OR status = ?)
                ORDER BY chunk_index, id
            """, (translation_id, status, status)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                try:
                    item["candidate"] = json.loads(item.pop("candidate_json") or "{}")
                except (TypeError, ValueError):
                    item["candidate"] = {}
                result.append(item)
            return result

    def resolve_narrator_voice_conflict(
        self, translation_id: str, conflict_id: int, resolution: str,
    ) -> bool:
        if resolution not in {"accepted_transition", "rejected_transition"}:
            return False
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                UPDATE context_narrator_conflicts SET status = 'resolved',
                    resolution = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND id = ? AND status = 'open'
            """, (resolution, translation_id, int(conflict_id)))
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def quarantine_narrator_voice_after(
        self, translation_id: str, chunk_index: int,
    ) -> None:
        """Quarantine unlocked inferred state downstream of a resync boundary."""

        with self._lock:
            conn = self._get_connection()
            conn.execute("""
                UPDATE context_narrator_observations SET status = 'quarantined'
                WHERE translation_id = ? AND chunk_index >= ?
            """, (translation_id, int(chunk_index)))
            conn.execute("""
                UPDATE context_narrator_profiles SET status = 'quarantined',
                    updated_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND start_chunk_index >= ?
                  AND is_locked = 0
            """, (translation_id, int(chunk_index)))
            conn.execute("""
                UPDATE context_narrator_transitions SET status = 'quarantined'
                WHERE translation_id = ? AND chunk_index >= ?
            """, (translation_id, int(chunk_index)))
            self._commit_connection(conn)

    def claim_narrator_bootstrap(
        self, translation_id: str, *, attempt_kind: str = "bootstrap",
        boundary_key: str = "", sampled_chunks: Optional[List[int]] = None,
    ) -> bool:
        """Claim a single durable bootstrap/transition-verification attempt."""

        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                INSERT OR IGNORE INTO context_narrator_bootstrap_attempts (
                    translation_id, attempt_kind, boundary_key, status,
                    sampled_chunks
                ) VALUES (?, ?, ?, 'running', ?)
            """, (translation_id, attempt_kind, boundary_key,
                  json.dumps(sampled_chunks or [])))
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def finish_narrator_bootstrap(
        self, translation_id: str, status: str, *, attempt_kind: str = "bootstrap",
        boundary_key: str = "", details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        with self._lock:
            conn = self._get_connection()
            cursor = conn.execute("""
                UPDATE context_narrator_bootstrap_attempts SET status = ?,
                    details = ?, updated_at = CURRENT_TIMESTAMP
                WHERE translation_id = ? AND attempt_kind = ? AND boundary_key = ?
            """, (status, json.dumps(details or {}, ensure_ascii=False),
                  translation_id, attempt_kind, boundary_key))
            self._commit_connection(conn)
            return cursor.rowcount > 0

    def get_narrator_bootstrap_attempts(
        self, translation_id: str, *, attempt_kind: str = "bootstrap",
    ) -> List[Dict[str, Any]]:
        """Return decoded bootstrap attempts for public diagnostics."""

        with self._lock:
            rows = self._get_connection().execute("""
                SELECT * FROM context_narrator_bootstrap_attempts
                WHERE translation_id = ? AND attempt_kind = ?
                ORDER BY created_at, boundary_key
            """, (translation_id, attempt_kind)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for key, fallback in (("sampled_chunks", []), ("details", {})):
                    try:
                        item[key] = json.loads(item.get(key) or json.dumps(fallback))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        item[key] = fallback
                result.append(item)
            return result

    def mark_narrator_voice_chunks_stale(
        self, translation_id: str, start_chunk_index: int,
        *, end_chunk_index: Optional[int] = None,
    ) -> int:
        """Mark completed historical chunks for an explicit Editor retry."""

        changed = 0
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute("""
                SELECT chunk_index, chunk_data FROM checkpoint_chunks
                WHERE translation_id = ? AND chunk_index >= ?
                  AND (? IS NULL OR chunk_index <= ?)
                  AND status IN ('completed', 'partial')
            """, (
                translation_id, int(start_chunk_index), end_chunk_index,
                int(end_chunk_index) if end_chunk_index is not None else None,
            )).fetchall()
            for row in rows:
                try:
                    data = json.loads(row["chunk_data"] or "{}")
                except (TypeError, ValueError):
                    data = {}
                data["narrator_voice_stale"] = True
                conn.execute("""
                    UPDATE checkpoint_chunks SET chunk_data = ?
                    WHERE translation_id = ? AND chunk_index = ?
                """, (json.dumps(data, ensure_ascii=False), translation_id,
                      int(row["chunk_index"])))
                changed += 1
            self._commit_connection(conn)
        return changed

    def close(self):
        """Close this thread's database connection."""
        conn = getattr(self._local, 'connection', None)
        if conn is not None:
            conn.close()
            self._local.connection = None
            with self._connections_lock:
                try:
                    self._all_connections.remove(conn)
                except ValueError:
                    pass

    def close_all(self):
        """Close every connection opened by any thread.

        Worker threads die without a chance to call close(), so their
        thread-local connections would otherwise stay open until the process
        exits. Called from the shutdown hook in src.api.translation_state.
        """
        with self._connections_lock:
            connections = list(self._all_connections)
            self._all_connections.clear()
        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._local = threading.local()
