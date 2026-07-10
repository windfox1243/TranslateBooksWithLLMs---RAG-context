"""
SQLite database manager for translation job persistence.
"""

import sqlite3
import json
import os
import time
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
import threading


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

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema
        self._initialize_schema()

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
            self._local.connection = conn
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
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Translation jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS translation_jobs (
                    translation_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    config JSON NOT NULL,
                    progress JSON NOT NULL,
                    translation_context JSON,
                    server_session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    paused_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            # Add server_session_id column if it doesn't exist (migration for existing DBs)
            cursor.execute("PRAGMA table_info(translation_jobs)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'server_session_id' not in columns:
                cursor.execute("ALTER TABLE translation_jobs ADD COLUMN server_session_id TEXT")

            # Checkpoint chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_chunks (
                    translation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    original_text TEXT NOT NULL,
                    translated_text TEXT,
                    chunk_data JSON,
                    status TEXT NOT NULL,
                    completed_at TIMESTAMP,
                    PRIMARY KEY (translation_id, chunk_index),
                    FOREIGN KEY (translation_id) REFERENCES translation_jobs(translation_id)
                        ON DELETE CASCADE
                )
            """)

            # Context Entities Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_entities (
                    translation_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    aliases JSON,
                    gender TEXT,
                    traits TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (translation_id, entity_id)
                )
            """)

            # Context Addressing Rules Table (Directed graph: speaker -> addressee)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_addressing_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    speaker_name TEXT NOT NULL,
                    addressee_name TEXT NOT NULL,
                    self_pronoun TEXT NOT NULL,
                    target_pronoun TEXT NOT NULL,
                    vocative TEXT,
                    register TEXT,
                    social_basis JSON,
                    scope TEXT NOT NULL DEFAULT 'durable',
                    contract_version INTEGER NOT NULL DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    is_locked INTEGER DEFAULT 0,
                    last_chunk_index INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(translation_id, speaker_name, addressee_name)
                )
            """)
            cursor.execute("PRAGMA table_info(context_addressing_rules)")
            addressing_columns = {row[1] for row in cursor.fetchall()}
            for column, definition in (
                ("social_basis", "JSON"),
                ("scope", "TEXT NOT NULL DEFAULT 'durable'"),
                ("contract_version", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if column not in addressing_columns:
                    cursor.execute(
                        f"ALTER TABLE context_addressing_rules ADD COLUMN {column} {definition}"
                    )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_addressing_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    speaker_name TEXT NOT NULL,
                    addressee_name TEXT NOT NULL,
                    source_form TEXT NOT NULL,
                    usage TEXT NOT NULL DEFAULT 'direct_address',
                    source_language TEXT,
                    evidence_quote TEXT,
                    scope TEXT NOT NULL DEFAULT 'durable',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    provenance TEXT NOT NULL DEFAULT 'unknown',
                    dialogue_turn_id TEXT,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(
                        translation_id, speaker_name, addressee_name,
                        source_form, usage, scope, evidence_quote
                    )
                )
            """)

            # Context Audit Logs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    speaker_name TEXT NOT NULL,
                    addressee_name TEXT NOT NULL,
                    old_state_json JSON,
                    new_state_json JSON,
                    trigger_source TEXT NOT NULL,
                    evidence_quote TEXT,
                    confidence REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Structured relationship graph. These tables are additive so
            # existing jobs and the directed-addressing API remain compatible.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_relationship_nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    aliases JSON,
                    entity_type TEXT NOT NULL DEFAULT 'character',
                    gender TEXT NOT NULL DEFAULT 'unknown',
                    is_locked INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(translation_id, normalized_name)
                )
            """)
            cursor.execute("PRAGMA table_info(context_relationship_nodes)")
            relationship_node_columns = {row[1] for row in cursor.fetchall()}
            if "gender" not in relationship_node_columns:
                cursor.execute(
                    "ALTER TABLE context_relationship_nodes "
                    "ADD COLUMN gender TEXT NOT NULL DEFAULT 'unknown'"
                )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_relationship_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    source_node_id INTEGER NOT NULL,
                    target_node_id INTEGER NOT NULL,
                    relationship_type TEXT NOT NULL,
                    direction TEXT NOT NULL DEFAULT 'symmetric',
                    scope TEXT NOT NULL DEFAULT 'durable',
                    hierarchy TEXT NOT NULL DEFAULT 'unknown',
                    relative_age TEXT NOT NULL DEFAULT 'unknown',
                    rank_relation TEXT NOT NULL DEFAULT 'unknown',
                    intimacy TEXT NOT NULL DEFAULT 'unknown',
                    register TEXT NOT NULL DEFAULT 'neutral',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    is_locked INTEGER NOT NULL DEFAULT 0,
                    last_chunk_index INTEGER NOT NULL DEFAULT 0,
                    provenance TEXT NOT NULL DEFAULT 'unknown',
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(
                        translation_id, source_node_id, target_node_id,
                        relationship_type, scope
                    ),
                    FOREIGN KEY (source_node_id)
                        REFERENCES context_relationship_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_node_id)
                        REFERENCES context_relationship_nodes(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("PRAGMA table_info(context_relationship_edges)")
            relationship_edge_columns = {row[1] for row in cursor.fetchall()}
            for column in ("relative_age", "rank_relation"):
                if column not in relationship_edge_columns:
                    cursor.execute(
                        f"ALTER TABLE context_relationship_edges ADD COLUMN "
                        f"{column} TEXT NOT NULL DEFAULT 'unknown'"
                    )

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_relationship_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    edge_id INTEGER,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    file_id TEXT,
                    dialogue_turn_id TEXT,
                    evidence_quote TEXT,
                    provenance TEXT NOT NULL DEFAULT 'unknown',
                    parser_status TEXT NOT NULL DEFAULT 'unknown',
                    confidence REAL NOT NULL DEFAULT 0.5,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (edge_id)
                        REFERENCES context_relationship_edges(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_relationship_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    edge_id INTEGER,
                    source_name TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'warning',
                    validator TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    reason TEXT NOT NULL,
                    remediation_hint TEXT,
                    candidate_json JSON,
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (edge_id)
                        REFERENCES context_relationship_edges(id) ON DELETE SET NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_relationship_derivations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    hierarchy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    path_json JSON NOT NULL,
                    basis TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    last_chunk_index INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(translation_id, source_name, target_name)
                )
            """)

            # Create indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status
                ON translation_jobs(status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_translation
                ON checkpoint_chunks(translation_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_addressing_translation
                ON context_addressing_rules(translation_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_translation
                ON context_audit_logs(translation_id, chunk_index)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_addressing_evidence_pair
                ON context_addressing_evidence(
                    translation_id, speaker_name, addressee_name
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_nodes_translation
                ON context_relationship_nodes(translation_id, normalized_name)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_edges_translation
                ON context_relationship_edges(translation_id, status)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_evidence_edge
                ON context_relationship_evidence(translation_id, edge_id, chunk_index)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_derivations_pair
                ON context_relationship_derivations(
                    translation_id, source_name, target_name
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationship_conflicts_translation
                ON context_relationship_conflicts(translation_id, status, chunk_index)
            """)

            conn.commit()

    def create_job(
        self,
        translation_id: str,
        file_type: str,
        config: Dict[str, Any],
        server_session_id: Optional[str] = None
    ) -> bool:
        """
        Create a new translation job record.

        Args:
            translation_id: Unique job identifier
            file_type: Type of file (txt, srt, epub)
            config: Full translation configuration
            server_session_id: Unique identifier for the current server session

        Returns:
            True if created successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                progress = {
                    'current_chunk_index': -1,
                    'total_chunks': 0,
                    'completed_chunks': 0,
                    'failed_chunks': 0,
                    'start_time': time.time(),  # Use timestamp for compatibility with existing code
                    # Marks the uniform checkpoint convention: current_chunk_index
                    # is the LAST COMPLETED unit for every format (resume = +1).
                    # Absent on pre-migration checkpoints, which load_checkpoint
                    # still handles via the legacy per-format branch.
                    'resume_index_semantics': 'completed',
                }

                cursor.execute("""
                    INSERT INTO translation_jobs
                    (translation_id, status, file_type, config, progress, server_session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    translation_id,
                    'running',
                    file_type,
                    json.dumps(sanitize_config_secrets(config)),
                    json.dumps(progress),
                    server_session_id
                ))

                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Job already exists
                return False
            except Exception as e:
                print(f"Error creating job: {e}")
                return False

    def update_job_progress(
        self,
        translation_id: str,
        current_chunk_index: Optional[int] = None,
        total_chunks: Optional[int] = None,
        completed_chunks: Optional[int] = None,
        failed_chunks: Optional[int] = None,
        status: Optional[str] = None,
        epub_accumulated_stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job progress information.

        Args:
            translation_id: Job identifier
            current_chunk_index: Current chunk being processed
            total_chunks: Total number of chunks
            completed_chunks: Number of completed chunks
            failed_chunks: Number of failed chunks
            status: Job status (running, paused, completed, error)
            epub_accumulated_stats: Snapshot of cross-file accumulated EPUB
                fallback counters. Stored verbatim in the progress JSON so the
                resume path can rehydrate counters that live above the
                per-file checkpoint (token_alignment_used, fallback_used, ...).

        Returns:
            True if updated successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Get current progress
                cursor.execute(
                    "SELECT progress FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return False

                progress = json.loads(row['progress'])

                # Update fields
                if current_chunk_index is not None:
                    progress['current_chunk_index'] = current_chunk_index
                if total_chunks is not None:
                    progress['total_chunks'] = total_chunks
                if completed_chunks is not None:
                    progress['completed_chunks'] = completed_chunks
                if failed_chunks is not None:
                    progress['failed_chunks'] = failed_chunks
                if epub_accumulated_stats is not None:
                    progress['epub_accumulated_stats'] = epub_accumulated_stats

                # Build update query
                updates = ["progress = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [json.dumps(progress)]

                if status:
                    updates.append("status = ?")
                    params.append(status)

                    if status == 'paused':
                        updates.append("paused_at = CURRENT_TIMESTAMP")
                    elif status == 'completed':
                        updates.append("completed_at = CURRENT_TIMESTAMP")

                params.append(translation_id)

                cursor.execute(
                    f"UPDATE translation_jobs SET {', '.join(updates)} WHERE translation_id = ?",
                    params
                )

                conn.commit()
                return True
            except Exception as e:
                print(f"Error updating job progress: {e}")
                return False

    def save_chunk(
        self,
        translation_id: str,
        chunk_index: int,
        original_text: str,
        translated_text: Optional[str] = None,
        chunk_data: Optional[Dict[str, Any]] = None,
        status: str = 'completed'
    ) -> bool:
        """
        Save a translated chunk to database.

        Args:
            translation_id: Job identifier
            chunk_index: Index of the chunk
            original_text: Original text
            translated_text: Translated text (if completed)
            chunk_data: Additional chunk metadata (context_before, context_after, etc.)
            status: Chunk status (completed, failed)

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO checkpoint_chunks
                    (translation_id, chunk_index, original_text, translated_text,
                     chunk_data, status, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    translation_id,
                    chunk_index,
                    original_text,
                    translated_text,
                    json.dumps(chunk_data) if chunk_data else None,
                    status
                ))

                conn.commit()
                return True
            except Exception as e:
                print(f"Error saving chunk: {e}")
                return False

    def get_job(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job information.

        Args:
            translation_id: Job identifier

        Returns:
            Job data dictionary or None if not found
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT * FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,)
                )
                row = cursor.fetchone()

                if not row:
                    return None

                return {
                    'translation_id': row['translation_id'],
                    'status': row['status'],
                    'file_type': row['file_type'],
                    'config': json.loads(row['config']),
                    'progress': json.loads(row['progress']),
                    'translation_context': json.loads(row['translation_context']) if row['translation_context'] else None,
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                    'paused_at': row['paused_at'],
                    'completed_at': row['completed_at']
                }
            except Exception as e:
                print(f"Error getting job: {e}")
                return None

    def update_job_config(self, translation_id: str, config: Dict[str, Any]) -> bool:
        """
        Update the configuration of an existing job.

        Args:
            translation_id: Job identifier
            config: New configuration dictionary

        Returns:
            True if updated successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                cursor.execute(
                    "UPDATE translation_jobs SET config = ?, updated_at = CURRENT_TIMESTAMP WHERE translation_id = ?",
                    (json.dumps(sanitize_config_secrets(config)), translation_id)
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error updating job config: {e}")
                return False

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

    def get_resumable_jobs(self, max_age_days: int = 30) -> List[Dict[str, Any]]:
        """
        Get all jobs that can resume or seed an Add New Content job.

        Args:
            max_age_days: Maximum age in days for resumable jobs (default 30)

        Returns:
            List of job dictionaries
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                # Only return jobs created within max_age_days
                cursor.execute("""
                    SELECT * FROM translation_jobs
                    WHERE status IN (
                        'paused', 'interrupted', 'error', 'partial', 'completed'
                    )
                    AND created_at > datetime('now', ? || ' days')
                    ORDER BY updated_at DESC
                """, (f'-{max_age_days}',))

                jobs = []
                for row in cursor.fetchall():
                    jobs.append({
                        'translation_id': row['translation_id'],
                        'status': row['status'],
                        'file_type': row['file_type'],
                        'config': json.loads(row['config']),
                        'progress': json.loads(row['progress']),
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'paused_at': row['paused_at']
                    })

                return jobs
            except Exception as e:
                print(f"Error getting resumable jobs: {e}")
                return []

    def cleanup_old_jobs(self, max_age_days: int = 30) -> int:
        """
        Delete old jobs that are no longer relevant.

        Removes jobs older than max_age_days that are in resumable states
        (paused, interrupted, error) to prevent database bloat.

        Args:
            max_age_days: Maximum age in days for jobs to keep (default 30)

        Returns:
            Number of jobs deleted
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
                    return 0

                # Delete old jobs (chunks deleted via CASCADE)
                cursor.execute("""
                    DELETE FROM translation_jobs
                    WHERE status IN ('paused', 'interrupted', 'error', 'partial', 'completed')
                    AND created_at <= datetime('now', ? || ' days')
                """, (f'-{max_age_days}',))

                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
            except Exception as e:
                print(f"Error cleaning up old jobs: {e}")
                return 0

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
        """
        Delete a job and all its chunks (CASCADE).

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()

                for table in (
                    "context_relationship_derivations",
                    "context_relationship_conflicts",
                    "context_relationship_evidence",
                    "context_relationship_edges",
                    "context_relationship_nodes",
                    "context_audit_logs",
                    "context_addressing_evidence",
                    "context_addressing_rules",
                    "context_entities",
                ):
                    cursor.execute(
                        f"DELETE FROM {table} WHERE translation_id = ?",
                        (translation_id,),
                    )
                cursor.execute(
                    "DELETE FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,)
                )

                conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting job: {e}")
                return False

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
        chunk_index: int = 0
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
                        social_basis, scope, contract_version, confidence,
                        is_locked, last_chunk_index, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(translation_id, speaker_name, addressee_name) DO UPDATE SET
                        self_pronoun = excluded.self_pronoun,
                        target_pronoun = excluded.target_pronoun,
                        vocative = excluded.vocative,
                        register = excluded.register,
                        social_basis = excluded.social_basis,
                        scope = excluded.scope,
                        contract_version = MAX(
                            context_addressing_rules.contract_version,
                            excluded.contract_version
                        ),
                        confidence = excluded.confidence,
                        is_locked = CASE WHEN context_addressing_rules.is_locked = 1 THEN 1 ELSE excluded.is_locked END,
                        last_chunk_index = excluded.last_chunk_index,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    translation_id, speaker_name, addressee_name,
                    self_pronoun, target_pronoun, vocative or "", register or "polite",
                    json.dumps(social_basis or [], ensure_ascii=False),
                    scope or "durable", int(contract_version or 1),
                    confidence, is_locked, chunk_index
                ))
                self._commit_connection(conn)
                return True
            except Exception as e:
                print(f"Error upserting addressing rule: {e}")
                return False

    def get_addressing_rules(self, translation_id: str) -> List[Dict[str, Any]]:
        """Fetch all addressing rules for a given translation job."""
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, speaker_name, addressee_name, self_pronoun,
                           target_pronoun, vocative, register, social_basis,
                           scope, contract_version, confidence, is_locked,
                           last_chunk_index, updated_at
                    FROM context_addressing_rules
                    WHERE translation_id = ?
                    ORDER BY speaker_name, addressee_name
                """, (translation_id,))
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
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    INSERT OR IGNORE INTO context_addressing_evidence (
                        translation_id, speaker_name, addressee_name,
                        source_form, usage, source_language, evidence_quote,
                        scope, confidence, provenance, dialogue_turn_id,
                        chunk_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    translation_id, speaker_name, addressee_name,
                    str(source_form).strip(), usage or "direct_address",
                    source_language or "", evidence_quote or "",
                    scope or "durable", confidence, provenance or "unknown",
                    dialogue_turn_id or "", chunk_index,
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
                        pass
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
        for node in self.get_relationship_nodes(translation_id):
            if str(node.get("normalized_name") or "").casefold() == wanted:
                return node
            if any(str(alias).casefold().strip() == wanted for alias in node.get("aliases") or []):
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
                        last_chunk_index, provenance, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    translation_id, source_node_id, target_node_id,
                    relationship_type, direction, scope, hierarchy,
                    relative_age, rank_relation, intimacy, register, confidence,
                    status, int(bool(is_locked)), chunk_index, provenance, details,
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
                self._commit_connection(conn)
                return int(row["id"]) if row else None
            except Exception as e:
                print(f"Error upserting relationship edge: {e}")
                return None

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
    ) -> Optional[int]:
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    INSERT INTO context_relationship_evidence (
                        translation_id, edge_id, chunk_index, file_id,
                        dialogue_turn_id, evidence_quote, provenance,
                        parser_status, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    translation_id, edge_id, chunk_index, file_id,
                    dialogue_turn_id, evidence_quote, provenance,
                    parser_status, confidence,
                ))
                self._commit_connection(conn)
                return int(cursor.lastrowid)
            except Exception as e:
                print(f"Error adding relationship evidence: {e}")
                return None

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
                cursor = conn.execute("""
                    INSERT INTO context_relationship_conflicts (
                        translation_id, edge_id, source_name, target_name,
                        severity, validator, status, reason,
                        remediation_hint, candidate_json, chunk_index
                    ) VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?)
                """, (
                    translation_id, edge_id, source_name, target_name,
                    severity, validator, reason, remediation_hint,
                    json.dumps(candidate or {}, ensure_ascii=False), chunk_index,
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

    def close(self):
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
