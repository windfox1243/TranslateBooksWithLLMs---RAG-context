"""
SQLite database manager for translation job persistence.
"""

import sqlite3
import json
import os
import shutil
import time
from contextlib import contextmanager
from typing import Optional, Dict, List, Any
import threading
from pathlib import Path


_TRANSLATION_CHILD_TABLES = (
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
            conn.execute("PRAGMA foreign_keys=ON")
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
                    notes TEXT,
                    scope TEXT NOT NULL DEFAULT 'durable',
                    contract_version INTEGER NOT NULL DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    is_locked INTEGER DEFAULT 0,
                    validation_status TEXT NOT NULL DEFAULT 'active',
                    validation_reason TEXT,
                    provenance TEXT NOT NULL DEFAULT 'unknown',
                    validated_at TIMESTAMP,
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
                ("notes", "TEXT"),
                ("validation_status", "TEXT NOT NULL DEFAULT 'active'"),
                ("validation_reason", "TEXT"),
                ("provenance", "TEXT NOT NULL DEFAULT 'unknown'"),
                ("validated_at", "TIMESTAMP"),
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

            # Contract-v2 beta builds could persist an indirect mention as a
            # durable directed rule. Quarantine only unlocked LLM-owned rules
            # whose evidence contains no exact spoken addressee form. Manual,
            # locked, and legacy rules remain active for compatibility.
            cursor.execute("""
                UPDATE context_addressing_rules AS rule
                SET validation_status = 'quarantined',
                    validation_reason = 'missing_direct_dialogue_evidence',
                    validated_at = CURRENT_TIMESTAMP
                WHERE rule.contract_version >= 2
                  AND rule.is_locked = 0
                  AND rule.validation_status = 'active'
                  AND EXISTS (
                      SELECT 1 FROM context_addressing_evidence AS evidence
                      WHERE evidence.translation_id = rule.translation_id
                        AND evidence.speaker_name = rule.speaker_name
                        AND evidence.addressee_name = rule.addressee_name
                        AND evidence.provenance NOT IN ('user_manual', 'manual_context')
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM context_addressing_evidence AS evidence
                      WHERE evidence.translation_id = rule.translation_id
                        AND evidence.speaker_name = rule.speaker_name
                        AND evidence.addressee_name = rule.addressee_name
                        AND evidence.usage IN ('direct_address', 'second_person')
                        AND length(trim(evidence.source_form)) > 0
                        AND instr(
                            lower(evidence.evidence_quote),
                            lower(evidence.source_form)
                        ) > 0
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
                    evidence_tier TEXT NOT NULL DEFAULT 'unknown',
                    reason_code TEXT NOT NULL DEFAULT '',
                    supporting_units INTEGER NOT NULL DEFAULT 0,
                    match_kind TEXT NOT NULL DEFAULT '',
                    validator_version INTEGER NOT NULL DEFAULT 2,
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
            edge_migrations = {
                "relative_age": "TEXT NOT NULL DEFAULT 'unknown'",
                "rank_relation": "TEXT NOT NULL DEFAULT 'unknown'",
                "evidence_tier": "TEXT NOT NULL DEFAULT 'unknown'",
                "reason_code": "TEXT NOT NULL DEFAULT ''",
                "supporting_units": "INTEGER NOT NULL DEFAULT 0",
                "match_kind": "TEXT NOT NULL DEFAULT ''",
                "validator_version": "INTEGER NOT NULL DEFAULT 2",
            }
            for column, definition in edge_migrations.items():
                if column not in relationship_edge_columns:
                    cursor.execute(
                        f"ALTER TABLE context_relationship_edges ADD COLUMN "
                        f"{column} {definition}"
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
                    match_kind TEXT NOT NULL DEFAULT '',
                    source_start INTEGER,
                    source_end INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (edge_id)
                        REFERENCES context_relationship_edges(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("PRAGMA table_info(context_relationship_evidence)")
            relationship_evidence_columns = {row[1] for row in cursor.fetchall()}
            for column, definition in {
                "match_kind": "TEXT NOT NULL DEFAULT ''",
                "source_start": "INTEGER",
                "source_end": "INTEGER",
            }.items():
                if column not in relationship_evidence_columns:
                    cursor.execute(
                        f"ALTER TABLE context_relationship_evidence ADD COLUMN "
                        f"{column} {definition}"
                    )

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
                    fingerprint TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (edge_id)
                        REFERENCES context_relationship_edges(id) ON DELETE SET NULL
                )
            """)
            cursor.execute("PRAGMA table_info(context_relationship_conflicts)")
            relationship_conflict_columns = {row[1] for row in cursor.fetchall()}
            if "fingerprint" not in relationship_conflict_columns:
                cursor.execute(
                    "ALTER TABLE context_relationship_conflicts "
                    "ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''"
                )

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
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_reasoning_migrations (
                    translation_id TEXT NOT NULL,
                    migration_key TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'running',
                    details JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (translation_id, migration_key)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_resync_runs (
                    run_id TEXT PRIMARY KEY,
                    translation_id TEXT NOT NULL,
                    start_chunk_index INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'staging',
                    initial_snapshot TEXT,
                    final_context TEXT,
                    last_processed_chunk INTEGER NOT NULL DEFAULT -1,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_resync_chunk_stage (
                    run_id TEXT NOT NULL,
                    translation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    original_text TEXT,
                    translated_text TEXT,
                    chunk_data JSON,
                    status TEXT NOT NULL,
                    relationship_candidates JSON,
                    addressing_candidates JSON,
                    parser_status TEXT,
                    PRIMARY KEY (run_id, chunk_index),
                    FOREIGN KEY (run_id) REFERENCES context_resync_runs(run_id)
                        ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS editor_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'translation',
                    provider TEXT,
                    model TEXT,
                    source_language TEXT,
                    target_language TEXT,
                    file_type TEXT,
                    prompt_version TEXT,
                    contract_version TEXT,
                    parse_status TEXT,
                    outcome TEXT NOT NULL DEFAULT 'running',
                    failure_class TEXT,
                    issue_count INTEGER NOT NULL DEFAULT 0,
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    resolved_issue_count INTEGER NOT NULL DEFAULT 0,
                    unresolved_issue_count INTEGER NOT NULL DEFAULT 0,
                    result_state TEXT NOT NULL DEFAULT 'unchanged_draft',
                    recovered_truncation INTEGER NOT NULL DEFAULT 0,
                    deterministic_count INTEGER NOT NULL DEFAULT 0,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    thinking_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    was_truncated INTEGER NOT NULL DEFAULT 0,
                    finish_reason TEXT,
                    blocked_reason TEXT,
                    response_hash TEXT,
                    diagnostics JSON,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS editor_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    attempt_index INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    parse_status TEXT,
                    failure_class TEXT,
                    reason_codes JSON,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    thinking_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    was_truncated INTEGER NOT NULL DEFAULT 0,
                    finish_reason TEXT,
                    blocked_reason TEXT,
                    response_hash TEXT,
                    excerpts JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES editor_runs(id) ON DELETE CASCADE
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_narrator_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    narrator_key TEXT NOT NULL DEFAULT 'default',
                    narrator_identity TEXT NOT NULL DEFAULT 'unknown',
                    point_of_view TEXT NOT NULL DEFAULT 'unknown',
                    self_reference TEXT NOT NULL DEFAULT '',
                    formality TEXT NOT NULL DEFAULT 'neutral',
                    speech_level TEXT NOT NULL DEFAULT '',
                    gender TEXT NOT NULL DEFAULT 'unknown',
                    number TEXT NOT NULL DEFAULT 'singular',
                    dialect TEXT NOT NULL DEFAULT '',
                    tense TEXT NOT NULL DEFAULT '',
                    stylistic_markers JSON NOT NULL DEFAULT '[]',
                    dimensions JSON NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    provenance TEXT NOT NULL DEFAULT 'unknown',
                    scope TEXT NOT NULL DEFAULT 'durable',
                    start_chunk_index INTEGER NOT NULL DEFAULT 0,
                    end_chunk_index INTEGER,
                    is_locked INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'provisional',
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(translation_id, narrator_key, start_chunk_index)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_narrator_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    profile_id INTEGER,
                    chunk_index INTEGER NOT NULL,
                    chapter_index INTEGER,
                    scene_key TEXT,
                    segment_id TEXT NOT NULL,
                    discourse_mode TEXT NOT NULL,
                    narrator_key TEXT NOT NULL DEFAULT 'default',
                    narrator_identity TEXT NOT NULL DEFAULT 'unknown',
                    point_of_view TEXT NOT NULL DEFAULT 'unknown',
                    dimensions JSON NOT NULL DEFAULT '{}',
                    source_quote TEXT NOT NULL,
                    target_quote TEXT NOT NULL,
                    transition_type TEXT NOT NULL DEFAULT 'none',
                    transition_evidence TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    provenance TEXT NOT NULL DEFAULT 'senior_editor',
                    status TEXT NOT NULL DEFAULT 'accepted',
                    rejection_reason TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(translation_id, chunk_index, segment_id, narrator_key),
                    FOREIGN KEY (profile_id)
                        REFERENCES context_narrator_profiles(id) ON DELETE SET NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_narrator_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    narrator_key TEXT NOT NULL,
                    from_profile_id INTEGER,
                    to_profile_id INTEGER,
                    chunk_index INTEGER NOT NULL,
                    chapter_index INTEGER,
                    scene_key TEXT,
                    transition_type TEXT NOT NULL,
                    evidence_quote TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'accepted',
                    provenance TEXT NOT NULL DEFAULT 'senior_editor',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (from_profile_id)
                        REFERENCES context_narrator_profiles(id) ON DELETE SET NULL,
                    FOREIGN KEY (to_profile_id)
                        REFERENCES context_narrator_profiles(id) ON DELETE SET NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_narrator_conflicts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    translation_id TEXT NOT NULL,
                    narrator_key TEXT NOT NULL DEFAULT 'default',
                    chunk_index INTEGER NOT NULL,
                    chapter_index INTEGER,
                    scene_key TEXT,
                    reason TEXT NOT NULL,
                    candidate_json JSON NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'open',
                    resolution TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_narrator_bootstrap_attempts (
                    translation_id TEXT NOT NULL,
                    attempt_kind TEXT NOT NULL,
                    boundary_key TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    sampled_chunks JSON NOT NULL DEFAULT '[]',
                    details JSON NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (translation_id, attempt_kind, boundary_key)
                )
            """)

            # Token-detail migration for databases created before beta.34.
            for table in ("editor_runs", "editor_attempts"):
                cursor.execute(f"PRAGMA table_info({table})")
                existing_columns = {row[1] for row in cursor.fetchall()}
                for column in ("thinking_tokens", "total_tokens"):
                    if column not in existing_columns:
                        cursor.execute(
                            f"ALTER TABLE {table} ADD COLUMN {column} "
                            "INTEGER NOT NULL DEFAULT 0"
                        )
            cursor.execute("PRAGMA table_info(editor_runs)")
            editor_run_columns = {row[1] for row in cursor.fetchall()}
            editor_run_migrations = {
                "warning_count": "INTEGER NOT NULL DEFAULT 0",
                "resolved_issue_count": "INTEGER NOT NULL DEFAULT 0",
                "unresolved_issue_count": "INTEGER NOT NULL DEFAULT 0",
                "result_state": "TEXT NOT NULL DEFAULT 'unchanged_draft'",
                "recovered_truncation": "INTEGER NOT NULL DEFAULT 0",
            }
            for column, definition in editor_run_migrations.items():
                if column not in editor_run_columns:
                    cursor.execute(
                        f"ALTER TABLE editor_runs ADD COLUMN {column} {definition}"
                    )

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
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_context_resync_translation
                ON context_resync_runs(translation_id, status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_editor_runs_translation
                ON editor_runs(translation_id, chunk_index, outcome)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_editor_attempts_run
                ON editor_attempts(run_id, attempt_index)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_narrator_profiles_effective
                ON context_narrator_profiles(
                    translation_id, status, start_chunk_index, end_chunk_index
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_narrator_observations_chunk
                ON context_narrator_observations(
                    translation_id, chunk_index, status
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_narrator_conflicts_status
                ON context_narrator_conflicts(translation_id, status, chunk_index)
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

                for translation_id in job_ids:
                    self._delete_job_rows(conn, translation_id)
                deleted_count = len(job_ids)
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
                self._delete_job_rows(conn, translation_id)
                conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting job: {e}")
                return False

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
        fields = (
            "translation_id", "chunk_index", "phase", "provider", "model",
            "source_language", "target_language", "file_type",
            "prompt_version", "contract_version", "outcome",
        )
        values = [payload.get(field) for field in fields]
        with self._lock:
            try:
                cursor = self._get_connection().cursor()
                cursor.execute(
                    f"INSERT INTO editor_runs ({','.join(fields)}) VALUES "
                    f"({','.join('?' for _ in fields)})",
                    values,
                )
                self._commit_connection(self._get_connection())
                return int(cursor.lastrowid)
            except Exception as exc:
                print(f"Error creating editor run: {exc}")
                return None

    def add_editor_attempt(self, run_id: int, payload: Dict[str, Any]) -> bool:
        """Append a bounded diagnostic record for one editor request."""
        fields = (
            "run_id", "attempt_index", "stage", "parse_status",
            "failure_class", "reason_codes", "prompt_tokens",
            "completion_tokens", "thinking_tokens", "total_tokens",
            "was_truncated", "finish_reason",
            "blocked_reason", "response_hash", "excerpts",
        )
        values = [
            run_id,
            int(payload.get("attempt_index", 0)),
            payload.get("stage") or "unknown",
            payload.get("parse_status"),
            payload.get("failure_class"),
            json.dumps(payload.get("reason_codes") or [], ensure_ascii=False),
            int(payload.get("prompt_tokens", 0) or 0),
            int(payload.get("completion_tokens", 0) or 0),
            int(payload.get("thinking_tokens", 0) or 0),
            int(payload.get("total_tokens", 0) or 0),
            int(bool(payload.get("was_truncated"))),
            payload.get("finish_reason"),
            payload.get("blocked_reason"),
            payload.get("response_hash"),
            json.dumps(payload.get("excerpts") or [], ensure_ascii=False),
        ]
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    f"INSERT INTO editor_attempts ({','.join(fields)}) VALUES "
                    f"({','.join('?' for _ in fields)})",
                    values,
                )
                self._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error saving editor attempt: {exc}")
                return False

    def finish_editor_run(self, run_id: int, payload: Dict[str, Any]) -> bool:
        """Finalize one editor run with a classified outcome."""
        with self._lock:
            try:
                conn = self._get_connection()
                conn.execute(
                    """
                    UPDATE editor_runs SET parse_status = ?, outcome = ?,
                        failure_class = ?, issue_count = ?,
                        warning_count = ?,
                        resolved_issue_count = ?, unresolved_issue_count = ?,
                        result_state = ?, recovered_truncation = ?,
                        deterministic_count = ?, prompt_tokens = ?,
                        completion_tokens = ?, thinking_tokens = ?,
                        total_tokens = ?, was_truncated = ?,
                        finish_reason = ?, blocked_reason = ?, response_hash = ?,
                        diagnostics = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        payload.get("parse_status"),
                        payload.get("outcome") or "review_required",
                        payload.get("failure_class"),
                        int(payload.get("issue_count", 0) or 0),
                        int(payload.get("warning_count", 0) or 0),
                        int(payload.get("resolved_issue_count", 0) or 0),
                        int(payload.get("unresolved_issue_count", 0) or 0),
                        payload.get("result_state") or "unchanged_draft",
                        int(bool(payload.get("recovered_truncation"))),
                        int(payload.get("deterministic_count", 0) or 0),
                        int(payload.get("prompt_tokens", 0) or 0),
                        int(payload.get("completion_tokens", 0) or 0),
                        int(payload.get("thinking_tokens", 0) or 0),
                        int(payload.get("total_tokens", 0) or 0),
                        int(bool(payload.get("was_truncated"))),
                        payload.get("finish_reason"),
                        payload.get("blocked_reason"),
                        payload.get("response_hash"),
                        json.dumps(payload.get("diagnostics") or {}, ensure_ascii=False),
                        int(run_id),
                    ),
                )
                self._commit_connection(conn)
                return True
            except Exception as exc:
                print(f"Error finishing editor run: {exc}")
                return False

    def get_editor_diagnostics(self, translation_id: str) -> Dict[str, Any]:
        """Return aggregate and per-run editor diagnostics for a job."""
        with self._lock:
            conn = self._get_connection()
            rows = [dict(row) for row in conn.execute(
                "SELECT * FROM editor_runs WHERE translation_id = ? ORDER BY id",
                (translation_id,),
            ).fetchall()]
            if not rows:
                return {
                    "translation_id": translation_id,
                    "classification": "legacy_unclassified",
                    "summary": {"total": 0},
                    "runs": [],
                }
            outcomes: Dict[str, int] = {}
            failures: Dict[str, int] = {}
            result_states: Dict[str, int] = {}
            attempts_by_run: Dict[int, list] = {}
            run_ids = [int(row["id"]) for row in rows]
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                for attempt_row in conn.execute(
                    f"SELECT * FROM editor_attempts WHERE run_id IN ({placeholders}) "
                    "ORDER BY run_id, attempt_index",
                    run_ids,
                ).fetchall():
                    attempt = dict(attempt_row)
                    for field in ("reason_codes", "excerpts"):
                        try:
                            attempt[field] = json.loads(attempt.get(field) or "[]")
                        except (TypeError, ValueError):
                            attempt[field] = []
                    # The API exposes classifications, not book excerpts.
                    attempt.pop("excerpts", None)
                    attempts_by_run.setdefault(int(attempt["run_id"]), []).append(
                        attempt
                    )
            for row in rows:
                stored_outcome = row.get("outcome") or "unknown"
                normalized_outcome = {
                    "repaired": "llm_repaired",
                    "draft_kept_review": "review_required",
                }.get(stored_outcome, stored_outcome)
                if normalized_outcome != stored_outcome:
                    row["legacy_outcome"] = stored_outcome
                    row["outcome"] = normalized_outcome
                outcomes[normalized_outcome] = outcomes.get(normalized_outcome, 0) + 1
                result_state = row.get("result_state") or "unchanged_draft"
                result_states[result_state] = result_states.get(result_state, 0) + 1
                if row.get("failure_class"):
                    failures[row["failure_class"]] = failures.get(
                        row["failure_class"], 0
                    ) + 1
                try:
                    row["diagnostics"] = json.loads(row.get("diagnostics") or "{}")
                except (TypeError, ValueError):
                    row["diagnostics"] = {}
                row["diagnostics"].pop("issues", None)
                row["attempts"] = attempts_by_run.get(int(row["id"]), [])
            successful = sum(
                outcomes.get(name, 0)
                for name in (
                    "no_issues", "warnings_only", "locally_repaired",
                    "llm_repaired",
                )
            )
            review_count = outcomes.get("review_required", 0)
            degraded = outcomes.get("transport_failed", 0)
            hard_failed = outcomes.get("blocked", 0)
            return {
                "translation_id": translation_id,
                "classification": "classified",
                "summary": {
                    "total": len(rows),
                    "outcomes": outcomes,
                    "failure_classes": failures,
                    "result_states": result_states,
                    "successful": successful,
                    "review_required": review_count,
                    "degraded": degraded,
                    "hard_failed": hard_failed,
                    "warnings": sum(
                        int(row.get("warning_count", 0) or 0) for row in rows
                    ),
                    "recovered": sum(
                        int(bool(row.get("recovered_truncation"))) for row in rows
                    ),
                },
                "runs": rows,
            }

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
        normalized = status if status in {"active", "quarantined"} else "quarantined"
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
                self._commit_connection(conn)
                return int(row["id"]) if row else None
            except Exception as e:
                print(f"Error upserting relationship edge: {e}")
                return None

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
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    INSERT INTO context_relationship_evidence (
                        translation_id, edge_id, chunk_index, file_id,
                        dialogue_turn_id, evidence_quote, provenance,
                        parser_status, confidence, match_kind, source_start,
                        source_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    translation_id, edge_id, chunk_index, file_id,
                    dialogue_turn_id, evidence_quote, provenance,
                    parser_status, confidence, match_kind, source_start, source_end,
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
        """Close database connection."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
