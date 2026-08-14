"""The SQLite schema, and the one place that applies it.

Every table, migration ALTER and index this application needs is created
here. Database._initialize_schema used to hold all 863 lines of it inline.

The sweep is two different things, and they have different rules:

`_create_tables_and_migrate_columns` is pure DDL -- CREATE ... IF NOT EXISTS
and PRAGMA-guarded ALTERs. It only ever has work to do on a file older than
the current build, so it runs behind a PRAGMA user_version stamp and is
skipped entirely once the file is known current. That is what takes ~126
statements off every Database() construction, and Database is constructed
independently by five or more modules.

`_repair_data` is the opposite: UPDATEs that fix rows the *current* version can
still write -- the contract-v2 addressing quarantine, the contract-v3
copied-vocative reclassification, and the two evidence-fingerprint backfills.
They must act on inherited and freshly-written rows alike, so they run
unconditionally on every construction regardless of the version stamp.

The repairs only read tables and columns the DDL creates, never the reverse,
and no index here is UNIQUE, so running all the DDL first and all the repairs
second is equivalent to the interleaved order they were written in.

Bumping SCHEMA_VERSION is required whenever a statement is added to the DDL
half -- otherwise an existing file skips it and is left missing the table or
column. tests/unit/test_schema_version_stamp.py fails if the DDL text changes
without the bump.
"""
import hashlib
from typing import Any, Callable

# Bump whenever _create_tables_and_migrate_columns changes.
SCHEMA_VERSION = 3


def _evidence_fingerprint(*parts: Any) -> str:
    """Stable identity for one piece of evidence, used by the backfill ALTERs.

    Lives here rather than in database.py because the schema migration below
    computes fingerprints for rows that predate the column, and database.py
    imports this module.
    """
    normalized = "".join(
        " ".join(str(part or "").casefold().strip().split()) for part in parts
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def apply_schema(lock: Any, get_connection: Callable[[], Any]) -> None:
    """Create the tables, run the migrations and repairs, build the indexes.

    Operates on the calling thread's connection, under the database's lock.
    """
    with lock:
        conn = get_connection()
        cursor = conn.cursor()

        if cursor.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            _create_tables_and_migrate_columns(cursor)
            # Stamped only once the sweep has succeeded. A failure part-way
            # through leaves the old stamp, so the next construction retries
            # the whole thing rather than assuming it landed.
            cursor.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

        _repair_data(cursor)

        conn.commit()


def _create_tables_and_migrate_columns(cursor: Any) -> None:
    """Every CREATE TABLE, column-adding ALTER and CREATE INDEX, in order.

    Idempotent, and skipped when the file already carries the current
    SCHEMA_VERSION stamp.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS translation_jobs (
            translation_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            file_type TEXT NOT NULL,
            config JSON NOT NULL,
            progress JSON NOT NULL,
            quality_status TEXT NOT NULL DEFAULT 'not_checked',
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
    if 'quality_status' not in columns:
        cursor.execute(
            "ALTER TABLE translation_jobs ADD COLUMN quality_status "
            "TEXT NOT NULL DEFAULT 'not_checked'"
        )

    # Checkpoint chunks table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkpoint_chunks (
            translation_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            original_text TEXT NOT NULL,
            translated_text TEXT,
            chunk_data JSON,
            status TEXT NOT NULL,
            quality_status TEXT NOT NULL DEFAULT 'not_checked',
            execution_failure_class TEXT,
            completed_at TIMESTAMP,
            PRIMARY KEY (translation_id, chunk_index),
            FOREIGN KEY (translation_id) REFERENCES translation_jobs(translation_id)
                ON DELETE CASCADE
        )
    """)

    cursor.execute("PRAGMA table_info(checkpoint_chunks)")
    chunk_columns = [row[1] for row in cursor.fetchall()]
    if 'quality_status' not in chunk_columns:
        cursor.execute(
            "ALTER TABLE checkpoint_chunks ADD COLUMN quality_status "
            "TEXT NOT NULL DEFAULT 'not_checked'"
        )
    if 'execution_failure_class' not in chunk_columns:
        cursor.execute(
            "ALTER TABLE checkpoint_chunks ADD COLUMN "
            "execution_failure_class TEXT"
        )

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
            fingerprint TEXT NOT NULL DEFAULT '',
            observation_count INTEGER NOT NULL DEFAULT 1,
            resolution_status TEXT NOT NULL DEFAULT 'open',
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(
                translation_id, speaker_name, addressee_name,
                source_form, usage, scope, evidence_quote
            )
        )
    """)
    cursor.execute("PRAGMA table_info(context_addressing_evidence)")
    addressing_evidence_columns = {row[1] for row in cursor.fetchall()}
    for column, definition in {
        "fingerprint": "TEXT NOT NULL DEFAULT ''",
        "observation_count": "INTEGER NOT NULL DEFAULT 1",
        "resolution_status": "TEXT NOT NULL DEFAULT 'open'",
        # SQLite cannot add a column with a non-constant default to an
        # existing table. Fresh databases still receive the default
        # from CREATE TABLE; migrations backfill below.
        "last_seen_at": "TIMESTAMP",
        "resolved_at": "TIMESTAMP",
    }.items():
        if column not in addressing_evidence_columns:
            cursor.execute(
                f"ALTER TABLE context_addressing_evidence ADD COLUMN "
                f"{column} {definition}"
            )
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_addressing_evidence_fingerprint
        ON context_addressing_evidence(translation_id, fingerprint)
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

    # An edge holds one current state per pair, so "enemies until chunk 40,
    # allies after" was not representable and the earlier half was simply
    # overwritten. This append-only companion keeps each state and the chunk it
    # started at, which is what a translation of chapter 5 needs -- chapter 5's
    # relationships, not the last chapter's.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_relationship_edge_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            translation_id TEXT NOT NULL,
            edge_id INTEGER NOT NULL,
            from_chunk_index INTEGER NOT NULL DEFAULT 0,
            relationship_type TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'symmetric',
            scope TEXT NOT NULL DEFAULT 'durable',
            hierarchy TEXT NOT NULL DEFAULT 'unknown',
            intimacy TEXT NOT NULL DEFAULT 'unknown',
            register TEXT NOT NULL DEFAULT 'neutral',
            status TEXT NOT NULL DEFAULT 'accepted',
            details TEXT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(translation_id, edge_id, from_chunk_index),
            FOREIGN KEY (edge_id)
                REFERENCES context_relationship_edges(id) ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relationship_edge_history_lookup
        ON context_relationship_edge_history (
            translation_id, edge_id, from_chunk_index
        )
    """)

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
            fingerprint TEXT NOT NULL DEFAULT '',
            observation_count INTEGER NOT NULL DEFAULT 1,
            resolution_status TEXT NOT NULL DEFAULT 'open',
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            resolved_at TIMESTAMP,
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
        "fingerprint": "TEXT NOT NULL DEFAULT ''",
        "observation_count": "INTEGER NOT NULL DEFAULT 1",
        "resolution_status": "TEXT NOT NULL DEFAULT 'open'",
        "last_seen_at": "TIMESTAMP",
        "resolved_at": "TIMESTAMP",
    }.items():
        if column not in relationship_evidence_columns:
            cursor.execute(
                f"ALTER TABLE context_relationship_evidence ADD COLUMN "
                f"{column} {definition}"
            )
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_relationship_evidence_fingerprint
        ON context_relationship_evidence(translation_id, fingerprint)
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
            refinement_pass_id TEXT,
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
            llm_issue_count INTEGER NOT NULL DEFAULT 0,
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
        CREATE TABLE IF NOT EXISTS editor_repair_batches (
            batch_id TEXT PRIMARY KEY,
            translation_id TEXT NOT NULL,
            scope TEXT NOT NULL,
            phase TEXT NOT NULL DEFAULT 'effective',
            status TEXT NOT NULL DEFAULT 'queued',
            stay_paused INTEGER NOT NULL DEFAULT 1,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            total_items INTEGER NOT NULL DEFAULT 0,
            completed_items INTEGER NOT NULL DEFAULT 0,
            succeeded_items INTEGER NOT NULL DEFAULT 0,
            failed_items INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS editor_repair_batch_items (
            batch_id TEXT NOT NULL,
            translation_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            phase TEXT NOT NULL DEFAULT 'effective',
            status TEXT NOT NULL DEFAULT 'queued',
            outcome TEXT,
            message TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            PRIMARY KEY (batch_id, chunk_index, phase),
            FOREIGN KEY (batch_id) REFERENCES editor_repair_batches(batch_id)
                ON DELETE CASCADE
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refinement_passes (
            pass_id TEXT PRIMARY KEY,
            translation_id TEXT NOT NULL,
            context_revision INTEGER NOT NULL DEFAULT 0,
            source_mode TEXT NOT NULL DEFAULT 'checkpoint',
            alignment_mode TEXT NOT NULL DEFAULT 'exact',
            status TEXT NOT NULL DEFAULT 'running',
            expected_units INTEGER NOT NULL DEFAULT 0,
            promoted INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS refinement_chunk_results (
            pass_id TEXT NOT NULL,
            translation_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            base_chunk_index INTEGER,
            source_text TEXT,
            refined_text TEXT,
            chunk_data JSON,
            status TEXT NOT NULL,
            quality_status TEXT NOT NULL DEFAULT 'not_checked',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (pass_id, chunk_index),
            FOREIGN KEY (pass_id) REFERENCES refinement_passes(pass_id)
                ON DELETE CASCADE
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
        "llm_issue_count": "INTEGER NOT NULL DEFAULT 0",
        "warning_count": "INTEGER NOT NULL DEFAULT 0",
        "resolved_issue_count": "INTEGER NOT NULL DEFAULT 0",
        "unresolved_issue_count": "INTEGER NOT NULL DEFAULT 0",
        "result_state": "TEXT NOT NULL DEFAULT 'unchanged_draft'",
        "recovered_truncation": "INTEGER NOT NULL DEFAULT 0",
        "refinement_pass_id": "TEXT",
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


def _repair_data(cursor: Any) -> None:
    """Data fixes that must run on every construction, not only on upgrade.

    These correct rows the current version is still capable of writing, so
    unlike the DDL above they are not gated on SCHEMA_VERSION.
    """
    addressing_rows = cursor.execute("""
        SELECT id, translation_id, speaker_name, addressee_name,
               source_form, usage, scope, evidence_quote
        FROM context_addressing_evidence
        WHERE fingerprint = '' OR fingerprint IS NULL
    """).fetchall()
    for row in addressing_rows:
        cursor.execute("""
            UPDATE context_addressing_evidence
            SET fingerprint = ?,
                last_seen_at = COALESCE(last_seen_at, created_at, CURRENT_TIMESTAMP)
            WHERE id = ?
        """, (
            _evidence_fingerprint(
                row["translation_id"], row["speaker_name"],
                row["addressee_name"], row["source_form"], row["usage"],
                row["scope"], row["evidence_quote"],
            ),
            row["id"],
        ))

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

    # Contract v3 could validate a source vocative and then promote the
    # entire Vietnamese target tuple. Reclassify those copied-vocative
    # tuples so existing jobs stop projecting unsupported pairs after
    # upgrade. Manual and locked facts always retain precedence.
    cursor.execute("""
        UPDATE context_addressing_rules AS rule
        SET validation_status = 'provisional',
            validation_reason = 'unsupported_vietnamese_target_pair',
            confidence = MIN(confidence, 0.79),
            social_basis = '[]',
            validated_at = CURRENT_TIMESTAMP
        WHERE rule.contract_version >= 3
          AND rule.is_locked = 0
          AND rule.validation_status = 'active'
          AND rule.provenance NOT IN (
              'user_manual', 'manual', 'manual_context', 'rest_api'
          )
          AND EXISTS (
              SELECT 1 FROM translation_jobs AS job
              WHERE job.translation_id = rule.translation_id
                AND lower(COALESCE(
                    CASE WHEN json_valid(job.config)
                        THEN json_extract(
                            job.config, '$.target_language'
                        )
                        ELSE ''
                    END,
                    ''
                )) IN ('vietnamese', 'vi')
          )
          AND trim(rule.target_pronoun) = trim(rule.vocative)
          AND EXISTS (
              SELECT 1 FROM context_addressing_evidence AS evidence
              WHERE evidence.translation_id = rule.translation_id
                AND evidence.speaker_name = rule.speaker_name
                AND evidence.addressee_name = rule.addressee_name
                AND lower(trim(evidence.source_form)) =
                    lower(trim(rule.target_pronoun))
          )
    """)

    relationship_rows = cursor.execute("""
        SELECT id, translation_id, edge_id, chunk_index,
               dialogue_turn_id, evidence_quote
        FROM context_relationship_evidence
        WHERE fingerprint = '' OR fingerprint IS NULL
    """).fetchall()
    for row in relationship_rows:
        cursor.execute("""
            UPDATE context_relationship_evidence
            SET fingerprint = ?,
                last_seen_at = COALESCE(last_seen_at, created_at, CURRENT_TIMESTAMP)
            WHERE id = ?
        """, (
            _evidence_fingerprint(
                row["translation_id"], row["edge_id"], row["chunk_index"],
                row["dialogue_turn_id"], row["evidence_quote"],
            ),
            row["id"],
        ))
