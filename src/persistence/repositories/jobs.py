"""Job, chunk and checkpoint persistence.

This repository owns the SQL for translation jobs and their chunks. The
connection and the write lock still belong to the `Database` facade -- it is
the object that opens the file, holds the per-thread connections and registers
the atexit close -- so the methods here reach through `self.database` for
those two primitives only.

`Database` keeps a same-named delegating method for every method below, so
existing call sites are unaffected. Those delegations are also why the
inherited `__getattr__` forwarder never fires for them: a concrete method
defined on this class shadows it, so `db.jobs.create_job` runs this code
rather than bouncing back to the facade.
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.persistence.database import sanitize_config_secrets
from src.persistence.repositories.base import DatabaseRepository


class JobRepository(DatabaseRepository):
    """Owns translation-job and chunk SQL."""

    prefixes = ("job", "chunk", "translation")
    methods = frozenset({
        "create_job",
        "delete_job",
        "get_job",
        "get_resumable_jobs",
        "save_chunk",
        "update_job_config",
        "update_job_progress",
    })

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
        with self.database._lock:
            try:
                conn = self.database._get_connection()
                cursor = conn.cursor()

                progress = {
                    'current_chunk_index': -1,
                    'total_chunks': 0,
                    'completed_chunks': 0,
                    'failed_chunks': 0,
                    'review_required_chunks': 0,
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
        review_required_chunks: Optional[int] = None,
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
        with self.database._lock:
            try:
                conn = self.database._get_connection()
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
                if review_required_chunks is not None:
                    progress['review_required_chunks'] = review_required_chunks
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

                quality_status = (
                    'review_required'
                    if int(progress.get('review_required_chunks') or 0) > 0
                    else 'passed'
                    if int(progress.get('completed_chunks') or 0) > 0
                    else 'not_checked'
                )
                updates.append("quality_status = ?")
                params.append(quality_status)

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
        status: str = 'completed',
        quality_status: Optional[str] = None,
        execution_failure_class: Optional[str] = None,
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
        with self.database._lock:
            try:
                conn = self.database._get_connection()
                cursor = conn.cursor()

                data = dict(chunk_data or {})
                resolved_quality_status = str(
                    quality_status
                    or data.get('quality_status')
                    or ('passed' if status == 'completed' else 'not_checked')
                )
                resolved_failure_class = (
                    execution_failure_class
                    or data.get('execution_failure_class')
                )
                cursor.execute("""
                    INSERT OR REPLACE INTO checkpoint_chunks
                    (translation_id, chunk_index, original_text, translated_text,
                     chunk_data, status, quality_status,
                     execution_failure_class, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    translation_id,
                    chunk_index,
                    original_text,
                    translated_text,
                    json.dumps(data) if data else None,
                    status,
                    resolved_quality_status,
                    resolved_failure_class,
                ))

                chunk_counts = cursor.execute("""
                    SELECT
                        SUM(CASE WHEN quality_status = 'review_required' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)
                    FROM checkpoint_chunks
                    WHERE translation_id = ?
                """, (translation_id,)).fetchone()
                quality_count = int(chunk_counts[0] or 0)
                completed_count = int(chunk_counts[1] or 0)
                job_row = cursor.execute(
                    "SELECT progress FROM translation_jobs WHERE translation_id = ?",
                    (translation_id,),
                ).fetchone()
                if job_row:
                    progress = json.loads(job_row['progress'])
                    progress['review_required_chunks'] = int(quality_count or 0)
                    job_quality = (
                        'review_required' if quality_count
                        else 'passed' if completed_count
                        else 'not_checked'
                    )
                    cursor.execute("""
                        UPDATE translation_jobs
                        SET progress = ?, quality_status = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE translation_id = ?
                    """, (
                        json.dumps(progress), job_quality, translation_id,
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
        with self.database._lock:
            try:
                conn = self.database._get_connection()
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
                    'quality_status': row['quality_status'],
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
        with self.database._lock:
            try:
                conn = self.database._get_connection()
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

    def get_resumable_jobs(self, max_age_days: int = 30) -> List[Dict[str, Any]]:
        """
        Get all jobs that can resume or seed an Add New Content job.

        Args:
            max_age_days: Maximum age in days for resumable jobs (default 30)

        Returns:
            List of job dictionaries
        """
        with self.database._lock:
            try:
                conn = self.database._get_connection()
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
                        'quality_status': row['quality_status'],
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

    def delete_job(self, translation_id: str) -> bool:
        """
        Delete a job and all its chunks (CASCADE).

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        with self.database._lock:
            try:
                conn = self.database._get_connection()
                self.database._delete_job_rows(conn, translation_id)
                conn.commit()
                return True
            except Exception as e:
                print(f"Error deleting job: {e}")
                return False
