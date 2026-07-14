"""Translation job and chunk persistence boundary."""

from .base import DatabaseRepository


class JobRepository(DatabaseRepository):
    prefixes = ("job", "chunk", "translation")
    methods = frozenset({
        "create_job", "delete_job", "get_job", "get_resumable_jobs",
        "save_chunk", "update_job_config", "update_job_progress",
    })
