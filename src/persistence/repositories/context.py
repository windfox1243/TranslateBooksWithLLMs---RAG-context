"""Structured relationship and addressing persistence boundary."""

from .base import DatabaseRepository


class ContextRepository(DatabaseRepository):
    prefixes = ("addressing_", "context_", "relationship_")
    methods = frozenset({
        "add_addressing_evidence", "add_context_audit_log",
        "add_relationship_evidence", "context_state_transaction",
        "delete_addressing_rule", "get_addressing_evidence",
        "get_addressing_rules", "get_context_audit_logs",
        "get_relationship_edges", "resolve_addressing_evidence",
        "resolve_relationship_evidence", "set_relationship_edge_status",
        "upsert_addressing_rule", "upsert_relationship_edge",
    })
