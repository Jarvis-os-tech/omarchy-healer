"""SQLite incident memory store for Omarchy-Healer."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import HealerGraphState

MAX_TRACE_JSON_CHARS = 64 * 1024


def _encode_trace(state: HealerGraphState) -> str:
    """Serialize the graph state, bounding it so a pathological state cannot bloat a row."""
    encoded = json.dumps(dict(state), ensure_ascii=False)
    if len(encoded) <= MAX_TRACE_JSON_CHARS:
        return encoded
    marker = {
        "truncated": True,
        "original_chars": len(encoded),
        "cap_chars": MAX_TRACE_JSON_CHARS,
    }

    def wrap(head: int) -> str:
        return json.dumps({**marker, "_trace": encoded[:head]}, ensure_ascii=False)

    # Encoded length grows monotonically with the head size, so bisect for the longest
    # head that still fits. Escaping quotes in the head make the mapping non-linear.
    low, high, best = 0, min(len(encoded), MAX_TRACE_JSON_CHARS), 0
    while low <= high:
        mid = (low + high) // 2
        if len(wrap(mid)) <= MAX_TRACE_JSON_CHARS:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return wrap(best)


class IncidentMemoryStore:
    """Persists incident triage, diagnosis, execution, and verification outcomes in SQLite."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection and guarantee it is closed, not merely committed.

        ``with sqlite3.connect(...) as conn`` commits or rolls back the transaction but
        leaves the file descriptor open, so a polling watchdog leaks one fd per store
        call until the garbage collector happens to reclaim it.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._session() as conn:
            self._configure_journal(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    title TEXT NOT NULL,
                    triage_provider TEXT NOT NULL,
                    diagnosis_provider TEXT NOT NULL,
                    playbook_id TEXT NOT NULL,
                    safety_tier INTEGER NOT NULL,
                    executed INTEGER NOT NULL,
                    resolved INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_incidents_fingerprint ON incidents(fingerprint)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS capability_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_row_id INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    title TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    missing_capability TEXT NOT NULL,
                    proposed_playbook_id TEXT NOT NULL,
                    proposed_feature_spec TEXT NOT NULL,
                    agent_prompt TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _configure_journal(conn: sqlite3.Connection) -> None:
        """Enable WAL so the watch loop can write while the CLI reads history."""
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            pass

    def find_successful_fixes(self, fingerprint: str, *, limit: int = 3) -> list[dict[str, Any]]:
        """Return up to ``limit`` previously executed and resolved incidents for a fingerprint."""
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT incident_id, fingerprint, playbook_id, safety_tier, triage_provider, created_at
                FROM incidents
                WHERE fingerprint = ? AND executed = 1 AND resolved = 1
                ORDER BY id DESC
                LIMIT ?
                """,
                (fingerprint, limit),
            ).fetchall()
            return [dict(row) for row in rows]

    def find_recent_failures(self, fingerprint: str, *, limit: int = 3) -> list[str]:
        """Return distinct recently executed and unresolved/failed playbooks for a fingerprint."""
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT playbook_id
                FROM incidents
                WHERE fingerprint = ? AND executed = 1 AND resolved = 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (fingerprint, limit),
            ).fetchall()
            return [str(row[0]) for row in rows if row[0]]

    def record_state(self, state: HealerGraphState) -> int:
        """Persist one incident row from a healer graph state and return its new row id."""
        symptom = state.get("symptom") or {}
        triage = state.get("triage") or {}
        diagnosis = state.get("diagnosis") or {}
        plan = state.get("plan") or {}
        execution = state.get("execution") or {}
        verification = state.get("verification") or {}
        escalation = state.get("escalation") or {}
        encoded_trace = _encode_trace(state)

        now = datetime.now(timezone.utc).isoformat()
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO incidents (
                    incident_id,
                    fingerprint,
                    domain,
                    title,
                    triage_provider,
                    diagnosis_provider,
                    playbook_id,
                    safety_tier,
                    executed,
                    resolved,
                    status,
                    trace_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(symptom.get("id") or "inc-unknown"),
                    str(symptom.get("fingerprint") or "unknown"),
                    str(symptom.get("domain") or "other"),
                    str(symptom.get("title") or ""),
                    str(triage.get("provider") or "heuristic"),
                    str(diagnosis.get("provider") or "heuristic"),
                    str(plan.get("playbook_id") or triage.get("selected_playbook") or "inspect_only"),
                    int(plan.get("safety_tier", triage.get("safety_tier", 0))),
                    1 if execution.get("executed") else 0,
                    1 if verification.get("resolved") else 0,
                    str(state.get("status") or "completed"),
                    encoded_trace,
                    now,
                ),
            )
            row_id = int(cursor.lastrowid or 0)
            if isinstance(escalation, dict) and escalation.get("agent_prompt"):
                conn.execute(
                    """
                    INSERT INTO capability_requests (
                        incident_row_id,
                        fingerprint,
                        domain,
                        title,
                        reason,
                        provider,
                        missing_capability,
                        proposed_playbook_id,
                        proposed_feature_spec,
                        agent_prompt,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row_id,
                        str(symptom.get("fingerprint") or "unknown"),
                        str(symptom.get("domain") or "other"),
                        str(symptom.get("title") or ""),
                        str(escalation.get("reason") or "remediation_failed"),
                        str(escalation.get("provider") or "heuristic"),
                        str(escalation.get("missing_capability") or ""),
                        str(escalation.get("proposed_playbook_id") or ""),
                        str(escalation.get("proposed_feature_spec") or ""),
                        str(escalation.get("agent_prompt") or ""),
                        now,
                    ),
                )
            return row_id

    def recent_incidents(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent incident rows, newest first, capped at ``limit``."""
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT id, incident_id, fingerprint, domain, title,
                       triage_provider, diagnosis_provider, playbook_id,
                       safety_tier, executed, resolved, status, created_at
                FROM incidents
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def recent_escalations(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent capability escalation requests, newest first."""
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT id, incident_row_id, fingerprint, domain, title,
                       reason, provider, missing_capability,
                       proposed_playbook_id, proposed_feature_spec,
                       agent_prompt, created_at
                FROM capability_requests
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_escalation(self, escalation_id: int) -> dict[str, Any] | None:
        """Fetch a single capability escalation request by ID."""
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT id, incident_row_id, fingerprint, domain, title,
                       reason, provider, missing_capability,
                       proposed_playbook_id, proposed_feature_spec,
                       agent_prompt, created_at
                FROM capability_requests
                WHERE id = ?
                """,
                (escalation_id,),
            ).fetchone()
            return dict(row) if row else None

