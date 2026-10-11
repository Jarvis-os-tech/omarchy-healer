"""SQLite incident memory store for Omarchy-Healer."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import HealerGraphState


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

    def _init_schema(self) -> None:
        with self._connect() as conn:
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

    def find_successful_fixes(self, fingerprint: str, *, limit: int = 3) -> list[dict[str, Any]]:
        with self._connect() as conn:
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

    def record_state(self, state: HealerGraphState) -> int:
        symptom = state.get("symptom") or {}
        triage = state.get("triage") or {}
        diagnosis = state.get("diagnosis") or {}
        plan = state.get("plan") or {}
        execution = state.get("execution") or {}
        verification = state.get("verification") or {}

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
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
                    json.dumps(dict(state), ensure_ascii=False),
                    now,
                ),
            )
            return int(cursor.lastrowid or 0)

    def recent_incidents(self, *, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
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
