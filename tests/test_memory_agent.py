"""Regression tests for IncidentMemoryStore resource safety and trace bounding.

The store is polled every 15s by the watchdog (``find_successful_fixes`` +
``record_state``) while the CLI reads history, so a connection that is merely
committed and never closed exhausts the process fd limit and kills the watchdog.
These tests count real file descriptors via ``/proc/self/fd`` to prove closure.
"""
from __future__ import annotations

import gc
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from omarchy_healer.memory import MAX_TRACE_JSON_CHARS, IncidentMemoryStore

FD_DIR = Path("/proc/self/fd")
CALLS = 200
FD_SLACK = 5


def _fd_count() -> int:
    return len(os.listdir(FD_DIR))


@contextmanager
def _gc_paused() -> Iterator[None]:
    """Hold the collector still so a leaked connection can never be reclaimed behind our back.

    Without this, a regression leaks fds only until a generational collection happens to
    run, which is exactly what makes the production failure intermittent.
    """
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()
        gc.collect()


def _state(index: int = 0, **extra: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "symptom": {
            "id": f"inc-{index:04d}",
            "fingerprint": "omarchy_ui:stuck_omarchy_osd",
            "domain": "ui_shell",
            "title": "Stuck Omarchy OSD Overlay Popup",
        },
        "triage": {"provider": "laya", "selected_playbook": "close_stuck_osd", "safety_tier": 0},
        "diagnosis": {"provider": "gemini"},
        "plan": {"playbook_id": "close_stuck_osd", "safety_tier": 0},
        "execution": {"executed": True},
        "verification": {"resolved": True},
        "status": "healed",
    }
    state.update(extra)
    return state


def _read_trace_json(db_path: Path, row_id: int) -> str:
    with IncidentMemoryStore(db_path)._session() as conn:
        row = conn.execute("SELECT trace_json FROM incidents WHERE id = ?", (row_id,)).fetchone()
    assert row is not None
    return str(row["trace_json"])


def test_store_operations_do_not_leak_file_descriptors(tmp_path: Path) -> None:
    store = IncidentMemoryStore(tmp_path / "incidents.db")

    with _gc_paused():
        baseline = _fd_count()
        for index in range(CALLS):
            store.record_state(_state(index))
            store.recent_incidents(limit=5)
            store.find_successful_fixes("omarchy_ui:stuck_omarchy_osd", limit=3)
        growth = _fd_count() - baseline

    assert len(store.recent_incidents(limit=CALLS)) == CALLS
    assert growth <= FD_SLACK, f"leaked {growth} fds across {CALLS} store call rounds"


def test_measured_leak_is_real_so_the_fd_assertion_cannot_pass_vacuously(tmp_path: Path) -> None:
    """The committed-but-unclosed pattern must show up as fd growth under the same measurement."""
    store = IncidentMemoryStore(tmp_path / "incidents.db")

    with _gc_paused():
        baseline = _fd_count()
        for _ in range(CALLS):
            with store._connect() as conn:
                conn.execute("SELECT 1 FROM incidents LIMIT 1").fetchall()
        growth = _fd_count() - baseline

    assert growth > CALLS // 2, (
        "fd counting no longer detects the original bug; the leak tests above are vacuous"
    )


def test_connection_is_closed_when_a_write_fails_mid_operation(tmp_path: Path) -> None:
    store = IncidentMemoryStore(tmp_path / "incidents.db")
    unserializable = _state(0, metrics={"payload": object()})

    with _gc_paused():
        baseline = _fd_count()
        for _ in range(50):
            with pytest.raises(TypeError):
                store.record_state(unserializable)
        growth = _fd_count() - baseline

    assert growth <= FD_SLACK, f"failed writes leaked {growth} fds"
    assert store.recent_incidents(limit=5) == []


def test_connection_is_closed_when_a_read_fails_mid_operation(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)
    with store._session() as conn:
        conn.execute("DROP TABLE incidents")

    with _gc_paused():
        baseline = _fd_count()
        for _ in range(50):
            with pytest.raises(sqlite3.OperationalError):
                store.recent_incidents(limit=5)
            with pytest.raises(sqlite3.OperationalError):
                store.find_successful_fixes("omarchy_ui:stuck_omarchy_osd", limit=3)
        growth = _fd_count() - baseline

    assert growth <= FD_SLACK, f"failed reads leaked {growth} fds"


def test_trace_json_is_capped_and_marked(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)
    oversized = _state(0, evidence="x" * (4 * 1024 * 1024))

    row_id = store.record_state(oversized)
    stored = _read_trace_json(db_path, row_id)

    assert len(stored) <= MAX_TRACE_JSON_CHARS
    decoded = json.loads(stored)
    assert decoded["truncated"] is True
    assert decoded["original_chars"] > MAX_TRACE_JSON_CHARS
    assert decoded["cap_chars"] == MAX_TRACE_JSON_CHARS
    assert len(decoded["_trace"]) > 0


def test_trace_json_cap_holds_when_json_escaping_expands_the_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)
    hostile = _state(0, raw_evidence='quote " backslash \\ newline \n' * 200_000)

    stored = _read_trace_json(db_path, store.record_state(hostile))

    assert len(stored) <= MAX_TRACE_JSON_CHARS
    assert json.loads(stored)["truncated"] is True


def test_normal_trace_json_is_stored_verbatim(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)
    state = _state(7)

    stored = _read_trace_json(db_path, store.record_state(state))

    assert stored == json.dumps(state, ensure_ascii=False)
    assert json.loads(stored) == state


def test_public_api_shape_is_unchanged(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)

    first = store.record_state(_state(1))
    second = store.record_state(_state(2))
    assert first > 0 and second > first

    rows = store.recent_incidents(limit=1)
    assert set(rows[0]) == {
        "id",
        "incident_id",
        "fingerprint",
        "domain",
        "title",
        "triage_provider",
        "diagnosis_provider",
        "playbook_id",
        "safety_tier",
        "executed",
        "resolved",
        "status",
        "created_at",
    }
    assert rows[0]["incident_id"] == "inc-0002"

    fixes = store.find_successful_fixes("omarchy_ui:stuck_omarchy_osd", limit=3)
    assert set(fixes[0]) == {
        "incident_id",
        "fingerprint",
        "playbook_id",
        "safety_tier",
        "triage_provider",
        "created_at",
    }

    store.record_state(_state(3, execution={"executed": False}, verification={"resolved": False}))
    assert len(store.find_successful_fixes("omarchy_ui:stuck_omarchy_osd", limit=3)) == 2


def test_wal_journal_mode_is_enabled(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    IncidentMemoryStore(db_path)

    with sqlite3.connect(str(db_path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_history_is_readable_while_a_writer_session_is_open(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)
    store.record_state(_state(1))

    with store._session() as writer:
        writer.execute("BEGIN")
        writer.execute(
            "INSERT INTO incidents (incident_id, fingerprint, domain, title, triage_provider,"
            " diagnosis_provider, playbook_id, safety_tier, executed, resolved, status,"
            " trace_json, created_at) VALUES ('inc-x', 'f', 'other', 't', 'heuristic',"
            " 'heuristic', 'inspect_only', 0, 1, 1, 'healed', '{}', 'now')"
        )
        rows = store.recent_incidents(limit=10)

    assert len(rows) == 1, "WAL reader was blocked by the concurrent writer"


def test_get_escalation_by_id(tmp_path: Path) -> None:
    db_path = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db_path)
    state = _state(1)
    state["escalation"] = {
        "reason": "missing_playbook",
        "provider": "heuristic",
        "missing_capability": "New playbook for sound",
        "proposed_playbook_id": "fix_pipewire",
        "proposed_feature_spec": "Restart pipewire audio",
        "agent_prompt": "Please implement fix_pipewire",
        "notified": True,
    }
    row_id = store.record_state(state)
    escalations = store.recent_escalations(limit=1)
    assert len(escalations) == 1
    esc_id = escalations[0]["id"]

    fetched = store.get_escalation(esc_id)
    assert fetched is not None
    assert fetched["id"] == esc_id
    assert fetched["proposed_playbook_id"] == "fix_pipewire"
    assert fetched["missing_capability"] == "New playbook for sound"

    assert store.get_escalation(99999) is None



@pytest.fixture(autouse=True)
def _require_proc_fd() -> None:
    if not FD_DIR.is_dir():
        pytest.skip("fd accounting requires /proc/self/fd")