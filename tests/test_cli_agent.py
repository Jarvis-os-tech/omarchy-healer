"""Regression tests for the CLI watch loop: opt-in auto-heal, suppression, robustness.

Three production bugs are pinned here:

1. ``watch --auto-heal`` used ``store_true`` with ``default=True``, so no flag could
   ever turn auto-healing off and a bare ``watch`` silently mutated the desktop.
2. The loop had no dedup/cooldown/backoff, so an unfixable symptom re-executed its
   playbook and re-fired its notification on every interval, forever.
3. The loop only caught ``KeyboardInterrupt``, so one transient probe/network/SQLite
   error killed the watchdog permanently.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from omarchy_healer import cli
from omarchy_healer.cli import WatchOrchestrator, WatchSuppressor, build_parser, run_watch
from omarchy_healer.config import HealerConfig
from omarchy_healer.graph import OmarchyHealerOrchestrator
from omarchy_healer.models import ExecutionResult, SymptomEvent, TriageDecision

FINGERPRINT = "mcp_and_systemd:failed_unit:omarchy-ai.service"


class _FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeJudge:
    """Offline triage judge so tests never touch the provider fallback chain."""

    def __init__(self, playbook: str, tier: int, domain: str = "systemd_service") -> None:
        self.playbook = playbook
        self.tier = tier
        self.domain = domain

    def evaluate(self, symptom: SymptomEvent) -> TriageDecision:
        return TriageDecision(
            provider="heuristic",
            domain=self.domain,
            severity_score=0.6,
            is_actionable=True,
            actionable_probability=0.9,
            safety_tier=self.tier,
            selected_playbook=self.playbook,
            fallback_chain_tried=["heuristic"],
        )


def _symptom(
    *,
    fingerprint_key: str = "failed_unit:omarchy-ai.service",
    playbook: str = "restart_systemd_user_unit",
    args: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    domain: str = "systemd_service",
    tier: int = 1,
) -> SymptomEvent:
    return SymptomEvent.create(
        probe_name="mcp_and_systemd",
        domain=domain,
        fingerprint_key=fingerprint_key,
        title=f"Failed Systemd User Service: {fingerprint_key}",
        summary="unit entered failed state",
        metrics=metrics if metrics is not None else {"unit": "omarchy-ai.service", "active": "failed"},
        suggested_playbook=playbook,
        suggested_args=args if args is not None else {"unit": "omarchy-ai.service"},
        default_tier=tier,
    )


def _state(
    symptom: SymptomEvent,
    *,
    executed: bool = True,
    resolved: bool = False,
) -> dict[str, Any]:
    return {
        "symptom": symptom.model_dump(),
        "execution": {
            "executed": executed,
            "playbook_id": symptom.suggested_playbook,
            "success": executed and resolved,
        },
        "verification": {
            "verified": True,
            "resolved": resolved,
            "post_check_summary": "post-check summary",
        },
        "status": "healed" if resolved else "execution_failed",
    }


def _failing_playbook_runner(calls: list[str]):
    def runner(playbook_id: str, _args: dict[str, Any]) -> ExecutionResult:
        calls.append(playbook_id)
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=False,
            stderr="unit refuses to start",
            duration_ms=3,
        )

    return runner


# --------------------------------------------------------------------------- parser


def test_watch_defaults_to_report_only() -> None:
    """A watchdog that mutates a live desktop must require an explicit opt-in."""
    args = build_parser().parse_args(["watch"])
    assert args.auto_heal is False


def test_auto_heal_flag_enables_remediation() -> None:
    args = build_parser().parse_args(["watch", "--auto-heal"])
    assert args.auto_heal is True


def test_no_auto_heal_flag_is_accepted_and_disables_remediation() -> None:
    args = build_parser().parse_args(["watch", "--no-auto-heal"])
    assert args.auto_heal is False


def test_auto_heal_help_documents_both_switches_and_the_default() -> None:
    parser = build_parser()
    watch_action = next(
        action
        for action in parser._subparsers._group_actions[0].choices["watch"]._actions
        if action.dest == "auto_heal"
    )
    assert "Defaults to OFF" in watch_action.help
    assert "--no-auto-heal" in watch_action.help


@pytest.mark.parametrize("bad", ["0", "-5", "-0.5", "nan", "inf", "-inf", "abc", ""])
def test_invalid_interval_is_rejected_at_parse_time(bad: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["watch", "--interval", bad])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("good", ["0.5", "1", "15", "3600"])
def test_valid_interval_is_parsed(good: str) -> None:
    args = build_parser().parse_args(["watch", "--interval", good])
    assert args.interval == float(good)
    assert math.isfinite(args.interval) and args.interval > 0


def test_history_limit_must_be_positive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["history", "--limit", "0"])
    assert build_parser().parse_args(["history", "--limit", "5"]).limit == 5


# ---------------------------------------------------------------------- suppression


def test_unchanged_repeat_is_suppressed_and_never_renotified() -> None:
    clock = _FakeClock()
    suppressor = WatchSuppressor(interval=15.0, repeat_after=300.0, clock=clock)
    symptom = _symptom()

    assert suppressor.should_process(symptom) is True
    suppressor.record(_state(symptom, executed=True, resolved=True))
    assert suppressor.should_notify(symptom) is False

    for _ in range(5):
        clock.advance(15.0)
        assert suppressor.should_process(symptom) is False
        assert suppressor.should_notify(symptom) is False

    assert suppressor.pending_retries()
    fingerprint, delay = suppressor.pending_retries()[0]
    assert fingerprint == symptom.fingerprint
    assert delay == pytest.approx(300.0 - 75.0)

    assert suppressor.failure_count(symptom.fingerprint) == 0, (
        "a resolved symptom must not accumulate failure backoff"
    )


def test_new_fingerprint_passes_through_immediately() -> None:
    suppressor = WatchSuppressor(interval=15.0, clock=_FakeClock())
    first = _symptom()
    second = _symptom(fingerprint_key="failed_unit:omarchy-shell.service")

    assert suppressor.should_process(first) is True
    suppressor.record(_state(first, executed=True, resolved=False))
    assert suppressor.should_process(first) is False
    assert suppressor.should_process(second) is True


def test_changed_evidence_and_arguments_pass_through_and_reset_pacing() -> None:
    clock = _FakeClock()
    suppressor = WatchSuppressor(interval=15.0, clock=clock)
    symptom = _symptom()

    assert suppressor.should_process(symptom) is True
    suppressor.record(_state(symptom, executed=True, resolved=False))
    assert suppressor.should_process(symptom) is False

    changed_args = _symptom(args={"unit": "omarchy-shell.service"})
    assert suppressor.should_process(changed_args) is True
    assert suppressor.failure_count(symptom.fingerprint) == 0

    suppressor.record(_state(changed_args, executed=True, resolved=False))
    assert suppressor.should_process(changed_args) is False

    changed_metrics = _symptom(metrics={"unit": "omarchy-ai.service", "active": "activating"})
    assert suppressor.should_process(changed_metrics) is True


def test_notification_is_emitted_once_until_the_symptom_changes() -> None:
    suppressor = WatchSuppressor(interval=15.0, clock=_FakeClock())
    symptom = _symptom()

    assert suppressor.should_notify(symptom) is True
    assert suppressor.should_notify(symptom) is False
    assert suppressor.should_notify(symptom) is False

    changed = _symptom(metrics={"unit": "omarchy-ai.service", "active": "activating"})
    assert suppressor.should_notify(changed) is True
    assert suppressor.should_notify(changed) is False


def test_drifting_numeric_metrics_do_not_relabel_the_same_problem() -> None:
    """RAM/PSI percentages move every cycle; they must not defeat suppression."""
    clock = _FakeClock()
    suppressor = WatchSuppressor(interval=15.0, clock=clock)

    def memory_event(used_percent: float) -> SymptomEvent:
        return SymptomEvent.create(
            probe_name="memory_probe",
            domain="memory_pressure",
            fingerprint_key="ram_pressure",
            title="RAM pressure",
            summary="above warn threshold",
            metrics={"used_percent": used_percent, "critical": True},
            suggested_playbook="reap_zombie_parents",
            default_tier=2,
        )

    first = memory_event(91.2)
    assert suppressor.should_process(first) is True
    suppressor.record(_state(first, executed=True, resolved=False))

    for drifting in (92.4, 88.1, 94.9):
        clock.advance(suppressor.retry_delay(first.fingerprint) - 0.001)
        assert suppressor.should_process(memory_event(drifting)) is False

    strict = WatchSuppressor(interval=15.0, metric_quantum=5.0, clock=_FakeClock())
    assert strict.should_process(memory_event(91.2)) is True
    assert strict.should_process(memory_event(96.4)) is True


def test_backoff_grows_for_repeat_failures_and_caps() -> None:
    clock = _FakeClock()
    suppressor = WatchSuppressor(
        interval=15.0,
        base_backoff=15.0,
        max_backoff=120.0,
        clock=clock,
    )
    symptom = _symptom()
    observed: list[float] = []

    for attempt in range(1, 8):
        assert suppressor.should_process(symptom) is True, f"attempt {attempt} was suppressed"
        suppressor.record(_state(symptom, executed=True, resolved=False))
        delay = suppressor.retry_delay(symptom.fingerprint)
        observed.append(delay)
        assert delay == pytest.approx(min(120.0, 15.0 * 2 ** (attempt - 1)))

        clock.advance(delay - 0.001)
        assert suppressor.should_process(symptom) is False, "retry fired before its backoff"
        clock.advance(0.001)

    assert observed == pytest.approx([15.0, 30.0, 60.0, 120.0, 120.0, 120.0, 120.0])
    assert max(observed) == suppressor.max_backoff


def test_dry_run_repeat_is_cooldown_paced_not_failure_backed() -> None:
    clock = _FakeClock()
    suppressor = WatchSuppressor(interval=15.0, repeat_after=300.0, clock=clock)
    symptom = _symptom()

    suppressor.should_process(symptom)
    suppressor.record(_state(symptom, executed=False, resolved=False))

    assert suppressor.failure_count(symptom.fingerprint) == 0
    clock.advance(299.0)
    assert suppressor.should_process(symptom) is False
    clock.advance(1.0)
    assert suppressor.should_process(symptom) is True


def test_stale_fingerprints_are_pruned() -> None:
    clock = _FakeClock()
    suppressor = WatchSuppressor(interval=15.0, stale_after=600.0, clock=clock)
    symptom = _symptom()

    suppressor.should_process(symptom)
    assert suppressor.known_fingerprints() == [symptom.fingerprint]
    clock.advance(601.0)
    suppressor.prune()
    assert suppressor.known_fingerprints() == []


# ------------------------------------------------------------- watch orchestrator


def test_unfixable_symptom_runs_its_playbook_once(tmp_path: Path) -> None:
    clock = _FakeClock()
    calls: list[str] = []
    symptom = _symptom()
    config = HealerConfig(max_auto_tier=2, db_path=tmp_path / "incidents.db")

    orchestrator = WatchOrchestrator(
        config,
        suppressor=WatchSuppressor(interval=15.0, clock=clock),
        triage_judge=_FakeJudge("restart_systemd_user_unit", 1),
        probe_runner=lambda _cfg: [symptom],
        playbook_runner=_failing_playbook_runner(calls),
    )

    first = orchestrator.scan_and_process(auto_heal=True)
    assert len(first) == 1
    assert first[0]["execution"]["executed"] is True
    assert first[0]["verification"]["resolved"] is False

    for _ in range(10):
        clock.advance(orchestrator.suppressor.retry_delay(symptom.fingerprint) - 0.001)
        assert orchestrator.scan_and_process(auto_heal=True) == []

    assert calls == ["restart_systemd_user_unit"], "playbook re-executed inside the backoff window"


def test_new_symptom_is_admitted_while_an_old_one_is_suppressed(tmp_path: Path) -> None:
    clock = _FakeClock()
    calls: list[str] = []
    current = {"symptom": _symptom()}
    config = HealerConfig(max_auto_tier=2, db_path=tmp_path / "incidents.db")

    orchestrator = WatchOrchestrator(
        config,
        suppressor=WatchSuppressor(interval=15.0, clock=clock),
        triage_judge=_FakeJudge("restart_systemd_user_unit", 1),
        probe_runner=lambda _cfg: [current["symptom"]],
        playbook_runner=_failing_playbook_runner(calls),
    )

    orchestrator.scan_and_process(auto_heal=True)
    clock.advance(orchestrator.suppressor.retry_delay(current["symptom"].fingerprint) - 0.001)
    current["symptom"] = _symptom(fingerprint_key="failed_unit:omarchy-shell.service")

    results = orchestrator.scan_and_process(auto_heal=True)
    assert len(results) == 1
    assert results[0]["symptom"]["fingerprint"] == current["symptom"].fingerprint
    assert len(calls) == 2


# ------------------------------------------------------------------------ notification gating


def test_watch_notifies_once_per_fingerprint_until_it_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unfixable symptom must not re-fire its desktop notification every interval."""
    clock = _FakeClock()
    sent: list[tuple[str, str]] = []
    symptom = _symptom()
    config = HealerConfig(max_auto_tier=2, db_path=tmp_path / "incidents.db")

    monkeypatch.setattr(
        cli,
        "send_completion_notification",
        lambda title, body: sent.append((title, body)) or True,
    )
    monkeypatch.setattr(
        cli,
        "send_escalation_notification",
        lambda title, body: sent.append((title, body)) or True,
    )

    orchestrator = WatchOrchestrator(
        config,
        suppressor=WatchSuppressor(interval=15.0, clock=clock),
        triage_judge=_FakeJudge("restart_systemd_user_unit", 1),
        probe_runner=lambda _cfg: [symptom],
        playbook_runner=_failing_playbook_runner([]),
    )

    orchestrator.scan_and_process(auto_heal=True)
    after_first = len(sent)
    assert after_first >= 1

    for _ in range(5):
        clock.advance(orchestrator.suppressor.retry_delay(symptom.fingerprint) + 1.0)
        orchestrator.scan_and_process(auto_heal=True)

    assert len(sent) == after_first, "repeat notifications fired for an unchanged symptom"

    # A genuinely different problem still alerts.
    other = _symptom(fingerprint_key="failed_unit:something-else.service")
    orchestrator.probe_runner = lambda _cfg: [other]  # type: ignore[assignment]
    clock.advance(orchestrator.suppressor.retry_delay(other.fingerprint) + 1.0)
    orchestrator.scan_and_process(auto_heal=True)
    assert len(sent) > after_first


# ------------------------------------------------------------------------ loop body


class _StubWatchOrchestrator:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.suppressor = WatchSuppressor(clock=_FakeClock())
        self.calls: list[dict[str, Any]] = []

    def scan_and_process(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else []
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _sqlite_lock_error() -> Exception:
    return sqlite3.OperationalError("database is locked")


def test_transient_cycle_exception_does_not_stop_the_watchdog() -> None:
    orchestrator = _StubWatchOrchestrator(
        [
            RuntimeError("probe crashed"),
            _sqlite_lock_error(),
            [_state(_symptom(), executed=False, resolved=False)],
        ]
    )
    sleeps: list[float] = []

    exit_code = run_watch(
        orchestrator,
        interval=15.0,
        auto_heal=True,
        max_auto_tier=1,
        sleep=sleeps.append,
        max_cycles=3,
        render=lambda _results: None,
    )

    assert exit_code == 0
    assert len(orchestrator.calls) == 3, "watchdog stopped instead of continuing to the next cycle"
    assert sleeps == [15.0, 15.0, 15.0]


def test_watch_loop_renders_every_processed_cycle() -> None:
    symptom = _symptom()
    orchestrator = _StubWatchOrchestrator([[_state(symptom)], []])
    rendered: list[list[dict[str, Any]]] = []

    run_watch(
        orchestrator,
        interval=2.0,
        auto_heal=True,
        max_auto_tier=1,
        sleep=lambda _seconds: None,
        max_cycles=2,
        render=rendered.append,
    )

    assert len(rendered) == 1
    assert rendered[0][0]["symptom"]["fingerprint"] == symptom.fingerprint
    assert orchestrator.calls == [
        {"auto_heal": True, "max_auto_tier": 1},
        {"auto_heal": True, "max_auto_tier": 1},
    ]


def test_keyboard_interrupt_during_sleep_stops_the_watchdog_cleanly() -> None:
    orchestrator = _StubWatchOrchestrator([[]])
    sleeps: list[float] = []

    def interrupt(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            raise KeyboardInterrupt

    exit_code = run_watch(
        orchestrator,
        interval=15.0,
        auto_heal=False,
        max_auto_tier=None,
        sleep=interrupt,
        max_cycles=99,
        render=lambda _results: None,
    )

    assert exit_code == 0
    assert len(orchestrator.calls) == 2


# --------------------------------------------------------------- max-tier and JSON


class _RecordingOrchestrator:
    instances: list[_RecordingOrchestrator] = []

    def __init__(self, config: HealerConfig, **_kwargs: Any) -> None:
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        _RecordingOrchestrator.instances.append(self)

    def scan_and_process(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.results


def test_heal_forwards_max_tier_two_to_the_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingOrchestrator.instances.clear()
    monkeypatch.setattr(cli, "OmarchyHealerOrchestrator", _RecordingOrchestrator)

    assert cli.main(["heal", "--max-tier", "2"]) == 0
    assert cli.main(["scan"]) == 0
    assert cli.main(["heal"]) == 0

    heal_two, scan, heal_default = _RecordingOrchestrator.instances
    assert heal_two.calls == [{"auto_heal": True, "max_auto_tier": 2}]
    assert scan.calls == [{"auto_heal": False}]
    assert heal_default.calls == [{"auto_heal": True, "max_auto_tier": None}]


def test_max_tier_defaults_to_config_and_two_genuinely_escalates(tmp_path: Path) -> None:
    """``max_auto_tier=None`` must fall back to config, and ``--max-tier 2`` must execute."""
    config = HealerConfig(max_auto_tier=1, db_path=tmp_path / "incidents.db")
    calls: list[str] = []
    zombies = SymptomEvent.create(
        probe_name="memory_probe",
        domain="memory_pressure",
        fingerprint_key="zombie_parents",
        title="Zombie parent processes detected",
        summary="unreaped children",
        metrics={"zombie_count": 4, "critical": True},
        suggested_playbook="reap_zombie_parents",
        suggested_args={"zombies": ["1234", "1235"]},
        default_tier=2,
    )
    orchestrator = OmarchyHealerOrchestrator(
        config,
        triage_judge=_FakeJudge("reap_zombie_parents", 2, domain="memory_pressure"),
        probe_runner=lambda _cfg: [zombies],
        playbook_runner=_failing_playbook_runner(calls),
    )

    blocked = orchestrator.scan_and_process(auto_heal=True, max_auto_tier=None)
    assert blocked[0]["plan"]["approved"] is False
    assert "execution" not in blocked[0]
    assert calls == []

    escalated = orchestrator.scan_and_process(auto_heal=True, max_auto_tier=2)
    assert escalated[0]["plan"]["approved"] is True
    assert escalated[0]["execution"]["executed"] is True
    assert calls == ["reap_zombie_parents"]


def test_json_output_stays_valid_for_non_serializable_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = [
        {
            "symptom": _symptom().model_dump() | {"suggested_args": {"db": Path("/tmp/incidents.db")}},
            "execution": {"executed": False, "playbook_id": "inspect_only"},
            "verification": {"resolved": False, "post_check_summary": "nan"},
            "extras": {"tags": {"a", "b"}, "ratio": float("nan"), "handle": object(), "score": 0.5},
        }
    ]

    class _JsonOrchestrator(_RecordingOrchestrator):
        def __init__(self, config: HealerConfig, **_kwargs: Any) -> None:
            super().__init__(config)
            self.results = payload

    monkeypatch.setattr(cli, "OmarchyHealerOrchestrator", _JsonOrchestrator)
    assert cli.main(["scan", "--json"]) == 0

    printed = capsys.readouterr().out
    decoded = json.loads(printed)
    assert decoded[0]["symptom"]["suggested_args"]["db"] == "/tmp/incidents.db"
    assert sorted(decoded[0]["extras"]["tags"]) == ["a", "b"]
    assert decoded[0]["extras"]["ratio"] == "nan"
    assert isinstance(decoded[0]["extras"]["handle"], str)
    assert decoded[0]["extras"]["score"] == 0.5


# --------------------------------------------------------------- improve command


def test_cli_improve_no_escalations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from omarchy_healer.memory import IncidentMemoryStore

    db = tmp_path / "incidents.db"
    IncidentMemoryStore(db)
    monkeypatch.setattr(cli.HealerConfig, "from_env", lambda: HealerConfig(db_path=db))
    code = cli.main(["improve"])
    assert code == 1
    out = capsys.readouterr().out
    assert "No capability escalation requests found" in out


def test_cli_improve_launches_omarchy_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omarchy_healer.memory import IncidentMemoryStore

    db = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db)
    state = _state(_symptom())
    state["escalation"] = {
        "reason": "missing_playbook",
        "provider": "heuristic",
        "missing_capability": "Fix bluetooth stack",
        "proposed_playbook_id": "restart_bluetooth",
        "proposed_feature_spec": "Restart systemd bluetooth unit",
        "agent_prompt": "Please implement restart_bluetooth",
        "notified": True,
    }
    store.record_state(state)

    launched: list[list[str]] = []
    chdirs: list[str] = []

    monkeypatch.setattr(cli.HealerConfig, "from_env", lambda: HealerConfig(db_path=db))
    monkeypatch.setattr(cli.os, "chdir", lambda p: chdirs.append(str(p)))
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: "/usr/share/omarchy/bin/omarchy-agent" if cmd == "omarchy-agent" else None)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: launched.append(cmd) or 0)

    code = cli.main(["improve"])
    assert code == 0
    assert len(launched) == 1
    cmd = launched[0]
    assert cmd[0] == "/usr/share/omarchy/bin/omarchy-agent"
    assert "--prompt" in cmd
    prompt_idx = cmd.index("--prompt")
    prompt_text = cmd[prompt_idx + 1]
    assert "restart_bluetooth" in prompt_text
    assert "Fix bluetooth stack" in prompt_text
    assert len(chdirs) == 1
    assert chdirs[0].endswith("omarchy-healer")


def test_cli_improve_inline_and_specific_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from omarchy_healer.memory import IncidentMemoryStore

    db = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db)

    # Record two escalations
    s1 = _state(_symptom())
    s1["escalation"] = {
        "reason": "missing_playbook",
        "provider": "heuristic",
        "missing_capability": "First capability",
        "proposed_playbook_id": "first_pb",
        "proposed_feature_spec": "Spec 1",
        "agent_prompt": "Prompt 1",
        "notified": True,
    }
    store.record_state(s1)

    s2 = _state(_symptom())
    s2["escalation"] = {
        "reason": "remediation_failed",
        "provider": "heuristic",
        "missing_capability": "Second capability",
        "proposed_playbook_id": "second_pb",
        "proposed_feature_spec": "Spec 2",
        "agent_prompt": "Prompt 2",
        "notified": True,
    }
    store.record_state(s2)

    launched: list[list[str]] = []
    monkeypatch.setattr(cli.HealerConfig, "from_env", lambda: HealerConfig(db_path=db))
    monkeypatch.setattr(cli.os, "chdir", lambda _p: None)
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: "/usr/share/omarchy/bin/omarchy-agent" if cmd == "omarchy-agent" else None)
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: launched.append(cmd) or 0)

    # Request ID 1 specifically with --inline
    code = cli.main(["improve", "1", "--inline"])
    assert code == 0
    assert len(launched) == 1
    cmd = launched[0]
    assert "--inline" in cmd
    prompt_text = cmd[cmd.index("--prompt") + 1]
    assert "Incident ID: #1" in prompt_text
    assert "first_pb" in prompt_text


def test_cli_improve_print_prompt_does_not_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from omarchy_healer.memory import IncidentMemoryStore

    db = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db)
    s = _state(_symptom())
    s["escalation"] = {
        "reason": "missing_playbook",
        "provider": "heuristic",
        "missing_capability": "Spec cap",
        "proposed_playbook_id": "spec_pb",
        "proposed_feature_spec": "Spec spec",
        "agent_prompt": "Prompt spec",
        "notified": True,
    }
    store.record_state(s)

    monkeypatch.setattr(cli.HealerConfig, "from_env", lambda: HealerConfig(db_path=db))
    monkeypatch.setattr(cli.subprocess, "run", lambda _c, **_k: pytest.fail("Should not launch"))

    code = cli.main(["improve", "--print-prompt"])
    assert code == 0
    out = capsys.readouterr().out
    assert "spec_pb" in out
    assert "Developer Instructions:" in out


def test_cli_improve_fallback_when_no_agent_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from omarchy_healer.memory import IncidentMemoryStore

    db = tmp_path / "incidents.db"
    store = IncidentMemoryStore(db)
    s = _state(_symptom())
    s["escalation"] = {
        "reason": "missing_playbook",
        "provider": "heuristic",
        "missing_capability": "Spec cap",
        "proposed_playbook_id": "spec_pb",
        "proposed_feature_spec": "Spec spec",
        "agent_prompt": "Prompt spec",
        "notified": True,
    }
    store.record_state(s)

    monkeypatch.setattr(cli.HealerConfig, "from_env", lambda: HealerConfig(db_path=db))
    monkeypatch.setattr(cli.shutil, "which", lambda _cmd: None)

    code = cli.main(["improve"])
    assert code == 0
    out = capsys.readouterr().out
    assert "'omarchy-agent' not found in PATH" in out
    assert "spec_pb" in out


