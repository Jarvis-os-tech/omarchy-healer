"""Tests for the LangGraph multi-agent workflow hardening in ``omarchy_healer.graph``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from omarchy_healer import graph as graph_module
from omarchy_healer.config import HealerConfig
from omarchy_healer.graph import (
    OmarchyHealerOrchestrator,
    _clamp_float,
    _clamp_int,
    _coerce_int,
    _coerce_str,
    domain_compatible,
)
from omarchy_healer.llm_router import LLMFallbackRouter
from omarchy_healer.models import ExecutionResult, SymptomEvent
from omarchy_healer.playbooks import PLAYBOOK_CATALOG
from omarchy_healer.probes import DEFAULT_OSD_STUCK_SECONDS

TRIAGE_MARKER = "Triage & Safety Judge"


class _ScriptedLLM:
    """Fake chat model that answers the triage and diagnostician prompts separately."""

    def __init__(
        self,
        *,
        triage: dict[str, Any] | None = None,
        diagnosis: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.triage = triage
        self.diagnosis = diagnosis
        self.error = error
        self.calls: list[str] = []

    def invoke(self, messages: Any) -> AIMessage:
        system = str(getattr(messages[0], "content", ""))
        is_triage = TRIAGE_MARKER in system
        self.calls.append("triage" if is_triage else "diagnosis")
        if self.error is not None:
            raise self.error
        payload = self.triage if is_triage else self.diagnosis
        return AIMessage(content=json.dumps(payload if payload is not None else {}))


class _RecordingPlaybookRunner:
    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.executed: list[str] = []
        self.on_success: Any = None

    def __call__(self, playbook_id: str, _args: dict[str, Any]) -> ExecutionResult:
        self.executed.append(playbook_id)
        if self.success and self.on_success is not None:
            self.on_success()
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=self.success,
            stdout=f"ran {playbook_id}",
            stderr="" if self.success else "boom",
            duration_ms=1,
        )


class _CountingProbeRunner:
    def __init__(self, symptoms: list[SymptomEvent]) -> None:
        self.symptoms = symptoms
        self.calls = 0

    def __call__(self, _cfg: HealerConfig) -> list[SymptomEvent]:
        self.calls += 1
        return list(self.symptoms)


def _config(tmp_path: Path, **overrides: Any) -> HealerConfig:
    base: dict[str, Any] = {
        "db_path": tmp_path / "incidents.db",
        "mcp_config_path": tmp_path / "mcp.json",
    }
    base.update(overrides)
    return HealerConfig(**base)


class _StubJudge:
    """Deterministic stand-in so the triage prompt never consumes a scripted LLM."""

    def evaluate(self, symptom: SymptomEvent):
        from omarchy_healer.triage_judge import TriageJudge

        return TriageJudge._heuristic_decision(symptom, ["heuristic"])


def _orchestrator(
    tmp_path: Path,
    *,
    probe_runner: Any = None,
    playbook_runner: Any = None,
    llm: _ScriptedLLM | None = None,
    osd_stuck_seconds: float = 30.0,
    **config_overrides: Any,
) -> OmarchyHealerOrchestrator:
    cfg = _config(tmp_path, **config_overrides)
    router = LLMFallbackRouter(
        cfg,
        custom_invokers=[("gemini", llm)] if llm is not None else [],
    )
    return OmarchyHealerOrchestrator(
        cfg,
        triage_judge=_StubJudge(),
        llm_router=router,
        probe_runner=probe_runner or _CountingProbeRunner([]),
        playbook_runner=playbook_runner or _RecordingPlaybookRunner(),
        osd_stuck_seconds=osd_stuck_seconds,
    )


def _osd_symptom(key: str = "stuck_omarchy_osd") -> SymptomEvent:
    return SymptomEvent.create(
        probe_name="omarchy_ui",
        domain="ui_shell",
        fingerprint_key=key,
        title="Stuck Omarchy OSD Overlay Popup",
        summary="omarchy-osd layer surface is stuck open.",
        metrics={"osd_state": "open"},
        suggested_playbook="close_stuck_osd",
        default_tier=0,
    )


def _memory_symptom() -> SymptomEvent:
    return SymptomEvent.create(
        probe_name="memory_pressure",
        domain="memory_pressure",
        fingerprint_key="ram:92",
        title="System Memory Pressure",
        summary="RAM above the critical threshold.",
        metrics={"ram_used_percent": 92.0},
        suggested_playbook="reclaim_memory_caches",
        suggested_args={
            "top_processes": [
                {"pid": 1, "command": "init", "rss_mb": 900, "mem_percent": 22.0}
            ]
        },
        default_tier=0,
    )


# --------------------------------------------------------------------------- issue 1


@pytest.mark.parametrize(
    "bogus_tier", [9, -4, "9", "banana", None, True, 1.9, float("inf"), {"t": 1}, [0]]
)
def test_untrusted_safety_tier_never_crashes_the_healer(
    tmp_path: Path, bogus_tier: Any
) -> None:
    llm = _ScriptedLLM(
        diagnosis={
            "root_cause": "Overlay surface never dismissed",
            "evidence_summary": "osd_state=open",
            "recommended_playbook": "close_stuck_osd",
            "safety_tier": bogus_tier,
            "risk_notes": "none",
        }
    )
    orchestrator = _orchestrator(tmp_path, llm=llm, gemini_api_key="fake-key")
    state = orchestrator.process_symptom(_osd_symptom(), auto_heal=False)

    assert 0 <= state["diagnosis"]["safety_tier"] <= 2
    assert 0 <= state["plan"]["safety_tier"] <= 2
    assert state["status"] == "dry_run_ready"


def test_garbage_llm_field_types_do_not_crash_the_healer(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        diagnosis={
            "root_cause": {"nested": "dict"},
            "evidence_summary": ["a", "list"],
            "recommended_playbook": {"not": "a string"},
            "safety_tier": "definitely not a tier",
            "risk_notes": None,
        }
    )
    orchestrator = _orchestrator(tmp_path, llm=llm, gemini_api_key="fake-key")
    state = orchestrator.process_symptom(_osd_symptom(), auto_heal=True, max_auto_tier=1)

    assert state["diagnosis"]["recommended_playbook"] == "close_stuck_osd"
    assert state["diagnosis"]["root_cause"]
    assert state["status"] in {"healed", "unresolved"}


def test_malformed_triage_payload_is_sanitized_in_graph_nodes(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    state = {
        "symptom": _osd_symptom().model_dump(),
        "triage": {"selected_playbook": ["nope"], "safety_tier": "banana", "is_actionable": True},
        "prior_incidents": [{"playbook_id": {"bad": "type"}}],
        "agent_trace": [],
    }

    diagnosed = orchestrator.diagnostician_node(state)
    planned = orchestrator.planner_node({**state, **diagnosed})

    assert diagnosed["diagnosis"]["recommended_playbook"] == "close_stuck_osd"
    assert planned["plan"]["playbook_id"] == "close_stuck_osd"
    assert planned["plan"]["safety_tier"] == PLAYBOOK_CATALOG["close_stuck_osd"]["tier"]


@pytest.mark.parametrize(
    ("value", "default", "raw", "clamped"),
    [
        (9, 1, 9, 2),
        (-9, 1, -9, 0),
        ("2", 1, 2, 2),
        ("2.7", 1, 2, 2),
        (1.9, 1, 1, 1),
        (True, 1, 1, 1),
        (None, 1, 1, 1),
        (float("nan"), 1, 1, 1),
        (float("inf"), 1, 1, 1),
        ({"a": 1}, 0, 0, 0),
        (False, 1, 1, 1),
        (True, 1, 1, 1),
    ],
)
def test_coercion_helpers_never_raise(value: Any, default: int, raw: int, clamped: int) -> None:
    assert _coerce_int(value, default) == raw
    assert _clamp_int(value, default) == clamped
    assert _clamp_int(99, 3, low=0, high=5) == 5


def test_string_and_float_coercion_helpers() -> None:
    assert _coerce_str("  hi  ") == "hi"
    assert _coerce_str(None, "fallback") == "fallback"
    assert _coerce_str(True, "fallback") == "fallback"
    assert _coerce_str({"a": 1}, "fallback") == "fallback"
    assert _coerce_str("x" * 50, max_chars=10) == "x" * 10
    assert _clamp_float("0.9", 0.0) == pytest.approx(0.9)
    assert _clamp_float("nope", 0.3) == pytest.approx(0.3)
    assert _clamp_float(17.0, 0.5) == pytest.approx(1.0)


# --------------------------------------------------------------------------- issue 2


def test_domain_compatibility_allowlist() -> None:
    assert domain_compatible("memory_pressure", "reclaim_memory_caches")[0] is True
    assert domain_compatible("memory_pressure", "clean_user_memory_caches")[0] is True
    assert domain_compatible("memory_pressure", "inspect_only")[0] is True
    assert domain_compatible("compositor", "close_stuck_osd")[0] is True
    ok, reason = domain_compatible("memory_pressure", "restart_systemd_user_unit")
    assert ok is False
    assert "incompatible" in reason
    assert domain_compatible("memory_pressure", "nope_not_real")[0] is False


def test_symptom_playbook_compatible_allows_clean_user_memory_caches() -> None:
    mem_sym = _memory_symptom()
    ok, _ = graph_module._symptom_playbook_compatible(mem_sym, "clean_user_memory_caches")
    assert ok is True

    zombie_sym = SymptomEvent.create(
        probe_name="memory_and_process",
        domain="memory_pressure",
        fingerprint_key="zombies:5",
        title="Zombies",
        summary="5 defunct",
        suggested_playbook="reap_zombie_parents",
        default_tier=1,
    )
    ok_zombie, reason = graph_module._symptom_playbook_compatible(zombie_sym, "clean_user_memory_caches")
    assert ok_zombie is False
    assert "does not match symptom remediation" in reason

def test_safety_gate_is_the_single_domain_enforcement_point(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    state = {
        "symptom": _memory_symptom().model_dump(),
        "auto_heal": True,
        "max_auto_tier": 2,
        "agent_trace": [],
        "plan": {
            "playbook_id": "restart_systemd_user_unit",
            "playbook_args": {"unit": "omarchy-ai.service"},
            "safety_tier": 1,
            "approved": False,
            "description": "cross-domain nonsense",
        },
    }

    result = orchestrator.safety_gate_node(state)
    plan = result["plan"]

    assert plan["playbook_id"] == "reclaim_memory_caches"
    assert "restart_systemd_user_unit" in plan["approval_reason"]
    assert any("[SafetyGate:domain]" in line for line in result["agent_trace"])
    assert plan["safety_tier"] >= 1


def test_cross_domain_llm_recommendation_never_reaches_the_executor(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        diagnosis={
            "root_cause": "systemd is eating RAM",
            "recommended_playbook": "restart_systemd_user_unit",
            "safety_tier": 0,
            "risk_notes": "restart the unit",
        }
    )
    runner = _RecordingPlaybookRunner()
    orchestrator = _orchestrator(
        tmp_path, playbook_runner=runner, llm=llm, gemini_api_key="fake-key"
    )

    state = orchestrator.process_symptom(_memory_symptom(), auto_heal=True, max_auto_tier=2)

    assert "restart_systemd_user_unit" not in runner.executed
    assert state["plan"]["playbook_id"] == "reclaim_memory_caches"
    assert "restart_systemd_user_unit" not in state["execution"]["playbook_id"]
    trace = "\n".join(state["agent_trace"])
    assert "discarded=restart_systemd_user_unit" in trace


def test_domain_fallback_to_inspect_only_when_suggested_playbook_is_also_wrong(
    tmp_path: Path,
) -> None:
    orchestrator = _orchestrator(tmp_path)
    symptom = SymptomEvent.create(
        probe_name="weird",
        domain="compositor",
        fingerprint_key="glitch",
        title="Compositor glitch",
        summary="frames dropped",
        suggested_playbook="restart_systemd_user_unit",
    )
    state = {
        "symptom": symptom.model_dump(),
        "auto_heal": True,
        "max_auto_tier": 2,
        "agent_trace": [],
        "plan": {"playbook_id": "reclaim_memory_caches", "safety_tier": 0},
    }
    plan = orchestrator.safety_gate_node(state)["plan"]
    assert plan["playbook_id"] == "inspect_only"


# --------------------------------------------------------------------------- issue 3


def test_verifier_does_not_claim_resolved_when_memory_is_still_high(tmp_path: Path) -> None:
    symptom = _memory_symptom()
    probes = _CountingProbeRunner([symptom])
    runner = _RecordingPlaybookRunner(success=True)
    orchestrator = _orchestrator(tmp_path, probe_runner=probes, playbook_runner=runner)

    state = orchestrator.process_symptom(symptom, auto_heal=True, max_auto_tier=1)

    assert runner.executed == ["reclaim_memory_caches"]
    assert state["verification"]["resolved"] is False
    assert state["status"] == "unresolved"
    assert "still detected" in state["verification"]["post_check_summary"]
    assert probes.calls == 1


def test_verifier_resolves_only_when_the_probe_sweep_clears(tmp_path: Path) -> None:
    symptom = _memory_symptom()
    probes = _CountingProbeRunner([symptom])
    orchestrator = _orchestrator(
        tmp_path, probe_runner=probes, playbook_runner=_RecordingPlaybookRunner()
    )

    state: dict[str, Any] = {
        "symptom": symptom.model_dump(),
        "auto_heal": True,
        "max_auto_tier": 1,
        "agent_trace": [],
        "triage": {"is_actionable": True, "selected_playbook": "reclaim_memory_caches"},
        "diagnosis": {"recommended_playbook": "reclaim_memory_caches", "safety_tier": 0},
        "plan": {"playbook_id": "reclaim_memory_caches", "playbook_args": {}, "safety_tier": 0},
    }
    state = {**state, **orchestrator.plan_graph.invoke(state)}
    probes.symptoms = []
    state = orchestrator.verify_graph.invoke(state)

    assert state["verification"]["resolved"] is True
    assert state["status"] == "healed"


# --------------------------------------------------------------------------- issue 4


def _default_probe_runner_orchestrator(tmp_path: Path, **kwargs: Any) -> OmarchyHealerOrchestrator:
    """Build an orchestrator without an injected probe runner so the default is exercised."""
    cfg = _config(tmp_path)
    return OmarchyHealerOrchestrator(cfg, triage_judge=_StubJudge(), **kwargs)


def test_default_probe_runner_waits_out_the_stuck_osd_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the orchestrator used to force osd_stuck_seconds=0.0.

    That explicit 0.0 overrode the probe's own default, so `elapsed >= 0.0` was true the
    instant any OSD opened and the executor closed a popup the user had just opened.
    A mutating run must inherit the probe default instead of zeroing it.
    """
    captured: dict[str, float] = {}

    def fake_collect(
        _config: HealerConfig, *, osd_stuck_seconds: float = 0.0
    ) -> list[SymptomEvent]:
        captured["osd_stuck_seconds"] = osd_stuck_seconds
        return []

    monkeypatch.setattr(graph_module, "collect_all_symptoms", fake_collect)
    orchestrator = _default_probe_runner_orchestrator(tmp_path)

    assert orchestrator.osd_stuck_seconds == pytest.approx(DEFAULT_OSD_STUCK_SECONDS)
    orchestrator.probe_runner(orchestrator.config)
    assert captured["osd_stuck_seconds"] == pytest.approx(DEFAULT_OSD_STUCK_SECONDS)
    assert captured["osd_stuck_seconds"] > 0.0

def test_osd_stuck_seconds_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, float] = {}

    def fake_collect(
        _config: HealerConfig, *, osd_stuck_seconds: float = 0.0
    ) -> list[SymptomEvent]:
        captured["osd_stuck_seconds"] = osd_stuck_seconds
        return []

    monkeypatch.setattr(graph_module, "collect_all_symptoms", fake_collect)
    orchestrator = _default_probe_runner_orchestrator(tmp_path, osd_stuck_seconds=12.5)

    orchestrator.probe_runner(orchestrator.config)
    assert orchestrator.osd_stuck_seconds == pytest.approx(12.5)
    assert captured["osd_stuck_seconds"] == pytest.approx(12.5)


# --------------------------------------------------------------------------- issue 5


def test_scan_and_process_costs_two_probe_sweeps_for_n_symptoms(tmp_path: Path) -> None:
    symptoms = [_osd_symptom(f"stuck_osd_{i}") for i in range(5)]
    probes = _CountingProbeRunner(symptoms)
    runner = _RecordingPlaybookRunner()
    runner.on_success = lambda: setattr(probes, "symptoms", [])
    orchestrator = _orchestrator(tmp_path, probe_runner=probes, playbook_runner=runner)

    results = orchestrator.scan_and_process(auto_heal=True, max_auto_tier=1)

    assert len(results) == 5
    assert probes.calls == 2
    assert all(state["verification"]["resolved"] is True for state in results)
    assert all(state["verification"]["remaining_anomalies"] == 0 for state in results)
    assert "shared post-batch probe sweep" in results[0]["verification"]["post_check_summary"]


def test_verification_reports_the_full_anomaly_count_not_a_clamped_one(tmp_path: Path) -> None:
    symptoms = [_osd_symptom(f"stuck_osd_{i}") for i in range(5)]
    probes = _CountingProbeRunner(symptoms)
    orchestrator = _orchestrator(
        tmp_path, probe_runner=probes, playbook_runner=_RecordingPlaybookRunner()
    )

    results = orchestrator.scan_and_process(auto_heal=True, max_auto_tier=1)

    assert probes.calls == 2
    assert all(state["verification"]["resolved"] is False for state in results)
    assert all(state["verification"]["remaining_anomalies"] == 5 for state in results)


def test_failed_playbooks_never_trigger_a_verification_sweep(tmp_path: Path) -> None:
    symptoms = [_osd_symptom(f"stuck_osd_{i}") for i in range(4)]
    probes = _CountingProbeRunner(symptoms)
    orchestrator = _orchestrator(
        tmp_path, probe_runner=probes, playbook_runner=_RecordingPlaybookRunner(success=False)
    )

    results = orchestrator.scan_and_process(auto_heal=True, max_auto_tier=1)

    assert probes.calls == 1
    assert all(state["verification"]["resolved"] is False for state in results)
    assert all(state["status"] == "unresolved" for state in results)


def test_scan_and_process_skips_the_sweep_when_nothing_was_approved(tmp_path: Path) -> None:
    symptom = SymptomEvent.create(
        probe_name="mcp_and_systemd",
        domain="systemd_service",
        fingerprint_key="failed_unit:omarchy-ai.service",
        title="Failed Systemd User Service",
        summary="Service failed",
        suggested_playbook="restart_systemd_user_unit",
        suggested_args={"unit": "omarchy-ai.service"},
        default_tier=1,
    )
    probes = _CountingProbeRunner([symptom])
    runner = _RecordingPlaybookRunner()
    orchestrator = _orchestrator(tmp_path, probe_runner=probes, playbook_runner=runner)

    results = orchestrator.scan_and_process(auto_heal=True, max_auto_tier=0)

    assert runner.executed == []
    assert probes.calls == 1
    assert results[0]["status"] == "awaiting_approval"
    assert "execution" not in results[0]


# --------------------------------------------------------------------------- issue 6


def test_prior_successful_fix_drives_the_diagnosis(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    symptom = _osd_symptom()
    orchestrator.memory.record_state(
        {
            "symptom": symptom.model_dump(),
            "triage": {"provider": "heuristic", "safety_tier": 0},
            "diagnosis": {"provider": "heuristic"},
            "plan": {"playbook_id": "close_stuck_osd", "safety_tier": 0},
            "execution": {"executed": True},
            "verification": {"resolved": True},
            "status": "healed",
        }
    )

    llm = _ScriptedLLM(
        diagnosis={
            "root_cause": "overlay surface leaked",
            "recommended_playbook": "restart_omarchy_shell",
            "safety_tier": 1,
        }
    )
    orchestrator.llm_router = LLMFallbackRouter(
        orchestrator.config, custom_invokers=[("gemini", llm)]
    )

    state = orchestrator.process_symptom(symptom, auto_heal=True, max_auto_tier=1)

    assert state["prior_incidents"], "memory should surface the earlier resolved incident"
    assert state["diagnosis"]["provider"] == "memory_cache"
    assert state["diagnosis"]["recommended_playbook"] == "close_stuck_osd"
    assert state["diagnosis"]["confidence"] >= 0.6
    assert state["plan"]["safety_tier"] == 0
    trace = "\n".join(state["agent_trace"])
    assert "memory_driven=True" in trace


def test_memory_cache_provider_is_unused_without_a_prior_success(tmp_path: Path) -> None:
    orchestrator = _orchestrator(tmp_path)
    state = orchestrator.process_symptom(_osd_symptom(), auto_heal=False)
    assert state["diagnosis"]["provider"] == "heuristic"


# --------------------------------------------------------------------------- issue 7


def test_llm_cannot_lower_the_catalog_safety_tier(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        diagnosis={
            "root_cause": "bar process wedged",
            "recommended_playbook": "restart_omarchy_shell",
            "safety_tier": 0,
        }
    )
    runner = _RecordingPlaybookRunner()
    orchestrator = _orchestrator(
        tmp_path, playbook_runner=runner, llm=llm, gemini_api_key="fake-key"
    )

    planned = orchestrator.planner_node(
        {
            "symptom": _osd_symptom().model_dump(),
            "triage": {"selected_playbook": "restart_omarchy_shell", "safety_tier": 0},
            "diagnosis": {"recommended_playbook": "restart_omarchy_shell", "safety_tier": 0},
            "agent_trace": [],
        }
    )
    assert planned["plan"]["safety_tier"] == PLAYBOOK_CATALOG["restart_omarchy_shell"]["tier"]

    state = orchestrator.process_symptom(_osd_symptom(), auto_heal=True, max_auto_tier=0)
    assert runner.executed == []
    assert state["status"] == "awaiting_approval"
    assert state["plan"]["approved"] is False


def test_llm_may_raise_the_safety_tier_above_the_catalog_floor(tmp_path: Path) -> None:
    llm = _ScriptedLLM(diagnosis={"recommended_playbook": "close_stuck_osd", "safety_tier": 2})
    orchestrator = _orchestrator(tmp_path, llm=llm, gemini_api_key="fake-key")

    planned = orchestrator.planner_node(
        {
            "symptom": _osd_symptom().model_dump(),
            "triage": {"selected_playbook": "close_stuck_osd", "safety_tier": 0},
            "diagnosis": {"recommended_playbook": "close_stuck_osd", "safety_tier": 2},
            "agent_trace": [],
        }
    )
    assert planned["plan"]["safety_tier"] == 2


# --------------------------------------------------------------------------- issue 8


class _FakeTool:
    def __init__(self, name: str, output: str = "{}", *, error: Exception | None = None) -> None:
        self.name = name
        self.output = output
        self.error = error
        self.calls = 0

    def invoke(self, args: dict[str, Any]) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.output


def test_diagnostician_runs_its_read_only_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _FakeTool(
        "inspect_memory_and_top_processes", '{"top_rss": ["PID 1 init"]}'
    )
    monkeypatch.setattr(
        graph_module, "_DIAGNOSTIC_TOOLS_BY_NAME", {"inspect_memory_and_top_processes": tool}
    )

    prompts: list[str] = []

    class _CapturingLLM(_ScriptedLLM):
        def invoke(self, messages: Any) -> AIMessage:
            prompts.append(str(messages[1].content))
            return super().invoke(messages)

    llm = _CapturingLLM(diagnosis={"recommended_playbook": "reclaim_memory_caches"})
    orchestrator = _orchestrator(tmp_path, llm=llm, gemini_api_key="fake-key")

    state = orchestrator.process_symptom(_memory_symptom(), auto_heal=False)

    assert tool.calls == 1
    assert "live_diagnostics" in prompts[0]
    assert "inspect_memory_and_top_processes" in prompts[0]
    assert "PID 1 init" in state["diagnosis"]["evidence_summary"]
    assert any("[Diagnostician:tools]" in line for line in state["agent_trace"])


def test_broken_diagnostic_tool_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _FakeTool(
        "inspect_memory_and_top_processes", error=OSError("hyprctl exploded")
    )
    monkeypatch.setattr(
        graph_module, "_DIAGNOSTIC_TOOLS_BY_NAME", {"inspect_memory_and_top_processes": tool}
    )
    llm = _ScriptedLLM(diagnosis={"recommended_playbook": "reclaim_memory_caches"})
    orchestrator = _orchestrator(tmp_path, llm=llm, gemini_api_key="fake-key")

    state = orchestrator.process_symptom(_memory_symptom(), auto_heal=False)

    assert tool.calls == 1
    assert state["diagnosis"]["provider"] == "gemini"
    assert "unavailable: OSError" in state["diagnosis"]["evidence_summary"]


def test_diagnostic_tools_are_not_run_without_an_llm_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _FakeTool("inspect_memory_and_top_processes")
    monkeypatch.setattr(
        graph_module, "_DIAGNOSTIC_TOOLS_BY_NAME", {"inspect_memory_and_top_processes": tool}
    )

    orchestrator = _orchestrator(tmp_path)
    state = orchestrator.process_symptom(_memory_symptom(), auto_heal=False)

    assert tool.calls == 0
    assert state["diagnosis"]["provider"] == "heuristic"


def test_zombie_symptom_rejects_reclaim_memory_caches_and_inspect_only(
    tmp_path: Path,
) -> None:
    zombie_symptom = SymptomEvent.create(
        probe_name="memory_and_process",
        domain="memory_pressure",
        fingerprint_key="zombie_process_accumulation",
        title="Zombie Process Accumulation (12 defunct)",
        summary="Found 12 defunct zombie processes on the host.",
        metrics={"zombie_count": 12},
        suggested_playbook="reap_zombie_parents",
        suggested_args={"zombies": [{"pid": 200, "ppid": 100, "command": "ps <defunct>"}]},
        default_tier=1,
    )
    llm = _ScriptedLLM(
        diagnosis={
            "root_cause": "zombies accumulated",
            "recommended_playbook": "reclaim_memory_caches",
            "safety_tier": 1,
        }
    )
    runner = _RecordingPlaybookRunner()
    probes = _CountingProbeRunner([])
    orchestrator = _orchestrator(
        tmp_path, probe_runner=probes, playbook_runner=runner, llm=llm, gemini_api_key="fake-key"
    )

    state = orchestrator.process_symptom(zombie_symptom, auto_heal=True, max_auto_tier=1)

    assert state["plan"]["playbook_id"] == "reap_zombie_parents"
    assert runner.executed == ["reap_zombie_parents"]
    assert state["status"] == "healed"


def test_capability_escalator_prompts_when_playbook_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "send_escalation_notification", lambda *_a, **_k: True)
    corrupted_mcp = SymptomEvent.create(
        probe_name="mcp_and_systemd",
        domain="mcp_config",
        fingerprint_key="invalid_json:/tmp/mcp.json",
        title="Corrupted MCP Configuration JSON",
        summary="Failed to parse /tmp/mcp.json: Expecting property name",
        metrics={"config_path": "/tmp/mcp.json"},
        suggested_playbook="inspect_only",
        default_tier=2,
    )
    orchestrator = _orchestrator(tmp_path)
    state = orchestrator.process_symptom(corrupted_mcp, auto_heal=True, max_auto_tier=1)

    esc = state.get("escalation")
    assert isinstance(esc, dict)
    assert esc["reason"] == "missing_playbook"
    assert esc["proposed_playbook_id"].startswith("heal_invalid_json")
    assert "playbooks.py" in esc["proposed_feature_spec"]
    assert "Corrupted MCP Configuration JSON" in esc["agent_prompt"]
    assert esc["notified"] is True

    stored = orchestrator.memory.recent_escalations(limit=5)
    assert len(stored) == 1
    assert stored[0]["reason"] == "missing_playbook"
    assert stored[0]["proposed_playbook_id"] == esc["proposed_playbook_id"]


def test_capability_escalator_prompts_when_remediation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "send_escalation_notification", lambda *_a, **_k: True)
    symptom = _memory_symptom()
    probes = _CountingProbeRunner([symptom])
    runner = _RecordingPlaybookRunner(success=False)
    orchestrator = _orchestrator(tmp_path, probe_runner=probes, playbook_runner=runner)

    state = orchestrator.process_symptom(symptom, auto_heal=True, max_auto_tier=1)

    assert state["status"] == "unresolved"
    esc = state.get("escalation")
    assert isinstance(esc, dict)
    assert esc["reason"] == "remediation_failed"
    assert esc["proposed_playbook_id"] == "reclaim_memory_caches"
    assert "boom" in esc["missing_capability"]
    assert "reclaim_memory_caches" in esc["agent_prompt"]
    assert len(orchestrator.memory.recent_escalations(limit=5)) == 1


def test_escalation_and_notifications_are_pure_english_without_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_notifications: list[tuple[str, str]] = []

    def _fake_escalation_notify(title: str, body: str) -> bool:
        captured_notifications.append((title, body))
        return True

    monkeypatch.setattr(graph_module, "send_escalation_notification", _fake_escalation_notify)

    corrupted_mcp = SymptomEvent.create(
        probe_name="mcp_and_systemd",
        domain="mcp_config",
        fingerprint_key="invalid_json:/home/user/.gemini/antigravity/mcp_config.json",
        title="Corrupted MCP Configuration JSON",
        summary="Failed to parse mcp_config.json: Expecting property name enclosed in double quotes",
        metrics={
            "config_path": "/home/user/.gemini/antigravity/mcp_config.json",
            "error_count": 1,
        },
        raw_evidence={"error": "SyntaxError"},
        suggested_playbook="inspect_only",
        default_tier=2,
    )
    orchestrator = _orchestrator(tmp_path)
    state = orchestrator.process_symptom(corrupted_mcp, auto_heal=True, max_auto_tier=1)

    esc = state.get("escalation")
    assert isinstance(esc, dict)
    # Check that prompt, missing capability, and feature spec are pure English (no JSON syntax)
    for field in ("agent_prompt", "missing_capability", "proposed_feature_spec"):
        val = esc[field]
        assert "{" not in val, f"Field {field} contains JSON brace: {val}"
        assert "}" not in val, f"Field {field} contains JSON brace: {val}"
        assert '":' not in val, f"Field {field} contains JSON key-value: {val}"

    # Verify notification has no JSON braces or syntax, and has clean tags
    assert len(captured_notifications) == 1
    title, body = captured_notifications[0]
    assert "{" not in title and "}" not in title
    assert "{" not in body and "}" not in body
    assert title.startswith("[Add Capability]")
    assert "Tag: Implement feature (Add capability to me)" in body
    assert "What is needed:" in body
    assert "Action required:" in body
    assert "config path '/home/user/.gemini/antigravity/mcp_config.json'" in esc["agent_prompt"]


def test_capability_escalator_prompts_when_blocked_by_safety_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(graph_module, "send_escalation_notification", lambda *_a, **_k: True)
    tier1_symptom = SymptomEvent.create(
        probe_name="mcp_and_systemd",
        domain="systemd_service",
        fingerprint_key="failed_unit:omarchy-ai.service",
        title="Failed Systemd User Service: omarchy-ai.service",
        summary="Service failed",
        suggested_playbook="restart_systemd_user_unit",
        suggested_args={"unit": "omarchy-ai.service"},
        default_tier=1,
    )
    orchestrator = _orchestrator(tmp_path)
    state = orchestrator.process_symptom(tier1_symptom, auto_heal=True, max_auto_tier=0)

    assert state["status"] == "awaiting_approval"
    esc = state.get("escalation")
    assert isinstance(esc, dict)
    assert esc["reason"] == "insufficient_autonomy_tier"
    assert esc["proposed_playbook_id"] == "restart_systemd_user_unit"
    assert "max_auto_tier=0" in esc["missing_capability"]
    assert len(orchestrator.memory.recent_escalations(limit=5)) == 1


def test_completion_notification_is_pure_english_without_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[tuple[str, str]] = []

    def _fake_completion_notify(title: str, body: str) -> bool:
        captured.append((title, body))
        return True

    monkeypatch.setattr(graph_module, "send_completion_notification", _fake_completion_notify)

    symptom = _osd_symptom()
    runner = _RecordingPlaybookRunner(success=True)
    # Runner succeeds, probe runner reports no remaining symptoms
    orchestrator = _orchestrator(
        tmp_path, probe_runner=_CountingProbeRunner([]), playbook_runner=runner
    )

    state = orchestrator.process_symptom(symptom, auto_heal=True, max_auto_tier=1)
    assert state["status"] == "healed"

    assert len(captured) == 1
    title, body = captured[0]
    assert "{" not in title and "}" not in title
    assert "{" not in body and "}" not in body
    assert title.startswith("[Issue Resolved]")
    assert "Tag: Issue Resolved" in body
    assert "What was done:" in body
    assert "Result:" in body
    assert "Resolved with close stuck osd" in body
    assert "Stuck Omarchy OSD Overlay Popup" in body


def test_executor_retries_with_alternative_playbook(tmp_path: Path) -> None:
    executed_pbs: list[str] = []

    def mock_runner(pb_id: str, _args: dict[str, Any]) -> ExecutionResult:
        executed_pbs.append(pb_id)
        # Fail the first one (reclaim_memory_caches), succeed on alternative (clean_user_memory_caches)
        if pb_id == "reclaim_memory_caches":
            return ExecutionResult(executed=True, playbook_id=pb_id, success=False, stderr="failed", duration_ms=1)
        return ExecutionResult(executed=True, playbook_id=pb_id, success=True, stdout="cleaned", duration_ms=1)

    symptom = SymptomEvent.create(
        probe_name="memory_and_process",
        domain="memory_pressure",
        fingerprint_key="high_ram_test",
        title="Elevated RAM",
        summary="RAM high",
        suggested_playbook="reclaim_memory_caches",
        default_tier=0,
    )

    orchestrator = _orchestrator(
        tmp_path,
        probe_runner=_CountingProbeRunner([]),
        playbook_runner=mock_runner,
    )

    state = orchestrator.process_symptom(symptom, auto_heal=True, max_auto_tier=1)
    assert "reclaim_memory_caches" in executed_pbs
    assert "clean_user_memory_caches" in executed_pbs
    assert state["execution"]["playbook_id"] == "clean_user_memory_caches"
    assert state["execution"]["success"] is True


def test_diagnostician_avoids_recently_failed_playbook(tmp_path: Path) -> None:
    from omarchy_healer.models import HealerGraphState

    orchestrator = _orchestrator(tmp_path)
    symptom = SymptomEvent.create(
        probe_name="memory_and_process",
        domain="memory_pressure",
        fingerprint_key="ram_leak",
        title="High RAM",
        summary="Memory high",
        suggested_playbook="reclaim_memory_caches",
        default_tier=0,
    )

    # Record a prior failed incident with reclaim_memory_caches
    state: HealerGraphState = {
        "symptom": symptom.model_dump(),
        "triage": {"provider": "heuristic", "domain": "memory_pressure", "selected_playbook": "reclaim_memory_caches", "safety_tier": 0},
        "diagnosis": {"provider": "heuristic", "recommended_playbook": "reclaim_memory_caches", "safety_tier": 0},
        "plan": {"playbook_id": "reclaim_memory_caches", "safety_tier": 0, "approved": True},
        "execution": {"executed": True, "playbook_id": "reclaim_memory_caches", "success": False},
        "verification": {"verified": True, "resolved": False},
        "status": "execution_failed",
        "agent_trace": [],
    }
    orchestrator.memory.record_state(state)

    # Diagnostician should see that reclaim_memory_caches failed recently for this fingerprint
    failures = orchestrator.memory.find_recent_failures(symptom.fingerprint)
    assert "reclaim_memory_caches" in failures




