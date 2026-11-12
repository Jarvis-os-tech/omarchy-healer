"""Unit and integration tests for Omarchy-Healer multi-agent workflow and provider fallbacks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from omarchy_healer.config import HealerConfig
from omarchy_healer.graph import OmarchyHealerOrchestrator
from omarchy_healer.triage_judge import TriageJudge
from omarchy_healer.llm_router import LLMFallbackRouter
from omarchy_healer.models import ExecutionResult, SymptomEvent
from omarchy_healer.playbooks import execute_playbook
from omarchy_healer.probes import McpAndServiceProbe


class _FakeLLM:
    def __init__(self, response_dict: dict[str, Any] | None = None, *, error: Exception | None = None):
        self.response_dict = response_dict
        self.error = error
        self.calls = 0

    def invoke(self, _messages: Any) -> AIMessage:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return AIMessage(content=json.dumps(self.response_dict or {}))


def _sample_osd_symptom() -> SymptomEvent:
    return SymptomEvent.create(
        probe_name="omarchy_ui",
        domain="ui_shell",
        fingerprint_key="stuck_omarchy_osd",
        title="Stuck Omarchy OSD Overlay Popup",
        summary="omarchy-osd layer surface is stuck open ('Launching Antigravity...').",
        metrics={"osd_state": "open", "osd_layer_present": True},
        suggested_playbook="close_stuck_osd",
        default_tier=0,
    )


def test_nvidia_nim_primary_triage_success(tmp_path: Path) -> None:
    cfg = HealerConfig(
        nvidia_api_key="test-nvidia-key",
        db_path=tmp_path / "incidents.db",
    )
    nvidia_fake = _FakeLLM(
        {
            "is_actionable": True,
            "actionable_probability": 0.95,
            "severity_score": 0.45,
            "selected_playbook": "close_stuck_osd",
            "safety_tier": 0,
            "rationale": "NVIDIA NIM primary triaged stuck OSD",
        }
    )
    router = LLMFallbackRouter(cfg, custom_invokers=[("nvidia_nim", nvidia_fake)])
    judge = TriageJudge(cfg, llm_router=router)
    decision = judge.evaluate(_sample_osd_symptom())

    assert decision.provider == "nvidia_nim"
    assert decision.is_actionable is True
    assert decision.selected_playbook == "close_stuck_osd"
    assert decision.safety_tier == 0
    assert decision.fallback_chain_tried == ["nvidia_nim"]
    assert nvidia_fake.calls == 1


def test_fallback_to_gemini_when_nvidia_nim_fails(tmp_path: Path) -> None:
    cfg = HealerConfig(
        nvidia_api_key="test-nvidia-key",
        gemini_api_key="test-gemini-key",
        db_path=tmp_path / "incidents.db",
    )
    nvidia_failing = _FakeLLM(error=RuntimeError("NVIDIA NIM 500 error"))
    gemini_working = _FakeLLM(
        {
            "is_actionable": True,
            "actionable_probability": 0.92,
            "severity_score": 0.5,
            "selected_playbook": "close_stuck_osd",
            "safety_tier": 0,
            "rationale": "Gemini fallback triaged stuck OSD",
        }
    )
    router = LLMFallbackRouter(
        cfg,
        custom_invokers=[
            ("nvidia_nim", nvidia_failing),
            ("gemini", gemini_working),
        ],
    )
    judge = TriageJudge(cfg, llm_router=router)
    decision = judge.evaluate(_sample_osd_symptom())

    assert decision.provider == "gemini"
    assert decision.selected_playbook == "close_stuck_osd"
    assert decision.fallback_chain_tried == ["nvidia_nim", "gemini"]
    assert nvidia_failing.calls == 1
    assert gemini_working.calls == 1


def test_fallback_to_heuristic_when_all_llms_fail(tmp_path: Path) -> None:
    cfg = HealerConfig(
        nvidia_api_key="test-nvidia-key",
        gemini_api_key="test-gemini-key",
        db_path=tmp_path / "incidents.db",
    )
    nvidia_failing = _FakeLLM(error=RuntimeError("NVIDIA NIM error"))
    gemini_failing = _FakeLLM(error=RuntimeError("Gemini error"))

    router = LLMFallbackRouter(
        cfg,
        custom_invokers=[
            ("nvidia_nim", nvidia_failing),
            ("gemini", gemini_failing),
        ],
    )
    judge = TriageJudge(cfg, llm_router=router)
    decision = judge.evaluate(_sample_osd_symptom())

    assert decision.provider == "heuristic"
    assert decision.selected_playbook == "close_stuck_osd"
    assert decision.fallback_chain_tried == ["nvidia_nim", "gemini", "heuristic"]

def test_mcp_probe_defers_to_user_when_binary_is_unrepairable(tmp_path: Path) -> None:
    """A missing MCP binary is a deployment problem, so Tier 1 must not 'fix' it by deletion."""
    mcp_file = tmp_path / "mcp.json"
    mcp_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "codebase-memory-mcp": {
                        "command": "/nonexistent/path/to/codebase-memory-mcp",
                        "args": [],
                        "type": "stdio",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = HealerConfig(mcp_config_path=mcp_file, db_path=tmp_path / "incidents.db")
    probe = McpAndServiceProbe(cfg)
    events = probe._check_mcp_config()

    assert len(events) == 1
    ev = events[0]
    assert ev.domain == "mcp_config"
    assert ev.suggested_playbook == "repair_or_disable_mcp_server"

    res = execute_playbook(ev.suggested_playbook, ev.suggested_args)
    assert res.success is False
    assert res.backup_path is None

    # Nothing was removed: the entry is still configured and still reported, so the
    # escalation path can ask the user to install it or run with --max-tier 2.
    updated = json.loads(mcp_file.read_text(encoding="utf-8"))
    assert "codebase-memory-mcp" in updated["mcpServers"]
    assert "_disabled_mcpServers" not in updated
    assert len(probe._check_mcp_config()) == 1


def test_mcp_quarantine_moves_entry_only_when_approved(tmp_path: Path) -> None:
    """Tier 2 quarantine still works once a human opts in with --max-tier 2."""
    mcp_file = tmp_path / "mcp.json"
    mcp_file.write_text(
        json.dumps({"mcpServers": {"gone": {"command": "/nonexistent/gone"}}}),
        encoding="utf-8",
    )

    res = execute_playbook(
        "quarantine_mcp_server", {"config_path": str(mcp_file), "server_name": "gone"}
    )

    assert res.success is True
    assert res.backup_path is not None and Path(res.backup_path).is_file()
    updated = json.loads(mcp_file.read_text(encoding="utf-8"))
    assert "gone" not in updated["mcpServers"]
    assert "gone" in updated["_disabled_mcpServers"]


def test_langgraph_end_to_end_auto_heal_and_verify(tmp_path: Path) -> None:
    cfg = HealerConfig(max_auto_tier=1, db_path=tmp_path / "incidents.db")
    osd_open = {"state": True}

    def fake_probes(_cfg: HealerConfig) -> list[SymptomEvent]:
        return [_sample_osd_symptom()] if osd_open["state"] else []

    def fake_playbook_runner(playbook_id: str, _args: dict[str, Any]) -> ExecutionResult:
        if playbook_id == "close_stuck_osd":
            osd_open["state"] = False
            return ExecutionResult(
                executed=True,
                playbook_id=playbook_id,
                success=True,
                stdout="closed osd",
                duration_ms=5,
            )
        return ExecutionResult(executed=False, playbook_id=playbook_id, success=False)

    orchestrator = OmarchyHealerOrchestrator(
        cfg,
        probe_runner=fake_probes,
        playbook_runner=fake_playbook_runner,
    )

    results = orchestrator.scan_and_process(auto_heal=True)
    assert len(results) == 1
    final_state = results[0]
    assert final_state["status"] == "healed"
    assert final_state["verification"]["resolved"] is True

    history = orchestrator.memory.recent_incidents(limit=5)
    assert len(history) == 1
    assert history[0]["status"] == "healed"
    assert history[0]["resolved"] == 1


def test_safety_gate_blocks_higher_tier_when_max_tier_exceeded(tmp_path: Path) -> None:
    cfg = HealerConfig(max_auto_tier=0, db_path=tmp_path / "incidents.db")
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

    orchestrator = OmarchyHealerOrchestrator(cfg)
    state = orchestrator.process_symptom(tier1_symptom, auto_heal=True, max_auto_tier=0)
    assert state["status"] == "awaiting_approval"
    assert state["plan"]["approved"] is False
    assert "execution" not in state
