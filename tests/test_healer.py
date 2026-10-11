"""Unit and integration tests for Omarchy-Healer multi-agent workflow and provider fallbacks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from omarchy_healer.config import HealerConfig
from omarchy_healer.graph import OmarchyHealerOrchestrator
from omarchy_healer.laya_judge import LayaTriageJudge
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


def test_laya_primary_triage_success(tmp_path: Path) -> None:
    cfg = HealerConfig(
        laya_endpoint_url="https://laya.example/v1/systemone",
        laya_api_key="test-laya-key",
        db_path=tmp_path / "incidents.db",
    )

    def fake_laya_http(state: str, questions: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        assert "stuck_omarchy_osd" in state or "Stuck Omarchy OSD" in state
        assert set(questions.keys()) == {"actionable", "playbook", "severity"}
        return {
            "answers": {
                "actionable": {"type": "noul", "noul": 0.97},
                "playbook": {
                    "type": "choice",
                    "choice": "close_stuck_osd",
                    "probabilities": {"close_stuck_osd": 0.94, "inspect_only": 0.06},
                },
                "severity": {"type": "score", "score": 1.2},
            }
        }

    judge = LayaTriageJudge(cfg, http_evaluator=fake_laya_http)
    decision = judge.evaluate(_sample_osd_symptom())

    assert decision.provider == "laya"
    assert decision.is_actionable is True
    assert decision.selected_playbook == "close_stuck_osd"
    assert decision.safety_tier == 0
    assert decision.fallback_chain_tried == ["laya"]


def test_fallback_to_gemini_when_laya_fails(tmp_path: Path) -> None:
    cfg = HealerConfig(
        laya_endpoint_url="https://laya.example/v1/systemone",
        laya_api_key="test-laya-key",
        gemini_api_key="test-gemini-key",
        db_path=tmp_path / "incidents.db",
    )

    def failing_laya_http(_state: str, _questions: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        raise RuntimeError("Laya endpoint 503 unavailable")

    gemini_fake = _FakeLLM(
        {
            "is_actionable": True,
            "actionable_probability": 0.93,
            "severity_score": 0.4,
            "selected_playbook": "close_stuck_osd",
            "safety_tier": 0,
            "rationale": "Gemini fallback selected close_stuck_osd",
        }
    )
    router = LLMFallbackRouter(cfg, custom_invokers=[("gemini", gemini_fake)])
    judge = LayaTriageJudge(cfg, llm_router=router, http_evaluator=failing_laya_http)

    decision = judge.evaluate(_sample_osd_symptom())
    assert decision.provider == "gemini"
    assert decision.selected_playbook == "close_stuck_osd"
    assert decision.fallback_chain_tried == ["laya", "gemini"]
    assert gemini_fake.calls == 1


def test_fallback_to_nvidia_nim_when_laya_and_gemini_fail(tmp_path: Path) -> None:
    cfg = HealerConfig(
        laya_endpoint_url="https://laya.example/v1/systemone",
        laya_api_key="test-laya-key",
        gemini_api_key="test-gemini-key",
        nvidia_api_key="test-nvidia-key",
        db_path=tmp_path / "incidents.db",
    )

    def failing_laya_http(_state: str, _questions: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
        raise RuntimeError("Laya connection refused")

    gemini_failing = _FakeLLM(error=RuntimeError("Gemini 429 quota exceeded"))
    nvidia_working = _FakeLLM(
        {
            "is_actionable": True,
            "actionable_probability": 0.91,
            "severity_score": 0.55,
            "selected_playbook": "close_stuck_osd",
            "safety_tier": 0,
            "rationale": "NVIDIA NIM fallback triaged stuck OSD",
        }
    )

    router = LLMFallbackRouter(
        cfg,
        custom_invokers=[
            ("gemini", gemini_failing),
            ("nvidia_nim", nvidia_working),
        ],
    )
    judge = LayaTriageJudge(cfg, llm_router=router, http_evaluator=failing_laya_http)

    decision = judge.evaluate(_sample_osd_symptom())
    assert decision.provider == "nvidia_nim"
    assert decision.selected_playbook == "close_stuck_osd"
    assert decision.fallback_chain_tried == ["laya", "gemini", "nvidia_nim"]
    assert gemini_failing.calls == 1
    assert nvidia_working.calls == 1


def test_mcp_probe_and_repair_playbook_with_backup(tmp_path: Path) -> None:
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

    res = execute_playbook("repair_or_disable_mcp_server", ev.suggested_args)
    assert res.success is True
    assert res.backup_path is not None and Path(res.backup_path).is_file()

    # Re-probing after repair must find 0 broken MCP servers
    post_events = probe._check_mcp_config()
    assert len(post_events) == 0

    updated = json.loads(mcp_file.read_text(encoding="utf-8"))
    assert "codebase-memory-mcp" not in updated["mcpServers"]
    assert "codebase-memory-mcp" in updated["_disabled_mcpServers"]


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
