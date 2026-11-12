"""Unit tests for omarchy_healer.formatting plain English helpers."""
from __future__ import annotations

import json
from omarchy_healer.formatting import (
    clean_english_text,
    format_completion_notification,
    format_escalation_notification,
    format_metrics_english,
    format_tool_evidence_english,
)


def test_format_metrics_english_standard_process() -> None:
    metrics = {
        "pid": 178812,
        "signal": "SIGSEGV",
        "unit": "omarchy-shell.service",
        "program": "omarchy-shell",
    }
    result = format_metrics_english(metrics)
    assert "{" not in result
    assert "}" not in result
    assert "process ID 178812" in result
    assert "signal SIGSEGV" in result
    assert "service unit omarchy-shell.service" in result
    assert "program omarchy-shell" in result


def test_format_metrics_english_memory_and_psi() -> None:
    metrics = {
        "used_percent": 88.5,
        "available_mb": 1200.0,
        "total_mb": 16000.0,
        "some_avg10": 15.2,
        "critical": False,
    }
    result = format_metrics_english(metrics)
    assert "{" not in result
    assert "}" not in result
    assert "memory used 88.5%" in result
    assert "1200 MiB memory available" in result
    assert "16000 MiB total memory" in result
    assert "PSI pressure avg10: 15.2" in result
    assert "severity critical: false" in result


def test_format_metrics_english_empty_and_primitives() -> None:
    assert format_metrics_english({}) == "none reported"
    assert format_metrics_english(None) == "none reported"
    assert format_metrics_english(["item1", "item2"]) == "item1, item2"
    assert format_metrics_english("already plain text") == "already plain text"


def test_clean_english_text_removes_json_code_blocks() -> None:
    raw = "Diagnosed problem:\n```json\n{\"pid\": 42, \"status\": \"crashed\"}\n```\nPlease investigate."
    cleaned = clean_english_text(raw)
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "```" not in cleaned
    assert "process ID 42" in cleaned


def test_clean_english_text_removes_inline_json_objects() -> None:
    raw = 'Symptom summary: Process died | Metrics: {"pid": 1234, "signal": "SIGKILL"}. Please fix.'
    cleaned = clean_english_text(raw)
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "process ID 1234" in cleaned
    assert "signal SIGKILL" in cleaned


def test_format_tool_evidence_english() -> None:
    evidence = {
        "inspect_memory_and_top_processes": "PID 1 init RSS 10MB\nPID 200 app RSS 500MB",
        "inspect_mcp_config": "Config is valid with 3 active servers",
    }
    result = format_tool_evidence_english(evidence)
    assert "{" not in result
    assert "}" not in result
    assert "- Inspect Memory And Top Processes:" in result
    assert "- Inspect Mcp Config:" in result
    assert "PID 1 init" in result


def test_format_completion_notification_has_tags_and_clean_description() -> None:
    title, body = format_completion_notification(
        "Stuck Omarchy OSD Overlay Popup",
        "close_stuck_osd",
        "Issue 'Stuck Omarchy OSD Overlay Popup' cleared after close_stuck_osd.",
    )
    assert title.startswith("[Issue Resolved]")
    assert "Stuck Omarchy OSD Overlay Popup" in title
    assert "Tag: Issue Resolved" in body
    assert "What was done:" in body
    assert "close stuck osd" in body
    assert "Result:" in body
    assert "{" not in title and "}" not in title
    assert "{" not in body and "}" not in body


def test_format_escalation_notification_tags_and_needs() -> None:
    # 1. Missing playbook -> Add capability / implement feature tag
    t1, b1 = format_escalation_notification(
        reason="missing_playbook",
        symptom_title="Corrupted MCP Configuration JSON",
        domain="mcp_config",
        proposed_playbook_id="heal_invalid_json",
    )
    assert t1.startswith("[Add Capability]")
    assert "Tag: Implement feature (Add capability to me)" in b1
    assert "What is needed:" in b1
    assert "Action required:" in b1
    assert "heal_invalid_json" in b1

    # 2. Insufficient tier -> Permission required tag
    t2, b2 = format_escalation_notification(
        reason="insufficient_autonomy_tier",
        symptom_title="Failed Systemd User Service",
        domain="systemd_service",
        proposed_playbook_id="restart_systemd_user_unit",
        plan_tier=1,
        max_tier=0,
    )
    assert t2.startswith("[Permission Required]")
    assert "Tag: Permission required (Higher tier approval)" in b2
    assert "What is needed:" in b2
    assert "Action required:" in b2
    assert "safety tier 1" in b2
    assert "max auto tier 0" in b2

    # 3. Remediation failed -> Upgrade capability tag
    t3, b3 = format_escalation_notification(
        reason="remediation_failed",
        symptom_title="High System RAM Usage",
        domain="memory_pressure",
        proposed_playbook_id="reclaim_memory_caches",
        failure_detail="memory remained at 88%",
    )
    assert t3.startswith("[Upgrade Capability]")
    assert "Tag: Upgrade capability (Feature enhancement needed)" in b3
    assert "What is needed:" in b3
    assert "Action required:" in b3
    assert "memory remained at 88%" in b3


def test_format_improvement_agent_prompt() -> None:
    from omarchy_healer.formatting import format_improvement_agent_prompt

    prompt = format_improvement_agent_prompt(
        incident_id=101,
        title="High System RAM Usage",
        domain="memory_pressure",
        reason="remediation_failed",
        missing_capability="Adaptive memory compaction and cache pressure management",
        proposed_playbook_id="compact_memory_slabs",
        proposed_feature_spec="Implement slab compaction in playbooks.py",
        failure_detail="memory remained at 92%",
        agent_prompt_override="Custom prompt override",
    )
    assert "Omarchy-Healer" in prompt
    assert "Incident ID: #101" in prompt
    assert "High System RAM Usage" in prompt
    assert "memory_pressure" in prompt
    assert "compact_memory_slabs" in prompt
    assert "uv run pytest" in prompt
    assert "src/omarchy_healer/playbooks.py" in prompt


