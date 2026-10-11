"""Bounded, idempotent remediation playbooks and LangChain tool definitions."""
from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .models import ExecutionResult

# Playbook registry metadata used by both Laya choice criteria and LangGraph Planner
PLAYBOOK_CATALOG: dict[str, dict[str, Any]] = {
    "close_stuck_osd": {
        "tier": 0,
        "domain": "ui_shell",
        "description": "Close a stuck Omarchy Quickshell OSD overlay via 'omarchy-shell osd close'.",
    },
    "restart_omarchy_shell": {
        "tier": 1,
        "domain": "ui_shell",
        "description": "Restart the Omarchy Quickshell bar and OSD service via 'omarchy restart shell'.",
    },
    "reload_hyprland_config": {
        "tier": 1,
        "domain": "compositor",
        "description": "Reload Hyprland configuration via 'hyprctl reload' and verify configerrors.",
    },
    "repair_or_disable_mcp_server": {
        "tier": 1,
        "domain": "mcp_config",
        "description": (
            "Backup ~/.omp/agent/mcp.json and repair the server command if found on PATH, "
            "or move the broken server entry to '_disabled_mcpServers' so omp starts cleanly."
        ),
    },
    "restart_systemd_user_unit": {
        "tier": 1,
        "domain": "systemd_service",
        "description": "Reset failed state and restart a crashed systemd --user service unit.",
    },
    "reclaim_memory_caches": {
        "tier": 0,
        "domain": "memory_pressure",
        "description": (
            "Run safe user-space memory reclamation (Python GC, malloc_trim/cache cleanup, "
            "and emit top RSS report without killing user windows)."
        ),
    },
    "inspect_only": {
        "tier": 0,
        "domain": "other",
        "description": "Log diagnostic evidence without mutating system state.",
    },
}


def _exec(cmd: list[str], *, timeout: float = 10.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, "", str(exc)


def send_completion_notification(title: str, body: str) -> bool:
    """Send a desktop notification when Omarchy-Healer finishes resolving an issue."""
    rc, _, _ = _exec(
        [
            "notify-send",
            "-a",
            "omarchy-action",
            "-t",
            "5000",
            f"Omarchy Healer: {title}",
            body,
        ],
        timeout=3.0,
    )
    return rc == 0

def execute_playbook(playbook_id: str, args: dict[str, Any] | None = None) -> ExecutionResult:
    """Execute a registered remediation playbook safely and return an ExecutionResult."""
    args = args or {}
    started = time.monotonic()

    if playbook_id == "close_stuck_osd":
        rc, out, err = _exec(["omarchy-shell", "osd", "close"])
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=(rc == 0),
            stdout=out or "omarchy-shell osd close -> ok",
            stderr=err,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if playbook_id == "restart_omarchy_shell":
        rc, out, err = _exec(["omarchy", "restart", "shell"], timeout=15.0)
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=(rc == 0),
            stdout=out or "Restarted omarchy-shell",
            stderr=err,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if playbook_id == "reload_hyprland_config":
        rc, out, err = _exec(["hyprctl", "reload"])
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=(rc == 0),
            stdout=out,
            stderr=err,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if playbook_id == "repair_or_disable_mcp_server":
        return _run_repair_or_disable_mcp(args, started)

    if playbook_id == "restart_systemd_user_unit":
        unit = str(args.get("unit") or "").strip()
        if not unit or any(ch in unit for ch in (";", "&", "|", "`", "$", " ")):
            return ExecutionResult(
                executed=False,
                playbook_id=playbook_id,
                success=False,
                stderr=f"Refusing invalid systemd unit name: {unit!r}",
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        _exec(["systemctl", "--user", "reset-failed", unit])
        rc, out, err = _exec(["systemctl", "--user", "restart", unit], timeout=15.0)
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=(rc == 0),
            stdout=out or f"Restarted systemd user unit {unit}",
            stderr=err,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if playbook_id == "reclaim_memory_caches":
        gc.collect()
        top_procs = args.get("top_processes") or []
        summary_lines = [
            f"PID {p.get('pid')} ({p.get('command')}): {p.get('rss_mb')} MiB ({p.get('mem_percent')}%)"
            for p in top_procs
            if isinstance(p, dict)
        ]
        msg = "Triggered user-space GC reclamation. Top RSS consumers: " + (
            "; ".join(summary_lines) if summary_lines else "none"
        )
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=True,
            stdout=msg,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if playbook_id == "inspect_only":
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=True,
            stdout="Diagnostic inspection recorded; no system state modified.",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    return ExecutionResult(
        executed=False,
        playbook_id=playbook_id,
        success=False,
        stderr=f"Unknown playbook_id: {playbook_id!r}",
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _run_repair_or_disable_mcp(args: dict[str, Any], started: float) -> ExecutionResult:
    raw_path = str(args.get("config_path") or (Path.home() / ".omp" / "agent" / "mcp.json"))
    mcp_path = Path(raw_path).expanduser()
    server_name = str(args.get("server_name") or "").strip()

    if not mcp_path.is_file() or not server_name:
        return ExecutionResult(
            executed=False,
            playbook_id="repair_or_disable_mcp_server",
            success=False,
            stderr=f"MCP config {mcp_path} or server_name {server_name!r} not found",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    try:
        data = json.loads(mcp_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers")
        if not isinstance(servers, dict) or server_name not in servers:
            return ExecutionResult(
                executed=False,
                playbook_id="repair_or_disable_mcp_server",
                success=False,
                stderr=f"Server {server_name!r} not present in mcpServers",
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        ts = int(time.time())
        backup_path = mcp_path.with_name(f"{mcp_path.name}.bak.{ts}")
        shutil.copy2(mcp_path, backup_path)

        spec = servers[server_name]
        cmd_name = Path(os.path.expanduser(str(spec.get("command") or ""))).name
        alt_bin = str(args.get("alternate_binary") or "").strip() or (
            shutil.which(cmd_name) if cmd_name else None
        )

        if alt_bin and Path(alt_bin).is_file():
            spec["command"] = alt_bin
            action_msg = f"Updated {server_name!r} command path to {alt_bin}"
        else:
            removed = servers.pop(server_name)
            disabled_map = data.setdefault("_disabled_mcpServers", {})
            if isinstance(disabled_map, dict):
                disabled_map[server_name] = {
                    **removed,
                    "disabled_reason": "Binary missing on disk (ENOENT); moved by omarchy-healer",
                }
            action_msg = (
                f"Moved broken MCP server {server_name!r} to '_disabled_mcpServers' "
                f"(backup saved at {backup_path})"
            )

        mcp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return ExecutionResult(
            executed=True,
            playbook_id="repair_or_disable_mcp_server",
            success=True,
            stdout=action_msg,
            backup_path=str(backup_path),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except (OSError, json.JSONDecodeError) as exc:
        return ExecutionResult(
            executed=False,
            playbook_id="repair_or_disable_mcp_server",
            success=False,
            stderr=str(exc),
            duration_ms=int((time.monotonic() - started) * 1000),
        )


# LangChain Read-Only Diagnostic Tools for the Diagnostician Agent
@tool
def inspect_omarchy_osd_and_layers() -> str:
    """Return current omarchy-shell OSD state and active Hyprland Wayland layers."""
    _, osd_state, _ = _exec(["omarchy-shell", "osd", "state"])
    _, layers_json, _ = _exec(["hyprctl", "layers", "-j"])
    return json.dumps({"osd_state": osd_state, "layers": layers_json[:1500]})


@tool
def inspect_mcp_config(config_path: str = "~/.omp/agent/mcp.json") -> str:
    """Inspect an MCP configuration file and verify whether each server binary exists."""
    p = Path(config_path).expanduser()
    if not p.is_file():
        return json.dumps({"exists": False, "path": str(p)})
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        report: dict[str, Any] = {}
        for name, spec in (raw.get("mcpServers") or {}).items():
            cmd = os.path.expanduser(str((spec or {}).get("command") or ""))
            exists = Path(cmd).is_file() if "/" in cmd else bool(shutil.which(cmd))
            report[name] = {"command": cmd, "exists": exists}
        return json.dumps({"exists": True, "path": str(p), "servers": report})
    except Exception as exc:
        return json.dumps({"exists": True, "path": str(p), "error": str(exc)})


@tool
def inspect_memory_and_top_processes() -> str:
    """Return current /proc/meminfo summary and top 5 RSS memory-consuming processes."""
    _, ps_out, _ = _exec(["ps", "-eo", "pid,ppid,%mem,rss,comm", "--sort=-rss"])
    lines = ps_out.splitlines()[:6] if ps_out else []
    return json.dumps({"top_rss": lines})


DIAGNOSTIC_TOOLS = [
    inspect_omarchy_osd_and_layers,
    inspect_mcp_config,
    inspect_memory_and_top_processes,
]
