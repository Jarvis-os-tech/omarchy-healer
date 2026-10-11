"""Deterministic system monitors for Omarchy UI/Hyprland, MCP/Systemd, and Memory/Processes."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import HealerConfig
from .models import SymptomEvent


def run_cmd(cmd: list[str], *, timeout: float = 4.0) -> tuple[int, str, str]:
    """Run a read-only diagnostic command with a hard timeout."""
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


class OmarchyUiProbe:
    """Monitors Omarchy Quickshell OSD overlays, status bar layers, and Hyprland config errors."""

    name = "omarchy_ui"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config
        self._osd_first_seen: float | None = None

    def check(self, *, osd_stuck_seconds: float = 0.0) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []

        # 1. Check omarchy-shell osd state + hyprctl layers for stuck OSD popup
        rc_osd, out_osd, _ = run_cmd(["omarchy-shell", "osd", "state"])
        rc_layers, out_layers, _ = run_cmd(["hyprctl", "layers", "-j"])

        osd_layer_present = False
        bar_layer_present = False
        layers_data: dict[str, Any] = {}
        if rc_layers == 0 and out_layers:
            try:
                layers_data = json.loads(out_layers)
                for monitor_data in layers_data.values():
                    levels = (monitor_data or {}).get("levels") or {}
                    for level_items in levels.values():
                        for item in level_items or []:
                            ns = item.get("namespace", "")
                            if ns == "omarchy-osd":
                                osd_layer_present = True
                            elif ns == "omarchy-bar":
                                bar_layer_present = True
            except json.JSONDecodeError:
                pass

        osd_is_open = (rc_osd == 0 and out_osd == "open") or osd_layer_present
        if osd_is_open:
            now = time.monotonic()
            if self._osd_first_seen is None:
                self._osd_first_seen = now
            elapsed = now - self._osd_first_seen
            if elapsed >= osd_stuck_seconds:
                events.append(
                    SymptomEvent.create(
                        probe_name=self.name,
                        domain="ui_shell",
                        fingerprint_key="stuck_omarchy_osd",
                        title="Stuck Omarchy OSD Overlay Popup",
                        summary=(
                            "omarchy-osd layer surface is open on screen (e.g. 'Launching ...' "
                            "with duration=0 that missed its close IPC)."
                        ),
                        metrics={
                            "osd_state": out_osd or "layer_active",
                            "osd_layer_present": osd_layer_present,
                            "observed_open_seconds": round(elapsed, 1),
                        },
                        raw_evidence={"osd_state": out_osd, "layers": layers_data},
                        suggested_playbook="close_stuck_osd",
                        default_tier=0,
                    )
                )
        else:
            self._osd_first_seen = None

        # 2. Check if Hyprland is up but omarchy-bar layer is missing
        if rc_layers == 0 and layers_data and not bar_layer_present:
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="ui_shell",
                    fingerprint_key="missing_omarchy_bar",
                    title="Omarchy Shell Bar Layer Missing",
                    summary="Hyprland compositor is active but the 'omarchy-bar' Quickshell layer is not mapped.",
                    metrics={"bar_layer_present": False},
                    raw_evidence={"layers": layers_data},
                    suggested_playbook="restart_omarchy_shell",
                    default_tier=1,
                )
            )

        # 3. Check hyprctl configerrors
        rc_err, out_err, _ = run_cmd(["hyprctl", "configerrors"])
        if rc_err == 0 and out_err and out_err.strip():
            error_lines = [line.strip() for line in out_err.splitlines() if line.strip()]
            if error_lines:
                events.append(
                    SymptomEvent.create(
                        probe_name=self.name,
                        domain="compositor",
                        fingerprint_key="hyprland_config_errors",
                        title="Hyprland Configuration Errors Detected",
                        summary=f"hyprctl reported {len(error_lines)} configuration error(s): {error_lines[0][:120]}",
                        metrics={"error_count": len(error_lines)},
                        raw_evidence={"errors": error_lines},
                        suggested_playbook="reload_hyprland_config",
                        default_tier=1,
                    )
                )

        return events


class McpAndServiceProbe:
    """Monitors MCP server configuration files and systemd --user services."""

    name = "mcp_and_systemd"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config

    def check(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        events.extend(self._check_mcp_config())
        events.extend(self._check_systemd_user())
        return events

    def _check_mcp_config(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        mcp_path = self.config.mcp_config_path
        if not mcp_path.is_file():
            return events

        try:
            data = json.loads(mcp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="mcp_config",
                    fingerprint_key=f"invalid_json:{mcp_path}",
                    title="Corrupted MCP Configuration JSON",
                    summary=f"Failed to parse {mcp_path}: {exc}",
                    metrics={"config_path": str(mcp_path)},
                    raw_evidence={"error": str(exc)},
                    suggested_playbook="inspect_only",
                    default_tier=2,
                )
            )
            return events

        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            return events

        for server_name, spec in servers.items():
            if not isinstance(spec, dict) or spec.get("disabled") is True:
                continue
            cmd = str(spec.get("command") or "").strip()
            if not cmd:
                continue
            expanded = os.path.expanduser(cmd)
            exists = (
                Path(expanded).is_file() and os.access(expanded, os.X_OK)
                if "/" in expanded
                else bool(shutil.which(expanded))
            )
            if not exists:
                alt_binary = shutil.which(Path(expanded).name)
                events.append(
                    SymptomEvent.create(
                        probe_name=self.name,
                        domain="mcp_config",
                        fingerprint_key=f"missing_mcp_bin:{server_name}",
                        title=f"MCP Server Binary Missing: {server_name}",
                        summary=(
                            f"MCP server {server_name!r} in {mcp_path} points to missing binary "
                            f"{expanded!r} (ENOENT on startup)."
                        ),
                        metrics={
                            "server_name": server_name,
                            "configured_command": cmd,
                            "alternate_found": bool(alt_binary),
                        },
                        raw_evidence={
                            "config_path": str(mcp_path),
                            "server_name": server_name,
                            "server_spec": spec,
                            "alternate_binary": alt_binary,
                        },
                        suggested_playbook="repair_or_disable_mcp_server",
                        suggested_args={
                            "config_path": str(mcp_path),
                            "server_name": server_name,
                            "alternate_binary": alt_binary or "",
                        },
                        default_tier=1,
                    )
                )

        return events

    def _check_systemd_user(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        rc, out, _ = run_cmd(
            ["systemctl", "--user", "--failed", "--output=json", "--no-pager"]
        )
        if rc != 0 or not out:
            return events

        try:
            failed_units = json.loads(out)
        except json.JSONDecodeError:
            return events

        if not isinstance(failed_units, list):
            return events

        for unit_info in failed_units:
            if not isinstance(unit_info, dict):
                continue
            unit_name = str(unit_info.get("unit") or "").strip()
            if not unit_name:
                continue
            _, journal_tail, _ = run_cmd(
                ["journalctl", "--user", "-u", unit_name, "-n", "12", "--no-pager"]
            )
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="systemd_service",
                    fingerprint_key=f"failed_unit:{unit_name}",
                    title=f"Failed Systemd User Service: {unit_name}",
                    summary=(
                        f"User unit {unit_name} is in state "
                        f"{unit_info.get('active', 'failed')}/{unit_info.get('sub', 'failed')}."
                    ),
                    metrics={
                        "unit": unit_name,
                        "load": unit_info.get("load", ""),
                        "active": unit_info.get("active", ""),
                        "sub": unit_info.get("sub", ""),
                    },
                    raw_evidence={
                        "unit_info": unit_info,
                        "journal_tail": journal_tail[-1200:],
                    },
                    suggested_playbook="restart_systemd_user_unit",
                    suggested_args={"unit": unit_name},
                    default_tier=1,
                )
            )
        return events


class MemoryProcessProbe:
    """Monitors /proc/meminfo, /proc/pressure/memory PSI, top RSS processes, and zombies."""

    name = "memory_and_process"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config

    def check(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        mem_stats = self._read_meminfo()
        psi_stats = self._read_memory_psi()

        used_pct = mem_stats.get("used_percent", 0.0)
        psi_some10 = psi_stats.get("some_avg10", 0.0)

        if used_pct >= self.config.ram_warn_percent or psi_some10 >= self.config.psi_some_avg10_warn:
            top_procs = self._top_rss_processes(limit=6)
            is_crit = used_pct >= self.config.ram_crit_percent
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="memory_pressure",
                    fingerprint_key="high_ram_pressure",
                    title=f"{'Critical' if is_crit else 'Elevated'} System RAM Usage ({used_pct:.1f}%)",
                    summary=(
                        f"RAM usage is at {used_pct:.1f}% "
                        f"({mem_stats.get('available_mb', 0):.0f} MiB available of "
                        f"{mem_stats.get('total_mb', 0):.0f} MiB; PSI some_avg10={psi_some10:.2f})."
                    ),
                    metrics={
                        **mem_stats,
                        **psi_stats,
                        "critical": is_crit,
                    },
                    raw_evidence={"top_rss_processes": top_procs},
                    suggested_playbook="reclaim_memory_caches",
                    suggested_args={"top_processes": top_procs[:3]},
                    default_tier=0 if not is_crit else 1,
                )
            )

        zombies = self._find_zombies()
        if len(zombies) >= 5:
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="memory_pressure",
                    fingerprint_key="zombie_process_accumulation",
                    title=f"Zombie Process Accumulation ({len(zombies)} defunct)",
                    summary=f"Found {len(zombies)} defunct zombie processes on the host.",
                    metrics={"zombie_count": len(zombies)},
                    raw_evidence={"zombies": zombies[:10]},
                    suggested_playbook="reap_zombie_parents",
                    suggested_args={"zombies": zombies[:10]},
                    default_tier=2,
                )
            )

        return events

    @staticmethod
    def _read_meminfo() -> dict[str, float]:
        info: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val_parts = parts[1].strip().split()
                    if val_parts and val_parts[0].isdigit():
                        info[key] = int(val_parts[0])
        except OSError:
            return {"used_percent": 0.0, "total_mb": 0.0, "available_mb": 0.0}

        total_kb = info.get("MemTotal", 0)
        avail_kb = info.get("MemAvailable", 0)
        if total_kb <= 0:
            return {"used_percent": 0.0, "total_mb": 0.0, "available_mb": 0.0}

        used_kb = max(0, total_kb - avail_kb)
        used_pct = round((used_kb / total_kb) * 100.0, 2)
        return {
            "used_percent": used_pct,
            "total_mb": round(total_kb / 1024.0, 1),
            "available_mb": round(avail_kb / 1024.0, 1),
        }

    @staticmethod
    def _read_memory_psi() -> dict[str, float]:
        result = {"some_avg10": 0.0, "full_avg10": 0.0}
        try:
            for line in Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines():
                tokens = line.split()
                if not tokens:
                    continue
                kind = tokens[0]
                for tok in tokens[1:]:
                    if tok.startswith("avg10="):
                        val = float(tok.split("=", 1)[1])
                        if kind == "some":
                            result["some_avg10"] = val
                        elif kind == "full":
                            result["full_avg10"] = val
        except (OSError, ValueError):
            pass
        return result

    @staticmethod
    def _top_rss_processes(limit: int = 6) -> list[dict[str, Any]]:
        rc, out, _ = run_cmd(
            ["ps", "-eo", "pid,ppid,%mem,rss,comm", "--sort=-rss"]
        )
        if rc != 0 or not out:
            return []
        rows: list[dict[str, Any]] = []
        for line in out.splitlines()[1 : limit + 1]:
            parts = line.split(None, 4)
            if len(parts) == 5:
                try:
                    rows.append(
                        {
                            "pid": int(parts[0]),
                            "ppid": int(parts[1]),
                            "mem_percent": float(parts[2]),
                            "rss_mb": round(int(parts[3]) / 1024.0, 1),
                            "command": parts[4],
                        }
                    )
                except ValueError:
                    continue
        return rows

    @staticmethod
    def _find_zombies() -> list[dict[str, Any]]:
        rc, out, _ = run_cmd(["ps", "-eo", "pid,ppid,stat,comm"])
        if rc != 0 or not out:
            return []
        zombies: list[dict[str, Any]] = []
        for line in out.splitlines()[1:]:
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[2].startswith("Z"):
                try:
                    zombies.append(
                        {"pid": int(parts[0]), "ppid": int(parts[1]), "command": parts[3]}
                    )
                except ValueError:
                    continue
        return zombies


def collect_all_symptoms(
    config: HealerConfig, *, osd_stuck_seconds: float = 0.0
) -> list[SymptomEvent]:
    """Run all configured deterministic probes and return deduplicated SymptomEvents."""
    ui_probe = OmarchyUiProbe(config)
    mcp_probe = McpAndServiceProbe(config)
    mem_probe = MemoryProcessProbe(config)

    events: list[SymptomEvent] = []
    events.extend(ui_probe.check(osd_stuck_seconds=osd_stuck_seconds))
    events.extend(mcp_probe.check())
    events.extend(mem_probe.check())
    return events
