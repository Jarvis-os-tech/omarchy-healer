"""Deterministic system monitors for Omarchy UI/Hyprland, MCP/Systemd, Memory/Processes, and coredumps."""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import HealerConfig
from .models import SymptomEvent
from .playbooks import is_zombie_acknowledged, reset_acknowledged_zombies
# An OSD popup normally dismisses itself in a few seconds, so anything still open this
# long has missed its close IPC for good. Keep this well above any time a human might
# plausibly be reading a popup: the watchdog closes it once the timer trips, and closing
# a popup someone is mid-sentence reading is worse than the stuck popup.
DEFAULT_OSD_STUCK_SECONDS = 90.0

_OSD_OPEN_TOKENS = frozenset(
    {"open", "opened", "on", "active", "visible", "showing", "true", "yes", "1"}
)


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


def _normalize_osd_state(raw: str) -> str:
    """Reduce `omarchy-shell osd state` output to one lowercased comparable token."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("state", "status", "osd_state", "osd", "value"):
                if key in parsed:
                    return str(parsed[key]).strip("\"'").rstrip(".").lower()
        if isinstance(parsed, (str, bool, int)):
            return str(parsed).strip("\"'").rstrip(".").lower()
    except (ValueError, TypeError):
        pass
    for sep in ("=", ":"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    return text.strip().strip("\"'").rstrip(".").lower()


def _osd_reports_open(rc: int, out: str) -> bool:
    """Report whether the shell command succeeded *and* says the OSD is open."""
    return rc == 0 and _normalize_osd_state(out) in _OSD_OPEN_TOKENS


def _scan_layer_namespaces(layers_data: Any) -> set[str]:
    """Collect every layer namespace from a `hyprctl layers -j` payload."""
    monitors = (
        list(layers_data.values())
        if isinstance(layers_data, dict)
        else (layers_data if isinstance(layers_data, list) else [])
    )
    return {
        item.get("namespace")
        for m in monitors
        if isinstance(m, dict) and isinstance(m.get("levels"), dict)
        for items in m["levels"].values()
        if isinstance(items, list)
        for item in items
        if isinstance(item, dict) and isinstance(item.get("namespace"), str) and item.get("namespace")
    }


class OmarchyUiProbe:
    """Monitors Omarchy Quickshell OSD overlays, status bar layers, and Hyprland config errors."""

    name = "omarchy_ui"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config
        self._osd_first_seen: float | None = None

    def check(
        self, *, osd_stuck_seconds: float = DEFAULT_OSD_STUCK_SECONDS
    ) -> list[SymptomEvent]:
        """Return UI/compositor symptoms, reporting an OSD only once it has been open long enough."""
        events: list[SymptomEvent] = []

        # 1. Check omarchy-shell osd state + hyprctl layers for stuck OSD popup
        rc_osd, out_osd, _ = run_cmd(["omarchy-shell", "osd", "state"])
        rc_layers, out_layers, _ = run_cmd(["hyprctl", "layers", "-j"])

        layers_payload: Any = None
        layers_evidence: dict[str, Any] = {}
        if rc_layers == 0 and out_layers:
            try:
                layers_payload = json.loads(out_layers)
            except (ValueError, TypeError):
                layers_payload = None
            if layers_payload is not None:
                layers_evidence = (
                    layers_payload
                    if isinstance(layers_payload, dict)
                    else {"monitors": layers_payload}
                )

        namespaces = _scan_layer_namespaces(layers_payload)
        osd_layer_present = "omarchy-osd" in namespaces
        bar_layer_present = "omarchy-bar" in namespaces

        osd_is_open = _osd_reports_open(rc_osd, out_osd) or osd_layer_present
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
                            "osd_state": _normalize_osd_state(out_osd) or "layer_active",
                            "osd_layer_present": osd_layer_present,
                            "observed_open_seconds": round(elapsed, 1),
                        },
                        raw_evidence={"osd_state": out_osd, "layers": layers_evidence},
                        suggested_playbook="close_stuck_osd",
                        default_tier=0,
                    )
                )
        else:
            self._osd_first_seen = None

        # 2. Check if Hyprland is up but omarchy-bar layer is missing
        if layers_payload and not bar_layer_present:
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="ui_shell",
                    fingerprint_key="missing_omarchy_bar",
                    title="Omarchy Shell Bar Layer Missing",
                    summary="Hyprland compositor is active but the 'omarchy-bar' Quickshell layer is not mapped.",
                    metrics={"bar_layer_present": False},
                    raw_evidence={"layers": layers_evidence},
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
        """Return MCP configuration and systemd --user unit symptoms."""
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


class CoreDumpProbe:
    """Watches systemd-coredump for process crashes.

    Reads the same journald MESSAGE_ID that omarchy-crash-watch tails, so the
    toast you click and the remediation below come from one signal:
    crash-watch announces the crash, this probe acts on it. A crash whose PID
    lived in a systemd --user service earns a tier-1 restart; anything else
    (GUI apps sit in transient ``app-*.scope`` groups, which are not units you
    can start) is recorded rather than blindly relaunched.
    """

    name = "coredump"
    # See systemd.journal-fields(7). Identical to what omarchy-crash-watch tails.
    COREDUMP_MESSAGE_ID = "fc2e22bc6ee647b6b90729ab34a250b1"
    # First sweep only: a host up for days must not replay every old dump as
    # though it had just happened.
    INITIAL_LOOKBACK_US = 3_600_000_000
    # Bounds this probe's contribution to a crash loop: one program earns one
    # incident, and at most one restart, per window. systemd still honours its
    # own Restart= policy outside this window.
    # ponytail: backoff that learns from a *failed* remediation would be
    # better; upgrade path is WatchOrchestrator's failure_count()/retry_delay(),
    # which is keyed per-fingerprint and so cannot see the changing fingerprint
    # a crash loop produces.
    COOLDOWN_US = 300_000_000
    # Crashes are the rare case this probe fires, so a bounded tail keeps idle
    # sweeps at "journalctl returned nothing" and a pathological lookback from
    # buffering megabytes of COREDUMP_PROC_MAPS into one string.
    MAX_ENTRIES = 100

    def __init__(self, config: HealerConfig) -> None:
        self.config = config
        # Shared across processes: crash-watch's trigger runs `heal` in a fresh
        # process while `watch` sweeps from its own, and they share no memory.
        self._state_path = config.db_path.parent / "coredump.state"

    def check(self) -> list[SymptomEvent]:
        state = self._read_state()
        cursor = self._as_int(state.get("cursor"))
        cooldowns = state.get("cooldowns")
        cooldowns = dict(cooldowns) if isinstance(cooldowns, dict) else {}

        now_us = int(time.time() * 1_000_000)
        if cursor <= 0:
            cursor = now_us - self.INITIAL_LOOKBACK_US

        rc, out, _ = run_cmd(
            [
                "journalctl",
                "-o", "json",
                f"MESSAGE_ID={self.COREDUMP_MESSAGE_ID}",
                f"--since=@{cursor // 1_000_000}",
                "-n", str(self.MAX_ENTRIES),
                "--no-pager",
            ],
            timeout=8.0,
        )
        if rc != 0 or not out:
            return []

        uid = str(os.getuid())
        events: list[SymptomEvent] = []
        latest = cursor

        for line in out.splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            # Only this user's crashes; a daemon dumping core is a sysadmin's
            # problem, matching omarchy-crash-watch.
            if str(entry.get("_UID", "")) != uid:
                continue
            stamp = self._as_int(entry.get("COREDUMP_TIMESTAMP"))
            # Strictly newer than the claim: journalctl re-reads the boundary
            # second, and re-acting on a dump is worse than missing one.
            if stamp <= cursor:
                continue

            exe = str(entry.get("COREDUMP_EXE") or entry.get("COREDUMP_COMM") or "unknown")
            program = os.path.basename(exe) or str(entry.get("COREDUMP_COMM") or "unknown")

            if stamp >= self._as_int(cooldowns.get(program)):
                event = self._event_for(entry, program, stamp)
                if event is not None:
                    events.append(event)
                    cooldowns[program] = stamp + self.COOLDOWN_US
            latest = max(latest, stamp)

        if latest > cursor:
            # Claim before returning: whichever process writes first owns the
            # dump, so the other one re-reads the cursor and finds nothing.
            # ponytail: a read-then-write race remains if both processes land
            # in the same few milliseconds; the consequence is one redundant
            # idempotent restart, so no file locking is worth it here.
            live = {k: v for k, v in cooldowns.items() if v > now_us}
            self._write_state({"cursor": latest, "cooldowns": live})

        return events

    def _event_for(
        self, entry: dict[str, Any], program: str, stamp: int
    ) -> SymptomEvent | None:
        pid = str(entry.get("COREDUMP_PID") or "")
        if not pid:
            return None

        signal_name = str(entry.get("COREDUMP_SIGNAL_NAME") or "unknown")
        unit = str(entry.get("COREDUMP_USER_UNIT") or "").strip()
        comm = str(entry.get("COREDUMP_COMM") or program)
        evidence = {
            "pid": pid,
            "signal": signal_name,
            "unit": unit,
            "comm": comm,
            "exe": str(entry.get("COREDUMP_EXE") or ""),
            "timestamp": str(stamp),
            "coredump": str(entry.get("COREDUMP_FILENAME") or ""),
        }

        # systemd-coredump records the owning unit directly. `user@*.service` is
        # pid 1's manager and `app-*.scope` is a transient GUI grouping; neither
        # is a unit restart_systemd_user_unit could usefully act on, so those
        # crashes are inspect-only.
        if unit.endswith(".service") and not unit.startswith("user@"):
            return SymptomEvent.create(
                probe_name=self.name,
                domain="systemd_service",
                fingerprint_key=f"{pid}:{stamp}",
                title=f"Process crashed in service: {unit}",
                summary=(
                    f"{comm} died from {signal_name} (PID {pid}) under systemd "
                    f"--user unit {unit}; restarting that unit."
                ),
                metrics={
                    "pid": pid,
                    "signal": signal_name,
                    "unit": unit,
                    "program": program,
                },
                raw_evidence=evidence,
                suggested_playbook="restart_systemd_user_unit",
                suggested_args={"unit": unit},
                default_tier=1,
            )

        return SymptomEvent.create(
            probe_name=self.name,
            domain="other",
            fingerprint_key=f"{pid}:{stamp}",
            title=f"Process crashed: {program}",
            summary=(
                f"{comm} died from {signal_name} (PID {pid}). It is not owned by a "
                "restartable systemd --user service, so only the evidence is recorded."
            ),
            metrics={
                "pid": pid,
                "signal": signal_name,
                "unit": unit,
                "program": program,
            },
            raw_evidence=evidence,
            suggested_playbook="inspect_only",
            default_tier=0,
        )

    @staticmethod
    def _as_int(value: Any) -> int:
        """Coerce an untrusted journal field to ``int``, returning 0 when unusable."""
        if isinstance(value, bool):
            return 0
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            return 0

    def _read_state(self) -> dict[str, Any]:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_state(self, state: dict[str, Any]) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(json.dumps(state), encoding="utf-8")
        except OSError:
            # Losing a claim only risks one redundant idempotent restart.
            pass


class MemoryProcessProbe:
    """Monitors /proc/meminfo, /proc/pressure/memory PSI, top RSS processes, and zombies."""

    name = "memory_and_process"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config

    def check(self) -> list[SymptomEvent]:
        """Return memory-pressure and zombie-process symptoms."""
        events: list[SymptomEvent] = []
        mem_stats = self._read_meminfo()
        psi_stats = self._read_memory_psi()

        used_pct = mem_stats.get("used_percent", 0.0)
        psi_some10 = psi_stats.get("some_avg10", 0.0)

        if used_pct >= self.config.ram_warn_percent or psi_some10 >= self.config.psi_some_avg10_warn:
            top_procs = self._top_rss_processes(limit=6)
            is_crit = used_pct >= self.config.ram_crit_percent
            total_mb = mem_stats.get("total_mb", 0.0)
            avail_mb = mem_stats.get("available_mb", 0.0)
            used_mb = max(0.0, total_mb - avail_mb)
            warn_target_mb = total_mb * (self.config.ram_warn_percent / 100.0)
            headroom_mb = max(512.0, total_mb * 0.08)
            needed_reclaim_mb = max(512.0, (used_mb - warn_target_mb) * 2.0 + headroom_mb)
            suggested_pb = "clean_user_memory_caches" if is_crit else "reclaim_memory_caches"
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="memory_pressure",
                    fingerprint_key="high_ram_pressure",
                    title=f"{'Critical' if is_crit else 'Elevated'} System RAM Usage ({used_pct:.1f}%)",
                    summary=(
                        f"RAM usage is at {used_pct:.1f}% "
                        f"({avail_mb:.0f} MiB available of "
                        f"{total_mb:.0f} MiB; PSI some_avg10={psi_some10:.2f})."
                    ),
                    metrics={
                        **mem_stats,
                        **psi_stats,
                        "critical": is_crit,
                    },
                    raw_evidence={"top_rss_processes": top_procs},
                    suggested_playbook=suggested_pb,
                    suggested_args={
                        "top_processes": top_procs,
                        "target_reclaim_mb": round(needed_reclaim_mb, 1),
                    },
                    default_tier=0 if not is_crit else 1,
                )
            )

        zombies = self._find_zombies()
        if len(zombies) >= 5:
            unreaped = [z for z in zombies if not z.get("acknowledged")]
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="memory_pressure",
                    fingerprint_key="zombie_process_accumulation",
                    title=f"Zombie Process Accumulation ({len(zombies)} defunct)",
                    summary=(
                        f"Found {len(zombies)} defunct zombie processes on the host "
                        f"({len(unreaped)} from parents not yet signalled)."
                    ),
                    metrics={
                        "zombie_count": len(zombies),
                        "unsignalled_count": len(unreaped),
                    },
                    raw_evidence={"zombies": zombies[:10]},
                    suggested_playbook="reap_zombie_parents",
                    suggested_args={"zombies": unreaped},
                    default_tier=1,
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
        """Return every zombie, flagging the ones already signalled by a previous reap.

        Acknowledged zombies stay in the list on purpose: a parent that never calls
        wait() keeps them forever, and hiding them would report a persistent leak as
        resolved (and fire a bogus "Fixed via reap_zombie_parents" notification). The
        flag only tells the playbook which parents are worth signalling again.
        """
        rc, out, _ = run_cmd(["ps", "-eo", "pid,ppid,stat,comm"])
        if rc != 0 or not out:
            return []
        zombies: list[dict[str, Any]] = []
        for line in out.splitlines()[1:]:
            parts = line.split(None, 3)
            if len(parts) == 4 and parts[2].startswith("Z"):
                try:
                    pid = int(parts[0])
                    ppid = int(parts[1])
                    zombies.append(
                        {
                            "pid": pid,
                            "ppid": ppid,
                            "command": parts[3],
                            "acknowledged": is_zombie_acknowledged(pid, ppid),
                        }
                    )
                except ValueError:
                    continue
        return zombies


class DiskSpaceProbe:
    """Monitors filesystem storage utilization on critical mounts (/, /home, /tmp)."""

    name = "disk_space"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config

    def check(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        mounts_to_check: list[Path] = [Path("/"), Path.home()]
        tmp_dir = Path("/tmp")
        if tmp_dir.is_dir():
            mounts_to_check.append(tmp_dir)

        checked_devs: set[int] = set()

        for mount in mounts_to_check:
            if not mount.exists():
                continue
            try:
                st = mount.stat()
                dev_id = st.st_dev
                if dev_id in checked_devs:
                    continue
                checked_devs.add(dev_id)
                usage = shutil.disk_usage(mount)
            except OSError:
                continue

            total_gb = usage.total / (1024**3)
            free_gb = usage.free / (1024**3)
            used_pct = round(((usage.total - usage.free) / usage.total) * 100.0, 1) if usage.total > 0 else 0.0

            is_crit = used_pct >= self.config.disk_crit_percent
            is_warn = used_pct >= self.config.disk_warn_percent

            if is_warn or is_crit:
                path_str = str(mount)
                severity_str = "Critical" if is_crit else "Elevated"
                events.append(
                    SymptomEvent.create(
                        probe_name=self.name,
                        domain="disk",
                        fingerprint_key=f"disk_full:{path_str}",
                        title=f"{severity_str} Disk Usage on {path_str} ({used_pct:.1f}%)",
                        summary=(
                            f"Filesystem at {path_str} is {used_pct:.1f}% full "
                            f"({free_gb:.1f} GiB free of {total_gb:.1f} GiB)."
                        ),
                        metrics={
                            "mount": path_str,
                            "used_percent": used_pct,
                            "free_gb": round(free_gb, 1),
                            "total_gb": round(total_gb, 1),
                            "critical": is_crit,
                        },
                        suggested_playbook="free_disk_space",
                        suggested_args={"path": path_str},
                        default_tier=1,
                    )
                )

        return events


class AudioServiceProbe:
    """Monitors PipeWire and WirePlumber user audio stack health."""

    name = "audio_service"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config

    def check(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        if not (
            os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("DISPLAY")
            or Path(f"/run/user/{os.getuid()}/hypr").exists()
        ):
            return events

        core_units = ["pipewire.service", "wireplumber.service"]
        failed_units: list[str] = []
        for unit in core_units:
            rc_fail, _, _ = run_cmd(["systemctl", "--user", "is-failed", unit])
            if rc_fail == 0:
                failed_units.append(unit)
                continue
            rc_active, out_act, _ = run_cmd(["systemctl", "--user", "is-active", unit])
            if rc_active != 0 or out_act != "active":
                failed_units.append(unit)

        if failed_units:
            events.append(
                SymptomEvent.create(
                    probe_name=self.name,
                    domain="audio",
                    fingerprint_key=f"audio_units_failed:{','.join(sorted(failed_units))}",
                    title=f"PipeWire Audio Service Degraded ({', '.join(failed_units)})",
                    summary=(
                        f"User audio service(s) {', '.join(failed_units)} are not running or in failed state; "
                        "sound output is unavailable."
                    ),
                    metrics={"failed_units": failed_units},
                    suggested_playbook="restart_pipewire",
                    default_tier=1,
                )
            )

        return events


class NetworkConnectivityProbe:
    """Monitors default route and network connectivity reachability."""

    name = "network_connectivity"

    def __init__(self, config: HealerConfig) -> None:
        self.config = config
        self._consecutive_failures = 0

    def check(self) -> list[SymptomEvent]:
        events: list[SymptomEvent] = []
        rc_route, out_route, _ = run_cmd(["ip", "route", "show", "default"])
        has_default_route = (rc_route == 0 and bool(out_route.strip()))

        can_connect = False
        if has_default_route:
            for host, port in [("1.1.1.1", 53), ("8.8.8.8", 53)]:
                try:
                    s = socket.create_connection((host, port), timeout=1.0)
                    s.close()
                    can_connect = True
                    break
                except OSError:
                    continue

        if not has_default_route or not can_connect:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 2:
                events.append(
                    SymptomEvent.create(
                        probe_name=self.name,
                        domain="network",
                        fingerprint_key="network_disconnected",
                        title="Network Connectivity Lost",
                        summary=(
                            f"Host has lost default gateway route or DNS reachability "
                            f"(has_default_route={has_default_route})."
                        ),
                        metrics={
                            "has_default_route": has_default_route,
                            "consecutive_failures": self._consecutive_failures,
                        },
                        suggested_playbook="reconnect_network",
                        default_tier=1,
                    )
                )
        else:
            self._consecutive_failures = 0

        return events


class _ProbeSet:
    """Long-lived probe instances for one configuration, so per-sweep state survives."""

    __slots__ = ("ui", "mcp", "memory", "crash", "disk", "audio", "network")

    def __init__(self, config: HealerConfig) -> None:
        self.ui = OmarchyUiProbe(config)
        self.mcp = McpAndServiceProbe(config)
        self.memory = MemoryProcessProbe(config)
        self.crash = CoreDumpProbe(config)
        self.disk = DiskSpaceProbe(config)
        self.audio = AudioServiceProbe(config)
        self.network = NetworkConnectivityProbe(config)


_PROBE_CACHE_KEY_FIELDS: tuple[str, ...] = (
    "mcp_config_path",
    "ram_warn_percent",
    "ram_crit_percent",
    "disk_warn_percent",
    "disk_crit_percent",
    "psi_some_avg10_warn",
    # CoreDumpProbe derives its shared claim file from the incidents DB path.
    "db_path",
)

_PROBE_CACHE: dict[tuple[Any, ...], _ProbeSet] = {}


def _probe_cache_key(config: HealerConfig) -> tuple[Any, ...]:
    """Build a hashable identity for *config* from the fields the probes actually read."""
    return tuple(str(getattr(config, field, None)) for field in _PROBE_CACHE_KEY_FIELDS)


def get_probe_set(config: HealerConfig) -> _ProbeSet:
    """Return the cached probe set for *config*, building it on first use.

    Probes must outlive a single sweep: the OSD open-timer in
    :class:`OmarchyUiProbe` is only meaningful when the same instance observes
    consecutive sweeps.
    """
    key = _probe_cache_key(config)
    probes = _PROBE_CACHE.get(key)
    if probes is None:
        probes = _ProbeSet(config)
        _PROBE_CACHE[key] = probes
    return probes


def reset_probe_cache() -> None:
    """Discard every cached probe instance and its accumulated state."""
    _PROBE_CACHE.clear()
    reset_acknowledged_zombies()


def collect_all_symptoms(
    config: HealerConfig, *, osd_stuck_seconds: float = DEFAULT_OSD_STUCK_SECONDS
) -> list[SymptomEvent]:
    """Run all configured deterministic probes and return deduplicated SymptomEvents."""
    probes = get_probe_set(config)

    events: list[SymptomEvent] = []
    events.extend(probes.ui.check(osd_stuck_seconds=osd_stuck_seconds))
    events.extend(probes.mcp.check())
    events.extend(probes.memory.check())
    events.extend(probes.crash.check())
    events.extend(probes.disk.check())
    events.extend(probes.audio.check())
    events.extend(probes.network.check())
    return events
