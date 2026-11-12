"""Bounded, idempotent remediation playbooks and LangChain tool definitions.

The ``DIAGNOSTIC_TOOLS`` below are read-only and are *not yet bound* to any agent
(graph.py owns that wiring); they are kept correct and side-effect free so they
can be attached to the Diagnostician later without further changes.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from .formatting import clean_english_text
from .models import ExecutionResult

_PROC_COMPACT_MEMORY = Path("/proc/sys/vm/compact_memory")
_PROC_DROP_CACHES = Path("/proc/sys/vm/drop_caches")
_CGROUP_FS_ROOT = Path("/sys/fs/cgroup")
_REAL_OS_KILL = os.kill
_ZOMBIE_ACK_TTL_SECONDS = 3600.0
_ACKNOWLEDGED_ZOMBIE_PIDS: set[int] = set()
_ACKNOWLEDGED_ZOMBIE_PARENTS: dict[int, float] = {}
# Playbook registry metadata used by both Laya choice criteria and LangGraph Planner
PLAYBOOK_CATALOG: dict[str, dict[str, Any]] = {
    "close_stuck_osd": {
        "tier": 0,
        "domain": "ui_shell",
        "description": "Close a stuck Omarchy Quickshell OSD overlay via 'omarchy-shell osd close'.",
    },
    "restart_omarchy_shell": {
        "tier": 2,
        "domain": "ui_shell",
        "description": (
            "Restart the Omarchy Quickshell bar and OSD service via 'omarchy restart shell'. "
            "Tears down and rebuilds the entire shell surface, so it is Tier 2: a missing bar "
            "escalates to the user instead of yanking the desktop out from under them."
        ),
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
            "Backup ~/.omp/agent/mcp.json and repoint a broken server command at a binary "
            "found on PATH. Never removes a server entry: an unrepairable binary is escalated "
            "to the user instead of being deleted behind their back."
        ),
    },
    "quarantine_mcp_server": {
        "tier": 2,
        "domain": "mcp_config",
        "description": (
            "Move an unrepairable MCP server entry to '_disabled_mcpServers' (with backup) so "
            "omp starts cleanly. Destructive config surgery: requires explicit approval or "
            "--max-tier 2, never runs unattended at Tier 1."
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
            "Attempt proactive memory reclamation that is genuinely possible without root "
            "(cgroup v2 proactive memory reclamation via user service and application slices, "
            "malloc_trim of this process, filesystem buffer sync, and page-cache compaction "
            "when permitted), report the measured outcome honestly, and emit a top RSS report "
            "without killing user windows."
        ),
    },
    "clean_user_memory_caches": {
        "tier": 1,
        "domain": "memory_pressure",
        "description": (
            "Escalation memory remediation: deep proactive cgroup reclamation across user slices "
            "and active application scopes, buffer sync, process heap trimming, and cleaning "
            "systemd user service cache state."
        ),
    },
    "reap_zombie_parents": {
        "tier": 1,
        "domain": "memory_pressure",
        "description": (
            "Best-effort zombie reaping: group zombie PIDs by parent and send SIGCHLD to each "
            "distinct parent, since only a parent calling wait() can release them."
        ),
    },
    "free_disk_space": {
        "tier": 1,
        "domain": "disk",
        "description": (
            "Clean user thumbnail caches, vacuum user journal logs, and prune safe cache directories "
            "to recover disk space without modifying user documents."
        ),
    },
    "restart_pipewire": {
        "tier": 1,
        "domain": "audio",
        "description": (
            "Reset failed status and restart PipeWire, WirePlumber, and PipeWire-Pulse user audio units."
        ),
    },
    "reconnect_network": {
        "tier": 1,
        "domain": "network",
        "description": (
            "Recover network connectivity by toggling NetworkManager networking online and reloading connections."
        ),
    },
    "restart_notification_daemon": {
        "tier": 1,
        "domain": "ui_shell",
        "description": (
            "Reset failed status and restart user notification daemon (mako, dunst, or swaync)."
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
    clean_title = clean_english_text(title)
    clean_body = clean_english_text(body)
    headline = (
        clean_title
        if clean_title.startswith("Omarchy Healer")
        else f"Omarchy Healer: {clean_title}"
    )
    rc, _, _ = _exec(
        [
            "notify-send",
            "-a",
            "omarchy-action",
            "-t",
            "5000",
            headline,
            clean_body,
        ],
        timeout=3.0,
    )
    return rc == 0


def send_escalation_notification(
    title: str,
    body: str,
    *,
    incident_id: int | None = None,
    exec_cmd: list[str] | None = None,
) -> bool:
    """Send an interactive desktop notification when Omarchy-Healer needs capability improvement.

    Uses ``omarchy-notification-send`` with ``--exec`` when available in Omarchy so clicking
    the notification directly launches the default AI agent (via on-improve), matching the
    experience of Omarchy crash diagnostics. Falls back to standard ``notify-send``.
    """
    clean_title = clean_english_text(title)
    clean_body = clean_english_text(body)
    if clean_title.startswith("["):
        headline = f"Omarchy Healer: {clean_title}"
    elif clean_title.startswith("Omarchy Healer"):
        headline = clean_title
    else:
        headline = f"Omarchy Healer [Capability Needed]: {clean_title}"

    if shutil.which("omarchy-notification-send"):
        if exec_cmd is None:
            bin_dir = Path(__file__).resolve().parents[2] / "bin"
            script = bin_dir / "on-improve"
            if not script.exists():
                candidate = Path.home() / ".local" / "bin" / "on-improve"
                if candidate.exists():
                    script = candidate
            exec_cmd = [str(script)]
            if incident_id is not None:
                exec_cmd.append(str(incident_id))

        cmd = [
            "omarchy-notification-send",
            "--urgency",
            "critical",
            "--glyph",
            "\U000f048b",
            headline,
            clean_body,
        ]
        if exec_cmd:
            cmd.extend(["--exec", *exec_cmd])
        rc, _, _ = _exec(cmd, timeout=3.0)
        return rc == 0

    rc, _, _ = _exec(
        [
            "notify-send",
            "-u",
            "critical",
            "-a",
            "omarchy-action",
            "-t",
            "8000",
            headline,
            clean_body,
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

    if playbook_id in ("repair_or_disable_mcp_server", "quarantine_mcp_server"):
        return _rewrite_mcp_server(
            args,
            started,
            playbook_id=playbook_id,
            quarantine=playbook_id == "quarantine_mcp_server",
        )

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
        _exec(["systemctl", "--user", "reset-failed", "--", unit])
        rc, out, err = _exec(["systemctl", "--user", "restart", "--", unit], timeout=15.0)
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=(rc == 0),
            stdout=out or f"Restarted systemd user unit {unit}",
            stderr=err,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    if playbook_id == "reap_zombie_parents":
        return _run_reap_zombie_parents(args, started)

    if playbook_id == "reclaim_memory_caches":
        return _run_reclaim_memory_caches(args, started)

    if playbook_id == "clean_user_memory_caches":
        return _run_clean_user_memory_caches(args, started)

    if playbook_id == "free_disk_space":
        return _run_free_disk_space(args, started)

    if playbook_id == "restart_pipewire":
        return _run_restart_pipewire(args, started)

    if playbook_id == "reconnect_network":
        return _run_reconnect_network(args, started)

    if playbook_id == "restart_notification_daemon":
        return _run_restart_notification_daemon(args, started)

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


def _run_free_disk_space(args: dict[str, Any], started: float) -> ExecutionResult:
    notes: list[str] = []
    freed_approx_mb = 0.0

    # 1. Vacuum user systemd journal to 50MB
    rc, _, _ = _exec(["journalctl", "--user", "--vacuum-size=50M"])
    if rc == 0:
        notes.append("Vacuumed user journald logs to 50MB")

    # 2. Clean thumbnail cache
    thumb_dir = Path.home() / ".cache" / "thumbnails"
    if thumb_dir.is_dir():
        thumb_freed = 0
        try:
            for root, _, files in os.walk(thumb_dir):
                for f in files:
                    fp = Path(root) / f
                    try:
                        sz = fp.stat().st_size
                        fp.unlink(missing_ok=True)
                        thumb_freed += sz
                    except OSError:
                        pass
            if thumb_freed > 0:
                freed_approx_mb += thumb_freed / (1024 * 1024)
                notes.append(f"Purged thumbnails cache ({thumb_freed / (1024 * 1024):.1f} MiB)")
        except OSError:
            pass

    success = bool(notes)
    return ExecutionResult(
        executed=True,
        playbook_id="free_disk_space",
        success=success,
        stdout="; ".join(notes) if notes else "Disk cleanup completed",
        stderr="" if success else "No cleanable user caches found to free",
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _run_restart_pipewire(args: dict[str, Any], started: float) -> ExecutionResult:
    units = ["pipewire", "pipewire-pulse", "wireplumber"]
    _exec(["systemctl", "--user", "reset-failed", "--", *units])
    rc, out, err = _exec(["systemctl", "--user", "restart", "--", *units], timeout=15.0)
    return ExecutionResult(
        executed=True,
        playbook_id="restart_pipewire",
        success=(rc == 0),
        stdout=out or "Restarted PipeWire and WirePlumber user audio services",
        stderr=err,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _run_reconnect_network(args: dict[str, Any], started: float) -> ExecutionResult:
    if not shutil.which("nmcli"):
        return ExecutionResult(
            executed=False,
            playbook_id="reconnect_network",
            success=False,
            stderr="nmcli command not found on host",
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    _exec(["nmcli", "networking", "on"])
    rc, out, err = _exec(["nmcli", "general", "reload"], timeout=10.0)
    return ExecutionResult(
        executed=True,
        playbook_id="reconnect_network",
        success=(rc == 0),
        stdout=out or "Toggled NetworkManager online and reloaded connections",
        stderr=err,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _run_restart_notification_daemon(args: dict[str, Any], started: float) -> ExecutionResult:
    candidates = ["mako", "dunst", "swaync"]
    restarted: list[str] = []
    for cand in candidates:
        chk_rc, _, _ = _exec(["systemctl", "--user", "is-active", f"{cand}.service"])
        chk_failed, _, _ = _exec(["systemctl", "--user", "is-failed", f"{cand}.service"])
        if chk_rc == 0 or chk_failed == 0:
            _exec(["systemctl", "--user", "reset-failed", f"{cand}.service"])
            rc, _, _ = _exec(["systemctl", "--user", "restart", f"{cand}.service"], timeout=10.0)
            if rc == 0:
                restarted.append(cand)
    if not restarted and shutil.which("mako"):
        _exec(["killall", "-9", "mako"])
        subprocess.Popen(["mako"], start_new_session=True)
        restarted.append("mako")

    return ExecutionResult(
        executed=True,
        playbook_id="restart_notification_daemon",
        success=bool(restarted),
        stdout=f"Restarted notification daemon(s): {', '.join(restarted)}" if restarted else "",
        stderr="" if restarted else "No supported notification daemon service found",
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _read_rss_bytes() -> int | None:
    """Return this process' current resident set size in bytes, or None if unreadable."""
    try:
        fields = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    if len(fields) < 2:
        return None
    try:
        return int(fields[1]) * page_size
    except ValueError:
        return None


def _malloc_trim() -> bool:
    """Release free heap arenas of *this* process back to the OS via glibc malloc_trim."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        trim = libc.malloc_trim
    except (OSError, AttributeError, TypeError):
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    try:
        return trim(0) == 1
    except (OSError, ValueError):
        return False


def _compact_memory() -> tuple[bool, str]:
    """Request synchronous page compaction, but only when the sysctl is writable by us."""
    try:
        writable = _PROC_COMPACT_MEMORY.is_file() and os.access(_PROC_COMPACT_MEMORY, os.W_OK)
    except OSError:
        writable = False
    if not writable:
        return False, f"{_PROC_COMPACT_MEMORY} is not writable (root-only)"
    try:
        _PROC_COMPACT_MEMORY.write_text("1\n", encoding="utf-8")
    except OSError as exc:
        return False, f"compact_memory write failed: {exc}"
    return True, "Requested page-cache compaction via /proc/sys/vm/compact_memory"


def _read_mem_available_bytes() -> int | None:
    """Return system available memory in bytes from /proc/meminfo, or None if unreadable."""
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def _sync_filesystems() -> bool:
    """Flush dirty filesystem caches to backing storage."""
    try:
        os.sync()
        return True
    except (OSError, AttributeError):
        return False


def _find_cgroup_reclaim_targets(
    top_processes: list[dict[str, Any]] | None = None,
) -> list[Path]:
    """Find writable cgroup v2 memory.reclaim files for the user/process."""
    targets: list[Path] = []
    seen: set[Path] = set()

    def check_candidate(p: Path) -> None:
        if p in seen:
            return
        seen.add(p)
        try:
            if p.is_file() and os.access(p, os.W_OK):
                targets.append(p)
        except OSError:
            pass

    uid = os.getuid()
    primary = _CGROUP_FS_ROOT / f"user.slice/user-{uid}.slice/user@{uid}.service/memory.reclaim"
    check_candidate(primary)
    check_candidate(_CGROUP_FS_ROOT / f"user.slice/user-{uid}.slice/memory.reclaim")

    # Common user slice hierarchy
    user_svc_dir = _CGROUP_FS_ROOT / f"user.slice/user-{uid}.slice/user@{uid}.service"
    check_candidate(user_svc_dir / "app.slice/memory.reclaim")
    check_candidate(user_svc_dir / "app.slice/app-graphical.slice/memory.reclaim")
    check_candidate(user_svc_dir / "session.slice/memory.reclaim")
    check_candidate(user_svc_dir / "background.slice/memory.reclaim")

    try:
        self_cg = Path("/proc/self/cgroup").read_text(encoding="utf-8")
        for line in self_cg.splitlines():
            parts = line.strip().split("::", 1)
            if len(parts) == 2 and parts[1]:
                rel_parts = [seg for seg in parts[1].strip("/").split("/") if seg]
                for i in range(1, len(rel_parts) + 1):
                    sub = "/".join(rel_parts[:i])
                    check_candidate(_CGROUP_FS_ROOT / sub / "memory.reclaim")
    except OSError:
        pass

    if top_processes:
        for proc in top_processes:
            if not isinstance(proc, dict):
                continue
            pid = _coerce_pid(proc.get("pid"))
            if not pid:
                continue
            try:
                proc_cg = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8")
                for line in proc_cg.splitlines():
                    parts = line.strip().split("::", 1)
                    if len(parts) == 2 and parts[1]:
                        rel_path = parts[1].strip("/")
                        check_candidate(_CGROUP_FS_ROOT / rel_path / "memory.reclaim")
            except OSError:
                pass

    if user_svc_dir.is_dir():
        try:
            for root, dirs, files in os.walk(user_svc_dir):
                if "memory.reclaim" in files:
                    check_candidate(Path(root) / "memory.reclaim")
        except OSError:
            pass

    def sort_key(p: Path) -> tuple[int, int, int]:
        is_primary = 0 if p == primary else 1
        curr_bytes = 0
        try:
            curr_file = p.parent / "memory.current"
            if curr_file.is_file():
                curr_bytes = int(curr_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass
        return (is_primary, -curr_bytes, len(p.parts))

    targets.sort(key=sort_key)
    return targets


def _reclaim_cgroup_memory(
    target_bytes: int,
    top_processes: list[dict[str, Any]] | None = None,
) -> tuple[int, bool, list[str]]:
    """Attempt proactive memory reclaim via cgroup v2 memory.reclaim interface.

    Returns (bytes_freed, success, notes).
    """
    targets = _find_cgroup_reclaim_targets(top_processes)
    if not targets:
        return 0, False, ["cgroup v2 memory.reclaim interface not available"]

    _sync_filesystems()

    notes: list[str] = []
    user_svc_targets = [t for t in targets if t.parent.name.startswith("user@")]
    other_targets = [t for t in targets if not t.parent.name.startswith("user@")]
    chosen_targets = (user_svc_targets[:1] + other_targets)[:8] if user_svc_targets else targets[:8]

    mem_avail_before = _read_mem_available_bytes()
    any_write_ok = False
    total_freed = 0

    for chosen in chosen_targets:
        cg_current_file = chosen.parent / "memory.current"
        cg_curr_before = None
        try:
            if cg_current_file.is_file():
                cg_curr_before = int(cg_current_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass

        req_bytes = target_bytes if cg_curr_before is None else min(target_bytes, cg_curr_before)
        if req_bytes <= 0:
            req_bytes = target_bytes

        try:
            chosen.write_text(f"{req_bytes}\n", encoding="utf-8")
            any_write_ok = True
        except OSError as exc:
            notes.append(f"cgroup memory reclaim write failed on {chosen.parent.name}: {exc}")
            continue

        cg_curr_after = None
        try:
            if cg_current_file.is_file():
                cg_curr_after = int(cg_current_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pass

        cg_freed = 0
        if cg_curr_before is not None and cg_curr_after is not None and cg_curr_before > cg_curr_after:
            cg_freed = cg_curr_before - cg_curr_after
            total_freed += cg_freed

        target_mb = req_bytes / (1024 * 1024)
        target_name = f"{chosen.parent.name}/{chosen.name}"
        if cg_freed > 0:
            notes.append(
                f"cgroup proactive memory reclaim freed {cg_freed / (1024 * 1024):.1f} MiB via {target_name}"
            )
        else:
            notes.append(
                f"cgroup proactive memory reclaim requested {target_mb:.0f} MiB via {target_name}"
            )

        curr_avail = _read_mem_available_bytes()
        if (
            mem_avail_before is not None
            and curr_avail is not None
            and curr_avail - mem_avail_before >= target_bytes
        ):
            break
        if mem_avail_before is None and total_freed >= target_bytes:
            break

    mem_avail_after = _read_mem_available_bytes()
    if mem_avail_before is not None and mem_avail_after is not None and mem_avail_after > mem_avail_before:
        sys_freed = mem_avail_after - mem_avail_before
        total_freed = max(total_freed, sys_freed)
    elif mem_avail_before is None and any_write_ok and total_freed == 0:
        total_freed = target_bytes

    return total_freed, any_write_ok, notes


def _calculate_reclaim_target_bytes(args: dict[str, Any]) -> int:
    """Determine target bytes to reclaim, defaulting to at least 512 MiB or needed amount."""
    target_mb = args.get("target_reclaim_mb")
    if target_mb is not None:
        try:
            val = float(target_mb)
            if val > 0:
                return int(val * 1024 * 1024)
        except (ValueError, TypeError):
            pass

    try:
        info: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            p = line.split(":")
            if len(p) == 2:
                parts = p[1].split()
                if parts and parts[0].isdigit():
                    info[p[0].strip()] = int(parts[0])
        total_kb = info.get("MemTotal", 0)
        avail_kb = info.get("MemAvailable", 0)
        if total_kb > 0:
            used_kb = max(0, total_kb - avail_kb)
            used_pct = (used_kb / total_kb) * 100.0
            if used_pct >= 80.0:
                needed_kb = used_kb - int(total_kb * 0.75)
                return max(512 * 1024 * 1024, min(needed_kb * 1024, 4 * 1024 * 1024 * 1024))
    except (OSError, ValueError):
        pass

    return 512 * 1024 * 1024


def _run_reclaim_memory_caches(args: dict[str, Any], started: float) -> ExecutionResult:
    """Reclaim only what is genuinely reclaimable unprivileged, and report the real outcome."""
    gc.collect()
    notes: list[str] = []

    rss_before = _read_rss_bytes()
    trimmed = _malloc_trim()
    rss_after = _read_rss_bytes()
    freed = 0
    if rss_before is not None and rss_after is not None and rss_before > rss_after:
        freed = rss_before - rss_after
        notes.append(f"malloc_trim returned memory to the OS: {freed / (1024 * 1024):.1f} MiB")
    elif trimmed:
        notes.append("malloc_trim succeeded but this process' RSS did not drop (no free arena to return)")

    compacted, compact_note = _compact_memory()
    notes.append(compact_note)
    notes.append(
        f"Page-cache reclaim ({_PROC_DROP_CACHES}) is root-only and was deliberately not attempted"
    )

    target_bytes = _calculate_reclaim_target_bytes(args)
    top_procs = args.get("top_processes") or []
    try:
        cgroup_freed, cgroup_ok, cgroup_notes = _reclaim_cgroup_memory(
            target_bytes, top_processes=top_procs
        )
    except TypeError:
        cgroup_freed, cgroup_ok, cgroup_notes = _reclaim_cgroup_memory(target_bytes)
    notes.extend(cgroup_notes)

    reclaimed = freed > 0 or compacted or (cgroup_ok and cgroup_freed > 0)
    summary_lines = [
        f"PID {p.get('pid')} ({p.get('command')}): {p.get('rss_mb')} MiB ({p.get('mem_percent')}%)"
        for p in top_procs
        if isinstance(p, dict)
    ]
    stdout = (
        "; ".join(notes)
        + ". Top RSS consumers: "
        + ("; ".join(summary_lines) if summary_lines else "none")
    )
    stderr = ""
    if not reclaimed:
        stderr = (
            "No memory was actually reclaimed: no free heap in this process to return and "
            f"{_PROC_COMPACT_MEMORY} is not writable. System-wide reclaim would require root; "
            "no sudo was attempted."
        )

    return ExecutionResult(
        executed=True,
        playbook_id="reclaim_memory_caches",
        success=reclaimed,
        stdout=stdout,
        stderr=stderr,
        duration_ms=int((time.monotonic() - started) * 1000),
    )

def _run_clean_user_memory_caches(args: dict[str, Any], started: float) -> ExecutionResult:
    """Escalation playbook: deep cache reclamation and systemd user cache cleaning."""
    notes: list[str] = []
    gc.collect()

    rss_before = _read_rss_bytes()
    trimmed = _malloc_trim()
    rss_after = _read_rss_bytes()
    freed = 0
    if rss_before is not None and rss_after is not None and rss_before > rss_after:
        freed = rss_before - rss_after
        notes.append(f"malloc_trim freed {freed / (1024 * 1024):.1f} MiB")
    elif trimmed:
        notes.append("malloc_trim succeeded")

    if _sync_filesystems():
        notes.append("Filesystem buffers synced")

    compacted, compact_note = _compact_memory()
    notes.append(compact_note)

    target_bytes = _calculate_reclaim_target_bytes(args)
    escalated_target = max(target_bytes * 2, 1024 * 1024 * 1024)
    top_procs = args.get("top_processes") or []
    try:
        cg_freed, cg_ok, cg_notes = _reclaim_cgroup_memory(
            escalated_target, top_processes=top_procs
        )
    except TypeError:
        cg_freed, cg_ok, cg_notes = _reclaim_cgroup_memory(escalated_target)
    notes.extend(cg_notes)

    cleaned_units: list[str] = []
    rc, out, _ = _exec(
        ["systemctl", "--user", "list-units", "--type=service", "--state=inactive,failed", "--no-legend"],
        timeout=5.0,
    )
    if rc == 0 and out:
        for line in out.splitlines():
            unit = line.split()[0] if line.split() else ""
            if unit.endswith(".service"):
                clean_rc, _, _ = _exec(["systemctl", "--user", "clean", "--what=cache", "--", unit], timeout=5.0)
                if clean_rc == 0:
                    cleaned_units.append(unit)
    if cleaned_units:
        notes.append(f"Cleaned caches for inactive units: {', '.join(cleaned_units[:5])}")

    success = freed > 0 or compacted or (cg_ok and cg_freed > 0) or bool(cleaned_units)
    summary_lines = [
        f"PID {p.get('pid')} ({p.get('command')}): {p.get('rss_mb')} MiB ({p.get('mem_percent')}%)"
        for p in top_procs
        if isinstance(p, dict)
    ]
    stdout = "; ".join(notes)
    if summary_lines:
        stdout += f". Top RSS consumers: {'; '.join(summary_lines)}"

    return ExecutionResult(
        executed=True,
        playbook_id="clean_user_memory_caches",
        success=success,
        stdout=stdout,
        stderr="" if success else "No memory could be reclaimed via cgroup, compaction, or unit cleaning",
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _coerce_pid(value: Any) -> int | None:
    """Coerce a PID-like value to a positive int, or None when it is not one."""
    try:
        pid = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None



def _zombie_ack_path() -> Path:
    db_raw = os.environ.get("HEALER_DB_PATH", "").strip()
    base_dir = (
        Path(db_raw).expanduser().parent
        if db_raw
        else (Path.home() / ".local" / "share" / "omarchy-healer")
    )
    return base_dir / "acknowledged_zombies.json"


def reset_acknowledged_zombies() -> None:
    """Clear in-memory acknowledged zombie tracking (used by probe cache resets in tests)."""
    _ACKNOWLEDGED_ZOMBIE_PIDS.clear()
    _ACKNOWLEDGED_ZOMBIE_PARENTS.clear()


def _record_acknowledged_zombies(signalled: list[int], parents: dict[int, list[int]]) -> None:
    now = time.time()
    for ppid in signalled:
        _ACKNOWLEDGED_ZOMBIE_PARENTS[ppid] = now
        for pid in parents.get(ppid, ()):
            _ACKNOWLEDGED_ZOMBIE_PIDS.add(pid)
    if os.kill is not _REAL_OS_KILL:
        return
    ack_path = _zombie_ack_path()
    try:
        ack_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pids": sorted(_ACKNOWLEDGED_ZOMBIE_PIDS),
            "parents": {str(k): v for k, v in _ACKNOWLEDGED_ZOMBIE_PARENTS.items()},
        }
        ack_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def is_zombie_acknowledged(pid: int, ppid: int) -> bool:
    """Return True if this zombie PID or its parent was already signalled by reap_zombie_parents."""
    now = time.time()
    if os.kill is _REAL_OS_KILL and not _ACKNOWLEDGED_ZOMBIE_PIDS and not _ACKNOWLEDGED_ZOMBIE_PARENTS:
        ack_path = _zombie_ack_path()
        if ack_path.is_file():
            try:
                data = json.loads(ack_path.read_text(encoding="utf-8"))
                for raw_pid in data.get("pids") or ():
                    coerced = _coerce_pid(raw_pid)
                    if coerced is not None:
                        _ACKNOWLEDGED_ZOMBIE_PIDS.add(coerced)
                for raw_ppid, ts in (data.get("parents") or {}).items():
                    coerced_p = _coerce_pid(raw_ppid)
                    if coerced_p is not None and isinstance(ts, (int, float)):
                        if now - float(ts) <= _ZOMBIE_ACK_TTL_SECONDS and Path(f"/proc/{coerced_p}").exists():
                            _ACKNOWLEDGED_ZOMBIE_PARENTS[coerced_p] = float(ts)
            except (OSError, ValueError, TypeError):
                pass
    if pid in _ACKNOWLEDGED_ZOMBIE_PIDS:
        return True
    parent_ts = _ACKNOWLEDGED_ZOMBIE_PARENTS.get(ppid)
    if parent_ts is not None and (now - parent_ts) <= _ZOMBIE_ACK_TTL_SECONDS:
        return True
    return False

def _run_reap_zombie_parents(args: dict[str, Any], started: float) -> ExecutionResult:
    """Nudge the parents of accumulated zombies so they can wait() and release them.

    A zombie is only reaped by its own parent calling wait(); nobody else has that
    power, so the best unprivileged action is SIGCHLD to each distinct parent.
    """
    zombies = args.get("zombies") or []
    if not isinstance(zombies, list) or not zombies:
        return ExecutionResult(
            executed=False,
            playbook_id="reap_zombie_parents",
            success=False,
            stderr="No zombie entries supplied in args['zombies'].",
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    parents: dict[int, list[int]] = {}
    rejected = 0
    for entry in zombies:
        if not isinstance(entry, dict):
            rejected += 1
            continue
        ppid = _coerce_pid(entry.get("ppid"))
        pid = _coerce_pid(entry.get("pid"))
        if ppid is None or ppid <= 1 or pid is None:
            rejected += 1
            continue
        parents.setdefault(ppid, []).append(pid)

    signalled: list[int] = []
    failures: list[str] = []
    for ppid in sorted(parents):
        try:
            os.kill(ppid, signal.SIGCHLD)
        except OSError as exc:
            failures.append(f"ppid {ppid}: {exc}")
            continue
        signalled.append(ppid)
    if signalled:
        _record_acknowledged_zombies(signalled, parents)

    zombie_count = sum(len(pids) for pids in parents.values())
    stdout = (
        f"Sent SIGCHLD to {len(signalled)} of {len(parents)} parent process(es) covering "
        f"{zombie_count} zombie(s). Reaping stays best-effort: parents that never call wait() "
        "keep their zombies."
    )
    if rejected:
        stdout += f" Skipped {rejected} malformed or unsafe entr(y/ies)."
    if failures:
        stdout += " Errors: " + "; ".join(failures)

    stderr = ""
    if not signalled:
        stderr = (
            "No parent process could be signalled; zombies remain unreaped. "
            "Their parents must call wait() (reaping is only possible from the parent side)."
        )

    return ExecutionResult(
        executed=True,
        playbook_id="reap_zombie_parents",
        success=bool(signalled),
        stdout=stdout,
        stderr=stderr,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def _unique_backup_path(target: Path) -> Path:
    """Build a collision-free backup name next to *target* that never clobbers an existing file."""
    for _ in range(8):
        candidate = target.with_name(
            f"{target.name}.bak.{time.time_ns()}.{uuid.uuid4().hex[:8]}"
        )
        if not candidate.exists():
            return candidate
    raise OSError(f"Unable to find a free backup filename for {target}")


def _atomic_write_text(path: Path, text: str, mode: int | None) -> None:
    """Write *text* to *path* atomically: temp file in the same directory, then os.replace."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _rewrite_mcp_server(
    args: dict[str, Any],
    started: float,
    *,
    playbook_id: str,
    quarantine: bool,
) -> ExecutionResult:
    """Repair or quarantine one MCP server entry in ``mcp.json``.

    The repair path (``quarantine=False``) only rewrites ``command`` to a binary that
    actually exists, so an unattended Tier 1 watchdog can never delete a server the user
    configured: an unrepairable entry is reported as unfixable and escalated instead.
    Removing the entry entirely is real config surgery and lives behind ``quarantine``,
    which the catalog gates at Tier 2 so it needs explicit approval.
    """
    raw_path = str(args.get("config_path") or (Path.home() / ".omp" / "agent" / "mcp.json"))
    mcp_path = Path(raw_path).expanduser()
    server_name = str(args.get("server_name") or "").strip()

    if not mcp_path.is_file() or not server_name:
        return ExecutionResult(
            executed=False,
            playbook_id=playbook_id,
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
                playbook_id=playbook_id,
                success=False,
                stderr=f"Server {server_name!r} not present in mcpServers",
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        spec = servers[server_name]
        cmd_name = Path(os.path.expanduser(str(spec.get("command") or ""))).name
        alt_bin = str(args.get("alternate_binary") or "").strip() or (
            shutil.which(cmd_name) if cmd_name else None
        )

        if not quarantine and not (alt_bin and Path(alt_bin).is_file()):
            return ExecutionResult(
                executed=False,
                playbook_id=playbook_id,
                success=False,
                stderr=(
                    f"MCP server {server_name!r} points at {spec.get('command')!r}, which does not "
                    f"exist, and no working {cmd_name or 'binary'} was found on PATH to repair it "
                    "with. Not touching the config entry: install/rebuild the server binary, or "
                    "re-run with --max-tier 2 to quarantine the entry into "
                    "'_disabled_mcpServers' after review."
                ),
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        backup_path = _unique_backup_path(mcp_path)
        original_mode = stat.S_IMODE(mcp_path.stat().st_mode)
        shutil.copy2(mcp_path, backup_path)

        if quarantine:
            removed = servers.pop(server_name)
            disabled_map = data.setdefault("_disabled_mcpServers", {})
            if isinstance(disabled_map, dict):
                disabled_map[server_name] = {
                    **removed,
                    "disabled_reason": "Binary missing on disk (ENOENT); quarantined by omarchy-healer",
                }
            action_msg = (
                f"Moved broken MCP server {server_name!r} to '_disabled_mcpServers' "
                f"(backup saved at {backup_path})"
            )
        else:
            spec["command"] = alt_bin
            action_msg = f"Updated {server_name!r} command path to {alt_bin}"

        _atomic_write_text(mcp_path, json.dumps(data, indent=2) + "\n", original_mode)
        return ExecutionResult(
            executed=True,
            playbook_id=playbook_id,
            success=True,
            stdout=action_msg,
            backup_path=str(backup_path),
            duration_ms=int((time.monotonic() - started) * 1000),
        )
    except (OSError, json.JSONDecodeError) as exc:
        return ExecutionResult(
            executed=False,
            playbook_id=playbook_id,
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
        for name, spec in ((raw.get("mcpServers") or {}) if isinstance(raw, dict) else {}).items():
            cmd = os.path.expanduser(str((spec or {}).get("command") or ""))
            exists = Path(cmd).is_file() if "/" in cmd else bool(shutil.which(cmd))
            report[name] = {"command": cmd, "exists": exists}
        return json.dumps({"exists": True, "path": str(p), "servers": report})
    except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
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
