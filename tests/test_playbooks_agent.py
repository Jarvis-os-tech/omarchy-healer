"""Tests for playbooks.py: zombie reaping, honest memory reclaim, backup/atomic-write safety."""
from __future__ import annotations

import json
import os
import signal
import stat
from pathlib import Path
from typing import Any

import pytest

from omarchy_healer import playbooks
from omarchy_healer.playbooks import DIAGNOSTIC_TOOLS, PLAYBOOK_CATALOG, execute_playbook


def _mcp_file(tmp_path: Path, servers: dict[str, Any]) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- catalog


def test_catalog_entries_have_stable_shape() -> None:
    for name, meta in PLAYBOOK_CATALOG.items():
        assert set(meta) == {"tier", "domain", "description"}, name
        assert isinstance(meta["tier"], int)
        assert isinstance(meta["description"], str) and meta["description"]


def test_reap_zombie_parents_registered_in_catalog() -> None:
    assert "reap_zombie_parents" in PLAYBOOK_CATALOG
    meta = PLAYBOOK_CATALOG["reap_zombie_parents"]
    assert meta["tier"] == 1
    assert meta["domain"] == "memory_pressure"


def test_diagnostic_tools_remain_importable() -> None:
    assert len(DIAGNOSTIC_TOOLS) == 3


# --------------------------------------------------------------------------- reap_zombie_parents


@pytest.fixture()
def recorded_signals(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    calls: list[tuple[int, int]] = []

    def fake_kill(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr(playbooks.os, "kill", fake_kill)
    return calls


def test_reap_signals_each_distinct_parent_once(recorded_signals: list[tuple[int, int]]) -> None:
    res = execute_playbook(
        "reap_zombie_parents",
        {
            "zombies": [
                {"pid": 501, "ppid": 400, "command": "defunct"},
                {"pid": 502, "ppid": 400, "command": "defunct"},
                {"pid": 503, "ppid": 777, "command": "defunct"},
            ]
        },
    )

    assert res.executed is True
    assert res.success is True
    assert sorted(recorded_signals) == [(400, signal.SIGCHLD), (777, signal.SIGCHLD)]
    assert "2 of 2 parent process(es)" in res.stdout
    assert "3 zombie(s)" in res.stdout


def test_reap_never_signals_pid_one_or_invalid_pids(recorded_signals: list[tuple[int, int]]) -> None:
    res = execute_playbook(
        "reap_zombie_parents",
        {
            "zombies": [
                {"pid": 10, "ppid": 1, "command": "defunct"},
                {"pid": 11, "ppid": 0, "command": "defunct"},
                {"pid": 12, "ppid": -5, "command": "defunct"},
                {"pid": "abc", "ppid": "xyz", "command": "defunct"},
                "not-a-dict",
                {"pid": 13, "ppid": 900, "command": "defunct"},
            ]
        },
    )

    assert res.success is True
    assert recorded_signals == [(900, signal.SIGCHLD)]
    assert [pid for pid, _ in recorded_signals if pid <= 1] == []
    assert "Skipped 5 malformed or unsafe" in res.stdout


def test_reap_reports_failure_when_no_parent_can_be_signalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(pid: int, sig: int) -> None:
        raise PermissionError(f"no permission for {pid}")

    monkeypatch.setattr(playbooks.os, "kill", boom)

    res = execute_playbook("reap_zombie_parents", {"zombies": [{"pid": 1, "ppid": 42, "command": "z"}]})

    assert res.executed is True
    assert res.success is False
    assert res.stderr != ""
    assert "no permission for 42" in res.stdout


def test_reap_without_zombie_evidence_is_not_executed() -> None:
    res = execute_playbook("reap_zombie_parents", {})

    assert res.executed is False
    assert res.success is False
    assert "No zombie entries" in res.stderr


def test_reap_signal_error_cannot_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    def weird(pid: int, sig: int) -> None:
        raise OSError(f"kill({pid}) failed")

    monkeypatch.setattr(playbooks.os, "kill", weird)

    res = execute_playbook("reap_zombie_parents", {"zombies": [{"pid": 1, "ppid": 55, "command": "z"}]})
    assert res.executed is True
    assert res.success is False
    assert "kill(55) failed" in res.stdout


# --------------------------------------------------------------------------- reclaim_memory_caches


def test_reclaim_reports_failure_when_nothing_reclaimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(playbooks, "_malloc_trim", lambda: False)
    monkeypatch.setattr(playbooks, "_compact_memory", lambda: (False, "compact_memory not writable"))
    monkeypatch.setattr(playbooks, "_read_rss_bytes", lambda: 100 * 1024 * 1024)
    monkeypatch.setattr(playbooks, "_reclaim_cgroup_memory", lambda _t: (0, False, ["cgroup memory.reclaim not available"]))
    res = execute_playbook("reclaim_memory_caches", {"top_processes": [{"pid": 7, "command": "x", "rss_mb": 12.0, "mem_percent": 1.0}]})

    assert res.executed is True
    assert res.success is False
    assert "No memory was actually reclaimed" in res.stderr
    assert "root" in res.stderr
    assert "sudo" not in res.stderr.lower().replace("no sudo was attempted", "")
    assert "PID 7" in res.stdout


def test_reclaim_never_writes_drop_caches_even_when_writable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playbooks, "_malloc_trim", lambda: False)
    monkeypatch.setattr(playbooks, "_compact_memory", lambda: (False, "compact_memory not writable"))
    monkeypatch.setattr(playbooks, "_read_rss_bytes", lambda: 100 * 1024 * 1024)
    monkeypatch.setattr(playbooks, "_reclaim_cgroup_memory", lambda _t: (0, False, ["cgroup memory.reclaim not available"]))
    monkeypatch.setattr(playbooks.os, "access", lambda _p, _mode: True)
    written: list[str] = []
    monkeypatch.setattr(
        Path, "write_text", lambda self, *a, **k: written.append(str(self)) or 0, raising=False
    )

    res = execute_playbook("reclaim_memory_caches", {})

    assert res.success is False
    assert written == []
    assert "drop_caches" in res.stdout


def test_reclaim_succeeds_when_compaction_was_performed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playbooks, "_malloc_trim", lambda: True)
    monkeypatch.setattr(playbooks, "_compact_memory", lambda: (True, "compacted"))
    monkeypatch.setattr(playbooks, "_read_rss_bytes", lambda: 10 * 1024 * 1024)

    res = execute_playbook("reclaim_memory_caches", {})

    assert res.executed is True
    assert res.success is True
    assert res.stderr == ""


def test_reclaim_succeeds_only_when_rss_actually_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playbooks, "_compact_memory", lambda: (False, "nope"))
    rss_values = iter([200 * 1024 * 1024, 150 * 1024 * 1024])
    monkeypatch.setattr(playbooks, "_read_rss_bytes", lambda: next(rss_values))
    monkeypatch.setattr(playbooks, "_malloc_trim", lambda: True)

    res = execute_playbook("reclaim_memory_caches", {})

    assert res.success is True
    assert "50.0 MiB" in res.stdout


def test_reclaim_does_not_kill_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_a: Any, **_k: Any) -> None:
        raise AssertionError("reclaim_memory_caches must not signal or kill processes")

    monkeypatch.setattr(playbooks.os, "kill", forbidden)
    monkeypatch.setattr(playbooks.subprocess, "run", forbidden)

    execute_playbook("reclaim_memory_caches", {"top_processes": [{"pid": 1, "command": "init"}]})


def test_reclaim_succeeds_when_cgroup_memory_reclaimed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playbooks, "_malloc_trim", lambda: False)
    monkeypatch.setattr(playbooks, "_compact_memory", lambda: (False, "not writable"))
    monkeypatch.setattr(playbooks, "_read_rss_bytes", lambda: 100 * 1024 * 1024)
    monkeypatch.setattr(
        playbooks,
        "_reclaim_cgroup_memory",
        lambda _t: (128 * 1024 * 1024, True, ["cgroup proactive memory reclaim freed 128.0 MiB via user@1000.service/memory.reclaim"]),
    )

    res = execute_playbook("reclaim_memory_caches", {})

    assert res.executed is True
    assert res.success is True
    assert res.stderr == ""
    assert "128.0 MiB" in res.stdout


def test_reclaim_cgroup_memory_writes_reclaim_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cg_dir = tmp_path / "user.slice" / "user-1000.slice" / "user@1000.service"
    cg_dir.mkdir(parents=True)
    reclaim_file = cg_dir / "memory.reclaim"
    reclaim_file.write_text("", encoding="utf-8")
    current_file = cg_dir / "memory.current"
    current_file.write_text(str(500 * 1024 * 1024), encoding="utf-8")

    monkeypatch.setattr(playbooks, "_CGROUP_FS_ROOT", tmp_path)
    monkeypatch.setattr(playbooks.os, "getuid", lambda: 1000)
    monkeypatch.setattr(playbooks, "_read_mem_available_bytes", lambda: None)

    original_write = Path.write_text
    def fake_write(self: Path, text: str, *a: Any, **k: Any) -> int:
        if self == reclaim_file:
            current_file.write_text(str(400 * 1024 * 1024), encoding="utf-8")
        return original_write(self, text, *a, **k)

    monkeypatch.setattr(Path, "write_text", fake_write)

    freed, ok, notes = playbooks._reclaim_cgroup_memory(100 * 1024 * 1024)
    assert ok is True
    assert freed == 100 * 1024 * 1024
    assert "100.0 MiB" in notes[0]


def test_calculate_reclaim_target_bytes_respects_args() -> None:
    target = playbooks._calculate_reclaim_target_bytes({"target_reclaim_mb": 750})
    assert target == 750 * 1024 * 1024


def test_reclaim_cgroup_memory_multi_target_across_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_svc = tmp_path / "user.slice" / "user-1000.slice" / "user@1000.service"
    app_slice = user_svc / "app.slice"
    user_svc.mkdir(parents=True)
    app_slice.mkdir(parents=True)

    user_rec = user_svc / "memory.reclaim"
    user_cur = user_svc / "memory.current"
    user_rec.write_text("", encoding="utf-8")
    user_cur.write_text(str(500 * 1024 * 1024), encoding="utf-8")

    app_rec = app_slice / "memory.reclaim"
    app_cur = app_slice / "memory.current"
    app_rec.write_text("", encoding="utf-8")
    app_cur.write_text(str(800 * 1024 * 1024), encoding="utf-8")

    monkeypatch.setattr(playbooks, "_CGROUP_FS_ROOT", tmp_path)
    monkeypatch.setattr(playbooks.os, "getuid", lambda: 1000)
    monkeypatch.setattr(playbooks, "_read_mem_available_bytes", lambda: None)

    written_targets: list[Path] = []
    orig_write = Path.write_text

    def fake_write(self: Path, text: str, *a: Any, **k: Any) -> int:
        if self in (user_rec, app_rec):
            written_targets.append(self)
            # Simulate 100 MiB drop on user_svc and 200 MiB drop on app_slice
            if self == user_rec:
                user_cur.write_text(str(400 * 1024 * 1024), encoding="utf-8")
            elif self == app_rec:
                app_cur.write_text(str(600 * 1024 * 1024), encoding="utf-8")
        return orig_write(self, text, *a, **k)

    monkeypatch.setattr(Path, "write_text", fake_write)

    freed, ok, notes = playbooks._reclaim_cgroup_memory(250 * 1024 * 1024)
    assert ok is True
    assert user_rec in written_targets
    assert app_rec in written_targets
    assert freed == 300 * 1024 * 1024
    assert len(notes) >= 2


def test_reclaim_finds_targets_for_top_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_svc = tmp_path / "user.slice" / "user-1000.slice" / "user@1000.service"
    proc_dir = user_svc / "app.slice" / "app-chrome-1234.scope"
    proc_dir.mkdir(parents=True)
    proc_rec = proc_dir / "memory.reclaim"
    proc_rec.write_text("", encoding="utf-8")

    monkeypatch.setattr(playbooks, "_CGROUP_FS_ROOT", tmp_path)
    monkeypatch.setattr(playbooks.os, "getuid", lambda: 1000)

    # Mock /proc/1234/cgroup read
    def fake_read_text(self: Path, *a: Any, **k: Any) -> str:
        if str(self) == "/proc/1234/cgroup":
            return "0::/user.slice/user-1000.slice/user@1000.service/app.slice/app-chrome-1234.scope\n"
        return ""

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    targets = playbooks._find_cgroup_reclaim_targets(
        top_processes=[{"pid": 1234, "command": "chrome"}]
    )
    assert proc_rec in targets


def test_sync_filesystems_called_before_cgroup_reclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cg_dir = tmp_path / "user.slice" / "user-1000.slice" / "user@1000.service"
    cg_dir.mkdir(parents=True)
    (cg_dir / "memory.reclaim").write_text("", encoding="utf-8")

    monkeypatch.setattr(playbooks, "_CGROUP_FS_ROOT", tmp_path)
    monkeypatch.setattr(playbooks.os, "getuid", lambda: 1000)
    monkeypatch.setattr(playbooks, "_read_mem_available_bytes", lambda: None)

    synced = []
    monkeypatch.setattr(playbooks, "_sync_filesystems", lambda: synced.append(True) or True)

    playbooks._reclaim_cgroup_memory(50 * 1024 * 1024)
    assert synced == [True]

def test_clean_user_memory_caches_registered_in_catalog() -> None:
    assert "clean_user_memory_caches" in PLAYBOOK_CATALOG
    meta = PLAYBOOK_CATALOG["clean_user_memory_caches"]
    assert meta["tier"] == 1
    assert meta["domain"] == "memory_pressure"
    assert "cgroup" in meta["description"].lower()


def test_clean_user_memory_caches_executes_successfully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(playbooks, "_malloc_trim", lambda: True)
    monkeypatch.setattr(playbooks, "_read_rss_bytes", lambda: 10 * 1024 * 1024)
    monkeypatch.setattr(playbooks, "_compact_memory", lambda: (False, "compaction not writable"))
    monkeypatch.setattr(
        playbooks,
        "_reclaim_cgroup_memory",
        lambda _t, **_k: (200 * 1024 * 1024, True, ["reclaimed 200 MiB"]),
    )
    monkeypatch.setattr(
        playbooks,
        "_exec",
        lambda cmd, **_k: (0, "unit1.service loaded active\n", "") if "list-units" in cmd else (0, "", ""),
    )

    res = execute_playbook("clean_user_memory_caches", {"top_processes": [{"pid": 100, "command": "app"}]})
    assert res.executed is True
    assert res.success is True
    assert res.playbook_id == "clean_user_memory_caches"
    assert "reclaimed 200 MiB" in res.stdout
    assert "Cleaned caches for inactive units: unit1.service" in res.stdout
# --------------------------------------------------------------------------- backup naming + atomic write


def test_backup_names_never_collide_across_rapid_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(playbooks.time, "time_ns", lambda: 1_700_000_000_000_000_000)
    mcp = _mcp_file(
        tmp_path,
        {
            "alpha": {"command": "/nonexistent/alpha"},
            "beta": {"command": "/nonexistent/beta"},
        },
    )
    original = mcp.read_text(encoding="utf-8")

    first = execute_playbook("quarantine_mcp_server", {"config_path": str(mcp), "server_name": "alpha"})
    second = execute_playbook("quarantine_mcp_server", {"config_path": str(mcp), "server_name": "beta"})

    assert first.success is True and second.success is True
    assert first.backup_path != second.backup_path
    assert Path(first.backup_path).read_text(encoding="utf-8") == original
    assert json.loads(Path(second.backup_path).read_text(encoding="utf-8"))["mcpServers"] == {
        "beta": {"command": "/nonexistent/beta"}
    }

    backups = sorted(p.name for p in tmp_path.glob("mcp.json.bak.*"))
    assert len(backups) == 2


def test_unique_backup_path_never_reuses_existing(tmp_path: Path) -> None:
    target = tmp_path / "mcp.json"
    target.write_text("{}", encoding="utf-8")

    first = playbooks._unique_backup_path(target)
    first.write_text("first", encoding="utf-8")
    second = playbooks._unique_backup_path(target)

    assert first != second
    assert first.read_text(encoding="utf-8") == "first"


def test_quarantine_preserves_valid_json_and_file_mode(tmp_path: Path) -> None:
    mcp = _mcp_file(tmp_path, {"gone": {"command": "/nonexistent/gone"}})
    os.chmod(mcp, 0o640)
    before_mode = stat.S_IMODE(mcp.stat().st_mode)

    res = execute_playbook("quarantine_mcp_server", {"config_path": str(mcp), "server_name": "gone"})

    assert res.success is True
    parsed = json.loads(mcp.read_text(encoding="utf-8"))
    assert "gone" in parsed["_disabled_mcpServers"]
    assert stat.S_IMODE(mcp.stat().st_mode) == before_mode
    assert list(tmp_path.glob(".mcp.json.*")) == []


def test_tier1_repair_never_deletes_an_unrepairable_server_entry(tmp_path: Path) -> None:
    """Regression: auto-heal used to 'heal' a missing MCP binary by deleting the entry.

    That silently removed a server the user had configured (reported as HEALED), so the
    Tier 1 playbook must refuse instead and let the escalation path ask the user.
    """
    mcp = _mcp_file(tmp_path, {"gone": {"command": "/nonexistent/codebase-memory-mcp"}})
    before = mcp.read_text(encoding="utf-8")

    res = execute_playbook(
        "repair_or_disable_mcp_server", {"config_path": str(mcp), "server_name": "gone"}
    )

    assert res.executed is False
    assert res.success is False
    assert "--max-tier 2" in res.stderr
    assert mcp.read_text(encoding="utf-8") == before
    assert list(tmp_path.glob("mcp.json.bak.*")) == []


def test_tier1_repair_repoints_command_when_binary_is_on_path(tmp_path: Path) -> None:
    real_binary = tmp_path / "some-server"
    real_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    real_binary.chmod(0o755)
    mcp = _mcp_file(tmp_path, {"srv": {"command": "/nonexistent/some-server"}})

    res = execute_playbook(
        "repair_or_disable_mcp_server",
        {"config_path": str(mcp), "server_name": "srv", "alternate_binary": str(real_binary)},
    )

    assert res.success is True
    parsed = json.loads(mcp.read_text(encoding="utf-8"))
    assert parsed["mcpServers"]["srv"]["command"] == str(real_binary)
    assert "_disabled_mcpServers" not in parsed


def test_quarantine_is_gated_above_tier1() -> None:
    assert PLAYBOOK_CATALOG["quarantine_mcp_server"]["tier"] == 2
    assert PLAYBOOK_CATALOG["repair_or_disable_mcp_server"]["tier"] == 1
    assert PLAYBOOK_CATALOG["quarantine_mcp_server"]["domain"] == "mcp_config"


def test_atomic_write_leaves_original_intact_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "mcp.json"
    target.write_text('{"mcpServers": {}}', encoding="utf-8")

    def boom(_src: str, _dst: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(playbooks.os, "replace", boom)

    with pytest.raises(OSError):
        playbooks._atomic_write_text(target, "BROKEN", 0o644)

    assert target.read_text(encoding="utf-8") == '{"mcpServers": {}}'
    assert list(tmp_path.iterdir()) == [target]


# --------------------------------------------------------------------------- systemd injection guard


def test_systemd_unit_is_passed_after_double_dash(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_exec(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        commands.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(playbooks, "_exec", fake_exec)

    execute_playbook("restart_systemd_user_unit", {"unit": "-evil.service"})

    assert commands[0] == ["systemctl", "--user", "reset-failed", "--", "-evil.service"]
    assert commands[1] == ["systemctl", "--user", "restart", "--", "-evil.service"]


def test_systemd_rejects_metacharacter_units(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_exec(_cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        raise AssertionError("must not execute invalid unit")

    monkeypatch.setattr(playbooks, "_exec", fake_exec)

    res = execute_playbook("restart_systemd_user_unit", {"unit": "foo.service; rm -rf /"})

    assert res.executed is False
    assert res.success is False


# --------------------------------------------------------------------------- diagnostic tools


def test_inspect_mcp_config_handles_bad_payloads(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    report = json.loads(playbooks.inspect_mcp_config.invoke({"config_path": str(broken)}))
    assert "error" in report

    odd = tmp_path / "odd.json"
    odd.write_text('["not", "a", "mapping"]', encoding="utf-8")
    report = json.loads(playbooks.inspect_mcp_config.invoke({"config_path": str(odd)}))
    assert report["servers"] == {}

    missing = json.loads(playbooks.inspect_mcp_config.invoke({"config_path": str(tmp_path / "nope.json")}))
    assert missing["exists"] is False


# --------------------------------------------------------------------------- escalation notification


def test_send_escalation_notification_uses_omarchy_notification_send_with_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_exec(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(playbooks, "_exec", fake_exec)
    monkeypatch.setattr(playbooks.shutil, "which", lambda cmd: "/usr/share/omarchy/bin/omarchy-notification-send" if cmd == "omarchy-notification-send" else None)

    ok = playbooks.send_escalation_notification("Add Capability: foo", "What is needed: bar", incident_id=42)
    assert ok is True
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "omarchy-notification-send"
    assert "--urgency" in cmd and "critical" in cmd
    assert "--glyph" in cmd
    assert "--exec" in cmd
    exec_idx = cmd.index("--exec")
    exec_args = cmd[exec_idx + 1 :]
    assert any("on-improve" in arg for arg in exec_args)
    assert "42" in exec_args


def test_send_escalation_notification_falls_back_to_notify_send_when_no_omarchy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_exec(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        calls.append(cmd)
        return 0, "", ""

    monkeypatch.setattr(playbooks, "_exec", fake_exec)
    monkeypatch.setattr(playbooks.shutil, "which", lambda _cmd: None)

    ok = playbooks.send_escalation_notification("Add Capability: foo", "What is needed: bar")
    assert ok is True
    assert len(calls) == 1
    assert calls[0][0] == "notify-send"


def test_bin_on_improve_exists_and_is_executable() -> None:
    bin_path = Path(__file__).resolve().parents[1] / "bin" / "on-improve"
    assert bin_path.exists()
    mode = bin_path.stat().st_mode
    assert mode & stat.S_IXUSR


def test_free_disk_space_playbook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_exec(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        calls.append(cmd)
        return 0, "Vacuumed 20MB", ""

    fake_cache = tmp_path / ".cache" / "thumbnails"
    fake_cache.mkdir(parents=True)
    fake_thumb = fake_cache / "test.png"
    fake_thumb.write_bytes(b"x" * 1024)

    monkeypatch.setattr(playbooks, "_exec", fake_exec)
    monkeypatch.setattr(playbooks.Path, "home", lambda: tmp_path)

    res = execute_playbook("free_disk_space", {"path": "/"})
    assert res.executed is True
    assert res.success is True
    assert any("journalctl" in c[0] for c in calls)
    assert not fake_thumb.exists()


def test_restart_pipewire_playbook(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_exec(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        calls.append(cmd)
        return 0, "Restarted", ""

    monkeypatch.setattr(playbooks, "_exec", fake_exec)
    res = execute_playbook("restart_pipewire")
    assert res.executed is True
    assert res.success is True
    assert any("pipewire" in " ".join(c) for c in calls)


def test_reconnect_network_playbook(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_exec(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        calls.append(cmd)
        return 0, "Online", ""

    monkeypatch.setattr(playbooks, "_exec", fake_exec)
    monkeypatch.setattr(playbooks.shutil, "which", lambda cmd: "/usr/bin/nmcli" if cmd == "nmcli" else None)

    res = execute_playbook("reconnect_network")
    assert res.executed is True
    assert res.success is True
    assert any("networking" in c for c in calls)



