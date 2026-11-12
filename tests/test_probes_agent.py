"""Probe regression tests: OSD stuck-timer persistence and hyprctl layers parsing."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from omarchy_healer import probes
from omarchy_healer.config import HealerConfig
from omarchy_healer.graph import domain_compatible
from omarchy_healer.models import SymptomEvent
from omarchy_healer.playbooks import PLAYBOOK_CATALOG
from omarchy_healer.probes import (
    DEFAULT_OSD_STUCK_SECONDS,
    CoreDumpProbe,
    MemoryProcessProbe,
    OmarchyUiProbe,
    collect_all_symptoms,
)

BAR_ONLY_LAYERS: dict[str, Any] = {
    "levels": {"0": [{"namespace": "omarchy-bar", "type": "quickshell"}]}
}


class _FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        """Wall clock too: CoreDumpProbe ages its claim state in real seconds."""
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeShell:
    """Canned responses for every command any probe runs."""

    def __init__(
        self,
        *,
        osd_state: str = "closed",
        osd_rc: int = 0,
        layers: str = json.dumps({"DP-1": BAR_ONLY_LAYERS}),
        layers_rc: int = 0,
        config_errors: str = "",
    ) -> None:
        self.osd_state = osd_state
        self.osd_rc = osd_rc
        self.layers = layers
        self.layers_rc = layers_rc
        self.config_errors = config_errors

    def __call__(self, cmd: list[str], *, timeout: float = 4.0) -> tuple[int, str, str]:
        key = " ".join(cmd)
        if key == "omarchy-shell osd state":
            return self.osd_rc, self.osd_state, ""
        if key == "hyprctl layers -j":
            return self.layers_rc, self.layers, ""
        if key == "hyprctl configerrors":
            return 0, self.config_errors, ""
        if key.startswith("systemctl --user"):
            return 0, "[]", ""
        if key.startswith("ps "):
            return 1, "", ""
        return 127, "", "not found"


@pytest.fixture(autouse=True)
def _clean_probe_cache() -> Iterator[None]:
    probes.reset_probe_cache()
    yield
    probes.reset_probe_cache()


@pytest.fixture()
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(probes, "time", fake)
    return fake


def _config(tmp_path: Path) -> HealerConfig:
    return HealerConfig(
        db_path=tmp_path / "incidents.db",
        mcp_config_path=tmp_path / "mcp.json",
    )


def _by_fingerprint(events: list[SymptomEvent], key: str) -> list[SymptomEvent]:
    return [ev for ev in events if ev.fingerprint.endswith(key)]


def test_osd_timer_persists_across_sweeps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    shell = _FakeShell(osd_state="open")
    monkeypatch.setattr(probes, "run_cmd", shell)
    cfg = _config(tmp_path)

    first = collect_all_symptoms(cfg, osd_stuck_seconds=30.0)
    assert _by_fingerprint(first, "stuck_omarchy_osd") == []

    clock.advance(31.0)
    second = collect_all_symptoms(cfg, osd_stuck_seconds=30.0)
    stuck = _by_fingerprint(second, "stuck_omarchy_osd")
    assert len(stuck) == 1
    assert stuck[0].metrics["observed_open_seconds"] == pytest.approx(31.0, abs=0.1)
    assert stuck[0].metrics["osd_layer_present"] is False


def test_transient_osd_under_threshold_emits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    shell = _FakeShell(osd_state="open")
    monkeypatch.setattr(probes, "run_cmd", shell)
    cfg = _config(tmp_path)

    for _ in range(3):
        events = collect_all_symptoms(cfg, osd_stuck_seconds=30.0)
        assert _by_fingerprint(events, "stuck_omarchy_osd") == []
        clock.advance(5.0)


def test_osd_timer_resets_when_osd_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock
) -> None:
    shell = _FakeShell(osd_state="open")
    monkeypatch.setattr(probes, "run_cmd", shell)
    cfg = _config(tmp_path)

    collect_all_symptoms(cfg, osd_stuck_seconds=30.0)
    clock.advance(45.0)
    assert _by_fingerprint(collect_all_symptoms(cfg, osd_stuck_seconds=30.0), "stuck_omarchy_osd")

    shell.osd_state = "closed"
    closed_sweep = collect_all_symptoms(cfg, osd_stuck_seconds=30.0)
    assert _by_fingerprint(closed_sweep, "stuck_omarchy_osd") == []

    shell.osd_state = "open"
    clock.advance(1.0)
    reopened = collect_all_symptoms(cfg, osd_stuck_seconds=30.0)
    assert _by_fingerprint(reopened, "stuck_omarchy_osd") == []


def test_probe_cache_is_reused_per_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    first = probes.get_probe_set(cfg)
    assert probes.get_probe_set(_config(tmp_path)) is first

    other = HealerConfig(
        db_path=tmp_path / "incidents.db",
        mcp_config_path=tmp_path / "other-mcp.json",
    )
    assert probes.get_probe_set(other) is not first


@pytest.mark.parametrize(
    "payload",
    [
        [{"levels": {"0": [{"namespace": "omarchy-osd"}, {"namespace": "omarchy-bar"}]}}],
        {"DP-1": {"levels": {"0": [{"namespace": "omarchy-osd"}, {"namespace": "omarchy-bar"}]}}},
    ],
    ids=["legacy-list", "monitor-dict"],
)
def test_layers_parser_handles_list_and_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clock: _FakeClock, payload: Any
) -> None:
    monkeypatch.setattr(probes, "run_cmd", _FakeShell(layers=json.dumps(payload)))
    probe = OmarchyUiProbe(_config(tmp_path))

    events = probe.check(osd_stuck_seconds=0.0)

    stuck = _by_fingerprint(events, "stuck_omarchy_osd")
    assert len(stuck) == 1
    assert stuck[0].metrics["osd_layer_present"] is True
    assert _by_fingerprint(events, "missing_omarchy_bar") == []
    assert stuck[0].raw_evidence["layers"]


@pytest.mark.parametrize(
    "raw",
    ['"not json at all"', "null", "{}", "[]", '{"DP-1": null}', '{"DP-1": {"levels": 5}}', "[1, 2, 3]"],
)
def test_malformed_layers_payload_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setattr(probes, "run_cmd", _FakeShell(osd_state="closed", layers=raw))
    events = OmarchyUiProbe(_config(tmp_path)).check(osd_stuck_seconds=0.0)
    assert isinstance(events, list)


def test_missing_bar_detected_in_legacy_list_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy_list = [{"levels": {"0": [{"namespace": "hyprland-overlay"}]}}]
    monkeypatch.setattr(probes, "run_cmd", _FakeShell(layers=json.dumps(legacy_list)))
    events = OmarchyUiProbe(_config(tmp_path)).check()
    assert len(_by_fingerprint(events, "missing_omarchy_bar")) == 1


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        ("open", True),
        ("  OPEN  ", True),
        ('{"state": "open"}', True),
        ("osd_state=open", True),
        ("state: open", True),
        ('"open"', True),
        ("closed", False),
        ("", False),
        ("not open", False),
    ],
)
def test_osd_state_matching_is_tolerant(out: str, expected: bool) -> None:
    assert probes._osd_reports_open(0, out) is expected


def test_osd_state_missing_command_is_not_open() -> None:
    assert probes._osd_reports_open(127, "open") is False
    assert probes._osd_reports_open(1, "open") is False


def test_default_osd_threshold_leaves_room_to_read_a_popup() -> None:
    """A popup a human is still reading must never be classified as stuck."""
    assert DEFAULT_OSD_STUCK_SECONDS == 90.0
    assert probes.DEFAULT_OSD_STUCK_SECONDS == 90.0


def _ps_with_zombies(*rows: tuple[int, int, str]) -> str:
    lines = ["  PID  PPID STAT COMMAND"]
    lines.extend(f"{pid:>5} {ppid:>5} Z    {cmd}" for pid, ppid, cmd in rows)
    return "\n".join(lines)


def test_already_signalled_zombies_stay_visible_in_the_symptom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: acknowledged zombies used to be dropped from the probe output.

    A parent that never calls wait() keeps its zombies forever. Filtering them out made
    a permanent leak look resolved, which reported "HEALED" and fired a "Fixed via
    reap_zombie_parents" notification for a problem that was still there.
    """
    monkeypatch.setattr(probes, "is_zombie_acknowledged", lambda pid, ppid: True)
    monkeypatch.setattr(
        probes,
        "run_cmd",
        lambda cmd, **kw: (0, _ps_with_zombies(*[(100 + i, 42, "leaky") for i in range(7)]), ""),
    )

    events = MemoryProcessProbe(_config(Path("/tmp"))).check()

    zombie = _by_fingerprint(events, "zombie_process_accumulation")
    assert len(zombie) == 1
    assert zombie[0].metrics["zombie_count"] == 7
    assert zombie[0].metrics["unsignalled_count"] == 0
    # Already-signalled parents are not offered to the playbook again.
    assert zombie[0].suggested_args["zombies"] == []


def test_unsignalled_zombies_are_the_ones_offered_for_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probes, "is_zombie_acknowledged", lambda pid, ppid: pid < 105)
    monkeypatch.setattr(
        probes,
        "run_cmd",
        lambda cmd, **kw: (0, _ps_with_zombies(*[(100 + i, 42, "leaky") for i in range(7)]), ""),
    )

    events = MemoryProcessProbe(_config(Path("/tmp"))).check()

    zombie = _by_fingerprint(events, "zombie_process_accumulation")[0]
    assert zombie.metrics["zombie_count"] == 7
    assert zombie.metrics["unsignalled_count"] == 2
    assert [z["pid"] for z in zombie.suggested_args["zombies"]] == [105, 106]

def test_memory_probe_suggests_target_reclaim_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MemoryProcessProbe,
        "_read_meminfo",
        lambda _self: {"used_percent": 85.0, "total_mb": 8000.0, "available_mb": 1200.0},
    )
    monkeypatch.setattr(MemoryProcessProbe, "_read_memory_psi", lambda _self: {"some_avg10": 0.0, "full_avg10": 0.0})
    monkeypatch.setattr(MemoryProcessProbe, "_top_rss_processes", lambda _self, limit=6: [])
    monkeypatch.setattr(MemoryProcessProbe, "_find_zombies", lambda _self: [])

    events = MemoryProcessProbe(_config(Path("/tmp"))).check()
    assert len(events) == 1
    ev = events[0]
    assert ev.domain == "memory_pressure"
    assert ev.suggested_playbook == "reclaim_memory_caches"
    assert "target_reclaim_mb" in ev.suggested_args
    assert ev.suggested_args["target_reclaim_mb"] > 256.0


def test_critical_memory_probe_suggests_clean_user_memory_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        MemoryProcessProbe,
        "_read_meminfo",
        lambda _self: {"used_percent": 93.5, "total_mb": 8000.0, "available_mb": 520.0},
    )
    monkeypatch.setattr(MemoryProcessProbe, "_read_memory_psi", lambda _self: {"some_avg10": 0.0, "full_avg10": 0.0})
    monkeypatch.setattr(MemoryProcessProbe, "_top_rss_processes", lambda _self, limit=6: [])
    monkeypatch.setattr(MemoryProcessProbe, "_find_zombies", lambda _self: [])

    events = MemoryProcessProbe(_config(Path("/tmp"))).check()
    assert len(events) == 1
    ev = events[0]
    assert ev.domain == "memory_pressure"
    assert ev.suggested_playbook == "clean_user_memory_caches"
    assert ev.default_tier == 1
    assert ev.metrics["critical"] is True

def test_restart_omarchy_shell_is_not_auto_approved() -> None:
    """The watchdog must never rebuild the whole shell surface unattended."""
    assert PLAYBOOK_CATALOG["restart_omarchy_shell"]["tier"] >= 2


# --- crash-watch / systemd-coredump wiring ----------------------------------


def _dump(**over: Any) -> dict[str, Any]:
    """One journal record shaped like the MESSAGE_ID omarchy-crash-watch tails."""
    record: dict[str, Any] = {
        "_UID": str(os.getuid()),
        "COREDUMP_PID": "4242",
        "COREDUMP_TIMESTAMP": str(int(time.time() * 1_000_000)),
        "COREDUMP_SIGNAL_NAME": "SIGSEGV",
        "COREDUMP_COMM": "buggy",
        "COREDUMP_EXE": "/usr/bin/buggy",
        "COREDUMP_FILENAME": "/var/lib/systemd/coredump/buggy.4242.sig",
    }
    record.update(over)
    return record


def _journal_cmd(entries: list[dict[str, Any]]):
    """run_cmd stand-in: coredump JSON for the journal query, benign answers elsewhere."""
    payload = "\n".join(json.dumps(entry) for entry in entries)

    def _run(cmd: list[str], **_kw: Any) -> tuple[int, str, str]:
        if any(str(part).startswith("MESSAGE_ID=") for part in cmd):
            return 0, payload, ""
        if cmd and str(cmd[0]).endswith("systemctl"):
            return 0, "[]", ""
        return 1, "", ""

    return _run


def test_service_owned_crash_offers_the_restart_playbook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probes, "run_cmd", _journal_cmd([_dump(COREDUMP_USER_UNIT="jarvis-oma.service")])
    )

    events = CoreDumpProbe(_config(tmp_path)).check()

    assert len(events) == 1
    crash = events[0]
    assert crash.domain == "systemd_service"
    assert crash.suggested_playbook == "restart_systemd_user_unit"
    assert crash.suggested_args == {"unit": "jarvis-oma.service"}
    # Tier 1 is what the running watchdog is allowed to execute unattended, so
    # a service-owned crash is fixed rather than queued for a human.
    assert PLAYBOOK_CATALOG["restart_systemd_user_unit"]["tier"] == 1
    # The pairing must survive the safety gate, or the tier above is irrelevant.
    assert domain_compatible(crash.domain, crash.suggested_playbook)[0] is True
    assert crash.default_tier <= PLAYBOOK_CATALOG["restart_systemd_user_unit"]["tier"]


def test_gui_crash_is_recorded_not_restarted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probes,
        "run_cmd",
        _journal_cmd([_dump(COREDUMP_USER_UNIT="app-org.chromium.Chromium-4587.scope")]),
    )

    events = CoreDumpProbe(_config(tmp_path)).check()

    assert len(events) == 1
    crash = events[0]
    assert crash.domain == "other"
    assert crash.suggested_playbook == "inspect_only"
    assert domain_compatible(crash.domain, crash.suggested_playbook)[0] is True


def test_crash_is_claimed_once_across_processes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """crash-watch triggers `heal` in a fresh process while `watch` sweeps from its own."""
    monkeypatch.setattr(probes, "run_cmd", _journal_cmd([_dump()]))

    assert len(CoreDumpProbe(_config(tmp_path)).check()) == 1
    assert CoreDumpProbe(_config(tmp_path)).check() == []


def test_foreign_crashes_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(probes, "run_cmd", _journal_cmd([_dump(_UID=str(os.getuid() + 1))]))

    assert CoreDumpProbe(_config(tmp_path)).check() == []


def test_one_crash_loop_earns_one_incident_per_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now_us = int(time.time() * 1_000_000)
    first = _dump(COREDUMP_PID="100", COREDUMP_TIMESTAMP=str(now_us))
    looped = _dump(COREDUMP_PID="101", COREDUMP_TIMESTAMP=str(now_us + 60_000_000))
    monkeypatch.setattr(probes, "run_cmd", _journal_cmd([first, looped]))

    assert len(CoreDumpProbe(_config(tmp_path)).check()) == 1


def test_collect_all_symptoms_includes_the_crash_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        probes, "run_cmd", _journal_cmd([_dump(COREDUMP_USER_UNIT="jarvis-oma.service")])
    )

    events = collect_all_symptoms(_config(tmp_path))

    assert any(ev.probe_name == "coredump" for ev in events)