"""Rich CLI for Omarchy-Healer multi-agent system monitor and auto-remediator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import HealerConfig
from .formatting import clean_english_text, format_improvement_agent_prompt
from .graph import OmarchyHealerOrchestrator
from .memory import IncidentMemoryStore
from .models import SymptomEvent
from .playbooks import send_completion_notification, send_escalation_notification

console = Console()


def _json_safe(value: Any) -> Any:
    """Coerce arbitrary graph state into something ``json.dumps`` always accepts.

    Probe evidence holds ``Path``/``set``/model instances, and ``json`` would
    either raise ``TypeError`` or emit bare ``NaN``/``Infinity`` tokens that no
    strict JSON parser accepts.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _json_safe(dump())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return str(value)


def _render_providers(config: HealerConfig) -> None:
    table = Table(title="Omarchy-Healer Model & Fallback Chain", show_lines=True)
    table.add_column("Order / Role", style="cyan", no_wrap=True)
    table.add_column("Provider", style="bold magenta")
    table.add_column("Configured", justify="center")
    table.add_column("Endpoint / Model Detail", style="dim")

    for row in config.provider_chain_summary():
        badge = "[green]YES[/green]" if row["configured"] == "yes" else "[yellow]NO[/yellow]"
        table.add_row(row["role"], row["provider"], badge, row["detail"])

    console.print(table)


def _tag_for_reason(reason: str) -> str:
    if reason == "missing_playbook":
        return "Implement feature (Add capability to me)"
    elif reason == "insufficient_autonomy_tier":
        return "Permission required (Higher tier approval)"
    elif reason == "remediation_failed":
        return "Upgrade capability (Feature enhancement needed)"
    return "Action required"


def _render_results(results: list[dict[str, Any]], *, auto_heal: bool) -> None:
    if not results:
        console.print(
            Panel(
                "[bold green]All system probes healthy — 0 anomalies detected.[/bold green]",
                title="Omarchy-Healer Status",
                border_style="green",
            )
        )
        return

    table = Table(
        title=f"Omarchy-Healer Multi-Agent Report ({'AUTO-HEAL' if auto_heal else 'DRY-RUN SCAN'})",
        show_lines=True,
    )
    table.add_column("Incident", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Triage (Fallback Chain)", style="yellow")
    table.add_column("Playbook (Tier)", style="bold")
    table.add_column("Outcome", style="green")

    for state in results:
        sym = state.get("symptom") or {}
        triage = state.get("triage") or {}
        plan = state.get("plan") or {}
        ver = state.get("verification") or {}
        status = str(state.get("status") or "unknown")

        chain_str = " -> ".join(triage.get("fallback_chain_tried") or [triage.get("provider", "?")])
        triage_cell = (
            f"provider: [bold]{triage.get('provider', '?')}[/bold]\n"
            f"chain: {chain_str}\n"
            f"sev={triage.get('severity_score', 0):.2f} p={triage.get('actionable_probability', 0):.2f}"
        )
        pb_cell = f"{plan.get('playbook_id', triage.get('selected_playbook', '-'))} (T{plan.get('safety_tier', 0)})"
        outcome_detail = ver.get("post_check_summary") or plan.get("approval_reason") or status
        table.add_row(
            f"[bold]{sym.get('title', '')}[/bold]\n[dim]{sym.get('summary', '')}[/dim]",
            str(sym.get("domain", "")),
            triage_cell,
            pb_cell,
            f"[bold]{status.upper()}[/bold]\n{outcome_detail}",
        )

    console.print(table)
    for state in results:
        esc = state.get("escalation")
        if not isinstance(esc, dict) or not esc.get("agent_prompt"):
            continue
        sym = state.get("symptom") or {}
        missing_cap = clean_english_text(esc.get("missing_capability", ""))
        feature_spec = clean_english_text(esc.get("proposed_feature_spec", ""))
        agent_prompt = clean_english_text(esc.get("agent_prompt", ""))
        tag = _tag_for_reason(str(esc.get("reason", "")))
        body = (
            f"[bold blue]Tag:[/bold blue] {tag}\n"
            f"[bold cyan]Incident:[/bold cyan] {sym.get('title', '')}\n"
            f"[bold magenta]Reason:[/bold magenta] {str(esc.get('reason', '')).replace('_', ' ')} "
            f"[dim](synthesized by {esc.get('provider', 'heuristic')})[/dim]\n"
            f"[bold red]What is Needed:[/bold red] {missing_cap}\n"
            f"[bold green]Proposed Playbook:[/bold green] {esc.get('proposed_playbook_id', '')}\n"
            f"[bold yellow]Feature Specification:[/bold yellow] {feature_spec}\n\n"
            f"[bold white]Prompt to Give Capacity (pass to coding agent / developer):[/bold white]\n"
            f"[italic]{agent_prompt}[/italic]"
        )
        console.print(
            Panel(
                body,
                title="Capability Escalation — New Feature / Capacity Required",
                border_style="yellow",
            )
        )


def _render_history(db_path, limit: int) -> None:
    store = IncidentMemoryStore(db_path)
    rows = store.recent_incidents(limit=limit)
    if not rows:
        console.print("[dim]No incidents recorded in SQLite memory yet.[/dim]")
        return

    table = Table(title=f"Recent Incidents ({db_path})", show_lines=True)
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Domain")
    table.add_column("Title")
    table.add_column("Triage / Diag")
    table.add_column("Playbook (Tier)")
    table.add_column("Status")

    for r in rows:
        table.add_row(
            str(r["id"]),
            str(r["created_at"])[:19],
            str(r["domain"]),
            str(r["title"]),
            f"{r['triage_provider']} / {r['diagnosis_provider']}",
            f"{r['playbook_id']} (T{r['safety_tier']})",
            str(r["status"]),
        )

    console.print(table)



def _render_escalations(db_path, limit: int) -> None:
    store = IncidentMemoryStore(db_path)
    rows = store.recent_escalations(limit=limit)
    if not rows:
        console.print("[dim]No capability escalation requests recorded yet.[/dim]")
        return

    for r in rows:
        tag = _tag_for_reason(str(r.get("reason", "")))
        missing_cap = clean_english_text(r["missing_capability"])
        feature_spec = clean_english_text(r["proposed_feature_spec"])
        agent_prompt = clean_english_text(r["agent_prompt"])
        body = (
            f"[bold cyan]ID #{r['id']}[/bold cyan] (Incident #{r['incident_row_id']}) · "
            f"[dim]{str(r['created_at'])[:19]}[/dim]\n"
            f"[bold blue]Tag:[/bold blue] {tag}\n"
            f"[bold]Title:[/bold] {r['title']}\n"
            f"[bold magenta]Reason:[/bold magenta] {str(r['reason']).replace('_', ' ')} (provider={r['provider']})\n"
            f"[bold red]What is Needed:[/bold red] {missing_cap}\n"
            f"[bold green]Proposed Playbook:[/bold green] {r['proposed_playbook_id']}\n"
            f"[bold yellow]Feature Spec:[/bold yellow] {feature_spec}\n\n"
            f"[bold white]Prompt to Give Capacity:[/bold white]\n"
            f"[italic]{agent_prompt}[/italic]"
        )
        console.print(
            Panel(
                body,
                title=f"Capability Request #{r['id']} — {r['proposed_playbook_id']}",
                border_style="yellow",
            )
        )

def _positive_seconds(raw: str) -> float:
    """argparse type for poll intervals: reject NaN, infinity and non-positive values."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"invalid interval {raw!r}: expected a number of seconds"
        ) from None
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            f"invalid interval {raw!r}: must be a finite number of seconds greater than 0"
        )
    return value


def _positive_int(raw: str) -> int:
    """argparse type for counts: reject zero and negative values."""
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid count {raw!r}: expected an integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"invalid count {raw!r}: must be greater than 0")
    return value


@dataclass
class _SymptomEntry:
    """Suppression bookkeeping for one symptom fingerprint in a watchdog process."""

    fingerprint: str
    signature: str = ""
    failures: int = 0
    next_allowed_at: float = 0.0
    notified: bool = False
    seen_count: int = 0
    suppressed_count: int = 0
    last_seen_at: float = 0.0


class WatchSuppressor:
    """Dedup, cooldown and exponential backoff for the ``watch`` loop.

    Without it, an unfixable symptom (a systemd unit that will not start, an MCP
    binary that gets re-disabled on every pass) re-executes its playbook and
    re-fires its desktop notification every single interval, forever. Identical
    repeat sightings are suppressed; a genuinely different problem — different
    fingerprint, playbook, arguments or categorical evidence — is admitted
    immediately and resets the failure pacing.
    """

    _IDENTITY_FIELDS = ("probe_name", "fingerprint", "domain", "title", "suggested_playbook")

    def __init__(
        self,
        *,
        interval: float = 15.0,
        base_backoff: float | None = None,
        max_backoff: float = 900.0,
        repeat_after: float | None = None,
        metric_quantum: float | None = None,
        stale_after: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Build a suppressor.

        Args:
            interval: Poll interval of the owning loop; the unit of retry pacing.
            base_backoff: Delay after the first failed remediation; defaults to ``interval``.
            max_backoff: Ceiling for the exponential retry delay.
            repeat_after: Delay before re-examining an unchanged symptom that is
                not being retried as a failure (dry-run, blocked by the safety
                gate, or successfully fixed and then seen again).
            metric_quantum: When set, numeric metrics join the signature after
                being rounded to this quantum instead of being ignored.
            stale_after: Forget a fingerprint that has not been seen for this long.
            clock: Monotonic time source; injectable for tests.
        """
        self.interval = max(0.001, float(interval))
        self.base_backoff = self.interval if base_backoff is None else max(0.0, float(base_backoff))
        self.max_backoff = max(self.base_backoff, float(max_backoff))
        self.repeat_after = self.interval if repeat_after is None else max(0.0, float(repeat_after))
        if metric_quantum is not None and float(metric_quantum) <= 0.0:
            raise ValueError("metric_quantum must be greater than 0")
        self.metric_quantum = None if metric_quantum is None else abs(float(metric_quantum))
        self.stale_after = max(0.0, float(stale_after))
        self._clock = clock
        self._entries: dict[str, _SymptomEntry] = {}

    @staticmethod
    def _symptom_payload(symptom: SymptomEvent | Mapping[str, Any]) -> dict[str, Any]:
        dump = getattr(symptom, "model_dump", None)
        return dump() if callable(dump) else dict(symptom)

    @staticmethod
    def _is_categorical(value: Any) -> bool:
        if isinstance(value, (str, bytes, bool)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(WatchSuppressor._is_categorical(item) for item in value)
        return False

    def _evidence(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        """Keep categorical evidence only unless ``metric_quantum`` opts numbers in.

        Probes emit live measurements — RAM percentages, PSI averages, the
        seconds an overlay has been open — that differ on every cycle by
        construction. Folding those into the signature would relabel one
        unchanged, unresolved problem as "changed" every interval, which is the
        hammer loop this class exists to stop; their drift is paced by the
        exponential backoff instead. Categorical evidence (unit name, load
        state, ``osd_state``, ``critical``) really does describe the problem.
        """
        evidence: dict[str, Any] = {}
        for key, value in sorted(metrics.items(), key=lambda item: str(item[0])):
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if self.metric_quantum is not None and numeric:
                evidence[str(key)] = round(float(value) / self.metric_quantum) * self.metric_quantum
            elif self._is_categorical(value):
                evidence[str(key)] = value
        return evidence

    def _signature(self, payload: Mapping[str, Any]) -> str:
        identity = {field: _json_safe(payload.get(field)) for field in self._IDENTITY_FIELDS}
        identity["suggested_args"] = _json_safe(payload.get("suggested_args") or {})
        identity["evidence"] = self._evidence(payload.get("metrics") or {})
        canonical = json.dumps(identity, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else float(now)

    def should_process(
        self,
        symptom: SymptomEvent | Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> bool:
        """Return True when this sighting must go through the agent pipeline."""
        moment = self._now(now)
        payload = self._symptom_payload(symptom)
        fingerprint = str(payload.get("fingerprint") or "")
        signature = self._signature(payload)

        self.prune(now=moment)

        entry = self._entries.get(fingerprint)
        if entry is None:
            entry = _SymptomEntry(fingerprint=fingerprint, signature=signature)
            self._entries[fingerprint] = entry
            entry.last_seen_at = moment
            entry.seen_count = 1
            return True

        if entry.signature != signature:
            entry.signature = signature
            entry.failures = 0
            entry.notified = False
            entry.next_allowed_at = 0.0
            entry.last_seen_at = moment
            entry.seen_count += 1
            return True

        entry.last_seen_at = moment
        entry.seen_count += 1
        if moment < entry.next_allowed_at:
            entry.suppressed_count += 1
            return False
        return True

    def record(self, state: Mapping[str, Any], *, now: float | None = None) -> None:
        """Fold the outcome of a processed symptom back into the suppression state."""
        moment = self._now(now)
        payload = self._symptom_payload(state.get("symptom") or {})
        fingerprint = str(payload.get("fingerprint") or "")
        entry = self._entries.get(fingerprint)
        if entry is None:
            entry = _SymptomEntry(fingerprint=fingerprint, signature=self._signature(payload))
            self._entries[fingerprint] = entry

        verification = state.get("verification") or {}
        execution = state.get("execution") or {}
        resolved = bool(verification.get("resolved"))
        executed = bool(execution.get("executed"))

        if resolved:
            entry.failures = 0
            entry.notified = True
            entry.next_allowed_at = moment + self.repeat_after
        elif executed:
            entry.failures += 1
            entry.next_allowed_at = moment + self.retry_delay_for(entry.failures)
        else:
            entry.failures = 0
            entry.next_allowed_at = moment + self.repeat_after
        entry.last_seen_at = moment

    def retry_delay_for(self, failures: int) -> float:
        """Exponential backoff for ``failures`` consecutive failed remediations."""
        if failures <= 0:
            return 0.0
        exponent = min(failures - 1, 32)
        return min(self.max_backoff, self.base_backoff * (2**exponent))

    def should_notify(
        self,
        symptom: SymptomEvent | Mapping[str, Any] | str,
        *,
        now: float | None = None,
    ) -> bool:
        """Return True at most once per fingerprint, until that fingerprint changes."""
        moment = self._now(now)
        if isinstance(symptom, str):
            entry = self._entries.get(symptom)
            if entry is None:
                return True
            return not entry.notified

        payload = self._symptom_payload(symptom)
        fingerprint = str(payload.get("fingerprint") or "")
        entry = self._entries.get(fingerprint)
        if entry is None:
            entry = _SymptomEntry(fingerprint=fingerprint, signature=self._signature(payload))
            self._entries[fingerprint] = entry
            entry.notified = True
            entry.last_seen_at = moment
            return True
        if entry.signature and entry.signature != self._signature(payload):
            entry.signature = self._signature(payload)
            entry.notified = False
            entry.failures = 0
            entry.next_allowed_at = 0.0
        if entry.notified:
            return False
        entry.notified = True
        entry.last_seen_at = moment
        return True

    def retry_delay(self, fingerprint: str, *, now: float | None = None) -> float:
        """Seconds until the given fingerprint is admitted again."""
        entry = self._entries.get(fingerprint)
        if entry is None:
            return 0.0
        return max(0.0, entry.next_allowed_at - self._now(now))

    def failure_count(self, fingerprint: str) -> int:
        """Consecutive failed remediations recorded for a fingerprint."""
        entry = self._entries.get(fingerprint)
        return 0 if entry is None else entry.failures

    def pending_retries(self, *, now: float | None = None) -> list[tuple[str, float]]:
        """Fingerprints currently suppressed, with their remaining delay."""
        moment = self._now(now)
        pending = [
            (entry.fingerprint, max(0.0, entry.next_allowed_at - moment))
            for entry in self._entries.values()
            if entry.suppressed_count and entry.next_allowed_at > moment
        ]
        return sorted(pending, key=lambda item: item[0])

    def prune(self, *, now: float | None = None) -> None:
        """Forget fingerprints that have not been seen for ``stale_after`` seconds."""
        if self.stale_after <= 0.0:
            return
        moment = self._now(now)
        for fingerprint, entry in list(self._entries.items()):
            if moment - entry.last_seen_at > self.stale_after:
                del self._entries[fingerprint]

    def known_fingerprints(self) -> list[str]:
        """Fingerprints currently tracked, for diagnostics and tests."""
        return sorted(self._entries)


class WatchOrchestrator(OmarchyHealerOrchestrator):
    """Orchestrator that suppresses repeated, unchanged, already-handled symptoms.

    Suppression has to happen *before* the graph runs: the Executor and the
    completion notifier live inside the pipeline, so filtering results
    afterwards would still re-execute the playbook and re-fire the desktop
    notification on every interval.
    """

    def __init__(
        self,
        config: HealerConfig | None = None,
        *,
        suppressor: WatchSuppressor | None = None,
        **kwargs: Any,
    ) -> None:
        """Build a watch orchestrator bound to a suppressor."""
        super().__init__(config, **kwargs)
        self.suppressor = suppressor or WatchSuppressor()

    def notify_completion(self, symptom: SymptomEvent, title: str, body: str) -> bool:
        """Notify at most once per fingerprint, until that fingerprint actually changes.

        ``WatchSuppressor.should_notify`` already exists for this; without it an
        unfixable symptom re-fired its desktop notification every single interval.
        """
        if not self.suppressor.should_notify(symptom):
            return False
        return send_completion_notification(title, body)

    def notify_escalation(
        self,
        symptom: SymptomEvent,
        title: str,
        body: str,
        *,
        incident_id: int | None = None,
        exec_cmd: list[str] | None = None,
    ) -> bool:
        """Same once-per-fingerprint gate as :meth:`notify_completion`."""
        if not self.suppressor.should_notify(symptom):
            return False
        try:
            return send_escalation_notification(title, body, incident_id=incident_id, exec_cmd=exec_cmd)
        except TypeError:
            return send_escalation_notification(title, body)

    def scan_and_process(
        self,
        *,
        auto_heal: bool = False,
        max_auto_tier: int | None = None,
    ) -> list[dict[str, Any]]:
        """Probe, drop already-handled unchanged symptoms, then run the pipeline."""
        results: list[dict[str, Any]] = []
        for symptom in self.probe_runner(self.config):
            if not self.suppressor.should_process(symptom):
                continue
            state = self.process_symptom(
                symptom,
                auto_heal=auto_heal,
                max_auto_tier=max_auto_tier,
            )
            self.suppressor.record(state)
            results.append(state)
        return results


def run_watch(
    orchestrator: WatchOrchestrator,
    *,
    interval: float,
    auto_heal: bool,
    max_auto_tier: int | None,
    sleep: Callable[[float], None] = time.sleep,
    max_cycles: int | None = None,
    render: Callable[[list[dict[str, Any]]], None] | None = None,
) -> int:
    """Run the watchdog until interrupted, surviving individual failed cycles.

    ``max_cycles`` bounds the loop for tests; production callers leave it unset.
    """
    renderer = render or (lambda results: _render_results(results, auto_heal=auto_heal))
    cycle = 0
    try:
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            try:
                results = orchestrator.scan_and_process(
                    auto_heal=auto_heal,
                    max_auto_tier=max_auto_tier,
                )
            except Exception as exc:
                console.print(
                    f"[red]Watch cycle failed ({type(exc).__name__}: {exc}); "
                    f"retrying in {interval:g}s[/red]"
                )
            else:
                if results:
                    renderer(results)
                pending = orchestrator.suppressor.pending_retries()
                if pending:
                    summary = ", ".join(f"{fp} in {delay:.0f}s" for fp, delay in pending)
                    console.print(f"[dim]Suppressed unchanged symptom(s): {summary}[/dim]")
            sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped Omarchy-Healer watchdog.[/yellow]")
        return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omarchy-healer",
        description="LangGraph + Laya (with Gemini & NVIDIA NIM fallbacks) system monitor and auto-healer",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("providers", help="Show configured model providers and fallback order")

    scan_p = sub.add_parser("scan", help="Run all system probes and multi-agent diagnosis in dry-run mode")
    scan_p.add_argument("--json", action="store_true", help="Output raw JSON state")

    heal_p = sub.add_parser("heal", help="Scan and automatically execute approved remediation playbooks")
    heal_p.add_argument(
        "--max-tier",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help="Maximum safety tier to auto-execute (0=silent safe, 1=auto-fix with backup/restart, 2=all)",
    )
    heal_p.add_argument("--json", action="store_true", help="Output raw JSON state")

    watch_p = sub.add_parser("watch", help="Continuously monitor system anomalies and auto-remediate")
    watch_p.add_argument(
        "--interval",
        type=_positive_seconds,
        default=15.0,
        help="Poll interval in seconds (must be a finite number greater than 0)",
    )
    watch_p.add_argument(
        "--auto-heal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Execute approved playbooks (tier <= --max-tier) every cycle. "
            "Defaults to OFF so an unattended watchdog never mutates the live "
            "desktop without an explicit opt-in: use --no-auto-heal to be "
            "explicit, --auto-heal to remediate."
        ),
    )
    watch_p.add_argument(
        "--max-tier",
        type=int,
        default=None,
        choices=[0, 1, 2],
        help=(
            "Maximum safety tier to auto-execute (0=silent safe, 1=auto-fix with "
            "backup/restart, 2=all); defaults to the configured tier"
        ),
    )

    hist_p = sub.add_parser("history", help="View past incidents and remediation outcomes from SQLite")
    hist_p.add_argument("--limit", type=_positive_int, default=20, help="Number of recent rows (greater than 0)")


    esc_p = sub.add_parser(
        "escalations",
        help="View capability escalation prompts for issues requiring new playbooks/features",
    )
    esc_p.add_argument("--limit", type=_positive_int, default=10, help="Number of recent rows (greater than 0)")
    esc_p.add_argument("--json", action="store_true", help="Output raw JSON capability requests")

    imp_p = sub.add_parser(
        "improve",
        help="Launch default AI coding agent with diagnostic context to improve omarchy-healer",
    )
    imp_p.add_argument(
        "incident_id",
        nargs="?",
        default=None,
        help="Specific capability request / incident ID (defaults to newest)",
    )
    imp_p.add_argument(
        "--inline",
        action="store_true",
        help="Launch agent inline in current terminal instead of opening TUI window",
    )
    imp_p.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print generated prompt and exit without launching agent",
    )
    return parser


def _run_improve(
    config: HealerConfig,
    incident_id: str | None = None,
    *,
    inline: bool = False,
    print_prompt: bool = False,
) -> int:
    store = IncidentMemoryStore(config.db_path)
    esc = None
    if incident_id:
        try:
            esc = store.get_escalation(int(incident_id))
        except ValueError:
            esc = None
    if not esc:
        recent = store.recent_escalations(limit=1)
        if recent:
            esc = recent[0]

    if not esc:
        console.print("[yellow]No capability escalation requests found in database.[/yellow]")
        return 1

    prompt = format_improvement_agent_prompt(
        incident_id=esc["id"],
        title=esc.get("title", ""),
        domain=esc.get("domain", ""),
        reason=esc.get("reason", ""),
        missing_capability=esc.get("missing_capability", ""),
        proposed_playbook_id=esc.get("proposed_playbook_id", ""),
        proposed_feature_spec=esc.get("proposed_feature_spec", ""),
        agent_prompt_override=esc.get("agent_prompt", ""),
        created_at=str(esc.get("created_at", "")),
    )

    if print_prompt:
        console.print(Panel(prompt, title=f"Improvement Prompt for #{esc['id']}", border_style="cyan"))
        return 0

    repo_dir = Path(__file__).resolve().parents[2]
    if not repo_dir.exists():
        repo_dir = Path("/home/g0pi/Projects/omarchy-healer")

    if repo_dir.is_dir():
        os.chdir(repo_dir)

    agent_bin = shutil.which("omarchy-agent")
    if not agent_bin:
        console.print(
            "[yellow]Warning: 'omarchy-agent' not found in PATH.[/yellow]\n"
            "Below is the prompt to improve omarchy-healer:\n"
        )
        console.print(Panel(prompt, title=f"Improvement Prompt for #{esc['id']}", border_style="cyan"))
        return 0

    cmd = [agent_bin]
    if inline:
        cmd.append("--inline")
    cmd.extend(["--prompt", prompt])

    console.print(
        f"[bold cyan]Launching default AI agent to improve omarchy-healer (Incident #{esc['id']})...[/bold cyan]"
    )
    res = subprocess.run(cmd)
    return getattr(res, "returncode", 0) or 0



def _orchestrator_for(config: HealerConfig, command: str) -> OmarchyHealerOrchestrator:
    """``watch`` needs the suppression-aware variant; the one-shot commands do not.

    ``scan`` is report-only (``auto_heal=False`` can never reach the executor), so it
    reports an OSD that is open right now instead of waiting out the stuck timer; a
    single sweep cannot know how long the popup has been up. Every command that can
    execute a playbook keeps the full default threshold.
    """
    if command == "watch":
        return WatchOrchestrator(config)
    if command == "scan":
        return OmarchyHealerOrchestrator(config, osd_stuck_seconds=0.0)
    return OmarchyHealerOrchestrator(config)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = HealerConfig.from_env()

    cmd = args.command or "scan"
    if cmd == "providers":
        _render_providers(config)
        return 0

    if cmd == "history":
        _render_history(config.db_path, args.limit)
        return 0

    if cmd == "escalations":
        if getattr(args, "json", False):
            rows = IncidentMemoryStore(config.db_path).recent_escalations(limit=args.limit)
            print(json.dumps(_json_safe(rows), indent=2, allow_nan=False))
        else:
            _render_escalations(config.db_path, args.limit)
        return 0

    if cmd == "improve":
        return _run_improve(
            config,
            args.incident_id,
            inline=args.inline,
            print_prompt=args.print_prompt,
        )


    orchestrator = _orchestrator_for(config, cmd)

    if cmd == "scan":
        results = orchestrator.scan_and_process(auto_heal=False)
        if getattr(args, "json", False):
            print(json.dumps(_json_safe(results), indent=2, allow_nan=False))
        else:
            _render_results(results, auto_heal=False)
        return 0

    if cmd == "heal":
        results = orchestrator.scan_and_process(
            auto_heal=True,
            max_auto_tier=args.max_tier,
        )
        if getattr(args, "json", False):
            print(json.dumps(_json_safe(results), indent=2, allow_nan=False))
        else:
            _render_results(results, auto_heal=True)
        return 0

    if cmd == "watch":
        console.print(
            f"[bold cyan]Starting Omarchy-Healer watchdog[/bold cyan] "
            f"(interval={args.interval:g}s, auto_heal={args.auto_heal}, "
            f"max_tier={args.max_tier if args.max_tier is not None else config.max_auto_tier})"
        )
        if not args.auto_heal:
            console.print(
                "[yellow]Report-only watchdog: no playbook will be executed. "
                "Pass --auto-heal to remediate automatically.[/yellow]"
            )
        return run_watch(
            orchestrator,  # type: ignore[arg-type]
            interval=args.interval,
            auto_heal=args.auto_heal,
            max_auto_tier=args.max_tier,
        )

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())