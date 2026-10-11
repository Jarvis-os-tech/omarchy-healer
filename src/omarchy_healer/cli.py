"""Rich CLI for Omarchy-Healer multi-agent system monitor and auto-remediator."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import HealerConfig
from .graph import OmarchyHealerOrchestrator
from .memory import IncidentMemoryStore

console = Console()


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
    watch_p.add_argument("--interval", type=float, default=15.0, help="Poll interval in seconds")
    watch_p.add_argument("--auto-heal", action="store_true", default=True, help="Enable automatic healing")
    watch_p.add_argument("--max-tier", type=int, default=None, choices=[0, 1, 2])

    hist_p = sub.add_parser("history", help="View past incidents and remediation outcomes from SQLite")
    hist_p.add_argument("--limit", type=int, default=20, help="Number of recent rows")

    return parser


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

    orchestrator = OmarchyHealerOrchestrator(config)

    if cmd == "scan":
        results = orchestrator.scan_and_process(auto_heal=False)
        if getattr(args, "json", False):
            print(json.dumps(results, indent=2))
        else:
            _render_results(results, auto_heal=False)
        return 0

    if cmd == "heal":
        results = orchestrator.scan_and_process(
            auto_heal=True,
            max_auto_tier=args.max_tier,
        )
        if getattr(args, "json", False):
            print(json.dumps(results, indent=2))
        else:
            _render_results(results, auto_heal=True)
        return 0

    if cmd == "watch":
        console.print(
            f"[bold cyan]Starting Omarchy-Healer watchdog[/bold cyan] "
            f"(interval={args.interval}s, auto_heal={args.auto_heal}, "
            f"max_tier={args.max_tier if args.max_tier is not None else config.max_auto_tier})"
        )
        try:
            while True:
                results = orchestrator.scan_and_process(
                    auto_heal=args.auto_heal,
                    max_auto_tier=args.max_tier,
                )
                if results:
                    _render_results(results, auto_heal=args.auto_heal)
                time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped Omarchy-Healer watchdog.[/yellow]")
            return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
