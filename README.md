# Omarchy-Healer

Multi-agent system monitor and self-healing orchestrator for Omarchy / Hyprland Linux built on **LangGraph**, **LangChain**, and **NVIDIA NIM** with automatic failover to **Google Gemini** and offline deterministic heuristics.
## Architecture

1. **Deterministic Probes (`probes.py`)**
   - `OmarchyUiProbe`: Detects stuck `omarchy-osd` layer overlays (e.g., `duration: 0` launch popups), missing `omarchy-bar` Quickshell surfaces, and `hyprctl configerrors`.
   - `McpAndServiceProbe`: Validates `~/.omp/agent/mcp.json` server binaries (`ENOENT` detection) and `systemctl --user --failed` units.
   - `MemoryProcessProbe`: Monitors `/proc/meminfo`, `/proc/pressure/memory` (PSI), top RSS consumers, and defunct zombie processes.

2. **Triage & Safety Judge (`laya_judge.py`)**
   - **Primary**: NVIDIA NIM (`NVIDIA_NIM_API_KEY` via `langchain-openai` against `https://integrate.api.nvidia.com/v1`).
   - **Fallback 1**: Google Gemini (`GEMINI_API_KEY` via `langchain-google-genai`).
   - **Fallback 2**: Calibrated deterministic rule matcher when offline.
3. **LangGraph Multi-Agent Workflow (`graph.py`)**
   - `triage` -> `diagnostician` -> `planner` -> `safety_gate` -> `executor` -> `verifier` -> `capability_escalator` (when an issue fails remediation or lacks a registered playbook/autonomy tier) -> `record_memory` (SQLite `~/.local/share/omarchy-healer/incidents.db`).
## Quick Start

```bash
cd ~/Projects/omarchy-healer
cp .env.example .env   # Or rely on automatic fallback to ~/Projects/Omarchy-AI/.env

# Check configured provider chain (NVIDIA NIM -> Gemini -> Heuristic)
uv run omarchy-healer providers

# Dry-run scan of live system anomalies
uv run omarchy-healer scan

# Auto-heal Tier 0 and Tier 1 issues and verify resolution
uv run omarchy-healer heal --max-tier 1

# Run continuous background watchdog
uv run omarchy-healer watch --interval 15 --auto-heal --max-tier 1

# View SQLite incident history and capability escalation prompts
uv run omarchy-healer history
uv run omarchy-healer escalations

# Launch default AI agent with diagnostic context to implement or upgrade playbooks
uv run omarchy-healer improve
uv run omarchy-healer improve <incident_id>
```

## Safety tiers

| Tier | Meaning | Auto-approved at `--max-tier 1` |
| ---- | ------- | ------------------------------- |
| 0 | Silent, no-op-ish (`close_stuck_osd`, `reclaim_memory_caches`, `inspect_only`) | yes |
| 1 | Mutating but reversible (`restart_systemd_user_unit`, `reload_hyprland_config`, `reap_zombie_parents`, `repair_or_disable_mcp_server`) | yes |
| 2 | Disruptive or destructive (`restart_omarchy_shell`, `quarantine_mcp_server`) | **no** — needs `--max-tier 2` or manual approval |

Two things deliberately refuse to act on their own:

- **A popup you are still reading.** `close_stuck_osd` only fires once an OSD has been
  open for `DEFAULT_OSD_STUCK_SECONDS` (90s), and only in the mutating commands. `scan`
  reports an open OSD immediately because a single sweep cannot know how long it has
  been up. An earlier default of `0.0` on the orchestrator meant *any* popup was
  classified stuck the instant it opened.
- **A zombie leak it cannot reap.** Only a parent calling `wait()` releases its
  children, so `reap_zombie_parents` nudges the parent with `SIGCHLD`. Zombies that
  survive stay in the probe output (flagged `acknowledged`) so a permanent leak keeps
  reporting instead of quietly turning into a "fixed" notification.

A missing MCP server binary is a *deployment* problem, so `repair_or_disable_mcp_server`
only repoints `command` at a binary that actually exists. It never deletes your config
entry: when nothing on `PATH` matches, the playbook reports the failure and the incident
escalates (desktop notification + `omarchy-healer escalations`) instead of silently
quarantining a server you configured. Use `quarantine_mcp_server` (`--max-tier 2`) if you
do want that entry moved to `_disabled_mcpServers` after review.

In `watch` mode each desktop notification fires once per fingerprint, not once per
interval, so an unfixable problem cannot spam you while it stays unresolved.

## Interactive AI Agent Notifications & Self-Improvement

Just like Omarchy's crash diagnostic triggers an AI agent when clicking a crash notification, Omarchy-Healer sends interactive desktop notifications whenever it needs assistance or new capabilities:

- **Notification trigger**: Fires via `omarchy-notification-send` with an `--exec` click handler pointing to `bin/on-improve`.
- **One-click agent launch**: Clicking the notification launches your default AI system agent (`omarchy-agent` — e.g. Gemini, Claude, Codex, OpenCode) with the working directory set to `omarchy-healer`.
- **Rich diagnostic prompt**: The prompt provides the exact anomaly context, failure detail, proposed playbook ID, code feature specification, and instructions on modifying `playbooks.py`/`probes.py` and verifying with `uv run pytest`.
- **Manual launch**: You can also launch or inspect the prompt anytime via `bin/on-improve` or `uv run omarchy-healer improve [incident_id]`.


## Run it unattended

The healer only helps if it is actually running. It ships as a systemd **user** service
bound to `graphical-session.target` (it needs the Hyprland session environment to probe
compositor state):

```bash
systemctl --user enable --now omarchy-healer.service
systemctl --user status  omarchy-healer.service
journalctl --user -u omarchy-healer.service -f
```

It runs `watch --interval 30 --auto-heal --max-tier 1`; stop it with
`systemctl --user disable --now omarchy-healer.service`. The 30s interval also matters
for the stuck-OSD probe, which needs a single long-lived process to observe how long an
overlay has been open.
