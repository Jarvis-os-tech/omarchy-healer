# Omarchy-Healer

Multi-agent system monitor and self-healing orchestrator for Omarchy / Hyprland Linux built on **LangGraph**, **LangChain**, and **TypeSafe Laya** with automatic failover to **Google Gemini** and **NVIDIA NIM**.

## Architecture

1. **Deterministic Probes (`probes.py`)**
   - `OmarchyUiProbe`: Detects stuck `omarchy-osd` layer overlays (e.g., `duration: 0` launch popups), missing `omarchy-bar` Quickshell surfaces, and `hyprctl configerrors`.
   - `McpAndServiceProbe`: Validates `~/.omp/agent/mcp.json` server binaries (`ENOENT` detection) and `systemctl --user --failed` units.
   - `MemoryProcessProbe`: Monitors `/proc/meminfo`, `/proc/pressure/memory` (PSI), top RSS consumers, and defunct zombie processes.

2. **Triage & Safety Judge (`laya_judge.py`)**
   - **Primary**: Evaluates `Noul` (`actionable`), `Choice` (`playbook`), and `Score` (`severity`) in parallel via **Laya** (`LAYA_ENDPOINT_URL` / `LAYA_API_KEY` or in-process `laya.Router` when `LAYA_LOCAL=1`).
   - **Fallback 1**: Google Gemini (`GEMINI_API_KEY` via `langchain-google-genai`).
   - **Fallback 2**: NVIDIA NIM (`NVIDIA_NIM_API_KEY` / `NVIDIA_API_KEY` via `langchain-openai` against `https://integrate.api.nvidia.com/v1`).
   - **Fallback 3**: Calibrated deterministic rule matcher when offline.

3. **LangGraph Multi-Agent Workflow (`graph.py`)**
   - `triage` -> `diagnostician` -> `planner` -> `safety_gate` -> `executor` -> `verifier` -> `record_memory` (SQLite `~/.local/share/omarchy-healer/incidents.db`).

## Quick Start

```bash
cd ~/Projects/omarchy-healer
cp .env.example .env   # Or rely on automatic fallback to ~/Projects/Omarchy-AI/.env

# Check configured provider chain (Laya -> Gemini -> NVIDIA NIM -> Heuristic)
uv run omarchy-healer providers

# Dry-run scan of live system anomalies
uv run omarchy-healer scan

# Auto-heal Tier 0 and Tier 1 issues and verify resolution
uv run omarchy-healer heal --max-tier 1

# Run continuous background watchdog
uv run omarchy-healer watch --interval 15 --auto-heal --max-tier 1

# View SQLite incident history
uv run omarchy-healer history
```
