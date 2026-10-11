"""Configuration loader with Laya -> Gemini -> NVIDIA NIM fallback resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import dotenv_values

DEFAULT_SEARCH_ENV_PATHS: tuple[Path, ...] = (
    Path.cwd() / ".env",
    Path(__file__).resolve().parents[2] / ".env",
    Path.home() / ".config" / "omarchy-healer" / ".env",
    Path.home() / "Projects" / "Omarchy-AI" / ".env",
    Path.home() / ".env",
)


def _is_placeholder(val: str | None) -> bool:
    if not val:
        return True
    cleaned = val.strip().strip('"').strip("'")
    if not cleaned:
        return True
    lower = cleaned.lower()
    return (
        lower.startswith("your_")
        or lower.endswith("_here")
        or "your-api-endpoint.example" in lower
        or "your-laya-endpoint.example" in lower
    )


def load_merged_env(extra_paths: list[Path] | None = None) -> dict[str, str]:
    """Merge .env files (lowest to highest priority) with os.environ taking top precedence."""
    merged: dict[str, str] = {}
    search_paths = list(DEFAULT_SEARCH_ENV_PATHS)
    if extra_paths:
        search_paths = list(extra_paths) + search_paths

    for path in reversed(search_paths):
        try:
            if path.is_file():
                for k, v in dotenv_values(path).items():
                    if v is not None and not _is_placeholder(v):
                        merged[k] = v.strip()
        except OSError:
            continue

    for k, v in os.environ.items():
        if not _is_placeholder(v):
            merged[k] = v.strip()

    return merged


def _first(env: dict[str, str], *keys: str, default: str = "") -> str:
    for key in keys:
        val = env.get(key, "").strip()
        if val and not _is_placeholder(val):
            return val
    return default


@dataclass
class HealerConfig:
    # 1. Primary: Laya / SystemOne
    laya_endpoint_url: str = ""
    laya_api_key: str = ""
    laya_model: str = "laya-rl-agent"
    laya_local: bool = False
    laya_timeout: float = 5.0

    # 2. Fallback 1: Google Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"

    # 3. Fallback 2: NVIDIA NIM (OpenAI-compatible)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.2-11b-vision-instruct"
    # 4. Autonomy & Thresholds
    max_auto_tier: int = 1
    ram_warn_percent: float = 80.0
    ram_crit_percent: float = 90.0
    psi_some_avg10_warn: float = 10.0
    db_path: Path = field(
        default_factory=lambda: Path.home() / ".local" / "share" / "omarchy-healer" / "incidents.db"
    )
    mcp_config_path: Path = field(
        default_factory=lambda: Path.home() / ".omp" / "agent" / "mcp.json"
    )

    @classmethod
    def from_env(cls, extra_env_paths: list[Path] | None = None) -> HealerConfig:
        env = load_merged_env(extra_env_paths)
        laya_local_raw = _first(env, "LAYA_LOCAL", default="0").lower()
        laya_local = laya_local_raw in ("1", "true", "yes", "on")

        max_tier_raw = _first(env, "HEALER_MAX_AUTO_TIER", default="1")
        try:
            max_auto_tier = max(0, min(2, int(max_tier_raw)))
        except ValueError:
            max_auto_tier = 1

        ram_warn = float(_first(env, "HEALER_RAM_WARN_PERCENT", default="80.0"))
        ram_crit = float(_first(env, "HEALER_RAM_CRIT_PERCENT", default="90.0"))
        db_raw = _first(env, "HEALER_DB_PATH")
        db_path = Path(db_raw).expanduser() if db_raw else (
            Path.home() / ".local" / "share" / "omarchy-healer" / "incidents.db"
        )
        mcp_raw = _first(env, "HEALER_MCP_CONFIG_PATH")
        mcp_path = Path(mcp_raw).expanduser() if mcp_raw else (
            Path.home() / ".omp" / "agent" / "mcp.json"
        )

        return cls(
            laya_endpoint_url=_first(
                env,
                "LAYA_ENDPOINT_URL",
                "ENDPOINT_URL",
                "JEV_ENDPOINT_URL",
                "SYSTEMONE_ENDPOINT_URL",
            ),
            laya_api_key=_first(
                env,
                "LAYA_API_KEY",
                "API_KEY",
                "JEV_API_KEY",
                "SYSTEMONE_API_KEY",
            ),
            laya_model=_first(
                env,
                "OMARCHY_LAYA_MODEL",
                "OMARCHY_JEV_MODEL",
                "TYPESAFE_MODEL",
                default="laya-rl-agent",
            ),
            laya_local=laya_local,
            gemini_api_key=_first(env, "GEMINI_API_KEY", "GOOGLE_API_KEY"),
            gemini_model=_first(env, "HEALER_GEMINI_MODEL", default="gemini-3.8-flash"),
            nvidia_api_key=_first(env, "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"),
            nvidia_base_url=_first(
                env, "NVIDIA_BASE_URL", default="https://integrate.api.nvidia.com/v1"
            ),
            nvidia_model=_first(
                env, "HEALER_NVIDIA_MODEL", default="meta/llama-3.2-11b-vision-instruct"
            ),
            max_auto_tier=max_auto_tier,
            ram_warn_percent=ram_warn,
            ram_crit_percent=ram_crit,
            db_path=db_path,
            mcp_config_path=mcp_path,
        )

    def has_laya(self) -> bool:
        return bool(self.laya_local or self.laya_endpoint_url or self.laya_api_key)

    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    def has_nvidia_nim(self) -> bool:
        return bool(self.nvidia_api_key)

    def provider_chain_summary(self) -> list[dict[str, str]]:
        return [
            {
                "role": "Primary Triage & Safety (System-1)",
                "provider": "laya",
                "configured": "yes" if self.has_laya() else "no",
                "detail": (
                    "local in-process laya.Router"
                    if self.laya_local
                    else (self.laya_endpoint_url or "no endpoint configured")
                ),
            },
            {
                "role": "Fallback 1 (Triage + LangGraph Agent)",
                "provider": "gemini",
                "configured": "yes" if self.has_gemini() else "no",
                "detail": self.gemini_model,
            },
            {
                "role": "Fallback 2 (Triage + LangGraph Agent)",
                "provider": "nvidia_nim",
                "configured": "yes" if self.has_nvidia_nim() else "no",
                "detail": f"{self.nvidia_model} ({self.nvidia_base_url})",
            },
            {
                "role": "Fallback 3 (Offline Deterministic)",
                "provider": "heuristic",
                "configured": "yes",
                "detail": "Built-in calibrated rule & playbook matcher",
            },
        ]
