"""Configuration loader with Laya -> Gemini -> NVIDIA NIM fallback resolution."""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import dotenv_values

log = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"

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


def _as_float(env: dict[str, str], key: str, default: float) -> float:
    """Parse a float env var, falling back to `default` on any malformed value.

    A typo in one threshold must not take down the whole CLI at startup.
    """
    raw = _first(env, key, default=str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        log.warning("Ignoring malformed %s=%r; falling back to %s.", key, raw, default)
        return default
    return value


def _as_int(
    env: dict[str, str],
    key: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Parse a clamped int env var, falling back to `default` on any malformed value."""
    raw = _first(env, key, default=str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        log.warning("Ignoring malformed %s=%r; falling back to %s.", key, raw, default)
        return default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


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
    gemini_model: str = DEFAULT_GEMINI_MODEL

    # 3. Fallback 2: NVIDIA NIM (OpenAI-compatible)
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = DEFAULT_NVIDIA_MODEL
    # 4. Autonomy & Thresholds
    max_auto_tier: int = 1
    ram_warn_percent: float = 80.0
    ram_crit_percent: float = 90.0
    disk_warn_percent: float = 85.0
    disk_crit_percent: float = 95.0
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

        max_auto_tier = _as_int(env, "HEALER_MAX_AUTO_TIER", 1, minimum=0, maximum=2)

        ram_warn = _as_float(env, "HEALER_RAM_WARN_PERCENT", 80.0)
        ram_crit = _as_float(env, "HEALER_RAM_CRIT_PERCENT", 90.0)
        disk_warn = _as_float(env, "HEALER_DISK_WARN_PERCENT", 85.0)
        disk_crit = _as_float(env, "HEALER_DISK_CRIT_PERCENT", 95.0)
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
                "JEV_ENDPOINT_URL",
                "SYSTEMONE_ENDPOINT_URL",
            ),
            laya_api_key=_first(
                env,
                "LAYA_API_KEY",
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
            gemini_model=_first(env, "HEALER_GEMINI_MODEL", default=DEFAULT_GEMINI_MODEL),
            nvidia_api_key=_first(env, "NVIDIA_NIM_API_KEY", "NVIDIA_API_KEY"),
            nvidia_base_url=_first(
                env, "NVIDIA_BASE_URL", default="https://integrate.api.nvidia.com/v1"
            ),
            nvidia_model=_first(env, "HEALER_NVIDIA_MODEL", default=DEFAULT_NVIDIA_MODEL),
            max_auto_tier=max_auto_tier,
            ram_warn_percent=ram_warn,
            ram_crit_percent=ram_crit,
            disk_warn_percent=disk_warn,
            disk_crit_percent=disk_crit,
            db_path=db_path,
            mcp_config_path=mcp_path,
        )

    def has_laya(self) -> bool:
        """True only when Laya is actually reachable: in-process router or an HTTP endpoint.

        An API key on its own cannot be called, so it must not advertise the
        primary provider as configured.
        """
        return bool(self.laya_local or self.laya_endpoint_url)

    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    def has_nvidia_nim(self) -> bool:
        return bool(self.nvidia_api_key)

    def provider_chain_summary(self) -> list[dict[str, str]]:
        return [
            {
                "role": "Primary Triage & Agent (NVIDIA NIM)",
                "provider": "nvidia_nim",
                "configured": "yes" if self.has_nvidia_nim() else "no",
                "detail": f"{self.nvidia_model} ({self.nvidia_base_url})",
            },
            {
                "role": "Fallback 1 (Google Gemini)",
                "provider": "gemini",
                "configured": "yes" if self.has_gemini() else "no",
                "detail": self.gemini_model,
            },
            {
                "role": "Fallback 2 (Offline Deterministic)",
                "provider": "heuristic",
                "configured": "yes",
                "detail": "Built-in calibrated rule & playbook matcher",
            },
        ]
