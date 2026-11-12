"""Tests for config hardening: safe numeric parsing and provider usability."""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from omarchy_healer import config as config_module
from omarchy_healer.config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NVIDIA_MODEL,
    HealerConfig,
    _is_placeholder,
    load_merged_env,
)


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Detach from the developer's real .env files and shell environment."""
    monkeypatch.setattr(config_module, "DEFAULT_SEARCH_ENV_PATHS", (tmp_path / "absent.env",))
    for key in list(os.environ):
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.mark.parametrize("bad_value", ["eighty", "", "   ", "80,0", "nan", "inf", "80%"])
def test_malformed_float_env_var_falls_back_to_default(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    monkeypatch.setenv("HEALER_RAM_WARN_PERCENT", bad_value)
    monkeypatch.setenv("HEALER_RAM_CRIT_PERCENT", bad_value)

    cfg = HealerConfig.from_env()

    assert cfg.ram_warn_percent == 80.0
    assert cfg.ram_crit_percent == 90.0


def test_malformed_float_env_var_does_not_raise(isolated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a single malformed threshold used to raise ValueError at startup."""
    monkeypatch.setenv("HEALER_RAM_WARN_PERCENT", "eighty")

    cfg = HealerConfig.from_env()

    assert isinstance(cfg.ram_warn_percent, float)
    assert cfg.ram_warn_percent == 80.0


def test_valid_numeric_env_vars_still_parse(isolated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALER_RAM_WARN_PERCENT", "65.5")
    monkeypatch.setenv("HEALER_RAM_CRIT_PERCENT", "97")
    monkeypatch.setenv("HEALER_MAX_AUTO_TIER", "2")

    cfg = HealerConfig.from_env()

    assert cfg.ram_warn_percent == 65.5
    assert cfg.ram_crit_percent == 97.0
    assert cfg.max_auto_tier == 2


def test_max_auto_tier_stays_clamped_and_guards_bad_ints(
    isolated_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HEALER_MAX_AUTO_TIER", "9")
    assert HealerConfig.from_env().max_auto_tier == 2

    monkeypatch.setenv("HEALER_MAX_AUTO_TIER", "-4")
    assert HealerConfig.from_env().max_auto_tier == 0

    monkeypatch.setenv("HEALER_MAX_AUTO_TIER", "two")
    assert HealerConfig.from_env().max_auto_tier == 1


def test_has_nvidia_nim_and_gemini_flags() -> None:
    cfg_empty = HealerConfig()
    assert cfg_empty.has_nvidia_nim() is False
    assert cfg_empty.has_gemini() is False

    cfg_nvidia = HealerConfig(nvidia_api_key="nv-key")
    assert cfg_nvidia.has_nvidia_nim() is True
    assert cfg_nvidia.has_gemini() is False

    cfg_gemini = HealerConfig(gemini_api_key="gm-key")
    assert cfg_gemini.has_nvidia_nim() is False
    assert cfg_gemini.has_gemini() is True


def test_placeholder_keys_are_filtered_out(isolated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "your_nvidia_nim_api_key_here")
    monkeypatch.setenv("GEMINI_API_KEY", "your_gemini_api_key_here")

    cfg = HealerConfig.from_env()
    assert cfg.nvidia_api_key == ""
    assert cfg.gemini_api_key == ""
    assert cfg.has_nvidia_nim() is False
    assert cfg.has_gemini() is False


def test_provider_chain_summary_reports_nvidia_nim_configured() -> None:
    cfg = HealerConfig(nvidia_api_key="nv-key")
    nv_row = cfg.provider_chain_summary()[0]
    assert nv_row["provider"] == "nvidia_nim"
    assert nv_row["configured"] == "yes"


def test_provider_chain_summary_order_is_nvidia_then_gemini(isolated_env: None) -> None:
    summary = HealerConfig.from_env().provider_chain_summary()
    assert [row["provider"] for row in summary] == ["nvidia_nim", "gemini", "heuristic"]
    assert summary[-1]["configured"] == "yes"


def test_model_defaults_are_non_empty_strings(isolated_env: None) -> None:
    for cfg in (HealerConfig(), HealerConfig.from_env()):
        assert isinstance(cfg.gemini_model, str) and cfg.gemini_model.strip()
        assert isinstance(cfg.nvidia_model, str) and cfg.nvidia_model.strip()


def test_model_defaults_are_text_models() -> None:
    assert DEFAULT_GEMINI_MODEL == "gemini-3.5-flash"
    assert DEFAULT_NVIDIA_MODEL == "nvidia/nemotron-3-super-120b-a12b"
    assert "vision" not in DEFAULT_NVIDIA_MODEL
    assert "8-flash" not in DEFAULT_GEMINI_MODEL


def test_model_env_overrides_still_win(isolated_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEALER_GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("HEALER_NVIDIA_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")

    cfg = HealerConfig.from_env()

    assert cfg.gemini_model == "gemini-2.5-flash"
    assert cfg.nvidia_model == "nvidia/llama-3.1-nemotron-70b-instruct"


def test_placeholder_and_merged_env_public_api_intact() -> None:
    assert _is_placeholder(None) is True
    assert _is_placeholder("  ") is True
    assert _is_placeholder("your_key_here") is True
    assert _is_placeholder("https://your-api-endpoint.example/v1") is True
    assert _is_placeholder("https://integrate.api.nvidia.com/v1") is False
    assert isinstance(load_merged_env(), dict)