"""Triage & Safety Judge with NVIDIA NIM (Primary), Gemini (Fallback), and Heuristic."""
from __future__ import annotations

import json
import logging
from typing import Any

from .config import HealerConfig
from .llm_router import LLMFallbackRouter
from .models import DomainType, SymptomEvent, TriageDecision
from .playbooks import PLAYBOOK_CATALOG

log = logging.getLogger(__name__)


class TriageJudge:
    """Evaluates SymptomEvents via NVIDIA NIM (primary) -> Gemini (fallback) -> Heuristic."""

    def __init__(
        self,
        config: HealerConfig,
        *,
        llm_router: LLMFallbackRouter | None = None,
        custom_evaluator: Any = None,
    ) -> None:
        self.config = config
        self.llm_router = llm_router or LLMFallbackRouter(config)
        self._custom_evaluator = custom_evaluator

    def evaluate(self, symptom: SymptomEvent) -> TriageDecision:
        tried: list[str] = []

        # 1. Custom / test evaluator
        if self._custom_evaluator is not None:
            tried.append("custom")
            try:
                state_str = json.dumps({"symptom": symptom.model_dump()}, ensure_ascii=False)
                res = self._custom_evaluator(state_str, {})
                if isinstance(res, TriageDecision):
                    return res
                if isinstance(res, dict):
                    pb = str(res.get("selected_playbook") or symptom.suggested_playbook or "inspect_only")
                    pb = self._resolve_playbook(symptom, pb)
                    tier = int(res.get("safety_tier", symptom.default_tier))
                    return TriageDecision(
                        provider=res.get("provider", "heuristic"),
                        domain=symptom.domain,
                        severity_score=float(res.get("severity_score", 0.5)),
                        is_actionable=bool(res.get("is_actionable", pb != "inspect_only")),
                        actionable_probability=float(res.get("actionable_probability", 0.85)),
                        safety_tier=tier,
                        selected_playbook=pb,
                        confidence=float(res.get("confidence", 0.85)),
                        rationale=str(res.get("rationale", "Custom evaluation")),
                        fallback_chain_tried=list(tried),
                    )
            except Exception as exc:
                log.warning("Custom evaluator failed (%s); falling back.", exc)

        # 2. Primary (NVIDIA NIM) & Fallback 1 (Gemini) via LangChain Router
        if self.config.has_nvidia_nim() or self.config.has_gemini():
            try:
                decision = self._eval_with_llm_fallback(symptom, tried)
                if decision is not None:
                    return decision
            except Exception as exc:
                log.warning("LLM triage failed (%s); using deterministic heuristic.", exc)

        # 3. Deterministic Heuristic Fallback (Always succeeds offline)
        tried.append("heuristic")
        return self._heuristic_decision(symptom, tried)

    @staticmethod
    def _resolve_playbook(symptom: SymptomEvent, candidate: str) -> str:
        """Ensure the selected playbook exists and matches the symptom's domain and type."""
        suggested = symptom.suggested_playbook if symptom.suggested_playbook in PLAYBOOK_CATALOG else ""
        if candidate not in PLAYBOOK_CATALOG:
            return suggested or "inspect_only"
        if suggested and suggested != "inspect_only":
            if candidate == "inspect_only":
                return suggested
            cand_domain = PLAYBOOK_CATALOG[candidate].get("domain")
            if cand_domain != symptom.domain:
                return suggested
            if symptom.domain == "memory_pressure":
                zombie_pbs = {"reap_zombie_parents"}
                cache_pbs = {"reclaim_memory_caches", "clean_user_memory_caches"}
                if (suggested in zombie_pbs and candidate not in zombie_pbs) or (
                    suggested in cache_pbs and candidate not in cache_pbs
                ):
                    return suggested
        return candidate

    def _eval_with_llm_fallback(
        self, symptom: SymptomEvent, tried: list[str]
    ) -> TriageDecision | None:
        catalog_desc = "; ".join(
            f"{pid} (domain={meta['domain']}, tier={meta['tier']}): {meta['description']}"
            for pid, meta in PLAYBOOK_CATALOG.items()
        )
        system_prompt = (
            "You are the Triage & Safety Judge for Omarchy-Healer on Linux. "
            "Available playbooks: " + catalog_desc + ". "
            "When `suggested_playbook` in the symptom JSON is a valid remediation playbook, "
            "prefer it along with its catalog `safety_tier` and set `is_actionable` to true. "
            "Given a system anomaly JSON, respond ONLY with a JSON object containing keys: "
            "\"is_actionable\" (bool), \"actionable_probability\" (float 0..1), "
            "\"severity_score\" (float 0..1), \"selected_playbook\" (one of: "
            + ", ".join(PLAYBOOK_CATALOG.keys())
            + "), \"safety_tier\" (int 0, 1, or 2), and \"rationale\" (short string)."
        )
        user_prompt = json.dumps(symptom.model_dump(), ensure_ascii=False)
        parsed, provider_used, llm_tried = self.llm_router.invoke_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        tried.extend(llm_tried)
        if not parsed or provider_used is None:
            return None

        raw_pb = str(parsed.get("selected_playbook") or symptom.suggested_playbook or "inspect_only")
        selected_pb = self._resolve_playbook(symptom, raw_pb)
        pb_meta = PLAYBOOK_CATALOG.get(selected_pb, PLAYBOOK_CATALOG["inspect_only"])
        catalog_tier = int(pb_meta.get("tier", symptom.default_tier))
        tier = int(parsed.get("safety_tier", catalog_tier))
        tier = max(0, min(2, tier))
        act_p = max(0.0, min(1.0, float(parsed.get("actionable_probability", 0.85))))
        sev = max(0.0, min(1.0, float(parsed.get("severity_score", 0.6))))
        has_remediation = selected_pb != "inspect_only"
        is_act = (bool(parsed.get("is_actionable", True)) or bool(symptom.suggested_playbook)) and has_remediation
        if is_act and act_p < 0.5:
            act_p = 0.85

        return TriageDecision(
            provider=provider_used,
            domain=symptom.domain,
            severity_score=round(sev, 3),
            is_actionable=is_act,
            actionable_probability=round(act_p, 3),
            safety_tier=tier,
            selected_playbook=selected_pb,
            confidence=round(act_p, 3),
            rationale=str(parsed.get("rationale") or f"Triaged by {provider_used} fallback"),
            fallback_chain_tried=list(tried),
        )

    @staticmethod
    def _heuristic_decision(symptom: SymptomEvent, tried: list[str]) -> TriageDecision:
        pb = symptom.suggested_playbook if symptom.suggested_playbook in PLAYBOOK_CATALOG else "inspect_only"
        meta = PLAYBOOK_CATALOG.get(pb, PLAYBOOK_CATALOG["inspect_only"])
        tier = int(meta.get("tier", symptom.default_tier))
        domain_sev: dict[DomainType, float] = {
            "ui_shell": 0.45,
            "mcp_config": 0.65,
            "systemd_service": 0.75,
            "memory_pressure": 0.70,
            "compositor": 0.80,
            "disk": 0.85,
            "network": 0.70,
            "audio": 0.60,
            "other": 0.30,
        }
        sev = domain_sev.get(symptom.domain, 0.5)
        if symptom.metrics.get("critical"):
            sev = 0.92
        return TriageDecision(
            provider="heuristic",
            domain=symptom.domain,
            severity_score=sev,
            is_actionable=(pb != "inspect_only"),
            actionable_probability=0.9 if pb != "inspect_only" else 0.2,
            safety_tier=tier,
            selected_playbook=pb,
            confidence=0.9,
            rationale=f"Deterministic rule matched {symptom.fingerprint} -> {pb} (tier {tier}).",
            fallback_chain_tried=list(tried),
        )


# Backward-compatibility alias
LayaTriageJudge = TriageJudge
