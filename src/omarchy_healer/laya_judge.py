"""TypeSafe / Laya System-1 Triage & Safety Judge with Gemini and NVIDIA NIM fallbacks."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .config import HealerConfig
from .llm_router import LLMFallbackRouter
from .models import DomainType, SymptomEvent, TriageDecision
from .playbooks import PLAYBOOK_CATALOG

log = logging.getLogger(__name__)

_LOCAL_LAYA_ROUTER: Any = None


def build_laya_questions() -> dict[str, dict[str, Any]]:
    """Construct parallel TypeSafe System-1 questions (Noul + Choice + Score)."""
    playbook_criteria = {
        pid: meta["description"] for pid, meta in PLAYBOOK_CATALOG.items()
    }
    return {
        "actionable": {
            "type": "noul",
            "instructions": (
                "Does this Linux desktop or service anomaly require automated remediation now, "
                "rather than being harmless normal operation?"
            ),
        },
        "playbook": {
            "type": "choice",
            "instructions": (
                "Select the single most appropriate bounded remediation playbook for the "
                "reported `symptom.domain`, `symptom.title`, and `symptom.summary`."
            ),
            "criteria": playbook_criteria,
        },
        "severity": {
            "type": "score",
            "instructions": "Rate the operational severity and urgency of this system issue.",
            "criteria": [
                "Benign or transient state requiring no action.",
                "Minor UI overlay or configuration drift safe for immediate non-disruptive cleanup.",
                "Degraded user service, broken MCP binary, or elevated memory pressure needing active remediation.",
                "Critical resource exhaustion or compositor failure risking session instability.",
            ],
        },
    }


class LayaTriageJudge:
    """Evaluates SymptomEvents via Laya (primary) -> Gemini -> NVIDIA NIM -> Heuristic."""

    def __init__(
        self,
        config: HealerConfig,
        *,
        llm_router: LLMFallbackRouter | None = None,
        http_evaluator: Any = None,
    ) -> None:
        self.config = config
        self.llm_router = llm_router or LLMFallbackRouter(config)
        self._http_evaluator = http_evaluator

    def evaluate(self, symptom: SymptomEvent) -> TriageDecision:
        tried: list[str] = []
        state_payload = {
            "symptom": {
                "domain": symptom.domain,
                "title": symptom.title,
                "summary": symptom.summary,
                "metrics": symptom.metrics,
                "suggested_playbook": symptom.suggested_playbook,
                "default_tier": symptom.default_tier,
            }
        }
        state_str = json.dumps(state_payload, ensure_ascii=False)
        questions = build_laya_questions()

        # 1. Try Primary: Laya (Local Router or HTTP Endpoint)
        if self.config.has_laya() or self._http_evaluator is not None:
            tried.append("laya")
            try:
                raw_answers = self._eval_with_laya(state_str, questions)
                return self._parse_laya_answers(symptom, raw_answers, tried)
            except Exception as exc:
                log.warning("Laya primary triage unavailable (%s); falling back to LLM chain.", exc)

        # 2. Try Fallback 1 (Gemini) & Fallback 2 (NVIDIA NIM) via LangChain Router
        if self.config.has_gemini() or self.config.has_nvidia_nim():
            try:
                decision = self._eval_with_llm_fallback(symptom, tried)
                if decision is not None:
                    return decision
            except Exception as exc:
                log.warning("LLM triage fallback failed (%s); using deterministic heuristic.", exc)

        # 3. Deterministic Heuristic Fallback (Always succeeds offline)
        tried.append("heuristic")
        return self._heuristic_decision(symptom, tried)

    def _eval_with_laya(
        self, state_str: str, questions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        if self._http_evaluator is not None:
            resp = self._http_evaluator(state_str, questions, timeout=self.config.laya_timeout)
            return resp.get("answers", resp) if isinstance(resp, dict) else {}

        if self.config.laya_local:
            global _LOCAL_LAYA_ROUTER
            if _LOCAL_LAYA_ROUTER is None:
                import laya

                _LOCAL_LAYA_ROUTER = laya.Router(device="cpu", preload=False)
            raw = _LOCAL_LAYA_ROUTER.predict(state_str, questions)
            return raw.get("answers", raw) if isinstance(raw, dict) else {}

        if not self.config.laya_endpoint_url:
            raise RuntimeError("LAYA_ENDPOINT_URL is not configured")

        payload = json.dumps(
            {
                "model": self.config.laya_model,
                "state": state_str,
                "questions": questions,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.laya_api_key:
            headers["Authorization"] = f"Bearer {self.config.laya_api_key}"
        if "ngrok" in self.config.laya_endpoint_url:
            headers["ngrok-skip-browser-warning"] = "1"

        req = urllib.request.Request(
            self.config.laya_endpoint_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.laya_timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("answers", data) if isinstance(data, dict) else {}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Laya HTTP request failed: {exc}") from exc

    def _parse_laya_answers(
        self, symptom: SymptomEvent, answers: dict[str, Any], tried: list[str]
    ) -> TriageDecision:
        if not isinstance(answers, dict) or not answers:
            raise ValueError("Empty answers from Laya")

        act_ans = answers.get("actionable") or {}
        act_p = float(
            act_ans.get("noul")
            if "noul" in act_ans
            else act_ans.get("probability", act_ans.get("p", 0.85))
        )

        pb_ans = answers.get("playbook") or {}
        selected_pb = str(pb_ans.get("choice") or symptom.suggested_playbook or "inspect_only")
        if selected_pb not in PLAYBOOK_CATALOG:
            selected_pb = symptom.suggested_playbook or "inspect_only"
        pb_probs = pb_ans.get("probabilities") or {}
        confidence = float(pb_probs.get(selected_pb, pb_ans.get("confidence", 0.85)))

        sev_ans = answers.get("severity") or {}
        raw_score = float(sev_ans.get("score", 1.5))
        severity_norm = round(max(0.0, min(1.0, raw_score / 3.0)), 3)

        pb_meta = PLAYBOOK_CATALOG.get(selected_pb, PLAYBOOK_CATALOG["inspect_only"])
        tier = int(pb_meta.get("tier", symptom.default_tier))

        return TriageDecision(
            provider="laya",
            domain=symptom.domain,
            severity_score=severity_norm,
            is_actionable=(act_p >= 0.5 and selected_pb != "inspect_only"),
            actionable_probability=round(act_p, 4),
            safety_tier=tier,
            selected_playbook=selected_pb,
            confidence=round(confidence, 4),
            rationale=(
                f"Laya System-1 judgment: actionable_p={act_p:.2f}, "
                f"playbook={selected_pb} (p={confidence:.2f}), severity={severity_norm:.2f}"
            ),
            fallback_chain_tried=list(tried),
        )

    def _eval_with_llm_fallback(
        self, symptom: SymptomEvent, tried: list[str]
    ) -> TriageDecision | None:
        system_prompt = (
            "You are the Triage & Safety Judge for Omarchy-Healer on Linux. "
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

        selected_pb = str(parsed.get("selected_playbook") or symptom.suggested_playbook or "inspect_only")
        if selected_pb not in PLAYBOOK_CATALOG:
            selected_pb = symptom.suggested_playbook or "inspect_only"
        pb_meta = PLAYBOOK_CATALOG.get(selected_pb, PLAYBOOK_CATALOG["inspect_only"])
        tier = int(parsed.get("safety_tier", pb_meta.get("tier", symptom.default_tier)))
        tier = max(0, min(2, tier))
        act_p = max(0.0, min(1.0, float(parsed.get("actionable_probability", 0.85))))
        sev = max(0.0, min(1.0, float(parsed.get("severity_score", 0.6))))
        is_act = bool(parsed.get("is_actionable", True)) and selected_pb != "inspect_only"

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
