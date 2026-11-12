"""LangGraph multi-agent workflow: Triage -> Diagnose -> Plan -> SafetyGate -> Execute -> Verify -> Memory."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, cast

from langgraph.graph import END, START, StateGraph

from .config import HealerConfig
from .laya_judge import LayaTriageJudge
from .llm_router import LLMFallbackRouter
from .memory import IncidentMemoryStore
from .formatting import (
    clean_english_text,
    format_completion_notification,
    format_escalation_notification,
    format_metrics_english,
    format_tool_evidence_english,
)
from .models import (
    MAX_SAFETY_TIER,
    MIN_SAFETY_TIER,
    VERIFICATION_SWEEP_KEY,
    CapabilityEscalation,
    DiagnosisReport,
    EscalationReason,
    ExecutionResult,
    HealerGraphState,
    ProviderType,
    RemediationPlan,
    SymptomEvent,
    VerificationReport,
)
from .playbooks import (
    DIAGNOSTIC_TOOLS,
    PLAYBOOK_CATALOG,
    execute_playbook,
    send_completion_notification,
    send_escalation_notification,
)
from .probes import DEFAULT_OSD_STUCK_SECONDS, collect_all_symptoms

MAX_UNTRUSTED_TEXT_CHARS = 4000
MAX_DIAGNOSTIC_TOOL_CHARS = 800

# The safety gate is the single enforcement point for playbook/domain pairing, so the
# allowlists below are deliberately explicit rather than inferred from probe metadata.
CROSS_DOMAIN_PLAYBOOKS: frozenset[str] = frozenset({"inspect_only"})
CROSS_DOMAIN_ALLOWED: dict[str, frozenset[str]] = {
    "compositor": frozenset({"ui_shell"}),
    "systemd_service": frozenset({"other"}),
}

_DIAGNOSTIC_TOOLS_BY_NAME: dict[str, Any] = {tool.name: tool for tool in DIAGNOSTIC_TOOLS}

# Read-only tools are only worth their subprocess cost when an LLM will actually read
# their output, so each domain selects the smallest set that explains its symptoms.
DOMAIN_DIAGNOSTIC_TOOLS: dict[str, tuple[str, ...]] = {
    "ui_shell": ("inspect_omarchy_osd_and_layers",),
    "compositor": ("inspect_omarchy_osd_and_layers",),
    "mcp_config": ("inspect_mcp_config",),
    "memory_pressure": ("inspect_memory_and_top_processes",),
}


def _coerce_int(value: Any, default: int) -> int:
    """Coerce an untrusted LLM value to an ``int``, returning *default* when it is not one.

    ``bool`` is excluded on purpose: it is an ``int`` subclass in Python but a model
    answering ``true`` for a safety tier is malformed input, not tier 1.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else default
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except (TypeError, ValueError):
            return default
    return default


def _clamp_int(
    value: Any, default: int, low: int = MIN_SAFETY_TIER, high: int = MAX_SAFETY_TIER
) -> int:
    """Coerce *value* to an int and clamp it into ``[low, high]`` so models can never
    produce a value the pydantic bounds would reject."""
    return max(low, min(high, _coerce_int(value, default)))


def _coerce_float(value: Any, default: float) -> float:
    """Coerce an untrusted LLM value to a finite ``float``, returning *default* otherwise."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else default
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default
    return default


def _clamp_float(value: Any, default: float, low: float = 0.0, high: float = 1.0) -> float:
    """Coerce *value* to a float and clamp it into ``[low, high]``."""
    return max(low, min(high, _coerce_float(value, default)))


def _coerce_str(value: Any, default: str = "", *, max_chars: int = MAX_UNTRUSTED_TEXT_CHARS) -> str:
    """Return a bounded, stripped string for *value*, or *default* when it is not textual."""
    if isinstance(value, bool):
        return default
    if isinstance(value, str):
        text = value.strip()
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        text = str(value)
    else:
        return default
    return text[:max_chars] if text else default


def playbook_domain(playbook_id: str) -> str:
    """Return the domain a catalog playbook belongs to, defaulting to ``other``."""
    meta = PLAYBOOK_CATALOG.get(playbook_id)
    return str(meta.get("domain", "other")) if meta else "other"


def domain_compatible(symptom_domain: str, playbook_id: str) -> tuple[bool, str]:
    """Return ``(compatible, reason)`` for pairing a symptom domain with a playbook.

    A playbook that mutates a subsystem the symptom never touched is a category error
    (memory pressure must never restart a systemd unit), so mismatches are rejected
    unless they are explicitly allowlisted.
    """
    if playbook_id not in PLAYBOOK_CATALOG:
        return False, f"playbook {playbook_id!r} is not in the catalog"
    pb_domain = playbook_domain(playbook_id)
    if pb_domain == symptom_domain:
        return True, f"playbook domain {pb_domain!r} matches symptom domain {symptom_domain!r}"
    if playbook_id in CROSS_DOMAIN_PLAYBOOKS:
        return True, f"{playbook_id!r} is a domain-agnostic no-op"
    if pb_domain in CROSS_DOMAIN_ALLOWED.get(symptom_domain, frozenset()):
        return True, f"cross-domain allowlist permits {pb_domain!r} for {symptom_domain!r}"
    return False, (
        f"playbook domain {pb_domain!r} is incompatible with symptom domain {symptom_domain!r}"
    )


def _symptom_playbook_compatible(symptom: SymptomEvent, playbook_id: str) -> tuple[bool, str]:
    """Check domain compatibility plus sub-domain specificity (e.g. zombies vs RAM cache)."""
    compatible, reason = domain_compatible(symptom.domain, playbook_id)
    if not compatible:
        return False, reason
    if symptom.domain == "memory_pressure":
        zombie_pbs = {"reap_zombie_parents"}
        cache_pbs = {"reclaim_memory_caches", "clean_user_memory_caches"}
        if (symptom.suggested_playbook in zombie_pbs and playbook_id not in zombie_pbs) or (
            symptom.suggested_playbook in cache_pbs and playbook_id not in cache_pbs
        ):
            return False, (
                f"playbook {playbook_id!r} does not match symptom remediation "
                f"{symptom.suggested_playbook!r}"
            )
    return True, reason

def _collect_tool_evidence(domain: str, mcp_config_path: Any) -> dict[str, str]:
    """Invoke the read-only diagnostic tools registered for *domain* and capture their output."""
    evidence: dict[str, str] = {}
    for tool_name in DOMAIN_DIAGNOSTIC_TOOLS.get(domain, ()):
        tool = _DIAGNOSTIC_TOOLS_BY_NAME.get(tool_name)
        if tool is None:
            continue
        args: dict[str, Any] = (
            {"config_path": str(mcp_config_path)} if tool_name == "inspect_mcp_config" else {}
        )
        try:
            output = tool.invoke(args)
        except Exception as exc:
            evidence[tool_name] = f"unavailable: {type(exc).__name__}: {exc}"
            continue
        evidence[tool_name] = _coerce_str(output, "", max_chars=MAX_DIAGNOSTIC_TOOL_CHARS)
    return evidence


class OmarchyHealerOrchestrator:
    """Multi-agent system monitor & self-healer built on LangGraph + Laya + Gemini/NVIDIA NIM."""

    def __init__(
        self,
        config: HealerConfig | None = None,
        *,
        triage_judge: LayaTriageJudge | None = None,
        llm_router: LLMFallbackRouter | None = None,
        memory_store: IncidentMemoryStore | None = None,
        probe_runner: Callable[[HealerConfig], list[SymptomEvent]] | None = None,
        playbook_runner: Callable[[str, dict[str, Any]], ExecutionResult] | None = None,
        osd_stuck_seconds: float = DEFAULT_OSD_STUCK_SECONDS,
    ) -> None:
        self.config = config or HealerConfig.from_env()
        self.llm_router = llm_router or LLMFallbackRouter(self.config)
        self.triage_judge = triage_judge or LayaTriageJudge(
            self.config, llm_router=self.llm_router
        )
        self.memory = memory_store or IncidentMemoryStore(self.config.db_path)
        self.osd_stuck_seconds = max(0.0, float(osd_stuck_seconds))
        self.probe_runner = probe_runner or (
            lambda cfg: collect_all_symptoms(cfg, osd_stuck_seconds=self.osd_stuck_seconds)
        )
        self.playbook_runner = playbook_runner or execute_playbook
        self.graph = self._build_graph()
        self.plan_graph = self._build_plan_graph()
        self.verify_graph = self._build_verify_graph()

    def notify_completion(self, symptom: SymptomEvent, title: str, body: str) -> bool:
        """Send the "fixed it" desktop notification. Overridden by the watch variant."""
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
        """Send the "needs a human" notification. Overridden by the watch variant."""
        try:
            return send_escalation_notification(title, body, incident_id=incident_id, exec_cmd=exec_cmd)
        except TypeError:
            return send_escalation_notification(title, body)

    def _add_remediation_nodes(self, workflow: StateGraph) -> None:
        workflow.add_node("triage", self.triage_node)
        workflow.add_node("diagnostician", self.diagnostician_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("safety_gate", self.safety_gate_node)
        workflow.add_node("capability_escalator", self.capability_escalator_node)
        workflow.add_node("record_memory", self.record_memory_node)
        workflow.add_edge("diagnostician", "planner")
        workflow.add_edge("planner", "safety_gate")
        workflow.add_edge("capability_escalator", "record_memory")
        workflow.add_edge("record_memory", END)

    def _route_from_triage(self, workflow: StateGraph) -> None:
        workflow.add_edge(START, "triage")
        workflow.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {
                "diagnose": "diagnostician",
                "escalate": "capability_escalator",
                "suppress": "record_memory",
            },
        )

    def _route_from_safety_gate(self, workflow: StateGraph) -> None:
        workflow.add_conditional_edges(
            "safety_gate",
            self._route_after_safety_gate,
            {
                "execute": "executor",
                "escalate": "capability_escalator",
                "skip": "record_memory",
            },
        )

    def _route_from_verifier(self, workflow: StateGraph) -> None:
        workflow.add_conditional_edges(
            "verifier",
            self._route_after_verifier,
            {
                "record": "record_memory",
                "escalate": "capability_escalator",
            },
        )

    def _build_graph(self):
        workflow = StateGraph(HealerGraphState)
        self._add_remediation_nodes(workflow)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("verifier", self.verifier_node)
        self._route_from_triage(workflow)
        self._route_from_safety_gate(workflow)
        workflow.add_edge("executor", "verifier")
        self._route_from_verifier(workflow)
        return workflow.compile()

    def _build_plan_graph(self):
        """Triage through Execute, stopping before verification.

        Splitting the batch lets ``scan_and_process`` execute every approved playbook
        first and then verify them all against one shared post-batch probe sweep.
        """
        workflow = StateGraph(HealerGraphState)
        self._add_remediation_nodes(workflow)
        workflow.add_node("executor", self.executor_node)
        self._route_from_triage(workflow)
        self._route_from_safety_gate(workflow)
        workflow.add_edge("executor", END)
        return workflow.compile()

    def _build_verify_graph(self):
        """Verify and record only, replaying a state already produced by the plan graph."""
        workflow = StateGraph(HealerGraphState)
        workflow.add_node("verifier", self.verifier_node)
        workflow.add_node("capability_escalator", self.capability_escalator_node)
        workflow.add_node("record_memory", self.record_memory_node)
        workflow.add_edge(START, "verifier")
        self._route_from_verifier(workflow)
        workflow.add_edge("capability_escalator", "record_memory")
        workflow.add_edge("record_memory", END)
        return workflow.compile()
    def triage_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        trace = list(state.get("agent_trace") or [])
        prior = self.memory.find_successful_fixes(symptom.fingerprint, limit=3)

        decision = self.triage_judge.evaluate(symptom)
        trace.append(
            f"[Triage:{decision.provider}] actionable={decision.is_actionable} "
            f"(p={decision.actionable_probability:.2f}) playbook={decision.selected_playbook} "
            f"tier={decision.safety_tier} tried={decision.fallback_chain_tried}"
        )
        status = "triaged" if decision.is_actionable else "suppressed"
        return {
            "prior_incidents": prior,
            "triage": decision.model_dump(),
            "status": status,
            "agent_trace": trace,
        }

    @staticmethod
    def _route_after_triage(state: HealerGraphState) -> str:
        triage = state.get("triage") or {}
        if triage.get("is_actionable"):
            return "diagnose"
        if _coerce_str(triage.get("selected_playbook"), "inspect_only") == "inspect_only":
            return "escalate"
        return "suppress"

    def _llm_diagnostician_available(self) -> bool:
        """True when the Diagnostician LLM chain has a configured provider to read tool output."""
        return self.config.has_gemini() or self.config.has_nvidia_nim()

    @staticmethod
    def _prior_successful_playbook(symptom: SymptomEvent, prior: list[dict[str, Any]]) -> str:
        """Return the newest previously successful playbook that is still domain-valid."""
        for row in prior:
            pb_id = _coerce_str(row.get("playbook_id"), "", max_chars=128)
            if pb_id in PLAYBOOK_CATALOG and domain_compatible(symptom.domain, pb_id)[0]:
                return pb_id
        return ""

    def diagnostician_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        triage = state.get("triage") or {}
        prior = state.get("prior_incidents") or []
        trace = list(state.get("agent_trace") or [])

        catalog_desc = "; ".join(
            f"{pid} (domain={meta['domain']}, tier={meta['tier']})"
            for pid, meta in PLAYBOOK_CATALOG.items()
        )
        system_prompt = (
            "You are the Diagnostician Agent for Omarchy-Healer on an Arch/Hyprland Linux desktop. "
            "Analyze the symptom, evidence, live read-only diagnostics, and prior fixes. "
            "Available playbooks: " + catalog_desc + ". "
            "When `symptom.suggested_playbook` is valid, recommend it with its catalog `safety_tier`. "
            "Respond ONLY with a JSON object containing: "
            "\"root_cause\" (string), \"evidence_summary\" (string), "
            "\"recommended_playbook\" (one of: "
            + ", ".join(PLAYBOOK_CATALOG.keys())
            + "), \"safety_tier\" (int 0..2), and \"risk_notes\" (string). "
            "Ensure all explanations, root causes, and evidence summaries are written in plain, natural English without JSON formatting or syntax."
        )
        tool_evidence: dict[str, str] = {}
        user_payload: dict[str, Any] = {
            "symptom": symptom.model_dump(),
            "triage": triage,
            "prior_successful_fixes": prior,
        }
        parsed: dict[str, Any] | None = None
        provider_used: ProviderType | None = None

        if self._llm_diagnostician_available():
            tool_evidence = _collect_tool_evidence(symptom.domain, self.config.mcp_config_path)
            if tool_evidence:
                user_payload["live_diagnostics"] = tool_evidence
            parsed, provider_used, _ = self.llm_router.invoke_json(
                system_prompt=system_prompt,
                user_prompt=json.dumps(user_payload, ensure_ascii=False),
            )

        prior_pb = self._prior_successful_playbook(symptom, prior)
        llm_pb = _coerce_str((parsed or {}).get("recommended_playbook"), "", max_chars=128)
        triage_pb = _coerce_str(triage.get("selected_playbook"), "", max_chars=128)
        candidates = [
            pb
            for pb in (prior_pb, llm_pb, triage_pb, symptom.suggested_playbook)
            if pb in PLAYBOOK_CATALOG
        ]
        unique_candidates = list(dict.fromkeys(candidates))
        compatible = [
            pb for pb in unique_candidates if _symptom_playbook_compatible(symptom, pb)[0]
        ]
        actionable_compatible = [pb for pb in compatible if pb != "inspect_only"]
        rejected = [
            f"{pb} ({_symptom_playbook_compatible(symptom, pb)[1]})"
            for pb in unique_candidates
            if not _symptom_playbook_compatible(symptom, pb)[0]
        ]
        recommended = (
            actionable_compatible[0]
            if actionable_compatible
            else (compatible[0] if compatible else "inspect_only")
        )
        memory_driven = bool(prior_pb) and recommended == prior_pb

        catalog_tier = _clamp_int(PLAYBOOK_CATALOG[recommended]["tier"], symptom.default_tier)
        llm_tier = _clamp_int((parsed or {}).get("safety_tier"), catalog_tier)
        triage_tier = _clamp_int(triage.get("safety_tier"), catalog_tier)
        tier = max(catalog_tier, llm_tier, triage_tier)

        metrics_english = format_metrics_english(symptom.metrics)
        default_evidence = f"Observed metrics: {metrics_english}."
        if parsed:
            root_cause = clean_english_text(_coerce_str(parsed.get("root_cause"), symptom.summary))
            evidence_summary = clean_english_text(
                _coerce_str(parsed.get("evidence_summary"), default_evidence)
            )
            risk_notes = clean_english_text(
                _coerce_str(parsed.get("risk_notes"), "Bounded playbook execution.")
            )
        else:
            root_cause = clean_english_text(symptom.summary)
            evidence_summary = default_evidence
            risk_notes = "Deterministic playbook matched from probe evidence."
        if tool_evidence:
            tool_findings = format_tool_evidence_english(tool_evidence)
            evidence_summary = _coerce_str(
                f"{evidence_summary}\nLive diagnostic findings:\n{tool_findings}",
                evidence_summary,
            )

        confidence = 0.6 if parsed else 0.4
        if memory_driven:
            confidence = min(1.0, confidence + 0.3)

        if memory_driven:
            provider: ProviderType = "memory_cache"
        elif provider_used:
            provider = provider_used
        else:
            provider = "heuristic"

        report = DiagnosisReport(
            provider=provider,
            root_cause=root_cause,
            evidence_summary=evidence_summary,
            recommended_playbook=recommended,
            playbook_args=dict(symptom.suggested_args),
            safety_tier=_clamp_int(tier, catalog_tier),
            risk_notes=risk_notes,
            confidence=_clamp_float(confidence, 0.4),
        )

        trace.append(
            f"[Diagnostician:{report.provider}] root_cause={report.root_cause[:90]!r} "
            f"recommended={report.recommended_playbook} tier={report.safety_tier} "
            f"confidence={report.confidence:.2f}"
        )
        if prior:
            trace.append(
                f"[Diagnostician:memory] prior_successes={len(prior)} replayed={recommended} "
                f"memory_driven={memory_driven}"
            )
        if rejected:
            trace.append(f"[Diagnostician:domain] discarded={'; '.join(rejected)}")
        if tool_evidence:
            trace.append(f"[Diagnostician:tools] invoked={sorted(tool_evidence)}")

        return {
            "diagnosis": report.model_dump(),
            "status": "diagnosed",
            "agent_trace": trace,
        }

    def planner_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        diagnosis = state.get("diagnosis") or {}
        triage = state.get("triage") or {}
        trace = list(state.get("agent_trace") or [])

        pb_id = _coerce_str(diagnosis.get("recommended_playbook"), "", max_chars=128)
        if pb_id not in PLAYBOOK_CATALOG:
            pb_id = _coerce_str(symptom.suggested_playbook, "", max_chars=128)
        if pb_id not in PLAYBOOK_CATALOG:
            pb_id = "inspect_only"
        pb_meta = PLAYBOOK_CATALOG[pb_id]
        catalog_tier = _clamp_int(pb_meta["tier"], symptom.default_tier)
        # Models may only raise the tier; the catalog floor keeps a mis-tiered LLM from
        # turning a high-risk playbook into an auto-approved one.
        tier = max(
            catalog_tier,
            _clamp_int(diagnosis.get("safety_tier"), catalog_tier),
            _clamp_int(triage.get("safety_tier"), catalog_tier),
        )
        args = diagnosis.get("playbook_args")
        plan_args = dict(args) if isinstance(args, dict) else dict(symptom.suggested_args)

        plan = RemediationPlan(
            playbook_id=pb_id,
            playbook_args=plan_args,
            safety_tier=_clamp_int(tier, catalog_tier),
            approved=False,
            description=str(pb_meta["description"]),
        )
        trace.append(
            f"[Planner] playbook={plan.playbook_id} tier={plan.safety_tier} "
            f"catalog_tier={catalog_tier}"
        )
        return {
            "plan": plan.model_dump(),
            "status": "planned",
            "agent_trace": trace,
        }

    def safety_gate_node(self, state: HealerGraphState) -> dict[str, Any]:
        plan = RemediationPlan.model_validate(state["plan"])
        symptom = SymptomEvent.model_validate(state["symptom"])
        auto_heal = bool(state.get("auto_heal", False))
        max_tier = _clamp_int(state.get("max_auto_tier"), self.config.max_auto_tier)
        trace = list(state.get("agent_trace") or [])

        catalog_tier = _clamp_int(
            PLAYBOOK_CATALOG.get(plan.playbook_id, {}).get("tier"), MIN_SAFETY_TIER
        )
        plan.safety_tier = max(_clamp_int(plan.safety_tier, MIN_SAFETY_TIER), catalog_tier)

        rejections: list[str] = []
        compatible, reason = _symptom_playbook_compatible(symptom, plan.playbook_id)
        if not compatible:
            fallback = _coerce_str(symptom.suggested_playbook, "", max_chars=128)
            if fallback not in PLAYBOOK_CATALOG:
                fallback = "inspect_only"
            elif not _symptom_playbook_compatible(symptom, fallback)[0]:
                fallback = "inspect_only"
            rejections.append(f"{plan.playbook_id}: {reason}")
            rejections.append(f"substituted={fallback}")
            plan.playbook_id = fallback
            plan.playbook_args = dict(symptom.suggested_args)
            plan.description = str(PLAYBOOK_CATALOG[fallback]["description"])
            # Keep the rejected proposal's tier so a cross-domain mistake can never
            # silently downshift into a silently auto-approved remediation.
            plan.safety_tier = max(
                plan.safety_tier,
                _clamp_int(PLAYBOOK_CATALOG[fallback]["tier"], symptom.default_tier),
            )

        if not auto_heal:
            plan.approved = False
            plan.approval_reason = "Dry-run / scan mode (auto_heal=False)"
            status = "dry_run_ready"
        elif plan.safety_tier <= max_tier:
            plan.approved = True
            plan.approval_reason = (
                f"Auto-approved: safety_tier {plan.safety_tier} <= max_auto_tier {max_tier}"
            )
            status = "approved"
        else:
            plan.approved = False
            plan.approval_reason = (
                f"Blocked by Safety Gate: playbook tier {plan.safety_tier} > max_auto_tier {max_tier}"
            )
            status = "awaiting_approval"

        if rejections:
            plan.approval_reason = (
                f"Domain validation rejected {'; '.join(rejections)} | {plan.approval_reason}"
            )
            trace.append(f"[SafetyGate:domain] rejected={'; '.join(rejections)}")

        trace.append(f"[SafetyGate] approved={plan.approved} ({plan.approval_reason})")
        return {
            "plan": plan.model_dump(),
            "status": status,
            "agent_trace": trace,
        }

    @staticmethod
    def _route_after_safety_gate(state: HealerGraphState) -> str:
        plan = state.get("plan") or {}
        if plan.get("approved"):
            return "execute"
        status = str(state.get("status") or "")
        pb_id = _coerce_str(plan.get("playbook_id"), "inspect_only")
        if status == "awaiting_approval" or pb_id == "inspect_only":
            return "escalate"
        return "skip"
    def executor_node(self, state: HealerGraphState) -> dict[str, Any]:
        plan = RemediationPlan.model_validate(state["plan"])
        trace = list(state.get("agent_trace") or [])

        result = self.playbook_runner(plan.playbook_id, plan.playbook_args)
        trace.append(
            f"[Executor] playbook={result.playbook_id} success={result.success} "
            f"duration_ms={result.duration_ms} out={result.stdout[:80]!r}"
        )
        return {
            "execution": result.model_dump(),
            "status": "executed" if result.success else "execution_failed",
            "agent_trace": trace,
        }

    def _sweep_fingerprints(self, sweep: dict[str, Any] | None) -> tuple[list[str], int, str]:
        """Return ``(fingerprints, total, source)`` from a shared sweep or a fresh probe run."""
        fingerprints = (sweep or {}).get("fingerprints")
        if isinstance(fingerprints, list):
            total = _coerce_int((sweep or {}).get("total"), len(fingerprints))
            return (
                [str(fp) for fp in fingerprints],
                max(0, total),
                "shared post-batch probe sweep",
            )
        events = self.probe_runner(self.config)
        return [ev.fingerprint for ev in events], len(events), "fresh probe sweep"

    def verifier_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        execution = ExecutionResult.model_validate(state["execution"])
        trace = list(state.get("agent_trace") or [])

        if not execution.success:
            report = VerificationReport(
                verified=True,
                resolved=False,
                post_check_summary=(
                    f"Playbook {execution.playbook_id} failed: "
                    f"{execution.stderr or execution.stdout or 'no output'}"
                ),
                remaining_anomalies=1,
            )
            trace.append("[Verifier] re-probe skipped: playbook reported failure, nothing changed")
        else:
            fingerprints, total, source = self._sweep_fingerprints(
                state.get(VERIFICATION_SWEEP_KEY)
            )
            still_present = symptom.fingerprint in set(fingerprints)
            report = VerificationReport(
                verified=True,
                resolved=not still_present,
                post_check_summary=(
                    f"Issue '{symptom.title}' cleared after {execution.playbook_id} "
                    f"({source})."
                    if not still_present
                    else f"Issue '{symptom.title}' still detected after "
                    f"{execution.playbook_id} ({source})."
                ),
                remaining_anomalies=total,
            )

        if report.resolved and execution.executed:
            notif_title, notif_body = format_completion_notification(
                symptom.title,
                execution.playbook_id,
                report.post_check_summary,
            )
            notified = self.notify_completion(
                symptom,
                notif_title,
                notif_body,
            )
            trace.append(f"[Notifier] desktop_notification_sent={notified}")

        trace.append(
            f"[Verifier] resolved={report.resolved} remaining={report.remaining_anomalies} "
            f"summary={report.post_check_summary!r}"
        )
        return {
            "verification": report.model_dump(),
            "status": "healed" if report.resolved else "unresolved",
            "agent_trace": trace,
        }


    @staticmethod
    def _route_after_verifier(state: HealerGraphState) -> str:
        ver = state.get("verification") or {}
        exe = state.get("execution") or {}
        if ver.get("resolved") and _coerce_str(exe.get("playbook_id"), "") != "inspect_only":
            return "record"
        return "escalate"

    @staticmethod
    def _propose_playbook_slug(symptom: SymptomEvent) -> str:
        raw_key = symptom.fingerprint.split(":", 1)[-1]
        slug = "".join(ch if ch.isalnum() else "_" for ch in raw_key.lower()).strip("_")
        while "__" in slug:
            slug = slug.replace("__", "_")
        return f"heal_{slug[:40]}" if slug else f"heal_{symptom.domain}"

    def capability_escalator_node(self, state: HealerGraphState) -> dict[str, Any]:
        """Synthesize a capability/feature escalation prompt when an issue cannot be fixed."""
        symptom = SymptomEvent.model_validate(state["symptom"])
        diagnosis = state.get("diagnosis") or {}
        plan = state.get("plan") or {}
        execution = state.get("execution") or {}
        verification = state.get("verification") or {}
        status = str(state.get("status") or "")
        auto_heal = bool(state.get("auto_heal", False))
        max_tier = _clamp_int(state.get("max_auto_tier"), self.config.max_auto_tier)
        trace = list(state.get("agent_trace") or [])

        pb_id = _coerce_str(
            execution.get("playbook_id")
            or plan.get("playbook_id")
            or symptom.suggested_playbook,
            "inspect_only",
            max_chars=128,
        )
        if status == "awaiting_approval":
            reason: EscalationReason = "insufficient_autonomy_tier"
        elif pb_id == "inspect_only":
            reason = "missing_playbook"
        else:
            reason = "remediation_failed"

        if reason == "missing_playbook":
            proposed_pb = self._propose_playbook_slug(symptom)
            default_missing = (
                f"No automated remediation playbook in catalog for "
                f"{symptom.domain.replace('_', ' ')} issue '{symptom.title}'."
            )
            default_spec = (
                f"Implement playbook '{proposed_pb}' (domain='{symptom.domain}') in "
                f"src/omarchy_healer/playbooks.py and wire suggested_playbook='{proposed_pb}' "
                f"in src/omarchy_healer/probes.py to remediate: {symptom.summary}"
            )
            metrics_english = format_metrics_english(symptom.metrics)
            metrics_text = (
                f" Observed metrics: {metrics_english}."
                if metrics_english and metrics_english != "none reported"
                else ""
            )
            default_prompt = (
                f"Omarchy-Healer detected '{symptom.title}' in domain "
                f"'{symptom.domain.replace('_', ' ')}', but lacks a remediation playbook (currently falls back to "
                f"'inspect_only'). Symptom summary: {symptom.summary}.{metrics_text} "
                f"Please add a new bounded, idempotent playbook "
                f"'{proposed_pb}' in src/omarchy_healer/playbooks.py and wire it in "
                f"src/omarchy_healer/probes.py so the healer agents have the capacity to fix it."
            )
        elif reason == "insufficient_autonomy_tier":
            proposed_pb = pb_id
            plan_tier = _clamp_int(plan.get("safety_tier"), symptom.default_tier)
            default_missing = (
                f"Playbook '{proposed_pb}' requires safety tier {plan_tier}, exceeding "
                f"current max_auto_tier={max_tier}."
            )
            default_spec = (
                f"Either add a safe non-destructive Tier {max_tier} remediation path for "
                f"'{proposed_pb}' in src/omarchy_healer/playbooks.py or run with "
                f"--max-tier {plan_tier} (HEALER_MAX_AUTO_TIER={plan_tier})."
            )
            default_prompt = (
                f"Omarchy-Healer planned '{proposed_pb}' for '{symptom.title}', "
                f"but SafetyGate blocked execution because "
                f"safety tier {plan_tier} exceeds maximum automatic tier {max_tier}. "
                f"Please add a safe Tier {max_tier} remediation feature for '{proposed_pb}' in "
                f"src/omarchy_healer/playbooks.py or grant Tier {plan_tier} autonomy."
            )
        else:
            proposed_pb = pb_id if pb_id != "inspect_only" else self._propose_playbook_slug(symptom)
            failure_detail = clean_english_text(
                _coerce_str(
                    execution.get("stderr")
                    or verification.get("post_check_summary")
                    or execution.get("stdout"),
                    "Symptom remained present after playbook execution.",
                    max_chars=500,
                )
            )
            root_cause = clean_english_text(
                _coerce_str(diagnosis.get("root_cause"), symptom.summary, max_chars=400)
            )
            default_missing = (
                f"Playbook '{proposed_pb}' failed or lacked capacity to resolve "
                f"'{symptom.title}': {failure_detail}"
            )
            default_spec = (
                f"Upgrade '{proposed_pb}' in src/omarchy_healer/playbooks.py (or add a "
                f"new escalation playbook for '{symptom.domain}') to handle root cause: "
                f"{root_cause} (failure: {failure_detail})."
            )
            default_prompt = (
                f"Omarchy-Healer attempted to fix '{symptom.title}' "
                f"using playbook '{proposed_pb}', but remediation could not resolve the issue. "
                f"Failure detail: {failure_detail} | Root cause: {root_cause}. "
                f"Please upgrade '{proposed_pb}' or add a new remediation feature in "
                f"src/omarchy_healer/playbooks.py and src/omarchy_healer/probes.py so "
                f"Omarchy-Healer has the capacity to heal this anomaly automatically."
            )

        provider: ProviderType = "heuristic"
        missing_cap = default_missing
        feature_spec = default_spec
        agent_prompt = default_prompt

        if self._llm_diagnostician_available():
            esc_system = (
                "You are the Capability Escalator Agent for Omarchy-Healer on Linux. "
                "The healer encountered an anomaly it could not fix with current playbooks. "
                "Respond ONLY with a JSON object containing: "
                "\"missing_capability\" (string), \"proposed_playbook_id\" (snake_case string), "
                "\"proposed_feature_spec\" (string describing code changes in playbooks.py/probes.py), "
                "and \"agent_prompt\" (a direct prompt asking the developer/coding agent to "
                "implement this feature in omarchy-healer). "
                "All text descriptions and explanations must be in clear, plain English without raw JSON formatting or syntax."
            )
            esc_user = json.dumps(
                {
                    "reason": reason,
                    "symptom": symptom.model_dump(),
                    "plan": plan,
                    "execution": execution,
                    "verification": verification,
                },
                ensure_ascii=False,
            )
            parsed, provider_used, _ = self.llm_router.invoke_json(
                system_prompt=esc_system,
                user_prompt=esc_user,
            )
            if parsed and provider_used and parsed.get("agent_prompt") and parsed.get("proposed_feature_spec"):
                provider = provider_used
                missing_cap = clean_english_text(_coerce_str(parsed.get("missing_capability"), default_missing))
                proposed_pb = _coerce_str(
                    parsed.get("proposed_playbook_id"), proposed_pb, max_chars=128
                )
                feature_spec = clean_english_text(_coerce_str(parsed.get("proposed_feature_spec"), default_spec))
                agent_prompt = clean_english_text(_coerce_str(parsed.get("agent_prompt"), default_prompt))

        notified = False
        if auto_heal or reason == "missing_playbook":
            notif_title, notif_body = format_escalation_notification(
                reason=reason,
                symptom_title=symptom.title,
                domain=symptom.domain,
                proposed_playbook_id=proposed_pb,
                failure_detail=failure_detail if reason == "remediation_failed" else "",
                plan_tier=plan_tier if reason == "insufficient_autonomy_tier" else 0,
                max_tier=max_tier if reason == "insufficient_autonomy_tier" else 0,
            )
            notified = self.notify_escalation(
                symptom,
                notif_title,
                notif_body,
            )

        escalation = CapabilityEscalation(
            reason=reason,
            provider=provider,
            missing_capability=missing_cap,
            proposed_playbook_id=proposed_pb,
            proposed_feature_spec=feature_spec,
            agent_prompt=agent_prompt,
            notified=notified,
        )
        trace.append(
            f"[Escalator:{escalation.provider}] reason={escalation.reason} "
            f"proposed={escalation.proposed_playbook_id} notified={escalation.notified}"
        )
        return {
            "escalation": escalation.model_dump(),
            "agent_trace": trace,
        }
    def record_memory_node(self, state: HealerGraphState) -> dict[str, Any]:
        trace = list(state.get("agent_trace") or [])
        payload = cast(
            HealerGraphState,
            {key: value for key, value in state.items() if key != VERIFICATION_SWEEP_KEY},
        )
        row_id = self.memory.record_state(payload)
        trace.append(f"[Memory] recorded incident row_id={row_id}")
        return {"agent_trace": trace}

    def _initial_state(
        self,
        symptom: SymptomEvent,
        *,
        auto_heal: bool,
        max_auto_tier: int | None,
    ) -> HealerGraphState:
        return {
            "symptom": symptom.model_dump(),
            "auto_heal": auto_heal,
            "max_auto_tier": (
                self.config.max_auto_tier if max_auto_tier is None else max_auto_tier
            ),
            "agent_trace": [],
        }

    def process_symptom(
        self,
        symptom: SymptomEvent,
        *,
        auto_heal: bool = False,
        max_auto_tier: int | None = None,
    ) -> HealerGraphState:
        """Run a single SymptomEvent through the full LangGraph multi-agent pipeline."""
        return self.graph.invoke(
            self._initial_state(symptom, auto_heal=auto_heal, max_auto_tier=max_auto_tier)
        )

    def scan_and_process(
        self,
        *,
        auto_heal: bool = False,
        max_auto_tier: int | None = None,
    ) -> list[HealerGraphState]:
        """Run all system probes and process each detected SymptomEvent through the graph.

        The whole cycle costs one detection sweep plus at most one post-batch
        verification sweep, no matter how many symptoms were healed, because every
        playbook in the batch executes before anything is verified.
        """
        symptoms = self.probe_runner(self.config)
        results: list[HealerGraphState | None] = [None] * len(symptoms)
        pending: list[tuple[int, HealerGraphState]] = []

        for index, symptom in enumerate(symptoms):
            state = cast(
                HealerGraphState,
                self.plan_graph.invoke(
                    self._initial_state(symptom, auto_heal=auto_heal, max_auto_tier=max_auto_tier)
                ),
            )
            if state.get("execution"):
                pending.append((index, state))
            else:
                results[index] = state

        needs_sweep = any(
            ExecutionResult.model_validate(state["execution"]).success
            for _, state in pending
        )
        sweep: dict[str, Any] | None = None
        if needs_sweep:
            events = self.probe_runner(self.config)
            sweep = {
                "fingerprints": [ev.fingerprint for ev in events],
                "total": len(events),
                "swept_at": datetime.now(timezone.utc).isoformat(),
            }

        for index, state in pending:
            if sweep is not None:
                state[VERIFICATION_SWEEP_KEY] = sweep
            results[index] = cast(HealerGraphState, self.verify_graph.invoke(state))

        return [state for state in results if state is not None]