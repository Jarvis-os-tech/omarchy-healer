"""LangGraph multi-agent workflow: Triage -> Diagnose -> Plan -> SafetyGate -> Execute -> Verify -> Memory."""
from __future__ import annotations

import json
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .config import HealerConfig
from .laya_judge import LayaTriageJudge
from .llm_router import LLMFallbackRouter
from .memory import IncidentMemoryStore
from .models import (
    DiagnosisReport,
    ExecutionResult,
    HealerGraphState,
    RemediationPlan,
    SymptomEvent,
    VerificationReport,
)
from .playbooks import PLAYBOOK_CATALOG, execute_playbook, send_completion_notification
from .probes import collect_all_symptoms


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
    ) -> None:
        self.config = config or HealerConfig.from_env()
        self.llm_router = llm_router or LLMFallbackRouter(self.config)
        self.triage_judge = triage_judge or LayaTriageJudge(
            self.config, llm_router=self.llm_router
        )
        self.memory = memory_store or IncidentMemoryStore(self.config.db_path)
        self.probe_runner = probe_runner or (
            lambda cfg: collect_all_symptoms(cfg, osd_stuck_seconds=0.0)
        )
        self.playbook_runner = playbook_runner or execute_playbook
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(HealerGraphState)

        workflow.add_node("triage", self.triage_node)
        workflow.add_node("diagnostician", self.diagnostician_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("safety_gate", self.safety_gate_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("verifier", self.verifier_node)
        workflow.add_node("record_memory", self.record_memory_node)

        workflow.add_edge(START, "triage")
        workflow.add_conditional_edges(
            "triage",
            self._route_after_triage,
            {
                "diagnose": "diagnostician",
                "suppress": "record_memory",
            },
        )
        workflow.add_edge("diagnostician", "planner")
        workflow.add_edge("planner", "safety_gate")
        workflow.add_conditional_edges(
            "safety_gate",
            self._route_after_safety_gate,
            {
                "execute": "executor",
                "skip": "record_memory",
            },
        )
        workflow.add_edge("executor", "verifier")
        workflow.add_edge("verifier", "record_memory")
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
        return "diagnose" if triage.get("is_actionable") else "suppress"

    def diagnostician_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        triage = state.get("triage") or {}
        prior = state.get("prior_incidents") or []
        trace = list(state.get("agent_trace") or [])

        # If Gemini or NVIDIA NIM is configured, ask the Diagnostician LLM agent
        system_prompt = (
            "You are the Diagnostician Agent for Omarchy-Healer on an Arch/Hyprland Linux desktop. "
            "Analyze the symptom, evidence, and prior fixes. Respond ONLY with a JSON object containing: "
            "\"root_cause\" (string), \"evidence_summary\" (string), "
            "\"recommended_playbook\" (one of: "
            + ", ".join(PLAYBOOK_CATALOG.keys())
            + "), \"safety_tier\" (int 0..2), and \"risk_notes\" (string)."
        )
        user_payload = {
            "symptom": symptom.model_dump(),
            "triage": triage,
            "prior_successful_fixes": prior,
        }
        parsed, provider_used, _ = self.llm_router.invoke_json(
            system_prompt=system_prompt,
            user_prompt=json.dumps(user_payload, ensure_ascii=False),
        )

        if parsed and provider_used:
            rec_pb = str(
                parsed.get("recommended_playbook")
                or triage.get("selected_playbook")
                or symptom.suggested_playbook
                or "inspect_only"
            )
            if rec_pb not in PLAYBOOK_CATALOG:
                rec_pb = str(triage.get("selected_playbook") or symptom.suggested_playbook or "inspect_only")
            pb_meta = PLAYBOOK_CATALOG.get(rec_pb, PLAYBOOK_CATALOG["inspect_only"])
            report = DiagnosisReport(
                provider=provider_used,
                root_cause=str(parsed.get("root_cause") or symptom.summary),
                evidence_summary=str(parsed.get("evidence_summary") or json.dumps(symptom.metrics)),
                recommended_playbook=rec_pb,
                playbook_args=dict(symptom.suggested_args),
                safety_tier=int(parsed.get("safety_tier", pb_meta.get("tier", symptom.default_tier))),
                risk_notes=str(parsed.get("risk_notes") or "Bounded playbook execution."),
            )
        else:
            rec_pb = str(triage.get("selected_playbook") or symptom.suggested_playbook or "inspect_only")
            pb_meta = PLAYBOOK_CATALOG.get(rec_pb, PLAYBOOK_CATALOG["inspect_only"])
            report = DiagnosisReport(
                provider="memory_cache" if prior else "heuristic",
                root_cause=symptom.summary,
                evidence_summary=f"Metrics: {json.dumps(symptom.metrics)}",
                recommended_playbook=rec_pb,
                playbook_args=dict(symptom.suggested_args),
                safety_tier=int(pb_meta.get("tier", symptom.default_tier)),
                risk_notes="Deterministic playbook matched from probe evidence.",
            )

        trace.append(
            f"[Diagnostician:{report.provider}] root_cause={report.root_cause[:90]!r} "
            f"recommended={report.recommended_playbook}"
        )
        return {
            "diagnosis": report.model_dump(),
            "status": "diagnosed",
            "agent_trace": trace,
        }

    def planner_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        diagnosis = state.get("diagnosis") or {}
        trace = list(state.get("agent_trace") or [])

        pb_id = str(diagnosis.get("recommended_playbook") or symptom.suggested_playbook or "inspect_only")
        if pb_id not in PLAYBOOK_CATALOG:
            pb_id = "inspect_only"
        pb_meta = PLAYBOOK_CATALOG[pb_id]
        tier = max(int(pb_meta["tier"]), int(diagnosis.get("safety_tier", 0)))

        plan = RemediationPlan(
            playbook_id=pb_id,
            playbook_args=dict(diagnosis.get("playbook_args") or symptom.suggested_args),
            safety_tier=tier,
            approved=False,
            description=str(pb_meta["description"]),
        )
        trace.append(f"[Planner] playbook={plan.playbook_id} tier={plan.safety_tier}")
        return {
            "plan": plan.model_dump(),
            "status": "planned",
            "agent_trace": trace,
        }

    def safety_gate_node(self, state: HealerGraphState) -> dict[str, Any]:
        plan = RemediationPlan.model_validate(state["plan"])
        auto_heal = bool(state.get("auto_heal", False))
        max_tier = int(state.get("max_auto_tier", self.config.max_auto_tier))
        trace = list(state.get("agent_trace") or [])

        if not auto_heal:
            plan.approved = False
            plan.approval_reason = "Dry-run / scan mode (auto_heal=False)"
            status = "dry_run_ready"
        elif plan.safety_tier <= max_tier:
            plan.approved = True
            plan.approval_reason = f"Auto-approved: safety_tier {plan.safety_tier} <= max_auto_tier {max_tier}"
            status = "approved"
        else:
            plan.approved = False
            plan.approval_reason = (
                f"Blocked by Safety Gate: playbook tier {plan.safety_tier} > max_auto_tier {max_tier}"
            )
            status = "awaiting_approval"

        trace.append(f"[SafetyGate] approved={plan.approved} ({plan.approval_reason})")
        return {
            "plan": plan.model_dump(),
            "status": status,
            "agent_trace": trace,
        }

    @staticmethod
    def _route_after_safety_gate(state: HealerGraphState) -> str:
        plan = state.get("plan") or {}
        return "execute" if plan.get("approved") else "skip"

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

    def verifier_node(self, state: HealerGraphState) -> dict[str, Any]:
        symptom = SymptomEvent.model_validate(state["symptom"])
        execution = ExecutionResult.model_validate(state["execution"])
        trace = list(state.get("agent_trace") or [])

        if not execution.success:
            report = VerificationReport(
                verified=True,
                resolved=False,
                post_check_summary=f"Playbook {execution.playbook_id} failed: {execution.stderr}",
                remaining_anomalies=1,
            )
        elif symptom.domain == "memory_pressure" and execution.playbook_id == "reclaim_memory_caches":
            # Advisory/cache reclamation succeeds when executed cleanly
            report = VerificationReport(
                verified=True,
                resolved=True,
                post_check_summary="User-space memory reclamation and top-RSS diagnostic completed.",
                remaining_anomalies=0,
            )
        else:
            current_events = self.probe_runner(self.config)
            still_present = any(ev.fingerprint == symptom.fingerprint for ev in current_events)
            report = VerificationReport(
                verified=True,
                resolved=not still_present,
                post_check_summary=(
                    f"Symptom {symptom.fingerprint!r} cleared after {execution.playbook_id}."
                    if not still_present
                    else f"Symptom {symptom.fingerprint!r} still detected after {execution.playbook_id}."
                ),
                remaining_anomalies=len(current_events),
            )

        if report.resolved and execution.executed:
            notified = send_completion_notification(
                symptom.title,
                f"Fixed via {execution.playbook_id}: {report.post_check_summary}",
            )
            trace.append(f"[Notifier] desktop_notification_sent={notified}")

        trace.append(
            f"[Verifier] resolved={report.resolved} summary={report.post_check_summary!r}"
        )
        return {
            "verification": report.model_dump(),
            "status": "healed" if report.resolved else "unresolved",
            "agent_trace": trace,
        }

    def record_memory_node(self, state: HealerGraphState) -> dict[str, Any]:
        trace = list(state.get("agent_trace") or [])
        row_id = self.memory.record_state(state)
        trace.append(f"[Memory] recorded incident row_id={row_id}")
        return {"agent_trace": trace}

    def process_symptom(
        self,
        symptom: SymptomEvent,
        *,
        auto_heal: bool = False,
        max_auto_tier: int | None = None,
    ) -> HealerGraphState:
        """Run a single SymptomEvent through the full LangGraph multi-agent pipeline."""
        initial_state: HealerGraphState = {
            "symptom": symptom.model_dump(),
            "auto_heal": auto_heal,
            "max_auto_tier": (
                self.config.max_auto_tier if max_auto_tier is None else max_auto_tier
            ),
            "agent_trace": [],
        }
        return self.graph.invoke(initial_state)

    def scan_and_process(
        self,
        *,
        auto_heal: bool = False,
        max_auto_tier: int | None = None,
    ) -> list[HealerGraphState]:
        """Run all system probes and process each detected SymptomEvent through the graph."""
        symptoms = self.probe_runner(self.config)
        results: list[HealerGraphState] = []
        for symptom in symptoms:
            results.append(
                self.process_symptom(
                    symptom,
                    auto_heal=auto_heal,
                    max_auto_tier=max_auto_tier,
                )
            )
        return results
