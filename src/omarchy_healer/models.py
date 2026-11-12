"""Data models and LangGraph state schema for Omarchy-Healer."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from pydantic import BaseModel, Field

DomainType = Literal[
    "ui_shell",
    "mcp_config",
    "systemd_service",
    "memory_pressure",
    "compositor",
    "other",
]

ProviderType = Literal["laya", "gemini", "nvidia_nim", "heuristic", "memory_cache"]
EscalationReason = Literal[
    "missing_playbook",
    "insufficient_autonomy_tier",
    "remediation_failed",
]


MIN_SAFETY_TIER = 0
MAX_SAFETY_TIER = 2

VERIFICATION_SWEEP_KEY = "verification_sweep"


class SymptomEvent(BaseModel):
    """Structured anomaly emitted by a deterministic system probe."""

    id: str
    fingerprint: str
    probe_name: str
    domain: DomainType
    title: str
    summary: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    raw_evidence: dict[str, Any] = Field(default_factory=dict)
    suggested_playbook: str = ""
    suggested_args: dict[str, Any] = Field(default_factory=dict)
    default_tier: int = 0
    detected_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def create(
        cls,
        *,
        probe_name: str,
        domain: DomainType,
        fingerprint_key: str,
        title: str,
        summary: str,
        metrics: dict[str, Any] | None = None,
        raw_evidence: dict[str, Any] | None = None,
        suggested_playbook: str = "",
        suggested_args: dict[str, Any] | None = None,
        default_tier: int = 0,
    ) -> SymptomEvent:
        digest = hashlib.sha256(f"{probe_name}:{fingerprint_key}".encode("utf-8")).hexdigest()[:12]
        return cls(
            id=f"inc-{digest}",
            fingerprint=f"{probe_name}:{fingerprint_key}",
            probe_name=probe_name,
            domain=domain,
            title=title,
            summary=summary,
            metrics=metrics or {},
            raw_evidence=raw_evidence or {},
            suggested_playbook=suggested_playbook,
            suggested_args=suggested_args or {},
            default_tier=default_tier,
        )


class TriageDecision(BaseModel):
    """Output of the Triage & Safety Judge (Laya primary -> Gemini -> NVIDIA NIM -> Heuristic)."""

    provider: ProviderType
    domain: DomainType
    severity_score: float = Field(ge=0.0, le=1.0)
    is_actionable: bool
    actionable_probability: float = Field(ge=0.0, le=1.0)
    safety_tier: int = Field(ge=MIN_SAFETY_TIER, le=MAX_SAFETY_TIER)
    selected_playbook: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str = ""
    fallback_chain_tried: list[str] = Field(default_factory=list)


class DiagnosisReport(BaseModel):
    """Output of the Diagnostician Agent."""

    provider: ProviderType
    root_cause: str
    evidence_summary: str
    recommended_playbook: str
    playbook_args: dict[str, Any] = Field(default_factory=dict)
    safety_tier: int = Field(default=MIN_SAFETY_TIER, ge=MIN_SAFETY_TIER, le=MAX_SAFETY_TIER)
    risk_notes: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RemediationPlan(BaseModel):
    """Output of the Remediation Planner + Safety Gate."""

    playbook_id: str
    playbook_args: dict[str, Any] = Field(default_factory=dict)
    safety_tier: int = Field(default=MIN_SAFETY_TIER, ge=MIN_SAFETY_TIER, le=MAX_SAFETY_TIER)
    approved: bool = False
    approval_reason: str = ""
    description: str = ""


class ExecutionResult(BaseModel):
    """Output of the Executor Agent."""

    executed: bool
    playbook_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    backup_path: str | None = None
    duration_ms: int = 0


class VerificationReport(BaseModel):
    """Output of the Verifier Agent."""

    verified: bool
    resolved: bool
    post_check_summary: str
    remaining_anomalies: int = 0



class CapabilityEscalation(BaseModel):
    """Output of the Capability Escalator Agent when an issue cannot be resolved."""

    reason: EscalationReason
    provider: ProviderType
    missing_capability: str
    proposed_playbook_id: str
    proposed_feature_spec: str
    agent_prompt: str
    notified: bool = False

class HealerGraphState(TypedDict, total=False):
    """State passed through the LangGraph multi-agent workflow."""

    symptom: dict[str, Any]
    auto_heal: bool
    max_auto_tier: int
    prior_incidents: list[dict[str, Any]]
    triage: dict[str, Any]
    diagnosis: dict[str, Any]
    plan: dict[str, Any]
    execution: dict[str, Any]
    verification: dict[str, Any]
    status: str
    agent_trace: list[str]
    verification_sweep: dict[str, Any]
    escalation: dict[str, Any]
