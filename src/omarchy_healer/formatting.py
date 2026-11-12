"""Plain-English formatters and sanitizers for Omarchy-Healer notifications and prompts."""
from __future__ import annotations

import json
import re
from typing import Any


def format_metrics_english(metrics: Any) -> str:
    """Format structured probe metrics into pure, human-readable English.

    Produces descriptive text without JSON braces, brackets, quotes, or colons.
    """
    if not metrics:
        return "none reported"

    if isinstance(metrics, str):
        stripped = metrics.strip()
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            try:
                parsed = json.loads(stripped)
                return format_metrics_english(parsed)
            except Exception:
                pass
        return metrics

    if not isinstance(metrics, dict):
        if isinstance(metrics, (list, tuple, set)):
            return ", ".join(clean_english_text(str(x)) for x in metrics)
        return str(metrics)

    items: list[str] = []
    for k, v in metrics.items():
        if v is None:
            continue
        key_str = str(k).strip()
        if isinstance(v, (dict, list, tuple)):
            val_str = format_metrics_english(v)
        else:
            val_str = str(v).strip()

        if not val_str:
            continue

        k_lower = key_str.lower()
        if k_lower == "pid":
            items.append(f"process ID {val_str}")
        elif k_lower == "signal":
            items.append(f"signal {val_str}")
        elif k_lower == "program":
            items.append(f"program {val_str}")
        elif k_lower == "unit":
            items.append(f"service unit {val_str}")
        elif k_lower == "used_percent":
            try:
                items.append(f"memory used {float(v):.1f}%")
            except (ValueError, TypeError):
                items.append(f"memory used {val_str}%")
        elif k_lower == "available_mb":
            try:
                items.append(f"{float(v):.0f} MiB memory available")
            except (ValueError, TypeError):
                items.append(f"{val_str} MiB memory available")
        elif k_lower == "total_mb":
            try:
                items.append(f"{float(v):.0f} MiB total memory")
            except (ValueError, TypeError):
                items.append(f"{val_str} MiB total memory")
        elif k_lower == "critical":
            items.append(f"severity critical: {str(v).lower()}")
        elif k_lower == "zombie_count":
            items.append(f"{val_str} defunct zombie processes")
        elif k_lower == "unsignalled_count":
            items.append(f"{val_str} unsignalled processes")
        elif k_lower == "error_count":
            items.append(f"{val_str} errors")
        elif k_lower == "server_name":
            items.append(f"server name '{val_str}'")
        elif k_lower == "config_path":
            items.append(f"config path '{val_str}'")
        elif k_lower == "configured_command":
            items.append(f"configured command '{val_str}'")
        elif k_lower == "alternate_found":
            items.append(f"alternate binary found: {str(v).lower()}")
        elif k_lower == "osd_state":
            items.append(f"OSD state '{val_str}'")
        elif k_lower == "osd_layer_present":
            items.append(f"OSD layer present: {str(v).lower()}")
        elif k_lower == "observed_open_seconds":
            items.append(f"open for {val_str} seconds")
        elif k_lower == "bar_layer_present":
            items.append(f"bar layer present: {str(v).lower()}")
        elif k_lower in ("some_avg10", "psi_some10"):
            items.append(f"PSI pressure avg10: {val_str}")
        elif k_lower in ("some_avg60", "psi_some60"):
            items.append(f"PSI pressure avg60: {val_str}")
        elif k_lower in ("full_avg10", "psi_full10"):
            items.append(f"PSI stall avg10: {val_str}")
        else:
            human_key = key_str.replace("_", " ")
            items.append(f"{human_key}: {val_str}")

    if not items:
        return "none reported"
    return "; ".join(items)


def clean_english_text(text: str) -> str:
    """Sanitize arbitrary text, stripping JSON syntax, braces, or code blocks into natural English."""
    if not text or not isinstance(text, str):
        return ""

    stripped = text.strip()
    # Check if the entire string is a JSON object or array
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            parsed = json.loads(stripped)
            return format_metrics_english(parsed)
        except Exception:
            pass

    # Replace markdown code blocks ```json ... ``` or ``` ... ```
    def _replace_code_block(match: re.Match[str]) -> str:
        code = match.group(1).strip()
        try:
            parsed = json.loads(code)
            return format_metrics_english(parsed)
        except Exception:
            return code

    cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", _replace_code_block, text)

    # Detect inline balanced JSON objects { ... }
    out: list[str] = []
    i = 0
    n = len(cleaned)
    while i < n:
        if cleaned[i] == "{" and (i == 0 or cleaned[i - 1] != "$"):
            depth = 1
            j = i + 1
            in_str = False
            escape = False
            while j < n and depth > 0:
                c = cleaned[j]
                if escape:
                    escape = False
                elif c == "\\":
                    escape = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                j += 1

            if depth == 0:
                candidate = cleaned[i:j]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, (dict, list)):
                        formatted = format_metrics_english(parsed)
                        out.append(f"({formatted})")
                        i = j
                        continue
                except Exception:
                    pass
        out.append(cleaned[i])
        i += 1

    cleaned = "".join(out)
    # Remove escaped quotes, redundant braces if any
    cleaned = cleaned.replace('\\"', '"')
    return cleaned.strip()


def format_tool_evidence_english(tool_evidence: dict[str, str]) -> str:
    """Format live diagnostic tool outputs into human-readable English bullet points."""
    if not tool_evidence:
        return ""
    lines: list[str] = []
    for tool_name, result in tool_evidence.items():
        clean_name = tool_name.replace("_", " ").title()
        res_str = str(result).strip()
        cleaned_res = clean_english_text(res_str)
        lines.append(f"- {clean_name}: {cleaned_res}")
    return "\n".join(lines)


def format_completion_notification(
    symptom_title: str, playbook_id: str, post_check_summary: str
) -> tuple[str, str]:
    """Build a tagged, clean-English completion notification title and body describing what was done."""
    friendly_pb = playbook_id.replace("_", " ")
    clean_title = clean_english_text(symptom_title)
    clean_summary = clean_english_text(post_check_summary)

    title = f"[Issue Resolved] {clean_title}"
    body = (
        f"Tag: Issue Resolved (Auto-Healed)\n"
        f"Status: Resolved with {friendly_pb}.\n"
        f"What was done: Fixed via playbook '{playbook_id}'. {clean_summary}\n"
        f"Result: '{clean_title}' is verified healthy."
    )
    return title, body


def format_escalation_notification(
    reason: str,
    symptom_title: str,
    domain: str,
    proposed_playbook_id: str,
    failure_detail: str = "",
    plan_tier: int = 0,
    max_tier: int = 0,
) -> tuple[str, str]:
    """Build a tagged, clean-English escalation notification title and body explaining what is needed."""
    clean_title = clean_english_text(symptom_title)
    clean_domain = domain.replace("_", " ")
    friendly_pb = proposed_playbook_id.replace("_", " ")
    clean_fail = clean_english_text(failure_detail)

    if reason == "missing_playbook":
        tag = "Implement feature (Add capability to me)"
        title = f"[Add Capability] {clean_title}"
        body = (
            f"Tag: {tag}\n"
            f"What is needed: Please implement a new remediation playbook in omarchy-healer.\n"
            f"Problem: Detected {clean_domain} issue '{clean_title}' with no remediation playbook available.\n"
            f"Action required: Implement playbook '{proposed_playbook_id}' ({friendly_pb}) to give me the capacity to fix this.\n"
            f"Details: Run 'omarchy-healer escalations' to view the code prompt."
        )
    elif reason == "insufficient_autonomy_tier":
        tag = "Permission required (Higher tier approval)"
        title = f"[Permission Required] {clean_title}"
        body = (
            f"Tag: {tag}\n"
            f"What is needed: Higher safety tier permission required to remediate.\n"
            f"Problem: Playbook '{proposed_playbook_id}' requires safety tier {plan_tier}, exceeding current max auto tier {max_tier}.\n"
            f"Action required: Grant Tier {plan_tier} permission or run with '--max-tier {plan_tier}' to allow automatic repair."
        )
    else:  # remediation_failed
        tag = "Upgrade capability (Feature enhancement needed)"
        title = f"[Upgrade Capability] {clean_title}"
        fail_snippet = f": {clean_fail}" if clean_fail else ""
        body = (
            f"Tag: {tag}\n"
            f"What is needed: Upgrade existing playbook feature in omarchy-healer.\n"
            f"Problem: Attempted playbook '{proposed_playbook_id}', but remediation could not resolve '{clean_title}'{fail_snippet}.\n"
            f"Action required: Upgrade '{proposed_playbook_id}' with improved remediation logic.\n"
            f"Details: Run 'omarchy-healer escalations' to view the code prompt."
        )

    return title, body


def format_improvement_agent_prompt(
    *,
    incident_id: int | str = "latest",
    title: str = "Unknown Anomaly",
    domain: str = "general",
    reason: str = "improvement_needed",
    missing_capability: str = "",
    proposed_playbook_id: str = "",
    proposed_feature_spec: str = "",
    failure_detail: str = "",
    agent_prompt_override: str = "",
    created_at: str = "",
) -> str:
    """Format a structured prompt instructing the default AI agent to improve omarchy-healer."""
    clean_t = clean_english_text(title)
    clean_d = clean_english_text(domain)
    clean_reason = reason.replace("_", " ")
    clean_missing = clean_english_text(missing_capability)
    clean_spec = clean_english_text(proposed_feature_spec)
    clean_fail = clean_english_text(failure_detail)

    prompt = (
        f"Omarchy-Healer detected an anomaly and requires code improvement or new remediation capabilities.\n\n"
        f"Incident Context:\n"
        f"  Incident ID: #{incident_id}"
        + (f" (recorded at {created_at})\n" if created_at else "\n")
        + f"  Symptom: {clean_t}\n"
        f"  Domain: {clean_d}\n"
        f"  Escalation Reason: {clean_reason}\n\n"
        f"What is Needed:\n"
        f"  {clean_missing or 'Implement or enhance autonomous remediation logic for this anomaly.'}\n\n"
        f"Proposed Playbook:\n"
        f"  {proposed_playbook_id or 'new_playbook'}\n\n"
        f"Feature Specification:\n"
        f"  {clean_spec or 'Add or enhance playbook in src/omarchy_healer/playbooks.py.'}\n\n"
    )
    if clean_fail:
        prompt += f"Failure Details:\n  {clean_fail}\n\n"

    if agent_prompt_override:
        prompt += f"Specific Request:\n  {clean_english_text(agent_prompt_override)}\n\n"

    prompt += (
        "Developer Instructions:\n"
        "1. Open and update `src/omarchy_healer/playbooks.py` to implement or enhance the required playbook.\n"
        "2. If needed, update symptom probes in `src/omarchy_healer/probes.py`.\n"
        "3. Ensure the playbook is registered in PLAYBOOK_CATALOG and properly handled in router/graph.\n"
        "4. Run the test suite to verify your changes:\n"
        "   uv run pytest\n"
        "5. Test scanning with:\n"
        "   uv run omarchy-healer scan\n"
    )
    return prompt

