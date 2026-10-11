"""LangChain multi-provider router with Gemini -> NVIDIA NIM automatic failover."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .config import HealerConfig
from .models import ProviderType

log = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract a JSON object from an LLM response string."""
    text = (raw_text or "").strip()
    if not text:
        return None
    match = _JSON_BLOCK_RE.search(text)
    if match:
        text = match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


class LLMFallbackRouter:
    """Routes LangChain prompts across Gemini (Fallback 1) -> NVIDIA NIM (Fallback 2)."""

    def __init__(
        self,
        config: HealerConfig,
        *,
        custom_invokers: list[tuple[ProviderType, Any]] | None = None,
    ) -> None:
        self.config = config
        self._custom_invokers = custom_invokers

    def _build_chain(self) -> list[tuple[ProviderType, Any]]:
        if self._custom_invokers is not None:
            return list(self._custom_invokers)

        models: list[tuple[ProviderType, Any]] = []
        if self.config.has_gemini():
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                gemini_llm = ChatGoogleGenerativeAI(
                    model=self.config.gemini_model,
                    google_api_key=self.config.gemini_api_key,
                    temperature=0.1,
                    timeout=10,
                    max_retries=0,
                )
                models.append(("gemini", gemini_llm))
            except Exception as exc:
                log.warning("Failed to initialize ChatGoogleGenerativeAI: %s", exc)

        if self.config.has_nvidia_nim():
            try:
                from langchain_openai import ChatOpenAI

                nvidia_llm = ChatOpenAI(
                    model=self.config.nvidia_model,
                    api_key=self.config.nvidia_api_key,
                    base_url=self.config.nvidia_base_url,
                    temperature=0.1,
                    timeout=8,
                    max_retries=0,
                )
                models.append(("nvidia_nim", nvidia_llm))
            except Exception as exc:
                log.warning("Failed to initialize NVIDIA NIM ChatOpenAI: %s", exc)

        return models

    def invoke_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any] | None, ProviderType | None, list[str]]:
        """Invoke the first healthy LangChain chat model and parse its JSON response."""
        tried: list[str] = []
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        for provider_name, llm in self._build_chain():
            tried.append(provider_name)
            try:
                response = llm.invoke(messages)
                content = getattr(response, "content", "")
                if isinstance(content, list):
                    content = " ".join(
                        str(part.get("text", part)) if isinstance(part, dict) else str(part)
                        for part in content
                    )
                parsed = extract_json_object(str(content))
                if parsed is not None:
                    return parsed, provider_name, tried
                log.warning("Provider %s returned non-JSON response", provider_name)
            except Exception as exc:
                log.warning("Provider %s invocation failed: %s", provider_name, exc)
                continue

        return None, None, tried
