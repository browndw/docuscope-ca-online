"""
Model provider boundaries for AI-assisted tools.

These classes keep assistant code independent from a specific model runner while
preserving the current protected OpenAI path as the default implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
from functools import lru_cache
import json
import os
from urllib.error import URLError
from urllib.request import urlopen

from webapp.utilities.ai.enterprise_integration import make_protected_openai_call


LOCAL_PROVIDER_NAMES = {"openai-compatible", "openai_compatible", "local"}
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_PLOTBOT_LOCAL_MODEL = "qwen2.5-coder:7b-instruct"


@dataclass(frozen=True)
class OpenAICompatibleProviderHealth:
    """Readiness result for an OpenAI-compatible model endpoint."""

    available: bool
    base_url: str
    configured_model: str | None = None
    resolved_model: str | None = None
    model_ids: list[str] = field(default_factory=list)
    error: str | None = None


def _collect_streamed_text(response: Any) -> str:
    """Collect text from an OpenAI-compatible streamed response."""
    full_response = ""
    for chunk in response:
        chunk_content = chunk.choices[0].delta.content
        if chunk_content:
            full_response += chunk_content
    return full_response


class ChatCompletionProvider(Protocol):
    """Minimal synchronous chat-completion provider contract."""

    def generate_text(
        self,
        api_key: str,
        request_params: dict[str, Any],
        cache_key: str | None = None
    ) -> str | dict[str, str]:
        """Return generated text or a normalized error dictionary."""


class ProtectedOpenAIChatProvider:
    """Chat provider backed by the existing protected OpenAI integration."""

    def generate_text(
        self,
        api_key: str,
        request_params: dict[str, Any],
        cache_key: str | None = None
    ) -> str | dict[str, str]:
        response = make_protected_openai_call(
            api_key=api_key,
            request_params=request_params,
            request_type="chat_completion",
            cache_key=cache_key
        )

        if isinstance(response, dict) and response.get("type") == "error":
            return response

        return _collect_streamed_text(response)


class OpenAICompatibleChatProvider:
    """Chat provider for local or hosted OpenAI-compatible endpoints."""

    def __init__(self, base_url: str, model: str | None = None):
        self.base_url = base_url
        self.model = model

    def generate_text(
        self,
        api_key: str,
        request_params: dict[str, Any],
        cache_key: str | None = None
    ) -> str | dict[str, str]:
        import openai

        try:
            params = dict(request_params)
            if self.model:
                params["model"] = self.model

            client = openai.OpenAI(
                api_key=api_key or "local-model",
                base_url=self.base_url
            )
            response = client.chat.completions.create(**params)
            return _collect_streamed_text(response)
        except Exception as exc:
            return {
                "type": "error",
                "value": f"Model provider request failed: {str(exc)}"
            }


def check_openai_compatible_provider_health(
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 1.0,
) -> OpenAICompatibleProviderHealth:
    """Check whether an OpenAI-compatible endpoint is reachable and model-ready."""
    resolved_base_url = (base_url or os.environ.get("DOCUSCOPE_AI_BASE_URL") or DEFAULT_LOCAL_BASE_URL).strip()
    configured_model = (model or os.environ.get("DOCUSCOPE_AI_MODEL") or "").strip() or None
    if not resolved_base_url:
        return OpenAICompatibleProviderHealth(
            available=False,
            base_url="",
            configured_model=configured_model,
            error="No OpenAI-compatible base URL is configured.",
        )

    models_url = f"{resolved_base_url.rstrip('/')}/models"
    try:
        with urlopen(models_url, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return OpenAICompatibleProviderHealth(
            available=False,
            base_url=resolved_base_url,
            configured_model=configured_model,
            error=f"Model endpoint health check failed: {str(exc)}",
        )

    models = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [
        str(model_payload.get("id", ""))
        for model_payload in models
        if isinstance(model_payload, dict) and model_payload.get("id")
    ]
    if configured_model and configured_model not in model_ids:
        return OpenAICompatibleProviderHealth(
            available=False,
            base_url=resolved_base_url,
            configured_model=configured_model,
            model_ids=model_ids,
            error=f"Configured model is not exposed by the endpoint: {configured_model}",
        )

    resolved_model = configured_model or (model_ids[0] if model_ids else None)
    return OpenAICompatibleProviderHealth(
        available=bool(resolved_model or not configured_model),
        base_url=resolved_base_url,
        configured_model=configured_model,
        resolved_model=resolved_model,
        model_ids=model_ids,
    )


@lru_cache(maxsize=1)
def _discover_local_openai_compatible_model() -> tuple[str, str | None] | None:
    """Return a reachable local OpenAI-compatible endpoint, if one is running."""
    auto_discover = os.environ.get("DOCUSCOPE_AI_AUTO_DISCOVER_LOCAL", "1").strip().lower()
    if auto_discover in {"0", "false", "no", "off"}:
        return None

    base_url = os.environ.get("DOCUSCOPE_AI_BASE_URL", DEFAULT_LOCAL_BASE_URL).strip()
    if not base_url:
        return None

    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urlopen(models_url, timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return None

    models = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [
        str(model.get("id", ""))
        for model in models
        if isinstance(model, dict) and model.get("id")
    ]
    configured_model = os.environ.get("DOCUSCOPE_AI_MODEL", "").strip()
    preferred_models = [
        configured_model,
        DEFAULT_PLOTBOT_LOCAL_MODEL,
        "Qwen2.5-Coder-7B-Instruct",
    ]
    for model in preferred_models:
        if model and model in model_ids:
            return base_url, model

    if model_ids:
        return base_url, model_ids[0]
    return base_url, configured_model or None


def get_openai_compatible_provider_config() -> tuple[str, str | None] | None:
    """Return configured or auto-discovered OpenAI-compatible provider settings."""
    provider = os.environ.get("DOCUSCOPE_AI_PROVIDER", "").strip().lower()
    base_url = os.environ.get("DOCUSCOPE_AI_BASE_URL", "").strip()
    model = os.environ.get("DOCUSCOPE_AI_MODEL", "").strip() or None
    if provider in LOCAL_PROVIDER_NAMES and base_url:
        return base_url, model

    if provider in {"", "auto", "local"}:
        return _discover_local_openai_compatible_model()

    return None


def get_default_chat_provider() -> ChatCompletionProvider:
    """Return the default chat provider for synchronous assistant calls."""
    provider_config = get_openai_compatible_provider_config()
    if provider_config is not None:
        base_url, model = provider_config
        return OpenAICompatibleChatProvider(base_url=base_url, model=model)

    return ProtectedOpenAIChatProvider()