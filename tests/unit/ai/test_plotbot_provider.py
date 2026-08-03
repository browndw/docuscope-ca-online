"""Tests for the Plotbot model-provider boundary."""

import pandas as pd
import webapp.utilities.ai.providers as provider_module

from webapp.utilities.ai.providers import (
    OpenAICompatibleChatProvider,
    ProtectedOpenAIChatProvider,
    check_openai_compatible_provider_health,
    get_openai_compatible_provider_config,
    get_default_chat_provider,
)
from webapp.utilities.ai.plotbot import plotbot_code_generate_or_update
from webapp.utilities.ai.plotbot import (
    generate_plotbot_code_and_result,
    run_plotbot_serialized_service,
    run_plotbot_service,
)


class FakeChatProvider:
    """Capture Plotbot request params and return deterministic code."""

    def __init__(self):
        self.request_params = None
        self.cache_key = None

    def generate_text(self, api_key, request_params, cache_key=None):
        self.request_params = request_params
        self.cache_key = cache_key
        return "```python\nfig = px.bar(df, x='label', y='value')\nfig.show()\n```"


def test_plotbot_code_generation_uses_injected_provider(monkeypatch):
    """Plotbot should generate through a provider without requiring OpenAI."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    df = pd.DataFrame({"label": ["a", "b"], "value": [1, 2]})
    provider = FakeChatProvider()

    code = plotbot_code_generate_or_update(
        df=df,
        user_request="Make a bar chart of value by label.",
        plot_lib="plotly.express",
        schema=df.dtypes.to_string(),
        api_key="test-key",
        llm_params={
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        chat_provider=provider
    )

    assert provider.request_params is not None
    assert provider.request_params["stream"] is True
    assert provider.request_params["messages"][1]["content"]
    prompt = provider.request_params["messages"][1]["content"]
    assert "fig.update_xaxes(showgrid=True)" in prompt
    assert "Preserve any valid columns named by the user" in prompt
    assert "choose a compatible replacement column near it" in prompt
    assert "Comments should help a student understand the plotting grammar" in prompt
    assert provider.cache_key.startswith("plotbot_code_")
    assert "```" not in code
    assert "fig.show()" not in code
    assert "fig = px.bar" in code


def test_plotbot_update_prompt_preserves_existing_mappings(monkeypatch):
    """Refinement prompts should tell providers to minimally edit current code."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    df = pd.DataFrame({"Tag": ["a", "b"], "RF": [1.0, 2.0]})
    provider = FakeChatProvider()

    plotbot_code_generate_or_update(
        df=df,
        user_request="Add a clearer title and light gridlines.",
        plot_lib="plotly.express",
        schema=df.dtypes.to_string(),
        api_key="test-key",
        llm_params={
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        code_chunk="fig = px.bar(df, x='Tag', y='RF')",
        chat_provider=provider
    )

    prompt = provider.request_params["messages"][1]["content"]
    assert "Make the minimum necessary edits" in prompt
    assert "style-only requests" in prompt
    assert "Preserve the current code's existing x, y, color" in prompt


def test_openai_compatible_provider_uses_base_url_and_model(monkeypatch):
    """Local model providers should use OpenAI-compatible client settings."""
    captured = {}

    class FakeDelta:
        content = "fig = px.bar(df, x='label', y='value')"

    class FakeChoice:
        delta = FakeDelta()

    class FakeChunk:
        choices = [FakeChoice()]

    class FakeCompletions:
        def create(self, **params):
            captured["params"] = params
            return [FakeChunk()]

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAIClient:
        chat = FakeChat()

    def fake_openai(api_key, base_url):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return FakeOpenAIClient()

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    provider = OpenAICompatibleChatProvider(
        base_url="http://localhost:11434/v1",
        model="Qwen2.5-Coder-7B-Instruct"
    )

    text = provider.generate_text(
        api_key="",
        request_params={
            "model": "ignored-default",
            "messages": [{"role": "user", "content": "plot"}],
            "stream": True,
        }
    )

    assert text == "fig = px.bar(df, x='label', y='value')"
    assert captured["api_key"] == "local-model"
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["params"]["model"] == "Qwen2.5-Coder-7B-Instruct"


def test_default_provider_selects_local_endpoint_from_environment(monkeypatch):
    """Environment config should enable local model testing without page changes."""
    provider_module._discover_local_openai_compatible_model.cache_clear()
    monkeypatch.setenv("DOCUSCOPE_AI_PROVIDER", "local")
    monkeypatch.setenv("DOCUSCOPE_AI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("DOCUSCOPE_AI_MODEL", "Qwen2.5-Coder-7B-Instruct")

    provider = get_default_chat_provider()

    assert isinstance(provider, OpenAICompatibleChatProvider)
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model == "Qwen2.5-Coder-7B-Instruct"


def test_default_provider_auto_discovers_local_ollama_endpoint(monkeypatch):
    """A running local OpenAI-compatible endpoint should be usable without env vars."""
    provider_module._discover_local_openai_compatible_model.cache_clear()
    monkeypatch.delenv("DOCUSCOPE_AI_PROVIDER", raising=False)
    monkeypatch.delenv("DOCUSCOPE_AI_BASE_URL", raising=False)
    monkeypatch.delenv("DOCUSCOPE_AI_MODEL", raising=False)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data": [{"id": "qwen2.5-coder:7b-instruct"}]}'

    monkeypatch.setattr(provider_module, "urlopen", lambda url, timeout: FakeResponse())

    provider_config = get_openai_compatible_provider_config()
    provider = get_default_chat_provider()

    assert provider_config == (
        "http://localhost:11434/v1",
        "qwen2.5-coder:7b-instruct"
    )
    assert isinstance(provider, OpenAICompatibleChatProvider)
    assert provider.base_url == "http://localhost:11434/v1"
    assert provider.model == "qwen2.5-coder:7b-instruct"


def test_default_provider_falls_back_to_protected_openai(monkeypatch):
    """OpenAI fallback should be preserved when local auto-discovery is disabled."""
    provider_module._discover_local_openai_compatible_model.cache_clear()
    monkeypatch.delenv("DOCUSCOPE_AI_PROVIDER", raising=False)
    monkeypatch.delenv("DOCUSCOPE_AI_BASE_URL", raising=False)
    monkeypatch.delenv("DOCUSCOPE_AI_MODEL", raising=False)
    monkeypatch.setenv("DOCUSCOPE_AI_AUTO_DISCOVER_LOCAL", "0")

    provider = get_default_chat_provider()

    assert isinstance(provider, ProtectedOpenAIChatProvider)
    provider_module._discover_local_openai_compatible_model.cache_clear()


def test_openai_compatible_provider_health_reports_ready_model(monkeypatch):
    """Health check should verify reachable endpoint and configured model."""
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data": [{"id": "qwen2.5-coder:7b-instruct"}]}'

    monkeypatch.setattr(provider_module, "urlopen", lambda url, timeout: FakeResponse())

    health = check_openai_compatible_provider_health(
        base_url="http://qwen-model:8000/v1",
        model="qwen2.5-coder:7b-instruct",
    )

    assert health.available is True
    assert health.base_url == "http://qwen-model:8000/v1"
    assert health.resolved_model == "qwen2.5-coder:7b-instruct"
    assert health.error is None


def test_openai_compatible_provider_health_reports_missing_model(monkeypatch):
    """Health check should fail when the configured model is not exposed."""
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"data": [{"id": "other-model"}]}'

    monkeypatch.setattr(provider_module, "urlopen", lambda url, timeout: FakeResponse())

    health = check_openai_compatible_provider_health(
        base_url="http://qwen-model:8000/v1",
        model="qwen2.5-coder:7b-instruct",
    )

    assert health.available is False
    assert health.model_ids == ["other-model"]
    assert "Configured model is not exposed" in health.error


def test_generate_plotbot_code_and_result_reuses_cached_code(monkeypatch):
    """Reusable Plotbot helper should execute cached code without a model call."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    df = pd.DataFrame({"label": ["a", "b"], "value": [1, 2]})
    plot_code, plot_result = generate_plotbot_code_and_result(
        df=df,
        plot_lib="plotly.express",
        user_input="Make a bar chart of value by label.",
        api_key="test-key",
        llm_params={
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        cached_code="fig = px.bar(df, x='label', y='value')"
    )

    assert plot_code == "fig = px.bar(df, x='label', y='value')"
    assert plot_result["type"] == "plot"
    assert hasattr(plot_result["value"], "to_image")


def test_run_plotbot_service_returns_normalized_result_from_cached_code(monkeypatch):
    """Service boundary should expose normalized status without Streamlit state."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    df = pd.DataFrame({"label": ["a", "b"], "value": [1, 2]})
    service_result = run_plotbot_service(
        df=df,
        plot_lib="plotly.express",
        user_input="Make a bar chart of value by label.",
        api_key="test-key",
        llm_params={
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        cached_code="fig = px.bar(df, x='label', y='value')"
    )

    assert service_result.code == "fig = px.bar(df, x='label', y='value')"
    assert service_result.success is True
    assert service_result.used_cached_code is True
    assert service_result.result["type"] == "plot"


def test_run_plotbot_serialized_service_returns_svg_payload(monkeypatch):
    """Queued Plotbot service should return JSON-friendly plot output."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    df = pd.DataFrame({"label": ["a", "b"], "value": [1, 2]})
    service_result = run_plotbot_serialized_service(
        df=df,
        plot_lib="plotly.express",
        user_input="Make a bar chart of value by label.",
        api_key="test-key",
        llm_params={
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.7,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        },
        cached_code="fig = px.bar(df, x='label', y='value')"
    )

    assert service_result.success is True
    assert service_result.result_type == "plot"
    assert service_result.used_cached_code is True
    assert service_result.code == "fig = px.bar(df, x='label', y='value')"
    assert service_result.plot_svg is not None
    assert "<svg" in service_result.plot_svg
