"""Tests for the Plotbot smoke benchmark harness."""

from webapp.utilities.ai.plotbot_smoke import make_plotbot_smoke_dataframe, run_plotbot_smoke_prompts


class FakeSmokeProvider:
    """Return deterministic Plotly code for smoke prompts."""

    def generate_text(self, api_key, request_params, cache_key=None):
        return "fig = px.bar(df, x='tag', y='frequency', color='group')"


class FakeErrorProvider:
    """Return a normalized provider error."""

    def generate_text(self, api_key, request_params, cache_key=None):
        return {"type": "error", "value": "provider unavailable"}


def test_plotbot_smoke_harness_validates_generated_code():
    results = run_plotbot_smoke_prompts(
        prompts=["Make a bar chart of frequency by tag."],
        provider=FakeSmokeProvider(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.code_only is True
    assert result.no_imports is True
    assert result.no_show_call is True
    assert result.creates_fig is True
    assert result.execution_type == "plot"


def test_plotbot_smoke_dataframe_includes_live_tag_frequency_columns():
    df = make_plotbot_smoke_dataframe()

    assert {"Tag", "RF", "tag", "frequency"}.issubset(df.columns)


def test_plotbot_smoke_harness_records_provider_errors():
    results = run_plotbot_smoke_prompts(
        prompts=["Make a bar chart of frequency by tag."],
        provider=FakeErrorProvider(),
    )

    assert len(results) == 1
    result = results[0]
    assert result.generated_code is None
    assert result.creates_fig is False
    assert result.execution_type == "error"
    assert result.error == "provider unavailable"