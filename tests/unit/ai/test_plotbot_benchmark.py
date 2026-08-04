"""Tests for the Plotbot benchmark harness."""

from webapp.utilities.ai.plotbot_benchmark import (
    PlotbotBenchmarkCase,
    run_plotbot_benchmark,
)


class FakeBenchmarkProvider:
    """Return simple executable Plotly code for benchmark prompts."""

    def generate_text(self, api_key, request_params, cache_key=None):
        return "fig = px.bar(df, x='category', y='value')"


class FakeHallucinatingProvider:
    """Return executable code that references a nonexistent column."""

    def generate_text(self, api_key, request_params, cache_key=None):
        return "fig = px.bar(df, x='ImaginaryColumn', y='value')"


def test_plotbot_benchmark_runs_with_fake_provider(monkeypatch):
    """Benchmark should run offline through the same Plotbot service path."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    results = run_plotbot_benchmark(
        api_key="test-key",
        cases=[
            PlotbotBenchmarkCase(
                name="basic_bar",
                prompt="Make a bar chart of value by category."
            )
        ],
        chat_provider=FakeBenchmarkProvider()
    )

    assert len(results) == 1
    assert results[0].name == "basic_bar"
    assert results[0].success is True
    assert results[0].result_type == "plot"
    assert results[0].has_import is False
    assert results[0].creates_fig is True
    assert results[0].uses_expected_columns is True
    assert results[0].hallucinated_columns == []


def test_plotbot_benchmark_flags_hallucinated_columns(monkeypatch):
    """Benchmark should fail cases that use unavailable dataframe columns."""
    monkeypatch.setattr("webapp.utilities.ai.plotbot.DESKTOP", True)

    results = run_plotbot_benchmark(
        api_key="test-key",
        cases=[
            PlotbotBenchmarkCase(
                name="missing_column",
                prompt="Make a chart with ImaginaryColumn.",
                expected_columns={"category", "value"},
                forbidden_columns={"ImaginaryColumn"}
            )
        ],
        chat_provider=FakeHallucinatingProvider()
    )

    assert len(results) == 1
    assert results[0].success is False
    assert results[0].uses_expected_columns is False
    assert results[0].hallucinated_columns == ["ImaginaryColumn"]
