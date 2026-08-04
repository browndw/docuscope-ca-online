"""Small Plotbot benchmark harness for local model evaluation."""

import json
import re
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from webapp.utilities.ai.plotbot import generate_plotbot_code_and_result
from webapp.utilities.ai.providers import ChatCompletionProvider


DEFAULT_LLM_PARAMS = {
    "temperature": 0.1,
    "max_tokens": 500,
    "top_p": 0.7,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@dataclass(frozen=True)
class PlotbotBenchmarkCase:
    """A single Plotbot prompt benchmark case."""

    name: str
    prompt: str
    plot_lib: str = "plotly.express"
    expected_columns: set[str] = field(default_factory=set)
    forbidden_columns: set[str] = field(default_factory=set)
    code_chunk: str | None = None


@dataclass(frozen=True)
class PlotbotBenchmarkResult:
    """Compact result for one Plotbot benchmark case."""

    name: str
    plot_lib: str
    success: bool
    result_type: str
    latency_seconds: float
    code_length: int
    has_import: bool
    creates_fig: bool
    uses_expected_columns: bool
    hallucinated_columns: list[str]
    error: str | None = None


DEFAULT_CASES = [
    PlotbotBenchmarkCase(
        name="basic_bar",
        prompt="Make a bar chart of value by category.",
        expected_columns={"category", "value"}
    ),
    PlotbotBenchmarkCase(
        name="axis_labels",
        prompt="Make a bar chart and label the x-axis Category and the y-axis Value.",
        expected_columns={"category", "value"}
    ),
    PlotbotBenchmarkCase(
        name="color_by_group",
        prompt="Make a scatter plot of value by score and color the points by group.",
        expected_columns={"score", "value", "group"}
    ),
    PlotbotBenchmarkCase(
        name="gridlines_title",
        prompt="Make a line chart with a clear title and visible gridlines.",
        expected_columns={"category", "value"}
    ),
    PlotbotBenchmarkCase(
        name="tag_rf_bar",
        prompt="Make a bar plot of RF vs Tag.",
        expected_columns={"Tag", "RF"}
    ),
    PlotbotBenchmarkCase(
        name="top_tags_by_rf",
        prompt="Show the top 3 tags by RF as a horizontal bar chart.",
        expected_columns={"Tag", "RF"}
    ),
    PlotbotBenchmarkCase(
        name="ignore_missing_column",
        prompt="Make a bar plot of RF vs ImaginaryColumn.",
        expected_columns={"RF"},
        forbidden_columns={"ImaginaryColumn"}
    ),
    PlotbotBenchmarkCase(
        name="refine_existing_plot",
        prompt="Update the previous plot with a clearer title and light gridlines.",
        expected_columns={"Tag", "RF"},
        code_chunk="fig = px.bar(df, x='Tag', y='RF')"
    ),
]


def sample_plotbot_dataframe() -> pd.DataFrame:
    """Return a small dataframe with categorical and numeric plotting options."""
    return pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "group": ["control", "control", "treatment", "treatment"],
        "value": [10, 14, 9, 18],
        "score": [0.2, 0.4, 0.7, 0.9],
        "Tag": ["AcademicTerms", "Reasoning", "Confidence", "Narrative"],
        "RF": [4.8, 3.4, 1.8, 2.6],
    })


def _referenced_dataframe_columns(code: str) -> set[str]:
    """Return likely dataframe column names referenced in generated code."""
    columns: set[str] = set()
    patterns = [
        r"df\[['\"]([^'\"]+)['\"]\]",
        (
            r"\b(?:x|y|color|labels|names|values|hover_name|facet_col|facet_row)"
            r"\s*=\s*['\"]([^'\"]+)['\"]"
        ),
    ]
    for pattern in patterns:
        columns.update(re.findall(pattern, code))
    return columns


def run_plotbot_benchmark(
    api_key: str = "",
    cases: list[PlotbotBenchmarkCase] | None = None,
    df: pd.DataFrame | None = None,
    llm_params: dict | None = None,
    chat_provider: ChatCompletionProvider | None = None,
) -> list[PlotbotBenchmarkResult]:
    """Run Plotbot prompts through the configured provider and execution path."""
    benchmark_cases = cases or DEFAULT_CASES
    data = df if df is not None else sample_plotbot_dataframe()
    params = llm_params or DEFAULT_LLM_PARAMS
    schema = data.dtypes.to_string()
    results = []

    for case in benchmark_cases:
        started = time.perf_counter()
        plot_code, plot_result = generate_plotbot_code_and_result(
            df=data,
            plot_lib=case.plot_lib,
            user_input=case.prompt,
            api_key=api_key,
            llm_params=params,
            schema=schema,
            code_chunk=case.code_chunk,
            chat_provider=chat_provider
        )
        latency = time.perf_counter() - started

        code = plot_code if isinstance(plot_code, str) else ""
        result_type = (
            plot_result.get("type", "error")
            if isinstance(plot_result, dict)
            else "error"
        )
        error = plot_result.get("value") if result_type == "error" else None
        referenced_columns = _referenced_dataframe_columns(code)
        available_columns = set(data.columns)
        hallucinated_columns = sorted(
            (referenced_columns - available_columns)
            | (referenced_columns & case.forbidden_columns)
        )
        uses_expected_columns = (
            not case.expected_columns or case.expected_columns.issubset(referenced_columns)
        )
        success = (
            result_type == "plot" and
            "fig" in code and
            uses_expected_columns and
            not hallucinated_columns
        )
        results.append(PlotbotBenchmarkResult(
            name=case.name,
            plot_lib=case.plot_lib,
            success=success,
            result_type=result_type,
            latency_seconds=round(latency, 4),
            code_length=len(code),
            has_import="import " in code,
            creates_fig="fig" in code,
            uses_expected_columns=uses_expected_columns,
            hallucinated_columns=hallucinated_columns,
            error=str(error) if error else None,
        ))

    return results


def main() -> int:
    """Run the benchmark using environment-selected provider settings."""
    results = run_plotbot_benchmark()
    payload = [asdict(result) for result in results]
    print(json.dumps(payload, indent=2))
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())