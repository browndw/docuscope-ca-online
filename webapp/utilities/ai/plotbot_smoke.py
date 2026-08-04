"""Smoke benchmark helpers for Plotbot model providers.

This module exercises the provider-backed Plotbot path with a small set of
plotting-grammar prompts. It is intentionally usable with either a fake provider
in tests or the default provider selected by environment variables for local
model evaluation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd

from webapp.utilities.ai import plotbot as plotbot_module
from webapp.utilities.ai.providers import ChatCompletionProvider, get_default_chat_provider


DEFAULT_PLOTBOT_SMOKE_PROMPTS = [
    "Make a bar chart of frequency by tag.",
    "Make a bar plot of RF vs Tag.",
    "Label the x-axis as Tag and the y-axis as Frequency.",
    "Add light gridlines to make the values easier to compare.",
    "Color the bars by group.",
    "Make the title clearer for a student reader.",
]


@dataclass(frozen=True)
class PlotbotSmokeResult:
    """Result for one Plotbot smoke prompt."""

    prompt: str
    generated_code: str | None
    code_only: bool
    no_imports: bool
    no_show_call: bool
    creates_fig: bool
    execution_type: str
    error: str | None = None


def make_plotbot_smoke_dataframe() -> pd.DataFrame:
    """Return a small dataframe with numeric and categorical plotting options."""
    return pd.DataFrame({
        "tag": ["AcademicTerms", "Reasoning", "Confidence", "Narrative"],
        "frequency": [24, 17, 9, 13],
        "relative_frequency": [4.8, 3.4, 1.8, 2.6],
        "group": ["A", "A", "B", "B"],
        "Tag": ["AcademicTerms", "Reasoning", "Confidence", "Narrative"],
        "RF": [4.8, 3.4, 1.8, 2.6],
    })


def _is_code_only(code: str) -> bool:
    """Detect obvious markdown or prose wrappers around generated code."""
    stripped = code.strip()
    if "```" in stripped:
        return False
    prose_markers = [
        "here is",
        "this code",
        "the following",
        "you can use",
    ]
    return not any(marker in stripped.lower() for marker in prose_markers)


def _has_imports(code: str) -> bool:
    return bool(re.search(r"^\s*(import|from)\s+", code, re.MULTILINE))


def _has_show_call(code: str) -> bool:
    return bool(re.search(r"\b(fig|plt)\.show\s*\(", code))


def run_plotbot_smoke_prompts(
    prompts: Iterable[str] = DEFAULT_PLOTBOT_SMOKE_PROMPTS,
    provider: ChatCompletionProvider | None = None,
    plot_lib: str = "plotly.express",
    api_key: str = "",
    llm_params: dict[str, Any] | None = None,
) -> list[PlotbotSmokeResult]:
    """Generate and execute Plotbot code for a fixed set of smoke prompts."""
    df = make_plotbot_smoke_dataframe()
    params = llm_params or {
        "temperature": 0.1,
        "max_tokens": 500,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    active_provider = provider or get_default_chat_provider()
    previous_code = None
    results: list[PlotbotSmokeResult] = []
    original_desktop = plotbot_module.DESKTOP
    plotbot_module.DESKTOP = True

    try:
        for prompt in prompts:
            generated = plotbot_module.plotbot_code_generate_or_update(
                df=df,
                user_request=prompt,
                plot_lib=plot_lib,
                schema=df.dtypes.to_string(),
                api_key=api_key,
                llm_params=params,
                code_chunk=previous_code,
                chat_provider=active_provider,
            )

            if isinstance(generated, dict):
                results.append(PlotbotSmokeResult(
                    prompt=prompt,
                    generated_code=None,
                    code_only=False,
                    no_imports=False,
                    no_show_call=False,
                    creates_fig=False,
                    execution_type="error",
                    error=generated.get("value", "Model provider returned an error."),
                ))
                previous_code = None
                continue

            code = generated or ""
            execution = plotbot_module.plotbot_code_execute(code, df=df, plot_lib=plot_lib)
            execution_type = (
                execution.get("type", "error")
                if isinstance(execution, dict)
                else "error"
            )
            creates_fig = execution_type == "plot"

            results.append(PlotbotSmokeResult(
                prompt=prompt,
                generated_code=code,
                code_only=_is_code_only(code),
                no_imports=not _has_imports(code),
                no_show_call=not _has_show_call(code),
                creates_fig=creates_fig,
                execution_type=execution_type,
                error=None if creates_fig else execution.get("value", "No plot generated."),
            ))
            previous_code = code if creates_fig else None
    finally:
        plotbot_module.DESKTOP = original_desktop

    return results


def main() -> None:
    """Run the Plotbot smoke benchmark against the configured provider."""
    parser = argparse.ArgumentParser(description="Run Plotbot provider smoke prompts.")
    parser.add_argument(
        "--api-key",
        default="",
        help="API key for the configured provider.",
    )
    parser.add_argument(
        "--plot-lib",
        default="plotly.express",
        help="Plotting library to request.",
    )
    args = parser.parse_args()

    results = run_plotbot_smoke_prompts(api_key=args.api_key, plot_lib=args.plot_lib)
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()