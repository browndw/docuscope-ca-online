"""Evidence-grounded interpretation assistant utilities.

This module avoids dataframe-agent execution. Application code builds compact,
deterministic evidence from a selected table, then asks a chat provider to explain
only that evidence in cautious student-facing language.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import polars as pl

from webapp.utilities.ai.providers import ChatCompletionProvider, get_default_chat_provider
from webapp.utilities.ai.shared import LLM_MODEL


DEFAULT_INTERPRETATION_MODEL = "qwen2.5:7b-instruct"
DEFAULT_TEACHING_LENS = "understand_table"

TEACHING_LENSES: dict[str, dict[str, str]] = {
    "understand_table": {
        "label": "Understand what this table shows",
        "instruction": (
            "Help the student understand what the selected table measures. "
            "Explain useful columns or values, what the table can support, "
            "and what it cannot show by itself."
        ),
        "sections": (
            "What This Table Can Help You Notice; What To Be Careful About; "
            "Questions To Ask Yourself; A Useful Next Step"
        ),
    },
    "find_pattern": {
        "label": "Find a pattern worth investigating",
        "instruction": (
            "Coach the student to identify a pattern that deserves follow-up. "
            "Do not present the pattern as a final interpretation; frame it "
            "as something to test with additional quantitative or context evidence."
        ),
        "sections": (
            "Pattern To Investigate; Why It Might Matter; What Could Complicate It; "
            "Questions To Ask Yourself; A Useful Next Step"
        ),
    },
    "think_beyond_average": {
        "label": "Think beyond the average",
        "instruction": (
            "Coach the student to reason beyond means, ranks, or top-line values. "
            "Emphasize spread, outliers, median, document-level concentration, "
            "sample size, and whether a few cases may drive the pattern."
        ),
        "sections": (
            "Start With The Average; Check Variation; Possible Pitfalls; "
            "Questions To Ask Yourself; A Better Quantitative Check"
        ),
    },
    "check_claim": {
        "label": "Check whether my claim is supported",
        "instruction": (
            "Evaluate the student's proposed claim against the supplied evidence. "
            "Separate what the table supports, what it does not support, and what "
            "would require additional context or quantitative checks."
        ),
        "sections": (
            "What Your Claim Uses From The Table; What The Table Does Not Yet Show; "
            "Quantitative Check; Context Check; A More Cautious Revision"
        ),
    },
    "choose_next_step": {
        "label": "Choose a next analysis",
        "instruction": (
            "Suggest a responsible next analysis step the student could take. "
            "Prefer concrete corpus-analysis actions such as KWIC, collocations, "
            "group comparison, plotting, or checking document-level distribution."
        ),
        "sections": (
            "What You Have So Far; What You Still Need; Recommended Next Analysis; "
            "Why That Step Helps; Question To Take Forward"
        ),
    },
    "draft_cautious_claim": {
        "label": "Draft cautious wording",
        "instruction": (
            "Help the student phrase a cautious evidence-based claim without doing "
            "the interpretation for them. Make uncertainty and limits explicit."
        ),
        "sections": (
            "What The Evidence Can Support; What To Avoid Overstating; "
            "Possible Cautious Wording; How To Strengthen The Claim"
        ),
    },
}


@dataclass(frozen=True)
class InterpretationEvidence:
    """Compact evidence supplied to the interpretation model."""

    table_name: str
    row_count: int
    columns: list[str]
    preview_rows: list[dict[str, Any]] = field(default_factory=list)
    numeric_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    categorical_top_values: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    measurement_rules: list[str] = field(default_factory=list)
    rhetorical_rules: list[str] = field(default_factory=list)
    coaching_observations: list[str] = field(default_factory=list)
    deterministic_note: str | None = None


@dataclass(frozen=True)
class InterpretationServiceResult:
    """Result from one interpretation request."""

    response: str | None
    evidence: InterpretationEvidence | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Return True when the provider produced interpretation text."""
        return self.error is None and bool(self.response)


def get_teaching_lens_options() -> dict[str, str]:
    """Return supported teaching-lens keys and display labels."""
    return {key: lens["label"] for key, lens in TEACHING_LENSES.items()}


def get_teaching_lens_label(teaching_lens: str) -> str:
    """Return the display label for a teaching lens."""
    return TEACHING_LENSES.get(
        teaching_lens,
        TEACHING_LENSES[DEFAULT_TEACHING_LENS],
    )["label"]


def _to_pandas_dataframe(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    """Convert supported dataframe inputs to pandas for compact summarization."""
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    return pd.DataFrame(df)


def _json_safe_value(value: Any) -> Any:
    """Convert dataframe scalar values to prompt-safe primitives."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _infer_measurement_rules(columns: list[str]) -> list[str]:
    """Infer deterministic quantitative rules from known table columns."""
    normalized_columns = {column.lower(): column for column in columns}
    rules: list[str] = []

    if "af" in normalized_columns:
        rules.append(
            "AF is absolute frequency, a raw count. It is sensitive to corpus, "
            "group, or document-set size and should not be the main basis for "
            "comparison unless sizes are known to be comparable."
        )
    if "rf" in normalized_columns:
        rules.append(
            "RF is relative frequency, a normalized frequency measure. Prefer RF "
            "over AF when comparing how frequent tags or words are across unequal "
            "corpora, groups, or document sets."
        )
    if "af" in normalized_columns or "rf" in normalized_columns:
        rules.extend([
            "High frequency means frequent in this table; it does not by itself mean "
            "important, crucial, distinctive, or rhetorically meaningful.",
            "Frequency tables do not show dispersion across documents by themselves. "
            "A pattern may be driven by many documents or by a few repeated cases.",
            "Before making an interpretive claim, check context with KWIC examples, "
            "group comparisons, or document-level distribution when available.",
        ])

    return rules


def _infer_rhetorical_rules(columns: list[str]) -> list[str]:
    """Infer DocuScope-specific rhetorical rules from known table columns."""
    normalized_columns = {column.lower() for column in columns}
    if "tag" not in normalized_columns:
        return []

    return [
        "DocuScope tags are rhetorical or semantic categories used to investigate "
        "communicative patterns in texts.",
        "Discuss tags as possible indicators of communicative function, not as "
        "automatic conclusions about meaning, author intention, or text quality.",
        "Do not discuss model accuracy, classifier errors, training labels, or "
        "prediction confidence unless the student explicitly asks about the model.",
        "When a tag has a high RF value, frame it as a pattern in language use worth "
        "checking in context, not as proof that the tag is important or unique.",
        "Encourage the student to ask what communicative work the language may be "
        "doing: defining, explaining, evaluating, organizing information, narrating, "
        "expressing stance, or positioning readers."
    ]


def _build_deterministic_note(
    measurement_rules: list[str],
    rhetorical_rules: list[str],
) -> str | None:
    """Build a student-visible measurement and rhetorical frame."""
    notes: list[str] = []
    if measurement_rules:
        notes.append(
            "Use RF, not AF, as the first comparison measure when corpus or group "
            "sizes may differ; high frequency is only a clue, not evidence of "
            "importance by itself."
        )
    if rhetorical_rules:
        notes.append(
            "Treat DocuScope tags as rhetorical patterns to investigate in context, "
            "not as automatic conclusions about meaning or text quality."
        )

    return "Coaching guardrail: " + " ".join(notes) if notes else None


def _build_coaching_observations(bounded_df: pd.DataFrame) -> list[str]:
    """Build row-specific observations that keep coaching tied to the table."""
    column_lookup = {str(column).lower(): column for column in bounded_df.columns}
    tag_column = column_lookup.get("tag")
    af_column = column_lookup.get("af")
    rf_column = column_lookup.get("rf")
    observations: list[str] = []

    if tag_column is not None and rf_column is not None:
        ranked_rows = (
            bounded_df[[tag_column, rf_column]]
            .dropna(subset=[rf_column])
            .sort_values(by=rf_column, ascending=False)
            .head(3)
        )
        top_tags = [
            f"{_json_safe_value(row[tag_column])} (RF={_json_safe_value(row[rf_column])})"
            for _, row in ranked_rows.iterrows()
        ]
        if top_tags:
            observations.append(
                "The strongest row-level starting point in this preview is RF: "
                f"{', '.join(top_tags)}. Coach from these normalized values before AF."
            )

    if tag_column is not None and af_column is not None and rf_column is not None:
        observations.append(
            "Do not foreground the mean of AF or RF for this table. For tag-frequency "
            "tables, row-level RF rankings and context checks are usually more useful "
            "for students than averages across unlike rhetorical categories."
        )

    if observations:
        observations.append(
            "Useful follow-up questions should point toward KWIC/context examples, "
            "group comparison, plotting RF, or document-level distribution rather than "
            "asking only which AF values are highest."
        )

    return observations


def _filter_untagged_docuscope_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove DocuScope Untagged rows from interpretation evidence."""
    column_lookup = {str(column).lower(): column for column in df.columns}
    tag_column = column_lookup.get("tag")
    if tag_column is None:
        return df

    return df[df[tag_column].astype(str) != "Untagged"].copy()


def build_interpretation_evidence(
    df: pd.DataFrame | pl.DataFrame,
    table_name: str,
    max_preview_rows: int = 8,
    max_columns: int = 10,
    max_categories: int = 5,
) -> InterpretationEvidence:
    """Build bounded deterministic evidence from the selected table."""
    pandas_df = _to_pandas_dataframe(df)
    evidence_df = _filter_untagged_docuscope_rows(pandas_df)
    selected_columns = [str(column) for column in pandas_df.columns[:max_columns]]
    bounded_df = evidence_df.loc[:, evidence_df.columns[:max_columns]]
    measurement_rules = _infer_measurement_rules(selected_columns)
    rhetorical_rules = _infer_rhetorical_rules(selected_columns)
    coaching_observations = _build_coaching_observations(bounded_df)

    preview_rows = [
        {str(key): _json_safe_value(value) for key, value in row.items()}
        for row in bounded_df.head(max_preview_rows).to_dict(orient="records")
    ]

    numeric_summaries: dict[str, dict[str, Any]] = {}
    categorical_top_values: dict[str, list[dict[str, Any]]] = {}
    for column in bounded_df.columns:
        series = bounded_df[column].dropna()
        column_name = str(column)
        if pd.api.types.is_numeric_dtype(series):
            numeric_summaries[column_name] = {
                "count": int(series.count()),
                "mean": _json_safe_value(series.mean()) if not series.empty else None,
                "median": _json_safe_value(series.median()) if not series.empty else None,
                "min": _json_safe_value(series.min()) if not series.empty else None,
                "max": _json_safe_value(series.max()) if not series.empty else None,
            }
        else:
            counts = series.astype(str).value_counts().head(max_categories)
            categorical_top_values[column_name] = [
                {"value": str(value), "count": int(count)}
                for value, count in counts.items()
            ]

    return InterpretationEvidence(
        table_name=table_name,
        row_count=len(evidence_df),
        columns=selected_columns,
        preview_rows=preview_rows,
        numeric_summaries=numeric_summaries,
        categorical_top_values=categorical_top_values,
        measurement_rules=measurement_rules,
        rhetorical_rules=rhetorical_rules,
        coaching_observations=coaching_observations,
        deterministic_note=_build_deterministic_note(
            measurement_rules,
            rhetorical_rules,
        ),
    )


def build_interpretation_prompt(
    user_prompt: str,
    evidence: InterpretationEvidence,
    teaching_lens: str = DEFAULT_TEACHING_LENS,
) -> str:
    """Build the grounded interpretation prompt from deterministic evidence."""
    lens = TEACHING_LENSES.get(teaching_lens, TEACHING_LENSES[DEFAULT_TEACHING_LENS])
    return (
        "You are helping a student interpret a corpus-analysis table.\n"
        "Your role is to coach quantitative reasoning, not to do the final "
        "interpretive work for the student.\n"
        "Use only the evidence supplied below. Do not invent rows, columns, "
        "statistics, p-values, causes, or corpus facts that are not present.\n"
        "Do not write Python code. Do not ask to inspect the full dataframe.\n"
        "If the user asks for something unsupported by the evidence, say what "
        "additional evidence would be needed.\n"
        "Use cautious language and distinguish observation from interpretation.\n\n"
        f"Teaching move: {lens['label']}\n"
        f"Teaching instruction: {lens['instruction']}\n"
        f"Use these response sections: {lens['sections']}\n\n"
        f"User question:\n{user_prompt}\n\n"
        f"Table name: {evidence.table_name}\n"
        f"Row count: {evidence.row_count}\n"
        f"Columns: {evidence.columns}\n"
        f"Preview rows: {evidence.preview_rows}\n"
        f"Numeric summaries: {evidence.numeric_summaries}\n"
        f"Categorical top values: {evidence.categorical_top_values}\n"
        f"Mandatory measurement rules: {evidence.measurement_rules}\n"
        f"Mandatory DocuScope rhetorical frame: {evidence.rhetorical_rules}\n\n"
        f"Table-specific coaching observations: {evidence.coaching_observations}\n\n"
        "You must follow the mandatory measurement and rhetorical rules. In particular, "
        "do not recommend AF as the first comparison basis when RF or another normalized "
        "measure is available; do not call frequent tags important or crucial unless "
        "the evidence supports that stronger claim. For DocuScope tag-frequency tables, "
        "do not foreground the mean of AF or RF unless the student explicitly asks about "
        "averages; use row-level RF patterns and context checks instead.\n\n"
        "Write concise coaching guidance using the requested sections. Include "
        "questions the student should ask themselves."
    )


def run_interpretation_service(
    df: pd.DataFrame | pl.DataFrame,
    user_prompt: str,
    table_name: str,
    teaching_lens: str = DEFAULT_TEACHING_LENS,
    api_key: str = "",
    llm_params: dict[str, Any] | None = None,
    chat_provider: ChatCompletionProvider | None = None,
) -> InterpretationServiceResult:
    """Generate an evidence-grounded interpretation for a selected table."""
    if df is None:
        return InterpretationServiceResult(response=None, error="No table is selected.")

    evidence = build_interpretation_evidence(df=df, table_name=table_name)
    if evidence.row_count == 0:
        return InterpretationServiceResult(
            response=None,
            evidence=evidence,
            error="The selected table has no rows to interpret.",
        )

    provider = chat_provider or get_default_chat_provider()
    params = dict(llm_params or {})
    request_params = {
        "model": params.get("model", LLM_MODEL),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You interpret deterministic corpus-analysis evidence. "
                    "You never compute hidden statistics or invent unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": build_interpretation_prompt(
                    user_prompt,
                    evidence,
                    teaching_lens=teaching_lens,
                ),
            },
        ],
        "temperature": params.get("temperature", 0.2),
        "max_tokens": params.get("max_tokens", 900),
        "stream": True,
    }

    provider_response = provider.generate_text(
        api_key=api_key,
        request_params=request_params,
        cache_key=None,
    )
    if isinstance(provider_response, dict) and provider_response.get("type") == "error":
        return InterpretationServiceResult(
            response=None,
            evidence=evidence,
            error=str(provider_response.get("value", "Interpretation provider failed.")),
        )

    response_text = str(provider_response).strip()
    if not response_text:
        return InterpretationServiceResult(
            response=None,
            evidence=evidence,
            error="Interpretation provider returned an empty response.",
        )

    if evidence.deterministic_note:
        response_text = f"{evidence.deterministic_note}\n\n{response_text}"

    return InterpretationServiceResult(response=response_text, evidence=evidence)
