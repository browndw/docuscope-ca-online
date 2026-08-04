"""Tests for evidence-grounded interpretation assistant utilities."""

import pandas as pd
import polars as pl

from webapp.utilities.ai.interpretation import (
    build_interpretation_evidence,
    build_interpretation_prompt,
    get_teaching_lens_label,
    get_teaching_lens_options,
    run_interpretation_service,
)


class FakeInterpretationProvider:
    """Capture request params and return deterministic interpretation text."""

    def __init__(self, response="The table suggests a cautious pattern."):
        self.response = response
        self.request_params = None

    def generate_text(self, api_key, request_params, cache_key=None):
        self.request_params = request_params
        return self.response


def test_build_interpretation_evidence_summarizes_selected_table():
    """Evidence should include bounded preview, column names, and summaries."""
    df = pd.DataFrame({
        "Tag": ["Academic", "Academic", "Narrative", "Narrative", "Narrative"],
        "RF": [12.5, 9.5, 3.0, 2.5, 2.0],
        "Category": ["A", "A", "B", "B", "B"],
    })

    evidence = build_interpretation_evidence(df, table_name="Tag Frequencies")

    assert evidence.table_name == "Tag Frequencies"
    assert evidence.row_count == 5
    assert evidence.columns == ["Tag", "RF", "Category"]
    assert evidence.preview_rows[0]["Tag"] == "Academic"
    assert evidence.numeric_summaries["RF"]["count"] == 5
    assert evidence.numeric_summaries["RF"]["max"] == 12.5
    assert evidence.categorical_top_values["Tag"][0] == {
        "value": "Narrative",
        "count": 3,
    }
    assert any("RF is relative frequency" in rule for rule in evidence.measurement_rules)
    assert any("DocuScope tags" in rule for rule in evidence.rhetorical_rules)
    assert evidence.deterministic_note is not None
    assert "Coaching guardrail" in evidence.deterministic_note
    assert "Use RF, not AF" in evidence.deterministic_note
    assert "rhetorical patterns" in evidence.deterministic_note


def test_frequency_table_rules_warn_against_af_first_comparison():
    """AF/RF tables should teach normalized comparison and rhetorical caution."""
    evidence = build_interpretation_evidence(
        pd.DataFrame({
            "Tag": ["AcademicTerms", "InformationExposition"],
            "AF": [250, 200],
            "RF": [125.0, 100.0],
        }),
        table_name="Tag Frequencies",
    )
    prompt = build_interpretation_prompt(
        "Which tags are most common?",
        evidence,
        teaching_lens="find_pattern",
    )

    assert "AF is absolute frequency" in prompt
    assert "RF is relative frequency" in prompt
    assert "do not recommend AF as the first comparison basis" in prompt
    assert "High frequency means frequent" in prompt
    assert "communicative patterns" in prompt
    assert "model accuracy" in prompt
    assert "Table-specific coaching observations" in prompt
    assert "row-level RF rankings" in prompt
    assert "do not foreground the mean of AF or RF" in prompt


def test_docuscope_untagged_rows_are_filtered_from_evidence():
    """Interpretation evidence should match DocuScope tools by excluding Untagged."""
    evidence = build_interpretation_evidence(
        pd.DataFrame({
            "Tag": ["Untagged", "AcademicTerms", "InformationExposition"],
            "AF": [125542, 34830, 22100],
            "RF": [9.8, 2.7, 1.9],
        }),
        table_name="Tag Frequencies",
    )
    prompt = build_interpretation_prompt("What stands out?", evidence)

    assert evidence.row_count == 2
    assert all(row["Tag"] != "Untagged" for row in evidence.preview_rows)
    assert evidence.numeric_summaries["AF"]["max"] == 34830
    assert "Untagged" not in prompt


def test_build_interpretation_evidence_accepts_polars_dataframe():
    """The interpretation service should work with Streamlit Polars tables."""
    df = pl.DataFrame({"Tag": ["A", "B"], "RF": [1.0, 2.0]})

    evidence = build_interpretation_evidence(df, table_name="Selected table")

    assert evidence.row_count == 2
    assert evidence.columns == ["Tag", "RF"]
    assert evidence.numeric_summaries["RF"]["mean"] == 1.5


def test_interpretation_prompt_forbids_hidden_dataframe_work_and_coaches_reasoning():
    """The prompt contract should constrain the model to supplied evidence."""
    evidence = build_interpretation_evidence(
        pd.DataFrame({"Tag": ["A"], "RF": [1.0]}),
        table_name="Tag Frequencies",
    )

    prompt = build_interpretation_prompt(
        "What stands out?",
        evidence,
        teaching_lens="think_beyond_average",
    )

    assert "coach quantitative reasoning" in prompt
    assert "Think beyond the average" in prompt
    assert "Check Variation" in prompt
    assert "Use only the evidence supplied below" in prompt
    assert "Do not write Python code" in prompt
    assert "Do not ask to inspect the full dataframe" in prompt
    assert "Numeric summaries" in prompt


def test_teaching_lens_options_expose_guided_prompting_moves():
    """The UI should be able to model good prompts through fixed lenses."""
    options = get_teaching_lens_options()

    assert options["understand_table"] == "Understand what this table shows"
    assert options["check_claim"] == "Check whether my claim is supported"
    assert options["think_beyond_average"] == "Think beyond the average"
    assert get_teaching_lens_label("missing") == "Understand what this table shows"


def test_run_interpretation_service_uses_provider_with_compact_evidence():
    """Provider calls should receive compact evidence rather than a dataframe agent."""
    df = pd.DataFrame({"Tag": ["A", "B"], "RF": [1.0, 2.0]})
    provider = FakeInterpretationProvider()

    result = run_interpretation_service(
        df=df,
        user_prompt="Explain the main difference.",
        table_name="Tag Frequencies",
        teaching_lens="check_claim",
        api_key="test-key",
        llm_params={"model": "test-model", "max_tokens": 300},
        chat_provider=provider,
    )

    assert result.success
    assert result.response is not None
    assert "Coaching guardrail" in result.response
    assert "The table suggests a cautious pattern." in result.response
    assert result.evidence is not None
    assert provider.request_params is not None
    assert provider.request_params["model"] == "test-model"
    assert provider.request_params["stream"] is True
    prompt = provider.request_params["messages"][1]["content"]
    assert "Explain the main difference." in prompt
    assert "Check whether my claim is supported" in prompt
    assert "A More Cautious Revision" in prompt
    assert "Preview rows" in prompt
    assert "Do not ask to inspect the full dataframe" in prompt


def test_run_interpretation_service_handles_empty_table():
    """Empty tables should fail locally without a provider call."""
    provider = FakeInterpretationProvider()

    result = run_interpretation_service(
        df=pd.DataFrame({"Tag": [], "RF": []}),
        user_prompt="What stands out?",
        table_name="Empty table",
        chat_provider=provider,
    )

    assert not result.success
    assert result.error == "The selected table has no rows to interpret."
    assert provider.request_params is None
