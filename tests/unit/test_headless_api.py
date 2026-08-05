"""Unit tests for headless API metric selection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest

from docuscope_ca import CorpusProcessingError
from docuscope_ca import api


def _documents() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "doc_id": ["1"],
            "doc_name": ["doc_1"],
            "text": ["Example text."],
            "n_chars": [13],
            "sha256": ["digest"],
        }
    )


def _tokens() -> pl.DataFrame:
    return pl.DataFrame({"doc_id": ["1"], "token": ["Example"]})


def test_process_corpus_materializes_generator_metrics_once():
    requested_metrics = (metric for metric in ("freq", "dtm"))

    with (
        patch.object(api, "_ingest_sources", return_value=_documents()),
        patch.object(
            api,
            "_load_model",
            return_value=SimpleNamespace(meta={"name": "test-model"}),
        ),
        patch.object(api, "_parse", return_value=_tokens()),
        patch.object(
            api,
            "_compute_metrics",
            return_value=(None, None, None, None, None, None),
        ) as mock_compute,
    ):
        result = api.process_corpus("Example text.", metrics=requested_metrics)

    computed_tokens, computed_metrics = mock_compute.call_args.args
    assert computed_tokens.equals(_tokens())
    assert computed_metrics == ("freq", "dtm")
    assert result.manifest["metrics"] == ["freq", "dtm"]


def test_process_corpus_deduplicates_metrics_in_caller_order():
    with (
        patch.object(api, "_ingest_sources", return_value=_documents()),
        patch.object(
            api,
            "_load_model",
            return_value=SimpleNamespace(meta={"name": "test-model"}),
        ),
        patch.object(api, "_parse", return_value=_tokens()),
        patch.object(
            api,
            "_compute_metrics",
            return_value=(None, None, None, None, None, None),
        ) as mock_compute,
    ):
        result = api.process_corpus(
            "Example text.",
            metrics=("tags", "freq", "tags", "dtm", "freq"),
        )

    computed_tokens, computed_metrics = mock_compute.call_args.args
    assert computed_tokens.equals(_tokens())
    assert computed_metrics == ("tags", "freq", "dtm")
    assert result.manifest["metrics"] == ["tags", "freq", "dtm"]


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ((), "At least one metric is required"),
        (("freq", "unknown"), "Unsupported metric selection: unknown"),
    ],
)
def test_process_corpus_rejects_invalid_metrics_before_ingest(metrics, message):
    with patch.object(api, "_ingest_sources") as mock_ingest:
        with pytest.raises(CorpusProcessingError, match=message):
            api.process_corpus("Example text.", metrics=metrics)

    mock_ingest.assert_not_called()
