"""Integration test exercising the headless processing API with bundled models."""

from __future__ import annotations

from pathlib import Path

import pytest

from docuscope_ca import process_corpus


@pytest.mark.integration
@pytest.mark.slow
def test_process_corpus_with_local_model():
    corpus_dir = Path("paper/data/test_corpus")
    model_dir = Path("webapp/_models/en_docusco_spacy")

    if not corpus_dir.exists():
        pytest.skip("Sample corpus not available in checkout")
    if not model_dir.exists():
        pytest.skip("Bundled DocuScope model missing from repository")

    sample_docs = sorted(corpus_dir.glob("*.txt"))[:3]
    assert sample_docs, "Expected at least one sample document"

    result = process_corpus(
        sources=[str(p) for p in sample_docs],
        model=str(model_dir),
        metrics=("freq", "tags"),
        export_dir=None,
    )

    assert result.tokens.height > 0
    assert result.freq_pos is not None
    assert result.freq_ds is not None
    assert result.tags_pos is not None
    assert result.tags_ds is not None

    manifest = result.manifest
    assert manifest["corpus"]["n_documents"] == len(sample_docs)
    assert manifest["corpus"]["total_tokens"] > 0
    assert manifest["model_name"]
    assert set(manifest["metrics"]) == {"freq", "tags"}
