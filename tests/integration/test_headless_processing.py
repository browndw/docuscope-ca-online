"""Integration test exercising the headless processing API with bundled models."""

from __future__ import annotations

import gzip
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st

from docuscope_ca import process_corpus
from webapp.utilities.session.session_core import (
    init_metadata_target,
    write_metadata_descriptor_sidecar,
)
from webapp.utilities.state import SessionKeys, MetadataKeys


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


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize(
    "corpus_dir",
    [
        Path("tests/test_data/tar_corpus"),
        Path("tests/test_data/ref_corpus"),
        Path("load_tests/test_data/tar_corpus"),
        Path("load_tests/test_data/ref_corpus"),
    ],
    ids=[
        "tests-target",
        "tests-reference",
        "load-tests-target",
        "load-tests-reference",
    ],
)
def test_fixture_corpora_support_sidecar_metadata_loading(corpus_dir: Path):
    model_dir = Path("webapp/_models/en_docusco_spacy")

    if not corpus_dir.exists():
        pytest.skip(f"Fixture corpus missing: {corpus_dir}")
    if not model_dir.exists():
        pytest.skip("Bundled DocuScope model missing from repository")

    sample_docs = sorted(corpus_dir.glob("*.txt"))[:3]
    assert sample_docs, f"Expected at least one sample document in {corpus_dir}"

    result = process_corpus(
        sources=[str(path) for path in sample_docs],
        model=str(model_dir),
        metrics=("freq",),
        export_dir=None,
    )

    with TemporaryDirectory() as temp_dir:
        corpus_artifact_dir = Path(temp_dir) / corpus_dir.name
        corpus_artifact_dir.mkdir(parents=True, exist_ok=True)

        ds_tokens_path = corpus_artifact_dir / "ds_tokens.gz"
        with gzip.open(ds_tokens_path, "wb") as file_handle:
            pickle.dump(result.tokens, file_handle)

        sidecar_path = write_metadata_descriptor_sidecar(
            corpus_artifact_dir,
            df=result.tokens,
        )
        assert sidecar_path.exists()

        session_id = f"sidecar_{corpus_dir.parent.name}_{corpus_dir.name}"
        st.session_state.clear()
        st.session_state[session_id] = {}

        mock_manager = MagicMock()
        mock_manager.session_corpus_data = {}
        mock_manager._get_artifact_refs.return_value = {
            "ds_tokens": {
                "storage_type": "gzip_pickle",
                "path": str(ds_tokens_path),
            }
        }
        mock_manager.get_core_data.side_effect = AssertionError(
            "sidecar-backed metadata init should not load ds_tokens"
        )

        with patch(
            "webapp.utilities.session.session_core.get_corpus_manager",
            return_value=mock_manager,
        ):
            init_metadata_target(session_id)

        metadata_df = st.session_state[session_id][SessionKeys.METADATA_TARGET]
        metadata = metadata_df.to_dict(as_series=False)

        assert metadata[MetadataKeys.NDOCS][0] == len(sample_docs)
        assert metadata[MetadataKeys.DOCIDS][0]["ids"]
        assert metadata[MetadataKeys.TAGS_POS][0]["tags"]
