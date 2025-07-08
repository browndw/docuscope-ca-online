"""
Pytest configuration and shared fixtures for DocuScope testing.

This module provides common fixtures and configuration for testing
the DocuScope Corpus Analysis application.
"""

import pytest
import tempfile
import polars as pl
import spacy
from pathlib import Path
from unittest.mock import MagicMock


# =============================================================================
# Test Data Factories
# =============================================================================

@pytest.fixture
def corpus_factory():
    """Factory for generating test corpus DataFrames with various content types."""
    def _make_corpus(doc_count: int = 2, content_type: str = "simple") -> pl.DataFrame:
        if content_type == "simple":
            texts = [f"This is test document {i}." for i in range(doc_count)]
        elif content_type == "academic":
            texts = [
                "However, the research demonstrates significant findings in this domain.",
                "Furthermore, this analysis reveals important implications for theory.",
                "Nevertheless, additional investigation is warranted to confirm results.",
                "Moreover, the methodology employed ensures robust data collection.",
                "Therefore, these conclusions represent meaningful contributions."
            ][:doc_count]
        elif content_type == "complex":
            texts = [
                ("Dr. Smith's research (published in 2023) shows that complex "
                 "sentences—with multiple clauses—are processed differently."),
                ("The data collected over 5 years indicates: (1) increased accuracy, "
                 "(2) improved performance, and (3) better outcomes."),
                ("Results suggest that while previous studies were limited, our "
                 "approach addresses these methodological concerns effectively.")
            ][:doc_count]
        elif content_type == "punctuation":
            texts = [
                "What?! Really... Yes, indeed!!! This is surprising: very surprising.",
                "Questions arise: How? Why? When? These need answers—immediate answers.",
                "Statements include; lists, items, and various punctuation marks!!!"
            ][:doc_count]
        elif content_type == "short":
            texts = ["Short.", "Brief.", "Tiny.", "Small.", "Mini."][:doc_count]
        elif content_type == "empty":
            texts = ["", "   ", "\n\n", "\t\t"][:doc_count]
        else:
            raise ValueError(f"Unknown content_type: {content_type}")

        return pl.DataFrame({
            "doc_id": [f"doc_{i}" for i in range(doc_count)],
            "text": texts,
        })

    return _make_corpus


@pytest.fixture
def tagged_corpus_factory():
    """Factory for generating test corpus DataFrames with pre-tagged content."""
    def _make_tagged_corpus(doc_count: int = 2) -> pl.DataFrame:
        # Simulate realistic tagged output structure with actual DocuScope tags
        data = []
        for doc_idx in range(doc_count):
            doc_id = f"doc_{doc_idx}"
            tokens = ["This", "is", "a", "test", "document", "."]
            pos_tags = ["DD1", "VBZ", "AT", "NN1", "NN1", "Y"]
            ds_tags = ["MetadiscourseCohesive", "InformationStates", "Untagged",
                       "AcademicTerms", "AcademicTerms", "Untagged"]

            for i, (token, pos_tag, ds_tag) in enumerate(zip(tokens, pos_tags, ds_tags)):
                data.append({
                    "doc_id": doc_id,
                    "token": token,
                    "pos_tag": pos_tag,
                    "ds_tag": ds_tag,
                    "pos_id": i,
                    "ds_id": i
                })

        # Ensure proper data types for schema validation
        df = pl.DataFrame(data)
        return df.with_columns([
            pl.col("pos_id").cast(pl.UInt32),
            pl.col("ds_id").cast(pl.UInt32)
        ])

    return _make_tagged_corpus


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def docuscope_models():
    """Load actual DocuScope models for testing."""
    models_dir = Path("webapp/_models/")

    if not models_dir.exists():
        pytest.skip("DocuScope models not found in webapp/_models/")

    models = {}

    # Try to load large dictionary model
    large_model_path = models_dir / "en_docusco_spacy"
    if large_model_path.exists():
        try:
            models["Large Dictionary"] = spacy.load(str(large_model_path))
        except OSError:
            pytest.skip("Could not load large dictionary model")

    # Try to load common dictionary model
    small_model_path = models_dir / "en_docusco_spacy_cd"
    if small_model_path.exists():
        try:
            models["Common Dictionary"] = spacy.load(str(small_model_path))
        except OSError:
            pytest.skip("Could not load common dictionary model")

    if not models:
        pytest.skip("No DocuScope models could be loaded")

    return models


@pytest.fixture
def mock_spacy_model():
    """Mock spaCy model for testing without loading actual models."""
    mock_model = MagicMock()
    mock_model.pipe.return_value = [
        MagicMock(text="test", ents=[], sents=[])
    ]
    return mock_model


# =============================================================================
# Session State Fixtures
# =============================================================================

@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state for unit testing."""
    class MockSessionState:
        def __init__(self):
            self.data = {}

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

        def __contains__(self, key):
            return key in self.data

        def get(self, key, default=None):
            return self.data.get(key, default)

        def keys(self):
            return self.data.keys()

        def pop(self, key, default=None):
            return self.data.pop(key, default)

    return MockSessionState()


@pytest.fixture
def sample_session_state(mock_streamlit_session, tagged_corpus_factory):
    """Sample session state with corpus data loaded."""
    user_session_id = "test_user_123"

    # Create basic session structure
    mock_streamlit_session[user_session_id] = {
        "session": {
            "has_target": True,
            "has_reference": False,
            "has_meta": False,
        },
        "target": {
            "ds_tokens": tagged_corpus_factory(doc_count=3)
        },
        "metadata_target": {
            "ndocs": 3,
            "model": "Large Dictionary",
            "tags_ds": {"tags": ["Confidence", "Academic", "Reasoning", "Information"]},
            "tags_pos": {"tags": ["DT", "VBZ", "NN", "JJ"]},
            "docids": {"ids": ["doc_0", "doc_1", "doc_2"]}
        }
    }

    return mock_streamlit_session, user_session_id


# =============================================================================
# File System Fixtures
# =============================================================================

@pytest.fixture
def temp_corpus_files():
    """Generate temporary text files for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        files = []
        test_contents = [
            "This is the first test document with academic content.",
            "The second document explores different rhetorical patterns.",
            "Finally, this document concludes with important findings."
        ]

        for i, content in enumerate(test_contents):
            file_path = Path(temp_dir) / f"test_doc_{i}.txt"
            file_path.write_text(content, encoding='utf-8')
            files.append(file_path)

        yield files


@pytest.fixture
def temp_parquet_file(tagged_corpus_factory):
    """Generate temporary parquet file for testing corpus uploads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        df = tagged_corpus_factory(doc_count=5)
        file_path = Path(temp_dir) / "test_corpus.parquet"
        df.write_parquet(file_path)
        yield file_path


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_config():
    """Test configuration values."""
    return {
        "desktop_mode": True,
        "check_size": False,
        "check_language": False,
        "max_text_size": 1000,
        "max_polars_size": 10000
    }


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def capture_streamlit_output():
    """Capture Streamlit output for testing UI components."""
    outputs = []

    def mock_success(message, icon=None):
        outputs.append(("success", message, icon))

    def mock_error(message, icon=None):
        outputs.append(("error", message, icon))

    def mock_warning(message, icon=None):
        outputs.append(("warning", message, icon))

    def mock_info(message, icon=None):
        outputs.append(("info", message, icon))

    return {
        "outputs": outputs,
        "success": mock_success,
        "error": mock_error,
        "warning": mock_warning,
        "info": mock_info
    }
