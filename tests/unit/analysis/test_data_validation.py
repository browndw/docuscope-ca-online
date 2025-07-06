"""
Unit tests for data validation functions.

This module tests the core data validation functions used throughout
the DocuScope application for validating corpus data, checking formats,
and handling file uploads.
"""

import pytest
import polars as pl
from unittest.mock import patch, MagicMock

# Note: This module tests pure data validation functions and session utilities.
# No Streamlit UI components are tested here, so AppTest is not needed.

from webapp.utilities.analysis.data_validation import (
    check_language,
    check_schema,
    check_corpus_new,
    validate_dataframe_content,
    normalize_text,
    has_target_corpus,
    has_reference_corpus,
    has_metadata,
    safe_get_categories,
    is_valid_df
)


class TestDataValidation:
    """Test data validation utility functions."""

    def test_normalize_text_basic(self):
        """Test basic text normalization."""
        input_text = "This is a Test with UPPERCASE and lowercase."
        result = normalize_text(input_text)

        # Should normalize whitespace but preserve case
        assert isinstance(result, str)
        assert result == "This is a Test with UPPERCASE and lowercase."
        assert "Test" in result
        assert "UPPERCASE" in result

    def test_normalize_text_special_characters(self):
        """Test text normalization with special characters."""
        input_text = "Text with émojis 😀 and açcénts!"
        result = normalize_text(input_text)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_text_empty(self):
        """Test text normalization with empty input."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""
        assert normalize_text("\n\t") == ""

    def test_is_valid_df_polars(self, corpus_factory):
        """Test DataFrame validation with valid Polars DataFrame."""
        df = corpus_factory(doc_count=3, content_type="simple")
        assert is_valid_df(df) is True

    def test_is_valid_df_empty(self):
        """Test DataFrame validation with empty DataFrame."""
        empty_df = pl.DataFrame({"doc_id": [], "text": []})
        assert is_valid_df(empty_df) is False

    def test_is_valid_df_none(self):
        """Test DataFrame validation with None input."""
        assert is_valid_df(None) is False

    def test_is_valid_df_wrong_type(self):
        """Test DataFrame validation with wrong type."""
        assert is_valid_df("not a dataframe") is False
        assert is_valid_df([1, 2, 3]) is False
        assert is_valid_df({}) is False

    def test_validate_dataframe_content_valid(self, tagged_corpus_factory):
        """Test content validation with valid corpus data."""
        df = tagged_corpus_factory(doc_count=10)  # More docs to avoid size warning
        result = validate_dataframe_content(df)

        assert result == []  # Empty list means no validation errors

    def test_validate_dataframe_content_empty_text(self):
        """Test content validation with empty DataFrame."""
        df = pl.DataFrame({
            "doc_id": [],
            "token": [],
            "pos_tag": [],
            "ds_tag": []
        })
        result = validate_dataframe_content(df)

        assert "DataFrame is empty" in result

    def test_validate_dataframe_content_missing_columns(self):
        """Test content validation with missing required columns."""
        df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "content": ["text1", "text2"]  # Wrong column name
        })
        result = validate_dataframe_content(df)

        assert any("Missing required columns" in msg for msg in result)


class TestCorpusValidation:
    """Test corpus-specific validation functions."""

    def test_check_schema_valid_corpus(self, tagged_corpus_factory):
        """Test schema validation with valid tagged corpus."""
        df = tagged_corpus_factory(doc_count=2)
        result = check_schema(df)

        assert result is True

    def test_check_schema_missing_columns(self):
        """Test schema validation with missing required columns."""
        df = pl.DataFrame({
            "doc_id": ["doc1"],
            "token": ["test"],
            # Missing pos_tag, ds_tag, etc.
        })
        result = check_schema(df)

        assert result is False

    def test_check_schema_empty_dataframe(self):
        """Test schema validation with empty DataFrame."""
        df = pl.DataFrame()
        result = check_schema(df)

        assert result is False

    @patch('webapp.utilities.analysis.data_validation.load_detector')
    def test_check_language_english(self, mock_load_detector):
        """Test language detection for English text."""
        # Mock the language detector
        mock_detector = MagicMock()
        mock_detector.compute_language_confidence.return_value = 0.95  # High confidence
        mock_load_detector.return_value = mock_detector

        text = "This is a clear English sentence with proper grammar."
        result = check_language(text)

        assert result is True

    @patch('webapp.utilities.analysis.data_validation.load_detector')
    def test_check_language_non_english(self, mock_load_detector):
        """Test language detection for non-English text."""
        mock_detector = MagicMock()
        mock_detector.compute_language_confidence.return_value = 0.5  # Low confidence
        mock_load_detector.return_value = mock_detector

        text = "Este es un texto en español."
        result = check_language(text)

        assert result is False

    @patch('webapp.utilities.analysis.data_validation.load_detector')
    def test_check_language_detection_error(self, mock_load_detector):
        """Test language detection when detector fails."""
        mock_detector = MagicMock()
        mock_detector.compute_language_confidence.side_effect = Exception(
            "Detection failed"
        )
        mock_load_detector.return_value = mock_detector

        text = "Some text"

        # The function should propagate the exception or handle it
        # Based on the implementation, it should raise the exception
        with pytest.raises(Exception):
            check_language(text)

    def test_check_corpus_new_basic(self, temp_corpus_files):
        """Test basic corpus validation for new uploads."""
        # Convert Path objects to file-like objects for testing
        mock_files = []
        for file_path in temp_corpus_files:
            mock_file = MagicMock()
            mock_file.name = file_path.name
            mock_file.getvalue.return_value = file_path.read_bytes()
            mock_files.append(mock_file)

        result = check_corpus_new(docs=mock_files)

        # Should return duplicate IDs (empty list if no duplicates)
        assert isinstance(result, list)

    def test_check_corpus_new_with_duplicates(self):
        """Test corpus validation with duplicate file names."""
        mock_files = []
        for i in range(3):
            mock_file = MagicMock()
            mock_file.name = "duplicate.txt"  # Same name for all
            mock_file.getvalue.return_value = b"Some content"
            mock_files.append(mock_file)

        result = check_corpus_new(docs=mock_files)

        # Should detect duplicates (filename without extension)
        assert "duplicate" in result

    def test_check_corpus_new_empty_files(self):
        """Test corpus validation with empty file list."""
        result = check_corpus_new(docs=[])

        assert isinstance(result, list)
        assert len(result) == 0


class TestSessionStateValidation:
    """Test session state validation functions."""

    def test_has_target_corpus_true(self, sample_session_state):
        """Test target corpus detection when corpus exists."""
        mock_session, user_id = sample_session_state
        session_data = mock_session[user_id]["session"]  # Get session data directly

        result = has_target_corpus(session_data)
        assert result is True

    def test_has_target_corpus_false(self):
        """Test target corpus detection when corpus doesn't exist."""
        session_data = {"has_target": False}

        result = has_target_corpus(session_data)
        assert result is False

    def test_has_target_corpus_missing_key(self):
        """Test target corpus detection with missing session keys."""
        session_data = {}

        result = has_target_corpus(session_data)
        assert result is False

    def test_has_reference_corpus_true(self):
        """Test reference corpus detection when corpus exists."""
        session_data = {"has_reference": True}

        result = has_reference_corpus(session_data)
        assert result is True

    def test_has_reference_corpus_false(self):
        """Test reference corpus detection when corpus doesn't exist."""
        session_data = {"has_reference": False}

        result = has_reference_corpus(session_data)
        assert result is False

    def test_has_metadata_true(self, sample_session_state):
        """Test metadata detection when metadata exists."""
        mock_session, user_id = sample_session_state
        # Add metadata flag
        mock_session[user_id]["session"]["has_meta"] = True
        session_data = mock_session[user_id]["session"]

        result = has_metadata(session_data)
        assert result is True

    def test_has_metadata_false(self):
        """Test metadata detection when metadata doesn't exist."""
        session_data = {"has_meta": False}

        result = has_metadata(session_data)
        assert result is False

    def test_safe_get_categories_valid(self, sample_session_state):
        """Test safe category extraction with valid data."""
        mock_session, user_id = sample_session_state

        # Add category data in the correct format
        metadata_target = {
            "doccats": [{"cats": ["academic", "technical", "general"]}]
        }

        categories = safe_get_categories(metadata_target)

        assert isinstance(categories, list)
        assert "academic" in categories

    def test_safe_get_categories_no_metadata(self):
        """Test safe category extraction with no metadata."""
        categories = safe_get_categories({})

        assert isinstance(categories, list)
        assert len(categories) == 0


class TestParameterizedValidation:
    """Parameterized tests for various validation scenarios."""

    @pytest.mark.parametrize("content_type,expected", [
        ("simple", True),
        ("academic", True),
        ("complex", True),
        ("punctuation", True),
        ("short", True),
    ])
    def test_validate_different_content_types(
        self, tagged_corpus_factory, content_type, expected
    ):
        """Test validation with different content types."""
        # More docs to avoid size warning
        df = tagged_corpus_factory(doc_count=10)  # Tagged corpus doesn't use content_type
        result = validate_dataframe_content(df)
        # For valid data, result should be empty list (no errors)
        assert (result == []) == expected

    @pytest.mark.parametrize("doc_count", [1, 2, 5, 10])
    def test_validate_different_corpus_sizes(self, tagged_corpus_factory, doc_count):
        """Test validation with different corpus sizes."""
        df = tagged_corpus_factory(doc_count=doc_count)
        result = validate_dataframe_content(df)

        if doc_count >= 5:
            assert result == []  # No warnings for sufficient docs
        else:
            # Should have warning about insufficient documents
            assert any("documents found - may be insufficient" in msg for msg in result)

    @pytest.mark.parametrize("invalid_text", ["", "   ", "\n", "\t", None])
    def test_normalize_text_edge_cases(self, invalid_text):
        """Test text normalization with various edge cases."""
        if invalid_text is None:
            with pytest.raises((TypeError, AttributeError)):
                normalize_text(invalid_text)
        else:
            result = normalize_text(invalid_text)
            assert result == ""
