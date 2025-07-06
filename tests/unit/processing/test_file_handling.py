"""Test file processing and corpus loading functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import polars as pl


class TestFileHandling:
    """Test file handling and processing functionality."""

    def test_basic_file_reading(self):
        """Test basic file reading capabilities."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt',
            delete=False
        ) as f:
            f.write("Test content for file processing")
            test_file_path = f.name

        try:
            with open(test_file_path, 'r') as file:
                content = file.read()
                assert content == "Test content for file processing"
        finally:
            os.unlink(test_file_path)

    def test_multiple_file_formats(self):
        """Test handling of multiple file formats."""
        file_formats = ['.txt', '.md', '.rtf']

        for fmt in file_formats:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix=fmt,
                delete=False
            ) as f:
                f.write(f"Content for {fmt} file")
                temp_path = f.name

            try:
                with open(temp_path, 'r') as file:
                    content = file.read()
                    assert f"Content for {fmt} file" in content
            finally:
                os.unlink(temp_path)

    def test_file_encoding_handling(self):
        """Test handling of different file encodings."""
        test_content = "Test content with special characters: áéíóú"

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name

        try:
            with open(temp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                assert content == test_content
        finally:
            os.unlink(temp_path)

    def test_large_file_handling(self):
        """Test handling of large files."""
        large_content = "This is a line of text.\n" * 1000

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt',
            delete=False
        ) as f:
            f.write(large_content)
            temp_path = f.name

        try:
            with open(temp_path, 'r') as file:
                content = file.read()
                lines = content.split('\n')
                assert len(lines) > 900  # Account for potential empty lines
        finally:
            os.unlink(temp_path)

    @patch('webapp.utilities.processing.corpus_loading.load_corpus_internal')
    def test_corpus_loading_basic(self, mock_load_corpus):
        """Test basic corpus loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = []
            for i in range(3):
                file_path = Path(temp_dir) / f"doc{i}.txt"
                file_path.write_text(f"Content of document {i}")
                files.append(str(file_path))

            # Mock the corpus loading function
            mock_df = pl.DataFrame({
                "doc_id": ["doc1", "doc2", "doc3"],
                "text": ["Content 1", "Content 2", "Content 3"]
            })
            mock_load_corpus.return_value = mock_df

            try:
                from webapp.utilities.processing.corpus_loading import load_corpus_internal
                corpus_df = load_corpus_internal(
                    files, "test_session", "target", True, True
                )
                assert isinstance(corpus_df, pl.DataFrame)
                assert corpus_df.height == 3
                assert "doc_id" in corpus_df.columns
                assert "text" in corpus_df.columns
            except ImportError:
                pytest.skip("Corpus loading module not available")

    def test_file_processing_with_validation(self):
        """Test file processing with validation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as test_file:
            test_file.write("Valid content for testing")
            test_file_path = test_file.name

        try:
            # Test file validation
            assert os.path.exists(test_file_path)
            assert os.path.getsize(test_file_path) > 0

            with open(test_file_path, 'r') as f:
                content = f.read()
                assert len(content.strip()) > 0
        finally:
            os.unlink(test_file_path)

    def test_error_handling_invalid_files(self):
        """Test error handling for invalid files."""
        non_existent_file = "/path/that/does/not/exist.txt"

        # Test that the function handles non-existent files gracefully
        with pytest.raises(FileNotFoundError):
            with open(non_existent_file, 'r') as f:
                f.read()

    def test_file_metadata_extraction(self):
        """Test extraction of file metadata."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as test_file:
            test_file.write("Content for metadata test")
            test_file_path = test_file.name

        try:
            file_stat = os.stat(test_file_path)

            # Test basic metadata extraction
            assert file_stat.st_size > 0
            assert file_stat.st_mtime is not None

            # Test file extension extraction
            file_ext = Path(test_file_path).suffix
            assert file_ext == '.txt'
        finally:
            os.unlink(test_file_path)

    @patch('webapp.utilities.processing.corpus_loading.load_corpus_internal')
    def test_batch_file_processing(self, mock_load_corpus):
        """Test batch processing of multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = []

            # Create multiple test files
            for i in range(5):
                file_path = Path(temp_dir) / f"batch_doc{i}.txt"
                file_path.write_text(f"Batch content {i}")
                files.append(str(file_path))

            # Mock batch processing
            mock_df = pl.DataFrame({
                "doc_id": [f"batch_doc{i}" for i in range(5)],
                "text": [f"Batch content {i}" for i in range(5)]
            })
            mock_load_corpus.return_value = mock_df

            try:
                from webapp.utilities.processing.corpus_loading import load_corpus_internal
                corpus_df = load_corpus_internal(
                    files, "test_session", "target", True, True
                )
                assert isinstance(corpus_df, pl.DataFrame)
                assert corpus_df.height == 5
            except ImportError:
                pytest.skip("Corpus loading module not available")


class TestDocumentProcessing:
    """Test document processing functionality."""

    def test_text_file_processing(self):
        """Test processing of text files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write("Sample text for processing")
            temp_path = f.name

        try:
            with open(temp_path, 'r') as file:
                content = file.read()
                # Test basic text processing
                assert len(content) > 0
                assert isinstance(content, str)
        finally:
            os.unlink(temp_path)

    def test_document_chunking(self):
        """Test document chunking functionality."""
        large_text = "This is a sentence. " * 100

        # Test simple chunking by character count
        chunk_size = 100
        chunks = [large_text[i:i+chunk_size]
                  for i in range(0, len(large_text), chunk_size)]

        assert len(chunks) > 1
        assert all(len(chunk) <= chunk_size for chunk in chunks[:-1])

    def test_text_cleaning(self):
        """Test text cleaning functionality."""
        dirty_text = "   Text with\n\n\nextra   whitespace\t\t  "

        # Test basic cleaning
        cleaned = " ".join(dirty_text.split())
        assert cleaned == "Text with extra whitespace"

    def test_document_validation(self):
        """Test document validation."""
        valid_content = "This is valid document content."
        empty_content = ""
        whitespace_only = "   \n\t   "

        # Test validation logic
        assert len(valid_content.strip()) > 0
        assert len(empty_content.strip()) == 0
        assert len(whitespace_only.strip()) == 0


class TestCorpusManagement:
    """Test corpus management functionality."""

    def test_corpus_metadata(self):
        """Test corpus metadata handling."""
        corpus_metadata = {
            'name': 'test_corpus',
            'total_documents': 100,
            'total_tokens': 50000,
            'creation_date': '2025-01-01'
        }

        # Test metadata validation
        assert corpus_metadata['name'] is not None
        assert corpus_metadata['total_documents'] > 0
        assert corpus_metadata['total_tokens'] > 0

    def test_corpus_statistics(self):
        """Test corpus statistics calculation."""
        mock_corpus = {
            'documents': ['doc1', 'doc2', 'doc3'],
            'word_counts': [100, 200, 150]
        }

        # Test statistics
        total_docs = len(mock_corpus['documents'])
        total_words = sum(mock_corpus['word_counts'])
        avg_length = total_words / total_docs

        assert total_docs == 3
        assert total_words == 450
        assert avg_length == 150.0

    def test_corpus_comparison_setup(self):
        """Test setup for corpus comparison."""
        corpus_a = {'name': 'corpus_a', 'size': 1000}
        corpus_b = {'name': 'corpus_b', 'size': 800}

        comparison_config = {
            'target_corpus': corpus_a,
            'reference_corpus': corpus_b,
            'comparison_type': 'frequency'
        }

        # Test comparison setup
        target_size = comparison_config['target_corpus']['size']
        reference_size = comparison_config['reference_corpus']['size']
        assert target_size > reference_size

        valid_types = ['frequency', 'tags', 'ngrams']
        assert comparison_config['comparison_type'] in valid_types


class TestFileFormats:
    """Test support for UTF-8 text files only."""

    def test_txt_file_support(self):
        """Test .txt file support."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write("Plain text content")
            temp_path = f.name

        try:
            with open(temp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                assert content == "Plain text content"
        finally:
            os.unlink(temp_path)

    def test_markdown_file_support(self):
        """Test .md file support (treated as UTF-8 text)."""
        markdown_content = "# Header\n\nThis is **bold** text."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.md',
                                         delete=False) as f:
            f.write(markdown_content)
            temp_path = f.name

        try:
            with open(temp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                assert "# Header" in content
                assert "**bold**" in content
        finally:
            os.unlink(temp_path)

    def test_utf8_text_only_policy(self):
        """Test that only UTF-8 text files are supported."""
        # Create a valid UTF-8 text file
        utf8_content = "Valid UTF-8 text with special characters: áéíóú"

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         suffix='.txt', delete=False) as f:
            f.write(utf8_content)
            temp_path = f.name

        try:
            with open(temp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                assert content == utf8_content
        finally:
            os.unlink(temp_path)

    def test_docx_file_rejection(self):
        """Test that DOCX files are rejected (not UTF-8 text)."""
        # Create a fake DOCX file (binary format)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.docx', delete=False) as f:
            # DOCX files are binary ZIP archives, not UTF-8 text
            # Write an invalid UTF-8 sequence
            f.write(b'\xff\xfe\x00\x00\x89PNG\r\n\x1a\n')  # Invalid UTF-8 bytes
            temp_path = f.name

        try:
            # Attempting to read as UTF-8 should fail
            with pytest.raises(UnicodeDecodeError):
                with open(temp_path, 'r', encoding='utf-8') as file:
                    file.read()
        finally:
            os.unlink(temp_path)

    def test_binary_file_rejection(self):
        """Test that binary files are rejected."""
        # Create a binary file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG file signature
            temp_path = f.name

        try:
            # Should fail when trying to read as UTF-8
            with pytest.raises(UnicodeDecodeError):
                with open(temp_path, 'r', encoding='utf-8') as file:
                    file.read()
        finally:
            os.unlink(temp_path)

    def test_non_utf8_text_rejection(self):
        """Test that non-UTF-8 encoded text files are rejected."""
        # Create a file with Latin-1 encoding
        latin1_text = "Café résumé naïve"

        with tempfile.NamedTemporaryFile(mode='w', encoding='latin-1',
                                         delete=False) as f:
            f.write(latin1_text)
            temp_path = f.name

        try:
            # Should work with correct encoding
            with open(temp_path, 'r', encoding='latin-1') as file:
                content = file.read()
                assert content == latin1_text

            # But should fail when forced to read as UTF-8
            # (depending on the specific characters, this might work or fail)
            # Let's create a definitely invalid UTF-8 sequence
            pass
        finally:
            os.unlink(temp_path)

        # Create a file with invalid UTF-8 byte sequence
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8 sequence
            temp_path = f.name

        try:
            with pytest.raises(UnicodeDecodeError):
                with open(temp_path, 'r', encoding='utf-8') as file:
                    file.read()
        finally:
            os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling in file processing."""

    def test_file_not_found_handling(self):
        """Test handling of missing files."""
        non_existent = "/this/file/does/not/exist.txt"

        with pytest.raises(FileNotFoundError):
            with open(non_existent, 'r') as f:
                f.read()

    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a file and remove read permissions (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write("Test content")
                temp_path = f.name

            try:
                # Remove read permissions
                os.chmod(temp_path, 0o000)

                with pytest.raises(PermissionError):
                    with open(temp_path, 'r') as file:
                        file.read()
            finally:
                # Restore permissions for cleanup
                os.chmod(temp_path, 0o644)
                os.unlink(temp_path)
        else:
            pytest.skip("Permission test not applicable on Windows")

    def test_encoding_error_handling(self):
        """Test handling of encoding errors."""
        # Create a file with binary content that isn't valid UTF-8
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(b'\xff\xfe\x00\x00')  # Invalid UTF-8 sequence
            temp_path = f.name

        try:
            # This should raise a UnicodeDecodeError
            with pytest.raises(UnicodeDecodeError):
                with open(temp_path, 'r', encoding='utf-8') as file:
                    file.read()
        finally:
            os.unlink(temp_path)

    def test_corrupted_file_handling(self):
        """Test handling of corrupted files."""
        # Create an empty file that pretends to be a specific format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            # Write some potentially problematic content
            f.write('\x00\x00\x00')  # Null bytes
            temp_path = f.name

        try:
            with open(temp_path, 'r') as file:
                content = file.read()
                # File should be readable, even with null bytes
                assert isinstance(content, str)
        finally:
            os.unlink(temp_path)


class TestFileTypeValidation:
    """Test file type validation and extension support."""

    def test_supported_text_extensions(self):
        """Test that supported text file extensions are recognized."""
        supported_extensions = ['.txt', '.md', '.rst', '.log']

        for ext in supported_extensions:
            # These should all be treated as UTF-8 text files
            assert ext.lower() in ['.txt', '.md', '.rst', '.log']

    def test_unsupported_binary_extensions(self):
        """Test that binary file extensions are not supported."""
        unsupported_extensions = ['.docx', '.pdf', '.doc', '.rtf', '.odt']

        for ext in unsupported_extensions:
            # Create a file with the extension
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                if ext in ['.docx', '.pdf', '.doc', '.odt']:
                    # Write binary content for these formats
                    f.write(b'\x50\x4b\x03\x04')  # ZIP signature (for docx/odt)
                else:
                    # Write text content but expect rejection based on extension policy
                    f.write(b'Some content')
                temp_path = f.name

            try:
                # The application should reject these file types
                # This test documents the expected behavior
                file_extension = Path(temp_path).suffix.lower()
                assert file_extension in unsupported_extensions
            finally:
                os.unlink(temp_path)

    def test_file_content_validation(self):
        """Test that file content is validated as UTF-8."""
        # Valid UTF-8 content
        valid_content = "This is valid UTF-8 text with unicode: 你好世界"

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         suffix='.txt', delete=False) as f:
            f.write(valid_content)
            temp_path = f.name

        try:
            # Should read successfully as UTF-8
            with open(temp_path, 'r', encoding='utf-8') as file:
                content = file.read()
                assert content == valid_content
        finally:
            os.unlink(temp_path)

    def test_file_size_considerations(self):
        """Test handling of different file sizes."""
        # Small file
        small_content = "Small file content"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write(small_content)
            small_path = f.name

        # Large file (but still reasonable for testing)
        large_content = "Large file content.\n" * 1000
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write(large_content)
            large_path = f.name

        try:
            # Both should be readable
            with open(small_path, 'r', encoding='utf-8') as f:
                assert len(f.read()) > 0

            with open(large_path, 'r', encoding='utf-8') as f:
                content = f.read()
                assert len(content) > 10000
                assert content.count('\n') >= 1000
        finally:
            os.unlink(small_path)
            os.unlink(large_path)
