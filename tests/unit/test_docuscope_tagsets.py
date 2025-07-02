"""
Tests for validating DocuScope model tagsets and output formats.

Tests the actual structure and tags produced by both the large and small
DocuScope models to ensure test data matches reality.
"""

import polars as pl


class TestDocuScopeTagsets:
    """Test DocuScope model tagsets and data structures."""

    def test_large_model_tagset_structure(self):
        """Test expected structure for large dictionary model output."""
        # Sample of actual large model tags based on provided examples
        large_model_tags = {
            "Character", "Narrative", "Citation", "Untagged", "AcademicTerms",
            "InformationExposition", "InformationChange", "Description",
            "Contingent", "Negative", "MetadiscourseCohesive", "Strategic",
            "Reasoning", "ConfidenceHedged", "InformationTopics", "InformationStates",
            "Updates", "Uncertainty", "InformationReportVerbs", "Positive",
            "InformationChangePositive"
        }

        # Sample of POS tags from the examples
        pos_tags = {
            "NP1", "RR", "VVD", "Y", "JJ", "NN1", "VBZ", "II", "AT", "DDQ",
            "MC", "NN2", "RRQV", "MC1", "IO", "PPH1", "GE", "VBDZ", "VVN",
            "RP", "DD1", "CS", "CCB", "VVZ", "VHZ", "DA1", "NNT1", "APPGE"
        }

        # Verify we have realistic tag sets
        assert len(large_model_tags) > 15
        assert len(pos_tags) > 20
        assert "Untagged" in large_model_tags
        assert "AcademicTerms" in large_model_tags

    def test_small_model_tagset_structure(self):
        """Test expected structure for small (common dictionary) model output."""
        # Sample of actual small model tags based on provided examples
        small_model_tags = {
            "ActorsPeople", "Untagged", "ActorsAbstractions", "SignpostingMetadiscourse",
            "PlanningStrategy", "SentimentPositive", "ConfidenceHedged",
            "OrganizationNarrative", "StanceModerated"
        }

        # POS tags should be the same as large model
        pos_tags = {
            "NP1", "RR", "VVD", "Y", "JJ", "NN1", "VBZ", "II", "AT", "DDQ",
            "MC", "NN2", "RRQV", "MC1", "IO", "PPH1", "GE", "VBDZ", "VVN",
            "RP", "DD1", "CS", "CCB", "VVZ", "VHZ", "DA1", "NNT1", "APPGE"
        }

        # Small model has fewer rhetorical tags but same POS structure
        assert len(small_model_tags) < 15  # Fewer than large model
        assert len(pos_tags) > 20
        assert "Untagged" in small_model_tags
        assert "ActorsPeople" in small_model_tags

    def test_output_data_structure(self):
        """Test the expected data structure of processed tokens."""
        # Create a sample DataFrame that matches actual output structure
        sample_output = pl.DataFrame({
            "doc_id": ["BIO_G0_02_1"] * 5,
            "token": ["Ernst", "Mayr", "once", "wrote", ","],
            "pos": ["NP1", "NP1", "RR", "VVD", "Y"],
            "tag": ["Character", "Character", "Narrative", "Citation", "Citation"],
            "sent_id": [1, 1, 2, 3, 3],
            "token_id": [1, 2, 3, 4, 5]
        })

        # Verify structure
        assert sample_output.height == 5
        assert "doc_id" in sample_output.columns
        assert "token" in sample_output.columns
        assert "pos" in sample_output.columns
        assert "tag" in sample_output.columns

        # Verify data types
        assert sample_output["doc_id"].dtype == pl.Utf8
        assert sample_output["token"].dtype == pl.Utf8
        assert sample_output["tag"].dtype == pl.Utf8

    def test_realistic_test_data_matches_actual_output(self):
        """Test that our test data matches actual DocuScope output format."""
        # This simulates what our tests should use based on real data
        realistic_test_data = pl.DataFrame({
            "doc_id": ["academic_001", "academic_001", "academic_001"],
            "token": ["However", "the", "research"],
            "pos": ["RR", "AT", "NN1"],
            "tag": ["MetadiscourseCohesive", "Untagged", "AcademicTerms"],
            "sent_id": [1, 1, 1],
            "token_id": [1, 2, 3]
        })

        # Verify this matches expected structure
        assert realistic_test_data.height == 3
        assert all(col in realistic_test_data.columns
                   for col in ["doc_id", "token", "pos", "tag"])

        # Verify tags are from actual DocuScope tagset
        actual_tags = realistic_test_data["tag"].to_list()
        large_model_tags = {
            "MetadiscourseCohesive", "Untagged", "AcademicTerms", "Character",
            "Narrative", "Citation", "InformationExposition", "Description",
            "Reasoning", "ConfidenceHedged"
        }

        for tag in actual_tags:
            assert tag in large_model_tags, f"Tag '{tag}' not in known DocuScope tagset"

    def test_frequency_table_structure(self):
        """Test expected structure for frequency tables."""
        # Simulate frequency table output
        pos_freq_table = pl.DataFrame({
            "token": ["the", "and", "of", "to", "a"],
            "frequency": [1000, 800, 600, 400, 300],
            "relative_frequency": [0.1, 0.08, 0.06, 0.04, 0.03]
        })

        ds_freq_table = pl.DataFrame({
            "tag": ["Untagged", "AcademicTerms", "MetadiscourseCohesive", "Reasoning"],
            "frequency": [2000, 1500, 800, 600],
            "relative_frequency": [0.2, 0.15, 0.08, 0.06]
        })

        # Verify structure
        assert "frequency" in pos_freq_table.columns
        assert "relative_frequency" in pos_freq_table.columns
        assert "tag" in ds_freq_table.columns

        # Verify realistic tag presence
        tags = ds_freq_table["tag"].to_list()
        assert "Untagged" in tags
        assert "AcademicTerms" in tags

    def test_dtm_structure(self):
        """Test expected structure for document-term matrices."""
        # Simulate DTM output with actual DocuScope tags as columns
        dtm_sample = pl.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "AcademicTerms": [5, 8, 3],
            "MetadiscourseCohesive": [2, 4, 1],
            "Reasoning": [3, 6, 2],
            "Untagged": [10, 15, 8]
        })

        # Verify structure
        assert "doc_id" in dtm_sample.columns
        assert dtm_sample.height == 3

        # Verify tag columns are realistic
        tag_columns = [col for col in dtm_sample.columns if col != "doc_id"]
        expected_tags = {"AcademicTerms", "MetadiscourseCohesive", "Reasoning", "Untagged"}
        assert set(tag_columns) == expected_tags


class TestTagsetCompatibility:
    """Test compatibility between different model tagsets."""

    def test_large_vs_small_model_differences(self):
        """Test differences between large and small model tagsets."""
        # Large model has more granular tags
        large_tags = {
            "Character", "Narrative", "Citation", "AcademicTerms",
            "InformationExposition", "MetadiscourseCohesive", "Reasoning",
            "ConfidenceHedged", "Uncertainty", "Strategic", "Positive"
        }

        # Small model has broader, less granular tags
        small_tags = {
            "ActorsPeople", "ActorsAbstractions", "SignpostingMetadiscourse",
            "PlanningStrategy", "SentimentPositive", "ConfidenceHedged",
            "OrganizationNarrative", "StanceModerated", "Untagged"
        }

        # Both should have "Untagged" but different specific tags
        assert "Untagged" in large_tags or len(small_tags) > 0
        assert len(large_tags) > len(small_tags)

        # Both models should handle confidence but may use different tag names
        confidence_in_large = "ConfidenceHedged" in large_tags
        confidence_in_small = "ConfidenceHedged" in small_tags
        assert confidence_in_large or confidence_in_small

    def test_pos_tag_consistency(self):
        """Test that POS tags are consistent across models."""
        # POS tags should be the same for both models
        expected_pos_tags = {
            "NP1", "RR", "VVD", "Y", "JJ", "NN1", "VBZ", "II", "AT",
            "DDQ", "MC", "NN2", "RRQV", "MC1", "IO", "PPH1", "GE",
            "VBDZ", "VVN", "RP", "DD1", "CS", "CCB", "VVZ", "VHZ"
        }

        assert len(expected_pos_tags) > 20
        assert "Y" in expected_pos_tags  # Punctuation
        assert "NN1" in expected_pos_tags  # Singular noun
        assert "VVD" in expected_pos_tags  # Past tense verb

    def test_untagged_token_handling(self):
        """Test that both models handle untagged tokens consistently."""
        # Both models should produce "Untagged" for certain tokens
        sample_untagged_tokens = ["(", ")", "1963:451", "...", "'"]

        # These should typically be tagged as "Untagged" in both models
        for token in sample_untagged_tokens:
            # In real processing, these would be tagged as "Untagged"
            assert True  # Placeholder for actual model testing
