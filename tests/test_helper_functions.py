"""
Unit tests for helper functions in Annotator/tools.py
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add Annotator directory to path
sys.path.append(str(Path(__file__).parent.parent / 'Annotator'))

from tools import (
    _extract_unmapped_conditions,
    _extract_existing_mesh_terms,
    _add_mesh_mappings,
    _filter_medical_conditions,
    _map_conditions_to_mesh,
    _extract_condition_from_context,
    _search_for_mesh_term,
)


class TestExtractUnmappedConditions:
    """Tests for _extract_unmapped_conditions function."""

    def test_fully_unmapped_trials(self, sample_dataframe):
        """Test extraction from trials with no matched conditions."""
        # Filter to only unmapped trials
        unmapped_df = sample_dataframe[sample_dataframe['matched_conditions'].isna()]

        result = _extract_unmapped_conditions(unmapped_df)

        assert isinstance(result, list)
        assert 'Type 2 Diabetes Mellitus (T2DM)' in result
        assert 'Obesity' in result
        assert 'Healthy' in result
        assert 'Quality of Life' in result

    def test_partially_mapped_trials(self, sample_dataframe):
        """Test that partially mapped trials are not included (only fully unmapped)."""
        result = _extract_unmapped_conditions(sample_dataframe)

        # Should only get conditions from rows with NaN/empty matched_conditions
        assert isinstance(result, list)
        assert len(result) > 0

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=['conditions', 'matched_conditions'])

        result = _extract_unmapped_conditions(empty_df)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_no_unmapped_trials(self):
        """Test DataFrame where all trials are fully mapped."""
        data = {
            'conditions': [['Diabetes Mellitus, Type 2']],
            'matched_conditions': [['Diabetes Mellitus, Type 2 (MeSH ID:68003924)']]
        }
        df = pd.DataFrame(data)

        result = _extract_unmapped_conditions(df)

        assert isinstance(result, list)
        assert len(result) == 0


class TestExtractExistingMeshTerms:
    """Tests for _extract_existing_mesh_terms function."""

    def test_extract_from_sample_data(self, sample_dataframe):
        """Test extraction from sample DataFrame."""
        result = _extract_existing_mesh_terms(sample_dataframe)

        assert isinstance(result, list)
        assert 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)' in result
        assert 'Obesity (MeSH ID:68009765)' in result

    def test_deduplication(self, sample_dataframe):
        """Test that duplicate MeSH terms are deduplicated."""
        result = _extract_existing_mesh_terms(sample_dataframe)

        # Diabetes appears in multiple trials, should only appear once
        diabetes_count = result.count('Diabetes Mellitus, Type 2 (MeSH ID:68003924)')
        assert diabetes_count == 1

    def test_empty_matched_conditions(self):
        """Test with DataFrame containing no matched conditions."""
        data = {
            'conditions': [['Diabetes'], ['Obesity']],
            'matched_conditions': [None, None]
        }
        df = pd.DataFrame(data)

        result = _extract_existing_mesh_terms(df)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_mixed_matched_conditions(self, sample_dataframe):
        """Test with mix of matched and unmapped trials."""
        result = _extract_existing_mesh_terms(sample_dataframe)

        assert len(result) == 2  # Only 2 unique MeSH terms in sample data


class TestAddMeshMappings:
    """Tests for _add_mesh_mappings function."""

    def test_normal_mapping(self, mock_mesh_mappings):
        """Test normal mapping application."""
        conditions = ['Type 2 Diabetes Mellitus (T2DM)', 'Hypothalamic Obesity']

        result = _add_mesh_mappings(conditions, mock_mesh_mappings)

        assert isinstance(result, list)
        assert 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)' in result
        assert 'Obesity (MeSH ID:68009765)' in result

    def test_empty_conditions(self, mock_mesh_mappings):
        """Test with empty conditions list."""
        result = _add_mesh_mappings([], mock_mesh_mappings)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_none_conditions(self, mock_mesh_mappings):
        """Test with None conditions."""
        result = _add_mesh_mappings(None, mock_mesh_mappings)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_unmapped_conditions(self, mock_mesh_mappings):
        """Test with conditions that have no mappings."""
        conditions = ['Unknown Disease', 'Made Up Condition']

        result = _add_mesh_mappings(conditions, mock_mesh_mappings)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_mixed_mapped_unmapped(self, mock_mesh_mappings):
        """Test with mix of mapped and unmapped conditions."""
        conditions = [
            'Type 2 Diabetes Mellitus (T2DM)',  # Mapped
            'Unknown Disease',  # Not mapped
            'Hypothalamic Obesity',  # Mapped
        ]

        result = _add_mesh_mappings(conditions, mock_mesh_mappings)

        assert len(result) == 2  # Only 2 mapped
        assert 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)' in result
        assert 'Obesity (MeSH ID:68009765)' in result

    def test_deduplication(self, mock_mesh_mappings):
        """Test that duplicate mappings are deduplicated."""
        # Both map to same MeSH term
        conditions = [
            'Type 2 Diabetes Mellitus (T2DM)',
            'Non-Insulin Dependent Diabetes'
        ]

        result = _add_mesh_mappings(conditions, mock_mesh_mappings)

        # Should only return one instance of the MeSH term
        assert len(result) == 1
        assert result[0] == 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)'


class TestFilterMedicalConditions:
    """Tests for _filter_medical_conditions function (with mocked GPT-5)."""

    def test_filter_with_mock_response(
        self,
        sample_unmapped_conditions,
        mock_gpt5_filter_response
    ):
        """Test filtering with mocked GPT-5 response."""
        # Mock the LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_gpt5_filter_response

        result = _filter_medical_conditions(sample_unmapped_conditions, mock_llm)

        assert isinstance(result, list)
        assert 'Type 2 Diabetes Mellitus (T2DM)' in result
        assert 'Obesity' in result
        assert 'Healthy' not in result
        assert 'Quality of Life' not in result

    def test_empty_input(self):
        """Test with empty condition list."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(
            medical_conditions=[],
            non_medical_items=[]
        )

        result = _filter_medical_conditions([], mock_llm)

        assert isinstance(result, list)
        assert len(result) == 0


class TestMapConditionsToMesh:
    """Tests for _map_conditions_to_mesh function (with mocked GPT-5)."""

    def test_successful_mapping(
        self,
        existing_mesh_terms,
        mock_gpt5_mapping_response
    ):
        """Test successful mapping with valid MeSH terms."""
        unmapped = ['Type 2 Diabetes Mellitus (T2DM)', 'Hypothalamic Obesity']

        # Mock the LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_gpt5_mapping_response

        result = _map_conditions_to_mesh(unmapped, existing_mesh_terms, mock_llm)

        assert isinstance(result, dict)
        assert 'Type 2 Diabetes Mellitus (T2DM)' in result
        assert result['Type 2 Diabetes Mellitus (T2DM)'] == 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)'

    def test_validation_rejects_invalid_mesh(
        self,
        existing_mesh_terms,
        mock_gpt5_invalid_mapping_response
    ):
        """Test that invalid MeSH terms are rejected."""
        unmapped = ['Type 2 Diabetes Mellitus (T2DM)', 'Hypothalamic Obesity']

        # Mock the LLM to return one valid and one invalid mapping
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_gpt5_invalid_mapping_response

        result = _map_conditions_to_mesh(unmapped, existing_mesh_terms, mock_llm)

        # Only the valid mapping should be included
        assert 'Type 2 Diabetes Mellitus (T2DM)' in result
        assert 'Hypothalamic Obesity' not in result  # Invalid MeSH term

    def test_empty_mappings(self, existing_mesh_terms):
        """Test when GPT-5 returns no mappings."""
        unmapped = ['Unknown Condition']

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(mappings=[])

        result = _map_conditions_to_mesh(unmapped, existing_mesh_terms, mock_llm)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_all_invalid_mesh_terms(self, existing_mesh_terms):
        """Test when all GPT-5 mappings are invalid."""
        from pydantic import BaseModel, Field

        # Define schemas locally (same as in conftest.py)
        class ConditionMapping(BaseModel):
            original_condition: str
            mesh_term: str
            confidence: str

        class ConditionMappings(BaseModel):
            mappings: list[ConditionMapping]

        unmapped = ['Condition 1', 'Condition 2']

        # Mock response with all invalid MeSH terms
        invalid_response = ConditionMappings(
            mappings=[
                ConditionMapping(
                    original_condition='Condition 1',
                    mesh_term='Invalid Term (MeSH ID:99999999)',
                    confidence='low'
                ),
                ConditionMapping(
                    original_condition='Condition 2',
                    mesh_term='Another Invalid (MeSH ID:88888888)',
                    confidence='low'
                ),
            ]
        )

        mock_llm = Mock()
        mock_llm.invoke.return_value = invalid_response

        result = _map_conditions_to_mesh(unmapped, existing_mesh_terms, mock_llm)

        assert isinstance(result, dict)
        assert len(result) == 0  # All should be rejected


class TestExtractConditionFromContext:
    """Tests for _extract_condition_from_context function."""

    def test_successful_extraction(self, mock_trial_row_complete, mock_gpt5_extraction_response):
        """Test successful condition extraction."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_gpt5_extraction_response

        result = _extract_condition_from_context(mock_trial_row_complete, mock_llm)

        assert result == 'Type 2 Diabetes Mellitus'
        mock_llm.invoke.assert_called_once()

    def test_llm_invocation_with_prompt(self, mock_trial_row_complete):
        """Test that LLM is invoked with correct prompt containing trial data."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(extracted_condition='Diabetes')

        _extract_condition_from_context(mock_trial_row_complete, mock_llm)

        # Verify LLM was called
        assert mock_llm.invoke.call_count == 1
        # Verify prompt includes trial data
        call_args = mock_llm.invoke.call_args[0][0]
        assert 'Type 2 Diabetes' in call_args or 'Diabetes' in call_args

    def test_extraction_returns_condition_field(self, mock_trial_row_minimal):
        """Test that function returns extracted_condition field."""
        mock_response = Mock()
        mock_response.extracted_condition = 'Obesity'
        mock_response.confidence = 'high'
        mock_response.reasoning = 'Clear from title'

        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_response

        result = _extract_condition_from_context(mock_trial_row_minimal, mock_llm)

        assert result == 'Obesity'

    def test_handles_missing_optional_fields(self):
        """Test extraction with missing optional fields (uses getattr defaults)."""
        # Create a minimal row with missing fields using spec to prevent auto-creation
        mock_row = Mock(spec=['brief_title', 'official_title', 'brief_summary'])
        mock_row.brief_title = 'Test Study'
        mock_row.official_title = 'Official Test'
        mock_row.brief_summary = 'Brief summary'
        # Missing: detailed_description, primary_outcomes, conditions (getattr will return default)

        mock_llm = Mock()
        mock_llm.invoke.return_value = Mock(extracted_condition='Test Condition')

        result = _extract_condition_from_context(mock_row, mock_llm)

        # Should not crash and return condition
        assert result == 'Test Condition'
        # Verify prompt was built with "Not Available" for missing fields
        call_args = mock_llm.invoke.call_args[0][0]
        assert 'Not Available' in call_args


class TestSearchForMeshTerm:
    """Tests for _search_for_mesh_term function."""

    @patch('tools.mesh_mapper')
    def test_finds_valid_mesh_term(self, mock_mesh_mapper, mock_mesh_mapper_success_response):
        """Test finding a valid MeSH term."""
        mock_mesh_mapper.search_mesh_term.return_value = mock_mesh_mapper_success_response

        result = _search_for_mesh_term('Type 2 Diabetes')

        assert result == 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)'
        mock_mesh_mapper.search_mesh_term.assert_called_once_with('Type 2 Diabetes', filter_diseases_only=True)

    @patch('tools.mesh_mapper')
    def test_returns_not_determined_when_no_match(self, mock_mesh_mapper):
        """Test return value when no match found."""
        mock_mesh_mapper.search_mesh_term.return_value = None

        result = _search_for_mesh_term('Unknown Condition')

        assert result == 'NOT DETERMINED'

    @patch('tools.mesh_mapper')
    def test_formats_mesh_id_correctly(self, mock_mesh_mapper):
        """Test MeSH ID formatting."""
        mock_mesh_mapper.search_mesh_term.return_value = {
            'mesh_term': 'Obesity',
            'mesh_id': '68009765',
            'tree_numbers': ['C23'],
            'categories': ['C'],
            'is_disease': True
        }

        result = _search_for_mesh_term('Obesity')

        assert result == 'Obesity (MeSH ID:68009765)'
        assert '(MeSH ID:' in result
        assert ')' in result

    @patch('tools.mesh_mapper')
    def test_uses_diseases_filter(self, mock_mesh_mapper):
        """Test that diseases filter is applied."""
        mock_mesh_mapper.search_mesh_term.return_value = None

        _search_for_mesh_term('Test')

        # Verify filter_diseases_only=True is passed
        mock_mesh_mapper.search_mesh_term.assert_called_with('Test', filter_diseases_only=True)

    @patch('tools.mesh_mapper')
    def test_handles_exception(self, mock_mesh_mapper):
        """Test exception handling."""
        mock_mesh_mapper.search_mesh_term.side_effect = Exception("API Error")

        result = _search_for_mesh_term('Test Condition')

        assert result == 'NOT DETERMINED'

    @patch('tools.mesh_mapper')
    def test_logs_success(self, mock_mesh_mapper, mock_mesh_mapper_success_response):
        """Test that success is logged."""
        mock_mesh_mapper.search_mesh_term.return_value = mock_mesh_mapper_success_response

        with patch('tools.logger') as mock_logger:
            _search_for_mesh_term('Diabetes')

            # Verify info log was called
            mock_logger.info.assert_called()

    @patch('tools.mesh_mapper')
    def test_logs_not_found(self, mock_mesh_mapper):
        """Test that not found is logged as warning."""
        mock_mesh_mapper.search_mesh_term.return_value = None

        with patch('tools.logger') as mock_logger:
            _search_for_mesh_term('Unknown')

            # Verify warning log was called
            mock_logger.warning.assert_called()
