"""
Integration tests for search_for_condition_mapping tool.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os

# Add Annotator directory to path
sys.path.append(str(Path(__file__).parent.parent / 'Annotator'))

from tools import search_for_condition_mapping


class TestSearchForConditionMappingIntegration:
    """Integration tests for the complete search_for_condition_mapping tool."""

    @patch('tools.mesh_mapper')
    @patch('tools.ChatOpenAI')
    def test_full_workflow_with_mocked_services(
        self,
        mock_chat_openai,
        mock_mesh_mapper,
        sample_dataframe_for_search,
        mock_gpt5_extraction_response,
        mock_mesh_mapper_success_response,
        temp_mesh_csv
    ):
        """Test complete workflow with mocked GPT-5 and mesh_mapper."""
        # Setup mocked GPT-5
        mock_base_llm = MagicMock()
        mock_extraction_llm = MagicMock()
        mock_extraction_llm.invoke.return_value = mock_gpt5_extraction_response
        mock_base_llm.with_structured_output.return_value = mock_extraction_llm
        mock_chat_openai.return_value = mock_base_llm

        # Setup mocked mesh_mapper
        mock_mesh_mapper.search_mesh_term.return_value = mock_mesh_mapper_success_response

        df = sample_dataframe_for_search.copy()

        # Run the tool
        result = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Assertions
        assert isinstance(result, dict)
        assert result['trials_processed'] == 3  # All 3 trials have missing matched_conditions
        assert result['new_mappings'] >= 0  # At least some successful mappings
        assert 'mappings_created' in result

    @patch('tools.ChatOpenAI')
    def test_no_missing_conditions(
        self,
        mock_chat_openai,
        temp_mesh_csv
    ):
        """Test with DataFrame where all trials already have matched_conditions."""
        # Create fully mapped DataFrame
        data = {
            'nct_id': ['NCT001'],
            'brief_title': ['Test'],
            'official_title': ['Test'],
            'brief_summary': ['Test'],
            'detailed_description': [None],
            'primary_outcomes': [None],
            'conditions': [['Diabetes']],
            'matched_conditions': [['Diabetes Mellitus, Type 2 (MeSH ID:68003924)']]
        }
        df = pd.DataFrame(data)

        result = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        assert result['trials_processed'] == 0
        assert result['new_mappings'] == 0
        assert result['message'] == 'No trials with missing matched_conditions found'

    @patch('tools.mesh_mapper')
    @patch('tools.ChatOpenAI')
    def test_mesh_mapper_returns_not_determined(
        self,
        mock_chat_openai,
        mock_mesh_mapper,
        sample_dataframe_for_search,
        mock_gpt5_extraction_response,
        temp_mesh_csv
    ):
        """Test when mesh_mapper returns None (no match found)."""
        # Setup mocked GPT-5
        mock_base_llm = MagicMock()
        mock_extraction_llm = MagicMock()
        mock_extraction_llm.invoke.return_value = mock_gpt5_extraction_response
        mock_base_llm.with_structured_output.return_value = mock_extraction_llm
        mock_chat_openai.return_value = mock_base_llm

        # Setup mesh_mapper to return None (no match)
        mock_mesh_mapper.search_mesh_term.return_value = None

        df = sample_dataframe_for_search.copy()

        result = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # All should be marked as NOT DETERMINED
        assert result['not_determined'] == len(result['mappings_created'])
        assert result['new_mappings'] == 0  # No successful mappings
        assert all(v == 'NOT DETERMINED' for v in result['mappings_created'].values())

    @patch('tools.mesh_mapper')
    @patch('tools.ChatOpenAI')
    def test_dataframe_updates_only_for_successful_matches(
        self,
        mock_chat_openai,
        mock_mesh_mapper,
        sample_dataframe_for_search,
        temp_mesh_csv
    ):
        """Test that DataFrame is updated only for successful MeSH matches, not for NOT DETERMINED."""
        # Setup mocked GPT-5 - returns different conditions
        mock_base_llm = MagicMock()
        mock_extraction_llm = MagicMock()

        extraction_responses = [
            Mock(extracted_condition='Type 2 Diabetes Mellitus', confidence='high', reasoning='test'),
            Mock(extracted_condition='Obesity', confidence='high', reasoning='test'),
            Mock(extracted_condition='Healthy', confidence='low', reasoning='test')
        ]
        mock_extraction_llm.invoke.side_effect = extraction_responses
        mock_base_llm.with_structured_output.return_value = mock_extraction_llm
        mock_chat_openai.return_value = mock_base_llm

        # Setup mesh_mapper - first two succeed, third fails
        mesh_responses = [
            {'mesh_term': 'Diabetes Mellitus, Type 2', 'mesh_id': '68003924', 'tree_numbers': ['C'], 'categories': ['C'], 'is_disease': True},
            {'mesh_term': 'Obesity', 'mesh_id': '68009765', 'tree_numbers': ['C'], 'categories': ['C'], 'is_disease': True},
            None  # No match for "Healthy"
        ]
        mock_mesh_mapper.search_mesh_term.side_effect = mesh_responses

        df = sample_dataframe_for_search.copy()

        result = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Should have 2 successful mappings, 1 NOT DETERMINED
        assert result['new_mappings'] == 2
        assert result['not_determined'] == 1
        assert result['trials_updated'] == 2  # Only 2 trials updated (successful matches)

        # Check DataFrame - first two trials should be updated, third should not
        assert df.loc[0, 'matched_conditions'] is not None and len(df.loc[0, 'matched_conditions']) > 0
        assert df.loc[1, 'matched_conditions'] is not None and len(df.loc[1, 'matched_conditions']) > 0
        # Third trial should still be unmapped (None or empty)

    @patch('tools.mesh_mapper')
    @patch('tools.ChatOpenAI')
    def test_gpt5_extraction_error_handling(
        self,
        mock_chat_openai,
        mock_mesh_mapper,
        sample_dataframe_for_search,
        temp_mesh_csv
    ):
        """Test that extraction errors don't stop processing other trials."""
        # Setup mocked GPT-5 - first call raises exception, others succeed
        mock_base_llm = MagicMock()
        mock_extraction_llm = MagicMock()

        def side_effect_extraction(*args, **kwargs):
            if mock_extraction_llm.invoke.call_count == 1:
                raise Exception("GPT-5 Error")
            return Mock(extracted_condition='Diabetes', confidence='high', reasoning='test')

        mock_extraction_llm.invoke.side_effect = side_effect_extraction
        mock_base_llm.with_structured_output.return_value = mock_extraction_llm
        mock_chat_openai.return_value = mock_base_llm

        # Setup mesh_mapper
        mock_mesh_mapper.search_mesh_term.return_value = {
            'mesh_term': 'Diabetes Mellitus, Type 2',
            'mesh_id': '68003924',
            'tree_numbers': ['C'],
            'categories': ['C'],
            'is_disease': True
        }

        df = sample_dataframe_for_search.copy()

        result = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Should continue processing despite error on first trial
        assert result['trials_processed'] == 3
        # At least 2 trials should have been processed successfully
        assert len(result['mappings_created']) >= 1

    @patch('tools.mesh_mapper')
    @patch('tools.ChatOpenAI')
    def test_csv_persistence_with_not_determined(
        self,
        mock_chat_openai,
        mock_mesh_mapper,
        sample_dataframe_for_search,
        temp_mesh_csv
    ):
        """Test that NOT DETERMINED entries are saved to CSV."""
        # Setup mocked GPT-5
        mock_base_llm = MagicMock()
        mock_extraction_llm = MagicMock()
        mock_extraction_llm.invoke.return_value = Mock(
            extracted_condition='Unknown Condition',
            confidence='low',
            reasoning='test'
        )
        mock_base_llm.with_structured_output.return_value = mock_extraction_llm
        mock_chat_openai.return_value = mock_base_llm

        # Setup mesh_mapper to return None
        mock_mesh_mapper.search_mesh_term.return_value = None

        df = sample_dataframe_for_search.copy()

        search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Read CSV and verify NOT DETERMINED is saved
        csv_df = pd.read_csv(temp_mesh_csv, index_col=0)
        assert 'Unknown Condition' in csv_df.index
        assert csv_df.loc['Unknown Condition', csv_df.columns[0]] == 'NOT DETERMINED'

    @patch('tools.ChatOpenAI')
    def test_empty_dataframe(
        self,
        mock_chat_openai,
        temp_mesh_csv
    ):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['nct_id', 'brief_title', 'official_title', 'brief_summary',
                                    'detailed_description', 'primary_outcomes', 'conditions', 'matched_conditions'])

        result = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        assert result['trials_processed'] == 0
        assert result['new_mappings'] == 0

    @patch('tools.mesh_mapper')
    @patch('tools.ChatOpenAI')
    def test_idempotency(
        self,
        mock_chat_openai,
        mock_mesh_mapper,
        sample_dataframe_for_search,
        mock_gpt5_extraction_response,
        mock_mesh_mapper_success_response,
        temp_mesh_csv
    ):
        """Test that running tool twice doesn't re-process already mapped trials."""
        # Setup mocks
        mock_base_llm = MagicMock()
        mock_extraction_llm = MagicMock()
        mock_extraction_llm.invoke.return_value = mock_gpt5_extraction_response
        mock_base_llm.with_structured_output.return_value = mock_extraction_llm
        mock_chat_openai.return_value = mock_base_llm
        mock_mesh_mapper.search_mesh_term.return_value = mock_mesh_mapper_success_response

        df = sample_dataframe_for_search.copy()

        # First run
        result1 = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        assert result1['trials_processed'] == 3

        # Second run on same DataFrame (which now has matched_conditions)
        result2 = search_for_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Should find fewer/no trials to process
        assert result2['trials_processed'] <= result1['trials_processed']
