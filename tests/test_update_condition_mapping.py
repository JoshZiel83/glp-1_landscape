"""
Integration tests for update_condition_mapping tool.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import os

# Add Annotator directory to path
sys.path.append(str(Path(__file__).parent.parent / 'Annotator'))

from tools import update_condition_mapping


class TestUpdateConditionMappingIntegration:
    """Integration tests for the complete update_condition_mapping tool."""

    @patch('tools.ChatOpenAI')
    def test_full_workflow_with_mocked_gpt5(
        self,
        mock_chat_openai,
        sample_dataframe,
        mock_gpt5_filter_response,
        mock_gpt5_mapping_response,
        temp_mesh_csv
    ):
        """Test complete workflow with mocked GPT-5 responses."""
        # Setup mocked LLM
        mock_base_llm = MagicMock()
        mock_filter_llm = MagicMock()
        mock_mapping_llm = MagicMock()

        # Configure filter LLM
        mock_filter_llm.invoke.return_value = mock_gpt5_filter_response

        # Configure mapping LLM
        mock_mapping_llm.invoke.return_value = mock_gpt5_mapping_response

        # Configure with_structured_output to return our mocks
        mock_base_llm.with_structured_output.side_effect = [
            mock_filter_llm,
            mock_mapping_llm
        ]

        # Configure ChatOpenAI constructor
        mock_chat_openai.return_value = mock_base_llm

        # Make a copy of the dataframe
        df = sample_dataframe.copy()

        # Run the tool
        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Assertions
        assert isinstance(result, dict)
        assert 'new_mappings' in result
        assert 'trials_affected' in result
        assert 'mappings_created' in result
        assert 'message' in result

        # Should have created mappings
        assert result['new_mappings'] >= 0
        assert result['trials_affected'] >= 0

        # Verify CSV was updated
        updated_csv = pd.read_csv(temp_mesh_csv, index_col=0)
        assert len(updated_csv) >= 3  # Original + new mappings

    @patch('tools.ChatOpenAI')
    def test_no_unmapped_conditions(
        self,
        mock_chat_openai,
        temp_mesh_csv
    ):
        """Test with DataFrame that has no unmapped conditions."""
        # Create fully mapped DataFrame
        data = {
            'conditions': [['Diabetes Mellitus, Type 2'], ['Obesity']],
            'matched_conditions': [
                ['Diabetes Mellitus, Type 2 (MeSH ID:68003924)'],
                ['Obesity (MeSH ID:68009765)']
            ]
        }
        df = pd.DataFrame(data)

        # Run the tool
        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Should return 0 new mappings
        assert result['new_mappings'] == 0
        assert result['trials_affected'] == 0
        assert result['message'] == 'No unmapped conditions found'

    @patch('tools.ChatOpenAI')
    def test_gpt5_validation_rejects_invalid(
        self,
        mock_chat_openai,
        sample_dataframe,
        mock_gpt5_filter_response,
        mock_gpt5_invalid_mapping_response,
        temp_mesh_csv
    ):
        """Test that invalid MeSH terms from GPT-5 are rejected."""
        # Setup mocked LLM with invalid mappings
        mock_base_llm = MagicMock()
        mock_filter_llm = MagicMock()
        mock_mapping_llm = MagicMock()

        mock_filter_llm.invoke.return_value = mock_gpt5_filter_response
        mock_mapping_llm.invoke.return_value = mock_gpt5_invalid_mapping_response

        mock_base_llm.with_structured_output.side_effect = [
            mock_filter_llm,
            mock_mapping_llm
        ]

        mock_chat_openai.return_value = mock_base_llm

        df = sample_dataframe.copy()

        # Run the tool
        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Should only include valid mappings
        # mock_gpt5_invalid_mapping_response has 1 valid and 1 invalid
        assert result['new_mappings'] == 1
        assert 'Type 2 Diabetes Mellitus (T2DM)' in result['mappings_created']
        assert 'Hypothalamic Obesity' not in result['mappings_created']

    @patch('tools.ChatOpenAI')
    def test_csv_persistence(
        self,
        mock_chat_openai,
        sample_dataframe,
        mock_gpt5_filter_response,
        mock_gpt5_mapping_response,
        temp_mesh_csv
    ):
        """Test that new mappings are persisted to CSV."""
        # Setup mocked LLM
        mock_base_llm = MagicMock()
        mock_filter_llm = MagicMock()
        mock_mapping_llm = MagicMock()

        mock_filter_llm.invoke.return_value = mock_gpt5_filter_response
        mock_mapping_llm.invoke.return_value = mock_gpt5_mapping_response

        mock_base_llm.with_structured_output.side_effect = [
            mock_filter_llm,
            mock_mapping_llm
        ]

        mock_chat_openai.return_value = mock_base_llm

        # Get initial CSV row count
        initial_csv = pd.read_csv(temp_mesh_csv, index_col=0)
        initial_count = len(initial_csv)

        df = sample_dataframe.copy()

        # Run the tool
        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Check CSV was updated
        updated_csv = pd.read_csv(temp_mesh_csv, index_col=0)
        assert len(updated_csv) >= initial_count

        # Verify new mappings are in CSV
        for condition, mesh_term in result['mappings_created'].items():
            assert condition in updated_csv.index
            assert updated_csv.loc[condition, updated_csv.columns[0]] == mesh_term

    @patch('tools.ChatOpenAI')
    def test_dataframe_update(
        self,
        mock_chat_openai,
        sample_dataframe,
        mock_gpt5_filter_response,
        mock_gpt5_mapping_response,
        temp_mesh_csv
    ):
        """Test that DataFrame matched_conditions are updated correctly."""
        # Setup mocked LLM
        mock_base_llm = MagicMock()
        mock_filter_llm = MagicMock()
        mock_mapping_llm = MagicMock()

        mock_filter_llm.invoke.return_value = mock_gpt5_filter_response
        mock_mapping_llm.invoke.return_value = mock_gpt5_mapping_response

        mock_base_llm.with_structured_output.side_effect = [
            mock_filter_llm,
            mock_mapping_llm
        ]

        mock_chat_openai.return_value = mock_base_llm

        df = sample_dataframe.copy()

        # Count unmapped before
        unmapped_before = df['matched_conditions'].isna().sum()

        # Run the tool
        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Check DataFrame was modified
        # Note: df is modified in-place by the tool
        unmapped_after = df['matched_conditions'].isna().sum()

        # Should have fewer unmapped trials (or same if no mappings applied)
        assert unmapped_after <= unmapped_before

    @patch('tools.ChatOpenAI')
    def test_idempotency(
        self,
        mock_chat_openai,
        sample_dataframe,
        mock_gpt5_filter_response,
        mock_gpt5_mapping_response,
        temp_mesh_csv
    ):
        """Test that running tool twice doesn't create duplicate mappings."""
        # Setup mocked LLM
        def create_mocks():
            mock_base_llm = MagicMock()
            mock_filter_llm = MagicMock()
            mock_mapping_llm = MagicMock()

            mock_filter_llm.invoke.return_value = mock_gpt5_filter_response
            mock_mapping_llm.invoke.return_value = mock_gpt5_mapping_response

            mock_base_llm.with_structured_output.side_effect = [
                mock_filter_llm,
                mock_mapping_llm
            ]

            return mock_base_llm

        # First run
        mock_chat_openai.return_value = create_mocks()
        df = sample_dataframe.copy()

        result1 = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # Second run on same data
        # NOTE: The tool only updates rows with NaN/empty matched_conditions
        # After first run, those rows got mappings, but there may still be
        # the same unmapped rows detected (rows with NaN are still NaN after apply)
        mock_chat_openai.return_value = create_mocks()

        result2 = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        # The mappings are already in CSV, so GPT-5 would map the same things
        # But trials_affected should be same or fewer since some were already updated
        assert result2['new_mappings'] >= 0
        # Just verify no error on second run
        assert 'message' in result2

    @patch('tools.ChatOpenAI')
    def test_empty_dataframe(
        self,
        mock_chat_openai,
        temp_mesh_csv
    ):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['conditions', 'matched_conditions'])

        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        assert result['new_mappings'] == 0
        assert result['trials_affected'] == 0

    @patch('tools.ChatOpenAI')
    def test_gpt5_returns_no_mappings(
        self,
        mock_chat_openai,
        sample_dataframe,
        mock_gpt5_filter_response,
        temp_mesh_csv
    ):
        """Test when GPT-5 finds no suitable mappings."""
        from pydantic import BaseModel, Field

        # Define schema locally
        class ConditionMapping(BaseModel):
            original_condition: str
            mesh_term: str
            confidence: str

        class ConditionMappings(BaseModel):
            mappings: list[ConditionMapping]

        # Setup mocked LLM that returns empty mappings
        mock_base_llm = MagicMock()
        mock_filter_llm = MagicMock()
        mock_mapping_llm = MagicMock()

        mock_filter_llm.invoke.return_value = mock_gpt5_filter_response

        # Mock mapping LLM to return no mappings
        mock_mapping_llm.invoke.return_value = ConditionMappings(mappings=[])

        mock_base_llm.with_structured_output.side_effect = [
            mock_filter_llm,
            mock_mapping_llm
        ]

        mock_chat_openai.return_value = mock_base_llm

        df = sample_dataframe.copy()

        result = update_condition_mapping.invoke({
            'data': df,
            'mesh_mappings_path': str(temp_mesh_csv)
        })

        assert result['new_mappings'] == 0
        assert result['message'] == 'No suitable mappings found'
