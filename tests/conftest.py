"""
Pytest configuration and fixtures for Annotator tool tests.
"""

import pytest
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List


# Pydantic schemas (matching those in tools.py)
class MedicalConditionFilter(BaseModel):
    medical_conditions: list[str]
    non_medical_items: list[str]


class ConditionMapping(BaseModel):
    original_condition: str
    mesh_term: str
    confidence: str


class ConditionMappings(BaseModel):
    mappings: list[ConditionMapping]


class ConditionExtraction(BaseModel):
    """Schema for extracting condition from trial context."""
    extracted_condition: str
    confidence: str
    reasoning: str


@pytest.fixture
def sample_dataframe():
    """
    Create a minimal DataFrame mimicking cleaned_trials.pkl structure.

    Includes:
    - Fully unmapped trials (NaN matched_conditions)
    - Partially mapped trials (some conditions mapped)
    - Fully mapped trials (all conditions mapped)
    """
    data = {
        'nct_id': [
            'NCT00000001',  # Fully unmapped
            'NCT00000002',  # Fully unmapped
            'NCT00000003',  # Partially mapped
            'NCT00000004',  # Fully mapped
            'NCT00000005',  # Empty conditions
        ],
        'conditions': [
            ['Type 2 Diabetes Mellitus (T2DM)', 'Obesity'],  # Unmapped
            ['Healthy', 'Quality of Life'],  # Non-medical conditions
            ['Diabetes Mellitus, Type 2', 'Hypothalamic Obesity'],  # Partial
            ['Diabetes Mellitus, Type 2', 'Obesity'],  # Fully mapped
            [],  # Empty
        ],
        'matched_conditions': [
            [],  # Unmapped
            [],  # Unmapped
            ['Diabetes Mellitus, Type 2 (MeSH ID:68003924)'],  # Partial
            ['Diabetes Mellitus, Type 2 (MeSH ID:68003924)', 'Obesity (MeSH ID:68009765)'],  # Full
            [],  # Empty
        ]
    }

    df = pd.DataFrame(data)

    # Convert empty lists to proper format to match real data
    df.loc[df['matched_conditions'].apply(len) == 0, 'matched_conditions'] = None

    return df


@pytest.fixture
def existing_mesh_terms():
    """Provide a list of existing MeSH terms for testing."""
    return [
        'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
        'Obesity (MeSH ID:68009765)',
        'Cardiovascular Diseases (MeSH ID:68002318)',
        'Diabetes Mellitus (MeSH ID:68003920)',
    ]


@pytest.fixture
def mock_mesh_mappings():
    """Provide sample MeSH term mappings dictionary."""
    return {
        'Type 2 Diabetes Mellitus (T2DM)': 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
        'Non-Insulin Dependent Diabetes': 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
        'Hypothalamic Obesity': 'Obesity (MeSH ID:68009765)',
    }


@pytest.fixture
def mock_gpt5_filter_response():
    """
    Mock GPT-5 response for filtering medical conditions.

    Returns a MedicalConditionFilter object.
    """
    return MedicalConditionFilter(
        medical_conditions=[
            'Type 2 Diabetes Mellitus (T2DM)',
            'Obesity',
            'Hypothalamic Obesity'
        ],
        non_medical_items=[
            'Healthy',
            'Quality of Life'
        ]
    )


@pytest.fixture
def mock_gpt5_mapping_response():
    """
    Mock GPT-5 response for mapping conditions to MeSH terms.

    Returns a ConditionMappings object.
    """
    return ConditionMappings(
        mappings=[
            ConditionMapping(
                original_condition='Type 2 Diabetes Mellitus (T2DM)',
                mesh_term='Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
                confidence='high'
            ),
            ConditionMapping(
                original_condition='Hypothalamic Obesity',
                mesh_term='Obesity (MeSH ID:68009765)',
                confidence='medium'
            ),
        ]
    )


@pytest.fixture
def mock_gpt5_invalid_mapping_response():
    """
    Mock GPT-5 response with invalid MeSH terms (not in existing list).

    Tests validation logic.
    """
    return ConditionMappings(
        mappings=[
            ConditionMapping(
                original_condition='Type 2 Diabetes Mellitus (T2DM)',
                mesh_term='Diabetes Mellitus, Type 2 (MeSH ID:68003924)',  # Valid
                confidence='high'
            ),
            ConditionMapping(
                original_condition='Hypothalamic Obesity',
                mesh_term='Obesity, Hypothalamic (MeSH ID:99999999)',  # Invalid!
                confidence='medium'
            ),
        ]
    )


@pytest.fixture
def temp_mesh_csv(tmp_path):
    """
    Create a temporary mesh_term_mappings.csv file for testing.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path to temporary CSV file
    """
    csv_path = tmp_path / "mesh_term_mappings.csv"

    # Create initial mappings
    df = pd.DataFrame({
        'condition': [
            'Diabetes Mellitus, Type 2',
            'Obesity',
            'Cardiovascular Disease',
        ],
        'mesh_term': [
            'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
            'Obesity (MeSH ID:68009765)',
            'Cardiovascular Diseases (MeSH ID:68002318)',
        ]
    })
    df.set_index('condition', inplace=True)
    df.to_csv(csv_path)

    return csv_path


@pytest.fixture
def sample_unmapped_conditions():
    """List of unmapped conditions for testing extraction."""
    return [
        'Type 2 Diabetes Mellitus (T2DM)',
        'Hypothalamic Obesity',
        'Non-Insulin Dependent Diabetes',
        'Healthy',
        'Quality of Life',
    ]


# Fixtures for search_for_condition_mapping tool

@pytest.fixture
def mock_trial_row_complete():
    """Complete trial row with all fields for _build_trial_context testing."""
    return pd.Series({
        'nct_id': 'NCT12345678',
        'brief_title': 'Study of Drug X in Type 2 Diabetes',
        'official_title': 'A Randomized, Double-Blind Study of Drug X in Patients with Type 2 Diabetes Mellitus',
        'brief_summary': 'This study investigates the efficacy of Drug X in patients with poorly controlled type 2 diabetes mellitus.',
        'detailed_description': 'Type 2 diabetes mellitus is a chronic metabolic disorder. This study will evaluate Drug X in 200 patients over 24 weeks.',
        'primary_outcomes': ['HbA1c change from baseline', 'Fasting glucose levels'],
        'conditions': ['Type 2 Diabetes Mellitus', 'Hyperglycemia'],
        'matched_conditions': None
    })


@pytest.fixture
def mock_trial_row_minimal():
    """Minimal trial row with only required fields."""
    return pd.Series({
        'nct_id': 'NCT87654321',
        'brief_title': 'Obesity Treatment Study',
        'official_title': 'Efficacy of Treatment in Obese Patients',
        'brief_summary': 'A study of obesity treatment interventions.',
        'detailed_description': None,  # Missing
        'primary_outcomes': None,  # Missing
        'conditions': ['Obesity'],
        'matched_conditions': None
    })


@pytest.fixture
def sample_dataframe_for_search():
    """DataFrame with trials needing search_for_condition_mapping."""
    data = {
        'nct_id': ['NCT00001', 'NCT00002', 'NCT00003'],
        'brief_title': [
            'Type 2 Diabetes Study',
            'Obesity Intervention Trial',
            'Quality of Life Assessment'
        ],
        'official_title': [
            'A Study of Type 2 Diabetes Mellitus Treatment',
            'Weight Loss Intervention in Obese Patients',
            'Assessment of Quality of Life in Healthy Volunteers'
        ],
        'brief_summary': [
            'This study evaluates treatment for type 2 diabetes.',
            'This study examines weight loss in obesity.',
            'This study assesses quality of life measures.'
        ],
        'detailed_description': [
            'Type 2 diabetes is a chronic disease...',
            None,
            None
        ],
        'primary_outcomes': [
            ['HbA1c reduction'],
            ['Weight loss percentage'],
            ['QOL score']
        ],
        'conditions': [
            ['T2DM'],  # Unclear abbreviation
            [],  # Empty
            ['Healthy']  # Non-medical
        ],
        'matched_conditions': [None, None, None]  # All unmapped
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_gpt5_extraction_response():
    """Mock GPT-5 ConditionExtraction response."""
    return ConditionExtraction(
        extracted_condition='Type 2 Diabetes Mellitus',
        confidence='high',
        reasoning='The trial clearly focuses on type 2 diabetes treatment based on title and description'
    )


@pytest.fixture
def mock_mesh_mapper_success_response():
    """Mock successful mesh_mapper.search_mesh_term() response."""
    return {
        'mesh_term': 'Diabetes Mellitus, Type 2',
        'mesh_id': '68003924',
        'tree_numbers': ['C18.452.394.750'],
        'categories': ['C'],
        'is_disease': True
    }


@pytest.fixture
def mock_mesh_mapper_not_found_response():
    """Mock mesh_mapper.search_mesh_term() returning None (no match)."""
    return None
