"""
Script to create test fixture files.
Run this from the project root: python tests/fixtures/create_fixtures.py
"""

import pandas as pd
import json
from pathlib import Path

def create_sample_data():
    """Create sample_data.pkl for testing."""
    data = {
        'nct_id': [
            'NCT00000001',
            'NCT00000002',
            'NCT00000003',
            'NCT00000004',
            'NCT00000005'
        ],
        'conditions': [
            ['Type 2 Diabetes Mellitus (T2DM)', 'Obesity'],
            ['Healthy', 'Quality of Life'],
            ['Diabetes Mellitus, Type 2', 'Hypothalamic Obesity'],
            ['Diabetes Mellitus, Type 2', 'Obesity'],
            []
        ],
        'matched_conditions': [
            None,
            None,
            ['Diabetes Mellitus, Type 2 (MeSH ID:68003924)'],
            ['Diabetes Mellitus, Type 2 (MeSH ID:68003924)', 'Obesity (MeSH ID:68009765)'],
            None
        ]
    }

    df = pd.DataFrame(data)

    # Save to pickle
    output_path = Path(__file__).parent / 'sample_data.pkl'
    df.to_pickle(output_path)
    print(f"Created {output_path}")
    return df


def create_mock_mesh_mappings():
    """Create mock_mesh_mappings.csv for testing."""
    mappings = {
        'Diabetes Mellitus, Type 2': 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
        'Obesity': 'Obesity (MeSH ID:68009765)',
        'Cardiovascular Disease': 'Cardiovascular Diseases (MeSH ID:68002318)',
        'Diabetes Mellitus': 'Diabetes Mellitus (MeSH ID:68003920)',
    }

    df = pd.DataFrame(list(mappings.items()), columns=['condition', 'mesh_term'])
    df.set_index('condition', inplace=True)

    # Save to CSV
    output_path = Path(__file__).parent / 'mock_mesh_mappings.csv'
    df.to_csv(output_path)
    print(f"Created {output_path}")


def create_expected_results():
    """Create expected_results.json for validation."""
    expected = {
        'extract_unmapped_conditions': [
            'Type 2 Diabetes Mellitus (T2DM)',
            'Obesity',
            'Healthy',
            'Quality of Life',
            'Hypothalamic Obesity'
        ],
        'extract_existing_mesh_terms': [
            'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
            'Obesity (MeSH ID:68009765)'
        ],
        'full_workflow': {
            'new_mappings': 2,
            'trials_affected': 2,
            'mappings_example': {
                'Type 2 Diabetes Mellitus (T2DM)': 'Diabetes Mellitus, Type 2 (MeSH ID:68003924)',
                'Hypothalamic Obesity': 'Obesity (MeSH ID:68009765)'
            }
        }
    }

    output_path = Path(__file__).parent / 'expected_results.json'
    with open(output_path, 'w') as f:
        json.dump(expected, f, indent=2)
    print(f"Created {output_path}")


if __name__ == '__main__':
    print("Creating test fixtures...")
    create_sample_data()
    create_mock_mesh_mappings()
    create_expected_results()
    print("All fixtures created successfully!")
