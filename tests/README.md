# Tests for Annotator Tools

Comprehensive test suite for the `update_condition_mapping` tool and its helper functions.

## Structure

```
tests/
├── __init__.py
├── conftest.py                      # Pytest fixtures and configuration
├── test_helper_functions.py         # Unit tests for helper functions
├── test_update_condition_mapping.py # Integration tests for main tool
├── fixtures/
│   ├── create_fixtures.py           # Script to generate test data
│   ├── sample_data.pkl              # Sample DataFrame for testing
│   ├── mock_mesh_mappings.csv       # Sample MeSH mappings
│   └── expected_results.json        # Expected test outputs
└── README.md                        # This file
```

## Setup

### Install Test Dependencies

```bash
poetry add --group dev pytest pytest-mock pytest-cov
```

### Generate Test Fixtures

Before running tests, generate the fixture data files:

```bash
cd /Users/joshziel/Documents/Coding/glp-1-landscape
poetry run python tests/fixtures/create_fixtures.py
```

## Running Tests

### Run All Tests

```bash
poetry run pytest tests/
```

### Run Specific Test File

```bash
# Unit tests
poetry run pytest tests/test_helper_functions.py

# Integration tests
poetry run pytest tests/test_update_condition_mapping.py
```

### Run with Coverage

```bash
poetry run pytest --cov=Annotator tests/
```

### Run with Verbose Output

```bash
poetry run pytest -v tests/
```

### Run Specific Test Class or Function

```bash
# Run a specific test class
poetry run pytest tests/test_helper_functions.py::TestExtractUnmappedConditions

# Run a specific test
poetry run pytest tests/test_helper_functions.py::TestExtractUnmappedConditions::test_fully_unmapped_trials
```

## Test Categories

### Unit Tests (`test_helper_functions.py`)

Tests for individual helper functions:

- **`_extract_unmapped_conditions`**: Extract unmapped conditions from DataFrame
- **`_extract_existing_mesh_terms`**: Extract existing MeSH terms
- **`_add_mesh_mappings`**: Apply mappings to condition lists
- **`_filter_medical_conditions`**: Filter non-medical conditions (mocked GPT-5)
- **`_map_conditions_to_mesh`**: Map conditions to MeSH terms (mocked GPT-5)

### Integration Tests (`test_update_condition_mapping.py`)

End-to-end tests for the complete tool:

- **Full workflow**: Complete pipeline with mocked GPT-5
- **Edge cases**: Empty DataFrames, no unmapped conditions
- **Validation**: Invalid MeSH terms are rejected
- **Persistence**: CSV and DataFrame updates
- **Idempotency**: Running twice doesn't create duplicates

## Mocking Strategy

GPT-5 API calls are mocked to:
- Avoid API costs during testing
- Ensure deterministic test results
- Speed up test execution
- Allow testing edge cases (invalid responses, etc.)

Mocks return proper Pydantic objects matching the tool's schema.

## Fixtures

### DataFrame Fixtures

- `sample_dataframe`: Minimal DataFrame with various trial types
- `existing_mesh_terms`: List of valid MeSH terms
- `mock_mesh_mappings`: Dictionary of condition → MeSH mappings

### GPT-5 Mock Fixtures

- `mock_gpt5_filter_response`: Mocked filtering response
- `mock_gpt5_mapping_response`: Mocked mapping response
- `mock_gpt5_invalid_mapping_response`: Response with invalid MeSH terms

### File Fixtures

- `temp_mesh_csv`: Temporary CSV file for testing file operations

## Writing New Tests

To add new tests:

1. Add test function to appropriate file
2. Use existing fixtures from `conftest.py`
3. Mock external dependencies (GPT-5, file I/O)
4. Use descriptive test names: `test_<what>_<condition>_<expected>`

Example:

```python
def test_extract_unmapped_conditions_empty_dataframe(self):
    """Test with empty DataFrame."""
    empty_df = pd.DataFrame(columns=['conditions', 'matched_conditions'])

    result = _extract_unmapped_conditions(empty_df)

    assert isinstance(result, list)
    assert len(result) == 0
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

- No external API calls (all mocked)
- Deterministic results
- Fast execution
- Temporary files cleaned up automatically

## Coverage Goals

Target coverage: **>90%** for `Annotator/tools.py`

Check current coverage:

```bash
poetry run pytest --cov=Annotator --cov-report=html tests/
open htmlcov/index.html
```

## Troubleshooting

### ImportError for Annotator modules

Ensure the path is added correctly:

```python
sys.path.append(str(Path(__file__).parent.parent / 'Annotator'))
```

### Fixture not found

Run the fixture generation script:

```bash
poetry run python tests/fixtures/create_fixtures.py
```

### Mock not working

Ensure you're patching the correct import path. Use `tools.ChatOpenAI`, not `langchain_openai.ChatOpenAI`.

## Future Tests

Potential additions:

- Performance tests with large DataFrames (1000+ trials)
- Stress tests for edge cases
- Property-based testing with Hypothesis
- Integration tests with real (but rate-limited) GPT-5 calls
