"""
Custom exceptions for the scientific intelligence utilities.

This module defines specific exception types for better error handling
and recovery guidance throughout the application.
"""


# ============================================================================
# NCBI/PubMed API Exceptions
# ============================================================================

class NCBIAPIError(Exception):
    """Base exception for NCBI API-related errors."""
    pass


class NCBIRateLimitError(NCBIAPIError):
    """Raised when NCBI API rate limit is exceeded."""

    def __init__(self, message="NCBI API rate limit exceeded. Wait 60 seconds before retrying."):
        self.message = message
        super().__init__(self.message)


class NCBIConnectionError(NCBIAPIError):
    """Raised when connection to NCBI API fails."""

    def __init__(self, message="Failed to connect to NCBI API. Check internet connection.", original_error=None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class NCBITimeoutError(NCBIAPIError):
    """Raised when NCBI API request times out."""

    def __init__(self, message="NCBI API request timed out. Try reducing max_articles or retry later."):
        self.message = message
        super().__init__(self.message)


class NCBIMalformedResponseError(NCBIAPIError):
    """Raised when NCBI API returns malformed or unexpected data."""

    def __init__(self, message="NCBI API returned malformed response.", missing_key=None):
        self.message = message
        self.missing_key = missing_key
        if missing_key:
            self.message += f" Missing key: '{missing_key}'"
        super().__init__(self.message)


# ============================================================================
# BioRxiv/MedRxiv API Exceptions
# ============================================================================

class BioRxivAPIError(Exception):
    """Base exception for BioRxiv/MedRxiv API-related errors."""
    pass


class BioRxivRateLimitError(BioRxivAPIError):
    """Raised when BioRxiv API rate limit is exceeded."""

    def __init__(self, message="BioRxiv API rate limit exceeded. Wait before retrying."):
        self.message = message
        super().__init__(self.message)


class BioRxivConnectionError(BioRxivAPIError):
    """Raised when connection to BioRxiv API fails."""

    def __init__(self, message="Failed to connect to BioRxiv API. Check internet connection.", original_error=None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class BioRxivTimeoutError(BioRxivAPIError):
    """Raised when BioRxiv API request times out."""

    def __init__(self, message="BioRxiv API request timed out. Try reducing max_records or retry later."):
        self.message = message
        super().__init__(self.message)


class BioRxivMalformedResponseError(BioRxivAPIError):
    """Raised when BioRxiv API returns malformed or unexpected data."""

    def __init__(self, message="BioRxiv API returned malformed response.", missing_key=None):
        self.message = message
        self.missing_key = missing_key
        if missing_key:
            self.message += f" Missing key: '{missing_key}'"
        super().__init__(self.message)


# ============================================================================
# ClinicalTrials.gov API Exceptions
# ============================================================================

class ClinicalTrialsAPIError(Exception):
    """Base exception for ClinicalTrials.gov API-related errors."""
    pass


class ClinicalTrialsRateLimitError(ClinicalTrialsAPIError):
    """Raised when ClinicalTrials.gov API rate limit is exceeded."""

    def __init__(self, message="ClinicalTrials.gov API rate limit exceeded. Wait before retrying."):
        self.message = message
        super().__init__(self.message)


class ClinicalTrialsConnectionError(ClinicalTrialsAPIError):
    """Raised when connection to ClinicalTrials.gov API fails."""

    def __init__(self, message="Failed to connect to ClinicalTrials.gov API. Check internet connection.", original_error=None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class ClinicalTrialsTimeoutError(ClinicalTrialsAPIError):
    """Raised when ClinicalTrials.gov API request times out."""

    def __init__(self, message="ClinicalTrials.gov API request timed out. Try reducing max_records or retry later."):
        self.message = message
        super().__init__(self.message)


class ClinicalTrialsMalformedResponseError(ClinicalTrialsAPIError):
    """Raised when ClinicalTrials.gov API returns malformed or unexpected data."""

    def __init__(self, message="ClinicalTrials.gov API returned malformed response.", missing_key=None):
        self.message = message
        self.missing_key = missing_key
        if missing_key:
            self.message += f" Missing key: '{missing_key}'"
        super().__init__(self.message)


# ============================================================================
# Semantic Scholar API Exceptions
# ============================================================================

class SemanticScholarAPIError(Exception):
    """Base exception for Semantic Scholar API-related errors."""
    pass


class SemanticScholarRateLimitError(SemanticScholarAPIError):
    """Raised when Semantic Scholar API rate limit is exceeded."""

    def __init__(self, message="Semantic Scholar API rate limit exceeded. Wait before retrying."):
        self.message = message
        super().__init__(self.message)


class SemanticScholarConnectionError(SemanticScholarAPIError):
    """Raised when connection to Semantic Scholar API fails."""

    def __init__(self, message="Failed to connect to Semantic Scholar API. Check internet connection.", original_error=None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)


class SemanticScholarTimeoutError(SemanticScholarAPIError):
    """Raised when Semantic Scholar API request times out."""

    def __init__(self, message="Semantic Scholar API request timed out. Try reducing max_records or retry later."):
        self.message = message
        super().__init__(self.message)


class SemanticScholarMalformedResponseError(SemanticScholarAPIError):
    """Raised when Semantic Scholar API returns malformed or unexpected data."""

    def __init__(self, message="Semantic Scholar API returned malformed response.", missing_key=None):
        self.message = message
        self.missing_key = missing_key
        if missing_key:
            self.message += f" Missing key: '{missing_key}'"
        super().__init__(self.message)


# ============================================================================
# LLM Exceptions
# ============================================================================

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM service fails."""

    def __init__(self, message="Failed to connect to LLM service.", model=None, original_error=None):
        self.message = message
        self.model = model
        self.original_error = original_error
        if model:
            self.message += f" Model: {model}"
        super().__init__(self.message)


class LLMTimeoutError(LLMError):
    """Raised when LLM query times out."""

    def __init__(self, message="LLM query timed out.", model=None):
        self.message = message
        self.model = model
        if model:
            self.message += f" Model: {model}"
        super().__init__(self.message)


class LLMParsingError(LLMError):
    """Raised when LLM response cannot be parsed into expected schema."""

    def __init__(self, message="Failed to parse LLM response into expected format.", model=None, response_preview=None):
        self.message = message
        self.model = model
        self.response_preview = response_preview
        if model:
            self.message += f" Model: {model}"
        if response_preview:
            self.message += f" Response preview: {response_preview[:100]}"
        super().__init__(self.message)


class LLMModelNotFoundError(LLMError):
    """Raised when requested LLM model is not available."""

    def __init__(self, message="Requested LLM model not found.", model=None, available_models=None):
        self.message = message
        self.model = model
        self.available_models = available_models
        if model:
            self.message += f" Requested: {model}"
        if available_models:
            self.message += f" Available: {', '.join(available_models[:5])}"
        super().__init__(self.message)


# ============================================================================
# Data Validation Exceptions
# ============================================================================

class DataValidationError(Exception):
    """Base exception for data validation errors."""
    pass


class EmptyQueryError(DataValidationError):
    """Raised when query string is empty or whitespace only."""

    def __init__(self, message="Query string cannot be empty or whitespace only."):
        self.message = message
        super().__init__(self.message)


class EmptyDataFrameError(DataValidationError):
    """Raised when DataFrame is unexpectedly empty."""

    def __init__(self, message="DataFrame is empty - no data to process.", operation=None):
        self.message = message
        self.operation = operation
        if operation:
            self.message += f" Operation: {operation}"
        super().__init__(self.message)


class MissingFieldError(DataValidationError):
    """Raised when required field is missing from data."""

    def __init__(self, message="Required field is missing.", field_name=None, record_id=None):
        self.message = message
        self.field_name = field_name
        self.record_id = record_id
        if field_name:
            self.message += f" Field: '{field_name}'"
        if record_id:
            self.message += f" Record ID: {record_id}"
        super().__init__(self.message)


class DataLengthMismatchError(DataValidationError):
    """Raised when related data structures have mismatched lengths."""

    def __init__(self, message="Data length mismatch detected.", expected=None, actual=None):
        self.message = message
        self.expected = expected
        self.actual = actual
        if expected is not None and actual is not None:
            self.message += f" Expected: {expected}, Got: {actual}"
        super().__init__(self.message)
