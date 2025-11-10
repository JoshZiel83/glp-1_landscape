"""
ClinicalTrials.gov API client utility.

This module provides a class-based interface for querying the ClinicalTrials.gov
API v2.0, with support for multiple query parameters, filtering, automatic
token-based pagination, and robust error handling.
"""

import logging
import requests
from typing import Optional, List, Dict, Any
from pydantic import validate_call

from utils.exceptions import (
    ClinicalTrialsAPIError,
    ClinicalTrialsRateLimitError,
    ClinicalTrialsConnectionError,
    ClinicalTrialsTimeoutError,
    ClinicalTrialsMalformedResponseError
)
from utils.retry_handler import (
    create_clinicaltrials_rate_limiter,
    retry_with_backoff,
    rate_limited
)

logger = logging.getLogger(__name__)

# Global rate limiter for ClinicalTrials.gov API (0.8 requests/second = 48/min)
_clinicaltrials_rate_limiter = create_clinicaltrials_rate_limiter(max_per_second=0.8)

# ClinicalTrials.gov API base URL
CLINICALTRIALS_API_BASE = "https://clinicaltrials.gov/api/v2/studies"


class clinicaltrials_api_call:
    """
    Client for querying ClinicalTrials.gov API v2.0.

    Supports searching clinical trials by condition, intervention, sponsor,
    location, and other parameters. Handles automatic token-based pagination
    and provides individual study retrieval by NCT ID.

    Example:
        >>> client = clinicaltrials_api_call(
        ...     query_condition="Diabetes",
        ...     query_intervention="Insulin",
        ...     filter_status="RECRUITING"
        ... )
        >>> count = client.get_study_count()
        >>> client.search_studies()
        >>> studies = client.study_list
    """

    @validate_call
    def __init__(
        self,
        query_condition: Optional[str] = None,
        query_intervention: Optional[str] = None,
        query_term: Optional[str] = None,
        query_location: Optional[str] = None,
        query_sponsor: Optional[str] = None,
        query_title: Optional[str] = None,
        filter_status: Optional[str] = None,
        filter_date: Optional[str]=None,
        page_size: int = 100,
        max_records: int = 10000
    ):
        """
        Initialize ClinicalTrials.gov API client.

        Args:
            query_condition: Disease or condition to search for
            query_intervention: Intervention or treatment to search for
            query_term: General search term
            query_location: Location to search (city, state, country)
            query_sponsor: Sponsor or collaborator name
            query_title: Title or acronym search
            filter_status: Study status filter (e.g., "RECRUITING", "COMPLETED")
            page_size: Number of results per page (default: 100, max: 1000)
            max_records: Maximum number of studies to retrieve (default: 10000)
        """
        self.query_condition = query_condition
        self.query_intervention = query_intervention
        self.query_term = query_term
        self.query_location = query_location
        self.query_sponsor = query_sponsor
        self.query_title = query_title
        self.filter_status = filter_status
        self.filter_date = filter_date
        self.page_size = min(page_size, 1000)  # API max is 1000
        self.max_records = max_records

        # State tracking
        self.studies_found = False
        self.study_list: List[Dict[str, Any]] = []
        self.total_count = 0

        logger.debug(
            f"Initialized ClinicalTrials.gov API client",
            extra={
                "query_condition": query_condition,
                "query_intervention": query_intervention,
                "filter_status": filter_status,
                "filter_date": filter_date,
                "page_size": page_size,
                "max_records": max_records
            }
        )

    def _build_query_params(self, include_pagination: bool = True) -> Dict[str, Any]:
        """
        Build query parameters for API request.

        Args:
            include_pagination: Whether to include pageSize parameter

        Returns:
            Dictionary of query parameters
        """
        params = {}

        # Add query parameters
        if self.query_condition:
            params["query.cond"] = self.query_condition
        if self.query_intervention:
            params["query.intr"] = self.query_intervention
        if self.query_term:
            params["query.term"] = self.query_term
        if self.query_location:
            params["query.locn"] = self.query_location
        if self.query_sponsor:
            params["query.spons"] = self.query_sponsor
        if self.query_title:
            params["query.titles"] = self.query_title

        # Add filters
        if self.filter_status:
            params["filter.overallStatus"] = self.filter_status
        if self.filter_date:
            params["filter.advanced"] = self.filter_date

        # Add pagination
        if include_pagination:
            params["pageSize"] = self.page_size

        # Request total count
        params["countTotal"] = "true"

        logger.debug(f"Built query parameters", extra={"params": params})
        return params

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(requests.RequestException, ConnectionError, TimeoutError)
    )
    @rate_limited(_clinicaltrials_rate_limiter)
    def get_study_count(self) -> int:
        """
        Get count of studies matching the search criteria.

        Makes an initial API call to get the total count without retrieving all studies.

        Returns:
            int: Number of studies matching criteria

        Raises:
            ClinicalTrialsConnectionError: If connection to ClinicalTrials.gov fails
            ClinicalTrialsTimeoutError: If request times out
            ClinicalTrialsMalformedResponseError: If response is malformed
        """
        logger.info(
            "Requesting study count from ClinicalTrials.gov",
            extra={"query_condition": self.query_condition}
        )

        try:
            params = self._build_query_params(include_pagination=True)
            params["pageSize"] = 1  # Minimal page size for count-only request

            response = requests.get(CLINICALTRIALS_API_BASE, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Validate response structure
            if "totalCount" not in data:
                raise ClinicalTrialsMalformedResponseError(missing_key="totalCount")

            total_count = int(data["totalCount"])
            logger.info(f"ClinicalTrials.gov reports {total_count} studies matching criteria")

            return total_count

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                raise ClinicalTrialsRateLimitError()
            else:
                raise ClinicalTrialsConnectionError(
                    f"HTTP error {e.response.status_code}: {e.response.reason}",
                    original_error=e
                )
        except requests.ConnectionError as e:
            raise ClinicalTrialsConnectionError(
                "Network connection failed - check internet connectivity",
                original_error=e
            )
        except requests.Timeout as e:
            raise ClinicalTrialsTimeoutError()
        except (ValueError, KeyError) as e:
            raise ClinicalTrialsMalformedResponseError(f"Invalid count value: {e}")

    def search_studies(self) -> None:
        """
        Search for studies and retrieve full records.

        Handles automatic token-based pagination to retrieve all studies
        (up to max_records). Updates self.studies_found, self.study_list,
        and self.total_count.

        Raises:
            ClinicalTrialsAPIError: If any API call fails
        """
        try:
            self._retrieve_studies()

        except ClinicalTrialsAPIError:
            # Re-raise ClinicalTrials-specific errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.error("Unexpected error during study search", exc_info=True)
            raise ClinicalTrialsAPIError(f"Unexpected error: {str(e)}") from e

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(requests.RequestException, ConnectionError, TimeoutError)
    )
    @rate_limited(_clinicaltrials_rate_limiter)
    def _retrieve_studies(self) -> None:
        """
        Retrieve study details with automatic token-based pagination.

        ClinicalTrials.gov API uses nextPageToken for pagination. This method
        handles pagination automatically until all studies (up to max_records)
        are retrieved.

        Updates self.studies_found, self.study_list, self.total_count

        Raises:
            ClinicalTrialsConnectionError: If connection fails
            ClinicalTrialsTimeoutError: If request times out
            ClinicalTrialsMalformedResponseError: If response is malformed
        """
        logger.info(
            f"Retrieving study details from ClinicalTrials.gov",
            extra={
                "query_condition": self.query_condition,
                "max_records": self.max_records
            }
        )

        studies_retrieved = 0
        all_studies = []
        page_token = None

        try:
            while studies_retrieved < self.max_records:
                params = self._build_query_params(include_pagination=True)

                # Add page token if we're fetching subsequent pages
                if page_token:
                    params["pageToken"] = page_token

                response = requests.get(CLINICALTRIALS_API_BASE, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                # Get total count on first iteration
                if page_token is None:
                    if "totalCount" not in data:
                        raise ClinicalTrialsMalformedResponseError(missing_key="totalCount")
                    self.total_count = int(data["totalCount"])
                    logger.info(
                        f"Total studies available: {self.total_count}",
                        extra={"will_retrieve": min(self.total_count, self.max_records)}
                    )

                # Get studies from this batch
                if "studies" not in data:
                    raise ClinicalTrialsMalformedResponseError(missing_key="studies")

                studies = data["studies"]

                # Break if no more studies
                if not studies:
                    logger.info("No more studies to retrieve")
                    break

                # Add studies to collection
                batch_size = len(studies)
                remaining_slots = self.max_records - studies_retrieved
                studies_to_add = studies[:remaining_slots]

                all_studies.extend(studies_to_add)
                studies_retrieved += len(studies_to_add)

                logger.debug(
                    f"Retrieved batch of {len(studies_to_add)} studies",
                    extra={"total_retrieved": studies_retrieved}
                )

                # Check if we've hit max_records limit
                if studies_retrieved >= self.max_records:
                    logger.info(f"Reached max_records limit of {self.max_records}")
                    break

                # Get next page token
                page_token = data.get("nextPageToken")
                if not page_token:
                    logger.info("No more pages available")
                    break

            # Update instance state
            self.study_list = all_studies
            self.studies_found = len(all_studies) > 0

            logger.info(
                f"Successfully retrieved {len(all_studies)} studies from ClinicalTrials.gov",
                extra={"total_available": self.total_count}
            )

            # Warn if we got fewer studies than expected
            if len(all_studies) < self.total_count and len(all_studies) < self.max_records:
                logger.warning(
                    f"Retrieved fewer studies than expected",
                    extra={
                        "expected": min(self.total_count, self.max_records),
                        "retrieved": len(all_studies)
                    }
                )

        except requests.HTTPError as e:
            self.studies_found = False
            self.study_list = []
            if e.response.status_code == 429:
                raise ClinicalTrialsRateLimitError()
            else:
                raise ClinicalTrialsConnectionError(
                    f"HTTP error {e.response.status_code}: {e.response.reason}",
                    original_error=e
                )
        except requests.ConnectionError as e:
            self.studies_found = False
            self.study_list = []
            raise ClinicalTrialsConnectionError(
                "Network connection failed - check internet connectivity",
                original_error=e
            )
        except requests.Timeout as e:
            self.studies_found = False
            self.study_list = []
            raise ClinicalTrialsTimeoutError()
        except (ValueError, KeyError) as e:
            self.studies_found = False
            self.study_list = []
            raise ClinicalTrialsMalformedResponseError(f"Invalid response data: {e}")

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(requests.RequestException, ConnectionError, TimeoutError)
    )
    @rate_limited(_clinicaltrials_rate_limiter)
    def get_study_by_nct_id(self, nct_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single study by its NCT ID.

        Args:
            nct_id: NCT identifier (e.g., "NCT04267848")

        Returns:
            Dict with study details if found, None if not found

        Raises:
            ClinicalTrialsConnectionError: If connection fails
            ClinicalTrialsTimeoutError: If request times out
        """
        logger.info(f"Retrieving study by NCT ID", extra={"nct_id": nct_id})

        try:
            url = f"{CLINICALTRIALS_API_BASE}/{nct_id}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check if study was found
            if "studies" in data and data["studies"]:
                study = data["studies"][0]
                logger.info(f"Study found", extra={"nct_id": nct_id})
                return study
            else:
                logger.info(f"Study not found", extra={"nct_id": nct_id})
                return None

        except requests.HTTPError as e:
            if e.response.status_code == 429:
                raise ClinicalTrialsRateLimitError()
            elif e.response.status_code == 404:
                logger.warning(f"NCT ID not found", extra={"nct_id": nct_id})
                return None
            else:
                raise ClinicalTrialsConnectionError(
                    f"HTTP error {e.response.status_code}: {e.response.reason}",
                    original_error=e
                )
        except requests.ConnectionError as e:
            raise ClinicalTrialsConnectionError(
                "Network connection failed - check internet connectivity",
                original_error=e
            )
        except requests.Timeout as e:
            raise ClinicalTrialsTimeoutError()
