from Bio import Entrez
import logging
from pydantic import validate_call, EmailStr
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from utils.exceptions import (
    NCBIAPIError,
    NCBIRateLimitError,
    NCBIConnectionError,
    NCBITimeoutError,
    NCBIMalformedResponseError
)
from utils.retry_handler import (
    create_ncbi_rate_limiter,
    retry_with_backoff,
    rate_limited
)

logger = logging.getLogger(__name__)

# Global rate limiter for NCBI API (3 requests/second without API key)
_ncbi_rate_limiter = create_ncbi_rate_limiter(has_api_key=False)

class pubmed_api_call:
    @validate_call
    def __init__(self, entrez_email : EmailStr, query : str, max_records : int = 10000):
        self.query = query
        self.credential = str(entrez_email)
        self.unique_ids = []
        self.total_records = 0
        self.max_records = max_records
        self.articles_found = False
        self.records = None

        Entrez.email = self.credential
        logger.debug(f"Initialized PubMed API client", extra={"query": query, "max_records": max_records})

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(HTTPError, URLError, IncompleteRead, ConnectionError, TimeoutError)
    )
    @rate_limited(_ncbi_rate_limiter)
    def article_count(self):
        """
        Get count of articles matching the query.

        Returns:
            int: Number of articles matching query

        Raises:
            NCBIConnectionError: If connection to NCBI fails
            NCBITimeoutError: If request times out
            NCBIMalformedResponseError: If response is malformed
        """
        logger.info("Requesting article count from NCBI", extra={"query": self.query})
        try:
            handle = Entrez.esearch(db="pubmed", term=self.query, rettype="count")
            records = Entrez.read(handle)
            handle.close()

            if "Count" not in records:
                raise NCBIMalformedResponseError(missing_key="Count")

            article_count = int(records["Count"])
            logger.info(f"NCBI reports {article_count} articles matching query")
            return article_count

        except HTTPError as e:
            if e.code == 429:
                raise NCBIRateLimitError()
            else:
                raise NCBIConnectionError(f"HTTP error {e.code}: {e.reason}", original_error=e)
        except (URLError, ConnectionError) as e:
            raise NCBIConnectionError("Network connection failed", original_error=e)
        except TimeoutError as e:
            raise NCBITimeoutError()
        except ValueError as e:
            raise NCBIMalformedResponseError(f"Invalid count value: {e}") 
    
    def search_articles(self):
        """
        Search for articles and retrieve full records.

        Sets self.articles_found, self.unique_ids, and self.records

        Raises:
            NCBIAPIError: If any API call fails
        """
        try:
            self._retrieve_pmids()
            # Only proceed if PMIDs were found
            if self.articles_found:
                self._retrieve_articles()
        except NCBIAPIError:
            # Re-raise NCBI-specific errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.error("Unexpected error during article search", exc_info=True)
            raise NCBIAPIError(f"Unexpected error: {str(e)}") from e


    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(HTTPError, URLError, IncompleteRead, ConnectionError, TimeoutError)
    )
    @rate_limited(_ncbi_rate_limiter)
    def _retrieve_pmids(self):
        """
        Retrieve PubMed IDs matching the query.

        Updates self.articles_found, self.unique_ids, self.total_records

        Raises:
            NCBIConnectionError: If connection fails
            NCBIRateLimitError: If rate limit exceeded
            NCBITimeoutError: If request times out
            NCBIMalformedResponseError: If response is malformed
        """
        logger.info("Retrieving PMIDs from NCBI", extra={"query": self.query, "max_records": self.max_records})

        try:
            handle = Entrez.esearch(db="pubmed", term=self.query, retmax=self.max_records)
            records = Entrez.read(handle)
            handle.close()

            # Validate response structure
            if "Count" not in records or "IdList" not in records:
                raise NCBIMalformedResponseError(
                    "NCBI response missing required keys",
                    missing_key="Count" if "Count" not in records else "IdList"
                )

            self.total_records = int(records["Count"])

            if self.total_records > 0:
                pmids = records["IdList"]
                self.articles_found = True
                self.unique_ids = pmids
                logger.info(f"Retrieved {len(pmids)} PMIDs from NCBI", extra={"total_available": self.total_records})
                if self.total_records > self.max_records:
                    logger.warning(f"Query returned {self.total_records} articles, but only retrieving {self.max_records} (max_records limit)")
            else:
                self.unique_ids = []
                self.articles_found = False
                logger.warning("NCBI search returned zero results", extra={"query": self.query})

        except HTTPError as e:
            self.articles_found = False
            self.unique_ids = []
            if e.code == 429:
                raise NCBIRateLimitError()
            else:
                raise NCBIConnectionError(f"HTTP error {e.code}: {e.reason}", original_error=e)
        except (URLError, ConnectionError) as e:
            self.articles_found = False
            self.unique_ids = []
            raise NCBIConnectionError("Network connection failed - check internet connectivity", original_error=e)
        except TimeoutError as e:
            self.articles_found = False
            self.unique_ids = []
            raise NCBITimeoutError()
        except (ValueError, KeyError) as e:
            self.articles_found = False
            self.unique_ids = []
            raise NCBIMalformedResponseError(f"Invalid response data: {e}")
            
    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(HTTPError, URLError, IncompleteRead, ConnectionError, TimeoutError)
    )
    @rate_limited(_ncbi_rate_limiter)
    def _retrieve_articles(self):
        """
        Retrieve full article records for the found PMIDs.

        Updates self.records

        Raises:
            NCBIConnectionError: If connection fails
            NCBITimeoutError: If request times out
            NCBIMalformedResponseError: If response is malformed
        """
        if not self.articles_found:
            logger.warning("Skipping article retrieval - no PMIDs found")
            return

        logger.info(f"Fetching full article records from NCBI", extra={"pmid_count": len(self.unique_ids)})

        try:
            handle = Entrez.efetch(db='pubmed', id=self.unique_ids, retmode='xml')
            self.records = Entrez.read(handle)
            handle.close()

            # Validate response structure
            if not isinstance(self.records, dict):
                raise NCBIMalformedResponseError("Response is not a dictionary")

            if 'PubmedArticle' not in self.records:
                raise NCBIMalformedResponseError(missing_key='PubmedArticle')

            article_count = len(self.records.get('PubmedArticle', []))
            logger.info(f"Successfully retrieved {article_count} article records from NCBI")

            # Warn if we got fewer articles than expected
            if article_count < len(self.unique_ids):
                logger.warning(f"Retrieved fewer articles than PMIDs",
                             extra={"pmids": len(self.unique_ids), "articles": article_count})

        except HTTPError as e:
            if e.code == 429:
                raise NCBIRateLimitError()
            else:
                raise NCBIConnectionError(f"HTTP error {e.code}: {e.reason}", original_error=e)
        except (URLError, ConnectionError) as e:
            raise NCBIConnectionError("Network connection failed - check internet connectivity", original_error=e)
        except TimeoutError as e:
            raise NCBITimeoutError("Request timed out - try reducing max_records")
        except (ValueError, KeyError, TypeError) as e:
            raise NCBIMalformedResponseError(f"Invalid response structure: {e}")

