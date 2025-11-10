from Bio import Entrez
import logging
from enum import Enum
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

class Database(Enum):
    PUBMED = "pubmed"
    PROTEIN = "protein"
    NUCCORE = "nuccore"
    IPG = "ipg"
    NUCLEOTIDE = "nucleotide"
    STRUCTURE = "structure"
    GENOME = "genome"
    ANNOTINFO = "annotinfo"
    ASSEMBLY = "assembly"
    BIOPROJECT = "bioproject"
    BIOSAMPLE = "biosample"
    BLASTDBINFO = "blastdbinfo"
    BOOKS = "books"
    CDD = "cdd"
    CLINVAR = "clinvar"
    GAP = "gap"
    GAPPLUS = "gapplus"
    GRASP = "grasp"
    DBVAR = "dbvar"
    GENE = "gene"
    GDS = "gds"
    GEOPROFILES = "geoprofiles"
    MEDGEN = "medgen"
    MESH = "mesh"
    NLMCATALOG = "nlmcatalog"
    OMIM = "omim"
    ORGTRACK = "orgtrack"
    PMC = "pmc"
    PROTEINCLUSTERS = "proteinclusters"
    PCASSAY = "pcassay"
    PROTFAM = "protfam"
    PCCOMPOUND = "pccompound"
    PCSUBSTANCE = "pcsubstance"
    SEQANNOT = "seqannot"
    SNP = "snp"
    SRA = "sra"
    TAXONOMY = "taxonomy"
    BIOCOLLECTIONS = "biocollections"
    GTR = "gtr"


class eutils_api_call:
    @validate_call
    def __init__(self, entrez_email : EmailStr, database: Database, query : str, max_records : int = 10000, sort:str = None):
        self.database = database.value
        self.sort = sort 
        self.query = query
        self.credential = str(entrez_email)
        self.unique_ids = []
        self.total_records = 0
        self.max_records = max_records
        self.records_found = False
        self.records = None

        Entrez.email = self.credential
        logger.debug(f"Initialized Entrez E-utilities API client", extra={"query": query, "max_records": max_records})

    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(HTTPError, URLError, IncompleteRead, ConnectionError, TimeoutError)
    )
    @rate_limited(_ncbi_rate_limiter)
    def id_count(self):
        """
        Get count of records matching the query.

        Returns:
            int: Number of records matching query

        Raises:
            NCBIConnectionError: If connection to NCBI fails
            NCBITimeoutError: If request times out
            NCBIMalformedResponseError: If response is malformed
        """
        logger.info("Requesting id count from NCBI", extra={"query": self.query})
        try:
            if self.sort:
                handle = Entrez.esearch(db=self.database, term=self.query, rettype="count", sort = self.sort)
            else:
                handle = Entrez.esearch(db=self.database, term=self.query, rettype="count")
            records = Entrez.read(handle)
            handle.close()

            if "Count" not in records:
                raise NCBIMalformedResponseError(missing_key="Count")

            record_count = int(records["Count"])
            logger.info(f"NCBI reports {record_count} records matching query")
            return record_count

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
    
    def search_records(self):
        """
        Search for records and retrieve full records.

        Sets self.rcordss_found, self.unique_ids, and self.records

        Raises:
            NCBIAPIError: If any API call fails
        """
        try:
            self._retrieve_ids()
            # Only proceed if PMIDs were found
            if self.records_found:
                self._retrieve_records()
        except NCBIAPIError:
            # Re-raise NCBI-specific errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            logger.error("Unexpected error during record search", exc_info=True)
            raise NCBIAPIError(f"Unexpected error: {str(e)}") from e


    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(HTTPError, URLError, IncompleteRead, ConnectionError, TimeoutError)
    )
    @rate_limited(_ncbi_rate_limiter)
    def _retrieve_ids(self):
        """
        Retrieve database Unique IDs matching the query.

        Updates self.records_found, self.unique_ids, self.total_records

        Raises:
            NCBIConnectionError: If connection fails
            NCBIRateLimitError: If rate limit exceeded
            NCBITimeoutError: If request times out
            NCBIMalformedResponseError: If response is malformed
        """
        logger.info("Retrieving IDs from NCBI", extra={"query": self.query, "max_records": self.max_records})

        try:
            if self.sort:
                handle = Entrez.esearch(db=self.database, term=self.query, sort = self.sort)
            else:
                handle = Entrez.esearch(db=self.database, term=self.query)
            records = Entrez.read(handle)
            handle.close()

            self.total_records = int(records["Count"])

            if self.total_records > 0:
                uids = records["IdList"]
                self.records_found = True
                self.unique_ids = uids
                logger.info(f"Retrieved {len(uids)} unique IDs from NCBI", extra={"total_available": self.total_records})
                if self.total_records > self.max_records:
                    logger.warning(f"Query returned {self.total_records} records, but only retrieving {self.max_records} (max_records limit)")
            else:
                self.unique_ids = []
                self.records_found = False
                logger.warning("Entrez search returned zero results", extra={"query": self.query})

        except HTTPError as e:
            self.records_found = False
            self.unique_ids = []
            if e.code == 429:
                raise NCBIRateLimitError()
            else:
                raise NCBIConnectionError(f"HTTP error {e.code}: {e.reason}", original_error=e)
        except (URLError, ConnectionError) as e:
            self.records_found = False
            self.unique_ids = []
            raise NCBIConnectionError("Network connection failed - check internet connectivity", original_error=e)
        except TimeoutError as e:
            self.records_found = False
            self.unique_ids = []
            raise NCBITimeoutError()
        except (ValueError, KeyError) as e:
            self.records_found = False
            self.unique_ids = []
            raise NCBIMalformedResponseError(f"Invalid response data: {e}")
            
    @retry_with_backoff(
        max_attempts=3,
        initial_delay=2.0,
        exceptions=(HTTPError, URLError, IncompleteRead, ConnectionError, TimeoutError)
    )
    @rate_limited(_ncbi_rate_limiter)
    def _retrieve_records(self):
        """
        Retrieve full records for the found Unique IDs.

        Updates self.records

        Raises:
            NCBIConnectionError: If connection fails
            NCBITimeoutError: If request times out
            NCBIMalformedResponseError: If response is malformed
        """
        if not self.records_found:
            logger.warning("Skipping record retrieval - no IDs found")
            return

        logger.info(f"Fetching full  records from Entrez", extra={"ID_count": len(self.unique_ids)})

        try:
            handle = Entrez.efetch(db=self.database, id=self.unique_ids, retmode='xml')
            self.records = Entrez.read(handle)
            handle.close()

            # Validate response structure
            if not isinstance(self.records, dict):
                raise NCBIMalformedResponseError("Response is not a dictionary")

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

