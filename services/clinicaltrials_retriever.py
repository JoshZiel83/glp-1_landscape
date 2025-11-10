
import logging
from datetime import datetime
import pandas as pd
from typing import Optional
from utils import clinicaltrials
from pydantic import validate_call
from utils.exceptions import (
    EmptyQueryError,
    DataValidationError,
    ClinicalTrialsMalformedResponseError
)

logger = logging.getLogger(__name__)

@validate_call
def get_study_count(
    query_condition: Optional[str] = None,
    query_intervention: Optional[str] = None,
    query_term: Optional[str] = None,
    query_location: Optional[str] = None,
    query_sponsor: Optional[str] = None,
    query_title: Optional[str] = None,
    filter_status: Optional[str] = None,
    look_back_date: Optional[datetime] = None
) -> int:
    """
    Get the count of ClinicalTrials.gov studies matching search criteria.

    At least one query parameter must be provided.

    Args:
        query_condition: Disease or condition to search for
        query_intervention: Intervention or treatment to search for
        query_term: General search term
        query_location: Location to search (city, state, country)
        query_sponsor: Sponsor or collaborator name
        query_title: Title or acronym search
        filter_status: Study status filter (e.g., "RECRUITING", "COMPLETED")

    Returns:
        int: Number of studies matching the criteria

    Raises:
        EmptyQueryError: If no query parameters are provided
        ClinicalTrialsAPIError: If ClinicalTrials.gov API call fails
    """
    #Add Date Filter
    if look_back_date:
        filter_date = _add_date_filter(look_back_date)
    else:
        filter_date = None
    
    # Validate at least one query parameter is provided
    query_params = [
        query_condition, query_intervention, query_term,
        query_location, query_sponsor, query_title
    ]
    if not any(query_params):
        raise EmptyQueryError("At least one query parameter must be provided")

    logger.info(
        "Getting study count from ClinicalTrials.gov",
        extra={
            "query_condition": query_condition,
            "query_intervention": query_intervention,
            "filter_status": filter_status
        }
    )

    try:
        client = clinicaltrials.clinicaltrials_api_call(
            query_condition=query_condition,
            query_intervention=query_intervention,
            query_term=query_term,
            query_location=query_location,
            query_sponsor=query_sponsor,
            query_title=query_title,
            filter_status=filter_status,
            filter_date=filter_date,
            max_records=1  # Only need count, not full studies
        )
        count = client.get_study_count()
        logger.info(
            f"ClinicalTrials.gov study count: {count}",
            extra={"query_condition": query_condition, "query_intervention": query_intervention}
        )
        return count
    except Exception as e:
        logger.error(
            "Failed to get study count from ClinicalTrials.gov",
            extra={
                "query_condition": query_condition,
                "query_intervention": query_intervention,
                "error": str(e)
            }
        )
        raise

@validate_call
def retrieve_batched_studies(
    query_condition: Optional[str] = None,
    query_intervention: Optional[str] = None,
    query_term: Optional[str] = None,
    query_location: Optional[str] = None,
    query_sponsor: Optional[str] = None,
    query_title: Optional[str] = None,
    filter_status: Optional[str] = None,
    look_back_date: Optional[datetime]=None,
    max_studies: int = 10000
) -> pd.DataFrame:
    """
    Retrieve and process ClinicalTrials.gov studies based on search criteria.

    At least one query parameter must be provided.

    Args:
        query_condition: Disease or condition to search for
        query_intervention: Intervention or treatment to search for
        query_term: General search term
        query_location: Location to search (city, state, country)
        query_sponsor: Sponsor or collaborator name
        query_title: Title or acronym search
        filter_status: Study status filter (e.g., "RECRUITING", "COMPLETED")
        filter_date: Filters for studies after the specified start_date
        max_studies: Maximum number of studies to retrieve (default: 10000)

    Returns:
        pd.DataFrame: DataFrame with study metadata. Empty if no results or error.
                      Check df.empty before processing.

    Raises:
        EmptyQueryError: If no query parameters are provided
        ValueError: If max_studies is not positive
    """
    logger.info("Starting ClinicalTrials.gov study retrieval")
    
    if look_back_date:
        filter_date = _add_date_filter(look_back_date)
    else:
        filter_date = None
    
    # Validate at least one query parameter is provided
    query_params = [
        query_condition, query_intervention, query_term,
        query_location, query_sponsor, query_title
    ]
    if not any(query_params):
        raise EmptyQueryError("At least one query parameter must be provided")

    if max_studies <= 0:
        raise ValueError(f"max_studies must be positive, got {max_studies}")

    logger.info(
        "Starting ClinicalTrials.gov study retrieval",
        extra={
            "query_condition": query_condition,
            "query_intervention": query_intervention,
            "filter_status": filter_status,
            "max_studies": max_studies
        }
    )

    try:
        client = clinicaltrials.clinicaltrials_api_call(
            query_condition=query_condition,
            query_intervention=query_intervention,
            query_term=query_term,
            query_location=query_location,
            query_sponsor=query_sponsor,
            query_title=query_title,
            filter_status=filter_status,
            filter_date = filter_date,
            max_records=max_studies
        )
        client.search_studies()
    except Exception as e:
        logger.error("ClinicalTrials.gov API call failed - returning empty DataFrame", extra={"error": str(e)})
        return pd.DataFrame()

    if client.studies_found:
        try:
            # Validate response structure
            if not client.study_list:
                logger.warning("No studies in study_list - returning empty DataFrame")
                return pd.DataFrame()

            studies_frame = _process_batched_studies(client.study_list)
            logger.info(f"Successfully processed {len(studies_frame)} studies")
            return studies_frame

        except (KeyError, TypeError, DataValidationError) as e:
            logger.error("Failed to process study records", exc_info=True,
                        extra={"study_count": len(client.study_list) if client.study_list else 0})
            return pd.DataFrame()
    else:
        logger.warning("Search returned no results - returning empty DataFrame")
        return pd.DataFrame()

def _process_batched_studies(studies):
    """
    Extract metadata from ClinicalTrials.gov study records.

    Args:
        studies: List of study records from ClinicalTrials.gov API

    Returns:
        pd.DataFrame: DataFrame with extracted study metadata
    """
    logger.debug(f"Processing {len(studies)} study records")
    rows_list = []
    errors_count = 0

    for study in studies:
        try:
            # Extract NCT ID directly from study record (top level)
            try:
                protocol_section = study.get('protocolSection', {})
                nct_id = protocol_section.get('identificationModule', {}).get('nctId', 'unknown')
            except (KeyError, TypeError, AttributeError) as e:
                logger.warning(f"Could not extract NCT ID from study record", extra={"error": str(e)})
                nct_id = "unknown"

            # Extract all metadata
            identification_info = _get_identification_info(study)
            status_info = _get_status_info(study)
            description_info = _get_description_info(study)
            conditions = _get_conditions(study)
            interventions_info = _get_interventions(study)
            sponsor_info = _get_sponsor_info(study)
            locations_info = _get_locations(study)
            eligibility_info = _get_eligibility_info(study)
            design_info = _get_design_info(study)
            outcomes_info = _get_outcomes(study)

            # Construct study URL
            study_url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id != "unknown" else None

            #construct citation
            citation = _process_citation(identification_info['official_title'], nct_id, study_url, status_info['last_update_date'])

            # Build row with all extracted data
            new_row_list = [
                nct_id,
                identification_info['official_title'],
                identification_info['brief_title'],
                identification_info['acronym'],
                status_info['overall_status'],
                status_info['start_date'],
                status_info['completion_date'],
                status_info['last_update_date'],
                description_info['brief_summary'],
                description_info['detailed_description'],
                conditions,
                interventions_info['interventions'],
                interventions_info['intervention_types'],
                sponsor_info['lead_sponsor'],
                sponsor_info['sponsor_class'],
                sponsor_info['collaborators'],
                locations_info['countries'],
                locations_info['facilities'],
                eligibility_info['min_age'],
                eligibility_info['max_age'],
                eligibility_info['gender'],
                eligibility_info['healthy_volunteers'],
                design_info['study_type'],
                design_info['phase'],
                design_info['enrollment'],
                outcomes_info['primary_outcomes'],
                outcomes_info['secondary_outcomes'],
                study_url, 
                citation
            ]
        except (KeyError, TypeError, AttributeError, IndexError) as e:
            logger.warning(f"Error processing study metadata", extra={"nct_id": nct_id if 'nct_id' in locals() else 'unknown', "error": str(e)})
            errors_count += 1
            # Create row with minimal data
            new_row_list = [
                nct_id if 'nct_id' in locals() else 'unknown',
                None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None
            ]

        rows_list.append(new_row_list)

    if errors_count > 0:
        logger.warning(f"Encountered errors processing {errors_count} out of {len(studies)} studies")

    studies_frame = pd.DataFrame(
        data=rows_list,
        columns=[
            'nct_id', 'official_title', 'brief_title', 'acronym',
            'overall_status', 'start_date', 'completion_date', 'last_update_date',
            'brief_summary', 'detailed_description',
            'conditions', 'interventions', 'intervention_types',
            'lead_sponsor', 'sponsor_class', 'collaborators',
            'countries', 'facilities',
            'min_age', 'max_age', 'gender', 'healthy_volunteers',
            'study_type', 'phase', 'enrollment',
            'primary_outcomes', 'secondary_outcomes',
            'study_url', 'citation'
        ]
    )
    return studies_frame

def _get_identification_info(study):
    """Extract identification information from study."""
    protocol_section = study.get('protocolSection', {})
    identification_module = protocol_section.get('identificationModule', {})

    return {
        'official_title': identification_module.get('officialTitle', ''),
        'brief_title': identification_module.get('briefTitle', ''),
        'acronym': identification_module.get('acronym', '')
    }

def _get_status_info(study):
    """Extract status and date information from study."""
    protocol_section = study.get('protocolSection', {})
    status_module = protocol_section.get('statusModule', {})

    # Extract dates
    start_date_struct = status_module.get('startDateStruct', {})
    start_date = start_date_struct.get('date', None)

    completion_date_struct = status_module.get('completionDateStruct', {}) or status_module.get('primaryCompletionDateStruct', {})
    completion_date = completion_date_struct.get('date', None)

    last_update_date = status_module.get('lastUpdatePostDateStruct', {}).get('date', None)

    return {
        'overall_status': status_module.get('overallStatus', None),
        'start_date': start_date,
        'completion_date': completion_date,
        'last_update_date': last_update_date
    }

def _get_description_info(study):
    """Extract description information from study."""
    protocol_section = study.get('protocolSection', {})
    description_module = protocol_section.get('descriptionModule', {})

    return {
        'brief_summary': description_module.get('briefSummary', None),
        'detailed_description': description_module.get('detailedDescription', None)
    }

def _get_conditions(study):
    """Extract conditions from study."""
    protocol_section = study.get('protocolSection', {})
    conditions_module = protocol_section.get('conditionsModule', {})
    conditions = conditions_module.get('conditions', [])

    return conditions if conditions else None

def _get_interventions(study):
    """Extract intervention information from study."""
    protocol_section = study.get('protocolSection', {})
    arms_interventions_module = protocol_section.get('armsInterventionsModule', {})
    interventions = arms_interventions_module.get('interventions', [])

    if interventions:
        intervention_names = []
        intervention_types = []

        for intervention in interventions:
            name = intervention.get('name', '')
            int_type = intervention.get('type', '')

            if name:
                intervention_names.append(name)
            if int_type:
                intervention_types.append(int_type)

        return {
            'interventions': intervention_names if intervention_names else None,
            'intervention_types': list(set(intervention_types)) if intervention_types else None
        }
    else:
        return {'interventions': None, 'intervention_types': None}

def _get_sponsor_info(study):
    """Extract sponsor information from study."""
    protocol_section = study.get('protocolSection', {})
    sponsor_collaborators_module = protocol_section.get('sponsorCollaboratorsModule', {})

    lead_sponsor = sponsor_collaborators_module.get('leadSponsor', {})
    collaborators = sponsor_collaborators_module.get('collaborators', [])

    collaborator_names = [collab.get('name', '') for collab in collaborators if collab.get('name')]

    return {
        'lead_sponsor': lead_sponsor.get('name', None),
        'sponsor_class': lead_sponsor.get('class', None),
        'collaborators': collaborator_names if collaborator_names else None
    }

def _get_locations(study):
    """Extract location information from study."""
    protocol_section = study.get('protocolSection', {})
    contacts_locations_module = protocol_section.get('contactsLocationsModule', {})
    locations = contacts_locations_module.get('locations', [])

    if locations:
        countries = []
        facilities = []

        for location in locations:
            country = location.get('country', '')
            facility = location.get('facility', '')

            if country:
                countries.append(country)
            if facility:
                facilities.append(facility)

        # Get unique countries
        unique_countries = list(set(countries)) if countries else None

        return {
            'countries': unique_countries,
            'facilities': facilities if facilities else None
        }
    else:
        return {'countries': None, 'facilities': None}

def _get_eligibility_info(study):
    """Extract eligibility criteria from study."""
    protocol_section = study.get('protocolSection', {})
    eligibility_module = protocol_section.get('eligibilityModule', {})

    return {
        'min_age': eligibility_module.get('minimumAge', None),
        'max_age': eligibility_module.get('maximumAge', None),
        'gender': eligibility_module.get('sex', None),
        'healthy_volunteers': eligibility_module.get('healthyVolunteers', None)
    }

def _get_design_info(study):
    """Extract design information from study."""
    protocol_section = study.get('protocolSection', {})
    design_module = protocol_section.get('designModule', {})

    # Get phases
    phases = design_module.get('phases', [])
    phase_str = ', '.join(phases) if phases else None

    # Get enrollment
    enrollment_info = design_module.get('enrollmentInfo', {})
    enrollment = enrollment_info.get('count', None)

    return {
        'study_type': design_module.get('studyType', None),
        'phase': phase_str,
        'enrollment': enrollment
    }

def _get_outcomes(study):
    """Extract outcome measures from study."""
    protocol_section = study.get('protocolSection', {})
    outcomes_module = protocol_section.get('outcomesModule', {})

    primary_outcomes = outcomes_module.get('primaryOutcomes', [])
    secondary_outcomes = outcomes_module.get('secondaryOutcomes', [])

    # Extract measure descriptions
    primary_measures = [outcome.get('measure', '') for outcome in primary_outcomes if outcome.get('measure')]
    secondary_measures = [outcome.get('measure', '') for outcome in secondary_outcomes if outcome.get('measure')]

    return {
        'primary_outcomes': primary_measures if primary_measures else None,
        'secondary_outcomes': secondary_measures if secondary_measures else None
    }

def _process_citation(title, nct_id, url, update_date):
    update_date_parts = update_date.split("-")
    update_date_obj = datetime(int(update_date_parts[0]), int(update_date_parts[1]), int(update_date_parts[2]))
    fmt_update_date = update_date_obj.strftime("%B %d, %Y")
    access_date =datetime.now()
    fmt_access_date = access_date.strftime("%B %d, %Y")
    citation = f"{title}. ClinicalTrials.gov identifier: {nct_id}. Updated {fmt_update_date}. Accessed {fmt_access_date}. {url}."
    return citation

def _add_date_filter(date_time_obj:datetime):
    """Add date filter to PubMed query"""
    start_date_str = f"AREA[StudyFirstPostDate]RANGE[{date_time_obj.strftime("%Y-%m-%d")},MAX]"
    return start_date_str
