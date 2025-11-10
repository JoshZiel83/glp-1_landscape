import os
import sys
from pathlib import Path
from langchain.tools import tool, InjectedState
from langchain_openai import ChatOpenAI
from typing import Annotated
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from annotator.schema import MedicalConditionFilter, ConditionMappings, ConditionExtraction

# Add parent directory to path to import services
sys.path.append(str(Path(__file__).parent.parent))
from services import logging_config, mesh_mapper

load_dotenv()

# Initialize logger
logger = logging_config.get_logger(__name__)



def _extract_condition_from_context(row, llm: ChatOpenAI) -> str:
    """
    Use GPT-5 to extract the primary medical condition from trial row.

    Args:
        row: Named tuple from DataFrame.itertuples() with trial data
        llm: ChatOpenAI instance configured for GPT-5 with structured output

    Returns:
        Extracted condition string
    """
    # Safely access fields with getattr, providing "Not Available" as default
    brief_title = getattr(row, 'brief_title', 'Not Available')
    official_title = getattr(row, 'official_title', 'Not Available')
    brief_summary = getattr(row, 'brief_summary', 'Not Available')
    detailed_description = getattr(row, 'detailed_description', 'Not Available')
    primary_outcomes = getattr(row, 'primary_outcomes', 'Not Available')
    conditions = getattr(row, 'conditions', 'Not Available')

    # Format list fields as comma-separated strings
    if isinstance(primary_outcomes, list):
        primary_outcomes = ', '.join(str(o) for o in primary_outcomes)
    elif primary_outcomes is None or (isinstance(primary_outcomes, float) and pd.isna(primary_outcomes)):
        primary_outcomes = 'Not Available'

    if isinstance(conditions, list):
        conditions = ', '.join(str(c) for c in conditions)
    elif conditions is None or (isinstance(conditions, float) and pd.isna(conditions)):
        conditions = 'Not Available'

    # Handle None/NaN for text fields
    if detailed_description is None or (isinstance(detailed_description, float) and pd.isna(detailed_description)):
        detailed_description = 'Not Available'

    prompt = f"""You are a medical expert analyzing a clinical trial description.

Based on the trial information below, identify the PRIMARY medical condition or disease being studied.

Instructions:
- Extract the main disease/condition, not symptoms or study parameters
- Ignore non-medical items like "Quality of Life" or "Healthy"
- Use standard medical terminology when possible
- If the listed condition seems unclear, infer from the trial description
- Return a single, clear condition name suitable for MeSH database search

Trial Information:
- Brief Title: {brief_title}
- Official Title: {official_title}
- Brief Summary: {brief_summary}
- Detailed Description: {detailed_description}
- Primary Outcomes: {primary_outcomes}
- Conditions Listed: {conditions}

What is the primary medical condition being studied?"""

    response = llm.invoke(prompt)
    return response.extracted_condition


def _search_for_mesh_term(condition: str) -> str:
    """
    Search for MeSH term using the mesh_mapper service.

    Args:
        condition: Medical condition string to search

    Returns:
        Formatted MeSH term string or "NOT DETERMINED"
    """
    try:
        result = mesh_mapper.search_mesh_term(condition, filter_diseases_only=True)

        if result:
            mesh_term = f"{result['mesh_term']} (MeSH ID:{result['mesh_id']})"
            logger.info(f"Found MeSH term for '{condition}': {mesh_term}")
            return mesh_term
        else:
            logger.warning(f"No MeSH term found for '{condition}'")
            return "NOT DETERMINED"

    except Exception as e:
        logger.error(f"Error searching for MeSH term for '{condition}': {e}")
        return "NOT DETERMINED"


def _add_mesh_mappings(conditions, term_map):
    """
    Apply MeSH term mappings to a list of conditions.

    Args:
        conditions: List of condition strings from a trial
        term_map: Dictionary mapping condition -> MeSH term

    Returns:
        List of unique matched MeSH terms
    """
    matched_conditions = []
    if conditions is not None and len(conditions) > 0:  # Skip null/empty conditions
        for condition in conditions:
            if term_map.keys() and condition in term_map.keys():
                matched_term = term_map[condition]
                logger.info(f"Found a MeSH term mapping for {condition}")
                matched_conditions.append(matched_term)
            else:
                logger.debug(f"No existing MeSH term mapping for {condition}")
    return list(set(matched_conditions))


def _extract_unmapped_conditions(df: pd.DataFrame) -> list:
    """
    Extract unique unmapped conditions from the DataFrame.

    Returns a list of condition strings that appear in 'conditions' column
    for rows without matched_conditions.
    """
    # Filter for rows without matched_conditions (NaN or empty lists)
    unmapped_rows = df[df['matched_conditions'].isna() | (df['matched_conditions'].str.len() == 0)]

    # Explode the conditions lists, get unique values, convert to list
    unmapped = unmapped_rows['conditions'].explode().unique().tolist()

    return unmapped


def _extract_existing_mesh_terms(df: pd.DataFrame) -> list:
    """
    Extract unique MeSH terms (with IDs) from matched_conditions column.

    Returns a list of strings in format: "Term Name (MeSH ID:########)"
    """
    # Filter for rows with matched_conditions
    matched_rows = df[df['matched_conditions'].notna() & (df['matched_conditions'].str.len() > 0)]

    # Explode the matched_conditions lists, get unique values, convert to list
    mesh_terms = matched_rows['matched_conditions'].explode().unique().tolist()

    return mesh_terms


def _filter_medical_conditions(conditions: list, llm: ChatOpenAI) -> list:
    """
    Use GPT-5 to filter out non-medical conditions from the list.

    Args:
        conditions: List of condition strings
        llm: ChatOpenAI instance configured for GPT-5 with structured output

    Returns:
        List of conditions that are actual medical conditions
    """
    prompt = f"""You are a medical terminology expert. Given the following list of items from clinical trials,
identify which ones ARE actual medical conditions or diseases, and which are NOT.

Items that are NOT medical conditions include:
- General states like "Healthy", "Quality of Life"
- Study parameters or methods
- Non-disease related terms

Items to classify:
{conditions}
"""

    response = llm.invoke(prompt)
    return response.medical_conditions


def _map_conditions_to_mesh(unmapped: list, existing_mesh: list, llm: ChatOpenAI) -> dict:
    """
    Use GPT-5 to map unmapped conditions to existing MeSH terms.

    Args:
        unmapped: List of unmapped condition strings
        existing_mesh: List of existing MeSH terms with IDs
        llm: ChatOpenAI instance configured for GPT-5 with structured output

    Returns:
        Dictionary mapping original condition -> MeSH term (with ID)
    """
    prompt = f"""You are a medical terminology expert. Map each unmapped condition to the most appropriate
existing MeSH term from the provided list.

Consider:
- Synonyms and alternative names
- Abbreviations (e.g., "T2DM" = "Type 2 Diabetes Mellitus")
- Broader/narrower term relationships
- Common medical terminology variations

Only return mappings where you are confident there is a clear match. Assign confidence levels:
- 'high': Very confident match (synonyms, exact equivalents)
- 'medium': Good match (related terms, broader/narrower relationships)
- 'low': Possible match but uncertain

If a condition has no good match, omit it from the results.

Unmapped conditions:
{unmapped}

Existing MeSH terms:
{existing_mesh}
"""
    response = llm.invoke(prompt)

    # Create set of valid MeSH terms for fast lookup
    valid_mesh_terms = set(existing_mesh)

    # Validate and convert mappings - only include MeSH terms that exist in our list
    validated_mappings = {}
    rejected_count = 0
    for mapping in response.mappings:
        if mapping.mesh_term in valid_mesh_terms:
            validated_mappings[mapping.original_condition] = mapping.mesh_term
        else:
            rejected_count += 1
            logger.warning(f"Rejected invalid mapping: '{mapping.original_condition}' -> '{mapping.mesh_term}' (not in existing MeSH terms)")

    if rejected_count > 0:
        logger.warning(f"Total rejected mappings: {rejected_count}")

    return validated_mappings


@tool
def update_condition_mapping(
    data: Annotated[pd.DataFrame, InjectedState("working_data")],
    mesh_mappings_path: str = "/Users/joshziel/Documents/Coding/glp-1-landscape/data/mesh_term_mappings.csv"
) -> dict:
    """
    Automatically map unmapped conditions to existing MeSH terms using GPT-5.

    This tool:
    1. Extracts unique unmapped conditions from the DataFrame
    2. Filters out non-medical conditions using GPT-5
    3. Gets existing MeSH terms from matched_conditions
    4. Uses GPT-5 to intelligently map unmapped -> existing MeSH terms
    5. Updates mesh_term_mappings.csv and DataFrame

    Args:
        data: DataFrame with 'conditions' and 'matched_conditions' columns
        mesh_mappings_path: Path to mesh_term_mappings.csv file

    Returns:
        Dictionary with results:
        - new_mappings: Number of new mappings created
        - trials_affected: Number of trials updated
        - mappings_created: Dict of condition -> MeSH term mappings
    """
    # Initialize GPT-5 with structured outputs
    base_llm = ChatOpenAI(
        model="gpt-5",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Step 1: Extract unmapped conditions
    unmapped_conditions = _extract_unmapped_conditions(data)
    logger.info(f"Found {len(unmapped_conditions)} unique unmapped conditions")

    if not unmapped_conditions:
        return {
            "new_mappings": 0,
            "trials_affected": 0,
            "mappings_created": {},
            "message": "No unmapped conditions found"
        }

    # Step 2: Filter out non-medical conditions
    filter_llm = base_llm.with_structured_output(MedicalConditionFilter)
    unmapped_list = list(unmapped_conditions)
    filtered_conditions = _filter_medical_conditions(unmapped_list, filter_llm)
    logger.info(f"After filtering: {len(filtered_conditions)} legitimate medical conditions")

    # Step 3: Get existing MeSH terms
    existing_mesh = list(_extract_existing_mesh_terms(data))
    logger.info(f"Found {len(existing_mesh)} unique existing MeSH terms")

    # Step 4: Map conditions to MeSH terms
    mapping_llm = base_llm.with_structured_output(ConditionMappings)
    mappings = _map_conditions_to_mesh(filtered_conditions, existing_mesh, mapping_llm)
    logger.info(f"GPT-5 created {len(mappings)} mappings")

    if not mappings:
        return {
            "new_mappings": 0,
            "trials_affected": 0,
            "mappings_created": {},
            "message": "No suitable mappings found"
        }

    # Step 5: Update mesh_term_mappings.csv
    try:
        mesh_df = pd.read_csv(mesh_mappings_path, index_col=0)
    except FileNotFoundError:
        mesh_df = pd.DataFrame(columns=[0])
        mesh_df.index.name = 'Unnamed: 0'

    # Add new mappings
    for original_cond, mesh_term in mappings.items():
        mesh_df.loc[original_cond] = mesh_term

    # Save updated mappings
    mesh_df.to_csv(mesh_mappings_path)

    # Step 6: Update DataFrame matched_conditions using apply
    # Filter for rows with unmapped conditions (NaN or empty lists)
    unmapped_rows = data['matched_conditions'].isna() | (data['matched_conditions'].str.len() == 0)

    # Apply mapping function only to unmapped rows
    data.loc[unmapped_rows, 'matched_conditions'] = data.loc[unmapped_rows, 'conditions'].apply(
        lambda conditions: _add_mesh_mappings(conditions, mappings)
    )

    # Count how many rows were affected (had mappings applied)
    trials_affected = unmapped_rows.sum()

    return {
        "new_mappings": len(mappings),
        "trials_affected": trials_affected,
        "mappings_created": mappings,
        "message": f"Successfully created {len(mappings)} new mappings affecting {trials_affected} trials"
    }


@tool
def search_for_condition_mapping(
    data: Annotated[pd.DataFrame, InjectedState("working_data")],
    mesh_mappings_path: str = f"/Users/joshziel/Documents/Coding/glp-1-landscape/data/mesh_term_mappings.csv"
) -> dict:
    """
    Identify new MeSH mappings using GPT-5 + mesh_mapper for trials with ambiguous or missing condition information.

    This tool:
    1. Finds trials with missing matched_conditions
    2. Uses GPT-5 to extract medical conditions from trial context
    3. Uses mesh_mapper service to find MeSH terms
    4. Marks as "NOT DETERMINED" if no MeSH term found
    5. Updates mesh_term_mappings.csv and DataFrame

    Args:
        data: DataFrame with trial data
        mesh_mappings_path: Path to mesh_term_mappings.csv

    Returns:
        Dictionary with results:
        - trials_processed: Number of trials analyzed
        - new_mappings: Number of new MeSH mappings created
        - not_determined: Number of conditions marked as NOT DETERMINED
        - trials_updated: Number of trials that got new matches
        - mappings_created: Dict of condition -> MeSH term
    """
    # Initialize GPT-5 with structured output
    base_llm = ChatOpenAI(
        model="gpt-5",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    extraction_llm = base_llm.with_structured_output(ConditionExtraction)

    # Step 1: Load existing mesh mappings
    try:
        mesh_df = pd.read_csv(mesh_mappings_path, index_col=0)
        existing_mappings = mesh_df.iloc[:, 0].to_dict()
        logger.info(f"Loaded {len(existing_mappings)} existing mappings from {mesh_mappings_path}")
    except FileNotFoundError:
        existing_mappings = {}
        logger.warning(f"Mesh mappings file not found at {mesh_mappings_path}, starting with empty mappings")

    # Step 2: Filter for trials with missing matched_conditions
    missing_mask = data['matched_conditions'].isna() | (data['matched_conditions'].str.len() == 0)
    trials_to_process = data[missing_mask]

    logger.info(f"Found {len(trials_to_process)} trials with missing matched_conditions")

    if len(trials_to_process) == 0:
        return {
            "trials_processed": 0,
            "new_mappings": 0,
            "not_determined": 0,
            "trials_updated": 0,
            "mappings_created": {},
            "message": "No trials with missing matched_conditions found"
        }

    # Step 3: Process each trial and update matched_conditions directly
    new_mappings = {}
    not_determined_count = 0
    trials_updated = 0
    existing_used = 0

    for row in trials_to_process.itertuples():
        try:
            # Extract condition using GPT-5 directly from row
            logger.info(f"Processing trial {row.nct_id}...")
            extracted_condition = _extract_condition_from_context(row, extraction_llm)
            logger.info(f"GPT-5 extracted condition: '{extracted_condition}'")

            # Check if extracted condition already has a mapping
            if extracted_condition in existing_mappings:
                mesh_term = existing_mappings[extracted_condition]
                logger.info(f"Found existing mapping for '{extracted_condition}': {mesh_term}")
                existing_used += 1
            else:
                # Search for MeSH term using mesh_mapper
                logger.info(f"No existing mapping found, searching mesh_mapper for '{extracted_condition}'")
                mesh_term = _search_for_mesh_term(extracted_condition)

            # Update row directly
            if mesh_term != "NOT DETERMINED":
                data.at[row.Index, 'matched_conditions'] = [mesh_term]
                trials_updated += 1

            # Store mapping for CSV only if it's a NEW mapping (not from existing_mappings)
            if extracted_condition not in existing_mappings and extracted_condition not in new_mappings:
                new_mappings[extracted_condition] = mesh_term
                if mesh_term == "NOT DETERMINED":
                    not_determined_count += 1

        except Exception as e:
            logger.error(f"Error processing trial {getattr(row, 'nct_id', 'unknown')}: {e}")
            # Continue processing other trials
            continue

    logger.info(f"Used {existing_used} existing mappings")
    logger.info(f"Created {len(new_mappings)} new mappings ({not_determined_count} marked as NOT DETERMINED)")
    logger.info(f"Updated {trials_updated} trials with new MeSH terms")

    # Step 4: Update mesh_term_mappings.csv with only NEW mappings
    if new_mappings:
        # Reload or create mesh_df
        if existing_mappings:
            # Already loaded, convert back to DataFrame
            mesh_df = pd.DataFrame.from_dict(existing_mappings, orient='index', columns=[0])
        else:
            mesh_df = pd.DataFrame(columns=[0])
            mesh_df.index.name = 'Unnamed: 0'

        # Add new mappings (including NOT DETERMINED)
        for condition, mesh_term in new_mappings.items():
            mesh_df.loc[condition] = mesh_term

        # Save updated mappings
        mesh_df.to_csv(mesh_mappings_path)
        logger.info(f"Updated {mesh_mappings_path} with {len(new_mappings)} new mappings")
    else:
        logger.info("No new mappings to save to CSV")

    return {
        "trials_processed": len(trials_to_process),
        "existing_mappings_used": existing_used,
        "new_mappings": len(new_mappings) - not_determined_count,
        "not_determined": not_determined_count,
        "trials_updated": trials_updated,
        "mappings_created": new_mappings,
        "message": f"Processed {len(trials_to_process)} trials: used {existing_used} existing mappings, created {len(new_mappings)} new mappings ({not_determined_count} NOT DETERMINED)"
    }

@tool
def submit_for_batch_annotation(data:pd.DataFrame):
    """
    Submit trials for batch annotation.

    TODO: Implementation pending.
    """
    pass

@tool
def check_batch_annotation_status(batch_name:str)->bool:
    """
    Check the status of a batch annotation job.

    TODO: Implementation pending.
    """
    pass

@tool
def retrieve_batch_annotations(batch_name:str):
    """
    Retrieve results from a completed batch annotation job.

    TODO: Implementation pending.
    """
    pass

