import pickle
from pathlib import Path
import pandas as pd
from datetime import datetime
from services import logging_config, mesh_mapper
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime



logger = logging_config.get_logger(__name__)

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

class MedicalConditionFilter(BaseModel):
    """Schema for filtering medical conditions from non-medical items."""
    medical_conditions: list[str] = Field(description="List of items that are actual medical conditions or diseases")
    non_medical_items: list[str] = Field(description="List of items that are NOT medical conditions (e.g., 'Healthy', 'Quality of Life', study parameters)")

class ConditionMapping(BaseModel):
    """Schema for a single condition to MeSH term mapping."""
    original_condition: str = Field(description="The unmapped condition string from the clinical trial")
    mesh_term: str = Field(description="The matched MeSH term with ID in format: 'Term Name (MeSH ID:########)'")
    confidence: ConfidenceLevel = Field(description="Confidence level for this mapping")

class ConditionMappings(BaseModel):
    """Schema for multiple condition to MeSH term mappings."""
    mappings: list[ConditionMapping] = Field(description="List of condition to MeSH term mappings. Only include mappings with clear matches.")

class ConditionExtraction(BaseModel):
    """Schema for condition extraction from trial context."""
    extracted_condition: str = Field(description="The primary medical condition/disease being studied, extracted from trial context")
    confidence: ConfidenceLevel = Field(description="Confidence level")
    reasoning: str = Field(description="Brief explanation of why this condition was identified")

class TherapeuticCategory(Enum):
    """Enum to define therapeutic categories."""
    ONCOLOGY = "oncology"
    CARDIOVASCULAR = "cardiovascular"
    CENTRAL_NERVOUS_SYSTEM = "central_nervous_system"
    RESPIRATORY = "respiratory"
    GASTROINTESTINAL = "gastrointestinal"
    ENDOCRINOLOGY = "endocrinology"
    INFECTIOUS_DISEASES = "infectious_diseases"
    IMMUNOLOGY = "immunology"
    DERMATOLOGY = "dermatology"
    OPHTHALMOLOGY = "ophthalmology"
    RHEUMATOLOGY = "rheumatology"
    NEPHROLOGY = "nephrology"
    HEMATOLOGY = "hematology"
    PAIN_MANAGEMENT = "pain_management"
    WOMENS_HEALTH = "womens_health"
    MENS_HEALTH = "mens_health"
    PEDIATRICS = "pediatrics"
    GERIATRICS = "geriatrics"
    RARE_DISEASES = "rare_diseases"
    VACCINES = "vaccines"
    MEDICAL_DEVICES = "medical_devices"
    DIAGNOSTICS = "diagnostics"
    OTHER = "other"

class TxCategoryAnnotation(BaseModel):
    """Schema for therapeutic category extraction and annotation"""
    tx_category: TherapeuticCategory = Field(..., description="the therapeutic category most relevant for the condition")
    confidence: ConfidenceLevel = Field(..., description="Your confidence in the organ system selection rated as very_high, high, moderate, low, or very_low.")
    explanation: str = Field(..., description="A brief explanation justifying the organ system selection and inclusion decision and any uncertainty.")

class TxCategoryClassifier(BaseModel):
    """Optional wrapper for category extraction and annotation"""
    tx_category_classification: Optional[TxCategoryAnnotation] = Field(description = "Classification of the therapeutic category for the condition possible")

class AnnotatorWorkflow:
    """
    A workflow manager for annotating clinical trial data with standardized medical terminology.
    
    This class orchestrates the process of enriching clinical trial datasets by mapping trial 
    conditions to Medical Subject Headings (MeSH) terms and classifying trials into therapeutic 
    categories. It uses a Large Language Model (LLM) to intelligently extract, match, and 
    categorize medical conditions from trial descriptions.
    
    The workflow performs three main annotation tasks:
    1. Maps unmapped trial conditions to existing MeSH terms using synonym matching
    2. Searches for new MeSH terms for conditions that couldn't be mapped to existing terms
    3. Classifies conditions into broad therapeutic categories (e.g., ONCOLOGY, CARDIOVASCULAR)
    
    Attributes:
        annotator: LLM instance compatible with LangChain structured outputs for medical annotation
        original_data (pd.DataFrame): Original unmodified trial dataset
        working_data (pd.DataFrame): Current state of the dataset being annotated
        annotated_data (pd.DataFrame): Final annotated dataset after workflow completion
        save_path (str): Directory path for saving/loading mappings and annotated data
        mesh_map (dict): Dictionary mapping condition names to MeSH terms with IDs
        tx_category_map (dict): Dictionary mapping MeSH terms to therapeutic categories
    
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> workflow = AnnotatorWorkflow(df=trials_df, llm=llm, data_loc="./data")
        >>> workflow.run_annotation_workflow()
        >>> annotated_trials = workflow.annotated_data
    
    Notes:
        - The workflow is designed to be fault-tolerant, continuing even if individual 
          annotation steps fail
        - All mappings are persisted to CSV files with timestamps for reproducibility
        - The class maintains both existing and newly discovered mappings to avoid 
          redundant LLM calls
        - Requires a LangChain-compatible LLM that supports structured output formatting
    """
    def __init__(self, df:pd.DataFrame, llm, data_loc:str = None):
        """
        Initialize the AnnotatorWorkflow with trial data and configuration.
    
        Sets up the annotation workflow by loading the input dataset, configuring the LLM 
        annotator, establishing the save location for outputs, and loading any existing 
        MeSH term mappings to avoid redundant annotation work.
    
        Args:
            df (pd.DataFrame): Clinical trials dataset generated by the data retriever notebook.
                Expected to contain columns including 'conditions', 'matched_conditions', 
                'brief_title', 'official_title', 'brief_summary', 'detailed_description',
                'primary_outcomes', and 'llm_annotations'.
            llm: LangChain-wrapped LLM instance (e.g., ChatOpenAI, ChatAnthropic) that 
                supports structured output generation via `.with_structured_output()`. 
                Used for all medical terminology extraction and mapping tasks.
            data_loc (str, optional): Directory path where annotation mappings and outputs 
                will be saved and retrieved. If None, defaults to the directory containing 
                the current file. Defaults to None.
    
        Raises:
            Warning: Logs a warning if no existing MeSH mappings file is found or if the 
                file cannot be loaded.
    
        Attributes Initialized:
            annotator: The provided LLM instance
            original_data: Immutable copy of the input DataFrame
            working_data: Mutable copy used during annotation process
            save_path: Resolved path for data persistence
            mesh_map: Dictionary of existing condition-to-MeSH mappings loaded from disk
            tx_category_map: Initially None, populated during therapeutic category mapping
            annotated_data: Initially None, populated after workflow completion
    
        Example:
            >>> from langchain_anthropic import ChatAnthropic
            >>> llm = ChatAnthropic(model="claude-sonnet-4-20250514")
            >>> trials_df = pd.read_csv("cleaned_trials.csv")
            >>> workflow = AnnotatorWorkflow(
            ...     df=trials_df, 
            ...     llm=llm, 
            ...     data_loc="./annotation_outputs"
            ... )
            >>> # Workflow is now ready to run
        """
        self.annotator = llm
        self.original_data = df
        if data_loc is None:
            data_loc = Path(__file__).resolve()
        self.save_path = data_loc
        self.working_data = df.copy()
        self.mesh_map = self._load_mesh_map()
        if len(self.mesh_map.keys())>0:
            logger.info(f"Loaded existing MeSH map containing {len(self.mesh_map.keys())} mappings as a {type(self.mesh_map)}")
        else:
            logger.warning("No items loaded from existing MeSH map")
        self.tx_category_map = None
        self.annotated_data = None
    
    def run_annotation_workflow(self):
        """
        Runs the entire workflow using annotator
        """
        try:
            logger.info("Trying to update MeSH mappings using existing terms")
            updated_data = self.update_using_existing_mesh_terms()
            if updated_data is not None:
                self.working_data = updated_data.copy()
                logger.info("Successfully updated MeSH mappings using existing terms")
        except Exception as e:
            logger.error(f"Warning - error when trying to update MeSH term mappings with existing terms: {e}")
        try:
            logger.info("Trying to update MeSH mappings by searching for missing terms")
            updated_data = self.search_for_missing_mesh_terms()
            if updated_data is not None:
                self.working_data = updated_data.copy()
                logger.info("Successfully updated MeSH mappings after searching for missing terms")
        except Exception as e:
            logger.error(f"Warning - error when trying to search for new MeSH term mapptings: {e}")
        try:
            logger.info("Trying to map therapeutic categores")
            updated_data = self.map_therapeutic_categories()
            if updated_data is not None:
                self.working_data = updated_data.copy()
                logger.info("Successfully mapped therapeutic categories")
        except Exception as e:
            logger.error(f"Warning - error when trying to map therapeutic categories: {e}")
        self.annotated_data = self.working_data.copy()
        self.annotated_data.to_csv(f"""{self.save_path}/annotated_trials_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv""")
        with open(f"""{self.save_path}/annotated_trials_{datetime.now().strftime("%Y%m%d%H%M%S")}.pkl""", 'wb') as f:
            pickle.dump(self.annotated_data, f)

    def update_using_existing_mesh_terms(self)->pd.DataFrame | None: 
        """
        Automatically try to map unmapped conditions to existing MeSH terms using Annotator.
        """
        data = self.working_data.copy()
        #Extract unmapped conditions
        unmapped_conditions = self._extract_unmapped_information(data, "matched_conditions", "conditions")
        logger.info(f"Found {len(unmapped_conditions)} unique unmapped conditions")

        if not unmapped_conditions:
            return None

        #Filter out non-medical conditions
        unmapped_list = list(unmapped_conditions)
        filtered_conditions = self._filter_for_medical_conditions(unmapped_list)
        logger.info(f"After filtering: {len(filtered_conditions)} legitimate medical conditions")

        #Get existing MeSH terms
        existing_mesh = list(self._extract_existing_mesh_terms(data))
        logger.info(f"Found {len(existing_mesh)} unique existing MeSH terms")

        #Map conditions to MeSH terms
        mappings = self._map_conditions_to_mesh(filtered_conditions, existing_mesh)
        logger.info(f"Created {len(mappings)} mappings")

        if not mappings:
            return None


        # Save new mappings
        pd.DataFrame.from_dict(mappings, orient = 'index').to_csv(f"""{self.save_path}/llm_mesh_term_mappings{datetime.now().strftime("%Y%m%d%H%M%S")}.csv""")
        mappings.update(self.mesh_map)
        self.mesh_map = mappings


        # Update DataFrame matched_conditions using apply
        # Filter for rows with unmapped conditions (NaN or empty lists)
        unmapped_rows = data['matched_conditions'].isna() | (data['matched_conditions'].str.len() == 0)

        # Apply mapping function only to unmapped rows
        
        data.loc[unmapped_rows, 'matched_conditions'] = data.loc[unmapped_rows, 'conditions'].apply(
            lambda conditions: self._apply_mappings(conditions, self.mesh_map)
        )
        data.loc[unmapped_rows, 'llm_annotations'] = data.loc[unmapped_rows].apply(
            lambda row: list(set(row['llm_annotations'] + ["matched_conditions"])) if len(row['matched_conditions']) > 0 else row['llm_annotations'],
            axis=1
        )
    
        # Count how many rows were affected (had mappings applied)
        trials_affected = unmapped_rows.sum()
        logger.info(f"Successfully created {len(mappings)} new mappings affecting {trials_affected} trials",)

        return data
    
    def search_for_missing_mesh_terms(self) -> pd.DataFrame | None:
        """
        Identify new MeSH mappings using Annotator + mesh_mapper for trials with ambiguous or missing condition information.
        """
        data = self.working_data.copy()


        # Filter for trials with missing matched_conditions
        missing_mask = data['matched_conditions'].isna() | (data['matched_conditions'].str.len() == 0)
        trials_to_process = data[missing_mask]

        logger.info(f"Found {len(trials_to_process)} trials with missing matched_conditions")

        if len(trials_to_process) == 0:
            return None

        # Process each trial and update matched_conditions directly
        new_mappings = {}
        not_determined_count = 0
        trials_updated = 0
        existing_used = 0

        for row in trials_to_process.itertuples():
            try:
                # Extract condition using Annotator directly from row
                logger.info(f"Processing trial {row.nct_id}...")
                extracted_condition = self._extract_condition_from_context(row)
                logger.info(f"Annotator extracted condition: '{extracted_condition}'")

                # Check if extracted condition already has a mapping
                if extracted_condition in self.mesh_map:
                    mesh_term = self.mesh_map[extracted_condition]
                    logger.info(f"Found existing mapping for '{extracted_condition}': {mesh_term}")
                    existing_used += 1
                else:
                    # Search for MeSH term using mesh_mapper
                    logger.info(f"No existing mapping found, searching mesh_mapper for '{extracted_condition}'")
                    mesh_term = self._search_for_mesh_term(extracted_condition)

                # Update row directly
                if mesh_term:
                    data.at[row.Index, 'matched_conditions'] = [mesh_term]
                    llm_annotations = data.at[row.Index, 'llm_annotations']
                    llm_annotations.append("matched_conditions")
                    data.at[row.Index, 'llm_annotations'] = list(set(llm_annotations))
                    trials_updated += 1
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

        if new_mappings:
            mesh_df = pd.DataFrame.from_dict(new_mappings, orient='index', columns=[0])
            mesh_df.to_csv(f"""{self.save_path}/llm_mesh_term_mappings{datetime.now().strftime("%Y%m%d%H%M%S")}.csv""")
    
        return data
    
    def map_therapeutic_categories(self) -> pd.DataFrame | None:
        """
        Uses Annotator to map each trial to a more general therapeutic category
        """
        data = self.working_data.copy()
        mask = data['matched_conditions'].notna() & data['matched_conditions'].apply(lambda x: x != ["NOT DETERMINED"])
        trials_to_process = data[mask]
        logger.info(f"Attempting to map {trials_to_process.shape[0]} trials to therapeutic categories")
        unique_conditions = trials_to_process["matched_conditions"].explode().dropna().unique().tolist()
        #logger.info(f"{unique_conditions}")
        self.tx_category_map = self._build_txcategory_map(unique_conditions)
        if len(self.tx_category_map.keys())>0:
            data["therapeutic_category"] = data["matched_conditions"].apply(self._apply_mappings, term_map=self.tx_category_map)
            mask = data["therapeutic_category"].notna()
            data.loc[mask, 'llm_annotations'] = data.loc[mask].apply(
                lambda row: list(set(row['llm_annotations'] + ["therapeutic_category"])),
                axis=1
            )
            return data
        else:
            return None
    
    def _load_mesh_map(self):
        try:
            existing_mappings = pd.read_csv(f"{self.save_path}/mesh_term_mappings.csv")
            existing_mappings.columns=["condition", "matched_mesh_term"]
            existing_mappings_dict = existing_mappings.set_index('condition')['matched_mesh_term'].to_dict()
        except Exception as e:
            logger.warning(f"Unable to load existing MeSH term mappings:{e}")
            existing_mappings_dict = {}
        return existing_mappings_dict

    def _extract_unmapped_information(self, df: pd.DataFrame, mapping_column:str, source_column:str) -> list:
        """
        Extract unique unmapped from the DataFrame.
        """
        unmapped_rows = df[df[mapping_column].isna() | (df[mapping_column].str.len() == 0)]
        unmapped = unmapped_rows[source_column].explode().unique().tolist()

        return unmapped
    
    def _filter_for_medical_conditions(self, conditions: list) -> list:
        """
        Use the Annotator to filter out non-medical conditions from the list.
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

        response = self.annotator.with_structured_output(MedicalConditionFilter).invoke(prompt)
        return response.medical_conditions
    
    def _extract_existing_mesh_terms(self, df: pd.DataFrame) -> list:
        """
        Extract unique MeSH terms (with IDs) from matched_conditions column.
        """
        matched_rows = df[df['matched_conditions'].notna() & (df['matched_conditions'].str.len() > 0)]
        mesh_terms = matched_rows['matched_conditions'].explode().unique().tolist()

        return mesh_terms
    
    def _map_conditions_to_mesh(self, unmapped: list, existing_mesh: list) -> dict:
        """
        Use Annotator to map unmapped conditions to existing MeSH terms.
        """
        prompt = f"""You are a medical terminology expert. Map each unmapped condition to the most appropriate
        existing MeSH term from the provided list.

        Consider:
        - Synonyms and alternative names
        - Abbreviations (e.g., "T2DM" = "Type 2 Diabetes Mellitus")
        - Broader/narrower term relationships
        - Common medical terminology variations

        Only return mappings where you are confident there is a clear match and assign your confidence level.

        If a condition has no good match, omit it from the results.

        Unmapped conditions:
        {unmapped}

        Existing MeSH terms:
        {existing_mesh}
        """
        response = self.annotator.with_structured_output(ConditionMappings).invoke(prompt)

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
                logger.info(f"Rejected invalid mapping: '{mapping.original_condition}' -> '{mapping.mesh_term}' (not in existing MeSH terms)")

            if rejected_count > 0:
                logger.info(f"Total rejected mappings: {rejected_count}")

        return validated_mappings

    def _extract_condition_from_context(self, row) -> str:
        """
        Use annotator to extract the primary medical condition from trial row.
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
        - If the main disease condition reflects a subset of patients, you should extract main disease only (for example: diabetes in patients with renal impairment -> diabetes)
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

        response = self.annotator.with_structured_output(ConditionExtraction).invoke(prompt)
        return response.extracted_condition
    
    def _search_for_mesh_term(self, condition: str) -> str:
        """
        Search for MeSH term using the mesh_mapper service.
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
    
    def _build_txcategory_map(self, conditions_list) -> dict:
        prompt = """You are an expert medical research assistant.  Your task is to classify the conditions below according therapeutic category:\n
        #REQUIREMENTS\n
        -To make the classification, consider the medical specialty most likely to treat the condition\n
            -for example, for metabolic conditions like diabetes or obesity, the most appropriate classification would be ENDOCRINOLOGY\n
        -Do not make any classifications you are unsure of\n

        #THERAPEUTIC CATEGORIES\n
        -ONCOLOGY - Cancers and malignant tumors of all types\n
        -CARDIOVASCULAR - Heart disease, hypertension, stroke, vascular conditions\n
        -CENTRAL_NERVOUS_SYSTEM - Neurological disorders, psychiatric conditions, brain and spinal cord diseases\n
        -RESPIRATORY - Lung diseases, asthma, COPD, pulmonary conditions\n
        -GASTROINTESTINAL - Digestive system disorders, liver disease, inflammatory bowel disease\n
        -ENDOCRINOLOGY - Diabetes, thyroid disorders, hormonal imbalances, metabolic conditions\n
        -INFECTIOUS_DISEASES - Bacterial, viral, fungal, and parasitic infections\n
        -IMMUNOLOGY - Autoimmune diseases, immune system disorders, allergies\n
        -DERMATOLOGY - Skin conditions, hair and nail disorders\n
        -OPHTHALMOLOGY - Eye diseases and vision disorders\n
        -RHEUMATOLOGY - Arthritis, joint diseases, musculoskeletal inflammatory conditions\n
        -NEPHROLOGY - Kidney diseases and renal disorders\n
        -HEMATOLOGY - Blood disorders, clotting disorders, anemia\n
        -PAIN_MANAGEMENT - Chronic pain conditions and pain-related disorders\n
        -WOMENS_HEALTH - Conditions specific to women including reproductive health, pregnancy, menopause\n
        -MENS_HEALTH - Conditions specific to men including reproductive and prostate health\n
        -PEDIATRICS - Conditions specifically affecting children and adolescents\n
        -GERIATRICS - Conditions primarily affecting elderly populations\n
        -RARE_DISEASES - Orphan diseases and ultra-rare conditions\n
        -VACCINES - Preventive immunizations and vaccine development\n
        -MEDICAL_DEVICES - Trials testing medical devices, implants, or equipment\n
        -DIAGNOSTICS - Diagnostic tests, biomarkers, and screening tools\n
        -OTHER - if the most appropriate category does not exist in this lsit\n
        
        #CONDITION TO CLASSIFY\n
        {condition}
        """
        mapping_dict = {}
        mapped_conditions = 0
        logger.info(f"Attempting to get mappings from Annotator for {len(conditions_list)} conditions")
        for condition in conditions_list:
            i = True
            count = 0
            while i and count <4:
                try:
                    formatted_prompt=prompt.format(condition = condition)
                    response = self.annotator.with_structured_output(TxCategoryClassifier).invoke(formatted_prompt)
                    count +=1
                    if hasattr(response, "tx_category_classification"):
                        logger.info(f"Recieved classification from Annotator")
                        classification = response.tx_category_classification
                        category = classification.tx_category.value.upper()
                        confidence = classification.confidence
                        explanation = classification.explanation
                        mapping_dict[condition] = category
                        mapped_conditions += 1
                        logger.info(f"Mapped {condition} to {category} with {confidence.name} confidence: {explanation}")
                        i = False
                    else:
                        logger.info(f"Annotator did not map a category for {condition} on try {count}")
                        mapping_dict[condition] = "NOT_DETERMINED"
                        i = False
                except Exception as e:
                    count += 1
                    logger.error(f"Error annotating {condition} on try {count}: {e}")
                    continue
        logger.info(f"Annotator mapped {mapped_conditions} of {len(conditions_list)} - {len(conditions_list)-mapped_conditions} are unmapped")
        mapping_frame = pd.DataFrame.from_dict(mapping_dict, orient = 'index')
        mapping_frame.to_csv(f"{self.save_path}/tx_category_mappings.csv")

        return mapping_dict     
        
    def _apply_mappings(self, mapping_list, term_map):
        mappings = []
        if mapping_list is not None and len(mapping_list)>0:  
            for item in mapping_list:
                if term_map.keys() and item in term_map.keys():
                    matched_term = term_map[item]
                    logger.info(f"Found a mapping for {item}")
                    mappings.append(matched_term)
                else:
                    logger.info(f"No existing mapping for {item}")
                return list(set(mappings))