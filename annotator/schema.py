from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal


class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"

class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROCESS = "in_process"
    COMPLETED ="completed"

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
    """Schema for GPT-5 condition extraction from trial context."""
    extracted_condition: str = Field(description="The primary medical condition/disease being studied, extracted from trial context")
    confidence: ConfidenceLevel = Field(description="Confidence level")
    reasoning: str = Field(description="Brief explanation of why this condition was identified")


class AgentDecision(BaseModel):
    """Schema for agent's decision on whether to continue working or end."""
    decision: Literal["work_on_a_task", "end"] = Field(
        description="Decision on next action: 'work_on_a_task' to continue with pending tasks, 'end' to stop when all tasks are complete"
    )
    reasoning: str = Field(description="Brief explanation for the decision")