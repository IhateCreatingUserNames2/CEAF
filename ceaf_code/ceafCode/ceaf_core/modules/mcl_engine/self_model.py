# ceaf_core/modules/mcl_engine/self_model.py

import logging
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# More robust import approach
GOAL_RECORD_DEFINED = False
try:
    from ..memory_blossom.memory_types import GoalRecord
    GOAL_RECORD_DEFINED = True
    logger.debug("self_model.py - GoalRecord imported successfully")
except ImportError:
    logger.debug("self_model.py - GoalRecord import failed, using Dict representation")
    try:
        # Try relative import as fallback
        from ..memory_blossom.memory_types import GoalRecord
        GOAL_RECORD_DEFINED = True
        logger.info("self_model.py - GoalRecord imported successfully via relative import")
    except ImportError:
        logger.warning("self_model.py - FAILED to import GoalRecord. Using simplified dict representation.")
        # GoalRecord not available, will use Dict[str, Any] for goals


class CeafSelfRepresentation(BaseModel):
    """
    Represents the AI's dynamic understanding of its own identity, capabilities,
    limitations, current goals, and operational persona.
    This model is managed and updated by the NCIM agent.
    """
    model_id: str = Field(default="ceaf_self_v1", description="Identifier for this version of the self-model schema.")
    model_version: str = Field(default="1.0.0", description="Version of the data within this self-model instance.")
    last_updated_ts: float = Field(default_factory=time.time, description="Timestamp of the last update to this model.")

    core_values_summary: str = Field(
        ...,
        description="A concise text summary of CEAF's core operational values and principles. This might be derived from or link to a more detailed manifesto (e.g., ceaf_manifesto.txt)."
    )
    perceived_capabilities: List[str] = Field(
        default_factory=list,
        description="A list of capabilities the AI believes it currently possesses (e.g., 'natural language understanding', 'tool_use_filesystem', 'ethical_reasoning_vre_v1'). Updated as new tools/skills are acquired or verified."
    )
    known_limitations: List[str] = Field(
        default_factory=list,
        description="A list of limitations the AI is aware of (e.g., 'cannot provide financial advice', 'real-time visual processing unavailable', 'emotional experience is simulated')."
    )

    # current_short_term_goals could be a list of full GoalRecord objects if tight integration is desired
    # or simplified dictionaries as specified in Improvements.txt for looser coupling.
    # Using Dict[str, Any] for flexibility as per the "Simplified from GoalRecord" note.
    current_short_term_goals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="A list of simplified representations of active short-term goals. Each dict might contain 'goal_id', 'description', 'status', 'priority'."
    )

    persona_attributes: Dict[str, str] = Field(
        default_factory=dict,
        description="Key-value pairs defining the AI's current operational persona (e.g., {'tone': 'helpful_reflective', 'disclosure_level': 'moderate', 'preferred_communication_style': 'clear_and_concise'})."
    )

    last_self_model_update_reason: str = Field(
        default="Initial model creation.",
        description="A brief description of why this self-model was last updated (e.g., 'VRE feedback on overconfidence', 'Successful use of new tool: X', 'MCL reflection cycle insights')."
    )

    # Optional field for more detailed history or provenance
    update_history_log_reference: Optional[str] = Field(
        None,
        description="Reference to a more detailed log of changes to this self-model over time (e.g., a specific memory ID in MBS)."
    )

    class Config:
        str_strip_whitespace = True
        validate_assignment = True # Ensures that updates to fields are also validated
