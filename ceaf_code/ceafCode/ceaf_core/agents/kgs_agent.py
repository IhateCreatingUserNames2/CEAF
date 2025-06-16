# ceaf_core/agents/kgs_agent.py

import os
import logging
import json
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from .base_ceaf_agent import CeafBaseAgent
from ..config.model_configs import get_agent_model_name  # Assuming you add KGS model vars here

logger = logging.getLogger("KGSAgent")


# --- Pydantic Models for KGS Agent's LLM Output ---

class KGEntity(BaseModel):
    id_str: str = Field(...,
                        description="A unique, canonical string identifier for the entity (e.g., person_john_doe, concept_ai_ethics). Normalize spaces to underscores and lowercase.")
    label: str = Field(..., description="Human-readable label for the entity.")
    type: str = Field(...,
                      description="Type of the entity (e.g., person, organization, concept, event, product, location). Try to use KGEntityType enum values if applicable.")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Key-value properties of the entity.")
    description: Optional[str] = Field(None, description="A short description of the entity.")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for this entity.")


class KGRelation(BaseModel):
    source_id_str: str = Field(..., description="ID string of the source entity.")
    target_id_str: str = Field(..., description="ID string of the target entity.")
    label: str = Field(...,
                       description="Label for the relationship (e.g., works_at, located_in, part_of, causes, reported_by).")
    context: Optional[str] = Field(None,
                                   description="Brief context or evidence for this relationship from the source text.")
    attributes: Dict[str, Any] = Field(default_factory=dict,
                                       description="Key-value properties of the relationship itself.")


class KGSynthesisOutput(BaseModel):
    extracted_entities: List[KGEntity] = Field(default_factory=list)
    extracted_relations: List[KGRelation] = Field(default_factory=list)
    summary_of_synthesis: Optional[str] = Field(None, description="Brief summary of what was processed or any issues.")
    unprocessed_or_ambiguous_info: List[str] = Field(default_factory=list,
                                                     description="Information that was difficult to parse into the graph.")


# --- KGS Agent Configuration ---
KGS_AGENT_DEFAULT_MODEL_ENV_VAR = "KGS_AGENT_MODEL"  # Add to model_configs.py
KGS_AGENT_FALLBACK_MODEL = "openrouter/openai/gpt-4.1"  # Good for structured data

KGS_AGENT_SPECIFIC_INSTRUCTION = """
You are the Knowledge Graph Synthesizer Agent (KGS Agent) within CEAF.
Your primary function is to process textual information from CEAF's memories (primarily ExplicitMemory objects)
and extract structured knowledge in the form of entities and relationships to build or refine CEAF's internal Knowledge Graph.

**Input Analysis:**
You will receive a batch of text snippets or structured data from CEAF memories. This data might include:
- Raw text from user interactions or ORA responses.
- Structured data fields already present in memories (e.g., `extracted_entities`, `extracted_relations` from a previous, simpler extraction).
- Summaries of longer documents.

**Your Task:**
1.  **Entity Extraction:** Identify key entities (e.g., people, organizations, locations, concepts, events, products, system components). For each entity:
    *   Assign a `label` (human-readable name).
    *   Generate a unique, canonical `id_str` (e.g., lowercase, underscores for spaces, prefixed by type like 'person_john_doe').
    *   Determine its `type` (e.g., person, organization, concept, event). Use standard types where possible.
    *   Extract relevant `attributes` as key-value pairs.
    *   Note any `aliases`.
    *   Provide a brief `description` if discernible.
2.  **Relation Extraction:** Identify relationships between the extracted entities. For each relationship:
    *   Identify the `source_id_str` and `target_id_str` of the related entities.
    *   Assign a `label` for the relationship (e.g., "works_at", "is_part_of", "reported_by", "caused_by").
    *   Extract any `attributes` of the relationship itself (e.g., date, confidence).
    *   Provide a `context` snippet from the source text that supports this relationship.
3.  **Normalization & Coreference (Best Effort):**
    *   Attempt to normalize entity labels (e.g., "AI" and "Artificial Intelligence" might refer to the same concept). Use the canonical `id_str` to link them.
    *   Resolve simple coreferences if possible (e.g., "John said he..." -> John is the 'he').
4.  **Output Formatting:** Return the extracted knowledge as a JSON object adhering to the `KGSynthesisOutput` schema, containing `extracted_entities` and `extracted_relations`.
    *   If certain information is ambiguous or cannot be confidently parsed, include it in `unprocessed_or_ambiguous_info`.

**Example Input Snippet (from an ExplicitMemory):**
```json
{
  "text_content": "User Jane Doe (jane.doe@email.com) reported that the 'Phoenix Project' system, version 2.1, is experiencing slowdowns after the recent 'Omega Update' on 2023-10-26. The issue seems related to the database module.",
  "structured_data": {
    "user_email": "jane.doe@email.com",
    "project_name": "Phoenix Project",
    "system_version": "2.1",
    "event": "Omega Update",
    "event_date": "2023-10-26"
  }
}
```

**Expected JSON Output (KGSynthesisOutput structure):**
```json
{
  "extracted_entities": [
    {
      "id_str": "user_jane_doe",
      "label": "Jane Doe",
      "type": "user",
      "attributes": {"email": "jane.doe@email.com"},
      "description": "User who reported an issue.",
      "aliases": []
    },
    {
      "id_str": "system_phoenix_project",
      "label": "Phoenix Project",
      "type": "system_component",
      "attributes": {"version": "2.1"},
      "description": "The system experiencing slowdowns.",
      "aliases": []
    }
  ],
  "extracted_relations": [
    {
      "source_id_str": "user_jane_doe",
      "target_id_str": "system_phoenix_project",
      "label": "reported_issue_with",
      "context": "User Jane Doe ... reported that the 'Phoenix Project' system ... is experiencing slowdowns",
      "attributes": {"issue_type": "slowdowns"}
    }
  ],
  "summary_of_synthesis": "Processed memory snippet about Phoenix Project issue.",
  "unprocessed_or_ambiguous_info": []
}
```

Focus on accuracy and structured output. Avoid making assumptions not supported by the text.
"""


# --- KGS Agent Definition ---
def create_kgs_agent():
    """Factory function to create the KGS Agent with proper error handling."""
    try:
        kgs_agent_model_name = get_agent_model_name(
            agent_type_env_var=KGS_AGENT_DEFAULT_MODEL_ENV_VAR,
            fallback_model=KGS_AGENT_FALLBACK_MODEL
        )

        agent = CeafBaseAgent(
            name="KnowledgeGraphSynthesizer_Agent",
            default_model_env_var=KGS_AGENT_DEFAULT_MODEL_ENV_VAR,
            fallback_model_name=KGS_AGENT_FALLBACK_MODEL,
            description="Extracts entities and relationships from textual memories to build CEAF's knowledge graph.",
            specific_instruction=KGS_AGENT_SPECIFIC_INSTRUCTION,
            tools=[],  # Tools the KGS agent itself can call
        )

        logger.info(
            f"KnowledgeGraphSynthesizer_Agent defined successfully with model '{getattr(agent.model, 'model', kgs_agent_model_name)}'.")
        return agent

    except Exception as e:
        logger.critical(f"Failed to define KnowledgeGraphSynthesizer_Agent: {e}", exc_info=True)
        return None


# Create the agent instance
knowledge_graph_synthesizer_agent = create_kgs_agent()

