# ceaf_core/tools/kgs_tools.py

import logging
import json
from typing import Dict, Any, Optional, List, cast
import asyncio



from google.adk.tools import FunctionTool, ToolContext

from .common_utils import create_successful_tool_response, create_error_tool_response, sanitize_text_for_logging

logger = logging.getLogger("KGSTools")

# --- Initialize flags for loaded modules/functions ---
MBS_AND_TYPES_LOADED = False
LIFECYCLE_FUNCTIONS_AVAILABLE = False


# --- Define Placeholders ---
# These will be used if the real imports fail.

class _Placeholder_MBSMemoryService:
    """Placeholder for MBSMemoryService when real import fails."""

    async def add_specific_memory(self, record_object: Any):
        logger.warning("Using dummy async MBSMemoryService.add_specific_memory")

    _in_memory_explicit_cache: List = []
    _in_memory_kg_entities_cache: List = []
    _in_memory_kg_relations_cache: List = []
    store_path: Any = type('DummyPath', (), {
        'exists': lambda: False,
        'mkdir': lambda **k: None,
        'unlink': lambda: None
    })()


class _Placeholder_KGS_KGEntity:
    """Placeholder for KGS_KGEntity when real import fails."""
    pass


class _Placeholder_KGS_KGRelation:
    """Placeholder for KGS_KGRelation when real import fails."""
    pass


class _Placeholder_ExplicitMemory:
    """Placeholder for ExplicitMemory when real import fails."""
    pass


class _Placeholder_KGEntityRecord:
    """Placeholder for KGEntityRecord when real import fails."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, **kwargs):
        return self.__dict__


class _Placeholder_KGRelationRecord:
    """Placeholder for KGRelationRecord when real import fails."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, **kwargs):
        return self.__dict__


class _Placeholder_BaseMemory:
    """Placeholder for BaseMemory when real import fails."""
    pass


class _Placeholder_MemorySourceType:
    """Placeholder for MemorySourceType when real import fails."""
    INTERNAL_REFLECTION = "internal_reflection"


class _Placeholder_MemorySalience:
    """Placeholder for MemorySalience when real import fails."""
    MEDIUM = "medium"


class _Placeholder_KGEntityType:
    """Placeholder for KGEntityType when real import fails."""
    OTHER = "other"


def _placeholder_update_dynamic_salience(mem, access_type):
    """Placeholder for update_dynamic_salience when real import fails."""
    pass


# --- Store reference to real class for type checking ---
_RealMBSClassForCheck = None

# --- Attempt to import real modules and functions ---
try:
    from ..services.mbs_memory_service import MBSMemoryService
    from ..agents.kgs_agent import KGEntity as KGS_KGEntity, KGRelation as KGS_KGRelation
    from ..modules.memory_blossom.memory_types import (
        ExplicitMemory, KGEntityRecord, KGRelationRecord, MemorySourceType, MemorySalience, KGEntityType, BaseMemory
    )

    # Store the real class for type checking
    _RealMBSClassForCheck = MBSMemoryService
    MBS_AND_TYPES_LOADED = True
    logger.info("KGS Tools: Successfully loaded core MBS and types.")

    try:
        from ..modules.memory_blossom.memory_lifecycle_manager import update_dynamic_salience

        LIFECYCLE_FUNCTIONS_AVAILABLE = True
        logger.info("KGS Tools: Successfully loaded lifecycle functions.")
    except ImportError as e_lifecycle:
        update_dynamic_salience = _placeholder_update_dynamic_salience  # type: ignore
        logger.warning(f"KGS Tools: Lifecycle function 'update_dynamic_salience' not loaded: {e_lifecycle}. "
                       "Salience updates will use placeholder.")

except ImportError as e_core:
    # Assign placeholders if core imports fail
    MBSMemoryService = _Placeholder_MBSMemoryService  # type: ignore
    _RealMBSClassForCheck = _Placeholder_MBSMemoryService
    KGS_KGEntity = _Placeholder_KGS_KGEntity  # type: ignore
    KGS_KGRelation = _Placeholder_KGS_KGRelation  # type: ignore
    ExplicitMemory = _Placeholder_ExplicitMemory  # type: ignore
    KGEntityRecord = _Placeholder_KGEntityRecord  # type: ignore
    KGRelationRecord = _Placeholder_KGRelationRecord  # type: ignore
    BaseMemory = _Placeholder_BaseMemory  # type: ignore
    MemorySourceType = _Placeholder_MemorySourceType  # type: ignore
    MemorySalience = _Placeholder_MemorySalience  # type: ignore
    KGEntityType = _Placeholder_KGEntityType  # type: ignore
    update_dynamic_salience = _placeholder_update_dynamic_salience  # type: ignore
    logger.error(f"KGS Tools: Failed to load necessary types or MBS service: {e_core}. "
                 "Tools may be non-functional.")
    logger.error(f"KGS Tools: CRITICAL IMPORT FAILURE (e_core): {e_core}", exc_info=True)  # Log with traceback


# --- Improved Helper to get MBS from context ---
# In ceaf_core/tools/kgs_tools.py

def _get_mbs_from_context(tool_context: ToolContext) -> Optional[MBSMemoryService]:
    """
    More robust memory service retrieval from tool context.
    """
    logger.info(f"KGS Tools: _get_mbs_from_context called. tool_context type: {type(tool_context)}")
    if tool_context is None:
        logger.error("KGS Tools: _get_mbs_from_context received tool_context as None!")
        return None
    logger.info(f"KGS Tools: dir(tool_context): {dir(tool_context)}")

    memory_service_candidate: Any = None
    ic = None

    if hasattr(tool_context, 'invocation_context'):
        ic = tool_context.invocation_context
    else:
        logger.warning("KGS Tools: ToolContext has no 'invocation_context'. Cannot retrieve MBS.")
        return None

    # Attempt 1: Direct access from invocation_context if explicitly set (good for dummy context)
    if hasattr(ic, 'memory_service'):
        memory_service_candidate = ic.memory_service
        if memory_service_candidate:
            logger.debug("KGS Tools: Found memory_service_candidate via ic.memory_service")

    # Attempt 2: Standard ADK runner services access
    if not memory_service_candidate and hasattr(ic, 'runner') and hasattr(ic.runner, '_services'):
        memory_service_candidate = ic.runner._services.get('memory_service')
        if memory_service_candidate:
            logger.debug("KGS Tools: Found memory_service_candidate via ic.runner._services")

    # Attempt 3: Alternative runner access
    if not memory_service_candidate and hasattr(ic, 'runner') and hasattr(ic.runner, 'memory_service'):
        memory_service_candidate = ic.runner.memory_service
        if memory_service_candidate:
            logger.debug("KGS Tools: Found memory_service_candidate via ic.runner.memory_service")

    # Attempt 4: Check if services are stored differently on invocation_context
    if not memory_service_candidate and hasattr(ic, 'services') and isinstance(ic.services, dict):
        memory_service_candidate = ic.services.get('memory_service')
        if memory_service_candidate:
            logger.debug("KGS Tools: Found memory_service_candidate via ic.services")

    if not memory_service_candidate:
        logger.warning("KGS Tools: Could not find memory_service_candidate through common context paths.")
        # Try global adk_components as a last resort for startup tasks
        try:
            from ceaf_project.main import adk_components as main_adk_components_module_level
            memory_service_candidate = main_adk_components_module_level.get('memory_service')
            if memory_service_candidate:
                logger.info("KGS Tools: Retrieved memory_service_candidate from main.adk_components (startup fallback)")
        except ImportError:
            logger.debug("KGS Tools: main.adk_components not found or importable for fallback.")
            pass  # Fall through

    if not memory_service_candidate:
        logger.error("KGS Tools: MBSMemoryService instance completely not found in context.")
        return None


    logger.debug(f"KGS Tools: Candidate type: {type(memory_service_candidate)}, "
                 f"MBS_AND_TYPES_LOADED here: {MBS_AND_TYPES_LOADED}, "
                 f"_RealMBSClassForCheck here: {_RealMBSClassForCheck}, "
                 f"Local MBSMemoryService here: {MBSMemoryService}")

    if MBS_AND_TYPES_LOADED and _RealMBSClassForCheck is not _Placeholder_MBSMemoryService:
        if isinstance(memory_service_candidate, _RealMBSClassForCheck):
            logger.info("KGS Tools: Validated memory service against _RealMBSClassForCheck.")
            return cast(MBSMemoryService, memory_service_candidate)
        else:
            logger.warning(f"KGS Tools: Type mismatch. Candidate type: {type(memory_service_candidate)}, "
                           f"expected real type from this module: {_RealMBSClassForCheck}.")

    logger.info(f"KGS Tools: memory_service_candidate type: {type(memory_service_candidate)}")
    logger.info(f"KGS Tools: _RealMBSClassForCheck type: {type(_RealMBSClassForCheck)}")
    logger.info(
        f"KGS Tools: Is candidate instance of _RealMBSClassForCheck? {isinstance(memory_service_candidate, _RealMBSClassForCheck)}")

    if (hasattr(memory_service_candidate, 'search_raw_memories') and
            hasattr(memory_service_candidate, 'add_specific_memory') and
            hasattr(memory_service_candidate, 'get_memory_by_id')):
        logger.warning(
            "KGS Tools: Using DUCK-TYPING for MBSMemoryService as strict isinstance checks failed or imports were incomplete.")
        # Even if MBSMemoryService is a placeholder here, the cast is for type hinting.
        # The actual object `memory_service_candidate` is what's returned.
        return cast(MBSMemoryService, memory_service_candidate)
    else:
        logger.error(f"KGS Tools: Found memory_service_candidate (type: {type(memory_service_candidate)}) "
                     "but it doesn't match expected MBSMemoryService interface (duck-typing failed).")
        return None


# --- Tool Functions with Enhanced Documentation ---

def get_explicit_memories_for_kg_synthesis(
        tool_context: ToolContext,
        max_memories_to_fetch: int = 20,
        min_salience_score: Optional[float] = 0.3,
        only_with_structured_data: bool = False

) -> Dict[str, Any]:
    """
    Retrieves a batch of explicit memories from CEAF's memory store for knowledge graph synthesis.

    This tool is designed to provide the KGS Agent with raw material that can be processed
    to extract entities and relations. It prioritizes recent memories with sufficient salience
    and optionally filters for memories that contain structured data.

    Use this tool when:
    - The KGS Agent needs fresh memory content to process
    - You want to gather memories suitable for knowledge extraction
    - Building knowledge graphs from recent user interactions or system observations

    Args:
        max_memories_to_fetch: Maximum number of memories to return (default: 20)
        min_salience_score: Minimum salience score threshold for filtering (default: 0.3)
        only_with_structured_data: If True, only return memories with structured_data field
        tool_context: ADK tool context (automatically injected by ADK)

    Returns:
        Dictionary with structure:
        - On success: {'status': 'success', 'explicit_memories_batch': [memory_dicts], 'message': str}
        - On error: {'status': 'error', 'error_message': str, 'details': Any, 'error_code': str}
    """
    logger.info(f"KGS Tool (get_explicit_memories_for_kg_synthesis): Fetching up to {max_memories_to_fetch} memories.")

    if not MBS_AND_TYPES_LOADED:
        return create_error_tool_response(
            "MBS or required types not loaded.",
            details="System configuration error - core dependencies unavailable.",
            error_code="MBS_UNAVAILABLE"
        )

    mbs: Optional[MBSMemoryService] = _get_mbs_from_context(tool_context)
    if not mbs or not hasattr(mbs, '_in_memory_explicit_cache'):
        return create_error_tool_response(
            "MBS not available or not properly initialized.",
            error_code="MBS_UNAVAILABLE"
        )

    try:
        candidate_memories = list(mbs._in_memory_explicit_cache)  # type: ignore

        # Apply filters
        if min_salience_score is not None:
            candidate_memories = [
                mem for mem in candidate_memories
                if getattr(mem, 'dynamic_salience_score', 0.0) >= min_salience_score
            ]

        if only_with_structured_data:
            candidate_memories = [
                mem for mem in candidate_memories
                if (hasattr(mem, 'content') and
                    getattr(getattr(mem, 'content', None), 'structured_data', None))
            ]

        # Sort by timestamp (most recent first)
        candidate_memories.sort(key=lambda m: getattr(m, 'timestamp', 0), reverse=True)

        # Extract memory data using model_dump
        selected_memories_data = []
        for mem in candidate_memories[:max_memories_to_fetch]:
            if hasattr(mem, 'model_dump'):
                selected_memories_data.append(mem.model_dump(exclude_none=True))
            else:
                logger.warning(f"KGS Tool: Memory object {getattr(mem, 'memory_id', 'Unknown ID')} "
                               "lacks model_dump method.")

        logger.info(f"KGS Tool (get_explicit_memories_for_kg_synthesis): "
                    f"Retrieved {len(selected_memories_data)} memories for KGS.")

        return create_successful_tool_response(
            data={"explicit_memories_batch": selected_memories_data},
            message=f"Retrieved {len(selected_memories_data)} explicit memories."
        )

    except Exception as e:
        logger.error(f"KGS Tool: Error retrieving memories: {e}", exc_info=True)
        return create_error_tool_response(
            f"Failed to retrieve memories: {str(e)}",
            error_code="MEMORY_RETRIEVAL_ERROR"
        )


async def commit_knowledge_graph_elements(
        tool_context: ToolContext,
        extracted_entities: List[Dict[str, Any]],
        extracted_relations: List[Dict[str, Any]],
        source_memory_ids: Optional[List[str]] = None

) -> Dict[str, Any]:
    """
    Commits extracted KG entities and relations to CEAF's MemoryBlossomService.

    This tool is typically called by the KGS Agent after it has processed text content
    and generated structured knowledge graph elements. It converts the extracted data
    into proper KGEntityRecord and KGRelationRecord objects and stores them in memory.

    Use this tool when:
    - The KGS Agent has extracted entities and relations from text
    - You need to persist knowledge graph elements in the memory system
    - Building up the system's knowledge base from processed content

    Args:
        extracted_entities: List of entity dictionaries with keys like 'id_str', 'label', 'type', etc.
        extracted_relations: List of relation dictionaries with keys like 'source_id_str', 'target_id_str', etc.
        source_memory_ids: Optional list of memory IDs that were the source of these extractions
        tool_context: ADK tool context (automatically injected by ADK)

    Returns:
        Dictionary with structure:
        - On success: {'status': 'success', 'committed_entity_ids': [str], 'committed_relation_ids': [str]}
        - On error: {'status': 'error', 'error_message': str, 'details': dict, 'errors': [str]}
    """
    logger.info(f"KGS Tool (commit_knowledge_graph_elements): Committing {len(extracted_entities)} entities "
                f"and {len(extracted_relations)} relations.")

    if not MBS_AND_TYPES_LOADED:
        return create_error_tool_response(
            "MBS or required KG types not loaded.",
            details="System configuration error - core dependencies unavailable.",
            error_code="MBS_UNAVAILABLE"
        )

    mbs: Optional[MBSMemoryService] = _get_mbs_from_context(tool_context)
    if not mbs or not hasattr(mbs, 'add_specific_memory'):
        return create_error_tool_response(
            "MBS not available or not properly initialized.",
            error_code="MBS_UNAVAILABLE"
        )

    committed_entity_ids = []
    committed_relation_ids = []
    errors = []

    # Process entities
    for entity_data in extracted_entities:
        try:
            # Use KGS_KGEntity for parsing if available, otherwise use dict directly
            if KGS_KGEntity != _Placeholder_KGS_KGEntity:
                parsed_entity_data = KGS_KGEntity(**entity_data)  # type: ignore
            else:
                parsed_entity_data = entity_data  # type: ignore

            # Resolve entity type enum
            entity_type_enum_val = KGEntityType.OTHER  # type: ignore
            kgs_entity_type_str = getattr(parsed_entity_data, 'type', 'OTHER')

            if MBS_AND_TYPES_LOADED and hasattr(KGEntityType, '__members__'):
                try:
                    entity_type_enum_val = KGEntityType[kgs_entity_type_str.upper()]  # type: ignore
                except KeyError:
                    logger.warning(f"KGS Tool: Unknown entity type '{kgs_entity_type_str}' "
                                   f"for entity '{getattr(parsed_entity_data, 'label', 'N/A')}'. "
                                   "Defaulting to OTHER.")
            elif not MBS_AND_TYPES_LOADED:
                entity_type_enum_val = kgs_entity_type_str  # type: ignore

            # Create entity record
            entity_record_data = {
                "entity_id_str": getattr(parsed_entity_data, 'id_str', None),
                "label": getattr(parsed_entity_data, 'label', None),
                "entity_type": entity_type_enum_val,
                "description": getattr(parsed_entity_data, 'description', None),
                "attributes": getattr(parsed_entity_data, 'attributes', {}),
                "aliases": getattr(parsed_entity_data, 'aliases', []),
                "source_type": MemorySourceType.INTERNAL_REFLECTION,  # type: ignore
                "source_agent_name": "KGS_Agent_CommitTool",
                "salience": MemorySalience.MEDIUM,  # type: ignore
                "metadata": {"source_explicit_memory_ids": source_memory_ids} if source_memory_ids else {}
            }

            entity_record = KGEntityRecord(**entity_record_data)  # type: ignore
            await mbs.add_specific_memory(entity_record)  # type: ignore
            committed_entity_ids.append(getattr(entity_record, 'entity_id_str', 'unknown_id'))

        except Exception as e:
            err_msg = f"Failed to process/commit entity data '{str(entity_data)[:100]}...': {e}"
            logger.error(f"KGS Tool Commit: {err_msg}", exc_info=True)
            errors.append(err_msg)

    # Process relations
    for relation_data in extracted_relations:
        try:
            # Use KGS_KGRelation for parsing if available, otherwise use dict directly
            if KGS_KGRelation != _Placeholder_KGS_KGRelation:
                parsed_relation_data = KGS_KGRelation(**relation_data)  # type: ignore
            else:
                parsed_relation_data = relation_data  # type: ignore

            # Create relation record
            relation_record_data = {
                "source_entity_id_str": getattr(parsed_relation_data, 'source_id_str', None),
                "target_entity_id_str": getattr(parsed_relation_data, 'target_id_str', None),
                "relation_label": getattr(parsed_relation_data, 'label', None),
                "description": getattr(parsed_relation_data, 'context', None),  # KGS_KGRelation uses 'context'
                "attributes": getattr(parsed_relation_data, 'attributes', {}),
                "source_type": MemorySourceType.INTERNAL_REFLECTION,  # type: ignore
                "source_agent_name": "KGS_Agent_CommitTool",
                "salience": MemorySalience.MEDIUM,  # type: ignore
                "metadata": {"source_explicit_memory_ids": source_memory_ids} if source_memory_ids else {}
            }

            relation_record = KGRelationRecord(**relation_record_data)  # type: ignore
            await mbs.add_specific_memory(relation_record)  # type: ignore
            committed_relation_ids.append(getattr(relation_record, 'relation_id_str', 'unknown_id'))

        except Exception as e:
            err_msg = f"Failed to process/commit relation data '{str(relation_data)[:100]}...': {e}"
            logger.error(f"KGS Tool Commit: {err_msg}", exc_info=True)
            errors.append(err_msg)

    # Return results
    msg = f"Committed {len(committed_entity_ids)} entities and {len(committed_relation_ids)} relations."
    if errors:
        msg += f" Encountered {len(errors)} errors during commit."
        return create_error_tool_response(
            error_message=msg,
            details={
                "committed_entities": committed_entity_ids,
                "committed_relations": committed_relation_ids,
                "errors": errors
            }
        )
    else:
        return create_successful_tool_response(
            data={
                "committed_entity_ids": committed_entity_ids,
                "committed_relation_ids": committed_relation_ids
            },
            message=msg
        )


def query_knowledge_graph(
        tool_context: ToolContext,
        entity_label: Optional[str] = None,
        relation_label: Optional[str] = None,
        target_entity_label: Optional[str] = None,
        limit: int = 5

) -> Dict[str, Any]:
    """
    Performs a basic query against CEAF's knowledge graph.

    This tool enables searching for entities by label, or relations connected to entities.
    It supports partial string matching and returns rich context including related entities.
    Dynamic salience scores are updated for accessed memory objects.

    Use this tool when:
    - You need to find specific entities or relations in the knowledge graph
    - Exploring connections between knowledge graph elements
    - Retrieving context about entities for reasoning or response generation

    Args:
        entity_label: Partial or full label of entities to search for
        relation_label: Label of relations to search for (used with or without entity_label)
        target_entity_label: Currently not used for filtering (reserved for future enhancement)
        limit: Maximum number of results to return (default: 5)
        tool_context: ADK tool context (automatically injected by ADK)

    Returns:
        Dictionary with structure:
        - On success: {'status': 'success', 'kg_query_results': [result_dicts], 'message': str}
        - On error: {'status': 'error', 'error_message': str, 'error_code': str}

    Note:
        When both entity_label and relation_label are provided, the tool searches for
        relations connected to the matching entities. When only relation_label is provided,
        it searches all relations matching that label.
    """
    logger.info(f"KGS Tool (query_knowledge_graph): Query for entity '{entity_label}', "
                f"relation '{relation_label}', limit {limit}")

    if not MBS_AND_TYPES_LOADED:
        return create_error_tool_response(
            "KG types not loaded.",
            details="System configuration error - core dependencies unavailable.",
            error_code="MBS_UNAVAILABLE"
        )

    mbs: Optional[MBSMemoryService] = _get_mbs_from_context(tool_context)
    if not mbs or not hasattr(mbs, '_in_memory_kg_entities_cache') or not hasattr(mbs, '_in_memory_kg_relations_cache'):
        return create_error_tool_response(
            "MBS not available or KG caches not initialized.",
            error_code="MBS_UNAVAILABLE"
        )

    try:
        results = []
        found_objects_for_salience_update: List[BaseMemory] = []  # type: ignore

        # Search for entities if entity_label is provided
        if entity_label:
            for ent_rec in mbs._in_memory_kg_entities_cache:  # type: ignore
                if (hasattr(ent_rec, 'label') and isinstance(ent_rec.label, str) and
                        entity_label.lower() in ent_rec.label.lower()):  # type: ignore

                    if LIFECYCLE_FUNCTIONS_AVAILABLE and isinstance(ent_rec, BaseMemory):  # type: ignore
                        found_objects_for_salience_update.append(cast(BaseMemory, ent_rec))  # type: ignore

                    entity_data = ent_rec.model_dump(exclude_none=True) if hasattr(ent_rec, 'model_dump') else vars(
                        ent_rec)
                    results.append(entity_data)

                    if len(results) >= limit:
                        break

        # Handle relation queries
        if relation_label:
            relations_found_this_pass = []

            # If entities were found, search relations connected to them
            if results and entity_label:
                entities_to_check_relations_for = results[:limit]
                for entity_dict_found in entities_to_check_relations_for:
                    source_ent_id_str = entity_dict_found.get('entity_id_str')
                    if not source_ent_id_str:
                        continue

                    for rel_rec in mbs._in_memory_kg_relations_cache:  # type: ignore
                        if (hasattr(rel_rec, 'source_entity_id_str') and
                                rel_rec.source_entity_id_str == source_ent_id_str and  # type: ignore
                                hasattr(rel_rec, 'relation_label') and
                                isinstance(rel_rec.relation_label, str) and  # type: ignore
                                relation_label.lower() in rel_rec.relation_label.lower()):  # type: ignore

                            # Find target entity information
                            target_entity_info = "Unknown Target Entity"
                            if hasattr(rel_rec, 'target_entity_id_str'):  # type: ignore
                                for ent_rec_tgt in mbs._in_memory_kg_entities_cache:  # type: ignore
                                    if (hasattr(ent_rec_tgt, 'entity_id_str') and
                                            ent_rec_tgt.entity_id_str == rel_rec.target_entity_id_str):  # type: ignore
                                        target_entity_info = (f"{getattr(ent_rec_tgt, 'label', 'N/A')} "
                                                              f"(ID: {ent_rec_tgt.entity_id_str})")  # type: ignore
                                        if LIFECYCLE_FUNCTIONS_AVAILABLE and isinstance(ent_rec_tgt,
                                                                                        BaseMemory):  # type: ignore
                                            found_objects_for_salience_update.append(
                                                cast(BaseMemory, ent_rec_tgt))  # type: ignore
                                        break

                            relation_dump = (rel_rec.model_dump(exclude_none=True)
                                             if hasattr(rel_rec, 'model_dump') else vars(rel_rec))

                            relations_found_this_pass.append({
                                "relation": relation_dump,
                                "source_entity_summary": entity_dict_found.get('label', source_ent_id_str),
                                "target_entity_summary": target_entity_info
                            })

                            if LIFECYCLE_FUNCTIONS_AVAILABLE and isinstance(rel_rec, BaseMemory):  # type: ignore
                                found_objects_for_salience_update.append(cast(BaseMemory, rel_rec))  # type: ignore

                            if len(relations_found_this_pass) >= limit:
                                break

                    if len(relations_found_this_pass) >= limit:
                        break

            # If no specific entity_label or no entities found, search all relations by relation_label
            elif not entity_label:
                for rel_rec in mbs._in_memory_kg_relations_cache:  # type: ignore
                    if (hasattr(rel_rec, 'relation_label') and
                            isinstance(rel_rec.relation_label, str) and  # type: ignore
                            relation_label.lower() in rel_rec.relation_label.lower()):  # type: ignore

                        # Find source entity information
                        source_info = "Unknown Source"
                        if hasattr(rel_rec, 'source_entity_id_str'):  # type: ignore
                            for src_ent in mbs._in_memory_kg_entities_cache:  # type: ignore
                                if (hasattr(src_ent, 'entity_id_str') and
                                        src_ent.entity_id_str == rel_rec.source_entity_id_str):  # type: ignore
                                    source_info = f"{getattr(src_ent, 'label', 'N/A')} (ID: {src_ent.entity_id_str})"  # type: ignore
                                    if LIFECYCLE_FUNCTIONS_AVAILABLE and isinstance(src_ent,
                                                                                    BaseMemory):  # type: ignore
                                        found_objects_for_salience_update.append(
                                            cast(BaseMemory, src_ent))  # type: ignore
                                    break

                        # Find target entity information
                        target_info = "Unknown Target"
                        if hasattr(rel_rec, 'target_entity_id_str'):  # type: ignore
                            for tgt_ent in mbs._in_memory_kg_entities_cache:  # type: ignore
                                if (hasattr(tgt_ent, 'entity_id_str') and
                                        tgt_ent.entity_id_str == rel_rec.target_entity_id_str):  # type: ignore
                                    target_info = f"{getattr(tgt_ent, 'label', 'N/A')} (ID: {tgt_ent.entity_id_str})"  # type: ignore
                                    if LIFECYCLE_FUNCTIONS_AVAILABLE and isinstance(tgt_ent,
                                                                                    BaseMemory):  # type: ignore
                                        found_objects_for_salience_update.append(
                                            cast(BaseMemory, tgt_ent))  # type: ignore
                                    break

                        relation_dump = (rel_rec.model_dump(exclude_none=True)
                                         if hasattr(rel_rec, 'model_dump') else vars(rel_rec))

                        relations_found_this_pass.append({
                            "relation": relation_dump,
                            "source_entity_summary": source_info,
                            "target_entity_summary": target_info
                        })

                        if LIFECYCLE_FUNCTIONS_AVAILABLE and isinstance(rel_rec, BaseMemory):  # type: ignore
                            found_objects_for_salience_update.append(cast(BaseMemory, rel_rec))  # type: ignore

                        if len(relations_found_this_pass) >= limit:
                            break

            # If relations were specifically searched and found, replace results with relations
            if relations_found_this_pass:
                results = relations_found_this_pass[:limit]

        # Update dynamic salience scores for accessed objects
        if LIFECYCLE_FUNCTIONS_AVAILABLE:
            for mem_obj in found_objects_for_salience_update:
                update_dynamic_salience(mem_obj, access_type="retrieval")  # type: ignore

        # Return results
        if not results:
            return create_successful_tool_response(
                message=f"No KG elements found matching criteria: entity '{entity_label}', "
                        f"relation '{relation_label}'."
            )

        return create_successful_tool_response(
            data={"kg_query_results": results},
            message=f"Found {len(results)} KG query results."
        )

    except Exception as e:
        logger.error(f"KGS Tool: Error querying knowledge graph: {e}", exc_info=True)
        return create_error_tool_response(
            f"Failed to query knowledge graph: {str(e)}",
            error_code="KG_QUERY_ERROR"
        )


# --- Create ADK Function Tools ---

get_memories_for_kg_synthesis_tool = FunctionTool(
    func=get_explicit_memories_for_kg_synthesis
)

commit_kg_elements_tool = FunctionTool(
    func=commit_knowledge_graph_elements
)

query_knowledge_graph_tool = FunctionTool(
    func=query_knowledge_graph
)

# --- Export tools for use by other modules ---
ceaf_kgs_tools = [
    get_memories_for_kg_synthesis_tool,
    commit_kg_elements_tool,
    query_knowledge_graph_tool
]