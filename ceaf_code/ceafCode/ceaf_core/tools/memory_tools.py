# Memory Tools
# ceaf_project/ceaf_core/tools/memory_tools.py

import logging
import json
import time
import asyncio # ADDED for async tool function
from typing import Dict, Any, Optional, List, Tuple, cast

from google.adk.tools import FunctionTool, ToolContext

# --- Type Imports and Placeholders ---
# Assuming memory_types.py and mbs_memory_service.py exist and are importable
# For brevity, the placeholder logic for MEMORY_TYPES_LOADED is kept but ideally it's always true.
try:
    from ..modules.memory_blossom.memory_types import (
        AnyMemoryType, # Ensure this Union includes all relevant memory types
        ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience, BaseMemory
    )
    from ..services.mbs_memory_service import MBSMemoryService # ADDED
    MEMORY_TYPES_LOADED = True
    MBS_SERVICE_LOADED = True
except ImportError as e:
    logging.warning(
        f"Memory Tools: Could not import full CEAF memory_types or MBSMemoryService: {e}. "
        "LTM tools might be limited or non-functional."
    )
    MEMORY_TYPES_LOADED = False
    MBS_SERVICE_LOADED = False
    class BaseMemoryPydantic: # Basic placeholder
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
        def model_dump(self, **kwargs): return self.__dict__
    class AnyMemoryType(BaseMemoryPydantic): pass # type: ignore
    class ExplicitMemory(BaseMemoryPydantic): pass
    class ExplicitMemoryContent(BaseMemoryPydantic): pass
    class MemorySourceType: USER_INTERACTION = "user"; ORA_RESPONSE = "agent"; INTERNAL_REFLECTION = "system"
    class MemorySalience: MEDIUM = "medium"; LOW = "low"; HIGH = "high"; CRITICAL = "critical"
    class MBSMemoryService: # Dummy service
        async def search_raw_memories(self, query: str, top_k: int) -> List[Tuple[AnyMemoryType, float]]: # type: ignore
            logger.warning("Using dummy MBSMemoryService.search_raw_memories")
            return []


from .common_utils import (
    create_successful_tool_response,
    create_error_tool_response,
    sanitize_text_for_logging
)

logger = logging.getLogger(__name__)

CONTEXT_QUERY_DELIMITER = " <CEAF_CONTEXT_SEPARATOR> "

# --- Helper to get MBS from context (can be shared or in common_utils) ---
def _get_mbs_from_context(tool_context: ToolContext) -> Optional[MBSMemoryService]:
    """
    More robust memory service retrieval from tool context.
    """
    logger.debug(f"Memory Tools: _get_mbs_from_context called. tool_context type: {type(tool_context)}")
    if tool_context is None:
        logger.error("Memory Tools: _get_mbs_from_context received tool_context as None!")
        return None

    memory_service_candidate: Any = None
    ic = None

    if hasattr(tool_context, 'invocation_context') and tool_context.invocation_context is not None:
        ic = tool_context.invocation_context
        logger.debug(f"Memory Tools: Found tool_context.invocation_context: {type(ic)}")
    else:
        logger.warning("Memory Tools: ToolContext has no 'invocation_context' or it is None.")
        # Fallback directly if invocation_context is missing
        try:
            from ceaf_project.main import adk_components as main_adk_components_module_level # type: ignore
            memory_service_candidate = main_adk_components_module_level.get('memory_service')
            if memory_service_candidate:
                logger.info("Memory Tools: Retrieved memory_service_candidate from main.adk_components (early fallback)")
                if isinstance(memory_service_candidate, MBSMemoryService):
                    return cast(MBSMemoryService, memory_service_candidate)
                else:
                    logger.warning(f"Memory Tools: main.adk_components['memory_service'] is not MBSMemoryService type: {type(memory_service_candidate)}")
                    return None # Or raise, or try duck typing
        except ImportError:
            logger.error("Memory Tools: main.adk_components not found or importable for early fallback, and no invocation_context.")
            return None
        return None # If fallback didn't work or wasn't MBS type

    # Proceed with ic if it was found
    if hasattr(ic, 'memory_service'): # Check if memory_service is directly on invocation_context
        memory_service_candidate = ic.memory_service
        if memory_service_candidate:
            logger.debug("Memory Tools: Found memory_service_candidate via ic.memory_service (direct)")

    if not memory_service_candidate and hasattr(ic, 'runner'):
        if hasattr(ic.runner, '_services'):
            memory_service_candidate = ic.runner._services.get('memory_service')
            if memory_service_candidate:
                logger.debug("Memory Tools: Found memory_service_candidate via ic.runner._services")
        if not memory_service_candidate and hasattr(ic.runner, 'memory_service'):
            memory_service_candidate = ic.runner.memory_service
            if memory_service_candidate:
                logger.debug("Memory Tools: Found memory_service_candidate via ic.runner.memory_service (direct on runner)")

    if not memory_service_candidate and hasattr(ic, 'services') and isinstance(ic.services, dict):
        memory_service_candidate = ic.services.get('memory_service')
        if memory_service_candidate:
            logger.debug("Memory Tools: Found memory_service_candidate via ic.services (dict)")

    if not memory_service_candidate:
        logger.warning("Memory Tools: Could not find memory_service_candidate through common context paths. Trying global fallback.")
        try:
            from ceaf_project.main import adk_components as main_adk_components_module_level # type: ignore
            memory_service_candidate = main_adk_components_module_level.get('memory_service')
            if memory_service_candidate:
                logger.info("Memory Tools: Retrieved memory_service_candidate from main.adk_components (last resort fallback)")
        except ImportError:
            logger.debug("Memory Tools: main.adk_components not found or importable for last resort fallback.")
            pass

    if not memory_service_candidate:
        logger.error("Memory Tools: MBSMemoryService instance completely not found in context or fallbacks.")
        return None

    # Type check before returning
    if MBS_SERVICE_LOADED and isinstance(memory_service_candidate, MBSMemoryService):
        logger.info("Memory Tools: Validated memory service instance successfully.")
        return cast(MBSMemoryService, memory_service_candidate)
    elif (hasattr(memory_service_candidate, 'search_raw_memories') and
            hasattr(memory_service_candidate, 'add_specific_memory')): # Duck typing as a last resort
        logger.warning(
            "Memory Tools: Using DUCK-TYPING for MBSMemoryService as strict isinstance check failed or MBS_SERVICE_LOADED is False.")
        return cast(MBSMemoryService, memory_service_candidate)
    else:
        logger.error(f"Memory Tools: Found memory_service_candidate (type: {type(memory_service_candidate)}) "
                     "but it doesn't match expected MBSMemoryService interface (duck-typing failed).")
        return None


# --- Short-Term Session Memory Tools (using ADK session state) ---
# These remain synchronous as they operate on tool_context.state directly.

def store_short_term_memory(
        memory_key: str,
        memory_value: str,
        tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Saves a key-value pair to the current session's short-term (volatile) memory.
    Useful for remembering details within the current conversational context.

    Args:
        memory_key (str): The key under which to store the value.
        memory_value (str): The string value to store.
        tool_context (ToolContext): The ADK tool context.

    Returns:
        Dict[str, Any]: A dictionary indicating success or failure.
    Example:
        store_short_term_memory(memory_key='user_preference_color', memory_value='blue')
    """
    # ... (implementation remains unchanged) ...
    agent_name_log = getattr(tool_context, 'agent_name', "UnknownAgent")
    if not memory_key or not isinstance(memory_key, str):
        return create_error_tool_response("Invalid memory_key provided. Must be a non-empty string.")
    if not isinstance(memory_value, str): # For V1, keeping it simple as string
        return create_error_tool_response("Invalid memory_value provided. Must be a string for V1 short-term memory.")

    state_key = f"stm:{memory_key}"
    try:
        tool_context.state[state_key] = memory_value
        logger.info(
            f"Memory Tool (STM): Stored '{sanitize_text_for_logging(memory_value)}' under state key '{state_key}' by agent '{agent_name_log}'.")
        return create_successful_tool_response(message=f"Information for '{memory_key}' stored in short-term memory.")
    except Exception as e:
        logger.error(f"Memory Tool (STM): Error storing to state for key '{state_key}': {e}", exc_info=True)
        return create_error_tool_response(f"Failed to store short-term memory for '{memory_key}'.")


store_stm_tool = FunctionTool(
    func=store_short_term_memory
)

def retrieve_short_term_memory(
        memory_key: str,
        tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Fetches a value from the current session's short-term (volatile) memory based on a key.

    Args:
        memory_key (str): The key of the value to retrieve.
        tool_context (ToolContext): The ADK tool context.

    Returns:
        Dict[str, Any]: A dictionary containing the retrieved value or an error.
    Example:
        retrieve_short_term_memory(memory_key='user_preference_color')
    """
    # ... (implementation remains unchanged) ...
    agent_name_log = getattr(tool_context, 'agent_name', "UnknownAgent")
    if not memory_key or not isinstance(memory_key, str):
        return create_error_tool_response("Invalid memory_key provided. Must be a non-empty string.")

    state_key = f"stm:{memory_key}"
    retrieved_value = tool_context.state.get(state_key)

    if retrieved_value is not None:
        logger.info(
            f"Memory Tool (STM): Retrieved '{sanitize_text_for_logging(str(retrieved_value))}' for state key '{state_key}' by agent '{agent_name_log}'.")
        return create_successful_tool_response(
            data={"memory_key": memory_key, "value": retrieved_value},
            message="Information retrieved from short-term memory."
        )
    else:
        logger.info(
            f"Memory Tool (STM): No value found in state for key '{state_key}' by agent '{agent_name_log}'.")
        return create_error_tool_response(
            f"No information found in short-term memory for key '{memory_key}'.",
            error_code="STM_KEY_NOT_FOUND"
        )


retrieve_stm_tool = FunctionTool(
    func=retrieve_short_term_memory
)

# --- Long-Term Memory Interaction Tools ---

async def query_long_term_memory_store( # CHANGED to async
        search_query: str,
        tool_context: ToolContext,
        top_k: Optional[int] = None,
        augmented_query_context: Optional[Dict[str, Any]] = None,
        # NEW parameter to control output type for internal vs. LLM consumption
        return_raw_memory_objects: bool = False
) -> Dict[str, Any]:
    """
    Searches CEAF's persistent long-term memory for information.
    Can return textual snippets for LLMs or raw memory objects for internal CEAF processes.

    Args:
        search_query (str): The primary text to search for.
        tool_context (ToolContext): The ADK tool context.
        top_k (Optional[int]): The maximum number of results to return (default 3, max 10).
        augmented_query_context (Optional[Dict[str, Any]]): Additional context to guide the search.
        return_raw_memory_objects (bool): If True, returns serialized raw memory objects.
                                         Otherwise, returns formatted text snippets. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary with search results or an error.
    Example:
        query_long_term_memory_store(search_query='Project Alpha failures', return_raw_memory_objects=True)
    """
    agent_name_log = getattr(tool_context, 'agent_name', "UnknownAgent")
    if not search_query or not isinstance(search_query, str):
        return create_error_tool_response("Invalid search_query provided. Must be a non-empty string.")

    actual_top_k = top_k if (top_k is not None and 0 < top_k <= 10) else 3
    if top_k is not None and not (0 < top_k <= 10):
        logger.warning(f"Memory Tool (LTM Query): Invalid top_k value '{top_k}', defaulting to {actual_top_k}.")

    # Prepare the full query string including augmented context if provided
    final_query_for_mbs = search_query
    if augmented_query_context and isinstance(augmented_query_context, dict):
        try:
            context_json_str = json.dumps(augmented_query_context)
            final_query_for_mbs = f"{search_query}{CONTEXT_QUERY_DELIMITER}{context_json_str}"
            logger.info(
                f"Memory Tool (LTM Query): Augmented query with context: {sanitize_text_for_logging(final_query_for_mbs, 200)}")
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Memory Tool (LTM Query): Could not serialize augmented_query_context to JSON: {e}. Proceeding with original query.")

    # Get MBS instance
    if not MBS_SERVICE_LOADED:
        return create_error_tool_response("MBSMemoryService not loaded.", error_code="MBS_SERVICE_UNAVAILABLE")

    mbs = _get_mbs_from_context(tool_context)
    if not mbs or not hasattr(mbs, 'search_raw_memories'): # Check for the new method
        logger.error("Memory Tool (LTM Query): MBSMemoryService not available or missing 'search_raw_memories' method.")
        return create_error_tool_response("LTM search capability (raw) not configured.", error_code="LTM_SERVICE_UNAVAILABLE")

    try:
        logger.info(
            f"Memory Tool (LTM Query): Searching LTM (raw) with final query: '{sanitize_text_for_logging(final_query_for_mbs, 150)}' (top_k={actual_top_k}) by agent '{agent_name_log}'.")

        # Call the new method on MBSMemoryService
        # This method is expected to be async and return List[Tuple[AnyMemoryType, float]]
        raw_memory_results_with_scores: List[Tuple[AnyMemoryType, float]] = await mbs.search_raw_memories(
            query=final_query_for_mbs, # The full query string
            top_k=actual_top_k
        )

        if not raw_memory_results_with_scores:
            logger.info(f"Memory Tool (LTM Query): No raw memories found for query '{search_query}'.")
            return create_successful_tool_response(
                data={"query": search_query, "augmented_context_used": bool(augmented_query_context),
                      "found_memories_count": 0,
                      "retrieved_memories": [],
                      "retrieved_memory_objects": []}, # Ensure consistent return structure
                message="No relevant long-term memories found."
            )

        if return_raw_memory_objects:
            # Serialize raw memory objects for internal use (e.g., by NCF tool)
            serialized_memory_objects = []
            for mem_obj, score_val in raw_memory_results_with_scores:
                try:
                    # Ensure memory_type is included in the dump
                    dumped_mem = mem_obj.model_dump(exclude_none=True) # type: ignore
                    dumped_mem['retrieval_score'] = score_val # Add score
                    if 'memory_type' not in dumped_mem and hasattr(mem_obj, 'memory_type'):
                         dumped_mem['memory_type'] = getattr(mem_obj, 'memory_type')
                    serialized_memory_objects.append(dumped_mem)
                except Exception as e_dump:
                    logger.error(f"Memory Tool (LTM Query): Failed to dump memory object {getattr(mem_obj, 'memory_id', 'UNKNOWN_ID')}: {e_dump}")
                    serialized_memory_objects.append({"error": "serialization_failed", "memory_id": getattr(mem_obj, 'memory_id', 'UNKNOWN_ID'), "score": score_val})

            logger.info(
                f"Memory Tool (LTM Query): Returning {len(serialized_memory_objects)} serialized raw memory objects for query '{search_query}'.")
            return create_successful_tool_response(
                data={"query": search_query, "augmented_context_used": bool(augmented_query_context),
                      "found_memories_count": len(serialized_memory_objects),
                      "retrieved_memory_objects": serialized_memory_objects},
                message=f"Retrieved {len(serialized_memory_objects)} raw memory objects."
            )
        else:
            # Format as text snippets for LLM consumption (similar to old logic but from raw objects)
            formatted_memories_for_llm = []
            for mem_obj, score_val in raw_memory_results_with_scores:
                # Create snippet (this part needs a robust way to get text from AnyMemoryType)
                # For now, a simplified approach:
                text_snippet = f"({getattr(mem_obj, 'memory_type', 'Memory')}, Score: {score_val:.2f}): "
                if hasattr(mem_obj, 'content') and hasattr(mem_obj.content, 'text_content') and mem_obj.content.text_content:
                    text_snippet += sanitize_text_for_logging(mem_obj.content.text_content, 150)
                elif hasattr(mem_obj, 'goal_description') and mem_obj.goal_description:
                    text_snippet += sanitize_text_for_logging(mem_obj.goal_description, 150)
                elif hasattr(mem_obj, 'label') and mem_obj.label: # For KGEntity
                    text_snippet += sanitize_text_for_logging(f"{mem_obj.label} - {getattr(mem_obj, 'description', '')}", 150)
                else:
                    text_snippet += f"ID {getattr(mem_obj, 'memory_id', 'Unknown')}"

                formatted_memories_for_llm.append({
                    "retrieval_score": score_val,
                    "content_snippets": [text_snippet], # Keep structure similar
                    "retrieved_memory_type": getattr(mem_obj, 'memory_type', 'unknown')
                })

            logger.info(
                f"Memory Tool (LTM Query): Returning {len(formatted_memories_for_llm)} formatted memory snippets for query '{search_query}'.")
            return create_successful_tool_response(
                data={"query": search_query, "augmented_context_used": bool(augmented_query_context),
                      "found_memories_count": len(formatted_memories_for_llm),
                      "retrieved_memories": formatted_memories_for_llm}, # For LLM
                message=f"Found {len(formatted_memories_for_llm)} potentially relevant long-term memories."
            )

    except Exception as e:
        logger.error(f"Memory Tool (LTM Query): Error searching LTM for '{search_query}': {e}", exc_info=True)
        return create_error_tool_response(f"Failed to search long-term memory: {str(e)}")


query_ltm_tool = FunctionTool(
    func=query_long_term_memory_store
)

def commit_explicit_fact_to_long_term_memory(
        fact_text_content: str,
        tool_context: ToolContext,
        keywords: Optional[List[str]] = None,
        salience_level: str = "medium",
        source_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identifies a specific, important piece of textual information as an 'explicit fact' and queues it
    for storage in CEAF's persistent long-term memory. Use this when a key insight, definition, or
    critical piece of data has been established that needs to be recallable across sessions.

    Args:
        fact_text_content (str): The textual content of the fact.
        tool_context (ToolContext): The ADK tool context.
        keywords (Optional[List[str]]): Optional keywords to associate with the fact.
        salience_level (str): Importance level ('low', 'medium', 'high', 'critical'). Default 'medium'.
        source_description (Optional[str]): Optional description of where this fact originated.

    Returns:
        Dict[str, Any]: A dictionary indicating success or failure of queuing the commit.
    Example:
        commit_explicit_fact_to_long_term_memory(
            fact_text_content='The CEAF manifesto emphasizes narrative coherence.',
            keywords=['ceaf', 'manifesto', 'coherence'],
            salience_level='high'
        )
    """
    # ... (implementation remains largely unchanged, as it queues for later processing by MBS) ...
    # This tool's job is to package the data into an ExplicitMemory-like structure and put it on the session state.
    # The actual saving is handled by MBSMemoryService.add_session_to_memory or a similar dedicated processor.
    # No direct async calls here.
    agent_name_log = getattr(tool_context, 'agent_name', "UnknownAgent")
    invocation_id_log = getattr(tool_context, 'invocation_id', "unknown_inv_id")
    session_id_log = getattr(tool_context, 'session_id', "unknown_session_id")

    logger.info(
        f"Memory Tool (LTM Commit): Request to commit explicit fact: '{sanitize_text_for_logging(fact_text_content)}' by agent '{agent_name_log}'.")

    if not fact_text_content or not isinstance(fact_text_content, str):
        return create_error_tool_response("Invalid fact_text_content provided.")

    salience_to_use: Any = MemorySalience.MEDIUM
    try:
        if MEMORY_TYPES_LOADED and hasattr(MemorySalience, salience_level.upper()):
            salience_to_use = getattr(MemorySalience, salience_level.upper())
        elif not MEMORY_TYPES_LOADED: # Placeholder logic
            valid_salience_strs = ["low", "medium", "high", "critical"]
            if salience_level.lower() in valid_salience_strs: salience_to_use = salience_level.lower()
            else:
                logger.warning(f"Memory Tool (LTM Commit): Invalid salience_level '{salience_level}', defaulting to medium (placeholder).")
                salience_to_use = "medium"
        else: # MEMORY_TYPES_LOADED is True but salience_level is not a valid enum member name
            logger.warning(f"Memory Tool (LTM Commit): Invalid salience_level '{salience_level}' for MemorySalience enum, defaulting to medium.")
            # salience_to_use remains MemorySalience.MEDIUM as initialized
    except (ValueError, AttributeError) as e_sal: # Catch broader errors during enum conversion
        logger.warning(f"Memory Tool (LTM Commit): Error processing salience_level '{salience_level}': {e_sal}. Defaulting to medium.")

    # If full memory types are not available, we queue a simpler dict.
    # Otherwise, we create a proper Pydantic model instance.
    if not MEMORY_TYPES_LOADED:
        logger.warning("Memory Tool (LTM Commit): Real CEAF Memory Types not loaded. Queuing placeholder data.")
        simplified_fact_data = {
            "memory_type": "explicit_placeholder", # Indicate it's a placeholder
            "text_content": fact_text_content,
            "keywords": keywords or [],
            "salience": salience_level, # Store as string
            "source_agent_name": agent_name_log,
            "source_turn_id": invocation_id_log,
            "source_interaction_id": session_id_log,
            "source_type": MemorySourceType.INTERNAL_REFLECTION, # String value
            "timestamp": time.time(),
            "metadata": {"commit_tool_source_description": source_description} if source_description else {}
        }
        pending_commits_key = "mbs:pending_memory_commits"
        if pending_commits_key not in tool_context.state:
            tool_context.state[pending_commits_key] = []
        tool_context.state[pending_commits_key].append(simplified_fact_data)
        # This tool should probably still return success if it successfully *queued* it,
        # even if it's placeholder data. The processing step is later.
        return create_successful_tool_response(
            message="Fact queued for commit with placeholder data due to missing MemoryType definitions.",
            data={"status": "queued_as_placeholder"}
        )

    # Create a full ExplicitMemory object
    fact_memory_data = {
        "timestamp": time.time(),
        "source_turn_id": invocation_id_log,
        "source_interaction_id": session_id_log,
        "source_type": MemorySourceType.INTERNAL_REFLECTION,
        "source_agent_name": agent_name_log,
        "salience": salience_to_use,
        "keywords": keywords or [],
        "content": ExplicitMemoryContent(text_content=fact_text_content),
        "metadata": {"commit_tool_source_description": source_description} if source_description else {}
        # memory_type will be set by ExplicitMemory model itself
    }
    fact_memory: Optional[ExplicitMemory] = None
    try:
        fact_memory = ExplicitMemory(**fact_memory_data)
    except Exception as e_create:
        logger.error(f"Memory Tool (LTM Commit): Failed to create ExplicitMemory object: {e_create}", exc_info=True)
        return create_error_tool_response("Failed to create memory object for LTM commit.", details=str(e_create))

    pending_commits_key = "mbs:pending_memory_commits"
    if pending_commits_key not in tool_context.state:
        tool_context.state[pending_commits_key] = []
    # Store the Pydantic model's dictionary representation
    tool_context.state[pending_commits_key].append(fact_memory.model_dump(exclude_none=True))
    fact_memory_id = getattr(fact_memory, 'memory_id', None) # Should be set by default_factory
    logger.info(f"Memory Tool (LTM Commit): Fact '{fact_memory_id}' (type: {getattr(fact_memory, 'memory_type', 'N/A')}) queued for commit to LTM via session state.")
    return create_successful_tool_response(
        data={"memory_id_queued": fact_memory_id},
        message="Fact has been queued for commit to long-term memory."
    )

commit_explicit_fact_ltm_tool = FunctionTool(
    func=commit_explicit_fact_to_long_term_memory
)

ceaf_memory_tools = [
    store_stm_tool,
    retrieve_stm_tool,
    query_ltm_tool, # This is now async
    commit_explicit_fact_ltm_tool,
]

