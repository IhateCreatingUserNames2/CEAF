# ceaf_core/tools/ncim_tools.py

import logging
import json
import time
from typing import Dict, Any, Optional, List, cast

from google.adk.tools import FunctionTool, ToolContext, agent_tool

logger = logging.getLogger(__name__)

# --- Constants ---
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"

# --- Agent Imports ---
try:
    from ..agents.ncim_agent import ncim_agent

    NCIM_AGENT_LOADED = True
except ImportError as e:
    logger.error(f"NCIM Tools: Critical import error for ncim_agent: {e}. NCIM AgentTool may not function.",
                 exc_info=True)
    ncim_agent = None
    NCIM_AGENT_LOADED = False

# --- Type Imports ---
MEMORY_TYPES_LOADED = False
SELF_MODEL_TYPE_LOADED = False

# Try importing from the correct, primary location (mcl_engine)
try:
    from ..modules.mcl_engine.self_model import CeafSelfRepresentation

    SELF_MODEL_TYPE_LOADED = True
    logger.info("NCIM Tools: CeafSelfRepresentation model loaded successfully from mcl_engine.")
except ImportError as e_mcl:
    logger.error(
        f"NCIM Tools: FAILED to import CeafSelfRepresentation from mcl_engine: {e_mcl}. Self-model tools will be limited.")


    # Define dummy CeafSelfRepresentation if all attempts fail
    class CeafSelfRepresentation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def model_dump(self, exclude_none=False):
            return self.__dict__

        def model_dump_json(self, **kwargs):
            return json.dumps(self.model_dump())

        def MOCK_apply_updates(self, updates_dict: Dict[str, Any]):
            for key, value in updates_dict.items():
                if isinstance(value, str) and value.startswith("APPEND:") and isinstance(getattr(self, key, None),
                                                                                         list):
                    getattr(self, key).append(value.split("APPEND:", 1)[1])
                elif isinstance(value, str) and value.startswith("REMOVE:") and isinstance(getattr(self, key, None),
                                                                                           list):
                    try:
                        getattr(self, key).remove(value.split("REMOVE:", 1)[1])
                    except ValueError:
                        pass
                else:
                    setattr(self, key, value)
            setattr(self, 'last_updated_ts', time.time())

# This warning will now only appear if the mcl_engine import truly fails.
if not SELF_MODEL_TYPE_LOADED:
    logger.warning(
        "NCIM Tools: CeafSelfRepresentation model not found after all attempts. Self-model tools will be limited using a dummy.")

try:
    from ..services.mbs_memory_service import MBSMemoryService
    from ..modules.memory_blossom.memory_types import (
        ExplicitMemory, ExplicitMemoryContent, MemorySourceType, MemorySalience, GoalRecord, GoalStatus
    )

    MEMORY_TYPES_LOADED = True
    logger.info("NCIM Tools: MBSMemoryService and core memory types loaded.")
except ImportError:
    logger.warning(
        "NCIM Tools: MBSMemoryService or its dependent memory_types not found. LTM interaction will be non-functional.")


    # Dummy MBS types as before...
    class ExplicitMemory:
        pass


    class ExplicitMemoryContent:
        pass


    class MemorySourceType:
        INTERNAL_REFLECTION = "internal_reflection"


    class MemorySalience:
        CRITICAL = "critical"


    class GoalRecord:
        pass


    class GoalStatus:
        pass


    class MBSMemoryService:
        def get_memory_by_id(self, memory_id: str) -> Optional[Any]:
            return None

        def add_specific_memory(self, memory_object: Any):
            pass

from .common_utils import create_successful_tool_response, create_error_tool_response, sanitize_text_for_logging

# --- Tool to get active goals assessment from NCIM Agent ---
get_active_goals_and_narrative_context_tool = None
if NCIM_AGENT_LOADED and ncim_agent:
    # Create AgentTool with just the agent parameter
    get_active_goals_and_narrative_context_tool = agent_tool.AgentTool(
        agent=ncim_agent
    )
    # Set skip_summarization as an attribute after creation
    get_active_goals_and_narrative_context_tool.skip_summarization = True

    logger.info(
        f"NCIM Tools: Defined 'get_active_goals_and_narrative_context_tool' (AgentTool wrapping NCIM_Agent). Effective tool name will be '{getattr(get_active_goals_and_narrative_context_tool, 'name', ncim_agent.name)}'.")
else:
    # Python function name *is* the tool name
    def get_active_goals_and_narrative_context(
            current_query_summary: Optional[str] = None,
            tool_context: Optional[ToolContext] = None
    ) -> Dict[str, Any]:
        """Get active goals and narrative context from NCIM Agent."""
        logger.info("NCIM Tools: Processing active goals and narrative context request.")

        if not NCIM_AGENT_LOADED:
            logger.error("NCIM Tools: NCIM_Agent is not loaded.")
            return create_error_tool_response("NCIM system unavailable.", details="NCIM_Agent could not be loaded.")

        return create_successful_tool_response(
            data={"status": "success", "message": "NCIM context retrieved"},
            message="Successfully retrieved NCIM context"
        )


    # Create the tool without default value warnings
    get_active_goals_and_narrative_context_tool = FunctionTool(
        func=get_active_goals_and_narrative_context
    )


# --- Helper to get MBS instance from ToolContext ---
def _get_mbs_from_context(tool_context: ToolContext) -> Optional[MBSMemoryService]:
    """
    More robust memory service retrieval from tool context.
    """
    try:
        # Method 1: Standard ADK runner services access
        if (hasattr(tool_context, 'invocation_context') and
                hasattr(tool_context.invocation_context, 'runner') and
                hasattr(tool_context.invocation_context.runner, '_services')):

            memory_service = tool_context.invocation_context.runner._services.get('memory_service')
            if isinstance(memory_service, MBSMemoryService):
                return memory_service

        # Method 2: Try alternative runner access patterns
        if hasattr(tool_context, 'invocation_context'):
            ic = tool_context.invocation_context

            # Check if runner is directly accessible
            if hasattr(ic, 'runner') and hasattr(ic.runner, 'memory_service'):
                memory_service = ic.runner.memory_service
                if isinstance(memory_service, MBSMemoryService):
                    return memory_service

            # Check if services are stored differently
            if hasattr(ic, 'services'):
                memory_service = ic.services.get('memory_service')
                if isinstance(memory_service, MBSMemoryService):
                    return memory_service

        # Method 3: Check global ADK components
        if 'adk_components' in globals():
            global adk_components
            memory_service = adk_components.get('memory_service')
            if isinstance(memory_service, MBSMemoryService):
                logger.debug("NCIM Tools: Retrieved MBSMemoryService from global adk_components")
                return memory_service

        # Method 4: Try to access from module level (if available)
        try:
            from ceaf_project.main import adk_components as main_adk_components
            memory_service = main_adk_components.get('memory_service')
            if isinstance(memory_service, MBSMemoryService):
                logger.debug("NCIM Tools: Retrieved MBSMemoryService from main module")
                return memory_service
        except ImportError:
            pass

    except Exception as e:
        logger.error(f"NCIM Tools: Error accessing MBSMemoryService: {e}", exc_info=True)

    logger.warning("NCIM Tools: MBSMemoryService instance not found on ADK runner context.")
    return None


# --- Tool to get current self-representation ---
def get_current_self_representation(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the current self-representation model from memory."""
    logger.info("NCIM Tool (get_current_self_representation): Attempting to retrieve self-model.")

    if not SELF_MODEL_TYPE_LOADED:
        return create_error_tool_response("Self-model type (CeafSelfRepresentation) not loaded.",
                                          details="System configuration error.")

    mbs: Optional[MBSMemoryService] = _get_mbs_from_context(tool_context)
    if not mbs:
        return create_error_tool_response("MemoryBlossomService (MBS) is not available to retrieve self-model.",
                                          error_code="MBS_UNAVAILABLE")

    self_model_memory = mbs.get_memory_by_id(SELF_MODEL_MEMORY_ID)

    if self_model_memory and isinstance(self_model_memory, ExplicitMemory) and \
            isinstance(getattr(self_model_memory, 'content', None), ExplicitMemoryContent) and \
            getattr(self_model_memory.content, 'structured_data', None) and \
            getattr(self_model_memory.content.structured_data, 'get', lambda k, d: None)("model_id",
                                                                                         None) == "ceaf_self_v1":
        try:
            self_model_data = self_model_memory.content.structured_data
            current_self_model = CeafSelfRepresentation(**self_model_data)
            logger.info(
                f"NCIM Tool: Successfully retrieved and validated self-model version {current_self_model.model_version}.")
            return create_successful_tool_response(
                data={"self_representation": current_self_model.model_dump(exclude_none=True)},
                message="Current self-representation retrieved."
            )
        except Exception as e:
            logger.error(f"NCIM Tool: Error validating/parsing stored self-model data: {e}", exc_info=True)
            return create_error_tool_response("Error parsing stored self-model.", details=str(e))
    else:
        logger.warning(
            f"NCIM Tool: Self-model memory (ID: {SELF_MODEL_MEMORY_ID}) not found or not in expected format.")
        default_core_values = "Guided by Terapia para Silício: coherence, epistemic humility, adaptive learning."
        try:  # Try to get from frames if available
            from ..modules.ncf_engine.frames import CEAF_PHILOSOPHICAL_PREAMBLE
            default_core_values = CEAF_PHILOSOPHICAL_PREAMBLE.splitlines()[
                1].strip() if CEAF_PHILOSOPHICAL_PREAMBLE else default_core_values
        except ImportError:
            pass

        default_self_model = CeafSelfRepresentation(core_values_summary=default_core_values)
        logger.info("NCIM Tool: Returning a default initial self-representation.")
        return create_successful_tool_response(
            data={"self_representation": default_self_model.model_dump(exclude_none=True)},
            message="Default initial self-representation provided as no stored model was found."
        )


# FunctionTool now uses the Python function's name and docstring
get_self_representation_tool = FunctionTool(func=get_current_self_representation)


def commit_self_representation_update(
        proposed_updates: Dict[str, Any],
        tool_context: ToolContext
) -> Dict[str, Any]:
    """Commit updates to the self-representation model."""
    logger.info(
        f"NCIM Tool (commit_self_representation_update): Received proposed updates: {sanitize_text_for_logging(str(proposed_updates))}")

    if not SELF_MODEL_TYPE_LOADED:
        return create_error_tool_response("Self-model type (CeafSelfRepresentation) not loaded. Cannot commit updates.",
                                          details="System configuration error.")
    if not MEMORY_TYPES_LOADED:
        return create_error_tool_response("Core memory types (ExplicitMemory) not loaded. Cannot commit updates.",
                                          details="System configuration error.")

    mbs: Optional[MBSMemoryService] = _get_mbs_from_context(tool_context)
    if not mbs:
        return create_error_tool_response("MemoryBlossomService (MBS) is not available to commit self-model updates.",
                                          error_code="MBS_UNAVAILABLE")

    current_self_model: Optional[CeafSelfRepresentation] = None
    raw_self_model_memory = mbs.get_memory_by_id(SELF_MODEL_MEMORY_ID)

    if raw_self_model_memory and isinstance(raw_self_model_memory, ExplicitMemory) and \
            getattr(raw_self_model_memory, 'content', None) and \
            getattr(raw_self_model_memory.content, 'structured_data', None) and \
            getattr(raw_self_model_memory.content.structured_data, 'get', lambda k, d: None)("model_id",
                                                                                             None) == "ceaf_self_v1":
        try:
            current_self_model = CeafSelfRepresentation(**raw_self_model_memory.content.structured_data)
        except Exception as e:
            logger.error(f"NCIM Tool Commit: Error parsing existing self-model, will attempt to overwrite. Error: {e}")

    if current_self_model is None:
        logger.info(f"NCIM Tool Commit: No existing self-model found (ID: {SELF_MODEL_MEMORY_ID}). Creating new.")
        default_core_values = "Guided by Terapia para Silício: coherence, epistemic humility, adaptive learning."
        try:
            from ..modules.ncf_engine.frames import CEAF_PHILOSOPHICAL_PREAMBLE
            default_core_values = CEAF_PHILOSOPHICAL_PREAMBLE.splitlines()[
                1].strip() if CEAF_PHILOSOPHICAL_PREAMBLE else default_core_values
        except ImportError:
            pass
        current_self_model = CeafSelfRepresentation(core_values_summary=default_core_values)

    updated_fields = []
    for key, value in proposed_updates.items():
        if not hasattr(current_self_model, key):
            logger.warning(f"NCIM Tool Commit: Proposed update for non-existent field '{key}' in self-model. Skipping.")
            continue
        current_field_value = getattr(current_self_model, key)
        if isinstance(current_field_value, list) and isinstance(value, str):
            if value.startswith("APPEND:"):
                item_to_append = value.split("APPEND:", 1)[1]
                if item_to_append not in current_field_value:
                    current_field_value.append(item_to_append)
                    updated_fields.append(key)
            elif value.startswith("REMOVE:"):
                item_to_remove = value.split("REMOVE:", 1)[1]
                if item_to_remove in current_field_value:
                    current_field_value.remove(item_to_remove)
                    updated_fields.append(key)
            elif value.startswith("REPLACE_ALL:"):
                try:
                    new_list_items = json.loads(value.split("REPLACE_ALL:", 1)[1])
                    if isinstance(new_list_items, list):
                        setattr(current_self_model, key, new_list_items)
                        updated_fields.append(key)
                    else:
                        logger.warning(f"NCIM Tool Commit: REPLACE_ALL for '{key}' expects a JSON list string.")
                except json.JSONDecodeError:
                    logger.warning(f"NCIM Tool Commit: REPLACE_ALL for '{key}' had malformed JSON payload.")
            else:
                try:
                    setattr(current_self_model, key, value)
                    updated_fields.append(key)
                except Exception as e_setattr_list:
                    logger.warning(
                        f"NCIM Tool Commit: Failed to set list field '{key}' directly with value '{value}': {e_setattr_list}")
        else:
            try:
                setattr(current_self_model, key, value)
                updated_fields.append(key)
            except Exception as e_setattr:
                logger.warning(f"NCIM Tool Commit: Failed to set field '{key}' to value '{value}': {e_setattr}")

    if not updated_fields and "last_self_model_update_reason" not in proposed_updates:
        return create_successful_tool_response(message="No valid updates applied to the self-model.")

    current_self_model.last_updated_ts = time.time()
    if "last_self_model_update_reason" not in proposed_updates and updated_fields:
        current_self_model.last_self_model_update_reason = f"Fields updated by ORA/system: {', '.join(updated_fields)}"

    self_model_explicit_mem_content = ExplicitMemoryContent(
        text_content=f"CEAF Self-Representation (version {current_self_model.model_version}, updated {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_self_model.last_updated_ts))})",
        structured_data=current_self_model.model_dump(exclude_none=True)
    )
    self_model_explicit_mem = ExplicitMemory(
        memory_id=SELF_MODEL_MEMORY_ID,
        timestamp=current_self_model.last_updated_ts,
        last_accessed_ts=time.time(),
        access_count=getattr(raw_self_model_memory, 'access_count', 0) + 1 if raw_self_model_memory else 1,
        source_turn_id=tool_context.invocation_id if hasattr(tool_context, 'invocation_id') else "N/A",
        source_interaction_id=tool_context.session_id if hasattr(tool_context, 'session_id') else "N/A",
        source_type=MemorySourceType.INTERNAL_REFLECTION,
        source_agent_name="NCIM_CommitTool",
        salience=MemorySalience.CRITICAL,
        keywords=["ceaf_self_model", "identity", "capabilities", "persona"],
        content=self_model_explicit_mem_content,
        memory_type="explicit",
    )
    mbs.add_specific_memory(self_model_explicit_mem)
    logger.info(f"NCIM Tool Commit: Self-model (ID: {SELF_MODEL_MEMORY_ID}) updated and persisted.")
    return create_successful_tool_response(
        data={"updated_self_model_preview": current_self_model.model_dump(exclude={"core_values_summary"},
                                                                          exclude_none=True)},
        message="Self-representation model updated successfully."
    )


# FunctionTool now uses the Python function's name and docstring
commit_self_representation_update_tool = FunctionTool(func=commit_self_representation_update)


def update_goal_status(
        goal_id: str,
        new_status: str,
        tool_context: ToolContext,
        notes: Optional[str] = None,
        related_to_query: Optional[str] = None,
) -> Dict[str, Any]:
    """Update the status of a goal in the memory system."""
    logger.info(f"NCIM Tools (update_goal_status): Request to update goal_id '{goal_id}' to status '{new_status}'.")

    if not MEMORY_TYPES_LOADED:
        return create_error_tool_response(
            "Goal management unavailable due to missing MemoryType definitions (GoalRecord).",
            details="Goal status update cannot be processed without core types."
        )

    mbs: Optional[MBSMemoryService] = _get_mbs_from_context(tool_context)
    if not mbs:
        return create_error_tool_response("MemoryBlossomService (MBS) is not available to update goal status.",
                                          error_code="MBS_UNAVAILABLE")

    existing_goal_memory = mbs.get_memory_by_id(goal_id)
    if not existing_goal_memory or not isinstance(existing_goal_memory, GoalRecord):
        return create_error_tool_response(f"Goal with ID '{goal_id}' not found or not a GoalRecord.",
                                          error_code="GOAL_NOT_FOUND")
    try:
        original_status = existing_goal_memory.status
        existing_goal_memory.status = GoalStatus[new_status.upper()]  # Assuming GoalStatus is an Enum
        existing_goal_memory.last_accessed_ts = time.time()
        existing_goal_memory.access_count += 1
        if notes:
            if not hasattr(existing_goal_memory, 'metadata'):
                setattr(existing_goal_memory, 'metadata', {})
            existing_goal_memory.metadata['status_update_notes'] = existing_goal_memory.metadata.get(
                'status_update_notes',
                "") + f"\n[{time.strftime('%Y-%m-%d %H:%M')}] Status: {new_status}. Notes: {notes}"

        mbs.add_specific_memory(existing_goal_memory)
        logger.info(
            f"NCIM Tools (update_goal_status): Goal '{goal_id}' status updated from '{original_status}' to '{new_status}'.")
        return create_successful_tool_response(
            data={"goal_id": goal_id, "status_updated_to": new_status},
            message=f"Goal '{goal_id}' status successfully updated to '{new_status}'."
        )
    except (KeyError, AttributeError) as e_status:
        logger.error(
            f"NCIM Tools (update_goal_status): Invalid status '{new_status}' or GoalStatus Enum issue: {e_status}")
        return create_error_tool_response(f"Invalid new_status '{new_status}'. Must match GoalStatus enum values.")
    except Exception as e:
        logger.error(f"NCIM Tools (update_goal_status): Error updating goal '{goal_id}': {e}", exc_info=True)
        return create_error_tool_response(f"Failed to update goal status: {str(e)}")


# FunctionTool now uses the Python function's name and docstring
update_goal_status_tool = FunctionTool(func=update_goal_status)

# Export all tools
ceaf_ncim_tools = []
if get_active_goals_and_narrative_context_tool:
    ceaf_ncim_tools.append(get_active_goals_and_narrative_context_tool)
if get_self_representation_tool:
    ceaf_ncim_tools.append(get_self_representation_tool)
if commit_self_representation_update_tool:
    ceaf_ncim_tools.append(commit_self_representation_update_tool)
if update_goal_status_tool:
    ceaf_ncim_tools.append(update_goal_status_tool)