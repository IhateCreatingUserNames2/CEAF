# MCL Callbacks - FIXED VERSION
# ceaf_project/ceaf_core/callbacks/mcl_callbacks.py

import logging
import time
import json  # For serializing complex data if needed

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from typing import Optional, Dict, Any, List
from ..services.persistent_log_service import PersistentLogService
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

logger_prompt_debug = logging.getLogger("LLMPromptDebug") # Use a specific logger

# --- Constants for State Keys ---
MCL_OBSERVATIONS_LIST_KEY = "mcl:ora_turn_observations_log"

try:
    persistent_log_service = PersistentLogService()  # Uses default DB path
    PLS_AVAILABLE = True
except Exception as e_pls:
    logger.error(f"MCL Callbacks: Failed to initialize PersistentLogService: {e_pls}. Persistent logging disabled.",
                 exc_info=True)
    persistent_log_service = None  # type: ignore
    PLS_AVAILABLE = False


# --- Helper Function to Add Observation to Log ---
def _add_mcl_observation(
        state: Dict[str, Any], # This is callback_context.state or tool_context.state
        observation_type: str,
        data: Any,
        turn_id: Optional[str] = None,
        source_agent_pls: Optional[str] = None,
        session_id_pls: Optional[str] = None # session_id specifically for PLS
):
    # Prepare the entry for the volatile session state log
    log_entry_for_session_state = {
        "timestamp": time.time(),
        # Use turn_id from context if available, otherwise what's passed or default
        "turn_id": turn_id or state.get("current_interaction_turn_id", "unknown_turn"),
        "observation_type": observation_type,
        "data": data
    }

    if MCL_OBSERVATIONS_LIST_KEY not in state:
        state[MCL_OBSERVATIONS_LIST_KEY] = []

    state[MCL_OBSERVATIONS_LIST_KEY].append(log_entry_for_session_state)

    MAX_LOG_ENTRIES_IN_STATE = 20
    if len(state[MCL_OBSERVATIONS_LIST_KEY]) > MAX_LOG_ENTRIES_IN_STATE:
        state[MCL_OBSERVATIONS_LIST_KEY] = state[MCL_OBSERVATIONS_LIST_KEY][-MAX_LOG_ENTRIES_IN_STATE:]

    logger.debug(
        f"MCL Observation Added to session state: {observation_type} for turn {log_entry_for_session_state['turn_id']}")

    if PLS_AVAILABLE and persistent_log_service:
        try:
            tags = ["mcl_observation"]
            if source_agent_pls: tags.append(source_agent_pls.lower())
            else: tags.append("unknown_source")

            if "error" in observation_type.lower() or \
               (isinstance(data, dict) and data.get("status") == "error") or \
               (isinstance(data, dict) and "error" in str(data.get("response_summary", "")).lower()):
                tags.append("error")

            persistent_log_service.log_event(
                event_type=f"MCL_OBSERVATION_{observation_type.upper()}",
                data_payload=data,
                source_agent=source_agent_pls or "MCL_Callback_System",
                session_id=session_id_pls, # Use the explicitly passed session_id for PLS
                turn_id=log_entry_for_session_state['turn_id'],
                tags=list(set(tags))
            )
            logger.debug(
                f"MCL Observation logged to PersistentLogService: {observation_type} for turn {log_entry_for_session_state['turn_id']}")
        except Exception as e_log:
            logger.error(
                f"MCL Callbacks: Failed to log observation '{observation_type}' to PersistentLogService: {e_log}",
                exc_info=True)


# --- Helper function to safely extract LlmRequest attributes ---
def _extract_request_summary(llm_request: LlmRequest) -> Dict[str, Any]:
    """
    Safely extract summary information from LlmRequest object.
    Handles different possible attribute names and structures.
    """
    summary = {}

    # Model information
    try:
        if hasattr(llm_request, 'model'):
            if hasattr(llm_request.model, 'model'):
                summary["model_name"] = llm_request.model.model
            else:
                summary["model_name"] = str(llm_request.model)
        else:
            summary["model_name"] = "Unknown"
    except Exception as e:
        summary["model_name"] = f"Error extracting model: {e}"

    # Instruction/Content information - try different possible attributes
    instruction_snippet = "N/A"
    try:
        # Try common attribute names for instruction/prompt content
        for attr_name in ['instruction', 'prompt', 'system_instruction', 'messages']:
            if hasattr(llm_request, attr_name):
                attr_value = getattr(llm_request, attr_name)
                if attr_value:
                    if isinstance(attr_value, str):
                        instruction_snippet = attr_value[:200] + ("..." if len(attr_value) > 200 else "")
                    else:
                        instruction_snippet = str(attr_value)[:200] + ("..." if len(str(attr_value)) > 200 else "")
                    break

        # If no instruction found, try to get from contents
        if instruction_snippet == "N/A" and hasattr(llm_request, 'contents'):
            contents = llm_request.contents
            if contents and len(contents) > 0:
                # Try to extract text from first content item
                first_content = contents[0]
                if hasattr(first_content, 'parts') and first_content.parts:
                    for part in first_content.parts:
                        if hasattr(part, 'text') and part.text:
                            instruction_snippet = part.text[:200] + ("..." if len(part.text) > 200 else "")
                            break
                elif hasattr(first_content, 'text'):
                    instruction_snippet = first_content.text[:200] + ("..." if len(first_content.text) > 200 else "")
    except Exception as e:
        instruction_snippet = f"Error extracting instruction: {e}"

    summary["instruction_snippet"] = instruction_snippet

    # Contents count
    try:
        if hasattr(llm_request, 'contents') and llm_request.contents:
            summary["num_contents"] = len(llm_request.contents)
        else:
            summary["num_contents"] = 0
    except:
        summary["num_contents"] = 0

    # Tools information
    try:
        if hasattr(llm_request, 'tools') and llm_request.tools:
            summary["tool_names"] = [getattr(tool, 'name', 'UnknownTool') for tool in llm_request.tools]
        else:
            summary["tool_names"] = []
    except:
        summary["tool_names"] = []

    # Generate config
    try:
        if hasattr(llm_request, 'generate_config') and llm_request.generate_config:
            if hasattr(llm_request.generate_config, 'model_dump'):
                summary["generate_config"] = llm_request.generate_config.model_dump()
            else:
                summary["generate_config"] = str(llm_request.generate_config)
        else:
            summary["generate_config"] = None
    except:
        summary["generate_config"] = None

    return summary


# --- ORA Callbacks for MCL ---

# Update your MCL callbacks to handle context access more robustly

def ora_before_model_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    if agent_name != "ORA":
        return None

    logger.info(f"MCL Callback (ora_before_model_callback) for ORA: Capturing LLM request.")

    # More robust session_id access with multiple fallbacks
    session_id_for_pls = None

    # Try multiple ways to get session_id
    try:
        # Method 1: Via invocation_context
        if hasattr(callback_context, 'invocation_context'):
            if hasattr(callback_context.invocation_context, 'session_id'):
                session_id_for_pls = callback_context.invocation_context.session_id
            elif hasattr(callback_context.invocation_context, 'session'):
                session_id_for_pls = getattr(callback_context.invocation_context.session, 'id', None)

        # Method 2: Direct session_id attribute
        if not session_id_for_pls and hasattr(callback_context, 'session_id'):
            session_id_for_pls = callback_context.session_id

        # Method 3: Via state if it contains session info
        if not session_id_for_pls and hasattr(callback_context, 'state'):
            session_id_for_pls = callback_context.state.get('session_id', None)

        # Method 4: Use invocation_id as fallback
        if not session_id_for_pls:
            session_id_for_pls = getattr(callback_context, 'invocation_id', "unknown_session")
            logger.debug(f"MCL Callback: Using invocation_id '{session_id_for_pls}' as session_id fallback")

    except Exception as e:
        logger.warning(f"MCL Callback (ora_before_model): Error accessing session_id: {e}")
        session_id_for_pls = "callback_error_session"

    if not session_id_for_pls:
        logger.warning("MCL Callback (ora_before_model): session_id is None, using 'none_session' fallback")
        session_id_for_pls = "none_session"

    # Continue with rest of callback logic...
    try:
        request_summary = _extract_request_summary(llm_request)
        _add_mcl_observation(
            state=callback_context.state,
            observation_type="ora_llm_request_prepared",
            data=request_summary,
            turn_id=callback_context.invocation_id,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    except Exception as e:
        logger.error(f"MCL Callback: Error in ora_before_model_callback processing: {e}", exc_info=True)

    return None


def ora_after_model_callback(
        callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    agent_name = callback_context.agent_name
    if agent_name != "ORA":
        return None

    logger.info(f"MCL Callback (ora_after_model_callback) for ORA: Capturing LLM response.")

    # More robust session_id access with multiple fallbacks (same as ora_before_model_callback)
    session_id_for_pls = None

    # Try multiple ways to get session_id
    try:
        # Method 1: Via invocation_context
        if hasattr(callback_context, 'invocation_context'):
            if hasattr(callback_context.invocation_context, 'session_id'):
                session_id_for_pls = callback_context.invocation_context.session_id
            elif hasattr(callback_context.invocation_context, 'session'):
                session_id_for_pls = getattr(callback_context.invocation_context.session, 'id', None)

        # Method 2: Direct session_id attribute
        if not session_id_for_pls and hasattr(callback_context, 'session_id'):
            session_id_for_pls = callback_context.session_id

        # Method 3: Via state if it contains session info
        if not session_id_for_pls and hasattr(callback_context, 'state'):
            session_id_for_pls = callback_context.state.get('session_id', None)

        # Method 4: Use invocation_id as fallback
        if not session_id_for_pls:
            session_id_for_pls = getattr(callback_context, 'invocation_id', "unknown_session")
            logger.debug(f"MCL Callback: Using invocation_id '{session_id_for_pls}' as session_id fallback")

    except Exception as e:
        logger.warning(f"MCL Callback (ora_after_model): Error accessing session_id: {e}")
        session_id_for_pls = "callback_error_session"

    if not session_id_for_pls:
        logger.warning("MCL Callback (ora_after_model): session_id is None, using 'none_session' fallback")
        session_id_for_pls = "none_session"

    try:
        response_text_snippet = "N/A"

        if llm_response.content and llm_response.content.parts:
            text_parts = [part.text for part in llm_response.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                full_text = " ".join(text_parts).strip()
                response_text_snippet = full_text[:100] + "..." if len(full_text) > 100 else full_text

        # FIX: Use hasattr to check for function_calls instead of calling get_function_calls()
        function_calls = []
        if hasattr(llm_response, 'function_calls') and llm_response.function_calls:
            function_calls = [fc.name for fc in llm_response.function_calls]
        elif hasattr(llm_response.content, 'parts'):
            # Alternative: Check for function calls in content parts
            for part in llm_response.content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append(part.function_call.name)

        # FIX: Safely access usage_metadata attribute
        usage_metadata = None
        if hasattr(llm_response, 'usage_metadata') and llm_response.usage_metadata:
            try:
                usage_metadata = llm_response.usage_metadata.model_dump()
            except AttributeError:
                # Fallback: try to access as dict or other format
                usage_metadata = dict(llm_response.usage_metadata) if llm_response.usage_metadata else None
        elif hasattr(llm_response, 'usage') and llm_response.usage:
            # Some versions might use 'usage' instead of 'usage_metadata'
            try:
                usage_metadata = llm_response.usage.model_dump() if hasattr(llm_response.usage, 'model_dump') else dict(llm_response.usage)
            except (AttributeError, TypeError):
                usage_metadata = str(llm_response.usage)

        response_summary = {
            "has_content": bool(llm_response.content),
            "num_parts": len(llm_response.content.parts) if llm_response.content else 0,
            "text_response_snippet": response_text_snippet,
            "finish_reason_from_llm": (
                getattr(llm_response, 'finish_reason', None).name
                if hasattr(getattr(llm_response, 'finish_reason', None), 'name')
                else str(getattr(llm_response, 'finish_reason', 'unknown'))
            ),
            "function_calls": function_calls,
            "usage_metadata": usage_metadata,  # FIX: Use safely extracted usage_metadata
        }

        _add_mcl_observation(
            state=callback_context.state,
            observation_type="ora_llm_response_received",
            data=response_summary,
            turn_id=callback_context.invocation_id,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    except Exception as e:
        logger.error(f"MCL Callback: Error in ora_after_model_callback processing: {e}", exc_info=True)
        _add_mcl_observation(
            state=callback_context.state,
            observation_type="ora_llm_response_received_error",
            data={"error": f"Failed to extract response summary: {str(e)}"},
            turn_id=callback_context.invocation_id,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    return None


def ora_before_tool_callback(
        tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    agent_name = getattr(tool_context, 'agent_name', "UnknownAgent")
    if agent_name != "ORA":
        return None

    tool_name_log = getattr(tool, 'name', "UnknownTool")
    logger.info(f"MCL Callback (ora_before_tool_callback) for ORA: Capturing tool call '{tool_name_log}'.")

    session_id_for_pls = getattr(tool_context, 'session_id',
                                getattr(getattr(tool_context, 'invocation_context', None), 'session_id', "unknown_session_id"))
    turn_id_for_pls_and_state = getattr(tool_context, 'invocation_id', "unknown_turn_id")

    try:
        # Robust handling of tool description
        tool_description_str = getattr(tool, 'description', None) # Get description, could be None
        if tool_description_str is None:
            tool_description_str = "N/A" # Fallback if description is None
        else:
            tool_description_str = str(tool_description_str)[:100] + "..." # Ensure it's a string and slice

        tool_call_data = {
            "tool_name": tool_name_log,
            "tool_description": tool_description_str, # Use the robustly handled string
            "arguments": {k: (str(v)[:100] + "..." if len(str(v)) > 100 else v) for k, v in args.items()},
        }
        _add_mcl_observation(
            state=tool_context.state,
            observation_type="ora_tool_call_attempted",
            data=tool_call_data,
            turn_id=turn_id_for_pls_and_state,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    except Exception as e:
        logger.error(f"MCL Callback: Error in ora_before_tool_callback processing: {e}", exc_info=True)
        _add_mcl_observation(
            state=tool_context.state,
            observation_type="ora_tool_call_attempted_error",
            data={"error": f"Failed to process tool call data: {str(e)}", "tool_name": tool_name_log},
            turn_id=turn_id_for_pls_and_state,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    return None


def ora_after_tool_callback(
        tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[Dict]:
    agent_name = getattr(tool_context, 'agent_name', "UnknownAgent")
    if agent_name != "ORA":
        return None

    tool_name_log = getattr(tool, 'name', "UnknownTool")
    logger.info(f"MCL Callback (ora_after_tool_callback) for ORA: Capturing tool response for '{tool_name_log}'.")

    # Use safe attribute access for ToolContext session_id (consistency with ora_before_tool_callback)
    session_id_for_pls = getattr(tool_context, 'session_id',
                                getattr(getattr(tool_context, 'invocation_context', None), 'session_id', "unknown_session_id"))
    turn_id_for_pls_and_state = getattr(tool_context, 'invocation_id', "unknown_turn_id")

    try:
        response_summary_str = ""
        if isinstance(tool_response, dict):
            status = tool_response.get("status", tool_response.get("result"))
            if status:
                response_summary_str = f"Status: {status}, Data snippet: {str(tool_response)[:150]}..."
            else:
                response_summary_str = f"Dict keys: {list(tool_response.keys())}, Snippet: {str(tool_response)[:150]}..."
        elif isinstance(tool_response, str):
            response_summary_str = tool_response[:200] + ("..." if len(tool_response) > 200 else "")
        else:
            response_summary_str = str(type(tool_response))

        tool_response_data = {
            "tool_name": tool_name_log,
            "arguments_used": {k: (str(v)[:100] + "..." if len(str(v)) > 100 else v) for k, v in args.items()},
            "response_summary": response_summary_str,
        }
        _add_mcl_observation(
            state=tool_context.state,
            observation_type="ora_tool_response_received",
            data=tool_response_data,
            turn_id=turn_id_for_pls_and_state,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    except Exception as e:
        logger.error(f"MCL Callback: Error in ora_after_tool_callback processing: {e}", exc_info=True)
        _add_mcl_observation(
            state=tool_context.state,
            observation_type="ora_tool_response_received_error",
            data={"error": f"Failed to process tool response data: {str(e)}", "tool_name": tool_name_log},
            turn_id=turn_id_for_pls_and_state,
            source_agent_pls=agent_name,
            session_id_pls=session_id_for_pls
        )
    return None


def simple_prompt_logger_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    A simple callback to print LLM prompt details to the console.
    """
    agent_name = callback_context.agent_name
    invocation_id = callback_context.invocation_id

    logger_prompt_debug.info(f"\n--- LLM PROMPT START (Agent: {agent_name}, Turn: {invocation_id}) ---")
    print(f"\n--- LLM PROMPT START (Agent: {agent_name}, Turn: {invocation_id}) ---")  # Use print

    # 1. System Instruction
    system_instruction = None
    if hasattr(llm_request, 'instruction') and llm_request.instruction:
        system_instruction = llm_request.instruction
    elif hasattr(llm_request, 'system_instruction') and llm_request.system_instruction:
        system_instruction = llm_request.system_instruction

    if system_instruction:
        logger_prompt_debug.info(f"[SYSTEM INSTRUCTION]:\n{system_instruction}\n")
    else:
        logger_prompt_debug.info("[SYSTEM INSTRUCTION]: None provided in LlmRequest")

    # 2. Messages / Conversation History
    if hasattr(llm_request, 'contents') and llm_request.contents:
        logger_prompt_debug.info("[MESSAGES]:")
        for i, content_item in enumerate(llm_request.contents):
            role = content_item.role
            parts_str_list = []
            if content_item.parts:
                for part in content_item.parts:
                    if hasattr(part, 'text') and part.text:
                        parts_str_list.append(f"  Text: {part.text}")
                    if hasattr(part, 'function_call') and part.function_call:
                        parts_str_list.append(
                            f"  FunctionCall: {part.function_call.name}(args={part.function_call.args})")
                    # Add other part types if needed
            logger_prompt_debug.info(f"  ({i + 1}) Role: {role}\n" + "\n".join(parts_str_list))
        logger_prompt_debug.info("-" * 20)
    else:
        logger_prompt_debug.info("[MESSAGES]: None provided in LlmRequest.contents")

    # 3. Tools / Function Declarations
    if hasattr(llm_request, 'tools') and llm_request.tools:
        logger_prompt_debug.info("[TOOLS DECLARED]:")
        for i, adk_tool_declaration in enumerate(llm_request.tools):
            # ADK tools are often FunctionDeclarations or similar Pydantic models
            tool_name = getattr(adk_tool_declaration, 'name', f"UnknownTool_{i}")
            tool_desc = getattr(adk_tool_declaration, 'description', "No description")
            # For Pydantic models, model_dump() is useful, otherwise str()
            try:
                params = adk_tool_declaration.parameters.model_dump_json(indent=2) if hasattr(adk_tool_declaration,
                                                                                              'parameters') and hasattr(
                    adk_tool_declaration.parameters, 'model_dump_json') else "{}"
            except:
                params = str(getattr(adk_tool_declaration, 'parameters', {}))

            logger_prompt_debug.info(f"  ({i + 1}) Name: {tool_name}")
            logger_prompt_debug.info(f"      Desc: {tool_desc}")
            logger_prompt_debug.info(f"      Params Schema: {params}")
        logger_prompt_debug.info("-" * 20)
    else:
        logger_prompt_debug.info("[TOOLS DECLARED]: None")

    # 4. Generation Config
    if hasattr(llm_request, 'generate_config') and llm_request.generate_config:
        try:
            gen_config_str = llm_request.generate_config.model_dump_json(indent=2) if hasattr(
                llm_request.generate_config, 'model_dump_json') else str(llm_request.generate_config)
        except:
            gen_config_str = str(llm_request.generate_config)
        logger_prompt_debug.info(f"[GENERATION CONFIG]:\n{gen_config_str}\n")
    else:
        logger_prompt_debug.info("[GENERATION CONFIG]: None")

    logger_prompt_debug.info(f"--- LLM PROMPT END (Agent: {agent_name}, Turn: {invocation_id}) ---\n")

    return None  # Don't modify the request


async def gather_mcl_input_from_state(session_state: Dict[str, Any], turn_id: str) -> Dict[str, Any]:
    """
    Gathers necessary information from session_state to prepare input for the MCL_Agent.
    This function reads from the volatile session state log.
    """
    logger.info(f"MCL: Gathering input from session state for turn_id '{turn_id}'.")

    turn_observations_from_state = [
        obs for obs in session_state.get(MCL_OBSERVATIONS_LIST_KEY, [])
        if obs.get("turn_id") == turn_id
    ]

    # It's also beneficial to fetch specific NCF params and user query for this turn if logged to state
    # Example keys, adjust if they are stored differently
    current_ncf_params = session_state.get(f"ora_turn_ncf_params:{turn_id}", {})
    last_user_query = session_state.get(f"ora_turn_user_query:{turn_id}", "N/A")

    mcl_input = {
        "turn_id": turn_id,
        # This uses the in-memory session log for immediate MCL reaction
        "interaction_log_snippet_from_session_state": turn_observations_from_state[-5:],
        # Last 5 observations from session state
        "full_observation_log_for_turn_from_session_state": turn_observations_from_state,
        "current_ncf_parameters_for_turn": current_ncf_params,
        "ora_last_user_query_for_turn": last_user_query
    }
    logger.debug(f"MCL Input (from session state) for turn {turn_id}: {json.dumps(mcl_input, default=str)[:500]}...")
    return mcl_input