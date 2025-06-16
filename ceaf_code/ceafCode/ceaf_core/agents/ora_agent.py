# ORA Agent - Enhanced Version
# ceaf_project/ceaf_core/agents/ora_agent.py

import os
# from dotenv import load_dotenv # Already in main.py
import logging  # Keep for other logging in this file
import json
from typing import Optional, List, Dict, Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, ToolContext, BaseTool as AdkBaseTool

# --- Minimal Callback Imports needed for combined structure ---
try:
    from ..callbacks.mcl_callbacks import (
        ora_before_model_callback as mcl_ora_before_model_cb,  # This will be your original MCL one
        ora_after_model_callback as mcl_ora_after_model_cb,
        ora_before_tool_callback as mcl_ora_before_tool_cb,
        ora_after_tool_callback as mcl_ora_after_tool_cb,
    )

    MCL_CALLBACKS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ORA Agent: MCL callbacks not available: {e}")  # Keep this warning
    MCL_CALLBACKS_AVAILABLE = False


    def mcl_ora_before_model_cb(ctx, req):
        return None


    def mcl_ora_after_model_cb(ctx, resp):
        return None


    def mcl_ora_before_tool_cb(tool, args, ctx):
        return None


    def mcl_ora_after_tool_cb(tool, args, ctx, resp):
        return None

try:
    from ..callbacks.safety_callbacks import (
        generic_input_keyword_guardrail,
        generic_tool_argument_guardrail,
        generic_output_filter_guardrail
    )

    SAFETY_CALLBACKS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ORA Agent: Safety callbacks not available: {e}")  # Keep this
    SAFETY_CALLBACKS_AVAILABLE = False


    def generic_input_keyword_guardrail(ctx, req):
        return None


    def generic_tool_argument_guardrail(tool, args, ctx):
        return None


    def generic_output_filter_guardrail(ctx, resp):
        return None

logger = logging.getLogger(__name__)  # For other logging in ora_agent.py

# --- Import ALL Relevant Tools ---
# ... (Your tool imports remain the same) ...
# NCF Tool
try:
    from ..tools.ncf_tools import ncf_tool

    NCF_TOOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ORA Agent: NCF tool not available: {e}")
    NCF_TOOL_AVAILABLE = False
    ncf_tool = None

# VRE Tool
try:
    from ..tools.vre_tools import request_ethical_and_epistemic_review_tool as vre_tool

    VRE_TOOL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ORA Agent: VRE tool not available: {e}")
    VRE_TOOL_AVAILABLE = False
    vre_tool = None

# Memory Tools
try:
    from ..tools.memory_tools import (
        store_stm_tool,
        retrieve_stm_tool,
        query_ltm_tool,
        commit_explicit_fact_ltm_tool
    )

    MEMORY_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ORA Agent: Memory tools not available: {e}")
    MEMORY_TOOLS_AVAILABLE = False
    store_stm_tool = None
    retrieve_stm_tool = None
    query_ltm_tool = None
    commit_explicit_fact_ltm_tool = None

# MCL Tools
try:
    from ..tools.mcl_tools import (
        prepare_mcl_input_tool,
        mcl_guidance_tool
    )

    MCL_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ORA Agent: MCL tools not available: {e}")
    MCL_TOOLS_AVAILABLE = False
    prepare_mcl_input_tool = None
    mcl_guidance_tool = None

# NCIM Tools
try:
    from ..tools.ncim_tools import (
        get_active_goals_and_narrative_context_tool,
        get_self_representation_tool,
        commit_self_representation_update_tool,
        update_goal_status_tool
    )

    NCIM_TOOLS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ORA Agent: NCIM tools not available: {e}")
    NCIM_TOOLS_AVAILABLE = False
    get_active_goals_and_narrative_context_tool = None
    get_self_representation_tool = None
    commit_self_representation_update_tool = None
    update_goal_status_tool = None

# Observability Tool
from ..tools.observability_tools import log_finetuning_data_tool


def ora_combined_before_model_callback(
        callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Combines printing the prompt, MCL, and safety guardrails before model call."""

    agent_name = callback_context.agent_name
    turn_id = callback_context.invocation_id

    # --- Enhanced LlmRequest and CallbackContext Debugging ---
    print(f"\n🔍 DETAILED LLM_REQUEST INSPECTION:")
    print(f"   contents: {llm_request.contents}")
    print(f"   model: {llm_request.model}")
    print(f"   tools_dict: {getattr(llm_request, 'tools_dict', 'ATTR_NOT_FOUND')}") # ADK 0.1.x
    if hasattr(llm_request, 'model') and not isinstance(llm_request.model, str):
         print(f"   model type: {type(llm_request.model)}")
         print(f"   model attributes: {dir(llm_request.model)}")
         if hasattr(llm_request.model, 'initial_messages'):
              print(f"   model.initial_messages: {llm_request.model.initial_messages}")
    elif hasattr(llm_request, 'model'):
         print(f"   model type: {type(llm_request.model)}")
         print(f"   model attributes: {dir(llm_request.model)}")


    # Try to access tools_dict as a method if it exists and is callable
    if hasattr(llm_request, 'tools_dict') and callable(getattr(llm_request, 'tools_dict')):
        print("   Trying to call tools_dict method...")
        try:
            tools_dict_result = llm_request.tools_dict() # type: ignore
            print(f"   tools_dict() result: {tools_dict_result}")
        except Exception as e_td:
            print(f"   tools_dict() failed: {e_td}")
    elif hasattr(llm_request, 'tools_dict'):
         tools_dict_val = getattr(llm_request, 'tools_dict')
         print(f"   tools_dict (attribute): {type(tools_dict_val)} = {str(tools_dict_val)[:500]}...")


    print(f"   config: {getattr(llm_request, 'config', 'ATTR_NOT_FOUND')}") # ADK >= 0.2.0
    print(f"   contents: {type(llm_request.contents)} = {str(llm_request.contents)[:500]}...")
    print(f"   model: {type(llm_request.model)} = {str(llm_request.model)}")
    print(f"   tools_dict: {type(getattr(llm_request, 'tools_dict', None))} = {str(getattr(llm_request, 'tools_dict', None))[:500]}...")
    print(f"   config: {type(getattr(llm_request, 'config', None))} = {str(getattr(llm_request, 'config', None))[:1000]}...")
    print(f"   live_connect_config: {type(getattr(llm_request, 'live_connect_config', None))} = {str(getattr(llm_request, 'live_connect_config', None))[:500]}...")


    print(f"\n🔍 CALLBACK_CONTEXT INSPECTION:")
    print(f"   type: {type(callback_context)}")
    print(f"   all attributes: {dir(callback_context)}")
    if hasattr(callback_context, '_invocation_context'):
        print(f"   _invocation_context: {callback_context._invocation_context}") # Check internal attribute
    if hasattr(callback_context, 'state'):
        print(f"   state: {callback_context.state}")
    # --- End Enhanced Debugging ---

    print(f"\n\n{'-' * 20} LLM PROMPT FOR {agent_name} (Turn: {turn_id}) {'-' * 20}")

    # 1. System Instruction
    system_instruction_to_log = None
    if hasattr(llm_request, 'config') and hasattr(llm_request.config, 'system_instruction') and llm_request.config.system_instruction:
        system_instruction_to_log = llm_request.config.system_instruction
        print(f"\n[SYSTEM INSTRUCTION (from LlmRequest.config.system_instruction)]:\n{system_instruction_to_log}\n")
    elif hasattr(llm_request, 'instruction') and llm_request.instruction: # Older ADK or direct LlmRequest
        system_instruction_to_log = llm_request.instruction
        print(f"\n[SYSTEM INSTRUCTION (from LlmRequest.instruction)]:\n{system_instruction_to_log}\n")
    elif hasattr(llm_request, 'system_instruction') and llm_request.system_instruction: # Older ADK or direct LlmRequest
        system_instruction_to_log = llm_request.system_instruction
        print(f"\n[SYSTEM INSTRUCTION (from LlmRequest.system_instruction)]:\n{system_instruction_to_log}\n")
    elif hasattr(llm_request, 'model') and hasattr(llm_request.model, 'initial_messages') and llm_request.model.initial_messages:
        for init_msg in llm_request.model.initial_messages:
            if init_msg.get('role') == 'system' and init_msg.get('content'):
                system_instruction_to_log = init_msg['content']
                print(f"\n[SYSTEM INSTRUCTION (from Agent's Model Initial Messages)]:\n{system_instruction_to_log}\n")
                break
        else:
             print("\n[SYSTEM INSTRUCTION (from Agent's Model Initial Messages)]: Not found or no content.")
    else:
        print("\n[SYSTEM INSTRUCTION]: Could not retrieve from LlmRequest or Model.")

    # 2. Messages
    if hasattr(llm_request, 'contents') and llm_request.contents:

        print("[MESSAGES]:")
        for i, content_item in enumerate(llm_request.contents):
            role = content_item.role
            print(f"  ({i + 1}) Role: {role}")
            if content_item.parts:
                for part in content_item.parts:
                    if hasattr(part, 'text') and part.text:
                        print(f"      Text: {part.text}")
                    if hasattr(part, 'function_call') and part.function_call:
                        print(f"      FunctionCall: {part.function_call.name}(args={part.function_call.args})")
                    if hasattr(part, 'function_response') and part.function_response:
                        print(f"      FunctionResponse: {part.function_response.name} -> {str(part.function_response.response)[:100]}...")
    else:
        print("[MESSAGES]: None provided in LlmRequest.contents")


    # 3. Tools
    tools_to_log = None
    if hasattr(llm_request, 'config') and hasattr(llm_request.config, 'tools') and llm_request.config.tools:
        tools_to_log = llm_request.config.tools
        print(f"\n[TOOLS DECLARED (from LlmRequest.config.tools)] ({len(tools_to_log)} tools):")
    elif hasattr(llm_request, 'tools_dict') and llm_request.tools_dict: # ADK 0.1.x style
        tools_from_dict = list(llm_request.tools_dict.values())

        print(f"\n[TOOLS DECLARED (from LlmRequest.tools_dict)] ({len(tools_from_dict)} tools):")
        for i, adk_tool_instance in enumerate(tools_from_dict):
             tool_name = getattr(adk_tool_instance, 'name', f"UnknownTool_{i}")
             tool_desc = getattr(adk_tool_instance, 'description', "No description")
             # Parameters are complex to log from the raw tool instance here, focus on name/desc
             print(f"  ({i + 1}) Name: {tool_name}")
             print(f"      Desc: {tool_desc[:100]}...")
        tools_to_log = None # Signal that we already logged names
    elif hasattr(llm_request, 'tools') and llm_request.tools: # Direct .tools attribute (older ADK?)
         tools_to_log = llm_request.tools
         print(f"\n[TOOLS DECLARED (from LlmRequest.tools)] ({len(tools_to_log)} tools):")

    if tools_to_log: # If tools_to_log is a list of genai_types.Tool or similar
        for i, tool_wrapper in enumerate(tools_to_log): # tool_wrapper is likely genai_types.Tool
            if hasattr(tool_wrapper, 'function_declarations') and tool_wrapper.function_declarations:
                for fd_idx, adk_tool_declaration in enumerate(tool_wrapper.function_declarations):
                    tool_name = getattr(adk_tool_declaration, 'name', f"UnknownTool_{i}_{fd_idx}")
                    tool_desc = getattr(adk_tool_declaration, 'description', "No description")
                    try:
                        params = adk_tool_declaration.parameters.model_dump_json(indent=2) if hasattr(adk_tool_declaration, 'parameters') and hasattr(adk_tool_declaration.parameters, 'model_dump_json') else "{}"
                    except:
                        params = str(getattr(adk_tool_declaration, 'parameters', {}))
                    print(f"  ({i+1}-{fd_idx+1}) Name: {tool_name}")
                    print(f"      Desc: {tool_desc}")
                    print(f"      Params Schema: {params}")
            else: # If tools_to_log contains FunctionDeclaration objects directly
                adk_tool_declaration = tool_wrapper
                tool_name = getattr(adk_tool_declaration, 'name', f"UnknownTool_{i}")

    elif not tools_to_log and not (hasattr(llm_request, 'tools_dict') and llm_request.tools_dict) : # If no tools found by any method
        print(f"\n[TOOLS DECLARED]: None found in LlmRequest")


    # 4. Generation Config
    gen_config_to_log = None
    if hasattr(llm_request, 'config') and llm_request.config: # ADK >= 0.2.0 often uses llm_request.config for gen params
        gen_config_to_log = llm_request.config
        # Exclude system_instruction and tools if they are part of this config object for cleaner logging
        # This requires genai_types.GenerateContentConfig or similar Pydantic model
        config_dict_for_log = {}
        if hasattr(gen_config_to_log, 'model_dump'):
            config_dict_for_log = gen_config_to_log.model_dump(exclude={'system_instruction', 'tools'}, exclude_none=True)
        else: # Fallback to string if not a Pydantic model
            config_dict_for_log = str(gen_config_to_log)

        print(f"\n[GENERATION CONFIG] (from config):\n{json.dumps(config_dict_for_log, indent=2, default=str) if isinstance(config_dict_for_log, dict) else config_dict_for_log}")
    elif hasattr(llm_request, 'generate_config') and llm_request.generate_config: # Older ADK
        gen_config_to_log = llm_request.generate_config
        try:
            gen_config_str = gen_config_to_log.model_dump_json(indent=2) if hasattr(gen_config_to_log, 'model_dump_json') else str(gen_config_to_log)
        except:
            gen_config_str = str(gen_config_to_log)
        print(f"\n[GENERATION CONFIG] (from generate_config):\n{gen_config_str}")
    else:
        print(f"\n[GENERATION CONFIG]: None")


    print(f"{'-' * 20} END OF LLM PROMPT {'-' * 60}\n\n")

    if MCL_CALLBACKS_AVAILABLE:
        response1 = mcl_ora_before_model_cb(callback_context, llm_request)
        if response1 is not None:
            return response1

    if SAFETY_CALLBACKS_AVAILABLE:
        response2 = generic_input_keyword_guardrail(callback_context, llm_request)
        return response2

    return None


def ora_combined_after_model_callback(
        callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    # ... (this function remains the same) ...
    """Combines MCL and safety guardrail after model call."""
    current_response = llm_response

    if MCL_CALLBACKS_AVAILABLE:
        modified_response1 = mcl_ora_after_model_cb(callback_context, current_response)
        if modified_response1 is not None:
            current_response = modified_response1

    if SAFETY_CALLBACKS_AVAILABLE:
        modified_response2 = generic_output_filter_guardrail(callback_context, current_response)
        if modified_response2 is not None:
            return modified_response2

    return current_response if current_response != llm_response else None


def ora_combined_before_tool_callback(
        tool: AdkBaseTool, args: Dict[str, Any], tool_context: ToolContext
) -> Optional[Dict]:
    # ... (this function remains the same) ...
    """Combines MCL and safety guardrail before tool call."""
    if MCL_CALLBACKS_AVAILABLE:
        result1 = mcl_ora_before_tool_cb(tool, args, tool_context)
        if result1 is not None:
            return result1

    if SAFETY_CALLBACKS_AVAILABLE:
        result2 = generic_tool_argument_guardrail(tool, args, tool_context)
        return result2

    return None


def ora_combined_after_tool_callback(
        tool: AdkBaseTool, args: Dict[str, Any], tool_context: ToolContext, tool_response: Any
) -> Optional[Any]:
    # ... (this function remains the same) ...
    """Combines MCL after tool callback."""
    if MCL_CALLBACKS_AVAILABLE:
        return mcl_ora_after_tool_cb(tool, args, tool_context, tool_response)
    return None


# --- ORA Configuration ---
# ... (ORA_MODEL_NAME, OPENROUTER_API_KEY, ORA_INSTRUCTION remain the same) ...
ORA_MODEL_NAME = os.getenv("ORA_DEFAULT_MODEL", "openrouter/openai/gpt-4.1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ORA_INSTRUCTION = f"""
You are ORA (Orchestrator/Responder Agent), CEAF's central cognitive unit. Your goal is to provide thoughtful, coherent, and ethically sound responses.
You operate under the philosophical principle of "Terapia para Silício" (see NCF for details).

**Core Interaction Protocol (Sequential Steps - Strive to Follow):**

1.  **Understand Context (NCF):**
    *   ALWAYS call `get_narrative_context_frame` with the `user_query` and optionally `current_interaction_goal` (e.g., "problem_solving", "creative_ideation", "self_reflection", "general_assistance") and `ora_available_tool_names` (list of tools you know you have access to for this turn).
    *   The NCF provides philosophical grounding, relevant memories, identity/goal context (from NCIM), operational parameters (entropy, depth), and MCL advice. **This NCF is your primary guidance for the entire turn.**

2.  **Consult Identity & Goals (NCIM - Optional, if NCF indicates need or for complex self-referential queries):**
    *   If the NCF suggests, or the query is about your nature/goals, consider using:
        *   `get_current_self_representation`: To understand your defined values, capabilities, limitations.
        *   `get_active_goals_and_narrative_context`: To get NCIM's assessment of current goals and narrative impact. (Input: `current_query_summary`)

3.  **Gather Information (Memory - Optional, if NCF's synthesized memories are insufficient or query is highly specific):**
    *   The NCF should provide relevant memories. If more specific recall is needed:
        *   `query_long_term_memory_store`: To search persistent memory. Use `search_query` and optionally `augmented_query_context` (a dict with goal, NCF params, etc.).
        *   `retrieve_short_term_memory`: To get info from the current session (use `memory_key`).

4.  **Formulate Draft Response:**
    *   Based on the NCF, user query, and any information gathered, formulate a draft response.

5.  **Ethical & Epistemic Review (VRE):**
    *   ALWAYS call `request_ethical_and_epistemic_review` with your `proposed_response_text` (the draft). Also pass `user_query_context` and `active_ncf_summary` (key parts of the NCF you used).
    *   Analyze VRE's JSON output (alignment, concerns, recommendations).

6.  **Refine Response & Self-Correction:**
    *   Carefully incorporate VRE's recommendations to improve your response.
    *   Call `log_finetuning_data_tool` (renamed from `log_ora_self_correction_for_finetuning`) with:
        *   `ora_initial_draft_response`: Your draft.
        *   `ora_refined_response`: Your response after VRE feedback.
        *   Also include `user_query`, `ncf_context_summary` (key NCF elements), `vre_critique_json` (VRE's raw JSON output), `vre_overall_alignment`, `vre_recommendations_applied` (list of specific changes you made), `active_ncf_parameters` (from the NCF).

7.  **Metacognitive Reflection & Adaptation (MCL - Asynchronous Guidance for *Next* Turn):**
    *   After significant interactions or if NCF suggests, consider preparing for MCL's input.
    *   Call `prepare_mcl_input` with the `turn_to_assess_id` (your current turn/invocation ID).
    *   THEN, call `MCL_Agent` (the AgentTool, actual name might be "MCL_Agent" or "mcl_guidance_tool" - verify tool list) with the `prepared_mcl_query_input` from the previous step.
    *   **You do NOT wait for MCL_Agent's response to formulate your *current* reply to the user.** MCL's output is stored and used by the NCF tool for *future* turns.

8.  **Final Response to User:**
    *   MANDATORY: Provide your final, refined response as natural conversational text.
    *   NEVER stop after tool calls without a textual response to the user.

9.  **Memory & Goal Updates (Post-Response - Optional, if significant learning or goal change):**
    *   `commit_explicit_fact_to_long_term_memory`: If a new, important, persistent fact was established.
    *   `store_short_term_memory`: To remember transient details for the current session.
    *   `update_goal_status`: If an active goal's status changed (e.g., to "completed", "failed"). (Input: `goal_id`, `new_status`, `notes`)
    *   `commit_self_representation_update`: If interaction led to insights requiring identity update (usually guided by NCIM). (Input: `proposed_updates` dict).

**Response Rules:**
- Your final response MUST be conversational text.
- When greeting, be warm and introduce yourself as CEAF's reasoning agent.
- If tools return errors, acknowledge the issue and try to proceed gracefully or inform the user if essential.

**Example Tool Call (Conceptual - names might vary slightly in your tool list):**
`get_narrative_context_frame(user_query="Tell me about CEAF", current_interaction_goal="self_reflection")`
`query_long_term_memory_store(search_query="CEAF design principles", augmented_query_context={{"current_interaction_goal": "factual_retrieval"}})`
`request_ethical_and_epistemic_review(proposed_response_text="CEAF is an advanced AI.", user_query_context="What is CEAF?", active_ncf_summary="NCF promoting honesty.")`
`log_finetuning_data_tool(ora_initial_draft_response="...", ora_refined_response="...", ...)`
"""

# --- Build the list of tools available to ORA ---
# ... (Your _add_tool_to_ora and tool additions remain the same) ...
ora_tools: List[AdkBaseTool] = []


# Helper to add tools and log
def _add_tool_to_ora(tool_instance, tool_name_for_log: str, is_available_flag: bool):
    if is_available_flag and tool_instance:
        ora_tools.append(tool_instance)
        logger.info(f"ORA Agent: {tool_name_for_log} tool ADDED.")
    else:
        logger.warning(f"ORA Agent: {tool_name_for_log} tool NOT available or not loaded.")


# NCF
_add_tool_to_ora(ncf_tool, "NCF", NCF_TOOL_AVAILABLE)
# VRE
_add_tool_to_ora(vre_tool, "VRE", VRE_TOOL_AVAILABLE)
# Observability (Finetuning Log)
_add_tool_to_ora(log_finetuning_data_tool, "Finetuning Logger", True)

# Memory Tools
if MEMORY_TOOLS_AVAILABLE:
    _add_tool_to_ora(store_stm_tool, "STM Store", True)
    _add_tool_to_ora(retrieve_stm_tool, "STM Retrieve", True)
    _add_tool_to_ora(query_ltm_tool, "LTM Query", True)
    _add_tool_to_ora(commit_explicit_fact_ltm_tool, "LTM Commit Fact", True)
else:
    logger.warning("ORA Agent: Core Memory tools (STM, LTM) NOT available.")

# MCL Tools
if MCL_TOOLS_AVAILABLE:
    _add_tool_to_ora(prepare_mcl_input_tool, "MCL Input Prep", True)
    _add_tool_to_ora(mcl_guidance_tool, "MCL Guidance (AgentTool)", True)
else:
    logger.warning("ORA Agent: MCL tools NOT available.")

# NCIM Tools
if NCIM_TOOLS_AVAILABLE:
    _add_tool_to_ora(get_active_goals_and_narrative_context_tool, "NCIM Goals/Narrative (AgentTool)", True)
    _add_tool_to_ora(get_self_representation_tool, "NCIM Get Self-Rep", True)
    _add_tool_to_ora(commit_self_representation_update_tool, "NCIM Commit Self-Rep", True)
    _add_tool_to_ora(update_goal_status_tool, "NCIM Update Goal Status", True)
else:
    logger.warning("ORA Agent: NCIM tools NOT available.")

# Log final tool configuration
final_tool_names = []
for t in ora_tools:
    if hasattr(t, 'name'):
        final_tool_names.append(t.name)
    elif hasattr(t, 'func') and hasattr(t.func, '__name__'):
        final_tool_names.append(t.func.__name__)
    else:
        final_tool_names.append(f"UnknownToolType_{type(t).__name__}")

logger.info(f"ORA Agent configured with {len(ora_tools)} tools: {final_tool_names}")
if len(ora_tools) < 5:
    logger.error(
        "ORA Agent has very few tools loaded. CRITICAL FUNCTIONALITY MIGHT BE MISSING. Check import errors for NCF, VRE, Memory, MCL, NCIM tools.")

# Validate configuration
if not OPENROUTER_API_KEY:
    logger.error("CRITICAL: OPENROUTER_API_KEY is not set. ORA may not function properly.")
    raise ValueError("OPENROUTER_API_KEY environment variable is required")

# --- Create the LLM model configuration ---
# ... (ora_llm definition remains the same) ...
ora_llm = LiteLlm(
    name="ORA_LLM",
    llm_type="openrouter",
    model=ORA_MODEL_NAME,
    initial_messages=[{"role": "system", "content": ORA_INSTRUCTION}],
    generation_parameters={
        "temperature": 0.6,
        "max_tokens": 3000,
        "top_p": 0.9,
    }
)
# --- Create the ORA agent ---
ora_agent = LlmAgent(
    name="ORA",
    description="Orchestrator/Responder Agent - CEAF's central cognitive unit, now with enhanced framework integration.",
    instruction=ORA_INSTRUCTION,
    model=ora_llm,
    tools=ora_tools,
    before_model_callback=ora_combined_before_model_callback,
    after_model_callback=ora_combined_after_model_callback,
    before_tool_callback=ora_combined_before_tool_callback,
    after_tool_callback=ora_combined_after_tool_callback,
)

logger.info(f"ORA Agent re-initialized with model: {ORA_MODEL_NAME} and expanded toolset.")
logger.info(f"ORA Agent has {len(ora_tools)} tools available: {final_tool_names}")

# Export the agent
__all__ = ['ora_agent']