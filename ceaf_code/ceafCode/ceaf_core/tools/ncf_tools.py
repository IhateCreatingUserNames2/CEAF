# ceaf_core/tools/ncf_tools.py

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List

from google.adk.tools import FunctionTool, ToolContext
from google.adk.tools.agent_tool import AgentTool  # For type checking

# NCF engine components
try:
    from ..modules.ncf_engine.frames import (
        DEFAULT_INTERACTION_FRAME_TEMPLATE,
        CEAF_PHILOSOPHICAL_PREAMBLE,
        DEFAULT_NCF_PARAMETERS,
        format_mcl_advice_section,
        get_goal_directive_section,
        format_tool_usage_guidance
    )

    NCF_FRAMES_LOADED = True
except ImportError as e_ncf_frames:
    logging.error(
        f"NCF Tools: Critical import error for NCF frames: {e_ncf_frames}. NCF tool may not function correctly.",
        exc_info=True)
    NCF_FRAMES_LOADED = False
    # Fallback definitions (as in your provided code)
    DEFAULT_INTERACTION_FRAME_TEMPLATE = """
{philosophical_preamble}
**Current Interaction Context:**
- User Query: "{user_query}"
- Synthesized Relevant Memories/Knowledge:
{synthesized_memories_narrative}
- Agent Identity & Goal Context (from NCIM):
{ncim_context_summary}
**Operational Directives for This Turn (Parameters):**
- Conceptual Entropy: {conceptual_entropy}
- Narrative Depth: {narrative_depth}
- Philosophical Framing Intensity: {philosophical_framing_intensity}
- Emotional Tone Target: {emotional_tone_target}
- Self Disclosure Level: {self_disclosure_level}
{additional_mcl_advice_section}
{specific_goal_section}
{tool_usage_guidance}
"""
    CEAF_PHILOSOPHICAL_PREAMBLE = "CEAF System - Cognitive Ethical AI Framework"
    DEFAULT_NCF_PARAMETERS = {
        "conceptual_entropy": "balanced", "narrative_depth": "medium",
        "philosophical_framing_intensity": "medium", "emotional_tone_target": "neutral_positive",
        "self_disclosure_level": "moderate"
    }


    def format_mcl_advice_section(guidance_json: Optional[str]) -> str:
        if not guidance_json: return ""
        try:
            guidance = json.loads(guidance_json)
            advice = guidance.get("operational_advice_for_ora")
            if advice: return f"\n**MCL Guidance for This Turn:**\n- {advice}\n"
        except:
            pass
        return ""


    def get_goal_directive_section(goal_type: Optional[str]) -> str:
        return f"\n**Specific Goal:** {goal_type or 'Default general response.'}\n"


    def format_tool_usage_guidance(tools: Optional[List[str]]) -> str:
        if not tools: return "No specific tools recommended."
        return f"Consider using tools: {', '.join(tools)}"

# Memory and NCIM components
MEMORY_TOOLS_LOADED = False
NCIM_TOOLS_LOADED = False

try:
    from .memory_tools import query_long_term_memory_store  # This is an async function

    MEMORY_TOOLS_LOADED = True
except ImportError as e_memory:
    logging.error(f"NCF Tools: Failed to import 'query_long_term_memory_store' from memory_tools: {e_memory}",
                  exc_info=True)
    query_long_term_memory_store = None

try:
    from .ncim_tools import get_current_self_representation  # This is a sync function
    from .ncim_tools import \
        get_active_goals_and_narrative_context_tool  # This is an ADK Tool (AgentTool or FunctionTool)

    NCIM_TOOLS_LOADED = True
except ImportError as e_ncim:
    logging.error(f"NCF Tools: Failed to import NCIM functions/tools: {e_ncim}", exc_info=True)
    get_current_self_representation = None
    get_active_goals_and_narrative_context_tool = None

from .common_utils import create_successful_tool_response, create_error_tool_response, sanitize_text_for_logging

logger = logging.getLogger(__name__)


# --- Enhanced Async NCF Generation Tool ---
async def get_narrative_context_frame(
        user_query: str,
        tool_context: ToolContext,
        current_interaction_goal: Optional[str] = None,
        ora_available_tool_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    agent_name_for_log = getattr(tool_context, 'agent_name', "UnknownAgent")
    invocation_id_for_log = getattr(tool_context, 'invocation_id', "unknown_invocation")

    logger.info(
        f"NCF TOOL: Generating NCF for query: '{sanitize_text_for_logging(user_query, 70)}' "
        f"by agent '{agent_name_for_log}', invocation '{invocation_id_for_log}'"
    )

    if not NCF_FRAMES_LOADED:
        error_msg = "NCF generation failed due to missing core NCF frame components."
        logger.critical(error_msg)
        return create_error_tool_response(
            error_message=error_msg,
            details={
                "ncf_fallback_provided": True,
                "fallback_ncf": f"CRITICAL ERROR: {error_msg}. Defaulting to minimal response.",
                "applied_ncf_parameters": DEFAULT_NCF_PARAMETERS
            }
        )

    try:
        # --- Step 1: Retrieve MCL Guidance and NCF Parameters ---
        mcl_operational_advice = None
        final_ncf_params = DEFAULT_NCF_PARAMETERS.copy()
        mcl_last_guidance_str = tool_context.state.get("mcl_last_guidance")
        if mcl_last_guidance_str:
            try:
                mcl_guidance_data = json.loads(mcl_last_guidance_str)
                if "ncf_target_parameters" in mcl_guidance_data:
                    final_ncf_params.update(mcl_guidance_data["ncf_target_parameters"])
                    logger.info(
                        f"NCF TOOL: Applied MCL-derived NCF parameters: {mcl_guidance_data['ncf_target_parameters']}")
                mcl_operational_advice = mcl_guidance_data.get("operational_advice_for_ora")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"NCF TOOL: Failed to parse MCL guidance from state: {e}")

        # --- Step 2: Memory Context Retrieval ---
        synthesized_memories_str = "- No relevant memories were strongly activated for this query (or LTM search disabled/failed)."
        if MEMORY_TOOLS_LOADED and query_long_term_memory_store is not None:
            try:
                memory_search_augmented_context = {
                    "current_interaction_goal": current_interaction_goal,
                    "active_ncf_parameters": final_ncf_params,
                    "active_narrative_thread_id": tool_context.state.get("active_narrative_thread_id")
                }
                logger.info(
                    f"NCF TOOL: Calling query_long_term_memory_store with query: '{sanitize_text_for_logging(user_query, 70)}', augmented_context: {memory_search_augmented_context}")

                # CORRECTED: Directly await the async function query_long_term_memory_store
                ltm_search_result = await query_long_term_memory_store(
                    search_query=user_query,
                    tool_context=tool_context,
                    augmented_query_context=memory_search_augmented_context,
                    top_k=5,  # Adjust as needed
                    return_raw_memory_objects=False  # We want formatted snippets for NCF
                )
                logger.debug(
                    f"NCF TOOL: LTM search result for NCF: {str(ltm_search_result)[:500]}...")  # Log more of the result

                if ltm_search_result.get("status") == "success":
                    memories_data = ltm_search_result.get("data", {}).get("retrieved_memories", [])
                    if memories_data:
                        formatted_snippets = []
                        for mem_res in memories_data:  # mem_res is a dict
                            if mem_res.get("content_snippets"):
                                for snippet in mem_res.get("content_snippets", []):  # snippet is a string
                                    # Add more context to the snippet string
                                    formatted_snippets.append(
                                        f"(From LTM - Type: {mem_res.get('retrieved_memory_type', 'N/A')}, Score: {mem_res.get('retrieval_score', 0.0):.2f}, ID: {mem_res.get('memory_id', 'N/A')}): {snippet}"
                                    )

                        if formatted_snippets:
                            synthesized_memories_str = "\n".join(f"- {s}" for s in formatted_snippets)
                            logger.info(
                                f"NCF TOOL: Using {len(formatted_snippets)} memory snippets for NCF context: {synthesized_memories_str[:300]}...")
                        else:
                            logger.info(
                                "NCF TOOL: LTM search successful but no usable content snippets found in results.")
                    else:
                        logger.info("NCF TOOL: LTM search successful but 'retrieved_memories' data was empty.")
                else:
                    error_detail = ltm_search_result.get("error_message", "Unknown error from LTM search.")
                    logger.warning(
                        f"NCF TOOL: LTM search was not successful. Status: {ltm_search_result.get('status')}, Error: {error_detail}")

            except Exception as e_mem:
                logger.error(f"NCF TOOL: Error during memory retrieval for NCF: {e_mem}", exc_info=True)
                synthesized_memories_str = f"- Error retrieving memories for context: {str(e_mem)[:100]}"
        else:
            logger.warning(
                "NCF TOOL: Memory function 'query_long_term_memory_store' not available. Skipping LTM search for NCF.")

        # --- Step 3: NCIM Context Summary ---
        ncim_context_summary_str = "- NCIM context (self & goals): Default state assumed or NCIM tools not fully available."
        self_representation_str = "  - Self-Representation: Default (Core Values: Focus on coherence, humility, learning)."
        active_goals_str = "  - Active Goals: Prioritize current user query."

        if NCIM_TOOLS_LOADED:
            try:
                # Get Self-Representation
                if get_current_self_representation is not None:
                    logger.info("NCF TOOL: Retrieving self-representation from NCIM function...")
                    self_repr_result = await asyncio.to_thread(  # get_current_self_representation is sync
                        get_current_self_representation,
                        tool_context=tool_context
                    )
                    if self_repr_result.get("status") == "success":
                        self_data = self_repr_result.get("data", {}).get("self_representation", {})
                        core_values = self_data.get('core_values_summary', 'Default core values.')
                        capabilities_preview = ", ".join(self_data.get('perceived_capabilities', [])[:2]) + (
                            "..." if len(self_data.get('perceived_capabilities', [])) > 2 else "")
                        limitations_preview = ", ".join(self_data.get('known_limitations', [])[:2]) + (
                            "..." if len(self_data.get('known_limitations', [])) > 2 else "")
                        self_representation_str = (
                            f"  - Self-Representation: Values: '{sanitize_text_for_logging(core_values, 70)}'. "
                            f"Capabilities preview: '{capabilities_preview}'. Limitations preview: '{limitations_preview}'."
                        )
                else:
                    logger.warning("NCF TOOL: NCIM function 'get_current_self_representation' not loaded.")

                # Get Goals Assessment (via AgentTool or FunctionTool)
                if get_active_goals_and_narrative_context_tool is not None:
                    if isinstance(get_active_goals_and_narrative_context_tool, AgentTool):
                        active_goals_str = "  - Active Goals: (NCIM Agent guidance typically obtained by ORA via separate tool call or state)."
                        logger.warning("NCF TOOL: 'get_active_goals_and_narrative_context_tool' is an AgentTool. "
                                       "Direct call from NCF tool is not standard. ORA should call it or NCIM should update state.")
                        ncim_goals_summary_from_state = tool_context.state.get("ncim_last_assessment", {}).get(
                            "summary_for_ncf")
                        if ncim_goals_summary_from_state:
                            active_goals_str = f"  - Active Goals & Context (from state): {sanitize_text_for_logging(ncim_goals_summary_from_state, 150)}"
                    elif isinstance(get_active_goals_and_narrative_context_tool, FunctionTool):
                        logger.info("NCF TOOL: Retrieving active goals from NCIM FunctionTool...")
                        ncim_goals_result = await asyncio.to_thread(  # Assuming .func is sync
                            get_active_goals_and_narrative_context_tool.func,
                            current_query_summary=user_query[:100],
                            tool_context=tool_context
                        )
                        if ncim_goals_result.get("status") == "success":
                            summary = ncim_goals_result.get("data", {}).get("summary_for_ncf",
                                                                            "NCIM provided no specific summary.")
                            active_goals_str = f"  - Active Goals & Context: {sanitize_text_for_logging(summary, 150)}"
                    else:
                        logger.warning(
                            f"NCF TOOL: 'get_active_goals_and_narrative_context_tool' is of unexpected type: {type(get_active_goals_and_narrative_context_tool)}")
                else:
                    logger.warning("NCF TOOL: NCIM tool 'get_active_goals_and_narrative_context_tool' not loaded.")

                ncim_context_summary_str = f"{self_representation_str}\n{active_goals_str}"

            except Exception as e_ncim:
                logger.error(f"NCF TOOL: Error during NCIM context retrieval: {e_ncim}", exc_info=True)
                ncim_context_summary_str = f"- Error retrieving NCIM context: {str(e_ncim)[:100]}"
        else:
            logger.warning("NCF TOOL: NCIM components not loaded, using default NCIM context.")

        # --- Step 4: Assemble Final NCF ---
        philosophical_preamble_str = CEAF_PHILOSOPHICAL_PREAMBLE.strip()
        mcl_advice_formatted_str = format_mcl_advice_section(
            json.dumps({"operational_advice_for_ora": mcl_operational_advice}) if mcl_operational_advice else None)
        goal_section_str = get_goal_directive_section(current_interaction_goal)
        tool_guidance_str = format_tool_usage_guidance(ora_available_tool_names or [])

        complete_params_for_template = DEFAULT_NCF_PARAMETERS.copy()
        complete_params_for_template.update(final_ncf_params)

        final_ncf_string = DEFAULT_INTERACTION_FRAME_TEMPLATE.format(
            philosophical_preamble=philosophical_preamble_str,
            user_query=user_query,
            synthesized_memories_narrative=synthesized_memories_str,
            ncim_context_summary=ncim_context_summary_str,
            additional_mcl_advice_section=mcl_advice_formatted_str,
            specific_goal_section=goal_section_str,
            tool_usage_guidance=tool_guidance_str,
            **complete_params_for_template
        ).strip()

        # Log the final NCF being returned for debugging
        logger.debug(f"NCF TOOL --- FINAL NCF STRING --- \n{final_ncf_string}\n--- END NCF STRING ---")

        logger.info(
            f"NCF TOOL: Successfully generated NCF for agent '{agent_name_for_log}', "
            f"turn '{invocation_id_for_log}'. Length: {len(final_ncf_string)} chars."
        )

        # Store NCF params used for this turn in session state for MCL analysis later
        tool_context.state[f"ora_turn_ncf_params:{invocation_id_for_log}"] = final_ncf_params
        tool_context.state[f"ora_turn_user_query:{invocation_id_for_log}"] = user_query

        return create_successful_tool_response(
            data={
                "ncf": final_ncf_string,
                "applied_ncf_parameters": final_ncf_params,
                "memory_context_included": MEMORY_TOOLS_LOADED and query_long_term_memory_store is not None and "Error" not in synthesized_memories_str and "No relevant memories" not in synthesized_memories_str,
                "ncim_context_included": NCIM_TOOLS_LOADED and "Error" not in ncim_context_summary_str and "Default state assumed" not in ncim_context_summary_str
            },
            message="Narrative Context Frame generated successfully."
        )

    except Exception as e:
        logger.error(
            f"NCF TOOL: Error generating NCF for agent '{agent_name_for_log}', "
            f"turn '{invocation_id_for_log}': {e}",
            exc_info=True
        )
        fallback_ncf = f"""
{CEAF_PHILOSOPHICAL_PREAMBLE.strip()}
**System Alert: NCF Generation Error**
- An error occurred while generating the full Narrative Context Frame.
- User Query: "{user_query}"
- Error: {str(e)}
**Primary Task for ORA:**
Respond to the user's query directly and cautiously, relying on your general knowledge and core CEAF principles using default operational parameters.
        """.strip()
        return create_error_tool_response(
            error_message=f"Failed to generate Narrative Context Frame: {str(e)}",
            details={
                "ncf_fallback_provided": True,
                "fallback_ncf": fallback_ncf,
                "applied_ncf_parameters": DEFAULT_NCF_PARAMETERS
            }
        )


# Create the FunctionTool instance
ncf_tool = FunctionTool(
    func=get_narrative_context_frame
)