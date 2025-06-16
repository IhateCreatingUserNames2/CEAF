# MCL Tools - Refactored for Google ADK
# ceaf_project/ceaf_core/tools/mcl_tools.py

import logging
import json
import asyncio
from typing import Dict, Any, Optional, List

# Correct imports for Google ADK
from google.adk.tools import FunctionTool, LongRunningFunctionTool, ToolContext
from google.adk.tools.agent_tool import AgentTool

logger = logging.getLogger(__name__)

# Assuming mcl_agent and self_state_analyzer are accessible
try:
    from ..agents.mcl_agent import mcl_agent  # The actual MCL_Agent instance
    from ..modules.mcl_engine.self_state_analyzer import (
        analyze_ora_turn_observations,
        ORAStateAnalysis,
        MCL_OBSERVATIONS_LIST_KEY
    )

    MCL_MODULES_LOADED = True
except ImportError as e:
    logger.error(
        f"MCL Tools: Critical import error: {e}. MCL tools may not function correctly.",
        exc_info=True)
    MCL_MODULES_LOADED = False


    class ORAStateAnalysis:  # type: ignore
        def __init__(self):
            self.turn_id = "dummy"
            self.eoc_assessment = "dummy"
            self.eoc_confidence = 0.0
            self.summary_notes = []
            self.flags = {}
            self.tool_calls_attempted = 0
            self.tool_calls_succeeded = 0
            self.tool_errors = []
            self.last_llm_finish_reason = None
            self.function_calls_in_last_response = []
            # Quantitative scores
            self.novelty_score = None  # Changed to None
            self.coherence_score = None  # Changed to None
            self.grounding_score = None  # Changed to None
            self.eoc_quantitative_score = None  # Changed to None
            self.raw_observations_count = 0  # Added

        def to_dict(self): return self.__dict__


    def analyze_ora_turn_observations(turn_id, turn_observations, **kwargs) -> ORAStateAnalysis:  # type: ignore
        return ORAStateAnalysis()


    mcl_agent = None  # type: ignore
    MCL_OBSERVATIONS_LIST_KEY = "mcl:dummy_obs_log"

from .common_utils import (
    create_successful_tool_response,
    create_error_tool_response,
    sanitize_text_for_logging
)


def _prepare_mcl_query_from_analysis(
        analysis: ORAStateAnalysis,  # type: ignore
        current_ncf_params_ora: Optional[Dict[str, Any]],  # Renamed for clarity
        ora_last_user_query: Optional[str] = None,  # Added for more context to MCL
        additional_context: Optional[str] = None
) -> str:
    """
    Prepares a structured query for the MCL Agent based on ORA state analysis.

    Args:
        analysis: ORAStateAnalysis object containing turn assessment
        current_ncf_params_ora: NCF parameters that were active during the assessed turn
        ora_last_user_query: The user query that ORA was responding to in the assessed turn
        additional_context: Any additional context for MCL assessment

    Returns:
        Formatted query string for MCL Agent
    """
    tool_errors_summary = []
    if hasattr(analysis, 'tool_errors') and analysis.tool_errors:  # type: ignore
        tool_errors_summary = [
            f"- Tool: {err.get('tool_name')}, Args: {sanitize_text_for_logging(str(err.get('args')), 50)}, Summary: {sanitize_text_for_logging(err.get('summary'), 70)}"
            for err in analysis.tool_errors[:3]  # type: ignore
        ]
        if len(analysis.tool_errors) > 3:  # type: ignore
            tool_errors_summary.append(f"...and {len(analysis.tool_errors) - 3} more errors.")  # type: ignore

    query_parts = [
        "Perform a metacognitive assessment of ORA's recent performance and provide guidance by determining optimal NCF parameters for the next turn.",
        f"Assessment Input for ORA's Turn ID '{getattr(analysis, 'turn_id', 'N/A')}':",
        # Analyzer's assessment
        f"- eoc_assessment_analyzer: {getattr(analysis, 'eoc_assessment', 'N/A')}",
        f"- eoc_confidence_analyzer: {getattr(analysis, 'eoc_confidence', 0.0):.2f}",
        # Quantitative Scores from Analyzer
        f"- quantitative_eoc_scores:",
        f"  - novelty_score: {getattr(analysis, 'novelty_score', 'N/A')}",
        f"  - coherence_score: {getattr(analysis, 'coherence_score', 'N/A')}",
        f"  - grounding_score: {getattr(analysis, 'grounding_score', 'N/A')}",
        f"  - combined_eoc_score: {getattr(analysis, 'eoc_quantitative_score', 'N/A')}",
        f"- analyzer_flags: {json.dumps(getattr(analysis, 'flags', {}))}",
        f"- analyzer_key_notes: {'; '.join(getattr(analysis, 'summary_notes', [])) if getattr(analysis, 'summary_notes', []) else 'None'}",
        # ORA's operational details for the turn
        f"- tool_usage_summary: Attempted {getattr(analysis, 'tool_calls_attempted', 0)}, Succeeded {getattr(analysis, 'tool_calls_succeeded', 0)}",
        f"- tool_errors_summary: {'; '.join(tool_errors_summary) if tool_errors_summary else 'None'}",
        f"- last_llm_finish_reason_ora: {getattr(analysis, 'last_llm_finish_reason', 'N/A') or 'N/A'}",
        f"- function_calls_in_ora_last_response: {json.dumps(getattr(analysis, 'function_calls_in_last_response', []))}",
        # Context ORA was operating under
        f"- current_ncf_parameters_ora: {json.dumps(current_ncf_params_ora or {})}",
        f"- ora_last_user_query_for_turn: {sanitize_text_for_logging(ora_last_user_query, 150) if ora_last_user_query else 'N/A'}",
    ]
    if additional_context:
        query_parts.append(f"- additional_mcl_context: {additional_context}")

    query_parts.append(
        "\nBased on ALL this input, provide your structured JSON guidance (ora_state_assessment_mcl, ncf_target_parameters, operational_advice_for_ora, reasoning_for_guidance) as per your primary instructions."
    )
    return "\n".join(query_parts)


# Function for preparing MCL assessment input - converted to a simple function for ADK
def prepare_input_for_mcl_assessment( # This is the Python function
        turn_to_assess_id: str,
        tool_context: ToolContext, # Added tool_context
        additional_mcl_context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Gathers ORA's turn observations for a specific turn, analyzes them using the
    SelfStateAnalyzer, and formats a structured query input string for the MCL_Agent.
    This helps MCL_Agent assess ORA's performance and determine optimal NCF parameters.
    This tool is typically called by ORA when it needs metacognitive guidance or after
    a significant interaction turn.
    (THIS DOCSTRING WILL BE USED BY FunctionTool)
    """
    logger.info(f"MCL Tool (prepare_input_for_mcl_assessment): Request for turn_id '{turn_to_assess_id}'.")
    if not MCL_MODULES_LOADED:
        return create_error_tool_response("MCL system components (analyzer or agent) not available.")

    # ADK session state access
    try:
        # In ADK, state is directly on tool_context.state (which is a dict-like State object)
        # The MCL_OBSERVATIONS_LIST_KEY holds a list of dicts.
        all_observations_from_state = tool_context.state.get(MCL_OBSERVATIONS_LIST_KEY, []) # type: ignore
        relevant_turn_observations = [obs for obs in all_observations_from_state if
                                      obs.get("turn_id") == turn_to_assess_id]

        if not relevant_turn_observations:
            logger.warning(
                f"No observation data found for turn {turn_to_assess_id} in tool_context.state under key '{MCL_OBSERVATIONS_LIST_KEY}'.")
            return create_error_tool_response(f"No observation data for turn {turn_to_assess_id} found in current session state.")

        # Retrieve context data - adapt these based on your actual data storage pattern in session state
        current_ncf_params_for_assessed_turn = tool_context.state.get(f"ora_turn_ncf_params:{turn_to_assess_id}", {})
        user_query_for_assessed_turn = tool_context.state.get(f"ora_turn_user_query:{turn_to_assess_id}")
        ora_response_text_for_assessed_turn = tool_context.state.get(f"ora_turn_final_response_text:{turn_to_assess_id}")

        # Analyze the turn
        ora_analysis: ORAStateAnalysis = analyze_ora_turn_observations(
            turn_id=turn_to_assess_id,
            turn_observations=relevant_turn_observations,
            ora_response_text=ora_response_text_for_assessed_turn,
            user_query_text=user_query_for_assessed_turn,
            ncf_text_summary=json.dumps(current_ncf_params_for_assessed_turn), # Pass NCF params as summary
            current_ncf_params=current_ncf_params_for_assessed_turn # Also pass them structured
        )

        mcl_query = _prepare_mcl_query_from_analysis(
            ora_analysis,
            current_ncf_params_ora=current_ncf_params_for_assessed_turn,
            ora_last_user_query=user_query_for_assessed_turn,
            additional_context=additional_mcl_context
        )

        return create_successful_tool_response(
            data={"ora_turn_analysis_summary": ora_analysis.to_dict(), "prepared_mcl_query_input": mcl_query},
            message="Input for MCL_Agent prepared."
        )

    except Exception as e:
        logger.error(f"MCL Tool: Error preparing input for MCL assessment for turn {turn_to_assess_id}: {e}", exc_info=True)
        return create_error_tool_response(f"Error in ORA state analysis: {e}", details=str(e))


# Long-running version of the preparation function for complex analysis
def prepare_input_for_mcl_assessment_generator(  # This is the Python function
        turn_to_assess_id: str,
        tool_context: ToolContext,  # Added tool_context
        additional_mcl_context: Optional[str] = None
):
    """
    Long-running version of prepare_input_for_mcl_assessment. Gathers ORA's turn
    observations, analyzes them, and formats a query input for the MCL_Agent.
    Yields progress updates and returns the final analysis and query.
    (THIS DOCSTRING WILL BE USED BY LongRunningFunctionTool)
    """
    yield {"status": "pending", "message": f"Starting analysis for turn {turn_to_assess_id}..."}

    logger.info(f"MCL Tool (prepare_input_for_mcl_assessment_generator): Request for turn_id '{turn_to_assess_id}'.")

    if not MCL_MODULES_LOADED:
        yield create_error_tool_response("MCL system components (analyzer or agent) not available.")
        return

    yield {"status": "pending", "message": "Gathering observation data..."}

    try:
        all_observations_from_state = tool_context.state.get(MCL_OBSERVATIONS_LIST_KEY, [])  # type: ignore
        relevant_turn_observations = [obs for obs in all_observations_from_state if
                                      obs.get("turn_id") == turn_to_assess_id]

        if not relevant_turn_observations:
            yield create_error_tool_response(
                f"No observation data for turn {turn_to_assess_id} found in current session state.")
            return

        yield {"status": "pending", "message": "Analyzing turn observations...", "progress": "50%"}

        current_ncf_params_for_assessed_turn = tool_context.state.get(f"ora_turn_ncf_params:{turn_to_assess_id}", {})
        user_query_for_assessed_turn = tool_context.state.get(f"ora_turn_user_query:{turn_to_assess_id}")
        ora_response_text_for_assessed_turn = tool_context.state.get(
            f"ora_turn_final_response_text:{turn_to_assess_id}")

        ora_analysis: ORAStateAnalysis = analyze_ora_turn_observations(
            turn_id=turn_to_assess_id,
            turn_observations=relevant_turn_observations,
            ora_response_text=ora_response_text_for_assessed_turn,
            user_query_text=user_query_for_assessed_turn,
            ncf_text_summary=json.dumps(current_ncf_params_for_assessed_turn),
            current_ncf_params=current_ncf_params_for_assessed_turn
        )

        yield {"status": "pending", "message": "Preparing MCL query...", "progress": "75%"}

        mcl_query = _prepare_mcl_query_from_analysis(
            ora_analysis,
            current_ncf_params_ora=current_ncf_params_for_assessed_turn,
            ora_last_user_query=user_query_for_assessed_turn,
            additional_context=additional_mcl_context
        )

        yield {"status": "pending", "message": "Finalizing analysis...", "progress": "90%"}

        yield create_successful_tool_response(
            data={"ora_turn_analysis_summary": ora_analysis.to_dict(), "prepared_mcl_query_input": mcl_query},
            message="Input for MCL_Agent prepared."
        )
        return  # Explicit return for generator

    except Exception as e:
        logger.error(
            f"MCL Tool: Error in generator preparing input for MCL assessment for turn {turn_to_assess_id}: {e}",
            exc_info=True)
        yield create_error_tool_response(f"Error in ORA state analysis: {e}", details=str(e))
        return


# Create the tools for ADK
prepare_mcl_input_tool = FunctionTool(
    func=prepare_input_for_mcl_assessment
)

# Long-running version
prepare_mcl_input_long_running_tool = LongRunningFunctionTool(
    func=prepare_input_for_mcl_assessment_generator
)

# MCL Agent Tool - using AgentTool for agent-as-a-tool pattern
mcl_guidance_tool = None
if mcl_agent and MCL_MODULES_LOADED:
    # MODIFICATION START
    mcl_guidance_tool = AgentTool(
        agent=mcl_agent

    )
    # MODIFICATION END
    logger.info(f"MCL Tool: Defined MCL guidance tool (AgentTool wrapping MCL_Agent). Effective tool name will be '{getattr(mcl_guidance_tool, 'name', mcl_agent.name)}'.")
else:
    logger.error(
        "MCL Tool: 'mcl_agent' or dependent modules not loaded. "
        "MCL guidance AgentTool cannot be defined."
    )

# Collect all available tools
ceaf_mcl_tools = []

# Add the basic function tool
if prepare_mcl_input_tool:
    ceaf_mcl_tools.append(prepare_mcl_input_tool)

# Add the long-running version
if prepare_mcl_input_long_running_tool:
    ceaf_mcl_tools.append(prepare_mcl_input_long_running_tool)

# Add the MCL agent tool
if mcl_guidance_tool:
    ceaf_mcl_tools.append(mcl_guidance_tool)


# Helper function to integrate with your agent setup
def get_mcl_tools_for_agent() -> List:
    """
    Returns a list of MCL tools ready to be added to an ADK Agent.

    Usage:
        from .mcl_tools import get_mcl_tools_for_agent

        agent = Agent(
            model="gemini-2.0-flash",
            name="ora_agent",
            instruction="Your agent instructions...",
            tools=get_mcl_tools_for_agent() + other_tools
        )
    """
    return ceaf_mcl_tools

