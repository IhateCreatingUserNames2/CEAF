import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

import litellm
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from ...services.persistent_log_service import PersistentLogService

# --- Environment Setup ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    dotenv_path = os.path.join(project_root, ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
except Exception:
    pass

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AuraReflector")

# --- Configuration ---
AURA_INSIGHT_GENERATION_MODEL = os.getenv("AURA_INSIGHT_MODEL", "openrouter/openai/gpt-4.1")
AURA_STRATEGY_GENERATION_MODEL = os.getenv("AURA_STRATEGY_MODEL", "openrouter/openai/gpt-4.1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# --- Pydantic Models ---
class PerformanceInsight(BaseModel):
    insight_id: str = Field(default_factory=lambda: f"insight_{uuid.uuid4().hex}")
    timestamp: float = Field(default_factory=time.time)
    description: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_refinement_type: Literal[
        "ncf_param_heuristic", "ncf_template_update", "ora_instruction_tweak",
        "mbs_retrieval_bias", "vre_principle_clarification", "mcl_eoc_threshold_adjustment"
    ]
    refinement_details: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)
    potential_impact_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class RefinementStrategy(BaseModel):
    strategy_id: str = Field(default_factory=lambda: f"strat_{uuid.uuid4().hex}")
    timestamp: float = Field(default_factory=time.time)
    description: str
    associated_insight_ids: List[str]
    strategy_type: Literal[
        "ncf_param_heuristic", "ncf_template_update", "ora_instruction_tweak",
        "mbs_retrieval_bias", "vre_principle_clarification", "mcl_eoc_threshold_adjustment",
        "data_collection_for_finetuning", "documentation_update"
    ]
    strategy_payload: Dict[str, Any]
    priority: int = Field(5, ge=1, le=10)
    estimated_effort: Optional[str] = None
    status: Literal["proposed", "approved", "implemented", "rejected", "deferred"] = "proposed"


# --- PersistentLogService Instance ---
try:
    persistent_log_service_instance = PersistentLogService()
    LOG_SERVICE_AVAILABLE = True
except Exception as e_log_service_init:
    logger.error(
        f"AuraReflector: Failed to initialize PersistentLogService: {e_log_service_init}. Aura Reflector analysis will fail.",
        exc_info=True)
    persistent_log_service_instance = None
    LOG_SERVICE_AVAILABLE = False


# --- Core Logic Functions ---

async def analyze_historical_performance(
        log_store: Optional[PersistentLogService] = None,
        time_window_days: int = 30,
        min_events_for_llm_sample: int = 100,
        min_total_events_for_analysis: int = 200
) -> List[PerformanceInsight]:
    """
    Retrieves historical data from PersistentLogService and uses an LLM to identify performance insights.
    """
    if not LOG_SERVICE_AVAILABLE and not log_store:
        logger.error(
            "AuraReflector (analyze_historical_performance): PersistentLogService is not available. Cannot proceed.")
        return []

    active_log_store = log_store if log_store else persistent_log_service_instance
    if not active_log_store:
        logger.error("AuraReflector: No active log store provided or initialized for analysis.")
        return []

    if not OPENROUTER_API_KEY:
        logger.error(
            "AuraReflector (analyze_historical_performance): OPENROUTER_API_KEY not set. Cannot proceed with LLM call.")
        return []

    logger.info(f"AuraReflector: Starting analysis of historical performance data (last {time_window_days} days).")

    start_ts = time.time() - (time_window_days * 86400)

    all_relevant_events_in_window = active_log_store.query_logs(
        time_window_start_ts=start_ts,
        limit=min_total_events_for_analysis * 2,
        order_by_timestamp_desc=True
    )

    if len(all_relevant_events_in_window) < min_total_events_for_analysis:
        logger.warning(
            f"AuraReflector: Insufficient historical data ({len(all_relevant_events_in_window)} events found, need {min_total_events_for_analysis}) in the last {time_window_days} days for meaningful analysis.")
        return []

    events_for_llm_sample_raw = all_relevant_events_in_window[:min_events_for_llm_sample]

    formatted_events_for_llm = []
    total_chars_for_llm = 0
    MAX_CHARS_FOR_LLM_EVENTS_PART = 15000

    for entry in reversed(events_for_llm_sample_raw):
        payload_str = json.dumps(entry.get("data_payload", {}), default=str)
        if len(payload_str) > 500:
            payload_str = payload_str[:497] + "..."

        event_summary = {
            "event_id": entry.get("id"),
            "ts_relative_days_ago": round((time.time() - entry.get("timestamp", 0)) / 86400, 1),
            "type": entry.get("event_type"),
            "src_agent": entry.get("source_agent"),
            "session": entry.get("session_id", "")[-8:],
            "turn": entry.get("turn_id", "")[-8:],
            "payload_summary": payload_str,
            "tags": entry.get("tags", [])
        }
        event_str_for_llm = json.dumps(event_summary, default=str)
        if total_chars_for_llm + len(event_str_for_llm) > MAX_CHARS_FOR_LLM_EVENTS_PART:
            logger.info(
                f"AuraReflector: Reached max char limit for LLM event sample. Sending {len(formatted_events_for_llm)} events.")
            break
        formatted_events_for_llm.append(event_summary)
        total_chars_for_llm += len(event_str_for_llm)

    if not formatted_events_for_llm:
        logger.warning("AuraReflector: No historical data events formatted for LLM after sampling. Cannot proceed.")
        return []

    data_summary_for_llm = json.dumps(formatted_events_for_llm, indent=2, default=str)

    prompt = f"""You are an AI Performance Analyst for the CEAF (Coherent Emergence Agent Framework).
Your task is to analyze the provided historical log events from CEAF's operations and identify significant patterns, anomalies, or areas for improvement.

Focus on aspects related to:
- Response quality (coherence, relevance, helpfulness - infer from events like 'MCL_ANALYSIS', 'VRE_ASSESSMENT', 'ORA_RESPONSE', user feedback if logged).
- Operational efficiency (e.g., excessive tool use indicated by multiple 'ORA_TOOL_CALL_ATTEMPTED' in a turn, repeated errors in tool responses).
- Alignment with CEAF principles (e.g., epistemic humility, narrative sanity - might be inferred from VRE assessments or specific ORA responses).
- Effectiveness of NCF parameters (look for correlations between 'NCF_PARAMETERS_USED' events and subsequent performance like EoC scores from 'MCL_ANALYSIS' or VRE flags).
- Effectiveness of MCL interventions (look for changes in EoC scores after 'MCL_AGENT_GUIDANCE_GENERATED' events).

Historical Log Events Sample (presented chronologically, 'ts_relative_days_ago' indicates days ago from now):
Each log event object has fields: 'event_id', 'ts_relative_days_ago', 'type' (event_type), 'src_agent' (source_agent), 'session' (session_id suffix), 'turn' (turn_id suffix), 'payload_summary' (a JSON string summary of specific data for that event type), and 'tags'.

{data_summary_for_llm}

Based on this data, generate a JSON response with a root key "insights" containing a list of performance insight objects.
For each distinct insight, provide the following information in a JSON object:

"description": A clear, concise description of the insight. What pattern did you observe? What is the implication?
"supporting_evidence_ids": A list of 'event_id's, 'turn_id's, or 'session_id's (up to 5 relevant IDs as strings) from the provided log data that directly support this insight.
"confidence": Your confidence (0.0 to 1.0) that this insight is valid and significant based only on the provided data sample.
"suggested_refinement_type": Choose ONE from: "ncf_param_heuristic", "ncf_template_update", "ora_instruction_tweak", "mbs_retrieval_bias", "vre_principle_clarification", "mcl_eoc_threshold_adjustment".
"refinement_details": A dictionary with specific details for the suggested refinement. For example:
  For "ncf_param_heuristic": {{"parameter_name": "conceptual_entropy", "observed_issue": "When event_type 'MCL_ANALYSIS' payload.eoc_assessment is 'chaotic' and NCF params show high conceptual_entropy for problem-solving goals, ORA responses are often off-topic.", "suggested_change": "If goal is 'problem_solving' and conceptual_entropy is high, advise reducing it for subsequent turns if EoC becomes chaotic."}}
  For "ora_instruction_tweak": {{"target_section": "Core Operational Protocol - Step 5 (VRE Review)", "observed_issue": "Multiple 'VRE_ASSESSMENT' events show 'minor_concerns' on epistemic_honesty but ORA responses don't always reflect changes.", "suggestion": "Strengthen ORA's instruction to explicitly state how to incorporate VRE feedback on epistemic honesty."}}
"tags": A list of relevant keywords (e.g., ["coherence", "tool_use", "ncf_effectiveness", "eoc_score_pattern", "vre_feedback_loop"]).
"potential_impact_score": Your estimate (0.0 to 1.0) of how much addressing this insight could improve CEAF performance.

If no significant insights are found from this data sample, return {{"insights": []}}.
Ensure your output is valid JSON.

Example insight structure:
{{
  "description": "ORA responses show low coherence (inferred from 'MCL_ANALYSIS' payload.coherence_score < 0.4) following turns where 'ORA_TOOL_CALL_ATTEMPTED' count exceeds 3, especially for user queries tagged 'complex_reasoning'.",
  "supporting_evidence_ids": ["turn_abc123", "event_789", "session_xyz456"],
  "confidence": 0.75,
  "suggested_refinement_type": "ora_instruction_tweak",
  "refinement_details": {{
    "target_section": "Tool Usage Guidance in NCF",
    "observed_issue": "Excessive tool use in complex reasoning leads to fragmented, low-coherence responses.",
    "suggestion": "Add guidance to ORA: 'For complex reasoning tasks, prioritize consolidating information from 1-2 key tools before attempting further tool calls to maintain coherence.'"
  }},
  "tags": ["coherence", "tool_use", "complex_reasoning", "mcl_analysis"],
  "potential_impact_score": 0.6
}}"""

    insights: List[PerformanceInsight] = []
    try:
        logger.info(
            f"AuraReflector: Sending insight generation request to LLM ({AURA_INSIGHT_GENERATION_MODEL}) with {len(formatted_events_for_llm)} summarized events.")

        response = await litellm.acompletion(
            model=AURA_INSIGHT_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=3500
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            llm_output_str = response.choices[0].message.content
            logger.debug(f"AuraReflector: LLM raw output for insights: {llm_output_str[:1000]}...")
            try:
                parsed_output = json.loads(llm_output_str)
                insight_data_list = parsed_output.get("insights", [])

                for item_data in insight_data_list:
                    try:
                        evidence_ids = item_data.get("supporting_evidence_ids", [])
                        if not isinstance(evidence_ids, list):
                            evidence_ids = [str(evidence_ids)]
                        else:
                            evidence_ids = [str(eid) for eid in evidence_ids]
                        item_data["supporting_evidence_ids"] = evidence_ids

                        insights.append(PerformanceInsight(**item_data))
                    except ValidationError as e_val:
                        logger.warning(
                            f"AuraReflector: Validation error for an insight item: {e_val}. Item: {item_data}")
                    except Exception as e_item:
                        logger.warning(f"AuraReflector: Error processing insight item: {e_item}. Item: {item_data}")
            except json.JSONDecodeError as e_json:
                logger.error(
                    f"AuraReflector: Failed to decode JSON from LLM insight response: {e_json}. Raw: {llm_output_str[:500]}")
        else:
            logger.warning("AuraReflector: LLM response for insights was empty or malformed.")

    except litellm.exceptions.APIConnectionError as e_conn:
        logger.error(f"AuraReflector: API Connection Error during LLM call for insight generation: {e_conn}",
                     exc_info=True)
    except litellm.exceptions.RateLimitError as e_rate:
        logger.error(f"AuraReflector: Rate Limit Error during LLM call: {e_rate}", exc_info=True)
    except litellm.exceptions.APIError as e_api:
        logger.error(f"AuraReflector: LiteLLM API Error during insight generation: {e_api}", exc_info=True)
    except Exception as e_llm:
        logger.error(f"AuraReflector: Unexpected error during LLM call for insight generation: {e_llm}", exc_info=True)

    logger.info(
        f"AuraReflector: Generated {len(insights)} performance insights from analyzing {len(formatted_events_for_llm)} log events.")
    return insights


async def generate_refinement_strategies(
        insights: List[PerformanceInsight]
) -> List[RefinementStrategy]:
    """
    Takes PerformanceInsight objects and generates actionable RefinementStrategy objects using an LLM.
    """
    if not OPENROUTER_API_KEY:
        logger.error("AuraReflector (generate_refinement_strategies): OPENROUTER_API_KEY not set. Cannot proceed.")
        return []

    if not insights:
        logger.info("AuraReflector: No insights provided to generate refinement strategies.")
        return []

    logger.info(f"AuraReflector: Generating refinement strategies for {len(insights)} insights.")

    insights_json = json.dumps([insight.model_dump() for insight in insights], indent=2, default=str)

    prompt = f"""You are an AI Strategy Developer for the CEAF (Coherent Emergence Agent Framework).
Your task is to take a list of performance insights and propose actionable refinement strategies.
For each strategy, ensure it directly addresses one or more insights.

Performance Insights:
{insights_json}

For each proposed strategy, provide the following information in a JSON object:
"description": A concise description of the strategy and what it aims to achieve.
"associated_insight_ids": A list of insight_ids that this strategy addresses.
"strategy_type": The type of strategy, must be one of: "ncf_param_heuristic", "ncf_template_update", "ora_instruction_tweak", "mbs_retrieval_bias", "vre_principle_clarification", "mcl_eoc_threshold_adjustment", "data_collection_for_finetuning", "documentation_update". This should generally match the suggested_refinement_type from the insight(s).
"strategy_payload": A dictionary containing the specific details or parameters of the strategy. Examples:
  For "ncf_param_heuristic": {{"target_param": "conceptual_entropy", "rules": [{{"condition": "goal == 'problem_solving'", "action": "set_max_value('medium')"}}]}}
  For "ora_instruction_tweak": {{"target_section": "Core Operational Protocol", "suggested_wording_change": "Replace 'Consider VRE review' with 'Mandatory VRE review for claims about external facts.'", "reasoning": "To improve factual grounding."}}
  For "ncf_template_update": {{"template_name": "DEFAULT_INTERACTION_FRAME_TEMPLATE", "section_to_modify": "Tool Usage Guidance", "change_description": "Add a reminder to check tool output for errors before proceeding."}}
  For "data_collection_for_finetuning": {{"focus_area": "ethical_dilemma_resolution", "example_prompt_response_pair_structure": "..." }}
"priority": An integer from 1 (highest) to 10 (lowest).
"estimated_effort": A string like "low", "medium", "high".

Generate a JSON response with a root key "strategies" containing a list of these strategy objects. If no strategies are warranted, return {{"strategies": []}}.
Ensure your output is valid JSON. For example:

{{
  "strategies": [
    {{
      "description": "Implement a heuristic to cap ORA's tool usage at 2 per turn when narrative_depth is 'deep', unless user query explicitly requires more tools.",
      "associated_insight_ids": ["insight_some_id_related_to_tool_use"],
      "strategy_type": "ncf_param_heuristic",
      "strategy_payload": {{
        "target_param": "implicit_tool_limit_based_on_narrative_depth",
        "rules": [
          {{"condition": "ncf.narrative_depth == 'deep'", "action": "set_tool_call_advice('Recommend max 2 tools')"}}
        ]
      }},
      "priority": 3,
      "estimated_effort": "medium"
    }}
  ]
}}"""

    strategies: List[RefinementStrategy] = []
    try:
        logger.info(f"AuraReflector: Sending strategy generation request to LLM ({AURA_STRATEGY_GENERATION_MODEL}).")
        response = await litellm.acompletion(
            model=AURA_STRATEGY_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=3000
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            llm_output_str = response.choices[0].message.content
            logger.debug(f"AuraReflector: LLM raw output for strategies: {llm_output_str[:1000]}...")
            try:
                parsed_output = json.loads(llm_output_str)
                strategy_data_list = parsed_output.get("strategies", [])
                for item_data in strategy_data_list:
                    try:
                        item_data.setdefault("status", "proposed")
                        strategies.append(RefinementStrategy(**item_data))
                    except ValidationError as e_val:
                        logger.warning(
                            f"AuraReflector: Validation error for a strategy item: {e_val}. Item: {item_data}")
            except json.JSONDecodeError as e_json:
                logger.error(
                    f"AuraReflector: Failed to decode JSON from LLM strategy response: {e_json}. Raw: {llm_output_str[:500]}")
        else:
            logger.warning("AuraReflector: LLM response for strategies was empty or malformed.")

    except Exception as e_llm:
        logger.error(f"AuraReflector: Error during LLM call for strategy generation: {e_llm}", exc_info=True)

    logger.info(f"AuraReflector: Generated {len(strategies)} refinement strategies.")
    return strategies


async def main_aura_reflector_cycle():
    """Main function to run the complete Aura Reflector cycle."""
    logger.info("--- Starting Aura Reflector Cycle ---")

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set. Aura Reflector cannot function.")
        return

    if not LOG_SERVICE_AVAILABLE:
        logger.error(
            "PersistentLogService is not available. Aura Reflector cannot effectively analyze historical data.")
        return

    # Analyze historical performance
    performance_insights = await analyze_historical_performance(
        min_events_for_llm_sample=50,
        min_total_events_for_analysis=100
    )

    if not performance_insights:
        logger.info("No performance insights generated. Ending cycle.")
        return

    print("\n--- Generated Performance Insights ---")
    for insight in performance_insights:
        print(json.dumps(insight.model_dump(), indent=2, default=str))

    # Generate refinement strategies
    refinement_strategies = await generate_refinement_strategies(performance_insights)

    if not refinement_strategies:
        logger.info("No refinement strategies generated. Ending cycle.")
        return

    print("\n--- Generated Refinement Strategies ---")
    for strategy in refinement_strategies:
        print(json.dumps(strategy.model_dump(), indent=2, default=str))

    logger.info("Aura Reflector Cycle Completed. Insights and Strategies generated.")
    logger.info("Next steps would involve review, approval, and implementation of strategies.")
