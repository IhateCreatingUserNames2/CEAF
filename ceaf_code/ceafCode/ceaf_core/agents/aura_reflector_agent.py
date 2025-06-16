# ceaf_core/agents/aura_reflector_agent.py

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, Optional, List

from .base_ceaf_agent import CeafBaseAgent
from ..config.model_configs import get_agent_model_name
from ..modules.mcl_engine.aura_reflector import (
    analyze_historical_performance,
    generate_refinement_strategies,
    PerformanceInsight,
    RefinementStrategy,
    persistent_log_service_instance as actual_persistent_log_service,
    LOG_SERVICE_AVAILABLE as ACTUAL_LOG_SERVICE_AVAILABLE
)
from ..tools.common_utils import parse_llm_json_output

logger = logging.getLogger("AuraReflectorAgent")

# --- AuraReflectorAgent Configuration ---
AURA_REFLECTOR_AGENT_DEFAULT_MODEL_ENV_VAR = "AURA_REFLECTOR_AGENT_MODEL"
AURA_REFLECTOR_AGENT_FALLBACK_MODEL = "openrouter/openai/gpt-4.1"

AURA_REFLECTOR_AGENT_SPECIFIC_INSTRUCTION = """
You are the AuraReflector Agent, a specialized component of the CEAF (Coherent Emergence Agent Framework).
Your primary purpose is to initiate and oversee the reflective learning cycle for the entire CEAF system, focusing on long-term performance improvement and adaptation.

**Core Functions:**
1.  **Initiate Performance Analysis:** When invoked (e.g., on a schedule or due to specific triggers), you will call internal functions to analyze historical performance data. This involves using sophisticated analysis routines (potentially including other LLMs) to identify patterns, anomalies, and areas for improvement across CEAF's operations (ORA responses, NCF effectiveness, MCL interventions, etc.).
2.  **Oversee Strategy Generation:** Based on the identified performance insights, you will guide the generation of actionable refinement strategies. These strategies aim to address the root causes of sub-optimal performance or to capitalize on identified strengths.
3.  **Report and Log:** You will summarize the findings (insights and proposed strategies) in a structured JSON format. This report is intended for system administrators, developers, or other CEAF components responsible for implementing refinements.
4.  **Trigger Assessment:** You can be triggered by specific conditions (e.g., MCL reporting consistently poor EoC scores for ORA, a high rate of VRE interventions, or explicit user/admin request) to perform an ad-hoc reflection cycle.

**Input Analysis (When invoked via direct message/tool call):**
You may receive:
-   `trigger_reason`: A string explaining why this reflection cycle is being initiated (e.g., "scheduled_weekly_review", "persistent_low_eoc_score_for_ora", "admin_request_deep_dive_on_coherence").
-   `analysis_parameters`: Optional JSON string with parameters to guide the analysis, such as:
    -   `time_window_days`: How far back to look (e.g., 7, 30).
    -   `focus_areas`: List of specific areas to prioritize (e.g., ["ncf_conceptual_entropy_tuning", "vre_ethical_alignment_consistency"]).
    -   `min_sessions_to_analyze`: Minimum data points required.

**Your Task (When invoked as an agent):**
1.  Acknowledge the trigger and any specific parameters.
2.  Internally execute the `analyze_historical_performance` function using the log store.
3.  Internally execute the `generate_refinement_strategies` function using the insights from step 2.
4.  Format the combined results (insights and strategies) into a clear, structured JSON output. This JSON should be the primary content of your response.

**Example Output Format (JSON - this is what *you* should produce as your final message content):**
```json
{
  "reflection_cycle_id": "reflect_cycle_xyz123",
  "trigger_reason": "scheduled_weekly_review",
  "analysis_summary": {
    "data_analyzed_period_days": 30,
    "insights_generated_count": 5,
    "key_themes_from_insights": ["Improving NCF parameter tuning for problem-solving goals.", "Enhancing memory retrieval relevance for creative tasks."]
  },
  "performance_insights": [
    { "insight_id": "insight_abc", "description": "...", "confidence": 0.9, "suggested_refinement_type": "ncf_param_heuristic" }
  ],
  "refinement_strategies_proposed": [
    { "strategy_id": "strat_def", "description": "...", "associated_insight_ids": ["insight_abc"], "strategy_type": "ncf_param_heuristic" }
  ],
  "overall_recommendation": "Proceed with review and prioritization of the proposed refinement strategies. Focus on 'ncf_param_heuristic' strategies first due to high potential impact and low estimated effort."
}
```

You do NOT implement the strategies yourself. You are the initiator and reporter of this reflective process.
"""

# Resolve model name for this specific agent
aura_reflector_agent_model_name = get_agent_model_name(
    agent_type_env_var=AURA_REFLECTOR_AGENT_DEFAULT_MODEL_ENV_VAR,
    fallback_model=AURA_REFLECTOR_AGENT_FALLBACK_MODEL
)


class AuraReflectorAgent(CeafBaseAgent):
    def __init__(self, mock_log_store_for_testing: Optional[Any] = None, **kwargs):
        super().__init__(
            name="AuraReflector_Agent",
            default_model_env_var=AURA_REFLECTOR_AGENT_DEFAULT_MODEL_ENV_VAR,
            fallback_model_name=AURA_REFLECTOR_AGENT_FALLBACK_MODEL,
            description="Initiates and oversees the reflective learning cycle for the CEAF system.",
            specific_instruction=AURA_REFLECTOR_AGENT_SPECIFIC_INSTRUCTION,
            tools=[],
            **kwargs
        )

        if mock_log_store_for_testing is not None:
            self.log_store = mock_log_store_for_testing
            logger.info("AuraReflector_Agent: Using provided mock log store.")
        elif ACTUAL_LOG_SERVICE_AVAILABLE and actual_persistent_log_service:
            self.log_store = actual_persistent_log_service
            logger.info("AuraReflector_Agent: Using actual PersistentLogService instance.")
        else:
            logger.warning(
                "AuraReflectorAgent: Real PersistentLogService is unavailable and no mock was provided. Using a minimal no-op log store.")

            class MinimalNoOpLogStore:
                def query_logs(self, *args, **kwargs) -> List[Any]:
                    return []

                def get_total_log_count(self) -> int:
                    return 0

            self.log_store = MinimalNoOpLogStore()

        logger.info(
            f"AuraReflector_Agent initialized with model '{getattr(self.model, 'model', self.model)}'. "
            f"Log store type: {type(self.log_store).__name__}"
        )

    async def _process_input_and_run_cycle(
            self,
            trigger_reason: str,
            analysis_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Internal method to run the reflection cycle and structure the output.
        """
        analysis_params = analysis_parameters or {}
        time_window = analysis_params.get("time_window_days", 30)
        min_events_for_analysis = analysis_params.get("min_events_to_analyze",
                                                      analysis_params.get("min_sessions_to_analyze", 100))

        logger.info(f"AuraReflector_Agent: Cycle triggered. Reason: {trigger_reason}. Params: {analysis_params}")

        # Ensure log_store is not None before proceeding
        if self.log_store is None:
            logger.error("AuraReflector_Agent: Log store is None. Cannot proceed with analysis.")
            return {
                "reflection_cycle_id": f"reflect_cycle_error_nologstore_{uuid.uuid4().hex[:8]}",
                "trigger_reason": trigger_reason,
                "error": "Log store not available for analysis.",
                "analysis_summary": {},
                "performance_insights": [],
                "refinement_strategies_proposed": [],
                "overall_recommendation": "Analysis failed due to missing log store."
            }

        try:
            insights = await analyze_historical_performance(
                log_store=self.log_store,
                time_window_days=time_window,
                min_total_events_for_analysis=min_events_for_analysis,
                min_events_for_llm_sample=analysis_params.get("min_events_for_llm_sample",
                                                              min_events_for_analysis // 2 or 50)
            )

            strategies: List[RefinementStrategy] = []
            if insights:
                strategies = await generate_refinement_strategies(insights)
            else:
                logger.info("AuraReflector_Agent: No insights generated, so no strategies will be generated.")

            cycle_id = f"reflect_cycle_{uuid.uuid4().hex[:12]}"
            key_themes: List[str] = []
            if insights:
                for insight in insights:
                    if insight.confidence > 0.7 and len(key_themes) < 3:
                        description_preview = insight.description.split('.')[0][:80]
                        key_themes.append(f"{description_preview}...")

            report = {
                "reflection_cycle_id": cycle_id,
                "trigger_reason": trigger_reason,
                "analysis_summary": {
                    "data_analyzed_period_days": time_window,
                    "insights_generated_count": len(insights),
                    "strategies_proposed_count": len(strategies),
                    "key_themes_from_insights": key_themes if key_themes else [
                        "No specific themes derived from insights."]
                },
                "performance_insights": [p.model_dump(exclude_none=True) for p in insights],
                "refinement_strategies_proposed": [s.model_dump(exclude_none=True) for s in strategies],
                "overall_recommendation": (
                    "Review insights and proposed strategies for prioritization and implementation planning."
                    if strategies else "No new strategies proposed. Monitor system performance."
                )
            }
            return report

        except Exception as e:
            logger.error(f"Error in reflection cycle: {e}", exc_info=True)
            return {
                "reflection_cycle_id": f"reflect_cycle_error_{uuid.uuid4().hex[:8]}",
                "trigger_reason": trigger_reason,
                "error": str(e),
                "analysis_summary": {
                    "data_analyzed_period_days": time_window,
                    "insights_generated_count": 0,
                    "strategies_proposed_count": 0,
                    "key_themes_from_insights": ["Error occurred during analysis"]
                },
                "performance_insights": [],
                "refinement_strategies_proposed": [],
                "overall_recommendation": "Analysis failed. Please check logs and retry."
            }

    async def generate_reflection_report(self, input_query: str) -> str:
        trigger_reason_from_query = "Ad-hoc request from query"
        analysis_params_from_query: Dict[str, Any] = {}

        try:
            parsed_input = json.loads(input_query)
            trigger_reason_from_query = parsed_input.get("trigger_reason", trigger_reason_from_query)
            analysis_params_from_query = parsed_input.get("analysis_parameters", analysis_params_from_query)
        except json.JSONDecodeError:
            if len(input_query) < 100:
                trigger_reason_from_query = input_query
            else:
                trigger_reason_from_query = f"Ad-hoc analysis requested: {input_query[:50]}..."

        logger.info(f"AuraReflector_Agent.generate_reflection_report called with query: {input_query}")

        report_dict = await self._process_input_and_run_cycle(
            trigger_reason=trigger_reason_from_query,
            analysis_parameters=analysis_params_from_query
        )

        return json.dumps(report_dict, indent=2, default=str)

    async def run_scheduled_reflection(
            self,
            time_window_days: int = 30,
            min_events_to_analyze: int = 100,
            focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        analysis_parameters = {
            "time_window_days": time_window_days,
            "min_events_to_analyze": min_events_to_analyze,
            "focus_areas": focus_areas or []
        }
        return await self._process_input_and_run_cycle(
            trigger_reason="scheduled_reflection_cycle",
            analysis_parameters=analysis_parameters
        )


# Instantiate the agent
try:
    aura_reflector_agent_instance = AuraReflectorAgent()
    logger.info("AuraReflector_Agent instance created successfully.")
except Exception as e:
    logger.critical(f"Failed to instantiate AuraReflectorAgent: {e}", exc_info=True)
    aura_reflector_agent_instance = None
