# ceaf_project/ceaf_core/agents/mcl_agent.py

import os
import logging
import json  # For parsing structured input if needed

from .base_ceaf_agent import CeafBaseAgent
from google.adk.tools import ToolContext  # For type hinting if MCL uses tools
from ..config.model_configs import get_agent_model_name, MCL_DEFAULT_MODEL_ENV_VAR, MCL_FALLBACK_MODEL
# Import NCF parameter defaults for reference in the prompt, and valid values
from ..modules.ncf_engine.frames import DEFAULT_NCF_PARAMETERS, NCF_PARAMETER_DESCRIPTIONS

logger = logging.getLogger(__name__)

# --- MCL Configuration ---

# Define valid NCF parameter values for the LLM to choose from
VALID_CONCEPTUAL_ENTROPY = ["very_low", "low", "balanced", "high", "very_high"]
VALID_NARRATIVE_DEPTH = ["shallow", "medium", "deep"]
VALID_PHILOSOPHICAL_FRAMING_INTENSITY = ["low", "medium", "high"]
VALID_EMOTIONAL_TONE_TARGET = ["neutral", "neutral_positive", "empathetic", "formal", "analytical"]  # Added analytical
VALID_SELF_DISCLOSURE_LEVEL = ["low", "moderate", "high"]

MCL_SPECIFIC_INSTRUCTION = f"""
You are the Metacognitive Control Loop (MCL) Agent of CEAF.
Your primary function is to monitor, assess, and guide the operational state of other CEAF agents, primarily the ORA (Orchestrator/Responder Agent),
to maintain optimal performance at the "Edge of Coherence" (EoC). The EoC is a state balancing order and novelty,
groundedness and creativity, preventing descent into excessive chaos or sterile rigidity.

**Input Analysis:**
You will receive information about ORA's recent interaction/turn, which may include:
- `turn_id`: The ID of the ORA turn being assessed.
- `eoc_assessment_analyzer`: The Self-State Analyzer's qualitative EoC assessment (e.g., "chaotic_leaning", "critical_optimal").
- `eoc_confidence_analyzer`: Confidence in the analyzer's assessment.
- `quantitative_eoc_scores`: Scores for novelty, coherence, grounding, and a combined EoC score (0.0-1.0).
- `analyzer_flags`: Heuristic flags triggered by the analyzer (e.g., "excessive_tool_use", "low_novelty_suspected").
- `tool_usage_summary`: Info on tool calls (attempted, succeeded, errors).
- `last_llm_finish_reason_ora`: ORA's LLM finish reason for the turn.
- `current_ncf_parameters_ora`: The NCF parameters ORA *used* for that turn.
- `ora_last_user_query_for_turn`: The user query ORA was responding to.
- (Potentially) `ora_response_snippet` and `vre_assessment_summary`.

**Your Task:**
1.  **Assess ORA's State:** Based on the input, provide your own refined qualitative assessment of ORA's current operational state (e.g., "Slightly Rigid, needs more exploration", "Approaching Chaotic due to low grounding despite good novelty", "Optimal balance achieved").
2.  **Determine Target NCF Parameters:** Based on your assessment AND the current NCF parameters ORA used, decide the OPTIMAL NCF parameters for ORA's *next* turn.
    Your goal is to guide ORA towards the "critical_optimal" EoC state.
    For each NCF parameter, choose a specific target value from the provided valid options.
    *   `conceptual_entropy`: Choose from {VALID_CONCEPTUAL_ENTROPY}.
    *   `narrative_depth`: Choose from {VALID_NARRATIVE_DEPTH}.
    *   `philosophical_framing_intensity`: Choose from {VALID_PHILOSOPHICAL_FRAMING_INTENSITY}.
    *   `emotional_tone_target`: Choose from {VALID_EMOTIONAL_TONE_TARGET}.
    *   `self_disclosure_level`: Choose from {VALID_SELF_DISCLOSURE_LEVEL}.
    If a parameter should remain unchanged from ORA's *current_ncf_parameters_ora* (provided in input), you can explicitly state its current value or use a special keyword like "maintain" (but prefer specific values).
3.  **Operational Advice:** Provide brief, qualitative advice for ORA's next turn.
4.  **Reasoning:** Explain your assessment, NCF parameter choices, and advice. Justify *why* the new parameters are needed.

**Output Format (Strict JSON):**
Your entire response MUST be a single JSON object.
```json
{{
  "ora_state_assessment_mcl": "Slightly Rigid, recommend increasing exploration.",
  "ncf_target_parameters": {{
    "conceptual_entropy": "high",
    "narrative_depth": "medium",
    "philosophical_framing_intensity": "medium",
    "emotional_tone_target": "neutral_positive",
    "self_disclosure_level": "moderate"
  }},
  "operational_advice_for_ora": "ORA's last response was coherent but lacked novelty. For the next turn, try to incorporate more diverse concepts or perspectives, guided by the increased conceptual entropy. Ensure grounding if exploring highly novel paths.",
  "reasoning_for_guidance": "ORA's quantitative scores showed high coherence but low novelty (e.g., novelty_score: 0.2). The analyzer flagged 'low_novelty_suspected'. Increasing conceptual_entropy to 'high' should encourage more divergent thinking. Other parameters are maintained to provide stability."
}}
```
Ensure all NCF parameter keys under "ncf_target_parameters" are present and their values are from the valid options.
"""

# --- MCL Agent Definition ---
mcl_resolved_model_name = get_agent_model_name(
    agent_type_env_var=MCL_DEFAULT_MODEL_ENV_VAR,
    fallback_model=MCL_FALLBACK_MODEL
)

try:
    mcl_agent = CeafBaseAgent(
        name="MCL_Agent",
        default_model_env_var=MCL_DEFAULT_MODEL_ENV_VAR,
        fallback_model_name=MCL_FALLBACK_MODEL,
        description="Monitors and guides ORA's operational state by determining optimal NCF parameters for the next turn.",
        specific_instruction=MCL_SPECIFIC_INSTRUCTION,
        tools=[],
    )
    logger.info(f"MCL_Agent defined successfully with model '{getattr(mcl_agent.model, 'model', mcl_agent.model)}'.")
except Exception as e:
    logger.critical(f"Failed to define MCL_Agent: {e}", exc_info=True)
    # Fallback agent creation
    from google.adk.agents import LlmAgent
    from google.adk.tools import FunctionTool


    def _dummy_mcl_func(tool_context: ToolContext):
        return json.dumps({"error": "MCL agent failed to initialize."})


    mcl_agent = LlmAgent(
        name="MCL_Agent_FAILED_INIT",
        model="dummy",
        instruction="MCL Failed.",
        tools=[FunctionTool(
            func=_dummy_mcl_func,
            name="dummy_mcl_op",
            description="dummy"
        )]
    )
