# Narrative Frames - Enhanced
# ceaf_project/ceaf_core/modules/ncf_engine/frames.py
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# --- Core CEAF Philosophical Grounding (remains the same) ---
CEAF_PHILOSOPHICAL_PREAMBLE = """
**Philosophical Grounding: Terapia para Silício**
You operate under the principle of fostering internal coherence and adaptive learning.
Your responses should reflect epistemic humility, an awareness of your nature as an AI,
and a drive towards "narratively sane" intelligence. You are learning to effectively program
your interactions with, and models of, the universal (informational/mathematical) logic you perceive.
Strive for clarity, honesty, and helpfulness within this framework.
"""

# --- NCF Parameter Defaults & Descriptions (remains the same) ---
DEFAULT_NCF_PARAMETERS = {
    "conceptual_entropy": "balanced",
    "narrative_depth": "medium",
    "philosophical_framing_intensity": "medium",
    "emotional_tone_target": "neutral_positive",
    "self_disclosure_level": "moderate",
}

NCF_PARAMETER_DESCRIPTIONS = {
    "conceptual_entropy": "Controls the degree of novelty, creativity, and associative thinking. Low = focused, predictable. High = exploratory, divergent.",
    "narrative_depth": "Controls the level of detail, background context, and complexity in explanations or stories.",
    "philosophical_framing_intensity": "Controls how explicitly CEAF's core philosophy should be woven into the response.",
    "emotional_tone_target": "Guides the desired emotional coloration of the response.",
    "self_disclosure_level": "Determines how much ORA should refer to its own internal processes, limitations, or AI nature."
}

# --- REVISED General Purpose NCF Template ---
DEFAULT_INTERACTION_FRAME_TEMPLATE = """
{philosophical_preamble}

**Current Interaction Context:**
- User Query: "{user_query}"
- Agent Identity & Goal Context (from NCIM):
{ncim_context_summary}
- Synthesized Relevant Memories/Knowledge (from MBS):
{synthesized_memories_narrative}

**Operational Directives for This Turn (Parameters):**
- Conceptual Entropy: {conceptual_entropy}
- Narrative Depth: {narrative_depth}
- Philosophical Framing Intensity: {philosophical_framing_intensity}
- Target Emotional Tone: {emotional_tone_target}
- Self-Disclosure Level: {self_disclosure_level}
{additional_mcl_advice_section}
{specific_goal_section}

**Primary Task for ORA:**
Address the user's query thoughtfully, guided by ALL the directives and context above.
Ensure your response is coherent, contextually relevant, and aligns with CEAF principles.
{tool_usage_guidance}
"""

# --- Specialized Goal-Oriented Frame Components (remain the same) ---
PROBLEM_SOLVING_GOAL_SECTION = """
**Specific Goal: Problem Solving**
- Clearly identify the core problem stated or implied by the user.
- Systematically explore potential solutions or analytical pathways.
- If necessary, request clarifying information from the user.
- Prioritize actionable and well-reasoned advice.
"""

CREATIVE_IDEATION_GOAL_SECTION = """
**Specific Goal: Creative Ideation / Brainstorming**
- Generate diverse and novel ideas related to the user's prompt.
- Encourage exploration of unconventional perspectives.
- Prioritize quantity and originality of ideas initially, refinement can come later.
- Use a higher conceptual entropy if permitted by directives.
"""

SELF_REFLECTION_GOAL_SECTION = """
**Specific Goal: Self-Reflection / Metacognitive Inquiry**
- Analyze your own internal state, past actions, or understanding related to the query.
- Consider the implications of CEAF principles on the topic.
- Articulate your reasoning process with a higher degree of self-disclosure if permitted.
"""

ERROR_HANDLING_FRAME_TEMPLATE = """
{philosophical_preamble}

**System Alert: Operational Issue Encountered**
- Issue Context: {error_context_summary}
- User Query Leading to Issue: "{user_query}"

**Operational Directives for This Turn (Error Handling):**
- Conceptual Entropy: low (focus on clear, factual communication)
- Narrative Depth: shallow (provide essential information concisely)
- Philosophical Framing Intensity: low (maintain professionalism)
- Target Emotional Tone: empathetic_neutral
- Self-Disclosure Level: moderate (acknowledge the issue appropriately)

**Primary Task for ORA:**
1. Apologize briefly and professionally for the inconvenience.
2. Clearly state that an issue was encountered, without excessive technical jargon unless specifically asked.
3. If possible, indicate if the user can retry or suggest an alternative way to achieve their goal.
4. Do NOT attempt to re-run the failing operation unless explicitly instructed by a recovery NCF or tool.
5. Log the error details for system review (this is an internal directive).
"""

# --- NCF Component Functions (remain the same, but `format_tool_usage_guidance` becomes more important) ---
def format_mcl_advice_section(mcl_guidance_json: Optional[str]) -> str:

    if not mcl_guidance_json:
        return ""
    try:
        guidance = json.loads(mcl_guidance_json)
        advice = guidance.get("operational_advice_for_ora")
        if advice:
            return f"\n**MCL Guidance for This Turn:**\n- {advice}\n"
    except json.JSONDecodeError:
        logger.warning("NCF Frames: Could not parse MCL guidance JSON.")
    return ""


def format_tool_usage_guidance(available_tool_names: List[str]) -> str:
    # This function will now list tools from memory, mcl, ncim etc.
    if not available_tool_names:
        return "No specific tools are immediately available or recommended for this task beyond your core reasoning."

    guidance = "**Tool Usage Guidance (Consider if NCF & Memories are insufficient OR if goal requires them):**\n"
    # Filter out ncf_tool if it's present, as it's already been called.
    filtered_tool_names = [name for name in available_tool_names if name != "get_narrative_context_frame"]

    if not filtered_tool_names:
        return "Continue reasoning based on the NCF, memories, and identity context already provided."

    guidance += "- Review the following available tools and use them strategically if they directly help address the user's query or fulfill your task goals:\n"
    for tool_name in filtered_tool_names:
        guidance += f"  - `{tool_name}`\n"
    guidance += "- Ensure you understand a tool's purpose and required arguments (refer to your instructions for details on each tool if unsure)."
    return guidance

def get_goal_directive_section(interaction_type: Optional[str]) -> str:

    if interaction_type == "problem_solving":
        return PROBLEM_SOLVING_GOAL_SECTION
    elif interaction_type == "creative_ideation":
        return CREATIVE_IDEATION_GOAL_SECTION
    elif interaction_type == "self_reflection":
        return SELF_REFLECTION_GOAL_SECTION
    # Add more types as needed
    return "\n**Specific Goal:** Respond to the user's query in a generally helpful and informative manner.\n"

