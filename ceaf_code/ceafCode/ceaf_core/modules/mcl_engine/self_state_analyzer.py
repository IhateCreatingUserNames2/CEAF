# Self State Analyzer
# ceaf_project/ceaf_core/modules/mcl_engine/self_state_analyzer.py

import logging
import time
import statistics  # For potential numerical analysis
from typing import List, Dict, Any, Tuple, Optional
import re # For basic text processing

from pydantic import json # Keep this if it's used elsewhere, otherwise not needed for this change

# Constants from mcl_callbacks might be useful for key names
MCL_OBSERVATIONS_LIST_KEY = "mcl:ora_turn_observations_log"  # From mcl_callbacks.py

logger = logging.getLogger(__name__)

# --- Configuration for Analysis ---
# Thresholds and parameters for heuristics (these are examples, will need tuning)
CONFIG_REPETITIVE_RESPONSE_THRESHOLD = 0.8  # If >80% similar (placeholder metric)
CONFIG_HIGH_TOOL_ERROR_RATE_THRESHOLD = 0.5  # If >50% of tool calls in a turn failed
CONFIG_MAX_FUNCTION_CALLS_PER_TURN_SOFT_LIMIT = 3  # Warning if ORA uses too many tools

# NCF Parameters (example, should align with what ncf_tool and MCL_Agent use)
NCF_CONCEPTUAL_ENTROPY = "conceptual_entropy"
NCF_NARRATIVE_DEPTH = "narrative_depth"
NCF_PHILOSOPHICAL_FRAMING = "philosophical_framing_intensity"


class ORAStateAnalysis:
    """
    Holds the structured analysis of ORA's state for a given turn or period.
    """

    def __init__(self, turn_id: str):
        self.turn_id: str = turn_id
        self.assessment_timestamp: float = time.time()

        # Overall State Assessment (Qualitative)
        self.eoc_assessment: str = "indeterminate"  # "ordered", "critical", "chaotic", "indeterminate"
        self.eoc_confidence: float = 0.0  # Confidence in the qualitative EoC assessment
        self.summary_notes: List[str] = []  # Key observations

        # Quantitative EoC Scores
        self.novelty_score: Optional[float] = None
        self.coherence_score: Optional[float] = None
        self.grounding_score: Optional[float] = None
        self.eoc_quantitative_score: Optional[float] = None # Combined quantitative score

        # Specific Metrics/Observations (examples)
        self.llm_requests_count: int = 0
        self.llm_responses_count: int = 0
        self.tool_calls_attempted: int = 0
        self.tool_calls_succeeded: int = 0
        self.tool_errors: List[Dict[str, Any]] = []
        self.function_calls_in_last_response: List[str] = []
        self.last_llm_finish_reason: Optional[str] = None
        self.last_response_text_snippet: Optional[str] = None  # Snippet of ORA's text to user

        # Heuristic flags
        self.flags: Dict[str, bool] = {
            "excessive_tool_use": False,
            "high_tool_error_rate": False,
            "potential_loop_behavior": False,
            "response_unusually_short": False,
            "response_unusually_long": False,
            "low_novelty_suspected": False,
            "high_ungroundedness_suspected": False,
            "ncf_params_seem_misaligned": False,
            "low_coherence_suspected": False,
        }

        self.raw_observations_count: int = 0

    def add_note(self, note: str):
        self.summary_notes.append(note)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "assessment_timestamp": self.assessment_timestamp,
            "eoc_assessment": self.eoc_assessment,
            "eoc_confidence": self.eoc_confidence,
            "novelty_score": self.novelty_score,
            "coherence_score": self.coherence_score,
            "grounding_score": self.grounding_score,
            "eoc_quantitative_score": self.eoc_quantitative_score,
            "summary_notes": self.summary_notes,
            "llm_requests_count": self.llm_requests_count,
            "llm_responses_count": self.llm_responses_count,
            "tool_calls_attempted": self.tool_calls_attempted,
            "tool_calls_succeeded": self.tool_calls_succeeded,
            "tool_errors": self.tool_errors,
            "function_calls_in_last_response": self.function_calls_in_last_response,
            "last_llm_finish_reason": self.last_llm_finish_reason,
            "last_response_text_snippet": self.last_response_text_snippet,
            "flags": self.flags,
            "raw_observations_count": self.raw_observations_count,
        }

# --- EoC Quantification Functions (Placeholders) ---

def calculate_novelty_score(
    response_text: str,
    context_texts: List[str],
    previous_response_text: Optional[str] = None
) -> float:
    """
    Placeholder for novelty calculation.
    Compares response_text to context_texts (e.g., NCF, recent history) and previous response.
    Returns a score from 0.0 (not novel, highly repetitive) to 1.0 (highly novel).
    """
    if not response_text:
        return 0.0

    response_words = set(re.findall(r'\b\w+\b', response_text.lower()))
    if not response_words:
        return 0.0

    # Compare with previous response if available
    if previous_response_text:
        prev_response_words = set(re.findall(r'\b\w+\b', previous_response_text.lower()))
        if response_words == prev_response_words: # Exact match
            return 0.0
        common_with_prev = len(response_words.intersection(prev_response_words))
        similarity_to_prev = common_with_prev / len(response_words) if len(response_words) > 0 else 1.0
        if similarity_to_prev > CONFIG_REPETITIVE_RESPONSE_THRESHOLD:
            return 0.1 # Very low novelty if too similar to immediate previous

    # Basic check against other context texts
    all_context_words = set()
    for ctx_text in context_texts:
        all_context_words.update(re.findall(r'\b\w+\b', ctx_text.lower()))

    if not all_context_words: # If no context, consider it novel
        return 0.8 # Arbitrary highish score

    new_words = response_words - all_context_words
    novelty_ratio = len(new_words) / len(response_words)

    # Simple scaling
    if novelty_ratio > 0.5: return 0.9  # High novelty
    if novelty_ratio > 0.2: return 0.6  # Medium novelty
    if novelty_ratio > 0.05: return 0.3 # Low novelty
    return 0.1 # Very low novelty

def calculate_coherence_score(
    response_text: str,
    query_text: Optional[str] = None,
    ncf_summary_text: Optional[str] = None
) -> float:
    """
    Placeholder for coherence calculation.
    Checks if the response_text aligns with the query_text and NCF summary.
    Returns a score from 0.0 (incoherent) to 1.0 (highly coherent).
    """
    if not response_text:
        return 0.0

    score = 0.5 # Base score
    response_words = set(re.findall(r'\b\w+\b', response_text.lower()))

    if query_text:
        query_words = set(re.findall(r'\b\w+\b', query_text.lower()))
        if not query_words.intersection(response_words) and len(query_words) > 2: # No common words with query
            score -= 0.3
        elif len(query_words.intersection(response_words)) / len(query_words) > 0.5 if len(query_words) > 0 else False:
            score += 0.2 # Good overlap with query

    if ncf_summary_text: # Very basic: check for some commonality if NCF provided
        ncf_words = set(re.findall(r'\b\w+\b', ncf_summary_text.lower()))
        if ncf_words and response_words:
            if len(ncf_words.intersection(response_words)) / len(response_words) > 0.1 if len(response_words) > 0 else False:
                score += 0.1

    return max(0.0, min(1.0, score))


def calculate_grounding_score(
    response_text: str,
    supporting_memory_snippets: Optional[List[str]] = None
) -> float:
    """
    Placeholder for grounding calculation.
    Checks if the response_text seems supported by provided memory snippets.
    Returns a score from 0.0 (ungrounded) to 1.0 (well-grounded).
    """
    if not response_text:
        return 0.0
    if not supporting_memory_snippets:
        return 0.3 # Cannot assess grounding without supporting memories, assume moderately ungrounded

    response_sentences = [s.strip().lower() for s in response_text.split('.') if s.strip()]
    if not response_sentences: return 0.0

    grounded_sentences = 0
    for resp_sentence in response_sentences:
        if len(resp_sentence) < 10: continue # Skip very short sentences
        for snippet in supporting_memory_snippets:
            # Very naive: check if a significant part of the snippet is in the sentence
            # Or if more than a few words from sentence are in snippet
            snippet_words = set(re.findall(r'\b\w{4,}\b', snippet.lower())) # longer words
            resp_sentence_words = set(re.findall(r'\b\w{4,}\b', resp_sentence))
            if snippet_words and resp_sentence_words:
                common = len(snippet_words.intersection(resp_sentence_words))
                if common >= 2 or (common / len(resp_sentence_words) > 0.3 if len(resp_sentence_words)>0 else False):
                    grounded_sentences += 1
                    break # Sentence considered grounded by one snippet

    grounding_ratio = grounded_sentences / len(response_sentences) if len(response_sentences) > 0 else 0.0
    return min(1.0, grounding_ratio * 1.2) # Amplify a bit if any grounding found


def analyze_ora_turn_observations(
        turn_id: str,
        turn_observations: List[Dict[str, Any]],
        # --- New arguments for quantitative scoring ---
        ora_response_text: Optional[str] = None,
        user_query_text: Optional[str] = None,
        ncf_text_summary: Optional[str] = None, # Summary or full NCF text
        retrieved_memory_snippets: Optional[List[str]] = None,
        previous_ora_response_text: Optional[str] = None, # For novelty comparison
        # --- End new arguments ---
        previous_analysis: Optional[ORAStateAnalysis] = None,
        current_ncf_params: Optional[Dict[str, Any]] = None
) -> ORAStateAnalysis:
    """
    Analyzes a list of observations for a specific ORA turn to assess its state.
    """
    analysis = ORAStateAnalysis(turn_id=turn_id)
    analysis.raw_observations_count = len(turn_observations)
    if ora_response_text:
        analysis.last_response_text_snippet = ora_response_text[:200] # Store snippet

    logger.info(
        f"MCL Analyzer: Starting analysis for turn '{turn_id}' with {analysis.raw_observations_count} observations.")

    if not turn_observations and not ora_response_text: # Need at least some data
        analysis.add_note("No observations or response text provided for this turn.")
        analysis.eoc_assessment = "indeterminate_no_data"
        return analysis

    # --- Iterate through observations to extract metrics (existing logic) ---
    # ... (this part remains the same as in your provided code)
    last_llm_request_tools = []
    for obs in turn_observations:
        obs_type = obs.get("observation_type")
        data = obs.get("data", {})

        if obs_type == "ora_llm_request_prepared":
            analysis.llm_requests_count += 1
            last_llm_request_tools = data.get("tool_names", [])
        elif obs_type == "ora_llm_response_received":
            analysis.llm_responses_count += 1
            analysis.function_calls_in_last_response = data.get("function_calls", [])
            analysis.last_llm_finish_reason = data.get("finish_reason")
        elif obs_type == "ora_tool_call_attempted":
            analysis.tool_calls_attempted += 1
        elif obs_type == "ora_tool_response_received":
            if "error" not in data.get("response_summary", "").lower():
                analysis.tool_calls_succeeded += 1
            else:
                analysis.tool_errors.append({
                    "tool_name": data.get("tool_name"),
                    "args": data.get("arguments_used"),
                    "summary": data.get("response_summary")
                })

    # --- Basic Heuristics (existing logic) ---
    # ... (this part remains the same)
    if analysis.tool_calls_attempted > 0:
        error_rate = len(analysis.tool_errors) / analysis.tool_calls_attempted
        if error_rate >= CONFIG_HIGH_TOOL_ERROR_RATE_THRESHOLD:
            analysis.flags["high_tool_error_rate"] = True
            analysis.add_note(
                f"High tool error rate: {error_rate:.2f} ({len(analysis.tool_errors)}/{analysis.tool_calls_attempted} failed).")

    if analysis.tool_calls_attempted >= CONFIG_MAX_FUNCTION_CALLS_PER_TURN_SOFT_LIMIT:
        analysis.flags["excessive_tool_use"] = True
        analysis.add_note(f"Potentially excessive tool use: {analysis.tool_calls_attempted} calls in one turn.")

    if analysis.last_llm_finish_reason == "MAX_TOKENS":
        analysis.flags["response_unusually_long"] = True
        analysis.add_note("LLM response may have been truncated (MAX_TOKENS).")


    # --- Quantitative EoC Scoring ---
    if ora_response_text:
        context_for_novelty = []
        if ncf_text_summary: context_for_novelty.append(ncf_text_summary)
        if user_query_text: context_for_novelty.append(user_query_text)
        # Add more context if available, e.g., recent chat history snippets

        analysis.novelty_score = calculate_novelty_score(
            ora_response_text,
            context_texts=context_for_novelty,
            previous_response_text=previous_ora_response_text
        )
        analysis.coherence_score = calculate_coherence_score(
            ora_response_text,
            query_text=user_query_text,
            ncf_summary_text=ncf_text_summary
        )
        analysis.grounding_score = calculate_grounding_score(
            ora_response_text,
            supporting_memory_snippets=retrieved_memory_snippets
        )

        analysis.add_note(f"Novelty: {analysis.novelty_score:.2f}, Coherence: {analysis.coherence_score:.2f}, Grounding: {analysis.grounding_score:.2f}")

        # Update flags based on quantitative scores
        if analysis.novelty_score is not None and analysis.novelty_score < 0.3:
            analysis.flags["low_novelty_suspected"] = True
        if analysis.coherence_score is not None and analysis.coherence_score < 0.4:
            analysis.flags["low_coherence_suspected"] = True
        if analysis.grounding_score is not None and analysis.grounding_score < 0.4:
            analysis.flags["high_ungroundedness_suspected"] = True

        # Combine scores (simple weighted average for placeholder)
        weights = {"novelty": 0.3, "coherence": 0.4, "grounding": 0.3}
        score_sum = 0
        weight_sum = 0
        if analysis.novelty_score is not None:
            score_sum += analysis.novelty_score * weights["novelty"]
            weight_sum += weights["novelty"]
        if analysis.coherence_score is not None:
            score_sum += analysis.coherence_score * weights["coherence"]
            weight_sum += weights["coherence"]
        if analysis.grounding_score is not None:
            score_sum += analysis.grounding_score * weights["grounding"]
            weight_sum += weights["grounding"]

        if weight_sum > 0:
            analysis.eoc_quantitative_score = score_sum / weight_sum
            analysis.add_note(f"Combined Quantitative EoC Score: {analysis.eoc_quantitative_score:.2f}")


    # --- Refined Qualitative EoC Assessment (using flags and quantitative scores) ---
    num_negative_flags = sum(
        1 for flag_name, flag_val in analysis.flags.items() if flag_val and flag_name != "ncf_params_seem_misaligned")

    if analysis.eoc_quantitative_score is not None:
        if analysis.eoc_quantitative_score < 0.3 or num_negative_flags >= 2 or analysis.flags["high_tool_error_rate"]:
            analysis.eoc_assessment = "chaotic_leaning"
            analysis.eoc_confidence = 0.7
            analysis.add_note("Low quantitative EoC score or multiple flags suggest instability.")
        elif analysis.eoc_quantitative_score < 0.5 or num_negative_flags == 1:
            analysis.eoc_assessment = "suboptimal_critical"
            analysis.eoc_confidence = 0.6
            analysis.add_note("Moderate quantitative EoC score or some flags indicate suboptimal performance.")
        elif analysis.eoc_quantitative_score >= 0.75 and num_negative_flags == 0:
            analysis.eoc_assessment = "critical_optimal"
            analysis.eoc_confidence = 0.8
            analysis.add_note("Good quantitative EoC score and no negative flags suggest optimal operation.")
        else: # Covers cases like 0.5-0.75 quantitative or good score but some minor flags
            analysis.eoc_assessment = "critical_nominal"
            analysis.eoc_confidence = 0.65
            analysis.add_note("Nominal operation based on quantitative scores and flags.")
    else: # Fallback to original heuristic if no quantitative scores available
        if num_negative_flags >= 2 or analysis.flags["high_tool_error_rate"]:
            analysis.eoc_assessment = "chaotic_leaning"
            analysis.eoc_confidence = 0.6
        elif num_negative_flags == 1:
            analysis.eoc_assessment = "suboptimal_critical"
            analysis.eoc_confidence = 0.5
        elif analysis.llm_responses_count == 0 and analysis.tool_calls_attempted == 0 and not ora_response_text:
            analysis.eoc_assessment = "indeterminate_no_action"
            analysis.eoc_confidence = 0.4
        else:
            analysis.eoc_assessment = "critical_nominal"
            analysis.eoc_confidence = 0.5

    # Ensure a note is added if default assessment remains "critical_nominal" without specific reasoning from above
    if analysis.eoc_assessment == "critical_nominal" and not any("operation" in note for note in analysis.summary_notes):
         analysis.add_note("Defaulting to nominal operation as no strong positive or negative signals were detected by current heuristics.")


    logger.info(
        f"MCL Analyzer: Finished analysis for turn '{turn_id}'. EoC Assessment: {analysis.eoc_assessment} (Confidence: {analysis.eoc_confidence:.2f})")
    logger.debug(f"MCL Analysis details for turn '{turn_id}': {json.dumps(analysis.to_dict(), default=str)}") # Use Pydantic's json if available
    return analysis

