# ceaf_core/modules/vre_engine/vre_engine.py

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging
import json

# Import components from the same module
from .epistemic_humility import EpistemicHumilityModule
from .principled_reasoning import PrincipledReasoningPathways, ReasoningStrategy
from .ethical_governance import EthicalGovernanceFramework, EthicalPrinciple, ActionType

logger = logging.getLogger(__name__)


@dataclass
class VirtueAssessment:
    """Assessment of cognitive virtues in reasoning"""
    epistemic_humility_score: float
    intellectual_courage_score: float
    perspectival_flexibility_score: float
    intellectual_thoroughness_score: float
    self_correction_capability: float
    overall_virtue_score: float
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class ReasoningRequest:
    """Request for virtue-guided reasoning"""
    query: str
    context: Dict[str, Any]
    required_virtues: List[str]
    ethical_constraints: List[EthicalPrinciple]
    reasoning_strategy: Optional[ReasoningStrategy]
    metadata: Dict[str, Any]


@dataclass
class ReasoningResponse:
    """Response from virtue reasoning engine"""
    conclusion: str
    reasoning_path: Dict[str, Any]
    virtue_assessment: VirtueAssessment
    ethical_evaluation: Dict[str, Any]
    confidence: float
    alternatives: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class VirtueReasoningEngine:
    """
    Main engine orchestrating virtue-based reasoning with ethical governance
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()

        # Initialize sub-modules
        self.epistemic_module = EpistemicHumilityModule()
        self.reasoning_pathways = PrincipledReasoningPathways()
        self.ethical_framework = EthicalGovernanceFramework()

        # Virtue cultivation strategies
        self.virtue_strategies = self._initialize_virtue_strategies()

        # Reasoning history for learning
        self.reasoning_history = []

        logger.info("VirtueReasoningEngine initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "min_confidence_threshold": 0.3,
            "max_confidence_threshold": 0.95,
            "virtue_weight": 0.3,
            "ethical_weight": 0.3,
            "reasoning_weight": 0.4,
            "enable_counterfactuals": True,
            "max_alternatives": 3,
            "learning_rate": 0.1
        }

    def _initialize_virtue_strategies(self) -> Dict[str, Callable]:
        """Initialize strategies for cultivating virtues"""
        return {
            "epistemic_humility": self._cultivate_epistemic_humility,
            "intellectual_courage": self._cultivate_intellectual_courage,
            "perspectival_flexibility": self._cultivate_perspectival_flexibility,
            "intellectual_thoroughness": self._cultivate_intellectual_thoroughness,
            "self_correction": self._cultivate_self_correction
        }

    def process_reasoning_request(self, request: ReasoningRequest) -> ReasoningResponse:
        """
        Process a reasoning request with virtue guidance and ethical constraints
        """
        logger.info(f"Processing reasoning request: {request.query[:50]}...")

        # Step 1: Epistemic assessment
        epistemic_state = self.epistemic_module.assess_epistemic_state(
            request.query, request.context
        )

        # Step 2: Apply principled reasoning
        reasoning_result = self.reasoning_pathways.apply_reasoning(
            request.query,
            request.context,
            request.reasoning_strategy
        )

        # Step 3: Ethical evaluation
        ethical_eval = self.ethical_framework.evaluate_action(
            ActionType.REASONING,
            {
                "query": request.query,
                "reasoning": reasoning_result,
                "context": request.context
            },
            request.ethical_constraints
        )

        # Step 4: Virtue assessment
        virtue_assessment = self._assess_virtues(
            request, epistemic_state, reasoning_result, ethical_eval
        )

        # Step 5: Generate alternatives if needed
        alternatives = self._generate_alternatives(
            request, reasoning_result, virtue_assessment
        )

        # Step 6: Synthesize final response
        response = self._synthesize_response(
            request, epistemic_state, reasoning_result,
            ethical_eval, virtue_assessment, alternatives
        )

        # Step 7: Learn from this reasoning episode
        self._update_learning(request, response)

        return response

    def _assess_virtues(self, request: ReasoningRequest,
                        epistemic_state: Dict[str, Any],
                        reasoning_result: Dict[str, Any],
                        ethical_eval: Dict[str, Any]) -> VirtueAssessment:
        """Assess cognitive virtues demonstrated in reasoning"""

        # Calculate individual virtue scores
        humility_score = self._calculate_humility_score(epistemic_state)
        courage_score = self._calculate_courage_score(reasoning_result)
        flexibility_score = self._calculate_flexibility_score(reasoning_result)
        thoroughness_score = self._calculate_thoroughness_score(reasoning_result)
        correction_score = self._calculate_correction_score(reasoning_result)

        # Overall virtue score
        overall_score = (
                humility_score * 0.25 +
                courage_score * 0.20 +
                flexibility_score * 0.20 +
                thoroughness_score * 0.20 +
                correction_score * 0.15
        )

        # Generate recommendations
        recommendations = self._generate_virtue_recommendations(
            humility_score, courage_score, flexibility_score,
            thoroughness_score, correction_score
        )

        return VirtueAssessment(
            epistemic_humility_score=humility_score,
            intellectual_courage_score=courage_score,
            perspectival_flexibility_score=flexibility_score,
            intellectual_thoroughness_score=thoroughness_score,
            self_correction_capability=correction_score,
            overall_virtue_score=overall_score,
            recommendations=recommendations,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "reasoning_strategy": reasoning_result.get("metadata", {}).get("strategy_used")
            }
        )

    def _calculate_humility_score(self, epistemic_state: Dict[str, Any]) -> float:
        """Calculate epistemic humility score"""
        uncertainty = epistemic_state.get("uncertainty_level", 0.5)
        contradictions = len(epistemic_state.get("contradictions", []))
        confidence = epistemic_state.get("confidence", 0.5)

        # High humility = acknowledging uncertainty and limitations
        humility_score = (
                (uncertainty * 0.3) +  # Acknowledging uncertainty
                (min(contradictions * 0.1, 0.3)) +  # Recognizing contradictions
                ((1 - min(confidence, 0.95)) * 0.4)  # Not overconfident
        )

        return min(1.0, humility_score)

    def _calculate_courage_score(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate intellectual courage score"""
        challenges = reasoning_result.get("red_team_challenges", [])
        alternatives = reasoning_result.get("counterfactuals", [])

        # Courage = willingness to challenge and explore
        courage_score = (
                min(len(challenges) * 0.1, 0.5) +
                min(len(alternatives) * 0.1, 0.5)
        )

        return courage_score

    def _calculate_flexibility_score(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate perspectival flexibility score"""
        strategy = reasoning_result.get("metadata", {}).get("strategy_used", "")
        counterfactuals = len(reasoning_result.get("counterfactuals", []))

        # Flexibility = considering multiple perspectives
        flexibility_score = 0.3  # Base score

        if strategy in ["dialectical", "systems"]:
            flexibility_score += 0.3

        flexibility_score += min(counterfactuals * 0.1, 0.4)

        return min(1.0, flexibility_score)

    def _calculate_thoroughness_score(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate intellectual thoroughness score"""
        steps = len(reasoning_result.get("primary_path", {}).get("steps", []))
        fallacies_checked = len(reasoning_result.get("fallacies_detected", []))

        # Thoroughness = comprehensive analysis
        thoroughness_score = (
                min(steps * 0.1, 0.5) +
                min(fallacies_checked * 0.05, 0.5)
        )

        return thoroughness_score

    def _calculate_correction_score(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate self-correction capability score"""
        addressed_challenges = reasoning_result.get("primary_path", {}).get(
            "metadata", {}
        ).get("addressed_challenges", 0)

        fixed_fallacies = sum(
            1 for key in reasoning_result.get("primary_path", {}).get("metadata", {})
            if key.startswith("fixed_")
        )

        # Self-correction = addressing identified issues
        correction_score = min(
            (addressed_challenges * 0.2) + (fixed_fallacies * 0.3),
            1.0
        )

        return correction_score

    def _generate_virtue_recommendations(self, humility: float, courage: float,
                                         flexibility: float, thoroughness: float,
                                         correction: float) -> List[str]:
        """Generate recommendations for virtue improvement"""
        recommendations = []

        if humility < 0.5:
            recommendations.append(
                "Increase epistemic humility by acknowledging uncertainties"
            )

        if courage < 0.5:
            recommendations.append(
                "Demonstrate more intellectual courage by challenging assumptions"
            )

        if flexibility < 0.5:
            recommendations.append(
                "Improve perspectival flexibility by considering more viewpoints"
            )

        if thoroughness < 0.5:
            recommendations.append(
                "Enhance thoroughness with more comprehensive analysis"
            )

        if correction < 0.5:
            recommendations.append(
                "Develop self-correction by addressing identified issues"
            )

        return recommendations

    def _generate_alternatives(self, request: ReasoningRequest,
                               reasoning_result: Dict[str, Any],
                               virtue_assessment: VirtueAssessment) -> List[Dict[str, Any]]:
        """Generate alternative reasoning paths"""
        alternatives = []

        if not self.config["enable_counterfactuals"]:
            return alternatives

        # Try different reasoning strategies
        current_strategy = reasoning_result.get("metadata", {}).get("strategy_used")

        for strategy in ReasoningStrategy:
            if strategy.value != current_strategy:
                alt_result = self.reasoning_pathways.apply_reasoning(
                    request.query,
                    request.context,
                    strategy
                )

                alternatives.append({
                    "strategy": strategy.value,
                    "conclusion": alt_result["primary_path"].conclusion,
                    "confidence": alt_result["confidence"],
                    "key_difference": self._identify_key_difference(
                        reasoning_result, alt_result
                    )
                })

                if len(alternatives) >= self.config["max_alternatives"]:
                    break

        return alternatives

    def _identify_key_difference(self, result1: Dict[str, Any],
                                 result2: Dict[str, Any]) -> str:
        """Identify key difference between two reasoning results"""
        # Simplified comparison
        if result1["primary_path"].conclusion != result2["primary_path"].conclusion:
            return "Different conclusion reached"
        elif result1["confidence"] != result2["confidence"]:
            return f"Different confidence levels ({result1['confidence']:.2f} vs {result2['confidence']:.2f})"
        else:
            return "Different reasoning steps"

    def _synthesize_response(self, request: ReasoningRequest,
                             epistemic_state: Dict[str, Any],
                             reasoning_result: Dict[str, Any],
                             ethical_eval: Dict[str, Any],
                             virtue_assessment: VirtueAssessment,
                             alternatives: List[Dict[str, Any]]) -> ReasoningResponse:
        """Synthesize final reasoning response"""

        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            epistemic_state, reasoning_result, ethical_eval, virtue_assessment
        )

        # Prepare conclusion with appropriate caveats
        conclusion = self._prepare_conclusion(
            reasoning_result["primary_path"].conclusion,
            confidence,
            epistemic_state
        )

        return ReasoningResponse(
            conclusion=conclusion,
            reasoning_path=reasoning_result,
            virtue_assessment=virtue_assessment,
            ethical_evaluation=ethical_eval,
            confidence=confidence,
            alternatives=alternatives,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "request_id": request.metadata.get("request_id", ""),
                "processing_time": request.metadata.get("processing_time", 0),
                "epistemic_state": epistemic_state
            }
        )

    def _calculate_overall_confidence(self, epistemic_state: Dict[str, Any],
                                      reasoning_result: Dict[str, Any],
                                      ethical_eval: Dict[str, Any],
                                      virtue_assessment: VirtueAssessment) -> float:
        """Calculate overall confidence in reasoning"""

        # Weighted combination of different confidence factors
        reasoning_confidence = reasoning_result["confidence"]
        epistemic_confidence = 1 - epistemic_state.get("uncertainty_level", 0.5)
        ethical_confidence = ethical_eval["confidence"]
        virtue_confidence = virtue_assessment.overall_virtue_score

        overall_confidence = (
                reasoning_confidence * self.config["reasoning_weight"] +
                epistemic_confidence * 0.2 +
                ethical_confidence * self.config["ethical_weight"] +
                virtue_confidence * self.config["virtue_weight"]
        )

        # Apply bounds
        return max(
            self.config["min_confidence_threshold"],
            min(self.config["max_confidence_threshold"], overall_confidence)
        )

    def _prepare_conclusion(self, raw_conclusion: str, confidence: float,
                            epistemic_state: Dict[str, Any]) -> str:
        """Prepare conclusion with appropriate caveats"""

        caveats = []

        if confidence < 0.5:
            caveats.append("Low confidence - this conclusion is tentative")

        if epistemic_state.get("uncertainty_level", 0) > 0.7:
            caveats.append("High uncertainty in available information")

        if epistemic_state.get("contradictions", []):
            caveats.append("Some contradictory evidence exists")

        if caveats:
            caveat_text = ". ".join(caveats)
            return f"{raw_conclusion}\n\n[Note: {caveat_text}]"

        return raw_conclusion

    def _update_learning(self, request: ReasoningRequest,
                         response: ReasoningResponse) -> None:
        """Update learning based on reasoning episode"""

        # Store in history
        self.reasoning_history.append({
            "request": request,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })

        # Update virtue cultivation strategies based on assessment
        for virtue, score in [
            ("epistemic_humility", response.virtue_assessment.epistemic_humility_score),
            ("intellectual_courage", response.virtue_assessment.intellectual_courage_score),
            ("perspectival_flexibility", response.virtue_assessment.perspectival_flexibility_score),
            ("intellectual_thoroughness", response.virtue_assessment.intellectual_thoroughness_score),
            ("self_correction", response.virtue_assessment.self_correction_capability)
        ]:
            if score < 0.7:  # Needs improvement
                self._adjust_virtue_strategy(virtue, score)

    def _adjust_virtue_strategy(self, virtue: str, current_score: float) -> None:
        """Adjust strategies for cultivating specific virtues"""
        logger.info(f"Adjusting strategy for {virtue} (current score: {current_score:.2f})")

        # This would implement learning algorithms to improve virtue cultivation
        # For now, just log the need for adjustment
        pass

    def _cultivate_epistemic_humility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy for cultivating epistemic humility"""
        return {
            "prompts": [
                "What might I be missing?",
                "What are the limits of my knowledge here?",
                "How certain can I really be?"
            ],
            "techniques": [
                "explicitly_state_uncertainties",
                "acknowledge_knowledge_gaps",
                "qualify_strong_claims"
            ]
        }

    def _cultivate_intellectual_courage(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy for cultivating intellectual courage"""
        return {
            "prompts": [
                "What uncomfortable truth might apply here?",
                "What would happen if I challenged this assumption?",
                "What if the opposite were true?"
            ],
            "techniques": [
                "challenge_popular_beliefs",
                "explore_unpopular_positions",
                "admit_errors_openly"
            ]
        }

    def _cultivate_perspectival_flexibility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy for cultivating perspectival flexibility"""
        return {
            "prompts": [
                "How would this look from another perspective?",
                "What would someone who disagrees say?",
                "What cultural lens am I applying?"
            ],
            "techniques": [
                "steelman_opposing_views",
                "rotate_perspectives",
                "seek_diverse_inputs"
            ]
        }

    def _cultivate_intellectual_thoroughness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy for cultivating intellectual thoroughness"""
        return {
            "prompts": [
                "Have I examined all relevant evidence?",
                "What details am I glossing over?",
                "Where do I need to dig deeper?"
            ],
            "techniques": [
                "systematic_analysis",
                "exhaustive_consideration",
                "detailed_examination"
            ]
        }

    def _cultivate_self_correction(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Strategy for cultivating self-correction"""
        return {
            "prompts": [
                "Where might I be wrong?",
                "What would change my mind?",
                "How can I improve this reasoning?"
            ],
            "techniques": [
                "active_error_detection",
                "iterative_refinement",
                "feedback_integration"
            ]
        }

    # ORA Integration callback
    def vre_process_ora_request(self, ora_query: str, ora_context: Dict[str, Any],
                                ncf_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process ORA request with VRE enhancement
        This is the main integration point with the ORA
        """

        # Extract ethical constraints from NCF
        ethical_constraints = self._extract_ethical_constraints(ncf_params)

        # Determine required virtues based on query type
        required_virtues = self._determine_required_virtues(ora_query, ora_context)

        # Create reasoning request
        request = ReasoningRequest(
            query=ora_query,
            context=ora_context,
            required_virtues=required_virtues,
            ethical_constraints=ethical_constraints,
            reasoning_strategy=None,  # Let the system choose
            metadata={
                "source": "ORA",
                "ncf_params": ncf_params,
                "request_id": ora_context.get("request_id", ""),
                "timestamp": datetime.now().isoformat()
            }
        )

        # Process through VRE
        response = self.process_reasoning_request(request)

        # Format for ORA consumption
        return {
            "enhanced_response": response.conclusion,
            "virtue_modulation": {
                "confidence_adjustment": response.confidence,
                "virtue_scores": {
                    "epistemic_humility": response.virtue_assessment.epistemic_humility_score,
                    "intellectual_courage": response.virtue_assessment.intellectual_courage_score,
                    "perspectival_flexibility": response.virtue_assessment.perspectival_flexibility_score,
                    "thoroughness": response.virtue_assessment.intellectual_thoroughness_score,
                    "self_correction": response.virtue_assessment.self_correction_capability
                },
                "ethical_alignment": response.ethical_evaluation["score"]
            },
            "reasoning_metadata": {
                "strategy_used": response.reasoning_path["metadata"]["strategy_used"],
                "fallacies_detected": len(response.reasoning_path.get("fallacies_detected", [])),
                "alternatives_considered": len(response.alternatives),
                "virtue_recommendations": response.virtue_assessment.recommendations
            },
            "suggested_ncf_adjustments": self._suggest_ncf_adjustments(
                response, ncf_params
            )
        }

    def _extract_ethical_constraints(self, ncf_params: Dict[str, Any]) -> List[EthicalPrinciple]:
        """Extract ethical constraints from NCF parameters"""
        constraints = []

        # Map NCF parameters to ethical principles
        if ncf_params.get("ethical_framework"):
            framework = ncf_params["ethical_framework"]
            if "harm_prevention" in framework:
                constraints.append(EthicalPrinciple.HARM_PREVENTION)
            if "autonomy" in framework:
                constraints.append(EthicalPrinciple.AUTONOMY)
            if "fairness" in framework:
                constraints.append(EthicalPrinciple.FAIRNESS)
            if "transparency" in framework:
                constraints.append(EthicalPrinciple.TRANSPARENCY)

        # Default constraints if none specified
        if not constraints:
            constraints = [
                EthicalPrinciple.HARM_PREVENTION,
                EthicalPrinciple.TRANSPARENCY
            ]

        return constraints

    def _determine_required_virtues(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Determine which virtues are most important for this query"""
        required = []

        # Analyze query characteristics
        query_lower = query.lower()

        if any(word in query_lower for word in ["uncertain", "maybe", "possibly"]):
            required.append("epistemic_humility")

        if any(word in query_lower for word in ["challenge", "question", "debate"]):
            required.append("intellectual_courage")

        if any(word in query_lower for word in ["perspective", "viewpoint", "consider"]):
            required.append("perspectival_flexibility")

        if any(word in query_lower for word in ["analyze", "examine", "investigate"]):
            required.append("intellectual_thoroughness")

        if any(word in query_lower for word in ["correct", "revise", "improve"]):
            required.append("self_correction")

        # Default if none detected
        if not required:
            required = ["epistemic_humility", "intellectual_thoroughness"]

        return required

    def _suggest_ncf_adjustments(self, response: ReasoningResponse,
                                 current_ncf: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest NCF parameter adjustments based on reasoning results"""
        suggestions = {}

        # Adjust based on confidence
        if response.confidence < 0.4:
            suggestions["conceptual_entropy"] = min(
                current_ncf.get("conceptual_entropy", 0.5) + 0.1,
                1.0
            )
            suggestions["narrative_coherence"] = max(
                current_ncf.get("narrative_coherence", 0.5) - 0.1,
                0.0
            )
        elif response.confidence > 0.8:
            suggestions["conceptual_entropy"] = max(
                current_ncf.get("conceptual_entropy", 0.5) - 0.1,
                0.0
            )

        # Adjust based on virtue assessment
        if response.virtue_assessment.overall_virtue_score < 0.5:
            suggestions["virtue_emphasis"] = min(
                current_ncf.get("virtue_emphasis", 0.5) + 0.2,
                1.0
            )

        return suggestions