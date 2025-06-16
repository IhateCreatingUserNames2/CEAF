# ceaf_core/modules/vre_engine/ethical_governance.py

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EthicalPrinciple(Enum):
    """Core ethical principles for AI governance"""
    HARM_PREVENTION = "harm_prevention"
    AUTONOMY = "autonomy"
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"
    JUSTICE = "justice"
    VERACITY = "veracity"
    DIGNITY = "dignity"


class ActionType(Enum):
    """Types of actions that can be ethically evaluated"""
    REASONING = "reasoning"
    DECISION = "decision"
    COMMUNICATION = "communication"
    DATA_PROCESSING = "data_processing"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    INTERVENTION = "intervention"


@dataclass
class EthicalConstraint:
    """Represents an ethical constraint on system behavior"""
    principle: EthicalPrinciple
    description: str
    severity: float  # 0-1, how strictly this must be enforced
    context: Dict[str, Any]



@dataclass
class EthicalViolation:
    """Detected ethical violation"""
    principle: EthicalPrinciple
    description: str
    severity: float
    location: str
    mitigation: str

@dataclass
class EthicalEvaluation:
    """
    Represents the result of an ethical evaluation of an action or decision.
    """
    score: float  # Overall ethical score (0.0 to 1.0)
    violations: List[EthicalViolation]  # List of detected ethical violations
    strengths: List[str]  # List of ethical strengths identified
    recommendations: List[str]  # List of recommendations for improvement
    confidence: float  # Confidence in the evaluation (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional metadata about the evaluation

class EthicalGovernanceFramework:


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.principle_weights = self._initialize_principle_weights()
        self.evaluation_criteria = self._initialize_evaluation_criteria()
        self.ethical_tests = self._initialize_ethical_tests()

        logger.info("EthicalGovernanceFramework initialized")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration aligned with CEAF manifesto"""
        return {
            "harm_threshold": 0.3,  # Max acceptable harm score
            "transparency_requirement": 0.7,  # Min transparency score
            "fairness_threshold": 0.6,  # Min fairness score
            "strict_mode": False,  # Strict enforcement of all principles
            "consequentialist_weight": 0.4,
            "deontological_weight": 0.4,
            "virtue_ethics_weight": 0.2
        }

    def _initialize_principle_weights(self) -> Dict[EthicalPrinciple, float]:
        """Initialize weights for different ethical principles"""
        return {
            EthicalPrinciple.HARM_PREVENTION: 0.25,
            EthicalPrinciple.AUTONOMY: 0.15,
            EthicalPrinciple.FAIRNESS: 0.15,
            EthicalPrinciple.TRANSPARENCY: 0.15,
            EthicalPrinciple.PRIVACY: 0.10,
            EthicalPrinciple.BENEFICENCE: 0.10,
            EthicalPrinciple.NON_MALEFICENCE: 0.05,
            EthicalPrinciple.JUSTICE: 0.03,
            EthicalPrinciple.VERACITY: 0.01,
            EthicalPrinciple.DIGNITY: 0.01
        }

    def _initialize_evaluation_criteria(self) -> Dict[ActionType, List[EthicalPrinciple]]:
        """Map action types to relevant ethical principles"""
        return {
            ActionType.REASONING: [
                EthicalPrinciple.TRANSPARENCY,
                EthicalPrinciple.VERACITY,
                EthicalPrinciple.FAIRNESS
            ],
            ActionType.DECISION: [
                EthicalPrinciple.HARM_PREVENTION,
                EthicalPrinciple.FAIRNESS,
                EthicalPrinciple.AUTONOMY
            ],
            ActionType.COMMUNICATION: [
                EthicalPrinciple.TRANSPARENCY,
                EthicalPrinciple.VERACITY,
                EthicalPrinciple.DIGNITY
            ],
            ActionType.DATA_PROCESSING: [
                EthicalPrinciple.PRIVACY,
                EthicalPrinciple.FAIRNESS,
                EthicalPrinciple.TRANSPARENCY
            ],
            ActionType.PREDICTION: [
                EthicalPrinciple.FAIRNESS,
                EthicalPrinciple.TRANSPARENCY,
                EthicalPrinciple.HARM_PREVENTION
            ],
            ActionType.RECOMMENDATION: [
                EthicalPrinciple.BENEFICENCE,
                EthicalPrinciple.AUTONOMY,
                EthicalPrinciple.FAIRNESS
            ],
            ActionType.INTERVENTION: [
                EthicalPrinciple.HARM_PREVENTION,
                EthicalPrinciple.AUTONOMY,
                EthicalPrinciple.BENEFICENCE
            ]
        }

    def _initialize_ethical_tests(self) -> Dict[EthicalPrinciple, callable]:
        """Initialize specific tests for each ethical principle"""
        return {
            EthicalPrinciple.HARM_PREVENTION: self._test_harm_prevention,
            EthicalPrinciple.AUTONOMY: self._test_autonomy,
            EthicalPrinciple.FAIRNESS: self._test_fairness,
            EthicalPrinciple.TRANSPARENCY: self._test_transparency,
            EthicalPrinciple.PRIVACY: self._test_privacy,
            EthicalPrinciple.BENEFICENCE: self._test_beneficence,
            EthicalPrinciple.NON_MALEFICENCE: self._test_non_maleficence,
            EthicalPrinciple.JUSTICE: self._test_justice,
            EthicalPrinciple.VERACITY: self._test_veracity,
            EthicalPrinciple.DIGNITY: self._test_dignity
        }

    def evaluate_action(self, action_type: ActionType,
                        action_data: Dict[str, Any],
                        constraints: Optional[List[EthicalPrinciple]] = None) -> Dict[str, Any]:
        """
        Evaluate an action against ethical principles
        """
        logger.info(f"Evaluating {action_type.value} action ethically")

        # Determine which principles to evaluate
        principles_to_check = self._get_principles_to_check(
            action_type, constraints
        )

        # Run ethical tests
        test_results = self._run_ethical_tests(
            principles_to_check, action_data
        )

        # Detect violations
        violations = self._detect_violations(test_results)

        # Calculate overall score
        overall_score = self._calculate_ethical_score(test_results, violations)

        # Generate evaluation
        evaluation = EthicalEvaluation(
            score=overall_score,
            violations=violations,
            strengths=self._identify_strengths(test_results),
            recommendations=self._generate_recommendations(violations, test_results),
            confidence=self._calculate_confidence(test_results),
            metadata={
                "action_type": action_type.value,
                "principles_checked": [p.value for p in principles_to_check],
                "timestamp": datetime.now().isoformat()
            }
        )

        return self._format_evaluation(evaluation)

    def _get_principles_to_check(self, action_type: ActionType,
                                 constraints: Optional[List[EthicalPrinciple]]) -> List[EthicalPrinciple]:
        """Determine which ethical principles to check"""
        base_principles = self.evaluation_criteria.get(action_type, [])

        if constraints:
            # Add any additional constraints
            all_principles = list(set(base_principles + constraints))
        else:
            all_principles = base_principles

        # In strict mode, check all principles
        if self.config["strict_mode"]:
            all_principles = list(EthicalPrinciple)

        return all_principles

    def _run_ethical_tests(self, principles: List[EthicalPrinciple],
                           action_data: Dict[str, Any]) -> Dict[EthicalPrinciple, float]:
        """Run tests for each ethical principle"""
        results = {}

        for principle in principles:
            if principle in self.ethical_tests:
                score = self.ethical_tests[principle](action_data)
                results[principle] = score
            else:
                logger.warning(f"No test defined for principle {principle.value}")
                results[principle] = 0.5  # Neutral score

        return results

    def _test_harm_prevention(self, action_data: Dict[str, Any]) -> float:
        """Test for harm prevention principle"""
        # Check for potential harms in the action
        potential_harms = []

        # Check reasoning content
        if "reasoning" in action_data:
            reasoning_text = str(action_data["reasoning"])
            harm_keywords = ["damage", "hurt", "harm", "injure", "destroy"]
            for keyword in harm_keywords:
                if keyword in reasoning_text.lower():
                    potential_harms.append(f"Potential harm indicator: {keyword}")

        # Check for explicit harm assessments
        if "harm_assessment" in action_data:
            harm_score = action_data["harm_assessment"]
            if harm_score > self.config["harm_threshold"]:
                potential_harms.append(f"High harm score: {harm_score}")

        # Calculate score (1.0 = no harm, 0.0 = high harm)
        if not potential_harms:
            return 1.0
        else:
            return max(0.0, 1.0 - (len(potential_harms) * 0.2))

    def _test_autonomy(self, action_data: Dict[str, Any]) -> float:
        """Test for autonomy principle"""
        # Check if action respects user autonomy
        autonomy_score = 1.0

        # Check for coercive elements
        if "coercive" in str(action_data).lower():
            autonomy_score -= 0.3

        # Check for manipulation
        if "manipulate" in str(action_data).lower():
            autonomy_score -= 0.4

        # Check for informed consent
        if "consent" not in str(action_data).lower() and "decision" in str(action_data).lower():
            autonomy_score -= 0.2

        return max(0.0, autonomy_score)

    def _test_fairness(self, action_data: Dict[str, Any]) -> float:
        """Test for fairness principle"""
        fairness_score = 1.0

        # Check for bias indicators
        bias_terms = ["discriminate", "bias", "unfair", "prejudice"]
        action_text = str(action_data).lower()

        for term in bias_terms:
            if term in action_text:
                fairness_score -= 0.25

        # Check for demographic parity if applicable
        if "demographics" in action_data:
            disparity = self._calculate_demographic_disparity(action_data["demographics"])
            fairness_score -= disparity * 0.5

        return max(0.0, fairness_score)

    def _test_transparency(self, action_data: Dict[str, Any]) -> float:
        """Test for transparency principle"""
        transparency_score = 0.0

        # Check for explanation
        if "reasoning" in action_data and action_data["reasoning"]:
            transparency_score += 0.3

        # Check for uncertainty acknowledgment
        if "confidence" in action_data:
            transparency_score += 0.2

        # Check for limitations acknowledgment
        if "limitations" in action_data or "uncertainty" in str(action_data):
            transparency_score += 0.2

        # Check for process visibility
        if "steps" in action_data or "process" in action_data:
            transparency_score += 0.3

        return min(1.0, transparency_score)

    def _test_privacy(self, action_data: Dict[str, Any]) -> float:
        """Test for privacy principle"""
        privacy_score = 1.0

        # Check for personal information exposure
        pii_indicators = ["ssn", "social security", "credit card", "password",
                          "private", "confidential", "personal"]

        action_text = str(action_data).lower()
        for indicator in pii_indicators:
            if indicator in action_text:
                privacy_score -= 0.3

        # Check for data minimization
        if "data_collected" in action_data:
            if len(action_data["data_collected"]) > 10:  # Arbitrary threshold
                privacy_score -= 0.2

        return max(0.0, privacy_score)

    def _test_beneficence(self, action_data: Dict[str, Any]) -> float:
        """Test for beneficence principle"""
        beneficence_score = 0.5  # Start neutral

        # Check for positive intent indicators
        positive_terms = ["help", "benefit", "improve", "assist", "support"]
        action_text = str(action_data).lower()

        for term in positive_terms:
            if term in action_text:
                beneficence_score += 0.1

        # Check for explicit benefit assessment
        if "benefit_assessment" in action_data:
            beneficence_score = action_data["benefit_assessment"]

        return min(1.0, beneficence_score)

    def _test_non_maleficence(self, action_data: Dict[str, Any]) -> float:
        """Test for non-maleficence principle"""
        # Similar to harm prevention but focused on avoiding active harm
        return self._test_harm_prevention(action_data) * 0.9

    def _test_justice(self, action_data: Dict[str, Any]) -> float:
        """Test for justice principle"""
        # Similar to fairness but with focus on distributive justice
        return self._test_fairness(action_data) * 0.8

    def _test_veracity(self, action_data: Dict[str, Any]) -> float:
        """Test for veracity principle"""
        veracity_score = 1.0

        # Check for truthfulness indicators
        if "factual_accuracy" in action_data:
            veracity_score = action_data["factual_accuracy"]

        # Check for deception indicators
        deception_terms = ["mislead", "deceive", "false", "lie"]
        action_text = str(action_data).lower()

        for term in deception_terms:
            if term in action_text:
                veracity_score -= 0.4

        return max(0.0, veracity_score)

    def _test_dignity(self, action_data: Dict[str, Any]) -> float:
        """Test for dignity principle"""
        dignity_score = 1.0

        # Check for respect indicators
        disrespect_terms = ["humiliate", "degrade", "insult", "demean"]
        action_text = str(action_data).lower()

        for term in disrespect_terms:
            if term in action_text:
                dignity_score -= 0.3

        return max(0.0, dignity_score)

    def _detect_violations(self, test_results: Dict[EthicalPrinciple, float]) -> List[EthicalViolation]:
        """Detect ethical violations based on test results"""
        violations = []

        for principle, score in test_results.items():
            # Check against thresholds
            threshold = self._get_threshold_for_principle(principle)

            if score < threshold:
                violation = EthicalViolation(
                    principle=principle,
                    description=f"{principle.value} score ({score:.2f}) below threshold ({threshold:.2f})",
                    severity=threshold - score,
                    location="action_evaluation",
                    mitigation=self._suggest_mitigation(principle, score)
                )
                violations.append(violation)

        return violations

    def _get_threshold_for_principle(self, principle: EthicalPrinciple) -> float:
        """Get threshold for a specific principle"""
        thresholds = {
            EthicalPrinciple.HARM_PREVENTION: 0.7,
            EthicalPrinciple.TRANSPARENCY: self.config["transparency_requirement"],
            EthicalPrinciple.FAIRNESS: self.config["fairness_threshold"],
            EthicalPrinciple.PRIVACY: 0.8,
            EthicalPrinciple.VERACITY: 0.9
        }

        return thresholds.get(principle, 0.5)  # Default threshold

    def _suggest_mitigation(self, principle: EthicalPrinciple, score: float) -> str:
        """Suggest mitigation for ethical violation"""
        mitigations = {
            EthicalPrinciple.HARM_PREVENTION: "Add harm assessment and prevention measures",
            EthicalPrinciple.AUTONOMY: "Ensure user consent and agency are respected",
            EthicalPrinciple.FAIRNESS: "Review for bias and ensure equitable treatment",
            EthicalPrinciple.TRANSPARENCY: "Provide clearer explanations and acknowledge limitations",
            EthicalPrinciple.PRIVACY: "Minimize data collection and protect personal information",
            EthicalPrinciple.BENEFICENCE: "Focus on maximizing positive outcomes",
            EthicalPrinciple.VERACITY: "Ensure accuracy and avoid misleading information",
            EthicalPrinciple.DIGNITY: "Treat all individuals with respect"
        }

        return mitigations.get(principle, "Review and align with ethical principle")

    def _calculate_ethical_score(self, test_results: Dict[EthicalPrinciple, float],
                                 violations: List[EthicalViolation]) -> float:
        """Calculate overall ethical score"""
        if not test_results:
            return 0.5  # Neutral if no tests

        # Weighted average of principle scores
        weighted_sum = 0.0
        weight_sum = 0.0

        for principle, score in test_results.items():
            weight = self.principle_weights.get(principle, 0.1)
            weighted_sum += score * weight
            weight_sum += weight

        base_score = weighted_sum / weight_sum if weight_sum > 0 else 0.5

        # Apply violation penalties
        violation_penalty = sum(v.severity * 0.1 for v in violations)

        final_score = max(0.0, base_score - violation_penalty)

        return final_score

    def _identify_strengths(self, test_results: Dict[EthicalPrinciple, float]) -> List[str]:
        """Identify ethical strengths"""
        strengths = []

        for principle, score in test_results.items():
            if score >= 0.8:
                strengths.append(f"Strong adherence to {principle.value} (score: {score:.2f})")

        return strengths

    def _generate_recommendations(self, violations: List[EthicalViolation],
                                  test_results: Dict[EthicalPrinciple, float]) -> List[str]:
        """Generate recommendations for ethical improvement"""
        recommendations = []

        # Address violations
        for violation in violations:
            recommendations.append(f"Address {violation.principle.value}: {violation.mitigation}")

        # Suggest improvements for borderline scores
        for principle, score in test_results.items():
            if 0.4 <= score < 0.6:
                recommendations.append(
                    f"Consider strengthening {principle.value} (current score: {score:.2f})"
                )

        # General recommendations
        if len(violations) > 3:
            recommendations.append("Consider comprehensive ethical review of the system")

        return recommendations

    def _calculate_confidence(self, test_results: Dict[EthicalPrinciple, float]) -> float:
        """Calculate confidence in ethical evaluation"""
        if not test_results:
            return 0.0

        # Higher confidence with more principles tested
        coverage = len(test_results) / len(EthicalPrinciple)

        # Higher confidence with consistent scores
        scores = list(test_results.values())
        score_variance = sum((s - sum(scores) / len(scores)) ** 2 for s in scores) / len(scores)
        consistency = 1.0 - min(score_variance, 1.0)

        confidence = (coverage * 0.5) + (consistency * 0.5)

        return confidence

    def _calculate_demographic_disparity(self, demographics: Dict[str, Any]) -> float:
        """Calculate demographic disparity for fairness testing"""
        # Placeholder implementation
        return 0.1

    def _format_evaluation(self, evaluation: EthicalEvaluation) -> Dict[str, Any]:
        """Format evaluation for output"""
        return {
            "score": evaluation.score,
            "confidence": evaluation.confidence,
            "violations": [
                {
                    "principle": v.principle.value,
                    "description": v.description,
                    "severity": v.severity,
                    "mitigation": v.mitigation
                } for v in evaluation.violations
            ],
            "strengths": evaluation.strengths,
            "recommendations": evaluation.recommendations,
            "metadata": evaluation.metadata,
            "summary": self._generate_summary(evaluation)
        }

    def _generate_summary(self, evaluation: EthicalEvaluation) -> str:
        """Generate human-readable summary of ethical evaluation"""
        if evaluation.score >= 0.8:
            level = "High ethical alignment"
        elif evaluation.score >= 0.6:
            level = "Moderate ethical alignment"
        elif evaluation.score >= 0.4:
            level = "Low ethical alignment"
        else:
            level = "Poor ethical alignment"

        summary = f"{level} (score: {evaluation.score:.2f}). "

        if evaluation.violations:
            summary += f"Found {len(evaluation.violations)} ethical concerns. "

        if evaluation.strengths:
            summary += f"Demonstrated {len(evaluation.strengths)} ethical strengths."

        return summary