# ceaf_core/modules/ncim_engine/narrative_thread_manager.py

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import re

logger = logging.getLogger(__name__)


class ThreadStatus(Enum):
    """Status of narrative threads"""
    ACTIVE = "active"
    DORMANT = "dormant"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"
    CONFLICTED = "conflicted"


class ThreadPriority(Enum):
    """Priority levels for narrative threads"""
    CRITICAL = "critical"  # Core identity threads
    HIGH = "high"  # Important ongoing narratives
    NORMAL = "normal"  # Regular threads
    LOW = "low"  # Background threads
    MINIMAL = "minimal"  # Barely active threads


class NarrativeArcType(Enum):
    """Types of narrative arcs"""
    DEVELOPMENT = "development"  # Growth/learning arc
    CONFLICT = "conflict"  # Conflict resolution arc
    EXPLORATION = "exploration"  # Discovery/curiosity arc
    RELATIONSHIP = "relationship"  # Social/interaction arc
    ACHIEVEMENT = "achievement"  # Goal accomplishment arc
    REFLECTION = "reflection"  # Self-examination arc


@dataclass
class NarrativeEvent:
    """Represents an event in a narrative thread"""
    event_id: str
    timestamp: datetime
    event_type: str  # "interaction", "reflection", "decision", "outcome"
    description: str
    significance: float  # 0-1, how significant this event is
    emotional_tone: str = "neutral"
    outcomes: List[str] = field(default_factory=list)
    related_components: List[str] = field(default_factory=list)


@dataclass
class NarrativeArc:
    """Represents a complete narrative arc within a thread"""
    arc_id: str
    arc_type: NarrativeArcType
    theme: str
    start_event_id: str
    end_event_id: Optional[str] = None
    events: List[str] = field(default_factory=list)  # Event IDs
    resolution_quality: Optional[float] = None  # 0-1 if resolved
    lessons_learned: List[str] = field(default_factory=list)


@dataclass
class ThreadDivergence:
    """Represents a point where narrative threads diverge or converge"""
    divergence_id: str
    timestamp: datetime
    divergence_type: str  # "split", "merge", "branch", "converge"
    parent_threads: List[str]
    child_threads: List[str]
    trigger_event: str
    significance: float


class NarrativeThreadManager:
    """
    Manages narrative threads, tracks storylines, and ensures narrative coherence.
    Handles thread lifecycle, arc development, and narrative consistency.
    """

    def __init__(self):
        self.threads: Dict[str, Any] = {}  # Will store enhanced thread objects
        self.narrative_events: Dict[str, NarrativeEvent] = {}
        self.narrative_arcs: Dict[str, NarrativeArc] = {}
        self.thread_divergences: Dict[str, ThreadDivergence] = {}

        # Thread management parameters
        self.max_active_threads = 10
        self.thread_decay_rate = 0.05  # How quickly inactive threads decay
        self.coherence_threshold = 0.6
        self.significance_threshold = 0.4

        # Narrative patterns and templates
        self.narrative_patterns = self._load_narrative_patterns()
        self.arc_templates = self._load_arc_templates()

        logger.info("Narrative Thread Manager initialized")

    def _load_narrative_patterns(self) -> Dict[str, Any]:
        """Load common narrative patterns for recognition"""
        return {
            "hero_journey": {
                "stages": ["call_to_adventure", "refusal", "mentor", "crossing_threshold",
                           "tests", "ordeal", "reward", "return"],
                "indicators": ["challenge", "guidance", "growth", "transformation"]
            },
            "problem_solving": {
                "stages": ["problem_identification", "exploration", "insight", "solution", "validation"],
                "indicators": ["question", "confusion", "understanding", "resolution"]
            },
            "relationship_building": {
                "stages": ["initial_contact", "discovery", "trust_building", "deeper_connection", "maintenance"],
                "indicators": ["introduction", "sharing", "empathy", "understanding", "support"]
            },
            "learning_journey": {
                "stages": ["ignorance", "awareness", "understanding", "application", "mastery"],
                "indicators": ["confusion", "curiosity", "insight", "practice", "expertise"]
            },
            "conflict_resolution": {
                "stages": ["tension", "escalation", "crisis", "resolution", "reconciliation"],
                "indicators": ["disagreement", "friction", "confrontation", "compromise", "harmony"]
            }
        }

    def _load_arc_templates(self) -> Dict[str, Any]:
        """Load templates for different narrative arc types"""
        return {
            NarrativeArcType.DEVELOPMENT: {
                "typical_duration": timedelta(days=7),
                "key_phases": ["baseline", "challenge", "growth", "integration"],
                "success_indicators": ["skill_improvement", "knowledge_gain", "confidence_increase"]
            },
            NarrativeArcType.CONFLICT: {
                "typical_duration": timedelta(days=3),
                "key_phases": ["tension", "confrontation", "resolution"],
                "success_indicators": ["understanding", "compromise", "harmony"]
            },
            NarrativeArcType.EXPLORATION: {
                "typical_duration": timedelta(days=5),
                "key_phases": ["curiosity", "investigation", "discovery", "understanding"],
                "success_indicators": ["new_knowledge", "insights", "connections"]
            },
            NarrativeArcType.RELATIONSHIP: {
                "typical_duration": timedelta(days=14),
                "key_phases": ["introduction", "rapport", "trust", "depth"],
                "success_indicators": ["understanding", "empathy", "connection"]
            },
            NarrativeArcType.ACHIEVEMENT: {
                "typical_duration": timedelta(days=10),
                "key_phases": ["goal_setting", "planning", "execution", "completion"],
                "success_indicators": ["progress", "milestones", "success"]
            },
            NarrativeArcType.REFLECTION: {
                "typical_duration": timedelta(days=2),
                "key_phases": ["observation", "analysis", "insight", "integration"],
                "success_indicators": ["self_awareness", "understanding", "wisdom"]
            }
        }

    def create_narrative_thread(self, theme: str, initial_event: str,
                                priority: ThreadPriority = ThreadPriority.NORMAL,
                                related_components: List[str] = None) -> str:
        """
        Create a new narrative thread.

        Args:
            theme: Main theme/topic of the thread
            initial_event: Description of the initiating event
            priority: Priority level of the thread
            related_components: Related identity components

        Returns:
            thread_id: Unique identifier for the thread
        """
        thread_id = f"thread_{len(self.threads)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create initial event
        initial_event_id = self.add_narrative_event(
            thread_id=thread_id,
            event_type="initiation",
            description=initial_event,
            significance=0.8,
            related_components=related_components or []
        )

        # Create thread object
        thread = {
            "thread_id": thread_id,
            "theme": theme,
            "priority": priority,
            "status": ThreadStatus.ACTIVE,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "events": [initial_event_id],
            "arcs": [],
            "coherence_score": 1.0,
            "activity_level": 1.0,
            "related_components": related_components or [],
            "tags": self._extract_tags_from_theme(theme),
            "narrative_momentum": 0.8,  # How much narrative drive this thread has
            "complexity_level": 0.1,  # How complex the narrative has become
            "emotional_tone_history": ["neutral"],
            "branching_points": [],  # Points where thread could branch
            "convergence_opportunities": []  # Opportunities to merge with other threads
        }

        self.threads[thread_id] = thread

        # Detect potential narrative pattern
        pattern = self._detect_narrative_pattern(theme, initial_event)
        if pattern:
            thread["detected_pattern"] = pattern

        logger.info(f"Created narrative thread {thread_id}: {theme}")

        return thread_id

    def add_narrative_event(self, thread_id: str, event_type: str, description: str,
                            significance: float, emotional_tone: str = "neutral",
                            outcomes: List[str] = None,
                            related_components: List[str] = None) -> str:
        """
        Add a new event to a narrative thread.

        Args:
            thread_id: ID of the thread to add event to
            event_type: Type of event
            description: Description of the event
            significance: Significance score (0-1)
            emotional_tone: Emotional tone of the event
            outcomes: Outcomes or consequences of the event
            related_components: Related identity components

        Returns:
            event_id: Unique identifier for the event
        """
        event_id = f"event_{len(self.narrative_events)}_{datetime.now().strftime('%H%M%S')}"

        event = NarrativeEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            description=description,
            significance=significance,
            emotional_tone=emotional_tone,
            outcomes=outcomes or [],
            related_components=related_components or []
        )

        self.narrative_events[event_id] = event

        # Update thread if it exists
        if thread_id in self.threads:
            thread = self.threads[thread_id]
            thread["events"].append(event_id)
            thread["last_activity"] = datetime.now()
            thread["activity_level"] = min(1.0, thread["activity_level"] + 0.2)
            thread["emotional_tone_history"].append(emotional_tone)

            # Update complexity based on event significance
            thread["complexity_level"] = min(1.0, thread["complexity_level"] + significance * 0.1)

            # Update narrative momentum
            momentum_change = significance * 0.3
            if emotional_tone in ["positive", "exciting", "hopeful"]:
                momentum_change *= 1.2
            elif emotional_tone in ["negative", "frustrating", "disappointing"]:
                momentum_change *= 0.8

            thread["narrative_momentum"] = max(0.1, min(1.0,
                                                        thread["narrative_momentum"] + momentum_change - 0.05
                                                        # Natural decay
                                                        ))

            # Recalculate coherence
            thread["coherence_score"] = self._calculate_thread_coherence(thread_id)

            # Check for arc completion or new arc initiation
            self._update_narrative_arcs(thread_id, event_id)

            # Detect branching opportunities
            if significance > 0.7:
                self._detect_branching_opportunities(thread_id, event_id)

        logger.info(f"Added event {event_id} to thread {thread_id}")

        return event_id

    def _extract_tags_from_theme(self, theme: str) -> List[str]:
        """Extract tags from thread theme for categorization"""
        theme_lower = theme.lower()

        # Common narrative tags
        tag_patterns = {
            "learning": ["learn", "study", "understand", "knowledge", "skill"],
            "problem_solving": ["problem", "challenge", "solve", "fix", "resolve"],
            "creativity": ["creative", "art", "design", "innovative", "imagination"],
            "relationship": ["friend", "relationship", "social", "communication", "connect"],
            "growth": ["improve", "develop", "grow", "progress", "advance"],
            "conflict": ["conflict", "disagree", "tension", "friction", "dispute"],
            "exploration": ["explore", "discover", "investigate", "research", "find"],
            "reflection": ["reflect", "think", "consider", "contemplate", "ponder"]
        }

        tags = []
        for tag, keywords in tag_patterns.items():
            if any(keyword in theme_lower for keyword in keywords):
                tags.append(tag)

        return tags

    def _detect_narrative_pattern(self, theme: str, initial_event: str) -> Optional[str]:
        """Detect which narrative pattern this thread might follow"""
        combined_text = f"{theme} {initial_event}".lower()

        pattern_scores = {}
        for pattern_name, pattern_info in self.narrative_patterns.items():
            score = 0
            for indicator in pattern_info["indicators"]:
                if indicator in combined_text:
                    score += 1
            pattern_scores[pattern_name] = score / len(pattern_info["indicators"])

        # Return pattern with highest score if above threshold
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            if pattern_scores[best_pattern] > 0.3:
                return best_pattern

        return None

    def _calculate_thread_coherence(self, thread_id: str) -> float:
        """Calculate coherence score for a narrative thread"""
        if thread_id not in self.threads:
            return 0.0

        thread = self.threads[thread_id]
        events = [self.narrative_events[eid] for eid in thread["events"] if eid in self.narrative_events]

        if len(events) < 2:
            return 1.0

        coherence_factors = []

        # Temporal coherence (events should flow logically in time)
        temporal_coherence = 1.0
        for i in range(1, len(events)):
            time_gap = (events[i].timestamp - events[i - 1].timestamp).total_seconds()
            if time_gap < 0:  # Events out of order
                temporal_coherence -= 0.2
        coherence_factors.append(max(0.0, temporal_coherence))

        # Thematic coherence (events should relate to thread theme)
        theme_keywords = set(thread["theme"].lower().split())
        thematic_coherence = 0.0
        for event in events:
            event_keywords = set(event.description.lower().split())
            overlap = len(theme_keywords.intersection(event_keywords))
            thematic_coherence += overlap / max(len(theme_keywords), 1)
        thematic_coherence /= len(events)
        coherence_factors.append(min(1.0, thematic_coherence))

        # Emotional coherence (emotional tone should make sense)
        emotional_coherence = 1.0
        emotion_transitions = {
            ("positive", "negative"): 0.7,  # Reasonable transition
            ("negative", "positive"): 0.8,  # Growth/recovery
            ("neutral", "positive"): 0.9,  # Good development
            ("neutral", "negative"): 0.8,  # Reasonable decline
            ("positive", "neutral"): 0.9,  # Settling
            ("negative", "neutral"): 0.9  # Recovery
        }

        emotional_tones = [event.emotional_tone for event in events]
        for i in range(1, len(emotional_tones)):
            transition = (emotional_tones[i - 1], emotional_tones[i])
            if transition in emotion_transitions:
                emotional_coherence *= emotion_transitions[transition]
            elif emotional_tones[i - 1] == emotional_tones[i]:
                emotional_coherence *= 0.95  # Slight penalty for no change
        coherence_factors.append(emotional_coherence)

        # Significance coherence (events should have reasonable significance distribution)
        significances = [event.significance for event in events]
        if significances:
            significance_variance = statistics.variance(significances) if len(significances) > 1 else 0
            # Lower variance is more coherent (not all events should be extremely significant)
            significance_coherence = max(0.3, 1.0 - significance_variance)
            coherence_factors.append(significance_coherence)

        return statistics.mean(coherence_factors)

    def _update_narrative_arcs(self, thread_id: str, new_event_id: str) -> None:
        """Update narrative arcs based on new events"""
        if thread_id not in self.threads:
            return

        thread = self.threads[thread_id]
        new_event = self.narrative_events[new_event_id]

        # Check if current arc should be completed
        current_arcs = [arc for arc in thread["arcs"] if
                        arc in self.narrative_arcs and not self.narrative_arcs[arc].end_event_id]

        for arc_id in current_arcs:
            arc = self.narrative_arcs[arc_id]
            template = self.arc_templates[arc.arc_type]

            # Check if arc should be completed
            arc_duration = datetime.now() - self.narrative_events[arc.start_event_id].timestamp
            if (arc_duration > template["typical_duration"] * 1.5 or
                    new_event.significance > 0.8 and "resolution" in new_event.description.lower()):
                self._complete_narrative_arc(arc_id, new_event_id)

        # Check if new arc should be started
        if new_event.significance > 0.6:
            arc_type = self._determine_arc_type(new_event, thread)
            if arc_type:
                self._start_narrative_arc(thread_id, arc_type, new_event_id)

    def _determine_arc_type(self, event: NarrativeEvent, thread: Dict) -> Optional[NarrativeArcType]:
        """Determine what type of narrative arc this event might start"""
        description_lower = event.description.lower()

        arc_indicators = {
            NarrativeArcType.DEVELOPMENT: ["learn", "improve", "develop", "grow", "skill", "ability"],
            NarrativeArcType.CONFLICT: ["conflict", "disagree", "problem", "tension", "challenge"],
            NarrativeArcType.EXPLORATION: ["explore", "discover", "investigate", "research", "curious"],
            NarrativeArcType.RELATIONSHIP: ["meet", "friend", "connect", "relationship", "social"],
            NarrativeArcType.ACHIEVEMENT: ["goal", "accomplish", "achieve", "complete", "finish"],
            NarrativeArcType.REFLECTION: ["think", "reflect", "consider", "realize", "understand"]
        }

        scores = {}
        for arc_type, indicators in arc_indicators.items():
            score = sum(1 for indicator in indicators if indicator in description_lower)
            if score > 0:
                scores[arc_type] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    def _start_narrative_arc(self, thread_id: str, arc_type: NarrativeArcType, start_event_id: str) -> str:
        """Start a new narrative arc"""
        arc_id = f"arc_{len(self.narrative_arcs)}_{arc_type.value}_{datetime.now().strftime('%H%M%S')}"

        # Determine theme based on event and arc type
        start_event = self.narrative_events[start_event_id]
        arc_theme = f"{arc_type.value.replace('_', ' ').title()}: {start_event.description[:50]}..."

        arc = NarrativeArc(
            arc_id=arc_id,
            arc_type=arc_type,
            theme=arc_theme,
            start_event_id=start_event_id,
            events=[start_event_id]
        )

        self.narrative_arcs[arc_id] = arc

        # Add to thread
        if thread_id in self.threads:
            self.threads[thread_id]["arcs"].append(arc_id)

        logger.info(f"Started {arc_type.value} arc {arc_id} in thread {thread_id}")

        return arc_id

    def _complete_narrative_arc(self, arc_id: str, end_event_id: str) -> None:
        """Complete a narrative arc"""
        if arc_id not in self.narrative_arcs:
            return

        arc = self.narrative_arcs[arc_id]
        arc.end_event_id = end_event_id

        # Calculate resolution quality
        arc.resolution_quality = self._calculate_arc_resolution_quality(arc_id)

        # Extract lessons learned
        arc.lessons_learned = self._extract_arc_lessons(arc_id)

        logger.info(f"Completed arc {arc_id} with resolution quality {arc.resolution_quality:.2f}")

    def _calculate_arc_resolution_quality(self, arc_id: str) -> float:
        """Calculate how well an arc was resolved"""
        if arc_id not in self.narrative_arcs:
            return 0.0

        arc = self.narrative_arcs[arc_id]
        template = self.arc_templates[arc.arc_type]

        # Get arc events
        arc_events = [self.narrative_events[eid] for eid in arc.events if eid in self.narrative_events]

        if not arc_events:
            return 0.0

        resolution_factors = []

        # Check if success indicators were met
        success_indicators = template["success_indicators"]
        indicator_score = 0
        for event in arc_events:
            for indicator in success_indicators:
                if indicator.replace("_", " ") in event.description.lower():
                    indicator_score += 1
        resolution_factors.append(min(1.0, indicator_score / len(success_indicators)))

        # Check emotional trajectory
        emotional_tones = [event.emotional_tone for event in arc_events]
        positive_ending = emotional_tones[-1] in ["positive", "neutral"] if emotional_tones else False
        resolution_factors.append(0.8 if positive_ending else 0.4)

        # Check significance of resolution
        if arc.end_event_id:
            end_event = self.narrative_events[arc.end_event_id]
            resolution_factors.append(end_event.significance)
        else:
            resolution_factors.append(0.3)  # No clear ending

        return statistics.mean(resolution_factors)

    def _extract_arc_lessons(self, arc_id: str) -> List[str]:
        """Extract lessons learned from a completed arc"""
        if arc_id not in self.narrative_arcs:
            return []

        arc = self.narrative_arcs[arc_id]
        lessons = []

        # Arc-type specific lesson extraction
        if arc.arc_type == NarrativeArcType.DEVELOPMENT:
            lessons.append("Growth requires consistent effort and learning from challenges")
        elif arc.arc_type == NarrativeArcType.CONFLICT:
            lessons.append("Conflicts can be resolved through understanding and communication")
        elif arc.arc_type == NarrativeArcType.EXPLORATION:
            lessons.append("Curiosity leads to new discoveries and understanding")
        elif arc.arc_type == NarrativeArcType.RELATIONSHIP:
            lessons.append("Relationships develop through trust and shared experiences")
        elif arc.arc_type == NarrativeArcType.ACHIEVEMENT:
            lessons.append("Goals are achieved through planning and persistent effort")
        elif arc.arc_type == NarrativeArcType.REFLECTION:
            lessons.append("Self-reflection leads to greater self-awareness and wisdom")

        # Add specific lessons based on arc outcomes
        if arc.resolution_quality > 0.8:
            lessons.append("This experience was handled particularly well")
        elif arc.resolution_quality < 0.4:
            lessons.append("This situation could have been handled better")

        return lessons

    def _detect_branching_opportunities(self, thread_id: str, event_id: str) -> None:
        """Detect opportunities for thread branching"""
        if thread_id not in self.threads:
            return

        thread = self.threads[thread_id]
        event = self.narrative_events[event_id]

        # High significance events can create branching opportunities
        if event.significance > 0.7:
            branching_point = {
                "event_id": event_id,
                "timestamp": event.timestamp,
                "potential_branches": self._identify_potential_branches(event),
                "branching_probability": event.significance * 0.8
            }
            thread["branching_points"].append(branching_point)

    def _identify_potential_branches(self, event: NarrativeEvent) -> List[str]:
        """Identify potential narrative branches from an event"""
        branches = []
        description_lower = event.description.lower()

        branch_triggers = {
            "new_capability": ["learned", "discovered", "realized", "can now"],
            "new_challenge": ["problem", "difficulty", "challenge", "obstacle"],
            "new_relationship": ["met", "introduced", "connected", "friend"],
            "new_interest": ["interested", "curious", "fascinated", "want to learn"],
            "decision_point": ["choose", "decide", "option", "alternative"]
        }

        for branch_type, triggers in branch_triggers.items():
            if any(trigger in description_lower for trigger in triggers):
                branches.append(branch_type)

        return branches

    def merge_narrative_threads(self, primary_thread_id: str, secondary_thread_id: str,
                                merge_reason: str) -> Dict[str, Any]:
        """
        Merge two narrative threads.

        Args:
            primary_thread_id: ID of the primary thread (will absorb the secondary)
            secondary_thread_id: ID of the secondary thread (will be merged into primary)
            merge_reason: Reason for the merge

        Returns:
            Merge results
        """
        if primary_thread_id not in self.threads or secondary_thread_id not in self.threads:
            return {"error": "One or both threads not found"}

        primary = self.threads[primary_thread_id]
        secondary = self.threads[secondary_thread_id]

        # Create divergence record
        divergence_id = f"merge_{len(self.thread_divergences)}_{datetime.now().strftime('%H%M%S')}"
        divergence = ThreadDivergence(
            divergence_id=divergence_id,
            timestamp=datetime.now(),
            divergence_type="merge",
            parent_threads=[secondary_thread_id],
            child_threads=[primary_thread_id],
            trigger_event=f"Merge: {merge_reason}",
            significance=0.7
        )
        self.thread_divergences[divergence_id] = divergence

        # Merge thread data
        merge_results = {
            "merged_events": len(secondary["events"]),
            "merged_arcs": len(secondary["arcs"]),
            "new_theme": f"{primary['theme']} + {secondary['theme']}",
            "combined_tags": list(set(primary["tags"] + secondary["tags"]))
        }

        # Merge events
        primary["events"].extend(secondary["events"])

        # Merge arcs
        primary["arcs"].extend(secondary["arcs"])

        # Merge related components
        primary["related_components"] = list(set(primary["related_components"] + secondary["related_components"]))

        # Update primary thread metadata
        primary["theme"] = merge_results["new_theme"]
        primary["tags"] = merge_results["combined_tags"]
        primary["complexity_level"] = min(1.0, primary["complexity_level"] + secondary["complexity_level"] * 0.5)
        primary["last_activity"] = max(primary["last_activity"], secondary["last_activity"])

        # Recalculate coherence
        primary["coherence_score"] = self._calculate_thread_coherence(primary_thread_id)

        # Remove secondary thread
        del self.threads[secondary_thread_id]

        logger.info(f"Merged thread {secondary_thread_id} into {primary_thread_id}")

        return merge_results

    def branch_narrative_thread(self, parent_thread_id: str, branching_event_id: str,
                                new_theme: str, branch_reason: str) -> str:
        """
        Create a new thread by branching from an existing one.

        Args:
            parent_thread_id: ID of the parent thread
            branching_event_id: Event that triggers the branch
            new_theme: Theme for the new branch
            branch_reason: Reason for branching

        Returns:
            new_thread_id: ID of the newly created branch thread
        """
        if parent_thread_id not in self.threads:
            raise ValueError(f"Parent thread {parent_thread_id} not found")

        parent = self.threads[parent_thread_id]

        # Create new branch thread
        new_thread_id = self.create_narrative_thread(
            theme=new_theme,
            initial_event=f"Branched from {parent['theme']}: {branch_reason}",
            priority=parent["priority"],
            related_components=parent["related_components"].copy()
        )

        # Copy relevant context from parent
        new_thread = self.threads[new_thread_id]
        new_thread["tags"] = parent["tags"].copy()
        new_thread["complexity_level"] = parent["complexity_level"] * 0.7
        new_thread["narrative_momentum"] = parent["narrative_momentum"] * 0.8

        # Create divergence record
        divergence_id = f"branch_{len(self.thread_divergences)}_{datetime.now().strftime('%H%M%S')}"
        divergence = ThreadDivergence(
            divergence_id=divergence_id,
            timestamp=datetime.now(),
            divergence_type="branch",
            parent_threads=[parent_thread_id],
            child_threads=[new_thread_id],
            trigger_event=branching_event_id,
            significance=0.6
        )
        self.thread_divergences[divergence_id] = divergence

        logger.info(f"Branched thread {new_thread_id} from {parent_thread_id}")

        return new_thread_id

    def update_thread_priorities(self) -> Dict[str, Any]:
        """Update thread priorities based on activity and significance"""
        updates = {"priority_changes": [], "status_changes": []}

        for thread_id, thread in self.threads.items():
            old_priority = thread["priority"]
            old_status = thread["status"]

            # Calculate new priority based on activity, momentum, and coherence
            activity_score = thread["activity_level"] * 0.4
            momentum_score = thread["narrative_momentum"] * 0.3
            coherence_score = thread["coherence_score"] * 0.2
            complexity_score = thread["complexity_level"] * 0.1

            total_score = activity_score + momentum_score + coherence_score + complexity_score

            # Determine new priority
            if total_score > 0.8:
                new_priority = ThreadPriority.HIGH
            elif total_score > 0.6:
                new_priority = ThreadPriority.NORMAL
            elif total_score > 0.4:
                new_priority = ThreadPriority.LOW
            else:
                new_priority = ThreadPriority.MINIMAL

            # Special case for critical threads (never downgrade below normal)
            if old_priority == ThreadPriority.CRITICAL:
                new_priority = max(new_priority, ThreadPriority.NORMAL)

            # Update priority if changed
            if new_priority != old_priority:
                thread["priority"] = new_priority
                updates["priority_changes"].append({
                    "thread_id": thread_id,
                    "old_priority": old_priority.value,
                    "new_priority": new_priority.value,
                    "reason": f"Score: {total_score:.2f}"
                })

            # Update status based on activity and time
            time_since_activity = datetime.now() - thread["last_activity"]

            new_status = thread["status"]
            if time_since_activity > timedelta(days=7) and thread["activity_level"] < 0.2:
                new_status = ThreadStatus.DORMANT
            elif time_since_activity > timedelta(days=30):
                new_status = ThreadStatus.ABANDONED
            elif thread["coherence_score"] < 0.4:
                new_status = ThreadStatus.CONFLICTED
            elif thread["activity_level"] > 0.6:
                new_status = ThreadStatus.ACTIVE

            if new_status != old_status:
                thread["status"] = new_status
                updates["status_changes"].append({
                    "thread_id": thread_id,
                    "old_status": old_status.value,
                    "new_status": new_status.value
                })

        return updates

    def decay_inactive_threads(self) -> Dict[str, Any]:
        """Apply decay to inactive threads"""
        decay_results = {"threads_decayed": [], "threads_archived": []}

        for thread_id, thread in list(self.threads.items()):
            time_since_activity = datetime.now() - thread["last_activity"]

            if time_since_activity > timedelta(hours=12):  # Start decay after 12 hours
                # Apply decay to activity level and momentum
                decay_amount = self.thread_decay_rate * (time_since_activity.total_seconds() / 3600)

                old_activity = thread["activity_level"]
                old_momentum = thread["narrative_momentum"]

                thread["activity_level"] = max(0.0, thread["activity_level"] - decay_amount)
                thread["narrative_momentum"] = max(0.1, thread["narrative_momentum"] - decay_amount * 0.5)

                if old_activity != thread["activity_level"] or old_momentum != thread["narrative_momentum"]:
                    decay_results["threads_decayed"].append({
                        "thread_id": thread_id,
                        "activity_change": thread["activity_level"] - old_activity,
                        "momentum_change": thread["narrative_momentum"] - old_momentum
                    })

                # Archive very inactive threads
                if (thread["activity_level"] < 0.1 and
                        thread["narrative_momentum"] < 0.2 and
                        time_since_activity > timedelta(days=14)):
                    self._archive_thread(thread_id)
                    decay_results["threads_archived"].append(thread_id)

        return decay_results

    def _archive_thread(self, thread_id: str) -> None:
        """Archive an inactive thread"""
        if thread_id in self.threads:
            thread = self.threads[thread_id]
            thread["status"] = ThreadStatus.ABANDONED
            thread["archived_at"] = datetime.now()
            # Could move to separate archive storage in full implementation
            logger.info(f"Archived inactive thread {thread_id}")

    def get_narrative_summary(self, thread_id: str = None) -> Dict[str, Any]:
        """
        Get narrative summary for a specific thread or all threads.

        Args:
            thread_id: Specific thread ID, or None for system-wide summary

        Returns:
            Narrative summary
        """
        if thread_id:
            return self._get_single_thread_summary(thread_id)
        else:
            return self._get_system_narrative_summary()

    def _get_single_thread_summary(self, thread_id: str) -> Dict[str, Any]:
        """Get summary for a single thread"""
        if thread_id not in self.threads:
            return {"error": f"Thread {thread_id} not found"}

        thread = self.threads[thread_id]

        # Get thread events
        events = [self.narrative_events[eid] for eid in thread["events"] if eid in self.narrative_events]

        # Get thread arcs
        arcs = [self.narrative_arcs[aid] for aid in thread["arcs"] if aid in self.narrative_arcs]

        return {
            "thread_id": thread_id,
            "theme": thread["theme"],
            "status": thread["status"].value,
            "priority": thread["priority"].value,
            "coherence_score": thread["coherence_score"],
            "activity_level": thread["activity_level"],
            "narrative_momentum": thread["narrative_momentum"],
            "complexity_level": thread["complexity_level"],
            "event_count": len(events),
            "arc_count": len(arcs),
            "completed_arcs": len([arc for arc in arcs if arc.end_event_id]),
            "tags": thread["tags"],
            "emotional_trajectory": thread["emotional_tone_history"][-5:],  # Last 5 emotions
            "recent_events": [
                {
                    "description": event.description,
                    "significance": event.significance,
                    "emotional_tone": event.emotional_tone,
                    "timestamp": event.timestamp.isoformat()
                }
                for event in sorted(events, key=lambda e: e.timestamp, reverse=True)[:3]
            ],
            "key_lessons": [
                               lesson for arc in arcs for lesson in arc.lessons_learned
                           ][:5]
        }

    def _get_system_narrative_summary(self) -> Dict[str, Any]:
        """Get system-wide narrative summary"""
        all_threads = list(self.threads.values())

        # Status distribution
        status_counts = {}
        for thread in all_threads:
            status = thread["status"].value
            status_counts[status] = status_counts.get(status, 0) + 1

        # Priority distribution
        priority_counts = {}
        for thread in all_threads:
            priority = thread["priority"].value
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        # Average metrics
        avg_coherence = statistics.mean([t["coherence_score"] for t in all_threads]) if all_threads else 0
        avg_activity = statistics.mean([t["activity_level"] for t in all_threads]) if all_threads else 0
        avg_momentum = statistics.mean([t["narrative_momentum"] for t in all_threads]) if all_threads else 0

        # Most active threads
        most_active = sorted(all_threads, key=lambda t: t["activity_level"], reverse=True)[:5]

        # Most coherent threads
        most_coherent = sorted(all_threads, key=lambda t: t["coherence_score"], reverse=True)[:5]

        return {
            "total_threads": len(all_threads),
            "total_events": len(self.narrative_events),
            "total_arcs": len(self.narrative_arcs),
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "average_coherence": avg_coherence,
            "average_activity": avg_activity,
            "average_momentum": avg_momentum,
            "most_active_threads": [
                {"id": t["thread_id"], "theme": t["theme"], "activity": t["activity_level"]}
                for t in most_active
            ],
            "most_coherent_threads": [
                {"id": t["thread_id"], "theme": t["theme"], "coherence": t["coherence_score"]}
                for t in most_coherent
            ],
            "narrative_health_score": self._calculate_narrative_health_score()
        }

    def _calculate_narrative_health_score(self) -> float:
        """Calculate overall narrative health score (0-1)"""
        if not self.threads:
            return 1.0

        health_factors = []

        # Coherence factor
        coherence_scores = [t["coherence_score"] for t in self.threads.values()]
        avg_coherence = statistics.mean(coherence_scores)
        health_factors.append(avg_coherence)

        # Activity factor
        activity_scores = [t["activity_level"] for t in self.threads.values()]
        avg_activity = statistics.mean(activity_scores)
        health_factors.append(avg_activity)

        # Status factor (proportion of healthy statuses)
        healthy_statuses = [ThreadStatus.ACTIVE, ThreadStatus.RESOLVING]
        healthy_count = sum(1 for t in self.threads.values() if t["status"] in healthy_statuses)
        status_factor = healthy_count / len(self.threads)
        health_factors.append(status_factor)

        # Complexity factor (moderate complexity is healthy)
        complexity_scores = [t["complexity_level"] for t in self.threads.values()]
        avg_complexity = statistics.mean(complexity_scores)
        # Ideal complexity is around 0.5-0.7
        complexity_factor = 1.0 - abs(avg_complexity - 0.6)
        health_factors.append(max(0.0, complexity_factor))

        return statistics.mean(health_factors)
