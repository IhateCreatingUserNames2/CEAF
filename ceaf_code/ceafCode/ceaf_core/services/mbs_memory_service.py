# ceaf_core/services/mbs_memory_service.py
import asyncio
import logging
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, cast, Literal
import re
import numpy as np  # For type hinting if used directly, though similarity is in utils

from google.adk.memory import BaseMemoryService
from google.adk.sessions import Session as AdkSession
from google.adk.events import Event as AdkEvent
from google.genai.types import Part as AdkPart

# --- Embedding Imports ---
from ..utils.embedding_utils import get_embedding_client, compute_adaptive_similarity, EmbeddingClient

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_MEMORY_STORE_PATH = "./data/mbs_memory_store"
MEMORY_STORE_ROOT_DIR = Path(os.getenv("MBS_MEMORY_STORE_PATH", DEFAULT_MEMORY_STORE_PATH))

AGGREGATED_EXPLICIT_MEMORIES_FILENAME = "all_explicit_memories.jsonl"
GOALS_STORE_FILENAME = "all_goal_records.jsonl"
KG_ENTITIES_FILENAME = "all_kg_entities.jsonl"
KG_RELATIONS_FILENAME = "all_kg_relations.jsonl"
# Add filenames for other types if they get dedicated stores
PROCEDURAL_MEMORIES_FILENAME = "all_procedural_memories.jsonl"
EMOTIONAL_MEMORIES_FILENAME = "all_emotional_memories.jsonl"

SESSION_HISTORY_FOR_MEMORY_TURNS = 10
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"

SEMANTIC_CONNECTION_THRESHOLD = float(os.getenv("MBS_SEMANTIC_CONNECTION_THRESHOLD", "0.78")) # Higher threshold for initial auto-connect
KEYWORD_CONNECTION_THRESHOLD_COUNT = int(os.getenv("MBS_KEYWORD_CONNECTION_THRESHOLD_COUNT", "2")) # Min common keywords to connect
MAX_RELATED_IDS_PER_MEMORY = int(os.getenv("MBS_MAX_RELATED_IDS", "10")) # Prevent overly dense connections initially

# --- Scoring Configuration ---
DEFAULT_SEMANTIC_SCORE_WEIGHT = float(os.getenv("MBS_SEMANTIC_SCORE_WEIGHT", "0.6"))
DEFAULT_KEYWORD_SCORE_WEIGHT = float(os.getenv("MBS_KEYWORD_SCORE_WEIGHT", "0.4"))
CONTEXT_QUERY_DELIMITER = " <CEAF_CONTEXT_SEPARATOR> "  # From memory_tools.py

# --- Type Imports and Placeholders ---
MEMORY_TYPES_LOADED_SUCCESSFULLY = False
MEMORY_LIFECYCLE_MANAGER_LOADED = False

try:
    from ..modules.memory_blossom.memory_types import (
        AnyMemoryType, BaseMemory, ExplicitMemory, ExplicitMemoryContent,
        MemorySourceType, MemorySalience, GoalRecord, GoalStatus,  # Added GoalStatus
        KGEntityRecord, KGRelationRecord, KGEntityType,
        EmotionalMemory, EmotionalContext, EmotionalTag,  # Added Emotional
        ProceduralMemory, ProceduralStep  # Added Procedural
        # Import other types like Flashbulb, Somatic, Liminal, Generative as they become fully integrated
    )

    MEMORY_TYPES_LOADED_SUCCESSFULLY = True
    logging.info("MBS Memory Service: Successfully loaded CEAF memory_types.")
except ImportError as e_mem_types:
    logging.error(f"MBS Memory Service: Could not import full CEAF memory_types: {e_mem_types}. Using placeholders.")


    # Placeholder definitions (ensure all used types have placeholders)
    class BaseMemoryPydanticPlaceholder:
        def __init__(self, **kwargs):
            self.memory_id = kwargs.get('memory_id', f'dummy_{time.time_ns()}')
            self.timestamp = kwargs.get('timestamp', time.time())
            self.last_accessed_ts: Optional[float] = None
            self.access_count: int = 0
            self.dynamic_salience_score: float = 0.5
            self.decay_rate: float = 0.01
            self.salience: str = "medium"
            self.source_type: str = "unknown"
            self.memory_type: str = kwargs.get('memory_type', 'unknown_placeholder')
            self.embedding_reference: Optional[str] = None
            self.keywords: List[str] = kwargs.get('keywords', [])
            self.narrative_thread_id: Optional[str] = kwargs.get('narrative_thread_id')  # For filtering
            self.entity_id_str: Optional[str] = kwargs.get('entity_id_str')
            self.relation_id_str: Optional[str] = kwargs.get('relation_id_str')
            self.source_entity_id_str: Optional[str] = kwargs.get('source_entity_id_str')
            self.target_entity_id_str: Optional[str] = kwargs.get('target_entity_id_str')
            self.relation_label: Optional[str] = kwargs.get('relation_label')
            self.goal_description: Optional[str] = kwargs.get('goal_description')  # For GoalRecord
            self.label: Optional[str] = kwargs.get('label')  # For KGEntityRecord
            self.description: Optional[str] = kwargs.get('description')  # For KGEntityRecord & KGRelationRecord
            self.primary_emotion: Optional[str] = kwargs.get('primary_emotion')  # For EmotionalMemory
            self.context: Any = kwargs.get('context')  # For EmotionalMemory context
            self.procedure_name: Optional[str] = kwargs.get('procedure_name')  # For ProceduralMemory
            self.steps: List[Any] = kwargs.get('steps', [])  # For ProceduralMemory
            self.__dict__.update(kwargs)

        def model_dump_json(self, **kwargs): return json.dumps(self.__dict__)

        def model_dump(self, **kwargs): return self.__dict__

        def mark_accessed(self): self.last_accessed_ts = time.time(); self.access_count += 1

        @property
        def content(self): return self  # For ExplicitMemory placeholder

        @property
        def text_content(self): return self.__dict__.get('text_content', '')  # For ExplicitMemory placeholder


    class ExplicitMemory(BaseMemoryPydanticPlaceholder):
        memory_type = "explicit"  # type: ignore


    class ExplicitMemoryContent(BaseMemoryPydanticPlaceholder):
        pass  # type: ignore


    class GoalRecord(BaseMemoryPydanticPlaceholder):
        memory_type = "goal_record"  # type: ignore


    class GoalStatus:
        PENDING = "pending"; ACTIVE = "active"; COMPLETED = "completed"; FAILED = "failed"; PAUSED = "paused"  # type: ignore


    class KGEntityRecord(BaseMemoryPydanticPlaceholder):
        memory_type = "kg_entity_record"  # type: ignore


    class KGRelationRecord(BaseMemoryPydanticPlaceholder):
        memory_type = "kg_relation_record"  # type: ignore


    class KGEntityType:
        OTHER = "other"  # type: ignore


    class EmotionalMemory(BaseMemoryPydanticPlaceholder):
        memory_type = "emotional"  # type: ignore


    class EmotionalContext(BaseMemoryPydanticPlaceholder):
        pass  # type: ignore


    class EmotionalTag:
        NEUTRAL = "neutral"  # type: ignore


    class ProceduralMemory(BaseMemoryPydanticPlaceholder):
        memory_type = "procedural"  # type: ignore


    class ProceduralStep(BaseMemoryPydanticPlaceholder):
        pass  # type: ignore


    class MemorySourceType:
        USER_INTERACTION = "user_interaction"; ORA_RESPONSE = "ora_response"; TOOL_OUTPUT = "tool_output"; INTERNAL_REFLECTION = "internal_reflection"  # type: ignore


    class MemorySalience:
        MEDIUM = "medium"; LOW = "low"; HIGH = "high"; CRITICAL = "critical"  # type: ignore


    AnyMemoryType = Union[
        ExplicitMemory, GoalRecord, KGEntityRecord, KGRelationRecord, EmotionalMemory, ProceduralMemory]  # type: ignore

try:
    from ..modules.memory_blossom.memory_lifecycle_manager import (
        initialize_dynamic_salience, update_dynamic_salience, apply_decay_to_all_memories,
        archive_or_forget_low_salience_memories, DEFAULT_ARCHIVE_SALIENCE_THRESHOLD,
        EPHEMERAL_SOURCE_TYPES_FOR_DELETION
    )

    MEMORY_LIFECYCLE_MANAGER_LOADED = True
    logging.info("MBS Memory Service: Successfully loaded CEAF memory_lifecycle_manager.")
except ImportError as e_lifecycle:
    logging.error(
        f"MBS Memory Service: Could not import CEAF memory_lifecycle_manager: {e_lifecycle}. Lifecycle features limited.")


    # Dummy lifecycle functions
    def initialize_dynamic_salience(mem):
        pass


    def update_dynamic_salience(mem, **kwargs):
        pass


    def apply_decay_to_all_memories(mems):
        pass


    def archive_or_forget_low_salience_memories(mems, **kwargs):
        return mems, [], []


    DEFAULT_ARCHIVE_SALIENCE_THRESHOLD = 0.1
    EPHEMERAL_SOURCE_TYPES_FOR_DELETION = []


class MBSMemoryService(BaseMemoryService):
    def __init__(self, memory_store_path: Optional[Union[str, Path]] = None):
        self.store_path = Path(memory_store_path or MEMORY_STORE_ROOT_DIR)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self._embedding_client: EmbeddingClient = get_embedding_client()
        logger.info(f"MBS Memory Service: Using EmbeddingClient with provider '{self._embedding_client.provider}' "
                    f"and default model '{self._embedding_client.default_model_name}'.")

        self._embedding_cache: Dict[str, List[float]] = {}

        self._in_memory_explicit_cache: List[ExplicitMemory] = self._load_all_explicit_memories()
        self._in_memory_goals_cache: List[GoalRecord] = self._load_all_goal_records()
        self._in_memory_kg_entities_cache: List[KGEntityRecord] = self._load_all_kg_entities()
        self._in_memory_kg_relations_cache: List[KGRelationRecord] = self._load_all_kg_relations()
        # Add caches for newly integrated types
        self._in_memory_emotional_cache: List[EmotionalMemory] = self._load_all_emotional_memories()
        self._in_memory_procedural_cache: List[ProceduralMemory] = self._load_all_procedural_memories()

        # Initialize lifecycle attributes for all loaded memories
        all_memory_caches_for_init = [
            (self._in_memory_explicit_cache, 0.01, 0.5),
            (self._in_memory_goals_cache, 0.02, 0.6),
            (self._in_memory_kg_entities_cache, 0.005, 0.7),
            (self._in_memory_kg_relations_cache, 0.005, 0.65),
            (self._in_memory_emotional_cache, 0.015, 0.55),
            (self._in_memory_procedural_cache, 0.008, 0.6)
        ]
        if MEMORY_LIFECYCLE_MANAGER_LOADED:
            for cache, default_decay, default_salience_on_load in all_memory_caches_for_init:
                for mem_item in cache:
                    if not hasattr(mem_item, 'dynamic_salience_score'):
                        setattr(mem_item, 'dynamic_salience_score', default_salience_on_load)
                    if not hasattr(mem_item, 'decay_rate'):
                        setattr(mem_item, 'decay_rate', default_decay)
                    if not hasattr(mem_item, 'access_count'):
                        setattr(mem_item, 'access_count', 0)
                    if not hasattr(mem_item, 'last_accessed_ts') or getattr(mem_item, 'last_accessed_ts') is None:
                        setattr(mem_item, 'last_accessed_ts', getattr(mem_item, 'timestamp', time.time()))
                    initialize_dynamic_salience(mem_item)

        logger.info(
            f"MBSMemoryService initialized. Storage path: {self.store_path}. "
            f"Loaded: {len(self._in_memory_explicit_cache)} explicit, "
            f"{len(self._in_memory_goals_cache)} goals, "
            f"{len(self._in_memory_kg_entities_cache)} KG entities, "
            f"{len(self._in_memory_kg_relations_cache)} KG relations, "
            f"{len(self._in_memory_emotional_cache)} emotional, "
            f"{len(self._in_memory_procedural_cache)} procedural."
        )
        logger.info(f"Embeddings in cache after load: {len(self._embedding_cache)}")

        self._stop_background_tasks = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []

    async def build_initial_connection_graph(self):
        """
        Builds an initial graph of connections between memories at startup.
        This involves:
        1. Connecting memories based on semantic similarity of their embeddings.
        2. Connecting memories based on shared keywords.
        3. Reinforcing KG structure by linking entities and relations.
        4. Creating links based on explicit cross-references in memory content.
        Modifies the 'related_memory_ids' field of BaseMemory objects in-place
        and then rewrites the persistent stores.
        """
        logger.info("MBSMemoryService: Starting initial memory connection graph build...")
        changes_made_to_any_cache = False

        all_memory_caches_for_graph: Dict[str, List[AnyMemoryType]] = {
            "explicit": self._in_memory_explicit_cache,
            "goals": self._in_memory_goals_cache,
            "kg_entities": self._in_memory_kg_entities_cache,
            "kg_relations": self._in_memory_kg_relations_cache,
            "emotional": self._in_memory_emotional_cache,
            "procedural": self._in_memory_procedural_cache,
        }

        # Helper to safely add a connection
        def _add_connection_if_new(mem1: BaseMemory, mem2_id: str) -> bool:
            if not hasattr(mem1, 'related_memory_ids'):
                setattr(mem1, 'related_memory_ids', [])
            if mem2_id != mem1.memory_id and mem2_id not in mem1.related_memory_ids and len(
                    mem1.related_memory_ids) < MAX_RELATED_IDS_PER_MEMORY:
                mem1.related_memory_ids.append(mem2_id)
                return True
            return False

        # Consolidate all memories into one list for easier pairwise comparison for semantic/keyword
        all_memories_flat: List[BaseMemory] = []
        for cache_name, cache_list in all_memory_caches_for_graph.items():
            all_memories_flat.extend(cast(List[BaseMemory], cache_list))

        logger.info(f"MBS: Processing {len(all_memories_flat)} total memories for connections.")

        # 1. Semantic Similarity & 2. Keyword Overlap
        # This is N^2, so be mindful for very large initial memory stores.
        # For now, we'll iterate. Optimized approaches (e.g., FAISS for semantic) are for later.
        for i in range(len(all_memories_flat)):
            mem1 = all_memories_flat[i]
            if not hasattr(mem1, 'memory_id'): continue  # Skip if malformed

            for j in range(i + 1, len(all_memories_flat)):
                mem2 = all_memories_flat[j]
                if not hasattr(mem2, 'memory_id'): continue

                # Ensure embeddings are present for semantic comparison
                if mem1.memory_id in self._embedding_cache and mem2.memory_id in self._embedding_cache:
                    emb1 = self._embedding_cache[mem1.memory_id]
                    emb2 = self._embedding_cache[mem2.memory_id]
                    semantic_sim = compute_adaptive_similarity(emb1, emb2)

                    if semantic_sim >= SEMANTIC_CONNECTION_THRESHOLD:
                        if _add_connection_if_new(mem1, mem2.memory_id): changes_made_to_any_cache = True
                        if _add_connection_if_new(mem2, mem1.memory_id): changes_made_to_any_cache = True
                        logger.debug(
                            f"MBS Graph: Semantic link ({semantic_sim:.2f}) between {mem1.memory_id} and {mem2.memory_id}")
                        continue  # Prioritize semantic link over keyword if strong enough

                # Keyword Overlap (if not already semantically linked strongly)
                keywords1 = set(getattr(mem1, 'keywords', []))
                keywords2 = set(getattr(mem2, 'keywords', []))
                if keywords1 and keywords2:  # Both must have keywords
                    common_keywords = keywords1.intersection(keywords2)
                    if len(common_keywords) >= KEYWORD_CONNECTION_THRESHOLD_COUNT:
                        if _add_connection_if_new(mem1, mem2.memory_id): changes_made_to_any_cache = True
                        if _add_connection_if_new(mem2, mem1.memory_id): changes_made_to_any_cache = True
                        logger.debug(
                            f"MBS Graph: Keyword link ({len(common_keywords)} common) between {mem1.memory_id} and {mem2.memory_id}")

        # 3. KG Structural Links
        logger.info("MBS Graph: Processing KG structural links...")
        entity_map_by_id_str: Dict[str, KGEntityRecord] = {
            getattr(e, 'entity_id_str'): e for e in self._in_memory_kg_entities_cache  # type: ignore
            if hasattr(e, 'entity_id_str')
        }

        for rel_mem_any in self._in_memory_kg_relations_cache:
            rel_mem = cast(KGRelationRecord, rel_mem_any)  # Assuming correct type in this cache
            if not hasattr(rel_mem, 'memory_id') or \
                    not hasattr(rel_mem, 'source_entity_id_str') or \
                    not hasattr(rel_mem, 'target_entity_id_str'):
                continue

            source_entity_id = getattr(rel_mem, 'source_entity_id_str')
            target_entity_id = getattr(rel_mem, 'target_entity_id_str')

            source_entity_mem = entity_map_by_id_str.get(source_entity_id)
            target_entity_mem = entity_map_by_id_str.get(target_entity_id)

            if source_entity_mem:
                if _add_connection_if_new(cast(BaseMemory, source_entity_mem),
                                          rel_mem.memory_id): changes_made_to_any_cache = True
                if _add_connection_if_new(cast(BaseMemory, rel_mem),
                                          source_entity_mem.memory_id): changes_made_to_any_cache = True
            if target_entity_mem:
                if _add_connection_if_new(cast(BaseMemory, target_entity_mem),
                                          rel_mem.memory_id): changes_made_to_any_cache = True
                if _add_connection_if_new(cast(BaseMemory, rel_mem),
                                          target_entity_mem.memory_id): changes_made_to_any_cache = True

            # Optionally, connect source and target entities directly if this relation implies a strong bond
            # For now, the relation memory itself acts as the bridge.

        # 4. Explicit Cross-References in ExplicitMemory
        logger.info("MBS Graph: Processing explicit cross-references...")
        # Build quick lookups for goals and procedural memories by ID
        goal_map_by_id: Dict[str, GoalRecord] = {
            g.memory_id: g for g in self._in_memory_goals_cache  # type: ignore
        }
        proc_map_by_id: Dict[str, ProceduralMemory] = {
            p.memory_id: p for p in self._in_memory_procedural_cache  # type: ignore
        }

        for explicit_mem_any in self._in_memory_explicit_cache:
            explicit_mem = cast(ExplicitMemory, explicit_mem_any)  # Assuming correct type
            if not hasattr(explicit_mem, 'memory_id'): continue

            if hasattr(explicit_mem, 'explains_procedure_step') and explicit_mem.explains_procedure_step:
                # This field should store the ProceduralMemory ID, not a step ID.
                # The actual step is within the ProceduralMemory's steps list.
                # For simplicity, we link to the whole ProceduralMemory.
                proc_mem_id = explicit_mem.explains_procedure_step
                linked_proc_mem = proc_map_by_id.get(proc_mem_id)
                if linked_proc_mem:
                    if _add_connection_if_new(cast(BaseMemory, explicit_mem),
                                              linked_proc_mem.memory_id): changes_made_to_any_cache = True
                    if _add_connection_if_new(cast(BaseMemory, linked_proc_mem),
                                              explicit_mem.memory_id): changes_made_to_any_cache = True
                    logger.debug(
                        f"MBS Graph: Linked explicit mem {explicit_mem.memory_id} to procedural mem {linked_proc_mem.memory_id} (explains_procedure).")

            if hasattr(explicit_mem, 'provides_evidence_for_goal') and explicit_mem.provides_evidence_for_goal:
                goal_mem_id = explicit_mem.provides_evidence_for_goal
                linked_goal_mem = goal_map_by_id.get(goal_mem_id)
                if linked_goal_mem:
                    if _add_connection_if_new(cast(BaseMemory, explicit_mem),
                                              linked_goal_mem.memory_id): changes_made_to_any_cache = True
                    if _add_connection_if_new(cast(BaseMemory, linked_goal_mem),
                                              explicit_mem.memory_id): changes_made_to_any_cache = True
                    logger.debug(
                        f"MBS Graph: Linked explicit mem {explicit_mem.memory_id} to goal mem {linked_goal_mem.memory_id} (provides_evidence).")

        if changes_made_to_any_cache:
            logger.info("MBS Graph: Rewriting memory stores due to new connections...")
            await self._rewrite_aggregated_store()
            await self._rewrite_goals_store()
            await self._rewrite_kg_entities_store()
            await self._rewrite_kg_relations_store()
            await self._rewrite_emotional_store()
            await self._rewrite_procedural_store()
            # Add rewrites for any other stores if they can have 'related_memory_ids'
        else:
            logger.info("MBS Graph: No new connections were made during the build process.")



    async def _get_searchable_text_and_keywords(self, memory: AnyMemoryType) -> Tuple[str, List[str]]:
        """Extracts a concatenated text string for searching and inherent keywords from a memory object."""
        texts_for_search: List[str] = []
        inherent_keywords: List[str] = list(getattr(memory, 'keywords', []))  # Start with existing keywords

        mem_type = getattr(memory, 'memory_type', 'unknown')

        if mem_type == "explicit":
            content_obj = getattr(memory, 'content', None)
            if content_obj:
                if hasattr(content_obj, 'text_content') and getattr(content_obj, 'text_content'):
                    texts_for_search.append(str(getattr(content_obj, 'text_content')))
                if hasattr(content_obj, 'structured_data') and getattr(content_obj, 'structured_data'):
                    # Simple stringify for search, could be more nuanced
                    texts_for_search.append(json.dumps(getattr(content_obj, 'structured_data')))
        elif mem_type == "goal_record":
            if hasattr(memory, 'goal_description') and getattr(memory, 'goal_description'):
                texts_for_search.append(str(getattr(memory, 'goal_description')))
        elif mem_type == "kg_entity_record":
            if hasattr(memory, 'label') and getattr(memory, 'label'):
                texts_for_search.append(str(getattr(memory, 'label')))
            if hasattr(memory, 'description') and getattr(memory, 'description'):
                texts_for_search.append(str(getattr(memory, 'description')))
            if hasattr(memory, 'aliases') and getattr(memory, 'aliases'):
                texts_for_search.extend([str(alias) for alias in getattr(memory, 'aliases')])
        elif mem_type == "kg_relation_record":
            if hasattr(memory, 'relation_label') and getattr(memory, 'relation_label'):
                texts_for_search.append(str(getattr(memory, 'relation_label')))
            if hasattr(memory, 'description') and getattr(memory, 'description'):
                texts_for_search.append(str(getattr(memory, 'description')))
        elif mem_type == "emotional":
            if hasattr(memory, 'primary_emotion') and getattr(memory, 'primary_emotion'):
                emotion_val = getattr(memory, 'primary_emotion')
                texts_for_search.append(str(emotion_val.value if hasattr(emotion_val, 'value') else emotion_val))
            emo_context = getattr(memory, 'context', None)
            if emo_context and hasattr(emo_context, 'triggering_event_summary') and getattr(emo_context,
                                                                                            'triggering_event_summary'):
                texts_for_search.append(str(getattr(emo_context, 'triggering_event_summary')))
        elif mem_type == "procedural":
            if hasattr(memory, 'procedure_name') and getattr(memory, 'procedure_name'):
                texts_for_search.append(str(getattr(memory, 'procedure_name')))
            if hasattr(memory, 'goal_description') and getattr(memory, 'goal_description'):
                texts_for_search.append(str(getattr(memory, 'goal_description')))
            if hasattr(memory, 'steps') and getattr(memory, 'steps'):
                for step in getattr(memory, 'steps'):
                    if hasattr(step, 'description') and getattr(step, 'description'):
                        texts_for_search.append(str(getattr(step, 'description')))
        # Add other memory types here...

        full_search_text = " ".join(filter(None, texts_for_search)).strip()
        return full_search_text, inherent_keywords

    async def _ensure_embedding_for_memory(self, memory: AnyMemoryType):
        if not hasattr(memory, 'memory_id') or not hasattr(memory, 'memory_type'):
            logger.warning(
                f"MBS: Memory object missing memory_id or memory_type, cannot ensure embedding. Obj: {memory}")
            return

        memory_id = memory.memory_id
        memory_type_str = str(getattr(memory, 'memory_type', 'unknown'))  # Use string memory_type for context

        if memory_id in self._embedding_cache:
            setattr(memory, 'embedding_reference', memory_id)
            return

        text_to_embed, _ = await self._get_searchable_text_and_keywords(memory)

        if text_to_embed:
            try:
                embedding = await self._embedding_client.get_embedding(text_to_embed, context_type=memory_type_str)
                if embedding:
                    self._embedding_cache[memory_id] = embedding
                    setattr(memory, 'embedding_reference', memory_id)
                    logger.debug(f"MBS: Generated and cached embedding for {memory_type_str} memory {memory_id}")
            except Exception as e:
                logger.error(f"MBS: Failed to generate embedding for {memory_type_str} {memory_id}: {e}")
        else:
            logger.debug(f"MBS: No text content to embed for {memory_type_str} memory {memory_id}.")

    # --- File Path Getters (add new ones) ---
    def _get_aggregated_memories_filepath(self) -> Path:
        return self.store_path / AGGREGATED_EXPLICIT_MEMORIES_FILENAME

    def _get_goals_store_filepath(self) -> Path:
        return self.store_path / GOALS_STORE_FILENAME

    def _get_kg_entities_filepath(self) -> Path:
        return self.store_path / KG_ENTITIES_FILENAME

    def _get_kg_relations_filepath(self) -> Path:
        return self.store_path / KG_RELATIONS_FILENAME

    def _get_emotional_memories_filepath(self) -> Path:
        return self.store_path / EMOTIONAL_MEMORIES_FILENAME

    def _get_procedural_memories_filepath(self) -> Path:
        return self.store_path / PROCEDURAL_MEMORIES_FILENAME

    # --- Generic Loader (no change needed if Pydantic models handle their types) ---
    def _load_from_jsonl_file(self, filepath: Path, pydantic_model_class: type, expected_memory_type_attr: str) -> List[
        Any]:
        records: List[Any] = []
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line)
                            # Ensure 'memory_type' exists and matches before Pydantic parsing
                            if data.get("memory_type") == expected_memory_type_attr:
                                record_instance = pydantic_model_class(**data)
                                records.append(record_instance)
                                if data.get("embedding_reference") and data.get("_embedding_vector_for_jsonl_"):
                                    self._embedding_cache[data["embedding_reference"]] = data[
                                        "_embedding_vector_for_jsonl_"]
                            # No warning for mismatched types if we iterate many files; let specific loaders handle specific files.
                        except (json.JSONDecodeError, TypeError,
                                ValueError) as e_parse:  # PydanticError can inherit from ValueError
                            logger.warning(
                                f"MBS Store: Skipping malformed/invalid line {line_num} in {filepath} for model {pydantic_model_class.__name__}: {line.strip()}. Error: {e_parse}")
            except Exception as e:
                logger.error(f"MBS Store: Failed to load records from {filepath}: {e}", exc_info=True)
        return records

    # --- Specific Loaders (add new ones) ---
    def _load_all_explicit_memories(self) -> List[ExplicitMemory]:
        return self._load_from_jsonl_file(self._get_aggregated_memories_filepath(), ExplicitMemory,
                                          "explicit")  # type: ignore

    def _load_all_goal_records(self) -> List[GoalRecord]:
        return self._load_from_jsonl_file(self._get_goals_store_filepath(), GoalRecord, "goal_record")  # type: ignore

    def _load_all_kg_entities(self) -> List[KGEntityRecord]:
        return self._load_from_jsonl_file(self._get_kg_entities_filepath(), KGEntityRecord,
                                          "kg_entity_record")  # type: ignore

    def _load_all_kg_relations(self) -> List[KGRelationRecord]:
        return self._load_from_jsonl_file(self._get_kg_relations_filepath(), KGRelationRecord,
                                          "kg_relation_record")  # type: ignore

    def _load_all_emotional_memories(self) -> List[EmotionalMemory]:
        return self._load_from_jsonl_file(self._get_emotional_memories_filepath(), EmotionalMemory,
                                          "emotional")  # type: ignore

    def _load_all_procedural_memories(self) -> List[ProceduralMemory]:
        return self._load_from_jsonl_file(self._get_procedural_memories_filepath(), ProceduralMemory,
                                          "procedural")  # type: ignore

    # --- Generic Saver (no change needed) ---
    async def _save_record_to_jsonl_file(self, filepath: Path, record_object: Any, record_id_attr: str = 'memory_id'):
        try:
            await self._ensure_embedding_for_memory(record_object)
            record_data_dict = {}
            if hasattr(record_object, "model_dump"):
                record_data_dict = record_object.model_dump(exclude_none=True)
            else:  # Fallback for placeholders
                record_data_dict = record_object.__dict__

            mem_id_for_embedding = record_data_dict.get(record_id_attr) or record_data_dict.get('memory_id')
            if mem_id_for_embedding and mem_id_for_embedding in self._embedding_cache:
                record_data_dict["_embedding_vector_for_jsonl_"] = self._embedding_cache[mem_id_for_embedding]

            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record_data_dict) + "\n")
        except Exception as e:
            record_id_val = getattr(record_object, record_id_attr, 'N/A_ID')
            logger.error(f"MBS Store: Failed to save record '{record_id_val}' to {filepath}: {e}", exc_info=True)

    # --- Specific Savers (add new ones) ---
    async def _save_explicit_memory_to_aggregated_store(self, memory: ExplicitMemory):
        await self._save_record_to_jsonl_file(self._get_aggregated_memories_filepath(), memory)

    async def _save_goal_record_to_store(self, goal_memory: GoalRecord):
        await self._save_record_to_jsonl_file(self._get_goals_store_filepath(), goal_memory)

    async def _save_kg_entity_to_store(self, entity_record: KGEntityRecord):
        await self._save_record_to_jsonl_file(self._get_kg_entities_filepath(), entity_record, 'entity_id_str')

    async def _save_kg_relation_to_store(self, relation_record: KGRelationRecord):
        await self._save_record_to_jsonl_file(self._get_kg_relations_filepath(), relation_record, 'relation_id_str')

    async def _save_emotional_memory_to_store(self, emotional_memory: EmotionalMemory):
        await self._save_record_to_jsonl_file(self._get_emotional_memories_filepath(), emotional_memory)

    async def _save_procedural_memory_to_store(self, procedural_memory: ProceduralMemory):
        await self._save_record_to_jsonl_file(self._get_procedural_memories_filepath(), procedural_memory)

    # --- Generic Rewriter (no change needed) ---
    async def _rewrite_jsonl_file(self, filepath: Path, records_cache: List[Any], temp_suffix: str):
        temp_filepath = filepath.with_suffix(f".{time.time_ns()}.{temp_suffix}.tmp")
        try:
            for record_obj_for_rewrite in records_cache:
                await self._ensure_embedding_for_memory(record_obj_for_rewrite)

            with open(temp_filepath, 'w', encoding='utf-8') as f:
                for record_obj in records_cache:
                    record_data_dict = {}
                    if hasattr(record_obj, "model_dump"):
                        record_data_dict = record_obj.model_dump(exclude_none=True)
                    else:  # Fallback for placeholders
                        record_data_dict = record_obj.__dict__

                    mem_id_for_embedding = record_data_dict.get(
                        getattr(record_obj, '_id_attr_for_embedding_lookup', 'memory_id')) \
                                           or record_data_dict.get('memory_id')
                    if mem_id_for_embedding and mem_id_for_embedding in self._embedding_cache:
                        record_data_dict["_embedding_vector_for_jsonl_"] = self._embedding_cache[mem_id_for_embedding]

                    f.write(json.dumps(record_data_dict) + "\n")
            os.replace(temp_filepath, filepath)
            logger.info(f"MBS Store: Successfully rewrote store at {filepath}")
        except Exception as e:
            logger.error(f"MBS Store: Failed to rewrite store {filepath}: {e}", exc_info=True)
            if temp_filepath.exists():
                try:
                    os.remove(temp_filepath)
                except OSError:
                    logger.error(f"MBS Store: Failed to remove temp file {temp_filepath}")

    # --- Specific Rewriters (add new ones) ---
    async def _rewrite_aggregated_store(self):
        await self._rewrite_jsonl_file(self._get_aggregated_memories_filepath(), self._in_memory_explicit_cache,
                                       "explicit")

    async def _rewrite_goals_store(self):
        await self._rewrite_jsonl_file(self._get_goals_store_filepath(), self._in_memory_goals_cache, "goals")

    async def _rewrite_kg_entities_store(self):
        await self._rewrite_jsonl_file(self._get_kg_entities_filepath(), self._in_memory_kg_entities_cache,
                                       "kg_entities")

    async def _rewrite_kg_relations_store(self):
        await self._rewrite_jsonl_file(self._get_kg_relations_filepath(), self._in_memory_kg_relations_cache,
                                       "kg_relations")

    async def _rewrite_emotional_store(self):
        await self._rewrite_jsonl_file(self._get_emotional_memories_filepath(), self._in_memory_emotional_cache,
                                       "emotional")

    async def _rewrite_procedural_store(self):
        await self._rewrite_jsonl_file(self._get_procedural_memories_filepath(), self._in_memory_procedural_cache,
                                       "procedural")

    # --- Core Public Methods (add_session_to_memory: no major changes needed initially) ---
    async def _extract_and_embed_explicit_memories_from_adk_session(self, session: AdkSession) -> List[ExplicitMemory]:

        extracted_memories: List[ExplicitMemory] = []
        events_to_process = session.events[-SESSION_HISTORY_FOR_MEMORY_TURNS:] if session.events else []

        for event in events_to_process:
            if not event.content or not event.content.parts:
                continue

            event_texts = [part.text for part in event.content.parts if
                           isinstance(part, AdkPart) and part.text and part.text.strip()]
            full_event_text = " ".join(event_texts).strip()

            if full_event_text:
                source_type_val_str = MemorySourceType.USER_INTERACTION.value if event.author == "user" else MemorySourceType.ORA_RESPONSE.value  # type: ignore
                if event.author not in ["user", "ORA", "SYSTEM"]:  # type: ignore
                    source_type_val_str = MemorySourceType.TOOL_OUTPUT.value  # type: ignore

                source_type_val = source_type_val_str
                if MEMORY_TYPES_LOADED_SUCCESSFULLY:
                    try:
                        source_type_val = MemorySourceType(source_type_val_str)
                    except ValueError:
                        logger.warning(
                            f"MBS Store: Unknown MemorySourceType str '{source_type_val_str}' for event by '{event.author}'")
                        source_type_val = MemorySourceType.INTERNAL_REFLECTION  # type: ignore

                keywords = list(set(re.findall(r'\b\w{4,15}\b', full_event_text.lower())))[:10]
                mem_data = {
                    "timestamp": event.timestamp or time.time(),
                    "source_turn_id": event.invocation_id or session.id,
                    "source_interaction_id": session.id,
                    "source_type": source_type_val,
                    "source_agent_name": event.author,
                    "salience": MemorySalience.MEDIUM,  # type: ignore
                    "keywords": keywords,
                    "content": ExplicitMemoryContent(
                        text_content=full_event_text) if MEMORY_TYPES_LOADED_SUCCESSFULLY else {
                        "text_content": full_event_text},  # type: ignore
                    "confidence_score": 0.85,  # Default confidence
                    "memory_type": "explicit"
                }
                try:
                    new_explicit_mem = ExplicitMemory(**mem_data)  # type: ignore
                    await self._ensure_embedding_for_memory(new_explicit_mem)

                    if MEMORY_LIFECYCLE_MANAGER_LOADED:
                        # Initialize lifecycle attributes if not present
                        if not hasattr(new_explicit_mem, 'dynamic_salience_score'): setattr(new_explicit_mem,
                                                                                            'dynamic_salience_score',
                                                                                            0.5)
                        if not hasattr(new_explicit_mem, 'decay_rate'): setattr(new_explicit_mem, 'decay_rate', 0.01)
                        if not hasattr(new_explicit_mem, 'access_count'): setattr(new_explicit_mem, 'access_count', 0)
                        if not hasattr(new_explicit_mem, 'last_accessed_ts') or getattr(new_explicit_mem,
                                                                                        'last_accessed_ts') is None: setattr(
                            new_explicit_mem, 'last_accessed_ts', getattr(new_explicit_mem, 'timestamp', time.time()))
                        initialize_dynamic_salience(new_explicit_mem)
                    extracted_memories.append(new_explicit_mem)
                except Exception as e_create:
                    logger.error(
                        f"MBS Store: Failed to create ExplicitMemory from event data: {e_create}. Data: {mem_data}",
                        exc_info=True)
        logger.debug(f"MBS Store: Extracted {len(extracted_memories)} explicit memories from ADK session {session.id}")
        return extracted_memories

    async def add_session_to_memory(self, session: AdkSession):

        if not isinstance(session, AdkSession):
            logger.error("MBS add_session_to_memory: Expected AdkSession object.")
            return
        logger.info(f"MBS: Processing ADK session '{session.id}' for memory ingestion.")
        new_memories = await self._extract_and_embed_explicit_memories_from_adk_session(session)
        if not new_memories:
            logger.info(f"MBS: No new explicit memories extracted from session '{session.id}'.")
            return

        added_count = 0
        for mem_obj in new_memories:
            self._in_memory_explicit_cache.append(mem_obj)
            await self._save_explicit_memory_to_aggregated_store(mem_obj)
            added_count += 1

        if added_count > 0:
            logger.info(
                f"MBS: Added {added_count} new memories from session '{session.id}' to store. Explicit Cache: {len(self._in_memory_explicit_cache)}")

    async def search_memory(self, *, app_name: str, user_id: str, query: str) -> 'SearchMemoryResponse':  # type: ignore
        top_k = 5  # Default, can be made configurable or part of augmented_query_context
        logger.info(
            f"MBS search_memory (Advanced CARS): Full Query='{query[:200]}...', App='{app_name}', User='{user_id}'")

        # 1. Parse query and augmented context
        main_query_text = query
        parsed_augmented_context: Dict[str, Any] = {}
        if CONTEXT_QUERY_DELIMITER in query:
            parts = query.split(CONTEXT_QUERY_DELIMITER, 1)
            main_query_text = parts[0]
            if len(parts) > 1 and parts[1]:
                try:
                    parsed_augmented_context = json.loads(parts[1])
                    logger.info(
                        f"MBS search_memory: Parsed augmented_query_context: {list(parsed_augmented_context.keys())}")
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"MBS search_memory: Failed to parse augmented_query_context JSON: {e}. Context part: {parts[1][:100]}")

        if not main_query_text or not main_query_text.strip():
            return {"memories": []}

        # 2. Dynamic Weighting based on context
        current_semantic_weight = parsed_augmented_context.get("semantic_score_weight_override",
                                                               DEFAULT_SEMANTIC_SCORE_WEIGHT)
        current_keyword_weight = parsed_augmented_context.get("keyword_score_weight_override",
                                                              DEFAULT_KEYWORD_SCORE_WEIGHT)
        if "current_interaction_goal" in parsed_augmented_context:
            goal = parsed_augmented_context["current_interaction_goal"]
            if goal == "creative_ideation":
                current_semantic_weight = max(current_semantic_weight, 0.75)  # Favor broader semantic connections
                current_keyword_weight = min(current_keyword_weight, 0.25)
            elif goal == "problem_solving" or goal == "factual_retrieval":
                current_keyword_weight = max(current_keyword_weight, 0.5)  # Favor precision
                current_semantic_weight = min(current_semantic_weight, 0.5)
        logger.debug(
            f"MBS search_memory: Using SemanticWeight={current_semantic_weight}, KeywordWeight={current_keyword_weight}")

        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = await self._embedding_client.get_embedding(main_query_text, context_type="default_query")
        except Exception as e_q_embed:
            logger.error(
                f"MBS search_memory: Failed to get embedding for query '{main_query_text}': {e_q_embed}. Proceeding with keyword search mainly.")

        scored_candidates: List[Tuple[AnyMemoryType, float]] = []
        query_keywords_set = set(re.findall(r'\b\w{3,15}\b', main_query_text.lower()))

        # Define all memory caches to iterate over with their type string
        all_memory_sources: List[Tuple[List[AnyMemoryType], str]] = [
            (self._in_memory_explicit_cache, "explicit"),
            (self._in_memory_goals_cache, "goal_record"),
            (self._in_memory_kg_entities_cache, "kg_entity_record"),
            (self._in_memory_kg_relations_cache, "kg_relation_record"),
            (self._in_memory_emotional_cache, "emotional"),
            (self._in_memory_procedural_cache, "procedural"),
            # Add other caches here as they are implemented
        ]

        # 3. Iterate All Relevant Caches & 5. Unified Scoring
        for memory_cache, memory_type_str_for_loop in all_memory_sources:
            # 3.1 Memory Type Prioritization (Example)
            type_priority_multiplier = 1.0
            if "current_interaction_goal" in parsed_augmented_context:
                goal = parsed_augmented_context["current_interaction_goal"]
                if goal == "recall_procedure" and memory_type_str_for_loop in ["procedural", "goal_record"]:
                    type_priority_multiplier = 1.5
                elif goal == "understand_user_emotion" and memory_type_str_for_loop == "emotional":
                    type_priority_multiplier = 1.3
                elif goal == "knowledge_graph_exploration" and memory_type_str_for_loop in ["kg_entity_record",
                                                                                            "kg_relation_record"]:
                    type_priority_multiplier = 1.4

            for mem in memory_cache:
                keyword_score = 0.0
                semantic_score = 0.0

                searchable_text, inherent_keywords = await self._get_searchable_text_and_keywords(mem)
                all_mem_keywords = set(str(k).lower() for k in inherent_keywords)

                # Keyword scoring
                if searchable_text:  # Check against extracted full text
                    for qk in query_keywords_set:
                        if qk in searchable_text.lower(): keyword_score += 1.0
                for mk_keyword_lower in all_mem_keywords:  # Check against inherent keywords
                    if mk_keyword_lower in query_keywords_set: keyword_score += 2.0  # Boost for explicit keywords

                # Semantic scoring
                mem_id_for_embedding = getattr(mem, 'memory_id', None)
                if query_embedding and mem_id_for_embedding and mem_id_for_embedding in self._embedding_cache:
                    memory_embedding = self._embedding_cache[mem_id_for_embedding]
                    semantic_score = compute_adaptive_similarity(query_embedding, memory_embedding)
                    semantic_score = max(0, semantic_score)  # Ensure non-negative
                elif query_embedding and searchable_text:  # Try to embed on-the-fly if not cached
                    try:
                        # Be cautious with on-the-fly embedding in search due to performance
                        await self._ensure_embedding_for_memory(mem)  # This will cache it if successful
                        if mem_id_for_embedding and mem_id_for_embedding in self._embedding_cache:
                            memory_embedding = self._embedding_cache[mem_id_for_embedding]
                            semantic_score = compute_adaptive_similarity(query_embedding, memory_embedding)
                            semantic_score = max(0, semantic_score)
                    except Exception as e_search_emb:
                        logger.warning(
                            f"MBS Search: On-the-fly embedding failed for {mem_id_for_embedding}: {e_search_emb}")

                if keyword_score > 0 or semantic_score > 0:
                    dynamic_salience = getattr(mem, 'dynamic_salience_score', 0.5)
                    current_timestamp_val = getattr(mem, 'timestamp', 0)
                    recency_factor = 1.0 / (1 + (time.time() - current_timestamp_val) / (
                                86400 * 30))  # Penalize older, up to ~30 days

                    final_score = (current_keyword_weight * keyword_score + current_semantic_weight * semantic_score)
                    final_score *= (1 + dynamic_salience)  # Boost by dynamic salience
                    final_score *= recency_factor  # Modulate by recency
                    final_score *= type_priority_multiplier  # Modulate by type priority

                    # 4. Contextual Filtering (Example: narrative_thread_id)
                    active_narrative_thread = parsed_augmented_context.get("active_narrative_thread_id")
                    mem_narrative_thread = getattr(mem, 'narrative_thread_id', None)
                    if active_narrative_thread and mem_narrative_thread == active_narrative_thread:
                        final_score *= 1.2  # Boost if matches active narrative thread
                    elif active_narrative_thread and mem_narrative_thread:  # Different thread
                        final_score *= 0.8  # Slightly penalize if different active thread

                    if final_score > 0.01:  # Some minimal threshold
                        scored_candidates.append((mem, final_score))

        scored_candidates.sort(key=lambda item: item[1], reverse=True)

        # 6. Result Formatting
        results: List[Dict[str, Any]] = []
        retrieved_for_salience_update: List[BaseMemory] = []

        final_top_k = parsed_augmented_context.get("top_k_override", top_k)

        for mem_obj, score_val in scored_candidates[:final_top_k]:
            source_agent_name = getattr(mem_obj, 'source_agent_name', "retrieved_memory")
            mem_type_attr = str(getattr(mem_obj, 'memory_type', 'unknown'))

            # Improved display text generation
            display_text_parts = [f"Retrieved {mem_type_attr} (Score: {score_val:.2f})"]
            searchable_content_for_display, _ = await self._get_searchable_text_and_keywords(mem_obj)
            if searchable_content_for_display:
                display_text_parts.append(searchable_content_for_display[:150] + "...")  # Snippet
            else:  # Fallback display
                if mem_type_attr == 'explicit':
                    explicit_content = getattr(mem_obj, 'content', None)
                    if explicit_content and hasattr(explicit_content, 'text_content'):
                        display_text_parts.append(str(getattr(explicit_content, 'text_content', ""))[:150] + "...")
                elif mem_type_attr == 'kg_entity_record':
                    display_text_parts.append(
                        f"Entity: {getattr(mem_obj, 'label', 'N/A')}, Desc: {getattr(mem_obj, 'description', '')[:100]}...")
                elif mem_type_attr == 'goal_record':
                    display_text_parts.append(f"Goal: {getattr(mem_obj, 'goal_description', '')[:150]}...")

            display_text_final = " ".join(filter(None, display_text_parts))

            if display_text_final:
                pseudo_event_part_dict = AdkPart(text=display_text_final).to_dict()
                pseudo_event_content_dict = {"parts": [pseudo_event_part_dict], "role": source_agent_name}
                pseudo_event_dict = {
                    "id": f"mem-event-{getattr(mem_obj, 'memory_id', 'dummy_id')}",
                    "author": source_agent_name, "timestamp": getattr(mem_obj, 'timestamp', time.time()),
                    "content": pseudo_event_content_dict,
                    "invocation_id": getattr(mem_obj, 'source_turn_id',
                                             getattr(mem_obj, 'source_interaction_id', 'unknown_session'))
                }
                results.append({
                    "session_id": getattr(mem_obj, 'source_interaction_id', 'unknown_session'),
                    "events": [pseudo_event_dict], "score": score_val,
                    "retrieved_memory_type": mem_type_attr,
                    "retrieved_memory_id": getattr(mem_obj, 'memory_id', 'unknown_id')  # Add memory ID
                })
                if MEMORY_LIFECYCLE_MANAGER_LOADED and isinstance(mem_obj,
                                                                  BaseMemoryPydanticPlaceholder if not MEMORY_TYPES_LOADED_SUCCESSFULLY else BaseMemory):
                    retrieved_for_salience_update.append(
                        cast(BaseMemory if MEMORY_TYPES_LOADED_SUCCESSFULLY else BaseMemoryPydanticPlaceholder,
                             mem_obj))

        if MEMORY_LIFECYCLE_MANAGER_LOADED:
            for mem_to_update in retrieved_for_salience_update:
                update_dynamic_salience(mem_to_update, access_type="retrieval")

        logger.info(
            f"MBS search_memory (Advanced CARS): Found {len(results)} relevant memories for query '{main_query_text}'.")
        return {"memories": results}

    async def search_raw_memories(self, query: str, top_k: int = 5) -> List[Tuple[AnyMemoryType, float]]:
        """
        Search for raw memory objects (not wrapped in ADK format).

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of tuples containing (memory_object, score)
        """
        # Parse the query similar to search_memory
        main_query_text = query
        parsed_augmented_context: Dict[str, Any] = {}

        if CONTEXT_QUERY_DELIMITER in query:
            parts = query.split(CONTEXT_QUERY_DELIMITER, 1)
            main_query_text = parts[0]
            if len(parts) > 1 and parts[1]:
                try:
                    parsed_augmented_context = json.loads(parts[1])
                except json.JSONDecodeError:
                    pass

        # Get query embedding
        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = await self._embedding_client.get_embedding(main_query_text, context_type="default_query")
        except Exception as e:
            logger.error(f"Failed to get embedding for query: {e}")

        # Search across all memory types
        scored_candidates: List[Tuple[AnyMemoryType, float]] = []
        query_keywords_set = set(re.findall(r'\b\w{3,15}\b', main_query_text.lower()))

        all_memory_sources: List[Tuple[List[AnyMemoryType], str]] = [
            (self._in_memory_explicit_cache, "explicit"),
            (self._in_memory_goals_cache, "goal_record"),
            (self._in_memory_kg_entities_cache, "kg_entity_record"),
            (self._in_memory_kg_relations_cache, "kg_relation_record"),
            (self._in_memory_emotional_cache, "emotional"),
            (self._in_memory_procedural_cache, "procedural"),
        ]

        for memory_cache, memory_type_str in all_memory_sources:
            for mem in memory_cache:
                keyword_score = 0.0
                semantic_score = 0.0

                searchable_text, inherent_keywords = await self._get_searchable_text_and_keywords(mem)
                all_mem_keywords = set(str(k).lower() for k in inherent_keywords)

                # Calculate keyword score
                if searchable_text:
                    for qk in query_keywords_set:
                        if qk in searchable_text.lower():
                            keyword_score += 1.0

                for mk in all_mem_keywords:
                    if mk in query_keywords_set:
                        keyword_score += 2.0

                # Calculate semantic score
                mem_id = getattr(mem, 'memory_id', None)
                if query_embedding and mem_id and mem_id in self._embedding_cache:
                    memory_embedding = self._embedding_cache[mem_id]
                    semantic_score = compute_adaptive_similarity(query_embedding, memory_embedding)
                    semantic_score = max(0, semantic_score)

                # Calculate final score
                if keyword_score > 0 or semantic_score > 0:
                    dynamic_salience = getattr(mem, 'dynamic_salience_score', 0.5)
                    timestamp_val = getattr(mem, 'timestamp', 0)
                    recency_factor = 1.0 / (1 + (time.time() - timestamp_val) / (86400 * 30))

                    final_score = (DEFAULT_KEYWORD_SCORE_WEIGHT * keyword_score +
                                   DEFAULT_SEMANTIC_SCORE_WEIGHT * semantic_score)
                    final_score *= (1 + dynamic_salience)
                    final_score *= recency_factor

                    if final_score > 0.01:
                        scored_candidates.append((mem, final_score))

        # Sort by score and return top k
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Update salience for retrieved memories
        if MEMORY_LIFECYCLE_MANAGER_LOADED:
            for mem, _ in scored_candidates[:top_k]:
                update_dynamic_salience(mem, access_type="retrieval")

        return scored_candidates[:top_k]


    async def add_specific_memory(self, memory_object: AnyMemoryType):
        # Ensure memory_type is a string for consistent handling
        memory_type_attr_val = getattr(memory_object, 'memory_type', type(memory_object).__name__)
        memory_type_attr = str(
            memory_type_attr_val.value if hasattr(memory_type_attr_val, 'value') else memory_type_attr_val)
        memory_id = getattr(memory_object, 'memory_id', f"unknown_id_{time.time_ns()}")

        logger.info(f"MBS: add_specific_memory called for ID '{memory_id}', type_attr '{memory_type_attr}'")

        await self._ensure_embedding_for_memory(memory_object)

        # Lifecycle initialization
        if MEMORY_LIFECYCLE_MANAGER_LOADED:
            # Default salience/decay can be based on memory_type_attr
            default_salience_map = {
                "goal_record": 0.65, "kg_entity_record": 0.7, "kg_relation_record": 0.65,
                "emotional": 0.6, "procedural": 0.65, "explicit": 0.55
            }
            default_decay_map = {
                "goal_record": 0.015, "kg_entity_record": 0.005, "kg_relation_record": 0.005,
                "emotional": 0.01, "procedural": 0.008, "explicit": 0.007
            }
            default_s = default_salience_map.get(memory_type_attr, 0.5)
            default_d = default_decay_map.get(memory_type_attr, 0.01)

            if not hasattr(memory_object, 'dynamic_salience_score'): setattr(memory_object, 'dynamic_salience_score',
                                                                             default_s)
            if not hasattr(memory_object, 'decay_rate'): setattr(memory_object, 'decay_rate', default_d)
            if not hasattr(memory_object, 'access_count'): setattr(memory_object, 'access_count', 0)
            if not hasattr(memory_object, 'last_accessed_ts') or getattr(memory_object,
                                                                         'last_accessed_ts') is None: setattr(
                memory_object, 'last_accessed_ts', getattr(memory_object, 'timestamp', time.time()))

            initialize_dynamic_salience(memory_object)
            update_dynamic_salience(memory_object, access_type="commit")

        # Determine target cache and save/rewrite methods based on memory_type_attr
        target_cache: Optional[List[AnyMemoryType]] = None
        save_method = None
        rewrite_method = None
        id_attr_for_cache_update = 'memory_id'

        if memory_type_attr == 'explicit':
            target_cache, save_method, rewrite_method = self._in_memory_explicit_cache, self._save_explicit_memory_to_aggregated_store, self._rewrite_aggregated_store
        elif memory_type_attr == 'goal_record':
            target_cache, save_method, rewrite_method = self._in_memory_goals_cache, self._save_goal_record_to_store, self._rewrite_goals_store
        elif memory_type_attr == 'kg_entity_record':
            target_cache, save_method, rewrite_method = self._in_memory_kg_entities_cache, self._save_kg_entity_to_store, self._rewrite_kg_entities_store
            id_attr_for_cache_update = 'entity_id_str'
        elif memory_type_attr == 'kg_relation_record':
            target_cache, save_method, rewrite_method = self._in_memory_kg_relations_cache, self._save_kg_relation_to_store, self._rewrite_kg_relations_store
            id_attr_for_cache_update = 'relation_id_str'
        elif memory_type_attr == 'emotional':
            target_cache, save_method, rewrite_method = self._in_memory_emotional_cache, self._save_emotional_memory_to_store, self._rewrite_emotional_store
        elif memory_type_attr == 'procedural':
            target_cache, save_method, rewrite_method = self._in_memory_procedural_cache, self._save_procedural_memory_to_store, self._rewrite_procedural_store
        # Add other types here...

        if target_cache is not None and save_method and rewrite_method:
            unique_id_value = getattr(memory_object, id_attr_for_cache_update, memory_id)
            idx = next((i for i, item in enumerate(target_cache) if
                        getattr(item, id_attr_for_cache_update, None) == unique_id_value), -1)

            if idx != -1:  # Update existing
                # Preserve some attributes from existing if new object doesn't have them (e.g. placeholder updates)
                existing_item = target_cache[idx]
                for attr_name in ['memory_id', 'access_count', 'timestamp', 'dynamic_salience_score', 'decay_rate',
                                  'last_accessed_ts']:
                    if not hasattr(memory_object, attr_name) or getattr(memory_object, attr_name) is None:
                        if hasattr(existing_item, attr_name):
                            setattr(memory_object, attr_name, getattr(existing_item, attr_name))

                target_cache[idx] = memory_object
                logger.info(f"MBS: Updated {memory_type_attr} '{unique_id_value}' (memory_id: {memory_id}) in cache.")
                await rewrite_method()
            else:  # Add new
                target_cache.append(memory_object)
                logger.info(f"MBS: Added new {memory_type_attr} '{unique_id_value}' (memory_id: {memory_id}).")
                await save_method(memory_object)
        else:
            logger.warning(
                f"MBS: add_specific_memory for unsupported/unmapped type '{memory_type_attr}', ID '{memory_id}'. Not stored.")

    async def get_memory_by_id(self, memory_id: str) -> Optional[AnyMemoryType]:
        # Simplified: iterate all known caches
        all_caches = [
            self._in_memory_explicit_cache, self._in_memory_goals_cache,
            self._in_memory_kg_entities_cache, self._in_memory_kg_relations_cache,
            self._in_memory_emotional_cache, self._in_memory_procedural_cache
        ]
        for cache in all_caches:
            for mem in cache:
                # Primary check by memory_id
                if getattr(mem, 'memory_id', None) == memory_id:
                    if MEMORY_LIFECYCLE_MANAGER_LOADED: update_dynamic_salience(mem, access_type="retrieval")
                    return mem
                # Fallback checks for KG types by their specific ID fields if memory_id doesn't match
                if hasattr(mem, 'entity_id_str') and getattr(mem, 'entity_id_str') == memory_id:
                    if MEMORY_LIFECYCLE_MANAGER_LOADED: update_dynamic_salience(mem, access_type="retrieval")
                    return mem
                if hasattr(mem, 'relation_id_str') and getattr(mem, 'relation_id_str') == memory_id:
                    if MEMORY_LIFECYCLE_MANAGER_LOADED: update_dynamic_salience(mem, access_type="retrieval")
                    return mem
        return None

    # --- KG Traversal Methods (existing are fine, ensure salience updates use the manager) ---
    async def get_entity_by_id_str(self, entity_id_str: str, update_salience: bool = True) -> Optional[KGEntityRecord]:
        for entity in self._in_memory_kg_entities_cache:
            if getattr(entity, 'entity_id_str', None) == entity_id_str:
                if update_salience and MEMORY_LIFECYCLE_MANAGER_LOADED:
                    update_dynamic_salience(entity, access_type="retrieval")  # type: ignore
                logger.debug(f"MBS: Retrieved KG entity by entity_id_str: {entity_id_str}")
                return entity  # type: ignore
        logger.debug(f"MBS: KG entity with entity_id_str '{entity_id_str}' not found.")
        return None

    async def get_direct_relations(self, entity_id_str: str,
                                   relation_label: Optional[str] = None,
                                   direction: Literal["outgoing", "incoming", "both"] = "both",
                                   update_salience: bool = True) -> List[KGRelationRecord]:
        found_relations: List[KGRelationRecord] = []
        if not entity_id_str: return found_relations

        for relation in self._in_memory_kg_relations_cache:
            match = False
            is_source = getattr(relation, 'source_entity_id_str', None) == entity_id_str
            is_target = getattr(relation, 'target_entity_id_str', None) == entity_id_str

            if direction == "outgoing" and is_source:
                match = True
            elif direction == "incoming" and is_target:
                match = True
            elif direction == "both" and (is_source or is_target):
                match = True

            if match and relation_label:
                if getattr(relation, 'relation_label', "").lower() != relation_label.lower():
                    match = False
            if match:
                found_relations.append(relation)  # type: ignore

        if update_salience and MEMORY_LIFECYCLE_MANAGER_LOADED:
            for rel in found_relations:
                update_dynamic_salience(rel, access_type="retrieval")  # type: ignore

        logger.debug(
            f"MBS: Found {len(found_relations)} relations for entity '{entity_id_str}' (label: {relation_label}, dir: {direction}).")
        return found_relations

    async def get_neighboring_entities(self, entity_id_str: str,
                                       relation_label: Optional[str] = None,
                                       direction: Literal["outgoing", "incoming", "both"] = "both",
                                       update_salience: bool = True) -> List[KGEntityRecord]:
        neighboring_entities_dict: Dict[str, KGEntityRecord] = {}
        if not entity_id_str: return []

        relations = await self.get_direct_relations(entity_id_str, relation_label, direction, update_salience=False)

        for relation in relations:
            neighbor_id_to_fetch = None
            is_source = getattr(relation, 'source_entity_id_str', None) == entity_id_str
            is_target = getattr(relation, 'target_entity_id_str', None) == entity_id_str

            if is_source and (direction == "outgoing" or direction == "both"):
                neighbor_id_to_fetch = getattr(relation, 'target_entity_id_str', None)
            elif is_target and (
                    direction == "incoming" or direction == "both"):  # Use elif to avoid double add for self-loops in "both"
                neighbor_id_to_fetch = getattr(relation, 'source_entity_id_str', None)

            if neighbor_id_to_fetch and neighbor_id_to_fetch not in neighboring_entities_dict:
                neighbor_entity = await self.get_entity_by_id_str(neighbor_id_to_fetch, update_salience=update_salience)
                if neighbor_entity:
                    neighboring_entities_dict[neighbor_id_to_fetch] = neighbor_entity

        logger.debug(
            f"MBS: Found {len(neighboring_entities_dict)} neighboring entities for '{entity_id_str}' (label: {relation_label}, dir: {direction}).")
        return list(neighboring_entities_dict.values())

    # --- Lifecycle Management Tasks (Adjusted for new caches) ---
    async def _periodic_decay_task(self, interval_seconds: int):
        while not self._stop_background_tasks.is_set():
            try:
                await asyncio.sleep(interval_seconds)
                if self._stop_background_tasks.is_set(): break
                logger.info("MBSMemoryService: Running periodic memory decay...")
                if MEMORY_LIFECYCLE_MANAGER_LOADED:
                    all_caches_for_decay = [
                        self._in_memory_explicit_cache, self._in_memory_goals_cache,
                        self._in_memory_kg_entities_cache, self._in_memory_kg_relations_cache,
                        self._in_memory_emotional_cache, self._in_memory_procedural_cache
                    ]
                    for mem_list_item in all_caches_for_decay:
                        apply_decay_to_all_memories(mem_list_item)  # type: ignore
                logger.info("MBSMemoryService: Periodic memory decay complete.")
            except asyncio.CancelledError:
                logger.info("MBS: Decay task cancelled."); break
            except Exception as e:
                logger.error(f"MBS: Error in decay task: {e}", exc_info=True)

    async def _periodic_archive_forget_task(self, interval_seconds: int):
        while not self._stop_background_tasks.is_set():
            try:
                await asyncio.sleep(interval_seconds)
                if self._stop_background_tasks.is_set(): break
                logger.info("MBSMemoryService: Running periodic archive/forget task...")
                if MEMORY_LIFECYCLE_MANAGER_LOADED:
                    ephemeral_cfg = os.getenv("MBS_EPHEMERAL_SOURCES", ",".join(
                        str(s.value if hasattr(s, 'value') else s) for s in EPHEMERAL_SOURCE_TYPES_FOR_DELETION))
                    ephemeral_types_str_list = [s.strip() for s in ephemeral_cfg.split(',') if s.strip()]
                    ephemeral_srcs = []
                    if MEMORY_TYPES_LOADED_SUCCESSFULLY:
                        for ets_str in ephemeral_types_str_list:
                            try:
                                ephemeral_srcs.append(MemorySourceType(ets_str))  # type: ignore
                            except ValueError:
                                logger.warning(
                                    f"MBS: Invalid MemorySourceType '{ets_str}' in env for ephemeral config.")
                    else:
                        ephemeral_srcs = ephemeral_types_str_list  # Use as strings if enums not loaded

                    # Process each cache type
                    cache_configs = [
                        ("explicit", self._in_memory_explicit_cache, self._rewrite_aggregated_store,
                         float(os.getenv("MBS_EXPLICIT_ARCHIVE_THRESHOLD", DEFAULT_ARCHIVE_SALIENCE_THRESHOLD))),
                        ("goals", self._in_memory_goals_cache, self._rewrite_goals_store,
                         float(os.getenv("MBS_GOAL_ARCHIVE_THRESHOLD", 0.05))),
                        ("kg_entities", self._in_memory_kg_entities_cache, self._rewrite_kg_entities_store,
                         float(os.getenv("MBS_KG_ENTITY_ARCHIVE_THRESHOLD", 0.02))),
                        ("kg_relations", self._in_memory_kg_relations_cache, self._rewrite_kg_relations_store,
                         float(os.getenv("MBS_KG_RELATION_ARCHIVE_THRESHOLD", 0.02))),
                        ("emotional", self._in_memory_emotional_cache, self._rewrite_emotional_store,
                         float(os.getenv("MBS_EMOTIONAL_ARCHIVE_THRESHOLD", 0.08))),
                        ("procedural", self._in_memory_procedural_cache, self._rewrite_procedural_store,
                         float(os.getenv("MBS_PROCEDURAL_ARCHIVE_THRESHOLD", 0.03))),
                    ]

                    any_changes = False
                    for mem_type_name, cache_ref, rewrite_func, threshold in cache_configs:
                        kept, archived, forgotten = archive_or_forget_low_salience_memories(cache_ref,
                                                                                            archive_threshold=threshold,
                                                                                            ephemeral_sources=ephemeral_srcs)  # type: ignore
                        if forgotten or archived:
                            any_changes = True
                            # Update the in-memory cache reference
                            if mem_type_name == "explicit":
                                self._in_memory_explicit_cache = kept  # type: ignore
                            elif mem_type_name == "goals":
                                self._in_memory_goals_cache = kept  # type: ignore
                            elif mem_type_name == "kg_entities":
                                self._in_memory_kg_entities_cache = kept  # type: ignore
                            elif mem_type_name == "kg_relations":
                                self._in_memory_kg_relations_cache = kept  # type: ignore
                            elif mem_type_name == "emotional":
                                self._in_memory_emotional_cache = kept  # type: ignore
                            elif mem_type_name == "procedural":
                                self._in_memory_procedural_cache = kept  # type: ignore

                            if forgotten: logger.info(f"MBS: Forgetting {len(forgotten)} {mem_type_name} memories.")
                            if archived: logger.info(
                                f"MBS: Archiving {len(archived)} {mem_type_name} memories."); await self._archive_memories(
                                archived, archive_type_name=mem_type_name)
                            await rewrite_func()

                    if not any_changes:
                        logger.info(
                            "MBSMemoryService: No memories met criteria for archiving or forgetting this cycle.")
                logger.info("MBSMemoryService: Periodic archive/forget task complete.")
            except asyncio.CancelledError:
                logger.info("MBS: Archive/forget task cancelled."); break
            except Exception as e:
                logger.error(f"MBS: Error in archive/forget task: {e}", exc_info=True)

    async def _archive_memories(self, memories_to_archive: List[AnyMemoryType], archive_type_name: str):
        # Simplified archive filename based on type_name
        archive_filename = f"archived_{archive_type_name}_memories.jsonl"
        archive_filepath = self.store_path / archive_filename
        try:
            for memory_for_archive in memories_to_archive:
                await self._ensure_embedding_for_memory(memory_for_archive)

            with open(archive_filepath, 'a', encoding='utf-8') as f:
                for memory in memories_to_archive:
                    record_data_dict = {}
                    if hasattr(memory, "model_dump"):
                        record_data_dict = memory.model_dump(exclude_none=True)
                    else:  # Fallback for placeholders
                        record_data_dict = memory.__dict__

                    mem_id_for_embedding = record_data_dict.get(
                        getattr(memory, '_id_attr_for_embedding_lookup', 'memory_id')) \
                                           or record_data_dict.get('memory_id')
                    if mem_id_for_embedding and mem_id_for_embedding in self._embedding_cache:
                        record_data_dict["_embedding_vector_for_jsonl_"] = self._embedding_cache[mem_id_for_embedding]
                    f.write(json.dumps(record_data_dict) + "\n")
            logger.info(f"MBS Store: Appended {len(memories_to_archive)} memories to archive: {archive_filepath}")
        except Exception as e:
            logger.error(f"MBS Store: Failed to archive memories to {archive_filepath}: {e}", exc_info=True)

    def start_lifecycle_management_tasks(self, decay_interval: int = 3600 * 6, archive_interval: int = 86400):

        if not MEMORY_LIFECYCLE_MANAGER_LOADED:
            logger.warning("MBSMemoryService: Memory Lifecycle Manager not loaded. Background tasks will not start.")
            return
        logger.info(
            f"MBSMemoryService: Starting background tasks (Decay: {decay_interval}s, Archive: {archive_interval}s).")
        self._stop_background_tasks.clear()
        if decay_interval > 0: self._background_tasks.append(
            asyncio.create_task(self._periodic_decay_task(decay_interval)))
        if archive_interval > 0: self._background_tasks.append(
            asyncio.create_task(self._periodic_archive_forget_task(archive_interval)))

    async def stop_lifecycle_management_tasks(self):

        if not self._background_tasks: return
        logger.info("MBSMemoryService: Stopping background lifecycle tasks...")
        self._stop_background_tasks.set()
        await asyncio.sleep(0.1)
        for task in self._background_tasks:
            if not task.done(): task.cancel()
        results = await asyncio.gather(*self._background_tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                logger.error(f"MBSMemoryService: Error during background task {i + 1} shutdown: {result}")
        self._background_tasks = []
        logger.info("MBSMemoryService: Background tasks stopped.")