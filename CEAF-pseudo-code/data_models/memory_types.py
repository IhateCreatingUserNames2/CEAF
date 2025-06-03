# data_models/memory_types.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Literal, Optional
import time

MemoryType = Literal[
    "explicit", "emotional", "procedural",
    "flashbulb", "somatic", "liminal", "generative"
]

class MemoryBlossomEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())) # Assuming uuid is imported
    content: str
    memory_type: MemoryType
    timestamp: float = Field(default_factory=time.time)
    embedding: Optional[List[float]] = None # Stored separately or on demand
    metadata: Dict[str, Any] = Field(default_factory=dict) # e.g., emotional_salience, source, narrative_thread_id
    source_event_id: Optional[str] = None
    user_id: str
    session_id: str