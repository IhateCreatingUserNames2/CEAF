# data_models/identity_state.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class IdentityElement(BaseModel):
    name: str
    description: str
    value: Any # Could be a belief, trait, persona aspect
    confidence: float = 1.0 # How strongly this element is held
    entropy: float = 0.0    # How much flux/change this element is undergoing
    source: str = "initial" # initial, learned, inferred, NCF_induced

class CeafIdentity(BaseModel):
    current_persona_name: str = "DefaultCEAF"
    core_self_description: str = "I am a Coherent Emergence Agent."
    elements: List[IdentityElement] = Field(default_factory=list)
    # Could include preferred NCF parameters here
    preferred_philosophical_framing: str = "epistemic_humility_focused"
    preferred_narrative_depth: float = 0.7
    # ... other identity-related state