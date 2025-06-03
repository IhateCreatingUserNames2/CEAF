    ceaf_prototype/
    │
    ├── .env                     # API keys (OPENROUTER_API_KEY, etc.)
    │
    ├── main_runner.py           # Main script to run the CEAF agent
    │
    ├── ceaf_agent.py            # Defines the Central Orchestrator/Responder Agent (ORA)
    │
    ├── modules/
    │   ├── __init__.py
    │   ├── ncf_engine.py        # Narrative Context Framing Engine (Context Weaver, NCF Param Controller)
    │   ├── memory_blossom.py    # MemoryBlossom System (Multi-Type Stores, CARS, Synthesizer)
    │   ├── identity_module.py   # Narrative Coherence & Identity Module (NCIM)
    │   ├── reasoning_module.py  # Virtue & Reasoning Engine (VRE)
    │   └── metacontrol_loop.py  # Metacognitive Control Loop (MCL)
    │
    ├── core_logic/
    │   ├── __init__.py
    │   ├── adk_setup.py         # ADK Runner, SessionService, ArtifactService setup
    │   ├── llm_models.py        # LiteLLM model definitions
    │   └── event_processing.py  # Helpers for handling ADK events
    │
    ├── data_models/
    │   ├── __init__.py
    │   ├── memory_types.py      # Pydantic models for MemoryBlossom types
    │   └── identity_state.py    # Pydantic model for agent identity
    │
    └── utils/
        ├── __init__.py
        └── helpers.py           # Utility functions (e.g., entropy calculation)
