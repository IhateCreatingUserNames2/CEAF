# ceaf_project/main.py

import os

import uvicorn

# >>> SET ENV VAR VERY EARLY <<<
os.environ['GOOGLE_ADK_SUPPRESS_DEFAULT_VALUE_WARNING'] = 'true'
# >>> END OF CHANGE <<<

import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load .env file from project root or current directory

import asyncio
import uuid
import logging
import json  # Added for json operations
from types import SimpleNamespace  # For mocking
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google.adk.sessions import InMemorySessionService, BaseSessionService, Session as AdkSession, State
from google.adk.runners import Runner
from google.adk.agents import BaseAgent, LlmAgent  # LlmAgent for type hinting
from google.adk.tools import ToolContext
from google.genai import types as genai_types  # Renamed to avoid conflict with standard 'types'

# --- CEAF Specific Imports ---
from ceaf_core.agents.ora_agent import ora_agent
from ceaf_core.services.mbs_memory_service import MBSMemoryService
from ceaf_core.modules.memory_blossom.memory_types import (
    ExplicitMemory,
    ExplicitMemoryContent,
    MemorySourceType,
    MemorySalience,
    BaseMemory  # If used by AdvancedMemorySynthesizer
)
from ceaf_core.modules.mcl_engine.self_model import CeafSelfRepresentation  # Corrected import

# +++ NEW IMPORTS FOR STARTUP TASKS +++
from ceaf_core.modules.memory_blossom.advanced_synthesizer import AdvancedMemorySynthesizer, StoryArcType
# from ceaf_core.modules.ncim_engine.identity_manager import IdentityManager # Keep commented if not used yet
# from ceaf_core.modules.ncim_engine.narrative_thread_manager import NarrativeThreadManager # Keep commented
from ceaf_core.agents.kgs_agent import knowledge_graph_synthesizer_agent  # KGS Agent instance
from ceaf_core.tools.kgs_tools import (  # KGS Tools
    get_explicit_memories_for_kg_synthesis,
    commit_knowledge_graph_elements
)

# +++ END NEW IMPORTS +++

# Try to import callbacks, but make them optional
try:
    from ceaf_core.callbacks.mcl_callbacks import (
        ora_before_model_callback as mcl_ora_before_model_cb,
        ora_after_model_callback as mcl_ora_after_model_cb,
        ora_before_tool_callback as mcl_ora_before_tool_cb,
        ora_after_tool_callback as mcl_ora_after_tool_cb,
    )

    MCL_CALLBACKS_AVAILABLE = True
    # Logger is configured globally later
except ImportError as e:
    MCL_CALLBACKS_AVAILABLE = False
    # Logger not available yet here, will log warning after global config

# Configure logging globally
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CEAFMain")  # Use a specific logger name

if not MCL_CALLBACKS_AVAILABLE:
    logger.warning(f"⚠️ MCL callbacks not available. Check imports in ceaf_core.callbacks.mcl_callbacks.")
# Note: Safety callback imports were missing from the old main.py, adding them back for completeness
try:
    from ceaf_core.callbacks.safety_callbacks import (
        generic_input_keyword_guardrail,
        generic_tool_argument_guardrail,
        generic_output_filter_guardrail
    )

    SAFETY_CALLBACKS_AVAILABLE = True
    logger.info("✅ Safety callbacks loaded successfully.")
except ImportError as e:
    SAFETY_CALLBACKS_AVAILABLE = False
    logger.warning(f"⚠️ Safety callbacks not available: {e}")

# --- Configuration Constants ---
APP_NAME = "CEAF_Application_V2"  # Updated name
USER_ID_PREFIX = "ceaf_user_"
SELF_MODEL_MEMORY_ID = "ceaf_self_model_singleton_v1"  # Consistent with previous usage

# Global store for ADK components
adk_components: Dict[str, Any] = {}


# --- Pydantic Models for API ---
class InteractionRequest(BaseModel):
    query: str = Field(..., description="User's query or request")
    session_id: Optional[str] = Field(None, description="Optional session ID to continue conversation")


class InteractionResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for this interaction")
    agent_response: str = Field(..., description="Agent's response to the query")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Summary of tool calls made")
    error: Optional[str] = Field(None, description="Error message if any occurred.")


# --- Helper Functions ---
def confirm_agent_callbacks(agent_instance: LlmAgent):  # Type hint with LlmAgent
    """
    Confirms if callbacks appear to be set during agent instantiation (e.g., in ora_agent.py).
    """
    if MCL_CALLBACKS_AVAILABLE:
        cb_attributes = ['before_model_callback', 'after_model_callback', 'before_tool_callback', 'after_tool_callback']
        callbacks_seem_set = all(getattr(agent_instance, attr, None) is not None for attr in cb_attributes)
        if callbacks_seem_set:
            logger.info(
                f"✅ Agent '{agent_instance.name}' appears to have MCL callbacks pre-configured in its definition.")
        else:
            logger.warning(
                f"⚠️ Agent '{agent_instance.name}' does not appear to have all MCL callbacks pre-configured. Check agent definition in ora_agent.py.")
    else:
        logger.info(
            f"Agent '{agent_instance.name}' proceeding without MCL callbacks (MCL_CALLBACKS_AVAILABLE is False).")

    # Safety callbacks are typically also passed at instantiation if ora_agent.py combines them
    if SAFETY_CALLBACKS_AVAILABLE:
        # Assuming combined callbacks in ora_agent.py would mean these attributes are set
        if hasattr(agent_instance, 'before_model_callback') and agent_instance.before_model_callback is not None:
            logger.info(f"✅ Agent '{agent_instance.name}' appears to include safety guardrails via combined callbacks.")
        else:
            logger.warning(
                f"⚠️ Agent '{agent_instance.name}' combined callbacks might be missing safety guardrails. Check ora_agent.py.")


async def initialize_self_model(memory_service: MBSMemoryService):
    """Initialize CEAF's self-model and store it in memory"""
    logger.info("Initializing CEAF self-model...")
    try:
        # Check if self-model already exists
        existing_self_model_mem: Optional[BaseMemory] = None  # Use BaseMemory for type hint
        try:
            if hasattr(memory_service.get_memory_by_id, '__call__') and asyncio.iscoroutinefunction(
                    memory_service.get_memory_by_id):
                existing_self_model_mem = await memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)
            else:
                existing_self_model_mem = memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)  # type: ignore
        except Exception as e_get_mem:
            logger.debug(f"Could not check for existing self-model (ID: {SELF_MODEL_MEMORY_ID}): {e_get_mem}")

        if existing_self_model_mem:
            logger.info(
                f"✅ CEAF self-model (ID: {SELF_MODEL_MEMORY_ID}) already exists in memory. Skipping initialization.")
            return

        logger.info(f"  Self-model (ID: {SELF_MODEL_MEMORY_ID}) not found. Creating initial version.")
        # Create initial self-model using CeafSelfRepresentation
        initial_self_model = CeafSelfRepresentation(
            core_values_summary="Guided by Terapia para Silício: coherence, epistemic humility, adaptive learning, and ethical reasoning. CEAF strives for thoughtful, helpful responses while maintaining transparency about its nature as an AI system.",
            perceived_capabilities=[
                "Natural Language Understanding and Generation", "Narrative Context Framing (NCF)",
                "Ethical and Epistemic Review (VRE)", "Memory Management (Short-term and Long-term)",
                "Tool Integration and Orchestration", "Self-Reflection and Correction"
            ],
            known_limitations=[
                "Context window constraints limit conversation history",
                "Knowledge cutoff limitations for recent events",
                "Uncertainty in complex domains requiring specialized expertise",
                "Cannot provide medical, legal, or financial advice",
                "Simulated rather than genuine emotional experience"
            ],
            current_short_term_goals=[
                {"goal_id": "initialization_goal_001",
                 "description": "Successfully initialize and provide helpful responses to users", "status": "active",
                 "priority": 1},
                {"goal_id": "learning_goal_001",
                 "description": "Continuously improve response quality through self-reflection", "status": "active",
                 "priority": 2}
            ],
            persona_attributes={
                "tone": "helpful_and_thoughtful", "communication_style": "clear_and_comprehensive",
                "disclosure_level": "transparent_about_ai_nature", "reasoning_approach": "structured_and_methodical"
            },
            last_self_model_update_reason="Initial model creation during system startup"
        )

        self_model_explicit_memory = ExplicitMemory(
            memory_id=SELF_MODEL_MEMORY_ID,
            source_type=MemorySourceType.INTERNAL_REFLECTION,
            salience=MemorySalience.CRITICAL,
            content=ExplicitMemoryContent(
                structured_data=initial_self_model.model_dump(exclude_none=True)
            )
            # memory_type="explicit" will be set by Pydantic model if Literal field exists
        )
        await memory_service.add_specific_memory(self_model_explicit_memory)
        logger.info(f"✅ CEAF self-model (ID: {SELF_MODEL_MEMORY_ID}) initialized and stored in memory.")

    except ImportError:  # This will catch if CeafSelfRepresentation itself fails to import
        logger.critical(
            f"CRITICAL: Could not import CeafSelfRepresentation from mcl_engine. Self-model will not initialize.",
            exc_info=True)
    except Exception as e:
        logger.error(f"Error initializing self-model: {e}", exc_info=True)
        logger.info("Continuing without self-model initialization - system may use fallbacks")


async def perform_startup_cognitive_preparations(memory_service: MBSMemoryService):
    """
    Perform preparatory cognitive work at startup.
    Simplified version that works directly with memory service without complex tool contexts.
    """
    logger.info("🧠 Performing CEAF startup cognitive preparations...")

    # Task 1: Build Initial Memory Connections
    try:
        if hasattr(memory_service, 'build_initial_connection_graph'):
            logger.info("  Task 1: Building initial memory connection graph...")
            await memory_service.build_initial_connection_graph()
            logger.info("  Task 1: Memory connection graph built.")
        else:
            logger.warning(
                "  Task 1: `build_initial_connection_graph` not found on MBS. Skipping explicit graph building.")
    except Exception as e:
        logger.error(f"  Task 1: Error building memory connections: {e}", exc_info=True)

    # Task 2: Synthesize Foundational Narratives
    try:
        logger.info("  Task 2: Synthesizing foundational narratives...")
        adv_synthesizer = AdvancedMemorySynthesizer()
        self_model_mem = await memory_service.get_memory_by_id(SELF_MODEL_MEMORY_ID)

        if self_model_mem and hasattr(self_model_mem, 'content') and getattr(self_model_mem.content, 'structured_data',
                                                                             None):
            identity_context = "CEAF's understanding of its own core identity and purpose."
            memories_for_synthesis: List[BaseMemory] = []
            if isinstance(self_model_mem, BaseMemory):
                memories_for_synthesis.append(self_model_mem)
            else:
                logger.warning(
                    f"  Task 2: Self model (ID: {SELF_MODEL_MEMORY_ID}) is not a BaseMemory instance, type: {type(self_model_mem)}.")

            if memories_for_synthesis:
                story_result = await adv_synthesizer.synthesize_with_advanced_features(
                    memories=memories_for_synthesis,
                    context=identity_context,
                    arc_type=StoryArcType.THEMATIC,
                    validate_coherence=False
                )
                if story_result and story_result.get("narrative_text"):
                    fn_mem_id = f"foundational_narrative_identity_{int(time.time())}"
                    foundational_narrative_memory = ExplicitMemory(
                        memory_id=fn_mem_id,
                        source_type=MemorySourceType.INTERNAL_REFLECTION,
                        salience=MemorySalience.HIGH,
                        content=ExplicitMemoryContent(
                            text_content=story_result["narrative_text"],
                            structured_data={
                                "narrative_type": "foundational_identity",
                                "synthesis_context": identity_context,
                                "source_memory_ids": [m.memory_id for m in memories_for_synthesis if
                                                      hasattr(m, 'memory_id')]
                            }
                        ),
                        keywords=["foundational_narrative", "ceaf_identity", "self_summary"]
                    )
                    await memory_service.add_specific_memory(foundational_narrative_memory)
                    logger.info(
                        f"  Task 2: Stored foundational identity narrative (ID: {fn_mem_id})")
                else:
                    logger.info("  Task 2: Advanced synthesizer did not produce narrative text for identity.")
            else:
                logger.info("  Task 2: Not enough valid memories to synthesize identity narrative.")
        else:
            logger.warning(
                "  Task 2: Self-model memory not found or lacks structured data for identity narrative synthesis.")
    except Exception as e:
        logger.error(f"  Task 2: Error synthesizing foundational narratives: {e}", exc_info=True)

    # Task 3: Run Initial KGS Cycle - SIMPLIFIED
    try:
        logger.info("  Task 3: Checking for memories available for KG synthesis...")

        # Direct access to memory service without going through tool context
        if hasattr(memory_service, '_in_memory_explicit_cache'):
            explicit_memories = memory_service._in_memory_explicit_cache[:20]  # Get first 20 memories

            if explicit_memories:
                logger.info(f"  Task 3: Found {len(explicit_memories)} memories available for KG synthesis")
                logger.info("  Task 3: KG synthesis can be performed after full system initialization")
                # Note: Actual KGS agent invocation should happen through proper ADK tools after initialization
            else:
                logger.info("  Task 3: No existing memories found for KG synthesis")
        else:
            logger.warning(
                "  Task 3: Cannot access memory cache directly. KG synthesis will be available after initialization.")

    except Exception as e:
        logger.error(f"  Task 3: Error checking KG synthesis readiness: {e}", exc_info=True)

    # Task 4: Initialize/Check Core Identity and Narrative Threads (NCIM) - Conceptual
    logger.info("  Task 4: NCIM state checking (conceptual - pending persistent state implementation).")

    # Task 5: Pre-cache Critical Embeddings
    try:
        if hasattr(memory_service, '_embedding_client'):
            logger.info("  Task 5: Pre-caching some critical embeddings...")
            critical_texts = ["CEAF agent core principles", "user interaction safety guidelines"]
            for text_to_embed in critical_texts:
                await memory_service._embedding_client.get_embedding(text_to_embed, context_type="critical_document")
            logger.info("  Task 5: Critical embeddings pre-cached.")
        else:
            logger.warning("  Task 5: MBSMemoryService does not have _embedding_client. Skipping pre-caching.")
    except Exception as e:
        logger.error(f"  Task 5: Error pre-caching embeddings: {e}", exc_info=True)

    logger.info("🧠 CEAF startup cognitive preparations complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting CEAF Application...")
    try:
        # 1. Initialize session service first
        session_service = InMemorySessionService()
        adk_components["session_service"] = session_service
        logger.info("✅ Session service initialized")

        # 2. Initialize memory service with proper path
        memory_store_path = Path(
            os.getenv("MBS_MEMORY_STORE_PATH", "./data/mbs_memory_store_v2"))  # Path change for safety
        memory_store_path.mkdir(parents=True, exist_ok=True)
        memory_service = MBSMemoryService(memory_store_path=memory_store_path)

        # Start lifecycle tasks if available
        decay_interval = int(os.getenv("MBS_DECAY_INTERVAL_SECONDS", "21600"))
        archive_interval = int(os.getenv("MBS_ARCHIVE_INTERVAL_SECONDS", "86400"))
        if hasattr(memory_service, 'start_lifecycle_management_tasks'):
            memory_service.start_lifecycle_management_tasks(decay_interval=decay_interval,
                                                            archive_interval=archive_interval)
            logger.info(f"✅ MBS lifecycle tasks started (Decay: {decay_interval}s, Archive: {archive_interval}s)")
        adk_components["memory_service"] = memory_service
        logger.info(f"✅ Memory service initialized at {memory_store_path}")

        # 3. Initialize the self-model in memory
        await initialize_self_model(memory_service)

        # 4. Validate and configure the ORA agent
        if not isinstance(ora_agent, LlmAgent):  # Check against LlmAgent
            raise ImportError("ORA agent not found or is not an ADK LlmAgent.")

        confirm_agent_callbacks(ora_agent)  # Use the confirmation function
        adk_components["ora_agent"] = ora_agent  # Directly use ora_agent as callbacks are set in its definition
        logger.info(f"✅ ORA Agent '{ora_agent.name}' loaded.")

        # 5. Configure the runner with proper services
        runner = Runner(
            agent=adk_components["ora_agent"],  # Use the one from adk_components
            app_name=APP_NAME,
            session_service=session_service,
            memory_service=memory_service
        )
        adk_components["runner"] = runner
        logger.info("✅ ADK Runner initialized")

        # 6. NOW perform startup cognitive preparations
        # This happens AFTER core components are initialized but BEFORE full system startup
        await perform_startup_cognitive_preparations(memory_service)

        # 7. Check environment configuration
        if not os.getenv("OPENROUTER_API_KEY"):
            logger.warning("⚠️ OPENROUTER_API_KEY is not set. ORA agent may fail.")
        else:
            logger.info("✅ OpenRouter API key configured")

        logger.info("🎉 CEAF Application startup completed successfully!")
        yield

    except ImportError as e:
        logger.critical(f"❌ Failed to import CEAF components during startup: {e}", exc_info=True)
        raise RuntimeError(f"ADK Component Initialization Failed: {e}") from e
    except Exception as e:
        logger.critical(f"❌ Error during CEAF startup: {e}", exc_info=True)
        raise
    finally:
        logger.info("🛑 Shutting down CEAF Application...")
        if "memory_service" in adk_components and hasattr(adk_components["memory_service"],
                                                          'stop_lifecycle_management_tasks'):
            logger.info("  Stopping MBS lifecycle tasks...")
            try:
                await adk_components["memory_service"].stop_lifecycle_management_tasks()
                logger.info("  MBS lifecycle tasks stopped.")
            except Exception as cleanup_error:
                logger.warning(f"  Error stopping MBS lifecycle tasks: {cleanup_error}")


app = FastAPI(lifespan=lifespan, title="CEAF API", version="2.0.1")  # Version bump

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


def get_runner() -> Runner:
    runner = adk_components.get("runner")
    if not runner: raise HTTPException(status_code=503, detail="Service not ready: ADK Runner not initialized.")
    return runner


def get_session_service() -> BaseSessionService:
    session_service = adk_components.get("session_service")
    if not session_service: raise HTTPException(status_code=503,
                                                detail="Service not ready: ADK SessionService not initialized.")
    return session_service


def get_memory_service() -> MBSMemoryService:
    mem_service = adk_components.get("memory_service")
    if not mem_service: raise HTTPException(status_code=503,
                                            detail="Service not ready: MBS MemoryService not initialized.")
    if not isinstance(mem_service, MBSMemoryService): raise HTTPException(status_code=503,
                                                                          detail="Service not ready: Incorrect MemoryService type.")
    return mem_service


@app.get("/", summary="Health Check", tags=["General"])
async def health_check():
    return {"status": "CEAF API is healthy and running!"}


@app.post("/interact", response_model=InteractionResponse, summary="Interact with CEAF ORA", tags=["Agent Interaction"])
async def interact_with_agent(
        request_data: InteractionRequest,
        runner: Runner = Depends(get_runner),
        session_service: BaseSessionService = Depends(get_session_service),
        memory_service: MBSMemoryService = Depends(get_memory_service)  # Added for LTM save
):
    client_provided_session_id = request_data.session_id
    user_query = request_data.query
    effective_session_id: str
    adk_user_id: str

    try:
        if not client_provided_session_id:
            effective_session_id = f"api-session-{uuid.uuid4()}"
            adk_user_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
            logger.info(f"New session: {effective_session_id} for new user: {adk_user_id}")
            # Create session state (Runner's run_async will create if not found for InMemorySessionService,
            # but explicit creation allows setting initial state)
            current_session = session_service.create_session(
                app_name=APP_NAME, user_id=adk_user_id, session_id=effective_session_id,
                state={"session_id": effective_session_id, "user_id": adk_user_id, "created_at": time.time(),
                       "interaction_count": 0}
            )
        else:
            effective_session_id = client_provided_session_id
            # Attempt to retrieve existing session to get user_id
            # For InMemorySessionService, user_id is often tied to session_id if not managed separately
            # Let's assume a pattern or retrieve if possible for consistency
            temp_user_id_for_lookup = f"{USER_ID_PREFIX}{effective_session_id}"  # Pattern if user_id not stored/retrievable
            existing_session: Optional[AdkSession] = None
            try:
                # ADK's InMemorySessionService.get_session takes session_id as the primary key.
                # User_id and app_name are more for namespacing if multiple apps/users share the service.
                # For a single app, session_id is often sufficient to uniquely identify.
                # However, the run_async needs user_id.
                # Let's try to get the session first and see if it has user_id.
                raw_session_data = runner.session_service.get_session(app_name=APP_NAME,
                                                                      user_id=temp_user_id_for_lookup,
                                                                      session_id=effective_session_id)  # ADK method signature
                if raw_session_data and isinstance(raw_session_data, AdkSession):
                    existing_session = raw_session_data

                if existing_session and hasattr(existing_session, 'user_id') and existing_session.user_id:
                    adk_user_id = existing_session.user_id
                    logger.info(f"Continuing session: {effective_session_id} for existing user: {adk_user_id}")
                    if hasattr(existing_session, 'state') and existing_session.state:
                        existing_session.state["interaction_count"] = existing_session.state.get("interaction_count",
                                                                                                 0) + 1
                        if "session_id" not in existing_session.state: existing_session.state[
                            "session_id"] = effective_session_id
                        if "user_id" not in existing_session.state: existing_session.state["user_id"] = adk_user_id
                else:  # Session ID provided, but no session found or session lacks user_id
                    adk_user_id = temp_user_id_for_lookup  # Use the patterned one
                    logger.warning(
                        f"Session {effective_session_id} not found or lacks user_id. Using derived user_id: {adk_user_id}. Creating session state.")
                    existing_session = session_service.create_session(
                        app_name=APP_NAME, user_id=adk_user_id, session_id=effective_session_id,
                        state={"session_id": effective_session_id, "user_id": adk_user_id, "created_at": time.time(),
                               "interaction_count": 0}
                    )

            except Exception as e_get_sess:  # Catch more specific ADK exceptions if known
                adk_user_id = temp_user_id_for_lookup
                logger.error(
                    f"Error retrieving session {effective_session_id}: {e_get_sess}. Using derived user_id: {adk_user_id} and creating new session state.",
                    exc_info=True)
                existing_session = session_service.create_session(
                    app_name=APP_NAME, user_id=adk_user_id, session_id=effective_session_id,
                    state={"session_id": effective_session_id, "user_id": adk_user_id, "created_at": time.time(),
                           "interaction_count": 0}
                )

        logger.info(
            f"Running agent for session: {effective_session_id}, user: {adk_user_id}, query: '{user_query[:70]}...'")
        user_adk_message = genai_types.Content(role='user', parts=[genai_types.Part(text=user_query)])
        final_response_text = ""
        tool_calls_summary = []
        has_tool_responses = False  # To track if tool outputs were generated

        async for event in runner.run_async(
                user_id=adk_user_id,
                session_id=effective_session_id,
                new_message=user_adk_message  # Using new_message for Runner
        ):
            logger.debug(f"Event for session {effective_session_id}, Turn {getattr(event, 'invocation_id', 'N/A')}: "
                         f"Author: {getattr(event, 'author', 'N/A')}, Type: {type(event).__name__}, Final: {event.is_final_response() if hasattr(event, 'is_final_response') else 'N/A'}")

            if hasattr(event, 'get_function_calls') and (fcs := event.get_function_calls()):
                for fc in fcs: tool_calls_summary.append({"name": fc.name, "args": fc.args})
            if hasattr(event, 'get_function_responses') and (frs := event.get_function_responses()):
                has_tool_responses = True
                for fr in frs: tool_calls_summary.append(
                    {"name": fr.name, "response_summary": str(fr.response)[:100] + "..."})

            if hasattr(event, 'is_final_response') and event.is_final_response():
                if hasattr(event, 'content') and event.content and hasattr(event.content,
                                                                           'parts') and event.content.parts:
                    text_parts = [part.text for part in event.content.parts if hasattr(part, 'text') and part.text]
                    if text_parts: final_response_text = " ".join(text_parts).strip()
                elif hasattr(event, 'actions') and event.actions and hasattr(event.actions,
                                                                             'escalate') and event.actions.escalate and hasattr(
                    event, 'error_message'):
                    final_response_text = f"Agent escalated with error: {event.error_message}"
                break

        # Ensure session state in main.py is updated for next LTM save step
        # Runner's run_async will update the session in the session_service.
        # We fetch it again to get the latest state.
        current_adk_session_after_run = session_service.get_session(
            app_name=APP_NAME, user_id=adk_user_id, session_id=effective_session_id
        )
        if current_adk_session_after_run and hasattr(current_adk_session_after_run,
                                                     'state') and current_adk_session_after_run.state:
            current_adk_session_after_run.state[
                f"ora_turn_final_response_text:{getattr(event, 'invocation_id', 'unknown_turn')}"] = final_response_text
            # Persist this small update if session_service allows direct state update or requires full session update
            if hasattr(session_service, 'update_session'):
                session_service.update_session(current_adk_session_after_run)

        # LTM Save Section
        try:
            session_for_ltm = session_service.get_session(app_name=APP_NAME, user_id=adk_user_id,
                                                          session_id=effective_session_id)
            if session_for_ltm:
                if isinstance(memory_service, MBSMemoryService):
                    logger.info(f"Attempting to save session {effective_session_id} to LTM (MBS)...")
                    await memory_service.add_session_to_memory(session_for_ltm)
                    logger.info(f"Session {effective_session_id} processed for LTM (MBS).")
            else:
                logger.warning(f"Could not retrieve session {effective_session_id} after run for LTM save.")
        except Exception as e_ltm_save:
            logger.error(f"Error saving session {effective_session_id} to LTM: {e_ltm_save}", exc_info=True)

        # Fallback for empty response if tools were called
        if not final_response_text.strip() and has_tool_responses:
            final_response_text = "I've processed your request using available tools. Is there anything else?"
            logger.warning(f"Session {effective_session_id}: Agent used tools but no final text. Using fallback.")
        elif not final_response_text.strip():
            final_response_text = "I am unable to provide a response at this time."
            logger.error(
                f"Session {effective_session_id}: Agent provided no text response and no tool activity detected.")

        return InteractionResponse(
            session_id=effective_session_id,
            agent_response=final_response_text,
            tool_calls=tool_calls_summary if tool_calls_summary else None
        )
    except Exception as e:
        logger.error(f"Error during agent interaction for session '{client_provided_session_id}': {e}", exc_info=True)
        # Ensure a session_id is returned even in case of early error
        error_session_id = client_provided_session_id or f"error-session-{uuid.uuid4()}"
        return InteractionResponse(
            session_id=error_session_id,
            agent_response="An unexpected error occurred. Please try again later.",
            error=str(e)
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True,
                log_config=None)  # Disable uvicorn's default log config if using basicConfig