# core_logic/adk_setup.py
from google.adk.sessions import InMemorySessionService # Start with in-memory, can upgrade later
from google.adk.artifacts import InMemoryArtifactService
# from google.adk.memory import InMemoryMemoryService # ADK's default, MemoryBlossom will be custom
from google.adk.runners import Runner

# CEAF specific MemoryBlossom instance will be created and passed if needed
# For now, ADK runner doesn't directly take MemoryBlossom.
# MemoryBlossom will be accessed via context/tools or directly by modules.

def create_session_service():
    return InMemorySessionService()

def create_artifact_service():
    return InMemoryArtifactService()

def create_runner(agent_instance, app_name, session_service, artifact_service):
    return Runner(
        agent=agent_instance,
        app_name=app_name,
        session_service=session_service,
        artifact_service=artifact_service
        # memory_service could be integrated if MemoryBlossom aligns with BaseMemoryService
    )

# Constants
APP_NAME = "CEAF_App"
USER_ID_DEFAULT = "user_ceaf"
SESSION_ID_DEFAULT = "session_ceaf_main"