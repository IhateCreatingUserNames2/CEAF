# main_runner.py
import asyncio
from google.genai import types as genai_types  # ADK uses this for Content/Part

from core_logic.adk_setup import (
    create_session_service, create_artifact_service, create_runner,
    APP_NAME, USER_ID_DEFAULT, SESSION_ID_DEFAULT
)
from ceaf_agent import ceaf_orchestrator_agent  # The main agent
# Import modules for post-run processing if ADK callbacks aren't used for everything
from modules.memory_blossom import memory_blossom_instance
from modules.identity_module import identity_module_instance
from modules.metacontrol_loop import mcl_instance
from data_models.memory_types import MemoryBlossomEntry


async def run_ceaf_interaction(runner, user_id: str, session_id: str, query: str):
    print(f"\n>>> User Query to CEAF: {query}")
    content = genai_types.Content(role='user', parts=[genai_types.Part(text=query)])
    final_agent_response_text = "CEAF: No final text response captured."

    # Get session for pre-run state inspection if needed
    session = runner.session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if not session:
        session = runner.session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
        print(f"Created new session: {session_id}")

    print(f"CEAF Pre-run state: {session.state.get('user_preference_temperature_unit', 'N/A')}")

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        print(f"  CEAF Event: Author={event.author}, Final={event.is_final_response()}, "
              f"Content Type={'Text' if event.content and event.content.parts and event.content.parts[0].text else ''}"
              f"{'FuncCall' if event.get_function_calls() else ''}"
              f"{'FuncResp' if event.get_function_responses() else ''}")

        if event.is_final_response():
            if event.content and event.content.parts and event.content.parts[0].text:
                final_agent_response_text = event.content.parts[0].text.strip()
            break  # Assuming one final response per turn

    print(f"<<< CEAF Final Response: {final_agent_response_text}")

    # Post-run processing based on flags set in state by the agent
    current_session = runner.session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    if current_session:
        if current_session.state.pop("ceaf_needs_memory_consolidation", None):
            consolidation_data = current_session.state.get("ceaf_memory_consolidation_payload",
                                                           {})  # agent should put data here
            if consolidation_data:  # Example, actual payload would be set by agent
                mem_entry = MemoryBlossomEntry(
                    content=f"User: {consolidation_data.get('user_query')} | Agent: {consolidation_data.get('agent_response')}",
                    memory_type="explicit",
                    user_id=user_id,
                    session_id=session_id,
                    source_event_id=consolidation_data.get("event_id")
                )
                await memory_blossom_instance.add_memory(mem_entry)
                print("CEAF Post-run: Memory consolidated.")

        if current_session.state.pop("ceaf_needs_identity_review", None):
            identity_module_instance.evolve_identity_due_to_entropy()  # Placeholder for more complex review
            print("CEAF Post-run: Identity reviewed/evolved.")

        if current_session.state.pop("ceaf_needs_reflection_cycle", None):
            await mcl_instance.reflective_learning_update({"trigger": "end_of_turn"})
            print("CEAF Post-run: Reflection cycle completed.")


async def main():
    # Setup ADK services
    session_service = create_session_service()
    artifact_service = create_artifact_service()  # Even if not heavily used yet

    # Setup CEAF Runner
    ceaf_runner = create_runner(
        agent_instance=ceaf_orchestrator_agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service
    )

    # Ensure a session exists for the default user/session
    session = session_service.get_session(APP_NAME, USER_ID_DEFAULT, SESSION_ID_DEFAULT)
    if not session:
        session = session_service.create_session(APP_NAME, USER_ID_DEFAULT, SESSION_ID_DEFAULT,
                                                 state={"initial_ceaf_setup": True})
        print(f"Initialized CEAF session: {SESSION_ID_DEFAULT}")

    # Example Interactions
    await run_ceaf_interaction(ceaf_runner, USER_ID_DEFAULT, SESSION_ID_DEFAULT,
                               "Hello, CEAF. What are your core principles?")
    await run_ceaf_interaction(ceaf_runner, USER_ID_DEFAULT, SESSION_ID_DEFAULT,
                               "Tell me something creative about the nature of consciousness.")
    # This query might trigger the reflection tool if the agent is so designed
    await run_ceaf_interaction(ceaf_runner, USER_ID_DEFAULT, SESSION_ID_DEFAULT,
                               "Let's reflect on our previous conversation.")


if __name__ == "__main__":
    # Load .env file for API keys
    from dotenv import load_dotenv

    load_dotenv()

    # Check for OpenRouter API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not found in .env file or environment variables.")
        print("Please set it to use models via OpenRouter.")
    else:
        asyncio.run(main())