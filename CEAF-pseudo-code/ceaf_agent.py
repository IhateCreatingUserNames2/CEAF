# ceaf_agent.py
from google.adk.agents import LlmAgent, InvocationContext, BaseAgent, ToolContext
from google.adk.events import Event
from typing import AsyncGenerator, Dict, Any

# Import modules (adjust paths if needed)
from core_logic.llm_models import get_ora_llm_instance
from modules.ncf_engine import ncf_engine_instance
from modules.identity_module import identity_module_instance
from modules.memory_blossom import memory_blossom_instance  # For post-processing
from modules.metacontrol_loop import mcl_instance
from modules.reasoning_module import vre_instance


# from data_models.memory_types import MemoryBlossomEntry # For post-processing

# For ADK, context objects are passed automatically to callbacks and tools.
# For direct module usage within the agent's run, we'll manage it.

# --- Define Tools that CEAF Agent might use directly (could be minimal) ---
# Most complex logic is handled by modules feeding into NCF.
# Example: a tool to explicitly trigger reflection or identity update.

def trigger_reflection_tool(tool_context: ToolContext) -> Dict[str, Any]:
    """Triggers a reflective learning cycle and identity evolution check."""
    print("ORA Tool: Triggering reflection.")
    # This would ideally call mcl_instance.reflective_learning_update()
    # and identity_module_instance.evolve_identity_due_to_entropy()
    # Since tool_context doesn't easily pass module instances, this tool might just set a flag in state
    # and the main loop or a callback would pick it up.
    # For simplicity, let's assume it can call them (this needs careful design in ADK).
    # asyncio.create_task(mcl_instance.reflective_learning_update({"trigger": "manual_tool"}))
    # identity_module_instance.evolve_identity_due_to_entropy()
    tool_context.state["ceaf_needs_reflection_cycle"] = True
    return {"status": "Reflection cycle initiated."}


class CEAFOrchestratorAgent(LlmAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Modules are typically instantiated globally or passed in for testability
        self.ncf_engine = ncf_engine_instance
        self.identity_module = identity_module_instance
        self.memory_blossom = memory_blossom_instance
        self.mcl = mcl_instance
        self.vre = vre_instance

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        print(f"\nCEAF ORA ({self.name}): Invocation {ctx.invocation_id} started.")
        user_query = ""
        if ctx.user_content and ctx.user_content.parts:
            user_query = ctx.user_content.parts[0].text or ""

        # 1. Get current identity
        current_identity = self.identity_module.get_current_identity()

        # 2. Build NCF Prompt using the NCF Engine
        # The NCF engine itself might use memory_blossom to fetch relevant memories
        ncf_full_prompt_str = await self.ncf_engine.build_ncf_prompt(
            user_query=user_query,
            user_id=ctx.session.user_id,
            session_id=ctx.session.id,
            current_identity=current_identity
        )
        print(f"CEAF ORA: NCF Prompt constructed (first 100 chars): {ncf_full_prompt_str[:100]}...")

        # 3. ORA (LlmAgent) processes the NCF Prompt
        # The LlmAgent's core logic will handle this. We inject the NCF as the primary "user message"
        # or by overriding the instruction provider if ADK allows that level of dynamic instruction.
        # For now, let's assume we modify the incoming context for the LlmAgent's default processing.

        # This is tricky in ADK's LlmAgent flow. _run_async_impl is usually for complex orchestration,
        # not direct LLM calls. The LlmAgent class itself makes the LLM call based on its
        # instruction, tools, and the history (including user_content).
        # We need a way for THIS agent's instruction to effectively BE the NCF prompt, or for
        # the NCF prompt to be the primary content it reasons over.

        # Option A: Yield a synthetic event that becomes the new "user input" for the LlmAgent's machinery
        # This feels a bit like a hack.
        # ncf_event_content = types.Content(role="user", parts=[types.Part(text=ncf_full_prompt_str)])
        # yield Event(author="user_internal_ncf", content=ncf_event_content, invocation_id=ctx.invocation_id)
        # The LlmAgent machinery would then pick this up.

        # Option B: A more direct way if LlmAgent is flexible enough (might need custom LlmAgent subclass)
        # For this prototype, we'll rely on the LlmAgent's default behavior, assuming its
        # 'instruction' parameter can be dynamically formed or its main reasoning will
        # use the 'user_content' which IS our NCF prompt if structured correctly.

        # Let's assume the NCF prompt effectively IS the content this LlmAgent works on.
        # We pass it to the superclass's logic by ensuring it's in `ctx.user_content`
        # or by designing the LlmAgent to use a dynamic instruction provider.

        # For now, we'll let the LlmAgent's default behavior handle the LLM call
        # after we've set up the context. The NCF would ideally be passed as instruction.
        # A custom instruction_provider for LlmAgent could achieve this:
        # def ncf_instruction_provider(readonly_context: ReadonlyContext) -> str:
        #    # ... logic to build NCF ... (would need async here, or careful sync design)
        #    return ncf_full_prompt_str
        # self.instruction = ncf_instruction_provider # If this were possible during __init__ or dynamically

        # Fallback: The NCF prompt becomes the system instruction if possible,
        # or the user_query is wrapped by NCF elements in the LlmAgent's prompt assembly.
        # The LlmAgent will use its configured self.model (which is LiteLlm)

        print(f"CEAF ORA: Handing NCF prompt to LlmAgent machinery...")
        # The LlmAgent's internal _run_llm_flow will be invoked.
        # We need to ensure it uses `ncf_full_prompt_str`.
        # This might involve setting `ctx.session.state['temp:current_ncf_prompt'] = ncf_full_prompt_str`
        # and having a `before_model_callback` that injects this into the `llm_request`.

        # Simulate yielding the NCF as the thing to process:
        # We need to modify how the LlmAgent gets its main prompt.
        # This is a simplification:
        if ctx.user_content and ctx.user_content.parts:
            ctx.user_content.parts[0].text = ncf_full_prompt_str  # Replace original query with NCF
        else:
            from google.genai import types as genai_types  # Ensure import
            ctx.user_content = genai_types.Content(role="user", parts=[genai_types.Part(text=ncf_full_prompt_str)])

        async for event in super()._run_async_impl(ctx):
            # 4. Post-Response Processing (within the event loop of the LlmAgent)
            if event.is_final_response() and event.content and event.content.parts:
                final_text_response = event.content.parts[0].text
                print(f"CEAF ORA: Final text response received: '{final_text_response[:100]}...'")

                # 4a. Metacognitive Control Loop - Evaluate output
                # This needs access to logprobs, which ADK's Event doesn't directly expose.
                # We'd need a custom LiteLlm model wrapper or after_model_callback to capture them.
                # For now, conceptual call:
                await self.mcl.evaluate_ora_output(final_text_response, logprobs=None,
                                                   output_entropy=None)  # Placeholder for logprobs

                # 4b. Virtue Reasoning Engine - Check humility (example)
                # is_humble, vre_feedback = await self.vre.check_epistemic_humility(final_text_response, ncf_full_prompt_str)
                # if not is_humble:
                #     print(f"CEAF ORA (VRE): Response lacked humility: {vre_feedback}")
                # Potentially trigger re-generation or add a disclaimer based on VRE feedback.

            # 4c. Memory Blossom - Add interaction to memory (after final response or key events)
            # This typically happens in an after_agent_callback or outside the run loop
            # For prototype, can simulate here:
            if event.is_final_response():  # Or other conditions
                # from data_models.memory_types import MemoryBlossomEntry # Ensure import
                # mem_entry = MemoryBlossomEntry(
                #     content=f"User: {user_query} | Agent: {final_text_response}",
                #     memory_type="explicit", # Or classify based on content/emotion
                #     user_id=ctx.session.user_id,
                #     session_id=ctx.session.id,
                #     source_event_id=event.id,
                #     metadata={"agent_name": self.name}
                # )
                # await self.memory_blossom.add_memory(mem_entry)
                ctx.session.state["ceaf_needs_memory_consolidation"] = {
                    "user_query": user_query, "agent_response": final_text_response, "event_id": event.id
                }

            # 4d. Identity Module - Potential update based on interaction
            # This usually happens less frequently, based on significant events or reflection.
            # self.identity_module.update_identity_element("last_interaction_summary", final_text_response[:50], 0.9)
            if event.is_final_response():
                ctx.session.state["ceaf_needs_identity_review"] = True

            # Yield the event from LlmAgent
            yield event

        print(f"CEAF ORA ({self.name}): Invocation {ctx.invocation_id} finished.")


# Instantiate the CEAF ORA
ceaf_orchestrator_agent = CEAFOrchestratorAgent(
    name="CEAF_Orchestrator_v1",
    model=get_ora_llm_instance(),  # Uses LiteLlm
    description="A Coherent Emergence Agent Framework orchestrator.",
    instruction="Your primary instruction will be dynamically constructed via NCF. Process it thoughtfully.",
    # Generic base
    tools=[trigger_reflection_tool],  # Example tool
    # Callbacks will be important for fine-grained control and module interaction
    # after_model_callback=capture_logprobs_and_trigger_mcl,
    # after_agent_callback=final_post_processing_hook,
)