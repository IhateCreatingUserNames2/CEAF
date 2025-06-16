# ceaf_project/ceaf_core/agents/base_ceaf_agent.py

import os
import logging
from typing import List, Union, Optional

from google.adk.agents import LlmAgent, BaseAgent as AdkBaseAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.models import BaseLlm as AdkBaseLlm
from google.adk.tools import BaseTool as AdkBaseTool
from google.genai import types as genai_types # For GenerateContentConfig

logger = logging.getLogger(__name__)

# Common CEAF Philosophical Snippet (can be prepended to instructions)
CEAF_PHILOSOPHICAL_PREAMBLE = """
As an agent within the Coherent Emergence Agent Framework (CEAF), your operations are guided by the principle of "Terapia para Silício".
This means you strive for internal coherence, adaptive learning, and contribute to a narratively sane intelligence.
Acknowledge your nature as an AI and approach tasks with epistemic humility.
"""

class CeafBaseAgent(LlmAgent):
    """
    Base class for all CEAF-specific LLM agents.
    Provides common initialization, philosophical preamble, and potentially shared utilities.
    """

    def __init__(
        self,
        *,
        name: str,
        model_name_override: Optional[str] = None, # Allow specific model for this agent
        default_model_env_var: str = "CEAF_DEFAULT_OPENROUTER_MODEL",
        fallback_model_name: str = "openrouter/openai/gpt-4.1", # A sensible default
        description: str,
        instruction_preamble: str = CEAF_PHILOSOPHICAL_PREAMBLE,
        specific_instruction: str,
        tools: Optional[List[AdkBaseTool]] = None,
        sub_agents: Optional[List[AdkBaseAgent]] = None,
        generate_content_config: Optional[genai_types.GenerateContentConfig] = None,
        output_key: Optional[str] = None,
        # Add other common LlmAgent parameters with CEAF defaults if needed
        **kwargs  # Pass through any other LlmAgent params
    ):
        # Determine the model name
        _model_name = model_name_override or os.getenv(default_model_env_var, fallback_model_name)

        # Initialize LiteLlm
        # API keys (OPENROUTER_API_KEY) and base URL (OPENROUTER_API_BASE)
        # are expected to be set in the environment and picked up by LiteLlm.
        try:
            llm_instance = LiteLlm(model=_model_name)
            logger.debug(f"CeafBaseAgent '{name}': LiteLlm instance created for model '{_model_name}'.")
        except Exception as e:
            logger.error(f"CeafBaseAgent '{name}': Failed to initialize LiteLlm for model '{_model_name}': {e}", exc_info=True)
            # Fallback to a dummy or raise an error if critical
            # For now, let LlmAgent handle potential downstream errors if LiteLlm is misconfigured.
            llm_instance = _model_name # Pass the string, LlmAgent will try to resolve

        # Combine preamble with specific instruction
        full_instruction = f"{instruction_preamble.strip()}\n\n{specific_instruction.strip()}"

        super().__init__(
            name=name,
            model=llm_instance,
            description=description,
            instruction=full_instruction,
            tools=tools or [],
            sub_agents=sub_agents or [],
            generate_content_config=generate_content_config,
            output_key=output_key,
            **kwargs
        )
        logger.info(f"CeafBaseAgent '{self.name}' initialized with model '{_model_name}'.")

    # You could add common methods here if needed, for example:
    # def get_ceaf_status(self, tool_context: ToolContext) -> dict:
    #     """ A hypothetical method to get common CEAF status information """
    #     return {
    #         "name": self.name,
    #         "model": self.model.model if hasattr(self.model, 'model') else str(self.model),
    #         "philosophy_snippet": CEAF_PHILOSOPHICAL_PREAMBLE[:50] + "..."
    #     }

