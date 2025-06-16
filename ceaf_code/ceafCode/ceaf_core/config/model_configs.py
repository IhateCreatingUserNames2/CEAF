# Model Configurations
# ceaf_project/ceaf_core/config/model_configs.py

import os
import logging

logger = logging.getLogger(__name__)

# --- OpenRouter Configuration ---
# These are typically set in .env and picked up by LiteLLM automatically,
# but defining them here provides a central reference and defaults if not set.
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", DEFAULT_OPENROUTER_API_BASE)

# API Key is critical and should be in .env, not defaulted here.
# This is more for awareness in other parts of the code.
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"


# --- Default Model Names on OpenRouter for CEAF Agents ---
# These can be overridden by environment variables.

# ORA (Orchestrator/Responder Agent) - Needs to be capable and good at instruction following.
ORA_DEFAULT_MODEL_ENV_VAR = "ORA_DEFAULT_MODEL"
ORA_FALLBACK_MODEL = "openrouter/openai/gpt-4.1" # Good balance

# NCIM (Narrative Coherence & Identity Module) - Needs good text understanding and nuance.
NCIM_DEFAULT_MODEL_ENV_VAR = "NCIM_DEFAULT_MODEL"
NCIM_FALLBACK_MODEL = "openrouter/openai/gpt-4.1" # Faster, good for focused analysis

# VRE (Virtue & Reasoning Engine) - Needs strong reasoning, potentially logical capabilities.
VRE_DEFAULT_MODEL_ENV_VAR = "VRE_DEFAULT_MODEL"
VRE_FALLBACK_MODEL = "openrouter/openai/gpt-4.1" # Sonnet for more complex reasoning

# MCL (Metacognitive Control Loop) - Needs analytical skills.
MCL_DEFAULT_MODEL_ENV_VAR = "MCL_DEFAULT_MODEL"
MCL_FALLBACK_MODEL = "openrouter/openai/gpt-4.1" # Could also be Sonnet if tasks are complex

# A general CEAF default if a specific agent doesn't have one defined
CEAF_GENERAL_DEFAULT_MODEL_ENV_VAR = "CEAF_DEFAULT_OPENROUTER_MODEL"
CEAF_GENERAL_FALLBACK_MODEL = "openrouter/openai/gpt-4.1" # A fast and capable general model


# --- Function to Get Model Name for an Agent ---
def get_agent_model_name(
    agent_type_env_var: str,
    fallback_model: str,
    general_default_env_var: str = CEAF_GENERAL_DEFAULT_MODEL_ENV_VAR,
    general_fallback_model: str = CEAF_GENERAL_FALLBACK_MODEL
) -> str:
    """
    Resolves the model name for an agent based on environment variables and fallbacks.
    Priority:
    1. Specific agent environment variable (e.g., ORA_DEFAULT_MODEL)
    2. Specific agent fallback model (e.g., ORA_FALLBACK_MODEL)
    3. General CEAF environment variable (e.g., CEAF_DEFAULT_OPENROUTER_MODEL) - though specific fallback is usually better.
    4. General CEAF fallback model (e.g., CEAF_GENERAL_FALLBACK_MODEL)
    """
    model_name = os.getenv(agent_type_env_var)
    if model_name:
        logger.info(f"Using model '{model_name}' from environment variable '{agent_type_env_var}'.")
        return model_name

    # If specific agent env var is not set, use its defined fallback.
    # The fallback_model parameter already serves this purpose.
    # No need to check general_default_env_var before specific fallback.
    logger.info(f"Environment variable '{agent_type_env_var}' not set. Using fallback model '{fallback_model}'.")
    return fallback_model
