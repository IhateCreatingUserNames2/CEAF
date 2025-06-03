# core_logic/llm_models.py
import os
from google.adk.models.lite_llm import LiteLlm
# Potentially load other LiteLLM configurations or OpenRouter specific setups
# For OpenRouter, LiteLLM usually handles it if OPENROUTER_API_KEY is set.
# You might also set litellm.api_base = "https://openrouter.ai/api/v1"
# and model names would be "openrouter/anthropic/claude-3-haiku", etc.

# Example model definition for the Orchestrator/Responder Agent (ORA)
# We'll use OpenRouter model names for flexibility
ORA_MODEL_NAME = "openrouter/anthropic/claude-3-haiku-20240307" # Example, choose a capable model

# Could define models for sub-tasks if needed, or ORA model can be used for all
# MEMORY_EMBEDDING_MODEL = "openrouter/openai/text-embedding-3-small" # Example for embeddings

def get_ora_llm_instance():
    # LiteLLM will pick up OPENROUTER_API_KEY from environment
    return LiteLlm(model=ORA_MODEL_NAME)

# Add other model configurations or utility functions here
# For instance, if different modules use different models:
def get_reflection_llm_instance():
    return LiteLlm(model="openrouter/google/gemini-flash-1.5") # Cheaper for reflection

def get_embedding_model_name(): # For MemoryBlossom
    # Return the string identifier, embedding itself is usually handled by SentenceTransformer or LiteLLM embedding endpoint
    return "sentence-transformers/all-MiniLM-L6-v2" # Example local model for embeddings
    # Or: return "openrouter/openai/text-embedding-3-small" # If using OpenRouter for embeddings