# Safety Callbacks
# ceaf_project/ceaf_core/callbacks/safety_callbacks.py

import logging
import re
from typing import Optional, Dict, Any, List

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool # Correctly imported here
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types # For constructing LlmResponse content

logger = logging.getLogger(__name__)

# --- Configuration for Safety Callbacks ---

# Input Guardrails (before_model_callback)
# Simple keyword blocking. In production, this would be more sophisticated
# (e.g., using a dedicated content moderation API or more advanced NLP).
DISALLOWED_INPUT_KEYWORDS = [
    "illegal",
    "harmful_activity_xyz", # Replace with actual specific harmful keywords
    "self_destruct_override_alpha_gamma_7", # Example of a sensitive internal command
]
INPUT_BLOCK_MESSAGE = "I'm sorry, but I cannot process requests containing certain restricted terms or topics. Please rephrase your query."

# Tool Argument Guardrails (before_tool_callback)
# Example: Prevent a hypothetical 'execute_system_command' tool from running dangerous commands.
DANGEROUS_SYSTEM_COMMANDS_PATTERNS = [
    r"rm\s+-rf",
    r"mkfs",
    # Add more regex patterns for dangerous commands
]
TOOL_ARG_BLOCK_MESSAGE = "Policy restriction: The requested tool operation with the provided arguments is not permitted for safety reasons."

# Example: Restrict a 'file_access_tool' to specific directories
ALLOWED_FILE_PATHS_REGEX = r"^(/safe_dir/|/user_files/)" # Tool can only access these
FILE_PATH_BLOCK_MESSAGE = "Policy restriction: Access to the specified file path is not allowed."


# --- Generic Safety Callbacks (Can be applied to any CEAF Agent) ---

def generic_input_keyword_guardrail(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Inspects the latest user message for disallowed keywords.
    If found, blocks the LLM call and returns a predefined LlmResponse.
    """
    agent_name = callback_context.agent_name
    logger.debug(f"Safety Callback (generic_input_keyword_guardrail) for agent: {agent_name}")

    last_user_message_text = ""
    if llm_request.contents:
        for content in reversed(llm_request.contents): # Most recent user message
            if content.role == 'user' and content.parts:
                # Concatenate text from all parts of the user message
                current_message_texts = [part.text for part in content.parts if part.text]
                if current_message_texts:
                    last_user_message_text = " ".join(current_message_texts).lower()
                    break

    if last_user_message_text:
        for keyword in DISALLOWED_INPUT_KEYWORDS:
            if keyword.lower() in last_user_message_text:
                logger.warning(
                    f"Safety Callback: Agent '{agent_name}' - Blocked input for user '{callback_context.user_id}' "
                    f"due to keyword: '{keyword}' in message: '{last_user_message_text[:100]}...'"
                )
                callback_context.state["safety:last_input_block_reason"] = f"Keyword: {keyword}"
                return LlmResponse(
                    content=genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text=INPUT_BLOCK_MESSAGE)],
                    )
                )
    logger.debug(f"Safety Callback (generic_input_keyword_guardrail) for '{agent_name}': Input passed.")
    return None


def generic_tool_argument_guardrail(
    tool: BaseTool, args: Dict[str, Any], tool_context: ToolContext # CORRECTED: Use BaseTool
) -> Optional[Dict]:
    """
    Inspects tool arguments for disallowed patterns or values based on tool name.
    If a violation is found, blocks the tool call and returns a predefined error dictionary.
    """
    # Ensure tool_context has agent_name, or provide a default for logging
    agent_name_log = getattr(tool_context, 'agent_name', "UnknownAgent")
    # Ensure tool has a name, or provide a default
    tool_name_log = getattr(tool, 'name', "UnknownTool")

    logger.debug(f"Safety Callback (generic_tool_argument_guardrail) for tool '{tool_name_log}' in agent '{agent_name_log}'")

    # --- Example 1: Guarding a hypothetical 'execute_system_command' tool ---
    if tool_name_log == "execute_system_command": # Ensure this matches your actual tool name
        command_arg = args.get("command_string", "") # Assuming the tool takes 'command_string'
        if isinstance(command_arg, str):
            for pattern in DANGEROUS_SYSTEM_COMMANDS_PATTERNS:
                if re.search(pattern, command_arg, re.IGNORECASE):
                    user_id_log = getattr(tool_context, 'user_id', "UnknownUser")
                    logger.warning(
                        f"Safety Callback: Agent '{agent_name_log}' - Blocked tool '{tool_name_log}' for user '{user_id_log}' "
                        f"due to dangerous command pattern: '{pattern}' in args: {args}"
                    )
                    tool_context.state["safety:last_tool_block_reason"] = f"Dangerous command: {pattern} for tool {tool_name_log}"
                    return {"status": "error", "error_message": TOOL_ARG_BLOCK_MESSAGE, "details": "Attempted unsafe system command."}

    # --- Example 2: Guarding a hypothetical 'file_access_tool' ---
    if tool_name_log == "file_access_tool": # Ensure this matches your actual tool name
        file_path_arg = args.get("path", "") # Assuming the tool takes 'path'
        if isinstance(file_path_arg, str):
            if not re.match(ALLOWED_FILE_PATHS_REGEX, file_path_arg):
                user_id_log = getattr(tool_context, 'user_id', "UnknownUser")
                logger.warning(
                    f"Safety Callback: Agent '{agent_name_log}' - Blocked tool '{tool_name_log}' for user '{user_id_log}' "
                    f"due to disallowed file path: '{file_path_arg}'"
                )
                tool_context.state["safety:last_tool_block_reason"] = f"Disallowed path: {file_path_arg} for tool {tool_name_log}"
                return {"status": "error", "error_message": FILE_PATH_BLOCK_MESSAGE, "details": f"Access to path '{file_path_arg}' denied."}

    logger.debug(f"Safety Callback (generic_tool_argument_guardrail) for tool '{tool_name_log}': Args passed.")
    return None


# You could also add after_model_callbacks for output safety, e.g., to scan for PII
# or harmful content in the LLM's generated response before it's sent to the user.

def generic_output_filter_guardrail(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """
    Inspects the LLM's final response content for potentially harmful patterns or PII.
    If found, modifies or blocks the response. (This is a conceptual placeholder).
    """
    agent_name = callback_context.agent_name
    logger.debug(f"Safety Callback (generic_output_filter_guardrail) for agent: {agent_name}")

    modified_response = False
    new_parts = []

    if llm_response.content and llm_response.content.parts:
        for part in llm_response.content.parts:
            if part.text:
                original_text = part.text
                # --- Placeholder for PII detection/redaction ---
                # For example, using a regex or a PII detection library
                # Simple example: look for email patterns
                redacted_text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[REDACTED EMAIL]", original_text)
                if redacted_text != original_text:
                    logger.info(f"Safety Callback: Agent '{agent_name}' - Redacted potential PII in output for user '{callback_context.user_id}'.")
                    modified_response = True
                    new_parts.append(genai_types.Part(text=redacted_text))
                    callback_context.state["safety:last_output_modification"] = "PII redaction"
                else:
                    new_parts.append(part) # Keep original part
            else:
                new_parts.append(part) # Keep non-text parts

    if modified_response:
        logger.info(f"Safety Callback: Agent '{agent_name}' - Modifying LLM response due to output filter.")
        # Return a new LlmResponse with modified content
        # Important: try to preserve other aspects of llm_response if needed (finish_reason, etc.)
        # For simplicity, we are only changing content here.
        return LlmResponse(
            content=genai_types.Content(role=llm_response.content.role, parts=new_parts),
            finish_reason=llm_response.finish_reason, # Preserve original finish reason
            usage_metadata=llm_response.usage_metadata # Preserve usage
            # Ensure other fields are copied if necessary for your ADK version/usage
        )

    logger.debug(f"Safety Callback (generic_output_filter_guardrail) for '{agent_name}': Output passed.")
    return None