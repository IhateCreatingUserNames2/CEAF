# Common Utilities
# ceaf_project/ceaf_core/tools/common_utils.py

import json
import logging
import re
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, ValidationError, Field

logger = logging.getLogger(__name__)


# --- Text Processing Utilities ---

def sanitize_text_for_logging(text: Optional[str], max_length: int = 100) -> str:
    """Sanitizes text for logging, truncating and escaping newlines."""
    if not text:
        return "<empty>"
    sanitized = text.replace("\n", "\\n").replace("\r", "\\r")
    if len(sanitized) > max_length:
        return sanitized[:max_length] + "..."
    return sanitized


def extract_json_from_text(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """
    Attempts to extract the first valid JSON object or array from a string.
    Handles cases where JSON might be embedded within other text or markdown code blocks.
    """
    if not text:
        return None

    # Try to find JSON within markdown code blocks first
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(
                f"JSON common_utils: Found JSON in markdown, but failed to parse: {e}. Content: {sanitize_text_for_logging(json_str)}")
            # Fall through to try parsing the whole text or other methods

    # If no markdown, or markdown parsing failed, try to find JSON embedded anywhere
    # This is a bit more heuristic. Looks for the start of a JSON object/array.
    json_starts = [i for i, char in enumerate(text) if char in ('{', '[')]
    for start_index in json_starts:
        # Try to parse from this start point
        balance = 0
        if text[start_index] == '{':
            open_char, close_char = '{', '}'
        else:
            open_char, close_char = '[', ']'

        for end_index in range(start_index, len(text)):
            if text[end_index] == open_char:
                balance += 1
            elif text[end_index] == close_char:
                balance -= 1

            if balance == 0:  # Found a potentially complete JSON structure
                potential_json_str = text[start_index: end_index + 1]
                try:
                    # Validate if it's actually parseable JSON
                    parsed_json = json.loads(potential_json_str)
                    logger.debug(
                        f"JSON common_utils: Successfully extracted JSON: {sanitize_text_for_logging(potential_json_str)}")
                    return parsed_json
                except json.JSONDecodeError:
                    # Not a valid JSON structure, continue searching
                    continue
                    # If balance never reaches 0 after a start, it's likely not well-formed from that start

    logger.warning(f"JSON common_utils: Could not extract valid JSON from text: {sanitize_text_for_logging(text)}")
    return None


# --- Pydantic Model Utilities ---

def pydantic_to_dict_str(model_instance: Optional[BaseModel], indent: Optional[int] = 2,
                         exclude_none: bool = True) -> str:
    """Converts a Pydantic model instance to a JSON string, handling None."""
    if model_instance is None:
        return "None"
    try:
        return model_instance.model_dump_json(indent=indent, exclude_none=exclude_none)
    except Exception as e:
        logger.error(f"Pydantic common_utils: Error serializing Pydantic model {type(model_instance)}: {e}")
        return f"<Error serializing model: {type(model_instance).__name__}>"


def parse_llm_json_output(
        json_str: Optional[str],
        pydantic_model: type[BaseModel],
        strict: bool = False  # If true, requires perfect match. If false, tries to extract first.
) -> Optional[BaseModel]:

    if not json_str:
        logger.warning(f"Pydantic common_utils: Received empty JSON string for model {pydantic_model.__name__}")
        return None

    parsed_dict = None
    try:
        # First, try direct parsing
        parsed_dict = json.loads(json_str)
    except json.JSONDecodeError as e1:
        if strict:
            logger.error(
                f"Pydantic common_utils (strict): Failed to parse JSON string for model {pydantic_model.__name__}: {e1}. Raw: {sanitize_text_for_logging(json_str)}")
            return None

        logger.warning(
            f"Pydantic common_utils: Direct JSON parsing failed for model {pydantic_model.__name__}, attempting extraction. Error: {e1}")
        extracted_content = extract_json_from_text(json_str)
        if isinstance(extracted_content, dict):
            parsed_dict = extracted_content
        else:
            logger.error(
                f"Pydantic common_utils: Could not extract valid JSON dict for model {pydantic_model.__name__} from text. Extracted: {type(extracted_content)}")
            return None

    if parsed_dict is None:  # Should be caught by above, but as a safeguard
        logger.error(
            f"Pydantic common_utils: No valid JSON dictionary found or extracted for model {pydantic_model.__name__}. Original: {sanitize_text_for_logging(json_str)}")
        return None

    try:
        model_instance = pydantic_model(**parsed_dict)
        return model_instance
    except ValidationError as e2:
        logger.error(
            f"Pydantic common_utils: Validation error for model {pydantic_model.__name__}: {e2}. Parsed Dict: {parsed_dict}")
        return None
    except Exception as e3:
        logger.error(
            f"Pydantic common_utils: Unexpected error instantiating model {pydantic_model.__name__}: {e3}. Parsed Dict: {parsed_dict}")
        return None


# --- Tool Output Formatting ---

def create_successful_tool_response(data: Optional[Dict[str, Any]] = None, message: Optional[str] = None) -> Dict[
    str, Any]:
    """Creates a standard success response dictionary for ADK tools."""
    response = {"status": "success"}
    if message:
        response["message"] = message
    if data:
        response.update(data)  # Merge data dict into response
    return response


def create_error_tool_response(error_message: str, details: Optional[Any] = None, error_code: Optional[str] = None) -> \
Dict[str, Any]:
    """Creates a standard error response dictionary for ADK tools."""
    response = {"status": "error", "error_message": error_message}
    if details:
        response["details"] = details
    if error_code:
        response["error_code"] = error_code
    return response
