# ceaf_core/tools/observability_tools.py

import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from google.adk.tools import FunctionTool, ToolContext

from .common_utils import create_successful_tool_response, \
    create_error_tool_response  # Removed sanitize_text_for_logging as it's not used here
from ..services.persistent_log_service import PersistentLogService  # ADDED IMPORT

# Attempt to import the Pydantic model
try:
    from ..modules.mcl_engine.finetuning_datatypes import FinetuningDataPoint

    FINETUNING_DATATYPE_LOADED = True
except ImportError:
    FINETUNING_DATATYPE_LOADED = False


    class FinetuningDataPoint:  # Dummy placeholder
        def __init__(self, **kwargs): self.data = kwargs  # type: ignore

        def model_dump_json(self, **kwargs): return json.dumps(self.data)  # type: ignore


    logging.warning("ObservabilityTools: FinetuningDataPoint model not found. Logging will use raw dicts.")

logger = logging.getLogger("ObservabilityTools")

# --- Configuration for Finetuning Data Logger ---
FINETUNING_LOG_ENABLED_ENV_VAR = "CEAF_FINETUNING_LOG_ENABLED"
DEFAULT_FINETUNING_LOG_FILE_PATH = "./data/ceaf_finetuning_log.jsonl"
FINETUNING_LOG_FILE = Path(os.getenv("CEAF_FINETUNING_LOG_FILE", DEFAULT_FINETUNING_LOG_FILE_PATH))

# --- Initialize PersistentLogService (similar to mcl_callbacks) ---
try:
    persistent_log_service = PersistentLogService()
    PLS_FOR_FINETUNING_AVAILABLE = True
except Exception as e_pls_obs:
    logger.error(
        f"ObservabilityTools: Failed to initialize PersistentLogService: {e_pls_obs}. Finetuning data might only go to JSONL if PLS fails.",
        exc_info=True)
    persistent_log_service = None  # type: ignore
    PLS_FOR_FINETUNING_AVAILABLE = False


def log_finetuning_data_point(
        tool_context: ToolContext,
        ora_initial_draft_response: str,  # MOVED: Mandatory, non-default
        ora_refined_response: str,  # MOVED: Mandatory, non-default
        # Optional arguments now follow mandatory ones
        user_query: Optional[str] = None,
        ncf_context_summary: Optional[str] = None,
        vre_critique_json: Optional[str] = None,
        vre_overall_alignment: Optional[str] = None,
        vre_recommendations_applied: Optional[List[str]] = None,
        mcl_eoc_assessment_at_turn_start: Optional[str] = None,
        active_ncf_parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Logs a structured data point for potential ORA fine-tuning.
    This data captures ORA's initial draft, VRE's critique, and ORA's refined response.
    Logging is controlled by the CEAF_FINETUNING_LOG_ENABLED environment variable.
    Attempts to log to PersistentLogService and then to a dedicated JSONL file.
    """
    is_enabled = os.getenv(FINETUNING_LOG_ENABLED_ENV_VAR, "false").lower() == "true"

    if not is_enabled:
        return create_successful_tool_response(
            message="Finetuning data logging is currently disabled by environment variable.")

    session_id = getattr(tool_context, 'session_id', None)
    turn_id = getattr(tool_context, 'invocation_id', None)
    source_agent = getattr(tool_context, 'agent_name', "ORA_SelfCorrection_Tool")  # Agent calling this tool

    # This dict will be the payload for PLS and basis for FinetuningDataPoint Pydantic model
    data_payload_dict = {
        "user_query": user_query,
        "ncf_context_summary": ncf_context_summary,
        "ora_initial_draft_response": ora_initial_draft_response,
        "vre_critique_json": vre_critique_json,
        "vre_overall_alignment": vre_overall_alignment,
        "vre_recommendations_applied": vre_recommendations_applied or [],
        "ora_refined_response": ora_refined_response,
        "mcl_eoc_assessment_at_turn_start": mcl_eoc_assessment_at_turn_start,
        "active_ncf_parameters": active_ncf_parameters or {},
        # 'tags' will be passed to PLS log_event separately.
        # For FinetuningDataPoint Pydantic model, it's part of the main dict.
    }

    pls_logged_successfully = False
    jsonl_logged_successfully = False
    pls_error_msg = None
    jsonl_error_msg = None

    # 1. Attempt to log via PersistentLogService
    if PLS_FOR_FINETUNING_AVAILABLE and persistent_log_service:
        try:
            # Add session_id and turn_id to the payload if they are not inherently part of it
            # for other event types, but for finetuning data, they are good context.
            # However, PLS log_event takes them as separate args, so data_payload_dict is fine as is.
            combined_tags_for_pls = ["finetuning_datapoint", "ora_self_correction"] + (tags or [])

            persistent_log_service.log_event(
                event_type="FINETUNING_DATA_POINT",  # Consistent event type for PLS
                data_payload=data_payload_dict,  # The core structured data
                source_agent=source_agent,
                session_id=session_id,
                turn_id=turn_id,
                tags=list(set(combined_tags_for_pls))
            )
            pls_logged_successfully = True
            logger.info(
                f"ObservabilityTools: Logged finetuning data point for turn '{turn_id}' to PersistentLogService.")
        except Exception as e_pls_log:
            pls_error_msg = str(e_pls_log)
            logger.error(
                f"ObservabilityTools: Error logging finetuning data point to PersistentLogService: {pls_error_msg}",
                exc_info=True)
    else:
        pls_error_msg = "PersistentLogService not available or not initialized."
        logger.warning(f"ObservabilityTools: Skipping PersistentLogService for finetuning data: {pls_error_msg}")

    # 2. Log to dedicated JSONL file (fallback or parallel logging)
    try:
        # Construct the full FinetuningDataPoint object for JSONL, including its own log_id and timestamp
        pydantic_input_dict = {
            "session_id": session_id,
            "turn_id": turn_id,
            **data_payload_dict,  # Spread the common payload
            "tags": tags or []
        }

        if FINETUNING_DATATYPE_LOADED:
            data_point_for_jsonl = FinetuningDataPoint(**pydantic_input_dict)  # type: ignore
            log_entry_str = data_point_for_jsonl.model_dump_json(exclude_none=True)
        else:
            # Fallback if Pydantic model not loaded (manual construction)
            data_point_for_jsonl_dict_fallback = {
                "log_id": f"ft_log_{time.time_ns()}",  # Generate ID if model doesn't
                "timestamp": time.time(),  # Generate timestamp if model doesn't
                **pydantic_input_dict  # Spread the rest
            }
            log_entry_str = json.dumps({k: v for k, v in data_point_for_jsonl_dict_fallback.items() if v is not None})

        FINETUNING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FINETUNING_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry_str + "\n")
        jsonl_logged_successfully = True
        logger.info(
            f"ObservabilityTools: Logged finetuning data point for turn '{turn_id}' to JSONL file: {FINETUNING_LOG_FILE.name}.")

    except Exception as e_jsonl:
        jsonl_error_msg = str(e_jsonl)
        logger.error(f"ObservabilityTools: Error logging finetuning data point to JSONL: {jsonl_error_msg}",
                     exc_info=True)

    # Determine overall success message
    if pls_logged_successfully and jsonl_logged_successfully:
        return create_successful_tool_response(
            message="Finetuning data point logged successfully to Persistent DB and JSONL file.")
    elif pls_logged_successfully:
        return create_successful_tool_response(
            message=f"Finetuning data point logged to Persistent DB, but JSONL logging failed: {jsonl_error_msg}")
    elif jsonl_logged_successfully:
        return create_successful_tool_response(
            message=f"Finetuning data point logged to JSONL file, but Persistent DB logging failed: {pls_error_msg}")
    else:
        return create_error_tool_response(
            error_message="Failed to log finetuning data to any store.",
            details={"turn_id": turn_id, "pls_error": pls_error_msg, "jsonl_error": jsonl_error_msg}
        )


def log_ora_self_correction_for_finetuning(
        tool_context: ToolContext,
        ora_initial_draft_response: str,
        ora_refined_response: str,
        user_query: Optional[str] = None,
        ncf_context_summary: Optional[str] = None,
        vre_critique_json: Optional[str] = None,
        vre_overall_alignment: Optional[str] = None,
        vre_recommendations_applied: Optional[List[str]] = None,
        mcl_eoc_assessment_at_turn_start: Optional[str] = None,
        active_ncf_parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
) -> Dict[str, Any]:

    # ... (implementation of the function remains the same) ...
    is_enabled = os.getenv(FINETUNING_LOG_ENABLED_ENV_VAR, "false").lower() == "true"

    if not is_enabled:
        return create_successful_tool_response(
            message="Finetuning data logging is currently disabled by environment variable.")

    session_id = getattr(tool_context, 'session_id', None)
    turn_id = getattr(tool_context, 'invocation_id', None)
    source_agent = getattr(tool_context, 'agent_name', "ORA_SelfCorrection_Tool")

    data_payload_dict = {
        "user_query": user_query,
        "ncf_context_summary": ncf_context_summary,
        "ora_initial_draft_response": ora_initial_draft_response,
        "vre_critique_json": vre_critique_json,
        "vre_overall_alignment": vre_overall_alignment,
        "vre_recommendations_applied": vre_recommendations_applied or [],
        "ora_refined_response": ora_refined_response,
        "mcl_eoc_assessment_at_turn_start": mcl_eoc_assessment_at_turn_start,
        "active_ncf_parameters": active_ncf_parameters or {},
    }

    pls_logged_successfully = False
    jsonl_logged_successfully = False
    pls_error_msg = None
    jsonl_error_msg = None

    if PLS_FOR_FINETUNING_AVAILABLE and persistent_log_service:
        try:
            combined_tags_for_pls = ["finetuning_datapoint", "ora_self_correction"] + (tags or [])
            persistent_log_service.log_event(
                event_type="FINETUNING_DATA_POINT",
                data_payload=data_payload_dict,
                source_agent=source_agent,
                session_id=session_id,
                turn_id=turn_id,
                tags=list(set(combined_tags_for_pls))
            )
            pls_logged_successfully = True
            logger.info(
                f"ObservabilityTools: Logged finetuning data point for turn '{turn_id}' to PersistentLogService.")
        except Exception as e_pls_log:
            pls_error_msg = str(e_pls_log)
            logger.error(
                f"ObservabilityTools: Error logging finetuning data point to PersistentLogService: {pls_error_msg}",
                exc_info=True)
    else:
        pls_error_msg = "PersistentLogService not available or not initialized."
        logger.warning(f"ObservabilityTools: Skipping PersistentLogService for finetuning data: {pls_error_msg}")

    try:
        pydantic_input_dict = {
            "session_id": session_id,
            "turn_id": turn_id,
            **data_payload_dict,
            "tags": tags or []
        }
        if FINETUNING_DATATYPE_LOADED:
            data_point_for_jsonl = FinetuningDataPoint(**pydantic_input_dict)
            log_entry_str = data_point_for_jsonl.model_dump_json(exclude_none=True)
        else:
            data_point_for_jsonl_dict_fallback = {
                "log_id": f"ft_log_{time.time_ns()}",
                "timestamp": time.time(),
                **pydantic_input_dict
            }
            log_entry_str = json.dumps({k: v for k, v in data_point_for_jsonl_dict_fallback.items() if v is not None})

        FINETUNING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FINETUNING_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry_str + "\n")
        jsonl_logged_successfully = True
        logger.info(
            f"ObservabilityTools: Logged finetuning data point for turn '{turn_id}' to JSONL file: {FINETUNING_LOG_FILE.name}.")
    except Exception as e_jsonl:
        jsonl_error_msg = str(e_jsonl)
        logger.error(f"ObservabilityTools: Error logging finetuning data point to JSONL: {jsonl_error_msg}",
                     exc_info=True)

    if pls_logged_successfully and jsonl_logged_successfully:
        return create_successful_tool_response(
            message="Finetuning data point logged successfully to Persistent DB and JSONL file.")
    elif pls_logged_successfully:
        return create_successful_tool_response(
            message=f"Finetuning data point logged to Persistent DB, but JSONL logging failed: {jsonl_error_msg}")
    elif jsonl_logged_successfully:
        return create_successful_tool_response(
            message=f"Finetuning data point logged to JSONL file, but Persistent DB logging failed: {pls_error_msg}")
    else:
        return create_error_tool_response(
            error_message="Failed to log finetuning data to any store.",
            details={"turn_id": turn_id, "pls_error": pls_error_msg, "jsonl_error": jsonl_error_msg}
        )

# MODIFIED: Removed name and description arguments
log_finetuning_data_tool = FunctionTool(
    func=log_ora_self_correction_for_finetuning # Use the renamed Python function
)

ceaf_observability_tools = [
    log_finetuning_data_tool,
]

