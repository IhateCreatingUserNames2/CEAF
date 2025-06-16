# ceaf_project/ceaf_core/external_integrations/a2a/client.py

import asyncio
import json
import uuid
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator

import httpx  # For async HTTP requests
from pydantic import BaseModel


# Import data models (ideally from a shared A2A lib or define consistently)
# For this example, we might redefine or assume they are accessible
# from .server import A2APart, A2AMessage, A2ATask, A2ASendParams # If in same package
# Or define them again if client is truly separate:

class A2APart(BaseModel):  # Copied from server for example
    type: str
    text: Optional[str] = None
    file: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):  # Copied
    role: str
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None


class A2ATaskStatus(BaseModel):  # Copied
    state: str
    message: Optional[A2AMessage] = None
    timestamp: Optional[str] = None


class A2AArtifact(BaseModel):  # Copied
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None
    index: int = 0
    append: Optional[bool] = False
    lastChunk: Optional[bool] = False


class A2ATask(BaseModel):  # Copied
    id: str
    sessionId: Optional[str] = None
    status: A2ATaskStatus
    history: Optional[List[A2AMessage]] = None
    artifacts: Optional[List[A2AArtifact]] = None
    metadata: Optional[Dict[str, Any]] = None


# --- End Copied Models ---

logger = logging.getLogger("A2AClient")

A2A_REQUEST_ID_COUNTER = 1


async def _make_a2a_request(server_url: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to make a generic A2A JSON-RPC request."""
    global A2A_REQUEST_ID_COUNTER
    request_payload = {
        "jsonrpc": "2.0",
        "id": A2A_REQUEST_ID_COUNTER,
        "method": method,
        "params": params
    }
    A2A_REQUEST_ID_COUNTER += 1

    async with httpx.AsyncClient(timeout=30.0) as client:  # Timeout for requests
        try:
            logger.debug(
                f"A2A Client: Sending POST to {server_url}, method: {method}, params: {json.dumps(params, default=str)[:200]}...")
            response = await client.post(server_url, json=request_payload)
            response.raise_for_status()  # Raise an exception for HTTP error codes
            response_json = response.json()
            logger.debug(f"A2A Client: Received response: {json.dumps(response_json, default=str)[:200]}...")

            if "error" in response_json:
                logger.error(f"A2A server returned an error for method {method}: {response_json['error']}")
                # You might want to raise a custom exception here
            return response_json
        except httpx.HTTPStatusError as e:
            logger.error(
                f"A2A HTTP error for method {method} to {server_url}: {e.response.status_code} - {e.response.text}")
            raise  # Re-raise to be handled by caller
        except httpx.RequestError as e:
            logger.error(f"A2A request error for method {method} to {server_url}: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(
                f"A2A JSON decode error for method {method} from {server_url}: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
            raise


async def discover_a2a_agent(server_base_url: str) -> Optional[Dict[str, Any]]:
    """Fetches the Agent Card from an A2A server."""
    agent_card_url = f"{server_base_url.rstrip('/')}/.well-known/agent.json"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            logger.info(f"A2A Client: Discovering agent at {agent_card_url}")
            response = await client.get(agent_card_url)
            response.raise_for_status()
            agent_card = response.json()
            logger.info(f"A2A Client: Discovered agent '{agent_card.get('name')}'")
            return agent_card
    except Exception as e:
        logger.error(f"A2A Client: Failed to discover agent at {agent_card_url}: {e}")
        return None


async def send_task_to_a2a_agent(
        server_url: str,
        task_id: str,
        message_text: str,
        session_id: Optional[str] = None
) -> Optional[A2ATask]:
    """Sends a new task (or message to an existing task) to an A2A agent."""
    user_message = A2AMessage(
        role="user",
        parts=[A2APart(type="text", text=message_text)]
    )
    params = {
        "id": task_id,
        "message": user_message.model_dump(exclude_none=True)
    }
    if session_id:
        params["sessionId"] = session_id

    response_json = await _make_a2a_request(server_url, "tasks/send", params)
    if response_json and "result" in response_json:
        try:
            return A2ATask(**response_json["result"])
        except Exception as e:  # Pydantic validation error
            logger.error(f"A2A Client: Could not parse task result from tasks/send: {e}")
            return None
    return None


async def stream_task_from_a2a_agent(
        server_url: str,
        task_id: str,
        message_text: str,
        session_id: Optional[str] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Sends a task using tasks/sendSubscribe and streams updates (status, artifacts).
    Yields raw event dictionaries from the SSE stream.
    """
    user_message = A2AMessage(
        role="user",
        parts=[A2APart(type="text", text=message_text)]
    )
    params = {
        "id": task_id,
        "message": user_message.model_dump(exclude_none=True)
    }
    if session_id:
        params["sessionId"] = session_id

    request_payload = {
        "jsonrpc": "2.0",
        "id": A2A_REQUEST_ID_COUNTER,  # Use global counter
        "method": "tasks/sendSubscribe",
        "params": params
    }
    # Increment counter after use (or manage IDs per request if truly parallel)
    global A2A_REQUEST_ID_COUNTER
    A2A_REQUEST_ID_COUNTER += 1

    try:
        async with httpx.AsyncClient(timeout=None) as client:  # Timeout None for streaming
            logger.info(f"A2A Client: Sending tasks/sendSubscribe to {server_url} for task {task_id}")
            async with client.stream("POST", server_url, json=request_payload) as response:
                response.raise_for_status()  # Check initial HTTP status
                logger.info(f"A2A Client: SSE Stream opened for task {task_id}. Status: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if data_str:
                            try:
                                event_data = json.loads(data_str)
                                logger.debug(
                                    f"A2A Client (Task {task_id}) SSE Event: {event_data.get('result', {}).get('status', {}).get('state', event_data)}")
                                yield event_data  # Yield the raw JSON-RPC like event
                                if event_data.get("result", {}).get("status", {}).get("final") is True:
                                    logger.info(f"A2A Client: SSE Stream indicated final event for task {task_id}.")
                                    break  # Stop if server marks as final
                            except json.JSONDecodeError:
                                logger.warning(f"A2A Client: Could not decode SSE JSON: {data_str}")
                    elif line.strip():  # Log other non-empty lines for debugging
                        logger.debug(f"A2A Client (Task {task_id}) SSE Raw Line: {line}")


    except httpx.HTTPStatusError as e:
        logger.error(
            f"A2A SSE HTTP error for task {task_id} to {server_url}: {e.response.status_code} - {e.response.text}")
        yield {"error": {"code": e.response.status_code, "message": f"HTTP error: {e.response.text}"}}
    except httpx.RequestError as e:
        logger.error(f"A2A SSE request error for task {task_id} to {server_url}: {e}")
        yield {"error": {"code": -1, "message": f"Request error: {str(e)}"}}
    except Exception as e:
        logger.error(f"A2A Client: Unexpected error in stream_task_from_a2a_agent for task {task_id}: {e}",
                     exc_info=True)
        yield {"error": {"code": -1, "message": f"Unexpected streaming error: {str(e)}"}}


async def get_a2a_task_status(server_url: str, task_id: str) -> Optional[A2ATask]:
    """Retrieves the status and details of an existing A2A task."""
    params = {"id": task_id}
    response_json = await _make_a2a_request(server_url, "tasks/get", params)
    if response_json and "result" in response_json:
        try:
            return A2ATask(**response_json["result"])
        except Exception as e:  # Pydantic validation error
            logger.error(f"A2A Client: Could not parse task result from tasks/get: {e}")
            return None
    return None

