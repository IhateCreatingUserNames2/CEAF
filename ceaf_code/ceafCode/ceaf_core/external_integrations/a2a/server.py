# A2A Server
# ceaf_project/ceaf_core/external_integrations/a2a/server.py

import asyncio
import json
import uuid
import time
import logging
from typing import Dict, Any, List, Tuple, Optional

from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


# --- A2A Protocol Data Structures (Simplified based on spec) ---
# In a real scenario, these would come from an official A2A library

class A2APart(BaseModel):
    type: str  # "text", "file", "data"
    text: Optional[str] = None
    file: Optional[Dict[str, Any]] = None  # e.g., {"name": "report.pdf", "mimeType": "application/pdf", "uri": "..."}
    data: Optional[Dict[str, Any]] = None  # For structured JSON
    metadata: Optional[Dict[str, Any]] = None


class A2AMessage(BaseModel):
    role: str  # "user", "agent"
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None


class A2AArtifact(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[A2APart]
    metadata: Optional[Dict[str, Any]] = None
    index: int = 0
    append: Optional[bool] = False
    lastChunk: Optional[bool] = False


class A2ATaskStatus(BaseModel):
    state: str  # "submitted", "working", "input-required", "completed", "failed", "canceled"
    message: Optional[A2AMessage] = None
    timestamp: Optional[str] = Field(
        default_factory=lambda: time.toISOString())  # Assuming time.toISOString() available


class A2ATask(BaseModel):
    id: str
    sessionId: Optional[str] = None
    status: A2ATaskStatus
    history: Optional[List[A2AMessage]] = None
    artifacts: Optional[List[A2AArtifact]] = None
    metadata: Optional[Dict[str, Any]] = None


class A2ASendParams(BaseModel):
    id: str  # Task ID
    sessionId: Optional[str] = None
    message: A2AMessage
    # historyLength: Optional[int] = None
    # pushNotification: Optional[Dict[str, Any]] = None # Simplified for now


class A2ARPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int  # Request ID
    method: str  # e.g., "tasks/send"
    params: Dict[str, Any]


# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearchAgentA2AServer")

# --- FastAPI App for ResearchAgent Server ---
research_agent_a2a_app = FastAPI(
    title="External ResearchAgent A2A Server",
    description="Exposes a 'perform_deep_research' skill via A2A protocol.",
)

# In-memory store for tasks for this simple server
# Production would use a persistent DB
active_tasks: Dict[str, A2ATask] = {}

# --- Agent Card ---
RESEARCH_AGENT_CARD = {
    "name": "Specialized Research Agent",
    "description": "Performs in-depth research on specified topics and generates reports.",
    "url": "http://localhost:8001",  # Assuming this server runs on port 8001
    "provider": {"organization": "Research Inc.", "url": "https://research.example.com"},
    "version": "1.0.0",
    "authentication": {"schemes": ["None"]},  # For simplicity, no auth
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["application/json", "text/plain"],
    "capabilities": {"streaming": True, "pushNotifications": False},  # Supports streaming
    "skills": [
        {
            "id": "perform_deep_research",
            "name": "Perform Deep Research",
            "description": "Takes a research topic, conducts deep research, and returns a structured report. Supports streaming updates.",
            "tags": ["research", "analysis", "reporting"],
            "examples": ["Research quantum computing advancements in 2024."],
            "inputModes": ["text/plain"],  # Expects topic as text
            "outputModes": ["application/json"]  # Returns report as JSON
        }
    ]
}


@research_agent_a2a_app.get("/.well-known/agent.json", summary="Get Agent Card")
async def get_agent_card():
    return JSONResponse(content=RESEARCH_AGENT_CARD)


# --- Core Skill Logic ---
async def _execute_deep_research(task_id: str, topic: str, stream_writer=None):
    """Simulates the research process, yielding updates if streaming."""
    logger.info(f"Task {task_id}: Starting research on '{topic}'")

    async def send_status_update(state: str, message_text: Optional[str] = None, final: bool = False):
        if stream_writer:
            status_obj = A2ATaskStatus(state=state)
            if message_text:
                status_obj.message = A2AMessage(role="agent", parts=[A2APart(type="text", text=message_text)])

            event_data = {
                "jsonrpc": "2.0",  # Part of SSE stream content
                "result": {"id": task_id, "status": status_obj.model_dump(exclude_none=True), "final": final}
            }
            # SSE format: data: <json_string>\n\n
            await stream_writer(f"data: {json.dumps(event_data)}\n\n")
            logger.debug(f"Task {task_id}: Sent SSE status update: {state}")

    async def send_artifact_update(artifact: A2AArtifact, final_artifact_chunk: bool = False):
        if stream_writer:
            artifact.lastChunk = final_artifact_chunk  # Mark if it's the last part of this artifact
            event_data = {
                "jsonrpc": "2.0",
                "result": {"id": task_id, "artifact": artifact.model_dump(exclude_none=True)}
            }
            await stream_writer(f"data: {json.dumps(event_data)}\n\n")
            logger.debug(f"Task {task_id}: Sent SSE artifact update for '{artifact.name}'")

    # 1. Initial "working" status
    active_tasks[task_id].status = A2ATaskStatus(state="working")
    await send_status_update("working", "Research initiated. Analyzing sources...")
    await asyncio.sleep(2)  # Simulate work

    # 2. Intermediate progress update
    await send_status_update("working", "Gathering primary data points...")
    await asyncio.sleep(3)  # Simulate more work

    # 3. Simulate generating parts of an artifact (report)
    report_artifact_name = f"research_report_{topic.replace(' ', '_')}.json"
    report_parts_content = []

    # Part 1 of the report
    part1_data = {"introduction": f"Preliminary findings for '{topic}'.", "confidence": "medium"}
    report_parts_content.append(A2APart(type="data", data=part1_data))
    if stream_writer:  # If streaming, send artifact part by part
        artifact_chunk1 = A2AArtifact(
            name=report_artifact_name,
            description=f"Research report on {topic}",
            parts=[A2APart(type="data", data=part1_data)],
            index=0,  # First artifact
            append=False  # This is the first chunk of this artifact
        )
        await send_artifact_update(artifact_chunk1)
    await asyncio.sleep(2)

    # Part 2 of the report
    part2_data = {"key_findings": ["Finding A", "Finding B related to " + topic, "Finding C"],
                  "data_sources": ["Source X", "Source Y"]}
    report_parts_content.append(A2APart(type="data", data=part2_data))
    if stream_writer:
        artifact_chunk2 = A2AArtifact(
            name=report_artifact_name,  # Same artifact name
            parts=[A2APart(type="data", data=part2_data)],
            index=0,  # Still the first artifact
            append=True  # Appending to the previous part of this artifact
        )
        await send_artifact_update(artifact_chunk2, final_artifact_chunk=True)  # Mark as last chunk for this artifact
    await asyncio.sleep(1)

    # 4. Final "completed" status
    final_status = A2ATaskStatus(state="completed")
    active_tasks[task_id].status = final_status
    active_tasks[task_id].artifacts = [
        A2AArtifact(name=report_artifact_name, description=f"Full research report on {topic}",
                    parts=report_parts_content, index=0)
    ]
    await send_status_update("completed", "Research complete. Report generated.", final=True)  # Mark stream as final
    logger.info(f"Task {task_id}: Research on '{topic}' completed.")


# --- A2A JSON-RPC Endpoint ---
@research_agent_a2a_app.post("/", summary="A2A JSON-RPC Endpoint")
async def a2a_rpc_handler(rpc_request_model: A2ARPCRequest, raw_request: Request):
    # raw_body = await raw_request.body()
    # rpc_request = json.loads(raw_body.decode()) # Manual parsing if not using Pydantic model for full body

    rpc_request = rpc_request_model.model_dump()
    method = rpc_request.get("method")
    params = rpc_request.get("params", {})
    req_id = rpc_request.get("id")

    logger.info(f"Received A2A request: method='{method}', params_keys='{list(params.keys()) if params else None}'")

    if method == "tasks/send" or method == "tasks/sendSubscribe":
        try:
            send_params = A2ASendParams(**params)
        except Exception as e:  # Pydantic validation error
            logger.error(f"Invalid params for {method}: {e}")
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Invalid params: {e}"}}
            )

        task_id = send_params.id
        user_message_text = ""
        if send_params.message.parts and send_params.message.parts[0].type == "text":
            user_message_text = send_params.message.parts[0].text

        if not user_message_text:
            return JSONResponse(
                status_code=400,
                content={"jsonrpc": "2.0", "id": req_id,
                         "error": {"code": -32602, "message": "Missing text part in user message for research topic."}}
            )

        # Check if it's a new task or continuation (simplified: always new if not exists)
        if task_id not in active_tasks:
            new_task = A2ATask(
                id=task_id,
                sessionId=send_params.sessionId or str(uuid.uuid4()),
                status=A2ATaskStatus(state="submitted"),
                history=[send_params.message]
            )
            active_tasks[task_id] = new_task
            logger.info(f"New A2A Task created: {task_id} for topic: '{user_message_text}'")
        else:
            # Handle multi-turn tasks if necessary (not for this simple research skill)
            active_tasks[task_id].history.append(send_params.message)
            logger.info(f"Continuing A2A Task: {task_id}")

        # If tasks/sendSubscribe, start streaming
        if method == "tasks/sendSubscribe":
            async def stream_generator():
                # Yield initial acknowledgment for SSE
                # According to A2A, the HTTP response for tasks/sendSubscribe is actually empty (200 OK)
                # and the stream starts immediately after. Some clients might expect an immediate event though.
                # For simplicity, we'll let the first status update from _execute_deep_research be the first SSE event.

                # This function will be responsible for writing to the SSE stream
                async def sse_writer(data_str: str):
                    # This is tricky with FastAPI's StreamingResponse which expects the generator to yield strings directly.
                    # We'll adapt _execute_deep_research to yield strings for SSE.
                    # For this example, we'll directly call and let it manage things.
                    # In a real app, you might pass a queue or a callback.
                    yield data_str  # This is what StreamingResponse expects

                # Start the background task
                # We need to ensure the client doesn't time out waiting for the first byte
                # The _execute_deep_research now takes a stream_writer which is our sse_writer
                asyncio.create_task(_execute_deep_research(task_id, user_message_text, stream_writer=None))

                # This generator will now poll the task status or receive events if _execute_deep_research used a queue
                # For this version, we'll make _execute_deep_research yield the SSE strings directly.

                current_task_state = active_tasks[task_id].status.state
                yield f"data: {json.dumps({'jsonrpc': '2.0', 'result': {'id': task_id, 'status': active_tasks[task_id].status.model_dump(exclude_none=True), 'final': False}})}\n\n"

                while current_task_state not in ["completed", "failed", "canceled"]:
                    await asyncio.sleep(0.5)  # Poll interval
                    if task_id not in active_tasks: break  # Task removed

                    updated_task = active_tasks[task_id]
                    if updated_task.status.state != current_task_state:
                        current_task_state = updated_task.status.state
                        is_final = current_task_state in ["completed", "failed", "canceled"]

                        # Send status update
                        status_event_data = {
                            "jsonrpc": "2.0",
                            "result": {"id": task_id, "status": updated_task.status.model_dump(exclude_none=True),
                                       "final": is_final}
                        }
                        yield f"data: {json.dumps(status_event_data)}\n\n"

                        # Send artifact updates if any (simplified: send all on completion for this poll version)
                        if is_final and updated_task.artifacts:
                            for artifact in updated_task.artifacts:
                                artifact_event_data = {
                                    "jsonrpc": "2.0",
                                    "result": {"id": task_id, "artifact": artifact.model_dump(exclude_none=True)}
                                }
                                yield f"data: {json.dumps(artifact_event_data)}\n\n"
                    if is_final:
                        break

            # Start the background task (non-blocking for FastAPI)
            asyncio.create_task(_execute_deep_research(task_id, user_message_text))
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:  # tasks/send (synchronous-like, but skill is async)
            await _execute_deep_research(task_id, user_message_text)  # Run skill
            final_task_state = active_tasks.get(task_id)
            if final_task_state:
                return JSONResponse(
                    content={"jsonrpc": "2.0", "id": req_id, "result": final_task_state.model_dump(exclude_none=True)})
            else:  # Should not happen if task creation was successful
                return JSONResponse(
                    status_code=500,
                    content={"jsonrpc": "2.0", "id": req_id,
                             "error": {"code": -32000, "message": "Task processing failed unexpectedly."}}
                )


    elif method == "tasks/get":
        task_id = params.get("id")
        if task_id and task_id in active_tasks:
            # historyLength = params.get("historyLength", 0) # Implement if needed
            return JSONResponse(
                content={"jsonrpc": "2.0", "id": req_id, "result": active_tasks[task_id].model_dump(exclude_none=True)})
        else:
            return JSONResponse(
                status_code=404,  # Or A2A specific error
                content={"jsonrpc": "2.0", "id": req_id, "error": {"code": -32001, "message": "Task not found."}}
            )

    # Implement tasks/cancel, tasks/pushNotification/set, tasks/pushNotification/get, tasks/resubscribe as needed

    else:
        return JSONResponse(
            status_code=400,
            content={"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found."}}
        )

# To run this server (save as server.py in the a2a directory):
# uvicorn ceaf_core.external_integrations.a2a.server:research_agent_a2a_app --port 8001 --reload