# MCP Server
# ceaf_project/ceaf_core/external_integrations/mcp/server.py

import asyncio
import json
import logging
import os

# --- Environment and Path Setup ---
# This helps if running the server directly and ceaf_core is not in PYTHONPATH yet
import sys
import time
import uuid
from pathlib import Path
from typing import List, Any

# Add the project root to sys.path to allow imports like 'from ceaf_core...'
project_root = Path(__file__).resolve().parents[
    3]  # Adjust if structure changes (root is 3 levels up from mcp/server.py)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# --- MCP Server Imports ---
from mcp import types as mcp_types  # Use alias to avoid conflict with genai.types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio  # For running over stdio

# --- ADK Tool Imports & Conversion Utility ---
from google.adk.tools import FunctionTool, ToolContext  # For dummy context
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type
from google.adk.sessions import InMemorySessionService, State  # For dummy context

# Import the specific ADK tool from CEAF to expose
try:
    from ceaf_core.tools.ncf_tools import ncf_tool as ceaf_ncf_adk_tool
except ImportError:
    logging.error("Failed to import 'ceaf_ncf_adk_tool' from 'ceaf_core.tools.ncf_tools'. Ensure the tool is defined.")


    # Define a dummy ADK tool if import fails, so server can still start (in degraded mode)
    def _dummy_ncf_func(user_query: str, tool_context: ToolContext) -> dict:
        return {"status": "error", "ncf": "Dummy NCF Tool: CEAF NCF ADK tool not loaded."}


    ceaf_ncf_adk_tool = FunctionTool(
        func=_dummy_ncf_func,
        description="Generates a Narrative Context Frame for a given query (Dummy Version)."
    )
    logging.warning("Using a DUMMY NCF ADK tool for the MCP server.")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CEAF_MCP_NCF_Server")

# --- Load Environment Variables (If ADK tools need them, e.g., API keys for sub-calls) ---
# NCF tool might eventually call other services needing keys.
load_dotenv(dotenv_path=project_root / ".env")  # Load .env from project root

# --- Prepare the ADK Tool to be Exposed ---
# The ceaf_ncf_adk_tool is already instantiated from the import.
logger.info(
    f"ADK tool '{ceaf_ncf_adk_tool.name}' (version: {'CEAF Core' if ceaf_ncf_adk_tool.func != _dummy_ncf_func else 'Dummy'}) will be exposed via MCP.")

# --- MCP Server Setup ---
logger.info("Creating MCP Server instance...")
# Create a named MCP Server instance
app = Server(
    name="ceaf-ncf-mcp-server",
    display_name="CEAF Narrative Context Frame Server",
    version="0.1.0"
)


@app.list_tools()
async def list_mcp_tools() -> list[mcp_types.Tool]:
    """MCP handler to list available tools."""
    logger.info("MCP Server: Received list_tools request.")
    try:
        # Convert the ADK tool's definition to MCP format
        mcp_tool_schema = adk_to_mcp_tool_type(ceaf_ncf_adk_tool)
        logger.info(
            f"MCP Server: Advertising tool: {mcp_tool_schema.name} (Input Schema: {mcp_tool_schema.input_schema})")
        return [mcp_tool_schema]
    except Exception as e:
        logger.error(f"MCP Server: Error converting ADK tool to MCP schema: {e}", exc_info=True)
        return []


@app.call_tool()
async def call_mcp_tool(
        name: str, arguments: dict[str, Any], context: mcp_types.ToolCallContext
) -> list[mcp_types.Content]:  # MCP expects a list of Content objects
    """MCP handler to execute a tool call."""
    logger.info(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

    # Check if the requested tool name matches our wrapped ADK tool
    if name == ceaf_ncf_adk_tool.name:
        try:
            # --- Prepare a dummy ADK ToolContext ---
            # The ADK tool expects a ToolContext. Since this MCP server is not running
            # within a full ADK Runner invocation, we need to mock or create a minimal one.
            # For the NCF tool, it might use tool_context.state if it becomes more dynamic.
            # For now, a simple one is fine.
            dummy_session_service = InMemorySessionService()
            # Use invocation_id from MCP context if available, else generate one
            invocation_id = context.invocation_id if context and context.invocation_id else f"mcp-inv-{uuid.uuid4().hex[:8]}"

            # Create a dummy session state for the ToolContext
            # The NCF tool might read/write to this state.
            # We need a 'State' object, not just a dict, if tool_context expects one.
            # ADK's ToolContext has `state` as a `google.adk.sessions.State` object.

            # A minimal InvocationContext for ToolContext (most fields can be None/default)
            # This part is tricky as InvocationContext is usually built by the ADK Runner.
            # We create a simplified version.
            class MinimalInvocationContext:
                def __init__(self, state_dict: dict, inv_id: str):
                    self.session = MinimalSession(state_dict, inv_id)
                    self.app_name = "mcp_exposed_app"  # Dummy app name
                    self.user_id = "mcp_user"  # Dummy user
                    self.session_id = self.session.id
                    self.invocation_id = inv_id
                    self.current_agent_name = "mcp_tool_host_agent"  # Dummy agent name
                    # Other fields like runner, services are harder to mock simply.
                    # Hope that ncf_tool doesn't rely on them too much.
                    self.runner = None  # Cannot easily provide the full runner here
                    self.session_service = dummy_session_service
                    self.memory_service = None
                    self.artifact_service = None
                    self.current_events: List[Any] = []
                    self.parent_agent: Any = None
                    self.run_config: Any = None  # Assuming default run_config

            class MinimalSession:
                def __init__(self, state_dict: dict, sess_id: str):
                    self.state = State(value=state_dict.copy(), delta={})  # ADK State object
                    self.id = sess_id
                    self.user_id = "mcp_user"
                    self.app_name = "mcp_exposed_app"
                    self.events = []
                    self.last_update_time = time.time()

            # Initial state for the dummy context for this call
            initial_state_dict = {"mcp_call_source": context.client_name if context else "unknown_mcp_client"}

            # MCP's ToolCallContext provides `tool_inputs` which are the files/resources.
            # Our NCF tool currently takes `user_query: str`. If it needed files, we'd map them.
            # `arguments` from MCP call_tool maps to ADK tool `args`.

            mock_invocation_context = MinimalInvocationContext(initial_state_dict, invocation_id)

            adk_tool_context = ToolContext(
                invocation_context=mock_invocation_context,
                function_call_id=invocation_id,  # Use a unique ID for the call
            )
            # Now, the adk_tool_context.state is an ADK State object
            # logger.debug(f"MCP Server: Prepared dummy ADK ToolContext. State: {adk_tool_context.state.to_dict()}")

            # Execute the ADK tool's run_async method
            # The 'args' for ADK tool come from 'arguments' in MCP call
            adk_response_dict = await ceaf_ncf_adk_tool.run_async(
                args=arguments,  # e.g., {"user_query": "AI ethics"}
                tool_context=adk_tool_context,
            )
            logger.info(f"MCP Server: ADK tool '{name}' executed. Response: {adk_response_dict}")

            # Format the ADK tool's response (a dict) into MCP Content format.
            # NCF tool returns {"status": "success", "ncf": "..."}
            if adk_response_dict.get("status") == "success" and "ncf" in adk_response_dict:
                ncf_text = adk_response_dict["ncf"]
                # MCP expects a list of mcp_types.Content
                return [mcp_types.TextContent(type="text", text=ncf_text)]
            else:
                error_detail = adk_response_dict.get("error_message",
                                                     adk_response_dict.get("message", "Unknown error from ADK tool"))
                logger.error(f"MCP Server: ADK tool '{name}' returned an error or unexpected format: {error_detail}")
                # Return error as TextContent for simplicity, MCP has more formal error structures
                return [mcp_types.TextContent(type="text",
                                              text=json.dumps({"error": f"ADK tool '{name}' failed: {error_detail}"}))]

        except Exception as e:
            logger.error(f"MCP Server: Error executing ADK tool '{name}': {e}", exc_info=True)
            return [mcp_types.TextContent(type="text", text=json.dumps(
                {"error": f"Server error executing tool '{name}': {str(e)}"}))]
    else:
        logger.warning(f"MCP Server: Tool '{name}' not found or not implemented.")
        return [mcp_types.TextContent(type="text",
                                      text=json.dumps({"error": f"Tool '{name}' not implemented by this MCP server."}))]


# --- MCP Server Runner ---
async def run_mcp_stdio_server():
    """Runs the MCP server over standard input/output."""
    # Use the stdio_server context manager from the MCP library
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("CEAF MCP Server (stdio) starting handshake...")
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                # server_name=app.name, # Server.name is already set
                # server_version=app.version, # Server.version is already set
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),  # Default no notifications
                    experimental_capabilities={},
                ),
            ),
        )
        logger.info("CEAF MCP Server (stdio) run loop finished.")


if __name__ == "__main__":
    logger.info("Launching CEAF MCP Server to expose ADK NCF tool via stdio...")
    try:
        asyncio.run(run_mcp_stdio_server())
    except KeyboardInterrupt:
        logger.info("\nCEAF MCP Server stopped by user.")
    except Exception as e:
        logger.error(f"CEAF MCP Server encountered a critical error: {e}", exc_info=True)
    finally:
        logger.info("CEAF MCP Server process exiting.")