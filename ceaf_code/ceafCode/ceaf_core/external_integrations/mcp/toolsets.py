# MCP Toolsets
# ceaf_project/ceaf_core/external_integrations/mcp/toolsets.py

import os
import logging
import asyncio
from typing import List, Tuple, Optional
from contextlib import AsyncExitStack

from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters, SseServerParams
from google.adk.tools import BaseTool as AdkBaseTool  # For type hinting

# --- Environment and Path Setup ---
# Not strictly needed here if .env is loaded by main.py, but good practice if run standalone
from pathlib import Path
from dotenv import load_dotenv
from pydantic import json

project_root = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=project_root / ".env")

logger = logging.getLogger(__name__)

# --- Configuration for External MCP Servers ---

# Filesystem MCP Server
MCP_FS_SERVER_ENABLED = os.getenv("MCP_FS_SERVER_ENABLED", "false").lower() == "true"
MCP_FS_SERVER_COMMAND = os.getenv("MCP_FS_SERVER_COMMAND", "npx")
MCP_FS_SERVER_ARGS_JSON = os.getenv(
    "MCP_FS_SERVER_ARGS_JSON",
    '["-y", "@modelcontextprotocol/server-filesystem"]'  # Default args without path
)
# CRITICAL: This path MUST be an ABSOLUTE path accessible by the npx command.
# It's the root directory the FS MCP server will expose.
MCP_FS_SERVER_ROOT_PATH = os.getenv("MCP_FS_SERVER_ROOT_PATH")  # e.g., "/mnt/ceaf_shared_files"

# Google Maps MCP Server
MCP_MAPS_SERVER_ENABLED = os.getenv("MCP_MAPS_SERVER_ENABLED", "false").lower() == "true"
MCP_MAPS_SERVER_COMMAND = os.getenv("MCP_MAPS_SERVER_COMMAND", "npx")
MCP_MAPS_SERVER_ARGS_JSON = os.getenv(
    "MCP_MAPS_SERVER_ARGS_JSON",
    '["-y", "@modelcontextprotocol/server-google-maps"]'
)
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY_FOR_MCP")  # Specific key for MCP server if needed


# --- MCP Toolset Initializers ---

async def initialize_filesystem_mcp_tools() -> Tuple[Optional[List[AdkBaseTool]], Optional[AsyncExitStack]]:
    """
    Initializes ADK tools from an external Filesystem MCP server.

    Returns:
        A tuple containing:
        - A list of ADK BaseTool objects if successful, else None.
        - An AsyncExitStack to manage the MCP server process lifecycle if successful, else None.
    """
    if not MCP_FS_SERVER_ENABLED:
        logger.info("Filesystem MCP Server integration is disabled via MCP_FS_SERVER_ENABLED.")
        return None, None

    if not MCP_FS_SERVER_ROOT_PATH:
        logger.error("MCP_FS_SERVER_ROOT_PATH is not set. Cannot initialize Filesystem MCP tools.")
        return None, None
    if not Path(MCP_FS_SERVER_ROOT_PATH).is_dir():
        logger.error(f"MCP_FS_SERVER_ROOT_PATH '{MCP_FS_SERVER_ROOT_PATH}' is not a valid directory.")
        return None, None

    try:
        fs_server_args = json.loads(MCP_FS_SERVER_ARGS_JSON)
        # IMPORTANT: The actual root path for the server-filesystem must be the LAST argument.
        fs_server_args.append(MCP_FS_SERVER_ROOT_PATH)

        logger.info(
            f"Attempting to connect to Filesystem MCP server with command: '{MCP_FS_SERVER_COMMAND}' and args: {fs_server_args}")

        tools, exit_stack = await MCPToolset.from_server(
            connection_params=StdioServerParameters(
                command=MCP_FS_SERVER_COMMAND,
                args=fs_server_args,
            )
        )
        logger.info(
            f"Successfully fetched {len(tools)} tools from Filesystem MCP server for path '{MCP_FS_SERVER_ROOT_PATH}'.")
        return tools, exit_stack
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MCP_FS_SERVER_ARGS_JSON: {e}. Value: {MCP_FS_SERVER_ARGS_JSON}")
        return None, None
    except FileNotFoundError:
        logger.error(
            f"Command '{MCP_FS_SERVER_COMMAND}' for Filesystem MCP server not found. Is npx (or specified command) in PATH?")
        return None, None
    except Exception as e:
        logger.error(f"Failed to initialize Filesystem MCP tools: {e}", exc_info=True)
        return None, None


async def initialize_maps_mcp_tools() -> Tuple[Optional[List[AdkBaseTool]], Optional[AsyncExitStack]]:
    """
    Initializes ADK tools from an external Google Maps MCP server.

    Returns:
        A tuple containing:
        - A list of ADK BaseTool objects if successful, else None.
        - An AsyncExitStack to manage the MCP server process lifecycle if successful, else None.
    """
    if not MCP_MAPS_SERVER_ENABLED:
        logger.info("Google Maps MCP Server integration is disabled via MCP_MAPS_SERVER_ENABLED.")
        return None, None

    if not GOOGLE_MAPS_API_KEY:
        logger.error("GOOGLE_MAPS_API_KEY_FOR_MCP is not set. Cannot initialize Google Maps MCP tools.")
        return None, None

    try:
        maps_server_args = json.loads(MCP_MAPS_SERVER_ARGS_JSON)
        logger.info(
            f"Attempting to connect to Google Maps MCP server with command: '{MCP_MAPS_SERVER_COMMAND}' and args: {maps_server_args}")

        tools, exit_stack = await MCPToolset.from_server(
            connection_params=StdioServerParameters(
                command=MCP_MAPS_SERVER_COMMAND,
                args=maps_server_args,
                env={"GOOGLE_MAPS_API_KEY": GOOGLE_MAPS_API_KEY}  # Pass API key as env var to the npx process
            )
        )
        logger.info(f"Successfully fetched {len(tools)} tools from Google Maps MCP server.")
        return tools, exit_stack
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MCP_MAPS_SERVER_ARGS_JSON: {e}. Value: {MCP_MAPS_SERVER_ARGS_JSON}")
        return None, None
    except FileNotFoundError:
        logger.error(
            f"Command '{MCP_MAPS_SERVER_COMMAND}' for Google Maps MCP server not found. Is npx (or specified command) in PATH?")
        return None, None
    except Exception as e:
        logger.error(f"Failed to initialize Google Maps MCP tools: {e}", exc_info=True)
        return None, None


async def initialize_all_mcp_toolsets() -> Tuple[List[AdkBaseTool], AsyncExitStack]:
    """
    Initializes all configured external MCP toolsets and aggregates their tools and exit stacks.

    Returns:
        A tuple containing:
        - A list of all ADK BaseTool objects from all successful MCP initializations.
        - A single AsyncExitStack managing all created MCP server processes.
    """
    all_tools: List[AdkBaseTool] = []
    # Create a master exit_stack that will manage individual exit_stacks from each toolset
    master_exit_stack = AsyncExitStack()

    fs_tools, fs_exit_stack = await initialize_filesystem_mcp_tools()
    if fs_tools and fs_exit_stack:
        all_tools.extend(fs_tools)
        # Enter the individual exit_stack into the master_exit_stack.
        # This means when master_exit_stack.aclose() is called, fs_exit_stack.aclose() will also be called.
        await master_exit_stack.enter_async_context(fs_exit_stack)
        logger.info(f"Filesystem MCP tools added. Total tools: {len(all_tools)}")

    maps_tools, maps_exit_stack = await initialize_maps_mcp_tools()
    if maps_tools and maps_exit_stack:
        all_tools.extend(maps_tools)
        await master_exit_stack.enter_async_context(maps_exit_stack)
        logger.info(f"Google Maps MCP tools added. Total tools: {len(all_tools)}")

    # Add more initializers for other MCP toolsets here

    if not all_tools:
        logger.warning(
            "No MCP tools were initialized. Returning empty list and an empty exit stack (will be closed immediately).")
        # If no tools were loaded, the master_exit_stack is still valid but might be empty.
        # To avoid issues with an empty stack that wasn't entered, we can close it if empty.
        # However, it's generally safe to return it as is.
        # If it's truly empty and `aclose` is called, it does nothing.

    return all_tools, master_exit_stack
