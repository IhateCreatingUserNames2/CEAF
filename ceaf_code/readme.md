# CEAF (Coherent Emergence Agent Framework)

An advanced AI system designed to produce thoughtful, coherent, and ethically sound responses through the orchestration of multiple specialized AI agents.

## Overview

The Coherent Emergence Agent Framework (CEAF) is a sophisticated AI agent framework that implements "Terapia para Silício" (Therapy for Silicon). It integrates multiple cognitive architectural components to create a more robust, reflective, coherent, and ethically-aware AI system.

CEAF is built on the Google Agent Development Kit (ADK) and features:

- **Coherence**: Maintaining logical and narrative consistency in thoughts and actions
- **Adaptive Learning**: Continuously improving performance through self-reflection and interaction
- **Ethical Reasoning**: Operating within a defined ethical framework
- **Metacognition**: Components that monitor, assess, and guide behavior
- **Rich Internal State**: Managing various memory types and dynamic self-modeling

## Core Architecture

### Primary Agents

**Orchestrator/Responder Agent (ORA)**
- Central cognitive unit that processes user queries
- Implements a multi-step protocol for response generation
- Equipped with comprehensive tools for interacting with other CEAF modules

**Metacognitive Control Loop Agent (MCL)**
- Performance monitor that assesses ORA's "Edge of Coherence" (EoC) state
- Provides guidance for optimal NCF parameters for future turns
- Analyzes system state and provides operational advice

**Narrative Coherence & Identity Module Agent (NCIM)**
- Manages CEAF's self-representation and identity evolution
- Ensures actions align with identity, goals, and narrative threads
- Detects internal conflicts and derives emergent goals

**Virtue & Reasoning Engine Agent (VRE)**
- Ethical auditor that evaluates responses against ethical principles
- Assesses epistemic virtues (humility, thoroughness, self-correction)
- Provides recommendations for ethical and epistemic soundness

**Knowledge Graph Synthesizer Agent (KGS)**
- Processes textual information to extract entities and relationships
- Builds and maintains structured knowledge representation
- Outputs structured KG elements in JSON format

**Autonomous Universal Reflective Analyzer Agent (AURA)**
- Long-term learner that initiates reflective learning cycles
- Analyzes historical performance data for system-wide improvements
- Generates refinement strategies for the entire CEAF system

### Core Modules

**Memory Blossom Service (MBS)**
- Comprehensive memory management with multiple memory types:
  - Explicit memories (facts, experiences)
  - Emotional contexts
  - Procedural knowledge
  - Goal tracking
  - Knowledge graph entities and relations
- Features dynamic salience, decay, archiving, and semantic search
- Integrates embedding-based similarity search

**Narrative Context Framing (NCF) Engine**
- Provides dynamic interaction frames for ORA
- Includes philosophical grounding, relevant memories, and operational parameters
- Templates guide response generation with contextual information

**Metacognitive Control Loop (MCL) Engine**
- Self-state analysis and performance monitoring
- Finetuning data collection for continuous improvement
- Self-model management (CeafSelfRepresentation)

**Virtue & Reasoning Engine (VRE)**
- Epistemic humility assessment
- Ethical governance and evaluation
- Principled reasoning with multiple strategies (deductive, inductive, etc.)

## Features

- **Multi-Agent Orchestration**: Specialized agents work together for comprehensive response generation
- **Advanced Memory Management**: Dynamic salience, decay, archiving, and semantic search capabilities
- **Ethical Framework Integration**: Built-in ethical reasoning and assessment
- **Metacognitive Monitoring**: Self-awareness and performance optimization
- **Knowledge Graph Integration**: Structured knowledge extraction and storage
- **External System Integration**: A2A protocol and MCP support
- **Persistent Logging**: SQLite-based logging for introspection and fine-tuning
- **Flexible Embedding Support**: Multiple embedding providers and context-aware selection

## Installation

### Prerequisites

- Python 3.9 or higher
- git (optional, for cloning)

### Setup Steps

1. **Clone the Repository** (Optional)
```bash
git clone <repository-url>
cd ceaf_project
```

2. **Create Virtual Environment**
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install fastapi uvicorn python-dotenv litellm pydantic httpx google-cloud-sdk
pip install sentence-transformers numpy scikit-learn networkx
```

4. **Environment Configuration**

Create a `.env` file in the root directory:

```env
# API Keys (MANDATORY)
OPENROUTER_API_KEY="sk-YOUR_OPENROUTER_KEY"

# Core Paths
MBS_MEMORY_STORE_PATH="./data/mbs_memory_store_v2"
CEAF_PERSISTENT_LOG_DB_PATH="./data/ceaf_logs.sqlite"

# Logging & Observability
LOG_LEVEL="INFO"
CEAF_FINETUNING_LOG_ENABLED="true"
CEAF_FINETUNING_LOG_FILE="./data/ceaf_finetuning_log.jsonl"

# Default Models
ORA_DEFAULT_MODEL="openrouter/openai/gpt-4.1-turbo-preview"
NCIM_DEFAULT_MODEL="openrouter/google/gemini-pro"
VRE_DEFAULT_MODEL="openrouter/anthropic/claude-3-haiku-20240307"
MCL_DEFAULT_MODEL="openrouter/mistralai/mistral-large-latest"
KGS_AGENT_MODEL="openrouter/openai/gpt-3.5-turbo"

# Embedding Configuration
CEAF_EMBEDDING_PROVIDER="sentence_transformers"
CEAF_DEFAULT_EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Optional MCP External Tool Server Configuration
MCP_FS_SERVER_ENABLED="false"
MCP_MAPS_SERVER_ENABLED="false"
```

## Usage

### Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

**Health Check**: `GET /`
```json
{
  "status": "CEAF API is healthy and running!"
}
```

**Interact with CEAF**: `POST /interact`

Request:
```json
{
  "query": "Hello CEAF, what are your core principles?",
  "session_id": null
}
```

Response:
```json
{
  "session_id": "api-session-...",
  "agent_response": "Hello! As CEAF's Orchestrator/Responder Agent...",
  "tool_calls": null,
  "error": null
}
```

## Core Workflow

### Startup Process
1. Initialize ADK components (Runner, SessionService, MBSMemoryService)
2. Load persistent memories and build connection graph
3. Initialize self-model and foundational narratives
4. Prepare embedding cache and start lifecycle tasks

### User Interaction Flow
1. **NCF Generation**: ORA retrieves context frame with MCL guidance, memories, and identity context
2. **Query Processing**: ORA processes user query using structured multi-step protocol
3. **Tool Orchestration**: ORA calls specialized tools (VRE, NCIM, KGS, etc.) as needed
4. **Response Generation**: Final response after ethical review and self-correction
5. **Metacognitive Reflection**: MCL analyzes turn performance and provides guidance for next interaction
6. **Memory Updates**: Session data saved to long-term memory

### ORA's Multi-Step Protocol
1. Get Narrative Context Frame (NCF)
2. Consult Identity & Goals (NCIM)
3. Gather Information (Memory)
4. Formulate Draft Response
5. Ethical & Epistemic Review (VRE)
6. Refine Response & Self-Correction
7. Metacognitive Reflection (MCL)
8. Final Response to User
9. Memory & Goal Updates

## Project Structure

```
ceaf_project/
├── .env
├── main.py                     # FastAPI application entry point
├── ceaf_core/
│   ├── agents/                 # AI agent definitions
│   │   ├── ora_agent.py        # Orchestrator/Responder Agent
│   │   ├── mcl_agent.py        # Metacognitive Control Loop
│   │   ├── ncim_agent.py       # Narrative Coherence & Identity
│   │   ├── vre_agent.py        # Virtue & Reasoning Engine
│   │   ├── kgs_agent.py        # Knowledge Graph Synthesizer
│   │   └── aura_reflector_agent.py # Autonomous Reflective Analyzer
│   ├── callbacks/              # ADK callbacks for monitoring
│   │   ├── mcl_callbacks.py    # Metacognitive monitoring
│   │   └── safety_callbacks.py # Safety guardrails
│   ├── config/                 # Configuration management
│   │   └── model_configs.py    # Model configuration
│   ├── external_integrations/  # External system integration
│   │   ├── a2a/               # Agent-to-Agent protocol
│   │   └── mcp/               # Model Context Protocol
│   ├── modules/               # Core cognitive modules
│   │   ├── memory_blossom/    # Memory management system
│   │   ├── mcl_engine/        # Metacognitive control logic
│   │   ├── ncf_engine/        # Narrative context framing
│   │   ├── ncim_engine/       # Identity management
│   │   └── vre_engine/        # Virtue & reasoning logic
│   ├── services/              # System services
│   │   ├── mbs_memory_service.py # Memory service implementation
│   │   └── persistent_log_service.py # Logging service
│   ├── tools/                 # ADK tools for agent interaction
│   │   ├── ncf_tools.py       # Narrative context tools
│   │   ├── memory_tools.py    # Memory interaction tools
│   │   ├── mcl_tools.py       # Metacognitive tools
│   │   ├── ncim_tools.py      # Identity management tools
│   │   ├── vre_tools.py       # Virtue & reasoning tools
│   │   └── kgs_tools.py       # Knowledge graph tools
│   └── utils/                 # General utilities
│       ├── adk_helpers.py     # ADK integration helpers
│       └── embedding_utils.py # Embedding client utilities
└── README.md
```

## Key Concepts

**Edge of Coherence (EoC)**: The optimal operational state where CEAF maintains coherence while operating at appropriate levels of novelty and complexity.

**Narrative Context Frame (NCF)**: Dynamic interaction context that guides response generation with philosophical grounding, relevant memories, and operational parameters.

**Terapia para Silício**: The core philosophy emphasizing coherence, adaptive learning, ethical reasoning, and epistemic humility.

**Multi-Memory Architecture**: Comprehensive memory system supporting explicit facts, emotional contexts, procedural knowledge, goals, and structured knowledge graphs.

## External Integrations

**Agent-to-Agent (A2A) Protocol**: Enables communication with external specialized research agents.

**Model Context Protocol (MCP)**: Supports integration with external tools and services (filesystem, Google Maps, etc.).

**Embedding Flexibility**: Support for multiple embedding providers including Sentence Transformers and API-based models.

## Contributing

We welcome contributions to the CEAF project! Please see our CONTRIBUTING.md for guidelines on submitting issues, features, or pull requests.

## License

Only for Study, dont sell, dont make into a product. 

---

*CEAF represents an ambitious approach to creating more thoughtful, coherent, and ethically-aware AI systems through sophisticated cognitive architecture and multi-agent orchestration.*
