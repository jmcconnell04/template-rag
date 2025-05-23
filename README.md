# RAG Workspace Template

This repository serves as a template for creating configurable, containerized Retrieval Augmented Generation (RAG) workspaces. It's designed to be local-first, with a straightforward path for Proof-of-Concept (PoC) or demonstration deployments on Azure App Service.

The goal is to provide a flexible foundation for building RAG applications that can connect to various LLMs, utilize different vector stores (starting with ChromaDB and SQLite locally), and be organized by "Projects" within each workspace instance.

_(Note: The current development repository is `git@github.com:jmcconnell04/template-rag.git`. This will be updated once the official GitHub organization is established.)_

## Project Structure

Below is the directory structure for this RAG workspace template. Each key file and folder's purpose is described to help you understand the layout and where to find or place different components.

```text
template-rag/
│
├── .gitignore                     # Specifies intentionally untracked files Git should ignore.
├── docker-compose.yml             # Main Docker Compose file for defining and running services (CPU default).
├── docker-compose.gpu.yml         # Optional Docker Compose override file for enabling GPU acceleration (primarily for Ollama).
├── example.env                    # Template for environment variables. Copy to '.env' and customize.
├── README.md                      # This file: provides an overview and documentation for the template.
│
├── front_end/                     # Contains configurations or notes related to the web UI.
│   └── openwebui/
│       └── README.md              # Specific notes for OpenWebUI integration, setup, and customization.
│
├── prompts/                       # Directory for customizable prompt templates.
│   └── default_rag_prompt.md      # Example RAG prompt template (Markdown file).
│
├── rag_files/                     # Host directories for persistent RAG data (bind-mounted into containers).
│   ├── data/                      # For ongoing/user-uploaded documents for RAG processing.
│   │   └── .gitkeep               # Ensures the directory is tracked by Git.
│   ├── metadata_db/               # For the SQLite database file (e.g., app.db will be created here).
│   │   └── .gitkeep
│   ├── seed/                      # Contains initial documents to automatically seed the RAG on first workspace setup.
│   │   └── example_document.pdf   # An example seed document.
│   └── vector_stores/             # Parent directory for vector store data.
│       └── chroma_data/           # For ChromaDB data files.
│           └── .gitkeep           # Ensures this host path for ChromaDB data is tracked.
│
├── scripts/                       # Utility scripts for setup and management.
│   ├── setup.sh                   # Setup helper script for Linux/macOS (e.g., clones repo, copies .env).
│   ├── setup.bat                  # Setup helper script for Windows.
│
├── server/                        # Contains the custom backend RAG API application (Python/FastAPI).
│   ├── Dockerfile                 # Instructions to build the Docker image for the 'server' application.
│   │                              # Includes COPY instructions for 'prompts/' and 'rag_files/seed/' for image builds.
│   ├── requirements.txt           # Python dependencies for the 'server' application.
│   └── app/                       # Source code for the 'server' application.
│       ├── main.py                # FastAPI application entry point, handles startup logic (DB init, default project, seeding).
│       ├── config.py              # Loads configuration from environment variables (via .env).
│       ├── core/                  # Core logic: LLM services, vector store interaction (ChromaDB adapter), RAG pipeline.
│       ├── models/                # Pydantic models for API requests/responses, data structures.
│       ├── parsers/               # Document parsing and chunking logic.
│       ├── routers/               # API endpoints (e.g., for chat, document upload, project management).
│       └── services/              # Business logic services (e.g., project_service, chat_service, audit_service).
│   └── tests/                     # (Recommended) Unit and integration tests for the 'server' app.
│
└── n8n/                           # (Placeholder for v1.5+) Contains n8n workflow definitions for automation.
└── README.md                  # Explains future intent for n8n integration.
```

## Key File and Folder Explanations

- **`.env` (created from `example.env`):** **Crucial.** This (untracked) file will contain all your workspace-specific configurations, including API keys, `WORKSPACE_NAME`, port settings, and feature flags like `AUTO_SEED_ON_STARTUP`.
- **`docker-compose.yml`:** Defines the services (OpenWebUI, your `server`, Ollama, etc.), their configurations, networks, and volume mappings for a CPU-based local environment.
- **`docker-compose.gpu.yml`:** An optional override file to enable GPU support, primarily for the Ollama service.
- **`server/`:** The heart of your custom RAG application.
  - `server/Dockerfile`: Builds the image for your backend. For Azure deployments, this will copy in `prompts/` and `rag_files/seed/`.
  - `server/app/main.py`: Entry point for the backend; will include startup logic like initializing the SQLite DB, creating a default project (named after `WORKSPACE_NAME`), and triggering auto-seeding if enabled.
  - `server/app/config.py`: Manages loading settings from the `.env` file.
- **`prompts/`:** Store your `.md` prompt templates here. Your `server` will load these to construct prompts for the LLMs.
- **`rag_files/`:** These subdirectories are bind-mounted from your host into the `server` container for persistent data:
  - `seed/`: Place documents here that you want to be automatically ingested into the RAG for new workspaces (if `AUTO_SEED_ON_STARTUP=true`).
  - `data/`: For documents uploaded during runtime or added manually for ongoing RAG processing.
  - `metadata_db/`: Your SQLite database file (`app.db`) will be stored here by the `server`.
  - `vector_stores/chroma_data/`: Your ChromaDB vector store files will be persisted here by the `server`.
- **`scripts/`:** For helper scripts like `setup.sh`/`.bat` which will streamline the initial cloning of this template and setting up the `.env` file for a new workspace instance.

---
