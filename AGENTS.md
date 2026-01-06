# Build & Test

- **Install Backend**: `pip install -r requirements.txt`
- **Install Frontend**: `cd src/frontend && npm install`
- **Build Frontend**: `cd src/frontend && npm run build`
- **Run Ingestion**: `python document_ingestion.py classify`
- **Run Watch Mode**: `python document_ingestion.py watch`
- **Start API**: `python src/backend/api/app.py`
- **Start Frontend**: `cd src/frontend && npm run dev`

# Architecture Overview

This project is a Local AI-powered document management system. It uses specialized agents to automate ingestion, classification, and search.

- **Frontend**: React + Vite + Tailwind CSS (`src/frontend/`)
- **Backend API**: Flask (`src/backend/api/app.py`)
- **Orchestration**: `DocumentAgent` (`src/backend/core/agent.py`) coordinates the ingestion pipeline.
- **Intelligence**:
  - `Classifier`: Categorizes documents via local LLM.
  - `RAGAgent`: Evaluates relevance and generates cited answers.
  - `Sam3ReceiptSegmenter`: Segments multi-receipt images using SAM3.
- **Storage**:
  - Metadata: SQLite (`data/databases/documents.db`)
  - Vectors: ChromaDB (`data/vector_store/`)

# Security

- **Local First**: All LLM interactions use local endpoints (default: `localhost:11434` for Ollama).
- **No Cloud**: Sensitive documents never leave the local environment.
- **Configuration**: API endpoints and model names are managed in `src/backend/config/config.yaml`.

# Git Workflows

- **Branching**: Branch from `main` for new features or bugfixes.
- **Commits**: Use descriptive, atomic commits.
- **Review**: Ensure linter passes before submitting PRs.

# Conventions & Patterns

- **Configuration**: Always refer to `src/backend/config/config.yaml` for environment-specific settings.
- **Data Directory**: All persistent state (logs, DBs, vectors, segmented images) lives in `data/`.
- **Hybrid Search**: The system combines BM25 keyword search with vector semantic search for optimal retrieval.
- **Content Hashing**: Documents are deduplicated using SHA256 hashes of their content.
- **Component Style**: Frontend uses functional components with Tailwind for styling and Lucide for iconography.
- **Error Handling**: Failed ingestion steps are recorded in the `failed_files` table in SQLite.
