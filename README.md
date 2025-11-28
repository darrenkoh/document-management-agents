# Document Management Agents

An intelligent Python agent that automatically classifies documents by content using a local LLM (Ollama), with advanced OCR capabilities for image-based PDFs via DeepSeek-OCR, stores classifications in a SQLite database, and provides semantic search (RAG) capabilities with a modern React frontend.

## Project Structure

```
/
├── src/
│   ├── backend/              # Python backend code
│   │   ├── api/              # Flask API application
│   │   ├── core/              # Core business logic (agent, classifier, RAG)
│   │   ├── database/          # Database modules
│   │   ├── services/          # Service modules (file handling, embeddings, vector store)
│   │   ├── scripts/           # Utility scripts (migrations, etc.)
│   │   ├── utils/             # Utility modules (config)
│   │   └── config/            # Configuration files
│   └── frontend/              # React frontend application
│       ├── src/               # React source files
│       ├── dist/              # Build output
│       └── package.json       # Frontend dependencies
├── tests/                     # Test files
├── data/                      # Data files
│   ├── databases/             # SQLite databases
│   ├── exports/               # JSON exports
│   ├── vector_store/          # Vector store data
│   └── input/                 # Input documents
├── docs/                      # Documentation
├── scripts/                   # Build and setup scripts
├── main.py                    # Main CLI entry point
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
./scripts/setup.sh

# Or manually:
pip install -r requirements.txt

# Install frontend dependencies
cd src/frontend
npm install
```

### 2. Configure the Agent

Edit `src/backend/config/config.yaml` to configure:
- Source directories (`source_paths`)
- Database paths
- Ollama endpoint and models
- Vector store settings

### 3. Run the Backend

```bash
# CLI mode
python main.py classify

# Web API mode
python src/backend/api/app.py
```

### 4. Run the Frontend

```bash
cd src/frontend
npm run dev
```

## Usage

### CLI Commands

```bash
# Classify files once
python main.py classify

# Watch for new files
python main.py watch

# Semantic search
python main.py search "your query"

# Search by category
python main.py category invoice
```

### Web Interface

```bash
# Start backend API (port 8081)
python src/backend/api/app.py

# Start frontend dev server (port 5173)
cd src/frontend && npm run dev
```

## Configuration

Configuration files are located in `src/backend/config/`. The main configuration file is `config.yaml`.

Key configuration options:
- `source_paths`: Directories to monitor for documents
- `database.path`: SQLite database location (default: `data/databases/documents.db`)
- `database.json_export_path`: JSON export location (default: `data/exports/classifications.json`)
- `ollama.endpoint`: Ollama API endpoint
- `ollama.model`: Classification model name
- `ollama.embedding_model`: Embedding model name

## Development

### Backend Development

Backend code is organized in `src/backend/`:
- `api/`: Flask API routes
- `core/`: Business logic (agent, classifier, RAG)
- `database/`: Database implementations
- `services/`: Supporting services
- `scripts/`: Utility scripts

### Frontend Development

Frontend code is in `src/frontend/`:
- `src/`: React components and pages
- `dist/`: Production build output

### Running Tests

```bash
# Run tests from project root
python -m pytest tests/
```

## Documentation

- [Main Documentation](docs/README.md)
- [React Frontend Guide](docs/README_REACT_FRONTEND.md)
- [Vector Database Guide](docs/README_VECTOR_DB.md)

## Requirements

- Python 3.8+
- Node.js 18+
- Ollama installed and running
- Required Ollama models:
  - Classification: `deepseek-r1:8b` (or similar)
  - Embeddings: `qwen3-embedding:8b`
  - OCR: `deepseek-ocr:3b`

## License

This project is open source and available for use and modification.

