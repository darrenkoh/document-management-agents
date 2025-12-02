# Document Management Agent

An intelligent document classification and search system that uses local AI models to automatically categorize documents, extract text content, and provide semantic search capabilities. Built with Python (FastAPI) backend and React frontend, featuring OCR support and vector-based semantic search.

## ✨ Features

- **🤖 AI-Powered Classification**: Automatically categorizes documents using local Ollama LLMs
- **🔍 Semantic Search**: Vector-based search with RAG (Retrieval-Augmented Generation) for precise results
- **📄 Multi-Format Support**: Handles PDFs, Word docs, text files, and images with OCR fallback
- **🖥️ Modern Web UI**: Clean React interface for browsing and searching documents
- **⚡ High Performance**: Batch processing, duplicate detection, and optimized embeddings
- **🔒 Local AI**: No cloud dependencies - runs entirely on your hardware
- **📊 Analytics Dashboard**: View statistics and processing metrics
- **🔄 Real-time Monitoring**: Watch mode for automatic processing of new files

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **Ollama** installed and running locally

### 1. Install Ollama Models

```bash
# Install required AI models
ollama pull deepseek-r1:8b        # Document classification
ollama pull qwen3-embedding:8b    # Text embeddings for search
ollama pull deepseek-ocr:3b       # OCR for image-based PDFs
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd document-management-agents

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd src/frontend
npm install
cd ../..
```

### 3. Configure (Optional)

Edit `src/backend/config/config.yaml` to customize:
- Source directories for document monitoring
- File extensions to include for processing (required)
- Database and vector store locations
- Ollama model settings
- Web server configuration

### 4. Running File Processing and Web App for Browsing

**File Processing**

```bash
# Classify documents in your input directory
python document_ingestion.py classify

# Monitor directory for new files
python document_ingestion.py watch
```

**Web Interface**
```bash
# Terminal 1: Start the backend API
python src/backend/api/app.py

# Terminal 2: Start the frontend
cd src/frontend && npm run dev
```

Then open http://localhost:5173 in your browser.


## 📁 Project Structure

```
├── src/
│   ├── backend/                 # Python FastAPI backend
│   │   ├── api/                 # REST API endpoints
│   │   ├── core/                # AI agents (classifier, RAG)
│   │   ├── database/            # SQLite database layer
│   │   ├── services/            # File handling, embeddings, OCR
│   │   ├── utils/               # Configuration and utilities
│   │   └── config/              # YAML configuration files
│   └── frontend/                # React TypeScript frontend
│       ├── src/
│       │   ├── components/      # Reusable UI components
│       │   ├── pages/          # Main application pages
│       │   ├── lib/            # API clients and utilities
│       │   └── types/          # TypeScript type definitions
│       └── dist/               # Built frontend assets
├── data/
│   ├── input/                  # Place your documents here
│   ├── databases/              # SQLite database files
│   ├── vector_store/           # ChromaDB vector embeddings
│   └── exports/                # JSON export of classifications
├── document_ingestion.py      # CLI entry point for document processing
└── requirements.txt            # Python dependencies
```

## 🛠️ Usage

### Adding Documents

Place your documents in the `data/input/` directory. Supported formats:
- PDF documents (text-based and image-based with OCR)
- Microsoft Word (.docx, .doc)
- Text files (.txt)
- Images (.png, .jpg, .jpeg, .gif, .tiff)

### CLI Commands

```bash
# Process all documents in input directory
python document_ingestion.py classify

# Continuous monitoring for new files
python document_ingestion.py watch

# Semantic search through documents
python document_ingestion.py search "travel booking confirmation"

# Find documents by category
python document_ingestion.py category invoice

# Enable verbose logging
python document_ingestion.py --verbose classify
```

### Web Interface

The web interface provides:

- **Dashboard**: Overview of processed documents and statistics
- **Document Browser**: View all classified documents with filtering
- **Search Interface**: Semantic search with AI-powered relevance ranking
- **Document Details**: View full content and metadata
- **Real-time Logs**: Monitor processing status

## ⚙️ Configuration

Key settings in `src/backend/config/config.yaml`:

```yaml
# Document source directories
source_paths:
  - "data/input"

# File extensions to process (REQUIRED)
# Only files with these extensions will be processed
# Empty list means NO files will be processed
file_extensions:
  - ".pdf"
  - ".docx"
  - ".doc"
  - ".txt"
  - ".png"
  - ".jpg"
  - ".jpeg"
  - ".gif"
  - ".tiff"

# Database settings
database:
  path: "data/databases/documents.db"

# AI model configuration
ollama:
  endpoint: "http://localhost:11434"
  model: "deepseek-r1:8b"
  embedding_model: "qwen3-embedding:8b"
  # OCR model: 'deepseek-ocr:3b' for Ollama or 'chandra' for vLLM
  ocr_model: "chandra"

# Chandra OCR configuration (when ocr_model is set to 'chandra')
chandra:
  endpoint: "http://localhost:11435"
  model: "chandra"

# Web server settings
webapp:
  port: 8081
  host: "0.0.0.0"
```

## 🔧 Development

### Backend Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
python src/backend/api/app.py --debug
```

### Frontend Development

```bash
cd src/frontend

# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Testing

```bash
# Backend tests
python -m pytest

# Frontend tests
cd src/frontend && npm test
```

## 🤖 How It Works

1. **Document Ingestion**: Files are processed in batches for optimal performance
2. **Text Extraction**: Content is extracted using format-specific parsers with OCR fallback
3. **Duplicate Detection**: Content-based hashing prevents reprocessing identical files
4. **AI Classification**: Local LLM analyzes content and assigns relevant categories
5. **Vector Embeddings**: Documents are converted to semantic vectors for search
6. **Storage**: Metadata and embeddings stored in SQLite + ChromaDB
7. **Search**: Semantic similarity search with optional RAG relevance filtering

## 📊 Performance Features

- **Batch Processing**: Handles multiple documents simultaneously
- **Content-Based Deduplication**: Skips files with identical content
- **Optimized Embeddings**: Efficient vector storage and retrieval
- **Caching**: Database lookups prevent redundant operations
- **Progress Tracking**: Real-time status updates and performance metrics

## 🔍 Supported Categories

The AI automatically detects categories including:
- invoice, receipt, contract, agreement
- confirmation, booking, ticket, itinerary
- report, memo, letter, email
- certificate, form, manual, presentation
- image, document, other

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Ollama connection failed**
- Ensure Ollama is running: `ollama serve`
- Check endpoint URL in config.yaml
- Verify required models are installed

**OCR not working**
- For Ollama OCR: Install deepseek-ocr model: `ollama pull deepseek-ocr:3b`
- For Chandra OCR: Install Chandra and start vLLM server on port 11435
- Check poppler-utils and tesseract are installed for PDF processing

**Chandra OCR Setup**
```bash
# Install Chandra OCR
pip install chandra-ocr

# Start Chandra vLLM server (runs on port 11435 by default)
chandra_vllm

# Or use custom configuration
VLLM_API_BASE=http://localhost:11435/v1 VLLM_MODEL_NAME=chandra chandra_vllm
```

Update `config.yaml` to use Chandra:
```yaml
ollama:
  ocr_model: "chandra"  # Instead of "deepseek-ocr:3b"

chandra:
  endpoint: "http://localhost:11435"
  model: "chandra"
```

**Frontend not loading**
- Ensure backend API is running on port 8081
- Check CORS settings if accessing from different domain

**Slow processing**
- Use batch processing for multiple files
- Consider GPU acceleration for Ollama if available
- Reduce model size for faster inference

### Getting Help

- Check the logs in `data/agent.log`
- Enable verbose mode: `python document_ingestion.py --verbose classify`
- Review configuration in `src/backend/config/config.yaml`

