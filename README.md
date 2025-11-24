# Document Classification Agent

An intelligent Python agent that automatically classifies documents by content using a local LLM (Ollama), stores classifications in a NoSQL database, and provides semantic search (RAG) capabilities.

## Features

- **Content-based Classification**: Uses Ollama LLM to analyze file content and classify documents (supports up to 3 categories per document)
- **Multiple File Formats**: Supports PDF, DOCX, TXT, and images (with OCR)
- **NoSQL Database Storage**: Stores all classifications in TinyDB (JSON-based NoSQL database)
- **JSON Export**: Automatically exports classification results to JSON file
- **Semantic Search (RAG)**: Perform semantic search on document content using embeddings
- **Embedding Generation**: Automatically generates embeddings for all documents using Ollama
- **Web Interface**: Beautiful web UI for browsing documents, semantic search, and viewing results
- **Watch Mode**: Continuously monitors a directory for new files
- **Batch Processing**: Process all files in a directory at once
- **Configurable**: Easy-to-use YAML configuration file

## Requirements

- Python 3.8+
- Ollama installed and running locally
- Classification model (e.g., `deepseek-r1:8b`, `llama3.2`)
- Embedding model (e.g., `nomic-embed-text`)

## Installation

1. Clone or download this repository

2. Install Python dependencies:

**Option A: Using setup script (recommended with uv):**
```bash
./setup.sh
```

**Option B: Manual installation:**
```bash
pip install -r requirements.txt
```

**Note:** The setup script uses [uv](https://github.com/astral-sh/uv) for faster package installation. Install uv first:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install Tesseract OCR (for image text extraction):
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

4. Install and start Ollama:
   - Visit [ollama.ai](https://ollama.ai) for installation instructions
   - Pull a classification model: `ollama pull deepseek-r1:8b` (or `llama3.2`)
   - Pull an embedding model: `ollama pull nomic-embed-text`

5. Configure the agent by editing `config.yaml`:
```yaml
source_path: "./input"          # Where to read files from
database:
  path: "documents.json"        # NoSQL database file
  json_export_path: "classifications.json"  # JSON export file
ollama:
  endpoint: "http://localhost:11434"
  model: "deepseek-r1:8b"       # Classification model
  embedding_model: "nomic-embed-text"  # Embedding model
```

## Usage

### Classify Files Once

Process all files in the source directory:

```bash
python main.py classify
```

This will:
- Extract text from files
- Classify each document (up to 3 categories)
- Generate embeddings for semantic search
- Store everything in the NoSQL database
- Export results to JSON file

### Watch Mode

Continuously monitor the source directory for new files:

```bash
python main.py watch
```

Press `Ctrl+C` to stop watch mode.

### Semantic Search

Search documents by content using natural language:

```bash
python main.py search "flight confirmation"
python main.py search "invoice from 2024" --top-k 5
```

### Search by Category

Find all documents in a specific category:

```bash
python main.py category invoice
python main.py category booking-confirmation
```

### Command Line Options

```bash
# Use a custom config file
python main.py --config custom_config.yaml classify

# Enable verbose logging (shows LLM requests and responses)
python main.py --verbose classify

# Override source directory
python main.py classify --source /path/to/files

# Semantic search with custom result count
python main.py search "your query" --top-k 20
```

## Web Interface

Launch the web application for a beautiful interface to browse and search your documents:

```bash
# Basic usage (uses config settings)
python app.py

# Override port and host
python app.py --port 8080 --host 127.0.0.1

# Use custom config and disable debug mode
python app.py --config custom_config.yaml --no-debug

# Command line options override config file settings
python app.py --port 3000 --debug
```

The web app will be available at the configured host and port (default: `http://localhost:5000`) with the following features:

### Home Page
- **Semantic Search Bar**: Search documents using natural language (e.g., "travel documents", "financial statements")
- **Recent Documents**: View the 10 most recently classified documents
- **Quick Stats**: Overview of total documents and categories

### Document Browser
- **Paginated List**: Browse all documents with pagination (20 per page)
- **Search & Filter**: Search documents by filename, categories, or content
- **File Information**: View file size, type, classification date, and categories
- **Category Badges**: Visual category tags for easy identification

### Semantic Search Results
- **Relevance Scoring**: Results ranked by similarity percentage
- **File Location**: Direct path to each document
- **Content Preview**: Excerpt of document content
- **Quick Actions**: Copy file path or view detailed information

### Document Details
- **Full Content**: Complete extracted text content
- **File Metadata**: Size, type, modification date, classification date
- **Category Information**: All assigned categories
- **Embedding Status**: Whether AI embeddings are available for search

### Statistics Dashboard
- **Category Distribution**: Charts and counts of document categories
- **File Type Analysis**: Breakdown by file extensions
- **Processing Stats**: Total documents, embedding status, etc.

### API Endpoints
The web app also provides REST API endpoints:
- `GET /api/documents`: Paginated document listing
- `POST /api/search`: Semantic search via API

## Configuration

Edit `config.yaml` to customize:

- **source_path**: Directory to monitor for files
- **database**: Database settings
  - **path**: Path to TinyDB JSON database file
  - **json_export_path**: Path for JSON export file
- **ollama**: Ollama API settings
  - **endpoint**: Ollama API endpoint URL
  - **model**: Model name for classification
  - **embedding_model**: Model name for embeddings (default: nomic-embed-text)
  - **timeout**: API timeout in seconds
  - **num_predict**: Maximum tokens to predict (higher for reasoning models)
- **file_extensions**: List of file types to process (empty = all files)
- **categories**: Optional predefined categories (empty = auto-detect)
- **prompt_template**: Optional custom classification prompt template
  - Use `{filename}` and `{content}` as placeholders
  - Should ask for up to 3 categories separated by commas
- **watch**: Watch mode settings (interval, recursive)
- **webapp**: Web application settings
  - **port**: Port to run the web app on (default: 5000)
  - **host**: Host to bind to (default: "0.0.0.0")
  - **debug**: Enable debug mode (default: true)
- **logging**: Log level and file path

## How It Works

1. **Scan**: The agent scans the source directory for files
2. **Extract**: Text content is extracted from files (PDF, DOCX, TXT, or OCR for images)
3. **Classify**: Content is sent to Ollama LLM for classification (supports up to 3 categories per document)
4. **Embed**: Embedding vector is generated for semantic search
5. **Store**: Classification, content, and embedding are stored in NoSQL database
6. **Export**: Results are exported to JSON file
7. **Search**: Use semantic search to find documents by content similarity

## Database Schema

Each document in the database contains:

```json
{
  "file_path": "/path/to/document.pdf",
  "filename": "document.pdf",
  "content": "Full extracted text content...",
  "content_preview": "First 500 characters...",
  "categories": "invoice-receipt-financial",
  "classification_date": "2025-11-24T10:30:00",
  "metadata": {
    "file_size": 12345,
    "file_extension": ".pdf",
    "file_modified": 1234567890
  },
  "embedding": [0.123, -0.456, ...]  // Vector for semantic search
}
```

## JSON Export Format

The JSON export file contains:

```json
{
  "export_date": "2025-11-24T10:30:00",
  "total_documents": 10,
  "documents": [
    {
      "file_path": "/path/to/doc.pdf",
      "filename": "doc.pdf",
      "content": "...",
      "categories": "invoice",
      "classification_date": "...",
      "metadata": {...}
    }
  ]
}
```

## Classification Categories

The agent automatically detects categories such as:
- invoice
- contract
- receipt
- letter
- report
- resume
- certificate
- form
- statement
- manual
- article
- email
- memo
- note
- presentation
- spreadsheet
- confirmation
- booking
- ticket
- itinerary
- image
- other

Documents can have up to 3 categories, which are sorted and joined with "-" (e.g., "booking-confirmation-travel").

## Example Workflow

```bash
# 1. Classify documents
python main.py classify --source ./documents

# 2. Launch the web interface (custom port)
python app.py --port 8080

# 3. Or use command-line search
python main.py search "flight booking confirmation"
python main.py category invoice

# 4. Check the JSON export
cat classifications.json
```

Open your browser to the configured URL (default: `http://localhost:5000`) to access the web interface with semantic search, document browsing, and statistics.

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check the endpoint in `config.yaml` matches your Ollama setup
- Verify the models are installed: `ollama list`

### Embedding Model Not Found
- Pull the embedding model: `ollama pull nomic-embed-text`
- Check the model name in `config.yaml` matches an installed model

### OCR Not Working
- Ensure Tesseract is installed and in your PATH
- For images, ensure they contain readable text

### Database Errors
- Check file permissions on the database file
- Ensure the database directory is writable
- Check logs for specific error messages

### All Files Classified as "uncategorized"
- Enable verbose logging: `python main.py --verbose classify`
- Check the LLM response in the logs
- For reasoning models (like deepseek-r1), increase `num_predict` in `config.yaml`
- If `done_reason: 'length'` appears, the model hit the token limit - increase `num_predict`
- Try customizing the `prompt_template` in `config.yaml`

### Semantic Search Not Working
- Ensure embeddings are being generated (check logs)
- Verify the embedding model is installed and accessible
- Check that documents have embeddings stored in the database

### Web App Not Starting
- Ensure Flask is installed: `pip install flask`
- Check that the database file exists and is readable
- Verify no other service is using the configured port
- Check the configured host and port in `config.yaml` under `webapp` section
- Run with debug mode: `python app.py --debug` (includes error details)

### Web App Search Not Working
- Ensure the database contains documents with embeddings
- Check that the embedding model is properly configured
- Verify the web app can connect to the database

## License

This project is open source and available for use and modification.
