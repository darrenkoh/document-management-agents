# Document Classification Agent

An intelligent Python agent that automatically classifies documents by content using a local LLM (Ollama) and organizes them into category-specific folders.

## Features

- **Content-based Classification**: Uses Ollama LLM to analyze file content and classify documents
- **Multiple File Formats**: Supports PDF, DOCX, TXT, and images (with OCR)
- **Automatic Organization**: Copies files to folders named after their classification
- **Watch Mode**: Continuously monitors a directory for new files
- **Batch Processing**: Process all files in a directory at once
- **Configurable**: Easy-to-use YAML configuration file

## Requirements

- Python 3.8+
- Ollama installed and running locally
- An Ollama model (default: `llama3.2`)

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
   - Pull a model: `ollama pull llama3.2`

5. Configure the agent by editing `config.yaml`:
```yaml
source_path: "./input"          # Where to read files from
destination_path: "./output"     # Where to copy classified files
ollama:
  endpoint: "http://localhost:11434"
  model: "llama3.2"
```

## Usage

### Classify Files Once

Process all files in the source directory:

```bash
python main.py classify
```

### Watch Mode

Continuously monitor the source directory for new files:

```bash
python main.py watch
```

Press `Ctrl+C` to stop watch mode.

### Command Line Options

```bash
# Use a custom config file
python main.py --config custom_config.yaml classify

# Enable verbose logging (shows LLM requests and responses)
python main.py --verbose classify

# Override source directory
python main.py classify --source /path/to/files

# Override output directory
python main.py classify --output /path/to/output

# Override both source and output
python main.py classify --source /path/to/files --output /path/to/output

# Watch with custom interval
python main.py watch --interval 10

# Watch with custom source and output
python main.py watch --source /path/to/files --output /path/to/output

# Combine verbose with other options
python main.py --verbose classify --source /path/to/files --output /path/to/output
```

## Configuration

Edit `config.yaml` to customize:

- **source_path**: Directory to monitor for files
- **destination_path**: Base directory for classified files
- **ollama**: Ollama API settings (endpoint, model, timeout, num_predict)
  - **endpoint**: Ollama API endpoint URL
  - **model**: Model name to use for classification
  - **timeout**: API timeout in seconds
  - **num_predict**: Maximum number of tokens to predict (higher for reasoning models like deepseek-r1)
- **file_extensions**: List of file types to process (empty = all files)
- **watch**: Watch mode settings (interval, recursive)
- **logging**: Log level and file path

## How It Works

1. **Scan**: The agent scans the source directory for files
2. **Extract**: Text content is extracted from files (PDF, DOCX, TXT, or OCR for images)
3. **Classify**: Content is sent to Ollama LLM for classification
4. **Organize**: Files are copied to `destination/<category>/filename`

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
- image
- other

## Example

```
Input directory structure:
input/
  document1.pdf
  invoice_2024.pdf
  contract.docx

After classification:
output/
  report/
    document1.pdf
  invoice/
    invoice_2024.pdf
  contract/
    contract.docx

Note: Original files remain in the input directory (files are copied, not moved)
```

## Troubleshooting

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check the endpoint in `config.yaml` matches your Ollama setup
- Verify the model is installed: `ollama list`

### OCR Not Working
- Ensure Tesseract is installed and in your PATH
- For images, ensure they contain readable text

### Files Not Copying
- Check file permissions on source and destination directories
- Verify the source path exists
- Check logs for error messages

### All Files Classified as "uncategorized"
- Enable verbose logging to see LLM requests and responses: `python main.py --verbose classify`
- Check the LLM response in the logs to see what the model is returning
- Verify Ollama is running and the model is loaded: `ollama list`
- For reasoning models (like deepseek-r1), increase `num_predict` in `config.yaml` (default: 200, try 300-500)
- If `done_reason: 'length'` appears in logs, the model hit the token limit - increase `num_predict`
- Try a different model in `config.yaml` if the current one isn't working well
- Check that file content is being extracted properly (verbose logs will show this)

## License

This project is open source and available for use and modification.

