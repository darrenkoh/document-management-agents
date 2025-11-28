# Running the Application After Refactoring

## Import Path Fixes

All import paths have been fixed to work correctly. The key changes:

1. **All backend files** now add the project root to `sys.path` before importing
2. **Project root detection** uses: `Path(__file__).parent.parent.parent.parent` (from files in `src/backend/`)
3. **Imports** use absolute paths: `from src.backend.utils.config import Config`

## Running the Application

### Prerequisites

1. Activate the virtual environment:
```bash
source venv/bin/activate
# or on Windows:
# venv\Scripts\activate
```

2. Install dependencies (if not already installed):
```bash
pip install -r requirements.txt
```

### Running the Flask API

From the project root:
```bash
python src/backend/api/app.py
```

Or with custom config:
```bash
python src/backend/api/app.py --config src/backend/config/config.yaml
```

### Running the CLI

From the project root:
```bash
python main.py classify
python main.py watch
python main.py search "your query"
```

### Running Scripts

From the project root:
```bash
python src/backend/scripts/add_embeddings.py
```

## How Import Paths Work

When you run `python src/backend/api/app.py`:

1. Python executes `app.py`
2. `app.py` calculates the project root: `Path(__file__).parent.parent.parent.parent`
   - `__file__` = `src/backend/api/app.py`
   - `.parent` = `src/backend/api/`
   - `.parent` = `src/backend/`
   - `.parent` = `src/`
   - `.parent` = project root ✓
3. Project root is added to `sys.path`
4. Imports like `from src.backend.utils.config import Config` work correctly

## Troubleshooting

### ModuleNotFoundError: No module named 'src'

- Make sure you're running from the project root
- Check that the file calculating the path is correct
- Verify `__init__.py` files exist in `src/` and `src/backend/`

### ModuleNotFoundError: No module named 'flask' (or other dependencies)

- Activate your virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Config file not found

- Make sure config paths are relative to project root
- Default config path: `src/backend/config/config.yaml`
- You can override with `--config` flag

