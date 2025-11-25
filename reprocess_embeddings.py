#!/usr/bin/env python3
"""Re-process documents to ensure they have embeddings in the vector store."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def reprocess_documents():
    """Re-process existing documents to ensure embeddings are in vector store."""
    try:
        # Simple approach: just run the process command
        import subprocess
        result = subprocess.run([
            sys.executable, 'main.py', 'process'
        ], cwd=str(Path(__file__).parent), capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        print(f"Return code: {result.returncode}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    reprocess_documents()
