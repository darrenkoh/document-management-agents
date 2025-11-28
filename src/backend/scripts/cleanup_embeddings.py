#!/usr/bin/env python3
"""Clean up embeddings from TinyDB JSON storage since we now use vector store."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def cleanup_embeddings_from_json(json_path: str = "documents.json") -> bool:
    """Remove embedding data from TinyDB JSON file.

    Args:
        json_path: Path to the TinyDB JSON file

    Returns:
        True if successful
    """
    try:
        json_file = Path(json_path)
        if not json_file.exists():
            logger.info(f"JSON file {json_path} does not exist, nothing to clean up")
            return True

        # Read the current JSON data
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Clean up documents table
        if 'documents' in data and isinstance(data['documents'], dict):
            cleaned_count = 0
            for doc_id, doc_data in data['documents'].items():
                if isinstance(doc_data, dict):
                    # Remove embedding field if it exists
                    if 'embedding' in doc_data:
                        del doc_data['embedding']
                        cleaned_count += 1
                    # Ensure embedding_stored is set appropriately
                    if 'embedding_stored' not in doc_data:
                        doc_data['embedding_stored'] = False

            logger.info(f"Cleaned embeddings from {cleaned_count} documents")

        # Write back the cleaned data
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Successfully cleaned up embeddings from {json_path}")
        return True

    except Exception as e:
        logger.error(f"Error cleaning up embeddings: {e}")
        return False


def main():
    """Main cleanup function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting embedding cleanup from TinyDB JSON...")

    success = cleanup_embeddings_from_json()

    if success:
        logger.info("Cleanup completed successfully!")
    else:
        logger.error("Cleanup failed!")
        exit(1)


if __name__ == "__main__":
    main()
