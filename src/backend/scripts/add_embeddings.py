#!/usr/bin/env python3
"""Add embeddings to existing documents that don't have them."""
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def add_embeddings_to_existing_docs():
    """Add embeddings to existing documents in the database."""
    try:
        from src.backend.core.agent import DocumentAgent
        from src.backend.utils.config import Config

        # Load configuration
        config = Config()

        # Initialize agent
        agent = DocumentAgent(config, verbose=True)

        # Get all documents from database
        documents = agent.database.get_all_documents()
        print(f"Found {len(documents)} documents in database")

        # Process each document to add embeddings (ChromaDB handles duplicates)
        processed = 0
        for doc in documents:
            try:
                file_path = Path(doc['file_path'])
                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue

                # Extract text content (similar to process_file but just for embedding)
                content = agent.file_handler.extract_text(file_path)
                if not content:
                    print(f"Warning: Could not extract content from {file_path} (no content)")
                    continue

                print(f"Extracted content length: {len(content)} for {file_path.name}")

                # Generate embedding
                embedding = agent.embedding_generator.generate_document_embedding(content)
                if not embedding:
                    print(f"Warning: Could not generate embedding for {file_path} (embedding generation failed)")
                    continue

                print(f"Generated embedding with {len(embedding)} dimensions for {file_path.name}")

                # Store embedding
                success = agent.database.store_embedding(str(file_path), embedding)
                if success:
                    processed += 1
                    print(f"✅ Successfully added embedding for: {file_path.name}")
                else:
                    print(f"❌ Failed to store embedding for: {file_path.name}")

            except Exception as e:
                print(f"Error processing {doc.get('filename', 'unknown')}: {e}")
                continue

        print(f"Successfully added embeddings to {processed} documents")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    add_embeddings_to_existing_docs()
