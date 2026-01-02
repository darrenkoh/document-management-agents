#!/usr/bin/env python3
"""Add embeddings to existing documents that don't have them."""
import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def add_embeddings_to_existing_docs(config_path=None, force=False):
    """Add embeddings to existing documents in the database.
    
    Args:
        config_path: Path to configuration file
        force: If True, regenerate embeddings even if they already exist
    """
    try:
        from src.backend.core.agent import DocumentAgent
        from src.backend.utils.config import Config

        # Load configuration
        if config_path:
            config = Config(config_path)
        else:
            config = Config('src/backend/config/config.yaml')

        # Initialize agent
        agent = DocumentAgent(config, verbose=True)

        # Get all documents from database
        documents = agent.database.get_all_documents()
        print(f"Found {len(documents)} documents in database")
        
        if force:
            print("Force mode enabled - regenerating ALL embeddings")

        # Process each document to add embeddings (ChromaDB handles duplicates)
        processed = 0
        skipped = 0
        for doc in documents:
            try:
                file_path = Path(doc['file_path'])
                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    skipped += 1
                    continue

                # Check if embedding already exists (skip unless force mode)
                if doc.get('embedding_stored', False) and not force:
                    print(f"Skipping {file_path.name} - embedding already exists (use --force to regenerate)")
                    skipped += 1
                    continue

                # Extract text content (similar to process_file but just for embedding)
                content, _, _ = agent.file_handler.extract_text(file_path)
                if not content:
                    print(f"Warning: Could not extract content from {file_path} (no content)")
                    continue

                print(f"Extracted content length: {len(content)} for {file_path.name}")

                # Generate embeddings (chunks + optional summary) and store them
                embedding_result = agent.embedding_generator.generate_document_embeddings(
                    content,
                    chunk_size=config.chunk_size,
                    overlap=config.chunk_overlap,
                    generate_summary=config.enable_summary_embedding
                )

                if not embedding_result.get('chunks') and not embedding_result.get('summary'):
                    print(f"Warning: Could not generate any embeddings for {file_path} (embedding generation failed)")
                    continue

                chunk_count = len(embedding_result.get('chunks', []))
                summary_dim = len(embedding_result['summary']) if embedding_result.get('summary') else None
                chunk_dim = len(embedding_result['chunks'][0]) if chunk_count else None
                print(f"Generated embeddings for {file_path.name}: chunks={chunk_count} (dim={chunk_dim}), summary_dim={summary_dim}")

                success = agent.database.store_document_embeddings(
                    str(file_path),
                    embedding_result.get('chunks', []),
                    embedding_result.get('summary')
                )
                if success:
                    processed += 1
                    print(f"✅ Successfully added embedding for: {file_path.name}")
                else:
                    print(f"❌ Failed to store embedding for: {file_path.name}")

            except Exception as e:
                print(f"Error processing {doc.get('filename', 'unknown')}: {e}")
                continue

        print(f"\nSummary:")
        print(f"  Successfully added embeddings: {processed}")
        print(f"  Skipped: {skipped}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add embeddings to documents')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force regeneration of ALL embeddings, even if they already exist. '
                             'Use this after resetting the vector store.')

    args = parser.parse_args()

    add_embeddings_to_existing_docs(args.config, force=args.force)
