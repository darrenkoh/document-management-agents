#!/usr/bin/env python3
"""Debug script to check vector store content."""
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from vector_store import create_vector_store

def debug_vector_store():
    """Debug vector store content."""
    try:
        config = Config()
        print(f"Vector store type: {config.vector_store_type}")
        print(f"Vector store directory: {config.vector_store_directory}")
        print(f"Collection name: {config.vector_store_collection}")

        # Initialize vector store
        vector_store = create_vector_store(
            store_type=config.vector_store_type,
            persist_directory=config.vector_store_directory,
            collection_name=config.vector_store_collection,
            dimension=config.embedding_dimension
        )

        # Check count
        count = vector_store.count()
        print(f"Vector store count: {count}")

        if count > 0:
            print("Vector store has embeddings!")

            # Try a simple search
            test_embedding = [0.1] * config.embedding_dimension  # Dummy embedding
            results = vector_store.search_similar(test_embedding, top_k=5)
            print(f"Search results: {len(results)}")
            for doc_id, similarity, metadata in results:
                print(f"  ID: {doc_id}, Similarity: {similarity:.3f}, File: {metadata.get('filename', 'Unknown')}")
        else:
            print("Vector store is empty!")

        vector_store.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_vector_store()
