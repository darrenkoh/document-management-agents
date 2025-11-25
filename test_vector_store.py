#!/usr/bin/env python3
"""Test vector store functionality without yaml dependency."""
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Mock config class
class MockConfig:
    def __init__(self):
        self.vector_store_type = "chromadb"
        self.vector_store_directory = "vector_store"
        self.vector_store_collection = "documents"
        self.embedding_dimension = 768

def test_vector_store():
    """Test basic vector store functionality."""
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))

        from vector_store import create_vector_store

        config = MockConfig()
        print(f"Testing vector store: {config.vector_store_type}")
        print(f"Directory: {config.vector_store_directory}")
        print(f"Collection: {config.vector_store_collection}")

        # Create vector store
        print("Creating vector store...")
        vs = create_vector_store(
            store_type=config.vector_store_type,
            persist_directory=config.vector_store_directory,
            collection_name=config.vector_store_collection,
            dimension=config.embedding_dimension
        )

        # Check current count
        count = vs.count()
        print(f"Current vector store count: {count}")

        # Add a test embedding
        print("Adding test embedding...")
        test_embedding = [0.1] * config.embedding_dimension
        test_metadata = {
            'filename': 'test.pdf',
            'file_path': '/test/test.pdf',
            'categories': 'test',
            'content_preview': 'test content'
        }

        success = vs.add_embeddings(
            embeddings=[test_embedding],
            metadata=[test_metadata],
            ids=['test_123']
        )

        print(f"Add embedding success: {success}")

        # Check count again
        count_after = vs.count()
        print(f"Vector store count after adding: {count_after}")

        # Test search
        print("Testing search...")
        search_results = vs.search_similar(test_embedding, top_k=5, threshold=-1.0)
        print(f"Search results: {len(search_results)}")
        for doc_id, similarity, metadata in search_results:
            print(f"  ID: {doc_id}, Similarity: {similarity:.3f}, File: {metadata.get('filename', 'Unknown')}")

        # Test search with different embedding
        print("Testing search with different embedding...")
        different_embedding = [0.2] * config.embedding_dimension
        search_results2 = vs.search_similar(different_embedding, top_k=5, threshold=-1.0)
        print(f"Different embedding search results: {len(search_results2)}")

        vs.close()
        print("Test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vector_store()
