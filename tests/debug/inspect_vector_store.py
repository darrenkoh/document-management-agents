#!/usr/bin/env python3
"""Tool to inspect and debug vector store contents."""
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def inspect_vector_store():
    """Inspect the contents of the vector store."""
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))

        from config import Config
        from vector_store import create_vector_store

        # Load config
        config = Config()
        print("=== Vector Store Inspector ===")
        print(f"Type: {config.vector_store_type}")
        print(f"Directory: {config.vector_store_directory}")
        print(f"Collection: {config.vector_store_collection}")
        print(f"Dimension: {config.embedding_dimension}")
        print()

        # Create vector store instance
        vs = create_vector_store(
            store_type=config.vector_store_type,
            persist_directory=config.vector_store_directory,
            collection_name=config.vector_store_collection,
            dimension=config.embedding_dimension
        )

        # Get basic stats
        count = vs.count()
        print(f"📊 Total embeddings stored: {count}")

        if count == 0:
            print("❌ Vector store is empty!")
            print("💡 This means no documents have been processed with embeddings yet.")
            vs.close()
            return

        print(f"✅ Vector store contains {count} embeddings")
        print()

        # Try to get all documents (this might not work for all vector stores)
        print("🔍 Attempting to retrieve stored data...")

        if hasattr(vs, 'collection') and vs.collection:  # ChromaDB specific
            try:
                # Get all data from ChromaDB
                all_data = vs.collection.get(include=['metadatas', 'embeddings'])
                print(f"📋 Found {len(all_data['ids'])} documents in collection:")

                for i, (doc_id, metadata) in enumerate(zip(all_data['ids'], all_data['metadatas'])):
                    print(f"  {i+1}. ID: {doc_id}")
                    print(f"     File: {metadata.get('filename', 'Unknown')}")
                    print(f"     Path: {metadata.get('file_path', 'Unknown')}")
                    print(f"     Categories: {metadata.get('categories', 'Unknown')}")
                    print(f"     Preview: {metadata.get('content_preview', 'Unknown')[:100]}...")
                    print()

            except Exception as e:
                print(f"❌ Error retrieving collection data: {e}")

        # Test search functionality
        print("🧪 Testing search functionality...")

        # Create a dummy query embedding
        import numpy as np
        dummy_embedding = np.random.rand(config.embedding_dimension).tolist()

        print(f"🔎 Searching with random {config.embedding_dimension}-dimensional embedding...")

        results = vs.search_similar(dummy_embedding, top_k=10, threshold=-1.0)

        print(f"📈 Search returned {len(results)} results:")
        for i, (doc_id, similarity, metadata) in enumerate(results):
            print(f"  {i+1}. ID: {doc_id}")
            print(".3f")
            print(f"     File: {metadata.get('filename', 'Unknown')}")
            print()

        if len(results) == 0:
            print("⚠️  Search returned no results - this indicates a problem!")
        else:
            print("✅ Search is working correctly!")

        vs.close()

    except Exception as e:
        print(f"❌ Error inspecting vector store: {e}")
        import traceback
        traceback.print_exc()

def test_basic_functionality():
    """Test basic vector store operations."""
    try:
        print("=== Basic Functionality Test ===")

        from config import Config
        from vector_store import create_vector_store

        config = Config()

        # Create vector store
        vs = create_vector_store(
            store_type=config.vector_store_type,
            persist_directory=config.vector_store_directory,
            collection_name=config.vector_store_collection,
            dimension=config.embedding_dimension
        )

        print("✅ Vector store created successfully")

        # Test adding an embedding
        test_embedding = [0.1] * config.embedding_dimension
        test_metadata = {
            'filename': 'test_document.pdf',
            'file_path': '/test/test_document.pdf',
            'categories': 'test-category',
            'content_preview': 'This is a test document for debugging purposes.'
        }

        success = vs.add_embeddings(
            embeddings=[test_embedding],
            metadata=[test_metadata],
            ids=['test_doc_001']
        )

        if success:
            print("✅ Test embedding added successfully")
        else:
            print("❌ Failed to add test embedding")

        # Check count
        count = vs.count()
        print(f"📊 Current count: {count}")

        # Test search
        results = vs.search_similar(test_embedding, top_k=5, threshold=-1.0)
        print(f"🔎 Search test returned {len(results)} results")

        vs.close()
        print("✅ Basic functionality test completed")

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--basic":
        test_basic_functionality()
    else:
        inspect_vector_store()
