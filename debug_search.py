#!/usr/bin/env python3
"""Debug script for semantic search functionality."""

import logging
from config import Config
from database import DocumentDatabase
from agent import DocumentAgent

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

def main():
    print("=== Semantic Search Debug ===")

    # Load configuration
    config = Config()

    # Initialize components
    database = DocumentDatabase(config.database_path)
    agent = DocumentAgent(config)

    # Check documents in database
    all_docs = database.get_all_documents()
    print(f"Total documents in database: {len(all_docs)}")

    docs_with_embeddings = sum(1 for doc in all_docs if doc.get('embedding'))
    print(f"Documents with embeddings: {docs_with_embeddings}")

    for doc in all_docs:
        has_embedding = 'embedding' in doc and doc['embedding'] is not None
        print(f"- {doc.get('filename', 'N/A')}: {'✓' if has_embedding else '✗'} embedding")

    # Test semantic search
    test_queries = ["flight booking", "amazon order", "travel documents"]

    for query in test_queries:
        print(f"\n=== Testing query: '{query}' ===")
        try:
            results = agent.search(query, top_k=10)
            print(f"Found {len(results)} results")

            for i, result in enumerate(results[:3]):  # Show top 3
                print(".3f")

        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
