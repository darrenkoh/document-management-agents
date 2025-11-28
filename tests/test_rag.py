#!/usr/bin/env python3
"""Test script for RAG functionality."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from agent import DocumentAgent

def test_rag_search():
    """Test RAG-enhanced semantic search."""
    print("Testing RAG-enhanced semantic search...")

    # Load configuration
    config = Config('config.yaml')

    # Initialize agent
    agent = DocumentAgent(config, verbose=True)

    # Test queries
    test_queries = [
        "flight confirmation",
        "amazon order",
        "travel documents"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"Testing query: '{query}'")
        print('='*80)

        try:
            # Perform RAG search
            results = agent.search(query, top_k=5, use_rag=True)

            print(f"Found {len(results)} results:")

            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"Filename: {result.get('filename', 'N/A')}")
                print(f"Similarity: {result.get('similarity', 'N/A')}")
                print(f"RAG Relevance: {result.get('relevance_score', 'N/A')}")
                print(f"Is Relevant: {result.get('is_relevant', 'N/A')}")
                if result.get('relevance_reasoning'):
                    print(f"RAG Reasoning: {result.get('relevance_reasoning')[:200]}...")

        except Exception as e:
            print(f"Error testing query '{query}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_rag_search()
