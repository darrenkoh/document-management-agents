#!/usr/bin/env python3
"""Test similarity score display in search results."""

def test_similarity_display():
    """Test how similarity scores are displayed."""

    # Test cases
    test_results = [
        {'filename': 'doc1.pdf', 'similarity': 1.0, 'doc_id': 1},
        {'filename': 'doc2.pdf', 'similarity': 0.8, 'doc_id': 2},
        {'filename': 'doc3.pdf', 'similarity': 0.6, 'doc_id': 3},
        {'filename': 'doc4.pdf', 'similarity': 0.3, 'doc_id': 4},
        {'filename': 'doc5.pdf', 'similarity': 0.0, 'doc_id': 5},
        {'filename': 'doc6.pdf', 'similarity': None, 'doc_id': 6},
    ]

    print("Testing similarity score display logic:")
    print("=" * 50)

    for result in test_results:
        similarity = result.get('similarity')
        filename = result['filename']

        # Simulate template logic
        has_similarity = similarity is not None

        if has_similarity:
            score_percent = similarity * 100
            if score_percent >= 85:
                score_class = 'similarity-score-high'
            elif score_percent >= 51:
                score_class = 'similarity-score-medium'
            else:
                score_class = 'similarity-score-low'

            display = ".1f"
        else:
            display = "No similarity score"

        print(f"{filename}: similarity={similarity} -> {display}")

if __name__ == "__main__":
    test_similarity_display()
