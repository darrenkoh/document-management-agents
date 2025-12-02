#!/usr/bin/env python3
"""Test script to verify garbage text detection works with the user's example."""

import re

def _is_garbage_text(text: str) -> bool:
    """Check if extracted text appears to be garbage/random characters rather than meaningful content.

    This is specifically designed to detect corrupted PDF text layers that contain
    random symbols instead of actual document content.

    Args:
        text: The text to check

    Returns:
        True if text appears to be garbage, False if it might be meaningful
    """
    if not text or not text.strip():
        return True

    # Remove whitespace for analysis
    cleaned_text = text.replace(' ', '').replace('\n', '').replace('\t', '').strip()

    # If too short after cleaning, it's likely empty
    if len(cleaned_text) < 10:
        return True

    # Calculate ratio of alphanumeric characters to total characters
    # Garbage text often has very low alphanumeric content
    alpha_numeric = sum(1 for c in cleaned_text if c.isalnum())
    alpha_numeric_ratio = alpha_numeric / len(cleaned_text)

    # If less than 30% alphanumeric characters, likely garbage
    if alpha_numeric_ratio < 0.3:
        return True

    # Check for excessive special character patterns common in corrupted PDFs
    # Patterns like repeated symbols, excessive punctuation, etc.
    special_chars = sum(1 for c in cleaned_text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / len(cleaned_text)

    # If more than 70% special characters, likely garbage
    if special_ratio > 0.7:
        return True

    # Check for repetitive patterns (common in corrupted text)
    # Look for sequences of 3+ identical characters
    for char in set(cleaned_text.lower()):
        if cleaned_text.lower().count(char * 3) > 0:
            return True

    # Check if text contains mostly symbols that don't form words
    words = text.split()
    if len(words) > 0:
        # Calculate what percentage of "words" are actually just symbols
        symbol_words = sum(1 for word in words if len(word) > 0 and all(not c.isalnum() for c in word))
        symbol_word_ratio = symbol_words / len(words)

        # If more than 50% of words are pure symbols, likely garbage
        if symbol_word_ratio > 0.5:
            return True

    # Check for common garbage patterns from corrupted PDFs
    garbage_patterns = [
        r'[~@#$%^&*+=|\\{}[\]:;"\'<>,.?/-]{5,}',  # 5+ consecutive special chars
        r'[a-zA-Z]{1,2}[~@#$%^&*+=|\\{}[\]:;"\'<>,.?/-]{3,}',  # short letters + many symbols
        r'[0-9]{1,3}[~@#$%^&*+=|\\{}[\]:;"\'<>,.?/-]{3,}',  # short numbers + many symbols
    ]

    for pattern in garbage_patterns:
        if re.search(pattern, text):
            return True

    return False

def _has_meaningful_content(text: str) -> bool:
    """Check if text contains meaningful content (not just boilerplate, formatting, or OCR metadata).

    Args:
        text: The text to check

    Returns:
        True if text appears to contain meaningful content, False otherwise
    """
    if not text:
        return False

    # Remove common markdown boilerplate
    text = text.replace('```markdown', '').replace('```', '').strip()

    # Remove DeepSeek OCR metadata tokens
    text = text.replace('<|ref|>', '').replace('<|/ref|>', '').replace('<|det|>', '').replace('<|/det|>', '').strip()

    # Remove coordinate data (patterns like [[x,y,w,h]])
    text = re.sub(r'\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]', '', text).strip()

    # Check for minimum length (at least 10 characters for meaningful content)
    if len(text) < 10:
        return False

    # Check if it's just whitespace, punctuation, or common OCR artifacts
    if text.replace(' ', '').replace('\n', '').replace('\t', '').replace('-', '').replace('_', '').replace('=', '').replace('*', '').replace('#', '') == '':
        return False

    # Check for common "no content" responses
    no_content_indicators = [
        'no text found', 'no content', 'empty', 'blank',
        'no readable text', 'unable to extract', 'no data',
        'image', 'photo', 'picture', 'diagram', 'chart', 'graph'
    ]
    if any(indicator in text.lower() for indicator in no_content_indicators):
        return False

    # Check if text contains actual readable words (not just symbols)
    words = [word for word in text.split() if word.isalnum() and len(word) > 1]
    if len(words) < 2:  # Need at least 2 meaningful words
        return False

    return True

def test_garbage_detection():
    """Test the garbage text detection with the user's example."""

    # Test cases
    test_cases = [
        # User's example - should be detected as garbage
        ("User's example", "~-2 t ea • ~MBN<l1545-@00~,-- •.·ra.so i ~u.uz !"),

        # More garbage examples
        ("Pure symbols", "~~~@@##$$%%^^&&**(())__++||{{}}[[]];;::\"\"''<<>>,,..//??--=="),
        ("Mixed garbage", "abc~~~def@@@ghi###jkl$$$"),
        ("Repeated chars", "aaaaa!!!!!bbbbb@@@@@ccccc#####"),

        # Valid content examples
        ("Normal text", "This is a normal document with regular text content."),
        ("Invoice text", "INVOICE\nCustomer: John Doe\nAmount: $150.00\nDate: 2024-01-15"),
        ("Mixed valid", "Hello world! This is a test document with normal content."),
    ]

    print("Testing garbage text detection:")
    print("=" * 50)

    for name, text in test_cases:
        is_garbage = _is_garbage_text(text)
        has_meaningful = _has_meaningful_content(text)
        status = "GARBAGE" if is_garbage else "VALID"
        print(f"{name:15}: {status:7} | Meaningful: {has_meaningful:5} | Text: {text[:50]}{'...' if len(text) > 50 else ''}")

    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    test_garbage_detection()
