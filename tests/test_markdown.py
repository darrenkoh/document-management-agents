#!/usr/bin/env python3
"""Test script for markdown rendering functionality."""

def test_markdown_cleaning():
    """Test the markdown cleaning and rendering logic."""

    # Sample DeepSeek-OCR output with reference tags
    sample_content = """<|ref|>text<|/ref|><|det|>[[93, 130, 275, 161]]<|/det|>
DARREN KOH 3773 BRANDING IRON PL DUBLIN CA 9...

<|ref|>text<|/ref|><|det|>[[670, 124, 711, 135]]<|/det|>
SSN:

<|ref|>text<|/ref|><|det|>[[814, 123, 920, 133]]<|/det|>
XXX-XX-2480

<|ref|>title<|/ref|><|det|>[[347, 281, 630, 310]]<|/det|>
# YOUR TOTAL ACCOUNT VALUE IS: $260.12 AS OF 06/30/2011

<|ref|>table<|/ref|><|det|>[[67, 355, 555, 444]]<|/det|>

<table><tr><td colspan=\"2\">Your Account At A Glance</td></tr><tr><td>Opening Value on 04/01/2011</td><td>$261.30</td></tr><tr><td>Contributions</td><td>$0.00</td></tr><tr><td>Investment Gain/(Loss)</td><td>($1.18)</td></tr><tr><td>Total Account Value on 06/30/2011</td><td>$260.12</td></tr></table>

<|ref|>text<|/ref|><|det|>[[67, 563, 532, 590]]<|/det|>
Due to rounding, Total Account Value percentage may not equal 100%."""

    print("Original content:")
    print(repr(sample_content[:200]))
    print("\n" + "="*50 + "\n")

    # Clean the content (simulate the JavaScript cleaning function)
    import re
    cleaned = sample_content
    cleaned = re.sub(r'<\|ref\|>[^<]*<\|/ref\|>', '', cleaned)  # Remove reference tags
    cleaned = re.sub(r'<\|det\|>[^<]*<\|/det\|>', '', cleaned)  # Remove bounding box data
    cleaned = re.sub(r'<\|image\|>[^<]*<\|/image\|>', '', cleaned)  # Remove image tags
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Clean up excessive newlines
    cleaned = cleaned.strip()

    print("Cleaned content:")
    print(repr(cleaned[:200]))
    print("\n" + "="*50 + "\n")

    # Test markdown rendering (simplified)
    try:
        import markdown
        html_output = markdown.markdown(cleaned, extensions=['tables', 'fenced_code'])
        print("HTML output (first 300 chars):")
        print(html_output[:300])
    except ImportError:
        print("Markdown library not available for testing")

if __name__ == "__main__":
    test_markdown_cleaning()
