#!/usr/bin/env python3
"""Test script for DeepSeek-OCR integration."""

import base64
import io
from pathlib import Path
from PIL import Image
import ollama

def test_deepseek_ocr():
    """Test DeepSeek-OCR with a simple image."""
    try:
        # Create a simple test image with text
        image = Image.new('RGB', (400, 100), color='white')
        # Note: In a real implementation, you'd add text to the image

        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        image_url = f"data:image/png;base64,{image_data}"

        # Test the OCR model
        client = ollama.Client(host="http://spark-7819.local:11434")

        response = client.generate(
            model="deepseek-ocr:3b",
            prompt="Extract all text from this image. Return only the text content, nothing else.",
            images=[image_url],
            options={
                'temperature': 0.0,
                'num_predict': 1000,
            }
        )

        print("DeepSeek-OCR Response:")
        print(response.get('response', 'No response'))

    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_deepseek_ocr()
