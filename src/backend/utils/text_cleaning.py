"""Text cleaning utilities for LLM responses."""
import re


def strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from LLM output.
    
    Reasoning models like DeepSeek-R1 and Claude with extended thinking
    include thinking blocks in their output. These should be stripped
    before storing in the database or using for downstream processing.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Text with thinking blocks removed
    """
    if not text:
        return text
    return re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()
