#!/usr/bin/env python3
"""Simple test script to verify config loading without external dependencies."""

import os
import sys
from pathlib import Path

# Add current directory to path to import our modules
sys.path.insert(0, os.getcwd())

# Mock yaml module since it's not available in system python
class MockYaml:
    @staticmethod
    def safe_load(f):
        # Simple mock implementation that just returns the content as dict
        content = f.read()
        # For our test, we'll manually create the expected dict
        return {
            'source_path': './input',
            'database': {'path': 'test_documents.json'}
        }

# Monkey patch yaml
sys.modules['yaml'] = MockYaml()

# Now import our config
from config import Config

def test_backward_compatibility():
    """Test backward compatibility with source_path."""
    config = Config('test_config_backward_compat.yaml')

    # Should work with old source_path format
    source_paths = config.source_paths
    print(f"Backward compatibility test - source_paths: {source_paths}")

    # Should return a list with the single path
    assert isinstance(source_paths, list), "source_paths should be a list"
    assert len(source_paths) == 1, "Should have one source path"
    assert 'input' in source_paths[0], "Should contain the input path"

    print("✓ Backward compatibility test passed")

def test_new_format():
    """Test new source_paths format."""
    # Create a mock config dict for new format
    config = Config.__new__(Config)
    config.config_path = Path('config.yaml')
    config._config = {
        'source_paths': ['./input', './documents'],
        'database': {'path': 'documents.json'}
    }

    source_paths = config.source_paths
    print(f"New format test - source_paths: {source_paths}")

    assert isinstance(source_paths, list), "source_paths should be a list"
    assert len(source_paths) == 2, "Should have two source paths"
    assert 'input' in source_paths[0], "Should contain first path"
    assert 'documents' in source_paths[1], "Should contain second path"

    print("✓ New format test passed")

if __name__ == '__main__':
    try:
        test_backward_compatibility()
        test_new_format()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
