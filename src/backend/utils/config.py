"""Configuration management for the document classification agent."""
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional


class Config:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}. "
                "Please create config.yaml in the project root."
            )
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expand user paths and make them absolute
        if 'source_path' in config:
            config['source_path'] = os.path.expanduser(config['source_path'])
            config['source_path'] = os.path.abspath(config['source_path'])

        if 'source_paths' in config:
            if isinstance(config['source_paths'], str):
                # Convert single string to list for consistency
                config['source_paths'] = [config['source_paths']]
            # Expand user paths and make them absolute
            config['source_paths'] = [
                os.path.abspath(os.path.expanduser(path))
                for path in config['source_paths']
            ]
        
        if 'destination_path' in config:
            config['destination_path'] = os.path.expanduser(config['destination_path'])
            config['destination_path'] = os.path.abspath(config['destination_path'])
        
        return config
    
    @property
    def source_path(self) -> str:
        """Get the source directory path (for backward compatibility)."""
        return self._config.get('source_path', './input')

    @property
    def source_paths(self) -> List[str]:
        """Get the list of source directory paths."""
        # Check if source_paths is configured
        if 'source_paths' in self._config:
            return self._config['source_paths']
        # Fall back to source_path for backward compatibility
        elif 'source_path' in self._config:
            return [self._config['source_path']]
        # Default to single input directory
        else:
            return ['./input']
    
    @property
    def destination_path(self) -> str:
        """Get the destination directory path."""
        return self._config.get('destination_path', './output')
    
    @property
    def ollama_endpoint(self) -> str:
        """Get the Ollama API endpoint."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('endpoint', 'http://localhost:11434')
    
    @property
    def ollama_model(self) -> str:
        """Get the Ollama model name."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('model', 'llama3.2')
    
    @property
    def ollama_timeout(self) -> int:
        """Get the Ollama API timeout."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('timeout', 30)
    
    @property
    def ollama_num_predict(self) -> int:
        """Get the maximum number of tokens to predict."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('num_predict', 200)
    
    @property
    def file_extensions(self) -> List[str]:
        """Get the list of file extensions to process."""
        extensions = self._config.get('file_extensions', [])
        return [ext.lower() for ext in extensions]
    
    @property
    def categories(self) -> List[str]:
        """Get predefined categories (if any)."""
        return self._config.get('categories', [])
    
    @property
    def watch_interval(self) -> int:
        """Get watch mode polling interval."""
        watch_config = self._config.get('watch', {})
        return watch_config.get('interval', 5)
    
    @property
    def watch_recursive(self) -> bool:
        """Get watch mode recursive setting."""
        watch_config = self._config.get('watch', {})
        return watch_config.get('recursive', True)
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        log_config = self._config.get('logging', {})
        return log_config.get('level', 'INFO')
    
    @property
    def log_file(self) -> Optional[str]:
        """Get log file path."""
        log_config = self._config.get('logging', {})
        return log_config.get('file')
    
    @property
    def prompt_template(self) -> Optional[str]:
        """Get the classification prompt template.

        Returns:
            Prompt template string with {filename} and {content} placeholders, or None for default
        """
        return self._config.get('prompt_template')

    @property
    def webapp_port(self) -> int:
        """Get the web app port."""
        webapp_config = self._config.get('webapp', {})
        return webapp_config.get('port', 5000)

    @property
    def webapp_host(self) -> str:
        """Get the web app host."""
        webapp_config = self._config.get('webapp', {})
        return webapp_config.get('host', '0.0.0.0')

    @property
    def webapp_debug(self) -> bool:
        """Get the web app debug mode."""
        webapp_config = self._config.get('webapp', {})
        return webapp_config.get('debug', True)
    
    @property
    def database_path(self) -> str:
        """Get the database file path."""
        db_config = self._config.get('database', {})
        return db_config.get('path', 'documents.json')

    @property
    def ollama_embedding_model(self) -> str:
        """Get the embedding model name."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('embedding_model', 'qwen3-embedding:8b')

    @property
    def ollama_summarizer_model(self) -> str:
        """Get the summarizer model name."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('summarizer_model', 'deepseek-r1:8b')

    @property
    def ollama_ocr_model(self) -> str:
        """Get the OCR model name."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('ocr_model', 'deepseek-ocr:3b')

    @property
    def ollama_ocr_timeout(self) -> int:
        """Get the OCR timeout in seconds."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('ocr_timeout', 60)

    @property
    def ollama_max_retries(self) -> int:
        """Get the maximum number of retry attempts for failed LLM API calls."""
        ollama_config = self._config.get('ollama', {})
        retry_config = ollama_config.get('retry', {})
        return retry_config.get('max_retries', 3)

    @property
    def ollama_retry_base_delay(self) -> float:
        """Get the base delay in seconds between retry attempts."""
        ollama_config = self._config.get('ollama', {})
        retry_config = ollama_config.get('retry', {})
        return retry_config.get('base_delay', 1.0)

    @property
    def ocr_provider(self) -> str:
        """Get the OCR provider ('ollama' or 'chandra')."""
        ollama_config = self._config.get('ollama', {})
        ocr_model = ollama_config.get('ocr_model', 'deepseek-ocr:3b')
        # If ocr_model is 'chandra', use chandra provider
        if ocr_model == 'chandra':
            return 'chandra'
        # Otherwise, assume ollama
        return 'ollama'

    @property
    def chandra_endpoint(self) -> str:
        """Get the Chandra vLLM API endpoint."""
        chandra_config = self._config.get('chandra', {})
        return chandra_config.get('endpoint', 'http://localhost:11435')

    @property
    def chandra_model(self) -> str:
        """Get the Chandra model name."""
        chandra_config = self._config.get('chandra', {})
        return chandra_config.get('model', 'chandra')

    @property
    def chandra_timeout(self) -> int:
        """Get the Chandra OCR timeout in seconds."""
        chandra_config = self._config.get('chandra', {})
        return chandra_config.get('timeout', 300)

    @property
    def chandra_max_tokens(self) -> int:
        """Get the maximum tokens for Chandra OCR."""
        chandra_config = self._config.get('chandra', {})
        return chandra_config.get('max_tokens', 8192)

    @property
    def chandra_max_retries(self) -> int:
        """Get the maximum number of retry attempts for Chandra API calls."""
        chandra_config = self._config.get('chandra', {})
        retry_config = chandra_config.get('retry', {})
        return retry_config.get('max_retries', 3)

    @property
    def chandra_retry_base_delay(self) -> float:
        """Get the base delay in seconds between Chandra retry attempts."""
        chandra_config = self._config.get('chandra', {})
        retry_config = chandra_config.get('retry', {})
        return retry_config.get('base_delay', 1.0)

    @property
    def chandra_frequency_penalty(self) -> float:
        """Get the frequency penalty for Chandra OCR."""
        chandra_config = self._config.get('chandra', {})
        return chandra_config.get('frequency_penalty', 0.02)

    @property
    def chandra_detect_repeat_tokens(self) -> bool:
        """Get whether to detect and retry on repetitive OCR output for Chandra."""
        chandra_config = self._config.get('chandra', {})
        return chandra_config.get('detect_repeat_tokens', True)

    @property
    def vector_store_type(self) -> str:
        """Get the vector store type."""
        db_config = self._config.get('database', {})
        vector_config = db_config.get('vector_store', {})
        return vector_config.get('type', 'chromadb')

    @property
    def vector_store_directory(self) -> str:
        """Get the vector store persistence directory."""
        db_config = self._config.get('database', {})
        vector_config = db_config.get('vector_store', {})
        return vector_config.get('persist_directory', 'vector_store')

    @property
    def vector_store_collection(self) -> str:
        """Get the vector store collection name."""
        db_config = self._config.get('database', {})
        vector_config = db_config.get('vector_store', {})
        return vector_config.get('collection_name', 'documents')

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        db_config = self._config.get('database', {})
        vector_config = db_config.get('vector_store', {})
        return vector_config.get('dimension', 768)

    @property
    def vector_store_distance_metric(self) -> str:
        """Get the vector store distance metric."""
        db_config = self._config.get('database', {})
        vector_config = db_config.get('vector_store', {})
        return vector_config.get('distance_metric', 'l2')

    @property
    def chunk_size(self) -> int:
        """Get the document chunk size for embeddings."""
        chunking_config = self._config.get('chunking', {})
        return chunking_config.get('chunk_size', 4000)

    @property
    def chunk_overlap(self) -> int:
        """Get the chunk overlap size."""
        chunking_config = self._config.get('chunking', {})
        return chunking_config.get('chunk_overlap', 200)

    @property
    def enable_summary_embedding(self) -> bool:
        """Get whether to generate summary embeddings."""
        chunking_config = self._config.get('chunking', {})
        return chunking_config.get('enable_summary_embedding', True)

    @property
    def semantic_search_top_k(self) -> int:
        """Get the default number of semantic search results."""
        search_config = self._config.get('semantic_search', {})
        return search_config.get('top_k', 10)

    @property
    def semantic_search_min_threshold(self) -> float:
        """Get the minimum similarity threshold for semantic search."""
        search_config = self._config.get('semantic_search', {})
        return search_config.get('min_similarity_threshold', 0.0)

    @property
    def semantic_search_max_candidates(self) -> int:
        """Get the maximum number of candidates to retrieve before filtering."""
        search_config = self._config.get('semantic_search', {})
        return search_config.get('max_candidates', 50)

    @property
    def semantic_search_debug(self) -> bool:
        """Get whether to enable debug logging for similarity calculations."""
        search_config = self._config.get('semantic_search', {})
        return search_config.get('debug_similarity', False)

