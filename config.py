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
        
        if 'destination_path' in config:
            config['destination_path'] = os.path.expanduser(config['destination_path'])
            config['destination_path'] = os.path.abspath(config['destination_path'])
        
        return config
    
    @property
    def source_path(self) -> str:
        """Get the source directory path."""
        return self._config.get('source_path', './input')
    
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
    def json_export_path(self) -> str:
        """Get the JSON export file path."""
        db_config = self._config.get('database', {})
        return db_config.get('json_export_path', 'classifications.json')
    
    @property
    def ollama_embedding_model(self) -> str:
        """Get the embedding model name."""
        ollama_config = self._config.get('ollama', {})
        return ollama_config.get('embedding_model', 'nomic-embed-text')

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

