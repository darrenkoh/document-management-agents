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

        def _expand_abs_path(p: str) -> str:
            return os.path.abspath(os.path.expanduser(p))
        
        # Expand user paths and make them absolute
        if 'source_path' in config:
            config['source_path'] = _expand_abs_path(config['source_path'])

        if 'source_paths' in config:
            if isinstance(config['source_paths'], str):
                # Convert single string to list for consistency
                config['source_paths'] = [config['source_paths']]
            # Expand user paths and make them absolute
            config['source_paths'] = [
                _expand_abs_path(path)
                for path in config['source_paths']
            ]
        
        if 'destination_path' in config:
            config['destination_path'] = _expand_abs_path(config['destination_path'])

        # Expand watch exclude paths (if any)
        watch_config = config.get('watch', {}) or {}
        exclude_paths = watch_config.get('exclude_paths')
        if exclude_paths:
            if isinstance(exclude_paths, str):
                exclude_paths = [exclude_paths]
            watch_config['exclude_paths'] = [_expand_abs_path(p) for p in exclude_paths]
            config['watch'] = watch_config

        # Expand segmentation paths (if any)
        seg_config = config.get('segmentation', {}) or {}
        if 'output_dir' in seg_config and seg_config['output_dir']:
            seg_config['output_dir'] = _expand_abs_path(seg_config['output_dir'])
        if 'checkpoint_path' in seg_config and seg_config['checkpoint_path']:
            seg_config['checkpoint_path'] = _expand_abs_path(seg_config['checkpoint_path'])
        if seg_config:
            config['segmentation'] = seg_config
        
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
    def llm_endpoint(self) -> str:
        """Get the OpenAI-compatible LLM API endpoint."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('endpoint', 'http://localhost:11434/v1')
    
    @property
    def llm_model(self) -> str:
        """Get the LLM model name."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('model', 'gpt-3.5-turbo')

    @property
    def llm_timeout(self) -> int:
        """Get the LLM API timeout."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('timeout', 30)

    @property
    def llm_num_predict(self) -> int:
        """Get the maximum number of tokens to predict."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('num_predict', 200)
    
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
    def watch_exclude_paths(self) -> List[str]:
        """Get watch mode exclude paths (absolute paths)."""
        watch_config = self._config.get('watch', {})
        exclude_paths = watch_config.get('exclude_paths', [])
        if not exclude_paths:
            return []
        if isinstance(exclude_paths, str):
            return [exclude_paths]
        return list(exclude_paths)
    
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
    def rag_answer_prompt_template(self) -> Optional[str]:
        """Get the RAG answer generation prompt template.

        Returns:
            Prompt template string with {query} and {documents} placeholders, or None for default
        """
        return self._config.get('rag_answer_prompt_template')

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
    def llm_embedding_endpoint(self) -> str:
        """Get the embedding API endpoint."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('embedding_endpoint', self.llm_endpoint)

    @property
    def llm_embedding_model(self) -> str:
        """Get the embedding model name."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('embedding_model', 'text-embedding-3-small')

    @property
    def llm_summarizer_model(self) -> str:
        """Get the summarizer model name."""
        llm_config = self._config.get('llm', {})
        return llm_config.get('summarizer_model', 'gpt-3.5-turbo')

    @property
    def llm_max_retries(self) -> int:
        """Get the maximum number of retry attempts for failed LLM API calls."""
        llm_config = self._config.get('llm', {})
        retry_config = llm_config.get('retry', {})
        return retry_config.get('max_retries', 3)

    @property
    def llm_retry_base_delay(self) -> float:
        """Get the base delay in seconds between retry attempts."""
        llm_config = self._config.get('llm', {})
        retry_config = llm_config.get('retry', {})
        return retry_config.get('base_delay', 1.0)

    # OCR configuration properties
    @property
    def ocr_provider(self) -> str:
        """Get the OCR provider ('ollama', 'chandra', or 'hunyuan')."""
        ocr_config = self._config.get('ocr', {})
        return ocr_config.get('provider', 'ollama')

    @property
    def max_ocr_pages(self) -> int:
        """Get the maximum number of PDF pages to process with OCR."""
        ocr_config = self._config.get('ocr', {})
        return ocr_config.get('max_pages', 12)

    # Ollama OCR properties
    @property
    def ollama_ocr_model(self) -> str:
        """Get the Ollama OCR model name."""
        ocr_config = self._config.get('ocr', {})
        ollama_ocr = ocr_config.get('ollama', {})
        return ollama_ocr.get('model', 'deepseek-ocr:3b')

    @property
    def ollama_ocr_timeout(self) -> int:
        """Get the Ollama OCR timeout in seconds."""
        ocr_config = self._config.get('ocr', {})
        ollama_ocr = ocr_config.get('ollama', {})
        return ollama_ocr.get('timeout', 300)

    @property
    def ollama_ocr_max_retries(self) -> int:
        """Get the maximum number of retry attempts for Ollama OCR API calls."""
        ocr_config = self._config.get('ocr', {})
        ollama_ocr = ocr_config.get('ollama', {})
        retry_config = ollama_ocr.get('retry', {})
        return retry_config.get('max_retries', 3)

    @property
    def ollama_ocr_retry_base_delay(self) -> float:
        """Get the base delay in seconds between Ollama OCR retry attempts."""
        ocr_config = self._config.get('ocr', {})
        ollama_ocr = ocr_config.get('ollama', {})
        retry_config = ollama_ocr.get('retry', {})
        return retry_config.get('base_delay', 1.0)

    @property
    def ollama_ocr_num_predict(self) -> int:
        """Get the maximum number of tokens to predict for Ollama OCR."""
        ocr_config = self._config.get('ocr', {})
        ollama_ocr = ocr_config.get('ollama', {})
        return ollama_ocr.get('num_predict', 12000)  # Higher default for reasoning models

    # Chandra OCR properties
    @property
    def chandra_endpoint(self) -> str:
        """Get the Chandra vLLM API endpoint."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        return chandra_ocr.get('endpoint', 'http://localhost:11435')

    @property
    def chandra_model(self) -> str:
        """Get the Chandra model name."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        return chandra_ocr.get('model', 'chandra')

    @property
    def chandra_timeout(self) -> int:
        """Get the Chandra OCR timeout in seconds."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        return chandra_ocr.get('timeout', 1800)

    @property
    def chandra_max_tokens(self) -> int:
        """Get the maximum tokens for Chandra OCR."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        return chandra_ocr.get('max_tokens', 16384)

    @property
    def chandra_max_retries(self) -> int:
        """Get the maximum number of retry attempts for Chandra API calls."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        retry_config = chandra_ocr.get('retry', {})
        return retry_config.get('max_retries', 3)

    @property
    def chandra_retry_base_delay(self) -> float:
        """Get the base delay in seconds between Chandra retry attempts."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        retry_config = chandra_ocr.get('retry', {})
        return retry_config.get('base_delay', 1.0)

    @property
    def chandra_frequency_penalty(self) -> float:
        """Get the frequency penalty for Chandra OCR."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        return chandra_ocr.get('frequency_penalty', 0.0)

    @property
    def chandra_detect_repeat_tokens(self) -> bool:
        """Get whether to detect and retry on repetitive OCR output for Chandra."""
        ocr_config = self._config.get('ocr', {})
        chandra_ocr = ocr_config.get('chandra', {})
        return chandra_ocr.get('detect_repeat_tokens', False)

    # HunyuanOCR properties
    @property
    def hunyuan_endpoint(self) -> str:
        """Get the HunyuanOCR vLLM API endpoint."""
        ocr_config = self._config.get('ocr', {})
        hunyuan_ocr = ocr_config.get('hunyuan', {})
        return hunyuan_ocr.get('endpoint', 'http://localhost:11435')

    @property
    def hunyuan_model(self) -> str:
        """Get the HunyuanOCR model name."""
        ocr_config = self._config.get('ocr', {})
        hunyuan_ocr = ocr_config.get('hunyuan', {})
        return hunyuan_ocr.get('model', 'tencent/HunyuanOCR')

    @property
    def hunyuan_timeout(self) -> int:
        """Get the HunyuanOCR timeout in seconds."""
        ocr_config = self._config.get('ocr', {})
        hunyuan_ocr = ocr_config.get('hunyuan', {})
        return hunyuan_ocr.get('timeout', 1800)

    @property
    def hunyuan_max_tokens(self) -> int:
        """Get the maximum tokens for HunyuanOCR."""
        ocr_config = self._config.get('ocr', {})
        hunyuan_ocr = ocr_config.get('hunyuan', {})
        return hunyuan_ocr.get('max_tokens', 16384)

    @property
    def hunyuan_max_retries(self) -> int:
        """Get the maximum number of retry attempts for HunyuanOCR API calls."""
        ocr_config = self._config.get('ocr', {})
        hunyuan_ocr = ocr_config.get('hunyuan', {})
        retry_config = hunyuan_ocr.get('retry', {})
        return retry_config.get('max_retries', 3)

    @property
    def hunyuan_retry_base_delay(self) -> float:
        """Get the base delay in seconds between HunyuanOCR retry attempts."""
        ocr_config = self._config.get('ocr', {})
        hunyuan_ocr = ocr_config.get('hunyuan', {})
        retry_config = hunyuan_ocr.get('retry', {})
        return retry_config.get('base_delay', 1.0)

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

    @property
    def semantic_search_enable_bm25(self) -> bool:
        """Get whether BM25 keyword search is enabled for hybrid search."""
        search_config = self._config.get('semantic_search', {})
        return search_config.get('enable_bm25', False)

    @property
    def semantic_search_bm25_weight(self) -> float:
        """Get weight for BM25 scores in hybrid search (0.0 to 1.0)."""
        search_config = self._config.get('semantic_search', {})
        return float(search_config.get('bm25_weight', 0.3))

    @property
    def semantic_search_semantic_weight(self) -> float:
        """Get weight for semantic search scores in hybrid search (0.0 to 1.0)."""
        search_config = self._config.get('semantic_search', {})
        return float(search_config.get('semantic_weight', 0.7))

    # ----------------------------
    # Optional image segmentation
    # ----------------------------

    @property
    def segmentation_enable(self) -> bool:
        """Whether receipt segmentation is enabled."""
        seg_config = self._config.get('segmentation', {})
        return bool(seg_config.get('enable', False))

    @property
    def segmentation_output_dir(self) -> str:
        """Directory to write segmented receipt PNGs to (absolute path when loaded from config.yaml)."""
        seg_config = self._config.get('segmentation', {})
        return seg_config.get('output_dir', os.path.abspath('data/segmented_receipts'))

    @property
    def segmentation_device(self) -> str:
        """Device for segmentation: auto|mps|cpu."""
        seg_config = self._config.get('segmentation', {})
        return seg_config.get('device', 'auto')

    @property
    def segmentation_checkpoint_path(self) -> Optional[str]:
        """Path to SAM3 checkpoint file/directory (absolute path when loaded from config.yaml)."""
        seg_config = self._config.get('segmentation', {})
        return seg_config.get('checkpoint_path')

    @property
    def segmentation_text_prompt(self) -> str:
        """Text prompt to use for SAM3 grounding on images."""
        seg_config = self._config.get('segmentation', {})
        return seg_config.get('text_prompt', 'receipt')

    @property
    def segmentation_confidence_threshold(self) -> float:
        """Minimum confidence threshold for masks/boxes returned by SAM3 processor."""
        seg_config = self._config.get('segmentation', {})
        return float(seg_config.get('confidence_threshold', 0.5))

    @property
    def segmentation_max_masks(self) -> int:
        seg_config = self._config.get('segmentation', {})
        return int(seg_config.get('max_masks', 128))

    @property
    def segmentation_max_segments(self) -> int:
        seg_config = self._config.get('segmentation', {})
        return int(seg_config.get('max_segments', 10))

    @property
    def segmentation_min_area_ratio(self) -> float:
        seg_config = self._config.get('segmentation', {})
        return float(seg_config.get('min_area_ratio', 0.02))

    @property
    def segmentation_min_width_px(self) -> int:
        seg_config = self._config.get('segmentation', {})
        return int(seg_config.get('min_width_px', 200))

    @property
    def segmentation_min_height_px(self) -> int:
        seg_config = self._config.get('segmentation', {})
        return int(seg_config.get('min_height_px', 200))

    @property
    def segmentation_min_fill_ratio(self) -> float:
        seg_config = self._config.get('segmentation', {})
        return float(seg_config.get('min_fill_ratio', 0.35))

    @property
    def segmentation_iou_dedup_threshold(self) -> float:
        seg_config = self._config.get('segmentation', {})
        return float(seg_config.get('iou_dedup_threshold', 0.85))

    @property
    def segmentation_bbox_padding_px(self) -> int:
        seg_config = self._config.get('segmentation', {})
        return int(seg_config.get('bbox_padding_px', 12))

