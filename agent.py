"""Core agent workflow for document classification."""
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from file_handler import FileHandler
from classifier import Classifier
from config import Config
from database import DocumentDatabase
from embeddings import EmbeddingGenerator
from vector_store import create_vector_store

logger = logging.getLogger(__name__)


class DocumentAgent:
    """Main agent that orchestrates the document classification workflow."""
    
    def __init__(self, config: Config, verbose: bool = False):
        """Initialize the document agent.
        
        Args:
            config: Configuration object
            verbose: If True, enable verbose logging for LLM interactions
        """
        self.config = config
        self.verbose = verbose
        self.file_handler = FileHandler(
            config.source_paths,
            ollama_endpoint=config.ollama_endpoint,
            ocr_model=config.ollama_ocr_model,
            ocr_timeout=config.ollama_ocr_timeout
        )
        self.classifier = Classifier(
            endpoint=config.ollama_endpoint,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict,
            prompt_template=config.prompt_template
        )
        # Initialize vector store (required for embeddings)
        try:
            vector_store = create_vector_store(
                store_type=config.vector_store_type,
                persist_directory=config.vector_store_directory,
                collection_name=config.vector_store_collection,
                dimension=config.embedding_dimension,
                distance_metric=config.vector_store_distance_metric
            )
            logger.info(f"Initialized {config.vector_store_type} vector store")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise RuntimeError("Vector store initialization failed - embeddings cannot be stored")

        self.database = DocumentDatabase(config.database_path, vector_store=vector_store, config=config)
        self.embedding_generator = EmbeddingGenerator(
            endpoint=config.ollama_endpoint,
            model=config.ollama_embedding_model,
            timeout=config.ollama_timeout
        )
    
    def process_files_batch(self, file_paths: List[Path], batch_size: int = 10) -> Dict[str, any]:
        """Process multiple files in batches for better performance.

        Args:
            file_paths: List of file paths to process
            batch_size: Number of files to process before generating/storing embeddings

        Returns:
            Dictionary with processing statistics and performance metrics
        """
        logger.info(f"Starting batch processing of {len(file_paths)} files")

        stats = {
            'total': len(file_paths),
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'performance': {
                'total_hash_duration': 0.0,
                'total_ocr_duration': 0.0,
                'total_classification_duration': 0.0,
                'total_db_lookup_duration': 0.0,
                'total_db_insert_duration': 0.0,
                'avg_hash_duration': 0.0,
                'avg_ocr_duration': 0.0,
                'avg_classification_duration': 0.0,
                'avg_db_lookup_duration': 0.0,
                'avg_db_insert_duration': 0.0
            }
        }

        processed_files_count = 0

        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")

            # Process batch
            for file_path in batch:
                try:
                    success, perf_metrics = self.process_file(file_path)
                    if success:
                        stats['processed'] += 1
                        processed_files_count += 1
                    else:
                        stats['failed'] += 1

                    # Accumulate performance metrics
                    stats['performance']['total_hash_duration'] += perf_metrics['hash_duration']
                    stats['performance']['total_ocr_duration'] += perf_metrics['ocr_duration']
                    stats['performance']['total_classification_duration'] += perf_metrics['classification_duration']
                    stats['performance']['total_db_lookup_duration'] += perf_metrics['db_lookup_duration']
                    stats['performance']['total_db_insert_duration'] += perf_metrics['db_insert_duration']

                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {e}")
                    stats['failed'] += 1

        # Calculate averages
        if processed_files_count > 0:
            stats['performance']['avg_hash_duration'] = stats['performance']['total_hash_duration'] / processed_files_count
            stats['performance']['avg_ocr_duration'] = stats['performance']['total_ocr_duration'] / processed_files_count
            stats['performance']['avg_classification_duration'] = stats['performance']['total_classification_duration'] / processed_files_count
            stats['performance']['avg_db_lookup_duration'] = stats['performance']['total_db_lookup_duration'] / processed_files_count
            stats['performance']['avg_db_insert_duration'] = stats['performance']['total_db_insert_duration'] / processed_files_count

        logger.info(
            f"Batch processing complete: {stats['processed']} processed, "
            f"{stats['failed']} failed, {stats['total']} total"
        )

        return stats
    
    def process_file(self, file_path: Path) -> tuple[bool, dict]:
        """Process a single file: extract, classify, generate embeddings, and store in database.

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (success boolean, performance metrics dictionary)
        """
        try:
            # Initialize performance metrics
            perf_metrics = {
                'hash_duration': 0.0,
                'ocr_duration': 0.0,
                'classification_duration': 0.0,
                'db_lookup_duration': 0.0,
                'db_insert_duration': 0.0
            }

            # Generate file hash first
            hash_result = self.file_handler.generate_file_hash(file_path)
            if not hash_result:
                logger.error(f"Failed to generate hash for {file_path.name}, skipping")
                return (False, {'hash_duration': 0.0, 'ocr_duration': 0.0, 'classification_duration': 0.0,
                               'db_lookup_duration': 0.0, 'db_insert_duration': 0.0})

            file_hash, perf_metrics['hash_duration'] = hash_result

            # Check if already processed by hash (content-based duplicate detection)
            db_lookup_start = time.time()
            existing = self.database.get_document_by_hash(file_hash)
            perf_metrics['db_lookup_duration'] = time.time() - db_lookup_start

            if existing:
                logger.debug(f"Skipping already processed file (duplicate content): {file_path.name}")
                logger.debug(f"Original file: {existing.get('filename', 'Unknown')}")

                # Log performance metrics for skipped files in verbose mode
                if self.verbose:
                    total_duration = perf_metrics['hash_duration'] + perf_metrics['db_lookup_duration']
                    logger.info(f"Performance for {file_path.name} (skipped - duplicate): "
                               f"hash={perf_metrics['hash_duration']:.3f}s, "
                               f"db_lookup={perf_metrics['db_lookup_duration']:.3f}s, "
                               f"total={total_duration:.3f}s")

                return (True, {**perf_metrics, 'ocr_duration': 0.0, 'classification_duration': 0.0, 'db_insert_duration': 0.0})

            logger.info(f"Processing file: {file_path.name} (hash: {file_hash[:16]}...)")

            # Extract text content
            content, ocr_duration = self.file_handler.extract_text(file_path)
            perf_metrics['ocr_duration'] = ocr_duration

            if not content or not content.strip():
                logger.warning(f"No content extracted from {file_path.name}, skipping")
                # Add missing metrics for early returns
                perf_metrics.update({'db_lookup_duration': perf_metrics.get('db_lookup_duration', 0.0),
                                   'db_insert_duration': 0.0})
                return (False, perf_metrics)
            
            # Classify content
            classification_result = self.classifier.classify(content, file_path.name, verbose=self.verbose)

            if not classification_result:
                logger.error(f"Failed to classify {file_path.name}")
                # Add missing metrics for early returns
                perf_metrics.update({'db_lookup_duration': perf_metrics.get('db_lookup_duration', 0.0),
                                   'db_insert_duration': 0.0})
                return (False, perf_metrics)

            categories, perf_metrics['classification_duration'] = classification_result
            
            # Generate embedding for semantic search
            logger.info(f"Generating embedding for {file_path.name}...")
            embedding = self.embedding_generator.generate_embedding(content)

            if not embedding:
                logger.error(f"Failed to generate embedding for {file_path.name}")
                # Add missing metrics for early returns
                perf_metrics.update({'db_lookup_duration': perf_metrics.get('db_lookup_duration', 0.0),
                                   'db_insert_duration': 0.0})
                return (False, perf_metrics)

            # Prepare metadata
            metadata = {
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'file_extension': file_path.suffix,
                'file_modified': file_path.stat().st_mtime if file_path.exists() else None
            }

            # Store in database
            db_insert_start = time.time()
            doc_id = self.database.store_classification(
                file_path=str(file_path),
                content=content,
                categories=categories,
                metadata=metadata,
                file_hash=file_hash
            )

            # Store embedding
            self.database.store_embedding(str(file_path), embedding)
            perf_metrics['db_insert_duration'] = time.time() - db_insert_start
            
            # Export to JSON file
            self.database.export_to_json(self.config.json_export_path)
            
            logger.info(f"Successfully processed {file_path.name} -> {categories} (DB ID: {doc_id})")

            # Log performance metrics for this file in verbose mode
            if self.verbose:
                total_duration = (perf_metrics['hash_duration'] + perf_metrics['ocr_duration'] +
                                perf_metrics['classification_duration'] + perf_metrics['db_lookup_duration'] +
                                perf_metrics['db_insert_duration'])
                logger.info(f"Performance for {file_path.name}: hash={perf_metrics['hash_duration']:.3f}s, "
                           f"ocr={perf_metrics['ocr_duration']:.3f}s, "
                           f"classify={perf_metrics['classification_duration']:.3f}s, "
                           f"db_lookup={perf_metrics['db_lookup_duration']:.3f}s, "
                           f"db_insert={perf_metrics['db_insert_duration']:.3f}s, "
                           f"total={total_duration:.3f}s")

            return (True, perf_metrics)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return (False, {'hash_duration': 0.0, 'ocr_duration': 0.0, 'classification_duration': 0.0,
                           'db_lookup_duration': 0.0, 'db_insert_duration': 0.0})
    
    def process_all(self) -> dict:
        """Process all files in the source directory.
        
        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting batch processing of all files")
        
        # Get files to process
        files = self.file_handler.get_files(
            extensions=self.config.file_extensions if self.config.file_extensions else None,
            recursive=self.config.watch_recursive
        )
        
        if not files:
            logger.info("No files found to process")
            return {
                'total': 0,
                'processed': 0,
                'failed': 0,
                'skipped': 0
            }
        
        logger.info(f"Found {len(files)} file(s) to process")
        
        # Use batch processing for better performance
        stats = self.process_files_batch(files, batch_size=10)
        
        logger.info(
            f"Processing complete: {stats['processed']} processed, "
            f"{stats['failed']} failed, {stats['total']} total"
        )
        
        # Final JSON export
        self.database.export_to_json(self.config.json_export_path)
        logger.info(f"Classification results exported to {self.config.json_export_path}")
        
        return stats
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess search query to improve semantic search quality.

        Args:
            query: Raw search query

        Returns:
            Processed query string
        """
        import re

        # Convert to lowercase
        query = query.lower().strip()

        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)

        # Common stop words that don't add much semantic value
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'what', 'when', 'where', 'who', 'why'
        }

        # Remove stop words but keep important domain terms
        words = query.split()
        filtered_words = []
        for word in words:
            # Keep numbers, short important words, and domain-specific terms
            if len(word) <= 2 or word not in stop_words or word in {'ai', 'ml', 'api', 'pdf', 'doc', 'txt'}:
                filtered_words.append(word)

        # If we filtered too much, use original query
        if len(filtered_words) < len(words) * 0.5 and len(words) > 3:
            filtered_words = words

        return ' '.join(filtered_words)


    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Perform semantic search on documents.

        Args:
            query: Search query text
            top_k: Number of results to return (uses config default if None)

        Returns:
            List of matching documents with similarity scores
        """
        # Use config values if not specified
        if top_k is None:
            top_k = self.config.semantic_search_top_k

        threshold = self.config.semantic_search_min_threshold
        debug_enabled = self.config.semantic_search_debug or logger.isEnabledFor(logging.DEBUG)

        # Preprocess query for better semantic search
        processed_query = self._preprocess_query(query)
        logger.info(f"Performing semantic search for: '{query}' -> '{processed_query}' (top_k={top_k}, threshold={threshold}, debug={debug_enabled})")

        # Check if embedding generator is available
        if not hasattr(self, 'embedding_generator') or self.embedding_generator is None:
            logger.error("Embedding generator not available")
            return []

        # Generate query embedding
        if debug_enabled:
            logger.debug("Generating query embedding...")
        try:
            query_embedding = self.embedding_generator.generate_query_embedding(processed_query)
        except Exception as e:
            logger.error(f"Exception during query embedding generation: {e}")
            return []

        if not query_embedding:
            logger.error("Failed to generate query embedding - returned None")
            return []

        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            logger.error(f"Invalid query embedding: {type(query_embedding)}, length: {len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}")
            return []

        if debug_enabled:
            logger.debug(f"Generated query embedding with {len(query_embedding)} dimensions")

        # Search database
        if debug_enabled:
            logger.debug("Searching database for semantic matches...")
        try:
            results = self.database.search_semantic(query_embedding, top_k=top_k, threshold=threshold)
        except Exception as e:
            logger.error(f"Exception during database search: {e}")
            return []

        logger.info(f"Found {len(results)} matching documents")
        if results:
            if debug_enabled:
                logger.info(f"Top result: {results[0].get('filename', 'N/A')} (similarity: {results[0].get('similarity', 'N/A')})")
                for i, result in enumerate(results[:3]):
                    logger.info(f"Result {i+1}: {result.get('filename', 'N/A')} (similarity: {results[i].get('similarity', 'N/A')})")
            else:
                logger.info(f"Top result: {results[0].get('filename', 'N/A')} (similarity: {results[0].get('similarity', 'N/A')})")
        return results
    
    def search_by_category(self, category: str) -> List[Dict]:
        """Search documents by category.
        
        Args:
            category: Category to search for
        
        Returns:
            List of matching documents
        """
        return self.database.search_by_category(category)
    
    def close(self):
        """Close database connections."""
        self.database.close()
    
    def watch(self, interval: Optional[int] = None):
        """Watch source directory and process new files continuously.
        
        Args:
            interval: Polling interval in seconds (uses config if not provided)
        """
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class WatchFileHandler(FileSystemEventHandler):
            """Watchdog event handler for file system events."""
            
            def __init__(self, agent):
                self.agent = agent
                self.processed = set()
            
            def on_created(self, event):
                """Handle file creation events."""
                if not event.is_directory:
                    file_path = Path(event.src_path)
                    # Wait a bit for file to be fully written
                    time.sleep(1)
                    if file_path.exists() and str(file_path) not in self.processed:
                        self.processed.add(str(file_path))
                        self.agent.process_file(file_path)
        
        source_paths_str = ", ".join(str(path) for path in self.config.source_paths)
        logger.info(f"Starting watch mode on: {source_paths_str}")
        logger.info(f"Polling interval: {interval or self.config.watch_interval} seconds")

        event_handler = WatchFileHandler(self)
        observer = Observer()

        # Schedule each source path for watching
        for source_path in self.config.source_paths:
            if Path(source_path).exists():
                observer.schedule(
                    event_handler,
                    source_path,
                    recursive=self.config.watch_recursive
                )
                logger.info(f"Watching directory: {source_path}")
            else:
                logger.warning(f"Source path does not exist, skipping: {source_path}")
        
        observer.start()
        
        try:
            while True:
                time.sleep(interval or self.config.watch_interval)
        except KeyboardInterrupt:
            logger.info("Stopping watch mode...")
            observer.stop()
        
        observer.join()
        logger.info("Watch mode stopped")

