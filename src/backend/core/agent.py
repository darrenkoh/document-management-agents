"""Core agent workflow for document classification."""
import logging
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Generator, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.services.file_handler import FileHandler
from src.backend.core.classifier import Classifier
from src.backend.utils.config import Config
from src.backend.database.database_sqlite_standalone import SQLiteDocumentDatabase
from src.backend.services.embeddings import EmbeddingGenerator
from src.backend.services.vector_store import create_vector_store
from src.backend.core.rag_agent import RAGAgent

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

        self.database = SQLiteDocumentDatabase(config.database_path, vector_store=vector_store, config=config)
        self.embedding_generator = EmbeddingGenerator(
            endpoint=config.ollama_endpoint,
            model=config.ollama_embedding_model,
            summarizer_model=config.ollama_summarizer_model,
            timeout=config.ollama_timeout
        )

        # Create method to get existing categories for classifier
        def get_existing_categories():
            """Get all unique categories from existing documents."""
            try:
                all_docs = self.database.get_all_documents()
                categories = [doc.get('categories', '') for doc in all_docs if doc.get('categories')]
                return categories
            except Exception as e:
                logger.warning(f"Failed to get existing categories: {e}")
                return []

        def get_existing_sub_categories():
            """Get all existing sub-categories from documents."""
            try:
                all_docs = self.database.get_all_documents()
                sub_categories = []
                for doc in all_docs:
                    sub_cats = doc.get('sub_categories', [])
                    if sub_cats:
                        sub_categories.append(sub_cats)
                return sub_categories
            except Exception as e:
                logger.warning(f"Failed to get existing sub-categories: {e}")
                return []

        # Create summarizer function for classifier
        def summarize_for_classification(text: str, max_length: int = 1500) -> str:
            """Summarize document text for classification purposes."""
            return self.embedding_generator.generate_document_summary(text, max_length=max_length)

        self.classifier = Classifier(
            endpoint=config.ollama_endpoint,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict,
            prompt_template=config.prompt_template,
            existing_categories_getter=get_existing_categories,
            existing_sub_categories_getter=get_existing_sub_categories,
            summarizer=summarize_for_classification
        )

        # Initialize RAG agent for document analysis
        self.rag_agent = RAGAgent(
            endpoint=config.ollama_endpoint,
            model=config.ollama_model,  # Use same model as classifier (deepseek-r1:8b)
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict
        )
    
    def process_files_batch(self, file_paths: List[Path], batch_size: int = 10, progress_callback=None) -> Dict[str, any]:
        """Process multiple files in batches for better performance.

        Args:
            file_paths: List of file paths to process
            batch_size: Number of files to process before generating/storing embeddings
            progress_callback: Optional callback function called after each file (current_count, total_count)

        Returns:
            Dictionary with processing statistics and performance metrics
        """
        logger.info(f"Starting batch processing of {len(file_paths)} files")

        stats = {
            'total': len(file_paths),
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'skipped_deleted': 0,
            'performance': {
                'total_hash_duration': 0.0,
                'total_ocr_duration': 0.0,
                'total_classification_duration': 0.0,
                'total_embedding_duration': 0.0,
                'total_db_lookup_duration': 0.0,
                'total_db_insert_duration': 0.0,
                'avg_hash_duration': 0.0,
                'avg_ocr_duration': 0.0,
                'avg_classification_duration': 0.0,
                'avg_embedding_duration': 0.0,
                'avg_db_lookup_duration': 0.0,
                'avg_db_insert_duration': 0.0
            }
        }

        processed_files_count = 0

        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")

            # Process batch
            for file_path in batch:
                try:
                    status, perf_metrics = self.process_file(file_path)
                    if status == 'processed':
                        stats['processed'] += 1
                        processed_files_count += 1
                    elif status == 'skipped_duplicate':
                        stats['skipped'] += 1
                        processed_files_count += 1  # Still count for performance averages
                    elif status == 'skipped_deleted':
                        stats['skipped_deleted'] += 1
                        processed_files_count += 1  # Still count for performance averages
                    elif status == 'failed':
                        stats['failed'] += 1

                            # Accumulate performance metrics
                    stats['performance']['total_hash_duration'] += perf_metrics['hash_duration']
                    stats['performance']['total_ocr_duration'] += perf_metrics['ocr_duration']
                    stats['performance']['total_classification_duration'] += perf_metrics['classification_duration']
                    stats['performance']['total_embedding_duration'] += perf_metrics['embedding_duration']
                    stats['performance']['total_db_lookup_duration'] += perf_metrics['db_lookup_duration']
                    stats['performance']['total_db_insert_duration'] += perf_metrics['db_insert_duration']

                    # Call progress callback after each file
                    if progress_callback:
                        progress_callback(stats['processed'] + stats['skipped'] + stats['skipped_deleted'] + stats['failed'], stats['total'])

                except Exception as e:
                    logger.error(f"Unexpected error processing {file_path}: {e}")
                    stats['failed'] += 1
                    # Call progress callback even on error
                    if progress_callback:
                        progress_callback(stats['processed'] + stats['skipped'] + stats['skipped_deleted'] + stats['failed'], stats['total'])

        # Calculate averages
        if processed_files_count > 0:
            stats['performance']['avg_hash_duration'] = stats['performance']['total_hash_duration'] / processed_files_count
            stats['performance']['avg_ocr_duration'] = stats['performance']['total_ocr_duration'] / processed_files_count
            stats['performance']['avg_classification_duration'] = stats['performance']['total_classification_duration'] / processed_files_count
            stats['performance']['avg_embedding_duration'] = stats['performance']['total_embedding_duration'] / processed_files_count
            stats['performance']['avg_db_lookup_duration'] = stats['performance']['total_db_lookup_duration'] / processed_files_count
            stats['performance']['avg_db_insert_duration'] = stats['performance']['total_db_insert_duration'] / processed_files_count

        logger.info(
            f"Batch processing complete: {stats['processed']} processed, "
            f"{stats['skipped']} skipped (duplicates), "
            f"{stats['skipped_deleted']} skipped (deleted), "
            f"{stats['failed']} failed, {stats['total']} total"
        )

        return stats
    
    def process_file(self, file_path: Path) -> tuple[str, dict]:
        """Process a single file: extract, classify, generate embeddings, and store in database.

        Args:
            file_path: Path to the file to process

        Returns:
            Tuple of (status string, performance metrics dictionary)
            Status can be: 'processed', 'skipped_duplicate', 'skipped_deleted', 'failed'
        """
        try:
            # Initialize performance metrics
            perf_metrics = {
                'hash_duration': 0.0,
                'ocr_duration': 0.0,
                'classification_duration': 0.0,
                'embedding_duration': 0.0,
                'db_lookup_duration': 0.0,
                'db_insert_duration': 0.0
            }

            # Generate file hash first
            hash_result = self.file_handler.generate_file_hash(file_path)
            if not hash_result:
                logger.error(f"Failed to generate hash for {file_path.name}, skipping")
                return ('failed', {'hash_duration': 0.0, 'ocr_duration': 0.0, 'classification_duration': 0.0,
                               'embedding_duration': 0.0, 'db_lookup_duration': 0.0, 'db_insert_duration': 0.0})

            file_hash, perf_metrics['hash_duration'] = hash_result

            # Check if file was previously deleted (skip processing deleted files)
            db_lookup_start = time.time()
            if self.database.is_file_hash_deleted(file_hash):
                logger.info(f"Skipping previously deleted file: {file_path.name} (hash: {file_hash[:16]}...)")
                perf_metrics['db_lookup_duration'] = time.time() - db_lookup_start

                # Log performance metrics for skipped files
                total_duration = perf_metrics['hash_duration'] + perf_metrics['db_lookup_duration']
                logger.info(f"Performance for {file_path.name} (skipped - previously deleted): "
                           f"hash={perf_metrics['hash_duration']:.3f}s, "
                           f"db_lookup={perf_metrics['db_lookup_duration']:.3f}s, "
                           f"total={total_duration:.3f}s")

                return ('skipped_deleted', {**perf_metrics, 'ocr_duration': 0.0, 'classification_duration': 0.0, 'embedding_duration': 0.0, 'db_insert_duration': 0.0})

            # Check if already processed by hash (content-based duplicate detection)
            existing = self.database.get_document_by_hash(file_hash)
            perf_metrics['db_lookup_duration'] = time.time() - db_lookup_start

            if existing:
                logger.info(f"Skipping already processed file (duplicate content): {file_path.name}")
                logger.info(f"Original file: {existing.get('filename', 'Unknown')}")

                # Log performance metrics for skipped files
                total_duration = perf_metrics['hash_duration'] + perf_metrics['db_lookup_duration']
                logger.info(f"Performance for {file_path.name} (skipped - duplicate): "
                           f"hash={perf_metrics['hash_duration']:.3f}s, "
                           f"db_lookup={perf_metrics['db_lookup_duration']:.3f}s, "
                           f"total={total_duration:.3f}s")

                return ('skipped_duplicate', {**perf_metrics, 'ocr_duration': 0.0, 'classification_duration': 0.0, 'embedding_duration': 0.0, 'db_insert_duration': 0.0})

            logger.info(f"Processing file: {file_path.name} (hash: {file_hash[:16]}...)")

            # Extract text content
            content, ocr_duration, ocr_used = self.file_handler.extract_text(file_path)
            perf_metrics['ocr_duration'] = ocr_duration

            if not content or not content.strip():
                logger.warning(f"No content extracted from {file_path.name}, skipping")
                # Add missing metrics for early returns
                perf_metrics.update({'db_lookup_duration': perf_metrics.get('db_lookup_duration', 0.0),
                                   'db_insert_duration': 0.0})
                return ('failed', perf_metrics)
            
            # Classify content
            classification_result = self.classifier.classify(content, file_path.name, verbose=self.verbose)

            if not classification_result:
                logger.error(f"Failed to classify {file_path.name}")
                # Add missing metrics for early returns
                perf_metrics.update({'db_lookup_duration': perf_metrics.get('db_lookup_duration', 0.0),
                                   'db_insert_duration': 0.0})
                return ('failed', perf_metrics)

            if len(classification_result) == 3:
                # New format with sub-categories
                categories, perf_metrics['classification_duration'], sub_categories = classification_result
            else:
                # Backward compatibility for old format
                categories, perf_metrics['classification_duration'] = classification_result
                sub_categories = []

            # Generate embeddings using semantic chunking and summary
            logger.info(f"Generating embeddings for {file_path.name}...")
            embedding_start = time.time()
            embedding_result = self.embedding_generator.generate_document_embeddings(
                content,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
                generate_summary=self.config.enable_summary_embedding
            )
            perf_metrics['embedding_duration'] = time.time() - embedding_start

            # Check if we have at least one embedding (chunks or summary)
            if not embedding_result['chunks'] and not embedding_result['summary']:
                logger.error(f"Failed to generate any embeddings for {file_path.name}")
                # Add missing metrics for early returns
                perf_metrics.update({'db_lookup_duration': perf_metrics.get('db_lookup_duration', 0.0),
                                   'db_insert_duration': 0.0})
                return ('failed', perf_metrics)

            # For backward compatibility, use the first chunk or summary as the main embedding
            # (We'll store all embeddings separately)
            main_embedding = embedding_result['chunks'][0] if embedding_result['chunks'] else embedding_result['summary']

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
                file_hash=file_hash,
                deepseek_ocr_used=ocr_used,
                sub_categories=sub_categories,
                summary=embedding_result.get('summary_text')
            )

            # Store embeddings (chunks and summary)
            self.database.store_document_embeddings(
                str(file_path),
                embedding_result['chunks'],
                embedding_result['summary']
            )
            perf_metrics['db_insert_duration'] = time.time() - db_insert_start

            logger.info(f"Successfully processed {file_path.name} -> {categories} (DB ID: {doc_id})")

            # Log performance metrics for this file
            total_duration = (perf_metrics['hash_duration'] + perf_metrics['ocr_duration'] +
                            perf_metrics['classification_duration'] + perf_metrics['embedding_duration'] +
                            perf_metrics['db_lookup_duration'] + perf_metrics['db_insert_duration'])
            logger.info(f"Performance for {file_path.name}: hash={perf_metrics['hash_duration']:.3f}s, "
                       f"ocr={perf_metrics['ocr_duration']:.3f}s, "
                       f"classify={perf_metrics['classification_duration']:.3f}s, "
                       f"embedding={perf_metrics['embedding_duration']:.3f}s, "
                       f"db_lookup={perf_metrics['db_lookup_duration']:.3f}s, "
                       f"db_insert={perf_metrics['db_insert_duration']:.3f}s, "
                       f"total={total_duration:.3f}s")

            return ('processed', perf_metrics)

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return ('failed', {'hash_duration': 0.0, 'ocr_duration': 0.0, 'classification_duration': 0.0,
                           'embedding_duration': 0.0, 'db_lookup_duration': 0.0, 'db_insert_duration': 0.0})

    def _collect_all_files(self) -> Dict[Path, List[Path]]:
        """Collect all files from all directories using BFS traversal.

        Returns:
            Dictionary mapping directory paths to lists of files in that directory
        """
        from collections import deque

        files_by_directory = {}
        directories_to_process = deque()

        # Start with source directories
        for source_path in self.file_handler.source_paths:
            if source_path.exists():
                directories_to_process.append(source_path)
            else:
                logger.warning(f"Source path does not exist: {source_path}")

        # BFS traversal to collect all files
        while directories_to_process:
            current_dir = directories_to_process.popleft()

            # Get files in current directory only (not subdirectories)
            try:
                files_in_dir = []
                for item in current_dir.iterdir():
                    if item.is_file():
                        # Check if file should be included based on extensions
                        if self.file_handler._is_included(item, self.config.file_extensions if self.config.file_extensions else None):
                            files_in_dir.append(item)
                        else:
                            logger.info(f"Skipping file with disallowed extension: {item}")
            except PermissionError:
                logger.warning(f"Permission denied accessing directory: {current_dir}")
                continue
            except Exception as e:
                logger.error(f"Error accessing directory {current_dir}: {e}")
                continue

            # Store files for this directory
            if files_in_dir:
                files_by_directory[current_dir] = files_in_dir
                logger.info(f"Collected {len(files_in_dir)} file(s) from directory {current_dir}")

            # Add subdirectories to queue for BFS traversal
            if self.config.watch_recursive:
                try:
                    for item in current_dir.iterdir():
                        if item.is_dir():
                            directories_to_process.append(item)
                except PermissionError:
                    logger.warning(f"Permission denied listing subdirectories of: {current_dir}")
                except Exception as e:
                    logger.error(f"Error listing subdirectories of {current_dir}: {e}")

        return files_by_directory
    
    def process_all(self) -> dict:
        """Process all files in the source directories using BFS directory traversal.

        Returns:
            Dictionary with processing statistics
        """
        logger.info("Starting file collection and processing of all directories")

        # Phase 1: Collect all files from all directories
        logger.info("Phase 1: Collecting all files from directories...")
        files_by_directory = self._collect_all_files()

        # Initialize statistics with complete total
        total_files = sum(len(files) for files in files_by_directory.values())
        logger.info(f"Found {total_files} total files across {len(files_by_directory)} directories")

        total_stats = {
            'total': total_files,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'performance': {
                'total_hash_duration': 0.0,
                'total_ocr_duration': 0.0,
                'total_classification_duration': 0.0,
                'total_embedding_duration': 0.0,
                'total_db_lookup_duration': 0.0,
                'total_db_insert_duration': 0.0,
                'avg_hash_duration': 0.0,
                'avg_ocr_duration': 0.0,
                'avg_classification_duration': 0.0,
                'avg_embedding_duration': 0.0,
                'avg_db_lookup_duration': 0.0,
                'avg_db_insert_duration': 0.0
            }
        }

        # Phase 2: Process files folder by folder
        logger.info("Phase 2: Processing files folder by folder...")

        current_file_index = 0
        def progress_callback(processed_in_batch, batch_total):
            """Progress callback to show current progress across all files."""
            nonlocal current_file_index
            current_file_index += 1
            print(f"\rProgress: {current_file_index}/{total_files} files processed", end='', flush=True)

        # Process each directory's files
        for directory_path, files_in_dir in files_by_directory.items():
            if not files_in_dir:
                continue

            logger.info(f"Processing directory: {directory_path} ({len(files_in_dir)} files)")

            # Process files in current directory in batches
            batch_stats = self.process_files_batch(files_in_dir, batch_size=10, progress_callback=progress_callback)

            # Accumulate statistics
            total_stats['processed'] += batch_stats['processed']
            total_stats['failed'] += batch_stats['failed']
            total_stats['skipped'] += batch_stats.get('skipped', 0)

            # Accumulate performance metrics
            total_stats['performance']['total_hash_duration'] += batch_stats['performance']['total_hash_duration']
            total_stats['performance']['total_ocr_duration'] += batch_stats['performance']['total_ocr_duration']
            total_stats['performance']['total_classification_duration'] += batch_stats['performance']['total_classification_duration']
            total_stats['performance']['total_db_lookup_duration'] += batch_stats['performance']['total_db_lookup_duration']
            total_stats['performance']['total_db_insert_duration'] += batch_stats['performance']['total_db_insert_duration']

        # Calculate averages
        if total_stats['processed'] > 0:
            total_stats['performance']['avg_hash_duration'] = total_stats['performance']['total_hash_duration'] / total_stats['processed']
            total_stats['performance']['avg_ocr_duration'] = total_stats['performance']['total_ocr_duration'] / total_stats['processed']
            total_stats['performance']['avg_classification_duration'] = total_stats['performance']['total_classification_duration'] / total_stats['processed']
            total_stats['performance']['avg_embedding_duration'] = total_stats['performance']['total_embedding_duration'] / total_stats['processed']
            total_stats['performance']['avg_db_lookup_duration'] = total_stats['performance']['total_db_lookup_duration'] / total_stats['processed']
            total_stats['performance']['avg_db_insert_duration'] = total_stats['performance']['total_db_insert_duration'] / total_stats['processed']

        # Print newline after progress display
        print()

        logger.info(
            f"File processing complete: {total_stats['processed']} processed, "
            f"{total_stats['failed']} failed, {total_stats['total']} total"
        )

        return total_stats
    
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

    def search(self, query: str, top_k: int = None, max_candidates: int = None, use_rag: bool = None, progress_callback=None) -> List[Dict]:
        """Perform semantic search on documents with optional RAG analysis and progress reporting.

        Args:
            query: Search query text
            top_k: Number of results to return (uses config default if None)
            max_candidates: Maximum number of candidates to retrieve before filtering (uses config default if None)
            use_rag: Whether to use RAG analysis for relevance assessment (uses config default if None)
            progress_callback: Optional callback function to report progress (message, type)

        Returns:
            List of matching documents with similarity scores and optional RAG analysis
        """
        # Use config values if not specified
        if top_k is None:
            top_k = self.config.semantic_search_top_k
        if max_candidates is None:
            max_candidates = self.config.semantic_search_max_candidates
        if use_rag is None:
            use_rag = getattr(self.config, 'semantic_search_enable_rag', True)

        threshold = self.config.semantic_search_min_threshold
        rag_threshold = getattr(self.config, 'semantic_search_rag_relevance_threshold', 0.3)
        debug_enabled = self.config.semantic_search_debug or logger.isEnabledFor(logging.DEBUG)

        # Preprocess query for better semantic search
        if progress_callback:
            progress_callback(f"Preprocessing query: '{query}'", "log")
        logger.info(f"Preprocessing query: '{query}'")
        processed_query = self._preprocess_query(query)
        logger.info(f"Query preprocessed: '{query}' -> '{processed_query}'")
        logger.info(f"Search parameters: top_k={top_k}, threshold={threshold}, rag={use_rag}, debug={debug_enabled}")

        # Check if embedding generator is available
        if not hasattr(self, 'embedding_generator') or self.embedding_generator is None:
            logger.error("Embedding generator not available")
            return []

        # Generate query embedding
        if progress_callback:
            progress_callback("Generating query embedding using AI model...", "log")
        logger.info("Generating query embedding...")
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

        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        if progress_callback:
            progress_callback(f"Generated {len(query_embedding)}-dimensional embedding vector", "log")

        # Search database
        if progress_callback:
            progress_callback("Searching document database for semantic matches...", "log")
        logger.info("Searching database for semantic matches...")
        try:
            results = self.database.search_semantic(query_embedding, top_k=top_k, threshold=threshold, max_candidates=max_candidates)
        except Exception as e:
            logger.error(f"Exception during database search: {e}")
            return []

        logger.info(f"Found {len(results)} matching documents via semantic search")
        if progress_callback:
            progress_callback(f"Found {len(results)} potential matches in database", "log")

        # Apply RAG analysis if enabled
        if use_rag and results:
            if progress_callback:
                progress_callback(f"Analyzing document relevance using AI (RAG) on {len(results)} matches...", "log")
            logger.info(f"Applying RAG analysis for relevance assessment on {len(results)} documents...")
            try:
                analyzed_results = self.rag_agent.analyze_relevance(query, results, verbose=debug_enabled)
                logger.info("RAG analysis completed, processing results...")

                # Filter results based on relevance threshold
                filtered_results = []
                for result in analyzed_results:
                    relevance_score = result.get('relevance_score', 1.0)  # Default to relevant if no score
                    if relevance_score >= rag_threshold:
                        filtered_results.append(result)

                results = filtered_results
                logger.info(f"RAG analysis completed. Filtered to {len(results)} relevant documents (threshold: {rag_threshold})")
                if progress_callback:
                    progress_callback(f"AI relevance analysis complete. Kept {len(results)} most relevant documents", "log")

            except Exception as e:
                logger.error(f"RAG analysis failed: {e}")
                # Continue with original results if RAG fails

        if results:
            logger.info(f"Returning {len(results)} search results")
            if progress_callback:
                progress_callback(f"Search completed successfully! Found {len(results)} relevant documents", "complete")
            if debug_enabled:
                relevance_info = ""
                if use_rag:
                    relevance_info = f", relevance: {results[0].get('relevance_score', 'N/A')}"
                logger.info(f"Top result: {results[0].get('filename', 'N/A')} (similarity: {results[0].get('similarity', 'N/A')}{relevance_info})")
                for i, result in enumerate(results[:3]):
                    relevance_info = ""
                    if use_rag:
                        relevance_info = f", relevance: {results[i].get('relevance_score', 'N/A')}"
                    logger.info(f"Result {i+1}: {result.get('filename', 'N/A')} (similarity: {results[i].get('similarity', 'N/A')}{relevance_info})")
            else:
                logger.info(f"Top result: {results[0].get('filename', 'N/A')} (similarity: {results[0].get('similarity', 'N/A')})")
        else:
            logger.info("No results found for the search query")
            if progress_callback:
                progress_callback("No documents found matching your search query", "complete")

        logger.info("Semantic search completed")
        return results

    def answer_question(self, query: str, top_k: int = None, progress_callback=None) -> Generator[Tuple[str, Optional[List[Dict]]], None, None]:
        """Answer a question using semantic search and RAG generation with streaming support.

        Args:
            query: User's question
            top_k: Number of documents to retrieve for context (uses config default if None)
            progress_callback: Optional callback function to report progress (message, type)

        Yields:
            Tuples of (chunk, None) for answer chunks during streaming, and (full_answer, citations) at the end
        """
        # Use config values if not specified
        if top_k is None:
            top_k = self.config.semantic_search_top_k

        debug_enabled = self.config.semantic_search_debug or logger.isEnabledFor(logging.DEBUG)

        # Step 1: Perform semantic search to find relevant documents
        if progress_callback:
            progress_callback(f"Searching for relevant documents to answer: '{query}'", "log")
        logger.info(f"Answering question: '{query}'")

        # Preprocess query
        processed_query = self._preprocess_query(query)

        # Check if embedding generator is available
        if not hasattr(self, 'embedding_generator') or self.embedding_generator is None:
            logger.error("Embedding generator not available")
            if progress_callback:
                progress_callback("Error: Embedding generator not available", "error")
            yield ("I'm unable to answer questions right now due to a configuration issue.", None)
            return

        # Generate query embedding
        if progress_callback:
            progress_callback("Generating query embedding...", "log")
        try:
            query_embedding = self.embedding_generator.generate_query_embedding(processed_query)
        except Exception as e:
            logger.error(f"Exception during query embedding generation: {e}")
            if progress_callback:
                progress_callback(f"Error generating embedding: {str(e)}", "error")
            yield ("I encountered an error while processing your question.", None)
            return

        if not query_embedding:
            logger.error("Failed to generate query embedding")
            if progress_callback:
                progress_callback("Error: Failed to generate query embedding", "error")
            yield ("I couldn't process your question. Please try again.", None)
            return

        # Search database
        if progress_callback:
            progress_callback("Searching document database...", "log")
        try:
            threshold = self.config.semantic_search_min_threshold
            max_candidates = self.config.semantic_search_max_candidates
            results = self.database.search_semantic(query_embedding, top_k=top_k, threshold=threshold, max_candidates=max_candidates)
        except Exception as e:
            logger.error(f"Exception during database search: {e}")
            if progress_callback:
                progress_callback(f"Error searching database: {str(e)}", "error")
            yield ("I encountered an error while searching for relevant documents.", None)
            return

        logger.info(f"Found {len(results)} matching documents")
        if progress_callback:
            progress_callback(f"Found {len(results)} relevant document(s)", "log")

        # Check if we have any documents
        if not results:
            if progress_callback:
                progress_callback("No relevant documents found", "log")
            yield ("I couldn't find any relevant documents to answer your question. Please try rephrasing or asking about a different topic.", None)
            return

        # Step 2: Generate answer using RAG agent
        if progress_callback:
            progress_callback(f"Generating answer from {len(results)} document(s)...", "log")
        logger.info(f"Generating answer using {len(results)} documents")

        try:
            # Use RAG agent to generate answer with streaming
            answer_generator = self.rag_agent.generate_answer(query, results, verbose=debug_enabled)
            
            full_answer = ""
            citations = None
            
            for chunk, chunk_citations in answer_generator:
                if chunk_citations is not None:
                    # This is the final yield with citations
                    citations = chunk_citations
                    full_answer = chunk
                    break
                else:
                    # This is a streaming chunk
                    full_answer += chunk
                    yield (chunk, None)

            # Yield final answer with citations
            if citations is not None:
                if progress_callback:
                    progress_callback("Answer generated successfully", "complete")
                yield (full_answer, citations)
            else:
                # Fallback if citations weren't extracted
                if progress_callback:
                    progress_callback("Answer generated (extracting citations...)", "log")
                # Try to extract citations from the full answer
                citations = self.rag_agent._extract_citations(full_answer, results)
                yield (full_answer, citations)

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            if progress_callback:
                progress_callback(f"Error generating answer: {str(e)}", "error")
            yield (f"I encountered an error while generating an answer: {str(e)}", None)
    
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
                    # Check if file is excluded before processing
                    if self.agent.file_handler._is_excluded(file_path):
                        logger.info(f"Skipping excluded file in watch mode: {file_path}")
                        return
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

