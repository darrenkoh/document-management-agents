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
        self.file_handler = FileHandler(config.source_path)
        self.classifier = Classifier(
            endpoint=config.ollama_endpoint,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict,
            prompt_template=config.prompt_template
        )
        self.database = DocumentDatabase(config.database_path)
        self.embedding_generator = EmbeddingGenerator(
            endpoint=config.ollama_endpoint,
            model=config.ollama_embedding_model,
            timeout=config.ollama_timeout
        )
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file: extract, classify, generate embeddings, and store in database.

        Args:
            file_path: Path to the file to process

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate file hash first
            file_hash = self.file_handler.generate_file_hash(file_path)
            if not file_hash:
                logger.error(f"Failed to generate hash for {file_path.name}, skipping")
                return False

            # Check if already processed by hash (content-based duplicate detection)
            existing = self.database.get_document_by_hash(file_hash)
            if existing:
                logger.debug(f"Skipping already processed file (duplicate content): {file_path.name}")
                logger.debug(f"Original file: {existing.get('filename', 'Unknown')}")
                return True

            logger.info(f"Processing file: {file_path.name} (hash: {file_hash[:16]}...)")

            # Extract text content
            content = self.file_handler.extract_text(file_path)

            if not content or not content.strip():
                logger.warning(f"No content extracted from {file_path.name}, skipping")
                return False
            
            # Classify content
            categories = self.classifier.classify(content, file_path.name, verbose=self.verbose)
            
            if not categories:
                logger.error(f"Failed to classify {file_path.name}")
                return False
            
            # Generate embedding for semantic search
            logger.info(f"Generating embedding for {file_path.name}...")
            embedding = self.embedding_generator.generate_embedding(content)

            if not embedding:
                logger.error(f"Failed to generate embedding for {file_path.name}")
                return False

            # Prepare metadata
            metadata = {
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'file_extension': file_path.suffix,
                'file_modified': file_path.stat().st_mtime if file_path.exists() else None
            }

            # Store in database
            doc_id = self.database.store_classification(
                file_path=str(file_path),
                content=content,
                categories=categories,
                metadata=metadata,
                file_hash=file_hash
            )

            # Store embedding
            self.database.store_embedding(str(file_path), embedding)
            
            # Export to JSON file
            self.database.export_to_json(self.config.json_export_path)
            
            logger.info(f"Successfully processed {file_path.name} -> {categories} (DB ID: {doc_id})")
            return True
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
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
        
        stats = {
            'total': len(files),
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for file_path in files:
            try:
                if self.process_file(file_path):
                    stats['processed'] += 1
                else:
                    stats['failed'] += 1
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}")
                stats['failed'] += 1
        
        logger.info(
            f"Processing complete: {stats['processed']} processed, "
            f"{stats['failed']} failed, {stats['total']} total"
        )
        
        # Final JSON export
        self.database.export_to_json(self.config.json_export_path)
        logger.info(f"Classification results exported to {self.config.json_export_path}")
        
        return stats
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Perform semantic search on documents.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of matching documents with similarity scores
        """
        logger.info(f"Performing semantic search for: '{query}'")

        # Check if embedding generator is available
        if not hasattr(self, 'embedding_generator') or self.embedding_generator is None:
            logger.error("Embedding generator not available")
            return []

        # Generate query embedding
        logger.debug("Generating query embedding...")
        try:
            query_embedding = self.embedding_generator.generate_query_embedding(query)
        except Exception as e:
            logger.error(f"Exception during query embedding generation: {e}")
            return []

        if not query_embedding:
            logger.error("Failed to generate query embedding - returned None")
            return []

        if not isinstance(query_embedding, list) or len(query_embedding) == 0:
            logger.error(f"Invalid query embedding: {type(query_embedding)}, length: {len(query_embedding) if isinstance(query_embedding, list) else 'N/A'}")
            return []

        logger.debug(f"Generated query embedding with {len(query_embedding)} dimensions")

        # Search database
        logger.debug("Searching database for semantic matches...")
        try:
            results = self.database.search_semantic(query_embedding, top_k=top_k)
        except Exception as e:
            logger.error(f"Exception during database search: {e}")
            return []

        logger.info(f"Found {len(results)} matching documents")
        if results:
            logger.debug(f"Top result: {results[0].get('filename', 'N/A')} (similarity: {results[0].get('similarity', 'N/A')})")
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
        
        logger.info(f"Starting watch mode on {self.config.source_path}")
        logger.info(f"Polling interval: {interval or self.config.watch_interval} seconds")
        
        event_handler = WatchFileHandler(self)
        observer = Observer()
        observer.schedule(
            event_handler,
            self.config.source_path,
            recursive=self.config.watch_recursive
        )
        
        observer.start()
        
        try:
            while True:
                time.sleep(interval or self.config.watch_interval)
        except KeyboardInterrupt:
            logger.info("Stopping watch mode...")
            observer.stop()
        
        observer.join()
        logger.info("Watch mode stopped")

