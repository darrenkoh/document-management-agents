"""Core agent workflow for document classification."""
import logging
import time
from pathlib import Path
from typing import List, Optional
from file_handler import FileHandler
from classifier import Classifier
from config import Config

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
            config.source_path,
            config.destination_path
        )
        self.classifier = Classifier(
            endpoint=config.ollama_endpoint,
            model=config.ollama_model,
            timeout=config.ollama_timeout,
            num_predict=config.ollama_num_predict,
            prompt_template=config.prompt_template
        )
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file: extract, classify, and copy.
        
        Args:
            file_path: Path to the file to process
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if already processed
            if self.file_handler.is_processed(file_path):
                logger.debug(f"Skipping already processed file: {file_path.name}")
                return True
            
            logger.info(f"Processing file: {file_path.name}")
            
            # Extract text content
            content = self.file_handler.extract_text(file_path)
            
            if not content or not content.strip():
                logger.warning(f"No content extracted from {file_path.name}, skipping")
                return False
            
            # Classify content
            category = self.classifier.classify(content, file_path.name, verbose=self.verbose)
            
            if not category:
                logger.error(f"Failed to classify {file_path.name}")
                return False
            
            # Copy file to category folder
            success = self.file_handler.move_file(file_path, category)
            
            if success:
                self.file_handler.mark_processed(file_path)
                logger.info(f"Successfully processed {file_path.name} -> {category}")
            
            return success
        
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
        
        return stats
    
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

