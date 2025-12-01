#!/usr/bin/env python3
"""Document ingestion entry point for processing and classifying documents."""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.backend.utils.config import Config
from src.backend.core.agent import DocumentAgent


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a nice millisecond string."""
    if seconds == 0.0:
        return "0ms"
    ms = seconds * 1000
    if ms < 1.0:
        return "<1ms"
    elif ms < 10.0:
        return f"{ms:.1f}ms"
    elif ms < 100.0:
        return f"{ms:.1f}ms"
    else:
        return f"{ms:.0f}ms"


def setup_logging(config: Config, verbose: bool = False):
    """Configure logging based on config.
    
    Args:
        config: Configuration object
        verbose: If True, set log level to DEBUG
    """
    if verbose:
        log_level = logging.DEBUG
    else:
        log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.log_file:
        # Ensure the log file directory exists
        log_dir = os.path.dirname(config.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(config.log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Document Classification Agent - Classify and organize files using AI'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='src/backend/config/config.yaml',
        help='Path to configuration file (default: src/backend/config/config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (shows LLM requests and responses)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Classify command
    classify_parser = subparsers.add_parser(
        'classify',
        help='Classify files in source directory once'
    )
    classify_parser.add_argument(
        '--source',
        type=str,
        nargs='+',
        help='Override source directories from config (can specify multiple)'
    )
    
    # Watch command
    watch_parser = subparsers.add_parser(
        'watch',
        help='Watch source directory and classify new files continuously'
    )
    watch_parser.add_argument(
        '--source',
        type=str,
        nargs='+',
        help='Override source directories from config (can specify multiple)'
    )
    watch_parser.add_argument(
        '--interval',
        type=int,
        help='Override polling interval from config (seconds)'
    )
    
    # Search command
    search_parser = subparsers.add_parser(
        'search',
        help='Perform semantic search on classified documents'
    )
    search_parser.add_argument(
        'query',
        type=str,
        help='Search query text'
    )
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )
    
    # Category search command
    category_parser = subparsers.add_parser(
        'category',
        help='Search documents by category'
    )
    category_parser.add_argument(
        'category',
        type=str,
        help='Category to search for'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override source paths if provided
        if hasattr(args, 'source') and args.source:
            source_paths = [os.path.abspath(os.path.expanduser(path)) for path in args.source]
            config._config['source_paths'] = source_paths
        
        # Setup logging
        verbose = getattr(args, 'verbose', False)
        setup_logging(config, verbose=verbose)
        
        logger = logging.getLogger(__name__)
        logger.info("Document Classification Agent starting...")
        source_paths_str = ", ".join(config.source_paths)
        logger.info(f"Source paths: {source_paths_str}")
        logger.info(f"Database: {config.database_path}")
        logger.info(f"Ollama Model: {config.ollama_model}")
        if verbose:
            logger.info("Verbose logging enabled - LLM requests and responses will be logged")
        
        # Initialize agent
        agent = DocumentAgent(config, verbose=verbose)
        
        try:
            # Execute command
            if args.command == 'classify':
                stats = agent.process_all()
                print(f"\nProcessing complete:")
                print(f"  Total files: {stats['total']}")
                print(f"  Processed: {stats['processed']}")
                print(f"  Skipped (duplicates): {stats.get('skipped', 0)}")
                print(f"  Skipped (deleted): {stats.get('skipped_deleted', 0)}")
                print(f"  Failed: {stats['failed']}")

                # Display and log performance metrics
                processed_or_skipped = stats['processed'] + stats.get('skipped', 0) + stats.get('skipped_deleted', 0)
                if 'performance' in stats and processed_or_skipped > 0:
                    perf = stats['performance']
                    print(f"\nPerformance metrics (per file averages, including skipped files):")
                    print(f"  SHA256 hash: {format_duration(perf['avg_hash_duration'])}")
                    print(f"  OCR: {format_duration(perf['avg_ocr_duration'])}")
                    print(f"  Classification: {format_duration(perf['avg_classification_duration'])}")
                    print(f"  Embeddings: {format_duration(perf['avg_embedding_duration'])}")
                    print(f"  DB lookup: {format_duration(perf['avg_db_lookup_duration'])}")
                    print(f"  DB insert: {format_duration(perf['avg_db_insert_duration'])}")
                    total_avg = (perf['avg_hash_duration'] + perf['avg_ocr_duration'] +
                               perf['avg_classification_duration'] + perf['avg_embedding_duration'] +
                               perf['avg_db_lookup_duration'] + perf['avg_db_insert_duration'])
                    print(f"  Total per file: {format_duration(total_avg)}")

                    # Log performance metrics
                    logger.info(f"Performance metrics - Processed {processed_or_skipped} files (including skipped):")
                    logger.info(f"  SHA256 hash: {format_duration(perf['avg_hash_duration'])}")
                    logger.info(f"  OCR: {format_duration(perf['avg_ocr_duration'])}")
                    logger.info(f"  Classification: {format_duration(perf['avg_classification_duration'])}")
                    logger.info(f"  Embeddings: {format_duration(perf['avg_embedding_duration'])}")
                    logger.info(f"  DB lookup: {format_duration(perf['avg_db_lookup_duration'])}")
                    logger.info(f"  DB insert: {format_duration(perf['avg_db_insert_duration'])}")
                    logger.info(f"  Total per file: {format_duration(total_avg)}")

                if stats['failed'] > 0:
                    sys.exit(1)
            
            elif args.command == 'watch':
                agent.watch(interval=getattr(args, 'interval', None))
            
            elif args.command == 'search':
                results = agent.search(args.query, top_k=args.top_k)
                print(f"\nSearch results for '{args.query}':")
                print(f"  Found {len(results)} documents\n")
                for i, doc in enumerate(results, 1):
                    print(f"{i}. {doc.get('filename', 'unknown')}")
                    print(f"   Categories: {doc.get('categories', 'N/A')}")
                    print(f"   Similarity: {doc.get('similarity', 0):.3f}")
                    print(f"   Path: {doc.get('file_path', 'N/A')}")
                    preview = doc.get('content_preview', '')
                    if preview:
                        print(f"   Preview: {preview[:100]}...")
                    print()
            
            elif args.command == 'category':
                results = agent.search_by_category(args.category)
                print(f"\nDocuments in category '{args.category}':")
                print(f"  Found {len(results)} documents\n")
                for i, doc in enumerate(results, 1):
                    print(f"{i}. {doc.get('filename', 'unknown')}")
                    print(f"   Categories: {doc.get('categories', 'N/A')}")
                    print(f"   Path: {doc.get('file_path', 'N/A')}")
                    preview = doc.get('content_preview', '')
                    if preview:
                        print(f"   Preview: {preview[:100]}...")
                    print()
        
        finally:
            agent.close()
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        logging.exception("Unexpected error occurred")
        sys.exit(1)


if __name__ == '__main__':
    main()

