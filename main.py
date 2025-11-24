#!/usr/bin/env python3
"""Main entry point for the document classification agent."""
import argparse
import logging
import os
import sys
from pathlib import Path
from config import Config
from agent import DocumentAgent


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
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
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
        help='Override source directory from config'
    )
    classify_parser.add_argument(
        '--output',
        type=str,
        help='Override output/destination directory from config'
    )
    
    # Watch command
    watch_parser = subparsers.add_parser(
        'watch',
        help='Watch source directory and classify new files continuously'
    )
    watch_parser.add_argument(
        '--source',
        type=str,
        help='Override source directory from config'
    )
    watch_parser.add_argument(
        '--output',
        type=str,
        help='Override output/destination directory from config'
    )
    watch_parser.add_argument(
        '--interval',
        type=int,
        help='Override polling interval from config (seconds)'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Override source path if provided
        if args.source:
            source_path = os.path.expanduser(args.source)
            config._config['source_path'] = os.path.abspath(source_path)
        
        # Override destination path if provided
        if hasattr(args, 'output') and args.output:
            dest_path = os.path.expanduser(args.output)
            config._config['destination_path'] = os.path.abspath(dest_path)
        
        # Setup logging
        verbose = getattr(args, 'verbose', False)
        setup_logging(config, verbose=verbose)
        
        logger = logging.getLogger(__name__)
        logger.info("Document Classification Agent starting...")
        logger.info(f"Source: {config.source_path}")
        logger.info(f"Destination: {config.destination_path}")
        logger.info(f"Ollama Model: {config.ollama_model}")
        if verbose:
            logger.info("Verbose logging enabled - LLM requests and responses will be logged")
        
        # Initialize agent
        agent = DocumentAgent(config, verbose=verbose)
        
        # Execute command
        if args.command == 'classify':
            stats = agent.process_all()
            print(f"\nProcessing complete:")
            print(f"  Total files: {stats['total']}")
            print(f"  Processed: {stats['processed']}")
            print(f"  Failed: {stats['failed']}")
            
            if stats['failed'] > 0:
                sys.exit(1)
        
        elif args.command == 'watch':
            agent.watch(interval=args.interval)
    
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

