#!/usr/bin/env python3
"""Migrate from TinyDB to SQLite for better performance."""
import json
import logging
from pathlib import Path
from typing import Dict, Any
import time

from database_sqlite import SQLiteDocumentDatabase

logger = logging.getLogger(__name__)


def migrate_tinydb_to_sqlite(tinydb_path: str = "documents.json",
                           sqlite_path: str = "documents.db",
                           backup_original: bool = True) -> bool:
    """Migrate documents from TinyDB JSON to SQLite database.

    Args:
        tinydb_path: Path to the TinyDB JSON file
        sqlite_path: Path for the new SQLite database
        backup_original: Whether to backup the original JSON file

    Returns:
        True if migration successful
    """
    tinydb_file = Path(tinydb_path)
    sqlite_file = Path(sqlite_path)

    if not tinydb_file.exists():
        logger.info(f"TinyDB file {tinydb_path} does not exist, nothing to migrate")
        return True

    if sqlite_file.exists():
        logger.warning(f"SQLite database {sqlite_path} already exists, skipping migration")
        return False

    try:
        # Backup original file
        if backup_original:
            backup_path = tinydb_file.with_suffix('.json.backup')
            tinydb_file.rename(backup_path)
            logger.info(f"Backed up {tinydb_path} to {backup_path}")

            # Read from backup
            source_file = backup_path
        else:
            source_file = tinydb_file

        # Read TinyDB JSON
        logger.info("Reading TinyDB JSON file...")
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents_table = data.get('documents', {})
        if not documents_table:
            logger.warning("No documents table found in TinyDB JSON")
            return False

        # Initialize SQLite database
        logger.info("Initializing SQLite database...")
        sqlite_db = SQLiteDocumentDatabase(sqlite_path)

        # Migrate documents
        logger.info(f"Migrating {len(documents_table)} documents...")
        migrated_count = 0
        start_time = time.time()

        for doc_id, doc_data in documents_table.items():
            try:
                # Convert TinyDB format to SQLite format
                sqlite_doc = {
                    'file_path': doc_data.get('file_path', ''),
                    'filename': doc_data.get('filename', ''),
                    'content': doc_data.get('content', ''),
                    'content_preview': doc_data.get('content_preview'),
                    'categories': doc_data.get('categories', ''),
                    'classification_date': doc_data.get('classification_date', ''),
                    'metadata': doc_data.get('metadata', {}),
                    'file_hash': doc_data.get('file_hash'),
                    'embedding_stored': doc_data.get('embedding_stored', False)
                }

                # Store in SQLite (this will create new IDs)
                sqlite_db.store_classification(**sqlite_doc)
                migrated_count += 1

                if migrated_count % 100 == 0:
                    logger.info(f"Migrated {migrated_count}/{len(documents_table)} documents...")

            except Exception as e:
                logger.error(f"Error migrating document {doc_id}: {e}")
                continue

        migration_time = time.time() - start_time
        logger.info(f"Successfully migrated {migrated_count} documents in {migration_time:.2f}s")
        logger.info(".2f")

        # Show database stats
        stats = sqlite_db.get_stats()
        logger.info(f"New database stats: {stats}")

        sqlite_db.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def benchmark_performance(sqlite_db: SQLiteDocumentDatabase, iterations: int = 100) -> Dict[str, float]:
    """Benchmark SQLite database performance vs TinyDB expectations."""
    logger.info(f"Benchmarking SQLite performance with {iterations} iterations...")

    # Get some sample hashes for testing
    docs = sqlite_db.get_all_documents()
    if not docs:
        logger.warning("No documents in database for benchmarking")
        return {}

    sample_hashes = [doc['file_hash'] for doc in docs[:min(10, len(docs))] if doc.get('file_hash')]

    if not sample_hashes:
        logger.warning("No file hashes found for benchmarking")
        return {}

    # Benchmark hash lookups
    times = []
    for i in range(iterations):
        hash_to_find = sample_hashes[i % len(sample_hashes)]
        start = time.time()
        result = sqlite_db.get_document_by_hash(hash_to_find)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    logger.info(".6f")
    logger.info(".6f")
    logger.info(".6f")

    return {
        'avg_lookup_time': avg_time,
        'min_lookup_time': min_time,
        'max_lookup_time': max_time,
        'expected_tinydb_time': avg_time * 10  # TinyDB scales linearly
    }


def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate from TinyDB to SQLite")
    parser.add_argument('--tinydb', default='documents.json', help='TinyDB JSON file path')
    parser.add_argument('--sqlite', default='documents.db', help='SQLite database path')
    parser.add_argument('--no-backup', action='store_true', help='Skip backing up original file')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark after migration')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Starting TinyDB to SQLite migration...")
    logger.info(f"Source: {args.tinydb}")
    logger.info(f"Destination: {args.sqlite}")

    success = migrate_tinydb_to_sqlite(args.tinydb, args.sqlite, not args.no_backup)

    if success and args.benchmark:
        logger.info("Running performance benchmark...")
        sqlite_db = SQLiteDocumentDatabase(args.sqlite)
        benchmark_performance(sqlite_db)
        sqlite_db.close()

    if success:
        logger.info("Migration completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Update your config.yaml to use 'documents.db' instead of 'documents.json'")
        logger.info("2. Update your code to import SQLiteDocumentDatabase instead of DocumentDatabase")
        logger.info("3. Test your application to ensure everything works")
    else:
        logger.error("Migration failed!")
        exit(1)


if __name__ == '__main__':
    main()
