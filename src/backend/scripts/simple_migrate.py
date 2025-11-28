#!/usr/bin/env python3
"""Simple migration from TinyDB JSON to SQLite database."""
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_documents(tinydb_path="documents.json", sqlite_path="documents.db"):
    """Migrate documents from TinyDB JSON to SQLite."""

    # Check if TinyDB file exists
    tinydb_file = Path(tinydb_path)
    if not tinydb_file.exists():
        logger.error(f"TinyDB file {tinydb_path} not found!")
        return False

    # Check if SQLite already exists
    sqlite_file = Path(sqlite_path)
    if sqlite_file.exists():
        logger.error(f"SQLite database {sqlite_path} already exists! Delete it first if you want to migrate.")
        return False

    try:
        # Read TinyDB JSON
        logger.info(f"Reading {tinydb_path}...")
        with open(tinydb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        documents_table = data.get('documents', {})
        if not documents_table:
            logger.warning("No documents table found in TinyDB JSON")
            return False

        logger.info(f"Found {len(documents_table)} documents to migrate")

        # Create SQLite database
        logger.info(f"Creating SQLite database {sqlite_path}...")
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()

        # Enable WAL mode for better performance
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")

        # Create table
        cursor.execute('''
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                content TEXT NOT NULL,
                content_preview TEXT,
                categories TEXT,
                classification_date TEXT,
                metadata TEXT,
                file_hash TEXT,
                embedding_stored BOOLEAN DEFAULT FALSE,
                created_at REAL,
                updated_at REAL
            )
        ''')

        # Create indexes for fast lookups
        cursor.execute('CREATE INDEX idx_file_hash ON documents(file_hash)')
        cursor.execute('CREATE INDEX idx_file_path ON documents(file_path)')
        cursor.execute('CREATE INDEX idx_categories ON documents(categories)')

        # Migrate documents
        migrated = 0
        now = datetime.now().timestamp()

        for doc_id, doc_data in documents_table.items():
            try:
                # Prepare document data
                doc = {
                    'file_path': doc_data.get('file_path', ''),
                    'filename': doc_data.get('filename', ''),
                    'content': doc_data.get('content', ''),
                    'content_preview': doc_data.get('content_preview'),
                    'categories': doc_data.get('categories', ''),
                    'classification_date': doc_data.get('classification_date', ''),
                    'metadata': json.dumps(doc_data.get('metadata', {})),
                    'file_hash': doc_data.get('file_hash'),
                    'embedding_stored': doc_data.get('embedding_stored', False),
                    'created_at': now,
                    'updated_at': now
                }

                # Insert document
                columns = ', '.join(doc.keys())
                placeholders = ', '.join('?' * len(doc))
                values = list(doc.values())

                cursor.execute(f'INSERT INTO documents ({columns}) VALUES ({placeholders})', values)
                migrated += 1

                if migrated % 100 == 0:
                    logger.info(f"Migrated {migrated}/{len(documents_table)} documents...")

            except Exception as e:
                logger.error(f"Error migrating document {doc_id}: {e}")
                continue

        conn.commit()

        # Verify migration
        cursor.execute('SELECT COUNT(*) FROM documents')
        final_count = cursor.fetchone()[0]

        conn.close()

        logger.info("✅ Migration completed successfully!")
        logger.info(f"   📊 Migrated: {migrated} documents")
        logger.info(f"   📁 Database: {sqlite_path}")
        logger.info(f"   📏 Size: {sqlite_file.stat().st_size / 1024:.0f} KB")

        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        tinydb_file = sys.argv[1]
    else:
        tinydb_file = "documents.json"

    if len(sys.argv) > 2:
        sqlite_file = sys.argv[2]
    else:
        sqlite_file = "documents.db"

    logger.info(f"Migrating from {tinydb_file} to {sqlite_file}")

    if migrate_documents(tinydb_file, sqlite_file):
        logger.info("\n🎉 Migration successful!")
        logger.info("\nNext steps:")
        logger.info("1. Update your config.yaml database path to 'documents.db'")
        logger.info("2. Update your code to use SQLiteDocumentDatabase")
        logger.info("3. Test your application")
        logger.info("4. Optionally delete the old documents.json file")
    else:
        logger.error("\n❌ Migration failed!")
        sys.exit(1)
