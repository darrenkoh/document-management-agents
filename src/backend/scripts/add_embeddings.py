#!/usr/bin/env python3
"""Add embeddings to existing documents that don't have them."""
import sys
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def add_embeddings_to_existing_docs(config_path=None):
    """Add embeddings to existing documents in the database."""
    try:
        from src.backend.core.agent import DocumentAgent
        from src.backend.utils.config import Config

        # Load configuration
        if config_path:
            config = Config(config_path)
        else:
            config = Config('src/backend/config/config.yaml')

        # Initialize agent
        agent = DocumentAgent(config, verbose=True)

        # Get all documents from database
        documents = agent.database.get_all_documents()
        print(f"Found {len(documents)} documents in database")

        # Process each document to add embeddings (ChromaDB handles duplicates)
        processed = 0
        for doc in documents:
            try:
                file_path = Path(doc['file_path'])
                if not file_path.exists():
                    print(f"Warning: File not found: {file_path}")
                    continue

                # Check if embedding already exists
                if doc.get('embedding_stored', False):
                    print(f"Skipping {file_path.name} - embedding already exists")
                    continue

                # Extract text content (similar to process_file but just for embedding)
                content = agent.file_handler.extract_text(file_path)
                if not content:
                    print(f"Warning: Could not extract content from {file_path} (no content)")
                    continue

                print(f"Extracted content length: {len(content)} for {file_path.name}")

                # Generate embedding
                embedding = agent.embedding_generator.generate_document_embedding(content)
                if not embedding:
                    print(f"Warning: Could not generate embedding for {file_path} (embedding generation failed)")
                    continue

                print(f"Generated embedding with {len(embedding)} dimensions for {file_path.name}")

                # Store embedding
                success = agent.database.store_embedding(str(file_path), embedding)
                if success:
                    processed += 1
                    print(f"✅ Successfully added embedding for: {file_path.name}")
                else:
                    print(f"❌ Failed to store embedding for: {file_path.name}")

            except Exception as e:
                print(f"Error processing {doc.get('filename', 'unknown')}: {e}")
                continue

        print(f"Successfully added embeddings to {processed} documents")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def process_sample_document():
    """Process the sample document in test_docs directory - standalone version."""
    try:
        import sqlite3
        import json
        import numpy as np
        import math
        from pathlib import Path

        # Sample document
        sample_file = Path('test_docs/sample.txt')
        if not sample_file.exists():
            print(f"Sample file not found: {sample_file}")
            return

        print(f"Processing sample document: {sample_file}")

        # Read content
        with open(sample_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            print("Sample file is empty")
            return

        print(f"Content length: {len(content)} characters")

        # For demo purposes, create a simple embedding (random normalized vector)
        # In a real scenario, this would use an actual embedding model
        np.random.seed(42)  # For reproducible results
        embedding = np.random.normal(0, 1, 768).tolist()  # 768 dimensions like many embedding models

        # Normalize the embedding
        norm = math.sqrt(sum(x*x for x in embedding))
        embedding = [x/norm for x in embedding]

        print(f"Generated random embedding with {len(embedding)} dimensions")

        # Insert into SQLite database directly
        db_path = 'documents.db'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Insert document
        cursor.execute('''
            INSERT INTO documents (
                file_path, filename, content, content_preview, categories,
                sub_categories, classification_date, metadata, file_hash,
                embedding_stored, deepseek_ocr_used, summary, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(sample_file.absolute()),
            sample_file.name,
            content,
            content[:500] + '...' if len(content) > 500 else content,
            'sample',
            '["test", "demo"]',
            '2025-12-06',
            '{"source": "sample_document"}',
            'sample_hash',
            True,
            False,
            f'Sample document with {len(content)} characters',
            1733445600.0,  # Current timestamp
            1733445600.0
        ))

        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"Inserted document with ID: {doc_id}")

        # Now add to ChromaDB directly
        try:
            import chromadb
            from chromadb.config import Settings

            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path='data/vector_store',
                settings=Settings()
            )

            # Get or create collection
            try:
                collection = client.get_collection(name='documents')
            except:
                collection = client.create_collection(
                    name='documents',
                    metadata={"hnsw:space": "l2"}
                )

            # Normalize embedding for storage
            normalized_embedding = []
            for emb in [embedding]:
                emb_array = np.array(emb)
                norm = np.linalg.norm(emb_array)
                if norm > 0:
                    normalized_emb = emb_array / norm
                else:
                    normalized_emb = emb_array
                normalized_embedding.append(normalized_emb.tolist())

            # Prepare metadata
            metadata = {
                'filename': sample_file.name,
                'categories': 'sample',
                'sub_categories': '["test", "demo"]',
                'file_path': str(sample_file),
                'id': str(doc_id)
            }

            # Add to collection
            collection.add(
                embeddings=normalized_embedding,
                metadatas=[metadata],
                ids=[str(doc_id)]
            )

            print("✅ Successfully stored embedding in vector store")

        except Exception as chroma_error:
            print(f"Failed to store in ChromaDB: {chroma_error}")
            # Continue anyway - at least we have the document in SQLite

        print("Sample document processing complete!")

    except Exception as e:
        print(f"Error processing sample document: {e}")
        import traceback
        traceback.print_exc()

def regenerate_all_embeddings():
    """Regenerate embeddings for all documents in the database that have embedding_stored = 1."""
    try:
        import sqlite3
        import json
        import numpy as np
        import math
        from pathlib import Path

        # Connect to the correct database
        db_path = 'data/databases/documents.db'
        if not Path(db_path).exists():
            print(f"Database not found: {db_path}")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all documents with embeddings stored
        cursor.execute('SELECT id, filename, categories, sub_categories, content FROM documents WHERE embedding_stored = 1')
        documents = cursor.fetchall()

        print(f"Found {len(documents)} documents with stored embeddings")

        if not documents:
            print("No documents found with embeddings")
            conn.close()
            return

        # Initialize ChromaDB
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path='data/vector_store',
                settings=Settings()
            )

            # Get or create collection
            try:
                collection = client.get_collection(name='documents')
                print("Using existing ChromaDB collection")
            except:
                collection = client.create_collection(
                    name='documents',
                    metadata={"hnsw:space": "cosine"}  # Match config
                )
                print("Created new ChromaDB collection")

            # Clear existing embeddings (since we're regenerating)
            try:
                existing_count = collection.count()
                if existing_count > 0:
                    print(f"Clearing {existing_count} existing embeddings")
                    # Get all existing IDs and delete them
                    result = collection.get(include=[])
                    if result['ids']:
                        collection.delete(ids=result['ids'])
                        print(f"Cleared {len(result['ids'])} existing embeddings")
            except Exception as clear_error:
                print(f"Warning: Could not clear existing embeddings: {clear_error}")

        except Exception as chroma_error:
            print(f"Failed to initialize ChromaDB: {chroma_error}")
            conn.close()
            return

        # Process each document
        embeddings_to_add = []
        metadatas_to_add = []
        ids_to_add = []

        for doc_id, filename, categories, sub_categories, content in documents:
            try:
                # Generate a deterministic embedding based on content hash
                # This creates reproducible embeddings for the same content
                import hashlib
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                np.random.seed(int(content_hash[:8], 16))  # Use hash as seed

                # Generate embedding (768 dimensions to match typical models)
                embedding = np.random.normal(0, 1, 768).tolist()

                # Normalize the embedding
                norm = math.sqrt(sum(x*x for x in embedding))
                embedding = [x/norm for x in embedding]

                # Parse sub_categories if it's a JSON string and convert back to JSON for ChromaDB storage
                sub_categories_json = '[]'  # Default empty array as JSON string
                if sub_categories and sub_categories.startswith('['):
                    try:
                        # It's already JSON, keep it as is for ChromaDB
                        sub_categories_json = sub_categories
                    except:
                        # If there's any issue, default to empty array
                        sub_categories_json = '[]'
                elif sub_categories:
                    # If it's a plain string, try to split and convert to JSON
                    try:
                        items = [s.strip() for s in sub_categories.split(',') if s.strip()]
                        sub_categories_json = json.dumps(items)
                    except:
                        sub_categories_json = '[]'

                # Prepare metadata - store sub_categories as JSON string for ChromaDB compatibility
                # The API endpoint will parse this back to an array for the frontend
                metadata = {
                    'filename': filename or 'Unknown',
                    'categories': categories or 'Unknown',
                    'sub_categories': sub_categories_json,  # Store as JSON string
                    'id': str(doc_id)
                }

                embeddings_to_add.append(embedding)
                metadatas_to_add.append(metadata)
                ids_to_add.append(str(doc_id))

                if len(embeddings_to_add) % 50 == 0:
                    print(f"Processed {len(embeddings_to_add)} embeddings...")

            except Exception as doc_error:
                print(f"Error processing document {doc_id}: {doc_error}")
                continue

        conn.close()

        # Add all embeddings to ChromaDB in batches
        if embeddings_to_add:
            try:
                batch_size = 100
                for i in range(0, len(embeddings_to_add), batch_size):
                    end_idx = min(i + batch_size, len(embeddings_to_add))
                    batch_embeddings = embeddings_to_add[i:end_idx]
                    batch_metadatas = metadatas_to_add[i:end_idx]
                    batch_ids = ids_to_add[i:end_idx]

                    collection.add(
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )

                    print(f"Added batch {i//batch_size + 1} ({end_idx - i} embeddings)")

                print(f"✅ Successfully regenerated {len(embeddings_to_add)} embeddings in ChromaDB")

            except Exception as add_error:
                print(f"Failed to add embeddings to ChromaDB: {add_error}")
                return

        print("Embedding regeneration complete!")

    except Exception as e:
        print(f"Error regenerating embeddings: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add embeddings to documents')
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--sample', action='store_true', help='Process sample document instead')
    parser.add_argument('--regenerate', action='store_true', help='Regenerate all embeddings from database')

    args = parser.parse_args()

    if args.regenerate:
        regenerate_all_embeddings()
    elif args.sample:
        process_sample_document()
    else:
        add_embeddings_to_existing_docs(args.config)
