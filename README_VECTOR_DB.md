# Vector Database Architecture for Document Management

## Overview

This document describes the refactored semantic search architecture that improves performance and scalability by replacing JSON-based embedding storage with dedicated vector databases.

## Architecture Changes

### Before (Legacy)
- **Storage**: Embeddings stored in TinyDB JSON file alongside document metadata
- **Search**: Load ALL embeddings into memory for each search, compute cosine similarity sequentially
- **Performance**: O(n) time complexity, high memory usage, slow with large document collections

### After (New Architecture)
- **Storage**: Document metadata in TinyDB, embeddings in dedicated vector database
- **Search**: Indexed vector search with approximate nearest neighbor algorithms
- **Performance**: Sub-linear time complexity, low memory usage, scalable to thousands of documents

## Vector Database Options

### ChromaDB (Recommended)
```yaml
database:
  vector_store:
    type: "chromadb"
    persist_directory: "vector_store"
    collection_name: "documents"
    dimension: 768
```

**Pros:**
- Easy setup and configuration
- Automatic persistence to disk
- RESTful API available if needed
- Good Python integration
- Handles metadata alongside vectors

**Cons:**
- Slightly slower than FAISS for very large datasets
- Requires more disk space

### FAISS (High Performance)
```yaml
database:
  vector_store:
    type: "faiss"
    persist_directory: "vector_store"
    dimension: 768
```

**Pros:**
- Extremely fast search performance
- Low memory footprint
- Industry standard for vector search

**Cons:**
- More complex setup
- Limited metadata support (requires separate storage)
- Index rebuilding required for deletions

## Installation

1. Install vector database dependencies:
```bash
pip install chromadb faiss-cpu
```

2. Update your `config.yaml` with vector database settings

3. Run migration script to convert existing embeddings:
```bash
python migrate_embeddings.py
```

## Configuration

Add to your `config.yaml`:

```yaml
database:
  vector_store:
    type: "chromadb"  # or "faiss"
    persist_directory: "vector_store"
    collection_name: "documents"
    dimension: 768  # Match your embedding model dimension
```

## Migration Process

The migration script (`migrate_embeddings.py`) will:

1. Read existing embeddings from TinyDB
2. Create vector database collection/index
3. Transfer embeddings with metadata
4. Update TinyDB documents to mark embeddings as migrated
5. Remove embedding data from JSON to save space

Run migration:
```bash
python migrate_embeddings.py [config_path]
```

## Performance Improvements

### Before Migration
- Search time: O(n) - proportional to number of documents
- Memory usage: High - loads all embeddings for each search
- Scalability: Poor - performance degrades linearly with document count

### After Migration
- Search time: O(log n) - sub-linear with indexing
- Memory usage: Low - vector database handles memory efficiently
- Scalability: Excellent - handles thousands of documents efficiently

### Benchmark Results (Estimated)

| Documents | Legacy Search | Vector DB Search | Improvement |
|-----------|---------------|------------------|-------------|
| 100       | 50ms         | 10ms            | 5x faster   |
| 1,000     | 500ms        | 15ms            | 33x faster  |
| 10,000    | 5s          | 50ms            | 100x faster |

## API Changes

### Backward Compatibility
- All existing APIs remain functional
- Automatic fallback to TinyDB if vector database unavailable
- No breaking changes to existing code

### New Features
- Batch processing for better performance
- Configurable vector database backends
- Migration tools for existing data

## File Structure

```
├── vector_store.py          # Vector database abstraction layer
├── database.py              # Updated DocumentDatabase with vector support
├── agent.py                 # Updated DocumentAgent with batch processing
├── migrate_embeddings.py    # Migration script for existing data
├── config.yaml              # Updated configuration
└── vector_store/            # Vector database persistence directory
    ├── chroma/             # ChromaDB data
    └── faiss/              # FAISS index files
```

## Troubleshooting

### Vector Database Not Available
If vector database fails to initialize, the system automatically falls back to TinyDB-based search with a warning log.

### Migration Issues
- Ensure old embeddings exist in TinyDB before migration
- Check vector database permissions for persistence directory
- Verify embedding dimensions match configuration

### Performance Issues
- For ChromaDB: Consider increasing `hnsw:space` parameter for larger datasets
- For FAISS: Experiment with different index types (IVF, HNSW) for optimal performance

## Future Enhancements

1. **Hybrid Search**: Combine semantic search with keyword matching
2. **Filtering**: Search within specific categories or date ranges
3. **Distributed Search**: Support for multiple vector database instances
4. **Real-time Updates**: Incremental index updates without full rebuilds

## Migration Checklist

- [ ] Backup existing `documents.json`
- [ ] Install vector database dependencies
- [ ] Update `config.yaml` with vector store settings
- [ ] Test vector database initialization
- [ ] Run migration script
- [ ] Verify search functionality
- [ ] Monitor performance improvements
