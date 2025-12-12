#!/usr/bin/env python3
"""Document Classification Agent - Web App Interface."""
import os
import sys
import argparse
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.utils.config import Config
from src.backend.database.database_sqlite_standalone import SQLiteDocumentDatabase
from src.backend.core.agent import DocumentAgent
from src.backend.services.file_handler import FileHandler
import math

ITEMS_PER_PAGE = 20

# Parse command line arguments
parser = argparse.ArgumentParser(description='Document Classification Agent - Web Interface')
parser.add_argument('--config', type=str, default='src/backend/config/config.yaml',
                   help='Path to configuration file (default: src/backend/config/config.yaml)')
parser.add_argument('--port', type=int,
                   help='Port to run the web app on (overrides config)')
parser.add_argument('--host', type=str,
                   help='Host to bind the web app to (overrides config)')
parser.add_argument('--debug', action='store_true', default=None,
                   help='Enable debug mode (overrides config)')
parser.add_argument('--no-debug', action='store_true',
                   help='Disable debug mode (overrides config)')
parser.add_argument('--verbose', action='store_true', default=None,
                   help='Enable verbose logging for debugging (overrides config)')
parser.add_argument('--no-verbose', action='store_true',
                   help='Disable verbose logging (overrides config)')

args = parser.parse_args()

# Initialize configuration
config = Config(args.config)

# Override config with command line arguments
if args.port:
    config._config.setdefault('webapp', {})['port'] = args.port
if args.host:
    config._config.setdefault('webapp', {})['host'] = args.host
if args.debug is True:
    config._config.setdefault('webapp', {})['debug'] = True
if args.no_debug:
    config._config.setdefault('webapp', {})['debug'] = False
if args.verbose is True:
    config._config.setdefault('logging', {})['level'] = 'DEBUG'
if args.no_verbose:
    config._config.setdefault('logging', {})['level'] = 'INFO'

# API-only backend (frontend is served separately by Node/Vite)
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Enable CORS for API endpoints
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:5000"],
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Cache-Control"]
    }
})

# Setup logging - import setup_logging from document_ingestion.py at root
from document_ingestion import setup_logging
setup_logging(config, verbose=(config.log_level.upper() == 'DEBUG'))

# Initialize components
database = SQLiteDocumentDatabase(config.database_path)
agent = DocumentAgent(config)

# Global function to refresh database data
def refresh_database():
    """Refresh the database data from disk."""
    try:
        database.refresh()
        app.logger.info("Database refreshed successfully")
        return True
    except Exception as e:
        app.logger.error(f"Failed to refresh database: {e}")
        return False

# Debug: Check if agent has embedding generator
app.logger.info("Initializing Document Agent...")
app.logger.info(f"Agent has embedding generator: {hasattr(agent, 'embedding_generator')}")
if hasattr(agent, 'embedding_generator'):
    app.logger.info(f"Embedding endpoint: {agent.embedding_generator.endpoint}")
    app.logger.info(f"Embedding model: {agent.embedding_generator.model}")
else:
    app.logger.error("Agent does not have embedding generator!")



















@app.route('/api/document/<int:doc_id>/file/view')
@app.route('/document/<int:doc_id>/file')  # legacy alias (UI should use /api/... going forward)
def serve_original_file(doc_id: int):
    """Serve the original file for viewing."""
    # Get the document by SQLite ID
    try:
        cursor = database.connection.cursor()
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        cursor.close()

        if row:
            doc = database._row_to_dict(row)
        else:
            doc = None
    except Exception as e:
        app.logger.error(f"Error fetching document {doc_id}: {e}")
        doc = None

    # Fallback: search through all documents
    if not doc:
        all_docs = database.get_all_documents()
        for d in all_docs:
            if d.get('id') == doc_id or d.get('doc_id') == doc_id:
                doc = d
                break

    if not doc:
        return 'Document not found', 404

    file_path = Path(doc['file_path'])
    if not file_path.exists():
        return 'File not found', 404

    # Determine MIME type
    file_extension = doc['metadata'].get('file_extension', '').lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword'
    }

    mime_type = mime_types.get(file_extension, 'application/octet-stream')

    try:
        return send_file(file_path, mimetype=mime_type, as_attachment=False)
    except Exception as e:
        app.logger.error(f"Error serving file {file_path}: {e}")
        return 'Error serving file', 500


@app.route('/api/document/<int:doc_id>/file')
def api_get_document_file(doc_id: int):
    """API endpoint to download the original file as a blob."""
    # Get the document by SQLite ID
    try:
        cursor = database.connection.cursor()
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        row = cursor.fetchone()
        cursor.close()

        if row:
            doc = database._row_to_dict(row)
        else:
            doc = None
    except Exception as e:
        app.logger.error(f"Error fetching document {doc_id}: {e}")
        doc = None

    # Fallback: search through all documents
    if not doc:
        all_docs = database.get_all_documents()
        for d in all_docs:
            if d.get('id') == doc_id or d.get('doc_id') == doc_id:
                doc = d
                break

    if not doc:
        return jsonify({'error': 'Document not found'}), 404

    file_path = Path(doc['file_path'])
    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404

    # Determine MIME type
    file_extension = doc['metadata'].get('file_extension', '').lower()
    mime_types = {
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.doc': 'application/msword'
    }

    mime_type = mime_types.get(file_extension, 'application/octet-stream')

    try:
        return send_file(file_path, mimetype=mime_type, as_attachment=True, download_name=doc['filename'])
    except Exception as e:
        app.logger.error(f"Error serving file {file_path}: {e}")
        return jsonify({'error': 'Error serving file'}), 500


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for semantic search (returns JSON)."""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        # Use config default for top_k instead of hardcoded 50
        top_k = config.semantic_search_top_k
        results = agent.search(query, top_k=top_k)
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        app.logger.error(f"API search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/stream', methods=['POST'])
def api_search_stream():
    """API endpoint for streaming semantic search with real-time logs."""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query required'}), 400

    def generate():
        import json
        import time
        import queue
        import threading

        # Thread-safe queue for real-time message passing
        message_queue = queue.Queue()
        search_completed = threading.Event()
        search_results = {}
        search_error = {}

        def progress_callback(message, message_type='log'):
            """Callback that immediately queues messages for streaming"""
            message_queue.put({'type': message_type, 'message': message})

        def search_worker():
            """Background thread that performs the search"""
            try:
                app.logger.info(f"Starting search for query: {query}")
                # Use config default for top_k instead of hardcoded 50
                top_k = config.semantic_search_top_k
                results = agent.search(query, top_k=top_k, progress_callback=progress_callback)
                search_results['data'] = results
                app.logger.info(f"Search completed with {len(results)} results")
            except Exception as e:
                app.logger.error(f"Search error: {e}")
                search_error['error'] = str(e)
            finally:
                search_completed.set()

        # Start the search in a background thread
        search_thread = threading.Thread(target=search_worker, daemon=True)
        search_thread.start()

        try:
            # Send initial message immediately
            start_data = json.dumps({'type': 'start', 'message': f'Starting AI-powered semantic search for: "{query}"'})
            yield f"data: {start_data}\n\n"

            # Stream messages as they become available
            while not search_completed.is_set():
                try:
                    # Wait for a message with timeout
                    message = message_queue.get(timeout=0.1)
                    data = json.dumps(message)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # No message available, continue checking
                    continue

            # Send any remaining messages
            while not message_queue.empty():
                try:
                    message = message_queue.get_nowait()
                    data = json.dumps(message)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    break

            # Check for errors
            if search_error:
                error_data = json.dumps({'type': 'error', 'message': search_error['error']})
                yield f"data: {error_data}\n\n"
                return

            # Send completion message
            results = search_results.get('data', [])
            complete_data = json.dumps({'type': 'complete', 'results': len(results), 'message': 'Search completed successfully'})
            yield f"data: {complete_data}\n\n"

            # Send final results
            results_data = json.dumps({'type': 'results', 'data': {'query': query, 'results': results, 'count': len(results)}})
            yield f"data: {results_data}\n\n"

        except Exception as e:
            error_data = json.dumps({'type': 'error', 'message': str(e)})
            yield f"data: {error_data}\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering if used
    return response


@app.route('/api/search/answer', methods=['POST'])
def api_search_answer():
    """API endpoint for streaming question answering with real-time answer generation."""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query required'}), 400

    def generate():
        import json
        import queue
        import threading

        # Thread-safe queue for real-time message passing
        message_queue = queue.Queue()
        answer_completed = threading.Event()
        answer_result = {}
        answer_error = {}
        citations_result = {}

        def progress_callback(message, message_type='log'):
            """Callback that immediately queues messages for streaming"""
            message_queue.put({'type': message_type, 'message': message})

        def answer_worker():
            """Background thread that generates the answer"""
            try:
                app.logger.info(f"Starting answer generation for query: {query}")
                answer_text = ""
                citations = None
                
                # Use agent.answer_question which yields (chunk, citations) tuples
                # Use config default for top_k instead of hardcoded 10
                top_k = config.semantic_search_top_k
                for chunk, chunk_citations in agent.answer_question(query, top_k=top_k, progress_callback=progress_callback):
                    if chunk_citations is not None:
                        # Final chunk with citations
                        answer_text = chunk
                        citations = chunk_citations
                        break
                    else:
                        # Streaming chunk
                        answer_text += chunk
                        # Stream answer chunks to frontend
                        message_queue.put({'type': 'answer_chunk', 'chunk': chunk})
                
                answer_result['answer'] = answer_text
                citations_result['citations'] = citations or []
                app.logger.info(f"Answer generation completed. Answer length: {len(answer_text)}, Citations: {len(citations_result['citations'])}")
            except Exception as e:
                app.logger.error(f"Answer generation error: {e}")
                answer_error['error'] = str(e)
            finally:
                answer_completed.set()

        # Start the answer generation in a background thread
        answer_thread = threading.Thread(target=answer_worker, daemon=True)
        answer_thread.start()

        try:
            # Send initial message immediately
            start_data = json.dumps({'type': 'start', 'message': f'Generating answer for: "{query}"'})
            yield f"data: {start_data}\n\n"

            # Stream messages as they become available
            while not answer_completed.is_set():
                try:
                    # Wait for a message with timeout
                    message = message_queue.get(timeout=0.1)
                    data = json.dumps(message)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # No message available, continue checking
                    continue

            # Send any remaining messages
            while not message_queue.empty():
                try:
                    message = message_queue.get_nowait()
                    data = json.dumps(message)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    break

            # Check for errors
            if answer_error:
                error_data = json.dumps({'type': 'error', 'message': answer_error['error']})
                yield f"data: {error_data}\n\n"
                return

            # Send citations if available
            citations = citations_result.get('citations', [])
            if citations:
                citations_data = json.dumps({'type': 'citations', 'citations': citations})
                yield f"data: {citations_data}\n\n"

            # Send completion message with full answer
            answer = answer_result.get('answer', '')
            complete_data = json.dumps({
                'type': 'complete',
                'answer': answer,
                'citations': citations,
                'message': 'Answer generated successfully'
            })
            yield f"data: {complete_data}\n\n"

        except Exception as e:
            error_data = json.dumps({'type': 'error', 'message': str(e)})
            yield f"data: {error_data}\n\n"

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable nginx buffering if used
    return response


@app.route('/api/debug/search')
@app.route('/debug/search')  # legacy alias
def debug_search():
    """Debug endpoint to check search functionality."""
    # Refresh data before testing
    refresh_database()

    # Test category search
    try:
        category_results = agent.search_by_category('confirmation')
        category_count = len(category_results)
    except Exception as e:
        category_results = []
        category_count = f"Error: {e}"

    # Test semantic search with a simple query
    try:
        # Use config default for top_k instead of hardcoded 10
        top_k = config.semantic_search_top_k
        semantic_results = agent.search('test', top_k=top_k)
        semantic_count = len(semantic_results)
    except Exception as e:
        semantic_results = []
        semantic_count = f"Error: {e}"

    # Check database
    all_docs = database.get_all_documents()
    docs_with_embeddings = sum(1 for doc in all_docs if doc.get('embedding'))

    return jsonify({
        'database': {
            'total_documents': len(all_docs),
            'documents_with_embeddings': docs_with_embeddings
        },
        'category_search': {
            'query': 'confirmation',
            'results_count': category_count
        },
        'semantic_search': {
            'query': 'test',
            'results_count': semantic_count
        },
        'documents': [
            {
                'filename': doc.get('filename', 'N/A'),
                'has_embedding': 'embedding' in doc and doc['embedding'] is not None,
                'categories': doc.get('categories', 'N/A')
            } for doc in all_docs[:3]  # Show first 3
        ]
    })


@app.route('/api/documents')
def api_documents():
    """API endpoint to get documents with pagination."""
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', ITEMS_PER_PAGE, type=int)
    search = request.args.get('search', '').strip()
    category = request.args.get('category', '').strip()
    sub_category = request.args.get('sub_category', '').strip()

    # Refresh data to ensure we have latest documents
    refresh_database()

    all_docs = database.get_all_documents()

    # Filter by category if provided (similar to search_by_category method)
    if category:
        filtered_docs = []
        category_lower = category.lower()
        for doc in all_docs:
            categories = doc.get('categories', '').lower()
            # Use LIKE-style matching (category appears anywhere in hyphen-separated string)
            if category_lower in categories:
                filtered_docs.append(doc)
        all_docs = filtered_docs

    # Filter by sub-category if provided
    if sub_category:
        filtered_docs = []
        sub_category_lower = sub_category.lower()
        for doc in all_docs:
            sub_categories = doc.get('sub_categories', [])
            if sub_categories and any(sub_cat.lower() == sub_category_lower for sub_cat in sub_categories):
                filtered_docs.append(doc)
        all_docs = filtered_docs

    # Filter by search if provided
    if search:
        filtered_docs = []
        search_lower = search.lower()
        for doc in all_docs:
            filename = doc.get('filename', '').lower()
            categories = doc.get('categories', '').lower()
            content_preview = doc.get('content_preview', '').lower()

            if (search_lower in filename or
                search_lower in categories or
                search_lower in content_preview):
                filtered_docs.append(doc)
        all_docs = filtered_docs

    # Sort by classification date
    all_docs.sort(key=lambda x: x.get('classification_date', ''), reverse=True)

    # Pagination
    total_docs = len(all_docs)
    total_pages = math.ceil(total_docs / limit)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    docs_page = all_docs[start_idx:end_idx]

    return jsonify({
        'documents': docs_page,
        'page': page,
        'total_pages': total_pages,
        'total_documents': total_docs,
        'limit': limit
    })


@app.route('/api/document/<int:doc_id>')
def api_get_document(doc_id):
    """API endpoint to get a single document by ID."""
    try:
        # Refresh data to ensure we have latest documents
        refresh_database()

        # Get document by ID
        doc = database.get_document_by_id(doc_id)
        if not doc:
            return jsonify({'error': f'Document with ID {doc_id} not found'}), 404

        # Add doc_id to the response
        doc_dict = dict(doc)
        doc_dict['id'] = doc_id

        return jsonify(doc_dict)

    except Exception as e:
        app.logger.error(f"Error getting document {doc_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/delete', methods=['POST'])
def api_delete_documents():
    """API endpoint to delete multiple documents by IDs.
    
    Expects JSON body: { "ids": [1, 2, 3, ...] }
    Deletes from both SQLite database and vector store.
    """
    try:
        data = request.get_json()
        if not data or 'ids' not in data:
            return jsonify({'error': 'Missing "ids" field in request body'}), 400
        
        doc_ids = data['ids']
        if not isinstance(doc_ids, list):
            return jsonify({'error': '"ids" must be a list of document IDs'}), 400
        
        if not doc_ids:
            return jsonify({'error': 'No document IDs provided'}), 400
        
        # Validate all IDs are integers
        try:
            doc_ids = [int(id) for id in doc_ids]
        except (ValueError, TypeError):
            return jsonify({'error': 'All IDs must be valid integers'}), 400
        
        app.logger.info(f"Deleting {len(doc_ids)} documents: {doc_ids}")
        
        # Delete documents from database and vector store
        result = database.delete_documents(doc_ids)
        
        # Refresh database after deletion
        refresh_database()
        
        return jsonify({
            'success': True,
            'deleted_count': result['deleted_count'],
            'vector_deleted_count': result['vector_deleted_count'],
            'errors': result['errors'],
            'message': f"Successfully deleted {result['deleted_count']} document(s)"
        })
        
    except Exception as e:
        app.logger.error(f"Error deleting documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/embeddings')
def api_get_embeddings():
    """Get all document embeddings with optional PCA processing."""
    try:
        # Check if vector store is available
        if not agent.database.vector_store:
            return jsonify({'error': 'Vector store not available'}), 503

        # Get query parameters
        raw = request.args.get('raw', 'false').lower() == 'true'
        n_components = int(request.args.get('components', 3))

        # Get all embeddings
        all_data = agent.database.vector_store.get_all_embeddings()

        if not all_data:
            return jsonify({'points': [], 'embeddings': [], 'count': 0})

        # Extract embeddings and metadata
        embeddings = []
        metadata_list = []
        ids = []

        for item in all_data:
            embeddings.append(item['embedding'])
            metadata_list.append(item['metadata'])
            ids.append(item['id'])

        # If raw embeddings requested, return them without PCA
        if raw:
            points = []
            for i, (emb, meta) in enumerate(zip(embeddings, metadata_list)):
                # Get sub_categories from metadata (may be JSON string, old string format, or already parsed)
                sub_categories = meta.get('sub_categories', [])
                if isinstance(sub_categories, str):
                    import json
                    try:
                        # Try parsing as JSON first (new format)
                        sub_categories = json.loads(sub_categories)
                    except (json.JSONDecodeError, TypeError):
                        # Fall back to old string format "['item1', 'item2']"
                        try:
                            # Remove brackets and quotes, split by comma
                            if sub_categories.startswith('[') and sub_categories.endswith(']'):
                                content = sub_categories[1:-1].strip()
                                if content:
                                    # Split by ', ' but be careful with spaces
                                    items = [item.strip().strip("'\"") for item in content.split(',')]
                                    sub_categories = [item for item in items if item]
                                else:
                                    sub_categories = []
                            else:
                                sub_categories = []
                        except:
                            sub_categories = []
                elif not isinstance(sub_categories, list):
                    sub_categories = []

            points.append({
                'id': ids[i],
                'embedding': emb.tolist() if hasattr(emb, 'tolist') else emb,
                'filename': meta.get('filename', 'Unknown'),
                'categories': meta.get('categories', 'Unknown'),
                'sub_categories': sub_categories,
                'metadata': meta
            })

            return jsonify({
                'embeddings': points,
                'count': len(points),
                'raw': True
            })

        # Perform PCA to reduce dimensions
        # Import here to avoid dependency if endpoint is not used
        from sklearn.decomposition import PCA
        import numpy as np

        # Convert to numpy array
        X = np.array(embeddings)

        # Validate n_components
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)
        n_components = min(n_components, max_components)

        if n_components < 2:
             # Not enough data for meaningful visualization
             # Just return some dummy coordinates or handle gracefully
             points = []
             for i, meta in enumerate(metadata_list):
                 points.append({
                     'id': ids[i],
                     'x': 0,
                     'y': 0,
                     'z': 0,
                     'filename': meta.get('filename', 'Unknown'),
                     'categories': meta.get('categories', 'Unknown'),
                     'metadata': meta
                 })
             return jsonify({'points': points, 'count': len(points)})

        # Fit PCA
        pca = PCA(n_components=n_components)

        X_pca = pca.fit_transform(X)
        
        # Prepare result points
        points = []
        for i, coords in enumerate(X_pca):
            # Create coordinate object based on number of components
            point_coords = {}
            for j in range(n_components):
                coord_name = 'xyz'[j] if j < 3 else f'pc{j+1}'
                point_coords[coord_name] = float(coords[j]) if j < len(coords) else 0.0

            meta = metadata_list[i]
            # Get sub_categories from metadata (may be JSON string, old string format, or already parsed)
            sub_categories = meta.get('sub_categories', [])
            if isinstance(sub_categories, str):
                import json
                try:
                    # Try parsing as JSON first (new format)
                    sub_categories = json.loads(sub_categories)
                except (json.JSONDecodeError, TypeError):
                    # Fall back to old string format "['item1', 'item2']"
                    try:
                        # Remove brackets and quotes, split by comma
                        if sub_categories.startswith('[') and sub_categories.endswith(']'):
                            content = sub_categories[1:-1].strip()
                            if content:
                                # Split by ', ' but be careful with spaces
                                items = [item.strip().strip("'\"") for item in content.split(',')]
                                sub_categories = [item for item in items if item]
                            else:
                                sub_categories = []
                        else:
                            sub_categories = []
                    except:
                        sub_categories = []
            elif not isinstance(sub_categories, list):
                sub_categories = []

            point = {
                'id': ids[i],
                'filename': meta.get('filename', 'Unknown'),
                'categories': meta.get('categories', 'Unknown'),
                'sub_categories': sub_categories,
                'metadata': meta,
                **point_coords  # Add coordinate data (x, y, z, pc4, etc.)
            }

            points.append(point)

        return jsonify({
            'points': points,
            'count': len(points),
            'components': n_components,
            'explained_variance': pca.explained_variance_ratio_.tolist() if hasattr(pca, 'explained_variance_ratio_') else []
        })

    except ImportError:
        app.logger.error("scikit-learn not installed")
        return jsonify({'error': 'scikit-learn not installed on server'}), 500
    except Exception as e:
        app.logger.error(f"Error generating embedding visualization: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/refresh')
@app.route('/refresh')  # legacy alias
def refresh_data():
    """Refresh database data from disk."""
    if refresh_database():
        return jsonify({'success': True, 'message': 'Database refreshed successfully'})
    else:
        return jsonify({'success': False, 'message': 'Failed to refresh database'}), 500


@app.route('/api/stats')
def api_get_stats():
    """Get document statistics."""
    try:
        # Refresh data to ensure we have latest documents
        refresh_database()

        all_docs = database.get_all_documents()
        total_docs = len(all_docs)

        # Count categories
        category_counts = {}
        file_type_counts = {}

        for doc in all_docs:
            # Count categories
            categories = doc.get('categories', '').split('-')
            for category in categories:
                category = category.strip()
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1

            # Count file types
            filename = doc.get('filename', '')
            if '.' in filename:
                file_ext = filename.split('.')[-1].lower()
                file_type_counts[file_ext] = file_type_counts.get(file_ext, 0) + 1

        # Get default categories from config and merge with database categories
        default_categories = config.categories or []
        # Ensure all default categories are included (with count 0 if not in database)
        for default_cat in default_categories:
            if default_cat not in category_counts:
                category_counts[default_cat] = 0

        # Count sub-categories
        sub_category_counts = {}
        for doc in all_docs:
            sub_categories = doc.get('sub_categories', [])
            for sub_category in sub_categories:
                sub_category_counts[sub_category] = sub_category_counts.get(sub_category, 0) + 1

        # Convert to sorted lists of tuples
        # Sort by count (descending), then by category name (ascending) for consistent ordering
        categories = sorted(category_counts.items(), key=lambda x: (-x[1], x[0]))
        sub_categories = sorted(sub_category_counts.items(), key=lambda x: (-x[1], x[0]))
        file_types = sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'total_docs': total_docs,
            'categories': categories,
            'sub_categories': sub_categories,
            'file_types': file_types
        })
    except Exception as e:
        app.logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


# Database management endpoints
@app.route('/api/database/tables')
def api_get_tables():
    """Get list of all tables in the database with their metadata."""
    try:
        cursor = database.connection.cursor()

        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = cursor.fetchall()

        table_info = []
        for table_row in tables:
            table_name = table_row['name']

            # Get row count
            cursor.execute(f'SELECT COUNT(*) as count FROM "{table_name}"')
            row_count = cursor.fetchone()['count']

            # Get column information
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            columns = cursor.fetchall()

            column_info = []
            for col in columns:
                column_info.append({
                    'name': col['name'],
                    'type': col['type'],
                    'nullable': not col['notnull'],
                    'primaryKey': bool(col['pk'])
                })

            table_info.append({
                'name': table_name,
                'rowCount': row_count,
                'columns': column_info
            })

        cursor.close()
        return jsonify({'tables': table_info})

    except Exception as e:
        app.logger.error(f"Error getting tables: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/tables/<table_name>/data')
def api_get_table_data(table_name):
    """Get paginated data from a specific table with optional sorting and filtering."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 50, type=int)
        sort = request.args.get('sort', '')
        direction = request.args.get('direction', 'asc')

        # Validate inputs
        if page < 1:
            page = 1
        if limit < 1 or limit > 1000:
            limit = 50
        if direction not in ['asc', 'desc']:
            direction = 'asc'

        cursor = database.connection.cursor()

        # Get column information to validate sort column
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        column_names = [col['name'] for col in columns]

        # Validate sort column exists
        if sort and sort not in column_names:
            app.logger.warning(f"Invalid sort column '{sort}' for table '{table_name}', resetting to default")
            sort = None

        # Validate filter columns exist
        for key, value in request.args.items():
            if key.startswith('filter_') and value.strip():
                column = key[7:]  # Remove 'filter_' prefix
                if column not in column_names:
                    app.logger.warning(f"Invalid filter column '{column}' for table '{table_name}', ignoring")
                    # Remove the invalid filter from request args
                    # We can't modify request.args directly, so we'll skip it in the loop below

        # Build WHERE clause from filters
        where_conditions = []
        where_params = []
        for key, value in request.args.items():
            if key.startswith('filter_') and value.strip():
                column = key[7:]  # Remove 'filter_' prefix
                if column in column_names:  # Only include valid columns
                    where_conditions.append(f'"{column}" LIKE ?')
                    where_params.append(f"%{value}%")

        where_clause = " AND ".join(where_conditions) if where_conditions else ""

        # Get total count
        count_query = f'SELECT COUNT(*) as count FROM "{table_name}"'
        if where_clause:
            count_query += f" WHERE {where_clause}"

        cursor.execute(count_query, where_params)
        total_rows = cursor.fetchone()['count']

        # Build ORDER BY clause
        order_clause = ""
        if sort:
            order_clause = f' ORDER BY "{sort}" {direction.upper()}'

        # Build LIMIT clause
        offset = (page - 1) * limit
        limit_clause = f" LIMIT {limit} OFFSET {offset}"

        # Get data
        data_query = f'SELECT * FROM "{table_name}"'
        if where_clause:
            data_query += f" WHERE {where_clause}"
        data_query += order_clause + limit_clause

        cursor.execute(data_query, where_params)
        rows = cursor.fetchall()

        # Convert rows to lists (we already have columns from PRAGMA)
        row_data = [list(row) for row in rows]

        cursor.close()

        total_pages = math.ceil(total_rows / limit)

        return jsonify({
            'columns': column_names,
            'rows': row_data,
            'totalRows': total_rows,
            'page': page,
            'totalPages': total_pages,
            'limit': limit
        })

    except Exception as e:
        app.logger.error(f"Error getting table data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/tables/<table_name>/record', methods=['POST'])
def api_create_record(table_name):
    """Create a new record in the specified table."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        cursor = database.connection.cursor()

        # Get column info to validate data types
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        column_names = [col['name'] for col in columns]

        # Filter data to only include valid columns
        insert_data = {k: v for k, v in data.items() if k in column_names}

        if not insert_data:
            return jsonify({'error': 'No valid columns provided'}), 400

        # Build INSERT query
        columns_str = ', '.join(f'"{col}"' for col in insert_data.keys())
        placeholders = ', '.join('?' for _ in insert_data.values())
        values = list(insert_data.values())

        query = f'INSERT INTO "{table_name}" ({columns_str}) VALUES ({placeholders})'
        cursor.execute(query, values)

        # Get the ID of the inserted record
        record_id = cursor.lastrowid

        database.connection.commit()
        cursor.close()

        return jsonify({
            'success': True,
            'id': record_id,
            'message': 'Record created successfully'
        })

    except Exception as e:
        database.connection.rollback()
        app.logger.error(f"Error creating record: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/tables/<table_name>/record/<int:record_id>', methods=['PUT'])
def api_update_record(table_name, record_id):
    """Update a record in the specified table."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        cursor = database.connection.cursor()

        # Get column info to validate data types
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        column_names = [col['name'] for col in columns]

        # Find primary key column
        pk_column = None
        for col in columns:
            if col['pk']:
                pk_column = col['name']
                break

        if not pk_column:
            return jsonify({'error': 'Table must have a primary key for updates'}), 400

        # Filter data to only include valid columns (exclude primary key)
        update_data = {k: v for k, v in data.items() if k in column_names and k != pk_column}

        if not update_data:
            return jsonify({'error': 'No valid columns to update'}), 400

        # Build UPDATE query
        set_clause = ', '.join(f'"{col}" = ?' for col in update_data.keys())
        values = list(update_data.values()) + [record_id]

        query = f'UPDATE "{table_name}" SET {set_clause} WHERE "{pk_column}" = ?'
        cursor.execute(query, values)

        if cursor.rowcount == 0:
            cursor.close()
            return jsonify({'error': 'Record not found'}), 404

        database.connection.commit()
        cursor.close()

        return jsonify({
            'success': True,
            'message': 'Record updated successfully'
        })

    except Exception as e:
        database.connection.rollback()
        app.logger.error(f"Error updating record: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/database/tables/<table_name>/record/<int:record_id>', methods=['DELETE'])
def api_delete_record(table_name, record_id):
    """Delete a record from the specified table."""
    try:
        cursor = database.connection.cursor()

        # Get column info to find primary key
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()

        pk_column = None
        for col in columns:
            if col['pk']:
                pk_column = col['name']
                break

        if not pk_column:
            return jsonify({'error': 'Table must have a primary key for deletion'}), 400

        # Delete the record
        query = f'DELETE FROM "{table_name}" WHERE "{pk_column}" = ?'
        cursor.execute(query, [record_id])

        if cursor.rowcount == 0:
            cursor.close()
            return jsonify({'error': 'Record not found'}), 404

        database.connection.commit()
        cursor.close()

        return jsonify({
            'success': True,
            'message': 'Record deleted successfully'
        })

    except Exception as e:
        database.connection.rollback()
        app.logger.error(f"Error deleting record: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/verbose', methods=['GET', 'POST'])
def toggle_verbose():
    """Get or set verbose logging state."""
    import logging

    current_level = app.logger.getEffectiveLevel()
    is_verbose = current_level == logging.DEBUG

    if request.method == 'POST':
        data = request.get_json() or {}
        enable_verbose = data.get('verbose', False)

        if enable_verbose:
            app.logger.setLevel(logging.DEBUG)
            # Also set other loggers to debug
            logging.getLogger('werkzeug').setLevel(logging.DEBUG)
            return jsonify({'success': True, 'verbose': True, 'message': 'Verbose logging enabled'})
        else:
            app.logger.setLevel(logging.INFO)
            logging.getLogger('werkzeug').setLevel(logging.INFO)
            return jsonify({'success': True, 'verbose': False, 'message': 'Verbose logging disabled'})

    return jsonify({'verbose': is_verbose})






if __name__ == '__main__':
    app.run(
        debug=config.webapp_debug,
        host=config.webapp_host,
        port=config.webapp_port
    )
