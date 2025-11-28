#!/usr/bin/env python3
"""Document Classification Agent - Web App Interface."""
import os
import sys
import argparse
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Enable CORS for API endpoints
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Cache-Control"]
    }
})

# Setup logging - import setup_logging from main.py at root
import sys
from pathlib import Path
root_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(root_path))
from main import setup_logging
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



















@app.route('/document/<int:doc_id>/file')
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


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for semantic search (returns JSON)."""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        results = agent.search(query, top_k=50)
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
                results = agent.search(query, top_k=50, progress_callback=progress_callback)
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


@app.route('/debug/search')
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
        semantic_results = agent.search('test', top_k=10)
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

    # Refresh data to ensure we have latest documents
    refresh_database()

    all_docs = database.get_all_documents()

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


@app.route('/refresh')
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
            categories = doc.get('categories', '').split(',')
            for category in categories:
                category = category.strip()
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1

            # Count file types
            filename = doc.get('filename', '')
            if '.' in filename:
                file_ext = filename.split('.')[-1].lower()
                file_type_counts[file_ext] = file_type_counts.get(file_ext, 0) + 1

        # Convert to sorted lists of tuples
        categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        file_types = sorted(file_type_counts.items(), key=lambda x: x[1], reverse=True)

        return jsonify({
            'total_docs': total_docs,
            'categories': categories,
            'file_types': file_types
        })
    except Exception as e:
        app.logger.error(f"Error getting stats: {e}")
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
