#!/usr/bin/env python3
"""Document Classification Agent - Web App Interface."""
import os
import argparse
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
from config import Config
from database import DocumentDatabase
from agent import DocumentAgent
from file_handler import FileHandler
from typing import List, Dict, Any
import math

# Parse command line arguments
parser = argparse.ArgumentParser(description='Document Classification Agent - Web Interface')
parser.add_argument('--config', type=str, default='config.yaml',
                   help='Path to configuration file (default: config.yaml)')
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

# Initialize components
database = DocumentDatabase(config.database_path)
agent = DocumentAgent(config)

# Debug: Check if agent has embedding generator
app.logger.info("Initializing Document Agent...")
app.logger.info(f"Agent has embedding generator: {hasattr(agent, 'embedding_generator')}")
if hasattr(agent, 'embedding_generator'):
    app.logger.info(f"Embedding endpoint: {agent.embedding_generator.endpoint}")
    app.logger.info(f"Embedding model: {agent.embedding_generator.model}")
else:
    app.logger.error("Agent does not have embedding generator!")

# Pagination settings
ITEMS_PER_PAGE = 20


def highlight_search_terms_in_results(results, search_terms):
    """Highlight search terms in content previews of search results.

    Args:
        results: List of search result dictionaries
        search_terms: List of terms to highlight

    Returns:
        List of results with highlighted content previews
    """
    if not results or not search_terms:
        return results

    highlighted_results = []

    for result in results:
        highlighted_result = result.copy()

        # Get content preview
        content_preview = result.get('content_preview', '')

        # Highlight each search term
        highlighted_content = content_preview
        for term in search_terms:
            if len(term.strip()) > 0:
                # Create case-insensitive regex pattern
                pattern = re.escape(term.strip())
                regex = re.compile(f'({pattern})', re.IGNORECASE)
                highlighted_content = regex.sub(
                    r'<mark class="search-highlight">\1</mark>',
                    highlighted_content
                )

        highlighted_result['content_preview'] = highlighted_content
        highlighted_results.append(highlighted_result)

    return highlighted_results


@app.route('/')
def index():
    """Home page - show recent documents and search interface."""
    # Get recent documents
    all_docs = database.get_all_documents()
    recent_docs = sorted(all_docs, key=lambda x: x.get('classification_date', ''), reverse=True)[:10]

    return render_template('index.html', recent_docs=recent_docs)


@app.route('/documents')
def documents():
    """List all documents with pagination."""
    page = request.args.get('page', 1, type=int)
    search = request.args.get('search', '').strip()
    category_filter = request.args.get('category', '').strip()

    # Get all documents
    all_docs = database.get_all_documents()

    # Filter by search if provided
    if search:
        filtered_docs = []
        search_lower = search.lower()
        for doc in all_docs:
            # Search in filename, categories, and content preview
            filename = doc.get('filename', '').lower()
            categories = doc.get('categories', '').lower()
            content_preview = doc.get('content_preview', '').lower()

            if (search_lower in filename or
                search_lower in categories or
                search_lower in content_preview):
                filtered_docs.append(doc)
        all_docs = filtered_docs

    # Filter by category if provided
    if category_filter:
        filtered_docs = []
        category_lower = category_filter.lower()
        for doc in all_docs:
            doc_categories = doc.get('categories', '').lower()
            if category_lower in doc_categories:
                filtered_docs.append(doc)
        all_docs = filtered_docs

    # Sort by classification date (newest first)
    all_docs.sort(key=lambda x: x.get('classification_date', ''), reverse=True)

    # Pagination
    total_docs = len(all_docs)
    total_pages = math.ceil(total_docs / ITEMS_PER_PAGE)
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    docs_page = all_docs[start_idx:end_idx]

    # Get all unique categories for filter dropdown
    all_categories = set()
    for doc in database.get_all_documents():
        for cat in doc.get('categories', '').split('-'):
            if cat.strip():
                all_categories.add(cat.strip())
    sorted_categories = sorted(all_categories)

    return render_template('documents.html',
                         documents=docs_page,
                         page=page,
                         total_pages=total_pages,
                         total_docs=total_docs,
                         search=search,
                         category_filter=category_filter,
                         available_categories=sorted_categories)


@app.route('/unified-search', methods=['POST'])
def unified_search():
    """Unified search that handles both semantic and category searches."""
    query = request.form.get('query', '').strip()

    if not query:
        flash('Please enter a search query.', 'warning')
        return redirect(url_for('index'))

    try:
        # Check if this is a category search (starts with "category:")
        if query.lower().startswith('category:'):
            category = query[9:].strip()  # Remove "category:" prefix
            return category_search_logic(category)

        # Check if the query matches a known category
        all_categories = set()
        for doc in database.get_all_documents():
            for cat in doc.get('categories', '').split('-'):
                if cat.strip():
                    all_categories.add(cat.strip().lower())

        if query.lower() in all_categories:
            # Perform category search with highlighting
            try:
                results = agent.search_by_category(query)
                highlighted_results = highlight_search_terms_in_results(results, [query])

                return render_template('search_results.html',
                                     query=f"Category: {query}",
                                     results=highlighted_results,
                                     search_type='category')
            except Exception as e:
                app.logger.error(f"Category search error: {e}")
                flash(f'Search failed: {str(e)}', 'error')
                return redirect(url_for('index'))

        # Default to semantic search
        return semantic_search_logic(query)

    except Exception as e:
        app.logger.error(f"Unified search error: {e}")
        flash(f'Search failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    """Perform semantic search."""
    query = request.form.get('query', '').strip()
    return semantic_search_logic(query)


def semantic_search_logic(query):
    """Core semantic search logic."""
    if not query:
        flash('Please enter a search query.', 'warning')
        return redirect(url_for('index'))

    try:
        app.logger.info(f"Starting semantic search for query: '{query}'")

        # Check if agent has embedding generator
        if not hasattr(agent, 'embedding_generator'):
            app.logger.error("Agent does not have embedding generator")
            flash('Semantic search is not available - embedding service not configured', 'error')
            return redirect(url_for('index'))

        # Perform semantic search
        results = agent.search(query, top_k=50)

        app.logger.info(f"Semantic search completed. Found {len(results)} results")

        # Highlight relevant terms in content previews
        highlighted_results = highlight_search_terms_in_results(results, query.split())

        # Debug: Log details about results
        if results:
            app.logger.debug(f"First result: {results[0].get('filename', 'N/A')} - similarity: {results[0].get('similarity', 'N/A')}")
            for i, result in enumerate(results[:3]):
                app.logger.debug(f"Result {i+1}: {result.get('filename', 'N/A')} - similarity: {result.get('similarity', 'N/A')}")
        else:
            app.logger.warning("No results found for semantic search")
            flash('No results found. Try different search terms or check if documents have embeddings.', 'warning')

        return render_template('search_results.html',
                             query=query,
                             results=highlighted_results,
                             search_type='semantic')

    except Exception as e:
        app.logger.error(f"Semantic search error: {e}")
        import traceback
        app.logger.error(f"Traceback: {traceback.format_exc()}")
        flash(f'Search failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/category-search', methods=['GET', 'POST'])
def category_search():
    """Search documents by category."""
    if request.method == 'POST':
        category = request.form.get('category', '').strip()
    else:
        category = request.args.get('category', '').strip()

    return category_search_logic(category)


def category_search_logic(category):
    """Core category search logic."""
    if not category:
        flash('Please select a category.', 'warning')
        return redirect(url_for('documents'))

    try:
        # Perform category search
        results = agent.search_by_category(category)

        # Highlight category terms in content previews
        highlighted_results = highlight_search_terms_in_results(results, [category])

        return render_template('search_results.html',
                             query=f"Category: {category}",
                             results=highlighted_results,
                             search_type='category')

    except Exception as e:
        app.logger.error(f"Category search error: {e}")
        flash(f'Search failed: {str(e)}', 'error')
        return redirect(url_for('documents'))


@app.route('/document/<int:doc_id>')
def document_detail(doc_id: int):
    """Show detailed document information."""
    # Note: TinyDB doesn't use sequential IDs, so we'll search by doc_id
    # For now, we'll get all documents and find the one with matching doc_id
    all_docs = database.get_all_documents()
    doc = None
    for d in all_docs:
        if d.get('doc_id') == doc_id:
            doc = d
            break

    if not doc:
        flash('Document not found.', 'error')
        return redirect(url_for('documents'))

    return render_template('document_detail.html', document=doc)


@app.route('/document/<int:doc_id>/view-original')
def view_original_content(doc_id: int):
    """Show the original file in an appropriate viewer based on file type."""
    # Get the document
    all_docs = database.get_all_documents()
    doc = None
    for d in all_docs:
        if d.get('doc_id') == doc_id:
            doc = d
            break

    if not doc:
        flash('Document not found.', 'error')
        return redirect(url_for('documents'))

    try:
        file_path = Path(doc['file_path'])
        if not file_path.exists():
            flash(f'Original file not found at: {file_path}', 'error')
            return redirect(url_for('document_detail', doc_id=doc_id))

        # Determine file type and viewer
        file_extension = doc['metadata'].get('file_extension', '').lower()
        file_size = doc['metadata'].get('file_size', 0)

        return render_template('view_original_content.html',
                             document=doc,
                             file_path=file_path,
                             file_extension=file_extension,
                             file_size=file_size)

    except Exception as e:
        app.logger.error(f"Error viewing original content for document {doc_id}: {e}")
        flash(f'Error loading original content: {str(e)}', 'error')
        return redirect(url_for('document_detail', doc_id=doc_id))


@app.route('/document/<int:doc_id>/file')
def serve_original_file(doc_id: int):
    """Serve the original file for viewing."""
    # Get the document
    all_docs = database.get_all_documents()
    doc = None
    for d in all_docs:
        if d.get('doc_id') == doc_id:
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


@app.route('/debug/search')
def debug_search():
    """Debug endpoint to check search functionality."""
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


@app.route('/stats')
def stats():
    """Show database statistics."""
    all_docs = database.get_all_documents()

    # Calculate statistics
    total_docs = len(all_docs)
    categories = {}
    file_types = {}

    for doc in all_docs:
        # Category stats
        doc_categories = doc.get('categories', '').split('-')
        for cat in doc_categories:
            if cat:
                categories[cat] = categories.get(cat, 0) + 1

        # File type stats
        ext = doc.get('metadata', {}).get('file_extension', '')
        if ext:
            file_types[ext] = file_types.get(ext, 0) + 1

    return render_template('stats.html',
                         total_docs=total_docs,
                         categories=sorted(categories.items(), key=lambda x: x[1], reverse=True),
                         file_types=sorted(file_types.items(), key=lambda x: x[1], reverse=True))


@app.template_filter('format_date')
def format_date(date_string):
    """Format ISO date string for display."""
    if not date_string:
        return 'Unknown'
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return date_string


@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    """Format Unix timestamp for display."""
    if not timestamp:
        return 'Unknown'
    try:
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return str(timestamp)


@app.template_filter('truncate_content')
def truncate_content(content, length=200):
    """Truncate content to specified length."""
    if not content:
        return ''
    if len(content) <= length:
        return content
    return content[:length] + '...'


@app.context_processor
def utility_processor():
    """Add utility functions to template context."""
    def get_file_icon(extension):
        icon_map = {
            '.pdf': '📄',
            '.docx': '📝',
            '.doc': '📝',
            '.txt': '📃',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.gif': '🖼️',
        }
        return icon_map.get(extension.lower(), '📄')

    return dict(get_file_icon=get_file_icon)


if __name__ == '__main__':
    app.run(
        debug=config.webapp_debug,
        host=config.webapp_host,
        port=config.webapp_port
    )
