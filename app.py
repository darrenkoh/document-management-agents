#!/usr/bin/env python3
"""Document Classification Agent - Web App Interface."""
import os
import argparse
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from config import Config
from database import DocumentDatabase
from agent import DocumentAgent
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

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Initialize components
database = DocumentDatabase(config.database_path)
agent = DocumentAgent(config)

# Pagination settings
ITEMS_PER_PAGE = 20


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

    # Sort by classification date (newest first)
    all_docs.sort(key=lambda x: x.get('classification_date', ''), reverse=True)

    # Pagination
    total_docs = len(all_docs)
    total_pages = math.ceil(total_docs / ITEMS_PER_PAGE)
    start_idx = (page - 1) * ITEMS_PER_PAGE
    end_idx = start_idx + ITEMS_PER_PAGE
    docs_page = all_docs[start_idx:end_idx]

    return render_template('documents.html',
                         documents=docs_page,
                         page=page,
                         total_pages=total_pages,
                         total_docs=total_docs,
                         search=search)


@app.route('/semantic-search', methods=['POST'])
def semantic_search():
    """Perform semantic search."""
    query = request.form.get('query', '').strip()

    if not query:
        flash('Please enter a search query.', 'warning')
        return redirect(url_for('index'))

    try:
        # Perform semantic search
        results = agent.search(query, top_k=50)

        return render_template('search_results.html',
                             query=query,
                             results=results)

    except Exception as e:
        app.logger.error(f"Semantic search error: {e}")
        flash(f'Search failed: {str(e)}', 'error')
        return redirect(url_for('index'))


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
