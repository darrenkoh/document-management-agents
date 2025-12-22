---
name: Embedding Search Page
overview: Create a new "Embedding Search" page that allows users to search the embedding database using a keyword. The page will show tokenization details (token IDs, words, and embedding vector info) and display similar documents with scores and previews.
todos:
  - id: backend-endpoint
    content: Add /api/embeddings/search endpoint to Flask backend
    status: completed
  - id: frontend-page
    content: Create EmbeddingSearchPage.tsx with search UI and results display
    status: completed
    dependencies:
      - backend-endpoint
  - id: types-update
    content: Update TypeScript types for embedding search response
    status: completed
  - id: routes-nav
    content: Add route and navigation link for new page
    status: completed
    dependencies:
      - frontend-page
---

# Embedding Search Page Implementation

## Overview

Create a new frontend page at `/embedding-search` that allows searching the embedding database by keyword. When a user enters a keyword, the system will:

1. Generate an embedding for the keyword using the existing `EmbeddingGenerator`
2. Display tokenization/embedding information (token IDs, words, vector dimensions)
3. Search the vector store for similar documents
4. Display results with similarity scores, content previews, and metadata

## Architecture

```mermaid
sequenceDiagram
    participant User
    participant Frontend as EmbeddingSearchPage
    participant API as Flask Backend
    participant EmbedGen as EmbeddingGenerator
    participant VectorStore as ChromaDB

    User->>Frontend: Enter keyword
    Frontend->>API: POST /api/embeddings/search
    API->>EmbedGen: generate_embedding(keyword)
    EmbedGen-->>API: embedding vector
    API->>VectorStore: search_similar(embedding)
    VectorStore-->>API: similar documents
    API-->>Frontend: {tokens, embedding_info, results}
    Frontend->>User: Display results
```



## Implementation

### 1. Backend API Endpoint

Add a new endpoint in [`src/backend/api/app.py`](src/backend/api/app.py):

```python
@app.route('/api/embeddings/search', methods=['POST'])
def api_search_embeddings():
    """Search embedding database using a keyword."""
```

This endpoint will:

- Accept JSON body with `keyword` and optional `limit` parameter (default: 10)
- Generate embedding for the keyword using `agent.embedding_generator`
- Search the vector store using `search_similar()`
- Return tokenization info, embedding metadata, and matching documents with scores

### 2. Frontend Page

Create [`src/frontend/src/pages/EmbeddingSearchPage.tsx`](src/frontend/src/pages/EmbeddingSearchPage.tsx):

- Search input with configurable result limit (dropdown: 10/20/50)
- Tokenization display section showing:
- Token IDs (if available from model)
- Words/subwords breakdown
- Embedding vector info (dimension, sample values)
- Results table showing:
- Document filename (clickable link to detail page)
- Categories and sub-categories
- Similarity score (percentage or decimal)
- Content preview
- Loading states and error handling

### 3. Route Registration

Update [`src/frontend/src/App.tsx`](src/frontend/src/App.tsx) to add the new route:

```tsx
<Route path="/embedding-search" element={<EmbeddingSearchPage />} />
```



### 4. Navigation

Update [`src/frontend/src/components/Layout.tsx`](src/frontend/src/components/Layout.tsx) to add navigation link to the new page.

### 5. Types (Already Exists)

The API client already has `searchEmbeddings()` method in [`src/frontend/src/lib/api.ts`](src/frontend/src/lib/api.ts) - will update the return type to include all required fields.

## Key Files to Modify

| File | Change ||------|--------|| `src/backend/api/app.py` | Add `/api/embeddings/search` endpoint || `src/frontend/src/pages/EmbeddingSearchPage.tsx` | New file - search page component || `src/frontend/src/App.tsx` | Add route for new page || `src/frontend/src/components/Layout.tsx` | Add navigation link || `src/frontend/src/types/index.ts` | Add types for search response |

## Notes