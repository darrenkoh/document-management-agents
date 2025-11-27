import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { Search, Filter, Download, Eye, FileText, Brain } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { StreamingLogs, useStreamingLogs } from '@/components/ui/StreamingLogs';
import { Document, DocumentsResponse } from '@/types';
import { apiClient, getFileIcon, formatFileSize } from '@/lib/api';
import toast from 'react-hot-toast';

export default function DocumentsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState(searchParams.get('search') || '');
  const [categoryFilter, setCategoryFilter] = useState(searchParams.get('category') || '');
  const [currentPage, setCurrentPage] = useState(parseInt(searchParams.get('page') || '1'));
  const [totalPages, setTotalPages] = useState(1);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [availableCategories, setAvailableCategories] = useState<string[]>([]);

  // Semantic search state
  const [isSemanticSearch, setIsSemanticSearch] = useState(false);
  const [semanticSearchQuery, setSemanticSearchQuery] = useState('');
  const [hasInitialized, setHasInitialized] = useState(false);

  // Streaming logs
  const { logs, isStreaming, startStreaming, clearLogs } = useStreamingLogs();

  // Handle semantic search results from navigation state
  useEffect(() => {
    console.log('DocumentsPage useEffect triggered, location.state:', location.state);
    const state = location.state as any;
    if (state?.semanticSearchResults) {
      console.log('Semantic search results found in state:', state.semanticSearchResults.length, 'results');
      setSemanticSearchQuery(state.semanticSearchQuery || '');
      setIsSemanticSearch(true);
      setDocuments(state.semanticSearchResults);
      setLoading(false);
      setHasInitialized(true);
      console.log('DocumentsPage state set, semantic search mode initialized');
    } else {
      console.log('No semantic search results in state, initializing regular mode');
      setHasInitialized(true);
    }
  }, []); // Only run once on mount

  useEffect(() => {
    // Only load documents after initialization and if we're not in semantic search mode
    if (hasInitialized && !isSemanticSearch) {
      loadDocuments();
    }
  }, [hasInitialized, currentPage, searchQuery, categoryFilter, isSemanticSearch]);

  useEffect(() => {
    // Update URL params when filters change
    const params = new URLSearchParams();
    if (searchQuery) params.set('search', searchQuery);
    if (categoryFilter) params.set('category', categoryFilter);
    if (currentPage > 1) params.set('page', currentPage.toString());
    setSearchParams(params);
  }, [searchQuery, categoryFilter, currentPage, setSearchParams]);

  const loadDocuments = async () => {
    setLoading(true);
    try {
      const response: DocumentsResponse = await apiClient.getDocuments({
        page: currentPage,
        limit: 20,
        search: searchQuery || undefined,
        category: categoryFilter || undefined,
      });

      setDocuments(response.documents);
      setTotalPages(response.total_pages);
      setTotalDocuments(response.total_documents);

      // Extract unique categories from current results
      const categories = new Set<string>();
      response.documents.forEach(doc => {
        doc.categories.split('-').forEach(cat => {
          if (cat.trim()) categories.add(cat.trim());
        });
      });
      setAvailableCategories(Array.from(categories).sort());
    } catch (error) {
      console.error('Failed to load documents:', error);
      toast.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const performSemanticSearch = async (query: string) => {
    console.log('performSemanticSearch called with query:', query);
    setLoading(true);
    clearLogs();

    const cleanup = startStreaming(query,
      (results) => {
        // Success callback
        setSemanticSearchQuery(query);
        setIsSemanticSearch(true);
        setDocuments(results.results);
        setTotalPages(1);
        setTotalDocuments(results.results.length);
        setCurrentPage(1);
        setLoading(false);
        toast.success(`Found ${results.results.length} results for "${query}"`);
      },
      (error) => {
        // Error callback
        console.error('Streaming semantic search failed:', error);
        toast.error(`Semantic search failed: ${error}`);
        setLoading(false);
      }
    );

    // Store cleanup function for potential cancellation
    return cleanup;
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (isSemanticSearch) {
      // Perform semantic search
      if (searchQuery.trim()) {
        await performSemanticSearch(searchQuery.trim());
      }
    } else {
      // Regular search - reset to first page
      setCurrentPage(1);
    }
  };

  const handleCategoryChange = (category: string) => {
    setCategoryFilter(category);
    setCurrentPage(1); // Reset to first page when filtering
  };

  const clearFilters = () => {
    setSearchQuery('');
    setCategoryFilter('');
    setCurrentPage(1);
    // If in semantic search mode, clear semantic search as well
    if (isSemanticSearch) {
      setIsSemanticSearch(false);
      setSemanticSearchQuery('');
      clearLogs();
      // Reload regular documents
      loadDocuments();
    }
  };

  const toggleSearchMode = () => {
    if (isSemanticSearch) {
      // Switching from semantic to regular search
      setIsSemanticSearch(false);
      setSemanticSearchQuery('');
      clearLogs();
      setSearchQuery('');
      setCategoryFilter('');
      setCurrentPage(1);
      loadDocuments();
    } else {
      // Switching to semantic search
      setIsSemanticSearch(true);
      setSearchQuery('');
      setCategoryFilter('');
      setCurrentPage(1);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return dateString;
    }
  };

  const truncateContent = (content: string, length = 200) => {
    if (content.length <= length) return content;
    return content.substring(0, length) + '...';
  };

  const hasActiveFilters = searchQuery || categoryFilter;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            {isSemanticSearch ? 'Semantic Search Results' : 'Documents'}
          </h1>
          <p className="text-gray-600 mt-1">
            {isSemanticSearch ? (
              semanticSearchQuery ? (
                <>Found {totalDocuments} document{totalDocuments !== 1 ? 's' : ''} matching "{semanticSearchQuery}"</>
              ) : (
                'Enter a search query to find documents using AI-powered semantic search'
              )
            ) : (
              <>{totalDocuments} document{totalDocuments !== 1 ? 's' : ''} found</>
            )}
          </p>
        </div>
        {/* Search Mode Toggle */}
        <Button
          variant={isSemanticSearch ? "default" : "outline"}
          onClick={toggleSearchMode}
          className="flex items-center gap-2"
        >
          <Brain className="w-4 h-4" />
          {isSemanticSearch ? 'AI Search' : 'Regular Search'}
        </Button>
      </div>

      {/* Search */}
      <Card>
        <CardContent className="p-6">
          <form onSubmit={handleSearch} className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              {isSemanticSearch ? (
                <Brain className="absolute left-3 top-1/2 transform -translate-y-1/2 text-purple-400 w-5 h-5" />
              ) : (
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              )}
              <Input
                type="text"
                placeholder={isSemanticSearch ? "Search using AI-powered semantic search..." : "Search documents..."}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {!isSemanticSearch && (
              <>
                {/* Category Filter */}
                <div className="flex items-center gap-2 min-w-0 lg:min-w-[200px]">
                  <Filter className="text-gray-400 w-5 h-5 flex-shrink-0" />
                  <select
                    value={categoryFilter}
                    onChange={(e) => handleCategoryChange(e.target.value)}
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg bg-white text-sm focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20"
                  >
                    <option value="">All Categories</option>
                    {availableCategories.map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                  </select>
                </div>
              </>
            )}

            {/* Clear Filters */}
            {hasActiveFilters && (
              <Button variant="outline" onClick={clearFilters}>
                Clear Filters
              </Button>
            )}
          </form>

          {/* Active Filters Display */}
          {hasActiveFilters && (
            <div className="mt-4 flex flex-wrap gap-2">
              {searchQuery && (
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
                  isSemanticSearch ? 'bg-purple-100 text-purple-800' : 'bg-primary-100 text-primary-800'
                }`}>
                  {isSemanticSearch ? 'AI Search' : 'Search'}: "{searchQuery}"
                  <button
                    onClick={() => setSearchQuery('')}
                    className={`ml-2 ${isSemanticSearch ? 'text-purple-600 hover:text-purple-800' : 'text-primary-600 hover:text-primary-800'}`}
                  >
                    ×
                  </button>
                </span>
              )}
              {categoryFilter && !isSemanticSearch && (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-secondary-100 text-secondary-800">
                  Category: {categoryFilter}
                  <button
                    onClick={() => setCategoryFilter('')}
                    className="ml-2 text-secondary-600 hover:text-secondary-800"
                  >
                    ×
                  </button>
                </span>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Streaming Logs - show during streaming regardless of search mode */}
      {(isStreaming || logs.length > 0) && (
        <StreamingLogs
          isVisible={true}
        />
      )}

      {/* Documents Grid */}
      {loading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : documents.length > 0 ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {documents.map((doc) => (
              <Card key={doc.id} className="hover:shadow-medium transition-shadow">
                <CardContent className="p-6">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="text-3xl flex-shrink-0">
                      {getFileIcon(doc.metadata.file_extension)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3
                        className="font-semibold text-gray-900 truncate cursor-pointer hover:text-primary-600 transition-colors"
                        onClick={() => navigate(`/document/${doc.id}`)}
                      >
                        {doc.filename}
                      </h3>
                      <p className="text-sm text-gray-500 mt-1">
                        {formatDate(doc.classification_date)}
                      </p>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-1 mb-4">
                    {doc.categories.split('-').map((category, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                      >
                        {category.trim()}
                      </span>
                    ))}
                  </div>

                  <p className="text-sm text-gray-600 mb-4 line-clamp-3">
                    {truncateContent(doc.content_preview)}
                  </p>

                  <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                    <span>Size: {formatFileSize(doc.metadata.file_size)}</span>
                    <span>Type: {doc.metadata.file_extension.toUpperCase()}</span>
                  </div>

                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => navigate(`/document/${doc.id}`)}
                      className="flex-1"
                    >
                      <Eye className="w-4 h-4 mr-2" />
                      View
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        // TODO: Implement download
                        toast.success('Download feature coming soon!');
                      }}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Pagination - only show for regular search mode */}
          {!isSemanticSearch && totalPages > 1 && (
            <div className="flex justify-center mt-8">
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>

                <span className="px-4 py-2 text-sm text-gray-600">
                  Page {currentPage} of {totalPages}
                </span>

                <Button
                  variant="outline"
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </>
      ) : (
        <Card>
          <CardContent className="text-center py-12">
            <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
            <p className="text-gray-600 mb-4">
              {hasActiveFilters
                ? 'Try adjusting your search criteria or clearing filters.'
                : 'No documents have been processed yet.'
              }
            </p>
            {hasActiveFilters && (
              <Button onClick={clearFilters}>Clear Filters</Button>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
