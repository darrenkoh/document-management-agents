import { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { Search, Filter, Download, Eye, FileText, Brain, ChevronDown, Check } from 'lucide-react';
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

  // Combobox state
  const [isCategoryOpen, setIsCategoryOpen] = useState(false);
  const [categorySearch, setCategorySearch] = useState('');
  const categoryRef = useRef<HTMLDivElement>(null);

  // Semantic search state
  const [isSemanticSearch, setIsSemanticSearch] = useState(false);
  const [semanticSearchQuery, setSemanticSearchQuery] = useState('');
  const [hasInitialized, setHasInitialized] = useState(false);

  // Streaming logs
  const { logs, isStreaming, startStreaming, clearLogs } = useStreamingLogs();

  // Close combobox when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (categoryRef.current && !categoryRef.current.contains(event.target as Node)) {
        setIsCategoryOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  // Load all categories on mount
  useEffect(() => {
    const fetchCategories = async () => {
      try {
        // Use stats API to get all categories from all documents
        const stats = await apiClient.getStats();
        // Extract category names from [category, count] tuples
        const categories = stats.categories.map(([category]) => category);
        setAvailableCategories(categories);
      } catch (error) {
        console.error('Failed to load categories:', error);
      }
    };
    fetchCategories();
  }, []);

  // Handle semantic search results from navigation state
  useEffect(() => {
    const state = location.state as any;
    if (state?.semanticSearchResults) {
      setSemanticSearchQuery(state.semanticSearchQuery || '');
      setIsSemanticSearch(true);
      setDocuments(state.semanticSearchResults);
      setLoading(false);
      setHasInitialized(true);
    } else {
      setHasInitialized(true);
    }
  }, []);

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
    } catch (error) {
      console.error('Failed to load documents:', error);
      toast.error('Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  const performSemanticSearch = async (query: string) => {
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

    return cleanup;
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();

    if (searchQuery.trim()) {
      // Always perform semantic search if there is a query
      await performSemanticSearch(searchQuery.trim());
    } else {
      // If empty, switch back to regular list
      setIsSemanticSearch(false);
      setCurrentPage(1);
      loadDocuments();
    }
  };

  const handleCategoryChange = (category: string) => {
    setCategoryFilter(category);
    setCurrentPage(1);
    // If we are filtering by category, we should probably exit semantic search mode
    // or filter the semantic results. For simplicity, let's switch to regular mode
    // as semantic search usually implies a text query.
    if (isSemanticSearch) {
      setIsSemanticSearch(false);
      setSemanticSearchQuery('');
      setSearchQuery(''); // Clear search query as we are now browsing by category
    }
  };

  const clearFilters = () => {
    setSearchQuery('');
    setCategoryFilter('');
    setCurrentPage(1);
    setIsSemanticSearch(false);
    setSemanticSearchQuery('');
    clearLogs();
    loadDocuments();
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

  const hasActiveFilters = searchQuery || categoryFilter || isSemanticSearch;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-primary-900">
            {isSemanticSearch ? 'Semantic Search Results' : 'Documents'}
          </h1>
          <p className="text-primary-600 mt-1">
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
      </div>

      {/* Search and Filter */}
      <Card>
        <CardContent className="p-6">
          <form onSubmit={handleSearch} className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              {isSemanticSearch ? (
                <Brain className="absolute left-3 top-1/2 transform -translate-y-1/2 text-accent-500 w-5 h-5" />
              ) : (
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-primary-400 w-5 h-5" />
              )}
              <Input
                type="text"
                placeholder="Search documents..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Category Combobox */}
            <div className="min-w-0 lg:min-w-[250px] relative" ref={categoryRef}>
              <div
                className="flex items-center justify-between w-full h-11 px-3 py-2 border border-primary-300 rounded-lg bg-white cursor-pointer hover:border-primary-400 transition-colors"
                onClick={() => setIsCategoryOpen(!isCategoryOpen)}
              >
                <div className="flex items-center gap-2 overflow-hidden">
                  <Filter className="text-primary-400 w-4 h-4 flex-shrink-0" />
                  <span className={`truncate text-sm ${categoryFilter ? 'text-primary-900 font-medium' : 'text-primary-500'}`}>
                    {categoryFilter || 'All Categories'}
                  </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-primary-400 transition-transform duration-200 ${isCategoryOpen ? 'rotate-180' : ''}`} />
              </div>

              {isCategoryOpen && (
                <div className="absolute z-20 w-full mt-1 bg-white border border-primary-200 rounded-lg shadow-lg max-h-80 overflow-hidden flex flex-col animate-fade-in">
                  <div className="p-2 border-b border-primary-100 bg-primary-50/50">
                    <Input
                      placeholder="Filter categories..."
                      value={categorySearch}
                      onChange={(e) => setCategorySearch(e.target.value)}
                      className="h-9 text-sm"
                      autoFocus
                      onClick={(e) => e.stopPropagation()}
                    />
                  </div>
                  <div className="overflow-y-auto flex-1 p-1">
                    <div
                      className={`px-3 py-2 rounded-md cursor-pointer text-sm flex items-center justify-between ${!categoryFilter ? 'bg-primary-100 text-primary-900' : 'text-primary-700 hover:bg-primary-50'}`}
                      onClick={() => {
                        handleCategoryChange('');
                        setIsCategoryOpen(false);
                      }}
                    >
                      <span>All Categories</span>
                      {!categoryFilter && <Check className="w-4 h-4 text-primary-600" />}
                    </div>
                    {availableCategories
                      .filter(c => c.toLowerCase().includes(categorySearch.toLowerCase()))
                      .map(category => (
                        <div
                          key={category}
                          className={`px-3 py-2 rounded-md cursor-pointer text-sm flex items-center justify-between ${categoryFilter === category ? 'bg-primary-100 text-primary-900' : 'text-primary-700 hover:bg-primary-50'}`}
                          onClick={() => {
                            handleCategoryChange(category);
                            setIsCategoryOpen(false);
                          }}
                        >
                          <span>{category}</span>
                          {categoryFilter === category && <Check className="w-4 h-4 text-primary-600" />}
                        </div>
                      ))}
                    {availableCategories.length === 0 && (
                      <div className="px-3 py-4 text-center text-sm text-primary-400">
                        No categories found
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Clear Filters */}
            {hasActiveFilters && (
              <Button variant="outline" onClick={clearFilters} type="button">
                Clear Filters
              </Button>
            )}
          </form>

          {/* Active Filters Display */}
          {hasActiveFilters && (
            <div className="mt-4 flex flex-wrap gap-2">
              {searchQuery && (
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${isSemanticSearch ? 'bg-accent-50 text-accent-800 border border-accent-200' : 'bg-primary-100 text-primary-800 border border-primary-200'
                  }`}>
                  {isSemanticSearch ? 'AI Search' : 'Search'}: "{searchQuery}"
                  <button
                    onClick={() => setSearchQuery('')}
                    className={`ml-2 ${isSemanticSearch ? 'text-accent-600 hover:text-accent-800' : 'text-primary-600 hover:text-primary-800'}`}
                  >
                    ×
                  </button>
                </span>
              )}
              {categoryFilter && (
                <span className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-primary-100 text-primary-800 border border-primary-200">
                  Category: {categoryFilter}
                  <button
                    onClick={() => setCategoryFilter('')}
                    className="ml-2 text-primary-600 hover:text-primary-800"
                  >
                    ×
                  </button>
                </span>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Streaming Logs */}
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
                        className="font-semibold text-primary-900 truncate cursor-pointer hover:text-primary-600 transition-colors"
                        onClick={() => navigate(`/document/${doc.id}`)}
                      >
                        {doc.filename}
                      </h3>
                      <p className="text-sm text-primary-500 mt-1">
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

                  <p className="text-sm text-primary-600 mb-4 line-clamp-3 bg-primary-50/50 p-3 rounded-lg border border-primary-100">
                    {truncateContent(doc.content_preview)}
                  </p>

                  <div className="flex items-center justify-between text-xs text-primary-500 mb-4">
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

          {/* Pagination */}
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

                <span className="px-4 py-2 text-sm text-primary-600">
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
            <FileText className="w-12 h-12 text-primary-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-primary-900 mb-2">No documents found</h3>
            <p className="text-primary-600 mb-4">
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
