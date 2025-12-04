import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { Download, Eye, FileText, Trash2, AlertTriangle, ScanText, Clock } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { StreamingLogs, useStreamingLogs } from '@/components/ui/StreamingLogs';
import { DocumentSearchAndQuestion } from '@/components/DocumentSearchAndQuestion';
import { Document, DocumentsResponse, AnswerCitation, AnswerStreamEvent } from '@/types';
import { apiClient, getFileIcon, formatFileSize, cleanContentPreview } from '@/lib/api';
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

  // Selection state for multi-delete
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);

  // Streaming logs
  const { logs, isStreaming, startStreaming, clearLogs } = useStreamingLogs();

  // Question answering state
  const [questionQuery, setQuestionQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [answerCitations, setAnswerCitations] = useState<AnswerCitation[]>([]);
  const [isAnswering, setIsAnswering] = useState(false);
  const [answerError, setAnswerError] = useState<string | undefined>();


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

  // Clear selection when documents change
  useEffect(() => {
    setSelectedIds(new Set());
  }, [documents]);

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

  const truncateContent = (content: string, length = 80) => {
    if (content.length <= length) return content;
    return content.substring(0, length) + '...';
  };

  // Helper function to format duration in minutes
  const formatProcessingTime = (seconds: number): string => {
    if (!seconds && seconds !== 0) return 'N/A';
    const totalSeconds = Math.round(seconds);
    if (totalSeconds < 60) return `${totalSeconds}s`;
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = totalSeconds % 60;
    return remainingSeconds > 0 ? `${minutes}m ${remainingSeconds}s` : `${minutes}m`;
  };

  // Selection handlers
  const handleSelectAll = () => {
    if (selectedIds.size === documents.length) {
      // Deselect all
      setSelectedIds(new Set());
    } else {
      // Select all
      setSelectedIds(new Set(documents.map(doc => doc.id)));
    }
  };

  const handleSelectRow = (id: number) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.size === 0) return;

    setIsDeleting(true);
    try {
      const idsToDelete = Array.from(selectedIds);
      const result = await apiClient.deleteDocuments(idsToDelete);

      if (result.success) {
        toast.success(result.message);
        setSelectedIds(new Set());
        setShowDeleteConfirm(false);
        // Reload documents after deletion
        if (isSemanticSearch) {
          // For semantic search, filter out deleted documents from current results
          setDocuments(prev => prev.filter(doc => !selectedIds.has(doc.id)));
          setTotalDocuments(prev => prev - result.deleted_count);
        } else {
          loadDocuments();
        }
      } else {
        toast.error('Failed to delete some documents');
        if (result.errors.length > 0) {
          result.errors.forEach(err => console.error(err));
        }
      }
    } catch (error) {
      console.error('Failed to delete documents:', error);
      toast.error('Failed to delete documents');
    } finally {
      setIsDeleting(false);
    }
  };

  const hasActiveFilters = searchQuery || categoryFilter || isSemanticSearch;
  const isAllSelected = documents.length > 0 && selectedIds.size === documents.length;
  const isSomeSelected = selectedIds.size > 0 && selectedIds.size < documents.length;

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

      {/* Combined Search and Question Component */}
      <DocumentSearchAndQuestion
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        onSearch={performSemanticSearch}
        isSearching={isStreaming}
        categoryFilter={categoryFilter}
        onCategoryFilterChange={handleCategoryChange}
        availableCategories={availableCategories}
        showCategoryFilter={true}
        searchPlaceholder="Search documents..."
        showClearFilters={!!hasActiveFilters}
        onClearFilters={clearFilters}
        isSemanticSearch={isSemanticSearch}
        showQuestionAnswering={true}
        questionQuery={questionQuery}
        onQuestionQueryChange={setQuestionQuery}
        answer={answer}
        answerCitations={answerCitations}
        isAnswering={isAnswering}
        answerError={answerError}
        onAnswerQuestion={async (query) => {
          setIsAnswering(true);
          setAnswer('');
          setAnswerCitations([]);
          setAnswerError(undefined);

          try {
            await apiClient.answerQuestion(
              query,
              (chunk) => {
                setAnswer(prev => prev + chunk);
              },
              (event: AnswerStreamEvent) => {
                if (event.type === 'citations' && event.citations) {
                  setAnswerCitations(event.citations);
                } else if (event.type === 'complete') {
                  if (event.answer) {
                    setAnswer(event.answer);
                  }
                  if (event.citations) {
                    setAnswerCitations(event.citations);
                  }
                  setIsAnswering(false);
                } else if (event.type === 'error') {
                  setAnswerError(event.message || 'An error occurred while generating the answer');
                  setIsAnswering(false);
                }
              }
            );
          } catch (error) {
            console.error('Failed to answer question:', error);
            setAnswerError(error instanceof Error ? error.message : 'Failed to generate answer');
            setIsAnswering(false);
            toast.error('Failed to generate answer');
          }
        }}
      />

      {/* Streaming Logs */}
      {(isStreaming || logs.length > 0) && (
        <StreamingLogs
          isVisible={true}
        />
      )}

      {/* Selection Actions Bar */}
      {selectedIds.size > 0 && (
        <div className="bg-primary-50 border border-primary-200 rounded-lg p-4 flex items-center justify-between">
          <span className="text-sm font-medium text-primary-700">
            {selectedIds.size} document{selectedIds.size !== 1 ? 's' : ''} selected
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowDeleteConfirm(true)}
            className="text-red-600 border-red-300 hover:bg-red-50 hover:border-red-400"
          >
            <Trash2 className="w-4 h-4 mr-2" />
            Delete Selected
          </Button>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full mx-4 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-full bg-red-100 flex items-center justify-center">
                <AlertTriangle className="w-5 h-5 text-red-600" />
              </div>
              <h3 className="text-lg font-semibold text-primary-900">Confirm Deletion</h3>
            </div>
            <p className="text-primary-600 mb-6">
              Are you sure you want to delete {selectedIds.size} document{selectedIds.size !== 1 ? 's' : ''}? 
              This action cannot be undone and will remove the documents from both the database and vector store.
            </p>
            <div className="flex justify-end gap-3">
              <Button
                variant="outline"
                onClick={() => setShowDeleteConfirm(false)}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                onClick={handleDeleteSelected}
                disabled={isDeleting}
                className="bg-red-600 hover:bg-red-700 text-white"
              >
                {isDeleting ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete
                  </>
                )}
              </Button>
            </div>
          </div>
        </div>
      )}

      {/* Documents Table */}
      {loading ? (
        <div className="flex justify-center py-12">
          <LoadingSpinner size="lg" />
        </div>
      ) : documents.length > 0 ? (
        <>
          <Card>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-primary-200 bg-primary-50/50">
                    <th className="px-4 py-3 text-left w-12">
                      <input
                        type="checkbox"
                        checked={isAllSelected}
                        ref={(el) => {
                          if (el) el.indeterminate = isSomeSelected;
                        }}
                        onChange={handleSelectAll}
                        className="w-4 h-4 rounded border-primary-300 text-primary-600 focus:ring-primary-500 cursor-pointer"
                      />
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      Filename
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      Categories
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      Size
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      Type
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        Processing Time
                      </div>
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold text-primary-600 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-primary-100">
                  {documents.map((doc) => (
                    <tr
                      key={doc.id}
                      className={`hover:bg-primary-50/50 transition-colors ${selectedIds.has(doc.id) ? 'bg-primary-50' : ''}`}
                    >
                      <td className="px-4 py-3">
                        <input
                          type="checkbox"
                          checked={selectedIds.has(doc.id)}
                          onChange={() => handleSelectRow(doc.id)}
                          className="w-4 h-4 rounded border-primary-300 text-primary-600 focus:ring-primary-500 cursor-pointer"
                        />
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-3">
                          <span className="text-xl flex-shrink-0">
                            {getFileIcon(doc.metadata.file_extension)}
                          </span>
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <p
                                className="font-medium text-primary-900 truncate cursor-pointer hover:text-primary-600 transition-colors max-w-xs"
                                onClick={() => navigate(`/document/${doc.id}`)}
                                title={doc.filename}
                              >
                                {doc.filename}
                              </p>
                              {!!doc.deepseek_ocr_used && (
                                <div
                                  className="flex items-center gap-1 px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium"
                                  title="Processed with DeepSeek-OCR"
                                >
                                  <ScanText className="w-3 h-3" />
                                  <span>OCR</span>
                                </div>
                              )}
                            </div>
                            <p className="text-xs text-primary-500 truncate max-w-xs" title={cleanContentPreview(doc.content_preview)}>
                              {truncateContent(cleanContentPreview(doc.content_preview))}
                            </p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex flex-wrap gap-1 max-w-xs">
                          {doc.categories.split('-').slice(0, 2).map((category, index) => (
                            <span
                              key={`main-${index}`}
                              className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-700"
                            >
                              {category.trim()}
                            </span>
                          ))}
                          {doc.categories.split('-').length > 2 && (
                            <span className="text-xs text-primary-500">
                              +{doc.categories.split('-').length - 2}
                            </span>
                          )}
                          {doc.sub_categories && doc.sub_categories.slice(0, 2).map((subCategory, index) => (
                            <span
                              key={`sub-${index}`}
                              className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200"
                            >
                              {subCategory}
                            </span>
                          ))}
                          {doc.sub_categories && doc.sub_categories.length > 2 && (
                            <span className="text-xs text-primary-500">
                              +{doc.sub_categories.length - 2} more
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm text-primary-600 whitespace-nowrap">
                        {formatDate(doc.classification_date)}
                      </td>
                      <td className="px-4 py-3 text-sm text-primary-600 whitespace-nowrap">
                        {formatFileSize(doc.metadata.file_size)}
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700 uppercase">
                          {doc.metadata.file_extension.replace('.', '')}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-primary-600 whitespace-nowrap">
                        {doc.metadata.performance_metrics ?
                          formatProcessingTime(doc.metadata.performance_metrics.total_processing_time) :
                          'N/A'
                        }
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex justify-end gap-1">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => navigate(`/document/${doc.id}`)}
                            className="px-2"
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => {
                              toast.success('Download feature coming soon!');
                            }}
                            className="px-2"
                          >
                            <Download className="w-4 h-4" />
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

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
