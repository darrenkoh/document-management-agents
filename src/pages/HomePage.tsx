import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, FileText, BarChart3, Clock, Eye, Download } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { StreamingLogs, useStreamingLogs, StreamingLogMessage } from '@/components/ui/StreamingLogs';
import { Document, SearchResult } from '@/types';
import { apiClient, getFileIcon, formatFileSize } from '@/lib/api';
import toast from 'react-hot-toast';

export default function HomePage() {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [recentDocs, setRecentDocs] = useState<Document[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);
  const [streamingLogs, setStreamingLogs] = useState<StreamingLogMessage[]>([]);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [hasSearchResults, setHasSearchResults] = useState(false);
  const navigate = useNavigate();

  // Streaming logs hook with callback
  const { isStreaming, startStreaming, clearLogs, cancelSearch } = useStreamingLogs((log) => {
    setStreamingLogs(prev => [...prev, log]);
  });


  useEffect(() => {
    loadRecentDocuments();
  }, []);

  const loadRecentDocuments = async () => {
    try {
      const response = await apiClient.getDocuments({ limit: 10 });
      setRecentDocs(response.documents);
    } catch (error) {
      console.error('Failed to load recent documents:', error);
      toast.error('Failed to load recent documents');
    } finally {
      setIsLoadingDocs(false);
    }
  };

  const handleSearch = async () => {
    if (!query.trim()) {
      toast.error('Please enter a search query');
      return;
    }

    setIsSearching(true);
    setStreamingLogs([]);
    setHasSearchResults(false);
    clearLogs();

    startStreaming(query.trim(),
      (results) => {
        // Success callback
        setSearchResults(results.results);
        setHasSearchResults(true);
        if (results.results.length === 0) {
          toast.error('No documents found matching your query');
        } else {
          toast.success(`Found ${results.results.length} documents matching "${query.trim()}"`);
        }
        setIsSearching(false);
      },
      (error) => {
        // Error callback - don't show error if cancelled
        if (error !== 'cancelled') {
          console.error('Streaming semantic search failed:', error);
          toast.error(`Search failed: ${error}`);
        }
        setIsSearching(false);
      }
    );
  };

  const handleCancel = () => {
    console.log('Cancelling search...');
    cancelSearch();
    setIsSearching(false);
    setStreamingLogs([]);
    setHasSearchResults(false);
    setSearchResults([]);
    clearLogs();
    toast('Search cancelled');
  };

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  const truncateContent = (content: string, length = 150) => {
    if (content.length <= length) return content;
    return content.substring(0, length) + '...';
  };

  const clearSearchResults = () => {
    setHasSearchResults(false);
    setSearchResults([]);
    setQuery('');
    setStreamingLogs([]);
    clearLogs();
  };

  return (
    <div className="space-y-8">
      {/* Hero Section with Search */}
      <div className="text-center">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            My Doc
            <span className="text-primary-600"> System</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Search, organize, and manage your documents with AI-powered classification
          </p>

          {/* Search Form */}
          <div className="max-w-2xl mx-auto">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <Input
                type="text"
                placeholder="Search documents, categories, or content..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !isSearching && query.trim()) {
                    handleSearch();
                  }
                }}
                className="pl-12 pr-24 py-4 text-lg h-14"
                disabled={isSearching}
              />
              <div className="absolute inset-y-0 right-0 pr-4 flex items-center">
                <Button
                  disabled={!query.trim()}
                  variant={isSearching ? "destructive" : "default"}
                  className="h-10 px-6"
                  onClick={isSearching ? handleCancel : handleSearch}
                >
                  {isSearching ? (
                    'Cancel'
                  ) : (
                    'Search'
                  )}
                </Button>
              </div>
            </div>
          </div>

          {/* Streaming Logs - show during search */}
          {(isStreaming || streamingLogs.length > 0) && (
            <div className="mt-6 max-w-2xl mx-auto">
              <StreamingLogs
                isVisible={true}
                logs={streamingLogs}
                isStreaming={isStreaming}
              />
            </div>
          )}

          {/* Search Examples */}
          <div className="mt-6 text-sm text-gray-500">
            <p>Try searching for: "travel documents", "financial statements", or "confirmation"</p>
          </div>
        </div>
      </div>

      {/* Search Results Section */}
      {hasSearchResults && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Search Results</h2>
              <p className="text-gray-600 mt-1">
                Found {searchResults.length} document{searchResults.length !== 1 ? 's' : ''} matching "{query.trim()}"
              </p>
            </div>
            <Button variant="outline" onClick={clearSearchResults}>
              Clear Results
            </Button>
          </div>

          {searchResults.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {searchResults.map((doc) => (
                <Card key={doc.id} className="hover:shadow-lg transition-all duration-300 border-0 shadow-md bg-white/80 backdrop-blur-sm">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4 mb-4">
                      <div className="text-4xl flex-shrink-0">
                        {getFileIcon(doc.metadata.file_extension)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-gray-900 truncate cursor-pointer hover:text-primary-600 transition-colors mb-2"
                            onClick={() => navigate(`/document/${doc.id}`)}>
                          {doc.filename}
                        </h3>
                        <div className="flex items-center gap-2 mb-3">
                          <span className="text-xs bg-primary-100 text-primary-800 px-2 py-1 rounded-full font-medium">
                            {(doc.similarity || 0 * 100).toFixed(1)}% match
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-1 mb-4">
                      {doc.categories.split('-').map((category, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gradient-to-r from-primary-50 to-primary-100 text-primary-700 border border-primary-200"
                        >
                          {category.trim()}
                        </span>
                      ))}
                    </div>

                    <p className="text-sm text-gray-600 mb-4 line-clamp-3 leading-relaxed">
                      {truncateContent(doc.content_preview)}
                    </p>

                    <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(doc.classification_date)}
                      </span>
                      <span>{formatFileSize(doc.metadata.file_size)}</span>
                    </div>

                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => navigate(`/document/${doc.id}`)}
                        className="flex-1 hover:bg-primary-50 hover:border-primary-300 transition-colors"
                      >
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          // TODO: Implement download
                          toast.success('Download feature coming soon!');
                        }}
                        className="hover:bg-gray-50 transition-colors"
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <Card className="border-2 border-dashed border-gray-200">
              <CardContent className="text-center py-12">
                <Search className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
                <p className="text-gray-600 mb-4">
                  Try adjusting your search query or check if documents have been processed with embeddings.
                </p>
                <Button onClick={clearSearchResults}>Clear Search</Button>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Quick Actions - hide when showing search results */}
      {!hasSearchResults && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card className="hover:shadow-medium transition-shadow cursor-pointer" onClick={() => navigate('/documents')}>
          <CardHeader>
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
              <FileText className="w-6 h-6 text-primary-600" />
            </div>
            <CardTitle className="text-lg">Browse Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600">
              View and manage all your classified documents with advanced filtering options.
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-medium transition-shadow cursor-pointer" onClick={() => navigate('/stats')}>
          <CardHeader>
            <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center mb-4">
              <BarChart3 className="w-6 h-6 text-success-600" />
            </div>
            <CardTitle className="text-lg">View Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600">
              Analyze your document collection with detailed statistics and insights.
            </p>
          </CardContent>
        </Card>

        <Card className="hover:shadow-medium transition-shadow">
          <CardHeader>
            <div className="w-12 h-12 bg-warning-100 rounded-lg flex items-center justify-center mb-4">
              <Clock className="w-6 h-6 text-warning-600" />
            </div>
            <CardTitle className="text-lg">Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-gray-600">
              {recentDocs.length > 0
                ? `${recentDocs.length} documents processed recently`
                : 'No recent documents'
              }
            </p>
          </CardContent>
        </Card>
      </div>
      )}

      {/* Recent Documents - hide when showing search results */}
      {!hasSearchResults && (
        <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Recent Documents</h2>
          <Button variant="outline" onClick={() => navigate('/documents')}>
            View All
          </Button>
        </div>

        {isLoadingDocs ? (
          <div className="flex justify-center py-12">
            <LoadingSpinner size="lg" />
          </div>
        ) : recentDocs.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {recentDocs.map((doc) => (
              <Card
                key={doc.id}
                className="hover:shadow-medium transition-shadow cursor-pointer"
                onClick={() => navigate(`/document/${doc.id}`)}
              >
                <CardContent className="p-6">
                  <div className="flex items-start gap-4">
                    <div className="text-3xl">
                      {getFileIcon(doc.metadata.file_extension)}
                    </div>
                    <div className="flex-1 min-w-0">
                      <h3 className="font-semibold text-gray-900 truncate mb-2">
                        {doc.filename}
                      </h3>
                      <div className="flex flex-wrap gap-1 mb-3">
                        {doc.categories.split('-').map((category, index) => (
                          <span
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                          >
                            {category.trim()}
                          </span>
                        ))}
                      </div>
                      <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                        {truncateContent(doc.content_preview)}
                      </p>
                      <div className="text-xs text-gray-500">
                        {formatDate(doc.classification_date)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="text-center py-12">
              <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
              <p className="text-gray-600">
                Start by uploading and processing some documents to see them here.
              </p>
            </CardContent>
          </Card>
        )}
        </div>
      )}
    </div>
  );
}
