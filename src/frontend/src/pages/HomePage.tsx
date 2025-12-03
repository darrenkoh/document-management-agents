import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, ArrowRight, Clock } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { StreamingLogs, useStreamingLogs, StreamingLogMessage } from '@/components/ui/StreamingLogs';
import { DocumentSearchAndQuestion } from '@/components/DocumentSearchAndQuestion';
import { Document, SearchResult, AnswerCitation, AnswerStreamEvent } from '@/types';
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
  
  // Question answering state
  const [questionQuery, setQuestionQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [answerCitations, setAnswerCitations] = useState<AnswerCitation[]>([]);
  const [isAnswering, setIsAnswering] = useState(false);
  const [answerError, setAnswerError] = useState<string | undefined>();
  
  const navigate = useNavigate();

  // Streaming logs hook with callback
  const { isStreaming, startStreaming, clearLogs } = useStreamingLogs((log) => {
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
    <div className="space-y-5">
      {/* Combined Search and Question Component */}
      <div className="max-w-4xl mx-auto">
        <DocumentSearchAndQuestion
          searchQuery={query}
          onSearchQueryChange={setQuery}
          onSearch={handleSearch}
          isSearching={isSearching}
          showCategoryFilter={false}
          searchPlaceholder="Search documents, categories, or content..."
          showClearFilters={false}
          isSemanticSearch={false}
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
      </div>

      {/* Streaming Logs - Always reserve space to prevent layout shifts */}
      <div className="max-w-3xl mx-auto">
        <div
          className={`bg-primary-900 rounded-xl shadow-2xl overflow-hidden border border-primary-800 transition-all duration-300 ease-in-out ${
            (isStreaming || streamingLogs.length > 0) ? 'p-6 opacity-100 max-h-96' : 'p-0 opacity-0 max-h-0'
          }`}
          style={{ overflowAnchor: 'auto' }}
        >
          {(isStreaming || streamingLogs.length > 0) && (
            <StreamingLogs
              isVisible={true}
              logs={streamingLogs}
              isStreaming={isStreaming}
            />
          )}
        </div>
      </div>

      {/* Content Section */}
      <div className="space-y-8">
        {hasSearchResults ? (
          <div className="space-y-6 animate-fade-in">
            <div className="flex items-center justify-between border-b border-primary-200 pb-4">
              <div>
                <h2 className="text-2xl font-bold text-primary-900">Search Results</h2>
                <p className="text-primary-500 mt-1">
                  Found {searchResults.length} document{searchResults.length !== 1 ? 's' : ''} matching "{query.trim()}"
                </p>
              </div>
              <Button variant="outline" onClick={clearSearchResults}>
                Clear Results
              </Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {searchResults.map((doc) => (
                <Card key={doc.id} className="group hover:-translate-y-1 transition-all duration-300">
                  <CardContent className="p-6">
                    <div className="flex items-start gap-4 mb-4">
                      <div className="w-10 h-10 rounded-lg bg-primary-50 flex items-center justify-center text-2xl group-hover:bg-primary-100 transition-colors">
                        {getFileIcon(doc.metadata.file_extension)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-primary-900 truncate cursor-pointer hover:text-primary-600 transition-colors mb-1"
                          onClick={() => navigate(`/document/${doc.id}`)}>
                          {doc.filename}
                        </h3>
                        <div className="flex items-center gap-2">
                          <span className="text-xs bg-green-50 text-green-700 px-2 py-0.5 rounded-full font-medium border border-green-100">
                            {((doc.similarity || 0) * 100).toFixed(0)}% Match
                          </span>
                        </div>
                      </div>
                    </div>

                    <p className="text-sm text-primary-600 mb-4 line-clamp-3 leading-relaxed bg-primary-50/50 p-3 rounded-lg border border-primary-100">
                      {truncateContent(doc.content_preview)}
                    </p>

                    <div className="flex items-center justify-between text-xs text-primary-400 pt-4 border-t border-primary-100">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(doc.classification_date)}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigate(`/document/${doc.id}`)}
                        className="h-auto py-1 px-2 text-primary-600 hover:text-primary-900"
                      >
                        View <ArrowRight className="w-3 h-3 ml-1" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ) : (
          <div className="space-y-12">
            {/* Recent Documents */}
            <div>
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-3xl font-bold text-primary-900">Recent Activity</h2>
                <Button variant="outline" onClick={() => navigate('/documents')}>
                  View All Documents
                </Button>
              </div>

              {isLoadingDocs ? (
                <div className="flex justify-center py-20">
                  <LoadingSpinner size="lg" />
                </div>
              ) : recentDocs.length > 0 ? (
                <div className="bg-white rounded-xl border border-primary-200 overflow-hidden shadow-sm">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="bg-primary-50 text-primary-900 font-medium border-b border-primary-200">
                        <tr>
                          <th className="px-6 py-4">Document</th>
                          <th className="px-6 py-4">Categories</th>
                          <th className="px-6 py-4">Date</th>
                          <th className="px-6 py-4">Size</th>
                          <th className="px-6 py-4 text-right">Action</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-primary-100">
                        {recentDocs.map((doc) => (
                          <tr key={doc.id} className="hover:bg-primary-50/50 transition-colors group cursor-pointer" onClick={() => navigate(`/document/${doc.id}`)}>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="text-xl text-primary-400 group-hover:text-primary-600 transition-colors">
                                  {getFileIcon(doc.metadata.file_extension)}
                                </div>
                                <span className="font-medium text-primary-900">{doc.filename}</span>
                              </div>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex flex-wrap gap-1">
                                {doc.categories.split('-').map((category, index) => (
                                  <span key={`main-${index}`} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700">
                                    {category.trim()}
                                  </span>
                                ))}
                                {doc.sub_categories && doc.sub_categories.map((subCategory, index) => (
                                  <span key={`sub-${index}`} className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200">
                                    {subCategory}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="px-6 py-4 text-primary-600">
                              {formatDate(doc.classification_date)}
                            </td>
                            <td className="px-6 py-4 text-primary-600 font-mono text-xs">
                              {formatFileSize(doc.metadata.file_size)}
                            </td>
                            <td className="px-6 py-4 text-right">
                              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                <ArrowRight className="w-4 h-4" />
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              ) : (
                <Card className="bg-primary-50 border-dashed">
                  <CardContent className="text-center py-16">
                    <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm">
                      <FileText className="w-8 h-8 text-primary-400" />
                    </div>
                    <h3 className="text-xl font-bold text-primary-900 mb-2">No documents found</h3>
                    <p className="text-primary-600 max-w-md mx-auto">
                      Start by uploading and processing some documents to see them appear in your recent activity.
                    </p>
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
