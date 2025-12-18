import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, FileText, Hash, Cpu, ArrowRight, Info, Sparkles, Layers } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { EmbeddingSearchResponse, EmbeddingSearchResult } from '@/types';
import { apiClient, getFileIcon, cleanContentPreview } from '@/lib/api';
import toast from 'react-hot-toast';

export default function EmbeddingSearchPage() {
  const [keyword, setKeyword] = useState('');
  const [limit, setLimit] = useState(10);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResponse, setSearchResponse] = useState<EmbeddingSearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const navigate = useNavigate();

  const handleSearch = async () => {
    if (!keyword.trim()) {
      toast.error('Please enter a keyword to search');
      return;
    }

    setIsSearching(true);
    setError(null);
    setSearchResponse(null);

    try {
      const response = await apiClient.searchEmbeddings(keyword.trim(), limit);
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      setSearchResponse(response);
      
      if (response.count === 0) {
        toast('No matching documents found', { icon: '🔍' });
      } else {
        toast.success(`Found ${response.count} matching document${response.count !== 1 ? 's' : ''}`);
      }
    } catch (err: any) {
      console.error('Embedding search failed:', err);
      const errorMessage = err.response?.data?.error || err.message || 'Failed to search embeddings';
      setError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isSearching) {
      handleSearch();
    }
  };

  const formatSimilarity = (similarity: number): string => {
    return (similarity * 100).toFixed(1) + '%';
  };

  const getSimilarityColor = (similarity: number): string => {
    if (similarity >= 0.8) return 'bg-green-100 text-green-700 border-green-200';
    if (similarity >= 0.6) return 'bg-blue-100 text-blue-700 border-blue-200';
    if (similarity >= 0.4) return 'bg-yellow-100 text-yellow-700 border-yellow-200';
    return 'bg-gray-100 text-gray-700 border-gray-200';
  };

  const truncateContent = (content: string, length = 200) => {
    if (!content) return '';
    const cleaned = cleanContentPreview(content);
    if (cleaned.length <= length) return cleaned;
    return cleaned.substring(0, length) + '...';
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold text-primary-900 flex items-center justify-center gap-3">
          <Sparkles className="w-8 h-8 text-primary-600" />
          Embedding Search
        </h1>
        <p className="text-primary-600 max-w-2xl mx-auto">
          Search the embedding database by keyword. See how your query is tokenized and find semantically similar documents.
        </p>
      </div>

      {/* Search Input */}
      <Card className="overflow-hidden">
        <CardContent className="p-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-primary-400" />
              <Input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter a keyword or phrase to search..."
                className="pl-12 h-12 text-lg"
                disabled={isSearching}
              />
            </div>
            <div className="flex items-center gap-3">
              <label className="text-sm text-primary-600 whitespace-nowrap">Results:</label>
              <select
                value={limit}
                onChange={(e) => setLimit(parseInt(e.target.value))}
                className="h-12 px-4 border border-primary-200 rounded-lg bg-white text-primary-900 focus:outline-none focus:ring-2 focus:ring-primary-500"
                disabled={isSearching}
              >
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
              <Button
                variant="primary"
                onClick={handleSearch}
                disabled={isSearching || !keyword.trim()}
                className="h-12 px-6"
              >
                {isSearching ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4 mr-2" />
                    Search
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error State */}
      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6 text-center">
            <p className="text-red-700">{error}</p>
          </CardContent>
        </Card>
      )}

      {/* Loading State */}
      {isSearching && (
        <Card>
          <CardContent className="p-12 text-center">
            <LoadingSpinner size="lg" className="mx-auto mb-4" />
            <p className="text-primary-600">Generating embedding and searching...</p>
          </CardContent>
        </Card>
      )}

      {/* Results */}
      {searchResponse && !isSearching && (
        <div className="space-y-6 animate-fade-in">
          {/* Embedding Info Section */}
          <Card className="bg-gradient-to-br from-primary-50 to-blue-50 border-primary-200">
            <CardContent className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <Cpu className="w-5 h-5 text-primary-600" />
                <h2 className="text-lg font-semibold text-primary-900">Embedding Information</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Word Breakdown */}
                <div className="bg-white/80 rounded-lg p-4 border border-primary-100">
                  <div className="flex items-center gap-2 mb-3">
                    <Hash className="w-4 h-4 text-primary-500" />
                    <h3 className="font-medium text-primary-800">Word Tokens</h3>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {searchResponse.words.map((word, index) => (
                      <span
                        key={index}
                        className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-700 border border-primary-200"
                      >
                        {word}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Vector Info */}
                <div className="bg-white/80 rounded-lg p-4 border border-primary-100">
                  <div className="flex items-center gap-2 mb-3">
                    <Layers className="w-4 h-4 text-primary-500" />
                    <h3 className="font-medium text-primary-800">Vector Metadata</h3>
                  </div>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-primary-600">Dimension:</span>
                      <span className="font-mono text-primary-900">{searchResponse.embedding_info.dimension}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-primary-600">Model:</span>
                      <span className="font-mono text-primary-900 text-xs">{searchResponse.embedding_info.model}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-primary-600">Min/Max:</span>
                      <span className="font-mono text-primary-900 text-xs">
                        {searchResponse.embedding_info.min_value.toFixed(4)} / {searchResponse.embedding_info.max_value.toFixed(4)}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-primary-600">Mean:</span>
                      <span className="font-mono text-primary-900 text-xs">
                        {searchResponse.embedding_info.mean_value.toFixed(6)}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Sample Values */}
                <div className="bg-white/80 rounded-lg p-4 border border-primary-100 md:col-span-2 lg:col-span-1">
                  <div className="flex items-center gap-2 mb-3">
                    <Info className="w-4 h-4 text-primary-500" />
                    <h3 className="font-medium text-primary-800">Sample Values (first 10)</h3>
                  </div>
                  <div className="font-mono text-xs text-primary-700 bg-primary-50 rounded p-2 overflow-x-auto">
                    [{searchResponse.embedding_info.sample_values.map(v => v.toFixed(4)).join(', ')}...]
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Search Results */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-primary-900">
                Search Results ({searchResponse.count})
              </h2>
              <span className="text-sm text-primary-500">
                Keyword: "<span className="font-medium">{searchResponse.keyword}</span>"
              </span>
            </div>

            {searchResponse.results.length === 0 ? (
              <Card className="bg-primary-50 border-dashed">
                <CardContent className="text-center py-12">
                  <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 shadow-sm">
                    <FileText className="w-8 h-8 text-primary-400" />
                  </div>
                  <h3 className="text-xl font-bold text-primary-900 mb-2">No matching documents</h3>
                  <p className="text-primary-600 max-w-md mx-auto">
                    No documents in the embedding database match your search keyword. Try a different term.
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="bg-white rounded-xl border border-primary-200 overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-primary-50 text-primary-900 font-medium border-b border-primary-200">
                      <tr>
                        <th className="px-6 py-4 w-16">#</th>
                        <th className="px-6 py-4">Document</th>
                        <th className="px-6 py-4">Categories</th>
                        <th className="px-6 py-4">Similarity</th>
                        <th className="px-6 py-4">Preview</th>
                        <th className="px-6 py-4 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-primary-100">
                      {searchResponse.results.map((result, index) => (
                        <ResultRow
                          key={result.id}
                          result={result}
                          index={index}
                          formatSimilarity={formatSimilarity}
                          getSimilarityColor={getSimilarityColor}
                          truncateContent={truncateContent}
                          navigate={navigate}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Initial State */}
      {!searchResponse && !isSearching && !error && (
        <Card className="bg-gradient-to-br from-primary-50 to-blue-50 border-primary-200">
          <CardContent className="text-center py-16">
            <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
              <Search className="w-10 h-10 text-primary-400" />
            </div>
            <h3 className="text-2xl font-bold text-primary-900 mb-3">Search by Embedding</h3>
            <p className="text-primary-600 max-w-lg mx-auto mb-6">
              Enter a keyword or phrase above to generate an embedding and find semantically similar documents in your database.
            </p>
            <div className="flex flex-wrap justify-center gap-2 text-sm text-primary-500">
              <span className="bg-white px-3 py-1 rounded-full border border-primary-200">Semantic Search</span>
              <span className="bg-white px-3 py-1 rounded-full border border-primary-200">Vector Similarity</span>
              <span className="bg-white px-3 py-1 rounded-full border border-primary-200">AI-Powered</span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Separate component for result rows to keep the main component cleaner
interface ResultRowProps {
  result: EmbeddingSearchResult;
  index: number;
  formatSimilarity: (similarity: number) => string;
  getSimilarityColor: (similarity: number) => string;
  truncateContent: (content: string, length?: number) => string;
  navigate: (path: string) => void;
}

function ResultRow({ result, index, formatSimilarity, getSimilarityColor, truncateContent, navigate }: ResultRowProps) {
  // Try to extract numeric ID for navigation
  const docId = typeof result.id === 'number' ? result.id : 
                typeof result.id === 'string' && result.id.includes('_') ? 
                parseInt(result.id.split('_')[0]) : null;
  
  const extension = result.metadata?.file_extension || 
                    (result.filename.includes('.') ? '.' + result.filename.split('.').pop() : '');

  return (
    <tr className="hover:bg-primary-50/50 transition-colors group">
      <td className="px-6 py-4 text-primary-400 font-mono text-xs">
        {index + 1}
      </td>
      <td className="px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="text-xl text-primary-400 group-hover:text-primary-600 transition-colors">
            {getFileIcon(extension)}
          </div>
          <span 
            className={`font-medium text-primary-900 ${docId ? 'cursor-pointer hover:text-primary-600' : ''}`}
            onClick={() => docId && navigate(`/document/${docId}`)}
          >
            {result.filename}
          </span>
        </div>
      </td>
      <td className="px-6 py-4">
        <div className="flex flex-wrap gap-1">
          {result.categories.split('-').map((category, idx) => (
            <span 
              key={`cat-${idx}`} 
              className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700"
            >
              {category.trim()}
            </span>
          ))}
          {result.sub_categories?.map((subCat, idx) => (
            <span 
              key={`sub-${idx}`} 
              className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200"
            >
              {subCat}
            </span>
          ))}
        </div>
      </td>
      <td className="px-6 py-4">
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border ${getSimilarityColor(result.similarity)}`}>
          {formatSimilarity(result.similarity)}
        </span>
      </td>
      <td className="px-6 py-4 max-w-md">
        <p className="text-xs text-primary-600 line-clamp-2">
          {truncateContent(result.content_preview, 150)}
        </p>
      </td>
      <td className="px-6 py-4 text-right">
        {docId && (
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={() => navigate(`/document/${docId}`)}
            className="h-8 w-8 p-0"
          >
            <ArrowRight className="w-4 h-4" />
          </Button>
        )}
      </td>
    </tr>
  );
}
