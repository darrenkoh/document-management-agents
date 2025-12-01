import { useState, useRef, useEffect } from 'react';
import { Search, Filter, Brain, MessageSquare, ChevronDown, Check } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { AnswerDisplay } from '@/components/AnswerDisplay';
import { AnswerCitation, AnswerStreamEvent } from '@/types';
import { apiClient } from '@/lib/api';
import toast from 'react-hot-toast';

interface DocumentSearchAndQuestionProps {
  // Search props
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  onSearch: (query: string) => void;
  isSearching?: boolean;
  categoryFilter?: string;
  onCategoryFilterChange?: (category: string) => void;
  availableCategories?: string[];
  showCategoryFilter?: boolean;
  searchPlaceholder?: string;
  showClearFilters?: boolean;
  onClearFilters?: () => void;
  isSemanticSearch?: boolean;
  
  // Question answering props
  showQuestionAnswering?: boolean;
  questionQuery?: string;
  onQuestionQueryChange?: (query: string) => void;
  answer?: string;
  answerCitations?: AnswerCitation[];
  isAnswering?: boolean;
  answerError?: string;
  onAnswerQuestion?: (query: string) => void;
  
  // Layout props
  variant?: 'default' | 'compact';
  className?: string;
}

export function DocumentSearchAndQuestion({
  searchQuery,
  onSearchQueryChange,
  onSearch,
  isSearching = false,
  categoryFilter = '',
  onCategoryFilterChange,
  availableCategories = [],
  showCategoryFilter = true,
  searchPlaceholder = 'Search documents...',
  showClearFilters = false,
  onClearFilters,
  isSemanticSearch = false,
  showQuestionAnswering = true,
  questionQuery: externalQuestionQuery,
  onQuestionQueryChange: externalOnQuestionQueryChange,
  answer: externalAnswer,
  answerCitations: externalAnswerCitations,
  isAnswering: externalIsAnswering,
  answerError: externalAnswerError,
  onAnswerQuestion: externalOnAnswerQuestion,
  variant = 'default',
  className = '',
}: DocumentSearchAndQuestionProps) {
  // Internal state for question answering if not controlled
  const [internalQuestionQuery, setInternalQuestionQuery] = useState('');
  const [internalAnswer, setInternalAnswer] = useState('');
  const [internalAnswerCitations, setInternalAnswerCitations] = useState<AnswerCitation[]>([]);
  const [internalIsAnswering, setIsInternalAnswering] = useState(false);
  const [internalAnswerError, setInternalAnswerError] = useState<string | undefined>();

  // Use external or internal state
  const questionQuery = externalQuestionQuery !== undefined ? externalQuestionQuery : internalQuestionQuery;
  const answer = externalAnswer !== undefined ? externalAnswer : internalAnswer;
  const answerCitations = externalAnswerCitations !== undefined ? externalAnswerCitations : internalAnswerCitations;
  const isAnswering = externalIsAnswering !== undefined ? externalIsAnswering : internalIsAnswering;
  const answerError = externalAnswerError !== undefined ? externalAnswerError : internalAnswerError;

  const setQuestionQuery = externalOnQuestionQueryChange || setInternalQuestionQuery;

  // Category combobox state
  const [isCategoryOpen, setIsCategoryOpen] = useState(false);
  const [categorySearch, setCategorySearch] = useState('');
  const categoryRef = useRef<HTMLDivElement>(null);

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

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onSearch(searchQuery.trim());
    }
  };

  const handleQuestionSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!questionQuery.trim()) {
      return;
    }

    // If external handler provided, use it
    if (externalOnAnswerQuestion) {
      externalOnAnswerQuestion(questionQuery.trim());
      return;
    }

    // Otherwise, handle internally
    setIsInternalAnswering(true);
    setInternalAnswer('');
    setInternalAnswerCitations([]);
    setInternalAnswerError(undefined);

    try {
      await apiClient.answerQuestion(
        questionQuery.trim(),
        (chunk) => {
          setInternalAnswer(prev => prev + chunk);
        },
        (event: AnswerStreamEvent) => {
          if (event.type === 'citations' && event.citations) {
            setInternalAnswerCitations(event.citations);
          } else if (event.type === 'complete') {
            if (event.answer) {
              setInternalAnswer(event.answer);
            }
            if (event.citations) {
              setInternalAnswerCitations(event.citations);
            }
            setIsInternalAnswering(false);
          } else if (event.type === 'error') {
            setInternalAnswerError(event.message || 'An error occurred while generating the answer');
            setIsInternalAnswering(false);
          }
        }
      );
    } catch (error) {
      console.error('Failed to answer question:', error);
      setInternalAnswerError(error instanceof Error ? error.message : 'Failed to generate answer');
      setIsInternalAnswering(false);
      toast.error('Failed to generate answer');
    }
  };

  const handleCategoryChange = (category: string) => {
    if (onCategoryFilterChange) {
      onCategoryFilterChange(category);
    }
    setIsCategoryOpen(false);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Question Answering Section */}
      {showQuestionAnswering && (
        <>
          <Card className="border-accent-200 bg-gradient-to-br from-accent-50/30 to-white">
            <CardContent className="p-6">
              <form onSubmit={handleQuestionSubmit} className="space-y-4">
                <div className="flex items-center gap-2 mb-2">
                  <MessageSquare className="w-5 h-5 text-accent-600" />
                  <h2 className="text-lg font-semibold text-primary-900">Ask a Question</h2>
                </div>
                <p className="text-sm text-primary-600 mb-4">
                  Get AI-powered answers based on your documents. The answer will be generated from the most relevant documents in your collection.
                </p>
                <div className="flex gap-3">
                  <div className="flex-1 relative">
                    <MessageSquare className="absolute left-3 top-1/2 transform -translate-y-1/2 text-accent-500 w-5 h-5" />
                    <Input
                      type="text"
                      placeholder="Ask a question about your documents..."
                      value={questionQuery}
                      onChange={(e) => setQuestionQuery(e.target.value)}
                      className="pl-10"
                      disabled={isAnswering}
                    />
                  </div>
                  <Button
                    type="submit"
                    disabled={!questionQuery.trim() || isAnswering}
                    className="bg-accent-600 hover:bg-accent-700 text-white"
                  >
                    {isAnswering ? (
                      <>
                        <LoadingSpinner size="sm" className="mr-2" />
                        Answering...
                      </>
                    ) : (
                      <>
                        <MessageSquare className="w-4 h-4 mr-2" />
                        Ask
                      </>
                    )}
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          {/* Answer Display */}
          <AnswerDisplay
            answer={answer}
            citations={answerCitations}
            isStreaming={isAnswering}
            error={answerError}
            question={questionQuery}
          />
        </>
      )}

      {/* Search and Filter Section */}
      <Card>
        <CardContent className="p-6">
          <form onSubmit={handleSearchSubmit} className="flex flex-col lg:flex-row gap-4">
            <div className="flex-1 relative">
              {isSemanticSearch ? (
                <Brain className="absolute left-3 top-1/2 transform -translate-y-1/2 text-accent-500 w-5 h-5" />
              ) : (
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-primary-400 w-5 h-5" />
              )}
              <Input
                type="text"
                placeholder={searchPlaceholder}
                value={searchQuery}
                onChange={(e) => onSearchQueryChange(e.target.value)}
                className="pl-10"
                disabled={isSearching}
              />
            </div>

            {/* Category Combobox */}
            {showCategoryFilter && availableCategories.length > 0 && (
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
                        onClick={() => handleCategoryChange('')}
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
                            onClick={() => handleCategoryChange(category)}
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
            )}

            {/* Clear Filters */}
            {showClearFilters && onClearFilters && (
              <Button variant="outline" onClick={onClearFilters} type="button">
                Clear Filters
              </Button>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

