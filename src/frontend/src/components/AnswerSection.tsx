import { useState, useEffect, useRef } from 'react';
import { MessageSquare, FileText, ExternalLink, AlertCircle, X } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { AnswerCitation, AnswerStreamEvent } from '@/types';
import { apiClient, cleanContentPreview } from '@/lib/api';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { ReasoningCollapse } from '@/components/ReasoningCollapse';

interface AnswerSectionProps {
  questionQuery?: string;
  onQuestionQueryChange?: (query: string) => void;
  answer?: string;
  answerCitations?: AnswerCitation[];
  isAnswering?: boolean;
  answerError?: string;
  onAnswerQuestion?: (query: string) => void;
  onClear?: () => void;
  className?: string;
}

export function AnswerSection({
  questionQuery: externalQuestionQuery,
  onQuestionQueryChange: externalOnQuestionQueryChange,
  answer: externalAnswer,
  answerCitations: externalAnswerCitations,
  isAnswering: externalIsAnswering,
  answerError: externalAnswerError,
  onAnswerQuestion: externalOnAnswerQuestion,
  onClear,
  className = '',
}: AnswerSectionProps) {
  const navigate = useNavigate();
  const answerContainerRef = useRef<HTMLDivElement>(null);
  const answerEndRef = useRef<HTMLDivElement>(null);

  // Internal state for question answering if not controlled externally
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

  // Auto-scroll to bottom when answer is streaming (only if user is already near bottom)
  useEffect(() => {
    if (isAnswering && answerContainerRef.current && answerEndRef.current) {
      const container = answerContainerRef.current;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;

      if (isNearBottom) {
        answerEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }
  }, [answer, isAnswering]);

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

  const hasAnswerContent = answer || answerError || isAnswering;

  return (
    <div className={className}>
      <Card className="border-accent-200 bg-gradient-to-br from-accent-50/30 to-white">
        <CardContent className="p-6">
          {/* Question Input */}
          <form onSubmit={handleQuestionSubmit}>
            <div className="flex items-center gap-2 mb-4">
              <MessageSquare className="w-5 h-5 text-accent-600" />
              <h2 className="text-lg font-semibold text-primary-900">Ask a Question</h2>
            </div>
            <div className="flex gap-3 mb-6">
              <div className="flex-1 relative">
                <MessageSquare className="absolute left-3 top-1/2 transform -translate-y-1/2 text-accent-500 w-5 h-5" />
                <Input
                  type="text"
                  placeholder="Ask a question about your documents..."
                  value={questionQuery}
                  onChange={(e) => setQuestionQuery(e.target.value)}
                  className="pl-10 pr-10"
                  disabled={isAnswering}
                />
                {questionQuery && onClear && (
                  <button
                    type="button"
                    onClick={onClear}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-accent-400 hover:text-accent-600 transition-colors"
                    disabled={isAnswering}
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
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

          {/* Answer Display */}
          {hasAnswerContent && (
            <div ref={answerContainerRef} className="border-t border-primary-200 pt-6">
              {/* Error State */}
              {answerError && (
                <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg mb-4">
                  <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-red-900">Error generating answer</p>
                    <p className="text-sm text-red-700 mt-1">{answerError}</p>
                  </div>
                </div>
              )}

              {/* Loading State */}
              {!answer && !answerError && isAnswering && (
                <div className="flex items-center gap-2 text-primary-600 py-4">
                  <LoadingSpinner size="sm" />
                  <span className="text-sm">Searching documents and generating answer...</span>
                </div>
              )}

              {/* Answer Content */}
              {answer && (
                <div className="mb-4">
                  <div className="prose prose-sm max-w-none">
                    <div className="text-primary-800 leading-relaxed whitespace-pre-wrap">
                      <ReasoningCollapse text={answer} isStreaming={isAnswering} />
                    </div>
                  </div>
                </div>
              )}

              {/* Citations */}
              {answerCitations && answerCitations.length > 0 && (
                <div className="mt-6 pt-6 border-t border-primary-200">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-primary-600" />
                      <h4 className="text-sm font-semibold text-primary-900">Sources</h4>
                      <span className="text-xs text-primary-500 bg-primary-100 px-2 py-0.5 rounded-full">
                        {answerCitations.length}
                      </span>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        navigate('/documents');
                      }}
                      className="text-xs"
                    >
                      View All Sources
                      <ExternalLink className="w-3 h-3 ml-1" />
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {answerCitations.map((citation, index) => (
                      <div
                        key={citation.id || index}
                        className="flex items-start gap-3 p-3 bg-white border border-primary-200 rounded-lg hover:border-accent-300 hover:shadow-sm transition-all cursor-pointer group"
                        onClick={() => citation.id && navigate(`/document/${citation.id}`)}
                      >
                        <div className="flex-shrink-0 w-8 h-8 rounded bg-accent-100 flex items-center justify-center text-xs font-semibold text-accent-700">
                          {index + 1}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <p className="text-sm font-medium text-primary-900 truncate group-hover:text-accent-700 transition-colors">
                              {citation.filename}
                            </p>
                            <ExternalLink className="w-3 h-3 text-primary-400 group-hover:text-accent-600 transition-colors flex-shrink-0" />
                          </div>
                          {citation.categories && (
                            <div className="flex flex-wrap gap-1 mb-2">
                              {citation.categories.split('-').slice(0, 2).map((category, catIndex) => (
                                <span
                                  key={catIndex}
                                  className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700"
                                >
                                  {category.trim()}
                                </span>
                              ))}
                            </div>
                          )}
                          {citation.content_preview && (
                            <p className="text-xs text-primary-600 line-clamp-2">
                              {cleanContentPreview(citation.content_preview)}
                            </p>
                          )}
                          {citation.similarity !== undefined && (
                            <p className="text-xs text-primary-500 mt-1">
                              Similarity: {(citation.similarity * 100).toFixed(1)}%
                            </p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Scroll anchor */}
              <div ref={answerEndRef} />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}














