import { useEffect, useRef } from 'react';
import { Card, CardContent } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { AnswerCitation } from '@/types';
import { MessageSquare, FileText, ExternalLink, AlertCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AnswerDisplayProps {
  answer: string;
  citations: AnswerCitation[];
  isStreaming: boolean;
  error?: string;
  question?: string;
}

export function AnswerDisplay({ answer, citations, isStreaming, error, question }: AnswerDisplayProps) {
  const navigate = useNavigate();
  const answerContainerRef = useRef<HTMLDivElement>(null);
  const answerEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when answer is streaming (only if user is already near bottom)
  useEffect(() => {
    if (isStreaming && answerContainerRef.current && answerEndRef.current) {
      const container = answerContainerRef.current;
      const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 100;

      if (isNearBottom) {
        answerEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      }
    }
  }, [answer, isStreaming]);

  // Don't render if there's no answer and no error and not streaming
  if (!answer && !error && !isStreaming) {
    return null;
  }

  return (
    <Card className="mb-6 border-accent-200 bg-gradient-to-br from-accent-50/50 to-white">
      <CardContent ref={answerContainerRef} className="p-6">
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-accent-100 flex items-center justify-center">
            <MessageSquare className="w-5 h-5 text-accent-600" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-primary-900">AI Answer</h3>
            {question && (
              <p className="text-sm text-primary-600 mt-0.5">Question: "{question}"</p>
            )}
          </div>
          {isStreaming && (
            <div className="flex items-center gap-2 text-sm text-accent-600">
              <LoadingSpinner size="sm" />
              <span>Generating...</span>
            </div>
          )}
        </div>

        {/* Error State */}
        {error && (
          <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg mb-4">
            <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-red-900">Error generating answer</p>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        )}

        {/* Answer Content */}
        {answer && (
          <div className="prose prose-sm max-w-none mb-4">
            <div className="text-primary-800 leading-relaxed whitespace-pre-wrap">
              {answer}
              {isStreaming && (
                <span className="inline-block w-2 h-4 bg-accent-500 ml-1 animate-pulse" />
              )}
            </div>
          </div>
        )}

        {/* Loading State */}
        {!answer && !error && isStreaming && (
          <div className="flex items-center gap-2 text-primary-600 py-4">
            <LoadingSpinner size="sm" />
            <span className="text-sm">Searching documents and generating answer...</span>
          </div>
        )}

        {/* Citations */}
        {citations && citations.length > 0 && (
          <div className="mt-6 pt-6 border-t border-primary-200">
            <div className="flex items-center gap-2 mb-3">
              <FileText className="w-4 h-4 text-primary-600" />
              <h4 className="text-sm font-semibold text-primary-900">Sources</h4>
              <span className="text-xs text-primary-500 bg-primary-100 px-2 py-0.5 rounded-full">
                {citations.length}
              </span>
            </div>
            <div className="space-y-2">
              {citations.map((citation, index) => (
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
                        {citation.content_preview}
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
      </CardContent>
    </Card>
  );
}

