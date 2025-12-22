import { useState } from 'react';
import { MessageSquare } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { AnswerDisplay } from '@/components/AnswerDisplay';
import { AnswerCitation, AnswerStreamEvent } from '@/types';
import { apiClient } from '@/lib/api';
import toast from 'react-hot-toast';

interface AnswerSectionProps {
  questionQuery?: string;
  onQuestionQueryChange?: (query: string) => void;
  answer?: string;
  answerCitations?: AnswerCitation[];
  isAnswering?: boolean;
  answerError?: string;
  onAnswerQuestion?: (query: string) => void;
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
  className = '',
}: AnswerSectionProps) {
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

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Question Answering Section */}
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
    </div>
  );
}













