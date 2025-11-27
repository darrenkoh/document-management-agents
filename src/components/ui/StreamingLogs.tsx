import { useState, useEffect, useRef } from 'react';
import { flushSync } from 'react-dom';
import { ChevronDown, ChevronRight, Brain, Loader2, CheckCircle, XCircle } from 'lucide-react';

export interface StreamingLogMessage {
  type: 'start' | 'log' | 'complete' | 'error' | 'results';
  message?: string;
  results?: number;
  data?: any;
}

interface StreamingLogsProps {
  isVisible: boolean;
  logs?: StreamingLogMessage[];
  isStreaming?: boolean;
}

export function StreamingLogs({ isVisible, logs = [], isStreaming = false }: StreamingLogsProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  console.log('StreamingLogs render - logs:', logs.length, 'isStreaming:', isStreaming);

  // Auto-expand when streaming starts
  useEffect(() => {
    if (isStreaming && !isExpanded) {
      setIsExpanded(true);
    }
  }, [isStreaming, isExpanded]);

  // Auto-scroll to bottom when new logs arrive (only if expanded)
  useEffect(() => {
    if (logs.length > 0 && isExpanded) {
      scrollToBottom();
    }
  }, [logs.length, isExpanded]);

  const scrollToBottom = () => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  };


  const getLogIcon = (type: string) => {
    switch (type) {
      case 'start':
        return <Brain className="w-3 h-3 text-blue-500 mt-0.5 flex-shrink-0" />;
      case 'complete':
        return <CheckCircle className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />;
      case 'error':
        return <XCircle className="w-3 h-3 text-red-500 mt-0.5 flex-shrink-0" />;
      default:
        return null;
    }
  };


  if (!isVisible) {
    return null;
  }

  return (
    <div className="border border-gray-200 rounded-lg bg-white shadow-sm mt-4">
      {/* Collapsible Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors rounded-lg"
      >
        <div className="flex items-center gap-3">
          <Brain className="w-4 h-4 text-blue-600" />
          <span className="text-sm font-medium text-gray-700">
            {isStreaming ? 'Thinking...' : 'Search completed'}
          </span>
          {isStreaming && (
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
          )}
        </div>
        <div className="flex items-center gap-2">
          {logs.length > 0 && (
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {logs.length} steps
            </span>
          )}
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </button>

      {/* Collapsible Content */}
      {isExpanded && (
        <div className="border-t border-gray-100">
          {logs.length === 0 && isStreaming ? (
            <div className="p-4 text-sm text-gray-500 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" />
              Initializing search...
            </div>
          ) : (
            <div ref={logsContainerRef} className="p-4 max-h-24 overflow-y-auto">
              <div className="space-y-1.5 text-sm">
                {logs.map((log, index) => (
                  <div key={index} className="flex items-start gap-2 text-gray-600">
                    {getLogIcon(log.type)}
                    <div className="flex-1 leading-relaxed">
                      {log.message && (
                        <span className={log.type === 'error' ? 'text-red-600' : ''}>
                          {log.message}
                        </span>
                      )}
                      {log.type === 'results' && log.data && (
                        <div className="mt-2 text-xs text-gray-500 bg-gray-50 px-3 py-2 rounded border">
                          Found {log.data.count} results for "{log.data.query}"
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Export hook for easier usage
export function useStreamingLogs(onLogReceived?: (log: StreamingLogMessage) => void) {
  const [logs, setLogs] = useState<StreamingLogMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const addLog = (log: StreamingLogMessage) => {
    setLogs(prev => [...prev, log]);
    onLogReceived?.(log);
  };

  const startStreaming = async (query: string, onComplete?: (results: any) => void, onError?: (error: string) => void) => {
    setLogs([]);
    setIsStreaming(true);

    try {
      const response = await fetch('/api/search/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');

        // Keep the last potentially incomplete line in buffer
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data: StreamingLogMessage = JSON.parse(line.slice(6));
              console.log('🎯 Real-time message received:', data);

              // Use React's flushSync to force immediate re-render for true real-time updates
              flushSync(() => {
                addLog(data);
              });

              if (data.type === 'complete') {
                console.log('✅ Search completed');
                setIsStreaming(false);
              } else if (data.type === 'error') {
                console.log('❌ Search error:', data.message);
                setIsStreaming(false);
                onError?.(data.message || 'Unknown error');
              } else if (data.type === 'results') {
                console.log('📊 Results received, count:', data.data?.count);
                setIsStreaming(false);
                onComplete?.(data.data);
              }
            } catch (error) {
              console.error('❌ Error parsing streaming data:', error, 'Line:', line);
            }
          }
        }

        // Small delay to prevent overwhelming the UI with too many rapid updates
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setIsStreaming(false);
      onError?.(error instanceof Error ? error.message : 'Connection error');
    }
  };

  const clearLogs = () => {
    setLogs([]);
    setIsStreaming(false);
  };

  return { logs, isStreaming, startStreaming, clearLogs };
}
