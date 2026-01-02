import { useState } from 'react';
import { ChevronDown, ChevronRight, Brain } from 'lucide-react';
import { Button } from './Button';

interface CollapsibleThinkProps {
  children: string;
  className?: string;
}

export function CollapsibleThink({ children, className = '' }: CollapsibleThinkProps) {
  // Parse the content to separate think blocks from regular content
  const parseContent = (content: string) => {
    // Match <think> tags (allowing for potential whitespace)
    const redactedRegex = /<think>\s*(.*?)\s*<\/redacted_reasoning>/gs;
    const parts: Array<{ type: 'text' | 'think'; content: string; start: number; end: number }> = [];
    
    // Find all redacted_reasoning blocks
    let match;
    while ((match = redactedRegex.exec(content)) !== null) {
      parts.push({
        type: 'think',
        content: match[1].trim(),
        start: match.index,
        end: match.index + match[0].length
      });
    }
    
    // Sort by position
    parts.sort((a, b) => a.start - b.start);
    
    // Build the final parts array with text and think blocks
    const result: Array<{ type: 'text' | 'think'; content: string }> = [];
    let lastIndex = 0;
    
    for (const part of parts) {
      // Add text before this think block
      if (part.start > lastIndex) {
        result.push({
          type: 'text',
          content: content.slice(lastIndex, part.start)
        });
      }
      
      // Add the think block
      result.push({
        type: 'think',
        content: part.content
      });
      
      lastIndex = part.end;
    }
    
    // Add remaining text after the last think block
    if (lastIndex < content.length) {
      result.push({
        type: 'text',
        content: content.slice(lastIndex)
      });
    }
    
    return result;
  };

  const parsedContent = parseContent(children);

  if (parsedContent.length === 0 || !parsedContent.some(part => part.type === 'think')) {
    // No complete think blocks found, render as normal text
    return <span className={className}>{children}</span>;
  }

  return (
    <div className={className}>
      {parsedContent.map((part, index) => {
        if (part.type === 'text') {
          return <span key={index}>{part.content}</span>;
        }

        // Each think block has its own expand/collapse state
        return <ThinkBlock key={index} content={part.content} />;
      })}
    </div>
  );
}

// Separate component for each think block with its own state
function ThinkBlock({ content }: { content: string }) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="my-3 border-l-4 border-blue-300 bg-blue-50/50 rounded-r-lg">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-blue-200/50">
        <Brain className="w-4 h-4 text-blue-600" />
        <span className="text-sm font-medium text-blue-800">Thinking Process</span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="ml-auto h-6 px-2 text-blue-700 hover:text-blue-900 hover:bg-blue-100"
        >
          {isExpanded ? (
            <>
              <ChevronDown className="w-3 h-3 mr-1" />
              Collapse
            </>
          ) : (
            <>
              <ChevronRight className="w-3 h-3 mr-1" />
              Expand
            </>
          )}
        </Button>
      </div>
      {isExpanded && (
        <div className="px-3 py-2 text-sm text-blue-900 leading-relaxed whitespace-pre-wrap">
          {content}
        </div>
      )}
    </div>
  );
}
