import { useState, useMemo, useCallback } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';

interface ReasoningSection {
  id: string;
  startIndex: number;
  endIndex: number;
  content: string;
}

interface ReasoningCollapseProps {
  text: string;
  isStreaming?: boolean;
}

export function ReasoningCollapse({ text, isStreaming = false }: ReasoningCollapseProps) {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set());

  // Parse the text to find all reasoning sections
  const { sections, processedText } = useMemo(() => {
    const sections: ReasoningSection[] = [];
    let processedText = text;
    let sectionId = 0;

    // Find all <think>...</think> blocks (case-insensitive to handle variations)
    // Also handle incomplete tags during streaming (opening tag without closing tag)
    // Note: The model produces <think> opening tag and </think> closing tag
    const reasoningRegex = /<think>([\s\S]*?)<\/think>/gi;
    let match;
    const matches: Array<{ start: number; end: number; content: string }> = [];

    // Collect all matches first
    while ((match = reasoningRegex.exec(text)) !== null) {
      matches.push({
        start: match.index,
        end: match.index + match[0].length,
        content: match[1], // Content inside the tags
      });
    }

    // Handle incomplete reasoning tags during streaming (opening tag without closing tag)
    // Only if we're streaming and there's an unclosed tag
    if (isStreaming) {
      const openTagRegex = /<think>/gi;
      const openMatches = Array.from(text.matchAll(openTagRegex));
      
      // Find open tags without matching close tags
      for (const openMatch of openMatches) {
        const openIndex = openMatch.index!;
        const hasMatchingClose = matches.some(m => m.start === openIndex);
        
        if (!hasMatchingClose) {
          // For incomplete tags, use everything from the opening tag to the end of text
          const endIndex = text.length;
          const content = text.slice(openIndex + '<think>'.length);
          
          // Only add if it's not already in matches
          if (!matches.some(m => m.start === openIndex)) {
            matches.push({
              start: openIndex,
              end: endIndex,
              content: content,
            });
          }
        }
      }
    }

    // Process matches in reverse order to maintain indices
    for (let i = matches.length - 1; i >= 0; i--) {
      const { start, end, content } = matches[i];
      const id = `reasoning-${sectionId++}`;
      
      sections.unshift({
        id,
        startIndex: start,
        endIndex: end,
        content: content.trim(),
      });

      // Replace the reasoning block with a placeholder
      const placeholder = `__REASONING_PLACEHOLDER_${id}__`;
      processedText = processedText.slice(0, start) + placeholder + processedText.slice(end);
    }

    return { sections, processedText };
  }, [text, isStreaming]);

  const toggleSection = useCallback((sectionId: string) => {
    setExpandedSections((prev) => {
      const next = new Set(prev);
      if (next.has(sectionId)) {
        next.delete(sectionId);
      } else {
        next.add(sectionId);
      }
      return next;
    });
  }, []);

  // Render the processed text with reasoning sections replaced by collapsible components
  const renderContent = useMemo(() => {
    // If no sections found, return original text
    if (sections.length === 0) {
      return <span>{text}</span>;
    }

    let result: React.ReactNode[] = [];
    let currentIndex = 0;
    let placeholderRegex = /__REASONING_PLACEHOLDER_(reasoning-\d+)__/g;
    let match;
    const processedTextCopy = processedText; // Use the processed text from useMemo

    while ((match = placeholderRegex.exec(processedTextCopy)) !== null) {
      // Add text before the placeholder
      if (match.index > currentIndex) {
        const beforeText = processedTextCopy.slice(currentIndex, match.index);
        if (beforeText) {
          result.push(<span key={`text-${currentIndex}`}>{beforeText}</span>);
        }
      }

      // Add the reasoning section component
      const sectionId = match[1];
      const section = sections.find((s) => s.id === sectionId);
      if (section) {
        const isExpanded = expandedSections.has(sectionId);
        result.push(
          <ReasoningSectionComponent
            key={sectionId}
            section={section}
            isExpanded={isExpanded}
            onToggle={() => toggleSection(sectionId)}
          />
        );
      }

      currentIndex = match.index + match[0].length;
    }

    // Add remaining text after last placeholder
    if (currentIndex < processedTextCopy.length) {
      const remainingText = processedTextCopy.slice(currentIndex);
      if (remainingText) {
        result.push(<span key={`text-${currentIndex}`}>{remainingText}</span>);
      }
    }

    return result.length > 0 ? result : <span>{text}</span>;
  }, [sections, processedText, expandedSections, text, toggleSection]);

  return (
    <div className="reasoning-collapse-container">
      {renderContent}
      {isStreaming && (
        <span className="inline-block w-2 h-4 bg-accent-500 ml-1 animate-pulse" />
      )}
    </div>
  );
}

interface ReasoningSectionComponentProps {
  section: ReasoningSection;
  isExpanded: boolean;
  onToggle: () => void;
}

function ReasoningSectionComponent({
  section,
  isExpanded,
  onToggle,
}: ReasoningSectionComponentProps) {
  const contentLines = section.content.split('\n').length;
  const previewLength = 100;
  const preview = section.content.length > previewLength
    ? section.content.slice(0, previewLength) + '...'
    : section.content;

  return (
    <div className="my-2 border border-primary-200 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-3 py-2 bg-primary-50 hover:bg-primary-100 transition-colors text-left"
        type="button"
      >
        <div className="flex items-center gap-2">
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-primary-600" />
          ) : (
            <ChevronDown className="w-4 h-4 text-primary-600" />
          )}
          <span className="text-xs font-medium text-primary-700">
            Reasoning {isExpanded ? '(expanded)' : '(collapsed)'}
          </span>
          {!isExpanded && (
            <span className="text-xs text-primary-500">
              {contentLines} line{contentLines !== 1 ? 's' : ''}
            </span>
          )}
        </div>
      </button>
      {isExpanded && (
        <div className="px-3 py-2 bg-white border-t border-primary-200">
          <pre className="text-xs text-primary-700 whitespace-pre-wrap font-mono max-h-96 overflow-y-auto">
            {section.content}
          </pre>
        </div>
      )}
      {!isExpanded && (
        <div className="px-3 py-1.5 bg-white border-t border-primary-200">
          <pre className="text-xs text-primary-500 whitespace-pre-wrap font-mono line-clamp-2">
            {preview}
          </pre>
        </div>
      )}
    </div>
  );
}
