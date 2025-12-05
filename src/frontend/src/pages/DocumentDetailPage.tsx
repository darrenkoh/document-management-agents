import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Eye, Trash2, AlertTriangle, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Document } from '@/types';
import { apiClient, downloadFile } from '@/lib/api';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

// New Components
import { FileInfoBar } from '@/components/FileInfoBar';
import { UnifiedFileViewer } from '@/components/UnifiedFileViewer';
import { CoolTooltip } from '@/components/ui/CoolTooltip';

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [copied, setCopied] = useState(false);

  // Function to filter out LLM encoding tokens
  const filterLLMEncoding = (content: string): string => {
    const tokenPattern = /<\|[^>]+\|>.*?<\|\/[^>]+\|>/g;
    return content.replace(tokenPattern, '').trim();
  };

  // Function to detect content type
  const detectContentType = (content: string): 'markdown' | 'html' | 'mixed' | 'plain' => {
    const cleanContent = filterLLMEncoding(content);

    const hasHtmlTables = cleanContent.includes('<table') || cleanContent.includes('<tr') || cleanContent.includes('<td') || cleanContent.includes('<th');

    // Check if content contains markdown table syntax (pipes)
    const hasMarkdownTables = cleanContent.includes('|') && cleanContent.includes('\n|') && (() => {
      const lines = cleanContent.split('\n');
      return lines.some(line => {
        const pipeCount = (line.match(/\|/g) || []).length;
        return pipeCount >= 2 && line.trim().startsWith('|') && line.trim().endsWith('|');
      });
    })();

    // If content has both HTML tables and markdown tables, it's mixed
    if (hasHtmlTables && hasMarkdownTables) {
      return 'mixed';
    }

    // Check for HTML tables first (since they might contain pipes that would confuse markdown detection)
    if (hasHtmlTables) {
      return 'html';
    }

    // Check if content contains markdown table syntax (pipes)
    if (hasMarkdownTables) {
      return 'markdown';
    }

    return 'plain';
  };

  // Function to clean and improve HTML table content from OCR
  const cleanHtmlTables = (htmlContent: string): string => {
    // Fix common OCR HTML table issues
    let cleaned = htmlContent;

    // Ensure tables have proper structure
    // Fix missing closing tags
    cleaned = cleaned.replace(/<table([^>]*)>((?!<\/table>).)*$/gi, '$&</table>');
    cleaned = cleaned.replace(/<tr([^>]*)>((?!<\/tr>).)*$/gi, '$&</tr>');
    cleaned = cleaned.replace(/<td([^>]*)>((?!<\/td>).)*$/gi, '$&</td>');
    cleaned = cleaned.replace(/<th([^>]*)>((?!<\/th>).)*$/gi, '$&</th>');

    // Add missing tbody tags for better structure
    cleaned = cleaned.replace(/(<table[^>]*>)\s*(<tr[^>]*>)/gi, '$1<tbody>$2');
    cleaned = cleaned.replace(/(<\/tr>\s*)(<\/table>)/gi, '$1</tbody>$2');

    // Fix spacing issues in table cells
    cleaned = cleaned.replace(/<td[^>]*>\s*<br\s*\/?>\s*/gi, '<td>');
    cleaned = cleaned.replace(/\s*<br\s*\/?>\s*<\/td>/gi, '</td>');

    return cleaned;
  };

  // Function to process mixed content (HTML tables + markdown tables)
  const processMixedContent = (content: string): string => {
    // First clean HTML tables
    let processed = cleanHtmlTables(content);

    // Then convert any markdown table syntax within the content to HTML tables
    // This handles cases where markdown tables appear within HTML content
    processed = processed.replace(/\|.*\|\n\|.*\|\n\|.*\|/g, (match) => {
      // Convert markdown table to HTML table
      const lines = match.split('\n');
      if (lines.length >= 3) {
        const headerLine = lines[0];
        const separatorLine = lines[1];
        const dataLines = lines.slice(2);

        // Parse header
        const headers = headerLine.split('|').filter(cell => cell.trim());
        // Parse data rows
        const rows = dataLines.map(line => line.split('|').filter(cell => cell.trim()));

        let htmlTable = '<table style="border-collapse: collapse; width: 100%; margin: 1rem 0; border: 1px solid #d1d5db;">';
        htmlTable += '<thead><tr>';
        headers.forEach(header => {
          htmlTable += `<th style="border: 1px solid #d1d5db; padding: 0.5rem; background-color: #f9fafb; font-weight: 600;">${header.trim()}</th>`;
        });
        htmlTable += '</tr></thead><tbody>';
        rows.forEach(row => {
          htmlTable += '<tr>';
          row.forEach(cell => {
            htmlTable += `<td style="border: 1px solid #d1d5db; padding: 0.5rem;">${cell.trim()}</td>`;
          });
          htmlTable += '</tr>';
        });
        htmlTable += '</tbody></table>';

        return htmlTable;
      }
      return match;
    });

    return processed;
  };

  // Function to render content based on type
  const renderContent = (content: string) => {
    const cleanContent = filterLLMEncoding(content);
    const contentType = detectContentType(content);

    switch (contentType) {
      case 'markdown':
        return (
          <div className="markdown-content prose prose-gray max-w-none">
            <ReactMarkdown
              components={{
                table: ({ children }) => (
                  <table className="w-full border-collapse border border-gray-300 my-4 rounded-lg overflow-hidden">
                    {children}
                  </table>
                ),
                th: ({ children }) => (
                  <th className="border border-gray-300 px-4 py-2 bg-gray-50 font-semibold text-gray-900 text-left">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="border border-gray-300 px-4 py-2 text-left">
                    {children}
                  </td>
                ),
                tr: ({ children }) => (
                  <tr className="hover:bg-gray-50">
                    {children}
                  </tr>
                ),
              }}
            >
              {cleanContent}
            </ReactMarkdown>
          </div>
        );

      case 'html':
        return (
          <div className="prose prose-gray max-w-none">
            {/* Enhanced HTML table styling with better spacing and visual hierarchy */}
            <style dangerouslySetInnerHTML={{
              __html: `
                .html-content table {
                  border-collapse: collapse;
                  width: 100%;
                  margin: 1rem 0;
                  border-radius: 0.5rem;
                  overflow: hidden;
                  box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
                }
                .html-content th,
                .html-content td {
                  border: 1px solid #d1d5db;
                  padding: 0.75rem 1rem;
                  text-align: left;
                  vertical-align: top;
                }
                .html-content th {
                  background-color: #f9fafb;
                  font-weight: 600;
                  color: #111827;
                }
                .html-content tr:nth-child(even) {
                  background-color: #f9fafb;
                }
                .html-content tr:hover {
                  background-color: #f3f4f6;
                }
                .html-content p {
                  margin: 0.5rem 0;
                  line-height: 1.6;
                }
                .html-content br {
                  margin: 0.25rem 0;
                }
              `
            }} />
            <div
              className="html-content leading-relaxed font-normal text-base font-sans text-gray-700"
              dangerouslySetInnerHTML={{ __html: cleanHtmlTables(cleanContent) }}
            />
          </div>
        );

      case 'mixed':
        return (
          <div className="prose prose-gray max-w-none">
            {/* Enhanced mixed content styling */}
            <style dangerouslySetInnerHTML={{
              __html: `
                .mixed-content table {
                  border-collapse: collapse;
                  width: 100%;
                  margin: 1rem 0;
                  border-radius: 0.5rem;
                  overflow: hidden;
                  box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
                }
                .mixed-content th,
                .mixed-content td {
                  border: 1px solid #d1d5db;
                  padding: 0.75rem 1rem;
                  text-align: left;
                  vertical-align: top;
                }
                .mixed-content th {
                  background-color: #f9fafb;
                  font-weight: 600;
                  color: #111827;
                }
                .mixed-content tr:nth-child(even) {
                  background-color: #f9fafb;
                }
                .mixed-content tr:hover {
                  background-color: #f3f4f6;
                }
                .mixed-content p {
                  margin: 0.5rem 0;
                  line-height: 1.6;
                }
                .mixed-content br {
                  margin: 0.25rem 0;
                }
              `
            }} />
            <div
              className="mixed-content leading-relaxed font-normal text-base font-sans text-gray-700"
              dangerouslySetInnerHTML={{ __html: processMixedContent(cleanContent) }}
            />
          </div>
        );

      default: // plain text
        return (
          <div className="leading-relaxed font-normal text-base font-sans prose prose-gray max-w-none whitespace-pre-wrap">
            {cleanContent}
          </div>
        );
    }
  };

  useEffect(() => {
    if (id) {
      loadDocument(parseInt(id));
    }
  }, [id]);

  const loadDocument = async (docId: number) => {
    setLoading(true);
    try {
      const doc = await apiClient.getDocument(docId);
      setDocument(doc);
    } catch (error) {
      console.error('Failed to load document:', error);
      toast.error('Failed to load document');
      navigate('/documents');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!document) return;

    setDownloading(true);
    try {
      await downloadFile(document);
      toast.success('Download started');
    } catch (error) {
      console.error('Download failed:', error);
      toast.error('Download failed');
    } finally {
      setDownloading(false);
    }
  };

  const handleViewOriginal = () => {
    if (!document) return;
    window.open(`/document/${document.id}/file`, '_blank');
  };

  const handleDelete = async () => {
    if (!document) return;

    setIsDeleting(true);
    try {
      const result = await apiClient.deleteDocuments([document.id]);
      if (result.success) {
        toast.success('Document deleted successfully');
        navigate('/documents');
      } else {
        toast.error('Failed to delete document');
        if (result.errors.length > 0) {
          result.errors.forEach(err => console.error(err));
        }
      }
    } catch (error) {
      console.error('Delete failed:', error);
      toast.error('Failed to delete document');
    } finally {
      setIsDeleting(false);
      setShowDeleteConfirm(false);
    }
  };

  const handleCopyContent = () => {
    if (!document) return;
    const cleanContent = filterLLMEncoding(document.content);
    navigator.clipboard.writeText(cleanContent);
    setCopied(true);
    toast.success('Content copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-[calc(100vh-100px)]">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!document) {
    return (
      <div className="text-center py-12">
        <h2 className="text-xl font-semibold text-primary-900 mb-2">Document not found</h2>
        <Button onClick={() => navigate('/documents')}>Back to Documents</Button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50/30 pb-12">
      {/* Top Bar with File Info */}
      <FileInfoBar document={document} />

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        {/* Header Actions */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={() => navigate('/documents')}
              className="flex items-center gap-2 hover:bg-white/80 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
            <h1 className="text-3xl font-bold text-gray-900 truncate max-w-2xl tracking-tight" title={document.filename}>
              {document.filename}
            </h1>
          </div>

          <div className="flex gap-3">
            <CoolTooltip content="View original file in new tab" side="bottom">
              <Button variant="outline" onClick={handleViewOriginal} className="bg-white/80 backdrop-blur-sm hover:shadow-sm transition-all">
                <Eye className="w-4 h-4 mr-2" />
                Open Original
              </Button>
            </CoolTooltip>

            <CoolTooltip content="Download original file" side="bottom">
              <Button onClick={handleDownload} disabled={downloading} className="bg-white/80 backdrop-blur-sm border border-primary-200 text-primary-700 hover:bg-white hover:shadow-sm transition-all">
                <Download className="w-4 h-4 mr-2" />
                {downloading ? 'Downloading...' : 'Download'}
              </Button>
            </CoolTooltip>

            <CoolTooltip content="Delete this document permanently" side="left">
              <Button
                variant="outline"
                onClick={() => setShowDeleteConfirm(true)}
                className="text-red-600 border-red-200 hover:bg-red-50 hover:border-red-300 bg-white/80 backdrop-blur-sm transition-all"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </Button>
            </CoolTooltip>
          </div>
        </motion.div>

        {/* Main Stacked Content */}
        <div className="flex flex-col gap-8">
          
          {/* 1. Extracted Content Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-4"
          >
            <div className="flex items-center justify-between px-1">
              <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                <span className="w-2 h-6 rounded-full bg-primary-500" />
                Extracted Content
              </h2>
               <div className="relative z-10">
                <CoolTooltip content="Copy text to clipboard" side="left">
                  <button
                    onClick={handleCopyContent}
                    className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 bg-white border border-gray-200 rounded-lg hover:text-primary-600 hover:border-primary-200 hover:bg-primary-50 transition-all shadow-sm"
                  >
                    {copied ? (
                      <>
                        <Check className="w-4 h-4 text-green-500" />
                        <span className="text-green-600">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-4 h-4" />
                        <span>Copy Text</span>
                      </>
                    )}
                  </button>
                </CoolTooltip>
              </div>
            </div>

            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden relative">
              <div className="max-h-[500px] overflow-y-auto p-8 scrollbar-thin scrollbar-thumb-gray-200 scrollbar-track-transparent hover:scrollbar-thumb-gray-300 transition-colors">
                {renderContent(document.content)}
              </div>
              {/* Bottom Fade for overflow indication */}
              <div className="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-white to-transparent pointer-events-none" />
            </div>
          </motion.section>

          {/* 2. Summary Section */}
          {document.summary && (
            <motion.section
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.3 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between px-1">
                <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                  <span className="w-2 h-6 rounded-full bg-blue-500" />
                  Summary
                </h2>
              </div>

              <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
                <div className="p-8">
                  {renderContent(document.summary)}
                </div>
              </div>
            </motion.section>
          )}

          {/* 3. Original File Preview Section */}
          <motion.section
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="space-y-4"
          >
             <div className="flex items-center justify-between px-1">
              <h2 className="text-xl font-semibold text-gray-800 flex items-center gap-2">
                <span className="w-2 h-6 rounded-full bg-green-500" />
                Original File Preview
              </h2>
            </div>
            
            <div className="h-[800px] rounded-2xl overflow-hidden border border-gray-200 shadow-lg bg-white">
               <UnifiedFileViewer document={document} className="h-full border-0 rounded-none shadow-none" />
            </div>
          </motion.section>

        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 backdrop-blur-sm">
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6 border border-gray-200"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-full bg-red-50 flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Delete Document</h3>
                <p className="text-sm text-gray-500">This action cannot be undone.</p>
              </div>
            </div>
            
            <div className="bg-gray-50 p-3 rounded-lg mb-6 border border-gray-100">
              <p className="text-sm font-medium text-gray-700 truncate">
                {document?.filename}
              </p>
            </div>

            <div className="flex justify-end gap-3">
              <Button
                variant="outline"
                onClick={() => setShowDeleteConfirm(false)}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                onClick={handleDelete}
                disabled={isDeleting}
                className="bg-red-600 hover:bg-red-700 text-white border-transparent shadow-md shadow-red-200"
              >
                {isDeleting ? (
                  <>
                    <LoadingSpinner size="sm" className="mr-2 text-white/90" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4 mr-2" />
                    Delete Permanently
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  );
}
