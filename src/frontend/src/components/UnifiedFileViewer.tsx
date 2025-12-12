import { useState, useEffect } from 'react';
import { Document } from '@/types';
import { apiClient } from '@/lib/api';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { AlertCircle, Download, ExternalLink, FileText } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { motion } from 'framer-motion';

interface UnifiedFileViewerProps {
  document: Document;
  className?: string;
}

export const UnifiedFileViewer = ({ document, className }: UnifiedFileViewerProps) => {
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Construct the URL directly since we have a proxy set up
    // But we can also fetch it as blob if we need to handle auth headers in the future
    // For now, direct URL is more efficient for browser caching and native handling
    setFileUrl(`/api/document/${document.id}/file/view`);
    setLoading(false);
  }, [document.id]);

  const getViewerContent = () => {
    if (loading) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-gray-400">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-sm">Loading preview...</p>
        </div>
      );
    }

    if (error || !fileUrl) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-red-500">
          <AlertCircle className="w-12 h-12 mb-4 opacity-50" />
          <p>Failed to load preview</p>
        </div>
      );
    }

    const extension = document.metadata.file_extension.toLowerCase();
    const mimeType = document.metadata.mime_type || '';

    // Image Viewer
    if (['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg'].includes(extension) || mimeType.startsWith('image/')) {
      return (
        <div className="flex items-center justify-center h-full overflow-auto p-4 bg-gray-100/50">
          <img 
            src={fileUrl} 
            alt={document.filename} 
            className="max-w-full max-h-full object-contain shadow-lg rounded-lg"
          />
        </div>
      );
    }

    // PDF and Text Viewer (Browser Native)
    if (extension === '.pdf' || extension === '.txt' || mimeType === 'application/pdf' || mimeType === 'text/plain') {
      return (
        <iframe
          src={fileUrl}
          className="w-full h-full border-0 bg-white"
          title="File Preview"
        />
      );
    }

    // Fallback for other types
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 bg-gray-50">
        <div className="w-20 h-20 bg-gray-200 rounded-2xl flex items-center justify-center mb-6">
          <FileText className="w-10 h-10 text-gray-400" />
        </div>
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Preview not available</h3>
        <p className="text-sm text-gray-500 mb-6 max-w-xs text-center">
          This file type ({extension}) cannot be previewed directly in the browser.
        </p>
        <div className="flex gap-3">
          <Button variant="outline" onClick={() => window.open(fileUrl, '_blank')}>
            <ExternalLink className="w-4 h-4 mr-2" />
            Open in New Tab
          </Button>
          <Button onClick={() => {
            const a = window.document.createElement('a');
            a.href = fileUrl;
            a.download = document.filename;
            a.click();
          }}>
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        </div>
      </div>
    );
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className={`bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden flex flex-col h-[calc(100vh-12rem)] ${className}`}
    >
      <div className="px-4 py-3 border-b border-gray-100 bg-gray-50/50 flex justify-between items-center">
        <span className="font-medium text-gray-700 text-sm flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          Original File
        </span>
        {fileUrl && (
          <a 
            href={fileUrl} 
            target="_blank" 
            rel="noreferrer"
            className="text-xs text-primary-600 hover:text-primary-700 hover:underline flex items-center gap-1"
          >
            Open raw <ExternalLink className="w-3 h-3" />
          </a>
        )}
      </div>
      <div className="flex-1 relative bg-gray-50">
        {getViewerContent()}
      </div>
    </motion.div>
  );
};

