import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Eye, Trash2, AlertTriangle, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Document } from '@/types';
import { apiClient, downloadFile } from '@/lib/api';
import toast from 'react-hot-toast';
import { motion } from 'framer-motion';

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
                <div className="prose prose-gray max-w-none prose-headings:text-gray-900 prose-p:text-gray-700 prose-a:text-primary-600 prose-strong:text-gray-900">
                  <style dangerouslySetInnerHTML={{
                    __html: `
                      table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
                      th, td { border: 1px solid #d1d5db; padding: 0.5rem 0.75rem; text-align: left; }
                      th { background-color: #f9fafb; font-weight: 600; }
                      tr:nth-child(even) { background-color: #f9fafb; }
                    `
                  }} />
                  <div
                    className="leading-relaxed font-normal text-base font-sans [&_table]:border-collapse [&_table]:w-full [&_table]:my-4 [&_th]:border [&_th]:border-gray-300 [&_th]:px-3 [&_th]:py-2 [&_th]:bg-gray-50 [&_th]:font-semibold [&_td]:border [&_td]:border-gray-300 [&_td]:px-3 [&_td]:py-2"
                    dangerouslySetInnerHTML={{ __html: filterLLMEncoding(document.content) }}
                  />
                </div>
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
                  <div className="prose prose-gray max-w-none prose-p:text-gray-700 prose-strong:text-gray-900">
                    <p className="text-base leading-relaxed font-normal text-gray-700 whitespace-pre-wrap">
                      {filterLLMEncoding(document.summary)}
                    </p>
                  </div>
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
