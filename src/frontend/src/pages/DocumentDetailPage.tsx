import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Download, Eye, FileText, Calendar, Tag, HardDrive } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Document } from '@/types';
import { apiClient, getFileIcon, formatFileSize, downloadFile } from '@/lib/api';
import toast from 'react-hot-toast';

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [document, setDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);

  // Function to filter out LLM encoding tokens
  const filterLLMEncoding = (content: string): string => {
    // Remove tokens like <|ref|>title<|/ref|> and <|det|>[[310, 110, 756, 127]]<|/det|>
    // Pattern: <|tag|>content<|/tag|>
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

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!document) {
    return (
      <div className="text-center py-12">
        <FileText className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-primary-900 mb-2">Document not found</h2>
        <p className="text-primary-600 mb-4">The document you're looking for doesn't exist.</p>
        <Button onClick={() => navigate('/documents')}>Back to Documents</Button>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <Button
          variant="ghost"
          onClick={() => navigate('/documents')}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Documents
        </Button>

        <div className="flex gap-2">
          <Button variant="outline" onClick={handleViewOriginal}>
            <Eye className="w-4 h-4 mr-2" />
            View Original
          </Button>
          <Button onClick={handleDownload} disabled={downloading}>
            <Download className="w-4 h-4 mr-2" />
            {downloading ? 'Downloading...' : 'Download'}
          </Button>
        </div>
      </div>

      {/* Document Header */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-start gap-4">
            <div className="text-5xl flex-shrink-0">
              {getFileIcon(document.metadata.file_extension)}
            </div>
            <div className="flex-1 min-w-0">
              <h1 className="text-2xl font-bold text-primary-900 mb-2">
                {document.filename}
              </h1>

              <div className="flex flex-wrap items-center gap-4 text-sm text-primary-600 mb-4">
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  {formatDate(document.classification_date)}
                </div>
                <div className="flex items-center gap-1">
                  <HardDrive className="w-4 h-4" />
                  {formatFileSize(document.metadata.file_size)}
                </div>
                <div className="flex items-center gap-1">
                  <FileText className="w-4 h-4" />
                  {document.metadata.file_extension.toUpperCase()}
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {document.categories.split('-').map((category, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-800"
                  >
                    <Tag className="w-3 h-3" />
                    {category.trim()}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Full Content */}
      <Card>
        <CardHeader>
          <CardTitle>Content</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="prose prose-gray max-w-none">
            <div className="markdown-content whitespace-pre-wrap text-primary-700 leading-relaxed max-h-96 overflow-y-auto border rounded p-4 bg-gray-50">
              {filterLLMEncoding(document.content)}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metadata */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>File Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-primary-600">Filename:</span>
              <span className="font-medium">{document.filename}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-primary-600">File Extension:</span>
              <span className="font-medium">{document.metadata.file_extension}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-primary-600">File Size:</span>
              <span className="font-medium">{formatFileSize(document.metadata.file_size)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-primary-600">MIME Type:</span>
              <span className="font-medium">{document.metadata.mime_type || 'Unknown'}</span>
            </div>
            {document.metadata.page_count && (
              <div className="flex justify-between">
                <span className="text-primary-600">Pages:</span>
                <span className="font-medium">{document.metadata.page_count}</span>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Classification Details</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex justify-between">
              <span className="text-primary-600">Document ID:</span>
              <span className="font-medium">{document.id}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-primary-600">Classification Date:</span>
              <span className="font-medium">{formatDate(document.classification_date)}</span>
            </div>
            <div>
              <span className="text-primary-600 block mb-2">Categories:</span>
              <div className="flex flex-wrap gap-1">
                {document.categories.split('-').map((category, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                  >
                    {category.trim()}
                  </span>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* File Path */}
      <Card>
        <CardHeader>
          <CardTitle>File Location</CardTitle>
        </CardHeader>
        <CardContent>
          <code className="text-sm bg-primary-100 px-3 py-2 rounded block font-mono break-all text-primary-800">
            {document.file_path}
          </code>
        </CardContent>
      </Card>
    </div>
  );
}
