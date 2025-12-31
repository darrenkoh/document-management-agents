import { Document } from '@/types';
import { CoolTooltip } from '@/components/ui/CoolTooltip';
import { formatFileSize } from '@/lib/api';
import { motion } from 'framer-motion';

// Helper function to format duration in milliseconds
const formatDuration = (seconds: number): string => {
  if (seconds === 0) return "0ms";
  const ms = seconds * 1000;
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
};

interface FileInfoBarProps {
  document: Document;
}

export const FileInfoBar = ({ document }: FileInfoBarProps) => {


  return (
    <motion.div 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="w-full bg-gray-50/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-20 shadow-sm"
    >
      <div className="max-w-7xl mx-auto px-6 py-3">
        <div className="flex items-center gap-6">

          {/* Left: Compact File Info */}
          <div className="flex items-center gap-4 flex-shrink-0">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Type</span>
              <span className="px-2 py-1 bg-gray-100 text-gray-800 rounded text-xs font-semibold border">
                {document.metadata.file_extension.toUpperCase()}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Size</span>
              <span className="text-xs font-semibold text-gray-800">
                {formatFileSize(document.metadata.file_size)}
              </span>
            </div>

            {!!document.ocr_used && (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200">
                  OCR
                </span>
              </div>
            )}

            {/* Performance Metrics */}
            {document.metadata.performance_metrics && (
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Total Time</span>
                <CoolTooltip content={`Processing breakdown:
• Hash: ${formatDuration(document.metadata.performance_metrics.hash_duration)}
• OCR: ${formatDuration(document.metadata.performance_metrics.ocr_duration)}
• Classification: ${formatDuration(document.metadata.performance_metrics.classification_duration)}
• Embeddings: ${formatDuration(document.metadata.performance_metrics.embedding_duration)}
• DB Lookup: ${formatDuration(document.metadata.performance_metrics.db_lookup_duration)}
• DB Insert: ${formatDuration(document.metadata.performance_metrics.db_insert_duration)}`}>
                  <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-semibold border border-green-200">
                    {formatDuration(document.metadata.performance_metrics.total_processing_time)}
                  </span>
                </CoolTooltip>
              </div>
            )}
          </div>

          {/* Categories */}
          <div className="flex items-center gap-2 flex-wrap">
            {document.categories.split('-').slice(0, 4).map((category, index) => (
              <CoolTooltip key={`main-${index}`} content={`Main Category: ${category.trim()}`}>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 text-primary-700 border border-primary-200/50">
                  {category.trim()}
                </span>
              </CoolTooltip>
            ))}
            {document.categories.split('-').length > 4 && (
              <span className="text-xs font-medium text-gray-500">
                +{document.categories.split('-').length - 4} main
              </span>
            )}
            {document.sub_categories && document.sub_categories.slice(0, 4).map((subCategory, index) => (
              <CoolTooltip key={`sub-${index}`} content={`Sub-category: ${subCategory}`}>
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200">
                  {subCategory}
                </span>
              </CoolTooltip>
            ))}
            {document.sub_categories && document.sub_categories.length > 4 && (
              <span className="text-xs font-medium text-gray-500">
                +{document.sub_categories.length - 4} sub
              </span>
            )}
          </div>

          {/* File Path - Takes remaining space */}
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wider hidden sm:inline flex-shrink-0">Path</span>
            <CoolTooltip content={`Full path: ${document.file_path}`}>
              <div className="px-3 py-1 bg-gray-50 border border-gray-200 rounded text-xs font-mono text-gray-700 truncate flex-1 min-w-0">
                {document.file_path}
              </div>
            </CoolTooltip>
          </div>

        </div>
      </div>
    </motion.div>
  );
};
