import { Document } from '@/types';
import { CoolTooltip } from '@/components/ui/CoolTooltip';
import { formatFileSize } from '@/lib/api';
import { motion } from 'framer-motion';

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

            {document.deepseek_ocr_used && (
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 border border-blue-200">
                  OCR: DeepSeek
                </span>
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
