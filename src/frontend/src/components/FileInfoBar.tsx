import { Document } from '@/types';
import { CoolTooltip } from '@/components/ui/CoolTooltip';
import { formatFileSize } from '@/lib/api';
import { Calendar, HardDrive, FileText, Layers, Hash, ScanText, Folder } from 'lucide-react';
import { motion } from 'framer-motion';

interface FileInfoBarProps {
  document: Document;
}

export const FileInfoBar = ({ document }: FileInfoBarProps) => {
  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  const InfoItem = ({ icon: Icon, label, value, detail }: { icon: any, label: string, value: string | number, detail?: string }) => (
    <CoolTooltip content={detail || `${label}: ${value}`}>
      <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-white/50 border border-gray-200 hover:bg-white hover:shadow-md hover:border-primary-200 transition-all duration-200 cursor-default group w-full">
        <div className="p-2 rounded-lg bg-primary-50 text-primary-500 group-hover:bg-primary-100 transition-colors">
           <Icon className="w-4 h-4" />
        </div>
        <div className="flex flex-col min-w-0">
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</span>
          <span className="text-sm font-semibold text-gray-800 truncate">{value}</span>
        </div>
      </div>
    </CoolTooltip>
  );

  return (
    <motion.div 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="w-full bg-gray-50/80 backdrop-blur-md border-b border-gray-200 sticky top-0 z-20 shadow-sm"
    >
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          
          {/* Left Column: File Details */}
          <div className="flex flex-col gap-4">
            {/* Main file details grid */}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <InfoItem
                icon={FileText}
                label="Type"
                value={document.metadata.file_extension.toUpperCase()}
                detail={`MIME Type: ${document.metadata.mime_type || 'Unknown'}`}
              />

              <InfoItem
                icon={HardDrive}
                label="Size"
                value={formatFileSize(document.metadata.file_size)}
              />

              {document.metadata.page_count && (
                <InfoItem
                  icon={Layers}
                  label="Pages"
                  value={document.metadata.page_count}
                  detail={`${document.metadata.page_count} Pages`}
                />
              )}

              {document.deepseek_ocr_used && (
                <InfoItem
                  icon={ScanText}
                  label="OCR"
                  value="DeepSeek"
                  detail="Document was processed using DeepSeek-OCR for text extraction"
                />
              )}
            </div>

            {/* File path in its own row */}
            <div className="w-full">
              <CoolTooltip content={`Full path: ${document.file_path}`}>
                <div className="flex items-center gap-3 px-4 py-3 rounded-xl bg-white/50 border border-gray-200 hover:bg-white hover:shadow-md hover:border-primary-200 transition-all duration-200 cursor-default group w-full">
                  <div className="p-2 rounded-lg bg-primary-50 text-primary-500 group-hover:bg-primary-100 transition-colors flex-shrink-0">
                    <Folder className="w-4 h-4" />
                  </div>
                  <div className="flex flex-col min-w-0 flex-1">
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">Path</span>
                    <span className="text-sm font-semibold text-gray-800 break-all">{document.file_path}</span>
                  </div>
                </div>
              </CoolTooltip>
            </div>
          </div>

          {/* Right Column: Classification & ID */}
          <div className="flex flex-col justify-center gap-3">
             <div className="flex items-center gap-3 w-full">
                <div className="flex-1">
                  <InfoItem 
                    icon={Calendar} 
                    label="Classified" 
                    value={formatDate(document.classification_date)}
                    detail={`Classified on ${formatDate(document.classification_date)}`}
                  />
                </div>
                 <div className="flex-1">
                   <div className="flex items-center gap-3 px-4 py-2 rounded-xl bg-white/50 border border-gray-200 hover:bg-white hover:shadow-md hover:border-primary-200 transition-all duration-200 cursor-default group w-full h-full">
                      <div className="p-2 rounded-lg bg-primary-50 text-primary-500 group-hover:bg-primary-100 transition-colors">
                         <Hash className="w-4 h-4" />
                      </div>
                      <div className="flex flex-col min-w-0 flex-1">
                        <span className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Categories</span>
                         <div className="flex flex-wrap gap-1">
                          {document.categories.split('-').slice(0, 2).map((category, index) => (
                            <CoolTooltip key={index} content={`Category: ${category.trim()}`}>
                              <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-primary-100 text-primary-700 border border-primary-200/50">
                                {category.trim()}
                              </span>
                            </CoolTooltip>
                          ))}
                           {document.categories.split('-').length > 2 && (
                              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-gray-100 text-gray-600 border border-gray-200">
                                +{document.categories.split('-').length - 2}
                              </span>
                           )}
                        </div>
                      </div>
                   </div>
                 </div>
             </div>
          </div>
          
        </div>
      </div>
    </motion.div>
  );
};
