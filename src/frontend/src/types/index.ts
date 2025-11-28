// API Response types
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  message?: string;
}

// Document types
export interface Document {
  id: number;
  doc_id?: number;
  filename: string;
  file_path: string;
  categories: string;
  content_preview: string;
  classification_date: string;
  metadata: DocumentMetadata;
  embedding?: number[];
}

export interface DocumentMetadata {
  file_extension: string;
  file_size: number;
  mime_type?: string;
  page_count?: number;
  creation_date?: string;
  modification_date?: string;
}

// Search types
export interface SearchResult extends Document {
  similarity?: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  count: number;
  search_type?: 'semantic' | 'category';
}

// Documents listing types
export interface DocumentsResponse {
  documents: Document[];
  page: number;
  total_pages: number;
  total_documents: number;
  limit: number;
}

// Statistics types
export interface DocumentStats {
  total_docs: number;
  categories: Array<[string, number]>;
  file_types: Array<[string, number]>;
}

// API request types
export interface SearchRequest {
  query: string;
  max_candidates?: number;
}

export interface DocumentsRequest {
  page?: number;
  limit?: number;
  search?: string;
  category?: string;
}

// Theme types
export type Theme = 'light' | 'dark' | 'auto';

// UI state types
export interface LoadingState {
  isLoading: boolean;
  error?: string;
}

export interface PaginationState {
  page: number;
  totalPages: number;
  totalItems: number;
  itemsPerPage: number;
}

// File viewer types
export interface FileViewerProps {
  document: Document;
  onClose?: () => void;
}

// Navigation types
export interface NavItem {
  label: string;
  href: string;
  icon?: React.ComponentType<{ className?: string }>;
  badge?: string | number;
}
