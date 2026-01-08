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
  sub_categories?: string[];
  content: string;
  content_preview: string;
  classification_date: string;
  metadata: DocumentMetadata;
  embedding?: number[];
  ocr_used?: boolean;
  summary?: string;
}

export interface DocumentMetadata {
  file_extension: string;
  file_size: number;
  mime_type?: string;
  page_count?: number;
  creation_date?: string;
  modification_date?: string;
  performance_metrics?: PerformanceMetrics;
}

export interface PerformanceMetrics {
  hash_duration: number;
  ocr_duration: number;
  classification_duration: number;
  embedding_duration: number;
  db_lookup_duration: number;
  db_insert_duration: number;
  total_processing_time: number;
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
  sub_categories: Array<[string, number]>;
  file_types: Array<[string, number]>;
}

export interface DocumentDuration {
  id: number;
  index: number;
  filename: string;
  total_duration: number;
  hash_duration: number;
  ocr_duration: number;
  classification_duration: number;
  embedding_duration: number;
  db_lookup_duration: number;
  db_insert_duration: number;
}

export interface DocumentDurationsResponse {
  documents: DocumentDuration[];
  count: number;
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
  sub_category?: string;
  sort?: string;
  direction?: 'asc' | 'desc';
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

// Database types
export interface ColumnInfo {
  name: string;
  type: string;
  nullable: boolean;
  primaryKey: boolean;
}

export interface TableInfo {
  name: string;
  rowCount: number;
  columns: ColumnInfo[];
}

export interface TableData {
  columns: string[];
  rows: any[][];
  totalRows: number;
  page: number;
  totalPages: number;
  limit: number;
}

export interface DatabaseTablesResponse {
  tables: TableInfo[];
}

export interface DatabaseTableDataResponse extends TableData {}

// CRUD operation types
export interface CreateRecordRequest {
  [key: string]: any;
}

export interface UpdateRecordRequest {
  [key: string]: any;
}

export interface CrudResponse {
  success: boolean;
  id?: number;
  message: string;
  error?: string;
}

// Embedding types
export interface EmbeddingPoint {
  id: number;
  x: number;
  y: number;
  z: number;
  filename: string;
  categories: string;
  sub_categories?: string[];
  metadata: DocumentMetadata;
  // Additional PCA components beyond x,y,z
  pc4?: number;
  pc5?: number;
  [key: `pc${number}`]: number | undefined;
}

export interface EmbeddingResponse {
  points: EmbeddingPoint[];
  count: number;
  components?: number;
  explained_variance?: number[];
  error?: string;
  // Raw embeddings response
  raw?: boolean;
  embeddings?: any[];
}

// Embedding Search types
export interface EmbeddingSearchResult {
  id: string | number;
  similarity: number;
  filename: string;
  categories: string;
  sub_categories: string[];
  content_preview: string;
  metadata: Record<string, any>;
}

export interface EmbeddingInfo {
  dimension: number;
  model: string;
  sample_values: number[];
  min_value: number;
  max_value: number;
  mean_value: number;
}

export interface EmbeddingSearchResponse {
  keyword: string;
  words: string[];
  embedding_info: EmbeddingInfo;
  results: EmbeddingSearchResult[];
  count: number;
  error?: string;
}

// Answer types
export interface AnswerCitation {
  id: number;
  filename: string;
  categories: string;
  similarity?: number;
  content_preview?: string;
}

export interface AnswerResponse {
  answer: string;
  citations: AnswerCitation[];
}

export interface AnswerStreamEvent {
  type: 'start' | 'log' | 'llm_chunk' | 'answer_chunk' | 'citations' | 'complete' | 'error';
  message?: string;
  chunk?: string;
  answer?: string;
  citations?: AnswerCitation[];
}
