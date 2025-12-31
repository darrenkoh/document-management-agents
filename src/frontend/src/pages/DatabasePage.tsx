import { useState, useEffect } from 'react';
import { Database, Table, Plus, Search, SortAsc, SortDesc, Edit, Trash2, FileSearch } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { apiClient } from '@/lib/api';
import { TableInfo, TableData, ColumnInfo } from '@/types';
import toast from 'react-hot-toast';

export default function DatabasePage() {
  const [tables, setTables] = useState<TableInfo[]>([]);
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [selectedTableInfo, setSelectedTableInfo] = useState<TableInfo | null>(null);
  const [tableData, setTableData] = useState<TableData | null>(null);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [sortColumn, setSortColumn] = useState<string | null>(null);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [filterInputs, setFilterInputs] = useState<Record<string, string>>({});

  // CRUD state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editingRecordId, setEditingRecordId] = useState<number | null>(null);
  const [formData, setFormData] = useState<Record<string, any>>({});

  // BM25 index state
  const [bm25Info, setBm25Info] = useState<{
    enabled: boolean;
    is_built: boolean;
    document_count: number;
    status: string;
    total_documents_in_db: number;
    index_coverage: string;
    coverage_percentage: number;
  } | null>(null);
  const [showBm25Info, setShowBm25Info] = useState(false);
  const [showBm25Details, setShowBm25Details] = useState(false);
  const [bm25Details, setBm25Details] = useState<{
    document_ids: number[];
    document_details: Array<{
      doc_id: number;
      filename: string;
      token_count: number;
      unique_token_count: number;
      sample_tokens: string[];
      content_preview: string;
    }>;
    total_documents: number;
    total_tokens: number;
    vocabulary_size: number;
    showing_first_n: number;
  } | null>(null);
  const [loadingBm25Details, setLoadingBm25Details] = useState(false);

  useEffect(() => {
    loadTables();
    loadBm25Info();
  }, []);

  const loadBm25Info = async () => {
    try {
      const info = await apiClient.getBm25Info();
      setBm25Info(info);
    } catch (error) {
      console.error('Failed to load BM25 info:', error);
      // Don't show error toast, BM25 might not be enabled
    }
  };

  const loadBm25Details = async () => {
    if (!bm25Info?.enabled || !bm25Info?.is_built) {
      return;
    }
    
    try {
      setLoadingBm25Details(true);
      const details = await apiClient.getBm25Details(200); // Get up to 200 documents
      setBm25Details(details);
    } catch (error) {
      console.error('Failed to load BM25 details:', error);
      toast.error('Failed to load BM25 index details');
    } finally {
      setLoadingBm25Details(false);
    }
  };

  useEffect(() => {
    if (showBm25Details && !bm25Details && bm25Info?.enabled && bm25Info?.is_built) {
      loadBm25Details();
    }
  }, [showBm25Details, bm25Info]);

  // Effect to set table info and defaults when table is selected
  useEffect(() => {
    if (selectedTable && tables.length > 0) {
      const tableInfo = tables.find(t => t.name === selectedTable);
      setSelectedTableInfo(tableInfo || null);

      if (tableInfo) {
        // Set default sort column if not set or invalid
        const currentSortValid = sortColumn && tableInfo.columns.some(col => col.name === sortColumn);
        if (!sortColumn || !currentSortValid) {
          const defaultSortColumn = getDefaultSortColumn(tableInfo.columns);
          if (defaultSortColumn !== sortColumn) {
            setSortColumn(defaultSortColumn);
            setSortDirection('asc');
          }
        }

        // Reset invalid filters
        const validFilters: Record<string, string> = {};
        Object.entries(filters).forEach(([key, value]) => {
          if (tableInfo.columns.some(col => col.name === key) && value.trim()) {
            validFilters[key] = value;
          }
        });
        setFilters(validFilters);
        
        // Sync filter inputs with valid filters (so inputs show current applied filters)
        const validFilterInputs: Record<string, string> = {};
        Object.entries(validFilters).forEach(([key, value]) => {
          validFilterInputs[key] = value;
        });
        setFilterInputs(validFilterInputs);
      }
    }
  }, [selectedTable, tables]);

  // Effect to load table data when parameters change
  useEffect(() => {
    if (selectedTable && tables.length > 0) {
      const tableInfo = tables.find(t => t.name === selectedTable);
      if (tableInfo && sortColumn) {
        loadTableData();
      }
    }
  }, [selectedTable, currentPage, sortColumn, sortDirection, filters, tables]);

  const loadTables = async () => {
    try {
      setLoading(true);
      const data = await apiClient.getDatabaseTables();
      setTables(data.tables);
    } catch (error) {
      console.error('Failed to load tables:', error);
      toast.error('Failed to load tables');
    } finally {
      setLoading(false);
    }
  };

  const loadTableData = async () => {
    if (!selectedTable) return;

    try {
      setLoading(true);
      console.log('Loading table data for:', selectedTable, {
        page: currentPage,
        sort: sortColumn,
        direction: sortDirection,
        filters
      });
      const data = await apiClient.getTableData(selectedTable, {
        page: currentPage,
        limit: 50,
        sort: sortColumn || undefined,
        direction: sortDirection,
        filters: Object.fromEntries(
          Object.entries(filters).filter(([_, value]) => value.trim() !== '')
        )
      });
      console.log('Table data loaded:', data);
      setTableData(data);
    } catch (error: any) {
      console.error('Failed to load table data:', error);
      console.error('Error response:', error.response?.data);
      toast.error(`Failed to load table data: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('asc');
    }
  };

  const handleFilterInputChange = (column: string, value: string) => {
    setFilterInputs(prev => ({ ...prev, [column]: value }));
  };

  const handleFilterApply = (column: string) => {
    const value = filterInputs[column] || '';
    setFilters(prev => ({ ...prev, [column]: value }));
    setCurrentPage(1); // Reset to first page when filtering
  };

  const handleFilterKeyDown = (e: React.KeyboardEvent<HTMLInputElement>, column: string) => {
    if (e.key === 'Enter') {
      handleFilterApply(column);
    }
  };

  const handleAddRecord = () => {
    if (!selectedTableInfo) return;

    // Initialize form data with empty values for all columns (except primary key)
    const initialData: Record<string, any> = {};
    selectedTableInfo.columns.forEach(col => {
      if (!col.primaryKey) {  // Don't include primary key in create form
        initialData[col.name] = '';
      }
    });

    setFormData(initialData);
    setShowCreateModal(true);
  };

  const handleEditRecord = (rowIndex: number) => {
    if (!tableData || !selectedTableInfo) return;

    const row = tableData.rows[rowIndex];
    const recordData: Record<string, any> = {};

    // Map row data to column names
    tableData.columns.forEach((col, index) => {
      recordData[col] = row[index];
    });

    // Find primary key value
    const pkColumn = selectedTableInfo.columns.find(col => col.primaryKey);
    if (pkColumn) {
      setEditingRecordId(recordData[pkColumn.name]);
    }

    // Remove primary key from editable fields
    if (pkColumn) {
      delete recordData[pkColumn.name];
    }

    setFormData(recordData);
    setShowEditModal(true);
  };

  const handleDeleteRecord = async (rowIndex: number) => {
    if (!tableData || !selectedTable || !selectedTableInfo) return;

    const row = tableData.rows[rowIndex];
    const pkColumn = selectedTableInfo.columns.find(col => col.primaryKey);
    if (!pkColumn) {
      toast.error('Cannot delete: No primary key found');
      return;
    }

    // Find the primary key value
    const pkIndex = tableData.columns.indexOf(pkColumn.name);
    const recordId = row[pkIndex];

    if (!confirm(`Are you sure you want to delete this record?`)) {
      return;
    }

    try {
      await apiClient.deleteRecord(selectedTable, recordId);
      toast.success('Record deleted successfully');
      loadTableData(); // Refresh data
    } catch (error) {
      console.error('Failed to delete record:', error);
      toast.error('Failed to delete record');
    }
  };

  const handleCreateRecord = async () => {
    if (!selectedTable) return;

    try {
      await apiClient.createRecord(selectedTable, formData);
      toast.success('Record created successfully');
      setShowCreateModal(false);
      setFormData({});
      loadTableData(); // Refresh data
    } catch (error) {
      console.error('Failed to create record:', error);
      toast.error('Failed to create record');
    }
  };

  const handleUpdateRecord = async () => {
    if (!selectedTable || editingRecordId === null) return;

    try {
      await apiClient.updateRecord(selectedTable, editingRecordId, formData);
      toast.success('Record updated successfully');
      setShowEditModal(false);
      setEditingRecordId(null);
      setFormData({});
      loadTableData(); // Refresh data
    } catch (error) {
      console.error('Failed to update record:', error);
      toast.error('Failed to update record');
    }
  };

  const handleFormChange = (column: string, value: any) => {
    setFormData(prev => ({ ...prev, [column]: value }));
  };

  const getDefaultSortColumn = (columns: ColumnInfo[]): string | null => {
    // Priority order for default sort columns
    const priorityColumns = ['id', 'created_at', 'updated_at', 'deleted_at', 'classification_date'];

    for (const colName of priorityColumns) {
      const column = columns.find(col => col.name === colName);
      if (column) {
        return colName;
      }
    }

    // Fallback to first non-primary key column, or first column if all are PK
    const nonPkColumns = columns.filter(col => !col.primaryKey);
    if (nonPkColumns.length > 0) {
      return nonPkColumns[0].name;
    }

    // Last resort: first column
    return columns.length > 0 ? columns[0].name : null;
  };

  const closeModals = () => {
    setShowCreateModal(false);
    setShowEditModal(false);
    setEditingRecordId(null);
    setFormData({});
  };

  const filteredTables = tables.filter(table =>
    table.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Database className="w-8 h-8 text-primary-600" />
          <div>
            <h1 className="text-3xl font-bold text-primary-900">Database</h1>
            <p className="text-primary-600">View and manage SQLite database tables</p>
          </div>
        </div>
        {bm25Info && (
          <Button
            variant="outline"
            onClick={() => setShowBm25Info(!showBm25Info)}
            className="flex items-center gap-2"
          >
            <FileSearch className="w-4 h-4" />
            BM25 Index
          </Button>
        )}
      </div>

      {/* BM25 Index Info Card */}
      {showBm25Info && bm25Info && (
        <Card className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <FileSearch className="w-5 h-5 text-primary-600" />
              <h2 className="text-lg font-semibold text-primary-900">BM25 Index Information</h2>
            </div>
            <button
              onClick={() => setShowBm25Info(false)}
              className="text-primary-400 hover:text-primary-600"
            >
              ✕
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-primary-600 mb-1">Status</div>
              <div className={`text-lg font-semibold ${
                bm25Info.status === 'active' ? 'text-green-600' : 'text-gray-500'
              }`}>
                {bm25Info.status === 'active' ? 'Active' : 'Inactive'}
              </div>
            </div>
            <div>
              <div className="text-sm text-primary-600 mb-1">Enabled</div>
              <div className={`text-lg font-semibold ${
                bm25Info.enabled ? 'text-green-600' : 'text-gray-500'
              }`}>
                {bm25Info.enabled ? 'Yes' : 'No'}
              </div>
            </div>
            <div>
              <div className="text-sm text-primary-600 mb-1">Index Built</div>
              <div className={`text-lg font-semibold ${
                bm25Info.is_built ? 'text-green-600' : 'text-yellow-600'
              }`}>
                {bm25Info.is_built ? 'Yes' : 'No'}
              </div>
            </div>
            <div>
              <div className="text-sm text-primary-600 mb-1">Documents Indexed</div>
              <div className="text-lg font-semibold text-primary-900">
                {bm25Info.document_count}
              </div>
            </div>
            {bm25Info.enabled && (
              <>
                <div>
                  <div className="text-sm text-primary-600 mb-1">Total Documents</div>
                  <div className="text-lg font-semibold text-primary-900">
                    {bm25Info.total_documents_in_db}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-primary-600 mb-1">Coverage</div>
                  <div className="text-lg font-semibold text-primary-900">
                    {bm25Info.index_coverage}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-primary-600 mb-1">Coverage %</div>
                  <div className={`text-lg font-semibold ${
                    bm25Info.coverage_percentage >= 95 ? 'text-green-600' :
                    bm25Info.coverage_percentage >= 50 ? 'text-yellow-600' : 'text-red-600'
                  }`}>
                    {bm25Info.coverage_percentage}%
                  </div>
                </div>
              </>
            )}
          </div>
          {bm25Info.enabled && !bm25Info.is_built && (
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
              <p className="text-sm text-yellow-800">
                BM25 index is enabled but not built. The index will be built automatically when documents are processed.
              </p>
            </div>
          )}
          {!bm25Info.enabled && (
            <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
              <p className="text-sm text-gray-600">
                BM25 keyword search is disabled. Enable it in config.yaml by setting <code className="bg-gray-100 px-1 rounded">semantic_search.enable_bm25: true</code>
              </p>
            </div>
          )}
          
          {/* Expandable Index Details Section */}
          {bm25Info.enabled && bm25Info.is_built && (
            <div className="mt-4">
              <button
                onClick={() => {
                  setShowBm25Details(!showBm25Details);
                  if (!showBm25Details && !bm25Details) {
                    loadBm25Details();
                  }
                }}
                className="w-full flex items-center justify-between p-3 bg-primary-50 hover:bg-primary-100 rounded-lg transition-colors"
              >
                <span className="font-medium text-primary-900">
                  {showBm25Details ? 'Hide' : 'Show'} Index Details
                </span>
                <span className="text-primary-600">
                  {showBm25Details ? '▲' : '▼'}
                </span>
              </button>
              
              {showBm25Details && (
                <div className="mt-4 space-y-4">
                  {loadingBm25Details ? (
                    <div className="flex items-center justify-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                    </div>
                  ) : bm25Details ? (
                    <>
                      {/* Index Statistics */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-primary-50 rounded-lg">
                        <div>
                          <div className="text-sm text-primary-600 mb-1">Total Documents</div>
                          <div className="text-lg font-semibold text-primary-900">
                            {bm25Details.total_documents}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-primary-600 mb-1">Total Tokens</div>
                          <div className="text-lg font-semibold text-primary-900">
                            {bm25Details.total_tokens.toLocaleString()}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-primary-600 mb-1">Vocabulary Size</div>
                          <div className="text-lg font-semibold text-primary-900">
                            {bm25Details.vocabulary_size.toLocaleString()}
                          </div>
                        </div>
                        <div>
                          <div className="text-sm text-primary-600 mb-1">Showing</div>
                          <div className="text-lg font-semibold text-primary-900">
                            {bm25Details.showing_first_n} / {bm25Details.total_documents}
                          </div>
                        </div>
                      </div>

                      {/* Document List */}
                      <div className="border border-primary-200 rounded-lg overflow-hidden">
                        <div className="bg-primary-100 px-4 py-2 font-semibold text-primary-900">
                          Indexed Documents ({bm25Details.document_details.length} shown)
                        </div>
                        <div className="max-h-96 overflow-y-auto">
                          <table className="w-full text-sm">
                            <thead className="bg-primary-50 sticky top-0">
                              <tr>
                                <th className="text-left py-2 px-4 font-semibold text-primary-900">ID</th>
                                <th className="text-left py-2 px-4 font-semibold text-primary-900">Filename</th>
                                <th className="text-right py-2 px-4 font-semibold text-primary-900">Tokens</th>
                                <th className="text-right py-2 px-4 font-semibold text-primary-900">Unique</th>
                                <th className="text-left py-2 px-4 font-semibold text-primary-900">Sample Tokens</th>
                              </tr>
                            </thead>
                            <tbody>
                              {bm25Details.document_details.map((doc, idx) => (
                                <tr key={doc.doc_id} className={idx % 2 === 0 ? 'bg-white' : 'bg-primary-50'}>
                                  <td className="py-2 px-4 text-primary-700 font-mono">{doc.doc_id}</td>
                                  <td className="py-2 px-4 text-primary-900 truncate max-w-xs" title={doc.filename}>
                                    {doc.filename}
                                  </td>
                                  <td className="py-2 px-4 text-right text-primary-700">{doc.token_count.toLocaleString()}</td>
                                  <td className="py-2 px-4 text-right text-primary-700">{doc.unique_token_count.toLocaleString()}</td>
                                  <td className="py-2 px-4">
                                    <div className="flex flex-wrap gap-1">
                                      {doc.sample_tokens.slice(0, 10).map((token, tokenIdx) => (
                                        <span
                                          key={tokenIdx}
                                          className="px-1.5 py-0.5 bg-primary-100 text-primary-700 rounded text-xs"
                                          title={token}
                                        >
                                          {token}
                                        </span>
                                      ))}
                                      {doc.sample_tokens.length > 10 && (
                                        <span className="text-primary-500 text-xs">+{doc.sample_tokens.length - 10}</span>
                                      )}
                                    </div>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>

                      {/* Document IDs List */}
                      <div className="border border-primary-200 rounded-lg p-4">
                        <div className="font-semibold text-primary-900 mb-2">
                          Document IDs in Index ({bm25Details.document_ids.length} shown)
                        </div>
                        <div className="max-h-32 overflow-y-auto">
                          <div className="flex flex-wrap gap-2">
                            {bm25Details.document_ids.map((docId) => (
                              <span
                                key={docId}
                                className="px-2 py-1 bg-primary-100 text-primary-700 rounded text-sm font-mono"
                              >
                                {docId}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="text-center py-4 text-primary-500">
                      Failed to load index details
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {/* Table Selection - Horizontal Layout */}
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-4">
          <Table className="w-5 h-5 text-primary-600" />
          <h2 className="text-lg font-semibold text-primary-900">Tables</h2>
          <div className="relative ml-4">
            <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-primary-400" />
            <Input
              placeholder="Search tables..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 w-64"
            />
          </div>
        </div>

        {/* Tables List - Horizontal */}
        <div className="flex flex-wrap gap-3">
          {filteredTables.map((table) => (
            <button
              key={table.name}
              onClick={() => setSelectedTable(table.name)}
              className={`flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-all min-w-[120px] ${
                selectedTable === table.name
                  ? 'bg-primary-100 border-primary-300 text-primary-900 shadow-md'
                  : 'bg-white border-primary-200 text-primary-700 hover:bg-primary-50 hover:border-primary-300'
              }`}
            >
              <div className="font-medium text-center">{table.name}</div>
              <div className="text-sm text-primary-500 mt-1">{table.rowCount} rows</div>
            </button>
          ))}
        </div>
      </Card>

      {/* Table Data */}
      <div>
          {selectedTable ? (
            <Card className="p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Table className="w-5 h-5 text-primary-600" />
                  <h2 className="text-xl font-semibold text-primary-900">{selectedTable}</h2>
                  {tableData && (
                    <span className="text-sm text-primary-500">
                      ({tableData.totalRows} rows)
                    </span>
                  )}
                </div>
                <Button onClick={handleAddRecord} size="sm">
                  <Plus className="w-4 h-4 mr-2" />
                  Add Record
                </Button>
              </div>

              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                </div>
              ) : tableData ? (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="border-b border-primary-200">
                        <th className="text-center py-2 px-4 font-semibold text-primary-900 w-20">Actions</th>
                        {tableData.columns.map((column) => {
                          const isContentColumn = column === 'content';
                          return (
                            <th
                              key={column}
                              className={`text-left py-2 px-4 font-semibold text-primary-900 ${isContentColumn ? 'w-[20vw] max-w-[20vw]' : ''}`}
                            >
                              <div className="flex items-center gap-2">
                                <button
                                  onClick={() => handleSort(column)}
                                  className="flex items-center gap-1 hover:text-primary-600"
                                >
                                  {column}
                                  {sortColumn === column && (
                                    sortDirection === 'asc' ? <SortAsc className="w-4 h-4" /> : <SortDesc className="w-4 h-4" />
                                  )}
                                </button>
                              </div>
                              {/* Filter input */}
                              <Input
                                placeholder={`Filter ${column}...`}
                                value={filterInputs[column] || ''}
                                onChange={(e) => handleFilterInputChange(column, e.target.value)}
                                onKeyDown={(e) => handleFilterKeyDown(e, column)}
                                className="mt-1 text-sm h-8"
                              />
                            </th>
                          );
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {tableData.rows.map((row, index) => (
                        <tr key={index} className="border-b border-primary-100 hover:bg-primary-50">
                          <td className="py-2 px-4 w-20">
                            <div className="flex gap-1 justify-center">
                              <button
                                onClick={() => handleEditRecord(index)}
                                className="p-1.5 text-primary-600 hover:text-primary-800 hover:bg-primary-100 rounded-md transition-colors duration-150"
                                title="Edit record"
                              >
                                <Edit className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => handleDeleteRecord(index)}
                                className="p-1.5 text-red-600 hover:text-red-800 hover:bg-red-100 rounded-md transition-colors duration-150"
                                title="Delete record"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                          </td>
                          {row.map((cell, cellIndex) => {
                            const columnName = tableData.columns[cellIndex];
                            const isContentColumn = columnName === 'content';
                            const isTimestampColumn = columnName === 'created_at' || columnName === 'updated_at' || columnName === 'deleted_at';

                            // Format timestamp columns
                            let displayValue = cell;
                            if (cell !== null && isTimestampColumn) {
                              try {
                                let timestamp: number;

                                // Convert cell to number
                                const numValue = typeof cell === 'number' ? cell : parseFloat(String(cell));
                                if (isNaN(numValue)) {
                                  throw new Error('Cannot convert to number');
                                }
                                timestamp = numValue;

                                // SQLite stores Unix timestamps in seconds, convert to milliseconds
                                timestamp = timestamp * 1000;

                                const date = new Date(timestamp);

                                // Validate the date is reasonable
                                if (date.getFullYear() < 2000 || date.getFullYear() > 2100) {
                                  throw new Error('Invalid date range');
                                }

                                // Format as ISO-like string in PST timezone
                                displayValue = date.toLocaleString('en-US', {
                                  timeZone: 'America/Los_Angeles',
                                  year: 'numeric',
                                  month: '2-digit',
                                  day: '2-digit',
                                  hour: '2-digit',
                                  minute: '2-digit',
                                  second: '2-digit',
                                  hour12: false
                                });

                              } catch (error) {
                                // Fallback to original value if parsing fails
                                console.warn(`Failed to format timestamp for ${columnName}:`, cell, error);
                                displayValue = String(cell);
                              }
                            }

                            return (
                              <td
                                key={cellIndex}
                                className={`py-2 px-4 text-sm ${isContentColumn ? 'w-[20vw] max-w-[20vw]' : ''}`}
                              >
                                {cell === null ? (
                                  <span className="text-primary-400 italic">NULL</span>
                                ) : (
                                  <div className={isContentColumn ? 'truncate max-w-full' : ''}>
                                    {String(displayValue)}
                                  </div>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>

                  {/* Pagination */}
                  {tableData.totalPages > 1 && (
                    <div className="flex items-center justify-between mt-4">
                      <div className="text-sm text-primary-600">
                        Page {tableData.page} of {tableData.totalPages}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                          disabled={currentPage === 1}
                        >
                          Previous
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setCurrentPage(Math.min(tableData.totalPages, currentPage + 1))}
                          disabled={currentPage === tableData.totalPages}
                        >
                          Next
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-primary-500">
                  No data available
                </div>
              )}
            </Card>
          ) : (
            <Card className="p-8 text-center">
              <Database className="w-16 h-16 text-primary-300 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-primary-900 mb-2">Select a Table</h3>
              <p className="text-primary-600">Choose a table from the sidebar to view its data</p>
            </Card>
          )}
        </div>

      {/* Create Record Modal */}
      {showCreateModal && selectedTableInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-primary-900">Create New Record</h3>
              <button
                onClick={closeModals}
                className="text-primary-400 hover:text-primary-600"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              {selectedTableInfo.columns
                .filter(col => !col.primaryKey)
                .map(column => (
                <div key={column.name}>
                  <label className="block text-sm font-medium text-primary-700 mb-1">
                    {column.name} {column.nullable ? '(optional)' : '(required)'}
                  </label>
                  <Input
                    type={column.type.toLowerCase().includes('int') || column.type.toLowerCase().includes('real') ? 'number' : 'text'}
                    value={formData[column.name] || ''}
                    onChange={(e) => handleFormChange(column.name, e.target.value)}
                    placeholder={`Enter ${column.name}...`}
                  />
                </div>
              ))}
            </div>

            <div className="flex gap-3 mt-6">
              <Button onClick={handleCreateRecord}>Create Record</Button>
              <Button variant="outline" onClick={closeModals}>Cancel</Button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Record Modal */}
      {showEditModal && selectedTableInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-primary-900">Edit Record</h3>
              <button
                onClick={closeModals}
                className="text-primary-400 hover:text-primary-600"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              {selectedTableInfo.columns
                .filter(col => !col.primaryKey)
                .map(column => (
                <div key={column.name}>
                  <label className="block text-sm font-medium text-primary-700 mb-1">
                    {column.name} {column.nullable ? '(optional)' : '(required)'}
                  </label>
                  <Input
                    type={column.type.toLowerCase().includes('int') || column.type.toLowerCase().includes('real') ? 'number' : 'text'}
                    value={formData[column.name] || ''}
                    onChange={(e) => handleFormChange(column.name, e.target.value)}
                    placeholder={`Enter ${column.name}...`}
                  />
                </div>
              ))}
            </div>

            <div className="flex gap-3 mt-6">
              <Button onClick={handleUpdateRecord}>Update Record</Button>
              <Button variant="outline" onClick={closeModals}>Cancel</Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
