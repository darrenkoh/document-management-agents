import { useState, useEffect } from 'react';
import { BarChart3, FileText, Tag, PieChart } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { DocumentStats } from '@/types';
import { apiClient } from '@/lib/api';
import toast from 'react-hot-toast';

export default function StatsPage() {
  const [stats, setStats] = useState<DocumentStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    setLoading(true);
    try {
      const data = await apiClient.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
      toast.error('Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    );
  }

  if (!stats) {
    return (
      <div className="text-center py-12">
        <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-gray-900 mb-2">Failed to load statistics</h2>
        <p className="text-gray-600">Please try refreshing the page.</p>
      </div>
    );
  }

  // Calculate percentages for categories
  const categoryPercentages = stats.categories.map(([category, count]) => ({
    category,
    count,
    percentage: ((count / stats.total_docs) * 100).toFixed(1),
  }));

  // Calculate percentages for file types
  const fileTypePercentages = stats.file_types.map(([type, count]) => ({
    type,
    count,
    percentage: ((count / stats.total_docs) * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Statistics</h1>
        <p className="text-gray-600 mt-1">
          Overview of your document collection and classification data
        </p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
                <FileText className="w-6 h-6 text-primary-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{stats.total_docs}</p>
                <p className="text-sm text-gray-600">Total Documents</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-success-100 rounded-lg flex items-center justify-center">
                <Tag className="w-6 h-6 text-success-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{stats.categories.length}</p>
                <p className="text-sm text-gray-600">Categories</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-warning-100 rounded-lg flex items-center justify-center">
                <PieChart className="w-6 h-6 text-warning-600" />
              </div>
              <div>
                <p className="text-2xl font-bold text-gray-900">{stats.file_types.length}</p>
                <p className="text-sm text-gray-600">File Types</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Categories Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Tag className="w-5 h-5" />
              Categories Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {categoryPercentages.length > 0 ? (
              <div className="space-y-4">
                {categoryPercentages.slice(0, 10).map(({ category, count, percentage }) => (
                  <div key={category} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-900">{category}</span>
                      <span className="text-sm text-gray-600">
                        {count} ({percentage}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
                {categoryPercentages.length > 10 && (
                  <p className="text-sm text-gray-600 text-center">
                    And {categoryPercentages.length - 10} more categories...
                  </p>
                )}
              </div>
            ) : (
              <p className="text-gray-600 text-center py-8">No categories found</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              File Types Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {fileTypePercentages.length > 0 ? (
              <div className="space-y-4">
                {fileTypePercentages.map(({ type, count, percentage }) => (
                  <div key={type} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium text-gray-900 uppercase">
                        {type}
                      </span>
                      <span className="text-sm text-gray-600">
                        {count} ({percentage}%)
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-success-600 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${percentage}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-600 text-center py-8">No file types found</p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Top Categories Table */}
      <Card>
        <CardHeader>
          <CardTitle>Top Categories</CardTitle>
        </CardHeader>
        <CardContent>
          {stats.categories.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-medium text-gray-900">Category</th>
                    <th className="text-right py-3 px-4 font-medium text-gray-900">Count</th>
                    <th className="text-right py-3 px-4 font-medium text-gray-900">Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.categories.map(([category, count]) => (
                    <tr key={category} className="border-b border-gray-100">
                      <td className="py-3 px-4 text-gray-900">{category}</td>
                      <td className="py-3 px-4 text-right text-gray-900">{count}</td>
                      <td className="py-3 px-4 text-right text-gray-600">
                        {((count / stats.total_docs) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-600 text-center py-8">No categories found</p>
          )}
        </CardContent>
      </Card>

      {/* File Types Table */}
      <Card>
        <CardHeader>
          <CardTitle>File Types Summary</CardTitle>
        </CardHeader>
        <CardContent>
          {stats.file_types.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-medium text-gray-900">File Type</th>
                    <th className="text-right py-3 px-4 font-medium text-gray-900">Count</th>
                    <th className="text-right py-3 px-4 font-medium text-gray-900">Percentage</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.file_types.map(([type, count]) => (
                    <tr key={type} className="border-b border-gray-100">
                      <td className="py-3 px-4 text-gray-900 uppercase">{type}</td>
                      <td className="py-3 px-4 text-right text-gray-900">{count}</td>
                      <td className="py-3 px-4 text-right text-gray-600">
                        {((count / stats.total_docs) * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-600 text-center py-8">No file types found</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
