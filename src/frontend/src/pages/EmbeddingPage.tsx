import React, { useEffect, useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { apiClient } from '@/lib/api';
import { EmbeddingPoint } from '@/types';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Card } from '@/components/ui/Card';
import { RefreshCw, Database, Info, Layers, Tag } from 'lucide-react';
import { Button } from '@/components/ui/Button';

type ViewMode = 'category' | 'subcategory';

export const EmbeddingPage: React.FC = () => {
  const [points, setPoints] = useState<EmbeddingPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [explainedVariance, setExplainedVariance] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('category');

  // Fetch embeddings
  const fetchEmbeddings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.getEmbeddings();
      if (response.error) {
        throw new Error(response.error);
      }
      setPoints(response.points);
      if (response.explained_variance) {
        setExplainedVariance(response.explained_variance);
      }
    } catch (err: any) {
      console.error('Failed to fetch embeddings:', err);
      setError(err.message || 'Failed to load embedding visualization');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEmbeddings();
  }, []);

  // Group points by category or sub-category for coloring
  const plotData = useMemo(() => {
    const groupedPoints: Record<string, { x: number[], y: number[], z: number[], text: string[], ids: number[] }> = {};
    
    points.forEach(point => {
      const subCategories = point.sub_categories || [];
      const subCategoriesDisplay = subCategories.length > 0 ? subCategories.join(', ') : 'None';
      
      // Determine grouping key based on view mode
      let groupKey: string;
      if (viewMode === 'subcategory') {
        // Use first sub-category for grouping, or 'Uncategorized' if none
        groupKey = subCategories.length > 0 ? subCategories[0] : 'Uncategorized';
      } else {
        // Use primary category (first part of hyphenated categories)
        const category = point.categories || 'Uncategorized';
        groupKey = category.split('-')[0].trim();
      }
      
      if (!groupedPoints[groupKey]) {
        groupedPoints[groupKey] = { x: [], y: [], z: [], text: [], ids: [] };
      }
      
      groupedPoints[groupKey].x.push(point.x);
      groupedPoints[groupKey].y.push(point.y);
      groupedPoints[groupKey].z.push(point.z);
      
      // Hover text - always show both categories and sub-categories
      const hoverText = `
        <b>File:</b> ${point.filename}<br>
        <b>Category:</b> ${point.categories}<br>
        <b>Sub-categories:</b> ${subCategoriesDisplay}<br>
        <b>ID:</b> ${point.id}
      `;
      groupedPoints[groupKey].text.push(hoverText);
      groupedPoints[groupKey].ids.push(point.id);
    });

    return Object.entries(groupedPoints).map(([groupName, data]) => ({
      x: data.x,
      y: data.y,
      z: data.z,
      mode: 'markers' as const,
      type: 'scatter3d' as const,
      name: groupName,
      text: data.text,
      hoverinfo: 'text' as const,
      marker: {
        size: 5,
        opacity: 0.8,
      }
    }));
  }, [points, viewMode]);

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <LoadingSpinner size="lg" />
        <span className="ml-3 text-gray-500">Loading 3D visualization...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center p-6">
        <Card className="w-full max-w-md p-6 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-red-100 text-red-500">
            <Database className="h-6 w-6" />
          </div>
          <h3 className="mb-2 text-lg font-medium text-gray-900">Visualization Error</h3>
          <p className="mb-4 text-gray-500">{error}</p>
          <Button onClick={fetchEmbeddings} variant="primary">
            <RefreshCw className="mr-2 h-4 w-4" />
            Retry
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - 80px)' }}>
      {/* Header */}
      <div className="flex-shrink-0 flex items-center justify-between border-b border-gray-200 bg-white px-6 py-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Embedding Projector</h1>
          <p className="text-sm text-gray-500">
            3D visualization of document embeddings using PCA
            {explainedVariance.length > 0 && (
              <span className="ml-2 text-xs bg-gray-100 px-2 py-0.5 rounded-full">
                Variance explained: {(explainedVariance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center space-x-3">
           <div className="text-sm text-gray-500 mr-4">
             {points.length} Documents
           </div>
           {/* View Mode Toggle */}
           <div className="flex items-center rounded-lg border border-gray-200 bg-gray-50 p-0.5">
             <button
               onClick={() => setViewMode('category')}
               className={`flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                 viewMode === 'category'
                   ? 'bg-white text-gray-900 shadow-sm'
                   : 'text-gray-500 hover:text-gray-700'
               }`}
             >
               <Layers className="mr-1.5 h-4 w-4" />
               Category
             </button>
             <button
               onClick={() => setViewMode('subcategory')}
               className={`flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                 viewMode === 'subcategory'
                   ? 'bg-white text-gray-900 shadow-sm'
                   : 'text-gray-500 hover:text-gray-700'
               }`}
             >
               <Tag className="mr-1.5 h-4 w-4" />
               Sub-category
             </button>
           </div>
           <Button onClick={fetchEmbeddings} variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 relative bg-gray-50 overflow-hidden" style={{ minHeight: '500px' }}>
        {points.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-gray-500">
              <Info className="mx-auto h-10 w-10 mb-2 text-gray-400" />
              <p>No embeddings found. Process some documents to generate embeddings.</p>
            </div>
          </div>
        ) : (
          <div className="absolute inset-0">
            <Plot
              data={plotData}
              layout={{
                autosize: true,
                margin: { l: 0, r: 0, b: 0, t: 0 },
                scene: {
                  xaxis: { title: 'PC1' },
                  yaxis: { title: 'PC2' },
                  zaxis: { title: 'PC3' },
                  aspectmode: 'cube',
                },
                legend: {
                  x: 0,
                  y: 1,
                  bgcolor: 'rgba(255, 255, 255, 0.8)',
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{
                displayModeBar: true,
                responsive: true,
                scrollZoom: true,
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

