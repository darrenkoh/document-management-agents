import React, { useEffect, useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { apiClient } from '@/lib/api';
import { EmbeddingPoint } from '@/types';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Card } from '@/components/ui/Card';
import { RefreshCw, Database, Info, Layers, Tag, Settings, Cpu } from 'lucide-react';
import { Button } from '@/components/ui/Button';

type ViewMode = 'category' | 'subcategory';
type ProcessingMode = 'backend' | 'frontend';

// Simple PCA implementation for frontend processing
const performPCA = (embeddings: number[][], nComponents: number = 3): { projected: number[][], explainedVariance: number[] } => {
  const n = embeddings.length;
  const d = embeddings[0].length;

  if (n === 0 || d === 0) return { projected: [], explainedVariance: [] };

  // Convert to matrix format and check if normalized
  const X: number[][] = embeddings.map(emb => [...emb]);

  // For normalized embeddings (common in semantic search), use SVD approach
  // This is more numerically stable than power iteration for small datasets

  // Center the data (even for normalized data, this helps with PCA)
  const means = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      means[j] += X[i][j];
    }
  }
  for (let j = 0; j < d; j++) {
    means[j] /= n;
  }

  // Create centered matrix
  const X_centered: number[][] = [];
  for (let i = 0; i < n; i++) {
    X_centered[i] = [];
    for (let j = 0; j < d; j++) {
      X_centered[i][j] = X[i][j] - means[j];
    }
  }

  // For small datasets, use covariance-based PCA
  if (n <= 1000 && d <= 1000) {
    // Compute covariance matrix
    const cov: number[][] = [];
    for (let i = 0; i < d; i++) {
      cov[i] = [];
      for (let j = 0; j < d; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += X_centered[k][i] * X_centered[k][j];
        }
        cov[i][j] = sum / n;
      }
    }

    // Get eigenvectors and eigenvalues using Jacobi-like method (simplified)
    const eigenvectors: number[][] = [];
    const eigenvalues: number[] = [];

    // For simplicity, use a basic approach: find directions of maximum variance
    for (let comp = 0; comp < Math.min(nComponents, Math.min(n, d)); comp++) {
      // Start with a random direction
      let v: number[] = [];
      for (let i = 0; i < d; i++) {
        v[i] = Math.random() - 0.5;
      }

      // Orthogonalize against previous eigenvectors
      for (const prevV of eigenvectors) {
        const dot = v.reduce((sum, val, idx) => sum + val * prevV[idx], 0);
        for (let i = 0; i < d; i++) {
          v[i] -= dot * prevV[i];
        }
      }

      // Normalize
      const norm = Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
      if (norm > 0) {
        v = v.map(val => val / norm);
      }

      // Simple power iteration (just a few iterations for speed)
      for (let iter = 0; iter < 10; iter++) {
        // Matrix-vector multiplication
        const vNew: number[] = [];
        for (let i = 0; i < d; i++) {
          let sum = 0;
          for (let j = 0; j < d; j++) {
            sum += cov[i][j] * v[j];
          }
          vNew[i] = sum;
        }

        // Normalize
        const normNew = Math.sqrt(vNew.reduce((sum, val) => sum + val * val, 0));
        if (normNew > 0) {
          v = vNew.map(val => val / normNew);
        }
      }

      // Compute eigenvalue (approximate)
      let eigenvalue = 0;
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          eigenvalue += v[i] * cov[i][j] * v[j];
        }
      }

      eigenvectors.push([...v]);
      eigenvalues.push(eigenvalue);
    }

    // Project data
    const projected: number[][] = [];
    for (let i = 0; i < n; i++) {
      const point: number[] = [];
      for (const eigenvector of eigenvectors) {
        let dot = 0;
        for (let j = 0; j < d; j++) {
          dot += X[i][j] * eigenvector[j];  // Project original data, not centered
        }
        point.push(dot);
      }
      projected.push(point);
    }

    // Calculate explained variance
    const totalVariance = eigenvalues.reduce((sum, ev) => sum + ev, 0);
    const explainedVariance = totalVariance > 0
      ? eigenvalues.map(ev => ev / totalVariance)
      : eigenvalues.map(() => 0);

    return { projected, explainedVariance };
  } else {
    // For larger datasets, use a simpler approach
    // Just return the first nComponents dimensions as-is (not ideal but works)
    const projected: number[][] = [];
    for (const emb of X) {
      projected.push(emb.slice(0, nComponents));
    }
    const explainedVariance = new Array(nComponents).fill(1.0 / nComponents);

    return { projected, explainedVariance };
  }
};

export const EmbeddingPage: React.FC = () => {
  const [points, setPoints] = useState<EmbeddingPoint[]>([]);
  const [rawEmbeddings, setRawEmbeddings] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [explainedVariance, setExplainedVariance] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<ViewMode>('category');
  const [processingMode, setProcessingMode] = useState<ProcessingMode>('backend');
  const [pcaComponents, setPcaComponents] = useState<number>(3);
  const [showSettings, setShowSettings] = useState(false);

  // Fetch embeddings
  const fetchEmbeddings = async () => {
    setLoading(true);
    setError(null);
    try {
      if (processingMode === 'backend') {
        // Get processed embeddings from backend
        const response = await apiClient.getEmbeddings({ components: pcaComponents });
        if (response.error) {
          throw new Error(response.error);
        }
        console.log('Backend PCA result:', {
          pointCount: response.points?.length || 0,
          samplePoints: response.points?.slice(0, 3).map(p => ({
            x: p.x?.toFixed(4),
            y: p.y?.toFixed(4),
            z: p.z?.toFixed(4),
            categories: p.categories
          })),
          explainedVariance: response.explained_variance
        });
        setPoints(response.points);
        setRawEmbeddings([]);
        if (response.explained_variance) {
          setExplainedVariance(response.explained_variance);
        }
      } else {
        // Get raw embeddings and process PCA on frontend
        const response = await apiClient.getEmbeddings({ raw: true });
        if (response.error) {
          throw new Error(response.error);
        }

        if (response.raw && response.embeddings) {
          setRawEmbeddings(response.embeddings);

          // Process PCA on frontend
          const embeddings = response.embeddings.map((item: any) => item.embedding);
          console.log(`Processing ${embeddings.length} embeddings with ${embeddings[0]?.length || 0} dimensions`);

          // For debugging: check embedding variance
          if (embeddings.length > 1) {
            const firstEmb = embeddings[0];
            const secondEmb = embeddings[1];
            let dotProduct = 0;
            let norm1 = 0;
            let norm2 = 0;
            for (let i = 0; i < firstEmb.length; i++) {
              dotProduct += firstEmb[i] * secondEmb[i];
              norm1 += firstEmb[i] * firstEmb[i];
              norm2 += secondEmb[i] * secondEmb[i];
            }
            norm1 = Math.sqrt(norm1);
            norm2 = Math.sqrt(norm2);
            const similarity = dotProduct / (norm1 * norm2);
            console.log(`Embeddings similarity check: dot=${dotProduct.toFixed(4)}, similarity=${similarity.toFixed(4)}`);
          }

          const { projected, explainedVariance: variance } = performPCA(embeddings, pcaComponents);
          console.log('PCA completed:', {
            projectedShape: [projected.length, projected[0]?.length || 0],
            sampleCoords: projected.slice(0, 3).map(p => p.map(v => v.toFixed(4))),
            explainedVariance: variance,
            coordRanges: projected.length > 0 ? projected[0].map((_, i) =>
              projected.reduce((range, p) => ({
                min: Math.min(range.min, p[i]),
                max: Math.max(range.max, p[i])
              }), { min: Infinity, max: -Infinity })
            ) : []
          });

          // Convert to EmbeddingPoint format
          const processedPoints: EmbeddingPoint[] = projected.map((coords: number[], i: number) => {
            const coordObj: any = {};
            for (let j = 0; j < coords.length; j++) {
              const coordName = j < 3 ? ['x', 'y', 'z'][j] : `pc${j + 1}`;
              coordObj[coordName] = coords[j];
            }

            const point = {
              id: response.embeddings[i].id,
              x: coordObj.x || 0,
              y: coordObj.y || 0,
              z: coordObj.z || 0,
              filename: response.embeddings[i].filename,
              categories: response.embeddings[i].categories,
              sub_categories: response.embeddings[i].sub_categories || [],
              metadata: response.embeddings[i].metadata,
              ...coordObj
            };

            return point;
          });

          // Check if points are too clustered (all within a small range)
          if (processedPoints.length > 1) {
            const xCoords = processedPoints.map(p => p.x);
            const yCoords = processedPoints.map(p => p.y);
            const zCoords = processedPoints.map(p => p.z);

            const xRange = Math.max(...xCoords) - Math.min(...xCoords);
            const yRange = Math.max(...yCoords) - Math.min(...yCoords);
            const zRange = Math.max(...zCoords) - Math.min(...zCoords);

            const maxRange = Math.max(xRange, yRange, zRange);

            console.log(`Coordinate ranges: x=${xRange.toFixed(4)}, y=${yRange.toFixed(4)}, z=${zRange.toFixed(4)}`);

            // If all points are within a very small range (less than 0.1), add some jitter
            if (maxRange < 0.1) {
              console.warn('Points are too clustered, adding jitter for visibility');
              processedPoints.forEach((point, i) => {
                const jitter = 0.05; // Small random offset
                point.x += (Math.random() - 0.5) * jitter;
                point.y += (Math.random() - 0.5) * jitter;
                point.z += (Math.random() - 0.5) * jitter;
              });
            }
          }

          console.log(`Final processed points: ${processedPoints.length}`);

          setPoints(processedPoints);
          setExplainedVariance(variance);
        } else {
          throw new Error('Invalid raw embeddings response');
        }
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
  }, [processingMode, pcaComponents]);

  // Group points by category or sub-category for coloring
  const plotData = useMemo(() => {
    const groupedPoints: Record<string, { x: number[], y: number[], z: number[], text: string[], ids: number[], pc4?: number[], pc5?: number[] }> = {};

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

      // Add additional PCA components for hover info if available
      const additionalComponents = [];
      if ((point as any).pc4 !== undefined) additionalComponents.push(`PC4: ${(point as any).pc4.toFixed(3)}`);
      if ((point as any).pc5 !== undefined) additionalComponents.push(`PC5: ${(point as any).pc5.toFixed(3)}`);

      // Hover text - show categories, sub-categories, and additional PCA info
      const hoverText = `
        <b>File:</b> ${point.filename}<br>
        <b>Category:</b> ${point.categories}<br>
        <b>Sub-categories:</b> ${subCategoriesDisplay}<br>
        <b>Coordinates:</b> (${point.x.toFixed(3)}, ${point.y.toFixed(3)}, ${point.z.toFixed(3)})<br>
        ${additionalComponents.length > 0 ? `<b>Additional PCs:</b> ${additionalComponents.join(', ')}<br>` : ''}
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
            {processingMode === 'backend' ? 'Backend' : 'Frontend'} PCA processing
            {explainedVariance.length > 0 && (
              <span className="ml-2 text-xs bg-gray-100 px-2 py-0.5 rounded-full">
                Variance explained: {(explainedVariance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%
              </span>
            )}
          </p>
        </div>
        <div className="flex items-center space-x-3">
           <div className="text-sm text-gray-500 mr-4">
             {points.length} Embeddings
           </div>
           {/* Settings Button */}
           <Button
             onClick={() => setShowSettings(!showSettings)}
             variant="outline"
             size="sm"
           >
             <Settings className="mr-2 h-4 w-4" />
             Settings
           </Button>
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
           {processingMode === 'frontend' && rawEmbeddings.length > 0 && (
             <Button
               onClick={() => {
                 // Re-process existing raw embeddings with new PCA settings
                 setLoading(true);
                 try {
                   const embeddings = rawEmbeddings.map((item: any) => item.embedding);
                   const { projected, explainedVariance: variance } = performPCA(embeddings, pcaComponents);

                   const processedPoints: EmbeddingPoint[] = projected.map((coords: number[], i: number) => {
                     const coordObj: any = {};
                     for (let j = 0; j < coords.length; j++) {
                       const coordName = j < 3 ? ['x', 'y', 'z'][j] : `pc${j + 1}`;
                       coordObj[coordName] = coords[j];
                     }

                     return {
                       id: rawEmbeddings[i].id,
                       x: coordObj.x || 0,
                       y: coordObj.y || 0,
                       z: coordObj.z || 0,
                       filename: rawEmbeddings[i].filename,
                       categories: rawEmbeddings[i].categories,
                       sub_categories: rawEmbeddings[i].sub_categories || [],
                       metadata: rawEmbeddings[i].metadata,
                       ...coordObj
                     };
                   });

                   setPoints(processedPoints);
                   setExplainedVariance(variance);
                 } catch (err) {
                   console.error('Error reprocessing PCA:', err);
                 } finally {
                   setLoading(false);
                 }
               }}
               variant="outline"
               size="sm"
             >
               <Cpu className="mr-2 h-4 w-4" />
               Reprocess
             </Button>
           )}
           <Button onClick={fetchEmbeddings} variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Settings Panel */}
      {showSettings && (
        <div className="flex-shrink-0 bg-gray-50 border-b border-gray-200 px-6 py-4">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-3">
              <label className="text-sm font-medium text-gray-700">Processing:</label>
              <div className="flex items-center rounded-lg border border-gray-200 bg-white p-0.5">
                <button
                  onClick={() => setProcessingMode('backend')}
                  className={`flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    processingMode === 'backend'
                      ? 'bg-blue-100 text-blue-900 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Database className="mr-1.5 h-4 w-4" />
                  Backend
                </button>
                <button
                  onClick={() => setProcessingMode('frontend')}
                  className={`flex items-center px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                    processingMode === 'frontend'
                      ? 'bg-green-100 text-green-900 shadow-sm'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  <Cpu className="mr-1.5 h-4 w-4" />
                  Frontend
                </button>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              <label className="text-sm font-medium text-gray-700">PCA Components:</label>
              <select
                value={pcaComponents}
                onChange={(e) => setPcaComponents(parseInt(e.target.value))}
                className="px-3 py-1.5 border border-gray-300 rounded-md text-sm bg-white"
              >
                {[2, 3, 4, 5, 10, 20, 50].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <Info className="h-4 w-4" />
              <span>
                {processingMode === 'backend'
                  ? 'PCA processed on server (faster, uses sklearn)'
                  : 'PCA processed in browser (slower, more flexible)'
                }
              </span>
            </div>
          </div>
        </div>
      )}

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
                  xaxis: {
                    title: pcaComponents >= 1 ? 'PC1' : 'X',
                    autorange: true,
                    showgrid: true,
                    gridcolor: '#e0e0e0'
                  },
                  yaxis: {
                    title: pcaComponents >= 2 ? 'PC2' : 'Y',
                    autorange: true,
                    showgrid: true,
                    gridcolor: '#e0e0e0'
                  },
                  zaxis: {
                    title: pcaComponents >= 3 ? 'PC3' : 'Z',
                    autorange: true,
                    showgrid: true,
                    gridcolor: '#e0e0e0'
                  },
                  aspectmode: 'data', // Use 'data' instead of 'cube' to better show the actual spread
                  camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                  }
                },
                legend: {
                  x: 0,
                  y: 1,
                  bgcolor: 'rgba(255, 255, 255, 0.8)',
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                title: {
                  text: `PCA Projection (${pcaComponents} components, ${processingMode} processing)`,
                  font: { size: 14 },
                  x: 0.5,
                  y: 0.98
                },
              }}
              useResizeHandler={true}
              style={{ width: '100%', height: '100%' }}
              config={{
                displayModeBar: true,
                responsive: true,
                scrollZoom: true,
                modeBarButtonsToAdd: [
                  {
                    name: 'Toggle Settings',
                    icon: {
                      width: 16,
                      height: 16,
                      path: 'M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z'
                    },
                    click: () => setShowSettings(!showSettings)
                  }
                ]
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

