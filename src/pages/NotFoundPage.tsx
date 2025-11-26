import { useNavigate } from 'react-router-dom';
import { Home, FileText } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent } from '@/components/ui/Card';

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <Card className="max-w-md w-full">
        <CardContent className="text-center py-12">
          <div className="text-6xl mb-4">🤖</div>
          <h1 className="text-2xl font-bold text-gray-900 mb-2">Page Not Found</h1>
          <p className="text-gray-600 mb-6">
            The page you're looking for doesn't exist in our document management system.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button onClick={() => navigate('/')} className="flex items-center gap-2">
              <Home className="w-4 h-4" />
              Go Home
            </Button>
            <Button variant="outline" onClick={() => navigate('/documents')} className="flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Browse Documents
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
