import { Outlet, Link, useLocation } from 'react-router-dom';
import { useState } from 'react';
import {
  Home,
  FileText,
  BarChart3,
  Database,
  Menu,
  X,
  RefreshCw,
  Terminal,
  ScatterChart,
  Sparkles
} from 'lucide-react';
import { NavItem } from '@/types';
import { Button } from '@/components/ui/Button';

const navigation: NavItem[] = [
  { label: 'Ask', href: '/', icon: Home },
  { label: 'Documents', href: '/documents', icon: FileText },
  { label: 'Database', href: '/database', icon: Database },
  { label: 'Embeddings', href: '/embeddings', icon: ScatterChart },
  { label: 'Embed Search', href: '/embedding-search', icon: Sparkles },
  { label: 'Statistics', href: '/stats', icon: BarChart3 },
];

export default function Layout() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const location = useLocation();

  const handleRefresh = async () => {
    try {
      const response = await fetch('/api/refresh');
      if (response.ok) {
        window.location.reload();
      }
    } catch (error) {
      console.error('Failed to refresh:', error);
    }
  };

  return (
    <div className="min-h-screen bg-primary-50 font-sans text-primary-900 flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-primary-50/80 backdrop-blur-md border-b border-primary-200">
        <div className="container mx-auto px-6">
          <div className="flex items-center justify-between h-20">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 bg-primary-900 rounded-lg flex items-center justify-center shadow-lg shadow-primary-900/20 group-hover:scale-105 transition-transform duration-200">
                <Terminal className="w-6 h-6 text-white" />
              </div>
              <span className="text-xl font-bold tracking-tight text-primary-900">
                DocAgent
              </span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-8">
              {navigation.map((item) => {
                const isActive = location.pathname === item.href;
                return (
                  <Link
                    key={item.href}
                    to={item.href}
                    className={`text-sm font-medium transition-colors duration-200 ${isActive
                        ? 'text-primary-900'
                        : 'text-primary-600 hover:text-primary-900'
                      }`}
                  >
                    {item.label}
                  </Link>
                );
              })}
            </nav>

            {/* Right Side Actions */}
            <div className="hidden md:flex items-center gap-4">
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                className="font-mono text-xs"
              >
                <RefreshCw className="w-3 h-3 mr-2" />
                REFRESH
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-primary-600 hover:bg-primary-100"
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t border-primary-200 bg-white/95 backdrop-blur-xl absolute w-full left-0 shadow-xl">
            <div className="container mx-auto px-6 py-6 space-y-4">
              <nav className="flex flex-col gap-4">
                {navigation.map((item) => {
                  const Icon = item.icon!;
                  const isActive = location.pathname === item.href;
                  return (
                    <Link
                      key={item.href}
                      to={item.href}
                      onClick={() => setIsMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-colors ${isActive
                          ? 'bg-primary-50 text-primary-900'
                          : 'text-primary-600 hover:bg-primary-50 hover:text-primary-900'
                        }`}
                    >
                      <Icon className="w-5 h-5" />
                      {item.label}
                    </Link>
                  );
                })}

                <div className="h-px bg-primary-100 my-2" />

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    handleRefresh();
                    setIsMobileMenuOpen(false);
                  }}
                  className="justify-start"
                >
                  <RefreshCw className="w-5 h-5 mr-3" />
                  Refresh Database
                </Button>
              </nav>
            </div>
          </div>
        )}
      </header>

      {/* Main Content */}
      <main className="flex-1 flex flex-col">
        <div className={`flex-1 flex flex-col ${location.pathname === '/embeddings' ? '' : 'container mx-auto px-6 py-3'}`}>
          <Outlet />
        </div>
      </main>
    </div>
  );
}
