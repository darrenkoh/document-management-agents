import { Outlet, Link, useLocation } from 'react-router-dom';
import { useState, useEffect } from 'react';
import {
  Home,
  FileText,
  BarChart3,
  Menu,
  X,
  RefreshCw,
  Volume2,
  VolumeX
} from 'lucide-react';
import { NavItem } from '@/types';
import { Button } from '@/components/ui/Button';
import { apiClient } from '@/lib/api';
import toast from 'react-hot-toast';

const navigation: NavItem[] = [
  { label: 'Home', href: '/', icon: Home },
  { label: 'Documents', href: '/documents', icon: FileText },
  { label: 'Statistics', href: '/stats', icon: BarChart3 },
];

export default function Layout() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isVerbose, setIsVerbose] = useState(false);
  const location = useLocation();

  const handleRefresh = async () => {
    try {
      const response = await fetch('/refresh');
      if (response.ok) {
        window.location.reload();
      }
    } catch (error) {
      console.error('Failed to refresh:', error);
    }
  };

  const handleToggleVerbose = async () => {
    try {
      const data = await apiClient.setVerboseState(!isVerbose);
      setIsVerbose(data.verbose);
      toast.success(data.message);
    } catch (error) {
      console.error('Failed to toggle verbose:', error);
      toast.error('Failed to toggle verbose logging');
    }
  };

  // Fetch initial verbose state
  useEffect(() => {
    const fetchVerboseState = async () => {
      try {
        const data = await apiClient.getVerboseState();
        setIsVerbose(data.verbose);
      } catch (error) {
        console.error('Failed to fetch verbose state:', error);
      }
    };

    fetchVerboseState();
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-white border-b border-gray-200 shadow-soft">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3 group">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center group-hover:bg-primary-700 transition-colors">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900 group-hover:text-primary-600 transition-colors">
                DocAgent
              </span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center gap-6">
              {navigation.map((item) => {
                const Icon = item.icon!;
                const isActive = location.pathname === item.href;

                return (
                  <Link
                    key={item.href}
                    to={item.href}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                );
              })}

              {/* Verbose Logging Toggle */}
              <Button
                variant="ghost"
                size="sm"
                onClick={handleToggleVerbose}
                className={`text-gray-600 hover:text-gray-900 ${isVerbose ? 'bg-yellow-50 text-yellow-700' : ''}`}
                title={isVerbose ? 'Disable Verbose Logging' : 'Enable Verbose Logging'}
              >
                {isVerbose ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </Button>

              {/* Refresh Button */}
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRefresh}
                className="text-gray-600 hover:text-gray-900"
                title="Refresh Database"
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </nav>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-gray-600 hover:text-gray-900 hover:bg-gray-50"
            >
              {isMobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <Menu className="w-6 h-6" />
              )}
            </button>
          </div>

          {/* Mobile Navigation */}
          {isMobileMenuOpen && (
            <div className="md:hidden border-t border-gray-200 py-4">
              <nav className="flex flex-col gap-2">
                {navigation.map((item) => {
                  const Icon = item.icon!;
                  const isActive = location.pathname === item.href;

                  return (
                    <Link
                      key={item.href}
                      to={item.href}
                      onClick={() => setIsMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-primary-50 text-primary-700'
                          : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      {item.label}
                    </Link>
                  );
                })}

                {/* Mobile Verbose Toggle */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    handleToggleVerbose();
                    setIsMobileMenuOpen(false);
                  }}
                  className={`justify-start ${isVerbose ? 'bg-yellow-50 text-yellow-700' : 'text-gray-600 hover:text-gray-900'}`}
                >
                  {isVerbose ? <Volume2 className="w-5 h-5 mr-3" /> : <VolumeX className="w-5 h-5 mr-3" />}
                  {isVerbose ? 'Disable Verbose' : 'Enable Verbose'}
                </Button>

                {/* Mobile Refresh Button */}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    handleRefresh();
                    setIsMobileMenuOpen(false);
                  }}
                  className="justify-start text-gray-600 hover:text-gray-900"
                >
                  <RefreshCw className="w-5 h-5 mr-3" />
                  Refresh Database
                </Button>
              </nav>
            </div>
          )}
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1">
        <div className="container mx-auto px-4 py-8">
          <Outlet />
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white mt-auto">
        <div className="container mx-auto px-4 py-6">
          <div className="text-center text-sm text-gray-500">
            <p>Document Management System - Powered by AI</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
