#!/bin/bash

# Build script for React frontend
# This script builds the React application and prepares it for deployment

set -e  # Exit on any error

echo "🚀 Building React Frontend for DOCRAG"
echo "========================================================="

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "📦 Installing dependencies..."
if ! npm install; then
    echo "⚠️  npm install failed (likely permission issue)"
    echo "   Please run the following commands manually:"
    echo "   npm install"
    echo "   npm run build"
    echo ""
    echo "   If you get permission errors, try:"
    echo "   sudo chown -R $(whoami) ~/.npm"
    echo "   npm install"
    exit 1
fi

echo "🔍 Running type check..."
if ! npm run type-check; then
    echo "❌ TypeScript type check failed"
    exit 1
fi

echo "🎨 Building for production..."
if ! npm run build; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build completed successfully!"
echo ""
echo "📁 Built files are in the 'dist/' directory"
echo "🌐 To serve the built app locally, run: npm run preview"
echo ""
echo "📋 Next steps:"
echo "1. Ensure Flask backend is running with CORS enabled"
echo "2. The Flask app will automatically serve the React build"
echo "3. Access the app at http://localhost:5000"
echo ""
echo "🎉 Frontend refactoring complete!"
