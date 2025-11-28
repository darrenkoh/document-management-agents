# React Frontend for DOCRAG

This document outlines the complete refactoring of the DOCRAG UI from Flask server-side rendered templates to a modern React frontend.

## Overview

### 🎯 **Clean Architecture**

**Before**: Flask handled both APIs and UI rendering (dual-mode)
**After**: Complete separation of concerns

- **Flask Backend**: Pure JSON API server (no UI rendering)
- **React Frontend**: Modern UI consuming Flask APIs
- **No more dual-mode confusion** - each service has one responsibility

The UI has been completely refactored to use:
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for modern, responsive styling
- **React Router** for client-side navigation
- **Axios** for API communication
- **React Hot Toast** for notifications

## Key Changes

### ✅ Completed Features

1. **Modern Design System**
   - Removed retro gaming themes (VT323 font, neon colors, scanlines)
   - Clean, professional color palette
   - Responsive design that works on mobile devices
   - Consistent component library

2. **React Architecture**
   - TypeScript for type safety
   - Component-based architecture
   - Client-side routing
   - State management with React hooks

3. **Backend Integration**
   - CORS-enabled API endpoints
   - JSON-based communication
   - Optimized for client-side rendering

4. **Pages Implemented**
   - **Home/Dashboard**: Search interface and recent documents
   - **Documents**: Full listing with pagination, search, and filtering
   - **Document Detail**: Complete document view with metadata
   - **Statistics**: Data visualization and analytics
   - **404 Page**: Error handling

5. **Developer Features**
   - **Verbose Logging Toggle**: Real-time control of backend logging levels
   - **Database Refresh**: Manual cache invalidation
   - **API Debugging**: Direct access to backend endpoints

### 🎨 Design System

#### Colors
- **Primary**: Blue tones for interactive elements
- **Secondary**: Gray tones for text and backgrounds
- **Success/Warning/Danger**: Semantic colors for feedback
- **Neutral**: Clean grays for content

#### Typography
- **Inter** for body text and UI elements
- **JetBrains Mono** for code and technical content
- Responsive text sizing

#### Components
- **Button**: Multiple variants (primary, secondary, ghost, outline)
- **Card**: Content containers with shadows
- **Input**: Form inputs with focus states
- **LoadingSpinner**: Consistent loading states

### 📱 Responsive Design

All components are fully responsive:
- **Mobile-first** approach
- **Breakpoint-based** layouts (sm, md, lg, xl)
- **Touch-friendly** interactions
- **Optimized** for phones and tablets

### 🔊 Verbose Logging Control

Real-time logging level management:
- **Toggle Button**: Volume icon in header (🔊 on, 🔇 off)
- **Visual Feedback**: Yellow highlight when verbose is enabled
- **API Integration**: `/api/verbose` endpoint for state management
- **Persistent**: Maintains state during session
- **User-Friendly**: Toast notifications for status changes

## Project Structure

```
src/
├── components/
│   ├── ui/                 # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Card.tsx
│   │   ├── Input.tsx
│   │   └── LoadingSpinner.tsx
│   └── Layout.tsx          # Main layout with navigation
├── lib/
│   ├── api.ts             # API client and utilities
│   └── utils.ts           # Utility functions
├── pages/                 # Page components
│   ├── HomePage.tsx
│   ├── DocumentsPage.tsx
│   ├── DocumentDetailPage.tsx
│   ├── StatsPage.tsx
│   └── NotFoundPage.tsx
├── types/                 # TypeScript type definitions
│   └── index.ts
├── App.tsx                # Main app component
├── main.tsx               # App entry point
└── index.css              # Global styles
```

## API Integration

The React frontend communicates with the Flask backend via RESTful APIs:

### Key Endpoints Used:
- `GET /api/documents` - List documents with pagination
- `GET /api/document/:id` - Get document details
- `POST /api/search` - Semantic search
- `GET /api/stats` - Statistics data
- `GET/POST /api/verbose` - Get/set verbose logging state
- `GET /document/:id/file` - Download files

### Features:
- **Error handling** with user-friendly messages
- **Loading states** for all async operations
- **Toast notifications** for user feedback
- **Automatic retries** for failed requests

## Development Setup

### Prerequisites
- Node.js 18+
- npm or yarn
- Python 3.8+ (for Flask backend)

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```
   The React app will run on `http://localhost:5173`

3. **Build for production:**
   ```bash
   npm run build
   ```

### Backend Setup

The Flask backend needs to be updated to serve the React app:

1. **Install Flask-CORS:**
   ```bash
   pip install flask-cors
   ```

2. **Run the Flask server:**
   ```bash
   python app.py
   ```
   The backend will run on `http://localhost:5000`

## Deployment

### Option 1: Serve React Build from Flask

1. Build the React app:
   ```bash
   npm run build
   ```

2. The `dist/` folder will contain the built files

3. Flask will automatically serve the React app from the root path

### Option 2: Separate Frontend Deployment

Deploy the React app separately and configure CORS on the backend.

## Remaining Tasks

### 🔄 In Progress / To Do

1. **Advanced Search Features**
   - Real-time search suggestions
   - Category-based filtering in search
   - Search history

2. **File Viewer Components**
   - PDF viewer integration
   - Image viewer
   - Text file viewer
   - Syntax highlighting for code

3. **Theme System**
   - Light/dark mode toggle
   - User preference persistence
   - Custom theme support

4. **Testing & Polish**
   - Unit tests for components
   - E2E testing with Cypress
   - Performance optimization
   - Accessibility improvements

## Migration Guide

### For Existing Users

1. **No data migration needed** - all document data remains in SQLite
2. **API endpoints preserved** - existing functionality works
3. **URL structure maintained** - bookmarks will continue to work

### For Developers

1. **Flask templates removed** - all UI is now in React
2. **API-first architecture** - backend focuses on data, frontend handles presentation
3. **TypeScript** - better development experience and fewer bugs

## Benefits of the Refactor

### Performance
- **Faster initial loads** with client-side rendering
- **Better caching** of static assets
- **Reduced server load** - Flask only serves APIs

### User Experience
- **Responsive design** works on all devices
- **Modern UI** that's clean and professional
- **Faster interactions** with client-side navigation
- **Better feedback** with loading states and notifications

### Developer Experience
- **Type safety** with TypeScript
- **Hot reloading** during development
- **Component reusability** across the application
- **Modern tooling** with Vite and Tailwind

### Maintainability
- **Separation of concerns** - UI and API are separate
- **Easier testing** - components can be tested in isolation
- **Scalable architecture** - easy to add new features
- **Future-proof** - React ecosystem provides long-term support

## Troubleshooting

### Common Issues

1. **CORS Errors**
   - Ensure Flask-CORS is installed and configured
   - Check that the React dev server is running on an allowed origin

2. **API Connection Issues**
   - Verify Flask backend is running on port 5000
   - Check network/firewall settings

3. **Build Issues**
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility

4. **TypeScript Errors**
   - Run `npm run type-check` to see detailed errors
   - Check import paths and type definitions

### Getting Help

- Check the browser console for JavaScript errors
- Verify API responses in browser network tab
- Test API endpoints directly with curl/Postman

## Troubleshooting

### Build Issues

If `./build_frontend.sh` fails with npm permission errors:

1. **Fix npm permissions:**
   ```bash
   sudo chown -R $(whoami) ~/.npm
   ```

2. **Manual build process:**
   ```bash
   # Install dependencies
   npm install

   # Run type check
   npm run type-check

   # Build for production
   npm run build
   ```

3. **If you get EPERM errors:**
   - Try using a Node version manager like `nvm`
   - Or run with elevated permissions: `sudo npm install`

### Common Issues

1. **CORS Errors**
   - Ensure Flask-CORS is installed: `pip install flask-cors`
   - Check that React dev server proxy is configured correctly

2. **API Connection Issues**
   - Verify Flask backend is running on port 5000
   - Check API endpoints are accessible

3. **TypeScript Errors**
   - Run `npm run type-check` to see detailed errors
   - Ensure all imports are correct

4. **Build Fails**
   - Clear node_modules: `rm -rf node_modules && npm install`
   - Check Node.js version (18+ required)

### Development Alternatives

If npm installation fails completely, you can:

1. **Use a different package manager:**
   ```bash
   yarn install
   yarn build
   ```

2. **Use a containerized environment:**
   ```bash
   docker run -it -v $(pwd):/app -w /app node:18 npm install
   ```

3. **Skip frontend build temporarily:**
   - The Flask backend will still work with existing functionality
   - Use the original Jinja2 templates as fallback

## Future Enhancements

1. **Progressive Web App (PWA)** - offline functionality
2. **Real-time updates** - WebSocket integration for live data
3. **Advanced analytics** - user behavior tracking
4. **Multi-language support** - internationalization
5. **Accessibility** - WCAG compliance
6. **Performance monitoring** - error tracking and analytics
