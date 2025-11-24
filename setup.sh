#!/bin/bash

# Document Classification Agent Setup Script
# This script uses uv to create a virtual environment and install dependencies

set -e  # Exit on error

echo "🚀 Setting up Document Classification Agent..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed."
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  or visit: https://github.com/astral-sh/uv"
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment using uv
echo "📦 Creating virtual environment..."
uv venv

# Determine the activation script path based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    VENV_ACTIVATE=".venv/Scripts/activate"
else
    VENV_ACTIVATE=".venv/bin/activate"
fi

source $VENV_ACTIVATE

# Install dependencies using uv
echo "📥 Installing dependencies..."
uv pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "  source .venv/Scripts/activate"
else
    echo "  source .venv/bin/activate"
fi
echo ""
echo "Then you can run the agent:"
echo "  python main.py classify"
echo "  python main.py watch"
echo ""

