#!/bin/bash
# PatternRAG Setup Script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Print welcome message
echo "=========================================="
echo "   PatternRAG Setup Script"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"
echo "Checking Python version..."
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "Python version $python_version is OK (>= $required_version)"
else
    echo "Error: Python version must be at least $required_version (found $python_version)"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/venv"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$PROJECT_ROOT/venv/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_ROOT/requirements.txt"

# Install spaCy model
echo "Installing spaCy model..."
if python -m spacy download en_core_web_sm; then
    echo "Installed spaCy model successfully"
else
    echo "Warning: Could not install spaCy model automatically"
    echo "Please run manually: python -m spacy download en_core_web_sm"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p "$PROJECT_ROOT/data/db"
mkdir -p "$PROJECT_ROOT/data/metadata"
mkdir -p "$PROJECT_ROOT/data/graph"
mkdir -p "$PROJECT_ROOT/documents"
mkdir -p "$PROJECT_ROOT/logs"

# Create configuration file if it doesn't exist
if [ ! -f "$PROJECT_ROOT/config/config.yaml" ]; then
    echo "Creating configuration file..."
    mkdir -p "$PROJECT_ROOT/config"
    cp "$PROJECT_ROOT/config/default_config.yaml" "$PROJECT_ROOT/config/config.yaml"
    echo "Configuration file created at config/config.yaml"
    echo "Please review and adjust settings as needed"
else
    echo "Configuration file already exists"
fi

# Check if Ollama is installed
echo "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "Ollama is installed"
    echo "You can pull a model with: ollama pull llama2"
else
    echo "Ollama not found"
    echo "If you want to use Ollama as your LLM provider,"
    echo "please install it from: https://ollama.ai"
    echo "You can also configure another LLM provider in config/config.yaml"
fi

echo ""
echo "=========================================="
echo "PatternRAG setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To process documents, run:"
echo "  python -m patternrag.ingest"
echo ""
echo "To start the API service, run:"
echo "  python -m patternrag.service"
echo ""
echo "For more information, see the documentation in the 'docs' directory"
