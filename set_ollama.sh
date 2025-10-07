#!/bin/bash

# 🦙 Ollama Setup Script for LangGraph Q&A Assistant
# This script helps you get started quickly with Ollama

set -e  # Exit on error

echo "=================================================="
echo "🦙 LangGraph Q&A Assistant - Ollama Setup"
echo "=================================================="
echo ""

# Check if Ollama is installed
echo "📋 Step 1/5: Checking for Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama is not installed!"
    echo ""
    echo "Please install Ollama first:"
    echo "  - Visit: https://ollama.com"
    echo "  - Or run: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    exit 1
else
    echo "✅ Ollama is installed: $(ollama --version)"
fi

# Check if Ollama is running
echo ""
echo "📋 Step 2/5: Checking if Ollama is running..."
if ! curl -s http://localhost:11434 > /dev/null 2>&1; then
    echo "⚠️ Ollama is not running!"
    echo "Starting Ollama server..."
    ollama serve > /dev/null 2>&1 &
    sleep 3
    
    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        echo "✅ Ollama server started successfully"
    else
        echo "❌ Failed to start Ollama server"
        echo "Please start it manually: ollama serve"
        exit 1
    fi
else
    echo "✅ Ollama is running"
fi

# Check for models
echo ""
echo "📋 Step 3/5: Checking for qwen3:4b model..."
if ollama list | grep -q "qwen3:4b"; then
    echo "✅ qwen3:4b is already downloaded"
else
    echo "⚠️ qwen3:4b not found. Downloading..."
    echo "This may take a few minutes (2.5 GB download)..."
    ollama pull qwen3:4b
    echo "✅ qwen3:4b downloaded successfully"
fi

# Install Python dependencies
echo ""
echo "📋 Step 4/5: Installing Python dependencies..."
if command -v pip &> /dev/null; then
    pip install -q -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "❌ pip not found! Please install Python and pip first."
    exit 1
fi

# Create .env file
echo ""
echo "📋 Step 5/5: Setting up environment configuration..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# LLM Configuration - Using Ollama (Local & Free)
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:4b
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Tavily API for web search
# Get free API key at: https://tavily.com
# TAVILY_API_KEY=your-tavily-key-here

# Optional: LangSmith for tracing
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-langsmith-key-here
# LANGCHAIN_PROJECT=LangGraph-QA-Assistant
EOF
    echo "✅ Created .env file with Ollama configuration"
else
    echo "⚠️ .env file already exists, skipping..."
fi

echo ""
echo "=================================================="
echo "✅ Setup Complete!"
echo "=================================================="
echo ""
echo "🚀 You're ready to go! Run the assistant with:"
echo ""
echo "    python main.py"
echo ""
echo "📚 Quick Tips:"
echo "  - Try different models: ollama pull llama3.1:8b"
echo "  - Update OLLAMA_MODEL in .env to switch models"
echo "  - Add TAVILY_API_KEY to enable web search"
echo ""
echo "📖 Documentation:"
echo "  - See OLLAMA_QUICKSTART.md for detailed guide"
echo "  - See LLM_PROVIDER_COMPARISON.md to compare with OpenAI"
echo ""
echo "Happy chatting! 🎉"
echo ""