#!/bin/bash

# Script to run Streamlit Dashboard
# Make sure the FastAPI service is running first on port 8000

echo "🚀 Starting Streamlit Dashboard..."
echo "Make sure your FastAPI service is running on http://localhost:8000"
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing..."
    pip install streamlit==1.40.2
fi

# Run streamlit
streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=localhost \
    --browser.gatherUsageStats=false

echo ""
echo "✅ Streamlit dashboard stopped"
