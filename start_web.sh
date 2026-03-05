#!/bin/bash

# AI Fitness Coach - Web Interface Startup Script

echo "=========================================="
echo "💪 AI Fitness Coach"
echo "=========================================="
echo ""

# Check if API server is running
echo "📡 Checking API server status..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API server is running"
else
    echo "❌ API server not running"
    echo "🚀 Starting API server..."
    python src/serving/api_server.py &
    API_PID=$!
    echo "Waiting for API server to start..."
    sleep 5

    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ API server started successfully (PID: $API_PID)"
    else
        echo "❌ Failed to start API server, please check logs"
        exit 1
    fi
fi

echo ""
echo "🌐 Starting Web Interface..."
echo ""

# Start Streamlit
streamlit run web_app_en.py --server.headless true --server.port 8501

echo ""
echo "=========================================="
echo "Web interface closed"
echo "=========================================="
