# Quick Start Guide

## Prerequisites

1. Python 3.8+
2. OpenAI API Key (configured in `.env` file)
3. Required packages installed: `pip install -r requirements.txt`

## Running the Application

### Option 1: Quick Start (Recommended)

```bash
./start_web.sh
```

This script will:
- Check if API server is running
- Start API server if needed
- Launch web interface

### Option 2: Manual Start

**Terminal 1 - Start API Server:**
```bash
python src/serving/api_server.py
```

**Terminal 2 - Start Web Interface:**
```bash
streamlit run web_app_en.py
```

### Option 3: Background Mode

```bash
# Start API server in background
python src/serving/api_server.py &

# Start web interface in background
streamlit run web_app_en.py --server.headless true --server.port 8501 &
```

## Accessing the Application

Open your browser and go to:
```
http://localhost:8501
```

## Main Features

### 1. Get Recommendation
- Input your body state data (readiness, sleep, HRV, etc.)
- Receive personalized training recommendations
- View safety checks and rationale

### 2. Submit Feedback
- Rate your completed workouts
- Provide RPE, mood, and satisfaction scores
- Help the system learn and improve

### 3. AI Coach Chat (NEW!)
- Ask questions about your training plan
- Get motivation and guidance
- Request plan adjustments
- Receive personalized coaching

### 4. Data Analysis
- View historical recommendations
- Analyze body state trends
- Download data for further analysis

## System Status

Check if services are running:

```bash
# Check API server
curl http://localhost:8000/health

# Check web interface
curl http://localhost:8501/_stcore/health
```

## Stopping Services

```bash
# Stop all services
pkill -f api_server
pkill -f streamlit
```

## Troubleshooting

**API server not responding:**
```bash
# Check if running
ps aux | grep api_server

# Restart
pkill -f api_server && python src/serving/api_server.py &
```

**Web interface not loading:**
```bash
# Check if running
ps aux | grep streamlit

# Restart
pkill -f streamlit && streamlit run web_app_en.py &
```

## Next Steps

- Read [README.md](README.md) for complete project documentation
- Check [DATA_VIEWING_AND_IOS_INTEGRATION.md](DATA_VIEWING_AND_IOS_INTEGRATION.md) for data viewing and iOS app integration guide
- Explore the web interface to get personalized recommendations!
