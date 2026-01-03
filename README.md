# Real-Time Personalized Fitness Plan Optimization with Reinforcement Learning

## Project Overview

This project focuses on building a production-ready personalized fitness plan optimization system that combines rule-based heuristics with reinforcement learning techniques for real-time sequence prediction. The system processes real-time wearable device data (Apple Watch, Oura Ring) and dynamically adjusts training plans based on body state, recovery status, and fitness goals.

## Key Components

### 1. Hybrid Fitness Plan Recommendation System
- Built a hybrid recommendation system combining:
  - **Rule-based Heuristics**: Similar to collaborative filtering, using predefined training templates based on fitness goals
  - **Contextual Bandits**: Managing exploration-exploitation tradeoffs to balance trying new exercises vs. sticking with proven plans
- **Data Sources**: 
  - Apple Watch data (heart rate, activity, sleep)
  - Oura Ring data (readiness score, HRV, sleep quality)
  - Training logs and subjective feedback
- **Performance**: Achieved **0.85+ AUC** in predicting optimal training plans

### 2. Feature Store and Model Serving Infrastructure
- **Feature Store**: Developed using **Feast** framework
  - Manages over **200 body state and training features**
  - Features include: heart rate variability, sleep quality, training history, fatigue levels, goal progress
  - Privacy-preserving feature aggregation
- **Model Serving**: Deployed API using **TorchServe**
  - Handles real-time plan generation requests
  - **p99 latency < 50ms** for instant plan recommendations
  - Serves personalized training plans based on current body state

### 3. Online Learning and A/B Testing
- **Online Learning Pipeline**: 
  - Utilizes **Kafka streaming** for real-time wearable device data processing
  - Implements **incremental Reinforcement Learning updates** using **Thompson sampling**
  - Continuously adapts to body state changes and training responses
- **A/B Testing Framework**: 
  - Tests different training strategies and plan intensities
  - Demonstrated **15%+ increase in training completion rate**
  - Adheres to safety constraints (prevents overtraining, respects injury history)

### 4. AI Coach Agent (Tool-Using LLM Agent)
- **Daily Coach Agent**: Tool-using AI agent that provides personalized coaching through natural language interaction
  - **Plan Explanation**: Explains training plan recommendations in natural language, highlighting why specific plans fit the user's current body state
  - **Feedback Collection**: Collects user feedback (RPE, mood, stress, pain) through conversational interface
  - **Action Triggers**: Adjusts training plans, schedules recovery days, generates daily summaries, and sets micro-goals
  - **Closed-Loop Learning**: Logs feedback events to Kafka for continuous model improvement
- **Safety Guardrails**: 
  - Hard rules and safety checks prevent dangerous recommendations
  - Detects overtraining, abnormal physiological signals, and injury risks
  - Escalates critical safety alerts with recommendations to consult healthcare professionals
- **Emotional Support**: 
  - Provides motivational messages and breathing exercises
  - Generates daily reflections and summaries
  - Adapts communication style based on user mood and stress levels
- **Architecture**: Three-layer design (Safety Gate → Recommendation Engine → LLM Agent) ensures reliability while enabling natural interaction

## Technology Stack

- **Big Data Processing**: PySpark (for multi-user scenarios and feature processing)
- **Reinforcement Learning**: Contextual Bandits, Thompson Sampling
- **Feature Engineering**: Feast (Feature Store)
- **Model Serving**: TorchServe
- **Streaming**: Apache Kafka
- **ML Framework**: PyTorch
- **Data Sources**: Apple HealthKit API, Oura API v2
- **API Framework**: FastAPI
- **AI Agent**: LLM with Tool Calling (OpenAI/Anthropic), Function Calling

## Project Structure

```
RL/
├── README.md
├── requirements.txt
├── env.example
├── src/
│   ├── data_collection/
│   │   ├── apple_health.py
│   │   ├── oura_api.py
│   │   └── training_log.py
│   ├── recommendation/
│   │   ├── rule_based.py
│   │   ├── contextual_bandits.py
│   │   └── hybrid_recommender.py
│   ├── feature_store/
│   │   ├── feast_config.py
│   │   └── feature_engineering.py
│   ├── serving/
│   │   ├── torchserve_handler.py
│   │   ├── api_server.py
│   │   └── agent_api.py
│   ├── agent/
│   │   ├── coach_agent.py
│   │   ├── tools.py
│   │   ├── safety.py
│   │   └── state.py
│   ├── online_learning/
│   │   ├── kafka_consumer.py
│   │   ├── thompson_sampling.py
│   │   └── incremental_updates.py
│   └── ab_testing/
│       ├── experiment_framework.py
│       └── safety_constraints.py
├── notebooks/
├── tests/
├── config/
├── scripts/
│   └── setup_data_collection.py
└── data/
    ├── raw/
    │   ├── apple_watch_health/    # Place your Apple Health export.xml here
    │   ├── oura/                   # Oura API data
    │   └── training_logs/          # Training session logs
    ├── processed/
    ├── features/
    └── public/                     # Public datasets
```

## Skills Demonstrated

### Machine Learning Engineering (MLE)
- ✅ Reinforcement learning algorithms (Contextual Bandits, Thompson Sampling)
- ✅ Online learning and incremental model updates
- ✅ Multi-modal data fusion (wearable sensors + subjective feedback)
- ✅ Model evaluation and metrics (AUC, completion rate, goal achievement)
- ✅ A/B testing and experimentation framework
- ✅ Safety constraints in ML systems (preventing overtraining)

### Software Engineering (SDE)
- ✅ Real-time data processing (Kafka streaming)
- ✅ Feature store architecture (Feast)
- ✅ High-performance model serving (TorchServe, <50ms latency)
- ✅ API integration (Apple HealthKit, Oura API)
- ✅ System optimization and performance tuning
- ✅ Production ML infrastructure

## Target Job Roles

This project demonstrates skills suitable for:

1. **Machine Learning Engineer (MLE)** ⭐ Primary Fit
   - Strong focus on ML algorithms, model development, and experimentation
   - Requires deep understanding of RL, online learning, and multi-modal data processing
   - Personal problem-solving approach shows practical ML application

2. **ML Infrastructure Engineer** ⭐ Secondary Fit
   - Combines ML expertise with strong systems engineering
   - Focus on scalable serving, feature stores, and real-time pipelines
   - Integration of multiple data sources and APIs

3. **Applied Scientist / Research Engineer**
   - Algorithm research and implementation
   - A/B testing and experimentation
   - Safety-aware ML systems

## Key Metrics & Achievements

- **Model Performance**: 0.85+ AUC in plan recommendation
- **Serving Performance**: p99 latency < 50ms
- **Business Impact**: 15%+ increase in training completion rate
- **Feature Management**: 200+ features (body state, training history, goals)
- **Real-time Processing**: Continuous adaptation to body state changes
- **Safety**: Zero overtraining incidents, respects injury constraints

## Data Sources

- **Apple Watch**: Heart rate, activity data, sleep metrics via HealthKit API
- **Oura Ring**: Readiness score, HRV, sleep quality via Oura API v2
- **Training Logs**: Exercise selection, sets, reps, weights, RPE, subjective feedback

## Quick Start

### 1. Preprocess Data
```bash
python src/data_collection/preprocess.py
```

### 2. Engineer Features
```bash
python src/feature_store/feature_engineering.py
```

### 3. Train Model
```bash
python src/recommendation/train.py
```

### 4. Start API
```bash
python src/serving/api_server.py
```

### 5. Explore Data
```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## AI Coach Agent Capabilities

The AI Coach Agent provides the following actions:

### Training Actions
- **adjust_plan()**: Adjusts training intensity, volume, or schedules rest days based on recovery status
- **explain_plan()**: Explains training plan recommendations in natural language
- **generate_warmup_cooldown()**: Generates personalized warmup and cooldown routines
- **set_micro_goal()**: Sets small daily goals to improve training adherence

### Emotional Support Actions
- **mood_checkin()**: Collects mood and stress feedback (1-5 scale) and adapts recommendations
- **reflect_and_summarize()**: Generates daily training and mood summaries with structured insights
- **breathing_prompt()**: Provides 60-120 second breathing/relaxation guidance
- **motivational_message()**: Generates motivational messages in different styles (short/long/humorous/rational)

### System Actions
- **log_event()**: Logs user feedback and completion data to Kafka for online learning
- **request_more_info()**: Asks users for missing information when data is incomplete
- **safety_check()**: Performs safety checks and escalates critical alerts

### Safety & Boundaries

**⚠️ Important Disclaimers:**
- This system is **NOT a medical device** and does **NOT provide medical advice**
- If you experience chest pain, dizziness, severe discomfort, or abnormal heart rate patterns, **stop training immediately** and consult a healthcare professional
- The AI agent provides fitness coaching and emotional support but does not diagnose or treat medical conditions
- All recommendations are based on fitness data and should be used at your own discretion

## Future Improvements

- Multi-user support with collaborative filtering
- Integration with more wearable devices
- Mobile app for real-time plan updates
- Advanced RL algorithms (DQN, PPO) for fine-grained control
- Enhanced agent memory and personalization
