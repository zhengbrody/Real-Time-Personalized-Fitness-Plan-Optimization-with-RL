"""
Streamlit Web Interface for Personalized Fitness Plan Optimizer (English Version)
"""

import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
# Get the directory where this script is located
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize OpenAI client for AI Coach
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")

    # Validate API key format (should start with 'sk-')
    if not api_key.startswith('sk-'):
        raise ValueError("Invalid OPENAI_API_KEY format")

    openai_client = OpenAI(api_key=api_key)
    AI_COACH_ENABLED = True
    print(f"✅ AI Coach enabled with API key: {api_key[:15]}...")
except Exception as e:
    AI_COACH_ENABLED = False
    print(f"❌ AI Coach disabled: {e}")

# Page configuration
st.set_page_config(
    page_title="AI Fitness Coach",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
    }
    .agent-message {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #6c757d;
        margin: 0.5rem 0;
        color: #212529;
        font-weight: 500;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 0.5rem 0;
        text-align: right;
        color: #1565c0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'user_001'
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []
if 'current_recommendation' not in st.session_state:
    st.session_state.current_recommendation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<div class="main-header">💪 AI Fitness Coach</div>', unsafe_allow_html=True)
st.markdown("### Personalized Training Plan Optimizer with Reinforcement Learning")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    # User ID input
    user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id,
        help="Enter your user ID"
    )
    st.session_state.user_id = user_id

    st.markdown("---")

    # API Status Check
    st.subheader("📡 System Status")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API Service Online")
            with st.expander("View Details"):
                st.json(health_data)
        else:
            st.error("❌ API Service Error")
    except Exception as e:
        st.error(f"❌ Cannot connect to API\n{str(e)}")

    st.markdown("---")

    # Information
    st.subheader("ℹ️ About")
    st.info("""
    **AI Fitness Coach**

    Personalized fitness plan recommendation system powered by reinforcement learning.

    - Real-time body state analysis
    - Intelligent training recommendations
    - Online learning optimization
    - Safety protection mechanisms
    """)

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Get Recommendation",
    "💬 Submit Feedback",
    "🤖 AI Coach Chat",
    "📈 Data Analysis",
    "📖 User Guide"
])

# Tab 1: Get Recommendation
with tab1:
    st.header("📊 Get Today's Training Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏃 Body State Data")

        # Input form
        with st.form("body_state_form"):
            readiness_score = st.slider(
                "Readiness Score",
                min_value=0,
                max_value=100,
                value=75,
                help="Overall readiness state score (0-100)"
            )

            sleep_score = st.slider(
                "Sleep Score",
                min_value=0,
                max_value=100,
                value=80,
                help="Sleep quality score (0-100)"
            )

            hrv = st.slider(
                "Heart Rate Variability (HRV)",
                min_value=20,
                max_value=100,
                value=50,
                help="Higher values indicate better recovery"
            )

            resting_hr = st.slider(
                "Resting Heart Rate",
                min_value=40,
                max_value=100,
                value=60,
                help="Resting heart rate (bpm)"
            )

            fatigue = st.slider(
                "Fatigue Level",
                min_value=1,
                max_value=10,
                value=5,
                help="Subjective fatigue feeling (1=no fatigue, 10=extremely fatigued)"
            )

            activity_score = st.slider(
                "Activity Score",
                min_value=0,
                max_value=100,
                value=70,
                help="Recent activity level score (0-100)"
            )

            submit_button = st.form_submit_button("🎯 Get Recommendation", use_container_width=True)

    with col2:
        st.subheader("💡 Recommendation Result")

        if submit_button:
            # Prepare request data
            request_data = {
                "user_id": st.session_state.user_id,
                "state": {
                    "readiness_score": readiness_score,
                    "sleep_score": sleep_score,
                    "hrv": hrv,
                    "resting_hr": resting_hr,
                    "fatigue": fatigue,
                    "activity_score": activity_score
                }
            }

            # Make API request
            try:
                with st.spinner("Generating recommendation..."):
                    response = requests.post(
                        f"{API_BASE_URL}/recommend",
                        json=request_data,
                        timeout=10
                    )

                if response.status_code == 200:
                    recommendation = response.json()
                    st.session_state.current_recommendation = recommendation

                    # Add to history
                    st.session_state.recommendation_history.append({
                        'timestamp': datetime.now(),
                        'recommendation': recommendation,
                        'state': request_data['state']
                    })

                    # Display recommendation
                    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)

                    # Workout type
                    workout_type = recommendation['workout_type']
                    intensity = recommendation['intensity']
                    duration = recommendation['duration_minutes']

                    if workout_type == "REST":
                        st.success("🛌 **Rest Day Recommended**")
                    else:
                        st.info(f"🏋️ **Workout Type**: {workout_type}")
                        st.info(f"⚡ **Intensity**: {intensity}")
                        st.info(f"⏱️ **Duration**: {duration} minutes")

                    # Description
                    st.markdown(f"**Description**: {recommendation['description']}")

                    # Rationale
                    st.markdown(f"**Rationale**: {recommendation['rationale']}")

                    # Safety check
                    safety = recommendation['safety_check']
                    if safety['is_safe']:
                        st.markdown('<div class="success-box">✅ Safety Check Passed</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="warning-box">⚠️ {safety["message"]}</div>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # Store action_id for feedback
                    st.session_state.last_action_id = recommendation['action_id']

                else:
                    st.error(f"❌ Failed to get recommendation: {response.status_code}")
                    st.error(response.text)

            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")

        elif st.session_state.current_recommendation:
            # Display last recommendation
            recommendation = st.session_state.current_recommendation

            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)

            workout_type = recommendation['workout_type']
            intensity = recommendation['intensity']
            duration = recommendation['duration_minutes']

            if workout_type == "REST":
                st.success("🛌 **Rest Day Recommended**")
            else:
                st.info(f"🏋️ **Workout Type**: {workout_type}")
                st.info(f"⚡ **Intensity**: {intensity}")
                st.info(f"⏱️ **Duration**: {duration} minutes")

            st.markdown(f"**Description**: {recommendation['description']}")
            st.markdown(f"**Rationale**: {recommendation['rationale']}")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Fill in the form on the left and click 'Get Recommendation' button")

# Tab 2: Submit Feedback
with tab2:
    st.header("💬 Submit Training Feedback")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Feedback Form")

        with st.form("feedback_form"):
            action_id = st.number_input(
                "Training Plan ID",
                min_value=0,
                max_value=20,
                value=st.session_state.get('last_action_id', 5),
                help="Recommended training plan ID"
            )

            completed = st.checkbox(
                "✅ Training Completed",
                value=True,
                help="Did you complete the recommended training plan?"
            )

            rpe = st.slider(
                "RPE (Rating of Perceived Exertion)",
                min_value=1,
                max_value=10,
                value=7,
                help="Post-training perceived exertion (1=very easy, 10=extremely hard)"
            )

            mood = st.slider(
                "Mood Rating",
                min_value=1,
                max_value=5,
                value=4,
                help="Post-training mood (1=very bad, 5=very good)"
            )

            satisfaction = st.slider(
                "Satisfaction",
                min_value=1,
                max_value=10,
                value=8,
                help="Training plan satisfaction (1=very unsatisfied, 10=very satisfied)"
            )

            feedback_notes = st.text_area(
                "Notes",
                placeholder="Optional: Add additional feedback...",
                help="Any additional feedback or feelings"
            )

            submit_feedback = st.form_submit_button("📤 Submit Feedback", use_container_width=True)

    with col2:
        st.subheader("📊 Feedback Result")

        if submit_feedback:
            # Prepare feedback data
            feedback_data = {
                "user_id": st.session_state.user_id,
                "action_id": action_id,
                "feedback": {
                    "completed": completed,
                    "rpe": rpe,
                    "mood": mood,
                    "satisfaction": satisfaction
                }
            }

            # Make API request
            try:
                with st.spinner("Submitting feedback..."):
                    response = requests.post(
                        f"{API_BASE_URL}/feedback",
                        json=feedback_data,
                        timeout=10
                    )

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Feedback submitted successfully!")

                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.metric("Reward Score", f"{result['reward']:.2f}")
                    st.markdown("Your feedback has been recorded and will be used to optimize future recommendations")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Display feedback summary
                    st.subheader("Feedback Summary")
                    feedback_df = pd.DataFrame([{
                        "Status": "✅ Completed" if completed else "❌ Not Completed",
                        "RPE": rpe,
                        "Mood": mood,
                        "Satisfaction": satisfaction,
                        "Reward": f"{result['reward']:.2f}"
                    }])
                    st.table(feedback_df)

                    if feedback_notes:
                        st.info(f"**Notes**: {feedback_notes}")

                else:
                    st.error(f"❌ Submission failed: {response.status_code}")
                    st.error(response.text)

            except Exception as e:
                st.error(f"❌ Request failed: {str(e)}")
        else:
            st.info("👈 Fill in the feedback form and click 'Submit Feedback' button")

            # Show feedback guide
            with st.expander("💡 Feedback Guide"):
                st.markdown("""
                **RPE (Rating of Perceived Exertion)**
                - 1-3: Very easy
                - 4-6: Moderate intensity
                - 7-8: Challenging
                - 9-10: Extremely hard

                **Mood Rating**
                - 1: Very bad
                - 2: Bad
                - 3: Neutral
                - 4: Good
                - 5: Very good

                **Satisfaction**
                - 1-3: Unsatisfied
                - 4-6: Neutral
                - 7-8: Satisfied
                - 9-10: Very satisfied
                """)

# Tab 3: AI Coach Chat
with tab3:
    st.header("🤖 AI Fitness Coach")
    st.markdown("Chat with your personal AI coach for guidance, motivation, and plan adjustments")

    # Display current health data that AI Coach can see
    if st.session_state.recommendation_history:
        with st.expander("📊 Your Current Health Data (AI Coach Context)", expanded=False):
            latest = st.session_state.recommendation_history[-1]
            state = latest.get('state', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Readiness Score", f"{state.get('readiness_score', 'N/A')}/100")
                st.metric("Sleep Score", f"{state.get('sleep_score', 'N/A')}/100")
            with col2:
                st.metric("HRV", f"{state.get('hrv', 'N/A')} ms")
                st.metric("Resting HR", f"{state.get('resting_hr', 'N/A')} bpm")
            with col3:
                st.metric("Fatigue Level", f"{state.get('fatigue', 'N/A')}/10")
                st.metric("Activity Score", f"{state.get('activity_score', 'N/A')}/100")

            st.info("💡 AI Coach uses this data to provide personalized guidance. Get a recommendation first to populate this data.")
    else:
        st.info("💡 **Tip**: Get a recommendation first in the 'Get Recommendation' tab. This will populate your health data for the AI Coach to analyze.")

    # Chat interface
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">👤 **You**: {message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-message">🤖 **AI Coach**: {message["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    st.markdown("---")

    col1, col2 = st.columns([4, 1])

    with col1:
        user_message = st.text_input(
            "Message",
            placeholder="Ask your coach anything... (e.g., 'Explain my plan', 'I'm feeling tired', 'Motivate me')",
            key="chat_input",
            label_visibility="collapsed"
        )

    with col2:
        send_button = st.button("📤 Send", use_container_width=True)

    if send_button and user_message:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now()
        })

        # Prepare agent request
        agent_request = {
            "user_id": st.session_state.user_id,
            "message": user_message,
            "context": {
                "current_recommendation": st.session_state.current_recommendation,
                "recent_feedback": st.session_state.recommendation_history[-3:] if st.session_state.recommendation_history else []
            }
        }

        # Call AI Coach with real health data
        try:
            with st.spinner("AI Coach is thinking..."):
                if AI_COACH_ENABLED:
                    # Get current body state from last recommendation or form
                    body_state = {}
                    if st.session_state.recommendation_history:
                        latest = st.session_state.recommendation_history[-1]
                        body_state = latest.get('state', {})

                    # Build context for AI Coach
                    system_prompt = """You are an expert AI Fitness Coach with access to the user's real-time health and fitness data.

Your role:
- Analyze user's body state data (readiness score, sleep quality, HRV, heart rate, fatigue levels, activity scores)
- Provide personalized training guidance based on their current physiological state
- Explain training recommendations with data-driven reasoning
- Offer motivation and emotional support
- Help adjust plans based on how the user feels

IMPORTANT BOUNDARIES:
- You are NOT a medical professional. Never diagnose or treat medical conditions.
- If user reports chest pain, severe dizziness, or abnormal symptoms, immediately recommend consulting a doctor.
- Focus on fitness coaching, recovery optimization, and performance improvement.
- Be supportive, conversational, and data-driven.

Current Health Data Available:
- Readiness Score: Overall body readiness (0-100)
- Sleep Score: Sleep quality (0-100)
- HRV: Heart Rate Variability (higher = better recovery)
- Resting Heart Rate: Morning resting HR
- Fatigue Level: Subjective fatigue (1-10, higher = more tired)
- Activity Score: Recent activity level (0-100)

Use this data to provide personalized, context-aware coaching."""

                    # Build user context message
                    context_parts = [f"User: {user_message}"]

                    if body_state:
                        context_parts.append("\n\n**Current Health Data:**")
                        context_parts.append(f"- Readiness Score: {body_state.get('readiness_score', 'N/A')}/100")
                        context_parts.append(f"- Sleep Score: {body_state.get('sleep_score', 'N/A')}/100")
                        context_parts.append(f"- HRV: {body_state.get('hrv', 'N/A')} ms")
                        context_parts.append(f"- Resting HR: {body_state.get('resting_hr', 'N/A')} bpm")
                        context_parts.append(f"- Fatigue Level: {body_state.get('fatigue', 'N/A')}/10")
                        context_parts.append(f"- Activity Score: {body_state.get('activity_score', 'N/A')}/100")

                    if st.session_state.current_recommendation:
                        rec = st.session_state.current_recommendation
                        context_parts.append("\n\n**Current Recommendation:**")
                        context_parts.append(f"- Workout: {rec.get('workout_type', 'N/A')}")
                        context_parts.append(f"- Intensity: {rec.get('intensity', 'N/A')}")
                        context_parts.append(f"- Duration: {rec.get('duration_minutes', 'N/A')} minutes")
                        context_parts.append(f"- Rationale: {rec.get('rationale', 'N/A')}")

                    user_context = "\n".join(context_parts)

                    # Call OpenAI API
                    response = openai_client.chat.completions.create(
                        model=os.getenv("AGENT_MODEL", "gpt-4"),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_context}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )

                    agent_response = response.choices[0].message.content

                else:
                    # Fallback if OpenAI is not available
                    agent_response = "AI Coach is currently unavailable. Please check your OpenAI API key configuration in the .env file."

                # Add agent response to history
                st.session_state.chat_history.append({
                    'role': 'agent',
                    'content': agent_response,
                    'timestamp': datetime.now()
                })

                # Rerun to update chat
                st.rerun()

        except Exception as e:
            st.error(f"❌ Agent request failed: {str(e)}")
            # Add error to chat
            st.session_state.chat_history.append({
                'role': 'agent',
                'content': f"I encountered an error: {str(e)}. Please make sure your OpenAI API key is configured correctly.",
                'timestamp': datetime.now()
            })
            st.rerun()

    # Quick action buttons
    st.markdown("---")
    st.subheader("💡 Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📖 Explain My Plan", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Explain my current training plan'})
            st.rerun()

    with col2:
        if st.button("😴 I'm Tired", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': "I'm feeling very tired today"})
            st.rerun()

    with col3:
        if st.button("💪 Motivate Me", use_container_width=True):
            st.session_state.chat_history.append({'role': 'user', 'content': 'Give me some motivation'})
            st.rerun()

    with col4:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Tab 4: Data Analysis
with tab4:
    st.header("📈 Data Analysis & History")

    if st.session_state.recommendation_history:
        # Create DataFrame from history
        history_data = []
        for item in st.session_state.recommendation_history:
            history_data.append({
                'Time': item['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Workout Type': item['recommendation']['workout_type'],
                'Intensity': item['recommendation']['intensity'],
                'Duration (min)': item['recommendation']['duration_minutes'],
                'Readiness': item['state']['readiness_score'],
                'Sleep': item['state']['sleep_score'],
                'HRV': item['state']['hrv'],
                'Fatigue': item['state']['fatigue']
            })

        df = pd.DataFrame(history_data)

        # Display table
        st.subheader("📋 Recommendation History")
        st.dataframe(df, use_container_width=True)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Body State Trends")

            # Line chart for body metrics
            metrics_df = df[['Time', 'Readiness', 'Sleep', 'HRV', 'Fatigue']].copy()
            metrics_df = metrics_df.set_index('Time')

            fig = go.Figure()
            for col in metrics_df.columns:
                fig.add_trace(go.Scatter(
                    x=metrics_df.index,
                    y=metrics_df[col],
                    mode='lines+markers',
                    name=col
                ))

            fig.update_layout(
                title="Body Metrics Over Time",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode='x unified',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏋️ Workout Distribution")

            # Pie chart for workout types
            workout_counts = df['Workout Type'].value_counts()

            fig = px.pie(
                values=workout_counts.values,
                names=workout_counts.index,
                title="Workout Type Distribution",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        # Download data
        st.subheader("💾 Export Data")
        csv = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Download History (CSV)",
            data=csv,
            file_name=f"fitness_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        st.info("No history available yet. Please generate recommendations in the 'Get Recommendation' tab first.")

# Tab 5: User Guide
with tab5:
    st.header("📖 User Guide")

    st.markdown("""
    ## Welcome to AI Fitness Coach!

    ### 🎯 System Features

    This is a **Reinforcement Learning** powered personalized fitness plan recommendation system that provides intelligent training suggestions based on your real-time body state.

    ### 📊 How to Use

    #### 1. Get Training Recommendations

    In the 'Get Recommendation' tab:
    - Input your body state data (readiness score, sleep quality, heart rate, etc.)
    - Click the 'Get Recommendation' button
    - Review your personalized training plan

    **Data Sources:**
    - Can be obtained from Apple Watch, Oura Ring, etc.
    - Can also be manually entered based on subjective feelings

    #### 2. Submit Training Feedback

    In the 'Submit Feedback' tab:
    - After completing training, submit your feedback
    - Include completion status, exertion, mood, satisfaction
    - The system will use feedback to optimize future recommendations

    #### 3. Chat with AI Coach

    In the 'AI Coach Chat' tab:
    - Ask questions about your training plan
    - Get motivation and guidance
    - Request plan adjustments based on how you feel
    - Receive personalized coaching advice

    **AI Coach can help you with:**
    - 📖 Explaining your training plan and reasoning
    - 💪 Providing motivation and encouragement
    - 🔄 Adjusting plans based on fatigue or stress
    - 📊 Analyzing your progress and trends
    - 🎯 Setting micro-goals for better adherence

    #### 4. View Data Analysis

    In the 'Data Analysis' tab:
    - View historical recommendation records
    - Analyze body state trends
    - Understand workout distribution
    - Export data for further analysis

    ### 🔒 Safety Protection

    The system includes multiple safety mechanisms:
    - ✅ Overtraining prevention
    - ✅ Recovery status monitoring
    - ✅ Fatigue accumulation analysis
    - ✅ Dynamic intensity adjustment

    ### 🤖 Technical Features

    - **Machine Learning**: Contextual Bandits algorithm
    - **Online Learning**: Thompson Sampling for real-time optimization
    - **Accuracy**: 0.85+ AUC
    - **Response Speed**: <50ms latency
    - **AI Assistance**: GPT-4 powered intelligent coach

    ### ⚠️ Important Disclaimer

    - ⚠️ This system is **NOT a medical device** and does NOT provide medical advice
    - ⚠️ If you experience chest pain, dizziness, severe discomfort, or abnormal heart rate, **stop training immediately** and consult a medical professional
    - ⚠️ All recommendations are for reference only, please use according to your own situation

    ### 📞 Support

    For questions or suggestions, please contact the development team.

    ### 🎓 Learn More

    - **Project Documentation**: README.md
    - **Quick Start**: QUICK_START.md
    - **Execution Order**: RUN_ORDER.md

    ---

    **Enjoy your intelligent fitness journey! 💪**
    """)

    # Quick tips
    st.subheader("💡 Usage Tips")

    tips_col1, tips_col2 = st.columns(2)

    with tips_col1:
        st.info("""
        **Tips for Better Recommendations**

        1. Update body state data regularly
        2. Submit honest training feedback
        3. Follow safety recommendations
        4. Pay attention to recovery and rest
        5. Use AI Coach for guidance
        """)

    with tips_col2:
        st.success("""
        **System Advantages**

        1. Personalized recommendations
        2. Continuous learning optimization
        3. Safety protection mechanisms
        4. Data-driven decision making
        5. AI-powered coaching
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💪 AI Fitness Coach | Personalized Training Optimization System with Reinforcement Learning</p>
    <p>Powered by FastAPI + Streamlit + PyTorch + GPT-4</p>
</div>
""", unsafe_allow_html=True)
