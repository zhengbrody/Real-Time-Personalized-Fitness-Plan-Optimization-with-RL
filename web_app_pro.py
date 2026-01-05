"""
ProFit AI: Personalized Fitness Plan Optimizer
High-Performance Interface with Real AI Integration
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# --- CONFIGURATION & ASSETS ---
st.set_page_config(
    page_title="ProFit AI | RL Coach",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
script_dir = Path(__file__).parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI client for AI Coach
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith('sk-'):
        openai_client = OpenAI(api_key=api_key)
        AI_COACH_ENABLED = True
        print(f"✅ AI Coach enabled with API key: {api_key[:20]}...")
    else:
        AI_COACH_ENABLED = False
        print("❌ AI Coach disabled: Invalid or missing API key")
except Exception as e:
    AI_COACH_ENABLED = False
    print(f"❌ AI Coach disabled: {e}")

# Custom CSS with improved contrast and readability
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1a1a1a;
    }

    .main {
        background-color: #ffffff;
    }

    /* Improve all text visibility */
    p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: #1a1a1a !important;
    }

    /* Card Styling with better contrast */
    .stMetric {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }

    .stMetric label {
        color: #495057 !important;
        font-weight: 600;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #1a1a1a !important;
        font-size: 1.5rem;
        font-weight: 700;
    }

    /* Recommendation Card with white text on dark background */
    .rec-card {
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: #ffffff !important;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(31, 119, 180, 0.3);
    }

    .rec-card h3, .rec-card p, .rec-card strong {
        color: #ffffff !important;
    }

    .rec-card hr {
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Chat Bubbles with high contrast */
    .coach-msg {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        border: 2px solid #dee2e6;
        margin-bottom: 10px;
    }

    .coach-msg b, .coach-msg {
        color: #1a1a1a !important;
        font-weight: 600;
    }

    .user-msg {
        background-color: #1f77b4;
        color: #ffffff !important;
        padding: 15px;
        border-radius: 15px 15px 0px 15px;
        margin-bottom: 10px;
        text-align: right;
        border: 2px solid #1565c0;
    }

    .user-msg b, .user-msg {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Status Indicators - more visible */
    .status-online {
        color: #28a745 !important;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-offline {
        color: #dc3545 !important;
        font-weight: bold;
        font-size: 1.1rem;
    }

    /* Button improvements */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.3s;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border: none;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.3);
        background-color: #1565c0;
    }

    /* Sidebar improvements */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Form labels */
    .stSlider label, .stCheckbox label, .stRadio label {
        color: #1a1a1a !important;
        font-weight: 600;
    }

    /* Expander header */
    .streamlit-expanderHeader {
        color: #1a1a1a !important;
        font-weight: 600;
    }

    /* Info/Warning/Success boxes */
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZATION ---
API_BASE_URL = "http://localhost:8000"

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = 'user_001'
if 'recommendation_history' not in st.session_state:
    st.session_state.recommendation_history = []

# --- SIDEBAR: BIOMETRIC SYNC STATUS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lightning-bolt.png", width=80)
    st.title("ProFit AI")
    st.caption("RL-Powered Performance Coach")

    st.divider()

    # User ID
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id

    st.divider()

    st.subheader("📡 System Status")

    # Check API Server
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.markdown('<span class="status-online">● API Server Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-offline">● API Server Error</span>', unsafe_allow_html=True)
    except:
        st.markdown('<span class="status-offline">● API Server Offline</span>', unsafe_allow_html=True)

    # AI Coach Status
    if AI_COACH_ENABLED:
        st.markdown('<span class="status-online">● AI Coach Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-offline">● AI Coach Offline</span>', unsafe_allow_html=True)

    st.divider()

    # Quick State Overview
    if st.session_state.recommendation_history:
        latest = st.session_state.recommendation_history[-1]
        state = latest.get('state', {})
        st.subheader("📊 Current State")
        st.progress(state.get('readiness_score', 0)/100, text=f"Readiness: {state.get('readiness_score', 0)}%")
        st.progress(state.get('fatigue', 5)/10, text=f"Fatigue: {state.get('fatigue', 5)}/10")

    st.divider()
    st.info("Agent v2.1: Online Learning via Thompson Sampling")

# --- MAIN INTERFACE ---

# 1. TOP HEADER SECTION
st.title("Welcome back, Athlete.")
st.write(f"Today is {datetime.now().strftime('%A, %B %d, %Y')}")

# 2. THE DASHBOARD (Tabbed Experience)
tab_today, tab_analytics, tab_coach, tab_settings = st.tabs([
    "🎯 Today's Protocol",
    "📈 Performance Insights",
    "🤖 AI Coach Agent",
    "⚙️ Settings"
])

# --- TAB 1: TODAY'S PROTOCOL ---
with tab_today:
    col_input, col_rec = st.columns([1, 2])

    with col_input:
        st.subheader("📝 Body State Input")
        with st.form("state_update"):
            st.write("Input your current state:")

            readiness_score = st.slider("Readiness Score", 0, 100, 75)
            sleep_score = st.slider("Sleep Score", 0, 100, 80)
            hrv = st.slider("HRV (ms)", 20, 100, 50)
            resting_hr = st.slider("Resting Heart Rate", 40, 100, 60)
            fatigue = st.slider("Fatigue Level", 1, 10, 5)
            activity_score = st.slider("Activity Score", 0, 100, 70)

            generate_btn = st.form_submit_button("⚡ Get Today's Plan", use_container_width=True)

            if generate_btn:
                # Real API Call
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

                try:
                    with st.spinner("Generating recommendation..."):
                        response = requests.post(
                            f"{API_BASE_URL}/recommend",
                            json=request_data,
                            timeout=10
                        )

                        if response.status_code == 200:
                            recommendation = response.json()
                            st.session_state.current_plan = recommendation

                            # Add to history
                            st.session_state.recommendation_history.append({
                                'timestamp': datetime.now(),
                                'recommendation': recommendation,
                                'state': request_data['state']
                            })

                            st.success("✅ Plan generated!")
                        else:
                            st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Request failed: {str(e)}")

    with col_rec:
        st.subheader("⚡ Recommended Session")
        if st.session_state.current_plan:
            plan = st.session_state.current_plan

            # Display metrics if available
            if st.session_state.recommendation_history:
                latest = st.session_state.recommendation_history[-1]
                state = latest.get('state', {})

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("HRV", f"{state.get('hrv', 'N/A')} ms")
                m2.metric("Sleep", f"{state.get('sleep_score', 'N/A')}/100")
                m3.metric("Readiness", f"{state.get('readiness_score', 'N/A')}%")
                m4.metric("RHR", f"{state.get('resting_hr', 'N/A')} bpm")

            st.markdown(f"""
            <div class="rec-card">
                <h3>{plan['workout_type']}</h3>
                <p><strong>Intensity:</strong> {plan['intensity']} | <strong>Duration:</strong> {plan['duration_minutes']} min</p>
                <hr style="opacity:0.3">
                <p style="font-style: italic;">"{plan['rationale']}"</p>
                <p><strong>Description:</strong> {plan['description']}</p>
                <div style="background: rgba(255,255,255,0.2); padding: 10px; border-radius: 8px; font-size: 0.9rem;">
                    {'✅ Safety Check Passed' if plan['safety_check']['is_safe'] else '⚠️ ' + plan['safety_check']['message']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feedback sub-section
            with st.expander("✅ Log Completion"):
                with st.form("feedback_form"):
                    action_id = plan['action_id']
                    completed = st.checkbox("Training Completed", value=True)
                    rpe = st.slider("Workout RPE (1-10)", 1, 10, 7)
                    mood = st.slider("Mood (1-5)", 1, 5, 4)
                    satisfaction = st.slider("Satisfaction (1-10)", 1, 10, 8)

                    if st.form_submit_button("📤 Submit Feedback"):
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

                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/feedback",
                                json=feedback_data,
                                timeout=10
                            )

                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"✅ Feedback submitted! Reward: {result['reward']:.2f}")
                            else:
                                st.error("❌ Feedback submission failed")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👈 Input your body state and click 'Get Today's Plan' to receive a personalized recommendation.")

# --- TAB 2: ANALYTICS ---
with tab_analytics:
    st.subheader("Performance Trends & System Insights")

    if st.session_state.recommendation_history:
        # Create DataFrame from history
        history_data = []
        for item in st.session_state.recommendation_history:
            history_data.append({
                'Time': item['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Workout': item['recommendation']['workout_type'],
                'Intensity': item['recommendation']['intensity'],
                'Duration': item['recommendation']['duration_minutes'],
                'Readiness': item['state']['readiness_score'],
                'Sleep': item['state']['sleep_score'],
                'HRV': item['state']['hrv'],
                'Fatigue': item['state']['fatigue']
            })

        df = pd.DataFrame(history_data)

        # Metrics Trend Chart
        st.subheader("📊 Body State Trends")
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
            height=400,
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Workout Distribution
        col_pie, col_table = st.columns(2)

        with col_pie:
            st.subheader("🏋️ Workout Distribution")
            workout_counts = df['Workout'].value_counts()
            fig2 = px.pie(values=workout_counts.values, names=workout_counts.index, hole=.4)
            st.plotly_chart(fig2, use_container_width=True)

        with col_table:
            st.subheader("📋 Recent Sessions")
            st.dataframe(df.tail(5), use_container_width=True)

        # Download data
        csv = df.to_csv(index=False, encoding='utf-8')
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv,
            file_name=f"fitness_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No data yet. Get your first recommendation to start tracking!")

# --- TAB 3: AI COACH AGENT ---
with tab_coach:
    st.subheader("💬 AI Coach Interface")

    if AI_COACH_ENABLED:
        st.caption("Powered by GPT-4 with Real-Time Context")
    else:
        st.warning("⚠️ AI Coach is offline. Check your OpenAI API key in .env file")

    # Show current health data
    if st.session_state.recommendation_history:
        with st.expander("📊 Coach's View of Your Data"):
            latest = st.session_state.recommendation_history[-1]
            state = latest.get('state', {})

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Readiness", f"{state.get('readiness_score', 'N/A')}/100")
                st.metric("Sleep", f"{state.get('sleep_score', 'N/A')}/100")
            with col2:
                st.metric("HRV", f"{state.get('hrv', 'N/A')} ms")
                st.metric("RHR", f"{state.get('resting_hr', 'N/A')} bpm")
            with col3:
                st.metric("Fatigue", f"{state.get('fatigue', 'N/A')}/10")
                st.metric("Activity", f"{state.get('activity_score', 'N/A')}/100")

    # Check if we need to generate a response (last message is from user)
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        user_prompt = st.session_state.chat_history[-1]["content"]

        if AI_COACH_ENABLED:
            with st.spinner("🤖 AI Coach is thinking..."):
                try:
                    # Get body state
                    body_state = {}
                    if st.session_state.recommendation_history:
                        latest = st.session_state.recommendation_history[-1]
                        body_state = latest.get('state', {})

                    # Build context
                    system_prompt = """You are an expert AI Fitness Coach with access to real-time health data.

Analyze user's body state (readiness, sleep, HRV, heart rate, fatigue, activity) and provide personalized guidance.
Be supportive, data-driven, and conversational. Focus on fitness coaching and recovery optimization.

IMPORTANT: You are NOT a medical professional. If user reports serious symptoms, recommend consulting a doctor."""

                    context_parts = [f"User: {user_prompt}"]

                    if body_state:
                        context_parts.append("\n\n**Current Health Data:**")
                        context_parts.append(f"- Readiness: {body_state.get('readiness_score', 'N/A')}/100")
                        context_parts.append(f"- Sleep: {body_state.get('sleep_score', 'N/A')}/100")
                        context_parts.append(f"- HRV: {body_state.get('hrv', 'N/A')} ms")
                        context_parts.append(f"- RHR: {body_state.get('resting_hr', 'N/A')} bpm")
                        context_parts.append(f"- Fatigue: {body_state.get('fatigue', 'N/A')}/10")

                    if st.session_state.current_plan:
                        rec = st.session_state.current_plan
                        context_parts.append("\n\n**Current Recommendation:**")
                        context_parts.append(f"- Workout: {rec.get('workout_type', 'N/A')}")
                        context_parts.append(f"- Intensity: {rec.get('intensity', 'N/A')}")
                        context_parts.append(f"- Duration: {rec.get('duration_minutes', 'N/A')} min")

                    user_context = "\n".join(context_parts)

                    # Call OpenAI
                    print(f"Calling OpenAI API with model: {os.getenv('AGENT_MODEL', 'gpt-4')}")
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
                    print(f"✅ Got response: {agent_response[:100]}...")
                    st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
                    st.rerun()

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(f"❌ OpenAI API error: {error_msg}")
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    st.rerun()
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "AI Coach is offline. Please configure your OpenAI API key."})
            st.rerun()

    # Chat Window
    st.markdown('<div style="max-height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #dee2e6; border-radius: 10px; background-color: #ffffff;">', unsafe_allow_html=True)
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.chat_history:
            div_class = "user-msg" if msg["role"] == "user" else "coach-msg"
            label = "👤 You" if msg["role"] == "user" else "🤖 Coach"
            st.markdown(f'<div class="{div_class}"><b>{label}:</b> {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Input Area
    col_input, col_button = st.columns([4, 1])
    with col_input:
        prompt = st.text_input("Ask your coach...", key="chat_input_text", placeholder="Ask about your plan, recovery, or get motivation...", label_visibility="collapsed")
    with col_button:
        send_button = st.button("Send", type="primary", use_container_width=True)

    if send_button and prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

    # Quick Actions
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📖 Explain My Plan", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Explain my current training plan"})
            st.rerun()
    with col2:
        if st.button("💪 Motivate Me", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": "Give me some motivation"})
            st.rerun()
    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- TAB 4: SETTINGS ---
with tab_settings:
    st.subheader("⚙️ System Configuration")

    st.markdown("""
    ### Technical Stack

    | Component | Status | Description |
    | :--- | :--- | :--- |
    | **Feature Engineering** | Active | 45+ body state features |
    | **Recommendation Engine** | Hybrid | Contextual Bandits + Rules |
    | **Model Serving** | FastAPI | Real-time inference |
    | **RL Algorithm** | Thompson Sampling | Online learning enabled |
    | **AI Coach** | GPT-4 | Natural language interface |
    """)

    st.divider()

    st.subheader("🔐 Privacy & Data")
    st.info("""
    - All data stored locally in SQLite database
    - No cloud sync (unless configured)
    - Feature store managed via Feast framework
    - Model updates via feedback loop
    """)

    st.divider()

    st.subheader("📡 API Configuration")
    st.code(f"API Base URL: {API_BASE_URL}", language="text")
    st.code(f"User ID: {st.session_state.user_id}", language="text")
    st.code(f"AI Coach: {'Enabled' if AI_COACH_ENABLED else 'Disabled'}", language="text")

# --- FOOTER ---
st.divider()
st.caption("ProFit AI © 2024 | RL-Powered Personalized Training | Research & Educational Use Only")
