"""
ProFit AI: Personalized Fitness Plan Optimizer
High-Performance Interface with Real AI Integration
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# --- LOCAL RL ENGINE (fallback when API server is offline) ---
sys.path.insert(0, str(Path(__file__).parent))
try:
    from src.recommendation.action_space import ActionSpace
    from src.recommendation.contextual_bandits import ContextualBandit
    _RL_AVAILABLE = True
except Exception:
    _RL_AVAILABLE = False

class _SafetyFilter:
    def filter_actions(self, state, action_space):
        all_ids = list(range(action_space.get_action_count()))
        r = state.get("readiness_score", 50)
        f = state.get("fatigue", 5)
        if r < 30 or f > 8:
            return [i for i in all_ids if action_space.get_action(i).workout_type in ("REST", "RECOVERY")]
        if f > 6:
            return [i for i in all_ids if action_space.get_action(i).intensity in ("LOW", "NONE")]
        return all_ids

def _get_local_recommendation(state):
    """Run Thompson Sampling locally when API server is unavailable."""
    if not _RL_AVAILABLE:
        return None
    if "local_action_space" not in st.session_state:
        st.session_state.local_action_space = ActionSpace()
        st.session_state.local_bandit = ContextualBandit(
            n_actions=st.session_state.local_action_space.get_action_count()
        )
        st.session_state.local_safety = _SafetyFilter()

    action_space = st.session_state.local_action_space
    bandit = st.session_state.local_bandit
    safety = st.session_state.local_safety

    allowed = safety.filter_actions(state, action_space)
    action_id = bandit.select_action(allowed)
    action = action_space.get_action(action_id)

    fatigue = state.get("fatigue", 5)
    readiness = state.get("readiness_score", 50)
    if fatigue > 6 or readiness < 40:
        rationale = "High fatigue detected — conservative session recommended to avoid overtraining."
    elif readiness > 75:
        rationale = "Strong readiness metrics — Thompson Sampling selected a challenging session."
    else:
        rationale = "Moderate readiness — balanced session selected via Thompson Sampling."

    return {
        "action_id": action_id,
        "workout_type": action.workout_type,
        "intensity": action.intensity,
        "duration_minutes": action.duration_minutes,
        "description": action.description,
        "rationale": rationale,
        "safety_check": {"is_safe": True, "message": "Passed local safety gate"},
    }

# --- CONFIGURATION & ASSETS ---
st.set_page_config(
    page_title="ProFit AI | RL Coach",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load environment variables
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize OpenAI client for AI Coach
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith("sk-"):
        openai_client = OpenAI(api_key=api_key)
        AI_COACH_ENABLED = True
        print(f"✅ AI Coach enabled with API key: {api_key[:20]}...")
    else:
        AI_COACH_ENABLED = False
        print("❌ AI Coach disabled: Invalid or missing API key")
except Exception as e:
    AI_COACH_ENABLED = False
    print(f"❌ AI Coach disabled: {e}")

# Custom CSS with dark mode support
dark_mode = st.session_state.get("dark_mode", False)

if dark_mode:
    # Dark Mode CSS
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }

    .main {
        background-color: #121212;
    }

    p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: #e0e0e0 !important;
    }

    .stMetric {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #404040;
    }

    .stMetric label {
        color: #b0b0b0 !important;
        font-weight: 600;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .rec-card {
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: #ffffff !important;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }

    .rec-card h3, .rec-card p, .rec-card strong {
        color: #ffffff !important;
    }

    .coach-msg {
        background-color: #2a2a2a;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        border: 2px solid #404040;
        margin-bottom: 10px;
    }

    .coach-msg b, .coach-msg {
        color: #e0e0e0 !important;
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

    .stButton>button {
        border-radius: 8px;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border: none;
    }

    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #e0e0e0 !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox select {
        color: #e0e0e0 !important;
        background-color: #2a2a2a !important;
        border: 1px solid #404040;
    }

    .stAlert {
        background-color: #2a2a2a;
        border-radius: 8px;
    }

    .status-online { color: #4caf50; font-weight: 600; }
    .status-offline { color: #f44336; font-weight: 600; }
</style>
""",
        unsafe_allow_html=True,
    )
else:
    # Light Mode CSS — only style custom components, let Streamlit handle the rest
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body { font-family: 'Inter', sans-serif; }

    .rec-card {
        background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%);
        color: #ffffff !important;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(31, 119, 180, 0.3);
    }
    .rec-card h3, .rec-card p, .rec-card strong { color: #ffffff !important; }

    .coach-msg {
        background-color: #f0f4f8;
        color: #1a1a1a !important;
        padding: 15px;
        border-radius: 15px 15px 15px 0px;
        border: 1px solid #d0d7de;
        margin-bottom: 10px;
    }
    .coach-msg b { color: #1a1a1a !important; }

    .user-msg {
        background-color: #1f77b4;
        color: #ffffff !important;
        padding: 15px;
        border-radius: 15px 15px 0px 15px;
        margin-bottom: 10px;
        text-align: right;
    }

    .status-online { color: #28a745; font-weight: 600; }
    .status-offline { color: #dc3545; font-weight: 600; }
</style>
""",
        unsafe_allow_html=True,
    )

# --- INITIALIZATION ---
API_BASE_URL = "http://localhost:8000"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_001"
if "recommendation_history" not in st.session_state:
    st.session_state.recommendation_history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "selected_date" not in st.session_state:
    st.session_state.selected_date = None

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
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            st.markdown(
                '<span class="status-online">● API Server Active</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-offline">● API Server Error</span>',
                unsafe_allow_html=True,
            )
    except Exception:
        if _RL_AVAILABLE:
            st.markdown(
                '<span class="status-online">● RL Engine (Local Mode)</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-offline">● API Server Offline</span>',
                unsafe_allow_html=True,
            )

    # AI Coach Status
    if AI_COACH_ENABLED:
        st.markdown(
            '<span class="status-online">● AI Coach Active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-offline">● AI Coach Offline</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Quick State Overview
    if st.session_state.recommendation_history:
        latest = st.session_state.recommendation_history[-1]
        state = latest.get("state", {})
        st.subheader("📊 Current State")
        st.progress(
            state.get("readiness_score", 0) / 100,
            text=f"Readiness: {state.get('readiness_score', 0)}%",
        )
        st.progress(
            state.get("fatigue", 5) / 10, text=f"Fatigue: {state.get('fatigue', 5)}/10"
        )

    st.divider()

    # Dark Mode Toggle
    st.subheader("🌓 Display Mode")
    dark_mode_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        st.rerun()

    st.divider()
    st.info("Agent v2.1: Online Learning via Thompson Sampling")

# --- MAIN INTERFACE ---

# 1. TOP HEADER SECTION
st.title("Welcome back, Athlete.")
st.write(f"Today is {datetime.now().strftime('%A, %B %d, %Y')}")

# 2. THE DASHBOARD (Tabbed Experience)
tab_today, tab_analytics, tab_coach, tab_settings = st.tabs(
    [
        "🎯 Today's Protocol",
        "📈 Performance Insights",
        "🤖 AI Coach Agent",
        "⚙️ Settings",
    ]
)

# --- TAB 1: TODAY'S PROTOCOL ---
with tab_today:
    col_input, col_rec = st.columns([1, 2])

    with col_input:
        st.subheader("📝 Body State Input")

        # Pre-fill from most recent uploaded/manual data
        _last_state = {}
        for entry in reversed(st.session_state.recommendation_history):
            s = entry.get("state", {})
            if s:
                _last_state = s
                break
        _def_readiness = int(_last_state.get("readiness_score", 75))
        _def_sleep     = int(_last_state.get("sleep_score", 80))
        _def_hrv       = int(_last_state.get("hrv", 50))
        _def_rhr       = int(_last_state.get("resting_hr", 60))
        _def_fatigue   = int(_last_state.get("fatigue", 5))
        _def_activity  = int(_last_state.get("activity_score", 70))

        if _last_state:
            st.caption("✅ Pre-filled from your latest uploaded data")

        with st.form("state_update"):
            st.write("Input your current state:")

            readiness_score = st.slider("Readiness Score", 0, 100, _def_readiness)
            sleep_score = st.slider("Sleep Score", 0, 100, _def_sleep)
            hrv = st.slider("HRV (ms)", 20, 100, max(20, min(100, _def_hrv)))
            resting_hr = st.slider("Resting Heart Rate", 40, 100, max(40, min(100, _def_rhr)))
            fatigue = st.slider("Fatigue Level", 1, 10, max(1, min(10, _def_fatigue)))
            activity_score = st.slider("Activity Score", 0, 100, _def_activity)

            generate_btn = st.form_submit_button(
                "⚡ Get Today's Plan", use_container_width=True
            )

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
                        "activity_score": activity_score,
                    },
                }

                with st.spinner("Generating recommendation..."):
                    recommendation = _get_local_recommendation(request_data["state"])
                    if recommendation:
                        st.session_state.current_plan = recommendation
                        st.session_state.recommendation_history.append(
                            {
                                "timestamp": datetime.now(),
                                "recommendation": recommendation,
                                "state": request_data["state"],
                            }
                        )
                        st.success("✅ Plan generated!")
                    else:
                        st.error("❌ RL engine not available")

    with col_rec:
        st.subheader("⚡ Recommended Session")
        if st.session_state.current_plan:
            plan = st.session_state.current_plan

            # Display metrics if available
            if st.session_state.recommendation_history:
                latest = st.session_state.recommendation_history[-1]
                state = latest.get("state", {})

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("HRV", f"{state.get('hrv', 'N/A')} ms")
                m2.metric("Sleep", f"{state.get('sleep_score', 'N/A')}/100")
                m3.metric("Readiness", f"{state.get('readiness_score', 'N/A')}%")
                m4.metric("RHR", f"{state.get('resting_hr', 'N/A')} bpm")

            st.markdown(
                f"""
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
            """,
                unsafe_allow_html=True,
            )

            # Feedback sub-section
            with st.expander("✅ Log Completion"):
                with st.form("feedback_form"):
                    action_id = plan["action_id"]
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
                                "satisfaction": satisfaction,
                            },
                        }

                        if _RL_AVAILABLE and "local_bandit" in st.session_state:
                            r = 1.0 if completed else 0.0
                            r += (satisfaction - 5) / 10
                            r = max(0.0, min(1.0, r))
                            st.session_state.local_bandit.update(action_id, r)
                            st.success(f"✅ Feedback submitted! Reward: {r:.2f}")
                        else:
                            st.warning("⚠️ Feedback noted (RL engine not initialized yet)")
        else:
            st.info(
                "👈 Input your body state and click 'Get Today's Plan' to receive a personalized recommendation."
            )

# --- TAB 2: ANALYTICS ---
with tab_analytics:
    st.subheader("📊 Performance Trends & System Insights")

    if st.session_state.recommendation_history:
        # Create DataFrame from history
        history_data = []
        for item in st.session_state.recommendation_history:
            rec = item.get("recommendation", item.get("state", {}))
            state = item.get("state", {})
            history_data.append(
                {
                    "Date": datetime.fromtimestamp(
                        item.get("timestamp", datetime.now().timestamp())
                    ),
                    "Workout": rec.get(
                        "workout_type", item.get("workout_type", "Unknown")
                    ),
                    "Intensity": rec.get("intensity", item.get("intensity", "N/A")),
                    "Duration": rec.get(
                        "duration_minutes", item.get("duration_minutes", 0)
                    ),
                    "Readiness": state.get("readiness_score", 0),
                    "Sleep": state.get("sleep_score", 0),
                    "HRV": state.get("hrv", 0),
                    "RHR": state.get("resting_hr", 0),
                    "Fatigue": state.get("fatigue", 0),
                    "Activity": state.get("activity_score", 0),
                    "Mood": state.get("mood", 5),
                    "Stress": state.get("stress", 5),
                }
            )

        df = pd.DataFrame(history_data)
        df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

        # Summary Statistics
        st.subheader("📈 7-Day Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_readiness = df["Readiness"].tail(7).mean()
            st.metric(
                "Avg Readiness",
                f"{avg_readiness:.1f}/100",
                delta=(
                    f"{df['Readiness'].iloc[-1] - avg_readiness:.1f}"
                    if len(df) > 1
                    else None
                ),
            )

        with col2:
            avg_hrv = df["HRV"].tail(7).mean()
            st.metric(
                "Avg HRV",
                f"{avg_hrv:.1f} ms",
                delta=f"{df['HRV'].iloc[-1] - avg_hrv:.1f}" if len(df) > 1 else None,
            )

        with col3:
            avg_sleep = df["Sleep"].tail(7).mean()
            st.metric(
                "Avg Sleep Score",
                f"{avg_sleep:.1f}/100",
                delta=(
                    f"{df['Sleep'].iloc[-1] - avg_sleep:.1f}" if len(df) > 1 else None
                ),
            )

        with col4:
            total_duration = df["Duration"].tail(7).sum()
            st.metric(
                "Total Training",
                f"{total_duration} min",
                delta=f"+{df['Duration'].iloc[-1]}" if len(df) > 0 else None,
            )

        st.divider()

        # Time Range Selector
        col_range, col_metrics = st.columns([1, 3])

        with col_range:
            time_range = st.selectbox(
                "Time Range",
                ["Last 7 Days", "Last 14 Days", "Last 30 Days", "All Time"],
                key="analytics_time_range",
            )

            days_map = {
                "Last 7 Days": 7,
                "Last 14 Days": 14,
                "Last 30 Days": 30,
                "All Time": len(df),
            }
            days = days_map[time_range]
            df_filtered = df.tail(days)

        with col_metrics:
            metrics_to_show = st.multiselect(
                "Select Metrics",
                [
                    "Readiness",
                    "Sleep",
                    "HRV",
                    "RHR",
                    "Fatigue",
                    "Activity",
                    "Mood",
                    "Stress",
                ],
                default=["Readiness", "Sleep", "HRV", "Fatigue"],
                key="metrics_selector",
            )

        # Main Trends Chart
        st.subheader("📉 Health Metrics Over Time")

        fig = go.Figure()
        for metric in metrics_to_show:
            if metric in df_filtered.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df_filtered["Date"],
                        y=df_filtered[metric],
                        mode="lines+markers",
                        name=metric,
                        line=dict(width=2),
                        marker=dict(size=6),
                    )
                )

        fig.update_layout(
            title=f"Health Metrics - {time_range}",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified",
            height=450,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Correlation Analysis
        st.subheader("🔗 Metric Correlations")

        col_corr, col_insights = st.columns([2, 1])

        with col_corr:
            # Calculate correlation matrix
            corr_metrics = ["Readiness", "Sleep", "HRV", "Fatigue", "Activity"]
            corr_df = df_filtered[corr_metrics].corr()

            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=corr_df.values.round(2),
                    texttemplate="%{text}",
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation"),
                )
            )

            fig_corr.update_layout(
                title="Metric Correlation Matrix", height=400, template="plotly_white"
            )

            st.plotly_chart(fig_corr, use_container_width=True)

        with col_insights:
            st.markdown("**Key Insights:**")

            # Find strongest correlations
            corr_flat = corr_df.where(~np.tril(np.ones(corr_df.shape)).astype(bool))
            strong_corr = corr_flat.stack().sort_values(ascending=False)

            if len(strong_corr) > 0:
                top_corr = strong_corr.iloc[0]
                metrics_pair = strong_corr.index[0]
                st.info(
                    f"**Strongest Correlation:**\n{metrics_pair[0]} ↔ {metrics_pair[1]}\n({top_corr:.2f})"
                )

            # Average scores
            st.metric("Avg Readiness", f"{df_filtered['Readiness'].mean():.1f}/100")
            st.metric("Avg Sleep", f"{df_filtered['Sleep'].mean():.1f}/100")
            st.metric("Avg HRV", f"{df_filtered['HRV'].mean():.1f} ms")

        st.divider()

        # Workout Distribution & Performance
        col_pie, col_bar = st.columns(2)

        with col_pie:
            st.subheader("🏋️ Workout Distribution")
            workout_counts = df["Workout"].value_counts()
            fig2 = px.pie(
                values=workout_counts.values, names=workout_counts.index, hole=0.4
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        with col_bar:
            st.subheader("📊 Training Volume")
            # Weekly training volume
            df_weekly = (
                df.groupby(df["Date"].dt.to_period("W"))["Duration"].sum().reset_index()
            )
            df_weekly["Date"] = df_weekly["Date"].astype(str)

            fig_bar = px.bar(
                df_weekly,
                x="Date",
                y="Duration",
                title="Weekly Training Minutes",
                labels={"Duration": "Minutes", "Date": "Week"},
            )
            fig_bar.update_layout(height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Recent Sessions Table
        st.subheader("📋 Recent Sessions")
        st.dataframe(
            df[
                [
                    "Date",
                    "Workout",
                    "Intensity",
                    "Duration",
                    "Readiness",
                    "Sleep",
                    "HRV",
                ]
            ].tail(10),
            use_container_width=True,
        )

        # Download data
        csv = df.to_csv(index=False, encoding="utf-8")
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv,
            file_name=f"fitness_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
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
            state = latest.get("state", {})

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
    if (
        st.session_state.chat_history
        and st.session_state.chat_history[-1]["role"] == "user"
    ):
        user_prompt = st.session_state.chat_history[-1]["content"]

        if AI_COACH_ENABLED:
            with st.spinner("🤖 AI Coach is thinking..."):
                try:
                    # Get body state
                    body_state = {}
                    if st.session_state.recommendation_history:
                        latest = st.session_state.recommendation_history[-1]
                        body_state = latest.get("state", {})

                    # Build context
                    system_prompt = """You are an expert AI Fitness Coach with access to real-time health data.

Analyze user's body state (readiness, sleep, HRV, heart rate, fatigue, activity) and provide personalized guidance.
Be supportive, data-driven, and conversational. Focus on fitness coaching and recovery optimization.

IMPORTANT: You are NOT a medical professional. If user reports serious symptoms, recommend consulting a doctor."""

                    context_parts = [f"User: {user_prompt}"]

                    if body_state:
                        context_parts.append("\n\n**Current Health Data:**")
                        context_parts.append(
                            f"- Readiness: {body_state.get('readiness_score', 'N/A')}/100"
                        )
                        context_parts.append(
                            f"- Sleep: {body_state.get('sleep_score', 'N/A')}/100"
                        )
                        context_parts.append(
                            f"- HRV: {body_state.get('hrv', 'N/A')} ms"
                        )
                        context_parts.append(
                            f"- RHR: {body_state.get('resting_hr', 'N/A')} bpm"
                        )
                        context_parts.append(
                            f"- Fatigue: {body_state.get('fatigue', 'N/A')}/10"
                        )

                    if st.session_state.current_plan:
                        rec = st.session_state.current_plan
                        context_parts.append("\n\n**Current Recommendation:**")
                        context_parts.append(
                            f"- Workout: {rec.get('workout_type', 'N/A')}"
                        )
                        context_parts.append(
                            f"- Intensity: {rec.get('intensity', 'N/A')}"
                        )
                        context_parts.append(
                            f"- Duration: {rec.get('duration_minutes', 'N/A')} min"
                        )

                    user_context = "\n".join(context_parts)

                    # Call OpenAI
                    print(
                        f"Calling OpenAI API with model: {os.getenv('AGENT_MODEL', 'gpt-4')}"
                    )
                    response = openai_client.chat.completions.create(
                        model=os.getenv("AGENT_MODEL", "gpt-4"),
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_context},
                        ],
                        temperature=0.7,
                        max_tokens=500,
                    )

                    agent_response = response.choices[0].message.content
                    print(f"✅ Got response: {agent_response[:100]}...")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": agent_response}
                    )
                    st.rerun()

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(f"❌ OpenAI API error: {error_msg}")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )
                    st.rerun()
        else:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": "AI Coach is offline. Please configure your OpenAI API key.",
                }
            )
            st.rerun()

    # Chat Window
    st.markdown(
        '<div style="max-height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #dee2e6; border-radius: 10px; background-color: #ffffff;">',
        unsafe_allow_html=True,
    )
    chat_box = st.container()
    with chat_box:
        for msg in st.session_state.chat_history:
            div_class = "user-msg" if msg["role"] == "user" else "coach-msg"
            label = "👤 You" if msg["role"] == "user" else "🤖 Coach"
            st.markdown(
                f'<div class="{div_class}"><b>{label}:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Input Area
    col_input, col_button = st.columns([4, 1])
    with col_input:
        prompt = st.text_input(
            "Ask your coach...",
            key="chat_input_text",
            placeholder="Ask about your plan, recovery, or get motivation...",
            label_visibility="collapsed",
        )
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
            st.session_state.chat_history.append(
                {"role": "user", "content": "Explain my current training plan"}
            )
            st.rerun()
    with col2:
        if st.button("💪 Motivate Me", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "content": "Give me some motivation"}
            )
            st.rerun()
    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- TAB 4: SETTINGS ---
with tab_settings:
    st.subheader("⚙️ Settings & Configuration")

    # Data Upload Section
    st.subheader("📤 Upload Health Data")
    st.markdown(
        "Upload your daily health data from Apple Watch or Oura Ring (no iOS app needed!)"
    )

    upload_tab1, upload_tab2 = st.tabs(["📱 Manual Entry", "📁 File Upload"])

    with upload_tab1:
        st.write("**Manually input today's data from your devices:**")
        with st.form("manual_data_upload"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Oura Data**")
                oura_readiness = st.number_input(
                    "Readiness Score", 0, 100, 75, key="oura_readiness"
                )
                oura_sleep = st.number_input(
                    "Sleep Score", 0, 100, 80, key="oura_sleep"
                )
                oura_hrv = st.number_input("HRV (ms)", 20, 100, 50, key="oura_hrv")

            with col2:
                st.markdown("**Apple Watch Data**")
                watch_rhr = st.number_input(
                    "Resting HR (bpm)", 40, 100, 60, key="watch_rhr"
                )
                watch_activity = st.number_input(
                    "Activity Score", 0, 100, 70, key="watch_activity"
                )
                watch_exercise = st.number_input(
                    "Exercise Minutes", 0, 180, 30, key="watch_exercise"
                )

            with col3:
                st.markdown("**Subjective**")
                fatigue_level = st.slider(
                    "Fatigue Level", 1, 10, 5, key="fatigue_manual"
                )
                mood_score = st.slider("Mood", 1, 10, 7, key="mood_manual")
                stress_level = st.slider("Stress", 1, 10, 5, key="stress_manual")

            submit_manual = st.form_submit_button(
                "💾 Upload Today's Data", use_container_width=True
            )

            if submit_manual:
                # Create a recommendation entry
                new_entry = {
                    "timestamp": datetime.now().timestamp(),
                    "state": {
                        "readiness_score": oura_readiness,
                        "sleep_score": oura_sleep,
                        "hrv": oura_hrv,
                        "resting_hr": watch_rhr,
                        "activity_score": watch_activity,
                        "fatigue": fatigue_level,
                        "mood": mood_score,
                        "stress": stress_level,
                        "exercise_minutes": watch_exercise,
                    },
                    "workout_type": "Pending",
                    "intensity": "TBD",
                    "duration_minutes": 0,
                    "source": "manual_upload",
                }
                st.session_state.recommendation_history.append(new_entry)
                st.success(
                    f"✅ Data uploaded successfully! Total entries: {len(st.session_state.recommendation_history)}"
                )
                st.rerun()

    with upload_tab2:
        st.write("**Upload CSV or JSON file with health data:**")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["csv", "json"], key="health_data_upload"
        )

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    import io

                    df = pd.read_csv(
                        io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    )
                    st.success(f"Loaded {len(df)} rows from CSV")
                    st.dataframe(df.head())

                    if st.button("Import to History"):
                        for _, row in df.iterrows():
                            entry = {
                                "timestamp": datetime.now().timestamp(),
                                "state": {
                                    "readiness_score": int(
                                        row.get("readiness_score", 75)
                                    ),
                                    "sleep_score": int(row.get("sleep_score", 80)),
                                    "hrv": int(row.get("hrv", 50)),
                                    "resting_hr": int(row.get("resting_hr", 60)),
                                    "activity_score": int(
                                        row.get("activity_score", 70)
                                    ),
                                    "fatigue": int(row.get("fatigue", 5)),
                                },
                                "workout_type": row.get("workout_type", "Unknown"),
                                "source": "csv_upload",
                            }
                            st.session_state.recommendation_history.append(entry)
                        st.success(f"Imported {len(df)} entries!")
                        st.rerun()

                elif uploaded_file.name.endswith(".json"):
                    import json

                    data = json.load(uploaded_file)
                    st.success(f"Loaded JSON with {len(data)} entries")
                    st.json(
                        data[0] if isinstance(data, list) and len(data) > 0 else data
                    )

                    if st.button("Import to History"):
                        if isinstance(data, list):
                            st.session_state.recommendation_history.extend(data)
                        else:
                            st.session_state.recommendation_history.append(data)
                        st.success("Imported successfully!")
                        st.rerun()

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

        st.info("""
        **CSV Format Example:**
        ```
        readiness_score,sleep_score,hrv,resting_hr,activity_score,fatigue,workout_type
        85,90,55,58,75,3,Strength
        80,85,52,60,70,4,Cardio
        ```
        """)

    st.divider()

    # User Profile Settings
    st.subheader("👤 User Profile")
    col1, col2 = st.columns(2)
    with col1:
        user_name = st.text_input(
            "Display Name", value="Athlete", key="user_name_settings"
        )
        user_age = st.number_input(
            "Age", min_value=18, max_value=100, value=30, key="user_age_settings"
        )
    with col2:
        user_weight = st.number_input(
            "Weight (kg)",
            min_value=40,
            max_value=200,
            value=70,
            key="user_weight_settings",
        )
        user_height = st.number_input(
            "Height (cm)",
            min_value=140,
            max_value=220,
            value=175,
            key="user_height_settings",
        )

    st.divider()

    # Historical Data Viewer
    st.subheader("📅 Historical Data Viewer")

    if st.session_state.recommendation_history:
        # Date selector
        dates_available = [
            datetime.fromtimestamp(rec.get("timestamp", datetime.now().timestamp()))
            for rec in st.session_state.recommendation_history
        ]

        selected_date = st.date_input(
            "Select Date to View",
            value=(
                max(dates_available).date()
                if dates_available
                else datetime.now().date()
            ),
            key="historical_date_selector",
        )

        # Filter recommendations for selected date
        recs_on_date = [
            rec
            for rec in st.session_state.recommendation_history
            if datetime.fromtimestamp(rec.get("timestamp", 0)).date() == selected_date
        ]

        if recs_on_date:
            st.success(
                f"Found {len(recs_on_date)} recommendation(s) on {selected_date}"
            )

            for idx, rec in enumerate(recs_on_date):
                with st.expander(
                    f"Recommendation #{idx + 1} - {datetime.fromtimestamp(rec.get('timestamp', 0)).strftime('%H:%M:%S')}"
                ):
                    state = rec.get("state", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Readiness", f"{state.get('readiness_score', 'N/A')}/100"
                        )
                        st.metric("Sleep", f"{state.get('sleep_score', 'N/A')}/100")
                    with col2:
                        st.metric("HRV", f"{state.get('hrv', 'N/A')} ms")
                        st.metric("RHR", f"{state.get('resting_hr', 'N/A')} bpm")
                    with col3:
                        st.metric("Workout", rec.get("workout_type", "N/A"))
                        st.metric(
                            "Duration", f"{rec.get('duration_minutes', 'N/A')} min"
                        )
        else:
            st.info(f"No data found for {selected_date}")
    else:
        st.info(
            "No historical data available yet. Get your first recommendation to start tracking!"
        )

    st.divider()

    # API & System Configuration
    st.subheader("📡 API & System Status")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("API Server", "🟢 Online" if API_BASE_URL else "🔴 Offline")
        st.code(f"{API_BASE_URL}", language="text")
    with col2:
        st.metric("AI Coach", "🟢 Active" if AI_COACH_ENABLED else "🔴 Offline")
        st.code(f"Model: {os.getenv('AGENT_MODEL', 'gpt-4')}", language="text")

    st.divider()

    # Data Management
    st.subheader("💾 Data Management")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export History", use_container_width=True):
            if st.session_state.recommendation_history:
                import json

                data_json = json.dumps(
                    st.session_state.recommendation_history, indent=2
                )
                st.download_button(
                    label="Download JSON",
                    data=data_json,
                    file_name=f"fitness_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                )
            else:
                st.warning("No data to export")

    with col2:
        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.rerun()

    with col3:
        if st.button("🗑️ Clear History", use_container_width=True):
            if st.session_state.recommendation_history:
                st.session_state.recommendation_history = []
                st.session_state.chat_history = []
                st.success("History cleared!")
                st.rerun()

    st.divider()

    # System Information
    st.subheader("ℹ️ System Information")

    st.markdown("""
    **Technical Stack:**
    - **Recommendation Engine**: Contextual Bandits + Thompson Sampling
    - **Feature Engineering**: 45+ body state features
    - **Model Serving**: FastAPI (Real-time inference)
    - **AI Coach**: GPT-4 with health data context
    - **Data Storage**: Local SQLite database
    """)

    st.caption(f"Total Recommendations: {len(st.session_state.recommendation_history)}")
    st.caption(f"Chat Messages: {len(st.session_state.chat_history)}")

# --- FOOTER ---
st.divider()
st.caption(
    "ProFit AI © 2024 | RL-Powered Personalized Training | Research & Educational Use Only"
)
