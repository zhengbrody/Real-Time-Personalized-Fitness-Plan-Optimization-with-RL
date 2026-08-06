"""
ProFit AI: Personalized Fitness Plan Optimizer
High-Performance Interface with Real AI Integration

Stage-2 UI: the recommendation form collects a structured TodayRequest and
dispatches through the single engine entry point `recommend_today`. The old
6-slider local-RL fake pipeline has been retired.
"""

import streamlit as st
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

# --- RECOMMENDATION ENGINE (single entry point shared with the API) ---
sys.path.insert(0, str(Path(__file__).parent))
try:
    from src.recommendation import recommend_today
    from src.recommendation.action_space import ActionSpace
    from src.recommendation.contextual_bandits import ContextualBandit
    from src.validation.schemas import (
        ManualCheckIn,
        RecentSession,
        TodayRequest,
        WearableData,
    )

    _RL_AVAILABLE = True
except Exception:
    _RL_AVAILABLE = False


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
    else:
        AI_COACH_ENABLED = False
except Exception:
    AI_COACH_ENABLED = False

# Custom CSS with dark mode support
dark_mode = st.session_state.get("dark_mode", False)

if dark_mode:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #e0e0e0; }
    .main { background-color: #121212; }
    p, span, div, h1, h2, h3, h4, h5, h6, label,
    [data-testid="stMarkdownContainer"] p, [data-testid="stText"],
    [class*="stSlider"] label, [class*="stSlider"] p,
    [data-testid="stWidgetLabel"] { color: #e0e0e0 !important; }
    .stMetric { background-color: #2a2a2a; padding: 15px; border-radius: 12px; border: 1px solid #404040; }
    .stMetric label { color: #b0b0b0 !important; font-weight: 600; }
    .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.5rem; font-weight: 700; }
    .rec-card { background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%); color: #ffffff !important; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; }
    .rec-card h3, .rec-card p, .rec-card strong { color: #ffffff !important; }
    .coach-msg { background-color: #2a2a2a; padding: 15px; border-radius: 15px 15px 15px 0px; border: 2px solid #404040; margin-bottom: 10px; }
    .coach-msg b, .coach-msg { color: #e0e0e0 !important; font-weight: 600; }
    .user-msg { background-color: #1f77b4; color: #ffffff !important; padding: 15px; border-radius: 15px 15px 0px 15px; margin-bottom: 10px; text-align: right; border: 2px solid #1565c0; }
    .stButton>button { border-radius: 8px; background-color: #1f77b4; color: white; font-weight: 600; border: none; }
    [data-testid="stSidebar"] { background-color: #1e1e1e; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #e0e0e0 !important; }
    .stTextInput input, .stNumberInput input, .stSelectbox select { color: #e0e0e0 !important; background-color: #2a2a2a !important; border: 1px solid #404040; }
    .stAlert { background-color: #2a2a2a; border-radius: 8px; }
    .status-online { color: #4caf50; font-weight: 600; }
    .status-offline { color: #f44336; font-weight: 600; }
    .risk-badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.04em; text-transform: uppercase; }
</style>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body { font-family: 'Inter', sans-serif; }
    .rec-card { background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%); color: #ffffff !important; padding: 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 10px 25px rgba(31, 119, 180, 0.3); }
    .rec-card h3, .rec-card p, .rec-card strong { color: #ffffff !important; }
    .coach-msg { background-color: #f0f4f8; color: #1a1a1a !important; padding: 15px; border-radius: 15px 15px 15px 0px; border: 1px solid #d0d7de; margin-bottom: 10px; }
    .coach-msg b { color: #1a1a1a !important; }
    .user-msg { background-color: #1f77b4; color: #ffffff !important; padding: 15px; border-radius: 15px 15px 0px 15px; margin-bottom: 10px; text-align: right; }
    .status-online { color: #28a745; font-weight: 600; }
    .status-offline { color: #dc3545; font-weight: 600; }
    .risk-badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: 0.85rem; letter-spacing: 0.04em; text-transform: uppercase; color: #ffffff; }
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


# --- BANDIT (for feedback on template/intensity tier) ---
# The bandit is fed a proxy action_id derived from the recommended intensity
# so thumbs up/down still exercises ContextualBandit.update(). Mapping:
#   rest/recovery -> REST/LOW action_ids; light -> STRENGTH LOW;
#   moderate -> STRENGTH MEDIUM; hard -> STRENGTH HIGH.
_INTENSITY_TO_LEGACY = {
    "rest": ("REST", "NONE"),
    "recovery": ("RECOVERY", "LOW"),
    "light": ("STRENGTH", "LOW"),
    "moderate": ("STRENGTH", "MEDIUM"),
    "hard": ("STRENGTH", "HIGH"),
}


def _proxy_action_id(intensity: str) -> int:
    """Map Recommendation.today_decision.recommended_intensity to a legacy
    action_id so the bandit can still receive reward updates."""
    if not _RL_AVAILABLE:
        return 0
    if "local_action_space" not in st.session_state:
        st.session_state.local_action_space = ActionSpace()
        st.session_state.local_bandit = ContextualBandit(
            action_space=st.session_state.local_action_space
        )
    wtype, wintensity = _INTENSITY_TO_LEGACY.get(intensity, ("STRENGTH", "MEDIUM"))
    try:
        return st.session_state.local_action_space.get_action_id(wtype, wintensity, 30)
    except Exception:
        return 0


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/lightning-bolt.png", width=80)
    st.title("ProFit AI")
    st.caption("RL-Powered Performance Coach")

    st.divider()
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.session_state.user_id = user_id

    st.divider()
    st.subheader("System Status")
    if _RL_AVAILABLE:
        st.markdown(
            '<span class="status-online">RL Engine Active</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-offline">RL Engine Unavailable</span>',
            unsafe_allow_html=True,
        )

    if AI_COACH_ENABLED:
        st.markdown(
            '<span class="status-online">AI Coach Active</span>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<span class="status-offline">AI Coach Offline</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    if st.session_state.recommendation_history:
        latest = st.session_state.recommendation_history[-1]
        state = latest.get("state", {})
        st.subheader("Current State")
        if state.get("readiness_score") is not None:
            st.progress(
                state.get("readiness_score", 0) / 100,
                text=f"Readiness: {state.get('readiness_score', 0)}%",
            )
        if state.get("fatigue") is not None:
            st.progress(
                state.get("fatigue", 5) / 10,
                text=f"Fatigue: {state.get('fatigue', 5)}/10",
            )

    st.divider()
    st.subheader("Display Mode")
    dark_mode_toggle = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        st.rerun()

    st.divider()
    st.info("Agent v2.2: dual-layer recommender (risk-aware)")


# --- MAIN INTERFACE ---
st.title("Welcome back, Athlete.")
st.write(f"Today is {datetime.now().strftime('%A, %B %d, %Y')}")

tab_today, tab_analytics, tab_coach, tab_settings = st.tabs(
    ["Today's Protocol", "Performance Insights", "AI Coach Agent", "Settings"]
)


# ============================================================================
# TAB 1: TODAY'S PROTOCOL
# ============================================================================


def _pre_fill_defaults():
    """Pull the most recent uploaded/manual row to seed slider defaults."""
    for entry in reversed(st.session_state.recommendation_history):
        s = entry.get("state", {})
        if s:
            return s
    return {}


_RISK_COLOR = {
    "low": "#22c55e",
    "moderate": "#eab308",
    "elevated": "#f97316",
    "high": "#ef4444",
}


def _render_recommendation(rec):
    """Render a Recommendation pydantic model in the UI."""
    td = rec.today_decision
    color = _RISK_COLOR.get(td.risk_level, "#64748b")

    st.markdown(
        f"""
<div class="rec-card">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom: 8px;">
        <span class="risk-badge" style="background-color:{color};">{td.risk_level} risk</span>
        <span style="font-size:1.4rem; font-weight:800;">{td.recommended_intensity.upper()}</span>
    </div>
    <h3 style="margin:0 0 6px 0;">{td.primary_goal_today}</h3>
    <p style="font-style: italic; margin: 0;">"{td.why_today}"</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # Session plan 5-column layout
    st.subheader("Session Plan")
    sp = rec.session_plan
    cols = st.columns(5)
    for col, (label, items) in zip(
        cols,
        [
            ("Warmup", sp.warmup),
            ("Main Lifts", sp.main_lifts),
            ("Accessories", sp.accessories),
            ("Conditioning", sp.conditioning),
            ("Cooldown", sp.cooldown),
        ],
    ):
        with col:
            st.markdown(f"**{label}**")
            if items:
                for it in items:
                    st.markdown(f"- {it}")
            else:
                st.caption("-")

    # Exercise prescription table
    st.subheader("Exercise Prescription")
    if rec.exercise_prescription:
        rx_df = pd.DataFrame(
            [
                {
                    "Exercise": rx.name,
                    "Sets": rx.sets,
                    "Reps": rx.reps,
                    "Target RPE": rx.target_rpe,
                    "Load": rx.load_guidance,
                    "Substitution": rx.substitution_if_needed or "-",
                }
                for rx in rec.exercise_prescription
            ]
        )
        st.dataframe(rx_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No loaded exercises prescribed today.")

    st.info(f"Volume cap: {rec.volume_cap}")

    if rec.forbidden_or_not_recommended:
        st.error("**Forbidden / Not Recommended Today**")
        for item in rec.forbidden_or_not_recommended:
            st.markdown(f"- {item}")

    st.info(f"**Override option:** {rec.override_option}")

    if rec.stop_conditions:
        st.warning("**Stop Conditions**")
        for item in rec.stop_conditions:
            st.markdown(f"- {item}")

    if rec.data_gaps:
        st.caption("Missing inputs reduced confidence: " + ", ".join(rec.data_gaps))


with tab_today:
    col_input, col_rec = st.columns([1, 2])

    with col_input:
        st.subheader("Today's Inputs")

        defaults = _pre_fill_defaults()
        if defaults:
            st.caption("Pre-filled from your latest uploaded data")

        with st.form("today_request_form"):
            goal_priority = st.selectbox(
                "Goal priority",
                [
                    "health+strength",
                    "strength",
                    "hypertrophy",
                    "conditioning",
                    "recovery",
                ],
                index=0,
            )
            equipment = st.selectbox(
                "Equipment",
                ["full_gym", "home_basic", "bodyweight"],
                index=0,
            )
            time_budget_min = st.slider("Time budget (min)", 20, 120, 60, step=5)

            with st.expander("Wearable signals (optional)", expanded=bool(defaults)):
                st.caption(
                    "Leave unset if not available. Missing fields are treated as gaps."
                )
                use_readiness = st.checkbox(
                    "Include readiness",
                    value=defaults.get("readiness_score") is not None,
                )
                readiness_score = (
                    st.slider(
                        "Readiness", 0, 100, int(defaults.get("readiness_score") or 70)
                    )
                    if use_readiness
                    else None
                )

                use_sleep_score = st.checkbox(
                    "Include sleep score", value=defaults.get("sleep_score") is not None
                )
                sleep_score = (
                    st.slider(
                        "Sleep score", 0, 100, int(defaults.get("sleep_score") or 75)
                    )
                    if use_sleep_score
                    else None
                )

                use_sleep_hours = st.checkbox(
                    "Include sleep hours",
                    value=defaults.get("sleep_duration_hours") is not None
                    or defaults.get("sleep_hours") is not None,
                )
                sleep_hours = (
                    st.slider(
                        "Sleep hours",
                        0.0,
                        12.0,
                        float(
                            defaults.get("sleep_duration_hours")
                            or defaults.get("sleep_hours")
                            or 7.5
                        ),
                        step=0.1,
                    )
                    if use_sleep_hours
                    else None
                )

                use_hrv = st.checkbox(
                    "Include HRV", value=defaults.get("hrv") is not None
                )
                hrv = (
                    st.slider("HRV (ms)", 20, 100, int(defaults.get("hrv") or 50))
                    if use_hrv
                    else None
                )

                use_rhr = st.checkbox(
                    "Include resting HR", value=defaults.get("resting_hr") is not None
                )
                resting_hr = (
                    st.slider(
                        "Resting HR (bpm)",
                        40,
                        100,
                        int(defaults.get("resting_hr") or 60),
                    )
                    if use_rhr
                    else None
                )

                use_activity = st.checkbox(
                    "Include activity score",
                    value=defaults.get("activity_score") is not None,
                )
                activity_score = (
                    st.slider(
                        "Activity score",
                        0,
                        100,
                        int(defaults.get("activity_score") or 65),
                    )
                    if use_activity
                    else None
                )

            st.markdown("**Manual check-in**")
            fatigue = st.slider(
                "Fatigue (1=fresh, 10=wrecked)",
                1,
                10,
                int(defaults.get("fatigue") or 5),
            )
            motivation = st.slider("Motivation (1-10)", 1, 10, 6)

            with st.expander("Pain (0-10, zeros are dropped)", expanded=False):
                pain_keys = [
                    "left_knee",
                    "right_knee",
                    "lower_back",
                    "left_shoulder",
                    "right_shoulder",
                    "neck",
                ]
                pain_values = {}
                for key in pain_keys:
                    pain_values[key] = st.slider(
                        f"Pain — {key}", 0, 10, 0, key=f"pain_{key}"
                    )
                custom_pain_name = st.text_input(
                    "Custom pain location (optional)", value="", key="pain_custom_name"
                )
                custom_pain_val = st.slider(
                    "Custom pain level", 0, 10, 0, key="pain_custom_val"
                )
                if custom_pain_name.strip():
                    pain_values[custom_pain_name.strip().lower()] = custom_pain_val

            with st.expander("Soreness (0-10, zeros are dropped)", expanded=False):
                soreness_keys = ["legs", "upper_body", "core", "full_body"]
                soreness_values = {}
                for key in soreness_keys:
                    soreness_values[key] = st.slider(
                        f"Soreness — {key}", 0, 10, 0, key=f"sore_{key}"
                    )

            with st.expander("Recent training (last 0-5 sessions)", expanded=False):
                n_sessions = st.number_input(
                    "How many recent sessions?", 0, 5, 0, step=1
                )
                recent_rows = []
                for i in range(int(n_sessions)):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c1:
                        days_ago = st.number_input(
                            f"Days ago #{i+1}", 0, 30, i + 1, step=1, key=f"rt_days_{i}"
                        )
                    with c2:
                        sess_type = st.text_input(
                            f"Type #{i+1}", value="lower heavy", key=f"rt_type_{i}"
                        )
                    with c3:
                        rpe = st.slider(
                            f"RPE #{i+1}", 0.0, 10.0, 7.0, step=0.5, key=f"rt_rpe_{i}"
                        )
                    recent_rows.append(
                        {
                            "days_ago": int(days_ago),
                            "type": sess_type,
                            "rpe": float(rpe),
                        }
                    )

            override_preference = st.radio(
                "Override preference",
                ["follow_recommendation", "go_heavier_if_possible", "go_lighter"],
                index=0,
                horizontal=False,
            )

            submitted = st.form_submit_button(
                "Get Recommendation", use_container_width=True, type="primary"
            )

            if submitted:
                if not _RL_AVAILABLE:
                    st.error("RL engine not available.")
                else:
                    wearable = WearableData(
                        readiness_score=readiness_score,
                        sleep_score=sleep_score,
                        sleep_hours=sleep_hours,
                        hrv=hrv,
                        resting_hr=resting_hr,
                        activity_score=activity_score,
                    )
                    pain_clean = {k: v for k, v in pain_values.items() if v > 0}
                    soreness_clean = {k: v for k, v in soreness_values.items() if v > 0}
                    manual = ManualCheckIn(
                        fatigue=fatigue,
                        motivation=motivation,
                        pain=pain_clean,
                        soreness=soreness_clean,
                    )
                    recent_training = [RecentSession(**row) for row in recent_rows]

                    req = TodayRequest(
                        goal_priority=goal_priority,
                        equipment=equipment,
                        time_budget_min=time_budget_min,
                        wearable=wearable,
                        manual=manual,
                        recent_training=recent_training,
                        override_preference=override_preference,
                    )

                    with st.spinner("Building recommendation..."):
                        rec = recommend_today(req)

                    st.session_state.current_plan = rec
                    st.session_state.recommendation_history.append(
                        {
                            "timestamp": datetime.now().timestamp(),
                            "recommendation": rec.model_dump(),
                            "state": {
                                "readiness_score": readiness_score,
                                "sleep_score": sleep_score,
                                "sleep_duration_hours": sleep_hours,
                                "hrv": hrv,
                                "resting_hr": resting_hr,
                                "activity_score": activity_score,
                                "fatigue": fatigue,
                                "motivation": motivation,
                                "pain": pain_clean,
                                "soreness": soreness_clean,
                            },
                            "workout_type": rec.today_decision.recommended_intensity,
                            "intensity": rec.today_decision.recommended_intensity,
                            "duration_minutes": time_budget_min,
                        }
                    )
                    st.success("Recommendation ready.")

    with col_rec:
        st.subheader("Recommended Session")
        if st.session_state.current_plan is not None:
            _render_recommendation(st.session_state.current_plan)

            with st.expander("Log completion / feedback"):
                with st.form("feedback_form"):
                    completed = st.checkbox("Training completed", value=True)
                    rpe = st.slider("Workout RPE (1-10)", 1, 10, 7)
                    satisfaction = st.slider("Satisfaction (1-10)", 1, 10, 8)
                    thumbs = st.radio(
                        "Overall",
                        ["thumbs up", "thumbs down", "skipped"],
                        horizontal=True,
                    )

                    if st.form_submit_button("Submit Feedback"):
                        if _RL_AVAILABLE:
                            intensity = (
                                st.session_state.current_plan.today_decision.recommended_intensity
                            )
                            aid = _proxy_action_id(intensity)
                            if thumbs == "thumbs up":
                                reward = 0.5 + (satisfaction - 5) / 10
                            elif thumbs == "thumbs down":
                                reward = 0.1
                            else:
                                reward = 0.0
                            reward = max(0.0, min(1.0, reward))
                            try:
                                st.session_state.local_bandit.update(aid, reward)
                                st.success(f"Feedback logged. reward={reward:.2f}")
                            except Exception as e:
                                st.warning(f"Could not update bandit: {e}")
                        else:
                            st.warning("Feedback noted (bandit unavailable).")
        else:
            st.info("Fill the form on the left and click Get Recommendation.")


# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================
with tab_analytics:
    st.subheader("Performance Trends & System Insights")

    if st.session_state.recommendation_history:
        history_data = []
        for item in st.session_state.recommendation_history:
            rec_dict = item.get("recommendation", {}) or {}
            td = (
                rec_dict.get("today_decision", {}) if isinstance(rec_dict, dict) else {}
            )
            state = item.get("state", {})
            history_data.append(
                {
                    "Date": datetime.fromtimestamp(
                        item.get("timestamp", datetime.now().timestamp())
                    ),
                    "Workout": td.get("primary_goal_today")
                    or item.get("workout_type", "Unknown"),
                    "Intensity": td.get("recommended_intensity")
                    or item.get("intensity", "N/A"),
                    "Duration": item.get("duration_minutes", 0),
                    "Readiness": state.get("readiness_score") or 0,
                    "Sleep": state.get("sleep_score") or 0,
                    "HRV": state.get("hrv") or 0,
                    "RHR": state.get("resting_hr") or 0,
                    "Fatigue": state.get("fatigue") or 0,
                    "Activity": state.get("activity_score") or 0,
                    "Motivation": state.get("motivation") or 0,
                }
            )

        df = pd.DataFrame(history_data)
        df["Date_str"] = df["Date"].dt.strftime("%Y-%m-%d")

        st.subheader("7-Day Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            avg_readiness = df["Readiness"].tail(7).mean()
            st.metric("Avg Readiness", f"{avg_readiness:.1f}/100")
        with col2:
            avg_hrv = df["HRV"].tail(7).mean()
            st.metric("Avg HRV", f"{avg_hrv:.1f} ms")
        with col3:
            avg_sleep = df["Sleep"].tail(7).mean()
            st.metric("Avg Sleep Score", f"{avg_sleep:.1f}/100")
        with col4:
            total_duration = df["Duration"].tail(7).sum()
            st.metric("Total Training", f"{total_duration} min")

        st.divider()

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
                    "Motivation",
                ],
                default=["Readiness", "Sleep", "HRV", "Fatigue"],
                key="metrics_selector",
            )

        st.subheader("Health Metrics Over Time")
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

        st.subheader("Recent Sessions")
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

        csv = df.to_csv(index=False, encoding="utf-8")
        st.download_button(
            label="Download Full History (CSV)",
            data=csv,
            file_name=f"fitness_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No data yet. Get your first recommendation to start tracking.")


# ============================================================================
# TAB 3: AI COACH AGENT
# ============================================================================
with tab_coach:
    st.subheader("AI Coach Interface")

    if AI_COACH_ENABLED:
        st.caption("Powered by OpenAI with real-time context")
    else:
        st.warning("AI Coach is offline. Configure OPENAI_API_KEY in .env to enable.")

    if st.session_state.recommendation_history:
        with st.expander("Coach's view of your latest data"):
            latest = st.session_state.recommendation_history[-1]
            state = latest.get("state", {})
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Readiness", f"{state.get('readiness_score', 'N/A')}")
                st.metric("Sleep", f"{state.get('sleep_score', 'N/A')}")
            with c2:
                st.metric("HRV", f"{state.get('hrv', 'N/A')}")
                st.metric("RHR", f"{state.get('resting_hr', 'N/A')}")
            with c3:
                st.metric("Fatigue", f"{state.get('fatigue', 'N/A')}")
                st.metric("Activity", f"{state.get('activity_score', 'N/A')}")

    if (
        st.session_state.chat_history
        and st.session_state.chat_history[-1]["role"] == "user"
    ):
        user_prompt = st.session_state.chat_history[-1]["content"]

        if AI_COACH_ENABLED:
            with st.spinner("Coach thinking..."):
                try:
                    body_state = {}
                    if st.session_state.recommendation_history:
                        latest = st.session_state.recommendation_history[-1]
                        body_state = latest.get("state", {})

                    system_prompt = (
                        "You are an expert AI Fitness Coach with access to real-time health data. "
                        "Analyze readiness/sleep/HRV/HR/fatigue and provide supportive, data-driven guidance. "
                        "You are NOT a medical professional; recommend a doctor for serious symptoms."
                    )

                    context_parts = [f"User: {user_prompt}"]
                    if body_state:
                        context_parts.append("\n\nCurrent Health Data:")
                        for k in (
                            "readiness_score",
                            "sleep_score",
                            "hrv",
                            "resting_hr",
                            "fatigue",
                        ):
                            context_parts.append(f"- {k}: {body_state.get(k, 'N/A')}")

                    if st.session_state.current_plan is not None:
                        rec = st.session_state.current_plan
                        td = rec.today_decision
                        context_parts.append("\n\nCurrent Recommendation:")
                        context_parts.append(f"- Risk: {td.risk_level}")
                        context_parts.append(f"- Intensity: {td.recommended_intensity}")
                        context_parts.append(f"- Goal: {td.primary_goal_today}")

                    user_context = "\n".join(context_parts)

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
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": agent_response}
                    )
                    st.rerun()

                except Exception as e:
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"Error: {e}"}
                    )
                    st.rerun()
        else:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "AI Coach is offline."}
            )
            st.rerun()

    st.markdown(
        '<div style="max-height: 400px; overflow-y: auto; padding: 1rem; border: 1px solid #dee2e6; border-radius: 10px; background-color: #ffffff;">',
        unsafe_allow_html=True,
    )
    with st.container():
        for msg in st.session_state.chat_history:
            div_class = "user-msg" if msg["role"] == "user" else "coach-msg"
            label = "You" if msg["role"] == "user" else "Coach"
            st.markdown(
                f'<div class="{div_class}"><b>{label}:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    col_input, col_button = st.columns([4, 1])
    with col_input:
        prompt = st.text_input(
            "Ask your coach...",
            key="chat_input_text",
            placeholder="Ask about your plan, recovery, or motivation.",
            label_visibility="collapsed",
        )
    with col_button:
        send_button = st.button("Send", type="primary", use_container_width=True)

    if send_button and prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Explain My Plan", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "content": "Explain my current training plan"}
            )
            st.rerun()
    with col2:
        if st.button("Motivate Me", use_container_width=True):
            st.session_state.chat_history.append(
                {"role": "user", "content": "Give me some motivation"}
            )
            st.rerun()
    with col3:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# TAB 4: SETTINGS
# ============================================================================
with tab_settings:
    st.subheader("Settings & Configuration")

    st.subheader("Upload Health Data")
    st.markdown("Upload daily health data from Apple Watch or Oura Ring.")

    upload_tab1, upload_tab2 = st.tabs(["Manual Entry", "File Upload"])

    with upload_tab1:
        st.write("Manually input today's data from your devices:")
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
                "Upload Today's Data", use_container_width=True
            )

            if submit_manual:
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
                    f"Data uploaded. Total entries: {len(st.session_state.recommendation_history)}"
                )
                st.rerun()

    with upload_tab2:
        st.write("Upload CSV or JSON file with health data:")
        uploaded_file = st.file_uploader(
            "Choose a file", type=["csv", "json"], key="health_data_upload"
        )
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    import io

                    df_up = pd.read_csv(
                        io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                    )
                    st.success(f"Loaded {len(df_up)} rows from CSV")
                    st.dataframe(df_up.head())
                    if st.button("Import to History"):
                        for _, row in df_up.iterrows():
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
                        st.success(f"Imported {len(df_up)} entries.")
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
                        st.success("Imported successfully.")
                        st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")

        st.info(
            "CSV format example:\n"
            "readiness_score,sleep_score,hrv,resting_hr,activity_score,fatigue,workout_type\n"
            "85,90,55,58,75,3,Strength"
        )

    st.divider()

    st.subheader("User Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Display Name", value="Athlete", key="user_name_settings")
        st.number_input(
            "Age", min_value=18, max_value=100, value=30, key="user_age_settings"
        )
    with col2:
        st.number_input(
            "Weight (kg)",
            min_value=40,
            max_value=200,
            value=70,
            key="user_weight_settings",
        )
        st.number_input(
            "Height (cm)",
            min_value=140,
            max_value=220,
            value=175,
            key="user_height_settings",
        )

    st.divider()

    st.subheader("API & System Status")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("API Server", "Online" if API_BASE_URL else "Offline")
        st.code(f"{API_BASE_URL}", language="text")
    with c2:
        st.metric("AI Coach", "Active" if AI_COACH_ENABLED else "Offline")
        st.code(f"Model: {os.getenv('AGENT_MODEL', 'gpt-4')}", language="text")

    st.divider()

    st.subheader("Data Management")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Export History", use_container_width=True):
            if st.session_state.recommendation_history:
                import json

                data_json = json.dumps(
                    st.session_state.recommendation_history, indent=2, default=str
                )
                st.download_button(
                    label="Download JSON",
                    data=data_json,
                    file_name=f"fitness_history_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                )
            else:
                st.warning("No data to export")
    with c2:
        if st.button("Refresh Stats", use_container_width=True):
            st.rerun()
    with c3:
        if st.button("Clear History", use_container_width=True):
            st.session_state.recommendation_history = []
            st.session_state.chat_history = []
            st.session_state.current_plan = None
            st.success("History cleared.")
            st.rerun()

    st.divider()

    st.subheader("System Information")
    st.markdown(
        "- Recommendation Engine: dual-layer (risk scorer -> session planner -> exercise prescriber)\n"
        "- Feedback: ContextualBandit updates on a proxy action_id mapped from recommended_intensity\n"
        "- Model Serving: FastAPI with /recommend_v2 (TodayRequest -> Recommendation)\n"
        "- AI Coach: OpenAI with real-time health context"
    )
    st.caption(f"Total entries: {len(st.session_state.recommendation_history)}")
    st.caption(f"Chat messages: {len(st.session_state.chat_history)}")


# --- FOOTER ---
st.divider()
st.caption(
    "ProFit AI | RL-Powered Personalized Training | Research & Educational Use Only"
)
