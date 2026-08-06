"""
Main API Server

Serves recommendations and handles user interactions.

Dual-layer contract (Stage-2):
    - POST /recommend_v2 accepts a `TodayRequest` (the new structured input)
      and returns a `Recommendation` (today_decision + session_plan +
      exercise_prescription + derived fields).
    - POST /recommend is the legacy endpoint; it now adapts the old dict-state
      body to a `TodayRequest` and calls the same recommender, returning the
      legacy-shaped dict so existing clients/tests keep working.
    - Both paths go through `recommend_today` from `src.recommendation`.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommendation import recommend_today  # noqa: E402
from src.recommendation.hybrid_recommender import HybridRecommender  # noqa: E402
from src.online_learning.loop import OnlineLearningLoop  # noqa: E402
from src.serving.feature_service import FeatureService  # noqa: E402
from src.serving.history_store import (  # noqa: E402
    ParquetHistoryStore,
    SyntheticHistoryStore,
)
from src.serving.policy_service import PolicyService  # noqa: E402
from src.validation.schemas import (  # noqa: E402
    Recommendation,
    TodayRequest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fitness Plan Optimization API", version="2.0.0")

# Initialize components
try:
    recommender = HybridRecommender(use_rl=True)
    learning_loop = OnlineLearningLoop(recommender)
    # The RL serving path: shared feature transform (Redis-cached rolling
    # block) + safety gate + the trained NeuralLinear checkpoint.
    feature_service = FeatureService()
    policy_service = PolicyService()

    # History is only read on a cache miss, to rebuild the rolling block.
    # Falls back to synthetic history when no table is configured, so a local
    # run exercises the real recompute cost instead of silently skipping it.
    import os as _os

    _history_path = _os.getenv("HISTORY_TABLE_PATH")
    history_store = (
        ParquetHistoryStore(Path(_history_path))
        if _history_path
        else SyntheticHistoryStore()
    )
    logger.info("Components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise


class RecommendationRequest(BaseModel):
    """Legacy request for recommendation (dict-shaped state)."""

    user_id: str
    state: Dict


class FeedbackRequest(BaseModel):
    """Feedback request."""

    user_id: str
    action_id: int
    feedback: Dict


class RLRecommendRequest(BaseModel):
    """Request for the contextual-bandit serving path."""

    user_id: str
    signals: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Raw body-state fields for today (readiness_score, hrv, "
            "sleep_duration_hours, fatigue, profile attributes...). Rolling "
            "window features are filled from the feature cache when absent."
        ),
    )
    explore: bool = Field(
        default=True,
        description=(
            "Sample from the posterior (Thompson Sampling). Set false only for "
            "deterministic replay — a greedy bandit stops learning."
        ),
    )


class RLFeedbackRequest(BaseModel):
    """Observed reward for a previously served RL recommendation."""

    user_id: str
    action_id: int
    reward: float = Field(ge=-1.0, le=1.0)
    signals: Dict[str, Any] = Field(default_factory=dict)


@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    """
    Legacy recommendation endpoint (kept for backward compatibility).

    Accepts a dict-shaped state, runs the same dual-layer engine, and returns
    the legacy dict shape so existing clients keep working unchanged. The
    embedded full Recommendation is available under `recommendation` for
    incremental migration.
    """
    try:
        logger.info(f"Recommendation request for user: {request.user_id}")
        recommendation = learning_loop.process_daily_cycle(
            request.user_id, request.state
        )
        return recommendation
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend_v2", response_model=Recommendation)
async def get_recommendation_v2(request: TodayRequest) -> Recommendation:
    """
    New dual-layer recommendation endpoint.

    Request body: TodayRequest (goal, equipment, time budget, wearable signals,
    manual check-in, recent training, optional override preference).

    Response body: Recommendation (today_decision, session_plan,
    exercise_prescription, volume_cap, forbidden_or_not_recommended,
    override_option, stop_conditions, data_gaps).
    """
    try:
        return recommend_today(request)
    except Exception as e:
        logger.error(f"Recommendation v2 error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback.

    Args:
        request: User ID, action ID, and feedback

    Returns:
        Computed reward
    """
    try:
        logger.info(
            f"Feedback from user: {request.user_id}, action: {request.action_id}"
        )
        reward = learning_loop.process_feedback(
            request.user_id, request.action_id, request.feedback
        )
        return {"reward": reward, "status": "updated"}
    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend_rl")
async def recommend_rl(request: RLRecommendRequest):
    """
    Contextual-bandit recommendation.

    The path is: request signals -> shared feature transform (rolling block
    served from the Redis cache when warm) -> safety gate -> Thompson Sampling
    over the gated action set.  Each stage is the same code the offline
    benchmark runs, which is what keeps the served policy and the measured
    policy identical.
    """
    try:
        fetch = feature_service.build_features(
            request.user_id,
            request.signals,
            # Lazy: only read on a cache miss.
            history=lambda: history_store.get(request.user_id),
        )
        decision = policy_service.recommend(
            request.signals, features=fetch.features, explore=request.explore
        )
        payload = decision.to_dict()
        payload["features"] = {
            "cache_hit": fetch.cache_hit,
            "source": fetch.rolling_source,
            "build_ms": round(fetch.elapsed_ms, 3),
        }
        return payload
    except Exception as e:
        logger.error(f"RL recommendation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback_rl")
async def feedback_rl(request: RLFeedbackRequest):
    """
    Fold an observed reward back into the bandit posterior.

    The features are rebuilt from the same signals rather than trusting a
    client-supplied vector: accepting model inputs from the caller would let a
    malformed client poison the posterior.
    """
    try:
        fetch = feature_service.build_features(
            request.user_id,
            request.signals,
            # Lazy: only read on a cache miss.
            history=lambda: history_store.get(request.user_id),
        )
        policy_service.update(fetch.features, request.action_id, request.reward)
        return {"status": "updated", "action_id": request.action_id}
    except Exception as e:
        logger.error(f"RL feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Status of API and dependencies
    """
    try:
        status = {
            "status": "healthy",
            "recommender": "initialized" if recommender else "not initialized",
            "learning_loop": "initialized" if learning_loop else "not initialized",
        }

        test_state = {
            "readiness_score": 75,
            "sleep_score": 80,
            "hrv": 50,
            "resting_hr": 60,
            "fatigue": 5,
        }
        recommender.recommend(test_state)
        status["test_recommendation"] = "working"
        status["feature_cache"] = feature_service.stats()
        status["policy"] = policy_service.stats()["model"]

        return status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
