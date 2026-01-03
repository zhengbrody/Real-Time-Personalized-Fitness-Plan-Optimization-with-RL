"""
Main API Server

Serves recommendations and handles user interactions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommendation.hybrid_recommender import HybridRecommender
from src.feature_store.feature_engineering import FeatureEngineer
from src.online_learning.loop import OnlineLearningLoop

app = FastAPI(title="Fitness Plan Optimization API")

# Initialize components
recommender = HybridRecommender(use_rl=True)
learning_loop = OnlineLearningLoop(recommender)


class RecommendationRequest(BaseModel):
    """Request for recommendation."""
    user_id: str
    state: Dict


class FeedbackRequest(BaseModel):
    """Feedback request."""
    user_id: str
    action_id: int
    feedback: Dict


@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    """
    Get training plan recommendation.
    
    Args:
        request: User ID and current state
    
    Returns:
        Recommended training plan
    """
    try:
        recommendation = learning_loop.process_daily_cycle(
            request.user_id,
            request.state
        )
        return recommendation
    except Exception as e:
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
        reward = learning_loop.process_feedback(
            request.user_id,
            request.action_id,
            request.feedback
        )
        return {"reward": reward, "status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

