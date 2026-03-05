"""
Main API Server

Serves recommendations and handles user interactions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.recommendation.hybrid_recommender import HybridRecommender
from src.feature_store.feature_engineering import FeatureEngineer
from src.online_learning.loop import OnlineLearningLoop

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fitness Plan Optimization API", version="1.0.0")

# Initialize components
try:
    recommender = HybridRecommender(use_rl=True)
    learning_loop = OnlineLearningLoop(recommender)
    logger.info("✅ Components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize components: {e}")
    raise


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
        logger.info(f"Recommendation request for user: {request.user_id}")
        recommendation = learning_loop.process_daily_cycle(
            request.user_id,
            request.state
        )
        return recommendation
    except Exception as e:
        logger.error(f"Recommendation error: {e}", exc_info=True)
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
        logger.info(f"Feedback from user: {request.user_id}, action: {request.action_id}")
        reward = learning_loop.process_feedback(
            request.user_id,
            request.action_id,
            request.feedback
        )
        return {"reward": reward, "status": "updated"}
    except Exception as e:
        logger.error(f"Feedback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status of API and dependencies
    """
    try:
        # Check if components are initialized
        status = {
            "status": "healthy",
            "recommender": "initialized" if recommender else "not initialized",
            "learning_loop": "initialized" if learning_loop else "not initialized",
        }
        
        # Try a simple recommendation to verify functionality
        test_state = {
            "readiness_score": 75,
            "sleep_score": 80,
            "hrv": 50,
            "resting_hr": 60,
            "fatigue": 5,
        }
        test_rec = recommender.recommend(test_state)
        status["test_recommendation"] = "working"
        
        return status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting API server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
