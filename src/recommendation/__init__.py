"""
Recommendation System Module.

Public surface:
    recommend_today(request: TodayRequest) -> Recommendation

Both the Streamlit UI and the FastAPI server import `recommend_today` from
this module so they share one code path.
"""

from src.recommendation.hybrid_recommender import recommend as recommend_today

__all__ = ["recommend_today"]
