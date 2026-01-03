"""
API endpoints for AI Coach Agent.

Provides REST API for agent interactions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import CoachAgent

app = FastAPI(title="AI Coach Agent API")

# Initialize agent (in production, use dependency injection)
agent = CoachAgent()


class ChatRequest(BaseModel):
    """Chat request model."""
    user_id: str
    message: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    message: str
    plan: Optional[Dict] = None
    tools_called: List[Dict] = []
    safety_alert: Optional[Dict] = None


@app.post("/agent/chat", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Chat with the AI Coach Agent.
    
    Args:
        request: Chat request with user_id and optional message
    
    Returns:
        Agent response with message, plan, and tool calls
    """
    try:
        response = agent.process_daily_coaching(
            user_id=request.user_id,
            user_message=request.message
        )
        
        return ChatResponse(
            message=response.get('message', ''),
            plan=response.get('plan'),
            tools_called=response.get('tools_called', []),
            safety_alert=response.get('safety_alert'),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/daily-plan/{user_id}")
async def get_daily_plan(user_id: str):
    """
    Get daily coaching plan and message.
    
    Args:
        user_id: User identifier
    
    Returns:
        Daily plan and coaching message
    """
    try:
        response = agent.process_daily_coaching(user_id=user_id)
        
        return {
            'plan': response.get('plan'),
            'message': response.get('message'),
            'tools_called': response.get('tools_called', []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai_coach_agent"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

