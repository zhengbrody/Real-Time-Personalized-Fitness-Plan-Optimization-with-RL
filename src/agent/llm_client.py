"""
LLM Client wrapper for OpenAI and Anthropic.
"""

import os
from typing import Optional, Dict, List, Any
from src.config import OPENAI_API_KEY, ANTHROPIC_API_KEY, AGENT_MODEL


class LLMClient:
    """Unified LLM client for OpenAI and Anthropic."""
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize LLM client.
        
        Args:
            provider: "openai" or "anthropic"
            model: Model name (defaults to AGENT_MODEL)
        """
        self.provider = provider.lower()
        self.model = model or AGENT_MODEL
        
        if self.provider == "openai":
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                raise ImportError("openai package not installed. Run: pip install openai")
        
        elif self.provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            try:
                from anthropic import Anthropic
                self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
            except ImportError:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Get chat completion from LLM.
        
        Args:
            messages: List of message dicts with "role" and "content"
            tools: List of tool definitions for function calling
            tool_choice: "auto", "none", or specific tool
            temperature: Sampling temperature
        
        Returns:
            Response dict with "content" and optionally "tool_calls"
        """
        try:
            if self.provider == "openai":
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice or "auto"
                
                response = self.client.chat.completions.create(**kwargs)
                
                result = {
                    "content": response.choices[0].message.content,
                    "role": response.choices[0].message.role,
                }
                
                if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    result["tool_calls"] = [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in response.choices[0].message.tool_calls
                    ]
                
                return result
            
            elif self.provider == "anthropic":
                # Anthropic API implementation
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 4096,
                }
                
                if tools:
                    kwargs["tools"] = tools
                
                response = self.client.messages.create(**kwargs)
                
                result = {
                    "content": response.content[0].text if response.content else "",
                    "role": "assistant",
                }
                
                return result
        
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {str(e)}")

