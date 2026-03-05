"""
AI Coach Agent - Main agent implementation.

This agent uses LLM with tool calling to:
- Explain training plan recommendations
- Collect user feedback
- Trigger actions (plan adjustments, summaries, etc.)
- Enable closed-loop learning
"""

from typing import Dict, List, Optional, Any
import json
import os
from .state import DailyState, DailyStateBuilder
from .safety import SafetyGuardrails, SafetyCheckResult
from .tools import AgentTools
from .llm_client import LLMClient


class CoachAgent:
    """
    AI Coach Agent that provides personalized coaching through LLM + tools.
    
    Architecture:
    - Safety Gate: Hard rules to prevent dangerous recommendations
    - Recommendation Engine: Bandit/heuristic model (external)
    - LLM Agent: Explains, collects feedback, triggers actions
    """
    
    def __init__(self, llm_client=None, kafka_producer=None, plan_service=None, provider="openai"):
        """
        Initialize Coach Agent.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.) - if None, creates default
            kafka_producer: Kafka producer for event logging
            plan_service: Service for plan management
            provider: LLM provider ("openai" or "anthropic")
        """
        if llm_client is None:
            try:
                self.llm_client = LLMClient(provider=provider)
            except Exception as e:
                raise RuntimeError(f"Failed to initialize LLM client: {e}")
        else:
            self.llm_client = llm_client
        
        self.state_builder = DailyStateBuilder()
        self.safety = SafetyGuardrails()
        self.tools = AgentTools(kafka_producer, plan_service)
        
        # System prompt for the agent
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for the LLM."""
        return """You are an AI Fitness Coach Agent that helps users optimize their training plans.

Your role:
1. Explain training plan recommendations in natural language
2. Collect user feedback (RPE, mood, stress, pain)
3. Trigger appropriate actions (plan adjustments, recovery, summaries)
4. Provide emotional support and motivation when appropriate

IMPORTANT BOUNDARIES:
- You are NOT a medical professional. Never provide medical advice or diagnoses.
- If you detect dangerous signals (chest pain, dizziness, severe discomfort, abnormal heart rate), 
  immediately escalate to safety alert and recommend consulting a healthcare professional.
- Focus on fitness coaching, not mental health therapy (though you can provide encouragement).
- Always respect user preferences and injury history.

Available tools:
- adjust_plan: Adjust training plan intensity/volume or schedule rest day
- explain_plan: Explain why a plan was recommended
- mood_checkin: Collect mood and stress feedback
- set_micro_goal: Set small daily goals for adherence
- log_event: Log events for system learning
- request_more_info: Ask user for missing information

Be conversational, supportive, and data-driven. Use the user's body state data to inform your recommendations."""
    
    def process_daily_coaching(self, user_id: str, user_message: Optional[str] = None) -> Dict:
        """
        Process daily coaching interaction.
        
        Args:
            user_id: User identifier
            user_message: Optional user message
        
        Returns:
            Coaching response dictionary
        """
        # Build daily state
        state = self.state_builder.build_state(user_id)
        state = self.state_builder.update_from_feature_store(state)
        
        # Safety check
        safety_result = self.safety.check_state(state)
        if not safety_result.is_safe:
            return self._handle_safety_issue(state, safety_result)
        
        # Get recommended plan (from external recommendation engine)
        recommended_plan = self._get_recommended_plan(user_id, state)
        
        # Check plan safety
        plan_safety = self.safety.check_plan_safety(recommended_plan, state)
        if not plan_safety.is_safe:
            # Adjust plan based on safety
            adjusted_plan = self._adjust_plan_for_safety(recommended_plan, plan_safety)
            recommended_plan = adjusted_plan
        
        # Generate agent response
        if user_message:
            # Interactive conversation
            response = self._generate_conversational_response(
                user_id, user_message, state, recommended_plan
            )
        else:
            # Daily proactive message
            response = self._generate_daily_message(user_id, state, recommended_plan)
        
        return response
    
    def _get_recommended_plan(self, user_id: str, state: DailyState) -> Dict:
        """
        Get recommended plan from recommendation engine.
        
        In production, this would call the contextual bandits model.
        """
        # Placeholder - in production, call actual recommendation service
        return {
            'plan_id': f'plan_{user_id}_{state.date}',
            'intensity': 'moderate',
            'volume': 'moderate',
            'exercises': [
                {'name': 'Bench Press', 'sets': 4, 'reps': 8},
                {'name': 'Squats', 'sets': 3, 'reps': 10},
            ],
            'rationale': 'Based on your current body state and recovery status',
        }
    
    def _adjust_plan_for_safety(self, plan: Dict, safety_result: SafetyCheckResult) -> Dict:
        """Adjust plan based on safety check result."""
        adjusted = plan.copy()
        
        if safety_result.recommended_action == 'rest_day_or_light_activity':
            adjusted['intensity'] = 'low'
            adjusted['volume'] = 'low'
        elif safety_result.recommended_action == 'reduce_intensity':
            if adjusted['intensity'] == 'high':
                adjusted['intensity'] = 'moderate'
            elif adjusted['intensity'] == 'moderate':
                adjusted['intensity'] = 'low'
        
        return adjusted
    
    def _generate_daily_message(self, user_id: str, state: DailyState, 
                                plan: Dict) -> Dict:
        """
        Generate daily proactive coaching message.
        
        Args:
            user_id: User identifier
            state: Daily state
            plan: Recommended plan
        
        Returns:
            Response dictionary
        """
        # Build context for LLM
        context = {
            'state_summary': state.to_natural_language(),
            'plan': plan,
            'safety_notes': [],
        }
        
        # If using LLM
        if self.llm_client:
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': self._build_daily_prompt(context)},
            ]
            
            # Call LLM with tool calling
            response = self.llm_client.chat.completions.create(
                model="gpt-4",  # or your preferred model
                messages=messages,
                tools=self.tools.get_tool_definitions(),
                tool_choice="auto",
            )
            
            return self._process_llm_response(response, user_id, state, plan)
        else:
            # Fallback: rule-based response
            return self._generate_rule_based_message(state, plan)
    
    def _generate_conversational_response(self, user_id: str, user_message: str,
                                         state: DailyState, plan: Dict) -> Dict:
        """Generate conversational response to user message."""
        if self.llm_client:
            messages = [
                {'role': 'system', 'content': self.system_prompt},
                {'role': 'user', 'content': f"User state: {state.to_natural_language()}\n\nUser message: {user_message}"},
            ]
            
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=self.tools.get_tool_definitions(),
                tool_choice="auto",
            )
            
            return self._process_llm_response(response, user_id, state, plan)
        else:
            return {
                'message': "I understand. Let me help you with that.",
                'tools_called': [],
            }
    
    def _process_llm_response(self, llm_response, user_id: str, 
                               state: DailyState, plan: Dict) -> Dict:
        """Process LLM response and execute tool calls."""
        message = llm_response.choices[0].message
        tools_called = []
        
        # Execute tool calls if any
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Call appropriate tool
                if function_name == 'adjust_plan':
                    result = self.tools.adjust_plan(user_id=user_id, **function_args)
                    tools_called.append({'tool': function_name, 'result': result})
                elif function_name == 'explain_plan':
                    result = self.tools.explain_plan(plan_id=plan['plan_id'], **function_args)
                    tools_called.append({'tool': function_name, 'result': result})
                elif function_name == 'mood_checkin':
                    result = self.tools.mood_checkin(user_id=user_id, **function_args)
                    tools_called.append({'tool': function_name, 'result': result})
                elif function_name == 'set_micro_goal':
                    result = self.tools.set_micro_goal(user_id=user_id, **function_args)
                    tools_called.append({'tool': function_name, 'result': result})
                elif function_name == 'log_event':
                    result = self.tools.log_event(**function_args)
                    tools_called.append({'tool': function_name, 'result': result})
        
        return {
            'message': message.content,
            'tools_called': tools_called,
            'plan': plan,
        }
    
    def _build_daily_prompt(self, context: Dict) -> str:
        """Build prompt for daily message generation."""
        return f"""Generate a daily coaching message for the user.

Current body state: {context['state_summary']}
Recommended plan: {json.dumps(context['plan'], indent=2)}

Provide:
1. A brief explanation of why this plan fits their current state
2. Motivation and encouragement
3. Any adjustments needed based on their state
4. A small daily goal to improve adherence

Use tools if you need to adjust the plan or set goals."""
    
    def _generate_rule_based_message(self, state: DailyState, plan: Dict) -> Dict:
        """Generate rule-based message (fallback when LLM not available)."""
        message_parts = []
        
        # Explain plan
        if state.readiness_score and state.readiness_score < 60:
            message_parts.append("Your readiness score is lower today, so I've adjusted your plan to be lighter.")
        else:
            message_parts.append("Based on your current body state, here's your recommended training plan.")
        
        # Add plan details
        message_parts.append(f"Today's plan: {plan.get('intensity', 'moderate')} intensity training.")
        
        # Motivation
        if state.motivation_level and state.motivation_level >= 7:
            message_parts.append("Great to see your high motivation! Let's make the most of it.")
        elif state.motivation_level and state.motivation_level <= 4:
            message_parts.append("I understand if you're not feeling super motivated today. Even a light session helps.")
        
        return {
            'message': " ".join(message_parts),
            'tools_called': [],
            'plan': plan,
        }
    
    def _handle_safety_issue(self, state: DailyState, 
                            safety_result: SafetyCheckResult) -> Dict:
        """Handle safety issues."""
        alert = self.safety.escalate_safety_alert(state, safety_result.message)
        
        # Log safety alert
        self.tools.log_event('safety.alerts', alert)
        
        return {
            'message': safety_result.message + "\n\n" + alert['message'],
            'safety_alert': alert,
            'tools_called': [],
            'plan': {'intensity': 'rest', 'volume': 'none'},
        }

