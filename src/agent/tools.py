"""
Agent Tools - Functions that the AI Coach Agent can call.

These tools enable the agent to take actions in the system.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class AgentTools:
    """
    Tools available to the AI Coach Agent.
    
    These functions can be called by the LLM to perform actions.
    """
    
    def __init__(self, kafka_producer=None, plan_service=None):
        """
        Initialize agent tools.
        
        Args:
            kafka_producer: Kafka producer for event logging
            plan_service: Service for plan management
        """
        self.kafka_producer = kafka_producer
        self.plan_service = plan_service
    
    # ==================== Training Plan Tools ====================
    
    def adjust_plan(self, user_id: str, intensity: Optional[str] = None, 
                   volume: Optional[str] = None, rest_day: bool = False,
                   reason: str = "") -> Dict:
        """
        Adjust training plan based on current state.
        
        Args:
            user_id: User identifier
            intensity: "low", "moderate", "high" (optional)
            volume: "low", "moderate", "high" (optional)
            rest_day: Whether to schedule a rest day
            reason: Reason for adjustment
        
        Returns:
            Result dictionary
        """
        action = {
            'type': 'plan_adjustment',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'intensity': intensity,
            'volume': volume,
            'rest_day': rest_day,
            'reason': reason,
        }
        
        # Log event
        self._log_event('training.plan.adjusted', action)
        
        # In production, call plan_service to actually adjust plan
        if self.plan_service:
            # self.plan_service.adjust_plan(user_id, intensity, volume, rest_day)
            pass
        
        return {
            'success': True,
            'message': f"Plan adjusted: intensity={intensity}, volume={volume}, rest_day={rest_day}",
            'action': action,
        }
    
    def explain_plan(self, plan_id: str, rationale: str, 
                    key_features: List[str]) -> Dict:
        """
        Explain why a training plan was recommended.
        
        Args:
            plan_id: Plan identifier
            rationale: Explanation of the plan
            key_features: List of key features that influenced the decision
        
        Returns:
            Explanation dictionary
        """
        explanation = {
            'type': 'plan_explanation',
            'plan_id': plan_id,
            'timestamp': datetime.now().isoformat(),
            'rationale': rationale,
            'key_features': key_features,
        }
        
        return {
            'success': True,
            'explanation': explanation,
        }
    
    def generate_warmup_cooldown(self, user_id: str, plan_type: str,
                                 body_state: Dict) -> Dict:
        """
        Generate warmup and cooldown routines for the day.
        
        Args:
            user_id: User identifier
            plan_type: Type of training plan
            body_state: Current body state
        
        Returns:
            Warmup/cooldown routine
        """
        # Simple template-based generation
        # In production, could use LLM or rule-based system
        
        warmup = {
            'duration_minutes': 10,
            'exercises': [
                {'name': 'Light cardio', 'duration': '5 min'},
                {'name': 'Dynamic stretching', 'duration': '5 min'},
            ],
        }
        
        cooldown = {
            'duration_minutes': 10,
            'exercises': [
                {'name': 'Static stretching', 'duration': '5 min'},
                {'name': 'Deep breathing', 'duration': '5 min'},
            ],
        }
        
        routine = {
            'type': 'warmup_cooldown',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'warmup': warmup,
            'cooldown': cooldown,
        }
        
        return {
            'success': True,
            'routine': routine,
        }
    
    def set_micro_goal(self, user_id: str, goal: str, 
                       completion_criteria: str) -> Dict:
        """
        Set a small daily goal to improve adherence.
        
        Args:
            user_id: User identifier
            goal: Goal description
            completion_criteria: How to measure completion
        
        Returns:
            Goal dictionary
        """
        micro_goal = {
            'type': 'micro_goal',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'goal': goal,
            'completion_criteria': completion_criteria,
            'status': 'pending',
        }
        
        self._log_event('agent.micro_goal.set', micro_goal)
        
        return {
            'success': True,
            'goal': micro_goal,
        }
    
    # ==================== Emotional Support Tools ====================
    
    def mood_checkin(self, user_id: str, mood_score: int, 
                    stress_level: int, notes: str = "") -> Dict:
        """
        Collect mood and stress check-in.
        
        Args:
            user_id: User identifier
            mood_score: Mood score (1-5)
            stress_level: Stress level (1-5)
            notes: Additional notes
        
        Returns:
            Check-in result
        """
        checkin = {
            'type': 'mood_checkin',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'mood_score': mood_score,
            'stress_level': stress_level,
            'notes': notes,
        }
        
        self._log_event('agent.user.feedback', checkin)
        
        # Generate appropriate response based on mood
        if mood_score <= 2 or stress_level >= 4:
            response = "I notice you're feeling stressed or low today. Let's adjust today's plan to be more gentle. Remember, rest is productive too."
        elif mood_score >= 4:
            response = "Great to hear you're feeling good! Let's make the most of this energy."
        else:
            response = "Thanks for checking in. Let's see how you feel during today's session."
        
        return {
            'success': True,
            'checkin': checkin,
            'response': response,
        }
    
    def reflect_and_summarize(self, user_id: str, training_session: Dict,
                             mood_trends: List[Dict]) -> Dict:
        """
        Generate daily reflection and summary.
        
        Args:
            user_id: User identifier
            training_session: Today's training session data
            mood_trends: Recent mood trends
        
        Returns:
            Summary dictionary
        """
        summary = {
            'type': 'daily_summary',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'training': {
                'completed': training_session.get('completed', False),
                'exercises': training_session.get('exercises', []),
                'rpe': training_session.get('rpe'),
            },
            'mood_trends': mood_trends,
            'key_insights': [],
            'recommendations': [],
        }
        
        # Generate insights
        if training_session.get('completed'):
            summary['key_insights'].append("Completed today's training session")
        
        if mood_trends:
            avg_mood = sum(m.get('mood_score', 3) for m in mood_trends) / len(mood_trends)
            if avg_mood < 3:
                summary['recommendations'].append("Consider lighter training or rest day")
        
        self._log_event('agent.daily_summary', summary)
        
        return {
            'success': True,
            'summary': summary,
        }
    
    def breathing_prompt(self, user_id: str, duration_seconds: int = 60) -> Dict:
        """
        Provide a breathing/relaxation prompt.
        
        Args:
            user_id: User identifier
            duration_seconds: Duration of breathing exercise (60-120)
        
        Returns:
            Breathing prompt
        """
        prompt = {
            'type': 'breathing_prompt',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': duration_seconds,
            'instructions': (
                f"Take {duration_seconds} seconds for this breathing exercise:\n"
                "1. Inhale slowly for 4 counts\n"
                "2. Hold for 4 counts\n"
                "3. Exhale slowly for 4 counts\n"
                "4. Repeat until time is up"
            ),
        }
        
        return {
            'success': True,
            'prompt': prompt,
        }
    
    def motivational_message(self, user_id: str, style: str = "balanced",
                            context: str = "") -> Dict:
        """
        Generate a motivational message.
        
        Args:
            user_id: User identifier
            style: "short", "long", "humorous", "rational", "balanced"
            context: Context for the message
        
        Returns:
            Motivational message
        """
        messages = {
            'short': "You've got this! 💪",
            'long': "Remember, progress isn't always linear. Every session counts, even the tough ones. You're building resilience along with strength.",
            'humorous': "Time to turn those muscles into 'mussels' - strong and ready! 🦪",
            'rational': "Based on your data, you're in a good position to train today. Let's optimize this session.",
            'balanced': "You've been consistent with your training. Today is another opportunity to move forward, even if it's just a small step.",
        }
        
        message = {
            'type': 'motivational_message',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'style': style,
            'message': messages.get(style, messages['balanced']),
            'context': context,
        }
        
        return {
            'success': True,
            'message': message,
        }
    
    # ==================== System Tools ====================
    
    def log_event(self, event_type: str, payload: Dict) -> Dict:
        """
        Log an event to Kafka.
        
        Args:
            event_type: Type of event
            payload: Event payload
        
        Returns:
            Logging result
        """
        return self._log_event(event_type, payload)
    
    def _log_event(self, event_type: str, payload: Dict) -> Dict:
        """Internal event logging."""
        event = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'payload': payload,
        }
        
        # Send to Kafka if producer available
        if self.kafka_producer:
            try:
                topic = self._get_topic_for_event_type(event_type)
                self.kafka_producer.send(topic, value=json.dumps(event))
                self.kafka_producer.flush()
            except Exception as e:
                print(f"Error logging event to Kafka: {e}")
        
        return {
            'success': True,
            'event': event,
        }
    
    def _get_topic_for_event_type(self, event_type: str) -> str:
        """Map event type to Kafka topic."""
        topic_mapping = {
            'training.plan.adjusted': 'training.plan.served',
            'agent.user.feedback': 'training.user.feedback',
            'agent.daily_summary': 'agent.conversation.events',
            'agent.micro_goal.set': 'agent.conversation.events',
        }
        return topic_mapping.get(event_type, 'agent.conversation.events')
    
    def request_more_info(self, user_id: str, question: str,
                         context: str = "") -> Dict:
        """
        Request more information from the user.
        
        Args:
            user_id: User identifier
            question: Question to ask
            context: Context for the question
        
        Returns:
            Request dictionary
        """
        request = {
            'type': 'info_request',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'context': context,
        }
        
        return {
            'success': True,
            'request': request,
        }
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        Get tool definitions for LLM function calling.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        return [
            {
                'type': 'function',
                'function': {
                    'name': 'adjust_plan',
                    'description': 'Adjust training plan intensity, volume, or schedule a rest day',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'intensity': {'type': 'string', 'enum': ['low', 'moderate', 'high']},
                            'volume': {'type': 'string', 'enum': ['low', 'moderate', 'high']},
                            'rest_day': {'type': 'boolean'},
                            'reason': {'type': 'string'},
                        },
                        'required': ['user_id'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'explain_plan',
                    'description': 'Explain why a training plan was recommended',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'plan_id': {'type': 'string'},
                            'rationale': {'type': 'string'},
                            'key_features': {'type': 'array', 'items': {'type': 'string'}},
                        },
                        'required': ['plan_id', 'rationale'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'mood_checkin',
                    'description': 'Collect mood and stress check-in from user',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'mood_score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                            'stress_level': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                            'notes': {'type': 'string'},
                        },
                        'required': ['user_id', 'mood_score', 'stress_level'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'set_micro_goal',
                    'description': 'Set a small daily goal to improve training adherence',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'user_id': {'type': 'string'},
                            'goal': {'type': 'string'},
                            'completion_criteria': {'type': 'string'},
                        },
                        'required': ['user_id', 'goal', 'completion_criteria'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'log_event',
                    'description': 'Log an event to the system for feedback and learning',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'event_type': {'type': 'string'},
                            'payload': {'type': 'object'},
                        },
                        'required': ['event_type', 'payload'],
                    },
                },
            },
        ]

