"""
Training Log Data Collection Module

This module provides functions to log and manage training session data.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import os
import json


class TrainingLogger:
    """Logger for training sessions and subjective feedback."""
    
    def __init__(self, log_file: str = "data/raw/training_logs/sessions.csv"):
        """
        Initialize training logger.
        
        Args:
            log_file: Path to CSV file for storing logs
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize CSV if it doesn't exist
        if not os.path.exists(log_file):
            self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create CSV file with headers."""
        columns = [
            'date',
            'exercise_name',
            'sets',
            'reps',
            'weight_kg',
            'rest_seconds',
            'rpe',
            'fatigue',
            'motivation',
            'muscle_soreness',
            'notes',
            'workout_type',
            'total_duration_minutes',
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.log_file, index=False)
    
    def log_exercise(
        self,
        exercise_name: str,
        sets: int,
        reps: int,
        weight_kg: float,
        rest_seconds: int = 60,
        rpe: Optional[int] = None,
        notes: str = "",
    ) -> Dict:
        """
        Log a single exercise.
        
        Args:
            exercise_name: Name of the exercise
            sets: Number of sets
            reps: Number of repetitions per set
            weight_kg: Weight in kilograms
            rest_seconds: Rest time between sets in seconds
            rpe: Rate of Perceived Exertion (1-10)
            notes: Additional notes
        
        Returns:
            Dictionary with exercise data
        """
        return {
            'exercise_name': exercise_name,
            'sets': sets,
            'reps': reps,
            'weight_kg': weight_kg,
            'rest_seconds': rest_seconds,
            'rpe': rpe,
            'notes': notes,
        }
    
    def log_session(
        self,
        exercises: List[Dict],
        fatigue: int,
        motivation: int,
        muscle_soreness: Optional[int] = None,
        workout_type: str = "",
        total_duration_minutes: Optional[int] = None,
        notes: str = "",
    ):
        """
        Log a complete training session.
        
        Args:
            exercises: List of exercise dictionaries from log_exercise()
            fatigue: Fatigue level (1-10)
            motivation: Motivation level (1-10)
            muscle_soreness: Muscle soreness level (1-10)
            workout_type: Type of workout (e.g., "Push", "Pull", "Legs", "Cardio")
            total_duration_minutes: Total workout duration in minutes
            notes: Additional session notes
        """
        session_records = []
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for exercise in exercises:
            record = {
                'date': date,
                'exercise_name': exercise.get('exercise_name', ''),
                'sets': exercise.get('sets', 0),
                'reps': exercise.get('reps', 0),
                'weight_kg': exercise.get('weight_kg', 0),
                'rest_seconds': exercise.get('rest_seconds', 60),
                'rpe': exercise.get('rpe'),
                'fatigue': fatigue,
                'motivation': motivation,
                'muscle_soreness': muscle_soreness,
                'notes': notes,
                'workout_type': workout_type,
                'total_duration_minutes': total_duration_minutes,
            }
            session_records.append(record)
        
        # Append to CSV
        df_new = pd.DataFrame(session_records)
        df_existing = pd.read_csv(self.log_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(self.log_file, index=False)
        
        print(f"Logged {len(exercises)} exercises for session on {date}")
    
    def get_recent_sessions(self, days: int = 7) -> pd.DataFrame:
        """
        Get recent training sessions.
        
        Args:
            days: Number of days to look back
        
        Returns:
            DataFrame with recent sessions
        """
        df = pd.read_csv(self.log_file)
        df['date'] = pd.to_datetime(df['date'])
        
        cutoff_date = datetime.now() - pd.Timedelta(days=days)
        recent_df = df[df['date'] >= cutoff_date]
        
        return recent_df
    
    def get_session_summary(self, days: int = 30) -> Dict:
        """
        Get summary statistics for recent sessions.
        
        Args:
            days: Number of days to analyze
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.get_recent_sessions(days)
        
        if df.empty:
            return {}
        
        summary = {
            'total_sessions': df['date'].nunique(),
            'total_exercises': len(df),
            'avg_fatigue': df['fatigue'].mean(),
            'avg_motivation': df['motivation'].mean(),
            'avg_muscle_soreness': df['muscle_soreness'].mean() if 'muscle_soreness' in df.columns else None,
            'workout_types': df['workout_type'].value_counts().to_dict(),
            'most_common_exercises': df['exercise_name'].value_counts().head(10).to_dict(),
        }
        
        return summary


def create_training_template():
    """
    Create a template for manual training log entry.
    Useful for quick logging via command line or scripts.
    """
    logger = TrainingLogger()
    
    print("Training Session Logger")
    print("=" * 50)
    
    # Get session-level info
    workout_type = input("Workout type (Push/Pull/Legs/Cardio): ").strip()
    fatigue = int(input("Fatigue level (1-10): "))
    motivation = int(input("Motivation level (1-10): "))
    muscle_soreness = int(input("Muscle soreness (1-10): "))
    duration = int(input("Total duration (minutes): "))
    
    exercises = []
    print("\nEnter exercises (press Enter with empty exercise name to finish):")
    
    while True:
        ex_name = input("Exercise name: ").strip()
        if not ex_name:
            break
        
        sets = int(input("  Sets: "))
        reps = int(input("  Reps: "))
        weight = float(input("  Weight (kg): "))
        rest = int(input("  Rest (seconds): "))
        rpe = input("  RPE (1-10, optional): ").strip()
        rpe = int(rpe) if rpe else None
        
        exercise = logger.log_exercise(
            exercise_name=ex_name,
            sets=sets,
            reps=reps,
            weight_kg=weight,
            rest_seconds=rest,
            rpe=rpe,
        )
        exercises.append(exercise)
    
    notes = input("\nAdditional notes: ").strip()
    
    logger.log_session(
        exercises=exercises,
        fatigue=fatigue,
        motivation=motivation,
        muscle_soreness=muscle_soreness,
        workout_type=workout_type,
        total_duration_minutes=duration,
        notes=notes,
    )
    
    print("\nSession logged successfully!")


if __name__ == "__main__":
    # Example usage
    logger = TrainingLogger()
    
    # Log a sample session
    exercises = [
        logger.log_exercise("Bench Press", sets=4, reps=8, weight_kg=80, rpe=8),
        logger.log_exercise("Overhead Press", sets=3, reps=10, weight_kg=50, rpe=7),
        logger.log_exercise("Tricep Dips", sets=3, reps=12, weight_kg=0, rpe=6),
    ]
    
    logger.log_session(
        exercises=exercises,
        fatigue=6,
        motivation=8,
        muscle_soreness=5,
        workout_type="Push",
        total_duration_minutes=60,
        notes="Good session, felt strong",
    )
    
    # Get summary
    summary = logger.get_session_summary(days=7)
    print("\nRecent Session Summary:")
    print(json.dumps(summary, indent=2))

