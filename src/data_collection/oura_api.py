"""
Oura Ring API Data Collection Module

This module provides functions to extract data from Oura Ring via Oura API v2.
"""

from python_oura import OuraClientPersonalV2
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

load_dotenv()


class OuraDataCollector:
    """Collector for Oura Ring data via API."""
    
    def __init__(self, personal_access_token: Optional[str] = None):
        """
        Initialize Oura API client.
        
        Args:
            personal_access_token: Oura Personal Access Token.
                                   If None, reads from OURA_TOKEN environment variable.
        """
        token = personal_access_token or os.getenv('OURA_TOKEN')
        if not token:
            raise ValueError("Oura token required. Set OURA_TOKEN environment variable or pass token directly.")
        
        self.client = OuraClientPersonalV2(personal_access_token=token)
    
    def get_sleep_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch sleep data from Oura API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with sleep metrics
        """
        try:
            sleep_data = self.client.sleep_summary(start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error fetching sleep data: {e}")
            return pd.DataFrame()
        
        records = []
        for day in sleep_data.get('data', []):
            records.append({
                'date': day.get('day'),
                'sleep_score': day.get('score'),
                'total_sleep_duration': day.get('total_sleep_duration'),
                'deep_sleep_duration': day.get('deep_sleep_duration'),
                'rem_sleep_duration': day.get('rem_sleep_duration'),
                'light_sleep_duration': day.get('light_sleep_duration'),
                'sleep_efficiency': day.get('efficiency'),
                'resting_heart_rate': day.get('resting_heart_rate'),
                'hrv': day.get('hrv'),
                'sleep_onset_latency': day.get('sleep_latency'),
                'awake_time': day.get('awake_time'),
            })
        
        return pd.DataFrame(records)
    
    def get_activity_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch activity data from Oura API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with activity metrics
        """
        try:
            activity_data = self.client.daily_activity(start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error fetching activity data: {e}")
            return pd.DataFrame()
        
        records = []
        for day in activity_data.get('data', []):
            records.append({
                'date': day.get('day'),
                'steps': day.get('steps'),
                'calories_total': day.get('calories_total'),
                'calories_active': day.get('calories_active'),
                'activity_score': day.get('score'),
                'met_minutes': day.get('met_minutes'),
                'low_activity_time': day.get('low_activity_time'),
                'medium_activity_time': day.get('medium_activity_time'),
                'high_activity_time': day.get('high_activity_time'),
            })
        
        return pd.DataFrame(records)
    
    def get_readiness_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch readiness/recovery data from Oura API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with readiness metrics
        """
        try:
            readiness_data = self.client.daily_readiness(start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error fetching readiness data: {e}")
            return pd.DataFrame()
        
        records = []
        for day in readiness_data.get('data', []):
            records.append({
                'date': day.get('day'),
                'readiness_score': day.get('score'),
                'temperature_trend': day.get('temperature_trend'),
                'hrv_balance': day.get('hrv_balance'),
                'recovery_index': day.get('recovery_index'),
                'resting_heart_rate': day.get('resting_heart_rate'),
            })
        
        return pd.DataFrame(records)
    
    def get_heart_rate_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch heart rate data from Oura API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            DataFrame with heart rate metrics
        """
        try:
            hr_data = self.client.heart_rate(start_date=start_date, end_date=end_date)
        except Exception as e:
            print(f"Error fetching heart rate data: {e}")
            return pd.DataFrame()
        
        records = []
        for entry in hr_data.get('data', []):
            records.append({
                'timestamp': entry.get('timestamp'),
                'bpm': entry.get('bpm'),
                'source': entry.get('source'),
            })
        
        return pd.DataFrame(records)
    
    def get_all_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all available data types.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            Dictionary with DataFrames for each data type
        """
        return {
            'sleep': self.get_sleep_data(start_date, end_date),
            'activity': self.get_activity_data(start_date, end_date),
            'readiness': self.get_readiness_data(start_date, end_date),
            'heart_rate': self.get_heart_rate_data(start_date, end_date),
        }
    
    def sync_recent_data(self, days: int = 7, save_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Sync recent data (last N days).
        
        Args:
            days: Number of days to sync
            save_path: Optional path to save CSV files
        
        Returns:
            Dictionary with DataFrames
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        data = self.get_all_data(start_date, end_date)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for data_type, df in data.items():
                if not df.empty:
                    file_path = os.path.join(save_path, f'oura_{data_type}.csv')
                    df.to_csv(file_path, index=False)
                    print(f"Saved {data_type} data to {file_path}")
        
        return data


if __name__ == "__main__":
    # Example usage
    collector = OuraDataCollector()
    
    # Sync last 30 days
    data = collector.sync_recent_data(days=30, save_path='data/raw/oura')
    
    print("Data collection complete!")
    for data_type, df in data.items():
        print(f"{data_type}: {len(df)} records")

