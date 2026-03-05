"""
Apple Health Data Collection Module

This module provides functions to parse and process Apple Health export data.
"""

import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
import os


class AppleHealthParser:
    """Parser for Apple Health export XML files."""
    
    # Relevant data types from Apple Health
    RELEVANT_TYPES = {
        'HKQuantityTypeIdentifierHeartRate': 'heart_rate',
        'HKQuantityTypeIdentifierRestingHeartRate': 'resting_heart_rate',
        'HKQuantityTypeIdentifierHeartRateVariabilitySDNN': 'hrv',
        'HKQuantityTypeIdentifierStepCount': 'steps',
        'HKQuantityTypeIdentifierActiveEnergyBurned': 'active_energy',
        'HKQuantityTypeIdentifierBasalEnergyBurned': 'basal_energy',
        'HKCategoryTypeIdentifierSleepAnalysis': 'sleep',
        'HKWorkoutTypeIdentifier': 'workout',
    }
    
    def __init__(self, xml_file_path: str):
        """
        Initialize parser with Apple Health export XML file.
        
        Args:
            xml_file_path: Path to exported Apple Health XML file
        """
        if not os.path.exists(xml_file_path):
            raise FileNotFoundError(
                f"Apple Health export file not found: {xml_file_path}\n"
                "Export your data: iPhone Settings → Privacy → Health → Export All Health Data\n"
                "Place export.xml in data/raw/apple_watch_health/"
            )
        
        self.xml_path = xml_file_path
        self.tree = ET.parse(xml_file_path)
        self.root = self.tree.getroot()
    
    def parse_records(self) -> pd.DataFrame:
        """
        Parse all health records from XML.
        
        Returns:
            DataFrame with all health records
        """
        records = []
        
        for record in self.root.findall('.//Record'):
            record_type = record.get('type')
            value = record.get('value')
            start_date = record.get('startDate')
            end_date = record.get('endDate')
            source_name = record.get('sourceName')
            
            # Map to readable type name
            type_name = self.RELEVANT_TYPES.get(record_type, record_type)
            
            records.append({
                'type': type_name,
                'original_type': record_type,
                'value': value,
                'start_date': start_date,
                'end_date': end_date,
                'source': source_name,
            })
        
        return pd.DataFrame(records)
    
    def get_heart_rate_data(self) -> pd.DataFrame:
        """
        Extract heart rate data.
        
        Returns:
            DataFrame with heart rate records
        """
        df = self.parse_records()
        hr_df = df[df['type'].isin(['heart_rate', 'resting_heart_rate'])].copy()
        
        if not hr_df.empty:
            hr_df['value'] = pd.to_numeric(hr_df['value'], errors='coerce')
            hr_df['start_date'] = pd.to_datetime(hr_df['start_date'], errors='coerce')
            hr_df = hr_df.sort_values('start_date')
        
        return hr_df
    
    def get_activity_data(self) -> pd.DataFrame:
        """
        Extract activity data (steps, calories).
        
        Returns:
            DataFrame with activity records
        """
        df = self.parse_records()
        activity_df = df[df['type'].isin(['steps', 'active_energy', 'basal_energy'])].copy()
        
        if not activity_df.empty:
            activity_df['value'] = pd.to_numeric(activity_df['value'], errors='coerce')
            activity_df['start_date'] = pd.to_datetime(activity_df['start_date'], errors='coerce')
            activity_df = activity_df.sort_values('start_date')
        
        return activity_df
    
    def get_sleep_data(self) -> pd.DataFrame:
        """
        Extract sleep data.
        
        Returns:
            DataFrame with sleep records
        """
        df = self.parse_records()
        sleep_df = df[df['type'] == 'sleep'].copy()
        
        if not sleep_df.empty:
            sleep_df['start_date'] = pd.to_datetime(sleep_df['start_date'], errors='coerce')
            sleep_df['end_date'] = pd.to_datetime(sleep_df['end_date'], errors='coerce')
            sleep_df['duration'] = (sleep_df['end_date'] - sleep_df['start_date']).dt.total_seconds() / 3600  # hours
            sleep_df = sleep_df.sort_values('start_date')
        
        return sleep_df
    
    def get_workout_data(self) -> pd.DataFrame:
        """
        Extract workout data.
        
        Returns:
            DataFrame with workout records
        """
        df = self.parse_records()
        workout_df = df[df['type'] == 'workout'].copy()
        
        if not workout_df.empty:
            workout_df['start_date'] = pd.to_datetime(workout_df['start_date'], errors='coerce')
            workout_df['end_date'] = pd.to_datetime(workout_df['end_date'], errors='coerce')
            workout_df['duration'] = (workout_df['end_date'] - workout_df['start_date']).dt.total_seconds() / 60  # minutes
            workout_df = workout_df.sort_values('start_date')
        
        return workout_df
    
    def get_hrv_data(self) -> pd.DataFrame:
        """
        Extract HRV (Heart Rate Variability) data.
        
        Returns:
            DataFrame with HRV records
        """
        df = self.parse_records()
        hrv_df = df[df['type'] == 'hrv'].copy()
        
        if not hrv_df.empty:
            hrv_df['value'] = pd.to_numeric(hrv_df['value'], errors='coerce')
            hrv_df['start_date'] = pd.to_datetime(hrv_df['start_date'], errors='coerce')
            hrv_df = hrv_df.sort_values('start_date')
        
        return hrv_df
    
    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Extract all relevant data types.
        
        Returns:
            Dictionary with DataFrames for each data type
        """
        return {
            'heart_rate': self.get_heart_rate_data(),
            'activity': self.get_activity_data(),
            'sleep': self.get_sleep_data(),
            'workouts': self.get_workout_data(),
            'hrv': self.get_hrv_data(),
        }
    
    def save_to_csv(self, output_dir: str):
        """
        Save all data to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        data = self.get_all_data()
        for data_type, df in data.items():
            if not df.empty:
                file_path = os.path.join(output_dir, f'apple_health_{data_type}.csv')
                df.to_csv(file_path, index=False)
                print(f"Saved {data_type} data to {file_path}")


if __name__ == "__main__":
    # Example usage
    # First, export Apple Health data from iPhone Settings
    # Place export.xml in data/raw/apple_watch_health/
    xml_path = "data/raw/apple_watch_health/export.xml"
    
    parser = AppleHealthParser(xml_path)
    data = parser.get_all_data()
    
    # Save to CSV
    parser.save_to_csv("data/raw/apple_watch_health")
    
    print("Apple Health data parsing complete!")
    for data_type, df in data.items():
        print(f"{data_type}: {len(df)} records")

