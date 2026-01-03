"""
Data Preprocessing Pipeline

Unifies multi-source data (Apple Watch, Oura, PMData) into a single daily schema.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection.apple_health import AppleHealthParser


class DataPreprocessor:
    """Preprocess and unify data from multiple sources."""
    
    def __init__(self, output_dir: str = "data/processed"):
        """
        Initialize preprocessor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_apple_watch(self, xml_path: str) -> pd.DataFrame:
        """
        Preprocess Apple Watch data to daily aggregates.
        
        Args:
            xml_path: Path to Apple Health export XML
        
        Returns:
            DataFrame with daily aggregated features
        """
        parser = AppleHealthParser(xml_path)
        
        # Get all data
        hr_df = parser.get_heart_rate_data()
        activity_df = parser.get_activity_data()
        sleep_df = parser.get_sleep_data()
        
        # Process heart rate
        daily_hr = self._aggregate_heart_rate(hr_df)
        
        # Process activity
        daily_activity = self._aggregate_activity(activity_df)
        
        # Process sleep
        daily_sleep = self._aggregate_sleep(sleep_df)
        
        # Merge daily data
        daily_df = daily_hr.merge(daily_activity, on='date', how='outer')
        daily_df = daily_df.merge(daily_sleep, on='date', how='outer')
        daily_df = daily_df.sort_values('date')
        
        return daily_df
    
    def _aggregate_heart_rate(self, hr_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate heart rate data to daily level."""
        if hr_df.empty:
            return pd.DataFrame(columns=['date'])
        
        hr_df['date'] = pd.to_datetime(hr_df['start_date']).dt.date
        hr_df['value'] = pd.to_numeric(hr_df['value'], errors='coerce')
        
        daily = hr_df.groupby('date').agg({
            'value': ['mean', 'min', 'max', 'std']
        }).reset_index()
        
        daily.columns = ['date', 'hr_mean', 'hr_min', 'hr_max', 'hr_std']
        daily['resting_hr'] = daily['hr_min']  # Approximate resting HR
        
        return daily
    
    def _aggregate_activity(self, activity_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate activity data to daily level."""
        if activity_df.empty:
            return pd.DataFrame(columns=['date'])
        
        activity_df['date'] = pd.to_datetime(activity_df['start_date']).dt.date
        activity_df['value'] = pd.to_numeric(activity_df['value'], errors='coerce')
        
        daily = activity_df.groupby(['date', 'type']).agg({
            'value': 'sum'
        }).reset_index()
        
        # Pivot to wide format
        daily_wide = daily.pivot(index='date', columns='type', values='value').reset_index()
        daily_wide.columns.name = None
        
        # Rename columns
        rename_map = {
            'steps': 'steps',
            'active_energy': 'active_calories',
            'basal_energy': 'basal_calories',
        }
        daily_wide = daily_wide.rename(columns=rename_map)
        
        return daily_wide
    
    def _aggregate_sleep(self, sleep_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate sleep data to daily level."""
        if sleep_df.empty:
            return pd.DataFrame(columns=['date'])
        
        sleep_df['date'] = pd.to_datetime(sleep_df['start_date']).dt.date
        
        daily = sleep_df.groupby('date').agg({
            'duration': 'sum'
        }).reset_index()
        
        daily.columns = ['date', 'sleep_duration_hours']
        
        return daily
    
    def preprocess_oura(self, csv_path: str) -> pd.DataFrame:
        """
        Preprocess Oura data to daily format.
        
        Args:
            csv_path: Path to Oura CSV file
        
        Returns:
            DataFrame with daily Oura features
        """
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Select key features
        key_features = [
            'date',
            'Readiness Score',
            'Sleep Score',
            'Activity Score',
            'Average HRV',
            'Average Resting Heart Rate',
            'Total Sleep Duration',
            'Steps',
            'Total Burn',
            'Activity Burn',
        ]
        
        available_features = [f for f in key_features if f in df.columns]
        oura_daily = df[available_features].copy()
        
        # Rename to canonical names
        rename_map = {
            'Readiness Score': 'readiness_score',
            'Sleep Score': 'sleep_score',
            'Activity Score': 'activity_score',
            'Average HRV': 'hrv',
            'Average Resting Heart Rate': 'resting_hr',
            'Total Sleep Duration': 'sleep_duration_hours',
            'Steps': 'steps',
            'Total Burn': 'total_calories',
            'Activity Burn': 'active_calories',
        }
        
        oura_daily = oura_daily.rename(columns=rename_map)
        
        return oura_daily
    
    def preprocess_pmdata(self, pmdata_dir: str) -> pd.DataFrame:
        """
        Preprocess PMData to daily format.
        
        Args:
            pmdata_dir: Path to PMData directory
        
        Returns:
            DataFrame with daily PMData features
        """
        pmdata_path = Path(pmdata_dir)
        participants = sorted([d for d in pmdata_path.iterdir() 
                              if d.is_dir() and d.name.startswith('p')])
        
        all_daily = []
        
        for p_dir in participants:
            user_id = p_dir.name
            
            # Load Fitbit data
            fitbit_dir = p_dir / 'fitbit'
            if fitbit_dir.exists():
                # Load sleep_score.csv if available
                sleep_file = fitbit_dir / 'sleep_score.csv'
                if sleep_file.exists():
                    sleep_df = pd.read_csv(sleep_file)
                    sleep_df['user_id'] = user_id
                    sleep_df['date'] = pd.to_datetime(sleep_df['timestamp']).dt.date
                    all_daily.append(sleep_df)
        
        if all_daily:
            combined = pd.concat(all_daily, ignore_index=True)
            return combined
        
        return pd.DataFrame()
    
    def unify_daily_data(self, apple_df: pd.DataFrame, oura_df: pd.DataFrame,
                        pmdata_df: Optional[pd.DataFrame] = None,
                        user_id: str = "user_001") -> pd.DataFrame:
        """
        Unify all daily data sources.
        
        Args:
            apple_df: Apple Watch daily data
            oura_df: Oura daily data
            pmdata_df: PMData daily data (optional)
            user_id: User identifier
        
        Returns:
            Unified daily DataFrame
        """
        # Add user_id
        apple_df['user_id'] = user_id
        oura_df['user_id'] = user_id
        
        # Merge Apple + Oura
        unified = apple_df.merge(
            oura_df,
            on=['date', 'user_id'],
            how='outer',
            suffixes=('_apple', '_oura')
        )
        
        # Merge PMData if available
        if pmdata_df is not None and not pmdata_df.empty:
            unified = unified.merge(
                pmdata_df,
                on=['date', 'user_id'],
                how='outer',
                suffixes=('', '_pmdata')
            )
        
        # Sort by date
        unified = unified.sort_values('date').reset_index(drop=True)
        
        # Fill missing user_id
        unified['user_id'] = unified['user_id'].fillna(user_id)
        
        return unified
    
    def save_unified_data(self, unified_df: pd.DataFrame, filename: str = "unified_daily.parquet"):
        """Save unified data to parquet."""
        output_path = self.output_dir / filename
        unified_df.to_parquet(output_path, index=False)
        print(f"✓ Saved unified data to {output_path}")
        return output_path


def main():
    """Main preprocessing function."""
    print("=" * 70)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    preprocessor = DataPreprocessor()
    
    # Process Apple Watch
    print("\n[1/3] Processing Apple Watch data...")
    apple_xml = Path("data/raw/apple/export.xml")
    if apple_xml.exists():
        apple_daily = preprocessor.preprocess_apple_watch(str(apple_xml))
        print(f"✓ Apple Watch: {len(apple_daily)} daily records")
    else:
        print("✗ Apple Watch XML not found")
        apple_daily = pd.DataFrame()
    
    # Process Oura
    print("\n[2/3] Processing Oura data...")
    oura_csv = Path("data/raw/oura/oura_2025-11-01_2026-01-01_trends.csv")
    if oura_csv.exists():
        oura_daily = preprocessor.preprocess_oura(str(oura_csv))
        print(f"✓ Oura: {len(oura_daily)} daily records")
    else:
        print("✗ Oura CSV not found")
        oura_daily = pd.DataFrame()
    
    # Process PMData (optional)
    print("\n[3/3] Processing PMData...")
    pmdata_dir = Path("data/public/pmdata")
    if pmdata_dir.exists():
        pmdata_daily = preprocessor.preprocess_pmdata(str(pmdata_dir))
        print(f"✓ PMData: {len(pmdata_daily)} records")
    else:
        print("⚠ PMData not found, skipping")
        pmdata_daily = None
    
    # Unify
    print("\n" + "=" * 70)
    print("UNIFYING DATA")
    print("=" * 70)
    
    if not apple_daily.empty or not oura_daily.empty:
        unified = preprocessor.unify_daily_data(
            apple_daily,
            oura_daily,
            pmdata_daily
        )
        
        print(f"\n✓ Unified dataset: {len(unified)} daily records")
        print(f"  Columns: {len(unified.columns)}")
        print(f"\nSample columns: {list(unified.columns[:10])}")
        
        # Save
        preprocessor.save_unified_data(unified)
        
        # Coverage report
        print("\n" + "=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70)
        
        if 'readiness_score' in unified.columns:
            coverage = unified['readiness_score'].notna().sum() / len(unified) * 100
            print(f"Readiness score coverage: {coverage:.1f}%")
        
        if 'sleep_duration_hours' in unified.columns:
            coverage = unified['sleep_duration_hours'].notna().sum() / len(unified) * 100
            print(f"Sleep coverage: {coverage:.1f}%")
        
        if 'steps' in unified.columns:
            coverage = unified['steps'].notna().sum() / len(unified) * 100
            print(f"Steps coverage: {coverage:.1f}%")
    else:
        print("✗ No data to unify")


if __name__ == "__main__":
    main()

