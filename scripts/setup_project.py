#!/usr/bin/env python3
"""
Project setup script - Initialize all required components.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def setup_project():
    """Setup project: database, check API keys, etc."""
    
    print("=" * 70)
    print("Project Setup")
    print("=" * 70)
    
    # 1. Check environment variables
    print("\n[1] Checking environment variables...")
    from src.config import OPENAI_API_KEY, DATABASE_URL
    
    if OPENAI_API_KEY:
        print(f"  ✅ OPENAI_API_KEY: {'*' * 20}...{OPENAI_API_KEY[-10:]}")
    else:
        print("  ⚠️  OPENAI_API_KEY not set")
        print("     Create .env file with: OPENAI_API_KEY=your_key")
    
    # 2. Initialize database
    print("\n[2] Initializing database...")
    try:
        from scripts.init_database import init_database
        init_database()
    except Exception as e:
        print(f"  ✗ Database initialization failed: {e}")
    
    # 3. Check data files
    print("\n[3] Checking data files...")
    data_files = [
        "data/processed/unified_daily.parquet",
        "data/features/daily_features.parquet",
    ]
    
    for df in data_files:
        p = Path(df)
        if p.exists():
            print(f"  ✅ {df}")
        else:
            print(f"  ⚠️  {df} not found")
            print(f"     Run: python src/data_collection/preprocess.py")
    
    # 4. Check model
    print("\n[4] Checking model...")
    model_path = Path("models/bandit_model.pkl")
    if model_path.exists():
        print(f"  ✅ Model exists: {model_path}")
    else:
        print(f"  ⚠️  Model not found")
        print(f"     Run: python src/recommendation/train.py")
    
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Train model: python src/recommendation/train.py")
    print("  2. Start API: python src/serving/api_server.py")


if __name__ == "__main__":
    setup_project()

