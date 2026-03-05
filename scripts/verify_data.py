#!/usr/bin/env python3
"""
Verify imported data and check structure.

This script checks:
1. Apple Watch Health data
2. Oura Ring data
3. PMData dataset
4. Data quality and structure
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def check_apple_watch_data():
    """Check Apple Watch Health data."""
    print("\n" + "=" * 70)
    print("APPLE WATCH HEALTH DATA")
    print("=" * 70)
    
    # Check both possible locations
    apple_dir1 = Path("data/raw/apple_watch_health")
    apple_dir2 = Path("data/raw/apple")
    
    xml_files = []
    if apple_dir1.exists():
        xml_files.extend(list(apple_dir1.glob("*.xml")))
    if apple_dir2.exists():
        xml_files.extend(list(apple_dir2.glob("*.xml")))
    
    if not xml_files:
        print("✗ No XML files found")
        print("  Expected locations:")
        print("    - data/raw/apple_watch_health/export.xml")
        print("    - data/raw/apple/export.xml")
        return False
    
    xml_file = xml_files[0]
    print(f"✓ Found: {xml_file} ({xml_file.stat().st_size / 1024 / 1024:.2f} MB)")
    
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_file)
        root = tree.getroot()
        records = root.findall('.//Record')
        
        print(f"\n✓ Health records: {len(records):,}")
        
        # Count by type (sample)
        types = {}
        for r in records[:1000]:  # Sample first 1000
            t = r.get('type', 'unknown')
            types[t] = types.get(t, 0) + 1
        
        print(f"\nSample record types:")
        for t, count in list(sorted(types.items(), key=lambda x: x[1], reverse=True))[:10]:
            print(f"  - {t}: {count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error parsing data: {e}")
        return False


def check_oura_data():
    """Check Oura Ring data."""
    print("\n" + "=" * 70)
    print("OURA RING DATA")
    print("=" * 70)
    
    oura_dir = Path("data/raw/oura")
    csv_files = list(oura_dir.glob("*.csv"))
    
    if not csv_files:
        print("✗ No CSV files found")
        return False
    
    print(f"✓ Found {len(csv_files)} CSV files:")
    
    total_records = 0
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            count = len(df)
            total_records += count
            print(f"\n  - {csv_file.name}:")
            print(f"    Records: {count:,}")
            print(f"    Columns: {len(df.columns)}")
            print(f"    Column names: {', '.join(df.columns.tolist()[:10])}")
            if len(df.columns) > 10:
                print(f"                ... and {len(df.columns) - 10} more")
            
            if 'date' in df.columns or 'Date' in df.columns:
                date_col = 'date' if 'date' in df.columns else 'Date'
                dates = pd.to_datetime(df[date_col], errors='coerce')
                dates = dates.dropna()
                if len(dates) > 0:
                    print(f"    Date range: {dates.min()} to {dates.max()}")
                    
        except Exception as e:
            print(f"  ✗ Error reading {csv_file.name}: {e}")
    
    print(f"\n✓ Total records: {total_records:,}")
    return True


def check_pmdata():
    """Check PMData dataset."""
    print("\n" + "=" * 70)
    print("PMDATA DATASET")
    print("=" * 70)
    
    pmdata_dir = Path("data/public/pmdata")
    
    if not pmdata_dir.exists():
        print("✗ PMData directory not found")
        return False
    
    # Check structure
    participants = [d for d in pmdata_dir.iterdir() if d.is_dir() and d.name.startswith('p')]
    csv_files = list(pmdata_dir.rglob("*.csv"))
    
    print(f"✓ Found:")
    print(f"  - Participants: {len(participants)} (p01-p{len(participants):02d})")
    print(f"  - CSV files: {len(csv_files)}")
    
    # Check structure of one participant
    if participants:
        p_dir = participants[0]
        print(f"\n  Structure (example: {p_dir.name}):")
        for subdir in sorted(p_dir.iterdir()):
            if subdir.is_dir():
                sub_csv = list(subdir.glob("*.csv"))
                print(f"    - {subdir.name}/: {len(sub_csv)} CSV files")
                if sub_csv:
                    # Show first file
                    try:
                        sample_df = pd.read_csv(sub_csv[0])
                        print(f"      Sample: {sub_csv[0].name} ({len(sample_df)} records)")
                    except:
                        pass
    
    # Count total records
    total_records = 0
    sample_files = 0
    for csv_file in csv_files[:100]:  # Sample first 100 files
        try:
            df = pd.read_csv(csv_file)
            total_records += len(df)
            sample_files += 1
        except:
            pass
    
    if sample_files > 0:
        avg_per_file = total_records / sample_files
        estimated_total = avg_per_file * len(csv_files)
        print(f"\n  Estimated total records: ~{estimated_total:,.0f}")
        print(f"  (Based on {sample_files} sample files)")
    
    return True


def main():
    """Main verification function."""
    print("=" * 70)
    print("DATA IMPORT VERIFICATION")
    print("=" * 70)
    
    results = {
        'Apple Watch': check_apple_watch_data(),
        'Oura': check_oura_data(),
        'PMData': check_pmdata(),
    }
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for source, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"{status} {source}")
    
    if all(results.values()):
        print("\n✅ All data imported successfully!")
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. EXPLORE DATA")
        print("   - Create notebooks/data_exploration.ipynb")
        print("   - Analyze data distributions, missing values, correlations")
        print("   - Understand PMData structure and features")
        
        print("\n2. PREPROCESS DATA")
        print("   - Standardize formats across datasets")
        print("   - Create unified feature schema")
        print("   - Handle missing values and outliers")
        print("   - Merge personal + PMData datasets")
        
        print("\n3. FEATURE ENGINEERING")
        print("   - Extract time-series features")
        print("   - Create body state indicators")
        print("   - Build training history features")
        print("   - Set up Feast feature store")
        
        print("\n4. BUILD MODEL")
        print("   - Implement contextual bandits")
        print("   - Implement Thompson sampling")
        print("   - Train on combined dataset")
        print("   - Evaluate performance")
        
        print("\n5. DEPLOY SYSTEM")
        print("   - Set up TorchServe API")
        print("   - Implement Kafka streaming")
        print("   - Deploy AI Coach Agent")
        print("   - Set up A/B testing")
        
        print("\n6. ITERATE & IMPROVE")
        print("   - Collect feedback")
        print("   - Online learning updates")
        print("   - Model retraining")
        print("   - Performance monitoring")
    else:
        print("\n⚠ Some data sources need attention")
        print("Check the errors above and fix missing data")


if __name__ == "__main__":
    main()
