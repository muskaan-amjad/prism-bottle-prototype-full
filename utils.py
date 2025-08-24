"""
Utility functions for bottle dataset management.
"""

import pandas as pd
from pathlib import Path
from typing import List
import json

def compare_datasets(csv1: str, csv2: str) -> None:
    """Compare two generated datasets."""
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    print(f"\n📊 Dataset Comparison:")
    print(f"Dataset 1 ({csv1}): {len(df1)} images, {df1['category'].nunique()} categories")
    print(f"Dataset 2 ({csv2}): {len(df2)} images, {df2['category'].nunique()} categories")
    
    cats1 = set(df1['category'].unique())
    cats2 = set(df2['category'].unique())
    
    if cats1 != cats2:
        print(f"\n🔍 Category Differences:")
        if cats1 - cats2:
            print(f"Only in dataset 1: {cats1 - cats2}")
        if cats2 - cats1:
            print(f"Only in dataset 2: {cats2 - cats1}")
    else:
        print("✅ Both datasets have identical categories")

def create_sample_config(filename: str, categories: dict, settings: dict | None = None) -> None:
    """Create a sample configuration file."""
    config = {
        "categories": categories,
        "settings": settings or {
            "images_per_category": 5000,
            "image_filename_format": "{:08d}.jpg",
            "output_filename": "bottle_labels.csv"
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Sample config saved to {filename}")

def validate_csv_structure(csv_file: str) -> bool:
    """Validate the structure of a generated CSV file."""
    try:
        df = pd.read_csv(csv_file)
        
        required_cols = ['image_filename', 'category', 'bottle_type']
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        feature_cols = [col for col in df.columns if col.endswith('_score')]
        if not feature_cols:
            print("⚠️  No feature score columns found")
        
        print(f"✅ CSV structure valid: {len(df)} rows, {len(df.columns)} columns")
        print(f"📊 Categories: {df['category'].nunique()}")
        print(f"🎯 Features: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating CSV: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Bottle Dataset Utilities")
    
    csv_files = list(Path('.').glob('*.csv'))
    if csv_files:
        print(f"\n📁 Found {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  - {csv_file}")
            validate_csv_structure(str(csv_file))
    
    csv_files = [str(f) for f in csv_files if 'bottle' in str(f)]
    if len(csv_files) >= 2:
        print(f"\n🔍 Comparing first two datasets...")
        compare_datasets(csv_files[0], csv_files[1])
