"""
Main script for generating bottle dataset labels.
"""

import argparse
from pathlib import Path

from config import BottleConfig
from label_generator import BottleLabelGenerator

def main():
    """Main function to generate bottle labels."""
    parser = argparse.ArgumentParser(description='Generate bottle dataset labels')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Output CSV file path')
    parser.add_argument('--sample', action='store_true', help='Show sample entries')
    
    args = parser.parse_args()
    
    config = BottleConfig(args.config)
    
    generator = BottleLabelGenerator(config)
    
    df = generator.generate_labels()
    
    if not generator.validate_labels(df):
        print("❌ Label validation failed")
        return
    
    output_path = generator.save_labels(df, args.output)
    
    generator.print_summary(df)
    
    if args.sample:
        print("\n🎯 Sample entries:")
        print(df.head())
    
    print(f"\n✅ Dataset generation complete: {output_path}")

if __name__ == "__main__":
    main()
