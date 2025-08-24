"""
Data generator for bottle dataset labels.
Handles label creation, validation, and export.
"""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
import logging

from config import BottleConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class BottleLabelGenerator:
    """Generates structured bottle dataset labels with configurable features."""
    
    def __init__(self, config: BottleConfig | None = None):
        """Initialize generator with configuration."""
        self.config = config or BottleConfig()
        self.labels_data: List[Dict[str, Any]] = []
    
    def generate_labels(self) -> pd.DataFrame:
        """Generate labels for all bottle categories."""
        logger.info("Starting label generation...")
        self.labels_data.clear()
        
        total_images = 0
        for category, features in self.config.categories.items():
            images_count = self._generate_category_labels(category, features)
            total_images += images_count
            logger.info(f"Generated {images_count} labels for {category}")
        
        df = pd.DataFrame(self.labels_data)
        logger.info(f"✅ Generated labels for {total_images} total images")
        
        return df
    
    def _generate_category_labels(self, category: str, features: Dict[str, int]) -> int:
        """Generate labels for a specific bottle category."""
        images_per_category = self.config.settings['images_per_category']
        filename_format = self.config.settings['image_filename_format']
        
        for i in range(images_per_category):
            image_filename = filename_format.format(i)
            
            label_entry = {
                'image_filename': image_filename,
                'category': category,
                'bottle_type': self._normalize_category_name(category)
            }
            
            for feature_name, score in features.items():
                label_entry[f'{feature_name}_score'] = score
            
            self.labels_data.append(label_entry)
        
        return images_per_category
    
    def _normalize_category_name(self, category: str) -> str:
        """Convert category name to normalized format."""
        return category.replace(' ', '_').lower()
    
    def save_labels(self, df: pd.DataFrame, output_path: str | None = None) -> str:
        """Save labels DataFrame to CSV file."""
        final_output_path = output_path or self.config.settings['output_filename']
        
        Path(final_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(final_output_path, index=False)
        logger.info(f"📁 Labels saved to: {final_output_path}")
        
        return final_output_path
    
    def validate_labels(self, df: pd.DataFrame) -> bool:
        """Validate generated labels for consistency."""
        required_columns = ['image_filename', 'category', 'bottle_type']
        feature_columns = [f'{feature}_score' for feature in self.config.get_feature_names()]
        all_required = required_columns + feature_columns
        
        missing_columns = set(all_required) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            return False
        
        if df.isnull().any().any():
            logger.error("Dataset contains missing values")
            return False
        
        for feature_col in feature_columns:
            if not df[feature_col].between(1, 10).all():
                logger.warning(f"Feature {feature_col} has scores outside 1-10 range")
        
        logger.info("✅ Label validation passed")
        return True
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print dataset summary statistics."""
        print("\n📊 Dataset Summary:")
        print(df.groupby('category').size())
        
        feature_cols = [f'{feature}_score' for feature in self.config.get_feature_names()]
        if feature_cols:
            print("\n🎯 Feature Score Averages by Category:")
            summary = df.groupby('category')[feature_cols].mean().round(1)
            print(summary)
        
        print(f"\n📈 Total Categories: {df['category'].nunique()}")
        print(f"📈 Total Images: {len(df)}")
