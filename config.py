"""
Configuration for bottle dataset generation.
Add new bottle types and features here.
"""

from typing import Dict, Any
import json
from pathlib import Path

class BottleConfig:
    """Configuration manager for bottle categories and features."""
    
    DEFAULT_CATEGORIES = {
        'Wine Bottle': {
            'durability': 9,        # Glass strength
            'chemical_safety': 9,   # Food-grade materials
            'ergonomics': 6         # Elegant but less practical
        },
        'Water Bottle': {
            'durability': 7,        # Daily use strength
            'chemical_safety': 8,   # BPA-free requirements
            'ergonomics': 9         # Optimized for drinking
        },
        'Soda Bottle': {
            'durability': 6,        # Carbonation pressure
            'chemical_safety': 7,   # Acidic drink compatibility
            'ergonomics': 8         # Easy grip design
        },
        'Plastic Bottles': {
            'durability': 5,        # Variable quality
            'chemical_safety': 6,   # Plastic grade dependent
            'ergonomics': 8         # Lightweight design
        },
        'Beer Bottles': {
            'durability': 8,        # Thick glass
            'chemical_safety': 9,   # Alcohol compatibility
            'ergonomics': 7         # Traditional shape
        }
    }
    
    DEFAULT_SETTINGS = {
        'images_per_category': 5000,
        'image_filename_format': '{:08d}.jpg',
        'output_filename': 'bottle_labels_essential_trio.csv'
    }
    
    def __init__(self, config_file: str | None = None):
        """Initialize configuration from file or defaults."""
        self.categories = self.DEFAULT_CATEGORIES.copy()
        self.settings = self.DEFAULT_SETTINGS.copy()
        
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                
            if 'categories' in config_data:
                self.categories.update(config_data['categories'])
            if 'settings' in config_data:
                self.settings.update(config_data['settings'])
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"⚠️  Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration.")
    
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        config_data = {
            'categories': self.categories,
            'settings': self.settings
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
    
    def add_category(self, name: str, features: Dict[str, int]) -> None:
        """Add a new bottle category."""
        self.categories[name] = features
    
    def get_feature_names(self) -> list:
        """Get list of all feature names."""
        if not self.categories:
            return []
        return list(next(iter(self.categories.values())).keys())
