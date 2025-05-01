from dataclasses import dataclass
from typing import Dict, Any
import yaml
from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for dir_path in [MODELS_DIR, DATABASE_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model paths
MODEL_PATHS = {
    "yolo": str(MODELS_DIR / "yolov8l.pt"),
    "resnet": str(MODELS_DIR / "resnet50.pth"),
    "inception": str(MODELS_DIR / "inception_v3.pth")
}

# Detection settings
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "classes": None,  # None means all classes
    "max_detections": 100
}

# Feature extraction settings
FEATURE_CONFIG = {
    "model": "inception",  # "resnet" or "inception"
    "device": "mps",  # "cpu", "cuda", or "mps"
    "batch_size": 32,
    "normalize": True
}

# Feature comparison settings
COMPARISON_CONFIG = {
    "similarity_threshold": 0.8,
    "top_k": 5,
    "use_pca": True,
    "pca_components": 128
}

# Database settings
DATABASE_CONFIG = {
    "name": "forensics",
    "tables": {
        "objects": """
            CREATE TABLE IF NOT EXISTS objects (
                obj_id INTEGER PRIMARY KEY,
                cls INTEGER,
                conf REAL,
                bbox TEXT,
                frame_ids TEXT,
                similarity REAL
            )
        """,
        "features": """
            CREATE TABLE IF NOT EXISTS features (
                obj_id INTEGER PRIMARY KEY,
                features BLOB,
                FOREIGN KEY (obj_id) REFERENCES objects (obj_id)
            )
        """
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "enabled": True,
    "window_name": "Object Tracking",
    "bbox_color": (0, 255, 0),  # BGR
    "text_color": (0, 255, 0),  # BGR
    "font_scale": 0.3,
    "thickness": 2
}

# Logging settings
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": str(LOGS_DIR / "forensics.log")
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration.
    
    Returns:
        Dictionary containing all configuration settings
    """
    return {
        "base_dir": BASE_DIR,
        "models_dir": MODELS_DIR,
        "database_dir": DATABASE_DIR,
        "logs_dir": LOGS_DIR,
        "model_paths": MODEL_PATHS,
        "detection": DETECTION_CONFIG,
        "feature": FEATURE_CONFIG,
        "comparison": COMPARISON_CONFIG,
        "database": DATABASE_CONFIG,
        "visualization": VISUALIZATION_CONFIG,
        "logging": LOGGING_CONFIG
    }

@dataclass
class DetectionConfig:
    min_confidence: float = 0.7
    model_path: str = "models/yolov8n.pt"
    tracker_config: str = "bytetrack.yaml"
    classes: list = None

@dataclass
class FeatureConfig:
    model_name: str = "inception"
    extraction_interval: int = 5
    comparison_threshold: float = 0.8

@dataclass
class DatabaseConfig:
    name: str = None
    tables: Dict[str, str] = None

@dataclass
class VisualizationConfig:
    enabled: bool = True
    show_bbox: bool = True
    show_similarity: bool = True
    show_id: bool = True

@dataclass
class ApplicationConfig:
    detection: DetectionConfig
    features: FeatureConfig
    database: DatabaseConfig
    visualization: VisualizationConfig
    performance_mode: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ApplicationConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            detection=DetectionConfig(**config_dict.get('detection', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            database=DatabaseConfig(**config_dict.get('database', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            performance_mode=config_dict.get('performance_mode', True)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'detection': {
                'min_confidence': self.detection.min_confidence,
                'model_path': self.detection.model_path,
                'tracker_config': self.detection.tracker_config,
                'classes': self.detection.classes
            },
            'features': {
                'model_name': self.features.model_name,
                'extraction_interval': self.features.extraction_interval,
                'comparison_threshold': self.features.comparison_threshold
            },
            'database': {
                'name': self.database.name,
                'tables': self.database.tables
            },
            'visualization': {
                'enabled': self.visualization.enabled,
                'show_bbox': self.visualization.show_bbox,
                'show_similarity': self.visualization.show_similarity,
                'show_id': self.visualization.show_id
            },
            'performance_mode': self.performance_mode
        } 