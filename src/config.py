from dataclasses import dataclass
from typing import Dict, Any
import yaml
from pathlib import Path

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