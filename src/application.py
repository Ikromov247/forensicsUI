from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import cv2
from pathlib import Path

from .config import ApplicationConfig
from .exceptions import (
    ForensicsError, DetectionError, FeatureExtractionError,
    DatabaseError, ResourceError
)
from .logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class TargetObjectData:
    """Data class for target object information"""
    obj_id: int
    cls: int
    conf: float
    bbox: List[int]
    feature: np.ndarray
    color: List[int]

class ForensicsApplication:
    """
    Main application class for object detection and tracking.
    Uses dependency injection for better testability and modularity.
    """
    
    def __init__(self, config: ApplicationConfig):
        """
        Initialize the application with configuration.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logger
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all application components"""
        try:
            # Initialize detector
            from .core.detection import ObjectDetector
            self.detector = ObjectDetector(
                model_path=self.config.detection.model_path,
                min_confidence=self.config.detection.min_confidence,
                tracker_config=self.config.detection.tracker_config
            )
            
            # Initialize feature extractor
            from .core.features import FeatureExtractor
            self.feature_extractor = FeatureExtractor(
                model_name=self.config.features.model_name
            )
            
            # Initialize feature comparator
            from .core.features import FeatureComparator
            self.feature_comparator = FeatureComparator(
                threshold=self.config.features.comparison_threshold
            )
            
            # Initialize database manager
            from .database import DatabaseManager
            self.db_manager = DatabaseManager(
                db_name=self.config.database.name,
                tables=self.config.database.tables
            )
            
            # Initialize visualizer
            from .visualization import Visualizer
            self.visualizer = Visualizer(
                enabled=self.config.visualization.enabled,
                show_bbox=self.config.visualization.show_bbox,
                show_similarity=self.config.visualization.show_similarity,
                show_id=self.config.visualization.show_id
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise ForensicsError(f"Component initialization failed: {str(e)}")
    
    def process_target_image(self, image_path: str) -> TargetObjectData:
        """
        Process target image and extract object information.
        
        Args:
            image_path: Path to target image
            
        Returns:
            TargetObjectData containing object information
        """
        try:
            # Load and detect object
            image = cv2.imread(image_path)
            if image is None:
                raise DetectionError(f"Failed to load image: {image_path}")
            
            detection_result = self.detector.detect(image, is_image=True)
            if not detection_result:
                raise DetectionError("No objects detected in target image")
            
            # Process detection result
            best_result = self._get_best_detection(detection_result)
            obj_id, cls, conf, bbox = self._process_detection_result(best_result)
            
            # Extract features
            cropped_obj = self._crop_object(image, bbox)
            feature = self.feature_extractor.extract(cropped_obj)
            color = self._get_dominant_color(cropped_obj)
            
            return TargetObjectData(
                obj_id=obj_id,
                cls=cls,
                conf=conf,
                bbox=bbox,
                feature=feature,
                color=color
            )
            
        except Exception as e:
            self.logger.error(f"Failed to process target image: {str(e)}")
            raise DetectionError(f"Target image processing failed: {str(e)}")
    
    def process_video(self, video_path: str, target_data: TargetObjectData):
        """
        Process video and compare objects with target.
        
        Args:
            video_path: Path to video file
            target_data: Target object information
        """
        try:
            from .core.tracking import VideoProcessor
            processor = VideoProcessor(
                detector=self.detector,
                feature_extractor=self.feature_extractor,
                feature_comparator=self.feature_comparator,
                visualizer=self.visualizer,
                config=self.config
            )
            
            return processor.process(video_path, target_data)
            
        except Exception as e:
            self.logger.error(f"Failed to process video: {str(e)}")
            raise ForensicsError(f"Video processing failed: {str(e)}")
    
    def _get_best_detection(self, detection_result):
        """Get the best detection result based on confidence"""
        if not detection_result:
            return None
        return max(detection_result, key=lambda x: x[-2])
    
    def _process_detection_result(self, result):
        """Process detection result and return object information"""
        bbox = result[:4].astype(int).tolist()
        conf = float(result[-2])
        cls = int(result[-1])
        obj_id = int(result[4]) if len(result) > 4 else 0
        return obj_id, cls, conf, bbox
    
    def _crop_object(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop object from image using bounding box"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
    
    def _get_dominant_color(self, image: np.ndarray, k: int = 3) -> List[int]:
        """Get dominant color from image using k-means clustering"""
        # Implementation from original code
        pixels = image.reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(
            pixels.astype(np.float32),
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        return centers[0].astype(int).tolist() 