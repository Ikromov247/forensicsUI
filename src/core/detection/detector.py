from typing import List, Optional, Union
import numpy as np
from ultralytics import YOLO

from ...exceptions import DetectionError
from ...logging_config import get_logger

logger = get_logger(__name__)

class ObjectDetector:
    """
    Handles object detection using YOLOv8.
    """
    
    def __init__(
        self,
        model_path: str = "models/trained_yolo.pt",
        min_confidence: float = 0.7,
        tracker_config: Optional[str] = None,
        device: str = "mps"
    ):
        """
        Initialize object detector.
        
        Args:
            model_path: Path to YOLO model
            min_confidence: Minimum confidence threshold
            tracker_config: Path to tracker configuration
            device: Device to run inference on (mps, cuda, cpu)
        """
        try:
            self.model = YOLO(model_path)
            self.min_confidence = min_confidence
            self.tracker_config = tracker_config
            self.device = device
            logger.info(f"Initialized object detector with model: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize object detector: {str(e)}")
            raise DetectionError(f"Detector initialization failed: {str(e)}")
    
    def detect(
        self,
        source: Union[str, np.ndarray],
        classes: Optional[List[int]] = None,
        is_image: bool = False,
        persist: bool = False
    ):
        """
        Detect objects in image or video frame.
        
        Args:
            source: Input image or video frame
            classes: List of class IDs to detect
            is_image: Whether source is an image
            persist: Whether to use tracking
            
        Returns:
            Detection results
        """
        try:
            if classes is None:
                classes = list(self.model.names.keys())
                
            kwargs = {
                'classes': classes,
                'device': self.device,
                'conf': self.min_confidence,
                'verbose': False
            }
            
            if persist and self.tracker_config:
                kwargs['tracker'] = self.tracker_config
                return self.model.track(source, **kwargs)
            else:
                return self.model(source, **kwargs)
            
        except Exception as e:
            logger.error(f"Detection failed: {str(e)}")
            raise DetectionError(f"Detection failed: {str(e)}") 