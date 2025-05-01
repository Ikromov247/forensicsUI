from typing import Dict, List, Optional
import numpy as np
import cv2
from pathlib import Path

from ...config import ApplicationConfig
from ...exceptions import ForensicsError, DetectionError, ResourceError
from ...logging_config import get_logger

logger = get_logger(__name__)

class VideoProcessor:
    """
    Handles video processing and object tracking.
    """
    
    def __init__(
        self,
        detector,
        feature_extractor,
        feature_comparator,
        visualizer,
        config: ApplicationConfig
    ):
        """
        Initialize video processor with required components.
        
        Args:
            detector: Object detector instance
            feature_extractor: Feature extractor instance
            feature_comparator: Feature comparator instance
            visualizer: Visualizer instance
            config: Application configuration
        """
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.feature_comparator = feature_comparator
        self.visualizer = visualizer
        self.config = config
        self.logger = logger
        
        self.tracked_objects: Dict[int, Dict] = {}
        self.extracted_objects = set()
    
    def process(self, video_path: str, target_data: Dict) -> Dict:
        """
        Process video and track objects.
        
        Args:
            video_path: Path to video file
            target_data: Target object information
            
        Returns:
            Dictionary containing tracking results
        """
        try:
            with self._open_video(video_path) as cap:
                while True:
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self._process_frame(frame, frame_index, target_data)
                    
                    if self.visualizer.enabled:
                        self.visualizer.display_frame(frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
            return self.tracked_objects
            
        except Exception as e:
            self.logger.error(f"Video processing failed: {str(e)}")
            raise ForensicsError(f"Video processing failed: {str(e)}")
    
    def _open_video(self, video_path: str):
        """Open video file with error handling"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ResourceError(f"Could not open video file: {video_path}")
        return cap
    
    def _process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        target_data: Dict
    ):
        """
        Process a single frame.
        
        Args:
            frame: Current video frame
            frame_index: Frame index
            target_data: Target object information
        """
        try:
            # Detect objects in frame
            results = self.detector.detect(
                frame,
                classes=[target_data['cls']],
                persist=True
            )
            
            if not results:
                return
                
            # Process each detection
            for result in results[0].boxes.data:
                self._process_detection(result, frame, frame_index, target_data)
                
        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")
            raise DetectionError(f"Frame processing failed: {str(e)}")
    
    def _process_detection(
        self,
        result: np.ndarray,
        frame: np.ndarray,
        frame_index: int,
        target_data: Dict
    ):
        """
        Process a single detection.
        
        Args:
            result: Detection result
            frame: Current frame
            frame_index: Frame index
            target_data: Target object information
        """
        # Extract detection information
        obj_id = int(result[4])
        bbox = result[:4].astype(int).tolist()
        
        # Update or create object tracking
        if obj_id in self.tracked_objects:
            self._update_tracked_object(obj_id, bbox, frame_index)
        else:
            self._create_tracked_object(obj_id, bbox, frame_index)
            
        # Extract features if needed
        if self._should_extract_features(obj_id):
            self._extract_and_compare_features(
                obj_id, frame, bbox, target_data
            )
            
        # Visualize if enabled
        if self.visualizer.enabled:
            self.visualizer.visualize_object(
                frame=frame,
                obj=self.tracked_objects[obj_id]
            )
    
    def _update_tracked_object(
        self,
        obj_id: int,
        bbox: List[int],
        frame_index: int
    ):
        """Update tracked object information"""
        obj = self.tracked_objects[obj_id]
        obj['bboxes'].append(bbox)
        obj['frame_indices'].append(frame_index)
    
    def _create_tracked_object(
        self,
        obj_id: int,
        bbox: List[int],
        frame_index: int
    ):
        """Create new tracked object"""
        self.tracked_objects[obj_id] = {
            'bboxes': [bbox],
            'frame_indices': [frame_index],
            'similarity': 0.0
        }
    
    def _should_extract_features(self, obj_id: int) -> bool:
        """Determine if features should be extracted"""
        if self.config.performance_mode:
            obj = self.tracked_objects[obj_id]
            frame_count = len(obj['frame_indices'])
            return frame_count % self.config.features.extraction_interval == 0
        return obj_id not in self.extracted_objects
    
    def _extract_and_compare_features(
        self,
        obj_id: int,
        frame: np.ndarray,
        bbox: List[int],
        target_data: Dict
    ):
        """Extract and compare features with target"""
        try:
            # Crop object
            cropped_obj = self._crop_object(frame, bbox)
            
            # Extract features
            feature = self.feature_extractor.extract(cropped_obj)
            
            # Compare with target
            similarity = self.feature_comparator.compare(
                target_data['feature'],
                feature
            )
            
            # Update tracking data
            self.tracked_objects[obj_id]['similarity'] = similarity
            self.extracted_objects.add(obj_id)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise ForensicsError(f"Feature extraction failed: {str(e)}")
    
    def _crop_object(self, frame: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Crop object from frame"""
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2] 