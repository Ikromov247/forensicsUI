import os
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
from pathlib import Path

from ..exceptions import ApplicationError
from ..logging_config import get_logger
from ..database.manager import DatabaseManager
from ..database.models import ObjectData, FeatureData
from .detection.detector import ObjectDetector
from .features.extractor import FeatureExtractor
from .features.comparator import FeatureComparator
from ..visualization.visualizer import Visualizer

logger = get_logger(__name__)

class Application:
    """
    Main application class that coordinates all components.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 db_name: str = "forensics",
                 visualization: bool = True):
        """
        Initialize the application.
        
        Args:
            config: Configuration dictionary
            db_name: Name of the database
            visualization: Whether to enable visualization
        """
        self.config = config or {}
        self.logger = logger
        
        # Initialize components
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.comparator = FeatureComparator()
        self.visualizer = Visualizer(is_on=visualization)
        
        # Initialize database
        self.db = DatabaseManager(db_name)
        self._setup_database()
        
        # Initialize state
        self.objects: Dict[int, ObjectData] = {}
        self.features: Dict[int, FeatureData] = {}
        self.frame_id = 0
        
    def _setup_database(self):
        """Set up database tables"""
        tables = {
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
        
        self.db.tables = tables
        self.db.create_tables()
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with visualizations
        """
        try:
            # Detect objects
            detections = self.detector.detect(frame)
            
            # Process each detection
            for det in detections:
                # Extract features
                features = self.extractor.extract(frame, det.bbox)
                
                # Compare with existing objects
                obj_id, similarity = self._match_object(features)
                
                # Update or create object
                if obj_id is not None:
                    self._update_object(obj_id, det, similarity)
                else:
                    obj_id = self._create_object(det, features)
                
                # Visualize object
                frame = self.visualizer.visualize_object(frame, self.objects[obj_id])
            
            # Display frame
            self.visualizer.display_frame(frame)
            
            # Increment frame ID
            self.frame_id += 1
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to process frame: {str(e)}")
            raise ApplicationError(f"Frame processing failed: {str(e)}")
    
    def _match_object(self, features: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """
        Match features with existing objects.
        
        Args:
            features: Feature vector to match
            
        Returns:
            Tuple of (object ID, similarity score)
        """
        if not self.features:
            return None, None
            
        # Get feature vectors
        feature_vectors = np.array([f.features for f in self.features.values()])
        obj_ids = list(self.features.keys())
        
        # Find top match
        top_idx, similarity = self.comparator.find_top_matches(features, feature_vectors, k=1)
        
        if top_idx and similarity[0] > 0.8:  # Threshold for matching
            return obj_ids[top_idx[0]], similarity[0]
            
        return None, None
    
    def _update_object(self, obj_id: int, detection: Dict, similarity: float):
        """
        Update an existing object.
        
        Args:
            obj_id: Object ID
            detection: Detection data
            similarity: Similarity score
        """
        obj = self.objects[obj_id]
        obj.bbox.append(detection["bbox"])
        obj.frame_ids.append(self.frame_id)
        obj.similarity = similarity
        
        # Update database
        self.db.save_object(obj)
    
    def _create_object(self, detection: Dict, features: np.ndarray) -> int:
        """
        Create a new object.
        
        Args:
            detection: Detection data
            features: Feature vector
            
        Returns:
            New object ID
        """
        # Generate new object ID
        obj_id = max(self.objects.keys(), default=0) + 1
        
        # Create object data
        obj = ObjectData(
            obj_id=obj_id,
            cls=detection["class"],
            conf=detection["confidence"],
            bbox=[detection["bbox"]],
            frame_ids=[self.frame_id]
        )
        
        # Create feature data
        feat = FeatureData(obj_id=obj_id, features=features)
        
        # Store in memory
        self.objects[obj_id] = obj
        self.features[obj_id] = feat
        
        # Save to database
        self.db.save_object(obj)
        self.db.save_feature(feat)
        
        return obj_id
    
    def run(self, video_path: str):
        """
        Run the application on a video file.
        
        Args:
            video_path: Path to video file
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ApplicationError(f"Failed to open video: {video_path}")
            
            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                self.process_frame(frame)
                
                # Check for exit
                if self.visualizer.wait_key(1) & 0xFF == ord('q'):
                    break
            
            # Clean up
            cap.release()
            self.visualizer.destroy_windows()
            
        except Exception as e:
            logger.error(f"Application run failed: {str(e)}")
            raise ApplicationError(f"Application run failed: {str(e)}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.visualizer.destroy_windows()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise ApplicationError(f"Cleanup failed: {str(e)}") 