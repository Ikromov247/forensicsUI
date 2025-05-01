import unittest
import numpy as np
import cv2
from pathlib import Path

from src.core.application import Application

class TestApplication(unittest.TestCase):
    """Test cases for Application"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = Application(db_name="test_app", visualization=False)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
    def test_process_frame(self):
        """Test frame processing"""
        processed_frame = self.app.process_frame(self.test_frame)
        self.assertIsInstance(processed_frame, np.ndarray)
        self.assertEqual(processed_frame.shape, self.test_frame.shape)
        
    def test_match_object(self):
        """Test object matching"""
        # Create test features
        features = np.random.rand(100)
        feature_vectors = np.array([np.random.rand(100) for _ in range(5)])
        
        # Test with empty features
        obj_id, similarity = self.app._match_object(features)
        self.assertIsNone(obj_id)
        self.assertIsNone(similarity)
        
        # Add features to app
        self.app.features = {
            i: type('obj', (), {'features': vec})()
            for i, vec in enumerate(feature_vectors)
        }
        
        # Test matching
        obj_id, similarity = self.app._match_object(features)
        if obj_id is not None:
            self.assertIn(obj_id, self.app.features)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
            
    def test_update_object(self):
        """Test object update"""
        # Create test object
        obj_id = 1
        self.app.objects[obj_id] = type('obj', (), {
            'bbox': [],
            'frame_ids': [],
            'similarity': None
        })()
        
        # Test update
        detection = {
            "bbox": [100, 100, 200, 200],
            "class": 0,
            "confidence": 0.9
        }
        similarity = 0.8
        
        self.app._update_object(obj_id, detection, similarity)
        
        obj = self.app.objects[obj_id]
        self.assertEqual(len(obj.bbox), 1)
        self.assertEqual(len(obj.frame_ids), 1)
        self.assertEqual(obj.similarity, similarity)
        
    def test_create_object(self):
        """Test object creation"""
        # Test creation
        detection = {
            "bbox": [100, 100, 200, 200],
            "class": 0,
            "confidence": 0.9
        }
        features = np.random.rand(100)
        
        obj_id = self.app._create_object(detection, features)
        
        self.assertIn(obj_id, self.app.objects)
        self.assertIn(obj_id, self.app.features)
        
        obj = self.app.objects[obj_id]
        self.assertEqual(len(obj.bbox), 1)
        self.assertEqual(len(obj.frame_ids), 1)
        self.assertEqual(obj.cls, detection["class"])
        self.assertEqual(obj.conf, detection["confidence"])
        
        feat = self.app.features[obj_id]
        np.testing.assert_array_equal(feat.features, features)
        
    def test_cleanup(self):
        """Test cleanup"""
        # This test just checks if the method runs without error
        try:
            self.app.cleanup()
        except Exception as e:
            self.fail(f"cleanup raised an exception: {str(e)}")

if __name__ == "__main__":
    unittest.main() 