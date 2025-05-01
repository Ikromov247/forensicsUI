import unittest
import numpy as np
import cv2
from pathlib import Path

from src.core.detection.detector import ObjectDetector

class TestObjectDetector(unittest.TestCase):
    """Test cases for ObjectDetector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = ObjectDetector()
        self.test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
    def test_detect(self):
        """Test object detection"""
        detections = self.detector.detect(self.test_image)
        self.assertIsInstance(detections, list)
        
        if detections:
            detection = detections[0]
            self.assertIn("bbox", detection)
            self.assertIn("confidence", detection)
            self.assertIn("class", detection)
            
            bbox = detection["bbox"]
            self.assertEqual(len(bbox), 4)
            self.assertTrue(all(isinstance(x, (int, float)) for x in bbox))
            
    def test_detect_with_threshold(self):
        """Test object detection with confidence threshold"""
        detections = self.detector.detect(self.test_image, confidence=0.5)
        self.assertIsInstance(detections, list)
        
        if detections:
            for detection in detections:
                self.assertGreaterEqual(detection["confidence"], 0.5)
                
    def test_detect_with_classes(self):
        """Test object detection with specific classes"""
        detections = self.detector.detect(self.test_image, classes=[0])  # Person class
        self.assertIsInstance(detections, list)
        
        if detections:
            for detection in detections:
                self.assertEqual(detection["class"], 0)

if __name__ == "__main__":
    unittest.main() 