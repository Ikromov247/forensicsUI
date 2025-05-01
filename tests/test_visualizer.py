import unittest
import numpy as np
import cv2
from pathlib import Path

from src.visualization.visualizer import Visualizer
from src.database.models import ObjectData

class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = Visualizer()
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.test_bbox = [100, 100, 200, 200]
        self.test_obj = ObjectData(
            obj_id=1,
            bbox=[self.test_bbox],
            similarity=0.8
        )
        
    def test_draw_bounding_box(self):
        """Test drawing bounding box"""
        frame = self.visualizer.draw_bounding_box(
            self.test_frame.copy(),
            self.test_bbox
        )
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, self.test_frame.shape)
        
    def test_draw_text(self):
        """Test drawing text"""
        frame = self.visualizer.draw_text(
            self.test_frame.copy(),
            "Test",
            (100, 100)
        )
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, self.test_frame.shape)
        
    def test_visualize_object(self):
        """Test object visualization"""
        frame = self.visualizer.visualize_object(
            self.test_frame.copy(),
            self.test_obj
        )
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, self.test_frame.shape)
        
    def test_display_frame(self):
        """Test frame display"""
        # This test just checks if the method runs without error
        # since we can't actually test the display
        try:
            self.visualizer.display_frame(self.test_frame)
        except Exception as e:
            self.fail(f"display_frame raised an exception: {str(e)}")
            
    def test_wait_key(self):
        """Test key wait"""
        # This test just checks if the method runs without error
        try:
            self.visualizer.wait_key(1)
        except Exception as e:
            self.fail(f"wait_key raised an exception: {str(e)}")
            
    def test_destroy_windows(self):
        """Test window destruction"""
        # This test just checks if the method runs without error
        try:
            self.visualizer.destroy_windows()
        except Exception as e:
            self.fail(f"destroy_windows raised an exception: {str(e)}")

if __name__ == "__main__":
    unittest.main() 