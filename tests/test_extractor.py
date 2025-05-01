import unittest
import numpy as np
import torch
from pathlib import Path

from src.core.features.extractor import FeatureExtractor

class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = FeatureExtractor()
        self.test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        self.test_bbox = [0, 0, 224, 224]
        
    def test_extract(self):
        """Test feature extraction"""
        features = self.extractor.extract(self.test_image, self.test_bbox)
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.ndim, 1)
        self.assertGreater(features.size, 0)
        
    def test_extract_with_different_models(self):
        """Test feature extraction with different models"""
        models = ["resnet", "inception"]
        
        for model in models:
            extractor = FeatureExtractor(model_name=model)
            features = extractor.extract(self.test_image, self.test_bbox)
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(features.ndim, 1)
            
    def test_extract_with_different_devices(self):
        """Test feature extraction with different devices"""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        if torch.backends.mps.is_available():
            devices.append("mps")
            
        for device in devices:
            extractor = FeatureExtractor(device=device)
            features = extractor.extract(self.test_image, self.test_bbox)
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(features.ndim, 1)
            
    def test_preprocess(self):
        """Test image preprocessing"""
        preprocessed = self.extractor._preprocess(self.test_image)
        self.assertIsInstance(preprocessed, torch.Tensor)
        self.assertEqual(preprocessed.ndim, 4)  # Batch dimension
        self.assertEqual(preprocessed.shape[1], 3)  # Channels
        self.assertEqual(preprocessed.shape[2], 224)  # Height
        self.assertEqual(preprocessed.shape[3], 224)  # Width

if __name__ == "__main__":
    unittest.main() 