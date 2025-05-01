import unittest
import numpy as np
from pathlib import Path

from src.core.features.comparator import FeatureComparator

class TestFeatureComparator(unittest.TestCase):
    """Test cases for FeatureComparator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.comparator = FeatureComparator()
        self.features1 = np.random.rand(100)
        self.features2 = np.random.rand(100)
        self.features_list = [np.random.rand(100) for _ in range(5)]
        
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        similarity = self.comparator.cosine_similarity(self.features1, self.features2)
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
        
    def test_euclidean_distance(self):
        """Test Euclidean distance calculation"""
        distance = self.comparator.euclidean_distance(self.features1, self.features2)
        self.assertIsInstance(distance, float)
        self.assertGreaterEqual(distance, 0.0)
        
    def test_find_top_matches(self):
        """Test finding top matches"""
        features_array = np.array(self.features_list)
        top_idx, similarity = self.comparator.find_top_matches(
            self.features1, 
            features_array,
            k=3
        )
        
        self.assertIsInstance(top_idx, list)
        self.assertIsInstance(similarity, list)
        self.assertEqual(len(top_idx), 3)
        self.assertEqual(len(similarity), 3)
        self.assertTrue(all(isinstance(x, float) for x in similarity))
        
    def test_compress_features(self):
        """Test feature compression"""
        features_array = np.array(self.features_list)
        compressed = self.comparator.compress_features(features_array, n_components=2)
        
        self.assertIsInstance(compressed, np.ndarray)
        self.assertEqual(compressed.shape[0], len(self.features_list))
        self.assertEqual(compressed.shape[1], 2)
        
    def test_is_match(self):
        """Test match determination"""
        # Test with high similarity
        self.assertTrue(self.comparator.is_match(0.9))
        
        # Test with low similarity
        self.assertFalse(self.comparator.is_match(0.5))
        
        # Test with custom threshold
        self.assertTrue(self.comparator.is_match(0.7, threshold=0.6))
        self.assertFalse(self.comparator.is_match(0.7, threshold=0.8))

if __name__ == "__main__":
    unittest.main() 