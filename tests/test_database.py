import unittest
import numpy as np
import os
from pathlib import Path

from src.database.manager import DatabaseManager
from src.database.models import ObjectData, FeatureData

class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.db_name = "test_db"
        self.db = DatabaseManager(self.db_name)
        self.test_obj = ObjectData(
            obj_id=1,
            cls=0,
            conf=0.9,
            bbox=[100, 100, 200, 200],
            frame_ids=[1, 2, 3],
            similarity=0.8
        )
        self.test_feat = FeatureData(
            obj_id=1,
            features=np.random.rand(100)
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        # Remove test database
        db_path = Path("database") / f"{self.db_name}.db"
        if db_path.exists():
            os.remove(db_path)
            
    def test_save_object(self):
        """Test saving object"""
        self.db.save_object(self.test_obj)
        
        # Verify object was saved
        objects = self.db.get_objects()
        self.assertIn(self.test_obj.obj_id, objects)
        saved_obj = objects[self.test_obj.obj_id]
        self.assertEqual(saved_obj.obj_id, self.test_obj.obj_id)
        self.assertEqual(saved_obj.cls, self.test_obj.cls)
        self.assertEqual(saved_obj.conf, self.test_obj.conf)
        self.assertEqual(saved_obj.bbox, self.test_obj.bbox)
        self.assertEqual(saved_obj.frame_ids, self.test_obj.frame_ids)
        self.assertEqual(saved_obj.similarity, self.test_obj.similarity)
        
    def test_save_feature(self):
        """Test saving feature"""
        self.db.save_feature(self.test_feat)
        
        # Verify feature was saved
        features = self.db.get_features()
        self.assertIn(self.test_feat.obj_id, features)
        saved_feat = features[self.test_feat.obj_id]
        self.assertEqual(saved_feat.obj_id, self.test_feat.obj_id)
        np.testing.assert_array_equal(saved_feat.features, self.test_feat.features)
        
    def test_get_objects(self):
        """Test getting objects"""
        # Save test object
        self.db.save_object(self.test_obj)
        
        # Get objects
        objects = self.db.get_objects()
        self.assertIsInstance(objects, dict)
        self.assertIn(self.test_obj.obj_id, objects)
        
    def test_get_features(self):
        """Test getting features"""
        # Save test feature
        self.db.save_feature(self.test_feat)
        
        # Get features
        features = self.db.get_features()
        self.assertIsInstance(features, dict)
        self.assertIn(self.test_feat.obj_id, features)
        
    def test_serialization(self):
        """Test data serialization"""
        # Test bbox serialization
        bbox_str = self.db._serialize_bbox(self.test_obj.bbox[-1])
        bbox = self.db._deserialize_bbox(bbox_str)
        self.assertEqual(bbox, self.test_obj.bbox[-1])
        
        # Test list serialization
        list_str = self.db._serialize_list(self.test_obj.frame_ids)
        lst = self.db._deserialize_list(list_str)
        self.assertEqual(lst, self.test_obj.frame_ids)

if __name__ == "__main__":
    unittest.main() 