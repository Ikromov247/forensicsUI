import os
import sqlite3
import contextlib
from helpers import convert_string_to_list, convert_list_to_string, convert_bbox

"""
CREATE TABLE orig_img (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_values BLOB  -- Or whatever datatype you use for frame data
);

CREATE TABLE objects (
    obj_id INTEGER PRIMARY KEY AUTOINCREMENT,
    cls TEXT,
    conf REAL,
    bbox TEXT,  -- Or appropriate datatype for bounding boxes 
    features BLOB, 
    similarity_score REAL,
    frame_id INTEGER, -- The column referencing orig_img
    FOREIGN KEY (frame_id) REFERENCES orig_img(frame_id) 
);
"""


class DatabaseConnection:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.cur = self.conn.cursor()

    def execute(self, query, parameters=None):
        if parameters:
            self.cur.execute(query, parameters)
        else:
            self.cur.execute(query)
        self.conn.commit()

    def query(self, query, parameters=None):
        if parameters:
            self.cur.execute(query, parameters)
        else:
            self.cur.execute(query)
        return self.cur.fetchall()

    def close(self):
        self.conn.close()


class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name

    @contextlib.contextmanager
    def connect_db(self):
        path = "database"
        db_conn = DatabaseConnection(f"{path}/{self.db_name}.db")
        try:
            yield db_conn
        finally:
            db_conn.close()

    def create_objects_table(self):
        with self.connect_db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS objects (
                            obj_id INTEGER PRIMARY KEY,
                            class INTEGER,
                            conf REAL,
                            bbox TEXT,
                            frame_ids TEXT,
                            similarity REAL
                        )""")

    def create_features_table(self):
        with self.connect_db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS features (
                                obj_id INTEGER PRIMARY KEY,
                                features BLOB
                                )""")

    def insert_to_objects(self, obj_id, cls, conf, bbox, frame_ids, similarity):
        with self.connect_db() as db:
            # todo store bbox-es better
            bbox = convert_list_to_string(bbox)
            frame_ids = convert_list_to_string(frame_ids)
            db.execute("""
                            INSERT INTO objects VALUES
                            (:obj_id, :class, :conf, :bbox, :frame_ids, :similarity)""",
                       {"obj_id": obj_id,
                        "class": cls,
                        "conf": conf,
                        "bbox": bbox,
                        "frame_ids": frame_ids,
                        "similarity": similarity})

    def insert_to_features(self, obj_id, features):
        with self.connect_db() as db:
            features = features.tobytes()
            db.execute("""
                            INSERT INTO features VALUES
                            (:obj_id, :features)
                """, {"obj_id": obj_id, "features": features})

    def query_objects(self):
        """:returns dict with obj_id as key, bbox, frames_id, similarity as value in array"""
        with self.connect_db() as db:
            cur = db.query("""
                            SELECT obj_id, bbox, frame_ids, similarity FROM objects
                """)
            converted_objs = {}
            for obj in cur:
                obj_id = obj[0]
                bbox = convert_bbox(obj[1])
                frames = convert_string_to_list(obj[2])
                similarity = obj[3]
                converted_objs[obj_id] = [frames, bbox, similarity]
            del cur
            del bbox
            del frames
            return converted_objs

    def query_features(self):
        with self.connect_db() as db:
            cur = db.query("""
                        SELECT obj_id, features FROM features
            """)
            return cur
