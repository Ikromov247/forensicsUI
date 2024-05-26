
class Objects:
    def __init__(self, obj_id, cls, conf, color, similarity=0):
        self.obj_id = obj_id
        self.cls = cls
        self.conf = conf
        self.similarity = similarity
        self.color = color
        self.bbox = []
        self.frame_ids = []

    def add_frame_index(self, frame_index):
        self.frame_ids.append(frame_index)

    def add_bbox(self, bbox):
        self.bbox.append(bbox)

    def __repr__(self):
        return f"Object(id={self.obj_id}, class={self.cls}, conf={self.conf}, similarity={self.similarity})"
