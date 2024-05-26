"""
1. Implement toggle logic
2. Method for drawing bounding boxes
"""
import cv2


class Visualization:
    def __init__(self, is_on=True):
        self.is_on = is_on

    def draw_bounding_box(self, frame, bbox, color=(0, 255, 0), thickness=2):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def draw_text(self, frame, text, position, font_scale=0.3, color=(0, 255, 0), thickness=1):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def visualize_object(self, frame, obj):
        bbox = obj.bbox[-1]
        self.draw_bounding_box(frame, bbox)
        text = f"ID: {obj.obj_id}, Sim: {obj.similarity:.2f}"
        text_position = (bbox[0], bbox[1] - 10)  # Above top-left corner
        color = (0, 255, 0) if obj.similarity < 0.9 else (255, 0, 0)
        self.draw_text(frame, text, text_position, color=color)

    def display_frame(self, frame, window_name="Object Tracking"):
        cv2.imshow(window_name, frame)
