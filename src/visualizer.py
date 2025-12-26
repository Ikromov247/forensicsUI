"""Visualization utilities for drawing annotations on frames."""
import cv2
import numpy as np
from config import Config


class Visualizer:
    """Draw bounding boxes and labels on video frames."""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def draw_detection(
        self,
        frame: np.ndarray,
        bbox: list,
        obj_id: int,
        similarity: float = None,
        is_match: bool = False
    ) -> np.ndarray:
        """
        Draw bounding box and label on frame.

        Args:
            frame: Video frame (BGR)
            bbox: Bounding box [x1, y1, x2, y2]
            obj_id: Object tracking ID
            similarity: Similarity score (0-1) if available
            is_match: Whether this is a high-confidence match

        Returns:
            Frame with annotations drawn
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = self.config.match_color if is_match else self.config.bbox_color

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.thickness)

        # Build label
        label = f"ID:{obj_id}"
        if similarity is not None:
            label += f" {similarity:.0%}"

        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - label_h - 10),
            (x1 + label_w + 4, y1),
            color,
            -1  # Filled
        )

        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (255, 255, 255),  # White text
            1
        )

        return frame

    def crop_object(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Crop object region from frame.

        Args:
            frame: Video frame
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Cropped image of the object
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2].copy()
