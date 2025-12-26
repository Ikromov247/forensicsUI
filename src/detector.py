"""Object detection using YOLOv8 with ByteTrack."""
from ultralytics import YOLO
from config import Config


class Detector:
    """YOLOv8 object detector with tracking support."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = YOLO(self.config.yolo_model)

    def detect(self, source, classes=None, is_image=False):
        """
        Detect objects in image or video frame.

        Args:
            source: Image path, numpy array, or video frame
            classes: List of class IDs to detect (None = all)
            is_image: If True, use single-frame detection; else use tracking

        Returns:
            YOLO results object
        """
        if classes is None:
            classes = list(self.model.names.keys())

        if is_image:
            return self.model(source, classes=classes, device=self.config.device)

        return self.model.track(
            source,
            classes=classes,
            tracker=self.config.tracker_config,
            persist=True,
            device=self.config.device,
            verbose=False
        )

    @property
    def class_names(self):
        """Get dictionary of class ID to name mappings."""
        return self.model.names
