"""Video processing pipeline for object detection and matching."""
import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from config import Config, OUTPUTS_DIR
from detector import Detector
from features import FeatureExtractor
from visualizer import Visualizer
from database import Database, DetectionRecord


@dataclass
class TrackedObject:
    """An object being tracked across frames."""
    obj_id: int
    class_id: int
    class_name: str
    confidence: float
    bboxes: List[List[int]] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)
    features: Optional[np.ndarray] = None
    similarity: float = 0.0


@dataclass
class ProcessingResult:
    """Result of video processing."""
    output_video_path: str
    matches: List[DetectionRecord]
    total_objects: int
    total_frames: int


class VideoProcessor:
    """Process video to find objects matching a target."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.detector = Detector(self.config)
        self.extractor = FeatureExtractor(self.config)
        self.visualizer = Visualizer(self.config)

    def process_target_image(self, image: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
        """
        Process target image to extract features.

        Args:
            image: Target image (BGR)

        Returns:
            Tuple of (feature_vector, class_id, bbox)
        """
        results = self.detector.detect(image, is_image=True)

        if not results or len(results[0].boxes) == 0:
            raise ValueError("No objects detected in target image")

        # Get the detection with highest confidence
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        best_box = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
        best_class = int(boxes.cls[best_idx].item())

        # Crop and extract features
        cropped = self.visualizer.crop_object(image, best_box)
        features = self.extractor.extract(cropped)

        return features, best_class, best_box

    def process_video(
        self,
        video_path: str,
        target_features: np.ndarray,
        target_class: int,
        progress_callback: Optional[Callable[[float], None]] = None,
        db_name: str = "forensics"
    ) -> ProcessingResult:
        """
        Process video to find matching objects.

        Args:
            video_path: Path to input video
            target_features: Feature vector of target object
            target_class: Class ID of target object
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            db_name: Database name for storing results

        Returns:
            ProcessingResult with output video path and matches
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        output_path = OUTPUTS_DIR / f"processed_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Track objects
        tracked: Dict[int, TrackedObject] = {}
        extracted_ids = set()

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and track objects of target class
            results = self.detector.detect(frame, classes=[target_class])

            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                for i in range(len(boxes)):
                    # Get tracking ID if available
                    if boxes.id is not None:
                        obj_id = int(boxes.id[i].item())
                    else:
                        obj_id = i

                    bbox = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                    conf = float(boxes.conf[i].item())
                    cls = int(boxes.cls[i].item())

                    # Update or create tracked object
                    if obj_id not in tracked:
                        tracked[obj_id] = TrackedObject(
                            obj_id=obj_id,
                            class_id=cls,
                            class_name=self.detector.class_names.get(cls, str(cls)),
                            confidence=conf
                        )

                    obj = tracked[obj_id]
                    obj.bboxes.append(bbox)
                    obj.frame_ids.append(frame_idx)
                    obj.confidence = max(obj.confidence, conf)

                    # Extract features periodically
                    should_extract = (
                        obj_id not in extracted_ids or
                        (self.config.performance_mode and
                         len(obj.frame_ids) % self.config.extraction_interval == 0)
                    )

                    if should_extract:
                        try:
                            cropped = self.visualizer.crop_object(frame, bbox)
                            if cropped.size > 0:
                                features = self.extractor.extract(cropped)
                                similarity = self.extractor.compare(target_features, features)
                                obj.features = features
                                obj.similarity = max(obj.similarity, similarity)
                                extracted_ids.add(obj_id)
                        except Exception:
                            pass  # Skip extraction errors

                    # Draw on frame
                    is_match = obj.similarity >= self.config.similarity_threshold
                    self.visualizer.draw_detection(
                        frame, bbox, obj_id, obj.similarity, is_match
                    )

            writer.write(frame)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx / total_frames)

        cap.release()
        writer.release()

        # Save to database
        db = Database(db_name)
        db.clear()
        for obj in tracked.values():
            record = DetectionRecord(
                obj_id=obj.obj_id,
                class_id=obj.class_id,
                class_name=obj.class_name,
                confidence=obj.confidence,
                similarity=obj.similarity,
                frame_ids=obj.frame_ids,
                bboxes=obj.bboxes,
                features=obj.features
            )
            db.save_detection(record)

        # Get top matches
        matches = db.get_top_matches(limit=10)

        return ProcessingResult(
            output_video_path=str(output_path),
            matches=matches,
            total_objects=len(tracked),
            total_frames=frame_idx
        )
