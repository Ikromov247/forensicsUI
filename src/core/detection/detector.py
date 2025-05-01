import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO


class ObjDetection:
    def __init__(self):
        self.model = self.load_model()

    @staticmethod
    def load_model():
        return YOLO("models/trained_yolo.pt")

    def detect(self, source, classes=None, tracker="bytetrack.yaml", persist=False, isImage=False):
        if classes is None:
            classes = list(self.model.names.keys())
        if isImage:
            return self.model(source, classes=classes, device="mps")
        return self.model.track(source, classes=classes, tracker=tracker, persist=persist, device="mps", verbose=False)
