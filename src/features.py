"""Feature extraction and comparison."""
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity

from config import Config, MODELS_DIR


class FeatureExtractor:
    """Extract feature vectors from images using deep learning models."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = torch.device(self.config.device)

        if self.config.feature_model == "resnet":
            self.model = self._load_resnet()
            self.preprocess = self._resnet_transforms()
        else:  # inception
            self.model = self._load_inception()
            self.preprocess = self._inception_transforms()

    def _resnet_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _inception_transforms(self):
        return transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_resnet(self):
        resnet = models.resnet50()
        resnet.fc = torch.nn.Linear(in_features=2048, out_features=196)
        resnet.to(self.device)

        model_path = MODELS_DIR / "ResNet50_v3.pth"
        if model_path.exists():
            state_dict = torch.load(str(model_path), map_location=self.device)
            resnet.load_state_dict(state_dict)

        # Remove classification layer to get features
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        resnet.eval()
        return resnet

    def _load_inception(self):
        model = models.inception_v3(weights=None, aux_logits=False)

        model_path = MODELS_DIR / "Inception.pth"
        if model_path.exists():
            state_dict = torch.load(str(model_path), map_location=self.device)
            model.load_state_dict(state_dict, strict=False)

        model.fc = torch.nn.Identity()
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.to(self.device)
        model.eval()
        return model

    def extract(self, image) -> np.ndarray:
        """
        Extract feature vector from an image.

        Args:
            image: PIL Image, numpy array (BGR from OpenCV), or torch Tensor

        Returns:
            Feature vector as numpy array
        """
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            # OpenCV uses BGR, convert to RGB
            image = Image.fromarray(image[:, :, ::-1]).convert('RGB')
        elif isinstance(image, torch.Tensor):
            pass  # Already tensor
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        # Preprocess and extract
        tensor = self.preprocess(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        return features.cpu().numpy().flatten()

    def compare(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        Compare two feature vectors using cosine similarity.

        Args:
            feature1: First feature vector
            feature2: Second feature vector

        Returns:
            Similarity score between 0 and 1
        """
        f1 = feature1.reshape(1, -1)
        f2 = feature2.reshape(1, -1)
        return float(cosine_similarity(f1, f2)[0, 0])
