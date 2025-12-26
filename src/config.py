"""Simple configuration for ForensicsUI."""
from dataclasses import dataclass
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Create directories
for dir_path in [MODELS_DIR, DATABASE_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(exist_ok=True)


@dataclass
class Config:
    """Application configuration with sensible defaults."""
    # Detection
    yolo_model: str = str(MODELS_DIR / "trained_yolo.pt")
    tracker_config: str = "bytetrack.yaml"
    confidence_threshold: float = 0.5

    # Feature extraction
    feature_model: str = "inception"  # "resnet" or "inception"
    extraction_interval: int = 5  # Extract features every N frames (performance mode)
    similarity_threshold: float = 0.8

    # Processing
    device: str = "mps"  # "cpu", "cuda", or "mps"
    performance_mode: bool = True

    # Visualization
    bbox_color: tuple = (0, 255, 0)  # Green in BGR
    match_color: tuple = (0, 0, 255)  # Red in BGR
    font_scale: float = 0.5
    thickness: int = 2
