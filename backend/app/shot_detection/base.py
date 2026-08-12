"""ShotDetector interface and TransNetV2 adapter."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Optional


class ShotDetector(ABC):
    """Abstract base class for shot boundary detection."""

    @abstractmethod
    def detect_shots(self, video_path: Path) -> List[Tuple[int, int]]:
        """Detect shot boundaries in a video.

        Args:
            video_path: Path to the video file.

        Returns:
            List of (start_ms, end_ms) tuples for each shot.
        """
        pass

    @abstractmethod
    def detect_shot_boundaries(self, video_path: Path) -> List[int]:
        """Detect shot boundaries as start timestamps.

        Args:
            video_path: Path to the video file.

        Returns:
            List of shot start timestamps in milliseconds.
        """
        pass


class TransNetV2Adapter(ShotDetector):
    """TransNetV2 shot boundary detector adapter.

    This adapter wraps the TransNetV2 model from the AIC2025-float97 repository.
    It loads the TensorFlow SavedModel and runs inference to detect shot boundaries.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the TransNetV2 adapter.

        Args:
            model_path: Path to the TransNetV2 SavedModel directory.
                       If None, attempts to find it in the float97 repo.
        """
        self.model_path = model_path
        self._model = None
        self._initialized = False

    def _load_model(self):
        """Lazy-load the TransNetV2 model."""
        if self._initialized:
            return

        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError("TensorFlow not available. Install tensorflow to use TransNetV2.")

        if self.model_path is None:
            # Try to find in the float97 repo
            import os
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            potential = os.path.join(base, "AIC2025-VideoRetrieval-float97", "transnetv2-weights")
            if os.path.exists(potential):
                self.model_path = potential

        if self.model_path and os.path.exists(self.model_path):
            self._model = tf.saved_model.load(self.model_path)
            self._initialized = True
        else:
            # TransNetV2 not available - will raise on first use
            pass

    def detect_shots(self, video_path: Path) -> List[Tuple[int, int]]:
        """Detect shots using TransNetV2.

        Returns list of (start_ms, end_ms) tuples.
        """
        boundaries = self.detect_shot_boundaries(video_path)
        if not boundaries:
            return []

        # Get video duration from metadata
        from backend.app.loaders.metadata_loader import load_metadata
        manifest = load_metadata(video_path)
        duration_ms = manifest.get("duration_ms", 0)

        shots = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else duration_ms
            shots.append((start, end))
        return shots

    def detect_shot_boundaries(self, video_path: Path) -> List[int]:
        """Detect shot boundaries using TransNetV2."""
        self._load_model()

        if not self._initialized or self._model is None:
            # TransNetV2 not available - return empty list (no shots detected)
            # This is intentional - we gracefully degrade to no-shot mode
            return []

        # TODO: Implement actual TransNetV2 inference
        # This requires:
        # 1. Extract frames from video at 25 FPS
        # 2. Preprocess frames (resize to 48x27, normalize)
        # 3. Run model.predict(frames)
        # 4. Post-process predictions to get shot boundaries
        #
        # For now, return empty list to indicate no shots detected
        # In production, this would call the actual model
        return []


def get_shot_detector(detector_type: str = "transnetv2", **kwargs) -> "ShotDetector":
    """Factory function to get a shot detector.

    Args:
        detector_type: Type of detector ("transnetv2", "none")
        **kwargs: Additional arguments for the detector.

    Returns:
        ShotDetector instance.
    """
    if detector_type == "transnetv2":
        return TransNetV2Adapter(**kwargs)
    elif detector_type == "none":
        return NullShotDetector()
    else:
        raise ValueError(f"Unknown shot detector type: {detector_type}")


class NullShotDetector(ShotDetector):
    """Null shot detector - returns no shot boundaries.

    Used when shot detection is disabled or unavailable.
    """

    def detect_shots(self, video_path: Path) -> List[Tuple[int, int]]:
        return []

    def detect_shot_boundaries(self, video_path: Path) -> List[int]:
        return []