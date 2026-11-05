"""Optional enhancement; brightness is not a proxy for recognition accuracy."""

import cv2
import numpy as np

from settings import PreprocessingConfig


class ImageProcessor:
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        # gamma > 1 brightens; build once, not per frame.
        self._lut = np.clip((np.arange(256) / 255.0) ** (1.0 / config.gamma) * 255, 0, 255).astype(np.uint8)
        self._clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8)) if config.clahe else None

    def process(self, frame: np.ndarray, *, force: bool = False) -> np.ndarray:
        if not self.config.enabled and not force:
            return frame
        result = frame
        if self.config.denoise:
            result = cv2.GaussianBlur(result, (3, 3), 0)
        if self.config.contrast != 1 or self.config.brightness != 0:
            # convertScaleAbs would incorrectly reflect negative brightness at zero.
            result = np.clip(result.astype(np.float32) * self.config.contrast + self.config.brightness, 0, 255).astype(np.uint8)
        if self.config.gamma != 1:
            result = cv2.LUT(result, self._lut)
        if self._clahe is not None:
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return result
