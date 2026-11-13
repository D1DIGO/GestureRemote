"""Optional enhancement; brightness is not a proxy for recognition accuracy."""

# Use OpenCV for camera capture, image operations, or preview windows.
import cv2
# Work with image arrays and numerical lookup tables.
import numpy as np

# Load validated configuration types, defaults, or configuration helpers.
from settings import PreprocessingConfig


# Apply optional deterministic enhancements without changing gesture logic.
class ImageProcessor:
    # Prepare reusable filters once for the configured session.
    def __init__(self, config: PreprocessingConfig):
        # Keep the enhancement settings for subsequent frames.
        self.config = config
        # gamma > 1 brightens; build once, not per frame.
        # Precompute byte-to-byte gamma correction; gamma above one brightens pixels.
        self._lut = np.clip((np.arange(256) / 255.0) ** (1.0 / config.gamma) * 255, 0, 255).astype(np.uint8)
        # Create a local-contrast filter only when CLAHE is requested.
        self._clahe = cv2.createCLAHE(clipLimit=config.clahe_clip_limit, tileGridSize=(8, 8)) if config.clahe else None

    # Enhance one BGR frame; force also supports a comparison-only preview.
    def process(self, frame: np.ndarray, *, force: bool = False) -> np.ndarray:
        # Bypass processing when neither inference nor comparison needs it.
        if not self.config.enabled and not force:
            # Return the same array without copying or modifying it.
            return frame
        # Start from the input image and replace the working result as needed.
        result = frame
        # Optionally smooth small image variations before contrast adjustment.
        if self.config.denoise:
            # Use a small Gaussian blur, which can also soften useful hand detail.
            result = cv2.GaussianBlur(result, (3, 3), 0)
        # Skip the pixel conversion when brightness and contrast are neutral.
        if self.config.contrast != 1 or self.config.brightness != 0:
            # convertScaleAbs would incorrectly reflect negative brightness at zero.
            # Use float arithmetic, clamp to byte range, and avoid reflecting negatives.
            result = np.clip(result.astype(np.float32) * self.config.contrast + self.config.brightness, 0, 255).astype(np.uint8)
        # Skip the lookup when gamma correction is neutral.
        if self.config.gamma != 1:
            # Apply the precomputed gamma mapping to every pixel channel.
            result = cv2.LUT(result, self._lut)
        # Run local contrast enhancement only when its filter exists.
        if self._clahe is not None:
            # Separate luminance from color using the LAB color space.
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            # Enhance only the luminance channel to limit color changes.
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            # Convert the enhanced image back to OpenCV's BGR format.
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # Return the final image; the caller chooses whether to recognize it.
        return result
