"""Continuously drain the camera, retaining only the newest frame."""

from dataclasses import dataclass
import logging
import platform
import threading
import time

import cv2
import numpy as np

from settings import CameraConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    timestamp: float


class Camera:
    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture = None
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest: CapturedFrame | None = None
        self._error: str | None = None

    def __enter__(self):
        backends = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}
        preferred = "dshow" if platform.system() == "Windows" else "v4l2" if platform.system() == "Linux" else "any"
        names = [preferred, "any"] if self.config.backend == "auto" else [self.config.backend]
        for name in dict.fromkeys(names):
            capture = cv2.VideoCapture(self.config.index, backends[name])
            if capture.isOpened():
                self._capture = capture
                break
            capture.release()
        if self._capture is None:
            raise CameraError(f"Cannot open camera {self.config.index}; check permissions, privacy shutter, index and other camera apps")
        try:
            for prop, value in [(cv2.CAP_PROP_FRAME_WIDTH, self.config.width),
                                (cv2.CAP_PROP_FRAME_HEIGHT, self.config.height),
                                (cv2.CAP_PROP_FPS, self.config.fps), (cv2.CAP_PROP_BUFFERSIZE, 1)]:
                self._capture.set(prop, value)  # drivers may ignore these requests
            log.info("Camera opened via %s: requested %dx%d; reported %dx%d at %.1f FPS",
                     name, self.config.width, self.config.height,
                     self._capture.get(cv2.CAP_PROP_FRAME_WIDTH), self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                     self._capture.get(cv2.CAP_PROP_FPS))
            self._thread = threading.Thread(target=self._capture_loop, name="webcam", daemon=True)
            self._thread.start()
        except BaseException:
            self._capture.release()
            raise
        return self

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None or frame.size == 0:
                    raise CameraError("Camera stopped delivering frames; check connection and camera access")
                with self._condition:
                    self._latest = CapturedFrame(frame, time.monotonic())
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = f"Camera capture failed: {exc}"
                self._condition.notify_all()
        finally:
            self._capture.release()

    def read(self, after: float = -1.0) -> CapturedFrame:
        deadline = time.monotonic() + self.config.read_timeout
        with self._condition:
            while True:
                if self._error:
                    raise CameraError(self._error)
                if self._latest is not None and self._latest.timestamp > after:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraError("Timed out waiting for a new webcam frame")
                self._condition.wait(min(remaining, 0.1))

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                # Never release concurrently with a native read: some drivers crash.
                log.warning("Camera driver is blocked; the OS will release it when the process exits")
        elif self._capture is not None:
            self._capture.release()

    def __exit__(self, *args) -> None:
        self.close()
