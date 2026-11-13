"""Continuously drain the camera, retaining only the newest frame."""

# Define structured records or create modified copies of configuration records.
from dataclasses import dataclass
# Report runtime status and errors through the standard logging system.
import logging
# Detect the operating system when choosing a camera backend.
import platform
# Coordinate background capture with the foreground processing loop.
import threading
# Measure monotonic elapsed time and perform short waits.
import time

# Use OpenCV for camera capture, image operations, or preview windows.
import cv2
# Work with image arrays and numerical lookup tables.
import numpy as np

# Load validated configuration types, defaults, or configuration helpers.
from settings import CameraConfig

# Keep webcam diagnostics under this module's logger name.
log = logging.getLogger(__name__)


# Distinguish capture failures from unrelated runtime errors.
class CameraError(RuntimeError):
    # The exception needs no behavior beyond RuntimeError.
    pass


# Generate data-class helpers and prevent reassignment of instance fields.
@dataclass(frozen=True)
# Package an image together with its host receipt time.
class CapturedFrame:
    # Store the uint8 BGR image supplied by OpenCV.
    image: np.ndarray
    # Record monotonic seconds when this frame was received.
    timestamp: float


# Own the capture device and a worker that keeps only the newest frame.
class Camera:
    # Prepare synchronization state without opening hardware yet.
    def __init__(self, config: CameraConfig):
        # Retain the validated camera settings.
        self.config = config
        # No native capture handle has been opened yet.
        self._capture = None
        # The worker thread will be created after successful initialization.
        self._thread: threading.Thread | None = None
        # Protect shared capture state and allow consumers to wait for updates.
        self._condition = threading.Condition()
        # Use a thread-safe flag to request worker shutdown.
        self._stop = threading.Event()
        # Reserve one slot for the newest frame instead of building a queue.
        self._latest: CapturedFrame | None = None
        # Store worker errors so the main thread can report them.
        self._error: str | None = None

    # Open the camera when entering a with block.
    def __enter__(self):
        # Translate configured backend names into OpenCV constants.
        backends = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}
        # Prefer DirectShow on Windows and V4L2 on Linux.
        preferred = "dshow" if platform.system() == "Windows" else "v4l2" if platform.system() == "Linux" else "any"
        # In automatic mode, also try OpenCV's default backend as a fallback.
        names = [preferred, "any"] if self.config.backend == "auto" else [self.config.backend]
        # Avoid trying the same backend twice while preserving order.
        for name in dict.fromkeys(names):
            # Try opening the configured camera index with this backend.
            capture = cv2.VideoCapture(self.config.index, backends[name])
            # Keep the first capture handle that successfully opened.
            if capture.isOpened():
                # Transfer ownership of the working native handle to this object.
                self._capture = capture
                # Stop trying backends after the first success.
                break
            # Release an unsuccessful handle before trying the next backend.
            capture.release()
        # Fail clearly if none of the requested backends could open the device.
        if self._capture is None:
            # Give the user concrete camera-access checks to perform.
            raise CameraError(f"Cannot open camera {self.config.index}; check permissions, privacy shutter, index and other camera apps")
        # Handle failures from the operations in this block.
        try:
            # Request the configured frame width.
            for prop, value in [(cv2.CAP_PROP_FRAME_WIDTH, self.config.width),
                                # Request the configured frame height.
                                (cv2.CAP_PROP_FRAME_HEIGHT, self.config.height),
                                # Request FPS and a minimal driver buffer where supported.
                                (cv2.CAP_PROP_FPS, self.config.fps), (cv2.CAP_PROP_BUFFERSIZE, 1)]:
                # Apply each preference; the driver may ignore unsupported settings.
                self._capture.set(prop, value)  # drivers may ignore these requests
            # Log both requested settings and actual values reported by the driver.
            log.info("Camera opened via %s: requested %dx%d; reported %dx%d at %.1f FPS",
                     # Include the chosen backend and requested image dimensions.
                     name, self.config.width, self.config.height,
                     # Read the actual width and height from the capture handle.
                     self._capture.get(cv2.CAP_PROP_FRAME_WIDTH), self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                     # Read the driver's reported capture frame rate.
                     self._capture.get(cv2.CAP_PROP_FPS))
            # Create a daemon worker so a blocked driver cannot prevent process exit.
            self._thread = threading.Thread(target=self._capture_loop, name="webcam", daemon=True)
            # Begin continuously draining the webcam.
            self._thread.start()
        # Also clean up if initialization is interrupted rather than merely failing.
        except BaseException:
            # Release the opened handle when worker setup did not complete.
            self._capture.release()
            # Preserve the original initialization failure for the caller.
            raise
        # Expose this camera instance as the with-block value.
        return self

    # Read frames in the background until shutdown or a capture failure.
    def _capture_loop(self) -> None:
        # Handle failures from the operations in this block.
        try:
            # Check the stop flag before starting another native read.
            while not self._stop.is_set():
                # Read a success flag and a newly captured frame from OpenCV.
                ok, frame = self._capture.read()
                # Reject failed reads and empty image buffers.
                if not ok or frame is None or frame.size == 0:
                    # Convert a broken capture stream into an actionable camera error.
                    raise CameraError("Camera stopped delivering frames; check connection and camera access")
                # Lock shared state before publishing the next frame.
                with self._condition:
                    # Replace the previous frame and timestamp receipt on the host.
                    self._latest = CapturedFrame(frame, time.monotonic())
                    # Wake consumers waiting for a newer frame.
                    self._condition.notify_all()
        # Capture worker exceptions instead of losing them silently in a thread.
        except Exception as exc:
            # Publish the error under the same lock as normal frame updates.
            with self._condition:
                # Preserve a descriptive message for the foreground thread.
                self._error = f"Camera capture failed: {exc}"
                # Wake readers so they notice failure instead of waiting until timeout.
                self._condition.notify_all()
        # Run cleanup whether the preceding operations succeeded or failed.
        finally:
            # Let the capture worker release its handle after its last read.
            self._capture.release()

    # Wait for a frame newer than the caller's last processed timestamp.
    def read(self, after: float = -1.0) -> CapturedFrame:
        # Set an absolute deadline for this read operation.
        deadline = time.monotonic() + self.config.read_timeout
        # Read shared frame and error state while holding the condition lock.
        with self._condition:
            # Recheck conditions whenever the worker wakes this reader.
            while True:
                # Give capture errors priority over any previously cached frame.
                if self._error:
                    # Raise the worker's error in the foreground thread.
                    raise CameraError(self._error)
                # Return only a frame that has not already been processed.
                if self._latest is not None and self._latest.timestamp > after:
                    # Return the latest frame without consuming a backlog of older ones.
                    return self._latest
                # Calculate how much of the read timeout remains.
                remaining = deadline - time.monotonic()
                # Stop waiting when the deadline has been reached.
                if remaining <= 0:
                    # Explain that no fresh frame arrived before the timeout.
                    raise CameraError("Timed out waiting for a new webcam frame")
                # Release the lock while waiting briefly for a notification.
                self._condition.wait(min(remaining, 0.1))

    # Ask the capture worker to stop without releasing during a native read.
    def close(self) -> None:
        # Set the worker's shutdown flag.
        self._stop.set()
        # Join the worker only if one was actually created.
        if self._thread is not None:
            # Bound the shutdown wait to one second.
            self._thread.join(timeout=1.0)
            # Detect a native driver read that has not returned yet.
            if self._thread.is_alive():
                # Never release concurrently with a native read: some drivers crash.
                # Explain why process exit must finish releasing a stuck device.
                log.warning("Camera driver is blocked; the OS will release it when the process exits")
        # Handle a device opened before any worker was created.
        elif self._capture is not None:
            # Release that handle directly because there is no concurrent read.
            self._capture.release()

    # Perform the same cleanup on normal and exceptional with-block exits.
    def __exit__(self, *args) -> None:
        # Delegate device cleanup to the common shutdown method.
        self.close()
