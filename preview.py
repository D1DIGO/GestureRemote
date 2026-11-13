"""Small OpenCV status window; never used in no-preview mode."""

# Use OpenCV for camera capture, image operations, or preview windows.
import cv2
# Work with image arrays and numerical lookup tables.
import numpy as np

# Share the timestamped gesture record across application layers.
from recognition.observation import GestureObservation

# Use a stable window name for display, key hints, and visibility checks.
WINDOW = "GestureRemote - Q / Esc to quit"


# Render the camera image and runtime feedback in a small OpenCV window.
class Preview:
    # Create the native preview window when preview mode is enabled.
    def __init__(self):
        # Allow the user to resize the window.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        # Choose its initial on-screen size independently of capture resolution.
        cv2.resizeWindow(WINDOW, 1100, 720)

    # Draw one preview frame and return whether the session should continue.
    def show(self, original: np.ndarray, enhanced: np.ndarray | None, observation: GestureObservation,
             # Require runtime status fields as named arguments for readability.
             *, active: bool, last_action: str, status: str, fps: float,
             # Include processing mode and the state machine's explanatory feedback.
             preprocessing: bool, dry_run: bool, feedback: str = "") -> bool:
        # Scale an image to the preview width while preserving its aspect ratio.
        def fit(frame):
            # Keep the resulting height positive even for unusual input dimensions.
            return cv2.resize(frame, (640, max(1, round(frame.shape[0] * 640 / frame.shape[1]))))

        # Resize the primary image for display.
        feed = fit(original)
        # Add a second panel only when an enhanced comparison image was supplied.
        if enhanced is not None:
            # Place original and enhanced images side by side.
            feed = np.hstack((feed, fit(enhanced)))
        # Separate header keeps labels readable even with 320x240 capture.
        # Reserve a black header above the image for readable status text.
        canvas = np.zeros((feed.shape[0] + 183, feed.shape[1], 3), dtype=np.uint8)
        # Copy the image into the region below the header.
        canvas[183:] = feed
        # Build the status lines in their displayed order.
        lines = [
            # Show whether control is enabled, the backend mode, and processed FPS.
            f"Control: {'ON' if active else 'OFF'}   {'DRY RUN' if dry_run else 'VLC'}   FPS: {fps:.1f}",
            # Display classification separately from the decision to execute a command.
            f"Gesture: {observation.gesture or 'None / ambiguous'}   Confidence: {observation.confidence:.2f}",
            # Explain hold progress, low confidence, cooldown, or release requirements.
            feedback,
            # Display the most recent successful or simulated media action.
            f"Last command: {last_action}",
            # Show the latest backend result or runtime warning.
            status,
            # Identify the inference input and the meaning of comparison panels.
            f"Input: {'enhanced' if preprocessing else 'original'} | " + ("Left: original / Right: enhanced" if enhanced is not None else "Hold Victory to toggle control"),
        ]
        # Draw each status line at a different vertical position.
        for i, line in enumerate(lines):
            # Highlight the first line in green while control is active.
            color = (100, 240, 100) if i == 0 and active else (230, 230, 230)
            # Render bounded-length antialiased text into the header.
            cv2.putText(canvas, line[:100], (12, 25 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        # Present the composed image in the native window.
        cv2.imshow(WINDOW, canvas)
        # Process window events and read an optional key press.
        key = cv2.waitKey(1) & 0xFF
        # Treat Q, q, and Escape as exit requests.
        if key in (ord("q"), ord("Q"), 27):
            # Tell the main loop to stop processing frames.
            return False
        # Handle failures from the operations in this block.
        try:
            # Continue only while the preview window remains visible.
            return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
        # Handle Qt destroying the window before the visibility query completes.
        except cv2.error:
            # Qt can destroy its receiver as soon as the close button is clicked.
            # Treat a closed native window as a clean exit request.
            return False

    # Release OpenCV's preview windows during cleanup.
    def close(self) -> None:
        # Destroy any windows created by this process.
        cv2.destroyAllWindows()
