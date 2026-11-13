"""Download Google's fixed version 1 model atomically. No ML training needed."""

# Parse command-line flags and display usage errors.
import argparse
# Calculate SHA-256 checksums for model integrity checks.
import hashlib
# Handle filesystem paths without platform-specific separators.
from pathlib import Path
# Import tempfile for use in this module.
import tempfile
# Import urllib.error for use in this module.
import urllib.error
# Download the official model over HTTPS.
import urllib.request
# Read and validate the model bundle as a ZIP archive.
import zipfile

# Use a fixed official model version instead of a moving latest URL.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
# SHA256 of the official version 1 asset, verified during development.
# Pin the expected SHA256 digest to detect a changed or corrupt download.
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
# Place the model under the project regardless of the shell's directory.
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "gesture_recognizer.task"


# Download, verify, and atomically replace the model; return its checksum.
def download(destination: Path) -> str:
    # Create the destination directory if it does not yet exist.
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Track whether a temporary file needs cleanup.
    temporary = None
    # Handle failures from the operations in this block.
    try:
        # Open the HTTPS response and a temporary output file in one resource scope.
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, tempfile.NamedTemporaryFile(
            # Keep the temporary file on the destination filesystem for atomic replacement.
            dir=destination.parent, suffix=".part", delete=False
        # Use the temporary file handle while streaming the response.
        ) as output:
            # Remember its path for validation and cleanup.
            temporary = Path(output.name)
            # Read up to one MiB per iteration instead of buffering the whole response.
            while chunk := response.read(1024 * 1024):
                # Append each received chunk to the temporary file.
                output.write(chunk)
        # Open the downloaded task bundle as a ZIP archive.
        with zipfile.ZipFile(temporary) as model:
            # Require the expected hand-landmark and gesture-recognition sub-bundles.
            if not {"hand_landmarker.task", "hand_gesture_recognizer.task"} <= set(model.namelist()):
                # Reject unrelated downloads before replacing the existing model.
                raise ValueError("Download is not the official Gesture Recognizer task bundle")
            # Check the archive members for CRC corruption.
            if model.testzip() is not None:
                # Reject a damaged ZIP bundle.
                raise ValueError("Downloaded model is corrupt")
        # Hash the complete downloaded bytes for an exact integrity check.
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        # Compare the asset with the pinned official-model digest.
        if digest != MODEL_SHA256:
            # Leave the previous model intact when verification fails.
            raise ValueError("Model SHA256 mismatch; existing model was not replaced")
        # Replace the destination only after all validation succeeded.
        temporary.replace(destination)
        # Return the verified digest for display or testing.
        return digest
    # Run cleanup whether the preceding operations succeeded or failed.
    finally:
        # Remove a partial file if one was created.
        if temporary is not None:
            # Ignore absence after a successful rename or earlier cleanup.
            temporary.unlink(missing_ok=True)


# Expose model downloading as a small standalone CLI.
def main() -> int:
    # Use the module description as command-line help text.
    parser = argparse.ArgumentParser(description=__doc__)
    # Allow callers to choose a destination other than the project default.
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    # Read the downloader's options from the command line.
    args = parser.parse_args()
    # Handle failures from the operations in this block.
    try:
        # Download and verify the requested model file.
        digest = download(args.output)
    # Convert network, filesystem, validation, and archive errors into a clear failure.
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        # Report the error without pretending the download completed.
        print(f"Model download failed: {exc}")
        # Report failure with a nonzero process exit status.
        return 1
    # Print the final path and verified checksum after successful replacement.
    print(f"Saved {args.output.resolve()}\nSHA256 {digest}")
    # Report successful completion to the calling process.
    return 0


# Run the downloader only when invoked directly.
if __name__ == "__main__":
    # Return its success or failure code to the shell.
    raise SystemExit(main())
