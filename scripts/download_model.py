"""Download Google's fixed version 1 model atomically. No ML training needed."""

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.error
import urllib.request
import zipfile

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
# SHA256 of the official version 1 asset, verified during development.
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "models" / "gesture_recognizer.task"


def download(destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".part", delete=False
        ) as output:
            temporary = Path(output.name)
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        with zipfile.ZipFile(temporary) as model:
            if not {"hand_landmarker.task", "hand_gesture_recognizer.task"} <= set(model.namelist()):
                raise ValueError("Download is not the official Gesture Recognizer task bundle")
            if model.testzip() is not None:
                raise ValueError("Downloaded model is corrupt")
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError("Model SHA256 mismatch; existing model was not replaced")
        temporary.replace(destination)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    try:
        digest = download(args.output)
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as exc:
        print(f"Model download failed: {exc}")
        return 1
    print(f"Saved {args.output.resolve()}\nSHA256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
