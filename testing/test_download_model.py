# Calculate SHA-256 checksums for model integrity checks.
import hashlib
# Use an in-memory byte stream instead of a physical file.
from io import BytesIO
# Read and validate the model bundle as a ZIP archive.
import zipfile

# Define test fixtures, parameterized cases, and expected failures.
import pytest

# Access the model-download helper for bundle validation tests.
from scripts import download_model


# Build an in-memory archive with the required member names, not a real model.
def bundle() -> bytes:
    # Collect ZIP bytes without using the network or a real model asset.
    stream = BytesIO()
    # Write a ZIP container into the memory buffer.
    with zipfile.ZipFile(stream, "w") as archive:
        # Include a fake hand-landmark sub-bundle with the expected name.
        archive.writestr("hand_landmarker.task", b"test hand model")
        # Include a fake gesture-recognition sub-bundle with the expected name.
        archive.writestr("hand_gesture_recognizer.task", b"test gesture model")
    # Return the complete bytes after closing the ZIP writer.
    return stream.getvalue()


# Test: download verifies hash then replaces atomically.
def test_download_verifies_hash_then_replaces_atomically(monkeypatch, tmp_path):
    # Prepare the structurally valid fake download payload.
    content = bundle()
    # Calculate the expected checksum for this test payload only.
    expected = hashlib.sha256(content).hexdigest()
    # Replace the production checksum so this fake bundle may pass validation.
    monkeypatch.setattr(download_model, "MODEL_SHA256", expected)
    # Serve the fake bytes in place of an HTTPS response.
    monkeypatch.setattr(download_model.urllib.request, "urlopen", lambda *a, **k: BytesIO(content))
    # Choose a temporary nested path to also exercise directory creation.
    target = tmp_path / "models" / "model.task"
    # Require the downloader to return the verified test checksum.
    assert download_model.download(target) == expected
    # Confirm the final model file contains exactly the downloaded bytes.
    assert target.read_bytes() == content
    # No partial download file should remain after success.
    assert not list(target.parent.glob("*.part"))


# Cover both an invalid archive and a valid archive with the wrong checksum.
@pytest.mark.parametrize("content", [b"not a model", bundle()])
# Test: corrupt or wrong hash does not destroy existing model.
def test_corrupt_or_wrong_hash_does_not_destroy_existing_model(monkeypatch, tmp_path, content):
    # Return the selected failure payload without making a network request.
    monkeypatch.setattr(download_model.urllib.request, "urlopen", lambda *a, **k: BytesIO(content))
    # Use a temporary destination that already contains a model.
    target = tmp_path / "model.task"
    # Create existing content that a failed download must preserve.
    target.write_bytes(b"existing model")
    # Accept either archive parsing failure or checksum validation failure.
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        # Attempt the invalid replacement through the actual downloader.
        download_model.download(target)
    # Confirm failure did not destroy the existing model.
    assert target.read_bytes() == b"existing model"
    # Require temporary files to be cleaned up after failure as well.
    assert not list(tmp_path.glob("*.part"))
