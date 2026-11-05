import hashlib
from io import BytesIO
import zipfile

import pytest

from scripts import download_model


def bundle() -> bytes:
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("hand_landmarker.task", b"test hand model")
        archive.writestr("hand_gesture_recognizer.task", b"test gesture model")
    return stream.getvalue()


def test_download_verifies_hash_then_replaces_atomically(monkeypatch, tmp_path):
    content = bundle()
    expected = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(download_model, "MODEL_SHA256", expected)
    monkeypatch.setattr(download_model.urllib.request, "urlopen", lambda *a, **k: BytesIO(content))
    target = tmp_path / "models" / "model.task"
    assert download_model.download(target) == expected
    assert target.read_bytes() == content
    assert not list(target.parent.glob("*.part"))


@pytest.mark.parametrize("content", [b"not a model", bundle()])
def test_corrupt_or_wrong_hash_does_not_destroy_existing_model(monkeypatch, tmp_path, content):
    monkeypatch.setattr(download_model.urllib.request, "urlopen", lambda *a, **k: BytesIO(content))
    target = tmp_path / "model.task"
    target.write_bytes(b"existing model")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        download_model.download(target)
    assert target.read_bytes() == b"existing model"
    assert not list(tmp_path.glob("*.part"))
