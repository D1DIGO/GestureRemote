# Verification record

Development checks on **2026-11-05** in a Fedora Linux 44 x86_64 environment,
Python **3.14.8**, using a local `.venv` and the pinned requirements.

| Check | Result |
| --- | --- |
| `python -m pytest -q` | 117 passed; includes real model blank-frame CPU smoke test |
| `python -m pip check` | No broken requirements |
| `python main.py --check-config` | Configuration valid |
| `python main.py --help` | CLI starts and lists options |
| `python main.py --check-model` | Official model loads on CPU and completes blank-frame inference, no hand |
| `python main.py --dry-run --no-preview --duration 1` | Exits with code 1 and clear missing-camera message; no camera device available |
| Actual application loop | Tested with fake camera/model and mocked VLC, including commands, dry-run, failures, stale frames and cleanup |
| HTTP syntax | Checked against VideoLAN's VLC 3.x source and request documentation; URL encoding tested |
| Windows Python 3.11 dependency resolution | pip binary-only cross-platform dry-run resolved the pinned dependencies; this is not a Windows runtime test |
| Real webcam recognition, distance, dim light, preview | **Not tested** — no camera/display validation available |
| Real VLC playback and focus independence | **Not tested** — VLC not installed in this environment |
| Windows 10/11 and Ubuntu hardware | **Not tested** — provided setup instructions and CI matrix, not claimed results |

Official model: float16/version 1, downloaded from Google's MediaPipe model
bucket. SHA256:

```text
97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482
```

The downloader validates the bundle and checksum before replacing a file.
The model is present locally and excluded from Git; a fresh checkout must run
`python scripts/download_model.py`.

Core pinned versions: MediaPipe 0.10.32, OpenCV contrib 4.13.0.92, NumPy 2.4.2,
Requests 2.32.5, pytest 9.0.2. MediaPipe's transitive dependencies were resolved
by pip; requirements are not a complete cross-platform lockfile.

MediaPipe emits informational native messages about CPU XNNPACK, absent custom
classifiers and feedback tensors. No custom classifier is used or required.
The successful inference/test result, rather than the absence of native log
messages, determines the smoke-test outcome.

Run the [physical procedure](PHYSICAL_TESTING.md) before claiming the hardware
items in the MVP definition of done or any distance/lighting performance target.
