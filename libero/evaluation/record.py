"""Per-episode video and action-trace recording helpers."""

import json
from pathlib import Path

import numpy as np


def upright_rgb(obs, key):
    """Return a contiguous upright uint8 RGB camera observation."""
    image = np.asarray(obs[key])
    if image.dtype != np.uint8:
        image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[::-1])


class VideoRecorder:
    def __init__(self, path, enabled=False, fps=10):
        self.path, self.enabled, self.fps, self._writer = Path(path), enabled, fps, None

    def __enter__(self):
        if self.enabled:
            import imageio.v2 as imageio
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(str(self.path), fps=self.fps)
        return self

    def append(self, frame):
        if self._writer is not None:
            self._writer.append_data(frame)

    def __exit__(self, *args):
        if self._writer is not None:
            self._writer.close()


class ActionTraceRecorder:
    """Write one compact, crash-readable JSON object per executed action."""

    def __init__(self, path, enabled=False):
        self.path, self.enabled, self._stream = Path(path), enabled, None

    def __enter__(self):
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self.path.open("w", encoding="utf-8")
        return self

    def append(self, record):
        if self._stream is not None:
            self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._stream.flush()

    def __exit__(self, *args):
        if self._stream is not None:
            self._stream.close()
