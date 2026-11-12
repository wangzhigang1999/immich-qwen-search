"""Load Qwen on demand and release it after the last inference finishes."""

import gc
import logging
import os
import threading
import time

from encoder import Encoder


LOGGER = logging.getLogger("uvicorn.error")


class ModelSession:
    def __init__(self):
        self.idle_seconds = int(os.environ.get("QWEN_ST_IDLE_SECONDS", "600"))
        if self.idle_seconds <= 0:
            raise ValueError("QWEN_ST_IDLE_SECONDS must be positive")
        self._lock = threading.Lock()
        self._encoder = None
        self._last_used = 0.0

    def encode(self, item, *, image=False):
        with self._lock:
            if self._encoder is None:
                started = time.monotonic()
                self._encoder = Encoder()
                LOGGER.info("Qwen model loaded in %.2f seconds", time.monotonic() - started)
            try:
                return self._encoder.encode([item], image=image)[0]
            finally:
                self._last_used = time.monotonic()

    def expire(self):
        if not self._lock.acquire(blocking=False):
            return
        try:
            if time.monotonic() - self._last_used >= self.idle_seconds:
                self._unload()
        finally:
            self._lock.release()

    def close(self):
        with self._lock:
            self._unload()

    def _unload(self):
        if self._encoder is None:
            return
        device = self._encoder.device
        self._encoder = None
        gc.collect()
        if device.startswith("cuda"):
            import torch

            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        LOGGER.info("Qwen model unloaded")
