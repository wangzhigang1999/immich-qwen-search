"""Load Qwen on demand and release it after the last inference finishes."""

import gc
import logging
import threading
import time

from config import Config
from encoder import Encoder

LOGGER = logging.getLogger("uvicorn.error")


class ModelSession:
    def __init__(self, config: Config):
        self.config = config
        self.idle_seconds = config.idle_seconds
        self._lock = threading.Lock()
        self._encoder = None
        self._last_used = 0.0

    def encode(self, item, *, image=False, deadline=None):
        acquired = (
            self._lock.acquire()
            if deadline is None
            else self._lock.acquire(
                timeout=min(threading.TIMEOUT_MAX, max(0, deadline - time.monotonic()))
            )
        )
        if not acquired:
            raise TimeoutError("Inference expired in the queue")
        try:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Inference expired in the queue")
            if self._encoder is None:
                started = time.monotonic()
                self._encoder = Encoder(self.config)
                LOGGER.info("Qwen model loaded in %.2f seconds", time.monotonic() - started)
            try:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Inference expired while loading the model")
                return self._encoder.encode([item], image=image)[0]
            finally:
                self._last_used = time.monotonic()
        finally:
            self._lock.release()

    def expire(self):
        if not self.idle_seconds or not self._lock.acquire(blocking=False):
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
