"""Runtime options for the Immich adapter."""

import math
import os
import re

MODEL = "Qwen/Qwen3-VL-Embedding-2B"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
MODEL_NAME = "sentence-qwen3-vl-embedding-2b/ViT-B-32__openai"


def number(name, default, convert=int, minimum=1):
    try:
        value = convert(os.environ.get(name, default))
        if math.isfinite(value) and value >= minimum:
            return value
    except (ValueError, OverflowError):
        pass
    raise ValueError(f"{name} must be a finite {convert.__name__} >= {minimum}")


class Config:
    def __init__(self):
        env = os.environ
        self.device = env.get("QWEN_ST_DEVICE", "cuda")
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("QWEN_ST_DEVICE must be cpu, cuda, or cuda:N")
        self.dtype = env.get("QWEN_ST_DTYPE", "auto")
        if self.dtype not in ("auto", "float32", "float16", "bfloat16"):
            raise ValueError("QWEN_ST_DTYPE must be auto, float32, float16, or bfloat16")
        if self.dtype == "auto":
            self.dtype = "float32" if self.device == "cpu" else "bfloat16"
        self.attention = env.get("QWEN_ST_ATTENTION", "sdpa")
        if self.attention not in ("sdpa", "eager"):
            raise ValueError("QWEN_ST_ATTENTION must be sdpa or eager")
        self.model = env.get("QWEN_ST_MODEL", MODEL)
        self.revision = env.get("QWEN_ST_REVISION") or (REVISION if self.model == MODEL else None)
        self.model_name = env.get("QWEN_ST_MODEL_NAME", MODEL_NAME)
        if not self.model.strip() or not self.model_name.strip():
            raise ValueError("QWEN_ST_MODEL and QWEN_ST_MODEL_NAME must be nonempty")
        self.dimension = number("QWEN_ST_DIMENSION", 512)
        self.query_prompt = env.get(
            "QWEN_ST_QUERY_PROMPT", "Retrieve images that match the given text."
        )
        self.image_prompt = env.get("QWEN_ST_IMAGE_PROMPT", "Represent the user's input.")
        self.image_max_edge = number("QWEN_ST_IMAGE_MAX_EDGE", 768)
        self.image_max_pixels = number("QWEN_ST_IMAGE_MAX_PIXELS", 80_000_000)
        self.max_body = number("QWEN_ST_MAX_BODY_MIB", 64) * 1024 * 1024
        self.idle_seconds = number("QWEN_ST_IDLE_SECONDS", 600, minimum=0)
        self.inference_timeout = number("QWEN_ST_INFERENCE_TIMEOUT", 0, float, minimum=0) or None
        self.upstream = env.get("IMMICH_ML_URL", "").rstrip("/")
        self.upstream_timeout = number("IMMICH_ML_TIMEOUT", 120, float, minimum=0.001)
        self.connect_timeout = number("IMMICH_ML_CONNECT_TIMEOUT", 3, float, minimum=0.001)
        self.health_timeout = number("IMMICH_ML_HEALTH_TIMEOUT", 2, float, minimum=0.001)
