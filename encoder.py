"""Local Qwen text and image embeddings for Immich Smart Search."""

from io import BytesIO
import os

import numpy as np
from PIL import Image, ImageOps


MODEL = "Qwen/Qwen3-VL-Embedding-2B"
REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
MODEL_NAME = "sentence-qwen3-vl-embedding-2b/ViT-B-32__openai"
DIMENSION = 512
MAX_EDGE = 768
MAX_PIXELS = 80_000_000
IMAGE_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images that match the given text."


def preview(data: bytes) -> Image.Image:
    with Image.open(BytesIO(data)) as original:
        if original.width * original.height > MAX_PIXELS:
            raise ValueError("Image exceeds the pixel limit")
        if original.format == "JPEG" and original.width * original.height > 36_000_000:
            original.draft("RGB", (MAX_EDGE, MAX_EDGE))
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
        return image


class Encoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        import torch

        self.device = os.environ.get("QWEN_ST_DEVICE", "cuda")
        cuda = self.device.startswith("cuda")
        self.model = SentenceTransformer(
            MODEL,
            revision=REVISION,
            device=self.device,
            model_kwargs={
                "dtype": torch.bfloat16 if cuda else torch.float32,
                "attn_implementation": "sdpa",
                "device_map": {"": self.device},
            },
        )

    def encode(self, inputs, *, image=False):
        vectors = self.model.encode(
            inputs,
            prompt=IMAGE_INSTRUCTION if image else QUERY_INSTRUCTION,
            batch_size=len(inputs),
            convert_to_tensor=True,
            show_progress_bar=False,
        ).float().cpu().numpy()[:, :DIMENSION]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape[1] != DIMENSION or not np.isfinite(vectors).all() or np.any(norms < 1e-8):
            raise RuntimeError("Invalid Qwen embedding")
        return vectors / norms
