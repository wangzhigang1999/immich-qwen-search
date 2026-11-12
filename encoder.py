"""Local Qwen text and image embeddings for Immich Smart Search."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from config import Config


def preview(data: bytes, config: Config) -> Image.Image:
    max_edge = config.image_max_edge
    with Image.open(BytesIO(data)) as original:
        if original.width * original.height > config.image_max_pixels:
            raise ValueError("Image exceeds the pixel limit")
        if original.format == "JPEG" and original.width * original.height > 36_000_000:
            original.draft("RGB", (max_edge, max_edge))
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        return image


class Encoder:
    def __init__(self, config: Config):
        import torch
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.device = config.device
        self.model = SentenceTransformer(
            config.model,
            revision=config.revision,
            device=self.device,
            model_kwargs={
                "dtype": getattr(torch, config.dtype),
                "attn_implementation": config.attention,
                "device_map": {"": self.device},
            },
        )

    def encode(self, inputs, *, image=False):
        dimension = self.config.dimension
        vectors = (
            self.model.encode(
                inputs,
                prompt=self.config.image_prompt if image else self.config.query_prompt,
                batch_size=len(inputs),
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            .float()
            .cpu()
            .numpy()[:, :dimension]
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if vectors.shape[1] != dimension or not np.isfinite(vectors).all() or np.any(norms < 1e-8):
            raise RuntimeError("Invalid Qwen embedding")
        return vectors / norms
