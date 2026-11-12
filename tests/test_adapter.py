import io
import json
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as adapter
from session import ModelSession


class FakeEncoder:
    device = "cpu"

    def __init__(self, config):
        self.dimension = config.dimension

    def encode(self, inputs, *, image=False):
        if image != isinstance(inputs[0], Image.Image):
            raise AssertionError("Unexpected input type")
        vector = np.zeros((1, self.dimension), dtype=np.float32)
        vector[0, 0] = 1
        return vector


class AdapterTests(unittest.TestCase):
    def test_text_image_and_idle(self):
        with patch("session.Encoder", side_effect=FakeEncoder) as load:
            with TestClient(adapter.app) as client:
                self.assertEqual(client.get("/ping").text, "pong")
                load.assert_not_called()
                text = {"text": (None, "猫咪")}
                self._predict(client, "textual", text)
                preview = io.BytesIO()
                Image.new("RGB", (48, 48)).save(preview, "JPEG")
                self._predict(client, "visual", {"image": ("preview.jpg", preview.getvalue())})
                load.assert_called_once()

                model = adapter.app.state.model
                self.assertIsInstance(model, ModelSession)
                model._last_used -= model.idle_seconds
                model.expire()
                self.assertEqual(client.get("/ping").text, "pong")
                load.assert_called_once()
                self._predict(client, "textual", text)
                self.assertEqual(load.call_count, 2)

    def test_non_clip_response_is_forwarded(self):
        with patch.dict(
            os.environ,
            {
                "IMMICH_ML_URL": "http://immich-ml:3003",
                "IMMICH_ML_TIMEOUT": "30",
                "IMMICH_ML_CONNECT_TIMEOUT": "4",
            },
        ):
            with TestClient(adapter.app) as client:
                self.assertEqual(adapter.app.state.client.timeout.read, 30)
                self.assertEqual(adapter.app.state.client.timeout.connect, 4)
                reply = httpx.Response(422, json={"detail": "example upstream response"})
                forward = AsyncMock(return_value=reply)
                with patch.object(adapter.app.state.client, "post", forward):
                    response = client.post(
                        "/predict",
                        files={"entries": (None, json.dumps({"ocr": {"detection": {}}}))},
                    )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json(), reply.json())
                self.assertEqual(forward.await_args.args[0], "http://immich-ml:3003/predict")

    def test_model_and_device_from_environment(self):
        overrides = {
            "QWEN_ST_DEVICE": "cpu",
            "QWEN_ST_MODEL": "/cache/local-model",
            "QWEN_ST_MODEL_NAME": "test-model",
            "QWEN_ST_DIMENSION": "32",
            "QWEN_ST_IDLE_SECONDS": "0",
        }
        with patch.dict(os.environ, overrides), patch("session.Encoder", side_effect=FakeEncoder):
            with TestClient(adapter.app) as client:
                config = adapter.app.state.config
                self.assertEqual(config.device, "cpu")
                self.assertEqual(config.dtype, "float32")
                self.assertEqual(config.model, "/cache/local-model")
                self.assertIsNone(config.revision)
                self.assertEqual(config.idle_seconds, 0)
                self._predict(client, "textual", {"text": (None, "test")})

    def test_inference_deadline_returns_504(self):
        class SlowEncoder(FakeEncoder):
            def encode(self, inputs, *, image=False):
                time.sleep(0.15)
                return super().encode(inputs, image=image)

        with (
            patch.dict(os.environ, {"QWEN_ST_INFERENCE_TIMEOUT": "0.03"}),
            patch("session.Encoder", side_effect=SlowEncoder),
            TestClient(adapter.app) as client,
        ):
            entries = {"clip": {"textual": {"modelName": adapter.app.state.config.model_name}}}
            response = client.post(
                "/predict",
                files={"entries": (None, json.dumps(entries)), "text": (None, "test")},
            )
            self.assertEqual(response.status_code, 504, response.text)

    def _predict(self, client, mode, files):
        config = adapter.app.state.config
        entries = {"clip": {mode: {"modelName": config.model_name}}}
        response = client.post(
            "/predict",
            files={"entries": (None, json.dumps(entries)), **files},
        )
        self.assertEqual(response.status_code, 200, response.text)
        vector = json.loads(response.json()["clip"])
        self.assertEqual(len(vector), config.dimension)
        self.assertEqual(np.linalg.norm(vector), 1)
