import io
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as adapter
from encoder import DIMENSION
from session import ModelSession


class FakeEncoder:
    device = "cpu"

    def encode(self, inputs, *, image=False):
        if image != isinstance(inputs[0], Image.Image):
            raise AssertionError("Unexpected input type")
        vector = np.zeros((1, DIMENSION), dtype=np.float32)
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
        with patch.object(adapter, "UPSTREAM", "http://immich-ml:3003"):
            with TestClient(adapter.app) as client:
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

    def _predict(self, client, mode, files):
        entries = {"clip": {mode: {"modelName": adapter.ACTIVE_MODEL}}}
        response = client.post(
            "/predict",
            files={"entries": (None, json.dumps(entries)), **files},
        )
        self.assertEqual(response.status_code, 200, response.text)
        vector = json.loads(response.json()["clip"])
        self.assertEqual(len(vector), DIMENSION)
        self.assertEqual(np.linalg.norm(vector), 1)
