"""Immich machine-learning HTTP adapter for Sentence Transformers."""

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from PIL import Image
from starlette.datastructures import UploadFile

from config import Config
from encoder import preview
from session import ModelSession


async def expire_when_idle(model):
    while model.idle_seconds:
        await asyncio.sleep(1)
        await asyncio.to_thread(model.expire)


@asynccontextmanager
async def lifespan(app):
    config = app.state.config = Config()
    app.state.model = ModelSession(config)
    timeout = httpx.Timeout(config.upstream_timeout, connect=config.connect_timeout)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        app.state.client = client
        expiry = asyncio.create_task(expire_when_idle(app.state.model))
        try:
            yield
        finally:
            expiry.cancel()
            with suppress(asyncio.CancelledError):
                await expiry
            await asyncio.to_thread(app.state.model.close)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/ping")
async def ping(request: Request):
    config = request.app.state.config
    if config.upstream:
        try:
            upstream = await request.app.state.client.get(
                f"{config.upstream}/ping", timeout=config.health_timeout
            )
            upstream.raise_for_status()
        except httpx.HTTPError as error:
            raise HTTPException(503, "Immich machine learning is unavailable") from error
    return Response("pong", media_type="text/plain")


@app.post("/predict")
async def predict(request: Request):
    config = request.app.state.config
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > config.max_body:
            raise HTTPException(413, "Request body is too large")
        body.extend(chunk)

    raw_body = bytes(body)
    del body

    async def replay():
        return {"type": "http.request", "body": raw_body, "more_body": False}

    async with Request(request.scope, receive=replay).form(max_files=2, max_fields=8) as form:
        try:
            entries = json.loads(form.get("entries", ""))
        except (TypeError, ValueError) as error:
            raise HTTPException(422, "Invalid Immich machine learning task") from error
        if not isinstance(entries, dict) or not entries:
            raise HTTPException(422, "Missing Immich machine learning task")
        if "clip" not in entries:
            if not config.upstream:
                raise HTTPException(503, "Set IMMICH_ML_URL to forward other Immich tasks")
            try:
                response = await request.app.state.client.post(
                    f"{config.upstream}/predict",
                    content=raw_body,
                    headers={"Content-Type": request.headers["content-type"]},
                )
            except httpx.HTTPError as error:
                raise HTTPException(502, "Immich machine learning is unavailable") from error
            return Response(
                response.content,
                response.status_code,
                media_type=response.headers.get("Content-Type"),
            )

        clip = entries["clip"]
        if len(entries) != 1 or not isinstance(clip, dict) or len(clip) != 1:
            raise HTTPException(422, "Expected one Smart Search operation")
        mode, settings = next(iter(clip.items()))
        if not isinstance(settings, dict) or settings.get("modelName") != config.model_name:
            raise HTTPException(409, "Unexpected Smart Search model name")
        if mode == "textual":
            item = form.get("text")
            if not isinstance(item, str) or not item.strip():
                raise HTTPException(422, "Missing search text")
        elif mode == "visual":
            image = form.get("image")
            if not isinstance(image, UploadFile):
                raise HTTPException(422, "Missing preview image")
            try:
                item = await asyncio.to_thread(preview, await image.read(), config)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                raise HTTPException(422, "Invalid preview image") from error
        else:
            raise HTTPException(422, "Unsupported Smart Search operation")

        timeout = config.inference_timeout
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            async with asyncio.timeout(timeout):
                vector = await asyncio.to_thread(
                    request.app.state.model.encode, item, image=mode == "visual", deadline=deadline
                )
        except TimeoutError as error:
            raise HTTPException(504, "Smart Search inference timed out") from error
        return {"clip": json.dumps(vector.tolist(), separators=(",", ":"))}
