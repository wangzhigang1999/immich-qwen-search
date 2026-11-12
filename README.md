# Immich Qwen Smart Search

Local text-to-image search for Immich using Qwen3-VL-Embedding-2B and Sentence Transformers.

## Setup

The image targets Linux x86-64 with an NVIDIA GPU. Use NVIDIA Container Toolkit
on Linux or Docker Desktop with WSL 2 GPU support on Windows.

Download [`compose.qwen.yaml`](https://github.com/wangzhigang1999/immich-qwen-search/releases/latest)
and put it next to your existing Immich Compose file.

```sh
docker compose -f docker-compose.yml -f compose.qwen.yaml up -d immich-qwen-search
```

In Immich's Machine Learning settings, set the URL to `http://immich-qwen-search:3003`
and the Smart Search model to `sentence-qwen3-vl-embedding-2b/ViT-B-32__openai`.
Run Smart Search for all assets when switching from a different model. Other ML
tasks are forwarded to the original `immich-machine-learning` service.

The first request downloads the weights. The model unloads after 10 minutes
without inference and reloads on demand.

## Model cache

The default Docker volume survives container recreation. To use a host directory,
add `QWEN_MODEL_CACHE` to the `.env` next to your Compose files.

```dotenv
QWEN_MODEL_CACHE=/srv/immich/qwen-cache
# On Docker Desktop for Windows: QWEN_MODEL_CACHE=D:/models/huggingface
```

Point it at the Hugging Face cache root containing `hub/`, or use an empty
directory to populate it on first use. Give the container's UID/GID
`10001:10001` write access if it will download weights. With the full
checkpoint cached, set `HF_HUB_OFFLINE=1` to reuse it offline.
