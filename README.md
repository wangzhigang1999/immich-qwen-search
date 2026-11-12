# Immich Qwen Smart Search

Local text-to-image search for Immich using Qwen3-VL-Embedding-2B and Sentence Transformers.

## Setup

The image targets Linux x86-64. For CUDA, use NVIDIA Container Toolkit on Linux
or Docker Desktop with WSL 2 GPU support on Windows.

Download [`compose.qwen.yaml`](https://github.com/wangzhigang1999/immich-qwen-search/releases/latest)
and put it next to your existing Immich Compose file.

```sh
docker compose -f docker-compose.yml -f compose.qwen.yaml up -d immich-qwen-search
```

On a CPU-only host, also download `compose.qwen.cpu.yaml` and append it as the
last Compose file.

```sh
docker compose -f docker-compose.yml -f compose.qwen.yaml -f compose.qwen.cpu.yaml up -d immich-qwen-search
```

In Immich's Machine Learning settings, set the URL to `http://immich-qwen-search:3003`
and the Smart Search model to `sentence-qwen3-vl-embedding-2b/ViT-B-32__openai`.
Run Smart Search for all assets when switching from a different model. Other ML
tasks are forwarded to the original `immich-machine-learning` service.

The first request downloads the weights if the cache is empty. The model unloads after 10 minutes
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

## Configuration

Copy the settings you need from [`qwen.env.example`](qwen.env.example) into your
existing Immich `.env`, then run the same Compose command to apply them. CUDA
uses BF16 and CPU uses FP32 by default. Set `QWEN_GPU_ID` to select the host GPU.

`QWEN_ST_MODEL` accepts a Hugging Face repository or a container path such as
`/cache/my-model`. Use a Sentence Transformers model that can encode both text
and images. Keep the model alias and dimension aligned with Immich. Changing
the checkpoint, prompts or dimension can require rebuilding the Smart Search index.

`QWEN_ST_INFERENCE_TIMEOUT` limits how long a request waits for inference,
including model loading. A timeout returns HTTP 504. Inference already running
can finish in the background. Zero leaves inference unlimited.
