# Immich Qwen Smart Search with Sentence Transformers

This adapter runs Qwen3-VL-Embedding-2B locally and exposes the
Immich machine-learning endpoints. The adapter returns 512-dimensional vectors
using the first 512 coordinates followed by L2 normalization. It uses the
official Hugging Face checkpoint pinned in `encoder.py` and the same text
and image instructions as the GGUF deployment.

## Docker

Download `compose.qwen.yaml` from the [release](https://github.com/wangzhigang1999/immich-qwen-search/releases/latest)
and put it next to your existing Immich Compose file. The image targets Linux
x86-64 with an NVIDIA GPU. Use NVIDIA Container Toolkit on Linux or the WSL 2
backend with GPU support on Docker Desktop for Windows.

```sh
docker compose -f docker-compose.yml -f compose.qwen.yaml up -d immich-qwen-search
```

In Immich's Machine Learning settings, use `http://immich-qwen-search:3003` as
the ML URL and `sentence-qwen3-vl-embedding-2b/ViT-B-32__openai` as the Smart
Search model. The adapter forwards other tasks to `immich-machine-learning`
on the same network. For a fresh index, run the Smart Search job for all
assets after changing models. The first request downloads the official
weights into the persistent cache volume.

For upgrades, download the next release's Compose file and repeat the same
command. Keep the same cache volume.

## Run locally on Windows

With Python 3.12, uv and an NVIDIA driver that supports CUDA 12.8, run these
commands from this directory.

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cu128 "torch==2.10.0" "torchvision==0.25.0" -r requirements.txt
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 17933
```

The first prediction downloads and loads the weights. CUDA uses BF16. Set
`QWEN_ST_DEVICE=cpu` to run with FP32 on the CPU. The model unloads after
600 seconds without inference by default. Set `QWEN_ST_IDLE_SECONDS` to
change the interval. The next prediction reloads the weights from disk.
Health probes leave the model unloaded. Run one Uvicorn worker per GPU.

The Windows launcher `run.ps1` reads optional environment overrides from a
local `deployment.psd1` PowerShell data file and writes logs under `logs`.

`GET /ping` checks whether the adapter has started. `POST /predict` accepts
Immich multipart requests. Set the Smart Search model name in a test request
to `sentence-qwen3-vl-embedding-2b/ViT-B-32__openai`. The suffix selects the
512-dimensional index in Immich 3.3.5. Set `QWEN_ST_MODEL_NAME` to an existing
Immich model name when intentionally reusing an index you have verified for
compatibility. For other machine-learning requests,
set `IMMICH_ML_URL` to the URL of the original ML service. Keep the prototype
on loopback for local testing or place it on a trusted private network for
integration testing.

For a Windows host with Docker Desktop, publish the built-in ML service on
`127.0.0.1:17934` and set `IMMICH_ML_URL=http://127.0.0.1:17934` for the
adapter. Verify `http://host.docker.internal:17933/ping` from inside the
Immich server container before using that URL for Machine Learning in
Immich. Keep one active URL for Smart Search and save the previous URL
before the cutover.

## Benchmark

Chinese text-to-image search on 512 Crossmodal-3600 images with an RTX 3060 Laptop GPU.

| Setup | Top-1 recall | Peak VRAM |
| --- | ---: | ---: |
| GGUF, Vulkan | 87.38% | 2.52 GiB |
| Sentence Transformers, CUDA | 86.79% | 4.40 GiB |

Sentence Transformers queries against the existing GGUF image vectors scored 87.18%.
