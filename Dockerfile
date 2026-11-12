FROM ghcr.io/astral-sh/uv:0.9.5@sha256:0f419824ea1810fe2a47af12aef8e8c39eabae8d19c728cd9612d18e17a98d17 AS uv
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=bind,from=uv,source=/uv,target=/usr/local/bin/uv \
    uv venv --no-cache --system-site-packages --python /usr/bin/python3 /opt/adapter \
    && /opt/adapter/bin/python -m pip install --no-cache-dir -r requirements.txt \
        "torch==2.10.0+cu128" "torchvision==0.25.0+cu128" \
    && mkdir -p /cache \
    && chown 10001:10001 /cache

ENV PATH=/opt/adapter/bin:$PATH \
    HF_HOME=/cache \
    HOME=/cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
COPY app.py encoder.py session.py ./

USER 10001:10001
EXPOSE 3003

HEALTHCHECK --interval=30s --timeout=5s \
    CMD ["python", "-c", "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:3003/ping', timeout=3).read() == b'pong'"]

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003", "--no-access-log"]
