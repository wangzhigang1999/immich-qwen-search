FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

ENV HF_HOME=/cache \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /cache \
    && chown 10001:10001 /cache

COPY app.py encoder.py session.py ./

USER 10001:10001
EXPOSE 3003

HEALTHCHECK --interval=30s --timeout=5s \
    CMD ["python", "-c", "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:3003/ping', timeout=3).read() == b'pong'"]

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3003", "--no-access-log"]
