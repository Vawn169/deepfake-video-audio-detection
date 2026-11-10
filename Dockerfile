FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES="" \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-api.txt .
RUN pip install torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-api.txt

COPY src/ src/
COPY models/ models/
COPY checkpoints_aug_s44/best_effnet_b0.pt checkpoints_aug_s44/best_effnet_b0.pt
COPY checkpoints_audio_v2b_s43/best_mlp.pt checkpoints_audio_v2b_s43/best_mlp.pt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "--app-dir", "src", "api:app", "--host", "0.0.0.0", "--port", "8000"]
