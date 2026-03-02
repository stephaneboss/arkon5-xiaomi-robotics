FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    TORCH_HOME=/torch_cache

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git wget curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt .
RUN pip install torch==2.8.0 torchvision==0.19.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
RUN pip install flash-attn==2.8.3 --no-build-isolation
RUN pip install -r requirements.txt

RUN git clone https://github.com/XiaomiRobotics/Xiaomi-Robotics-0.git /app/xiaomi_model \
    && pip install -e /app/xiaomi_model 2>/dev/null || true

COPY inference_server.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
