# RunPod Serverless GPU Video Renderer (FFmpeg + NVENC + libass + Python)
# Base com suporte CUDA/NVENC para renderização ultra-rápida em GPU (RTX 3090 / 4090 / A40 / L40S)
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Instalar FFmpeg (com libass para legendas estilizadas .ASS/.SRT), Python, fontes e utilitários
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libass9 \
    fontconfig \
    fonts-dejavu-core \
    fonts-liberation \
    fonts-noto-core \
    curl \
    ca-certificates \
    && fc-cache -f -v \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalar dependências Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 3. Copiar o handler Serverless
COPY handler.py .

# 4. Iniciar worker RunPod Serverless
CMD ["python3", "-u", "handler.py"]
