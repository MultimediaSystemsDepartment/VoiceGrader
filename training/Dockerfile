# GPU image -- used for all ML pipeline steps (pair generation, reward scoring,
# SFT, DPO, evaluation). Requires nvidia-container-toolkit on the host.
#
# Build: docker compose build gpu
# Run a pipeline step: docker compose run --rm generate

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# ---- System packages -------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        libsndfile1 \
        ffmpeg \
        git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- PyTorch with CUDA (large layer, cached separately) --------------------
RUN python3.12 -m pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

# ---- Remaining Python dependencies -----------------------------------------
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# ---- Source code -----------------------------------------------------------
COPY . .

CMD ["python3", "-c", "print('GPU image ready. Use: docker compose run --rm <service>')"]
