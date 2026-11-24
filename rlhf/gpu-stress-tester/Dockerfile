FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN python3.12 -m pip install --no-cache-dir \
        torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
    && python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY src ./src
RUN python3.12 -m pip install --no-cache-dir --no-deps .

EXPOSE 8080
CMD ["python3.12", "-m", "uvicorn", "gpu_stress_tester.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
