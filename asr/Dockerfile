# Build docker image for asr_workflow

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

ENV WS_SERVER_URL=ws://whisper_live:9090

# Install Python 3.13 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13-full python3.13-dev ffmpeg git git-lfs \
    curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.13 /usr/bin/python

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

RUN git lfs install

# Add user and group needed by Kubernetes
RUN groupadd -g 1100 appuser && \
    useradd -u 1100 -d /app -g appuser appuser

WORKDIR /app

#RUN chown appuser:appuser /app

COPY requirements.txt .
RUN python3.13 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Install whisqa
#RUN python3.13 -m pip install --no-deps --no-cache-dir git+https://github.com/leto19/WhiSQA.git@py_module
RUN git clone -b py_module --depth 1 https://github.com/leto19/WhiSQA.git /tmp/WhiSQA && \
    cd /tmp/WhiSQA && \
    git lfs pull && \
    python3.13 -m pip install --no-deps --no-cache-dir . && \
    rm -rf /tmp/WhiSQA

# Download nisqalib
# RUN ./download_nisqa

# Ensure that appuser owns everything.
RUN chown -R appuser:appuser /app

# Switch user
USER 1100

# Preload whisper model
RUN PYTHONPATH=. python3.13 -c "from asr_processor.whisperx_transcriber import WhisperXTranscriber; WhisperXTranscriber(use_gpu=False)"

# Set this to 1 so that at RUNTIME it never tries to check for updates
#ENV HF_HUB_OFFLINE=1

#EXPOSE 8000

# Run the app.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "60"]

