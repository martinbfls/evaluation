FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git\
    cmake\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app




RUN  pip install --no-cache-dir --upgrade pip setuptools wheel
RUN  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126


RUN pip install --no-cache-dir --no-build-isolation transformers evaluate pandas