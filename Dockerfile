# VeriFace — GPU-enabled image for training and/or serving the Streamlit app.
#
# Base: CUDA 12.9.2 (verified to exist as an official `nvidia/cuda` tag at
# build-authoring time: 12.9.2-cudnn-runtime-ubuntu24.04). Ubuntu 24.04 ships
# Python 3.12 by default, matching the cp312 wheels pinned below.
#
# torch/torchvision are installed from the PyTorch cu129 wheel index rather
# than plain PyPI, because plain PyPI `torch` wheels are CPU-only.
# torch==2.13.0+cu129 / torchvision==0.28.0+cu129 were confirmed to exist as
# cp312 linux/x86_64 wheels at pin time (2026-08-16).
#
# GPU target: this environment has no GPU, so none of this has been
# validated on real hardware. It is written for an RTX Pro 6000
# (Blackwell, compute capability sm_120) class card — see HANDOFF.md
# "Environment Requirements" for the caveat: verify at build time against
# https://pytorch.org/get-started/locally/ that the pinned torch build
# still lists Blackwell/sm_120 support, since that can't be confirmed from
# here.

FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ffmpeg: required by src/data/augmentations.py (video re-encoding) and by
# opencv-python-headless's video decode backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

WORKDIR /workspace

COPY requirements.txt .

# Install the CUDA build of torch/torchvision explicitly FIRST (plain
# `pip install -r requirements.txt` alone would resolve torch/torchvision to
# CPU-only wheels from PyPI). The subsequent `-r requirements.txt` install
# sees torch==2.13.0 / torchvision==0.28.0 already satisfied by the
# +cu129 local versions installed here (PEP 440: a local version segment
# still satisfies an exact-version requirement) and leaves them alone.
RUN pip install --no-cache-dir \
        torch==2.13.0 torchvision==0.28.0 \
        --index-url https://download.pytorch.org/whl/cu129 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Sanity check torch can see the GPU at container start (does not fail the
# build if run without --gpus, since this is CMD not RUN).
CMD ["sh", "-c", "python -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'; streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501"]
