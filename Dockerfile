FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# 切换 apt 源（Ubuntu18.04 国内源，加快速度，可选）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git wget curl vim python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# 升级 pip
RUN pip install --upgrade pip setuptools wheel

# 替换 pynvml 为 nvidia-ml-py（避免警告）
RUN pip install nvidia-ml-py

# 安装 vLLM（会自动处理 PyTorch/TorchVision 兼容性）
RUN pip install vllm

# 把项目代码拷贝进容器
WORKDIR /workspace
COPY . /workspace

# 安装项目依赖
RUN pip install -r requirements.txt \
    && pip install pybase64 \
    orjson \
    uvicorn \
    uvloop \
    fastapi \
    zmq \
    partial-json-parser \
    huggingface-hub \
    transformers \
    sentencepiece

ENTRYPOINT ["/bin/bash", "/workspace/vllm.sh"]