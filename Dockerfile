# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm
ARG DATASET=cifar10

FROM ${PYTHON_IMAGE} AS dataset-builder

ARG DATASET
COPY docker/datasets/${DATASET}/prepare.py /usr/local/bin/prepare-dataset
RUN python /usr/local/bin/prepare-dataset /dataset


FROM ${PYTHON_IMAGE} AS agent-base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ODBENCH_DATA_ROOT=/opt/odbench/data \
    PYTHONPATH=/opt/odbench/python

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        file \
        git \
        jq \
        tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0+cpu torchvision==0.28.0+cpu

COPY docker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --index-url https://pypi.org/simple \
        --requirement /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python -m pip check

COPY docker/odbench /opt/odbench/python/odbench

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /workspace agent \
    && mkdir -p /workspace \
    && chown agent:agent /workspace \
    && chmod -R a-w /opt/odbench \
    && python -c "import onnx, onnxruntime, torch, torchvision"

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]


FROM agent-base AS dataset-runtime

ARG DATASET
ENV ODBENCH_DATASET=${DATASET}

COPY --from=dataset-builder /dataset /opt/odbench/data/${DATASET}
COPY docker/datasets/${DATASET}/runtime.py /opt/odbench/python/odbench_dataset.py

RUN python -m odbench.validate_dataset
