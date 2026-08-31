# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.11-slim-bookworm
ARG DATASET=cifar10

FROM ${PYTHON_IMAGE} AS dataset-builder

ARG DATASET
COPY docker/datasets/${DATASET}/prepare.py /usr/local/bin/prepare-dataset
RUN --mount=type=bind,from=odbench_dataset_source,source=.,target=/source,ro \
    python /usr/local/bin/prepare-dataset /dataset /source


FROM ${PYTHON_IMAGE} AS agent-base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ODBENCH_DATA_ROOT=/opt/odbench/data \
    ODBENCH_PRETRAINED_ROOT=/opt/odbench/pretrained \
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

COPY docker/pretrained/manifest.json docker/pretrained/fetch.py /tmp/odbench-pretrained/
RUN python /tmp/odbench-pretrained/fetch.py \
        /tmp/odbench-pretrained/manifest.json \
        /opt/odbench/pretrained \
    && rm -rf /tmp/odbench-pretrained

COPY docker/odbench /opt/odbench/python/odbench
COPY docker/train/odbench_train /opt/odbench/python/odbench_train

RUN groupadd --gid 10001 agent \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /workspace agent \
    && mkdir -p /workspace \
    && chown agent:agent /workspace \
    && chmod -R a-w /opt/odbench \
    && python -c "import albumentations, odbench_train, onnx, onnxruntime, timm, torch, torchvision"

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
