FROM python:3.12-slim

ARG NUMPY_VERSION=2.3.2

RUN python -m pip install --no-cache-dir "numpy==${NUMPY_VERSION}" \
    && python -I -c "import numpy; print(numpy.__version__)"
