ARG BASE_IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG RAY_VERSION=2.55.1
ARG RUNTIME_LAYER_REVISION=2

# The publisher's dedicated GLM image deliberately omits the optional Ray
# runtime. RayExecutorV2 needs the default runtime but not Ray's cgraph extra;
# cgraph would incorrectly install cupy-cuda12x alongside the image's CUDA 13
# CuPy package. Ray 2.54+ also prevents stale actor cleanup from aborting the
# interpreter and hiding the actual EngineCore initialization error.
RUN python3 -m pip install --no-cache-dir "ray[default]==${RAY_VERSION}" \
    && EXPECTED_RAY_VERSION="${RAY_VERSION}" python3 -c \
       'import importlib.metadata as m; import os; import ray; import vllm; from vllm import envs; names={d.metadata.get("Name", "").lower().replace("_", "-") for d in m.distributions()}; assert ray.__version__ == os.environ["EXPECTED_RAY_VERSION"]; assert "cupy-cuda12x" not in names; assert hasattr(envs, "VLLM_USE_RAY_V2_EXECUTOR_BACKEND"); print("vllm", m.version("vllm"), "ray", ray.__version__)'

LABEL org.ml-compute.runtime="glm53-flash-ray" \
      org.ml-compute.base-image="${BASE_IMAGE}" \
      org.ml-compute.ray.version="${RAY_VERSION}" \
      org.ml-compute.runtime.revision="${RUNTIME_LAYER_REVISION}"
