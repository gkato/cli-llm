ARG BASE_IMAGE=vllm/vllm-openai:glm53-flash-arm64-cu130@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce
FROM ${BASE_IMAGE}

ARG BASE_IMAGE
ARG RAY_VERSION=2.55.1

# The publisher's dedicated GLM image deliberately omits the optional Ray
# runtime. Multi-node vLLM needs Ray, including the cgraph extra used by the
# current executor. Ray 2.54+ also prevents stale actor cleanup from aborting
# the interpreter and hiding the actual EngineCore initialization error.
RUN python3 -m pip install --no-cache-dir "ray[cgraph,default]==${RAY_VERSION}" \
    && EXPECTED_RAY_VERSION="${RAY_VERSION}" python3 -c \
       'import importlib.metadata as m; import os; import ray; import vllm; assert ray.__version__ == os.environ["EXPECTED_RAY_VERSION"]; print("vllm", m.version("vllm"), "ray", ray.__version__)'

LABEL org.ml-compute.runtime="glm53-flash-ray" \
      org.ml-compute.base-image="${BASE_IMAGE}" \
      org.ml-compute.ray.version="${RAY_VERSION}"
