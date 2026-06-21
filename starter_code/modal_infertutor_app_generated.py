"""
InferTutor Arena - Modal + vLLM server.

Students should usually not edit this file first. Start by changing
configuration from run_infertutor_experiment.py CLI flags.

This app runs Qwen/Qwen3-VL-4B-Instruct behind an OpenAI-compatible vLLM
HTTP server on Modal.
"""

import os
import subprocess

import modal


# vLLM 0.21.0 supports Qwen3-VL and the OpenAI-compatible multimodal API.
# The CUDA base image gives vLLM access to the GPU runtime it needs.
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "qwen-vl-utils==0.0.14")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("infertutor-FINAL-mixed")

# Persistent caches reduce repeated model download and compilation overhead.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)


# These constants are patched by run_infertutor_experiment.py before deploy.
MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
TENSOR_PARALLEL = 1
GPU_TYPE = "H100"
GPU_COUNT = 1
DTYPE = "bfloat16"
ENABLE_PREFIX_CACHING = True
ENABLE_CHUNKED_PREFILL = True
MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 4096
MAX_NUM_SEQS = 32
CONCURRENT_INPUTS = 64
MIN_CONTAINERS = 1
MAX_CONTAINERS = 1
FAST_BOOT = False
MM_MAX_PIXELS = 401408

MINUTES = 60
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{GPU_COUNT}",
    scaledown_window=10 * MINUTES,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    timeout=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_name("huggingface", required_keys=["HF_TOKEN"])],
)
@modal.concurrent(max_inputs=CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    """Start vLLM inside the Modal container."""

    # FastAPI >=0.116 introduced `_IncludedRouter`, which has no `.path`.
    # `prometheus-fastapi-instrumentator` (a vLLM dependency) crashes on
    # every request because its routing helper assumes `.path` exists.
    # Skip routers that don't have it.
    import pathlib

    routing_file = pathlib.Path(
        "/usr/local/lib/python3.12/site-packages/prometheus_fastapi_instrumentator/routing.py"
    )
    if routing_file.exists():
        src = routing_file.read_text()
        marker = 'if not hasattr(route, "path"):'
        if marker not in src:
            patched = src.replace(
                "for route in routes:",
                f'for route in routes:\n        {marker}\n            continue',
            )
            routing_file.write_text(patched)
            print("Patched prometheus_fastapi_instrumentator/routing.py", flush=True)

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL),
        "--dtype",
        DTYPE,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        "0.90",
        "--uvicorn-log-level=warning",
        "--limit-mm-per-prompt",
        '{"image": 1, "video": 0}',
        "--mm-processor-kwargs",
        f'{{"min_pixels": 784, "max_pixels": {MM_MAX_PIXELS}, "fps": 1}}',
    ]

    # Eager mode starts faster. Compiled mode can improve text-only throughput
    # after a longer warmup.
    if FAST_BOOT:
        cmd += ["--enforce-eager"]
    else:
        cmd += ["--no-enforce-eager"]

    if ENABLE_PREFIX_CACHING:
        cmd += ["--enable-prefix-caching"]

    if ENABLE_CHUNKED_PREFILL:
        cmd += ["--enable-chunked-prefill"]

    print("Starting vLLM:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)

