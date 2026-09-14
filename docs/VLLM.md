# Local vLLM backend

PRVR-Agent uses the OpenAI-compatible chat-completions API. For the local Qwen3-VL deployment used in this project, expose vLLM on port 8000 and point PRVR-Agent to `http://127.0.0.1:8000/v1`.

For a same-machine deployment, bind explicitly to loopback:

```bash
ENV_DIR=/home/omnisky/miniconda3/envs/chart-vllm

env -u PYTHONHOME -u PYTHONPATH \
  LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_VISIBLE_DEVICES=1 \
  "$ENV_DIR/bin/vllm" serve \
  /home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9
```

PRVR-Agent does not require tool calling for the current planner/verifier path, so `--tool-call-parser hermes` is optional.

Configure the client:

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
export PRVR_LLM_TIMEOUT=120
export PRVR_LLM_TEMPERATURE=0
export PRVR_LLM_MAX_TOKENS=2048
export PRVR_LLM_VALIDATION_RETRIES=2
export PRVR_LLM_HTTP_MAX_RETRIES=2
```

Then test connectivity:

```bash
prvr-agent check-llm
```

The model id returned by `/v1/models` must match `PRVR_LLM_MODEL`. If vLLM reports a different id, use that exact id in `PRVR_LLM_MODEL` or start vLLM with `--served-model-name <name>` and use the same name in PRVR-Agent.

## Security note

If the API must be reachable from another machine, use a firewall and/or reverse proxy to restrict the exposed network surface. Current vLLM documentation explicitly warns that `--api-key` authenticates only selected path prefixes and that other endpoints on the same server may remain unauthenticated. Do not treat `--api-key` as the sole perimeter control.

For reproducible experiments, keep the service on a trusted network, avoid development/debug endpoint modes, and record the exact vLLM version, model checkpoint, served model id, and launch arguments with each benchmark run.
