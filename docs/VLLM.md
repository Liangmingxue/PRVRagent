# Local vLLM backend

PRVR-Agent can use any OpenAI-compatible chat-completions server. For the local Qwen3-VL deployment used in this project, keep the service on loopback unless remote access is intentionally configured.

Example same-machine server command:

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

PRVR-Agent does not require tool calling for the current CQHG planner, prospective world planner, or world-observer path, so `--tool-call-parser hermes` is optional. If the server must be reachable from other machines, protect it at the network/reverse-proxy layer rather than exposing the unauthenticated development configuration directly.

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

Generate a CQHG:

```bash
prvr-agent plan-llm "a man washes his hands and then opens the refrigerator"
```

Generate three CQHG-anchored prospective event worlds:

```bash
prvr-agent imagine-llm "a man washes his hands and then opens the refrigerator" --num-worlds 3
```

The model id returned by `/v1/models` must match `PRVR_LLM_MODEL`. If vLLM reports a different id, use that exact id in `PRVR_LLM_MODEL` or start vLLM with `--served-model-name <name>` and use the same name in PRVR-Agent.
