# Local vLLM backend

PRVR-Agent can use any OpenAI-compatible chat-completions server. For the local Qwen3-VL deployment used in this project, expose vLLM on port 8000 and point PRVR-Agent to `http://127.0.0.1:8000/v1`.

Example server command:

```bash
ENV_DIR=/home/omnisky/miniconda3/envs/chart-vllm

env -u PYTHONHOME -u PYTHONPATH \
  LD_LIBRARY_PATH="$ENV_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  CUDA_VISIBLE_DEVICES=1 \
  "$ENV_DIR/bin/vllm" serve \
  /home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8 \
  --port 8000 \
  --trust-remote-code \
  --tool-call-parser hermes \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9
```

PRVR-Agent does not require tool calling for the current planner/verifier path, so `--tool-call-parser hermes` is harmless but not required by this repository.

Configure the client:

```bash
export PRVR_LLM_BASE_URL=http://127.0.0.1:8000/v1
export PRVR_LLM_API_KEY=EMPTY
export PRVR_LLM_MODEL=/home/omnisky/xlm/newtask/charttrans-workspace/model/Qwen3-VL-8B-Instruct-FP8
```

Then test connectivity:

```bash
prvr-agent check-llm
```

The model id returned by `/v1/models` must match `PRVR_LLM_MODEL`. If vLLM reports a different id, use that exact id in `PRVR_LLM_MODEL` or start vLLM with `--served-model-name <name>` and use the same name in PRVR-Agent.
