# Local vLLM backend

PRVR-Agent uses an OpenAI-compatible local Qwen3-VL server for three structured tasks:

1. CQHG generation;
2. prospective event-world generation;
3. sparse global candidate observation and world-belief revision evidence.

Example deployment:

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
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9
```

Tool calling is not required by the current repository.

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

Then:

```bash
prvr-agent check-llm
prvr-agent plan-llm "a man washes his hands and then opens the refrigerator"
prvr-agent imagine-llm "a man puts a cake into the oven" --worlds 3
```

The model id returned by `/v1/models` must match `PRVR_LLM_MODEL`. If vLLM reports a different id, use that exact id or launch vLLM with `--served-model-name <name>` and configure the same name in PRVR-Agent.
