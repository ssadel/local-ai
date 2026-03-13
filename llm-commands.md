# Local LLM Commands

## Setup (one time)
mkdir ~/Developer/local-ai && cd ~/Developer/local-ai
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm

## Activate venv (every new terminal session)
cd ~/Developer/local-ai && source .venv/bin/activate

## Models

### Qwen3-Coder-30B — fast coding, lightweight (17.2GB)
mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --port 8765

### Qwen3-Coder-Next — fast coding, heavier (~46GB, tight on 64GB)
mlx_lm.server --model mlx-community/Qwen3-Coder-Next-4bit --port 8765

### DeepSeek-R1-32B — reasoning model, slower but smarter (18.4GB)
mlx_lm.server --model mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit --port 8765

## Pre-download a model without running it
# Just start the server — it auto-downloads on first run then Ctrl+C.
# Or:
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit')"

## If downloads are slow
pip uninstall hf-xet -y
HF_HUB_ENABLE_HF_TRANSFER=1 mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --port 8765

## HuggingFace login (for faster downloads)
python -c "from huggingface_hub import login; login(token='hf_your_token_here')"

## API endpoint (point Claude Code here)
# http://localhost:8765/v1

## Test a model
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", "max_tokens": 4096, "stream": true, "messages": [{"role": "user", "content": "hello"}]}'

## Kill the server
# Ctrl+C in the terminal running it

## Check memory pressure
memory_pressure

## List cached models
ls ~/.cache/huggingface/hub/

## Delete a cached model
rm -rf ~/.cache/huggingface/hub/models--mlx-community--MODEL-NAME

## Models are stored in ~/.cache/huggingface/hub/ (separate from venv)
## Deleting venv does NOT delete models