# Local LLM Commands

## Setup (one time)
mkdir ~/Developer/local-ai && cd ~/Developer/local-ai
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm

## Activate venv (every new terminal session)
cd ~/Developer/local-ai && source .venv/bin/activate

## Start coding model (fast, non-reasoning)
mlx_lm.server --model mlx-community/Qwen3-Coder-Next-4bit --port 8080

## Start coding model — lighter, more room for Xcode
mlx_lm.server --model mlx-community/Qwen3-Coder-30B-A3B-4bit --port 8080

## Start reasoning model (slower, smarter)
mlx_lm.server --model mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit --port 8080

## Quick test a model from terminal
mlx_lm.generate --model mlx-community/Qwen3-Coder-30B-A3B-4bit --prompt "hello"

## Pre-download a model without running it
pip install huggingface_hub
huggingface-cli download mlx-community/Qwen3-Coder-Next-4bit
huggingface-cli download mlx-community/Qwen3-Coder-30B-A3B-4bit
huggingface-cli download mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit

## API endpoints (point Claude Code here)
# MLX server: http://localhost:8080/v1

## Kill the server
# Ctrl+C in the terminal running it

## Check memory pressure
memory_pressure

## List cached models
ls ~/.cache/huggingface/hub/models--mlx-community--*

## Delete a cached model
rm -rf ~/.cache/huggingface/hub/models--mlx-community--MODEL-NAME
