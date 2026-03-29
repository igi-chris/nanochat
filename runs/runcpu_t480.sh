#!/bin/bash

# Modified for Thinkpad T480 (CPU-only, 32GB RAM)
# This is a conservative, slower run - expect 2-4 hours total

# Run as:
# bash runs/runcpu_t480.sh

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~100M characters (much faster than 2B)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=100000000
python -m scripts.tok_eval

# train a small 4 layer model
# Reduced depth, iterations, and batch sizes for CPU
python -m scripts.base_train \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=256 \
    --device-batch-size=8 \
    --total-batch-size=4096 \
    --eval-every=200 \
    --eval-tokens=262144 \
    --core-metric-every=-1 \
    --sample-every=200 \
    --num-iterations=2000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# SFT (reduced iterations)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft \
    --max-seq-len=256 \
    --device-batch-size=8 \
    --total-batch-size=4096 \
    --eval-every=200 \
    --eval-tokens=262144 \
    --num-iterations=500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
