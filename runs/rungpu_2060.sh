#!/bin/bash

# RTX 2060 (6GB VRAM, CUDA)
# Larger model than T480 CPU run, but conservative for 6GB VRAM
# Expect ~15-30 min for base training

# Run as:
# bash runs/rungpu_2060.sh

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~100M characters
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=100000000
python -m scripts.tok_eval

# train a 6 layer model
# depth=6, max-seq-len=512 for better quality
# device-batch-size=8 to stay safe within 6GB VRAM
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=8192 \
    --eval-every=200 \
    --eval-tokens=262144 \
    --core-metric-every=-1 \
    --sample-every=200 \
    --num-iterations=2000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=4 --split-tokens=16384 --max-per-task=16

# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# LRs scaled by sqrt(8192/524288) = 0.125 to match batch size
# (chat_sft doesn't auto-scale like base_train does)
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=8192 \
    --embedding-lr=0.0375 \
    --unembedding-lr=0.001 \
    --matrix-lr=0.0025 \
    --eval-every=200 \
    --eval-tokens=262144 \
    --num-iterations=500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
