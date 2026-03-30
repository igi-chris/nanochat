#!/bin/bash

# RTX 2060 (6GB VRAM, CUDA) - Ambitious longer run
# depth=8 (~100M params), 5000 iterations, full tokenizer
# Expect ~45-60 min for base training
# Monitor GPU temp - 85°C observed on shorter run, throttles ~88°C

# Run as:
# bash runs/rungpu_2060_long.sh

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

# train tokenizer on full 2B characters for better vocab coverage
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# train an 8 layer model (~100M params)
# device-batch-size=8 kept conservative for VRAM headroom
# total-batch-size=16384 for training stability
# 5000 iterations for better convergence
python -m scripts.base_train \
    --depth=8 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --eval-every=500 \
    --eval-tokens=262144 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=4 --split-tokens=16384 --max-per-task=16

# SFT (more iterations for the larger model)
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
# LRs scaled by sqrt(16384/524288) = 0.177 to match batch size
# (chat_sft doesn't auto-scale like base_train does)
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --embedding-lr=0.053 \
    --unembedding-lr=0.0014 \
    --matrix-lr=0.0035 \
    --eval-every=200 \
    --eval-tokens=262144 \
    --num-iterations=1000 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
