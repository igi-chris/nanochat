# Picochat - Tiny LLM Training on Thinkpad T480

## Goal
Train a small LLM on a CPU-only Thinkpad T480 (32GB RAM, no GPU) using Karpathy's nanochat.

## Hardware Setups

### Thinkpad T480 (Primary - CPU-only)
- 32GB RAM, no GPU
- Use `uv sync --extra cpu`
- Use `runs/runcpu_t480.sh` (conservative settings)
- Settings: depth=4, max-seq-len=256, device-batch-size=8
- ~36M params, expect 1-2+ hours training

### Work PC (Alternative - GPU)
- RTX 2060 (8GB VRAM), 16GB RAM
- Use `uv sync --extra gpu`
- Can use larger model: depth=6-8, max-seq-len=512
- device-batch-size=16-32 should fit in 8GB VRAM
- ~50-100M params, much faster training

## Progress

### Completed
- [x] Cloned nanochat repo
- [x] Installed dependencies (`uv sync --extra cpu`)
- [x] Created `runs/runcpu_t480.sh` with conservative settings for CPU
- [x] Trained tokenizer on 100M chars (~5 sec)
- [x] Installed setuptools (required for PyTorch CPU codegen)
- [ ] Base training (currently running)

### In Progress
- Base model pretraining with:
  - depth=4, head-dim=64, max-seq-len=256
  - device-batch-size=8, total-batch-size=4096
  - num-iterations=2000
  - ~36M parameters

### Pending
- [ ] Base training completes
- [ ] Run base_eval to check model quality
- [ ] SFT (Supervised Fine-Tuning) for assistant behavior
- [ ] Test chat_cli with the model
- [ ] Optional: Try on work PC with RTX 2060 for larger model

## Model Architecture (depth=4)
```
Vocab size: 32,768
n_layer: 4
n_head: 4
n_embd: 256
max_seq_len: 256
Total params: ~36.7M
```

## Key Commands

```bash
# Thinkpad T480 (CPU-only)
uv sync --extra cpu
bash runs/runcpu_t480.sh

# Work PC (GPU)
uv sync --extra gpu
python -m scripts.base_train --depth=6 --head-dim=64 --max-seq-len=512 \
    --device-batch-size=16 --total-batch-size=8192 --num-iterations=2000 --run=test

# Test text completion (after base training)
python -m scripts.chat_cli -p "Once upon a time"

# Test assistant mode (after SFT)
python -m scripts.chat_cli -p "What is the capital of France?"
```

## Notes
- Base model = text completion only (not assistant-like)
- SFT step required for assistant behavior
- Tokenizer trained on 100M chars (vs 2B in original script)
- Expect 1-2+ hours for base training on CPU
