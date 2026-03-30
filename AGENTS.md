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
- RTX 2060 (6GB VRAM), 16GB RAM
- Use `uv sync --extra gpu`
- Use `runs/rungpu_2060.sh`
- Settings: depth=6, max-seq-len=512, device-batch-size=8
- ~55M params, much faster training (~15-30 min)

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

## Ambitious Run (RTX 2060 - longer training)

Observed during current run (depth=6, device-batch-size=8):
- VRAM: 4.3/6.0 GB (1.7 GB headroom)
- GPU utilisation: 94%
- Temperature: 85°C (near throttle point ~88°C, monitor on longer runs)
- System RAM: 13.8/15.8 GB (87% - fairly tight)

Suggested settings for a longer, higher-quality run:
- **depth=8** — use the VRAM headroom for more layers (~100M params)
- **device-batch-size=8** — keep as-is, VRAM is tight for larger model
- **max-seq-len=512** — keep as-is (longer would blow VRAM with depth=8)
- **total-batch-size=16384** — more gradient accumulation for stability
- **num-iterations=5000** — longer training for better convergence
- **Tokenizer**: train on full 2B chars (`--max-chars=2000000000`) for better vocab
- Estimated time: ~45-60 min base training
- If depth=8 OOMs, fall back to depth=6 with device-batch-size=16

## Notes
- Base model = text completion only (not assistant-like)
- SFT step required for assistant behavior
- Tokenizer trained on 100M chars (vs 2B in original script)
- Expect 1-2+ hours for base training on CPU
