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

### Completed (T480 CPU)
- [x] Cloned nanochat repo
- [x] Installed dependencies (`uv sync --extra cpu`)
- [x] Created `runs/runcpu_t480.sh` with conservative settings for CPU
- [x] Trained tokenizer on 100M chars (~5 sec)
- [x] Installed setuptools (required for PyTorch CPU codegen)

### Completed (Work PC - RTX 2060)
- [x] Base training (depth=6, 2000 iterations, ~13.5 min)
  - Loss: ~10 → 4.38, val bpb: 1.39
  - Samples show English structure but very repetitive — model is undertrained
  - Peak VRAM: 2770 MiB (plenty of headroom)
- [x] base_eval ran — CORE metric: -0.02 (near random, expected for small model)
- [x] SFT ran but **failed** — loss went NaN at step 7

### Known Issues Fixed
- KV cache dtype mismatch on pre-Ampere GPUs (fixed in engine.py)
- Python.h missing — need `pyenv install 3.10` (system python3.10 has no dev headers)
- SFT NaN: `chat_sft.py` doesn't scale LRs for small batch sizes like `base_train.py` does.
  The base_train scales by `√(batch_size/524288)` but chat_sft uses raw unscaled LRs.
  Workaround: pass explicit `--embedding-lr`, `--unembedding-lr`, `--matrix-lr` in run scripts.

### Next Steps (next session)
1. **Re-run SFT only** (base checkpoint at step 2000 is fine, no need to retrain):
   ```bash
   source .venv/bin/activate
   python -m scripts.chat_sft \
       --max-seq-len=512 --device-batch-size=8 --total-batch-size=8192 \
       --embedding-lr=0.0375 --unembedding-lr=0.001 --matrix-lr=0.0025 \
       --eval-every=200 --eval-tokens=262144 --num-iterations=500 --run=dummy
   ```
2. **Test chat**: `python -m scripts.chat_cli -p "What is the capital of France?"`
3. **If quality is poor**, try the ambitious run: `bash runs/rungpu_2060_long.sh`

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

# Work PC (GPU) - standard run
uv sync --extra gpu
bash runs/rungpu_2060.sh

# Work PC (GPU) - ambitious longer run
bash runs/rungpu_2060_long.sh

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
