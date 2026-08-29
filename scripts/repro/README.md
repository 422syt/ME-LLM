# Reproducibility re-run scripts (single GPU, BERT, pred_len=96)

These scripts are a **reduced, local-machine variant** of the original runners in
`scripts/`. They were created (not editing the originals) to regenerate the
Reviewer-4 reproducibility artifacts on one RTX 4060 Ti 8GB.

## Differences from `scripts/*.sh`

| Item | Original `scripts/` | Here (`scripts/repro/`) |
| --- | --- | --- |
| Backbone | `llama_layers=32` (ambiguous) | BERT-base (`--llm_model BERT --llm_dim 768 --llm_layers 12`) |
| Precision | `accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 8` | single GPU + `--use_amp` (bf16 autocast in-code) |
| Epoch cap | 50 / 100 | `train_epochs=10` |
| Prediction lengths | 96 / 192 / 336 / 720 | `pred_len=96` only (M4 keeps all 6 seasonal patterns) |
| Data paths | `./dataset/ETT-small/`, lowercase `electricity`/`weather`/`traffic` | `./dataset/<Dataset>/` + actual CSV filenames |

## Run

Each script runs one `python run_main.py` / `run_m4.py` / `run_pretrain.py`
invocation. Example:

```bash
cd E:/ME-LLM
bash scripts/repro/MELLM_ETTh1.sh
```

## Known code-level caveats (not yet fixed)

- `run_m4.py` and `run_pretrain.py` default `--llm_model LLAMA --llm_dim 4096`;
  these scripts override them to BERT/768 explicitly.
- `run_pretrain.py` still uses fp16 `torch.cuda.amp.autocast()` in its `--use_amp`
  path (the bf16 fix was only applied to `run_main.py` / `utils/tools.py`). Until
  that is changed to `dtype=torch.bfloat16`, the pretrain run may hit fp16 NaN.
