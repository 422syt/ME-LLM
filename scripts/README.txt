Regenerate the reported values.

The model source code lives on the `main` branch of this repository. This release
branch ships the per-value `configs/` and the regeneration scripts under `scripts/repro/`.

1) Check out the code and prepare datasets (see data_provider/).
2) Run one per-value experiment (base seed 2021, itr 10 -> seeds 2021-2030):
   python run_main.py --config configs/<config>.json
   or use the single-GPU scripts:
   bash scripts/repro/MELLM_<dataset>.sh
3) Each run writes its raw result JSON to results/ and its checkpoint to
   checkpoints/; the reported value is the mean over the 10 itrs.
