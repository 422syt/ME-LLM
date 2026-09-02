# ME-LLM package contents

Archived release tag: `asoc-r2-v1.0`

This archive is a compact reproducibility package for ME-LLM. It contains:

- the project README documenting the recorded experimental protocol and tagged-repository execution steps;
- reference copies of `models/MELLM.py` and `run_main.py` from the supplied source snapshot;
- environment specification files (`environment.yml` and `requirements.txt`);
- machine-readable copies of the reported result tables (Tables 4–18 and Supplementary Table S1);
- result and source indexes, checksums, and a package verification script.

## Scope of this archive

This ZIP is **not** intended to be a standalone runnable source distribution. It does not bundle every runtime module from the public repository (for example, `data_provider/` and `layers/`), benchmark datasets, local BERT-base model files, or the full set of checkpoint binaries.

To run training or evaluation, check out the public repository at the archived release tag `asoc-r2-v1.0` and follow the command examples in `README.md`.

Public repository:
https://github.com/422syt/ME-LLM

Archived release:
https://github.com/422syt/ME-LLM/releases/tag/asoc-r2-v1.0
