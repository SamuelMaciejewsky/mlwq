# MLWQ Reproducibility Notes

This repository includes the MLWQ paper in `docs/` as PDF and Markdown. The PDF is the source of truth because the Markdown extraction loses formulas and figures.

## Paper Equations Implemented

- Equation 1: uniform quantization with scale and zero-point.
- Equation 2 and 5: Hessian-inspired channel salience.
- Equation 3: channel-wise distribution loss based on activation mean and variance.
- Equation 4: BPLL constrained bit-width allocation.
- Equations 6 and 7: grouped clipping/quantization parameter tuning model for TQP.

## Code Mapping

- BPLL allocation lives in `mlwq/bpll.py` and is used by `mlwq/run.py`; the bit budget is weighted by each sublayer's parameter count.
- Channel-wise distribution helpers live in `mlwq/utils/reconstruct.py`.
- Uniform grouped quantization lives in `mlwq/utils/mixed_quantizer.py`.
- The GPTQ-compatible adapter lives in `mlwq/mlwq_gptq.py`.
- Dataset loading and WikiText2 caching live in `mlwq/datautils.py`.

## Reproducibility Boundary

The public upstream repository named in the paper does not include the custom `MLWQGPTQ`, mixed quantizer, `model_utils.LMClass`, or AutoGPTQ/SliM-LLM kernel integration files. This fork restores the missing Python APIs with a conservative PyTorch implementation so the algorithmic flow can run and be tested.

Expected reproducible now:

- Python imports and static compilation.
- BPLL bit allocation under a fixed budget using activation distribution loss.
- MQSA mixed bit-width application across non-salient, ordinary, and salient channels.
- TQP-style clipping grid search per salience group.
- WikiText2 loading and caching.
- Small-model smoke runs for calibration, quantization, and PPL evaluation.

Not expected to reproduce exactly yet:

- Table-level perplexity numbers from the paper.
- RTX 4090 token/s and packed mixed-precision kernel results.
- Bit-identical AutoGPTQ artifacts.

## Suggested Validation Order

From the standalone `mlwq` repository:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r mlwq/requirements.txt
uv pip install -e .
```

From the parent research repository:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r mlwq/mlwq/requirements.txt
uv pip install -e ./mlwq
```

The import name is `mlwq`; `wlmq` is a typo and is not provided by this
package.

1. Run `python -c "import mlwq.run; print(mlwq.run.__file__)"`.
2. Run `python -m py_compile mlwq/*.py mlwq/utils/*.py mlwq/model_utils/*.py`.
3. Run `pytest tests`.
4. Run a tiny OPT smoke test with `--nsamples 1`.
5. Run the FP16 baseline first, then the paper-like setting only after confirming CUDA memory availability:

```bash
python -m mlwq.run facebook/opt-125m wikitext2 3bit --device cuda:0 --eval_fp16_only --nsamples 128 --groupsize 128 --run_name opt125_fp16 --save_metrics metrics/opt125_fp16.json
python -m mlwq.run facebook/opt-125m wikitext2 3bit --device cuda:0 --nsamples 128 --groupsize 128
```

For a staged reproduction with logs and JSON metrics:

```bash
bash mlwq/scripts/reproduce_opt.sh
```

With a 16 GB GPU, start with `facebook/opt-125m`. If that completes, try:

```bash
MODEL=facebook/opt-350m bash mlwq/scripts/reproduce_opt.sh
```

Reference PPL targets from the paper for context only:

| Model | FP16 WikiText2 | MLWQ 3-bit WikiText2 |
|---|---:|---:|
| OPT-125M | 27.65 | 28.98 |
| OPT-350M | 22.01 | 23.81 |
