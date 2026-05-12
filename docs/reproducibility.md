# MLWQ Reproducibility Notes

This repository includes the MLWQ paper in `docs/` as PDF and Markdown. The PDF is the source of truth because the Markdown extraction loses formulas and figures.

## Paper Equations Implemented

- Equation 1: uniform quantization with scale and zero-point.
- Equation 2 and 5: Hessian-inspired channel salience.
- Equation 3: channel-wise distribution loss based on activation mean and variance.
- Equation 4: BPLL constrained bit-width allocation.
- Equations 6 and 7: grouped clipping/quantization parameter tuning model for TQP.

## Code Mapping

- BPLL allocation lives in `mlwq/bpll.py` and is used by `mlwq/run.py`.
- Channel-wise distribution helpers live in `mlwq/utils/reconstruct.py`.
- Uniform grouped quantization lives in `mlwq/utils/mixed_quantizer.py`.
- The GPTQ-compatible adapter lives in `mlwq/mlwq_gptq.py`.
- Dataset loading and WikiText2 caching live in `mlwq/datautils.py`.

## Reproducibility Boundary

The public upstream repository named in the paper does not include the custom `MLWQGPTQ`, mixed quantizer, `model_utils.LMClass`, or AutoGPTQ/SliM-LLM kernel integration files. This fork restores the missing Python APIs with a conservative PyTorch implementation so the algorithmic flow can run and be tested.

Expected reproducible now:

- Python imports and static compilation.
- BPLL bit allocation under a fixed budget.
- WikiText2 loading and caching.
- Small-model smoke runs for calibration, quantization, and PPL evaluation.

Not expected to reproduce exactly yet:

- Table-level perplexity numbers from the paper.
- RTX 4090 token/s and packed mixed-precision kernel results.
- Bit-identical AutoGPTQ artifacts.

## Suggested Validation Order

1. Run `python -m py_compile mlwq/*.py mlwq/utils/*.py mlwq/model_utils/*.py`.
2. Run `pytest tests`.
3. Run a tiny OPT smoke test with `--nsamples 1`.
4. Run the paper-like setting only after confirming CUDA memory availability:

```bash
python -m mlwq.run facebook/opt-125m wikitext2 3bit --device cuda:0 --nsamples 128 --groupsize 128
```
