# MLWQ: Efficient Small Language Model Deployment via Multi-Level Weight Quantization

This fork stabilizes the public MLWQ code enough to run reproducible smoke tests and small experiments from the paper:

- BPLL: bit-width preallocation based on layer loss.
- MQSA: intra-layer salient, ordinary, and non-salient channel grouping.
- TQP-compatible grouped quantization parameters.

The original paper reports custom AutoGPTQ/SliM-LLM mixed-precision kernels. Those kernels are not included in the public upstream repository, so this fork provides a conservative Python/PyTorch implementation for reproducibility and debugging. It is not expected to match the paper's throughput numbers.

## Requirements

- Python 3.10
- NVIDIA GPU for realistic model runs
- `uv` for environment management

## Install

```bash
git clone https://github.com/SamuelMaciejewsky/mlwq.git
cd mlwq
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r mlwq/requirements.txt
uv pip install -e .
```

When working from the parent research repository, install the submodule in
editable mode instead:

```bash
uv pip install -r mlwq/mlwq/requirements.txt
uv pip install -e ./mlwq
```

## Verify

```bash
python -c "import mlwq.run; print(mlwq.run.__file__)"
python -m py_compile mlwq/*.py mlwq/utils/*.py mlwq/model_utils/*.py
pytest tests
```

## Smoke Run

Use a small Hugging Face causal language model first. This validates the pipeline, not paper-level quality.

```bash
python -m mlwq.run facebook/opt-125m wikitext2 3bit --device cuda:0 --nsamples 1 --groupsize 128
```

Zero-shot evaluation requires a compatible `lm-eval` installation:

```bash
python -m mlwq.run facebook/opt-125m wikitext2 3bit --device cuda:0 --tasks piqa
```

## Paper-Like OPT Reproduction

On a CUDA machine, run the staged OPT reproduction script:

```bash
bash mlwq/scripts/reproduce_opt.sh
```

Use a larger OPT model only after `facebook/opt-125m` completes:

```bash
MODEL=facebook/opt-350m bash mlwq/scripts/reproduce_opt.sh
```

The script writes raw logs to `logs/` and structured metrics to `metrics/`.

For the 16 GB VRAM target, treat `facebook/opt-125m` as the acceptance run. Move to `facebook/opt-350m` only after the FP16 baseline, smoke run, and full `nsamples=128` MLWQ run complete.

## Paper

The project paper is in `docs/` as both PDF and extracted Markdown. See `docs/reproducibility.md` for the implementation mapping and current reproducibility boundary.
