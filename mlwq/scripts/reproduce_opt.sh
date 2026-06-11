#!/usr/bin/env bash
set -euo pipefail

# Script de reprodução do MLWQ para modelos OPT
#
# Variáveis de ambiente (ou use valores padrão):
#   MODEL=facebook/opt-125m
#   DEVICE=cuda:0
#   GROUPSIZE=128
#   NSAMPLES=128
#   SMOKE_NSAMPLES=1
#   BPLL_LOSS=activation_dist
#   TQP_GRID=1.0,0.95,0.9,0.85
#
# Uso: ./reproduce_opt.sh [flags_extras]
# Exemplo: ./reproduce_opt.sh --keep_on_gpu
#       MODEL=facebook/opt-350m ./reproduce_opt.sh --keep_on_gpu

MODEL="${MODEL:-facebook/opt-125m}"
DEVICE="${DEVICE:-cuda:0}"
GROUPSIZE="${GROUPSIZE:-128}"
SEED="${SEED:-0}"
NSAMPLES="${NSAMPLES:-128}"
SMOKE_NSAMPLES="${SMOKE_NSAMPLES:-1}"
BPLL_LOSS="${BPLL_LOSS:-activation_dist}"
TQP_GRID="${TQP_GRID:-1.0,0.95,0.9,0.85}"
MODEL_SLUG="${MODEL//\//_}"

mkdir -p logs metrics

echo "=========================================="
echo "MLWQ Reprodução - OPT Models"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Groupsize: $GROUPSIZE"
echo "Samples: $NSAMPLES"
echo "Flags extras: $@"
echo "=========================================="

python -V | tee logs/env.txt
if python -m pip freeze >/tmp/mlwq-pip-freeze.txt 2>/tmp/mlwq-pip-freeze.err; then
  cat /tmp/mlwq-pip-freeze.txt | tee -a logs/env.txt
elif command -v uv >/dev/null 2>&1; then
  uv pip freeze | tee -a logs/env.txt
else
  cat /tmp/mlwq-pip-freeze.err | tee -a logs/env.txt
  echo "dependency freeze skipped: pip and uv are unavailable" | tee -a logs/env.txt
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee logs/nvidia-smi.txt
else
  echo "nvidia-smi not found" | tee -a logs/env.txt
fi

python -m py_compile mlwq/*.py mlwq/utils/*.py mlwq/model_utils/*.py
python -m pytest tests

# FP16 baseline
python -m mlwq.run "$MODEL" wikitext2 3bit \
  --device "$DEVICE" \
  --seed "$SEED" \
  --nsamples "$NSAMPLES" \
  --groupsize "$GROUPSIZE" \
  --eval_fp16_only \
  --run_name "${MODEL_SLUG}_fp16_wikitext2" \
  --save_metrics "metrics/${MODEL_SLUG}_fp16_wikitext2.json" \
  "$@" 2>&1 | tee "logs/${MODEL_SLUG}_fp16_wikitext2.log"

# Smoke test
python -m mlwq.run "$MODEL" wikitext2 3bit \
  --device "$DEVICE" \
  --seed "$SEED" \
  --nsamples "$SMOKE_NSAMPLES" \
  --groupsize "$GROUPSIZE" \
  --run_name "${MODEL_SLUG}_smoke_ns${SMOKE_NSAMPLES}" \
  --save_metrics "metrics/${MODEL_SLUG}_smoke_ns${SMOKE_NSAMPLES}.json" \
  "$@" 2>&1 | tee "logs/${MODEL_SLUG}_smoke_ns${SMOKE_NSAMPLES}.log"

# MLWQ 3-bit quantização
python -m mlwq.run "$MODEL" wikitext2 3bit \
  --device "$DEVICE" \
  --seed "$SEED" \
  --nsamples "$NSAMPLES" \
  --groupsize "$GROUPSIZE" \
  --bpll_loss "$BPLL_LOSS" \
  --tqp_grid "$TQP_GRID" \
  --run_name "${MODEL_SLUG}_mlwq_3bit_ns${NSAMPLES}" \
  --save_metrics "metrics/${MODEL_SLUG}_mlwq_3bit_ns${NSAMPLES}.json" \
  "$@" 2>&1 | tee "logs/${MODEL_SLUG}_mlwq_3bit_ns${NSAMPLES}.log"
