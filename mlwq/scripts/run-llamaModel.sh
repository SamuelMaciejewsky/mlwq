#!/usr/bin/bash
# Script para rodar quantização MLWQ em modelos Llama
# Output salvo em: logs/YYYY-MM-DD_HH-MM-SS/
#
# Uso: ./run-llamaModel.sh [flags_extras]
# Exemplo: ./run-llamaModel.sh --keep_on_gpu --nsamples 256

# Criar diretório de logs com data/hora
LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/logs/$(date +'%Y-%m-%d_%H-%M-%S')"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/mlwq.log"

echo "=========================================="
echo "MLWQ Quantization - Llama Models"
echo "=========================================="
echo "Log directory: $LOG_DIR"
echo "Flags extras: $@"
echo "=========================================="

# Executar e salvar log (mostra no terminal E salva no arquivo)
# "$@" permite passar flags extras como --keep_on_gpu, --nsamples, etc.
python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit --device "cuda:0" --tasks piqa --torch_compile "$@" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log salvo em: $LOG_FILE"
