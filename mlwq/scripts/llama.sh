#!/usr/bin/bash
# Script simples para rodar quantização MLWQ em modelos Llama
#
# Uso: ./llama.sh [flags_extras]
# Exemplo: ./llama.sh --keep_on_gpu --torch_compile

python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit --groupsize 128 --device "cuda:0" "$@"
