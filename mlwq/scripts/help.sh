#!/usr/bin/bash
# Script Help - Guia completo do MLWQ
#
# Este script explica como usar run.py, suas flags e os scripts disponíveis.

cat << 'EOF'
─────────────────────────────────────────────────────────────────────────────
 1. COMO RODAR O RUN.PY                                                      
─────────────────────────────────────────────────────────────────────────────

FORMA BÁSICA:
  python -m mlwq.run <modelo> <dataset> <bits> [flags]

EXEMPLOS:
  # Quantização 3-bit do Llama-3.2-3B
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit

  # Com GPU acelerada
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit --keep_on_gpu

  # Com torch compile
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit --torch_compile

─────────────────────────────────────────────────────────────────────────────
 2. PARÂMETROS OBRIGATÓRIOS                                                  
─────────────────────────────────────────────────────────────────────────────

  modelo    : Nome do modelo HuggingFace
              Exemplos: meta-llama/llama-3.2-3B, facebook/opt-125m

  dataset   : Dataset para calibração
              Opções: wikitext2, ptb, c4, mix

  bits      : Bits de quantização
              Opções: 2bit, 3bit, 4bit, 5bit, 6bit, 7bit, 8bit

─────────────────────────────────────────────────────────────────────────────
 3. FLAGS PRINCIPAIS                                                         
─────────────────────────────────────────────────────────────────────────────

  --device              Dispositivo (default: cuda:0)
                        Opções: cuda:0, cuda:1, cpu

  --keep_on_gpu         Mantém modelo na GPU (MUITO mais rápido!)
                        Requer mais VRAM (~8-12GB vs 4-6GB)
                        Uso GPU: 70-95% (vs 10-40% sem flag)
                        TEMPO: ~1-1.5h (vs ~4h sem flag)

  --torch_compile       Compila modelo com PyTorch 2.0+ (20-40% mais rápido)
                        Modos disponíveis via --compile_mode:
                          - default (padrão)
                          - reduce-overhead (reduz overhead)
                          - max-autotune (máxima otimização)

  --tasks               Tasks zero-shot para avaliar
                        Exemplo: --tasks piqa,hellaswag,arc_easy

  --nsamples            Número de amostras para calibração (default: 128)
                        Aumentar para melhor calibração (mais lento)

  --groupsize           Tamanho do grupo para quantização (default: 128)
                        Opções comuns: 64, 128, 256

─────────────────────────────────────────────────────────────────────────────
 4. FLAGS DE QUANTIZAÇÃO                                                     
─────────────────────────────────────────────────────────────────────────────

  --bpll_loss            Tipo de perda para BPLL (default: activation_dist)
                        Opções: activation_dist, weight_mse

  --tqp_grid             Grid de clipping ratios (default: 1.0,0.95,0.9,0.85)
                        Formato: lista separada por vírgulas

  --hessian_mode         Modo de cálculo Hessian (default: diag)
                        Opções: diag, full

  --percdamp             Porcentagem de dampening (default: 0.01)

  --eval_fp16_only       Apenas avaliação FP16 (pula quantização)

─────────────────────────────────────────────────────────────────────────────
 5. FLAGS DE SAÍDA                                                           
─────────────────────────────────────────────────────────────────────────────

  --save                 Salvar modelo quantizado

  --pack                 Empacotar modelo para AutoGPTQ

  --save_metrics         Caminho para salvar métricas JSON
                        Exemplo: --save_metrics metrics/meu_run.json

  --run_name             Nome para identificar a execução
                        Exemplo: --run_name "llama-3b-test-1"

  --log_wandb            Habilitar logging no Weights & Biases

─────────────────────────────────────────────────────────────────────────────
 6. FLAGS DE CONTROLE                                                        
─────────────────────────────────────────────────────────────────────────────

  --seed                Semente para reproducibilidade (default: 0)

  --minlayer            Quant apenas camadas com ID >= este valor

  --maxlayer            Quant apenas camadas com ID < este valor

  --quant_only           Quant apenas camadas contendo este texto

  --invert              Inverte seleção de camadas

  --load_quantized       Carrega modelo já quantizado

─────────────────────────────────────────────────────────────────────────────
 7. SCRIPTS DISPONÍVEIS                                                      
─────────────────────────────────────────────────────────────────────────────

  ./run-llamaModel.sh    [flags]
    → Executa quantização em Llama-3.2-3B
    → Salva log em logs/YYYY-MM-DD_HH-MM-SS/
    → Aceita flags extras: --keep_on_gpu, --nsamples, etc.

  ./llama.sh              [flags]
    → Script simples para Llama
    → Usa configurações básicas
    → Aceita flags extras

  ./reproduce_opt.sh     [flags]
    → Reproduz experimentos do paper para OPT
    → Executa 3 fases: FP16 baseline, smoke test, MLWQ 3-bit
    → Variáveis via export: MODEL, DEVICE, GROUPSIZE, etc.
    → Exemplo: MODEL=facebook/opt-350m ./reproduce_opt.sh --keep_on_gpu

  ./help.sh              → Este guia

─────────────────────────────────────────────────────────────────────────────
 8. EXEMPLOS PRÁTICOS                                                        
─────────────────────────────────────────────────────────────────────────────

  # Smoke test rápido (1 amostra)
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit --nsamples 1

  # Quantização rápida com GPU acelerada
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit --keep_on_gpu

  # Quantização completa com todas otimizações
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit \
    --keep_on_gpu --torch_compile --nsamples 256

  # Avaliação zero-shot em múltiplas tasks
  python -m mlwq.run meta-llama/llama-3.2-3B wikitext2 3bit \
    --tasks "piqa,hellaswag,arc_easy,arc_challenge"

  # Usar script com log
  ./run-llamaModel.sh --keep_on_gpu

  # Reproduzir paper OPT-350m
  MODEL=facebook/opt-350m ./reproduce_opt.sh --keep_on_gpu

─────────────────────────────────────────────────────────────────────────────
 9. SOLUÇÃO DE PROBLEMAS                                                    
─────────────────────────────────────────────────────────────────────────────

  ERRO: CUDA out of memory
    → Não use --keep_on_gpu
    → Reduza --nsamples
    → Use modelo menor

  ERRO: AssertionError (lm_eval)
    → Verifique se --tasks tem nomes corretos
    → Problema comum já corrigido no código

  WARNING: "Not enough SMs to use max_autotune_gemm mode"
    → Normal, não afeta resultados
    → Use --compile_mode "reduce-overhead" ao invés

  TEMPO MUITO LENTO:
    → Use --keep_on_gpu
    → Verifique com "watch -n 1 nvidia-smi"
    → GPU usage deve ser 70-95% com flag
EOF
