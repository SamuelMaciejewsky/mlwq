import torch
from transformers import AutoModelForCausalLM  # noqa: F401


def measure_memory_decomposition(model, seq_len=2048):
  """
    Input the model and sequence length
    Extract config (num layers, num heads and head dimension)
    Calc KV cache given by 2 * layers * heads * dimension* seq_len * 2bytes
    Estimate activation (~10% from total)
    Output a dictionary with memory (MB) and percentages
  """

  # 1. Weights memory
  weight_memory = 0
  for name, param in model.named_parameters():
    weight_memory += param.numel() * param.elements_size()

  # 2. KV-Cache memory
  config = model.config

  # Access memory specific architecture
  if hasattr(config, 'num_hidden_layers'):
    num_layers = config.num_hidden_layers
  elif hasattr(config, 'n_layer'):
    num_layers = config.n_layers
  else:
    num_layers = config.num_layers

  if hasattr(config, 'num_attention_heads'):
    head_dim = config.hidden_size
  elif hasattr(config, 'n_head'):
    num_heads = config.n_head
  else:
    num_heads = config.num_attention_heads

  if hasattr(config, 'hidden_size'):
    head_dim = config.hidden_size // num_heads
  elif hasattr(config, 'n_embd'):
    head_dim = config.n_embd // num_heads
  else:
    head_dim = config.hidden_size // num_heads

  # Assume FP16/BF16 for KV cache (2 bytes)
  kv_memory = 2 * num_layers * num_heads * head_dim * seq_len * 2

  # 3. Total memory (estimated)
  total_memory = weight_memory + kv_memory

  # 4. Activation memory (estimated)
  activation_memory = total_memory * 0.1 # ~10% of total

  results = {

    'weight_memory_mb': weight_memory / (1024**2),
    'kv_cache_memory_mb': kv_memory / (1024**2),
    'activation_memory_mb': activation_memory / (1024**2),
    'total_memory_mb': (weight_memory + kv_memory + activation_memory) / (1024**2),
    'weight_percentage': weight_memory / total_memory * 100,
    'kv_percentage': kv_memory / total_memory * 100

  }

  return results

def measure_inference_memory(model, tokenizer, prompt, max_new_tokens=100):

  """
  Input the model, tokenizer, prompt and max_token_size and measure the: 
  - Static memory (weights + model overhead)
  - Peak(max) memory used during the inference
  - Difference given by KV Cache (increases linearly)
  - How much tokens are generated
  Output all results
  """

  # Clear memory
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.empty_cache()

  # Initial memory (Model loaded)
  initial_memory = torch.cuda.memory_allocated()

  # Encode input
  inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

  # Memory after tokenizer + preparation input
  input_memory = torch.cuda.memory_allocated()

  # Generate
  with torch.no_grad():
    outputs = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      return_dict_in_generate=True,
      use_cache=True
    )
  
  # Memory peak during the generation
  peak_memory = torch.cuda.memory_allocated()

  # KV cache size
  kv_cache_memory = peak_memory - input_memory
  
  results = {
    'initial_mb': initial_memory / (1024**2),
    'peak_mb': peak_memory / (1024**2),
    'kv_cache_mb': kv_cache_memory / (1024**2),
    # How much tokens have been generated 
    'tokens_generated': len(outputs[0][0]) - inputs['input_ids'].shape[1]
  }

  return results
