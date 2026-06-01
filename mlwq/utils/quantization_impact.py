"""
Compare the model FP16 vs quantized model
Calculate the weights memory reduction
Show the total impact including KV Cahce
"""

def estimate_kv_cache(config, seq_len):
  """Helper function"""

def compare_quantization_impact(fp16_model, quantized_model, seq_len=2048):

  """
  Intputs the fp16, quantized model and the seq_len
  Calculates the FP16 memory: param_count * 2 bytes
  Calculates the Quantized memory: param_count * 3 / 8
  Estimate KC cahce
  Cacculate the percetual reduction
  Outputs a dict with the reduction metrics
  """

  # FP16 Memory
  fp16_weight = sum(p.numel() * 2 for p in fp16_model.parameters())
  fp16_total = fp16_weight + estimate_kv_cache(fp16_model.config, seq_len) * 2