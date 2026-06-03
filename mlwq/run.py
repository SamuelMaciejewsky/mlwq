import json
import platform
import subprocess
import time

import torch
import tqdm
from torch import nn

from mlwq.bpll import compute_bit_assignments
from mlwq.mlwq_gptq import MLWQGPTQ
from mlwq.modelutils import find_layers
from mlwq.utils.mixed_quantizer import Quantizer


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    name = model
    if "opt" in model:
        from transformers import OPTForCausalLM

        model = OPTForCausalLM.from_pretrained(model, torch_dtype="auto")
        model.seqlen = model.config.max_position_embeddings
    elif "llama" in model:
        from transformers import LlamaForCausalLM
       
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
        if "llama-3" in name.lower():
            model.seqlen = 2048
        else:
            model.seqlen = 2048
    elif "Smo" or "gemma" in model:  # noqa: SIM222
        from transformers import AutoModelForCausalLM
        
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
        print(model)
        model.seqlen=2048
    return model

def bit_options_for_target(target_bits):
    target_bits = int(target_bits)
    if target_bits <= 2:
        return (2, 3), 2
    if target_bits == 3:
        return (2, 3), 3
    return (target_bits - 1, target_bits), target_bits


def salience_ratios_for_bits(bits):
    if int(bits) <= 2:
        return 0.02, 0.10
    if int(bits) == 3:
        return 0.05, 0.05
    return 0.10, 0.02


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def cuda_metrics_for_device(device):
    metric_device = torch.device(device)
    if not (torch.cuda.is_available() and metric_device.type == "cuda"):
        return {}

    device_index = metric_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(device_index),
        "gpu_memory_total_gb": round(properties.total_memory / (1024 ** 3), 2),
    }


def compute_bit(gptq):
    requested_bits = int(args.low_quant_method[0])
    bit_options, target_average = bit_options_for_target(requested_bits)
    layer_sizes = {
        name: int(sublayer.layer.weight.data.numel())
        for name, sublayer in gptq.items()
    }
    target_total = target_average * sum(layer_sizes.values())
    if len(gptq) > 1 and target_average > min(bit_options):
        target_total -= min(layer_sizes.values())
    best_combinations, min_total_error = compute_bit_assignments(
        gptq,
        bit_options=bit_options,
        target_total=target_total,
        layer_sizes=layer_sizes,
    )
    print("BPLL min total error:", min_total_error)
    return best_combinations


@torch.no_grad()
def quant_sequential(model, dataloader, dev, saved_block_precision):
 

    for name, module in model.named_modules():
        module.global_name = args.model + name

    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "opt" in args.model:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            dev
        )
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    elif "llama" in args.model or "Smo" in args.model:
        layers = model.model.layers
    
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        model.model.rotary_emb=model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    if "opt" in args.model:
        cache = {"i": 0, "attention_mask": None}
    else:
        cache = {"i": 0, "attention_mask": None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if "llama" in args.model or "Smo" in args.model:
                cache["position_ids"] = kwargs["position_ids"]
         
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    if "opt" in args.model:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if (
            hasattr(model.model.decoder, "project_out")
            and model.model.decoder.project_out
        ):
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if (
            hasattr(model.model.decoder, "project_in")
            and model.model.decoder.project_in
        ):
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif "llama" in args.model or "Smo" in args.model:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
        model.model.rotary_emb=model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    if "llama" in args.model or "Smo" in args.model:
        position_ids = cache["position_ids"]

    print("Ready.")
    bpll_bit_options, _ = bit_options_for_target(int(args.low_quant_method[0]))
    index = 0
    mixed_block_precision = {}
    quantizers = {}
    svgm=0

    for i in tqdm.tqdm(range(len(layers)), desc="Running Mixed-Precision Quantization"):
        layer = layers[i].to(dev)
       
    

        subset = find_layers(layer)

        gptq = {}
        for name in subset:
            index += 1
          
            if (
                not (args.minlayer <= i < args.maxlayer and args.quant_only in name)
            ) == (not args.invert):
                continue
            quantizer = Quantizer(
                subset[name].weight,
                method=args.low_quant_method,
                groupsize=groupsize,
                metric = args.quantizer_matric,
                lambda_salience=args.lambda_salience        
            )
            gptq[name] = MLWQGPTQ(
                subset[name],
                quantizer,
                bit_width = int(args.low_quant_method[0]),
                bpll_loss=args.bpll_loss,
                tqp_grid=args.tqp_grid,
                hessian_mode=args.hessian_mode,
                hessian_full_threshold=args.hessian_full_threshold,
                bit_options=bpll_bit_options,
            )

        if not gptq:
            if not (args.minlayer <= i < args.maxlayer) and not args.invert:
                for j in range(args.nsamples):
                    if "opt" in args.model:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
                layers[i] = layer.cpu()
                del layer
                torch.cuda.empty_cache()
                inps, outs = outs, inps
                continue
            raise ValueError(
                f"no quantizable sublayers selected in layer {i}; "
                "check --quant_only/--minlayer/--maxlayer/--invert"
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)  # noqa: B023, F821

            return tmp

        handles = []
        for name in gptq:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            if "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles:
            h.remove()
              
       
            
        best_combinations=compute_bit(gptq)
       
    
        sv=0
        sumc=0
        

        for name in gptq:
            high_ratio, low_ratio = salience_ratios_for_bits(best_combinations[name])
            gptq[name].get_salience1(name, i, high_ratio, low_ratio, blocksize=args.groupsize)
            sv+=best_combinations[name]*subset[name].weight.data.shape[0]*subset[name].weight.data.shape[1]
            sumc+=subset[name].weight.data.shape[0]*subset[name].weight.data.shape[1]
        sv=sv/sumc 
        
        svgm+=sv
        
       
     
        mixed_block_precision[i] = {}
        
        for name in gptq:
            print(i, name)
            print("Quantizing ...")
            layer_block_precision, scales, zeros, g_idx = gptq[name].fasterquant1(
                percdamp=args.percdamp, 
                blocksize=args.groupsize,
                layer_name=name,
                best_combination=best_combinations[name],
                saved_block_precision=saved_block_precision[i][name] if (saved_block_precision is not None) else None,
            )
            mixed_block_precision[i][name] = layer_block_precision
            quantizers[f"model.layers.{i}.{name}"] = (scales, zeros, g_idx)


            gptq[name].free()
        
        for j in range(args.nsamples):
            if "opt" in args.model:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask,position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
  
   
    model.config.use_cache = use_cache
    return quantizers, mixed_block_precision

def pack_model(model, save_path, bits, group_size, quantizers, block_bits):
    import os
    import sys
    sys.path.append(os.path.join(os.getcwd(), '../AutoGPTQ'))
  
    from auto_gptq import BaseQuantizeConfig, LlamaGPTQForCausalLM, OPTGPTQForCausalLM
    quantize_config = BaseQuantizeConfig(
        bits=bits,  
        group_size=group_size, 
        desc_act=False,  
    )
    scales={k : quantizers[k][0] for k in quantizers}
    zeros={k : quantizers[k][1] for k in quantizers}
    g_idxes={k : quantizers[k][2] for k in quantizers}
    modified_block_bits = {}
    for k in block_bits:
        for name in block_bits[k]:
            modified_block_bits['model.layers.%d.%s' % (k, name)] = torch.Tensor(block_bits[k][name]).int()  # noqa: UP031

    if "llama" in args.model:
        model = LlamaGPTQForCausalLM(model, quantized=False, quantize_config=quantize_config)
        model.quantize([], scales, zeros, g_idxes, modified_block_bits)
        model.save_quantized(save_path, use_safetensors=False)
    elif "opt" in args.model:
        model = OPTGPTQForCausalLM(model, quantized=False, quantize_config=quantize_config)
        model.quantize([], scales, zeros, g_idxes, modified_block_bits)
        print("save_path:",save_path)
        model.save_quantized(save_path, use_safetensors=False)

if __name__ == "__main__":
    import argparse
    import os

    from mlwq.datautils import get_loaders

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "low_quant_method",
        type=str,
        choices=["2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="weight bit-width",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="set the device to use for quantization.",
    )
    parser.add_argument(
        "--disable_gptq",
        action="store_true",
        help="disable GPTQ for quantization.",
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Quant all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Quant all layers with id < this."
    )
    parser.add_argument(
        "--quant_only",
        type=str,
        default="",
        help="Quant only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )

    parser.add_argument(
        "--salient_block",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--nonsalient_block",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--quantizer_matric",
        type=str,
        default="mse",
        help="quantizer parameter determination metric",
    )

    parser.add_argument(
        "--lambda_salience",
        type=float,
        default=1,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default="",
        help="Test datasets",
    )
    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )
    parser.add_argument(
        "--eval_fp16_only", action="store_true", help="Skip quantization and only evaluate FP16 PPL."
    )
    parser.add_argument(
        "--bpll_loss",
        type=str,
        default="activation_dist",
        choices=["activation_dist", "weight_mse"],
        help="Loss used by BPLL when ranking bit-width choices.",
    )
    parser.add_argument(
        "--tqp_grid",
        type=list_of_floats,
        default=[1.0, 0.95, 0.9, 0.85],
        help="Comma-separated clipping ratios for TQP grid search.",
    )
    parser.add_argument(
        "--hessian_mode",
        type=str,
        default="diag",
        choices=["diag", "full"],
        help="Use diagonal Hessian approximation or full inverse-Hessian diagonal when small enough.",
    )
    parser.add_argument(
        "--hessian_full_threshold",
        type=int,
        default=1024,
        help="Maximum input dimension for full Hessian inverse in salience.",
    )
    parser.add_argument("--run_name", type=str, default="", help="Name used in metrics output.")
    parser.add_argument("--save_metrics", type=str, default="", help="Path to a JSON metrics file.")

    args = parser.parse_args()
   

   
    groupsize = args.groupsize
    net = args.model.split("/")[-1]
    block_configurations = f'../block_precision_{args.groupsize}_{args.low_quant_method}/{net}.pt'
    if args.eval_fp16_only:
        block_precision = None
    elif os.path.exists(block_configurations):
        block_precision = torch.load(block_configurations)
    else:
        print(f'Block precisions of {net} does not exist. Start aware!')
        block_precision = None
    

    device = args.device
  
    save_title = f"{args.model}_{args.dataset}_{args.low_quant_method}_{groupsize}"
    save_file = "./output/" + save_title.replace("/", "_") + ".pt"
    quantizers = {}
    mixed_block_precision = {}
    metrics = {
        "run_name": args.run_name,
        "model": args.model,
        "dataset": args.dataset,
        "low_quant_method": args.low_quant_method,
        "seed": args.seed,
        "nsamples": args.nsamples,
        "groupsize": args.groupsize,
        "device": args.device,
        "bpll_loss": args.bpll_loss,
        "tqp_grid": args.tqp_grid,
        "hessian_mode": args.hessian_mode,
        "hessian_full_threshold": args.hessian_full_threshold,
        "eval_fp16_only": args.eval_fp16_only,
        "quantized": False,
        "git_commit": current_git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    metrics.update(cuda_metrics_for_device(device))

    if args.load_quantized:
        model = get_model(save_file)
        model.eval()
    else:
        model = get_model(args.model)
        model.eval()
        if not args.eval_fp16_only:
            tick = time.time()
            dataloader, testloader = get_loaders(
                args.dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=model.seqlen,
            )
            quantizers, mixed_block_precision = quant_sequential(model, dataloader, device, block_precision)
            metrics["quantization_time_s"] = time.time() - tick
            metrics["quantized"] = True
            print("quantization time:", metrics["quantization_time_s"], "s")
    metrics["model_seqlen"] = getattr(model, "seqlen", None)
   
    
    for dataset in ["wikitext2"]:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, seqlen=model.seqlen, model=args.model
        )
        print(dataset)
        if "opt" in args.model:
            from mlwq.eval_ppl_utils import opt_eval

            metrics[f"{dataset}_ppl"] = opt_eval(model, testloader, device, dataset, args.log_wandb)
        elif "llama" in args.model or "Smo" in args.model:
            from mlwq.eval_ppl_utils import llama_eval

            metrics[f"{dataset}_ppl"] = llama_eval(model, testloader, device, dataset, args.log_wandb)
    if args.tasks != "":
        from mlwq.eval_ppl_utils import zeroshot_evaluate
        zeroshot_evaluate(model, args)

    if args.save:
      
        if args.pack:
            if not quantizers:
                raise RuntimeError("--pack requires quantization metadata from a fresh run")
            bits = int(args.low_quant_method[0])
            block_bits = block_precision if block_precision is not None else mixed_block_precision
            pack_model(model, save_file, bits, groupsize, quantizers, block_bits)
       
        else:
            save_path = os.path.dirname(save_file)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save_pretrained(save_file)

    if args.save_metrics:
        metrics_path = args.save_metrics
        metrics_dir = os.path.dirname(metrics_path)
        if metrics_dir and not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
