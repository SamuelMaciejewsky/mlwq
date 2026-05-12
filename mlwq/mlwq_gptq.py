import torch
import torch.nn.functional as F

from mlwq.utils.reconstruct import channel_wise_distribution_loss


class MLWQGPTQ:
    """Minimal GPTQ-compatible adapter used to make MLWQ reproducible.

    This implementation preserves the public methods used by run.py. It performs
    uniform grouped weight quantization and records enough metadata for smoke
    tests and small experiments. It is not a replacement for the paper's custom
    CUDA/AutoGPTQ packing path.
    """

    def __init__(
        self,
        layer,
        quantizer,
        bit_width=3,
        bpll_loss="activation_dist",
        tqp_grid=None,
        hessian_mode="diag",
        hessian_full_threshold=1024,
        bit_options=(2, 3),
    ):
        self.layer = layer
        self.quantizer = quantizer
        self.bit_width = int(bit_width)
        self.bpll_loss = bpll_loss
        self.tqp_grid = tqp_grid or (1.0, 0.95, 0.9, 0.85)
        self.hessian_mode = hessian_mode
        self.hessian_full_threshold = int(hessian_full_threshold)
        self.bit_options = tuple(sorted(set(int(bit) for bit in bit_options)))
        self.block_errors = []
        self.hessian_diag = None
        self.hessian = None
        self.salience_labels = None
        self._quantized_weight_cache = {}

    @torch.no_grad()
    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1]).float()
        diag = inp.pow(2).mean(dim=0).clamp(min=1e-8)
        if self.hessian_diag is None:
            self.hessian_diag = diag
        else:
            self.hessian_diag = 0.5 * (self.hessian_diag + diag)

        if (
            self.hessian_mode == "full"
            and inp.shape[-1] <= self.hessian_full_threshold
        ):
            hessian = inp.t().matmul(inp) / max(inp.shape[0], 1)
            if self.hessian is None:
                self.hessian = hessian
            else:
                self.hessian = 0.5 * (self.hessian + hessian)

        errors = []
        for bits in self.bit_options:
            qweight = self._quantized_weight(bits)
            errors.append(self._quantization_loss(inp, out, qweight).item())
        self.block_errors.append(errors)

    @torch.no_grad()
    def _quantized_weight(self, bits):
        bits = int(bits)
        if bits not in self._quantized_weight_cache:
            qweight, _, _, _ = self.quantizer.quantize(self.layer.weight.data, bits=bits)
            self._quantized_weight_cache[bits] = qweight
        return self._quantized_weight_cache[bits]

    @torch.no_grad()
    def _quantization_loss(self, inp, out, qweight):
        if self.bpll_loss == "weight_mse":
            return torch.mean((self.layer.weight.data.float() - qweight.float()) ** 2)
        if self.layer.weight.data.dim() == 2 and hasattr(self.layer, "bias"):
            bias = self.layer.bias.float() if self.layer.bias is not None else None
            qout = F.linear(inp, qweight.float(), bias)
            return channel_wise_distribution_loss(qout, out.reshape_as(qout).float())
        return torch.mean((self.layer.weight.data.float() - qweight.float()) ** 2)

    @torch.no_grad()
    def _inverse_hessian_diag(self):
        if self.hessian is not None:
            damp = self.hessian.diag().mean().clamp(min=1e-8) * 0.01
            eye = torch.eye(self.hessian.shape[0], device=self.hessian.device)
            try:
                chol = torch.linalg.cholesky(self.hessian + damp * eye)
                inv_hessian = torch.cholesky_inverse(chol)
                return inv_hessian.diag().clamp(min=1e-8)
            except torch.linalg.LinAlgError:
                pass
        if self.hessian_diag is not None:
            return (1.0 / self.hessian_diag).clamp(min=1e-8)
        return None

    @torch.no_grad()
    def _compute_salience(self, high_ratio, low_ratio):
        weight = self.layer.weight.data.float()
        inv_hessian_diag = self._inverse_hessian_diag()
        if inv_hessian_diag is not None and inv_hessian_diag.numel() == weight.shape[1]:
            channel_score = weight.pow(2).sum(dim=0) / inv_hessian_diag.pow(2)
        else:
            channel_score = weight.pow(2).sum(dim=0)

        num_channels = channel_score.numel()
        high_count = min(max(int(round(num_channels * float(high_ratio))), 0), num_channels)
        low_count = min(max(int(round(num_channels * float(low_ratio))), 0), num_channels - high_count)
        labels = torch.full((num_channels,), 1, device=weight.device, dtype=torch.int32)

        if low_count:
            labels[torch.topk(channel_score, low_count, largest=False).indices] = 0
        if high_count:
            labels[torch.topk(channel_score, high_count, largest=True).indices] = 2
        self.salience_labels = labels
        return labels

    def get_salience1(self, name, layer_index, high_count, low_count, blocksize=128):
        return self._compute_salience(high_count, low_count)

    def get_salience(self, name, layer_index, high_count, blocksize=128):
        return self._compute_salience(high_count, high_count)

    @staticmethod
    def bit_map_for_layer(base_bits):
        base_bits = int(base_bits)
        if base_bits <= 2:
            return {0: 2, 1: 2, 2: 3}
        if base_bits == 3:
            return {0: 2, 1: 3, 2: 4}
        return {0: max(base_bits - 1, 2), 1: base_bits, 2: min(base_bits + 1, 8)}

    @torch.no_grad()
    def _tqp_clip_ratios(self, bit_map, blocksize):
        if self.salience_labels is None:
            return {}

        weight = self.layer.weight.data
        clip_ratios = {}
        for label, bits in bit_map.items():
            mask = self.salience_labels.to(weight.device) == int(label)
            if not mask.any():
                continue
            subset = weight[:, mask]
            best_ratio = 1.0
            best_error = float("inf")
            for ratio in self.tqp_grid:
                qsubset, _, _, _ = self.quantizer.quantize(
                    subset,
                    bits=bits,
                    blocksize=blocksize,
                    clip_ratio=ratio,
                )
                error = torch.mean((subset.float() - qsubset.float()) ** 2).item()
                if error < best_error:
                    best_error = error
                    best_ratio = float(ratio)
            clip_ratios[int(label)] = best_ratio
        return clip_ratios

    @torch.no_grad()
    def fasterquant1(
        self,
        percdamp=0.01,
        blocksize=128,
        layer_name="",
        best_combination=None,
        saved_block_precision=None,
    ):
        bits = int(best_combination or self.bit_width)
        bit_map = self.bit_map_for_layer(bits)
        clip_ratios = self._tqp_clip_ratios(bit_map, blocksize)
        qweight, scales, zeros, g_idx = self.quantizer.quantize_mixed(
            self.layer.weight.data,
            channel_labels=self.salience_labels,
            bit_map=bit_map,
            blocksize=blocksize,
            clip_ratios=clip_ratios,
        )
        self.layer.weight.data.copy_(qweight.to(self.layer.weight.dtype))

        if saved_block_precision is not None:
            block_precision = saved_block_precision
        elif self.salience_labels is not None:
            block_precision = self.salience_labels.detach().cpu().tolist()
        else:
            block_precision = [bits]

        if isinstance(scales, dict):
            scales = {key: value.detach().cpu() for key, value in scales.items()}
            zeros = {key: value.detach().cpu() for key, value in zeros.items()}
        else:
            scales = scales.detach().cpu()
            zeros = zeros.detach().cpu()

        return block_precision, scales, zeros, g_idx.detach().cpu()

    def free(self):
        self.hessian_diag = None
        self.hessian = None
        self.salience_labels = None
        self._quantized_weight_cache = {}
