import torch


class MLWQGPTQ:
    """Minimal GPTQ-compatible adapter used to make MLWQ reproducible.

    This implementation preserves the public methods used by run.py. It performs
    uniform grouped weight quantization and records enough metadata for smoke
    tests and small experiments. It is not a replacement for the paper's custom
    CUDA/AutoGPTQ packing path.
    """

    def __init__(self, layer, quantizer, bit_width=3):
        self.layer = layer
        self.quantizer = quantizer
        self.bit_width = int(bit_width)
        self.block_errors = []
        self.hessian_diag = None
        self.salience_labels = None

    @torch.no_grad()
    def add_batch(self, inp, out):
        inp = inp.reshape(-1, inp.shape[-1]).float()
        diag = inp.pow(2).mean(dim=0)
        if self.hessian_diag is None:
            self.hessian_diag = diag
        else:
            self.hessian_diag = 0.5 * (self.hessian_diag + diag)

        errors = []
        weight = self.layer.weight.data
        for bits in (2, 3):
            qweight, _, _, _ = self.quantizer.quantize(weight, bits=bits)
            errors.append(torch.mean((weight - qweight) ** 2).item())
        self.block_errors.append(errors)

    @torch.no_grad()
    def _compute_salience(self, high_count, low_count):
        weight = self.layer.weight.data.float()
        channel_score = weight.pow(2).mean(dim=0)
        if self.hessian_diag is not None and self.hessian_diag.numel() == channel_score.numel():
            channel_score = channel_score / self.hessian_diag.clamp(min=1e-8).pow(2)

        num_channels = channel_score.numel()
        high_count = min(max(int(high_count), 0), num_channels)
        low_count = min(max(int(low_count), 0), num_channels - high_count)
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
        qweight, scales, zeros, g_idx = self.quantizer.quantize(
            self.layer.weight.data,
            bits=bits,
            blocksize=blocksize,
        )
        self.layer.weight.data.copy_(qweight.to(self.layer.weight.dtype))

        if saved_block_precision is not None:
            block_precision = saved_block_precision
        elif self.salience_labels is not None:
            block_precision = self.salience_labels.detach().cpu().tolist()
        else:
            block_precision = [bits]

        return block_precision, scales.detach().cpu(), zeros.detach().cpu(), g_idx.detach().cpu()

    def free(self):
        self.hessian_diag = None
        self.salience_labels = None
