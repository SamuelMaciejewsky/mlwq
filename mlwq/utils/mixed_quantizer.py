import torch


class Quantizer:
    """Small uniform weight quantizer compatible with the MLWQ script API."""

    def __init__(
        self,
        weight,
        method="3bit",
        groupsize=128,
        metric="mse",
        lambda_salience=1.0,
    ):
        self.weight = weight
        self.bits = int(str(method)[0])
        self.groupsize = groupsize
        self.metric = metric
        self.lambda_salience = lambda_salience

    @torch.no_grad()
    def quantize(self, weight=None, bits=None, blocksize=None):
        weight = self.weight if weight is None else weight
        bits = self.bits if bits is None else int(bits)
        blocksize = self.groupsize if blocksize is None else blocksize

        original_shape = weight.shape
        flat = weight.reshape(weight.shape[0], -1)
        qweight = torch.empty_like(flat)
        scale_chunks = []
        zero_chunks = []
        group_indices = []

        levels = (1 << bits) - 1
        for row_index in range(flat.shape[0]):
            row = flat[row_index]
            row_scales = []
            row_zeros = []
            for start in range(0, row.numel(), blocksize):
                end = min(start + blocksize, row.numel())
                chunk = row[start:end]
                minimum = chunk.min()
                maximum = chunk.max()
                scale = (maximum - minimum).clamp(min=1e-8) / levels
                zero = torch.round(-minimum / scale).clamp(0, levels)
                q = torch.round(chunk / scale + zero).clamp(0, levels)
                qweight[row_index, start:end] = (q - zero) * scale
                row_scales.append(scale)
                row_zeros.append(zero)
                group_indices.extend([len(row_scales) - 1] * (end - start))
            scale_chunks.append(torch.stack(row_scales))
            zero_chunks.append(torch.stack(row_zeros))

        return (
            qweight.reshape(original_shape),
            torch.stack(scale_chunks),
            torch.stack(zero_chunks),
            torch.tensor(group_indices, device=weight.device, dtype=torch.int32),
        )
