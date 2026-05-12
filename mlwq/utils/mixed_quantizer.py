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
    def quantize(self, weight=None, bits=None, blocksize=None, clip_ratio=1.0):
        weight = self.weight if weight is None else weight
        bits = self.bits if bits is None else int(bits)
        blocksize = self.groupsize if blocksize is None else blocksize
        clip_ratio = float(clip_ratio)

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
                if clip_ratio < 1.0:
                    center = (minimum + maximum) * 0.5
                    half_range = (maximum - minimum) * 0.5 * clip_ratio
                    minimum = center - half_range
                    maximum = center + half_range
                    chunk = chunk.clamp(minimum, maximum)
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

    @torch.no_grad()
    def quantize_mixed(
        self,
        weight,
        channel_labels,
        bit_map,
        blocksize=None,
        clip_ratios=None,
    ):
        """Quantize input channels with label-specific bit widths."""
        blocksize = self.groupsize if blocksize is None else blocksize
        clip_ratios = clip_ratios or {}
        qweight = torch.empty_like(weight)
        scales = {}
        zeros = {}

        if channel_labels is None:
            qweight, scale, zero, g_idx = self.quantize(
                weight,
                bits=self.bits,
                blocksize=blocksize,
            )
            return qweight, scale, zero, g_idx

        channel_labels = channel_labels.to(weight.device)
        missing_labels = sorted(
            set(int(label) for label in channel_labels.detach().cpu().tolist())
            - set(int(label) for label in bit_map)
        )
        if missing_labels:
            raise ValueError(f"bit_map is missing channel labels: {missing_labels}")

        for label, bits in bit_map.items():
            mask = channel_labels == int(label)
            if not mask.any():
                continue
            subset = weight[:, mask]
            qsubset, scale, zero, _ = self.quantize(
                subset,
                bits=int(bits),
                blocksize=blocksize,
                clip_ratio=clip_ratios.get(int(label), 1.0),
            )
            qweight[:, mask] = qsubset
            scales[int(label)] = scale
            zeros[int(label)] = zero

        g_idx = channel_labels.to(dtype=torch.int32)
        return qweight, scales, zeros, g_idx
