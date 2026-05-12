import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

torch = pytest.importorskip("torch")

from mlwq.mlwq_gptq import MLWQGPTQ
from mlwq.utils.mixed_quantizer import Quantizer


def test_quantize_mixed_applies_label_specific_bit_widths():
    weight = torch.tensor([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
    labels = torch.tensor([0, 1, 2], dtype=torch.int32)
    quantizer = Quantizer(weight, method="3bit", groupsize=8)

    qweight, scales, zeros, g_idx = quantizer.quantize_mixed(
        weight,
        channel_labels=labels,
        bit_map={0: 2, 1: 3, 2: 4},
    )

    assert qweight.shape == weight.shape
    assert set(scales) == {0, 1, 2}
    assert set(zeros) == {0, 1, 2}
    assert torch.equal(g_idx.cpu(), labels)


def test_quantize_mixed_rejects_missing_label_mapping():
    weight = torch.tensor([[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]])
    labels = torch.tensor([0, 1, 2], dtype=torch.int32)
    quantizer = Quantizer(weight, method="3bit", groupsize=8)

    with pytest.raises(ValueError, match="missing channel labels"):
        quantizer.quantize_mixed(
            weight,
            channel_labels=labels,
            bit_map={0: 2, 1: 3},
        )


def test_tqp_grid_never_selects_unknown_label():
    layer = torch.nn.Linear(3, 2, bias=False)
    quantizer = Quantizer(layer.weight, method="3bit", groupsize=8)
    gptq = MLWQGPTQ(layer, quantizer, tqp_grid=[1.0, 0.9])
    gptq.salience_labels = torch.tensor([0, 1, 2], dtype=torch.int32)

    ratios = gptq._tqp_clip_ratios({0: 2, 1: 3, 2: 4}, blocksize=8)

    assert set(ratios) == {0, 1, 2}
    assert all(value in {1.0, 0.9} for value in ratios.values())
