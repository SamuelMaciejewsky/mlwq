import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mlwq.bpll import compute_bit_assignments, find_best_bitwidth_combination


class FakeGPTQ:
    def __init__(self, block_errors):
        self.block_errors = block_errors


def test_find_best_bitwidth_combination_respects_budget():
    errors = [
        [10.0, 1.0],
        [2.0, 1.5],
        [3.0, 2.0],
    ]

    combination, total_error = find_best_bitwidth_combination(
        errors,
        bit_options=(2, 3),
        target_total=8,
    )

    assert combination == (3, 2, 3)
    assert total_error == 5.0


def test_compute_bit_assignments_uses_average_block_errors():
    gptq = {
        "a": FakeGPTQ([[10.0, 1.0], [8.0, 1.0]]),
        "b": FakeGPTQ([[2.0, 1.5], [2.0, 1.5]]),
        "c": FakeGPTQ([[3.0, 2.0], [3.0, 2.0]]),
    }

    assignments, total_error = compute_bit_assignments(
        gptq,
        bit_options=(2, 3),
        target_total=8,
    )

    assert assignments == {"a": 3, "b": 2, "c": 3}
    assert total_error == 5.0


def test_find_best_bitwidth_combination_supports_non_contiguous_bits():
    errors = [
        [4.0, 1.0],
        [2.0, 3.0],
    ]

    combination, total_error = find_best_bitwidth_combination(
        errors,
        bit_options=(2, 4),
        target_total=6,
    )

    assert combination == (4, 2)
    assert total_error == 3.0


def test_compute_bit_assignments_supports_3_4_bit_candidates():
    gptq = {
        "a": FakeGPTQ([[4.0, 1.0], [4.0, 1.0]]),
        "b": FakeGPTQ([[2.0, 3.0], [2.0, 3.0]]),
    }

    assignments, total_error = compute_bit_assignments(
        gptq,
        bit_options=(3, 4),
        target_total=7,
    )

    assert assignments == {"a": 4, "b": 3}
    assert total_error == 3.0


def test_compute_bit_assignments_respects_weighted_budget():
    gptq = {
        "small": FakeGPTQ([[10.0, 0.1]]),
        "large": FakeGPTQ([[0.2, 10.0]]),
    }

    assignments, total_error = compute_bit_assignments(
        gptq,
        bit_options=(2, 3),
        target_total=11,
        layer_sizes={"small": 1, "large": 4},
    )

    assert assignments == {"small": 3, "large": 2}
    assert abs(total_error - 0.3) < 1e-12


if __name__ == "__main__":
    test_find_best_bitwidth_combination_respects_budget()
    test_compute_bit_assignments_uses_average_block_errors()
    test_find_best_bitwidth_combination_supports_non_contiguous_bits()
    test_compute_bit_assignments_supports_3_4_bit_candidates()
    test_compute_bit_assignments_respects_weighted_budget()
    print("bpll tests passed")
