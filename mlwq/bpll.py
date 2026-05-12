import itertools


def find_best_bitwidth_combination(average_errors, bit_options, target_total):
    """Return the minimum-error bit allocation under a total bit budget."""
    if not average_errors:
        return (), 0.0

    bit_options = tuple(sorted(set(int(bit) for bit in bit_options)))
    bit_index = {bit: index for index, bit in enumerate(bit_options)}
    num_sublayers = len(average_errors)
    best_combination = None
    min_total_error = float("inf")

    for combination in itertools.product(bit_options, repeat=num_sublayers):
        if sum(combination) != target_total:
            continue

        total_error = 0.0
        feasible = True
        for index, bit_width in enumerate(combination):
            try:
                total_error += float(average_errors[index][bit_index[bit_width]])
            except IndexError:
                feasible = False
                break

        if feasible and total_error < min_total_error:
            min_total_error = total_error
            best_combination = combination

    if best_combination is None:
        raise ValueError(
            "no feasible bit-width combination for "
            f"{num_sublayers} layers, bits={bit_options}, target_total={target_total}"
        )

    return best_combination, min_total_error


def compute_bit_assignments(gptq, bit_options=(2, 3), target_total=None):
    """Compute BPLL bit assignments from each sublayer's block_errors."""
    names = list(gptq)
    if target_total is None:
        target_total = 3 * len(names) - 1

    average_errors = []
    for name in names:
        sublayer = gptq[name]
        if not sublayer.block_errors:
            raise ValueError(f"sublayer {name!r} has no block_errors")
        average_errors.append(
            [sum(values) / len(values) for values in zip(*sublayer.block_errors)]
        )

    best_combination, min_total_error = find_best_bitwidth_combination(
        average_errors,
        bit_options=bit_options,
        target_total=target_total,
    )
    return dict(zip(names, best_combination)), min_total_error
