"""Learning-rate formulas shared by iteration-based training and CPU tests."""

import math


def cosine_learning_rate_factor(
    step: int,
    *,
    total_iters: int,
    warmup_iters: int = 0,
    min_factor: float = 0.0,
) -> float:
    """Return a multiplier for the optimizer's initial learning rate.

    ``step`` is the number of completed training batches. During warmup,
    the first batch uses 1 / warmup_iters of the initial learning rate.
    Cosine decay starts at the initial rate after warmup and reaches
    ``min_factor`` at total_iters. Values beyond the budget stay at the floor.
    Each parameter group keeps its initial learning-rate ratio.
    """
    if total_iters <= 0:
        raise ValueError('total_iters must be positive')
    if not 0 <= warmup_iters < total_iters:
        raise ValueError('Require 0 <= warmup_iters < total_iters')
    if step < 0:
        raise ValueError('step must be nonnegative')
    if not 0.0 <= min_factor <= 1.0:
        raise ValueError('min_factor must be between 0 and 1')

    if step < warmup_iters:
        return (step + 1) / warmup_iters

    decay_iters = total_iters - warmup_iters
    progress = min(1.0, (step - warmup_iters) / decay_iters)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine_factor
