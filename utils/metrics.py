"""Mathematical utilities for entropy, perplexity, divergence and calibration."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


def shannon_entropy(probs: Tensor, base: float = 2.0) -> float:
    """Compute the Shannon entropy: H(p) = -sum_i p_i * log_b(p_i).

    Args:
        probs: Probability vector of shape [V] that sums to 1.
        base: Logarithm base (2.0 yields bits).

    Returns:
        Entropy as a Python float in units determined by ``base``.
    """
    support = probs[probs > 0]
    entropy_nats = -(support * support.log()).sum()
    return float(entropy_nats / math.log(base))


def perplexity_from_nll(total_nll: float, token_count: int) -> float:
    """Convert summed negative log-likelihood (nats) into perplexity: PPL = exp(NLL / N).

    Args:
        total_nll: Sum of per-token negative log-likelihoods in nats.
        token_count: Number of evaluated tokens N.

    Returns:
        Perplexity as a float.
    """
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return math.exp(total_nll / token_count)


def kl_divergence(p: Tensor, q: Tensor, eps: float = 1e-12) -> float:
    """Compute Kullback-Leibler divergence: D_KL(p || q) = sum_i p_i * log(p_i / q_i).

    Args:
        p: Reference distribution of shape [V].
        q: Approximating distribution of shape [V].
        eps: Lower clamp applied to q to prevent division by zero.

    Returns:
        KL divergence in nats as a float.
    """
    mask = p > 0
    return float((p[mask] * (p[mask].log() - q[mask].clamp_min(eps).log())).sum())


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error: ECE = sum_b (n_b / N) * |acc_b - conf_b|.

    Args:
        confidences: Top-1 predicted probabilities, length N.
        correct: Boolean match indicators, length N.
        n_bins: Number of equal-width calibration partitions on [0, 1].

    Returns:
        Tuple of (ece, mean confidence per bin, accuracy per bin, count per bin).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hits = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with interior edges maps confidence 1.0 into the last bin instead of an out-of-range bin.
    bin_ids = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bin_conf = np.zeros(n_bins)
    bin_acc = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    ece = 0.0
    for b in range(n_bins):
        members = bin_ids == b
        bin_count[b] = members.sum()
        if bin_count[b] == 0:
            continue
        bin_conf[b] = conf[members].mean()
        bin_acc[b] = hits[members].mean()
        ece += (bin_count[b] / len(conf)) * abs(bin_acc[b] - bin_conf[b])
    return float(ece), bin_conf, bin_acc, bin_count


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute the Wilson score interval for a binomial proportion.

    Args:
        successes: Number of positive outcomes.
        trials: Number of Bernoulli trials.
        z: Standard-normal quantile; 1.96 yields a 95% interval.

    Returns:
        Tuple (lower, upper) bounds on the underlying proportion.
    """
    if trials <= 0:
        return 0.0, 0.0
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)

