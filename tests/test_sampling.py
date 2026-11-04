"""Unit tests for tensor-level sampling operations, metrics and guardrail helpers."""
from __future__ import annotations

import math

import pytest
import torch

from core.gpt_engine import apply_temperature, apply_top_k, apply_top_p, filter_logits, sample_next_token
from core.guardrails import has_unsupported_specific, is_correct
from core.lstm_baseline import LSTMLanguageModel, evaluate_perplexity
from utils.metrics import expected_calibration_error, perplexity_from_nll, shannon_entropy, wilson_interval


def test_temperature_scales_logits() -> None:
    """Dividing by T=0.5 doubles logits and sharpens the softmax."""
    logits = torch.tensor([2.0, 1.0, 0.0])
    scaled = apply_temperature(logits, 0.5)
    assert torch.allclose(scaled, logits * 2)
    assert torch.softmax(scaled, -1)[0] > torch.softmax(logits, -1)[0]


def test_top_k_keeps_exactly_k() -> None:
    """Top-k masking leaves exactly k finite logits."""
    logits = torch.tensor([0.1, 3.0, 2.0, -1.0, 1.0])
    filtered = apply_top_k(logits, 2)
    assert torch.isfinite(filtered).sum().item() == 2
    assert torch.isfinite(filtered[1]) and torch.isfinite(filtered[2])


def test_top_p_keeps_minimal_nucleus() -> None:
    """With probabilities [0.5, 0.3, 0.15, 0.05] and p=0.7 the nucleus is the first two tokens."""
    probs = torch.tensor([0.05, 0.5, 0.15, 0.3])
    filtered = apply_top_p(probs.log(), 0.7)
    assert torch.isfinite(filtered).tolist() == [False, True, False, True]


def test_top_p_always_keeps_top_token() -> None:
    """A tiny threshold still retains the argmax token."""
    logits = torch.tensor([5.0, 1.0, 0.0])
    assert torch.isfinite(apply_top_p(logits, 1e-6)).sum().item() == 1


def test_sampling_is_reproducible_and_respects_mask() -> None:
    """Equal seeds give equal draws and masked tokens are never selected."""
    logits = torch.randn(50)
    first, probs = sample_next_token(logits, 1.0, 5, 1.0, torch.Generator().manual_seed(1))
    second, _ = sample_next_token(logits, 1.0, 5, 1.0, torch.Generator().manual_seed(1))
    assert first == second
    assert probs[first] > 0
    assert math.isclose(float(probs.sum()), 1.0, rel_tol=1e-5)
    assert (probs > 0).sum().item() == 5


def test_filter_logits_rejects_nonpositive_temperature() -> None:
    """Temperature must be strictly positive."""
    with pytest.raises(ValueError):
        filter_logits(torch.zeros(4), 0.0, 0, 1.0)


def test_entropy_of_uniform_distribution() -> None:
    """A uniform distribution over 8 outcomes has 3 bits of entropy."""
    assert math.isclose(shannon_entropy(torch.full((8,), 1 / 8)), 3.0, rel_tol=1e-6)


def test_perplexity_of_uniform_model() -> None:
    """Uniform prediction over V tokens yields perplexity V."""
    assert math.isclose(perplexity_from_nll(10 * math.log(50), 10), 50.0, rel_tol=1e-9)


def test_calibration_of_perfectly_calibrated_bins() -> None:
    """Confidence equal to accuracy in every bin yields zero ECE."""
    ece, _, _, counts = expected_calibration_error([1.0, 1.0, 0.5, 0.5], [True, True, True, False], n_bins=2)
    assert math.isclose(ece, 0.0, abs_tol=1e-9)
    assert counts.sum() == 4


def test_wilson_interval_bounds() -> None:
    """The interval brackets the point estimate and stays within [0, 1]."""
    low, high = wilson_interval(5, 20)
    assert 0.0 <= low < 0.25 < high <= 1.0


def test_answer_matching_uses_word_boundaries() -> None:
    """The symbol 'na' must not match inside 'nation'."""
    assert is_correct("Na", ("na",))
    assert not is_correct("nation", ("na",))
    assert is_correct("Brasília", ("brasilia",))


def test_unsupported_specific_detection() -> None:
    """A capitalised entity absent from the question is flagged; echoed words are not."""
    assert has_unsupported_specific("What is the capital of France?", "Lyon")
    assert not has_unsupported_specific("What is the capital of France?", "France")


def test_lstm_forward_shape_and_untrained_perplexity() -> None:
    """Logits have shape [B, T, V] and an untrained model scores near vocabulary-size perplexity."""
    model = LSTMLanguageModel(vocab_size=100, embed_dim=16, num_layers=1, dropout=0.0)
    assert model(torch.randint(0, 100, (2, 7))).shape == (2, 7, 100)
    ppl, count = evaluate_perplexity(model, torch.randint(0, 100, (400,)), block_size=64, device="cpu")
    assert count > 0 and 50 < ppl < 200

