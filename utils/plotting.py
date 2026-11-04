"""Plotly figure builders for probability distributions, attention matrices and calibration."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RAW_COLOR = "#8A94A6"
ADJUSTED_COLOR = "#2F6FED"
SECONDARY_COLOR = "#E8743B"


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply a shared transparent layout, title and height to a figure."""
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=20, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def token_distribution_figure(
    tokens: Sequence[str], raw_probs: Sequence[float], adjusted_probs: Sequence[float], title: str
) -> go.Figure:
    """Grouped bars of raw (T=1, unfiltered) versus sampling-adjusted token probabilities.

    Args:
        tokens: Token labels, length n.
        raw_probs: Raw softmax probabilities, length n.
        adjusted_probs: Probabilities after temperature, top-k and top-p, length n.
        title: Figure title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    fig.add_bar(x=list(tokens), y=list(raw_probs), name="Raw model (T=1)", marker_color=RAW_COLOR)
    fig.add_bar(x=list(tokens), y=list(adjusted_probs), name="After sampling filters", marker_color=ADJUSTED_COLOR)
    fig.update_layout(barmode="group", xaxis_title="Candidate next token", yaxis_title="Probability")
    return _base_layout(fig, title)


def temperature_comparison_figure(
    tokens: Sequence[str], probs_low: Sequence[float], probs_high: Sequence[float], t_low: float, t_high: float
) -> go.Figure:
    """Two aligned panels showing the same candidate tokens at a low and a high temperature.

    Args:
        tokens: Token labels, length n.
        probs_low: Adjusted probabilities at t_low, length n.
        probs_high: Adjusted probabilities at t_high, length n.
        t_low: Low temperature value.
        t_high: High temperature value.

    Returns:
        Plotly figure with two columns sharing the y-axis.
    """
    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, subplot_titles=(f"Temperature = {t_low}", f"Temperature = {t_high}")
    )
    fig.add_bar(x=list(tokens), y=list(probs_low), marker_color=ADJUSTED_COLOR, name=f"T={t_low}", row=1, col=1)
    fig.add_bar(x=list(tokens), y=list(probs_high), marker_color=SECONDARY_COLOR, name=f"T={t_high}", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    return _base_layout(fig, "Next-token distribution: low versus high temperature")


def nucleus_mass_figure(sorted_probs: Sequence[float], top_p: float, max_rank: int = 100) -> go.Figure:
    """Cumulative probability mass over ranked tokens with the top-p threshold marked.

    Args:
        sorted_probs: Probabilities sorted in descending order.
        top_p: Nucleus threshold.
        max_rank: Number of ranks plotted.

    Returns:
        Plotly figure.
    """
    cumulative = np.cumsum(np.asarray(sorted_probs)[:max_rank])
    fig = go.Figure()
    fig.add_scatter(x=np.arange(1, len(cumulative) + 1), y=cumulative, mode="lines", line=dict(color=ADJUSTED_COLOR))
    fig.add_hline(y=top_p, line_dash="dash", line_color=SECONDARY_COLOR, annotation_text=f"top-p = {top_p:.2f}")
    fig.update_layout(xaxis_title="Token rank", yaxis_title="Cumulative probability", yaxis_range=[0, 1.02])
    return _base_layout(fig, "Cumulative probability mass (nucleus boundary)", height=320)


def attention_heatmap_figure(matrix: np.ndarray, tokens: List[str], title: str) -> go.Figure:
    """Heatmap of an attention matrix with query tokens on rows and key tokens on columns.

    Args:
        matrix: Array of shape [T, T] with rows summing to 1.
        tokens: Token labels, length T.
        title: Figure title.

    Returns:
        Plotly figure.
    """
    fig = go.Figure(
        go.Heatmap(z=matrix, x=tokens, y=tokens, colorscale="Blues", zmin=0.0, zmax=float(matrix.max()),
                   colorbar=dict(title="Weight"))
    )
    # Reversing the y-axis places the first query token at the top, matching the usual matrix orientation.
    fig.update_layout(xaxis_title="Key (attended-to) token", yaxis_title="Query token", yaxis_autorange="reversed")
    return _base_layout(fig, title, height=520)


def gpt_bert_comparison_figure(words: Sequence[str], gpt2_probs: Sequence[float], bert_probs: Sequence[float]) -> go.Figure:
    """Horizontal grouped bars contrasting GPT-2 and BERT probabilities for candidate words.

    Args:
        words: Candidate words, length n.
        gpt2_probs: GPT-2 probabilities, length n.
        bert_probs: BERT probabilities, length n.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    fig.add_bar(y=list(words), x=list(gpt2_probs), orientation="h", name="GPT-2 (causal)", marker_color=SECONDARY_COLOR)
    fig.add_bar(y=list(words), x=list(bert_probs), orientation="h", name="BERT (masked)", marker_color=ADJUSTED_COLOR)
    fig.update_layout(barmode="group", xaxis_title="Probability", yaxis_autorange="reversed")
    return _base_layout(fig, "Causal versus masked next-word probabilities", height=120 + 36 * len(words))


def perplexity_figure(labels: Sequence[str], values: Sequence[float]) -> go.Figure:
    """Bar chart of perplexity per model on a logarithmic axis.

    Args:
        labels: Model names.
        values: Perplexity values.

    Returns:
        Plotly figure.
    """
    fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=[SECONDARY_COLOR, ADJUSTED_COLOR][: len(labels)],
                           text=[f"{v:.1f}" for v in values], textposition="outside"))
    fig.update_layout(yaxis_title="Perplexity (log scale)", yaxis_type="log")
    return _base_layout(fig, "Perplexity on identical token blocks", height=360)


def calibration_figure(bin_conf: np.ndarray, bin_acc: np.ndarray, bin_count: np.ndarray) -> go.Figure:
    """Reliability diagram of top-1 confidence versus empirical accuracy.

    Args:
        bin_conf: Mean confidence per bin, shape [B].
        bin_acc: Accuracy per bin, shape [B].
        bin_count: Sample count per bin, shape [B].

    Returns:
        Plotly figure.
    """
    populated = bin_count > 0
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=RAW_COLOR), name="Perfect calibration")
    fig.add_scatter(x=bin_conf[populated], y=bin_acc[populated], mode="lines+markers",
                    line=dict(color=ADJUSTED_COLOR), name="GPT-2 top-1",
                    text=[f"n={int(c)}" for c in bin_count[populated]])
    fig.update_layout(xaxis_title="Mean predicted confidence", yaxis_title="Empirical accuracy")
    return _base_layout(fig, "Token probability calibration (reliability diagram)", height=380)


def error_vs_temperature_figure(
    temperatures: Sequence[float], error_rates: Sequence[float], hallucination_rates: Sequence[float]
) -> go.Figure:
    """Line chart of factual-error and hallucination rates against temperature.

    Args:
        temperatures: Temperature values, length n.
        error_rates: Factual error rates, length n.
        hallucination_rates: Hallucination rates, length n.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    fig.add_scatter(x=list(temperatures), y=list(error_rates), mode="lines+markers", name="Factual error rate",
                    line=dict(color=ADJUSTED_COLOR))
    fig.add_scatter(x=list(temperatures), y=list(hallucination_rates), mode="lines+markers", name="Hallucination rate",
                    line=dict(color=SECONDARY_COLOR))
    fig.update_layout(xaxis_title="Temperature", yaxis_title="Rate", yaxis_range=[0, 1.02])
    return _base_layout(fig, "Error versus temperature", height=360)

