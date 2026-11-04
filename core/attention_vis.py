"""Extraction of multi-head self-attention tensors from GPT-2 via forward hooks."""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class AttentionExtractor:
    """Captures per-layer attention weights of a GPT-2 model."""

    def __init__(self, model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
        """Store the model and tokenizer.

        Args:
            model: GPT-2 model loaded with eager attention.
            tokenizer: Matching tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def extract(self, text: str) -> Tensor:
        """Return attention weights of shape [L, H, T, T] for the input text.

        Forward hooks on each ``transformer.h[i].attn`` module record the attention probability
        tensor of shape [1, H, T, T]; the built-in ``output_attentions`` tuple is the fallback.

        Args:
            text: Input prompt.

        Returns:
            Tensor [layers, heads, query positions, key positions]; each row sums to 1 over keys.
        """
        device = next(self.model.parameters()).device
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(device)
        captured: List[Tensor] = []

        def hook(_module: torch.nn.Module, _inputs: tuple, output: tuple) -> None:
            # The attention probabilities are the last element of the module output tuple when requested.
            candidate = output[-1]
            if isinstance(candidate, torch.Tensor) and candidate.dim() == 4:
                captured.append(candidate.detach().cpu())

        handles = [block.attn.register_forward_hook(hook) for block in self.model.transformer.h]
        try:
            outputs = self.model(input_ids, output_attentions=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) == len(self.model.transformer.h):
            # Stacking along a new leading axis yields [L, 1, H, T, T]; dropping the singleton batch axis gives [L, H, T, T].
            return torch.stack(captured, dim=0)[:, 0]
        return torch.stack([a.detach().cpu() for a in outputs.attentions], dim=0)[:, 0]

    def tokens(self, text: str) -> List[str]:
        """Return whitespace-visible token strings for the input text."""
        ids = self.tokenizer(text).input_ids
        return [self.tokenizer.decode([i]).replace("\n", "⏎").replace(" ", "▁") for i in ids]

    @staticmethod
    def select(attentions: Tensor, layer: int, head: int | None) -> Tensor:
        """Select one head's matrix [T, T], or the head-mean matrix when head is None.

        Args:
            attentions: Tensor [L, H, T, T].
            layer: Zero-based layer index.
            head: Zero-based head index, or None to average over the H axis.

        Returns:
            Attention matrix of shape [T, T].
        """
        layer_attention = attentions[layer]
        return layer_attention.mean(dim=0) if head is None else layer_attention[head]

    @staticmethod
    def attention_received(matrix: Tensor) -> Tensor:
        """Average attention each key position receives across query rows.

        Args:
            matrix: Attention matrix [T, T] with rows as queries.

        Returns:
            Vector [T] of column means.
        """
        return matrix.mean(dim=0)

