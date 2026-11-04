"""Lightweight LSTM language model baseline with a perplexity evaluation harness."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LSTMLanguageModel(nn.Module):
    """Two-layer LSTM language model with tied input and output embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int = 256, num_layers: int = 2, dropout: float = 0.2) -> None:
        """Build the network.

        Args:
            vocab_size: Vocabulary size V (shared with the GPT-2 BPE tokenizer for comparable perplexity).
            embed_dim: Embedding and hidden width.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability on embeddings, inter-layer and output activations.
        """
        super().__init__()
        self.config: Dict[str, float] = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_layers": num_layers,
            "dropout": dropout,
        }
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying shares one [V, D] matrix between input embeddings and output projection.
        self.decoder.weight = self.embedding.weight

    def forward(self, tokens: Tensor) -> Tensor:
        """Compute next-token logits.

        Args:
            tokens: LongTensor of shape [B, T].

        Returns:
            Logits of shape [B, T, V].
        """
        embedded = self.dropout(self.embedding(tokens))
        hidden, _ = self.lstm(embedded)
        return self.decoder(self.dropout(hidden))


def load_wikitext(split: str = "test", max_chars: int = 200_000) -> str:
    """Load a WikiText-2 (raw) split as a single string truncated to max_chars.

    Args:
        split: One of 'train', 'validation', 'test'.
        max_chars: Maximum number of characters returned.

    Returns:
        Concatenated non-empty lines.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(line for line in dataset["text"] if line.strip())
    return text[:max_chars]


def encode_corpus(tokenizer, text: str) -> Tensor:
    """Tokenise text into a flat LongTensor of shape [N] using the supplied tokenizer."""
    return torch.tensor(tokenizer(text).input_ids, dtype=torch.long)


def train_lstm(
    model: LSTMLanguageModel,
    train_ids: Tensor,
    block_size: int = 128,
    batch_size: int = 8,
    steps: int = 300,
    learning_rate: float = 2e-3,
    seed: int = 0,
    device: Optional[str] = None,
    progress: Optional[Callable[[int, float], None]] = None,
) -> List[float]:
    """Train the LSTM with next-token cross-entropy on randomly sampled contiguous blocks.

    Args:
        model: Model to optimise in place.
        train_ids: LongTensor of shape [N] with N > block_size + 1.
        block_size: Sequence length T of each training block.
        batch_size: Number of blocks B per step.
        steps: Number of optimiser steps.
        learning_rate: AdamW learning rate.
        seed: Seed for block sampling.
        device: Torch device string; defaults to CUDA when available, else CPU.
        progress: Optional callback receiving (step, loss).

    Returns:
        List of per-step training losses (nats per token).
    """
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(run_device).train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    max_start = train_ids.numel() - block_size - 1
    losses: List[float] = []
    for step in range(steps):
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        # Inputs are tokens [s, s+T) and targets are tokens [s+1, s+T+1): a one-position shift.
        inputs = torch.stack([train_ids[s : s + block_size] for s in starts.tolist()]).to(run_device)
        targets = torch.stack([train_ids[s + 1 : s + block_size + 1] for s in starts.tolist()]).to(run_device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        # Global-norm clipping bounds the update magnitude against exploding recurrent gradients.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        losses.append(float(loss))
        if progress is not None:
            progress(step + 1, float(loss))
    model.eval()
    return losses


@torch.no_grad()
def evaluate_perplexity(
    model: LSTMLanguageModel, token_ids: Tensor, block_size: int = 128, device: Optional[str] = None
) -> Tuple[float, int]:
    """Compute perplexity over non-overlapping blocks, identical in protocol to GPT2Engine.perplexity.

    Args:
        model: Trained LSTM language model.
        token_ids: LongTensor of shape [N].
        block_size: Context length of each evaluation block.
        device: Torch device string; defaults to CUDA when available, else CPU.

    Returns:
        Tuple of (perplexity, number of predicted tokens).
    """
    run_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(run_device).eval()
    total_nll = 0.0
    count = 0
    for start in range(0, token_ids.numel() - 1, block_size):
        block = token_ids[start : start + block_size]
        if block.numel() < 2:
            continue
        logits = model(block[:-1].unsqueeze(0).to(run_device))[0]
        targets = block[1:].to(run_device)
        total_nll += float(F.cross_entropy(logits, targets, reduction="sum"))
        count += targets.numel()
    return math.exp(total_nll / count), count


def save_checkpoint(model: LSTMLanguageModel, path: str) -> None:
    """Serialise model configuration and weights to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str, device: str = "cpu") -> LSTMLanguageModel:
    """Restore an LSTMLanguageModel from a checkpoint written by ``save_checkpoint``."""
    payload = torch.load(path, map_location=device, weights_only=True)
    config = payload["config"]
    model = LSTMLanguageModel(
        vocab_size=int(config["vocab_size"]),
        embed_dim=int(config["embed_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()

