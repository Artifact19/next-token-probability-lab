"""GPT-2 wrapper with manual temperature, top-k and top-p sampling operations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils.metrics import shannon_entropy

# A logit processor maps (token ids generated so far [T], next-token logits [V]) to modified logits [V].
LogitProcessor = Callable[[Tensor, Tensor], Tensor]


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits by inverse temperature 1/T for softmax normalization.

    Args:
        logits: Unnormalized logit scores of shape [V].
        temperature: Strictly positive temperature parameter T.

    Returns:
        Scaled logits of shape [V].
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive")
    return logits / temperature


def apply_top_k(logits: Tensor, top_k: int) -> Tensor:
    """Mask every logit below the k-th largest value with -inf.

    Args:
        logits: Scores of shape [V].
        top_k: Number of candidates to keep; values <= 0 or >= V disable the filter.

    Returns:
        Logits of shape [V] with non-top-k entries set to -inf.
    """
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    # topk returns values sorted descending, so index -1 is the k-th largest logit (the inclusion threshold).
    kth_value = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def apply_top_p(logits: Tensor, top_p: float) -> Tensor:
    """Keep the smallest set of highest-probability tokens whose cumulative mass reaches top_p.

    Args:
        logits: Scores of shape [V].
        top_p: Cumulative probability threshold in (0, 1]; 1.0 disables the filter.

    Returns:
        Logits of shape [V] with tokens outside the nucleus set to -inf.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_index = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Subtracting each token's own mass keeps the first token whose inclusion crosses the threshold.
    remove_sorted = (cumulative - sorted_probs) >= top_p
    # scatter inverts the sort permutation so the boolean mask aligns with the original vocabulary order.
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_index, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def filter_logits(logits: Tensor, temperature: float, top_k: int, top_p: float) -> Tensor:
    """Apply temperature scaling, then top-k masking, then top-p masking.

    Args:
        logits: Scores of shape [V].
        temperature: Positive temperature.
        top_k: Top-k size (<= 0 disables).
        top_p: Nucleus threshold in (0, 1].

    Returns:
        Filtered logits of shape [V].
    """
    return apply_top_p(apply_top_k(apply_temperature(logits, temperature), top_k), top_p)


def sample_next_token(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[int, Tensor]:
    """Draw one token id from the filtered, renormalised distribution.

    Args:
        logits: Raw scores of shape [V] on the CPU.
        temperature: Positive temperature.
        top_k: Top-k size (<= 0 disables).
        top_p: Nucleus threshold in (0, 1].
        generator: Optional seeded CPU generator for reproducible multinomial draws.

    Returns:
        Tuple of (sampled token id, adjusted probability vector of shape [V]).
    """
    probs = F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)
    token_id = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
    return token_id, probs


def format_token(text: str) -> str:
    """Render whitespace visibly: spaces become '▁' and newlines become '⏎'."""
    return text.replace("\n", "⏎").replace(" ", "▁")


@dataclass(frozen=True)
class SamplingStep:
    """Trace record of one autoregressive sampling step."""

    position: int
    token_id: int
    token_text: str
    probability: float
    kept_candidates: int
    entropy_bits: float


@dataclass(frozen=True)
class GenerationResult:
    """Output of an autoregressive generation run."""

    prompt: str
    completion: str
    token_ids: List[int]
    steps: List[SamplingStep]

    @property
    def text(self) -> str:
        """Prompt concatenated with the generated completion."""
        return self.prompt + self.completion


@dataclass(frozen=True)
class TokenDistribution:
    """Raw and adjusted next-token probabilities for the top-n raw candidates."""

    token_ids: List[int]
    tokens: List[str]
    raw_probs: List[float]
    adjusted_probs: List[float]
    raw_entropy_bits: float
    adjusted_entropy_bits: float
    kept_candidates: int


class GPT2Engine:
    """GPT-2 language model wrapper exposing logits, distributions and manual sampling."""

    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None) -> None:
        """Load tokenizer and model weights.

        Args:
            model_name: Hugging Face model identifier.
            device: Torch device string; defaults to CUDA when available, else CPU.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        # Eager attention is required because fused kernels do not materialise attention weight tensors.
        self.model = (
            GPT2LMHeadModel.from_pretrained(model_name, attn_implementation="eager").to(self.device).eval()
        )
        self.max_positions: int = int(self.model.config.n_positions)

    def encode(self, text: str) -> Tensor:
        """Tokenise text into a LongTensor of shape [1, T] on the engine device."""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def token_text(self, token_id: int) -> str:
        """Decode a single token id into a whitespace-visible string."""
        return format_token(self.tokenizer.decode([token_id]))

    @torch.no_grad()
    def next_token_logits(self, prompt: str) -> Tensor:
        """Return next-token logits of shape [V] (float32, CPU) for the final prompt position."""
        input_ids = self.encode(prompt)
        # logits has shape [1, T, V]; the last time step parameterises the next-token distribution.
        return self.model(input_ids).logits[0, -1, :].float().cpu()

    def next_token_probs(self, prompt: str) -> Tensor:
        """Return the softmax next-token distribution of shape [V] on the CPU."""
        return F.softmax(self.next_token_logits(prompt), dim=-1)

    def distribution(
        self, prompt: str, temperature: float, top_k: int, top_p: float, top_n: int = 20
    ) -> TokenDistribution:
        """Compute raw and sampling-adjusted probabilities for the top-n raw candidates.

        Args:
            prompt: Input text.
            temperature: Positive temperature.
            top_k: Top-k size (<= 0 disables).
            top_p: Nucleus threshold in (0, 1].
            top_n: Number of raw-probability-ranked tokens to report.

        Returns:
            TokenDistribution with entropies and kept-candidate count over the full vocabulary.
        """
        logits = self.next_token_logits(prompt)
        raw_probs = F.softmax(logits, dim=-1)
        adjusted_probs = F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)
        top = torch.topk(raw_probs, top_n)
        ids = top.indices.tolist()
        return TokenDistribution(
            token_ids=ids,
            tokens=[self.token_text(i) for i in ids],
            raw_probs=top.values.tolist(),
            adjusted_probs=adjusted_probs[top.indices].tolist(),
            raw_entropy_bits=shannon_entropy(raw_probs),
            adjusted_entropy_bits=shannon_entropy(adjusted_probs),
            kept_candidates=int((adjusted_probs > 0).sum().item()),
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        seed: int = 0,
        processors: Sequence[LogitProcessor] = (),
    ) -> GenerationResult:
        """Run the autoregressive sampling loop with a key-value cache.

        Args:
            prompt: Input text.
            max_new_tokens: Number of tokens to sample.
            temperature: Positive temperature.
            top_k: Top-k size (<= 0 disables).
            top_p: Nucleus threshold in (0, 1].
            seed: Seed of the CPU generator driving multinomial draws.
            processors: Logit processors applied to raw logits before temperature scaling.

        Returns:
            GenerationResult containing the completion and a per-step trace.
        """
        input_ids = self.encode(prompt)
        if input_ids.size(1) + max_new_tokens > self.max_positions:
            raise ValueError("prompt length plus max_new_tokens exceeds the model context window")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        outputs = self.model(input_ids=input_ids, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[0, -1, :].float().cpu()
        sequence = input_ids[0].cpu()
        new_ids: List[int] = []
        steps: List[SamplingStep] = []
        for position in range(max_new_tokens):
            for processor in processors:
                logits = processor(sequence, logits)
            token_id, probs = sample_next_token(logits, temperature, top_k, top_p, generator)
            steps.append(
                SamplingStep(
                    position=position,
                    token_id=token_id,
                    token_text=self.token_text(token_id),
                    probability=float(probs[token_id]),
                    kept_candidates=int((probs > 0).sum().item()),
                    entropy_bits=shannon_entropy(probs),
                )
            )
            new_ids.append(token_id)
            sequence = torch.cat([sequence, torch.tensor([token_id])])
            if token_id == self.tokenizer.eos_token_id:
                break
            # Feeding one token with the cached keys/values avoids recomputing attention over the whole prefix.
            outputs = self.model(
                input_ids=torch.tensor([[token_id]], device=self.device),
                past_key_values=cache,
                use_cache=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits[0, -1, :].float().cpu()
        return GenerationResult(
            prompt=prompt,
            completion=self.tokenizer.decode(new_ids),
            token_ids=new_ids,
            steps=steps,
        )

    @torch.no_grad()
    def perplexity(self, token_ids: Tensor, block_size: int = 128) -> Tuple[float, int]:
        """Compute perplexity over non-overlapping blocks of a token stream.

        Args:
            token_ids: LongTensor of shape [N].
            block_size: Context length of each evaluation block.

        Returns:
            Tuple of (perplexity, number of predicted tokens).
        """
        total_nll = 0.0
        count = 0
        for start in range(0, token_ids.numel() - 1, block_size):
            block = token_ids[start : start + block_size]
            if block.numel() < 2:
                continue
            # Logits at positions 0..T-2 predict targets at positions 1..T-1 (one-token shift).
            logits = self.model(block.unsqueeze(0).to(self.device)).logits[0, :-1].float()
            targets = block[1:].to(self.device)
            total_nll += float(F.cross_entropy(logits, targets, reduction="sum"))
            count += targets.numel()
        return float(np.exp(total_nll / count)), count

    @torch.no_grad()
    def top1_calibration(self, token_ids: Tensor, block_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """Collect top-1 confidence and correctness over non-overlapping blocks.

        Args:
            token_ids: LongTensor of shape [N].
            block_size: Context length of each evaluation block.

        Returns:
            Tuple of (confidences of shape [M], boolean correctness of shape [M]).
        """
        confidences: List[np.ndarray] = []
        correct: List[np.ndarray] = []
        for start in range(0, token_ids.numel() - 1, block_size):
            block = token_ids[start : start + block_size]
            if block.numel() < 2:
                continue
            logits = self.model(block.unsqueeze(0).to(self.device)).logits[0, :-1].float()
            probs = F.softmax(logits, dim=-1)
            top_prob, top_index = probs.max(dim=-1)
            confidences.append(top_prob.cpu().numpy())
            correct.append((top_index.cpu() == block[1:]).numpy())
        return np.concatenate(confidences), np.concatenate(correct)

