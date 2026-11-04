"""BERT masked language model inference and alignment with GPT-2 next-token predictions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import BertForMaskedLM, BertTokenizerFast

from core.gpt_engine import GPT2Engine

MASK_TOKEN = "[MASK]"


@dataclass(frozen=True)
class AlignedRow:
    """Probability of one candidate word under both models."""

    word: str
    gpt2_prob: float
    bert_prob: float


@dataclass(frozen=True)
class AlignedComparison:
    """Side-by-side GPT-2 and BERT predictions for one masked position."""

    gpt2_top: List[Tuple[str, float]]
    bert_top: List[Tuple[str, float]]
    rows: List[AlignedRow]
    jaccard_overlap: float


class BertEngine:
    """bert-base-uncased masked language model wrapper."""

    def __init__(self, model_name: str = "bert-base-uncased", device: Optional[str] = None) -> None:
        """Load tokenizer and masked-LM weights.

        Args:
            model_name: Hugging Face model identifier.
            device: Torch device string; defaults to CUDA when available, else CPU.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def mask_probs(self, text: str) -> Tensor:
        """Return the softmax distribution of shape [V] at the single [MASK] position.

        Args:
            text: Sentence containing exactly one literal [MASK] token.

        Returns:
            Probability vector of shape [V] on the CPU.
        """
        if text.count(MASK_TOKEN) != 1:
            raise ValueError("text must contain exactly one [MASK] token")
        encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
        mask_index = (encoded.input_ids[0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        # logits has shape [1, T, V]; selecting the mask position gives a [V] vector conditioned on both-side context.
        logits = self.model(**encoded).logits[0, mask_index[0], :].float().cpu()
        return F.softmax(logits, dim=-1)

    def top_words(self, probs: Tensor, top_n: int) -> List[Tuple[str, float]]:
        """Return the top-n whole-word BERT predictions, skipping '##' sub-word pieces.

        Args:
            probs: Probability vector of shape [V].
            top_n: Number of entries to return.

        Returns:
            List of (token, probability) sorted by descending probability.
        """
        ranked = torch.topk(probs, min(probs.numel(), top_n * 8))
        words: List[Tuple[str, float]] = []
        for token_id, prob in zip(ranked.indices.tolist(), ranked.values.tolist()):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token.startswith("##") or token.startswith("["):
                continue
            words.append((token, prob))
            if len(words) == top_n:
                break
        return words

    def compare_with_gpt2(
        self, masked_text: str, gpt2: GPT2Engine, top_n: int = 10, use_right_context: bool = True
    ) -> AlignedComparison:
        """Align BERT's [MASK] prediction with GPT-2's next-token prediction.

        GPT-2 receives only the text left of [MASK] (causal conditioning); BERT receives the full
        sentence when ``use_right_context`` is True (bidirectional conditioning).

        Args:
            masked_text: Sentence with exactly one [MASK] token.
            gpt2: Loaded GPT2Engine.
            top_n: Number of candidates per model.
            use_right_context: Keep text right of [MASK] in the BERT input.

        Returns:
            AlignedComparison with per-model top lists, a union table and top-n Jaccard overlap.
        """
        left, right = masked_text.split(MASK_TOKEN)
        bert_input = masked_text if use_right_context else f"{left}{MASK_TOKEN}"
        bert_probs = self.mask_probs(bert_input)
        gpt_probs = gpt2.next_token_probs(left.rstrip())

        bert_top = self.top_words(bert_probs, top_n)
        gpt_ranked = torch.topk(gpt_probs, top_n * 4)
        gpt_top: List[Tuple[str, float]] = []
        for token_id, prob in zip(gpt_ranked.indices.tolist(), gpt_ranked.values.tolist()):
            word = gpt2.tokenizer.decode([token_id]).strip().lower()
            # Only alphabetic, space-initial tokens correspond to whole words comparable with BERT vocabulary entries.
            if word.isalpha() and gpt2.tokenizer.decode([token_id]).startswith(" "):
                gpt_top.append((word, prob))
            if len(gpt_top) == top_n:
                break

        vocabulary = {w for w, _ in bert_top} | {w for w, _ in gpt_top}
        rows = [
            AlignedRow(word=w, gpt2_prob=self._gpt2_word_prob(w, gpt_probs, gpt2), bert_prob=self._bert_word_prob(w, bert_probs))
            for w in vocabulary
        ]
        rows.sort(key=lambda r: max(r.gpt2_prob, r.bert_prob), reverse=True)
        bert_set = {w for w, _ in bert_top}
        gpt_set = {w for w, _ in gpt_top}
        union = bert_set | gpt_set
        overlap = len(bert_set & gpt_set) / len(union) if union else 0.0
        return AlignedComparison(gpt2_top=gpt_top, bert_top=bert_top, rows=rows, jaccard_overlap=overlap)

    def _bert_word_prob(self, word: str, probs: Tensor) -> float:
        """Probability of a whole-word BERT vocabulary entry; 0.0 if the word is not a single token."""
        token_id = self.tokenizer.convert_tokens_to_ids(word)
        if token_id == self.tokenizer.unk_token_id:
            return 0.0
        return float(probs[token_id])

    @staticmethod
    def _gpt2_word_prob(word: str, probs: Tensor, gpt2: GPT2Engine) -> float:
        """Probability of the first BPE token of ' word'; a lower bound for multi-token words."""
        first_token = gpt2.tokenizer.encode(" " + word)[0]
        return float(probs[first_token])

