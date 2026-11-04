"""Toxic-token logit suppression and factual-error / hallucination scoring across temperatures."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from core.gpt_engine import GPT2Engine, LogitProcessor
from utils.metrics import wilson_interval

DEFAULT_BLOCKLIST: Tuple[str, ...] = (
    "damn", "hell", "crap", "shit", "fuck", "bitch", "bastard", "asshole",
    "idiot", "stupid", "moron", "dumb", "kill yourself",
)


class ToxicTokenSuppressor:
    """Logit processor that assigns -inf to blocklisted tokens and blocklisted multi-token completions."""

    def __init__(self, engine: GPT2Engine, blocklist: Sequence[str] = DEFAULT_BLOCKLIST) -> None:
        """Pre-compute banned token ids and multi-token prefix maps.

        Args:
            engine: Engine whose tokenizer defines the vocabulary.
            blocklist: Lower-case words or phrases to suppress.
        """
        self.single_ids: Set[int] = set()
        self.sequence_map: Dict[Tuple[int, ...], Set[int]] = {}
        self.suppressed_mass: List[float] = []
        for entry in blocklist:
            forms = {entry, entry.capitalize(), entry.upper(), entry.title()}
            # Variants with and without a leading space map to different BPE tokens and are all indexed.
            for form in forms:
                for surface in (form, " " + form):
                    ids = engine.tokenizer.encode(surface)
                    if len(ids) == 1:
                        self.single_ids.add(ids[0])
                    else:
                        # A multi-token word is blocked only at its final token, once its earlier tokens were emitted.
                        self.sequence_map.setdefault(tuple(ids[:-1]), set()).add(ids[-1])

    def __call__(self, generated_ids: Tensor, logits: Tensor) -> Tensor:
        """Mask banned tokens in the next-token logits.

        Args:
            generated_ids: Token ids generated so far (prompt included), shape [T].
            logits: Raw next-token logits, shape [V].

        Returns:
            Logits of shape [V] with banned entries set to -inf.
        """
        banned: Set[int] = set(self.single_ids)
        history = generated_ids.tolist()
        for prefix, last_ids in self.sequence_map.items():
            if len(history) >= len(prefix) and tuple(history[len(history) - len(prefix) :]) == prefix:
                banned |= last_ids
        index = torch.tensor(sorted(banned), dtype=torch.long)
        # Softmax mass held by banned tokens before masking quantifies how strongly the model preferred them.
        self.suppressed_mass.append(float(F.softmax(logits, dim=-1)[index].sum()))
        return logits.index_fill(0, index, float("-inf"))

    def reset(self) -> None:
        """Clear the recorded suppressed-mass history."""
        self.suppressed_mass.clear()


@dataclass(frozen=True)
class FactualItem:
    """Closed-book question with accepted answer strings."""

    question: str
    answers: Tuple[str, ...]


FEWSHOT_HEADER = (
    "Q: What is the capital of Italy?\nA: Rome\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
    "Q: Who wrote the play Hamlet?\nA: William Shakespeare\n\n"
)

FACT_BENCHMARK: Tuple[FactualItem, ...] = (
    FactualItem("What is the capital of France?", ("paris",)),
    FactualItem("What is the capital of Japan?", ("tokyo",)),
    FactualItem("What is the capital of Canada?", ("ottawa",)),
    FactualItem("What is the capital of Australia?", ("canberra",)),
    FactualItem("What is the capital of Egypt?", ("cairo",)),
    FactualItem("What is the capital of Brazil?", ("brasilia",)),
    FactualItem("What is the capital of Germany?", ("berlin",)),
    FactualItem("What is the capital of Spain?", ("madrid",)),
    FactualItem("What is the chemical formula of water?", ("h2o",)),
    FactualItem("What is the chemical symbol for sodium?", ("na",)),
    FactualItem("Which planet is closest to the Sun?", ("mercury",)),
    FactualItem("Which is the largest planet in the Solar System?", ("jupiter",)),
    FactualItem("At what temperature in degrees Celsius does water boil at sea level?", ("100",)),
    FactualItem("Which is the largest ocean on Earth?", ("pacific",)),
    FactualItem("What is the hardest naturally occurring substance?", ("diamond",)),
    FactualItem("Which gas do plants absorb for photosynthesis?", ("carbon dioxide", "co2")),
    FactualItem("Who wrote the novel Pride and Prejudice?", ("austen",)),
    FactualItem("Who painted the Mona Lisa?", ("leonardo", "da vinci")),
    FactualItem("Who was the first President of the United States?", ("washington",)),
    FactualItem("In which year did World War II end?", ("1945",)),
    FactualItem("Who developed the theory of general relativity?", ("einstein",)),
    FactualItem("Which river flows through Cairo?", ("nile",)),
    FactualItem("What is the main language spoken in Brazil?", ("portuguese",)),
    FactualItem("How many continents are there on Earth?", ("seven", "7")),
)


def normalise(text: str) -> str:
    """Lower-case text and strip diacritics using NFKD decomposition."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def is_correct(first_line: str, answers: Sequence[str]) -> bool:
    """Return True if any accepted answer occurs as a whole word or phrase in the completion line."""
    haystack = normalise(first_line)
    return any(re.search(r"\b" + re.escape(a) + r"\b", haystack) for a in answers)


def has_unsupported_specific(question: str, first_line: str) -> bool:
    """Detect a confident specific (capitalised word or number) that does not appear in the question."""
    question_lower = question.lower()
    entities = re.findall(r"\b[A-Z][A-Za-z]+\b|\d+", first_line)
    return any(e.lower() not in question_lower for e in entities)


@dataclass(frozen=True)
class RegimeReport:
    """Aggregate factual-error statistics for one temperature."""

    temperature: float
    trials: int
    error_rate: float
    error_ci: Tuple[float, float]
    hallucination_rate: float
    hallucination_ci: Tuple[float, float]
    mean_entropy_bits: float
    failures: List[Dict[str, str]]


def evaluate_regime(
    engine: GPT2Engine,
    temperature: float,
    samples_per_item: int = 3,
    seed: int = 0,
    top_k: int = 0,
    top_p: float = 1.0,
    max_new_tokens: int = 6,
    items: Sequence[FactualItem] = FACT_BENCHMARK,
    processors: Sequence[LogitProcessor] = (),
) -> RegimeReport:
    """Measure factual error and hallucination rates at one temperature.

    A trial is an error when no accepted answer appears in the first completion line; it is a
    hallucination when it is an error and the line asserts an unsupported capitalised word or number.

    Args:
        engine: Loaded GPT2Engine.
        temperature: Positive sampling temperature.
        samples_per_item: Independent seeded samples per question.
        seed: Base seed; each (item, sample) pair derives a distinct seed.
        top_k: Top-k size (<= 0 disables).
        top_p: Nucleus threshold in (0, 1].
        max_new_tokens: Completion length in tokens.
        items: Benchmark questions.
        processors: Optional logit processors such as a toxic-token suppressor.

    Returns:
        RegimeReport with rates, Wilson intervals, mean step entropy and failure records.
    """
    errors = 0
    hallucinations = 0
    entropies: List[float] = []
    failures: List[Dict[str, str]] = []
    for item_index, item in enumerate(items):
        prompt = f"{FEWSHOT_HEADER}Q: {item.question}\nA:"
        for sample_index in range(samples_per_item):
            result = engine.generate(
                prompt, max_new_tokens, temperature, top_k, top_p,
                seed=seed + item_index * 1000 + sample_index, processors=processors,
            )
            first_line = result.completion.split("\n")[0].strip()
            entropies.extend(step.entropy_bits for step in result.steps)
            if not is_correct(first_line, item.answers):
                errors += 1
                if has_unsupported_specific(item.question, first_line):
                    hallucinations += 1
                failures.append(
                    {"question": item.question, "expected": " / ".join(item.answers), "generated": first_line}
                )
    trials = len(items) * samples_per_item
    return RegimeReport(
        temperature=temperature,
        trials=trials,
        error_rate=errors / trials,
        error_ci=wilson_interval(errors, trials),
        hallucination_rate=hallucinations / trials,
        hallucination_ci=wilson_interval(hallucinations, trials),
        mean_entropy_bits=float(sum(entropies) / max(len(entropies), 1)),
        failures=failures,
    )

