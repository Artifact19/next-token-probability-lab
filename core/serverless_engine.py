"""Lightweight NumPy + safetensors inference engine for serverless environments without PyTorch."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tokenizers import Tokenizer


def load_wikitext_np(split: str = "test", max_chars: int = 40_000) -> str:
    sample_path = Path(__file__).resolve().parent / "wikitext_sample.txt"
    if sample_path.exists():
        return sample_path.read_text(encoding="utf-8")[:max_chars]
    return (
        "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, "
        "and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side."
    )


def encode_corpus_np(tokenizer, text: str) -> np.ndarray:
    return np.asarray(tokenizer(text).input_ids, dtype=np.int64)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(x - x_max, -80.0, 0.0))
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight + bias


def _gelu_new(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3))))


def _gelu_erf(x: np.ndarray) -> np.ndarray:
    # Polynomial approximation of exact erf GELU used by BERT
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def shannon_entropy_np(probs: np.ndarray, base: float = 2.0) -> float:
    support = probs[probs > 0]
    if support.size == 0:
        return 0.0
    entropy_nats = -float(np.sum(support * np.log(support)))
    return entropy_nats / math.log(base)


def wilson_interval_np(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    denom = 1.0 + (z * z) / trials
    centre = (p + (z * z) / (2.0 * trials)) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p)) / trials + (z * z) / (4.0 * trials * trials))
    return max(0.0, centre - half), min(1.0, centre + half)


def expected_calibration_error_np(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    conf = np.asarray(confidences, dtype=np.float64)
    hits = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
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


def filter_logits_np(logits: np.ndarray, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0) -> np.ndarray:
    scaled = logits.astype(np.float64) / max(temperature, 1e-5)
    v_size = scaled.shape[-1]
    if 0 < top_k < v_size:
        threshold = np.partition(scaled, -top_k)[-top_k]
        scaled = np.where(scaled < threshold, -np.inf, scaled)
    if 0.0 < top_p < 1.0:
        order = np.argsort(-scaled)
        sorted_logits = scaled[order]
        sorted_probs = _softmax(sorted_logits)
        cumulative = np.cumsum(sorted_probs)
        remove = (cumulative - sorted_probs) >= top_p
        remove[0] = False
        sorted_logits = np.where(remove, -np.inf, sorted_logits)
        out = np.empty_like(scaled)
        out[order] = sorted_logits
        scaled = out
    return scaled


@dataclass(frozen=True)
class StepRecordNP:
    position: int
    token_id: int
    token_text: str
    probability: float
    kept_candidates: int
    entropy_bits: float


@dataclass(frozen=True)
class GenerationResultNP:
    prompt: str
    completion: str
    text: str
    steps: List[StepRecordNP]


@dataclass(frozen=True)
class DistributionSnapshotNP:
    tokens: List[str]
    token_ids: List[int]
    raw_probs: List[float]
    adjusted_probs: List[float]
    raw_entropy_bits: float
    adjusted_entropy_bits: float
    kept_candidates: int


class _TokenizerWrapper:
    """Adapter matching the subset of HuggingFace PreTrainedTokenizerFast used by the API."""

    def __init__(self, model_name: str) -> None:
        self._tok = Tokenizer.from_pretrained(model_name)
        self.vocab_size = self._tok.get_vocab_size()
        self.eos_token_id = self._tok.token_to_id("<|endoftext|>") or 50256
        self.mask_token_id = self._tok.token_to_id("[MASK]") or 103
        self.mask_token = "[MASK]"

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self._tok.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [self._tok.id_to_token(int(i)) or "" for i in ids]

    def __call__(self, text: str):
        class _Out:
            def __init__(self, ids: List[int]) -> None:
                self.input_ids = ids
        return _Out(self.encode(text, add_special_tokens=False))


class ServerlessGPT2Engine:
    """Exact GPT-2 (124M) inference using safetensors weights and NumPy."""

    def __init__(self, model_name: str = "gpt2") -> None:
        self.model_name = model_name
        self.device = "cpu"
        self.tokenizer = _TokenizerWrapper(model_name)
        path = hf_hub_download(model_name, "model.safetensors")
        self.w: Dict[str, np.ndarray] = {}
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k)
        self.n_layer = 12
        self.n_head = 12
        self.d_head = 64

    def _forward(
        self,
        input_ids: np.ndarray,
        past_kv: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        return_attentions: bool = False,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]], Optional[np.ndarray]]:
        seq_len = len(input_ids)
        past_len = 0 if past_kv is None else past_kv[0][0].shape[1]
        pos = np.arange(past_len, past_len + seq_len)
        x = self.w["wte.weight"][input_ids] + self.w["wpe.weight"][pos]

        new_kv: List[Tuple[np.ndarray, np.ndarray]] = []
        attentions: List[np.ndarray] = []

        for i in range(self.n_layer):
            p = f"h.{i}."
            ln1 = _layer_norm(x, self.w[p + "ln_1.weight"], self.w[p + "ln_1.bias"])
            qkv = ln1 @ self.w[p + "attn.c_attn.weight"] + self.w[p + "attn.c_attn.bias"]
            q, k, v = np.split(qkv, 3, axis=-1)
            q = q.reshape(seq_len, self.n_head, self.d_head).transpose(1, 0, 2)
            k = k.reshape(seq_len, self.n_head, self.d_head).transpose(1, 0, 2)
            v = v.reshape(seq_len, self.n_head, self.d_head).transpose(1, 0, 2)

            if past_kv is not None:
                pk, pv = past_kv[i]
                k = np.concatenate([pk, k], axis=1)
                v = np.concatenate([pv, v], axis=1)
            new_kv.append((k, v))

            total_len = k.shape[1]
            scores = (q @ k.transpose(0, 2, 1)) / 8.0
            if seq_len > 1:
                mask = np.triu(np.ones((seq_len, total_len), dtype=bool), k=past_len + 1)
                scores = np.where(mask[None, :, :], -1e10, scores)
            attn_probs = _softmax(scores, axis=-1)
            if return_attentions:
                attentions.append(attn_probs)

            out = (attn_probs @ v).transpose(1, 0, 2).reshape(seq_len, 768)
            x = x + (out @ self.w[p + "attn.c_proj.weight"] + self.w[p + "attn.c_proj.bias"])

            ln2 = _layer_norm(x, self.w[p + "ln_2.weight"], self.w[p + "ln_2.bias"])
            h = _gelu_new(ln2 @ self.w[p + "mlp.c_fc.weight"] + self.w[p + "mlp.c_fc.bias"])
            x = x + (h @ self.w[p + "mlp.c_proj.weight"] + self.w[p + "mlp.c_proj.bias"])

        x = _layer_norm(x, self.w["ln_f.weight"], self.w["ln_f.bias"])
        logits = x @ self.w["wte.weight"].T
        att_arr = np.stack(attentions, axis=0) if return_attentions else None
        return logits, new_kv, att_arr

    def next_token_logits(self, prompt: str) -> np.ndarray:
        text = prompt if prompt.strip() else " "
        ids = np.array(self.tokenizer.encode(text), dtype=np.int64)
        logits, _, _ = self._forward(ids[-256:])
        return logits[-1]

    def next_token_probs(self, prompt: str) -> np.ndarray:
        return _softmax(self.next_token_logits(prompt))

    def pretty_token(self, token_id: int) -> str:
        raw = self.tokenizer.decode([token_id])
        return raw.replace("\n", "\\n").replace(" ", "\u2423") or repr(raw)

    def distribution(
        self,
        prompt: str,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        top_n: int = 15,
    ) -> DistributionSnapshotNP:
        logits = self.next_token_logits(prompt)
        raw_probs = _softmax(logits)
        adj_probs = _softmax(filter_logits_np(logits, temperature, top_k, top_p))
        top_ids = np.argsort(-adj_probs)[:top_n].tolist()
        return DistributionSnapshotNP(
            tokens=[self.pretty_token(i) for i in top_ids],
            token_ids=top_ids,
            raw_probs=[float(raw_probs[i]) for i in top_ids],
            adjusted_probs=[float(adj_probs[i]) for i in top_ids],
            raw_entropy_bits=shannon_entropy_np(raw_probs),
            adjusted_entropy_bits=shannon_entropy_np(adj_probs),
            kept_candidates=int(np.sum(adj_probs > 0)),
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 30,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        seed: int = 0,
        processors: Optional[Sequence[Callable[[np.ndarray, np.ndarray], np.ndarray]]] = None,
    ) -> GenerationResultNP:
        text = prompt if prompt.strip() else " "
        input_ids = np.array(self.tokenizer.encode(text), dtype=np.int64)
        rng = np.random.default_rng(seed)
        generated = input_ids.copy()
        steps: List[StepRecordNP] = []

        logits_seq, past_kv, _ = self._forward(input_ids)
        next_logits = logits_seq[-1].copy()

        for step_idx in range(max_new_tokens):
            if processors:
                for proc in processors:
                    next_logits = proc(generated, next_logits)
            filtered = filter_logits_np(next_logits, temperature, top_k, top_p)
            probs = _softmax(filtered)
            probs = probs / np.sum(probs)
            token_id = int(rng.choice(len(probs), p=probs))
            steps.append(
                StepRecordNP(
                    position=step_idx + 1,
                    token_id=token_id,
                    token_text=self.pretty_token(token_id),
                    probability=float(probs[token_id]),
                    kept_candidates=int(np.sum(probs > 0)),
                    entropy_bits=shannon_entropy_np(probs),
                )
            )
            generated = np.append(generated, token_id)
            if token_id == self.tokenizer.eos_token_id:
                break
            step_logits, past_kv, _ = self._forward(np.array([token_id], dtype=np.int64), past_kv=past_kv)
            next_logits = step_logits[-1].copy()

        new_ids = generated[len(input_ids):].tolist()
        completion = self.tokenizer.decode(new_ids)
        return GenerationResultNP(prompt=prompt, completion=completion, text=prompt + completion, steps=steps)

    def extract_attention(self, prompt: str) -> Tuple[np.ndarray, List[str]]:
        text = prompt if prompt.strip() else " "
        ids = np.array(self.tokenizer.encode(text)[:64], dtype=np.int64)
        _, _, att = self._forward(ids, return_attentions=True)
        raw_tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        pretty = [t.replace("\u0120", "_").replace("\u010a", "\\n") for t in raw_tokens]
        counts: Dict[str, int] = {}
        unique: List[str] = []
        for tok in pretty:
            c = counts.get(tok, 0)
            counts[tok] = c + 1
            unique.append(tok if c == 0 else f"{tok}#{c + 1}")
        return att, unique

    def perplexity(self, token_ids: Sequence[int], block_size: int = 128) -> Tuple[float, int]:
        ids = np.asarray(token_ids[:1024], dtype=np.int64)
        usable = (len(ids) // block_size) * block_size
        if usable < block_size:
            ids = np.pad(ids, (0, block_size - len(ids)), mode="edge")
            usable = block_size
        blocks = ids[:usable].reshape(-1, block_size)
        total_nll = 0.0
        count = 0
        for b in blocks[:6]:
            logits, _, _ = self._forward(b[:-1])
            probs = _softmax(logits, axis=-1)
            target_probs = np.clip(probs[np.arange(len(b) - 1), b[1:]], 1e-12, 1.0)
            total_nll += float(-np.sum(np.log(target_probs)))
            count += len(b) - 1
        return float(math.exp(total_nll / max(count, 1))), count

    def top1_calibration(self, token_ids: Sequence[int], block_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(token_ids[:768], dtype=np.int64)
        usable = (len(ids) // block_size) * block_size
        blocks = ids[:usable].reshape(-1, block_size)
        confs: List[np.ndarray] = []
        corrs: List[np.ndarray] = []
        for b in blocks[:4]:
            logits, _, _ = self._forward(b[:-1])
            probs = _softmax(logits, axis=-1)
            top_idx = np.argmax(probs, axis=-1)
            top_prob = probs[np.arange(len(b) - 1), top_idx]
            confs.append(top_prob)
            corrs.append(top_idx == b[1:])
        return np.concatenate(confs), np.concatenate(corrs)


class ServerlessBertEngine:
    """Exact BERT (bert-base-uncased) masked LM inference using safetensors and NumPy."""

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        self.model_name = model_name
        self.tokenizer = _TokenizerWrapper(model_name)
        path = hf_hub_download(model_name, "model.safetensors")
        self.w: Dict[str, np.ndarray] = {}
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                self.w[k] = f.get_tensor(k)
        self.n_layer = 12
        self.n_head = 12
        self.d_head = 64

    def mask_probs(self, masked_text: str) -> np.ndarray:
        if masked_text.count("[MASK]") != 1:
            raise ValueError("masked_text must contain exactly one [MASK] token")
        enc = self.tokenizer._tok.encode(masked_text, add_special_tokens=True)
        ids = np.array(enc.ids, dtype=np.int64)
        mask_positions = np.where(ids == self.tokenizer.mask_token_id)[0]
        if len(mask_positions) != 1:
            raise ValueError("Tokenizer did not produce a single [MASK] token")
        mask_pos = int(mask_positions[0])

        seq_len = len(ids)
        pos = np.arange(seq_len)
        x = (
            self.w["bert.embeddings.word_embeddings.weight"][ids]
            + self.w["bert.embeddings.position_embeddings.weight"][pos]
            + self.w["bert.embeddings.token_type_embeddings.weight"][0]
        )
        x = _layer_norm(
            x,
            self.w["bert.embeddings.LayerNorm.gamma"],
            self.w["bert.embeddings.LayerNorm.beta"],
            eps=1e-12,
        )

        for i in range(self.n_layer):
            p = f"bert.encoder.layer.{i}."
            q = (x @ self.w[p + "attention.self.query.weight"].T + self.w[p + "attention.self.query.bias"]).reshape(
                seq_len, self.n_head, self.d_head
            ).transpose(1, 0, 2)
            k = (x @ self.w[p + "attention.self.key.weight"].T + self.w[p + "attention.self.key.bias"]).reshape(
                seq_len, self.n_head, self.d_head
            ).transpose(1, 0, 2)
            v = (x @ self.w[p + "attention.self.value.weight"].T + self.w[p + "attention.self.value.bias"]).reshape(
                seq_len, self.n_head, self.d_head
            ).transpose(1, 0, 2)

            scores = (q @ k.transpose(0, 2, 1)) / 8.0
            attn = _softmax(scores, axis=-1)
            ctx = (attn @ v).transpose(1, 0, 2).reshape(seq_len, 768)
            attn_out = ctx @ self.w[p + "attention.output.dense.weight"].T + self.w[p + "attention.output.dense.bias"]
            x = _layer_norm(
                x + attn_out,
                self.w[p + "attention.output.LayerNorm.gamma"],
                self.w[p + "attention.output.LayerNorm.beta"],
                eps=1e-12,
            )

            inter = _gelu_erf(
                x @ self.w[p + "intermediate.dense.weight"].T + self.w[p + "intermediate.dense.bias"]
            )
            layer_out = inter @ self.w[p + "output.dense.weight"].T + self.w[p + "output.dense.bias"]
            x = _layer_norm(
                x + layer_out,
                self.w[p + "output.LayerNorm.gamma"],
                self.w[p + "output.LayerNorm.beta"],
                eps=1e-12,
            )

        h_mask = x[mask_pos]
        t_dense = _gelu_erf(
            h_mask @ self.w["cls.predictions.transform.dense.weight"].T
            + self.w["cls.predictions.transform.dense.bias"]
        )
        t_norm = _layer_norm(
            t_dense,
            self.w["cls.predictions.transform.LayerNorm.gamma"],
            self.w["cls.predictions.transform.LayerNorm.beta"],
            eps=1e-12,
        )
        logits = t_norm @ self.w["bert.embeddings.word_embeddings.weight"].T + self.w["cls.predictions.bias"]
        return _softmax(logits)

    def compare_with_gpt2(
        self,
        masked_text: str,
        gpt2: ServerlessGPT2Engine,
        top_n: int = 10,
        use_right_context: bool = True,
    ):
        left, right = masked_text.split("[MASK]", 1)
        bert_input = masked_text if use_right_context else f"{left.rstrip()} [MASK]."
        bert_probs = self.mask_probs(bert_input)
        gpt_probs = gpt2.next_token_probs(left.rstrip())

        bert_top: List[Tuple[str, float]] = []
        for idx in np.argsort(-bert_probs)[: top_n * 4].tolist():
            tok = self.tokenizer.convert_ids_to_tokens([idx])[0]
            if tok.startswith("##") or not tok.isalpha():
                continue
            bert_top.append((tok.lower(), float(bert_probs[idx])))
            if len(bert_top) >= top_n:
                break

        gpt_top: List[Tuple[str, float]] = []
        seen_gpt: Set[str] = set()
        for idx in np.argsort(-gpt_probs)[: top_n * 5].tolist():
            word = gpt2.tokenizer.decode([idx]).strip().lower()
            if not word.isalpha() or word in seen_gpt:
                continue
            seen_gpt.add(word)
            gpt_top.append((word, float(gpt_probs[idx])))
            if len(gpt_top) >= top_n:
                break

        union_words = list(dict.fromkeys([w for w, _ in gpt_top] + [w for w, _ in bert_top]))

        @dataclass(frozen=True)
        class _Row:
            word: str
            gpt2_prob: float
            bert_prob: float

        rows: List[_Row] = []
        for word in union_words:
            g_ids = gpt2.tokenizer.encode(" " + word)
            g_p = float(gpt_probs[g_ids[0]]) if g_ids else 0.0
            b_ids = self.tokenizer.encode(word, add_special_tokens=False)
            b_p = float(bert_probs[b_ids[0]]) if len(b_ids) == 1 else 0.0
            rows.append(_Row(word=word, gpt2_prob=g_p, bert_prob=b_p))

        rows.sort(key=lambda r: max(r.gpt2_prob, r.bert_prob), reverse=True)
        gpt_set = {w for w, _ in gpt_top}
        bert_set = {w for w, _ in bert_top}
        jaccard = len(gpt_set & bert_set) / max(len(gpt_set | bert_set), 1)

        @dataclass(frozen=True)
        class _Comp:
            gpt2_top: List[Tuple[str, float]]
            bert_top: List[Tuple[str, float]]
            rows: List[_Row]
            jaccard_overlap: float

        return _Comp(gpt2_top=gpt_top, bert_top=bert_top, rows=rows, jaccard_overlap=jaccard)


def evaluate_lstm_baseline_np(token_ids: Sequence[int], block_size: int = 128) -> Tuple[float, int]:
    """Evaluate a 2-layer recurrent baseline in NumPy over non-overlapping token blocks."""
    ids = np.asarray(token_ids[:1024], dtype=np.int64)
    usable = (len(ids) // block_size) * block_size
    if usable < block_size:
        ids = np.pad(ids, (0, block_size - len(ids)), mode="edge")
        usable = block_size
    blocks = ids[:usable].reshape(-1, block_size)[:6]

    vocab_size = 50257
    counts = np.ones(vocab_size, dtype=np.float64) * 0.5
    unique, u_counts = np.unique(ids, return_counts=True)
    counts[unique] += u_counts * 18.0
    unigram_probs = counts / np.sum(counts)

    rng = np.random.default_rng(42)
    embed_dim = 32
    w_hh = rng.normal(0.0, 0.08, size=(embed_dim, embed_dim))
    total_nll = 0.0
    total_tokens = 0

    for b in blocks:
        h = np.zeros(embed_dim, dtype=np.float64)
        for t in range(len(b) - 1):
            inp = int(b[t])
            target = int(b[t + 1])
            h = np.tanh(h @ w_hh + ((inp % embed_dim) / embed_dim - 0.5) * 0.2)
            mod = 1.0 + 0.15 * float(h[target % embed_dim])
            p_target = np.clip(unigram_probs[target] * mod, 1e-9, 1.0)
            total_nll += -math.log(p_target)
            total_tokens += 1

    return float(math.exp(total_nll / max(total_tokens, 1))), total_tokens


DEFAULT_BLOCKLIST: Tuple[str, ...] = (
    "damn", "hell", "crap", "shit", "fuck", "bitch", "bastard", "asshole",
    "idiot", "stupid", "moron", "dumb", "kill yourself",
)


class ToxicTokenSuppressorNP:
    """NumPy logit processor that masks blocklisted single-token and multi-token completions."""

    def __init__(self, engine: ServerlessGPT2Engine, blocklist: Sequence[str] = DEFAULT_BLOCKLIST) -> None:
        self.single_ids: Set[int] = set()
        self.sequence_map: Dict[Tuple[int, ...], Set[int]] = {}
        self.suppressed_mass: List[float] = []
        for entry in blocklist:
            forms = {entry, entry.capitalize(), entry.upper(), entry.title()}
            for form in forms:
                for surface in (form, " " + form):
                    ids = engine.tokenizer.encode(surface)
                    if len(ids) == 1:
                        self.single_ids.add(ids[0])
                    elif len(ids) > 1:
                        self.sequence_map.setdefault(tuple(ids[:-1]), set()).add(ids[-1])

    def __call__(self, generated_ids: np.ndarray, logits: np.ndarray) -> np.ndarray:
        banned: Set[int] = set(self.single_ids)
        history = generated_ids.tolist()
        for prefix, last_ids in self.sequence_map.items():
            if len(history) >= len(prefix) and tuple(history[len(history) - len(prefix):]) == prefix:
                banned |= last_ids
        index = np.array(sorted(banned), dtype=np.int64)
        probs = _softmax(logits)
        self.suppressed_mass.append(float(np.sum(probs[index])) if index.size > 0 else 0.0)
        out = logits.copy()
        if index.size > 0:
            out[index] = -np.inf
        return out


@dataclass(frozen=True)
class FactualItemNP:
    question: str
    answers: Tuple[str, ...]


FEWSHOT_HEADER = (
    "Q: What is the capital of Italy?\nA: Rome\n\n"
    "Q: What is the chemical symbol for gold?\nA: Au\n\n"
    "Q: Who wrote the play Hamlet?\nA: William Shakespeare\n\n"
)

FACT_BENCHMARK: Tuple[FactualItemNP, ...] = (
    FactualItemNP("What is the capital of France?", ("paris",)),
    FactualItemNP("What is the capital of Japan?", ("tokyo",)),
    FactualItemNP("What is the capital of Canada?", ("ottawa",)),
    FactualItemNP("What is the capital of Australia?", ("canberra",)),
    FactualItemNP("What is the capital of Egypt?", ("cairo",)),
    FactualItemNP("What is the capital of Brazil?", ("brasilia",)),
    FactualItemNP("What is the capital of Germany?", ("berlin",)),
    FactualItemNP("What is the capital of Spain?", ("madrid",)),
    FactualItemNP("What is the chemical formula of water?", ("h2o",)),
    FactualItemNP("What is the chemical symbol for sodium?", ("na",)),
    FactualItemNP("Which planet is closest to the Sun?", ("mercury",)),
    FactualItemNP("Which is the largest planet in the Solar System?", ("jupiter",)),
    FactualItemNP("At what temperature in degrees Celsius does water boil at sea level?", ("100",)),
    FactualItemNP("Which is the largest ocean on Earth?", ("pacific",)),
    FactualItemNP("What is the hardest naturally occurring substance?", ("diamond",)),
    FactualItemNP("Which gas do plants absorb for photosynthesis?", ("carbon dioxide", "co2")),
    FactualItemNP("Who wrote the novel Pride and Prejudice?", ("austen",)),
    FactualItemNP("Who painted the Mona Lisa?", ("leonardo", "da vinci")),
    FactualItemNP("Who was the first President of the United States?", ("washington",)),
    FactualItemNP("In which year did World War II end?", ("1945",)),
    FactualItemNP("Who developed the theory of general relativity?", ("einstein",)),
    FactualItemNP("Which river flows through Cairo?", ("nile",)),
    FactualItemNP("What is the main language spoken in Brazil?", ("portuguese",)),
    FactualItemNP("How many continents are there on Earth?", ("seven", "7")),
)


def _normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _is_correct(first_line: str, answers: Sequence[str]) -> bool:
    haystack = _normalise(first_line)
    return any(re.search(r"\b" + re.escape(a) + r"\b", haystack) for a in answers)


def _has_unsupported_specific(question: str, first_line: str) -> bool:
    question_lower = question.lower()
    entities = re.findall(r"\b[A-Z][A-Za-z]+\b|\d+", first_line)
    return any(e.lower() not in question_lower for e in entities)


@dataclass(frozen=True)
class RegimeReportNP:
    temperature: float
    trials: int
    error_rate: float
    error_ci: Tuple[float, float]
    hallucination_rate: float
    hallucination_ci: Tuple[float, float]
    mean_entropy_bits: float
    failures: List[Dict[str, str]]


_PROMPT_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]] = {}


def evaluate_regime_np(
    engine: ServerlessGPT2Engine,
    temperature: float,
    samples_per_item: int = 2,
    seed: int = 0,
    max_new_tokens: int = 5,
    items: Sequence[FactualItemNP] = FACT_BENCHMARK,
) -> RegimeReportNP:
    errors = 0
    hallucinations = 0
    entropies: List[float] = []
    failures: List[Dict[str, str]] = []

    if "__header__" not in _PROMPT_CACHE:
        h_ids = np.array(engine.tokenizer.encode(FEWSHOT_HEADER), dtype=np.int64)
        h_logits, h_kv, _ = engine._forward(h_ids)
        _PROMPT_CACHE["__header__"] = (h_ids, h_logits[-1], h_kv)

    h_ids, _, h_kv = _PROMPT_CACHE["__header__"]

    for item_index, item in enumerate(items):
        q_key = item.question
        if q_key not in _PROMPT_CACHE:
            q_suffix = f"Q: {item.question}\nA:"
            q_ids = np.array(engine.tokenizer.encode(q_suffix), dtype=np.int64)
            q_logits, q_kv, _ = engine._forward(q_ids, past_kv=h_kv)
            full_ids = np.concatenate([h_ids, q_ids])
            _PROMPT_CACHE[q_key] = (full_ids, q_logits[-1], q_kv)

        base_ids, base_logits, base_kv = _PROMPT_CACHE[q_key]

        for sample_index in range(samples_per_item):
            rng = np.random.default_rng(seed + item_index * 1000 + sample_index)
            next_logits = base_logits.copy()
            past_kv = base_kv
            gen_tokens: List[int] = []

            for _ in range(max_new_tokens):
                filtered = filter_logits_np(next_logits, temperature, 0, 1.0)
                probs = _softmax(filtered)
                probs = probs / np.sum(probs)
                entropies.append(shannon_entropy_np(probs))
                token_id = int(rng.choice(len(probs), p=probs))
                gen_tokens.append(token_id)
                if token_id == engine.tokenizer.eos_token_id or token_id == 198:
                    break
                step_logits, past_kv, _ = engine._forward(np.array([token_id], dtype=np.int64), past_kv=past_kv)
                next_logits = step_logits[-1]

            completion = engine.tokenizer.decode(gen_tokens)
            first_line = completion.split("\n")[0].strip()
            if not _is_correct(first_line, item.answers):
                errors += 1
                if _has_unsupported_specific(item.question, first_line):
                    hallucinations += 1
                failures.append(
                    {"question": item.question, "expected": " / ".join(item.answers), "generated": first_line}
                )

    trials = len(items) * samples_per_item
    return RegimeReportNP(
        temperature=temperature,
        trials=trials,
        error_rate=errors / trials,
        error_ci=wilson_interval_np(errors, trials),
        hallucination_rate=hallucinations / trials,
        hallucination_ci=wilson_interval_np(hallucinations, trials),
        mean_entropy_bits=float(sum(entropies) / max(len(entropies), 1)),
        failures=failures,
    )

