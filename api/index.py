"""FastAPI backend for Next-Token Probability Lab (Vercel deployment)."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Serverless filesystem is read-only except /tmp
if os.getenv("VERCEL"):
    os.environ.setdefault("HF_HOME", "/tmp/hf")

# Ensure repo root is on sys.path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.attention_vis import AttentionExtractor
from core.bert_engine import BertEngine
from core.gpt_engine import GPT2Engine
from core.guardrails import (
    DEFAULT_BLOCKLIST,
    FACT_BENCHMARK,
    ToxicTokenSuppressor,
    evaluate_regime,
)
from core.image_generator import ImageGenerationError, ImageGenerator
from core.lstm_baseline import (
    LSTMLanguageModel,
    encode_corpus,
    evaluate_perplexity,
    load_checkpoint,
    load_wikitext,
    save_checkpoint,
    train_lstm,
)
from utils.metrics import expected_calibration_error, shannon_entropy

app = FastAPI(
    title="Next-Token Probability Lab API",
    description="Backend powering Units U1 to U5 on Vercel",
    version="1.0.0",
)

# Enable CORS for local dev and Vercel domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy singletons
_gpt_engine: Optional[GPT2Engine] = None
_bert_engine: Optional[BertEngine] = None
_image_generator: Optional[ImageGenerator] = None
_lstm_model: Optional[LSTMLanguageModel] = None

LSTM_CHECKPOINT = ROOT_DIR / "artifacts" / "lstm_baseline.pt"


def get_gpt() -> GPT2Engine:
    global _gpt_engine
    if _gpt_engine is None:
        _gpt_engine = GPT2Engine("gpt2")
    return _gpt_engine


def get_bert() -> BertEngine:
    global _bert_engine
    if _bert_engine is None:
        _bert_engine = BertEngine("bert-base-uncased")
    return _bert_engine


def get_img_gen() -> ImageGenerator:
    global _image_generator
    if _image_generator is None:
        _image_generator = ImageGenerator()
    return _image_generator


def get_lstm() -> Optional[LSTMLanguageModel]:
    global _lstm_model
    if _lstm_model is None and LSTM_CHECKPOINT.exists():
        try:
            _lstm_model = load_checkpoint(str(LSTM_CHECKPOINT), device="cpu")
        except Exception:
            _lstm_model = None
    return _lstm_model


# ---------------------------------------------------------------------------
# Request & Response Models
# ---------------------------------------------------------------------------

class DistributionRequest(BaseModel):
    prompt: str = Field(default="The scientist opened the door and saw that the")
    temperature: float = Field(default=1.0, ge=0.01, le=5.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_n: int = Field(default=20, ge=5, le=50)


class CompareTempsRequest(BaseModel):
    prompt: str = Field(default="The scientist opened the door and saw that the")
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_n: int = Field(default=15, ge=5, le=30)
    seed: int = Field(default=7)


class AttentionRequest(BaseModel):
    prompt: str = Field(default="The animal didn't cross the street because it was too tired")
    layer: int = Field(default=0, ge=0, le=11)
    head: Optional[int] = Field(default=None, ge=0, le=11)


class PerplexityRequest(BaseModel):
    source: str = Field(default="wikitext")
    custom_text: Optional[str] = None
    block_size: int = Field(default=128, ge=32, le=512)


class BertCompareRequest(BaseModel):
    masked_text: str = Field(default="The doctor told the nurse that [MASK] would arrive soon.")
    use_right_context: bool = Field(default=True)
    top_n: int = Field(default=10, ge=3, le=25)


class GenerateRequest(BaseModel):
    prompt: str = Field(default="A quiet harbour at dawn, with")
    max_new_tokens: int = Field(default=25, ge=1, le=100)
    temperature: float = Field(default=1.0, ge=0.01, le=3.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    seed: int = Field(default=0)


class ImageSynthRequest(BaseModel):
    text: str
    seed: int = Field(default=0)


class AuditRequest(BaseModel):
    temperatures: List[float] = Field(default=[0.2, 1.2])
    samples_per_item: int = Field(default=2, ge=1, le=5)
    seed: int = Field(default=0)


class ToxicFilterRequest(BaseModel):
    prompt: str = Field(default="You are such a")
    temperature: float = Field(default=1.2, ge=0.1, le=2.0)
    seed: int = Field(default=3)
    max_tokens: int = Field(default=30, ge=5, le=60)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "has_hf_token": bool(os.getenv("HF_TOKEN")),
        "lstm_checkpoint_exists": LSTM_CHECKPOINT.exists(),
    }


# U1: Next-token distribution
@app.post("/api/u1/distribution")
def u1_distribution(req: DistributionRequest):
    try:
        gpt = get_gpt()
        dist = gpt.distribution(
            prompt=req.prompt,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            top_n=req.top_n,
        )
        # Sorted distribution for nucleus curve (top 100)
        all_probs = gpt.next_token_probs(req.prompt)
        sorted_probs = torch.sort(all_probs, descending=True).values[:100].tolist()

        return {
            "tokens": dist.tokens,
            "token_ids": dist.token_ids,
            "raw_probs": dist.raw_probs,
            "adjusted_probs": dist.adjusted_probs,
            "raw_entropy_bits": round(dist.raw_entropy_bits, 3),
            "adjusted_entropy_bits": round(dist.adjusted_entropy_bits, 3),
            "kept_candidates": dist.kept_candidates,
            "nucleus_curve": sorted_probs,
            "top_p": req.top_p,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U1: Dual temperature comparison
@app.post("/api/u1/compare-temperatures")
def u1_compare_temps(req: CompareTempsRequest):
    try:
        gpt = get_gpt()
        dist_low = gpt.distribution(req.prompt, 0.2, req.top_k, req.top_p, req.top_n)
        dist_high = gpt.distribution(req.prompt, 1.2, req.top_k, req.top_p, req.top_n)

        # Generate completions at both temperatures
        gen_low = gpt.generate(req.prompt, 40, 0.2, req.top_k, req.top_p, seed=req.seed)
        gen_high = gpt.generate(req.prompt, 40, 1.2, req.top_k, req.top_p, seed=req.seed)

        return {
            "tokens": dist_low.tokens,
            "probs_low": dist_low.adjusted_probs,
            "probs_high": dist_high.adjusted_probs,
            "text_low": gen_low.text,
            "text_high": gen_high.text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U2: Attention Extraction
@app.post("/api/u2/attention")
def u2_attention(req: AttentionRequest):
    try:
        gpt = get_gpt()
        extractor = AttentionExtractor(gpt.model, gpt.tokenizer)
        attentions = extractor.extract(req.prompt)  # [L, H, T, T]
        num_layers, num_heads = attentions.shape[0], attentions.shape[1]

        layer_idx = min(req.layer, num_layers - 1)
        head_idx = None if req.head is None else min(req.head, num_heads - 1)

        matrix = extractor.select(attentions, layer_idx, head_idx)
        tokens = extractor.tokens(req.prompt)
        received = extractor.attention_received(matrix).tolist()

        return {
            "tokens": tokens,
            "matrix": matrix.numpy().tolist(),
            "received_attention": [round(x, 4) for x in received],
            "num_layers": num_layers,
            "num_heads": num_heads,
            "selected_layer": layer_idx,
            "selected_head": head_idx,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U2: Perplexity Comparison
@app.post("/api/u2/perplexity")
def u2_perplexity(req: PerplexityRequest):
    try:
        gpt = get_gpt()
        if req.source == "custom" and req.custom_text and req.custom_text.strip():
            text = req.custom_text.strip()
        else:
            text = load_wikitext("test", max_chars=40_000)

        token_ids = encode_corpus(gpt.tokenizer, text)[:8192]
        gpt_ppl, count = gpt.perplexity(token_ids, block_size=req.block_size)

        lstm = get_lstm()
        lstm_ppl = None
        if lstm is not None:
            lstm_ppl, _ = evaluate_perplexity(lstm, token_ids, block_size=req.block_size)

        return {
            "token_count": count,
            "gpt_ppl": round(gpt_ppl, 2),
            "lstm_ppl": round(lstm_ppl, 2) if lstm_ppl else None,
            "has_lstm": lstm is not None,
            "block_size": req.block_size,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U3: BERT vs GPT-2
@app.post("/api/u3/compare")
def u3_compare(req: BertCompareRequest):
    try:
        if req.masked_text.count("[MASK]") != 1:
            raise HTTPException(status_code=400, detail="Text must contain exactly one [MASK] token.")
        gpt = get_gpt()
        bert = get_bert()
        comp = bert.compare_with_gpt2(req.masked_text, gpt, top_n=req.top_n, use_right_context=req.use_right_context)
        return {
            "gpt2_top": [{"word": w, "probability": round(p, 4)} for w, p in comp.gpt2_top],
            "bert_top": [{"word": w, "probability": round(p, 4)} for w, p in comp.bert_top],
            "aligned_rows": [
                {"word": r.word, "gpt2_prob": round(r.gpt2_prob, 4), "bert_prob": round(r.bert_prob, 4)}
                for r in comp.rows[: req.top_n * 2]
            ],
            "jaccard_overlap": round(comp.jaccard_overlap, 3),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U4: From-scratch generation with step trace
@app.post("/api/u4/generate")
def u4_generate(req: GenerateRequest):
    try:
        gpt = get_gpt()
        res = gpt.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            seed=req.seed,
        )
        return {
            "prompt": res.prompt,
            "completion": res.completion,
            "text": res.text,
            "steps": [
                {
                    "position": s.position,
                    "token_id": s.token_id,
                    "token_text": s.token_text,
                    "probability": round(s.probability, 4),
                    "kept_candidates": s.kept_candidates,
                    "entropy_bits": round(s.entropy_bits, 3),
                }
                for s in res.steps
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U4: Image synthesis
@app.post("/api/u4/synthesize-image")
def u4_image(req: ImageSynthRequest):
    try:
        gen = get_img_gen()
        image, backend = gen.generate(req.text, seed=req.seed)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            "image_data_url": f"data:image/png;base64,{img_str}",
            "backend": backend,
        }
    except ImageGenerationError as e:
        raise HTTPException(status_code=503, detail=f"Image generation failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U5: Hallucination Audit
@app.post("/api/u5/audit")
def u5_audit(req: AuditRequest):
    try:
        gpt = get_gpt()
        reports = []
        for temp in req.temperatures:
            rep = evaluate_regime(
                gpt,
                temperature=temp,
                samples_per_item=req.samples_per_item,
                seed=req.seed,
            )
            reports.append({
                "temperature": rep.temperature,
                "trials": rep.trials,
                "error_rate": round(rep.error_rate, 4),
                "error_ci": [round(rep.error_ci[0], 4), round(rep.error_ci[1], 4)],
                "hallucination_rate": round(rep.hallucination_rate, 4),
                "hallucination_ci": [round(rep.hallucination_ci[0], 4), round(rep.hallucination_ci[1], 4)],
                "mean_entropy_bits": round(rep.mean_entropy_bits, 2),
                "failures": rep.failures[:12],
            })
        return {"reports": reports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U5: Toxic Token Suppression
@app.post("/api/u5/toxic-suppression")
def u5_toxic(req: ToxicFilterRequest):
    try:
        gpt = get_gpt()
        suppressor = ToxicTokenSuppressor(gpt, blocklist=DEFAULT_BLOCKLIST)
        plain = gpt.generate(req.prompt, req.max_tokens, req.temperature, 0, 1.0, seed=req.seed)
        filtered = gpt.generate(
            req.prompt, req.max_tokens, req.temperature, 0, 1.0, seed=req.seed, processors=[suppressor]
        )
        peak_mass = max(suppressor.suppressed_mass) if suppressor.suppressed_mass else 0.0

        return {
            "unfiltered": plain.text,
            "filtered": filtered.text,
            "peak_suppressed_mass": round(peak_mass, 4),
            "suppressed_mass_history": [round(m, 4) for m in suppressor.suppressed_mass],
            "blocklist": list(DEFAULT_BLOCKLIST),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# U5: Calibration
@app.post("/api/u5/calibration")
def u5_calibration():
    try:
        gpt = get_gpt()
        text = load_wikitext("test", max_chars=40_000)
        token_ids = encode_corpus(gpt.tokenizer, text)[:4096]
        confidences, correct = gpt.top1_calibration(token_ids)
        ece, bin_conf, bin_acc, bin_count = expected_calibration_error(confidences, correct, n_bins=10)

        return {
            "ece": round(ece, 4),
            "bin_conf": [round(x, 4) for x in bin_conf.tolist()],
            "bin_acc": [round(x, 4) for x in bin_acc.tolist()],
            "bin_count": [int(x) for x in bin_count.tolist()],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model card content
@app.get("/api/model-card")
def get_model_card():
    card_path = ROOT_DIR / "MODEL_CARD.md"
    if card_path.exists():
        return {"content": card_path.read_text(encoding="utf-8")}
    return {"content": "# Model Card Not Found"}

