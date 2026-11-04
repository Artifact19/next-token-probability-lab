# Model Card: Next-Token Probability Lab (P5)

## 1. Overview

The Next-Token Probability Lab is an interactive computational laboratory that analyzes how autoregressive and masked language models assign probabilities to vocabulary tokens and how sampling hyperparameters (temperature, top-k, top-p) alter generation. It composes pre-trained models and evaluates a recurrent LSTM baseline trained for comparative benchmarking.

| Component | Role | Source |
|---|---|---|
| GPT-2 small (`gpt2`, 124M parameters) | Causal next-token prediction, sampling, attention extraction | OpenAI, via Hugging Face |
| BERT base uncased (`bert-base-uncased`, 110M) | Masked-word prediction for contrast with GPT-2 | Google, via Hugging Face |
| LSTM baseline (2 layers, 256 units, tied embeddings) | Recurrent perplexity baseline trained in this repository | This repository |
| sd-turbo (`stabilityai/sd-turbo`) | Single-step text-to-image synthesis | Stability AI, via Hugging Face |

## 2. Intended Use

- Academic research and empirical analysis of token-level probability distributions, entropy, sampling filters, attention patterns, and masked versus causal autoregressive architectures.
- Comparative evaluation of factual reliability and hallucination rates across sampling temperatures on a closed-book cloze benchmark.

**Out-of-scope uses:** Factual question answering, automated moderation decisions, medical, legal, or financial advice, unreviewed production text generation, or any high-stakes autonomous decision pipeline.

## 3. Training and Evaluation Data

| Dataset | Used by | License | Notes |
|---|---|---|---|
| WebText (not redistributed) | GPT-2 pre-training | Model weights under MIT | Web-scraped outbound Reddit links |
| BookCorpus + English Wikipedia | BERT pre-training | Apache-2.0 | Encyclopedic and fiction text corpus |
| WikiText-2 (raw) | LSTM training; perplexity evaluation | CC BY-SA 3.0 | Verified Wikipedia articles |
| LAION-derived data | sd-turbo lineage | Stability AI Community License | Text-to-image synthesis dataset |

## 4. Metrics

- **Perplexity:** Evaluated over non-overlapping 128-token blocks using the shared GPT-2 BPE vocabulary to enable direct architectural comparison between LSTM and GPT-2.
- **Expected Calibration Error (ECE):** Top-1 confidence calibration evaluated over equal-width confidence partitions.
- **Factual Error Rate:** Proportion of cloze question completions containing no accepted ground-truth entity in the initial line, reported with 95% Wilson score intervals.
- **Hallucination Rate:** Subset of errors asserting unsupported entities or numeric quantities absent from context.

Empirical measurements are computed by `evaluate_temperature_errors.py` and recorded in `README.md`.

## 5. Limitations

- Benchmark coverage is focused on 24 reference factual prompts; confidence intervals reflect small-sample binomial uncertainty.
- GPT-2 small (124M) exhibits limited parametric factual recall; high error rates at elevated temperatures are expected.
- Heuristic hallucination detection captures entity intrusions but does not detect all contextually subtle inaccuracies.
- Lexical logit suppression operates on exact token sequences and does not resolve semantic paraphrasing.
- Tokenizer alignment between BERT WordPiece and GPT-2 BPE operates over first sub-word bounds for multi-token words.

## 6. Ethical Considerations and Bias

Language models inherit distributional skews present in web pre-training data, including occupational and gender associations. The laboratory provides visualization tools to audit these patterns. Generated images from diffusion models may reflect demographic skews and should not be used to represent real individuals.

## 7. Safety Measures

- Logit-level suppression sets disallowed single-token and multi-token prefixes to `-inf` prior to softmax normalization, ensuring zero probability mass.
- Authentication tokens are accessed via `HF_TOKEN` environment variables and are never committed to version control.
- Image synthesis requests utilize local pipeline fallbacks when remote inference endpoints are unconfigured.

## 8. Licensing Summary

| Asset | License |
|---|---|
| Repository source code | MIT |
| GPT-2 | MIT |
| BERT base uncased | Apache-2.0 |
| sd-turbo | Stability AI Community License |
| WikiText-2 | CC BY-SA 3.0 |
