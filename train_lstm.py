"""Command-line trainer that fits the LSTM baseline on WikiText-2 and reports held-out perplexity."""
from __future__ import annotations

import argparse

from transformers import GPT2TokenizerFast

from core.lstm_baseline import (
    LSTMLanguageModel,
    encode_corpus,
    evaluate_perplexity,
    load_wikitext,
    save_checkpoint,
    train_lstm,
)


def main() -> None:
    """Parse arguments, train the LSTM and write the checkpoint."""
    parser = argparse.ArgumentParser(description="Train the LSTM language-model baseline.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-chars", type=int, default=400_000)
    parser.add_argument("--output", type=str, default="artifacts/lstm_baseline.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_ids = encode_corpus(tokenizer, load_wikitext("train", args.max_chars))
    test_ids = encode_corpus(tokenizer, load_wikitext("test", 40_000))
    model = LSTMLanguageModel(vocab_size=tokenizer.vocab_size)
    train_lstm(
        model, train_ids, steps=args.steps, seed=args.seed,
        progress=lambda step, loss: print(f"step {step}/{args.steps} loss {loss:.4f}") if step % 25 == 0 else None,
    )
    perplexity, count = evaluate_perplexity(model, test_ids[:8192])
    print(f"test perplexity {perplexity:.2f} over {count} tokens")
    save_checkpoint(model, args.output)


if __name__ == "__main__":
    main()

