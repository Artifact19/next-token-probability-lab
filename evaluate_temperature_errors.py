"""Standalone benchmark: factual error and hallucination rates of GPT-2 across sampling temperatures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from core.gpt_engine import GPT2Engine
from core.guardrails import FACT_BENCHMARK, RegimeReport, ToxicTokenSuppressor, evaluate_regime


def reports_to_frame(reports: Sequence[RegimeReport]) -> pd.DataFrame:
    """Flatten regime reports into a table with rates and 95% Wilson interval bounds."""
    return pd.DataFrame(
        [
            {
                "temperature": r.temperature,
                "trials": r.trials,
                "error_rate": r.error_rate,
                "error_ci_low": r.error_ci[0],
                "error_ci_high": r.error_ci[1],
                "hallucination_rate": r.hallucination_rate,
                "hallucination_ci_low": r.hallucination_ci[0],
                "hallucination_ci_high": r.hallucination_ci[1],
                "mean_entropy_bits": r.mean_entropy_bits,
            }
            for r in reports
        ]
    )


def frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render the benchmark table as GitHub-flavoured Markdown."""
    header = "| Temperature | Trials | Factual error rate (95% CI) | Hallucination rate (95% CI) | Mean step entropy (bits) |"
    divider = "|---:|---:|---:|---:|---:|"
    lines = [header, divider]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.temperature:.1f} | {row.trials} "
            f"| {row.error_rate:.1%} ({row.error_ci_low:.1%}-{row.error_ci_high:.1%}) "
            f"| {row.hallucination_rate:.1%} ({row.hallucination_ci_low:.1%}-{row.hallucination_ci_high:.1%}) "
            f"| {row.mean_entropy_bits:.2f} |"
        )
    return "\n".join(lines)


def update_readme(readme_path: Path, table: str) -> None:
    """Replace the region between the RESULTS_START and RESULTS_END markers with the benchmark table."""
    start, end = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"
    content = readme_path.read_text(encoding="utf-8")
    head, remainder = content.split(start, 1)
    _, tail = remainder.split(end, 1)
    readme_path.write_text(f"{head}{start}\n{table}\n{end}{tail}", encoding="utf-8")


def main() -> None:
    """Run the benchmark for every requested temperature and write CSV, JSON and Markdown artefacts."""
    parser = argparse.ArgumentParser(description="Error-versus-temperature benchmark.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2, 0.6, 1.0, 1.2, 1.6])
    parser.add_argument("--samples", type=int, default=5, help="Samples per question per temperature.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--toxic-filter", action="store_true", help="Apply the toxic-token suppressor during sampling.")
    parser.add_argument("--show-failures", type=int, default=5, help="Failure examples printed per temperature.")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--update-readme", action="store_true", help="Write the table between README result markers.")
    args = parser.parse_args()

    engine = GPT2Engine("gpt2")
    processors = [ToxicTokenSuppressor(engine)] if args.toxic_filter else []
    reports: List[RegimeReport] = []
    for temperature in args.temperatures:
        report = evaluate_regime(
            engine, temperature, samples_per_item=args.samples, seed=args.seed,
            top_k=args.top_k, top_p=args.top_p, processors=processors,
        )
        reports.append(report)
        print(f"T={temperature:.1f}: error {report.error_rate:.1%}, hallucination {report.hallucination_rate:.1%}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame = reports_to_frame(reports)
    frame.to_csv(out / "error_vs_temperature.csv", index=False)
    table = frame_to_markdown(frame)
    (out / "error_vs_temperature.md").write_text(table + "\n", encoding="utf-8")
    failures = {f"{r.temperature:.1f}": r.failures for r in reports}
    (out / "failure_cases.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print("\n" + table)
    if args.update_readme:
        update_readme(Path("README.md"), table)
    for report in reports:
        print(f"\nFailure examples at T={report.temperature:.1f} ({len(FACT_BENCHMARK)} questions):")
        for failure in report.failures[: args.show_failures]:
            gen_str = failure['generated'].encode('ascii', errors='replace').decode('ascii')
            print(f"  Q: {failure['question']} | expected: {failure['expected']} | generated: {gen_str!r}")


if __name__ == "__main__":
    main()

