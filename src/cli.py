"""CLI entrypoint (`funnel-debug`) for Funnel-GPT."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from src.report import write_html_report
from src.score import score_non_converting_sessions
from src.train import train_funnel_gpt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="funnel-debug",
        description="Debug product conversion funnels with a from-scratch PyTorch GPT model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Train on converting sessions, score abandoned sessions, and generate report.html",
    )
    run_parser.add_argument("csv_path", help="Path to 3-column CSV (session_id, timestamp, event_name)")
    run_parser.add_argument("--goal", required=True, help="Goal event name defining a converting session (e.g. purchase)")
    run_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    run_parser.add_argument("--out", default="report.html", help="Output standalone HTML report path (default: report.html)")
    run_parser.add_argument("--artifacts-dir", default="artifacts", help="Directory for model.pt, tokenizer.json, metrics.json")
    run_parser.add_argument("--batch-size", type=int, default=64, help="Training batch size (default: 64)")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        t0 = time.perf_counter()
        print(f"[funnel-debug] Training on converting sessions (goal='{args.goal}', epochs={args.epochs})...")
        train_result = train_funnel_gpt(
            csv_path=args.csv_path,
            goal_event=args.goal,
            epochs=args.epochs,
            batch_size=args.batch_size,
            artifacts_dir=args.artifacts_dir,
            seed=args.seed,
        )

        metrics = train_result["metrics"]
        print(
            f"[funnel-debug] Trained on {metrics['converting_sessions']:,} converting sessions "
            f"(device={metrics['device']}, params={metrics['param_count']:,}, "
            f"train_loss={metrics['final_train_loss']:.4f}, val_loss={metrics['final_val_loss']:.4f}, "
            f"p95={metrics['p95_threshold']:.4f})"
        )

        print(f"[funnel-debug] Scoring {metrics['non_converting_sessions']:,} non-converting sessions...")
        score_result = score_non_converting_sessions(
            model=train_result["model"],
            tokenizer=train_result["tokenizer"],
            non_converting_sessions=train_result["non_converting_sessions"],
            p95_threshold=train_result["p95_threshold"],
            device=train_result["device"],
            top_k=5,
            sample_traces_count=10,
        )

        report_path = write_html_report(
            out_path=args.out,
            metrics=metrics,
            top_transitions=score_result["top_transitions"],
            sample_traces=score_result["sample_traces"],
            source_csv=Path(args.csv_path),
        )

        elapsed = time.perf_counter() - t0
        print(f"\nTop {len(score_result['top_transitions'])} Broken Funnel Transitions:")
        for rank, cluster in enumerate(score_result["top_transitions"], start=1):
            print(
                f"  {rank}. {cluster.transition_label}  "
                f"({cluster.count:,} sessions | {cluster.pct_dropoffs:.1f}% | mean surprisal {cluster.mean_surprisal:.3f})"
            )
        print(f"\n[funnel-debug] Saved artifacts to {train_result['artifacts_dir']} and report to {report_path} in {elapsed:.1f}s.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
