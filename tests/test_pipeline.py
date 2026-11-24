"""End-to-end integration tests for training, surprisal scoring, report generation, and CLI."""

from pathlib import Path

from data.generate_sample import generate_dataset
from src.cli import main as cli_main
from src.score import score_non_converting_sessions
from src.train import train_funnel_gpt


def test_end_to_end_pipeline_surfaces_planted_bugs(tmp_path: Path) -> None:
    csv_path = tmp_path / "mini_events.csv"
    artifacts_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.html"

    generate_dataset(output_path=csv_path, n_converting=350, n_abandoned=150, seed=42)

    train_res = train_funnel_gpt(
        csv_path=csv_path,
        goal_event="purchase",
        epochs=4,
        batch_size=32,
        artifacts_dir=artifacts_dir,
        seed=42,
    )

    assert (artifacts_dir / "model.pt").exists()
    assert (artifacts_dir / "tokenizer.json").exists()
    assert (artifacts_dir / "metrics.json").exists()
    assert train_res["metrics"]["converting_sessions"] == 350
    assert train_res["metrics"]["non_converting_sessions"] == 150

    score_res = score_non_converting_sessions(
        model=train_res["model"],
        tokenizer=train_res["tokenizer"],
        non_converting_sessions=train_res["non_converting_sessions"],
        p95_threshold=train_res["p95_threshold"],
        device=train_res["device"],
        top_k=5,
        sample_traces_count=10,
    )

    top_labels = [c.transition_label for c in score_res["top_transitions"]]
    assert len(top_labels) >= 3
    assert any("error_payment" in label for label in top_labels), f"Missing payment error bug in {top_labels}"
    assert any("add_shipping" in label and "page_view" in label for label in top_labels), f"Missing shipping bounce bug in {top_labels}"
    assert any("add_to_cart" in label and "DT_LONG" in label for label in top_labels), f"Missing cart stall bug in {top_labels}"
    assert len(score_res["sample_traces"]) == 10

    # Verify CLI end-to-end
    exit_code = cli_main(
        [
            "run",
            str(csv_path),
            "--goal",
            "purchase",
            "--epochs",
            "2",
            "--out",
            str(report_path),
            "--artifacts-dir",
            str(tmp_path / "cli_artifacts"),
        ]
    )
    assert exit_code == 0
    assert report_path.exists()
    html_content = report_path.read_text(encoding="utf-8")
    assert "Funnel-GPT Diagnostic Report" in html_content
    assert "Top 5 Broken Funnel Transitions" in html_content
    assert "Color-Coded Session Traces" in html_content
