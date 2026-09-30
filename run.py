"""Install dependencies, build both indexes, evaluate retrieval, and launch the app."""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"


def run_step(label: str, command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up YASCO, build indexes, evaluate retrieval, and launch Gradio."
    )
    parser.add_argument(
        "--skip-install", action="store_true",
        help="Skip pip installation when dependencies are already installed.",
    )
    parser.add_argument(
        "--evaluation-output", default=str(ROOT / "results" / "evaluation.txt"),
        help="Path for the retrieval evaluation report.",
    )
    args = parser.parse_args()

    if not args.skip_install:
        run_step(
            "Install Python dependencies",
            [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        )

    run_step("Build BM25 lexical index", [sys.executable, str(ROOT / "src" / "bm25.py")])
    run_step("Build dense embedding index", [sys.executable, str(ROOT / "src" / "embed.py")])
    run_step(
        "Evaluate retrieval",
        [
            sys.executable,
            str(ROOT / "src" / "evals.py"),
            "--out",
            args.evaluation_output,
        ],
    )
    run_step("Launch Gradio app", [sys.executable, str(ROOT / "src" / "generate.py")])


if __name__ == "__main__":
    main()
