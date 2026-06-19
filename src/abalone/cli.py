"""Command-line interface for the abalone project."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Abalone age prediction pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", help="Cross-validate models and train the winner")
    subparsers.add_parser("evaluate", help="Evaluate the saved model on the test set")
    subparsers.add_parser("eda", help="Generate EDA figures and summary")
    subparsers.add_parser(
        "build-dashboard",
        help="Build ExplainerDashboard YAML and explainer artifacts",
    )
    subparsers.add_parser("serve", help="Serve the ExplainerDashboard web app")

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run train, evaluate, EDA, and dashboard build",
    )
    pipeline_parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA figure generation",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        from abalone.train import train_and_select_model

        train_and_select_model()
    elif args.command == "evaluate":
        from abalone.evaluate import run_evaluation

        run_evaluation()
    elif args.command == "eda":
        from abalone.eda_report import generate_eda_report

        generate_eda_report()
    elif args.command == "build-dashboard":
        from abalone.dashboard import build_dashboard_config

        build_dashboard_config()
    elif args.command == "serve":
        from abalone.dashboard import serve_dashboard

        serve_dashboard()
    elif args.command == "pipeline":
        from abalone.dashboard import build_dashboard_config
        from abalone.eda_report import generate_eda_report
        from abalone.evaluate import run_evaluation
        from abalone.train import train_and_select_model

        train_and_select_model()
        run_evaluation()
        if not args.skip_eda:
            generate_eda_report()
        build_dashboard_config()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
