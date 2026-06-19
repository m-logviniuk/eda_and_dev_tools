"""Exploratory data analysis report generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from abalone.config import (
    EDA_SUMMARY_PATH,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from abalone.data import clean_raw_data, load_raw_data


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[TARGET_COLUMN], bins=30, color="steelblue", edgecolor="white")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of abalone age")
    _save_figure(fig, output_dir / "age_distribution.png")


def plot_feature_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes_flat = axes.flatten()
    columns = NUMERIC_FEATURES + ["Sex"]

    for ax, column in zip(axes_flat, columns, strict=False):
        if column == "Sex":
            counts = df[column].value_counts()
            ax.bar(counts.index, counts.values, color="slategray")
            ax.set_title(column)
        else:
            ax.hist(df[column], bins=25, color="steelblue", edgecolor="white")
            ax.set_title(column)

    for ax in axes_flat[len(columns):]:
        ax.axis("off")

    fig.suptitle("Feature distributions", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir / "feature_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    corr = df[NUMERIC_FEATURES + [TARGET_COLUMN]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    _save_figure(fig, output_dir / "correlation_heatmap.png")


def plot_sex_vs_age(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Sex", y=TARGET_COLUMN, ax=ax)
    ax.set_xlabel("Sex")
    ax.set_ylabel("Age (years)")
    ax.set_title("Age by sex")
    _save_figure(fig, output_dir / "sex_vs_age.png")


def plot_pairplot_sample(df: pd.DataFrame, output_dir: Path) -> None:
    sample_columns = ["Length", "Diameter", "Whole_weight", TARGET_COLUMN]
    sample = df[sample_columns].sample(n=min(300, len(df)), random_state=42)
    pairplot = sns.pairplot(sample, diag_kind="hist", corner=False)
    pairplot.figure.suptitle("Pairwise relationships (sample)", y=1.02)
    _save_figure(pairplot.figure, output_dir / "pairplot_sample.png")


def write_eda_summary(df: pd.DataFrame, path: Path = EDA_SUMMARY_PATH) -> None:
    """Write a short markdown summary of EDA findings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = df.isna().sum()
    zero_height = int((df["Height"] == 0).sum())

    lines = [
        "# EDA summary",
        "",
        "## Dataset",
        f"- Rows: {len(df)}",
        f"- Features: {len(df.columns) - 1} (+ target `{TARGET_COLUMN}`)",
        "",
        "## Data quality",
        f"- Missing values handled: {int(missing.sum())} total before cleaning",
        f"- Zero `Height` values replaced: {zero_height}",
        "- `Sex` values normalized (`f` → `F`)",
        "",
        "## Target",
        f"- Age mean: {df[TARGET_COLUMN].mean():.2f}",
        f"- Age std: {df[TARGET_COLUMN].std():.2f}",
        f"- Age range: [{df[TARGET_COLUMN].min():.1f}, {df[TARGET_COLUMN].max():.1f}]",
        "",
        "## Notes",
        "- Age is derived as `Rings + 1.5`.",
        "- Shell dimensions and weights are expected to correlate with age.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_eda_report(
    source: str | Path | None = None,
    output_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Generate EDA figures and summary markdown."""
    df = clean_raw_data(load_raw_data(source))

    plot_target_distribution(df, output_dir)
    plot_feature_distributions(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_sex_vs_age(df, output_dir)
    plot_pairplot_sample(df, output_dir)
    write_eda_summary(df)

    return df


def main() -> None:
    """Generate all EDA artifacts."""
    generate_eda_report()
    print(f"EDA figures saved to {FIGURES_DIR}")
    print(f"EDA summary saved to {EDA_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
