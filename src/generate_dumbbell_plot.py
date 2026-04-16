"""Generate Figure 4.2 p4m ablation dumbbell plot.

This script compares category-wise image AUROC between:
- Baseline (no p4m): results_ablation_no_p4m.csv
- Ours (8-bit + p4m): trilemma_results.csv filtered at Bits == 8

It saves a horizontal Cleveland-style dumbbell plot as:
fig_4_2_p4m_ablation.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT_DIR = Path(__file__).resolve().parent
BASELINE_CSV = ROOT_DIR / "results_ablation_no_p4m.csv"
TRILEMMA_CSV = ROOT_DIR / "trilemma_results.csv"
OUTPUT_PATH = ROOT_DIR / "fig_4_2_p4m_ablation.png"


def main() -> None:
    baseline_df = pd.read_csv(BASELINE_CSV)
    ours_df = pd.read_csv(TRILEMMA_CSV)

    baseline_df = baseline_df.rename(
        columns={
            "category": "Category",
            "image_auroc": "Baseline_AUROC",
        }
    )[["Category", "Baseline_AUROC"]]

    ours_df["Bits_num"] = pd.to_numeric(ours_df["Bits"], errors="coerce")
    ours_df = ours_df.loc[ours_df["Bits_num"] == 8, ["Category", "AUROC"]].rename(
        columns={"AUROC": "Ours_AUROC"}
    )

    merged = pd.merge(baseline_df, ours_df, on="Category", how="inner")
    if merged.empty:
        raise ValueError("No overlapping categories found between baseline and 8-bit results.")

    merged["Delta"] = merged["Ours_AUROC"] - merged["Baseline_AUROC"]
    merged = merged.sort_values("Delta", ascending=True).reset_index(drop=True)

    y = list(range(len(merged)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, row in merged.iterrows():
        ax.plot(
            [row["Baseline_AUROC"], row["Ours_AUROC"]],
            [idx, idx],
            color="#B0B0B0",
            linewidth=2.0,
            zorder=1,
        )

    ax.scatter(
        merged["Baseline_AUROC"],
        y,
        color="#7A7A7A",
        s=48,
        label="Baseline (No p4m)",
        zorder=2,
    )
    ax.scatter(
        merged["Ours_AUROC"],
        y,
        color="#1F77B4",
        s=48,
        label="Ours (8-bit + p4m)",
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(merged["Category"])
    ax.set_xlabel("Image-level AUROC")
    ax.set_ylabel("Category")
    ax.set_title("Figure 4.2: p4m Ablation (Baseline vs Ours)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    min_x = min(merged["Baseline_AUROC"].min(), merged["Ours_AUROC"].min())
    max_x = max(merged["Baseline_AUROC"].max(), merged["Ours_AUROC"].max())
    pad = max(0.01, (max_x - min_x) * 0.08)
    ax.set_xlim(max(0.0, min_x - pad), min(1.0, max_x + pad))

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
