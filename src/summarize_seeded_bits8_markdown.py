import argparse
from pathlib import Path
import sys

import pandas as pd

REQUIRED_COLUMNS = {"category", "image_auroc", "pixel_auroc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate 5 seed-level 8-bit MVTec CSV logs into a Markdown table "
            "with Image/Pixel AUROC Mean ± SD per category."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help=(
            "Exactly 5 CSV files (or glob patterns) for seed-level runs, "
            "for example: results/seed_*.csv"
        ),
    )
    return parser.parse_args()


def resolve_input_paths(raw_inputs: list[str]) -> list[str]:
    resolved: list[str] = []
    for item in raw_inputs:
        if any(ch in item for ch in "*?[]"):
            matches = sorted(str(path) for path in Path().glob(item))
            resolved.extend(matches)
        else:
            resolved.append(item)
    return resolved


def load_csv(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required column(s) in {path}: {missing_list}")

    return df[["category", "image_auroc", "pixel_auroc"]].copy()


def format_mean_sd(mean_value: float, std_value: float) -> str:
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.4f} ± {std_value:.4f}"


def build_markdown_table(df: pd.DataFrame) -> str:
    grouped = (
        df.groupby("category", as_index=True)
        .agg(
            image_mean=("image_auroc", "mean"),
            image_sd=("image_auroc", lambda x: x.std(ddof=1)),
            pixel_mean=("pixel_auroc", "mean"),
            pixel_sd=("pixel_auroc", lambda x: x.std(ddof=1)),
        )
        .sort_index()
    )

    lines = [
        "| Category | Image AUROC (Mean ± SD) | Pixel AUROC (Mean ± SD) |",
        "|---|---:|---:|",
    ]

    for category, row in grouped.iterrows():
        image_text = format_mean_sd(row["image_mean"], row["image_sd"])
        pixel_text = format_mean_sd(row["pixel_mean"], row["pixel_sd"])
        lines.append(f"| {category} | {image_text} | {pixel_text} |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if any("path/to/" in p.replace("\\", "/") for p in args.csv_files):
        print(
            "Error: placeholder path(s) detected. Replace path/to/seedX.csv with real CSV paths.",
            file=sys.stderr,
        )
        print(
            "Example: python src/summarize_seeded_bits8_markdown.py results/seed111.csv results/seed333.csv results/seed999.csv results/seed2026.csv results/seed3407.csv",
            file=sys.stderr,
        )
        sys.exit(2)

    csv_files = resolve_input_paths(args.csv_files)
    if len(csv_files) != 5:
        print(
            f"Error: expected exactly 5 CSV files after glob expansion, got {len(csv_files)}.",
            file=sys.stderr,
        )
        print(
            "Tip: pass 5 concrete files or a glob that expands to 5 files.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        frames = [load_csv(path) for path in csv_files]
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    markdown = build_markdown_table(merged)
    print(markdown)


if __name__ == "__main__":
    main()
