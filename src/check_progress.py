from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check live WinCLIP benchmark progress.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("results/winclip_mvtec_results.csv"),
        help="Path to the live benchmark CSV.",
    )
    parser.add_argument(
        "--n-shot",
        type=int,
        default=1,
        help="N-shot setting to analyze (default: 1).",
    )
    parser.add_argument(
        "--expected-categories",
        type=int,
        default=15,
        help="Expected number of categories per seed for completion (default: 15).",
    )
    return parser.parse_args()


def safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Read-only mode to avoid interfering with a live writer.
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        seed_scores: dict[int, list[float]] = defaultdict(list)
        seed_categories: dict[int, set[str]] = defaultdict(set)
        category_scores: dict[str, list[float]] = defaultdict(list)

        for row in reader:
            n_value = safe_int(row.get("N-shot") or row.get("N"))
            if n_value != args.n_shot:
                continue

            seed = safe_int(row.get("Seed"))
            category = (row.get("Category") or "").strip()
            score = safe_float(row.get("Image AUROC") or row.get("AUROC"))

            if seed is None or category == "" or score is None:
                # Skip partial/corrupted rows while benchmark is actively writing.
                continue

            seed_scores[seed].append(score)
            seed_categories[seed].add(category)
            category_scores[category].append(score)

    if not seed_scores:
        print(f"No valid rows found for N={args.n_shot} in {csv_path}.")
        return

    completed_seeds = [
        seed for seed in sorted(seed_scores) if len(seed_categories[seed]) >= args.expected_categories
    ]

    print(f"Progress snapshot for N={args.n_shot} from: {csv_path}")
    print(f"Total valid rows used: {sum(len(v) for v in seed_scores.values())}")

    if completed_seeds:
        print("\nAverage Image AUROC for completed seeds:")
        for seed in completed_seeds:
            avg = mean(seed_scores[seed])
            print(
                f"  Seed {seed}: {avg:.4f} "
                f"(categories={len(seed_categories[seed])}/{args.expected_categories})"
            )
    else:
        print("\nNo seed has all expected categories yet.")

    print("\nAverage Image AUROC for observed seeds (including in-progress):")
    for seed in sorted(seed_scores):
        avg = mean(seed_scores[seed])
        print(
            f"  Seed {seed}: {avg:.4f} "
            f"(categories={len(seed_categories[seed])}/{args.expected_categories})"
        )

    best_category, best_values = max(category_scores.items(), key=lambda kv: mean(kv[1]))
    worst_category, worst_values = min(category_scores.items(), key=lambda kv: mean(kv[1]))

    print("\nCategory ranking snapshot (N=1 rows seen so far):")
    print(f"  Best category: {best_category} (avg AUROC={mean(best_values):.4f}, n={len(best_values)})")
    print(
        f"  Bottleneck category: {worst_category} "
        f"(avg AUROC={mean(worst_values):.4f}, n={len(worst_values)})"
    )


if __name__ == "__main__":
    main()
