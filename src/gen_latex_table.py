from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table comparing WinCLIP and Ours on 1-shot Image AUROC."
    )
    parser.add_argument(
        "--winclip-csv",
        type=Path,
        default=Path("results/winclip_mvtec_results.csv"),
        help="Path to WinCLIP CSV.",
    )
    parser.add_argument(
        "--ours-csv",
        type=Path,
        default=Path("results/nshot_sensitivity.csv"),
        help="Path to PatchCore/Ours CSV.",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("results/latex_table_1shot.tex"),
        help="Output .tex path.",
    )
    return parser.parse_args()


def safe_int(text: str | None) -> int | None:
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    try:
        return int(float(t))
    except ValueError:
        return None


def safe_float(text: str | None) -> float | None:
    if text is None:
        return None
    t = str(text).strip()
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None


def normalize_auroc(value: float) -> float:
    # Some files store AUROC in [0,1], others in [0,100].
    if value > 1.0:
        return value / 100.0
    return value


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def fmt_pct(x01: float) -> str:
    return f"{x01 * 100.0:.2f}"


def collect_winclip_oneshot(csv_path: Path) -> dict[str, list[float]]:
    by_cat: dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_shot = safe_int(row.get("N-shot") or row.get("N"))
            category = (row.get("Category") or "").strip()
            score = safe_float(row.get("Image AUROC") or row.get("AUROC"))
            if n_shot != 1 or category == "" or score is None:
                continue
            by_cat.setdefault(category, []).append(normalize_auroc(score))
    return by_cat


def collect_ours_oneshot(csv_path: Path) -> dict[str, list[float]]:
    by_cat: dict[str, list[float]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_shot = safe_int(row.get("N-shot") or row.get("N"))
            category = (row.get("Category") or "").strip()
            score = safe_float(row.get("Image AUROC") or row.get("AUROC"))
            if n_shot != 1 or category == "" or score is None:
                continue
            by_cat.setdefault(category, []).append(normalize_auroc(score))
    return by_cat


def build_latex_table(winclip: dict[str, list[float]], ours: dict[str, list[float]]) -> str:
    categories = sorted(set(winclip.keys()) & set(ours.keys()))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Category & WinCLIP (1-shot) & Ours (1-shot) & $\Delta$ \\")
    lines.append(r"\midrule")

    winclip_row_means: list[float] = []
    ours_row_means: list[float] = []

    for cat in categories:
        wc = mean(winclip[cat])
        oc = mean(ours[cat])
        delta = oc - wc

        winclip_row_means.append(wc)
        ours_row_means.append(oc)

        wc_text = fmt_pct(wc)
        oc_text = fmt_pct(oc)
        if oc >= wc:
            oc_text = rf"\textbf{{{oc_text}}}"
        else:
            wc_text = rf"\textbf{{{wc_text}}}"

        delta_text = f"{delta * 100.0:+.2f}"
        lines.append(f"{latex_escape(cat)} & {wc_text} & {oc_text} & {delta_text} " + r"\\")

    lines.append(r"\midrule")
    mean_wc = mean(winclip_row_means)
    mean_oc = mean(ours_row_means)
    mean_delta = mean_oc - mean_wc

    mean_wc_text = fmt_pct(mean_wc)
    mean_oc_text = fmt_pct(mean_oc)
    if mean_oc >= mean_wc:
        mean_oc_text = rf"\textbf{{{mean_oc_text}}}"
    else:
        mean_wc_text = rf"\textbf{{{mean_wc_text}}}"

    lines.append(f"Mean & {mean_wc_text} & {mean_oc_text} & {mean_delta * 100.0:+.2f} " + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines) + "\n"


def build_markdown_table(winclip: dict[str, list[float]], ours: dict[str, list[float]]) -> str:
    categories = sorted(set(winclip.keys()) & set(ours.keys()))

    lines: list[str] = []
    lines.append("| Category | WinCLIP (1-shot) | Ours (1-shot) | Delta |")
    lines.append("|---|---:|---:|---:|")

    winclip_row_means: list[float] = []
    ours_row_means: list[float] = []

    for cat in categories:
        wc = mean(winclip[cat])
        oc = mean(ours[cat])
        delta = oc - wc

        winclip_row_means.append(wc)
        ours_row_means.append(oc)

        wc_text = fmt_pct(wc)
        oc_text = fmt_pct(oc)
        if oc >= wc:
            oc_text = f"**{oc_text}**"
        else:
            wc_text = f"**{wc_text}**"

        delta_text = f"{delta * 100.0:+.2f}"
        lines.append(f"| {cat} | {wc_text} | {oc_text} | {delta_text} |")

    mean_wc = mean(winclip_row_means)
    mean_oc = mean(ours_row_means)
    mean_delta = mean_oc - mean_wc

    mean_wc_text = fmt_pct(mean_wc)
    mean_oc_text = fmt_pct(mean_oc)
    if mean_oc >= mean_wc:
        mean_oc_text = f"**{mean_oc_text}**"
    else:
        mean_wc_text = f"**{mean_wc_text}**"

    lines.append(f"| Mean | {mean_wc_text} | {mean_oc_text} | {mean_delta * 100.0:+.2f} |")
    lines.append("")
    lines.append("{ #tbl-winclip-comparison }")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    if not args.winclip_csv.exists():
        raise FileNotFoundError(f"Missing WinCLIP CSV: {args.winclip_csv}")
    if not args.ours_csv.exists():
        raise FileNotFoundError(f"Missing Ours CSV: {args.ours_csv}")

    winclip = collect_winclip_oneshot(args.winclip_csv)
    ours = collect_ours_oneshot(args.ours_csv)

    if not winclip:
        raise RuntimeError("No 1-shot rows found for WinCLIP.")
    if not ours:
        raise RuntimeError("No 1-shot rows found for Ours.")

    categories = sorted(set(winclip.keys()) & set(ours.keys()))
    if not categories:
        raise RuntimeError("No overlapping categories between WinCLIP and Ours for 1-shot.")

    latex = build_latex_table(winclip, ours)
    markdown = build_markdown_table(winclip, ours)

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(latex, encoding="utf-8")

    print(latex)
    print("Markdown table:")
    print(markdown)
    print(f"Saved LaTeX table to: {args.output_tex}")
    print(f"Overlapping categories used: {len(categories)}")


if __name__ == "__main__":
    main()
