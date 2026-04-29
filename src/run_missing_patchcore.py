from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path

ALL_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

REPO_ROOT = Path(__file__).resolve().parent
OUT_CSV = REPO_ROOT / "results" / "patchcore_n10_seed42_15cat.csv"
LOG_DIR = REPO_ROOT / "results" / "patchcore_logs"
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
AUROC_PATTERN = re.compile(r"Image AUROC:\s*([0-9]*\.?[0-9]+)")


def parse_auroc(text: str) -> float | None:
    matches = AUROC_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def load_existing(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    if not path.exists():
        return values

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return values
        for row in reader:
            if len(row) != 2:
                continue
            category, val = row[0].strip(), row[1].strip()
            if category in {"OVERALL_MEAN", "OVERALL"}:
                continue
            if category in ALL_CATEGORIES:
                values[category] = float(val)
    return values


def run_category(category: str) -> float:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{category}.txt"

    cmd = [
        str(PYTHON_EXE),
        "src/evaluate_patchcore.py",
        "--category",
        category,
        "--num-workers",
        "0",
    ]
    print(f"RUNNING {category}")
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    log_text = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log_text, encoding="utf-8")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Category {category} failed with code {proc.returncode}. See {log_path}."
        )

    parsed = parse_auroc(log_text)
    if parsed is None:
        raise RuntimeError(f"Could not parse Image AUROC for {category}. See {log_path}.")

    value = parsed
    print(f"{category},{value:.6f}")
    return value


def load_from_log(category: str) -> float | None:
    log_path = LOG_DIR / f"{category}.txt"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return None
    return parse_auroc(text)


def write_output(path: Path, values: dict[str, float]) -> float:
    ordered = [values[c] for c in ALL_CATEGORIES]
    overall = sum(ordered) / len(ordered)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "image_auroc"])
        for c in ALL_CATEGORIES:
            writer.writerow([c, f"{values[c]:.6f}"])
        writer.writerow(["OVERALL_MEAN", f"{overall:.6f}"])

    return overall


def main() -> None:
    values = load_existing(OUT_CSV)
    missing = [c for c in ALL_CATEGORIES if c not in values]

    print(f"Found {len(values)} existing categories, {len(missing)} missing.")
    for category in missing:
        cached = load_from_log(category)
        if cached is not None:
            values[category] = cached
            print(f"{category},{cached:.6f} (from log)")
            continue
        values[category] = run_category(category)

    still_missing = [c for c in ALL_CATEGORIES if c not in values]
    if still_missing:
        raise RuntimeError(f"Missing categories after run: {still_missing}")

    overall = write_output(OUT_CSV, values)
    print(f"OVERALL_MEAN,{overall:.6f}")
    print(f"DONE: {OUT_CSV}")


if __name__ == "__main__":
    main()
