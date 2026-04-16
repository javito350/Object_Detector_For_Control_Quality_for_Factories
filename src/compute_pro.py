from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from skimage.measure import label

import generate_heatmaps as gh


ROOT = Path(__file__).resolve().parent
QUAL_DIR = ROOT / "qualitative_results"
OUT_CSV = ROOT / "pro_results.csv"


def normalize_map(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        if pred.shape[0] == 1:
            pred = pred[0]
        elif pred.shape[-1] == 1:
            pred = pred[..., 0]
        elif pred.shape[0] <= 4:
            pred = pred.mean(axis=0)
        else:
            pred = pred.mean(axis=-1)

    mn, mx = float(pred.min()), float(pred.max())
    if mx > mn:
        pred = (pred - mn) / (mx - mn)
    else:
        pred = np.zeros_like(pred, dtype=np.float32)
    return pred.astype(np.float32)


def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def main() -> None:
    pat = re.compile(r"^(?P<cat>[a-z_]+)_heatmap_rank(?P<rank>[1-9]\d*)\.npy$")
    by_cat: dict[str, dict[int, Path]] = {}

    for p in QUAL_DIR.glob("*.npy"):
        match = pat.match(p.name)
        if match:
            category = match.group("cat")
            rank = int(match.group("rank"))
            by_cat.setdefault(category, {})[rank] = p

    if not by_cat:
        raise RuntimeError("No .npy files found in qualitative_results/")

    rows: list[dict[str, float | str]] = []

    for cat, rank_map in sorted(by_cat.items()):
        print(f"Calculating REAL PRO for: {cat}...")
        top_paths = gh.recover_topk_paths_for_category(cat, max(rank_map.keys()))

        preds: list[np.ndarray] = []
        gts: list[np.ndarray] = []

        for rank, npy_p in sorted(rank_map.items()):
            img_p = top_paths[rank - 1]

            pred = normalize_map(np.load(npy_p))
            gt = gh.load_ground_truth_mask(gh.DATA_DIR / cat, img_p, pred.shape)

            preds.append(pred)
            gts.append(gt.astype(np.uint8))

        all_vals = np.concatenate([p.ravel() for p in preds])
        t_min, t_max = float(all_vals.min()), float(all_vals.max())

        if t_max <= t_min:
            rows.append({"category": cat, "pro_auc_fpr0.3": 0.0})
            continue

        thresholds = np.linspace(t_min, t_max, 300)
        fprs: list[float] = []
        pros: list[float] = []

        for threshold in thresholds:
            fp = 0
            neg = 0
            overlaps: list[float] = []

            for pred, gt in zip(preds, gts):
                pred_bin = pred >= threshold
                gt_bin = gt > 0

                fp += int(np.logical_and(pred_bin, ~gt_bin).sum())
                neg += int((~gt_bin).sum())

                cc = label(gt_bin.astype(np.uint8), connectivity=1)
                for region_id in range(1, int(cc.max()) + 1):
                    region = cc == region_id
                    if region.any():
                        overlaps.append(float(pred_bin[region].mean()))

            fpr = (fp / neg) if neg > 0 else 0.0
            if fpr <= 0.3:
                fprs.append(fpr)
                pros.append(float(np.mean(overlaps)) if overlaps else 0.0)

        if len(fprs) < 2:
            auc_score = 0.0
        else:
            order = np.argsort(fprs)
            fprs_sorted = np.asarray(fprs)[order]
            pros_sorted = np.asarray(pros)[order]
            auc_score = integrate_trapezoid(pros_sorted, fprs_sorted / 0.3)

        rows.append({"category": cat, "pro_auc_fpr0.3": auc_score})

    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    df.loc[len(df)] = ["macro_avg", float(df["pro_auc_fpr0.3"].mean())]
    df.to_csv(OUT_CSV, index=False)

    print("\n--- RIGOROUS PRO RESULTS ---")
    print(df)
    print(f"\nSaved: {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
