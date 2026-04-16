from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.measure import label
from PIL import Image
import torch

import generate_heatmaps_visa as gh
from models.anomaly_inspector import EnhancedAnomalyInspector

ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "pro_results_full_visa.csv"

def normalize_map(pred: np.ndarray) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred.mean(axis=0 if pred.shape[0] <= 4 else -1)
    mn, mx = float(pred.min()), float(pred.max())
    if mx > mn:
        return (pred - mn) / (mx - mn)
    return np.zeros_like(pred, dtype=np.float32)

def integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))

def load_mask_robust(category_root: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    if "good" in image_path.parts:
        return np.zeros(output_size, dtype=np.uint8)

    gt_root = category_root / "ground_truth"
    image_id = image_path.stem

    folders_to_try = ["anomaly", "bad", "defect", image_path.parent.name]
    suffixes_to_try = [".png", "_mask.png"]

    # 1. Try combinations
    for folder in folders_to_try:
        for suffix in suffixes_to_try:
            candidate = gt_root / folder / f"{image_id}{suffix}"
            if candidate.exists():
                mask = Image.open(candidate).convert("L").resize((output_size[1], output_size[0]), Image.NEAREST)
                return (np.asarray(mask) > 0).astype(np.uint8)

    # 2. Try recursive search
    for candidate in gt_root.rglob(f"{image_id}*.png"):
        if candidate.is_file():
            mask = Image.open(candidate).convert("L").resize((output_size[1], output_size[0]), Image.NEAREST)
            return (np.asarray(mask) > 0).astype(np.uint8)

    print(f"  [WARNING] Missing mask for: {image_path.name}")
    return np.zeros(output_size, dtype=np.uint8)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["candle", "macaroni1", "pcb4"])
    args = parser.parse_args()

    rows = []
    transform = gh.build_transform()

    for cat in args.categories:
        print(f"\nEvaluating FULL TEST SET for: {cat}...")
        cat_root = gh.DATA_DIR / cat
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inspector = EnhancedAnomalyInspector(backbone=gh.BACKBONE, device=device, use_pq=True)
        
        # Build memory bank
        support_loader = gh.build_support_loader(cat_root, transform)
        inspector.fit(support_loader, apply_p4m=True)
        
        # Evaluate every image in the test set
        test_loader = gh.build_test_loader(cat_root, transform)
        preds, gts = [], []
        
        for images, _, paths in test_loader:
            img_p = Path(paths[0])
            # Predict
            result = inspector.predict(images, apply_p4m=False)[0]
            pred = normalize_map(result.anomaly_map)
            # Load Mask
            gt = load_mask_robust(cat_root, img_p, pred.shape)
            
            preds.append(pred)
            gts.append(gt.astype(np.uint8))
            
        all_vals = np.concatenate([p.ravel() for p in preds])
        t_min, t_max = float(all_vals.min()), float(all_vals.max())

        if t_max <= t_min:
            rows.append({"category": cat, "pro_auc_fpr0.3": 0.0})
            continue

        thresholds = np.linspace(t_min, t_max, 300)
        fprs, pros = [], []

        for threshold in thresholds:
            fp, neg = 0, 0
            overlaps = []

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
            auc_score = integrate_trapezoid(np.asarray(pros)[order], np.asarray(fprs)[order] / 0.3)

        rows.append({"category": cat, "pro_auc_fpr0.3": auc_score})
        print(f"  -> PRO-AUC: {auc_score:.4f}")

    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    df.loc[len(df)] = ["macro_avg", float(df["pro_auc_fpr0.3"].mean())]
    df.to_csv(OUT_CSV, index=False)

    print("\n--- FINAL FULL DATASET PRO RESULTS ---")
    print(df)

if __name__ == "__main__":
    main()