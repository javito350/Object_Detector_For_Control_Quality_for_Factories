"""
Compute PRO-AUC metrics for subquantizer ablation study (bottle, metal_nut, screw)
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import label
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
QUAL_DIR = ROOT / "qualitative_results_m_ablation"
OUT_CSV = ROOT / "results" / "pro_auc_m_ablation.csv"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

MVTEC_ROOT = ROOT / "data" / "mvtec"
CATEGORIES = ["bottle", "metal_nut", "screw"]
M_VALUES = [16, 32, 64]


def normalize_map(pred: np.ndarray) -> np.ndarray:
    """Normalize prediction map to [0, 1]."""
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
    """Compute trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def load_ground_truth_mask(category_root: Path, image_path: Path, output_size: tuple[int, int]) -> np.ndarray:
    """Load ground truth mask, handling both good and defect cases."""
    if image_path.parent.name == "good":
        return np.zeros(output_size, dtype=np.uint8)
    
    defect_type = image_path.parent.name
    mask_path = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
    
    if not mask_path.exists():
        print(f"[WARNING] Mask not found: {mask_path}")
        return np.zeros(output_size, dtype=np.uint8)
    
    mask = Image.open(mask_path).convert("L").resize((output_size[1], output_size[0]), Image.NEAREST)
    return (np.asarray(mask) > 0).astype(np.uint8)


def get_top_k_image_paths(category: str, k: int) -> list[Path]:
    """Get the top-k scoring images for a category (deterministic ordering)."""
    test_dir = MVTEC_ROOT / category / "test"
    images = []
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir():
            for img_file in sorted(defect_dir.glob("*.png")):
                images.append(img_file)
    return images[:k]


def compute_pro_for_m_value(m_value: int, category: str) -> dict:
    """Compute PRO-AUC for a specific M value and category."""
    m_subdir = QUAL_DIR / f"m_{m_value}"
    
    # Pattern to match heatmap files
    pat = re.compile(rf"^{re.escape(category)}_m{m_value}_heatmap_rank(?P<rank>[1-9]\d*)\.npy$")
    
    rank_map = {}
    for p in m_subdir.glob("*.npy"):
        match = pat.match(p.name)
        if match:
            rank = int(match.group("rank"))
            rank_map[rank] = p
    
    if not rank_map:
        print(f"[WARNING] No .npy files found for {category} M={m_value} in {m_subdir}")
        return {
            "category": category,
            "m_value": m_value,
            "pro_auc_fpr0.3": np.nan,
            "num_samples": 0,
        }
    
    # Get the corresponding image paths
    top_paths = get_top_k_image_paths(category, max(rank_map.keys()))
    category_root = MVTEC_ROOT / category
    
    preds = []
    gts = []
    
    for rank, npy_p in sorted(rank_map.items()):
        if rank > len(top_paths):
            print(f"[WARNING] Rank {rank} exceeds available paths ({len(top_paths)})")
            continue
        
        img_p = top_paths[rank - 1]
        
        pred = normalize_map(np.load(npy_p))
        gt = load_ground_truth_mask(category_root, img_p, pred.shape)
        
        preds.append(pred)
        gts.append(gt.astype(np.uint8))
    
    if not preds:
        print(f"[WARNING] No valid predictions for {category} M={m_value}")
        return {
            "category": category,
            "m_value": m_value,
            "pro_auc_fpr0.3": np.nan,
            "num_samples": 0,
        }
    
    # Compute PRO score
    all_vals = np.concatenate([p.ravel() for p in preds])
    t_min, t_max = float(all_vals.min()), float(all_vals.max())
    
    if t_max <= t_min:
        print(f"[WARNING] No variation in predictions for {category} M={m_value}")
        return {
            "category": category,
            "m_value": m_value,
            "pro_auc_fpr0.3": 0.0,
            "num_samples": len(preds),
        }
    
    thresholds = np.linspace(t_min, t_max, 300)
    fprs = []
    pros = []
    
    for threshold in thresholds:
        fp = 0
        neg = 0
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
        fprs_sorted = np.asarray(fprs)[order]
        pros_sorted = np.asarray(pros)[order]
        auc_score = integrate_trapezoid(pros_sorted, fprs_sorted / 0.3)
    
    return {
        "category": category,
        "m_value": m_value,
        "pro_auc_fpr0.3": auc_score,
        "num_samples": len(preds),
    }


def main():
    print("=" * 70)
    print("Computing PRO-AUC for Subquantizer Ablation Study")
    print("=" * 70)
    
    rows = []
    
    for category in CATEGORIES:
        for m_value in M_VALUES:
            print(f"\nComputing PRO-AUC for {category} M={m_value}...")
            result = compute_pro_for_m_value(m_value, category=category)
            rows.append(result)
            print(f"  PRO-AUC (FPR@0.3): {result['pro_auc_fpr0.3']:.4f}")
            print(f"  Num samples: {result['num_samples']}")
    
    df = pd.DataFrame(rows)
    df = df.sort_values(["category", "m_value"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS: Subquantizer Ablation PRO-AUC")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nSaved to: {OUT_CSV}")
    
    # Extract results for bottle and metal_nut at M=16
    print("\n" + "=" * 70)
    print("REQUESTED RESULTS: Bottle & Metal_Nut at M=16")
    print("=" * 70)
    for cat in ["bottle", "metal_nut"]:
        row = df[(df["category"] == cat) & (df["m_value"] == 16)]
        if not row.empty:
            pro_auc = row["pro_auc_fpr0.3"].values[0]
            print(f"{cat:12s} @ M=16: PRO-AUC = {pro_auc:.4f}")
        else:
            print(f"{cat:12s} @ M=16: NOT FOUND")
    
    return df


if __name__ == "__main__":
    main()
