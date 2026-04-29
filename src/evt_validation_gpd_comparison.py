"""
evt_validation_gpd_comparison.py
=================================
EVT Validation: Compare GPD shape parameters (ξ) across 4-bit vs 8-bit quantization.

Purpose:
    Demonstrate that the sign inversion in GPD shape (ξ) from 4-bit to 8-bit
    is consistent across multiple MVTec categories and seeds.
    
    Expected behavior:
        - 4-bit: ξ ≈ +0.008  (heavy-tailed, quantization collapse)
        - 8-bit: ξ ≈ -0.067  (light-tailed, stable quantization)
    
    This inversion indicates that 4-bit quantization creates artificial
    long-tailed distance distributions (due to geometry collapse), while
    8-bit maintains a well-behaved tail.

Output:
    results/evt_validation_gpd_comparison.csv
    - Columns: category, seed, bit_depth, gpd_shape_xi, gpd_scale_sigma
    - Aggregated summary showing mean ± std across seeds for each category
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FINAL_CSV_EXPORTS = PROJECT_ROOT / "final_csv_exports"

# List of categories to compare (at least 5)
TARGET_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "metal_nut",
    "screw",
]

# List of seeds to analyze (should be ≥ 5)
SEEDS = [111, 333, 999, 2026, 3407]

# Output file
OUTPUT_CSV = RESULTS_DIR / "evt_validation_gpd_comparison.csv"


def load_evt_stability_results(seed: int, category: str) -> Optional[pd.DataFrame]:
    """
    Load EVT quantization stability results for a given seed and category.
    
    Expected file structure:
        {PROJECT_ROOT}/results/evt_quantization_stability_{seed}_{category}.csv
    
    Or fallback to aggregated file if available:
        {FINAL_CSV_EXPORTS}/evt_quantization_stability.csv
    """
    # Try seed-specific file first
    seed_file = RESULTS_DIR / f"evt_quantization_stability_seed{seed}_{category}.csv"
    if seed_file.exists():
        log.info(f"Loading {seed_file.name}")
        return pd.read_csv(seed_file)
    
    # Try aggregated file
    agg_file = FINAL_CSV_EXPORTS / "evt_quantization_stability.csv"
    if agg_file.exists():
        log.info(f"Loading aggregated data from {agg_file.name}")
        df = pd.read_csv(agg_file)
        # Filter by seed and category if applicable
        if "seed" in df.columns and "category" in df.columns:
            df = df[(df["seed"] == seed) & (df["category"] == category)]
            if not df.empty:
                return df
    
    log.warning(f"No EVT data found for seed={seed}, category={category}")
    return None


def extract_gpd_parameters(
    category: str,
    bit_depths: list[int] = [4, 8],
) -> dict:
    """
    Extract GPD shape (ξ) and scale (σ) parameters across seeds for given bit depths.
    
    Returns:
        {
            "4bit": {"xis": [...], "sigmas": [...], "mean_xi": ..., "std_xi": ...},
            "8bit": {"xis": [...], "sigmas": [...], "mean_xi": ..., "std_xi": ...},
            ...
        }
    """
    results = {}
    
    for bits in bit_depths:
        xis = []
        sigmas = []
        
        for seed in SEEDS:
            evt_df = load_evt_stability_results(seed, category)
            if evt_df is None:
                log.warning(f"Missing data for {category} seed={seed}")
                continue
            
            # Filter by bit depth (handle both "4-bit" string and 4 integer formats)
            bit_rows = evt_df[
                (evt_df["Bit_Depth"] == bits) | 
                (evt_df["Bit_Depth"] == f"{bits}-bit")
            ]
            
            if bit_rows.empty:
                log.warning(f"No {bits}-bit data found for {category} seed={seed}")
                continue
            
            # Take the first matching row (should be only one per bit depth)
            row = bit_rows.iloc[0]
            
            xi = row.get("GPD_Shape_xi")
            sigma = row.get("GPD_Scale_sigma")
            
            if pd.notna(xi) and pd.notna(sigma):
                xis.append(float(xi))
                sigmas.append(float(sigma))
                log.info(f"  {category} | seed={seed} | {bits}-bit: ξ={float(xi):.6f}, σ={float(sigma):.6f}")
        
        if xis:
            results[f"{bits}bit"] = {
                "xis": xis,
                "sigmas": sigmas,
                "mean_xi": float(np.mean(xis)),
                "std_xi": float(np.std(xis)),
                "mean_sigma": float(np.mean(sigmas)),
                "std_sigma": float(np.std(sigmas)),
                "n_seeds": len(xis),
            }
        else:
            results[f"{bits}bit"] = {
                "xis": [],
                "sigmas": [],
                "mean_xi": np.nan,
                "std_xi": np.nan,
                "mean_sigma": np.nan,
                "std_sigma": np.nan,
                "n_seeds": 0,
            }
    
    return results


def main() -> None:
    """
    Run EVT GPD validation: extract shape parameters across categories and seeds.
    """
    log.info("=" * 80)
    log.info("EVT Validation: GPD Shape Parameter Comparison (4-bit vs 8-bit)")
    log.info("=" * 80)
    
    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect results
    detailed_rows = []
    summary_rows = []
    
    for category in TARGET_CATEGORIES:
        log.info(f"\n➤ Processing category: {category}")
        results = extract_gpd_parameters(category, bit_depths=[4, 8])
        
        # Log summary for this category
        for bits_str, data in results.items():
            bits = int(bits_str.replace("bit", ""))
            log.info(f"  {bits}-bit Summary: "
                    f"ξ = {data['mean_xi']:.6f} ± {data['std_xi']:.6f} "
                    f"(n={data['n_seeds']} seeds)")
            
            # Add summary row
            summary_rows.append({
                "category": category,
                "bit_depth": bits,
                "gpd_shape_xi_mean": data["mean_xi"],
                "gpd_shape_xi_std": data["std_xi"],
                "gpd_scale_sigma_mean": data["mean_sigma"],
                "gpd_scale_sigma_std": data["std_sigma"],
                "n_seeds": data["n_seeds"],
            })
            
            # Add detailed rows (one per seed)
            for xi, sigma in zip(data.get("xis", []), data.get("sigmas", [])):
                detailed_rows.append({
                    "category": category,
                    "bit_depth": bits,
                    "gpd_shape_xi": xi,
                    "gpd_scale_sigma": sigma,
                })
    
    # Write summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_CSV.parent / "evt_validation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    log.info(f"\n✓ Saved summary to: {summary_file}")
    
    # Print summary statistics
    log.info("\n" + "=" * 80)
    log.info("SUMMARY: Sign Inversion Check (4-bit → 8-bit)")
    log.info("=" * 80)
    
    for category in TARGET_CATEGORIES:
        cat_data_4bit = summary_df[(summary_df["category"] == category) & (summary_df["bit_depth"] == 4)]
        cat_data_8bit = summary_df[(summary_df["category"] == category) & (summary_df["bit_depth"] == 8)]
        
        if not cat_data_4bit.empty and not cat_data_8bit.empty:
            xi_4bit = float(cat_data_4bit["gpd_shape_xi_mean"].iloc[0])
            xi_8bit = float(cat_data_8bit["gpd_shape_xi_mean"].iloc[0])
            sign_flipped = (xi_4bit > 0) and (xi_8bit < 0)
            status = "✓ SIGN FLIP" if sign_flipped else "✗ NO FLIP"
            
            log.info(
                f"{category:12s}: ξ(4-bit)={xi_4bit:+.6f}, ξ(8-bit)={xi_8bit:+.6f} "
                f"  →  {status}"
            )
    
    log.info("=" * 80)
    log.info(f"Detailed results (all seeds): {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
