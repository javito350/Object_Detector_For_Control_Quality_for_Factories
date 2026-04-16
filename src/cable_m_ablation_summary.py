"""
Summary: FAISS Subquantizer (M) Ablation Study - Cable Category
================================================================

This script compiles the results from the cable M-ablation study and 
compares them with the previously published screw results.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Load cable results
cable_pro_csv = ROOT / "results" / "cable_pro_auc_m_ablation.csv"
cable_latency_csv = ROOT / "qualitative_results_m_ablation" / "cable_subquantizer_summary.csv"

cable_pro = pd.read_csv(cable_pro_csv)
cable_latency = pd.read_csv(cable_latency_csv)

# Merge on m_value
cable_results = pd.merge(
    cable_pro,
    cable_latency[['m_value', 'avg_retrieval_latency_ms']],
    on='m_value'
)

# Screw results (from plot_rq1.py - previously published)
screw_results = pd.DataFrame({
    'm_value': [16, 32, 64],
    'pro_auc_fpr0.3': [0.2201, 0.2932, 0.9778],
    'avg_retrieval_latency_ms': [577.07, 599.54, 600.02]
})

print("=" * 80)
print("FAISS SUBQUANTIZER (M) ABLATION STUDY - RESULTS SUMMARY")
print("=" * 80)

print("\n" + "CABLE CATEGORY RESULTS (NEW)".center(80))
print("-" * 80)
print("\nM Value | PRO-AUC (FPR@0.3) | Latency (ms)")
print("-" * 80)
for _, row in cable_results.iterrows():
    m = int(row['m_value'])
    pro = row['pro_auc_fpr0.3']
    lat = row['avg_retrieval_latency_ms']
    print(f"  {m:3d}   |      {pro:6.4f}       |   {lat:7.2f}")

print("\n" + "SCREW CATEGORY RESULTS (BASELINE)".center(80))
print("-" * 80)
print("\nM Value | PRO-AUC (FPR@0.3) | Latency (ms)")
print("-" * 80)
for _, row in screw_results.iterrows():
    m = int(row['m_value'])
    pro = row['pro_auc_fpr0.3']
    lat = row['avg_retrieval_latency_ms']
    print(f"  {m:3d}   |      {pro:6.4f}       |   {lat:7.2f}")

print("\n" + "KEY OBSERVATIONS".center(80))
print("-" * 80)

cable_16 = cable_results[cable_results['m_value'] == 16].iloc[0]
cable_64 = cable_results[cable_results['m_value'] == 64].iloc[0]
screw_16 = screw_results[screw_results['m_value'] == 16].iloc[0]
screw_64 = screw_results[screw_results['m_value'] == 64].iloc[0]

print(f"\n1. PRO-AUC Trend (Accuracy):")
print(f"   - Cable:  M=16 ({cable_16['pro_auc_fpr0.3']:.4f}) -> M=64 ({cable_64['pro_auc_fpr0.3']:.4f}) "
      f"(Delta: {cable_64['pro_auc_fpr0.3'] - cable_16['pro_auc_fpr0.3']:+.4f})")
print(f"   - Screw:  M=16 ({screw_16['pro_auc_fpr0.3']:.4f}) -> M=64 ({screw_64['pro_auc_fpr0.3']:.4f}) "
      f"(Delta: {screw_64['pro_auc_fpr0.3'] - screw_16['pro_auc_fpr0.3']:+.4f})")

print(f"\n2. Latency Trend (Efficiency):")
print(f"   - Cable:  M=16 ({cable_16['avg_retrieval_latency_ms']:.2f} ms) -> M=64 ({cable_64['avg_retrieval_latency_ms']:.2f} ms) "
      f"(Reduction: {cable_16['avg_retrieval_latency_ms'] - cable_64['avg_retrieval_latency_ms']:.2f} ms)")
print(f"   - Screw:  M=16 ({screw_16['avg_retrieval_latency_ms']:.2f} ms) -> M=64 ({screw_64['avg_retrieval_latency_ms']:.2f} ms) "
      f"(Reduction: {screw_16['avg_retrieval_latency_ms'] - screw_64['avg_retrieval_latency_ms']:.2f} ms)")

print(f"\n3. Quantization Trade-off (@M=64 vs M=16):")
cable_lat_reduction_pct = 100 * (cable_16['avg_retrieval_latency_ms'] - cable_64['avg_retrieval_latency_ms']) / cable_16['avg_retrieval_latency_ms']
screw_lat_reduction_pct = 100 * (screw_16['avg_retrieval_latency_ms'] - screw_64['avg_retrieval_latency_ms']) / screw_16['avg_retrieval_latency_ms']
print(f"   - Cable latency improvement: {cable_lat_reduction_pct:.1f}%")
print(f"   - Screw latency improvement: {screw_lat_reduction_pct:.1f}%")

print(f"\n   - Cable accuracy change: {(cable_64['pro_auc_fpr0.3'] / cable_16['pro_auc_fpr0.3'] - 1) * 100:.1f}%")
print(f"   - Screw accuracy change: {(screw_64['pro_auc_fpr0.3'] / screw_16['pro_auc_fpr0.3'] - 1) * 100:.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION: M-Parameter Effect on Accuracy-Latency Tradeoff")
print("=" * 80)
print(f"""
For the CABLE category:
  - Decreasing M (16→32→64) generally DECREASES latency (faster inference)
  - However, the effect on accuracy (PRO-AUC) is non-monotonic
  - M=16 achieves the best PRO-AUC ({cable_16['pro_auc_fpr0.3']:.4f})
  - M=64 achieves the best latency ({cable_64['avg_retrieval_latency_ms']:.2f} ms)
  - This demonstrates the accuracy-efficiency quantization trade-off

The cable results show different precision-latency characteristics compared to 
screw, highlighting that optimal M selection is task/category-dependent.
""")

print("=" * 80)
