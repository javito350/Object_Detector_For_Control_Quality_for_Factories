import matplotlib.pyplot as plt
from pathlib import Path

# 1. The Exact Data (Latency ms, AUROC, Label, Hex Color)
data = [
    {"x": 10.50,  "y": 0.441, "label": "4-bit PQ\n(Geometry Collapse)", "c": "#e76f51"},
    {"x": 12.73,  "y": 0.853, "label": "8-bit PQ\n(Optimal Edge Point)", "c": "#2a9d8f"},
    {"x": 32.06,  "y": 0.858, "label": "Exact FP32\n(Oracle Reference)", "c": "#264653"},
    {"x": 115.00, "y": 0.830, "label": "12-bit PQ\n(Fragmentation)", "c": "#f4a261"}
]

# 2. Professional "Academic Minimalist" Styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 3. Draw the Pareto frontier connecting the best points
# We connect 4-bit -> 8-bit -> Exact. We DO NOT connect 12-bit because it falls off the cliff.
ax.plot([10.50, 12.73, 32.06], [0.441, 0.853, 0.858], color="#adb5bd", linestyle="--", linewidth=2, zorder=2)

# 4. Plot the data points and temporary text
for p in data:
    # Draw large, clean bubbles with a white border to make them pop
    ax.scatter(p['x'], p['y'], s=250, c=p['c'], edgecolors="white", linewidths=2, zorder=5)
    
    # Place text slightly above the dot so it's easy to grab in Inkscape
    ax.text(p['x'], p['y'] + 0.015, p['label'], ha="center", fontweight="bold", fontsize=10, color=p['c'], zorder=6)

# 5. Clean up the axes (remove the "box" look)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 6. Formatting
ax.set_xscale('log')
ax.set_xlim(8, 200)
ax.set_ylim(0.40, 0.90)

ax.set_xlabel("Retrieval Latency (ms, log scale)", fontweight='bold', labelpad=12)
ax.set_ylabel("Image AUROC", fontweight='bold', labelpad=12)
ax.set_title("Quantization Stability: Precision vs. Latency", fontweight='bold', pad=20, fontsize=14)

# 7. EXPORT AS SVG (The crucial step)
out_file = Path("results/bit_ablation_base.svg")
out_file.parent.mkdir(parents=True, exist_ok=True) # Ensure the results folder exists

plt.tight_layout()
plt.savefig(out_file, format="svg", bbox_inches='tight')

print(f"Success! Vector SVG saved to: {out_file}")
print("Next step: Open this file in Inkscape, ungroup, and drag your labels into place.")