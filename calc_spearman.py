import scipy.stats as stats

# 15 Categories (Order: Bottle, Zipper, Grid, Capsule, Leather, Toothbrush, 
# Carpet, Pill, Screw, Metal Nut, Hazelnut, Transistor, Wood, Tile, Cable)

nndv_values = [
    0.2947, 0.0784, 0.1331, 0.0658, 0.0245, 0.2416, 0.0250, 0.1637, 
    0.1315, 0.2380, 0.0975, 0.2108, 0.0434, 0.0309, 0.1550
]

auroc_sd_values = [
    0.0072, 0.0487, 0.0502, 0.0433, 0.0007, 0.0216, 0.0062, 0.0375, 
    0.0428, 0.1067, 0.0146, 0.0850, 0.0027, 0.0041, 0.0271
]

# Calculate Spearman rank correlation
correlation, p_value = stats.spearmanr(nndv_values, auroc_sd_values)

print(f"Spearman Rank Correlation (rho): {correlation:.3f}")
print(f"p-value: {p_value:.4f}")