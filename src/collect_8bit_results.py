import pandas as pd
import glob
import os

seeds = [111, 333, 999, 2026, 3407]
results = []

for seed in seeds:
    path = f"final_csv_exports/results_8bit_seed{seed}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        bottle = df[df['category'] == 'bottle']['image_auroc'].values[0]
        leather = df[df['category'] == 'leather']['image_auroc'].values[0]
        screw = df[df['category'] == 'screw']['image_auroc'].values[0]
        mean = df['image_auroc'].mean()
        results.append({
            'seed': seed,
            'bottle': bottle,
            'leather': leather,
            'screw': screw,
            'overall': mean
        })

df_res = pd.DataFrame(results)
print(df_res)
