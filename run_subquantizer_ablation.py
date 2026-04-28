import itertools
import subprocess


# The 3 categories requested by the reviewer for the NNDV proof
categories = ["bottle", "metal_nut", "screw"]

# The 3 subquantizer depths (M)
subquantizers = [16, 32, 64]

# Your 5 standard deployment seeds
seeds = [111, 333, 999, 2026, 3407]


print("Starting WACV Subquantizer Ablation...")

for cat, m, seed in itertools.product(categories, subquantizers, seeds):
    print("\n=======================================================")
    print(f"RUNNING: Category={cat} | Subquantizers (M)={m} | Seed={seed}")
    print("=======================================================\n")

    cmd = (
        f"python src/faiss_subquantizer_ablation_cable.py "
        f"--category {cat} --m-values {m}"
    )

    subprocess.run(cmd, shell=True)

print("Ablation Complete!")