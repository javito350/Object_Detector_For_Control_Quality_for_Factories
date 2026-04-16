import os

# Measure your 8-bit index file (once it's saved)
index_size_bytes = os.path.getsize('results/bottle_index.faiss') # or your index path
index_size_mb = index_size_bytes / (1024 * 1024)

# A standard ResNet-50 (Teacher) + Student (EfficientAD style) is roughly:
# (25M parameters + 25M parameters) * 4 bytes (float32) 
baseline_mb = (50 * 10**6 * 4) / (1024 * 1024)

print(f"8-bit FAISS Index Size: {index_size_mb:.2f} MB")
print(f"Distillation Baseline Size: ~{baseline_mb:.2f} MB")
print(f"Reduction Factor: {baseline_mb / index_size_mb:.1f}x")