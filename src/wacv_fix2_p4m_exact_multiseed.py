import os
import sys
import numpy as np
import torch
import faiss
import csv
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(r"c:\Users\Javier\Documents\Junior_Seminar\Project Root\Quality_control_for_Factories")
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.symmetry_feature_extractor import SymmetryAwareFeatureExtractor
from models.memory_bank import MemoryBank
from utils.image_loader import MVTecStyleDataset

MVTEC_ROOT = PROJECT_ROOT / "data" / "mvtec"
CATEGORIES = ["cable", "capsule", "screw"]
SEEDS = [0, 1, 42, 123, 999]
FEATURE_DIM = 1536
N_SHOT = 10
IMG_SIZE = 256
OUTPUT_CSV = PROJECT_ROOT / "results" / "wacv" / "fix2_p4m_exact_multiseed.csv"

def sample_seeded_support(samples, n_shot, seed):
    sorted_samples = sorted(samples, key=lambda item: item[0])
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(sorted_samples), size=n_shot, replace=False)
    return [sorted_samples[int(i)] for i in idx]

def build_loaders(category, seed):
    train_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=True, img_size=IMG_SIZE
    )
    train_ds.samples = sample_seeded_support(train_ds.samples, N_SHOT, seed)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False)

    test_ds = MVTecStyleDataset(
        root_dir=str(MVTEC_ROOT), category=category, is_train=False, img_size=IMG_SIZE
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    return train_loader, test_loader

def main():
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    feature_extractor = SymmetryAwareFeatureExtractor(backbone="wide_resnet50_2", device=device)
    
    all_results = []
    
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "seed", "image_auroc"])

    print(f"Starting evaluations for {len(CATEGORIES)} categories over {len(SEEDS)} seeds...")
    
    for category in CATEGORIES:
        for seed in SEEDS:
            train_loader, test_loader = build_loaders(category, seed)
            
            support_features = []
            for images, _, _ in train_loader:
                images = images.to(device)
                feats = feature_extractor.extract_patch_features(images, apply_p4m=True)
                support_features.append(feats)
                
            memory_bank = MemoryBank(dimension=FEATURE_DIM, use_gpu=(device=="cuda"), use_pq=False)
            memory_bank.build(support_features, coreset_percentage=0.1)
            
            # Force exact search
            exact_index = faiss.IndexFlatL2(memory_bank.dimension)
            exact_index.add(memory_bank.features)
            memory_bank.index = exact_index
            memory_bank.is_trained = True
            
            image_scores, image_labels = [], []
            for images, labels, _ in test_loader:
                images = images.to(device)
                B = images.shape[0]
                with torch.no_grad():
                    patch_feats = feature_extractor.extract_patch_features(images, apply_p4m=False)
                    distances, _ = memory_bank.query(patch_feats, k=1)
                patch_grid = distances.reshape(B, 28, 28)
                for idx in range(B):
                    img_score = float(np.max(patch_grid[idx]))
                    image_scores.append(img_score)
                    image_labels.append(int(labels[idx].item()))
                    
            auroc = roc_auc_score(image_labels, image_scores)
            
            print(f"{category:<10} | seed {seed:<3} | AUROC: {auroc:.6f}")
            with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([category, seed, auroc])
            
            all_results.append((category, auroc))
            
    print("\n" + "="*50)
    print("FINAL EXACT SEARCH (WITH P4M) RESULTS (5-Seed Mean):")
    print("="*50)
    for cat in CATEGORIES:
        aurocs = [r[1] for r in all_results if r[0] == cat]
        print(f"{cat.capitalize():<10}: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")

if __name__ == "__main__":
    main()
