import numpy as np
import faiss
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from typing import List, Tuple
import os

class EnhancedCoresetSampler:
    """Coreset sampler that picks a smaller but diverse feature set."""
    def __init__(self, percentage: float = 0.1, method: str = "statistical"):
        self.percentage = percentage
        self.method = method
        
    def sample(self, features: np.ndarray) -> np.ndarray:
        n_samples = int(len(features) * self.percentage)
        if self.method == "kcenter_greedy":
            return self._kcenter_greedy(features, n_samples)
        elif self.method == "statistical":
            return self._statistical_sampling(features, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")

    def _kcenter_greedy(self, features: np.ndarray, k: int) -> np.ndarray:
        n = len(features)
        center = np.mean(features, axis=0)
        distances = np.linalg.norm(features - center, axis=1)
        selected_indices = [np.argmax(distances)]
        distances = np.full(n, np.inf)
        
        for _ in tqdm(range(k - 1), desc="Coreset selection"):
            last_selected = features[selected_indices[-1]]
            dist_to_last = np.linalg.norm(features - last_selected, axis=1)
            distances = np.minimum(distances, dist_to_last)
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
        return features[selected_indices]

    def _statistical_sampling(self, features: np.ndarray, k: int) -> np.ndarray:
        pca = PCA(n_components=min(50, features.shape[1]))
        features_reduced = pca.fit_transform(features)
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            init='k-means++',
            random_state=42,
            n_init=10,
            batch_size=1024,
        )
        kmeans.fit(features_reduced)
        selected_indices = []
        all_distances = kmeans.transform(features_reduced)
        for cluster_idx in range(k):
            closest_idx = np.argmin(all_distances[:, cluster_idx])
            selected_indices.append(closest_idx)
        return features[selected_indices]

class MemoryBank:
    """Symmetry-Aware Quantized Memory Bank for Anomaly Detection."""
    def __init__(self, dimension: int, use_gpu: bool = True, use_pq: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.use_pq = use_pq
        
        self.index = None 
        self.features = None 
        self.is_trained = False 
        
        # Statistical models
        self.distance_mean = None 
        self.distance_std = None 
        self.feature_mean = None
        self.feature_cov_inv = None

    def _orbit_aware_reduction(self, features: np.ndarray, similarity_threshold: float = 0.98) -> np.ndarray:
        """Filters redundant features in the same geometric orbit (p4m symmetry)."""
        if len(features) < 2: return features
        norm_features = self._normalize_features(features)
        keep_indices = [0]
        for i in range(1, len(features)):
            sim = np.dot(norm_features[i], norm_features[keep_indices[-1]])
            if sim < similarity_threshold:
                keep_indices.append(i)
        return features[keep_indices]

    def build(self, features_list: List[np.ndarray], coreset_percentage: float = 0.1, pq_bits: int = 8) -> None:
        """Builds the bank using Orbit-Aware reduction and optional PQ compression."""
        all_features = np.vstack(features_list).astype(np.float32)
        
        # 1. Reduction Stages
        all_features = self._orbit_aware_reduction(all_features)
        sampler = EnhancedCoresetSampler(percentage=coreset_percentage, method="statistical")
        coreset_features = sampler.sample(all_features)
        
        self.features = self._normalize_features(coreset_features).astype(np.float32)
        
        # 2. FAISS Index Selection
        nlist = max(1, min(100, len(self.features) // 10))
        quantizer = faiss.IndexFlatL2(self.dimension)

        if self.use_pq:
            # M = sub-vectors. For WRN50 (dimension 1024), 8 or 16 is standard.
            M = 8 
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, M, pq_bits)
        else:
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_L2)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
        self.index.train(self.features)
        self.index.add(self.features)
        self.is_trained = True
        self._learn_statistics(all_features)

    def query(self, query_features: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained: raise ValueError("Bank not built.")
        query_features = self._normalize_features(query_features).astype(np.float32)
        distances, indices = self.index.search(query_features, k)
        return distances, indices

    def get_margin_pressure(self, exact_dist: np.ndarray, quant_dist: np.ndarray, labels: np.ndarray):
        """Calculates r = epsilon_q / Delta_sep (The Stability Diagnostic)."""
        epsilon_q = np.max(np.abs(exact_dist - quant_dist))
        nominals = exact_dist[labels == 0]
        anomalies = exact_dist[labels == 1]
        
        if len(anomalies) == 0 or len(nominals) == 0:
            return 0, epsilon_q, 0
            
        delta_sep = np.percentile(anomalies, 5) - np.percentile(nominals, 95)
        r = epsilon_q / delta_sep if delta_sep > 0 else float('inf')
        return r, epsilon_q, delta_sep

    def _learn_statistics(self, all_features: np.ndarray) -> None:
        normalized_features = self._normalize_features(all_features)
        self.feature_mean = np.mean(normalized_features, axis=0)
        cov = np.cov(normalized_features.T) + 1e-6 * np.eye(self.dimension)
        self.feature_cov_inv = np.linalg.inv(cov)
        
        distances, _ = self.query(normalized_features, k=1)
        self.distance_mean = np.mean(distances)
        self.distance_std = np.std(distances)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1: features = features.reshape(1, -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return features / norms
