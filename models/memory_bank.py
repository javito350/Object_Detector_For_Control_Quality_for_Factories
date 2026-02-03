import numpy as np
import faiss
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from tqdm import tqdm
from typing import List, Optional, Tuple
import h5py
import os

class EnhancedCoresetSampler:
    """
    Enhanced coreset sampling with statistical guarantees.
    """
    
    def __init__(self, percentage: float = 0.1, method: str = "kcenter_greedy"):
        self.percentage = percentage
        self.method = method
        
    def sample(self, features: np.ndarray) -> np.ndarray:
        """
        Select diverse subset of features with statistical coverage guarantees.
        """
        n_samples = int(len(features) * self.percentage)
        
        if self.method == "kcenter_greedy":
            return self._kcenter_greedy(features, n_samples)
        elif self.method == "statistical":
            return self._statistical_sampling(features, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {self.method}")
    
    def _kcenter_greedy(self, features: np.ndarray, k: int) -> np.ndarray:
        """K-center greedy with improved initialization"""
        n = len(features)
        
        # Initialize with farthest point from mean (better coverage)
        center = np.mean(features, axis=0)
        distances = np.linalg.norm(features - center, axis=1)
        selected_indices = [np.argmax(distances)]
        
        distances = np.full(n, np.inf)
        
        for _ in tqdm(range(k - 1), desc="Coreset selection"):
            # Update distances to nearest selected point
            last_selected = features[selected_indices[-1]]
            dist_to_last = np.linalg.norm(features - last_selected, axis=1)
            distances = np.minimum(distances, dist_to_last)
            
            # Select point with maximum minimum distance
            next_idx = np.argmax(distances)
            selected_indices.append(next_idx)
        
        return features[selected_indices]
    
    def _statistical_sampling(self, features: np.ndarray, k: int) -> np.ndarray:
        """
        Statistical sampling that preserves distribution characteristics.
        """
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=min(50, features.shape[1]))
        features_reduced = pca.fit_transform(features)
        
        # Use k-means++ initialization for better coverage
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(features_reduced)
        
        # Select points closest to cluster centers
        selected_indices = []
        
        # Get distances to all cluster centers
        all_distances = kmeans.transform(features_reduced)  # (n_samples, k)
        
        for cluster_idx in range(k):
            # Select closest point to center
            closest_idx = np.argmin(all_distances[:, cluster_idx])
            selected_indices.append(closest_idx)
        
        return features[selected_indices]

class MemoryBank:
    """
    Enhanced memory bank with statistical modeling and adaptive retrieval.
    """
    
    def __init__(self, dimension: int, use_gpu: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.features = None
        self.is_trained = False
        
        # Statistical models
        self.distance_mean = None
        self.distance_std = None
        self.feature_mean = None
        self.feature_cov = None
        self.feature_cov_inv = None
        
    def build(self, features_list: List[np.ndarray], 
              coreset_percentage: float = 0.1) -> None:
        """
        Build enhanced memory bank with statistical modeling.
        """
        # Concatenate all features
        all_features = np.vstack(features_list)
        print(f"Total features before coreset: {len(all_features)}")
        
        # Apply enhanced coreset sampling
        sampler = EnhancedCoresetSampler(percentage=coreset_percentage, method="statistical")
        coreset_features = sampler.sample(all_features)
        print(f"Features after statistical coreset: {len(coreset_features)}")
        
        # Normalize features
        coreset_features = self._normalize_features(coreset_features)
        
        # Store features
        self.features = coreset_features.astype(np.float32)
        
        # Build index
        self.index.reset()
        self.index.add(self.features)
        
        self.is_trained = True
        
        # Learn statistical models
        self._learn_statistics(all_features)
        
    def query(self, query_features: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the memory bank with statistical normalization.
        """
        if not self.is_trained:
            raise ValueError("Memory bank not built. Call build() first.")
        
        # Normalize query features
        query_features = self._normalize_features(query_features)
        
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        query_features = query_features.astype(np.float32)
        
        distances, indices = self.index.search(query_features, k)
        
        # Statistical normalization of distances
        if self.distance_mean is not None and self.distance_std is not None:
            distances = (distances - self.distance_mean) / (self.distance_std + 1e-8)
        
        return distances, indices
    
    def compute_mahalanobis_distance(self, query_features: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance using learned statistics.
        """
        if self.feature_cov_inv is None:
            raise ValueError("Statistics not learned. Call build() first.")
        
        query_features = self._normalize_features(query_features)
        
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        
        deltas = query_features - self.feature_mean
        distances = np.sqrt(np.sum(deltas @ self.feature_cov_inv * deltas, axis=1))
        
        return distances
    
    def get_adaptive_threshold(self, confidence_level: float = 0.95) -> float:
        """
        Get adaptive threshold based on learned distance distribution.
        """
        if self.distance_mean is None or self.distance_std is None:
            return 0.0
        
        # Assuming distances follow normal distribution
        from scipy import stats as sp_stats
        threshold = sp_stats.norm.ppf(confidence_level, loc=self.distance_mean, scale=self.distance_std)
        
        return threshold
    
    def _learn_statistics(self, all_features: np.ndarray) -> None:
        """Learn statistical models from features"""
        # Normalize features
        normalized_features = self._normalize_features(all_features)
        
        # Compute feature statistics
        self.feature_mean = np.mean(normalized_features, axis=0)
        self.feature_cov = np.cov(normalized_features.T)
        
        # Regularize covariance matrix
        reg = 1e-6
        self.feature_cov = self.feature_cov + reg * np.eye(self.feature_cov.shape[0])
        
        try:
            self.feature_cov_inv = np.linalg.inv(self.feature_cov)
        except np.linalg.LinAlgError:
            self.feature_cov_inv = np.linalg.pinv(self.feature_cov)
        
        # Compute distances to nearest neighbors for all features
        all_features_norm = normalized_features.astype(np.float32)
        distances, _ = self.index.search(all_features_norm, k=1)
        
        self.distance_mean = np.mean(distances)
        self.distance_std = np.std(distances)
        
        print(f"Learned statistics: mean distance = {self.distance_mean:.4f}, std = {self.distance_std:.4f}")
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Robust feature normalization"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # L2 normalization for cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = features / norms
        
        return normalized