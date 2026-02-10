import numpy as np  # Import NumPy for array operations.
import faiss  # Import FAISS for fast nearest-neighbor search.
import torch  # Import torch to check GPU availability.
from sklearn.decomposition import PCA  # Import PCA for dimensionality reduction.
from sklearn.cluster import KMeans  # Import KMeans for clustering in coreset sampling.
from scipy import stats  # Import stats for distributions and thresholds.
from tqdm import tqdm  # Import tqdm for progress bars.
from typing import List, Optional, Tuple  # Import typing helpers.
import h5py  # Import h5py for potential feature storage.
import os  # Import os for filesystem utilities.

# Coreset sampler that picks a smaller but diverse feature set.
class EnhancedCoresetSampler:  # Define a class for coreset sampling.
    # Initialize the sampler.
    def __init__(self, percentage: float = 0.1, method: str = "kcenter_greedy"):  # Set defaults.
        self.percentage = percentage  # Store fraction of samples to keep.
        self.method = method  # Store the sampling method name.
        
    # Select a subset of features to keep.
    def sample(self, features: np.ndarray) -> np.ndarray:  # Choose a diverse coreset.
        n_samples = int(len(features) * self.percentage)  # Compute target coreset size.
        
        if self.method == "kcenter_greedy":  # Use k-center greedy if selected.
            return self._kcenter_greedy(features, n_samples)  # Call k-center sampler.
        elif self.method == "statistical":  # Use statistical sampler if selected.
            return self._statistical_sampling(features, n_samples)  # Call statistical sampler.
        else:  # If method is unknown.
            raise ValueError(f"Unknown sampling method: {self.method}")  # Raise error.
    
    # K-center greedy sampling.
    def _kcenter_greedy(self, features: np.ndarray, k: int) -> np.ndarray:  # Pick k diverse points.
        n = len(features)  # Number of feature vectors.
        
        # Initialize with farthest point from mean (better coverage).
        center = np.mean(features, axis=0)  # Compute feature mean.
        distances = np.linalg.norm(features - center, axis=1)  # Distances from mean.
        selected_indices = [np.argmax(distances)]  # Start with farthest point.
        
        distances = np.full(n, np.inf)  # Initialize min distances as infinity.
        
        for _ in tqdm(range(k - 1), desc="Coreset selection"):  # Select remaining points.
            # Update distances to nearest selected point.
            last_selected = features[selected_indices[-1]]  # Get last chosen feature.
            dist_to_last = np.linalg.norm(features - last_selected, axis=1)  # Distances to last.
            distances = np.minimum(distances, dist_to_last)  # Update min distances.
            
            # Select point with maximum minimum distance.
            next_idx = np.argmax(distances)  # Choose farthest remaining point.
            selected_indices.append(next_idx)  # Add it to the selection.
        
        return features[selected_indices]  # Return selected features.
    
    # Statistical sampling using PCA and KMeans.
    def _statistical_sampling(self, features: np.ndarray, k: int) -> np.ndarray:  # Pick representatives.
        # Perform PCA for dimensionality reduction.
        pca = PCA(n_components=min(50, features.shape[1]))  # Reduce to <= 50 dims.
        features_reduced = pca.fit_transform(features)  # Apply PCA.
        
        # Use k-means++ initialization for better coverage.
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)  # Build KMeans.
        kmeans.fit(features_reduced)  # Fit KMeans on reduced features.
        
        # Select points closest to cluster centers.
        selected_indices = []  # Prepare list of selected points.
        
        # Get distances to all cluster centers.
        all_distances = kmeans.transform(features_reduced)  # Distances to centers.
        
        for cluster_idx in range(k):  # Loop over each cluster.
            # Select closest point to center.
            closest_idx = np.argmin(all_distances[:, cluster_idx])  # Find nearest point.
            selected_indices.append(closest_idx)  # Add index to selection.
        
        return features[selected_indices]  # Return chosen features.

# Memory bank class that stores normal features and queries them.
class MemoryBank:  # Define the memory bank for nearest neighbor search.
    # Initialize the memory bank.
    def __init__(self, dimension: int, use_gpu: bool = True):  # Set feature size and GPU usage.
        self.dimension = dimension  # Store feature dimension.
        self.use_gpu = use_gpu and torch.cuda.is_available()  # Use GPU only if available.
        
        # FAISS index.
        self.index = faiss.IndexFlatL2(dimension)  # Create L2 distance index.
        
        if self.use_gpu:  # If GPU is enabled.
            res = faiss.StandardGpuResources()  # Create GPU resources.
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # Move index to GPU.
        
        self.features = None  # Placeholder for stored features.
        self.is_trained = False  # Track whether the memory bank is ready.
        
        # Statistical models.
        self.distance_mean = None  # Mean of distances for normalization.
        self.distance_std = None  # Std of distances for normalization.
        self.feature_mean = None  # Mean of features for Mahalanobis.
        self.feature_cov = None  # Covariance of features for Mahalanobis.
        self.feature_cov_inv = None  # Inverse covariance for Mahalanobis.
        
    # Build the memory bank using normal features.
    def build(self, features_list: List[np.ndarray],  # List of feature arrays.
              coreset_percentage: float = 0.1) -> None:  # Coreset percentage.
        # Build enhanced memory bank with statistical modeling.
        all_features = np.vstack(features_list)  # Concatenate all features.
        print(f"Total features before coreset: {len(all_features)}")  # Print total count.
        
        # Apply enhanced coreset sampling.
        sampler = EnhancedCoresetSampler(percentage=coreset_percentage, method="statistical")  # Create sampler.
        coreset_features = sampler.sample(all_features)  # Sample a smaller set.
        print(f"Features after statistical coreset: {len(coreset_features)}")  # Print coreset count.
        
        # Normalize features.
        coreset_features = self._normalize_features(coreset_features)  # Normalize vectors.
        
        # Store features.
        self.features = coreset_features.astype(np.float32)  # Store in float32 for FAISS.
        
        # Build index.
        self.index.reset()  # Reset the FAISS index.
        self.index.add(self.features)  # Add features to the index.
        
        self.is_trained = True  # Mark memory bank as ready.
        
        # Learn statistical models.
        self._learn_statistics(all_features)  # Compute mean/std and covariance.
        
    # Query the memory bank for nearest neighbors.
    def query(self, query_features: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:  # Query k-NN.
        # Query the memory bank with statistical normalization.
        if not self.is_trained:  # Ensure memory bank is ready.
            raise ValueError("Memory bank not built. Call build() first.")  # Raise error if not ready.
        
        # Normalize query features.
        query_features = self._normalize_features(query_features)  # Normalize inputs.
        
        if query_features.ndim == 1:  # If a single vector.
            query_features = query_features.reshape(1, -1)  # Convert to 2D.
        
        query_features = query_features.astype(np.float32)  # Convert to float32.
        
        distances, indices = self.index.search(query_features, k)  # Search nearest neighbors.
        
        # Statistical normalization of distances.
        if self.distance_mean is not None and self.distance_std is not None:  # If stats exist.
            distances = (distances - self.distance_mean) / (self.distance_std + 1e-8)  # Normalize distances.
        
        return distances, indices  # Return distances and indices.
    
    # Compute Mahalanobis distance for a query.
    def compute_mahalanobis_distance(self, query_features: np.ndarray) -> np.ndarray:  # Use covariance stats.
        # Compute Mahalanobis distance using learned statistics.
        if self.feature_cov_inv is None:  # Ensure stats exist.
            raise ValueError("Statistics not learned. Call build() first.")  # Raise error if missing.
        
        query_features = self._normalize_features(query_features)  # Normalize features.
        
        if query_features.ndim == 1:  # If a single vector.
            query_features = query_features.reshape(1, -1)  # Make it 2D.
        
        deltas = query_features - self.feature_mean  # Compute offset from mean.
        distances = np.sqrt(np.sum(deltas @ self.feature_cov_inv * deltas, axis=1))  # Compute distance.
        
        return distances  # Return Mahalanobis distances.
    
    # Compute an adaptive threshold based on stats.
    def get_adaptive_threshold(self, confidence_level: float = 0.95) -> float:  # Choose threshold by confidence.
        # Get adaptive threshold based on learned distance distribution.
        if self.distance_mean is None or self.distance_std is None:  # If stats are missing.
            return 0.0  # Return a safe default.
        
        # Assuming distances follow normal distribution.
        from scipy import stats as sp_stats  # Import stats locally to avoid global name clash.
        threshold = sp_stats.norm.ppf(confidence_level, loc=self.distance_mean, scale=self.distance_std)  # Compute threshold.
        
        return threshold  # Return threshold.
    
    # Learn statistical properties of the feature space.
    def _learn_statistics(self, all_features: np.ndarray) -> None:  # Compute mean/cov and distance stats.
        # Normalize features.
        normalized_features = self._normalize_features(all_features)  # Normalize features for stats.
        
        # Compute feature statistics.
        self.feature_mean = np.mean(normalized_features, axis=0)  # Mean vector.
        self.feature_cov = np.cov(normalized_features.T)  # Covariance matrix.
        
        # Regularize covariance matrix.
        reg = 1e-6  # Small value for numerical stability.
        self.feature_cov = self.feature_cov + reg * np.eye(self.feature_cov.shape[0])  # Add to diagonal.
        
        try:  # Try to invert covariance.
            self.feature_cov_inv = np.linalg.inv(self.feature_cov)  # Invert covariance.
        except np.linalg.LinAlgError:  # If inversion fails.
            self.feature_cov_inv = np.linalg.pinv(self.feature_cov)  # Use pseudo-inverse instead.
        
        # Compute distances to nearest neighbors for all features.
        all_features_norm = normalized_features.astype(np.float32)  # Convert to float32 for FAISS.
        distances, _ = self.index.search(all_features_norm, k=1)  # Query nearest neighbors.
        
        self.distance_mean = np.mean(distances)  # Store mean distance.
        self.distance_std = np.std(distances)  # Store distance std.
        
        print(f"Learned statistics: mean distance = {self.distance_mean:.4f}, std = {self.distance_std:.4f}")  # Print stats.
    
    # Normalize feature vectors.
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:  # L2 normalize features.
        # Robust feature normalization.
        if features.ndim == 1:  # If input is 1D.
            features = features.reshape(1, -1)  # Convert to 2D.
        
        # L2 normalization for cosine similarity.
        norms = np.linalg.norm(features, axis=1, keepdims=True)  # Compute vector norms.
        norms[norms == 0] = 1  # Avoid division by zero.
        normalized = features / norms  # Normalize each vector.
        
        return normalized  # Return normalized features.