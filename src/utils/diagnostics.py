import numpy as np
from scipy.linalg import orthogonal_procrustes

def calculate_cka(X, Y):
    """Centered Kernel Alignment (Linear). Measures feature similarity."""
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    dot_prod = np.sum((X.T @ Y) ** 2)
    norm_x = np.sum((X.T @ X) ** 2)
    norm_y = np.sum((Y.T @ Y) ** 2)
    return dot_prod / np.sqrt(norm_x * norm_y)

def calculate_procrustes_alignment(X, Y):
    """Measures how well X can be rotated to match Y."""
    R, scale = orthogonal_procrustes(X, Y)
    # Return the normalized disparity (0 is perfect, 1 is total drift)
    return np.mean(np.square(X @ R - Y))