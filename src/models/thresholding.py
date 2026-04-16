import numpy as np
from scipy.stats import genpareto
import warnings

class EVTCalibrator:
    """
    Implements Extreme Value Theory (EVT) threshold calibration for Few-Shot Anomaly Detection.
    Fits a Generalized Pareto Distribution (GPD) to the right tail of nominal patch distances.
    """
    def __init__(self, tail_fraction: float = 0.10, target_fpr: float = 0.01):
        """
        Args:
            tail_fraction: The percentage of highest distances to use for the tail (Peaks Over Threshold).
            target_fpr: The acceptable False Positive Rate (e.g., 0.01 = 1% false rejection).
        """
        self.tail_fraction = tail_fraction
        self.target_fpr = target_fpr
        
        self.gpd_shape = None # xi
        self.gpd_scale = None # sigma
        self.threshold = None # u (the threshold where the tail begins)
        self.calibrated_decision_boundary = None
        
    def fit(self, nominal_distances: np.ndarray) -> float:
        """
        Fits the GPD to the extreme distances and calculates the decision boundary.
        
        Args:
            nominal_distances: Flattened 1D numpy array of nearest-neighbor distances from the normal support set.
            
        Returns:
            float: The calibrated decision threshold tau.
        """
        if len(nominal_distances) < 100:
            warnings.warn("Very few patches available for EVT. Calibration may be unstable.")

        # 1. Sort distances to find the extremes
        sorted_distances = np.sort(nominal_distances)
        
        # 2. Define the threshold 'u' where the tail begins
        tail_index = int(len(sorted_distances) * (1.0 - self.tail_fraction))
        self.threshold = sorted_distances[tail_index]
        
        # 3. Extract the exceedances (values above u)
        tail_data = sorted_distances[tail_index:] - self.threshold
        
        if len(tail_data) == 0 or np.max(tail_data) == 0:
             # Fallback if there is zero variance in the tail
             self.calibrated_decision_boundary = self.threshold * 1.05
             return self.calibrated_decision_boundary

        # 4. Fit the Generalized Pareto Distribution using Maximum Likelihood
        # genpareto.fit returns (shape, location, scale). We force loc=0 because we subtracted u.
        shape, loc, scale = genpareto.fit(tail_data, floc=0)
        self.gpd_shape = shape
        self.gpd_scale = scale
        
        # 5. Calculate the analytical decision boundary tau for the target FPR
        # Formula: tau = u + (sigma / xi) * (((alpha / tail_fraction)^-xi) - 1)
        # where alpha is the target_fpr
        
        if abs(shape) < 1e-5:
            # If shape is near zero, it's an exponential distribution limit
            margin = -scale * np.log(self.target_fpr / self.tail_fraction)
        else:
            margin = (scale / shape) * (((self.target_fpr / self.tail_fraction) ** -shape) - 1)
            
        self.calibrated_decision_boundary = self.threshold + margin
        
        return self.calibrated_decision_boundary