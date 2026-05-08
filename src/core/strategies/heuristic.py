"""Heuristic multi-feature weighting strategy for SVD components.

Combines multiple features (SFM, energy, temporal structure) to estimate
the probability that each SVD component is signal vs noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class HeuristicMultiFeatureStrategy:
    """Multi-feature heuristic weighting for SVD components.
    
    Combines three features to estimate signal probability:
    1. SFM (Spectral Flatness Measure): low = signal, high = noise
    2. Energy: high = likely signal, low = likely noise
    3. Temporal structure: high autocorrelation = signal, low = noise
    
    The final weight is a weighted combination of these features.
    """
    
    def __init__(
        self,
        sfm_weight: float = 0.4,
        energy_weight: float = 0.4,
        temporal_weight: float = 0.2,
        sfm_threshold_low: float = 0.2,
        sfm_threshold_high: float = 0.6,
        energy_threshold: float = 0.01,
        temporal_threshold: float = 0.3,
    ) -> None:
        """Initialize multi-feature strategy.
        
        Args:
            sfm_weight: Weight for SFM feature in final combination
            energy_weight: Weight for energy feature
            temporal_weight: Weight for temporal structure feature
            sfm_threshold_low: SFM below this is definitely signal
            sfm_threshold_high: SFM above this is definitely noise
            energy_threshold: Energy below this is considered noise
            temporal_threshold: Temporal autocorrelation above this is signal
        """
        # Normalize weights to sum to 1
        total = sfm_weight + energy_weight + temporal_weight
        self.sfm_weight = sfm_weight / total
        self.energy_weight = energy_weight / total
        self.temporal_weight = temporal_weight / total
        
        self.sfm_threshold_low = sfm_threshold_low
        self.sfm_threshold_high = sfm_threshold_high
        self.energy_threshold = energy_threshold
        self.temporal_threshold = temporal_threshold
    
    def compute_sfm(self, signal: NDArray[np.float64], frame_size: int = 1024) -> float:
        """Compute Spectral Flatness Measure for a signal.
        
        SFM = geometric_mean / arithmetic_mean
        Range: [0, 1], where 1 = white noise (flat), 0 = pure tone (peaky)
        """
        sfm_values = []
        # Need at least 2 frames for meaningful SFM
        if len(signal) < frame_size * 2:
            # Use smaller frame size
            frame_size = max(256, len(signal) // 4)
        
        for i in range(0, len(signal) - frame_size, frame_size):
            frame = signal[i:i + frame_size]
            spectrum = np.abs(np.fft.rfft(frame))
            # Avoid zeros
            spectrum = spectrum + 1e-10
            # Geometric mean
            geo_mean = np.exp(np.mean(np.log(spectrum)))
            # Arithmetic mean
            arith_mean = np.mean(spectrum)
            if arith_mean > 1e-10:  # Avoid division by zero
                sfm = geo_mean / arith_mean
                sfm_values.append(sfm)
        
        if not sfm_values:
            # Fallback: compute for entire signal
            spectrum = np.abs(np.fft.rfft(signal))
            spectrum = spectrum + 1e-10
            geo_mean = np.exp(np.mean(np.log(spectrum)))
            arith_mean = np.mean(spectrum)
            if arith_mean > 1e-10:
                return float(geo_mean / arith_mean)
            else:
                return 0.5  # Default value
        
        return float(np.mean(sfm_values))
    
    def compute_temporal_structure(self, signal: NDArray[np.float64], max_lag: int = 100) -> float:
        """Compute temporal structure measure (autocorrelation).
        
        High autocorrelation indicates periodic structure (signal).
        Low autocorrelation indicates random noise.
        """
        # Normalize signal
        signal_norm = signal - np.mean(signal)
        signal_std = np.std(signal_norm)
        if signal_std < 1e-10:
            return 0.0
        signal_norm = signal_norm / signal_std
        
        # Compute autocorrelation for different lags
        autocorr_values = []
        for lag in range(1, min(max_lag, len(signal_norm) // 4)):
            # Pearson correlation between signal and shifted signal
            corr = np.corrcoef(signal_norm[:-lag], signal_norm[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr_values.append(abs(corr))
        
        if not autocorr_values:
            return 0.0
        
        # Return mean absolute autocorrelation
        return float(np.mean(autocorr_values))
    
    def compute_feature_weights(
        self,
        component_signal: NDArray[np.float64],
        component_energy: float,
        total_energy: float,
    ) -> Tuple[float, float, float, float]:
        """Compute signal probability for a component based on multiple features.
        
        Returns:
            Tuple of (signal_probability, sfm, energy_ratio, temporal_structure)
        """
        # Feature 1: SFM
        sfm = self.compute_sfm(component_signal)
        # Convert SFM to signal probability: low SFM = high probability
        if sfm <= self.sfm_threshold_low:
            sfm_signal_prob = 1.0
        elif sfm >= self.sfm_threshold_high:
            sfm_signal_prob = 0.0
        else:
            # Linear interpolation
            sfm_signal_prob = 1.0 - (sfm - self.sfm_threshold_low) / (self.sfm_threshold_high - self.sfm_threshold_low)
        
        # Feature 2: Energy
        energy_ratio = component_energy / total_energy if total_energy > 0 else 0.0
        if energy_ratio >= self.energy_threshold:
            energy_signal_prob = 1.0
        else:
            energy_signal_prob = energy_ratio / self.energy_threshold
        
        # Feature 3: Temporal structure
        temporal = self.compute_temporal_structure(component_signal)
        if temporal >= self.temporal_threshold:
            temporal_signal_prob = 1.0
        else:
            temporal_signal_prob = temporal / self.temporal_threshold
        
        # Weighted combination
        signal_prob = (
            self.sfm_weight * sfm_signal_prob +
            self.energy_weight * energy_signal_prob +
            self.temporal_weight * temporal_signal_prob
        )
        
        return signal_prob, sfm, energy_ratio, temporal
    
    def get_weights(
        self,
        u: NDArray[np.float64],
        s: NDArray[np.float64],
        vh: NDArray[np.float64],
        signal_length: int,
    ) -> NDArray[np.float64]:
        """Compute weights for each SVD component based on multi-feature analysis.
        
        Args:
            u: Left singular vectors (m x k)
            s: Singular values (k,)
            vh: Right singular vectors (k x n)
            signal_length: Original signal length for reconstruction
            
        Returns:
            weights: Array of weights for each component (k,)
        """
        k = len(s)
        weights = np.zeros(k, dtype=np.float64)
        
        # Total energy
        total_energy = float(np.sum(s * s))
        
        # Only compute features for top components (to save time)
        # For components with very low energy, assign low weight directly
        min_energy_threshold = 0.001 * total_energy  # 0.1% of total energy
        
        # Limit to top 20 components for performance (instead of 50)
        max_components = min(20, k)
        
        for i in range(max_components):
            component_energy = s[i] * s[i]
            
            # Skip expensive feature computation for low-energy components
            if component_energy < min_energy_threshold:
                weights[i] = 0.1  # Low weight for low-energy components
                continue
            
            # Reconstruct component i
            component_matrix = u[:, i:i+1] @ (s[i] * vh[i:i+1, :])
            
            # Reconstruct signal from Hankel matrix using diagonal averaging
            component_signal = self._diagonal_average(component_matrix, signal_length)
            
            # Compute features
            signal_prob, _, _, _ = self.compute_feature_weights(
                component_signal, component_energy, total_energy
            )
            
            # Use signal probability as weight
            weights[i] = signal_prob
        
        # For remaining components, assign low weight
        for i in range(max_components, k):
            weights[i] = 0.1
        
        return weights
    
    def _diagonal_average(self, matrix: NDArray[np.float64], signal_length: int) -> NDArray[np.float64]:
        """Average along diagonals to reconstruct signal from Hankel matrix (vectorized)."""
        rows, cols = matrix.shape
        
        # Create index arrays for diagonal averaging
        # Each element (i, j) contributes to output index i + j
        i_indices, j_indices = np.indices((rows, cols))
        diag_indices = i_indices + j_indices
        
        # Flatten arrays for bincount
        flat_diag_indices = diag_indices.ravel()
        flat_matrix = matrix.ravel()
        
        # Use bincount for efficient diagonal averaging
        result = np.bincount(flat_diag_indices, weights=flat_matrix, minlength=signal_length)
        counts = np.bincount(flat_diag_indices, minlength=signal_length)
        
        # Avoid division by zero
        counts[counts == 0] = 1
        return result / counts


def apply_heuristic_weighting(
    u: NDArray[np.float64],
    s: NDArray[np.float64],
    vh: NDArray[np.float64],
    signal_length: int,
    strategy: HeuristicMultiFeatureStrategy | None = None,
) -> NDArray[np.float64]:
    """Apply heuristic multi-feature weighting to SVD components.
    
    Args:
        u: Left singular vectors
        s: Singular values
        vh: Right singular vectors
        signal_length: Original signal length
        strategy: Heuristic strategy (uses default if None)
        
    Returns:
        Reconstructed matrix with weighted components
    """
    if strategy is None:
        strategy = HeuristicMultiFeatureStrategy()
    
    weights = strategy.get_weights(u, s, vh, signal_length)
    weighted_s = s * weights
    
    # Reconstruct
    return (u * weighted_s) @ vh
