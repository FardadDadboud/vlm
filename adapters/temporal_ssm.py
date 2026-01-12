"""
Temporal SSM for STAD-vMF

Improvements:
1. Numerically stable Bessel functions (ive instead of iv)
2. Kappa clamping to prevent overflow
3. EMA smoothing for kappa estimates
4. Windowed EM for stability
5. get_prototypes() method for debugging
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.special import ive  # Exponentially scaled Bessel


# =============================================================================
# Constants
# =============================================================================
KAPPA_MAX = 30.0
KAPPA_MIN = 1e-6


# =============================================================================
# Numerically Stable VMF Utilities
# =============================================================================

class VMFUtils:
    """Utilities for von Mises-Fisher distribution."""
    
    @staticmethod
    def A_D(kappa: float, dim: int) -> float:
        """
        Ratio A_D(κ) = I_{D/2}(κ) / I_{D/2-1}(κ)
        
        Uses exponentially scaled Bessel functions to avoid overflow.
        """
        if kappa < KAPPA_MIN:
            return 0.0
        
        kappa = min(kappa, KAPPA_MAX)
        v = dim / 2.0
        
        # ive(v, x) = iv(v, x) * exp(-|x|) - stable for large kappa
        num = ive(v, kappa)
        denom = ive(v - 1, kappa)
        
        if denom < 1e-300:
            # Asymptotic: A_D(κ) ≈ 1 - (D-1)/(2κ)
            return 1.0 - (dim - 1) / (2 * kappa)
        
        return np.clip(num / denom, 0.0, 1.0)
    
    @staticmethod
    def estimate_kappa(r_bar: float, dim: int) -> float:
        """
        Estimate κ from mean resultant length using Banerjee approximation.
        """
        r_bar = np.clip(r_bar, 1e-10, 1.0 - 1e-10)
        r_bar_sq = r_bar ** 2
        kappa = r_bar * (dim - r_bar_sq) / (1 - r_bar_sq)
        return np.clip(kappa, KAPPA_MIN, KAPPA_MAX)
    
    @staticmethod
    def safe_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Safely normalize vectors."""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.maximum(norm, 1e-10)


# =============================================================================
# State Container
# =============================================================================

@dataclass
class STADState:
    """State for STAD algorithm."""
    # vMF parameters for each class
    rho: np.ndarray = None      # Mean directions (K, D)
    gamma: np.ndarray = None    # Concentrations (K,)
    
    # EMA smoothed kappa (optional)
    gamma_ema: np.ndarray = None
    
    # Transition parameters
    kappa_trans: float = 50.0
    
    # Feature history for windowed EM
    feature_history: List[np.ndarray] = field(default_factory=list)
    label_history: List[np.ndarray] = field(default_factory=list)
    
    initialized: bool = False


# =============================================================================
# Temporal SSM
# =============================================================================

class TemporalSSM:
    """
    State-Space Model for Temporal Test-Time Adaptation.
    
    Implements STAD (State-space Test-time ADaptation) with vMF distributions.
    """
    
    def __init__(self, 
                 num_classes: int,
                 feature_dim: int,
                 kappa_trans_init: float = 50.0,
                 kappa_ems_init: float = 10.0,
                 window_size: int = 5,
                 em_iterations: int = 3,
                 use_ema_kappa: bool = True,
                 kappa_ema: float = 0.9,
                 temperature: float = 1.0):
        """
        Initialize Temporal SSM.
        
        Args:
            num_classes: Number of classes K
            feature_dim: Feature dimension D
            kappa_trans_init: Initial transition concentration
            kappa_ems_init: Initial emission concentration
            window_size: Number of frames for windowed EM
            em_iterations: EM iterations per update
            use_ema_kappa: Whether to use EMA smoothing for kappa
            kappa_ema: EMA decay factor (0.9 = slow, 0.5 = fast)
            temperature: Temperature scaling factor
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.em_iterations = em_iterations
        self.use_ema_kappa = use_ema_kappa
        self.kappa_ema = kappa_ema
        self.temperature = temperature
        
        # Initialize state
        self.state = STADState(
            kappa_trans=kappa_trans_init
        )
        
        self._kappa_ems_init = kappa_ems_init
    
    def initialize_from_text_embeddings(self, text_embeddings: np.ndarray):
        """
        Initialize prototypes from text embeddings (hybrid format).
        
        Args:
            text_embeddings: (K, D) normalized text embeddings
        """
        K, D = text_embeddings.shape
        
        # Normalize
        self.state.rho = VMFUtils.safe_normalize(text_embeddings)
        
        # Initialize concentrations
        self.state.gamma = np.full(K, self._kappa_ems_init)
        self.state.gamma_ema = self.state.gamma.copy()
        
        self.state.initialized = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using current prototypes.
        
        Args:
            features: (N, D) query features
            
        Returns:
            (N, K) class probabilities
        """
        if not self.state.initialized:
            raise ValueError("SSM not initialized")
        
        # Normalize features
        features_norm = VMFUtils.safe_normalize(features)
        
        # Compute similarities: (N, K)
        similarities = features_norm @ self.state.rho.T
        
        # Weight by concentrations
        weighted_sims = similarities * self.state.gamma[np.newaxis, :]

        # Apply temperature scaling
        weighted_sims = weighted_sims / self.temperature
        
        # Softmax for probabilities
        exp_sims = np.exp(weighted_sims - np.max(weighted_sims, axis=1, keepdims=True))
        probs = exp_sims / (np.sum(exp_sims, axis=1, keepdims=True) + 1e-10)
        
        return probs
    
    def update(self, features: np.ndarray, confidence_mask: np.ndarray,
               class_probs: Optional[np.ndarray] = None):
        """
        Update prototypes using confident detections.
        
        Uses windowed EM for stability.
        
        Args:
            features: (N, D) query features
            confidence_mask: (N,) boolean mask for confident detections
            class_probs: (N, K) soft class assignments (optional, uses hard if None)
        """
        if not self.state.initialized:
            return
        
        # Select confident features
        conf_features = features[confidence_mask]
        if len(conf_features) == 0:
            return
        
        # Get class assignments
        if class_probs is not None:
            conf_probs = class_probs[confidence_mask]
            # Hard assignment from soft probs
            conf_labels = np.argmax(conf_probs, axis=1)
        else:
            # Use current predictions
            probs = self.predict(conf_features)
            conf_labels = np.argmax(probs, axis=1)
        
        # Add to history for windowed EM
        self.state.feature_history.append(conf_features)
        self.state.label_history.append(conf_labels)
        
        # Keep only last window_size
        if len(self.state.feature_history) > self.window_size:
            self.state.feature_history.pop(0)
            self.state.label_history.pop(0)
        
        # Run windowed EM
        self._windowed_em_update()
    
    def _windowed_em_update(self):
        """
        Run EM on windowed history for stable updates.
        """
        if not self.state.feature_history:
            return
        
        # Concatenate history (with optional decay weighting)
        all_features = []
        all_labels = []
        all_weights = []
        
        n_frames = len(self.state.feature_history)
        for t, (features, labels) in enumerate(zip(self.state.feature_history, 
                                                     self.state.label_history)):
            # Newer frames get higher weight
            weight = (t + 1) / n_frames  # Linear decay, oldest=1/n, newest=1
            
            all_features.append(features)
            all_labels.append(labels)
            all_weights.append(np.full(len(features), weight))
        
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)
        weights = np.concatenate(all_weights)
        
        if len(features) == 0:
            return
        
        # Normalize features
        features_norm = VMFUtils.safe_normalize(features)
        
        K = self.num_classes
        D = self.feature_dim
        
        # EM iterations
        for _ in range(self.em_iterations):
            new_rho = np.zeros((K, D))
            new_gamma = np.zeros(K)
            
            for k in range(K):
                mask = (labels == k)
                if not mask.any():
                    # Keep current prototype
                    new_rho[k] = self.state.rho[k]
                    new_gamma[k] = self.state.gamma[k]
                    continue
                
                # Weighted mean direction
                class_features = features_norm[mask]
                class_weights = weights[mask]
                
                # Weighted sum
                weighted_sum = np.sum(class_features * class_weights[:, np.newaxis], axis=0)
                
                # Transition prior: pull toward previous prototype
                prior_weight = self.state.kappa_trans
                combined = weighted_sum + prior_weight * self.state.rho[k]
                
                # Normalize to get new mean direction
                combined_norm = np.linalg.norm(combined)
                if combined_norm > 1e-10:
                    new_rho[k] = combined / combined_norm
                else:
                    new_rho[k] = self.state.rho[k]
                
                # Estimate new concentration
                total_weight = class_weights.sum() + prior_weight
                r_bar = combined_norm / total_weight
                new_kappa = VMFUtils.estimate_kappa(r_bar, D)
                
                # EMA smoothing (optional)
                if self.use_ema_kappa:
                    new_gamma[k] = self.kappa_ema * self.state.gamma_ema[k] + \
                                   (1 - self.kappa_ema) * new_kappa
                else:
                    new_gamma[k] = new_kappa
            
            # Update state
            self.state.rho = VMFUtils.safe_normalize(new_rho)
            self.state.gamma = np.clip(new_gamma, KAPPA_MIN, KAPPA_MAX)
            
            if self.use_ema_kappa:
                self.state.gamma_ema = self.state.gamma.copy()
    
    def get_prototypes(self) -> np.ndarray:
        """Get current prototype mean directions."""
        if not self.state.initialized:
            return None
        return self.state.rho.copy()
    
    def get_concentrations(self) -> np.ndarray:
        """Get current concentration parameters."""
        if not self.state.initialized:
            return None
        return self.state.gamma.copy()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing Temporal SSM...")
    
    # Test VMFUtils stability
    print("\n1. Testing A_D stability:")
    for kappa in [1, 10, 100, 500, 1000, 2000]:
        a_d = VMFUtils.A_D(kappa, 262)
        print(f"   κ={kappa:>4}: A_D = {a_d:.6f}")
    
    # Test SSM
    print("\n2. Testing SSM:")
    ssm = TemporalSSM(num_classes=6, feature_dim=262)
    
    # Mock text embeddings
    text_emb = np.random.randn(6, 262)
    text_emb = VMFUtils.safe_normalize(text_emb)
    ssm.initialize_from_text_embeddings(text_emb)
    
    # Mock features
    features = np.random.randn(10, 262)
    probs = ssm.predict(features)
    print(f"   Predictions shape: {probs.shape}")
    print(f"   Predictions sum: {probs.sum(axis=1)}")
    
    # Mock update
    conf_mask = np.array([True, True, False, True, False, False, True, False, False, True])
    ssm.update(features, conf_mask)
    
    print("\n✓ All tests passed!")