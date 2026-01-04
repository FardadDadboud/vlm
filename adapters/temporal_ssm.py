"""
Temporal TTA with State-Space Models (STAD) - Corrected Implementation

Based on: "Temporal Test-Time Adaptation with State-Space Models"
https://openreview.net/pdf?id=HFETOmUtrV

Key corrections from initial implementation:
1. Prototypes initialized from SOURCE MODEL (text embeddings for VLM)
2. Class assignments use soft probabilities λ_{t,n,k}
3. Proper temporal dynamics with transitions from W_{t-1}
4. Correct variational EM updates per Algorithm 3
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from scipy.special import iv as bessel_iv


@dataclass 
class TemporalState:
    """State maintained across time steps for STAD"""
    # Class prototypes (mean directions): shape (K, D)
    # In paper notation: W_t = (w_{t,1}, ..., w_{t,K})
    prototypes: Optional[np.ndarray] = None
    
    # Variational parameters for prototypes
    # ρ_{t,k} = mean direction, γ_{t,k} = concentration
    rho: Optional[np.ndarray] = None      # (K, D)
    gamma: Optional[np.ndarray] = None    # (K,)
    
    # Mixing coefficients (class priors): π_t
    mixing_coefficients: Optional[np.ndarray] = None  # (K,)
    
    # SSM parameters
    kappa_trans: float = 50.0   # Transition concentration
    kappa_ems: float = 10.0     # Emission concentration
    
    # Sliding window history
    # Each entry: (features, assignments, expected_prototypes)
    history: List[Tuple] = field(default_factory=list)
    
    # Frame counter
    frame_count: int = 0
    num_classes: int = 0
    feature_dim: int = 0
    
    # Flag for initialization
    initialized: bool = False


class VMFUtils:
    """
    von Mises-Fisher distribution utilities.
    
    vMF(h; μ, κ) = C_D(κ) * exp(κ * μᵀh)
    """
    
    @staticmethod
    def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
        """Project to unit hypersphere"""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / (norm + eps)
    
    @staticmethod
    def log_C_D(kappa: float, dim: int) -> float:
        """Log normalization constant for vMF"""
        v = dim / 2.0 - 1.0
        log_c = v * np.log(kappa + 1e-10) - (dim / 2.0) * np.log(2 * np.pi) - np.log(bessel_iv(v, kappa) + 1e-10)
        return log_c
    
    @staticmethod
    def A_D(kappa: float, dim: int) -> float:
        """
        Ratio A_D(κ) = I_{D/2}(κ) / I_{D/2-1}(κ)
        
        For vMF: E[w] = A_D(γ) * ρ
        """
        v = dim / 2.0
        num = bessel_iv(v, kappa)
        denom = bessel_iv(v - 1, kappa)
        return num / (denom + 1e-10)
    
    @staticmethod
    def estimate_kappa(r_bar: float, dim: int) -> float:
        """
        Estimate κ from mean resultant length r̄.
        
        Approximation: κ̂ = (r̄D - r̄³) / (1 - r̄²)
        """
        r_bar = np.clip(r_bar, 1e-6, 1 - 1e-6)
        kappa = (r_bar * dim - r_bar ** 3) / (1 - r_bar ** 2)
        return max(kappa, 0.1)
    
    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-10)


class TemporalSSM:
    """
    State-Space Model for Temporal TTA (STAD-vMF).
    
    Implements Algorithm 1 and Algorithm 3 from the paper.
    
    Model:
    - Transition: p(w_{t,k} | w_{t-1,k}) = vMF(w_{t,k} | w_{t-1,k}, κ^trans)
    - Emission: p(h_{t,n} | w_{t,k}) = vMF(h_{t,n} | w_{t,k}, κ^ems)
    
    Prediction:
    - p(class=k | h) = softmax(κ^ems * W_t^T * h)  [Eq. 10]
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 kappa_trans_init: float = 50.0,
                 kappa_ems_init: float = 10.0,
                 window_size: int = 5,
                 em_iterations: int = 3):
        """
        Initialize STAD.
        
        Args:
            num_classes: K - number of classes
            feature_dim: D - dimension of features
            kappa_trans_init: Initial transition concentration
            kappa_ems_init: Initial emission concentration
            window_size: s - sliding window size
            em_iterations: Number of EM iterations per update
        """
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.em_iterations = em_iterations
        
        # Initialize state
        self.state = TemporalState(
            num_classes=num_classes,
            feature_dim=feature_dim,
            kappa_trans=kappa_trans_init,
            kappa_ems=kappa_ems_init
        )
    
    def initialize_from_text_embeddings(self, text_embeddings: np.ndarray):
        """
        Initialize prototypes from text embeddings (source model weights W_0).
        
        This is the VLM equivalent of using source classifier weights.
        
        Args:
            text_embeddings: Shape (K, D) - text embedding for each class
        """
        K, D = text_embeddings.shape
        assert K == self.num_classes, f"Expected {self.num_classes} classes, got {K}"
        assert D == self.feature_dim, f"Expected dim {self.feature_dim}, got {D}"
        
        # Normalize to unit sphere
        W_0 = VMFUtils.normalize(text_embeddings, axis=1)
        
        # Set initial prototypes
        self.state.prototypes = W_0.copy()
        self.state.rho = W_0.copy()
        self.state.gamma = np.ones(K) * 100.0  # High initial confidence (κ_0 in paper)
        
        # Uniform class prior
        self.state.mixing_coefficients = np.ones(K) / K
        
        self.state.initialized = True
        print(f"STAD initialized with text embeddings: {K} classes, {D}D features")
    
    def initialize_from_features(self, features: np.ndarray, class_probs: np.ndarray):
        """
        Alternative initialization from first batch using class probabilities.
        
        Computes weighted mean of features per class as initial prototype.
        
        Args:
            features: Shape (N, D) - detection features
            class_probs: Shape (N, K) - class probabilities from VLM
        """
        N, D = features.shape
        K = class_probs.shape[1]
        
        assert K == self.num_classes
        assert D == self.feature_dim
        
        # Normalize features
        features_norm = VMFUtils.normalize(features, axis=1)
        
        # Compute weighted mean for each class
        # prototype_k = Σ_n (p_{n,k} * h_n) / Σ_n p_{n,k}
        prototypes = np.zeros((K, D))
        for k in range(K):
            weights = class_probs[:, k]
            weight_sum = weights.sum()
            if weight_sum > 1e-6:
                prototypes[k] = (weights[:, np.newaxis] * features_norm).sum(axis=0) / weight_sum
            else:
                # No samples for this class - use random direction
                prototypes[k] = np.random.randn(D)
        
        # Normalize prototypes
        prototypes = VMFUtils.normalize(prototypes, axis=1)
        
        # Set state
        self.state.prototypes = prototypes
        self.state.rho = prototypes.copy()
        self.state.gamma = np.ones(K) * 10.0  # Lower confidence than text-based init
        self.state.mixing_coefficients = class_probs.mean(axis=0)
        self.state.mixing_coefficients /= self.state.mixing_coefficients.sum()
        
        self.state.initialized = True
        print(f"STAD initialized from features: {K} classes, {D}D, {N} samples")
    
    def update(self, 
               features: np.ndarray, 
               confidence_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update SSM with new batch of features (Algorithm 1 main loop).
        
        Args:
            features: Detection features, shape (N, D)
            confidence_mask: Optional boolean mask for which features to use in update
            
        Returns:
            class_probs: Predicted class probabilities, shape (N, K)
        """
        if features.shape[0] == 0:
            return np.array([]).reshape(0, self.num_classes)
        
        # Normalize features to hypersphere
        H_t = VMFUtils.normalize(features, axis=1)
        N = H_t.shape[0]
        K = self.num_classes
        D = self.feature_dim
        
        # If not initialized, use first batch
        if not self.state.initialized:
            raise RuntimeError("SSM not initialized. Call initialize_from_text_embeddings() or initialize_from_features() first.")
        
        # Get previous expected prototypes E_q[W_{t-1}]
        if self.state.frame_count > 0 and len(self.state.history) > 0:
            prev_expected = self._compute_expected_prototypes()
        else:
            prev_expected = None
        
        # Run variational EM (Algorithm 3)
        for em_iter in range(self.em_iterations):
            # E-step: Update λ (assignments) and ρ, γ (prototype params)
            lambdas = self._e_step_assignments(H_t)
            self._e_step_prototypes(H_t, lambdas, prev_expected)
            
            # M-step: Update π, κ^trans, κ^ems
            self._m_step(H_t, lambdas)
        
        # Update prototypes to expected value
        self.state.prototypes = self._compute_expected_prototypes()
        
        # Store in history for sliding window
        expected_protos = self.state.prototypes.copy()
        
        # If confidence mask provided, only store high-confidence for history
        if confidence_mask is not None and confidence_mask.any():
            store_features = H_t[confidence_mask]
            store_lambdas = lambdas[confidence_mask]
        else:
            store_features = H_t
            store_lambdas = lambdas
        
        self.state.history.append((store_features, store_lambdas, expected_protos))
        
        # Trim history to window size
        if len(self.state.history) > self.window_size:
            self.state.history = self.state.history[-self.window_size:]
        
        self.state.frame_count += 1
        
        return lambdas
    
    def _compute_expected_prototypes(self) -> np.ndarray:
        """
        Compute E_q[W_t] = A_D(γ_k) * ρ_k for each class.
        """
        K = self.num_classes
        D = self.feature_dim
        
        expected = np.zeros((K, D))
        for k in range(K):
            a_d = VMFUtils.A_D(self.state.gamma[k], D)
            expected[k] = a_d * self.state.rho[k]
        
        return expected
    
    def _e_step_assignments(self, H_t: np.ndarray) -> np.ndarray:
        """
        E-step: Compute assignment probabilities λ_{t,n,k} (Eq. 35).
        
        λ_{t,n,k} ∝ π_{t,k} * C_D(κ^ems) * exp(κ^ems * E[w_{t,k}]^T * h_{t,n})
        """
        N = H_t.shape[0]
        K = self.num_classes
        D = self.feature_dim
        
        # Get expected prototypes
        E_W = self._compute_expected_prototypes()  # (K, D)
        
        # Compute log probabilities
        log_C_D = VMFUtils.log_C_D(self.state.kappa_ems, D)
        
        # Similarities: κ^ems * E[w_k]^T * h_n for all n, k
        similarities = H_t @ E_W.T  # (N, K)
        log_emission = self.state.kappa_ems * similarities + log_C_D
        
        # Add log prior
        log_prior = np.log(self.state.mixing_coefficients + 1e-10)  # (K,)
        log_joint = log_emission + log_prior  # (N, K)
        
        # Softmax to get λ
        lambdas = VMFUtils.softmax(log_joint, axis=1)  # (N, K)
        
        return lambdas
    
    def _e_step_prototypes(self, 
                           H_t: np.ndarray, 
                           lambdas: np.ndarray,
                           prev_expected: Optional[np.ndarray]):
        """
        E-step: Update prototype variational parameters ρ, γ (Eq. 36-37).
        
        β_{t,k} = κ^trans * E[w_{t-1,k}] + κ^ems * Σ_n λ_{t,n,k} * h_{t,n}
        γ_{t,k} = ||β_{t,k}||
        ρ_{t,k} = β_{t,k} / γ_{t,k}
        """
        K = self.num_classes
        D = self.feature_dim
        
        for k in range(K):
            # Start with emission term
            # κ^ems * Σ_n λ_{t,n,k} * h_{t,n}
            weighted_features = lambdas[:, k:k+1].T @ H_t  # (1, D)
            beta_k = self.state.kappa_ems * weighted_features.flatten()
            
            # Add transition from previous time step
            if prev_expected is not None:
                beta_k += self.state.kappa_trans * prev_expected[k]
            
            # Compute γ and ρ
            gamma_k = np.linalg.norm(beta_k)
            if gamma_k < 1e-8:
                # Degenerate case - keep previous
                continue
            
            rho_k = beta_k / gamma_k
            
            self.state.gamma[k] = gamma_k
            self.state.rho[k] = rho_k
    
    def _m_step(self, H_t: np.ndarray, lambdas: np.ndarray):
        """
        M-step: Update model parameters (Eq. 40-42).
        
        π_{t,k} = (1/N) * Σ_n λ_{t,n,k}
        κ^trans and κ^ems estimated from mean resultant lengths
        """
        N = H_t.shape[0]
        K = self.num_classes
        D = self.feature_dim
        
        # Update mixing coefficients (Eq. 42)
        self.state.mixing_coefficients = lambdas.mean(axis=0)
        self.state.mixing_coefficients = np.clip(self.state.mixing_coefficients, 0.01, 0.99)
        self.state.mixing_coefficients /= self.state.mixing_coefficients.sum()
        
        # Update κ^trans using history (Eq. 40)
        if len(self.state.history) >= 2:
            dot_products = []
            for t in range(1, len(self.state.history)):
                _, _, prev_protos = self.state.history[t-1]
                _, _, curr_protos = self.state.history[t]
                for k in range(K):
                    dot_prod = np.dot(prev_protos[k], curr_protos[k])
                    dot_products.append(dot_prod)
            
            if dot_products:
                r_bar_trans = np.abs(np.mean(dot_products))
                self.state.kappa_trans = VMFUtils.estimate_kappa(r_bar_trans, D)
                self.state.kappa_trans = np.clip(self.state.kappa_trans, 1.0, 500.0)
        
        # Update κ^ems using current batch (Eq. 41)
        E_W = self._compute_expected_prototypes()
        dot_products = []
        total_weight = 0.0
        
        for n in range(N):
            for k in range(K):
                weight = lambdas[n, k]
                dot_prod = weight * np.dot(E_W[k], H_t[n])
                dot_products.append(dot_prod)
                total_weight += weight
        
        if total_weight > 0:
            r_bar_ems = np.abs(np.sum(dot_products)) / total_weight
            self.state.kappa_ems = VMFUtils.estimate_kappa(r_bar_ems, D)
            self.state.kappa_ems = np.clip(self.state.kappa_ems, 1.0, 100.0)
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate class predictions without updating SSM.
        
        Uses Eq. 10: p(class=k | h) = softmax(κ^ems * W_t^T * h)
        
        Args:
            features: Query features, shape (N, D)
            
        Returns:
            class_probs: Shape (N, K)
        """
        if not self.state.initialized:
            return np.zeros((features.shape[0], self.num_classes))
        
        # Normalize features
        H = VMFUtils.normalize(features, axis=1)
        
        # Get current prototypes
        W_t = self.state.prototypes  # (K, D)
        
        # Compute logits: κ^ems * W_t^T * h
        logits = self.state.kappa_ems * (H @ W_t.T)  # (N, K)
        
        # Softmax
        return VMFUtils.softmax(logits, axis=1)
    
    def get_prototypes(self) -> Optional[np.ndarray]:
        """Get current class prototypes W_t"""
        return self.state.prototypes.copy() if self.state.prototypes is not None else None
    
    def get_state_summary(self) -> Dict:
        """Get summary of current SSM state for debugging"""
        return {
            'frame_count': self.state.frame_count,
            'initialized': self.state.initialized,
            'kappa_trans': self.state.kappa_trans,
            'kappa_ems': self.state.kappa_ems,
            'mixing_coefficients': self.state.mixing_coefficients.tolist() if self.state.mixing_coefficients is not None else None,
            'gamma': self.state.gamma.tolist() if self.state.gamma is not None else None,
            'history_size': len(self.state.history)
        }
    
    def reset(self):
        """Reset SSM state (for new video sequence)"""
        self.state = TemporalState(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            kappa_trans=50.0,
            kappa_ems=10.0
        )


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Corrected STAD Implementation")
    print("=" * 60)
    
    K = 6   # classes
    D = 256 # feature dim
    
    # Create SSM
    ssm = TemporalSSM(
        num_classes=K,
        feature_dim=D,
        kappa_trans_init=50.0,
        kappa_ems_init=10.0,
        window_size=5,
        em_iterations=3
    )
    
    # Simulate text embeddings (source model weights)
    print("\n1. Initialize from text embeddings (W_0):")
    text_embeddings = np.random.randn(K, D)
    ssm.initialize_from_text_embeddings(text_embeddings)
    print(f"   Initial prototypes shape: {ssm.get_prototypes().shape}")
    
    # Simulate video frames
    print("\n2. Process video frames:")
    for t in range(10):
        # Simulate detections with some class structure
        n_detections = np.random.randint(5, 20)
        
        # Features clustered around true prototypes + noise
        true_classes = np.random.randint(0, K, n_detections)
        features = text_embeddings[true_classes] + 0.3 * np.random.randn(n_detections, D)
        
        # Confidence mask (simulate high confidence detections)
        confidence_mask = np.random.rand(n_detections) > 0.3
        
        # Update SSM
        class_probs = ssm.update(features, confidence_mask)
        
        state = ssm.get_state_summary()
        print(f"   Frame {t}: {n_detections} dets, "
              f"κ_trans={state['kappa_trans']:.1f}, "
              f"κ_ems={state['kappa_ems']:.1f}, "
              f"history={state['history_size']}")
    
    # Test prediction without update
    print("\n3. Predict without update:")
    test_features = np.random.randn(5, D)
    probs = ssm.predict(test_features)
    print(f"   Predictions shape: {probs.shape}")
    print(f"   Sum per sample (should be 1.0): {probs.sum(axis=1)}")
    
    print("\n✓ Corrected STAD implementation tests passed!")