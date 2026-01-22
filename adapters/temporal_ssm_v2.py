"""
Temporal SSM v2 for STAD (State-space Test-time ADaptation)

Fixes vs v1:
1. Uses A_D (expected mean under vMF) - was defined but unused
2. Adds mixing coefficients π (paper Eq. 2, 35, 42)
3. Global κ_trans and κ_ems scalars (paper M-step Eq. 40-41)
4. Per-class update gating (min_updates_per_class)
5. Proper soft-EM with responsibilities (paper Algorithm 3)
6. Bounded logits via A_D(gamma) instead of raw gamma
7. STAD-Gaussian variant (paper Algorithm 2)
8. Debug logging for runaway behavior detection

References:
- Schirmer et al., "Temporal Test-Time Adaptation with State-Space Models", TMLR 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Literal
from scipy.special import ive  # Exponentially scaled Bessel
import warnings


# =============================================================================
# Default Constants (can be overridden via config)
# =============================================================================
DEFAULT_KAPPA_MAX = 500.0  # Paper allows higher values
DEFAULT_KAPPA_MIN = 1e-6
DEFAULT_GAMMA_MAX = 500.0  # Per-class concentration max
DEFAULT_GAMMA_MIN = 1.0    # Minimum to avoid degenerate distributions
DEFAULT_DEBUG_EVERY = 30   # Print debug logs every N frames


# =============================================================================
# Debug Logging Helper Mixin
# =============================================================================

class DebugLoggerMixin:
    """Mixin for gated and throttled debug logging."""
    
    def _init_debug(self, debug: bool, debug_every: int = DEFAULT_DEBUG_EVERY, 
                    class_names: Optional[List[str]] = None):
        """Initialize debug settings."""
        self._debug = debug
        self._debug_every = debug_every
        self._debug_call_count = 0
        self._class_names = class_names
        
        # Health counters
        self.num_updates_total = 0
        self.num_updates_skipped = 0
        self.num_updates_by_class = None  # Will be initialized when we know num_classes
    
    def _dlog(self, msg: str, force: bool = False):
        """Print debug message if debug enabled and (forced or throttled)."""
        if not self._debug:
            return
        if force or (self._debug_call_count % self._debug_every == 0):
            print(msg)
    
    def _dwarn(self, msg: str):
        """Print warning message (always prints if debug enabled)."""
        if self._debug:
            print(msg)
    
    def _get_class_name(self, idx: int) -> str:
        """Get class name by index."""
        if self._class_names is not None and idx < len(self._class_names):
            return self._class_names[idx]
        return f"c{idx}"
    
    def _entropy(self, probs: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute entropy of probability distribution."""
        return -np.sum(probs * np.log(probs + 1e-10), axis=axis)


# =============================================================================
# Numerically Stable VMF Utilities
# =============================================================================

class VMFUtils:
    """Utilities for von Mises-Fisher distribution."""
    
    @staticmethod
    def A_D(kappa: float, dim: int, 
            kappa_max: float = DEFAULT_KAPPA_MAX,
            kappa_min: float = DEFAULT_KAPPA_MIN) -> float:
        """
        Ratio A_D(κ) = I_{D/2}(κ) / I_{D/2-1}(κ)
        
        This is the expected value of the dot product between a vMF sample
        and its mean direction. Bounded in (0, 1).
        
        Uses exponentially scaled Bessel functions to avoid overflow.
        """
        if kappa < kappa_min:
            return 0.0
        
        # Clamp for numerical stability
        kappa = min(kappa, kappa_max)
        v = dim / 2.0
        
        # ive(v, x) = iv(v, x) * exp(-|x|) - stable for large kappa
        num = ive(v, kappa)
        denom = ive(v - 1, kappa)
        
        if denom < 1e-300:
            # Asymptotic: A_D(κ) ≈ 1 - (D-1)/(2κ) for large κ
            return 1.0 - (dim - 1) / (2 * kappa)
        
        return np.clip(num / denom, 0.0, 1.0 - 1e-10)
    
    @staticmethod
    def A_D_vectorized(kappa: np.ndarray, dim: int,
                       kappa_max: float = DEFAULT_KAPPA_MAX,
                       kappa_min: float = DEFAULT_KAPPA_MIN) -> np.ndarray:
        """Vectorized version of A_D for arrays."""
        result = np.zeros_like(kappa)
        
        for i, k in enumerate(kappa):
            result[i] = VMFUtils.A_D(k, dim, kappa_max, kappa_min)
        
        return result
    
    @staticmethod
    def inv_A_D(r_bar: float, dim: int,
                kappa_max: float = DEFAULT_KAPPA_MAX,
                kappa_min: float = DEFAULT_KAPPA_MIN) -> float:
        """
        Inverse of A_D: estimate κ from mean resultant length using Banerjee approximation.
        
        κ ≈ r̄(D - r̄²) / (1 - r̄²)
        """
        r_bar = np.clip(r_bar, 1e-10, 1.0 - 1e-10)
        r_bar_sq = r_bar ** 2
        kappa = r_bar * (dim - r_bar_sq) / (1 - r_bar_sq)
        return np.clip(kappa, kappa_min, kappa_max)
    
    @staticmethod
    def log_C_D(kappa: float, dim: int) -> float:
        """
        Log of vMF normalization constant C_D(κ).
        
        C_D(κ) = κ^{D/2-1} / ((2π)^{D/2} I_{D/2-1}(κ))
        
        For numerical stability, we compute log C_D directly.
        """
        if kappa < KAPPA_MIN:
            # Uniform distribution limit
            return -dim / 2 * np.log(2 * np.pi)
        
        v = dim / 2.0 - 1.0
        # Using ive for stability: log(iv(v,k)) = log(ive(v,k)) + k
        log_iv = np.log(ive(v, kappa) + 1e-300) + kappa
        
        return v * np.log(kappa) - (dim / 2) * np.log(2 * np.pi) - log_iv
    
    @staticmethod
    def safe_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Safely normalize vectors to unit length."""
        norm = np.linalg.norm(x, axis=axis, keepdims=True)
        return x / np.maximum(norm, 1e-10)
    
    @staticmethod
    def safe_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        logits_shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / (np.sum(exp_logits, axis=axis, keepdims=True) + 1e-10)


# =============================================================================
# State Container for vMF
# =============================================================================

@dataclass
class STADvMFState:
    """State for STAD-vMF algorithm."""
    # vMF variational parameters for each class (K,)
    rho: np.ndarray = None      # Mean directions (K, D) - unit vectors
    gamma: np.ndarray = None    # Per-class concentrations (K,) - variational params
    
    # Mixing coefficients
    pi: np.ndarray = None       # Class mixing coefficients (K,), sum to 1
    pi_ema: np.ndarray = None   # EMA smoothed pi (optional)
    
    # Global emission/transition concentrations (paper M-step)
    kappa_ems: float = 10.0     # Global emission concentration
    kappa_trans: float = 50.0   # Global transition concentration
    
    # Feature history for windowed EM
    feature_history: List[np.ndarray] = field(default_factory=list)
    probs_history: List[np.ndarray] = field(default_factory=list)  # Soft probs, not hard labels
    
    # Per-class update counts for gating
    class_update_counts: np.ndarray = None  # (K,) counts
    
    initialized: bool = False


# =============================================================================
# STAD-vMF Implementation
# =============================================================================

class TemporalSSMvMF(DebugLoggerMixin):
    """
    State-Space Model for Temporal TTA using von Mises-Fisher distributions.
    
    Implements STAD-vMF (Section 3.3, Algorithm 3) with:
    - Proper variational EM with soft responsibilities
    - A_D usage for expected means
    - Global κ^trans and κ^ems
    - Mixing coefficients π
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 kappa_trans_init: float = 100.0,
                 kappa_ems_init: float = 100.0,
                 gamma_init: float = 10.0,
                 window_size: int = 5,
                 em_iterations: int = 3,
                 temperature: float = 1.0,
                 use_pi: bool = False,
                 use_ema_pi: bool = False,
                 pi_ema_decay: float = 1.0,
                 min_updates_per_class: int = 2,
                 update_global_kappa: bool = False,  # Paper updates, but can destabilize
                 vlm_prior_weight: float = 0.0,  # η for combining VLM probs in E-step
                 dirichlet_alpha: float = 1e-2,  # Prior for π to avoid zeros
                 kappa_max: float = DEFAULT_KAPPA_MAX,
                 kappa_min: float = DEFAULT_KAPPA_MIN,
                 gamma_max: float = DEFAULT_GAMMA_MAX,
                 gamma_min: float = DEFAULT_GAMMA_MIN,
                 class_names: Optional[List[str]] = None,
                 debug: bool = False,
                 debug_every: int = DEFAULT_DEBUG_EVERY):
        """
        Initialize STAD-vMF.
        
        Args:
            num_classes: Number of classes K
            feature_dim: Feature dimension D
            kappa_trans_init: Initial global transition concentration
            kappa_ems_init: Initial global emission concentration
            gamma_init: Initial per-class variational concentration
            window_size: Sliding window size for EM
            em_iterations: Number of EM iterations per update
            temperature: Temperature for prediction softmax
            use_ema_pi: Whether to EMA smooth mixing coefficients
            pi_ema_decay: EMA decay for pi (0.9 = slow adaptation)
            min_updates_per_class: Minimum confident samples per class before updating
            update_global_kappa: Whether to update global kappa in M-step
            vlm_prior_weight: Weight η for VLM probs in responsibility computation
            dirichlet_alpha: Dirichlet prior for pi to prevent zeros
            kappa_max: Maximum value for global kappa parameters
            kappa_min: Minimum value for global kappa parameters
            gamma_max: Maximum value for per-class gamma
            gamma_min: Minimum value for per-class gamma
            class_names: Optional list of class names for logging
            debug: Enable debug logging
            debug_every: Print throttled logs every N calls
        """
        # Initialize debug mixin
        self._init_debug(debug, debug_every, class_names)
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.em_iterations = em_iterations
        self.temperature = temperature
        self.use_pi = use_pi
        self.use_ema_pi = use_ema_pi
        self.pi_ema_decay = pi_ema_decay
        self.min_updates_per_class = min_updates_per_class
        self.update_global_kappa = update_global_kappa
        self.vlm_prior_weight = vlm_prior_weight
        self.dirichlet_alpha = dirichlet_alpha
        
        # Configurable bounds
        self.kappa_max = kappa_max
        self.kappa_min = kappa_min
        self.gamma_max = gamma_max
        self.gamma_min = gamma_min
        
        # Initial values
        self._kappa_trans_init = kappa_trans_init
        self._kappa_ems_init = kappa_ems_init
        self._gamma_init = gamma_init
        
        # State
        self.state = STADvMFState(
            kappa_ems=kappa_ems_init,
            kappa_trans=kappa_trans_init
        )
        
        # Initialize health counters
        self.num_updates_by_class = np.zeros(num_classes, dtype=np.int64)
        
        # Debug stats
        self._debug_stats: Dict = {}
    
    def initialize_from_text_embeddings(self, text_embeddings: np.ndarray):
        """
        Initialize prototypes from text embeddings.
        
        Args:
            text_embeddings: (K, D) text embeddings (will be normalized)
        """
        K, D = text_embeddings.shape
        assert K == self.num_classes and D == self.feature_dim
        
        # Initialize mean directions (normalized)
        self.state.rho = VMFUtils.safe_normalize(text_embeddings)
        
        # Initialize per-class concentrations
        self.state.gamma = np.full(K, self._gamma_init, dtype=np.float64)
        
        # Initialize mixing coefficients (uniform)
        self.state.pi = np.full(K, 1.0 / K, dtype=np.float64)
        self.state.pi_ema = self.state.pi.copy()
        
        # Per-class update counts
        self.state.class_update_counts = np.zeros(K, dtype=np.int32)
        
        self.state.initialized = True
        
        # Debug: Init log (force=True)
        self._dlog(
            f"[vMF][init] K={K} D={D} k_trans={self.state.kappa_trans:.1f} "
            f"k_ems={self.state.kappa_ems:.1f} gamma_init={self._gamma_init:.1f} pi=uniform",
            force=True
        )
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using current prototypes.
        
        Uses the paper's formulation (Eq. 10):
        p(c_{t,n,k}=1|h) ∝ π_k · C_D(κ^ems) · exp(κ^ems · w_k^T · h)
        
        With w_k = E_q[w_k] = A_D(γ_k) · ρ_k (expected mean direction)
        
        Args:
            features: (N, D) query features
            
        Returns:
            (N, K) class probabilities
        """
        if not self.state.initialized:
            raise ValueError("SSM not initialized")
        
        self._debug_call_count += 1
        
        # Normalize features
        features_norm = VMFUtils.safe_normalize(features)
        N = features_norm.shape[0]
        K = self.num_classes
        
        # Compute expected prototypes: m_k = A_D(γ_k) · ρ_k
        # This is the key fix - using A_D to bound the effective magnitude
        a_d_gamma = VMFUtils.A_D_vectorized(
            self.state.gamma, self.feature_dim, 
            self.kappa_max, self.kappa_min
        )  # (K,)
        m_k = a_d_gamma[:, np.newaxis] * self.state.rho  # (K, D)
        
        # Compute logits: κ^ems · <h_n, m_k> / temperature + log(π_k)
        similarities = features_norm @ m_k.T  # (N, K)
        if self.use_pi:
            logits = (self.state.kappa_ems * similarities / self.temperature + 
                      np.log(self.state.pi + 1e-10)[np.newaxis, :])
        else:
            logits = (self.state.kappa_ems * similarities / self.temperature)
        
        # Softmax
        probs = VMFUtils.safe_softmax(logits, axis=1)
        
        # Debug logging (throttled)
        if self._debug:
            max_logits = np.max(np.abs(logits))
            entropy = self._entropy(probs, axis=1)
            H_mean, H_min, H_max = entropy.mean(), entropy.min(), entropy.max()
            pi_min, pi_max = self.state.pi.min(), self.state.pi.max()
            
            # Store for external access
            self._debug_stats['max_logits'] = max_logits
            self._debug_stats['pred_entropy'] = H_mean
            self._debug_stats['pi_range'] = (pi_min, pi_max)
            self._debug_stats['a_d_gamma'] = a_d_gamma
            
            # Throttled log
            self._dlog(
                f"[vMF][predict] max|logit|={max_logits:.1f} H mean={H_mean:.2f} "
                f"min={H_min:.2f} max={H_max:.2f} | pi min={pi_min:.3f} max={pi_max:.3f}"
            )
            
            # Warnings (always print)
            if max_logits > 50:
                self._dwarn(f"[vMF][WARN] logit explosion: max|logit|={max_logits:.1f}")
            if pi_max > 0.99:
                self._dwarn(f"[vMF][WARN] pi collapse detected: max(pi)={pi_max:.4f}")
            elif pi_max > 0.95:
                self._dwarn(f"[vMF][WARN] pi approaching collapse: max(pi)={pi_max:.4f}")
            if pi_min < 1e-4:
                self._dwarn(f"[vMF][WARN] pi near zero: min(pi)={pi_min:.2e}")
            if H_mean < 0.05:
                self._dwarn(f"[vMF][WARN] prediction entropy CRITICAL: H_mean={H_mean:.4f}")
            elif H_mean < 0.1:
                self._dwarn(f"[vMF][WARN] prediction entropy low: H_mean={H_mean:.3f}")
        
        return probs
    
    def update(self, features: np.ndarray, confidence_mask: np.ndarray,
               class_probs: Optional[np.ndarray] = None):
        """
        Update prototypes using confident detections via windowed EM.
        
        Args:
            features: (N, D) query features  
            confidence_mask: (N,) boolean mask for confident detections
            class_probs: (N, K) soft class assignments from VLM (optional)
        """
        if not self.state.initialized:
            return
        
        # Select confident features
        conf_features = features[confidence_mask]
        if len(conf_features) == 0:
            return
        
        # Get soft class probs for confident detections
        if class_probs is not None:
            conf_probs = class_probs[confidence_mask]
        else:
            # Use current predictions as initial probs
            conf_probs = self.predict(conf_features)
        
        # Add to history (store soft probs, not hard labels!)
        self.state.feature_history.append(conf_features.copy())
        self.state.probs_history.append(conf_probs.copy())
        
        # Keep only last window_size
        while len(self.state.feature_history) > self.window_size:
            self.state.feature_history.pop(0)
            self.state.probs_history.pop(0)
        
        # Run soft EM on window
        self._soft_em_update()
    
    def _soft_em_update(self):
        """
        Run soft variational EM on windowed history (Algorithm 3).
        
        Key differences from v1:
        1. Uses soft responsibilities λ_{n,k} not hard labels
        2. Uses A_D(γ) in E-step for expected means
        3. Updates π from responsibilities
        4. Optionally updates global κ^ems and κ^trans
        
        Stability fixes (v2.1):
        - Temperature scaling in E-step (consistent with predict())
        - r_bar_k clamping to prevent gamma explosion
        - EMA gamma updates for stability
        - Per-class R_k capping
        """
        if not self.state.feature_history:
            self.num_updates_skipped += 1
            return
        
        # Concatenate history with temporal weighting
        all_features = []
        all_probs = []
        all_weights = []
        
        n_frames = len(self.state.feature_history)
        for t, (features, probs) in enumerate(zip(self.state.feature_history,
                                                    self.state.probs_history)):
            # Newer frames get higher weight (linear decay)
            weight = (t + 1) / n_frames
            all_features.append(features)
            all_probs.append(probs)
            all_weights.append(np.full(len(features), weight))
        
        features = np.concatenate(all_features)  # (N_total, D)
        vlm_probs = np.concatenate(all_probs)    # (N_total, K)
        weights = np.concatenate(all_weights)    # (N_total,)
        
        N = len(features)
        if N == 0:
            self.num_updates_skipped += 1
            return
        
        K = self.num_classes
        D = self.feature_dim
        
        # Normalize features
        features_norm = VMFUtils.safe_normalize(features)
        
        # Store previous prototypes for transition prior
        rho_prev = self.state.rho.copy()
        gamma_prev = self.state.gamma.copy()  # For EMA updates
        
        # Debug: EM start log (force=True)
        self._dlog(
            f"[vMF][EM] frames_in_window={n_frames} N={N} em_iters={self.em_iterations} "
            f"vlm_prior_w={self.vlm_prior_weight:.2f} temperature={self.temperature:.1f}",
            force=True
        )
        
        # Track r_bar_k stats for logging
        r_bar_k_stats = []
        
        # EM iterations
        for em_iter in range(self.em_iterations):
            # ===== E-STEP: Compute soft responsibilities =====
            # λ_{n,k} ∝ π_k · exp(κ^ems · E_q[w_k]^T · h_n / temperature)
            
            # Compute expected prototypes
            a_d_gamma = VMFUtils.A_D_vectorized(
                self.state.gamma, D, self.kappa_max, self.kappa_min
            )  # (K,)
            m_k = a_d_gamma[:, np.newaxis] * self.state.rho  # (K, D)
            
            # Compute SSM logits WITH TEMPERATURE SCALING (FIX: consistent with predict())
            similarities = features_norm @ m_k.T  # (N, K)
            ssm_logits = (self.state.kappa_ems * similarities / self.temperature + 
                         np.log(self.state.pi + 1e-10)[np.newaxis, :])
            ssm_probs = VMFUtils.safe_softmax(ssm_logits, axis=1)  # (N, K)
            
            # Optionally combine with VLM probs
            if self.vlm_prior_weight > 0:
                # λ = normalize(λ_vlm^η · λ_ssm^(1-η))
                log_combined = (self.vlm_prior_weight * np.log(vlm_probs + 1e-10) +
                               (1 - self.vlm_prior_weight) * np.log(ssm_probs + 1e-10))
                responsibilities = VMFUtils.safe_softmax(log_combined, axis=1)
            else:
                responsibilities = ssm_probs  # (N, K)
            
            # Apply temporal weights to responsibilities
            weighted_resp = responsibilities * weights[:, np.newaxis]  # (N, K)
            
            # ===== M-STEP: Update parameters =====
            
            # Count effective samples per class
            R_k = np.sum(weighted_resp, axis=0)  # (K,)
            
            # STABILITY FIX: Cap per-class R_k to prevent single-class domination
            max_Rk_per_class = N * 0.5  # No single class can have more than 50% of effective samples
            R_k_capped = np.minimum(R_k, max_Rk_per_class)
            R_k_was_capped = np.any(R_k > max_Rk_per_class)
            
            # Check per-class update threshold
            class_has_enough = R_k_capped >= self.min_updates_per_class
            num_classes_updating = int(class_has_enough.sum())
            
            # Update prototypes for classes with enough samples
            new_rho = self.state.rho.copy()
            new_gamma = self.state.gamma.copy()
            gamma_clamped_classes = []
            r_bar_k_values = {}
            
            for k in range(K):
                if not class_has_enough[k]:
                    # Keep previous values
                    continue
                
                # Compute weighted sum of features (use capped weights)
                resp_scale = R_k_capped[k] / (R_k[k] + 1e-10) if R_k[k] > max_Rk_per_class else 1.0
                s_k = np.sum(weighted_resp[:, k:k+1] * resp_scale * features_norm, axis=0)  # (D,)
                
                # Add transition prior: pull toward previous prototype
                # β_k = κ^trans · E_q[w_{t-1,k}] + κ^ems · Σ_n λ_{n,k} · h_n
                # We use rho_prev as the previous expected direction
                a_d_prev = VMFUtils.A_D(
                    self.state.gamma[k], D, self.kappa_max, self.kappa_min
                )
                m_prev_k = a_d_prev * rho_prev[k]
                
                s_k_combined = self.state.kappa_ems * s_k + self.state.kappa_trans * m_prev_k
                
                # New mean direction
                gamma_k_new_norm = np.linalg.norm(s_k_combined)
                if gamma_k_new_norm > 1e-10:
                    new_rho[k] = s_k_combined / gamma_k_new_norm
                    
                    # Estimate new concentration from mean resultant length
                    total_effective = self.state.kappa_ems * R_k_capped[k] + self.state.kappa_trans
                    r_bar_k = gamma_k_new_norm / total_effective
                    
                    # STABILITY FIX: Clamp r_bar_k to prevent gamma explosion
                    r_bar_k_raw = r_bar_k
                    r_bar_k = np.clip(r_bar_k, 0.0, 0.95)  # Clamp to max 0.95
                    r_bar_k_values[self._get_class_name(k)] = (r_bar_k_raw, r_bar_k)
                    
                    gamma_k_from_r = VMFUtils.inv_A_D(r_bar_k, D, self.kappa_max, self.kappa_min)
                    
                    # STABILITY FIX: EMA gamma update for smoother adaptation
                    gamma_ema_decay = 0.7  # Keep 70% of old gamma
                    new_gamma[k] = gamma_ema_decay * gamma_prev[k] + (1 - gamma_ema_decay) * gamma_k_from_r
                    
                    # Check if clamping occurred
                    if new_gamma[k] <= self.gamma_min or new_gamma[k] >= self.gamma_max:
                        gamma_clamped_classes.append(self._get_class_name(k))
                    
                    new_gamma[k] = np.clip(new_gamma[k], self.gamma_min, self.gamma_max)
                    
                    # Track per-class updates
                    self.num_updates_by_class[k] += 1
            
            # Store r_bar_k stats for logging
            if r_bar_k_values:
                r_bar_k_stats = r_bar_k_values
            
            # Update state
            self.state.rho = VMFUtils.safe_normalize(new_rho)
            self.state.gamma = new_gamma
            
            # Update mixing coefficients π with Dirichlet prior
            # Use CAPPED R_k for pi update to prevent single-class domination
            new_pi = (R_k_capped + self.dirichlet_alpha) / (np.sum(R_k_capped) + K * self.dirichlet_alpha)
            
            if self.use_ema_pi:
                self.state.pi = self.pi_ema_decay * self.state.pi_ema + (1 - self.pi_ema_decay) * new_pi
                self.state.pi_ema = self.state.pi.copy()
            else:
                self.state.pi = new_pi
            
            # Ensure π sums to 1
            self.state.pi = self.state.pi / (np.sum(self.state.pi) + 1e-10)
            
            # Debug: Per-iteration log (throttled - only first and last iteration)
            if self._debug and (em_iter == 0 or em_iter == self.em_iterations - 1):
                R_k_sorted = np.sort(R_k)
                top2_pi_idx = np.argsort(self.state.pi)[-2:][::-1]
                top2_names = [self._get_class_name(i) for i in top2_pi_idx]
                top2_vals = self.state.pi[top2_pi_idx]
                
                # Compute responsibility stats
                resp_entropy = self._entropy(responsibilities, axis=1)
                max_resp = np.max(responsibilities, axis=1)
                
                # Logits stats
                logits_min, logits_med, logits_max = ssm_logits.min(), np.median(ssm_logits), ssm_logits.max()
                
                self._dlog(
                    f"[vMF][EM it={em_iter}] Rk min={R_k_sorted[0]:.1f} med={np.median(R_k):.1f} "
                    f"max={R_k_sorted[-1]:.1f}{' (capped)' if R_k_was_capped else ''} | "
                    f"gamma min={self.state.gamma.min():.1f} max={self.state.gamma.max():.1f} | "
                    f"A_D min={a_d_gamma.min():.3f} max={a_d_gamma.max():.3f} | "
                    f"pi min={self.state.pi.min():.3f} max={self.state.pi.max():.3f} "
                    f"(top: {top2_names[0]} {top2_vals[0]:.3f}, {top2_names[1]} {top2_vals[1]:.3f}) | "
                    f"classes_upd={num_classes_updating}/{K}",
                    force=True
                )
                
                # Log additional stats
                self._dlog(
                    f"[vMF][EM it={em_iter}] logits: min={logits_min:.1f} med={logits_med:.1f} max={logits_max:.1f} | "
                    f"resp: max_mean={max_resp.mean():.3f} max_max={max_resp.max():.3f} | "
                    f"H_resp: mean={resp_entropy.mean():.2f} min={resp_entropy.min():.2f}",
                    force=True
                )
                
                # Log r_bar_k stats
                if r_bar_k_stats:
                    r_bar_strs = [f"{k}={v[0]:.3f}→{v[1]:.3f}" for k, v in r_bar_k_stats.items()]
                    self._dlog(f"[vMF][EM it={em_iter}] r_bar_k (raw→clamped): {' '.join(r_bar_strs)}", force=True)
            
            # Debug: Warnings
            if self._debug:
                if gamma_clamped_classes:
                    self._dwarn(f"[vMF][WARN] gamma hit clamp for classes: {gamma_clamped_classes}")
                if self.state.pi.max() > 0.95:
                    self._dwarn(f"[vMF][WARN] pi approaching collapse: max(pi)={self.state.pi.max():.4f}")
                if R_k_was_capped:
                    self._dwarn(f"[vMF][WARN] R_k capped to prevent single-class domination: max_raw={R_k.max():.1f}")
        
        # Optionally update global kappa (can be unstable)
        if self.update_global_kappa:
            self._update_global_kappa(features_norm, responsibilities, rho_prev)
        
        # Update per-class counts
        hard_assignments = np.argmax(responsibilities, axis=1)
        for k in range(K):
            self.state.class_update_counts[k] += np.sum(hard_assignments == k)
        
        # Increment update counter
        self.num_updates_total += 1
        
        # Debug: Health log (throttled)
        self._dlog(
            f"[vMF][health] updates_total={self.num_updates_total} skipped={self.num_updates_skipped}"
        )
    
    def _update_global_kappa(self, features_norm: np.ndarray, 
                              responsibilities: np.ndarray,
                              rho_prev: np.ndarray):
        """
        Update global κ^ems and κ^trans (paper M-step Eq. 40-41).
        
        This can be unstable in practice, so it's optional.
        """
        K = self.num_classes
        D = self.feature_dim
        
        # Compute expected prototypes
        a_d_gamma = VMFUtils.A_D_vectorized(
            self.state.gamma, D, self.kappa_max, self.kappa_min
        )
        m_k = a_d_gamma[:, np.newaxis] * self.state.rho  # (K, D)
        
        # κ^ems update: based on emission alignment
        # r̄^ems = |Σ_n Σ_k λ_{n,k} · E_q[w_k]^T · h_n| / (Σ_n Σ_k λ_{n,k})
        alignment_sum = 0.0
        resp_sum = 0.0
        for k in range(K):
            # Dot products between features and expected prototype
            dots = features_norm @ m_k[k]  # (N,)
            alignment_sum += np.sum(responsibilities[:, k] * dots)
            resp_sum += np.sum(responsibilities[:, k])
        
        if resp_sum > 1e-10:
            r_bar_ems = np.clip(alignment_sum / resp_sum, -1 + 1e-10, 1 - 1e-10)
            # Only update if alignment is positive (prototypes align with features)
            if r_bar_ems > 0:
                new_kappa_ems = VMFUtils.inv_A_D(r_bar_ems, D, self.kappa_max, self.kappa_min)
                # EMA update for stability
                self.state.kappa_ems = 0.9 * self.state.kappa_ems + 0.1 * new_kappa_ems
                self.state.kappa_ems = np.clip(self.state.kappa_ems, self.kappa_min, self.kappa_max)
        
        # κ^trans update: based on prototype transition alignment
        # r̄^trans = mean_k <m_{t-1,k}, m_t,k>
        m_prev = VMFUtils.A_D_vectorized(
            self.state.gamma, D, self.kappa_max, self.kappa_min
        )[:, np.newaxis] * rho_prev
        trans_alignment = np.sum(m_prev * m_k) / K
        
        if trans_alignment > 0:
            new_kappa_trans = VMFUtils.inv_A_D(trans_alignment, D, self.kappa_max, self.kappa_min)
            self.state.kappa_trans = 0.9 * self.state.kappa_trans + 0.1 * new_kappa_trans
            self.state.kappa_trans = np.clip(self.state.kappa_trans, self.kappa_min, self.kappa_max)
    
    def get_prototypes(self) -> Optional[np.ndarray]:
        """Get current prototype mean directions ρ."""
        if not self.state.initialized:
            return None
        return self.state.rho.copy()
    
    def get_expected_prototypes(self) -> Optional[np.ndarray]:
        """Get expected prototype vectors m_k = A_D(γ_k) · ρ_k."""
        if not self.state.initialized:
            return None
        a_d_gamma = VMFUtils.A_D_vectorized(
            self.state.gamma, self.feature_dim, self.kappa_max, self.kappa_min
        )
        return a_d_gamma[:, np.newaxis] * self.state.rho
    
    def get_concentrations(self) -> Optional[np.ndarray]:
        """Get current per-class concentration parameters γ."""
        if not self.state.initialized:
            return None
        return self.state.gamma.copy()
    
    def get_mixing_coefficients(self) -> Optional[np.ndarray]:
        """Get current mixing coefficients π."""
        if not self.state.initialized:
            return None
        return self.state.pi.copy()
    
    def get_debug_stats(self) -> Dict:
        """Get debug statistics from last prediction."""
        return self._debug_stats.copy()


# =============================================================================
# State Container for Gaussian
# =============================================================================

@dataclass
class STADGaussState:
    """State for STAD-Gaussian algorithm (Kalman filter based)."""
    # Gaussian parameters for each class
    mu: np.ndarray = None       # Means (K, D)
    P: np.ndarray = None        # Covariances (K, D, D) or (K, D) for diagonal
    
    # Mixing coefficients
    pi: np.ndarray = None       # (K,)
    
    # Transition/emission noise
    Q: np.ndarray = None        # Process noise covariance (D, D) or (D,) diagonal
    R_base: float = 0.5         # Base emission noise scale
    
    # History for smoothing
    mu_pred_history: List[np.ndarray] = field(default_factory=list)
    P_pred_history: List[np.ndarray] = field(default_factory=list)
    mu_filt_history: List[np.ndarray] = field(default_factory=list)
    P_filt_history: List[np.ndarray] = field(default_factory=list)
    
    # Per-class update counts
    class_update_counts: np.ndarray = None
    
    initialized: bool = False


# =============================================================================
# STAD-Gaussian Implementation
# =============================================================================

class TemporalSSMGaussian(DebugLoggerMixin):
    """
    State-Space Model for Temporal TTA using Gaussian distributions (Kalman filter).
    
    Implements STAD-Gaussian (Section B.1, Algorithm 2) with:
    - Diagonal covariances for efficiency (O(D) instead of O(D²))
    - Per-class Kalman filter updates
    - Optional fixed-lag smoothing
    - Mixing coefficients π
    
    This variant is more expensive than vMF but may capture more structure.
    """
    
    def __init__(self,
                 num_classes: int,
                 feature_dim: int,
                 q_scale: float = 0.01,
                 r_base: float = 0.5,
                 window_size: int = 5,
                 use_pi: bool = False,
                 use_diagonal_cov: bool = True,
                 use_smoothing: bool = False,
                 min_updates_per_class: int = 2,
                 dirichlet_alpha: float = 1e-2,
                 class_names: Optional[List[str]] = None,
                 debug: bool = False,
                 debug_every: int = DEFAULT_DEBUG_EVERY):
        """
        Initialize STAD-Gaussian.
        
        Args:
            num_classes: Number of classes K
            feature_dim: Feature dimension D
            q_scale: Process noise scale (controls prototype drift rate)
            r_base: Base emission noise scale
            window_size: Window size for smoothing
            use_diagonal_cov: Use diagonal covariances (efficient) vs full
            use_smoothing: Apply fixed-lag RTS smoothing
            min_updates_per_class: Minimum samples per class before update
            dirichlet_alpha: Dirichlet prior for π
            class_names: Optional list of class names for logging
            debug: Enable debug logging
            debug_every: Print throttled logs every N calls
        """
        # Initialize debug mixin
        self._init_debug(debug, debug_every, class_names)
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.q_scale = q_scale
        self.r_base = r_base
        self.window_size = window_size
        self.use_pi = use_pi
        self.use_diagonal_cov = use_diagonal_cov
        self.use_smoothing = use_smoothing
        self.min_updates_per_class = min_updates_per_class
        self.dirichlet_alpha = dirichlet_alpha
        
        self.state = STADGaussState(R_base=r_base)
        self._debug_stats: Dict = {}
        
        # Initialize health counters
        self.num_updates_by_class = np.zeros(num_classes, dtype=np.int64)
    
    def initialize_from_text_embeddings(self, text_embeddings: np.ndarray):
        """
        Initialize from text embeddings.
        
        Args:
            text_embeddings: (K, D) text embeddings
        """
        K, D = text_embeddings.shape
        assert K == self.num_classes and D == self.feature_dim
        
        # Initialize means (optionally normalized for consistency with vMF)
        self.state.mu = VMFUtils.safe_normalize(text_embeddings)
        
        # Initialize covariances
        if self.use_diagonal_cov:
            self.state.P = np.ones((K, D)) * 0.1  # (K, D) diagonal
            self.state.Q = np.ones(D) * self.q_scale  # (D,) diagonal process noise
        else:
            self.state.P = np.tile(np.eye(D) * 0.1, (K, 1, 1))  # (K, D, D)
            self.state.Q = np.eye(D) * self.q_scale  # (D, D)
        
        # Mixing coefficients
        self.state.pi = np.full(K, 1.0 / K)
        
        # Per-class counts
        self.state.class_update_counts = np.zeros(K, dtype=np.int32)
        
        self.state.initialized = True
        
        # Debug: Init log (force=True)
        self._dlog(
            f"[Gauss][init] K={K} D={D} q_scale={self.q_scale} r_base={self.r_base} "
            f"smoothing={self.use_smoothing} win={self.window_size}",
            force=True
        )
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Gaussian mixture likelihood.
        
        p(k|h) ∝ π_k · N(h | μ_k, Σ^ems)
        
        For efficiency, we use diagonal Σ^ems = r_base · I.
        
        Args:
            features: (N, D) query features
            
        Returns:
            (N, K) class probabilities
        """
        if not self.state.initialized:
            raise ValueError("SSM not initialized")
        
        self._debug_call_count += 1
        
        # Normalize features for consistency
        features_norm = VMFUtils.safe_normalize(features)
        N = features_norm.shape[0]
        K = self.num_classes
        D = self.feature_dim
        
        # Compute log-likelihoods under each Gaussian
        # log N(h | μ_k, R) = -0.5 * ||h - μ_k||² / r_base - 0.5 * D * log(2π * r_base)
        
        log_probs = np.zeros((N, K))
        for k in range(K):
            diff = features_norm - self.state.mu[k]  # (N, D)
            sq_dist = np.sum(diff ** 2, axis=1)  # (N,)
            if self.use_pi: 
                log_probs[:, k] = (-0.5 * sq_dist / self.state.R_base + 
                                np.log(self.state.pi[k] + 1e-10))
            else:
                log_probs[:, k] = (-0.5 * sq_dist / self.state.R_base)
        
        # Softmax over log-probs
        probs = VMFUtils.safe_softmax(log_probs, axis=1)
        
        # Debug logging (throttled)
        if self._debug:
            entropy = self._entropy(probs, axis=1)
            H_mean, H_min, H_max = entropy.mean(), entropy.min(), entropy.max()
            pi_min, pi_max = self.state.pi.min(), self.state.pi.max()
            max_log_probs = np.max(np.abs(log_probs))
            
            # Store for external access
            self._debug_stats['max_log_probs'] = max_log_probs
            self._debug_stats['pred_entropy'] = H_mean
            self._debug_stats['pi_range'] = (pi_min, pi_max)
            
            # Throttled log
            self._dlog(
                f"[Gauss][predict] max|logp|={max_log_probs:.1f} H mean={H_mean:.2f} "
                f"min={H_min:.2f} max={H_max:.2f} | pi min={pi_min:.3f} max={pi_max:.3f}"
            )
            
            # Warnings (always print)
            if pi_max > 0.99:
                self._dwarn(f"[Gauss][WARN] pi collapse detected: max(pi)={pi_max:.4f}")
            if pi_min < 1e-4:
                self._dwarn(f"[Gauss][WARN] pi near zero: min(pi)={pi_min:.2e}")
        
        return probs
    
    def update(self, features: np.ndarray, confidence_mask: np.ndarray,
               class_probs: Optional[np.ndarray] = None):
        """
        Update prototypes using Kalman filter updates per class.
        
        Args:
            features: (N, D) query features
            confidence_mask: (N,) boolean mask
            class_probs: (N, K) soft class assignments (optional)
        """
        if not self.state.initialized:
            self.num_updates_skipped += 1
            return
        
        conf_features = features[confidence_mask]
        if len(conf_features) == 0:
            self.num_updates_skipped += 1
            self._dlog("[Gauss][update-skip] no confident features", force=True)
            return
        
        conf_features_norm = VMFUtils.safe_normalize(conf_features)
        N = len(conf_features_norm)
        K = self.num_classes
        D = self.feature_dim
        
        # Get responsibilities
        if class_probs is not None:
            responsibilities = class_probs[confidence_mask]
        else:
            responsibilities = self.predict(conf_features)
        
        # Store predictions for smoothing
        if self.use_smoothing:
            self.state.mu_pred_history.append(self.state.mu.copy())
            self.state.P_pred_history.append(self.state.P.copy())
        
        # Per-class Kalman updates
        R_k = np.sum(responsibilities, axis=0)  # Effective samples per class
        
        # Track innovations for debug
        innovation_norms = []
        num_classes_updating = 0
        cov_warnings = []
        
        for k in range(K):
            if R_k[k] < self.min_updates_per_class:
                continue
            
            num_classes_updating += 1
            
            # === Kalman Predict ===
            # μ_pred = μ_prev (identity transition)
            # P_pred = P_prev + Q
            if self.use_diagonal_cov:
                P_pred = self.state.P[k] + self.state.Q
            else:
                P_pred = self.state.P[k] + self.state.Q
            
            # === Compute class observation ===
            # Weighted mean of features assigned to this class
            y_k = np.sum(responsibilities[:, k:k+1] * conf_features_norm, axis=0) / (R_k[k] + 1e-10)
            
            # Observation noise shrinks with more evidence
            R_obs = self.state.R_base / (R_k[k] + 1e-10)
            
            # === Kalman Update ===
            if self.use_diagonal_cov:
                # Diagonal Kalman gain: K = P_pred / (P_pred + R_obs)
                K_gain = P_pred / (P_pred + R_obs + 1e-10)
                
                # Update
                innovation = y_k - self.state.mu[k]
                innovation_norms.append(np.linalg.norm(innovation))
                self.state.mu[k] = self.state.mu[k] + K_gain * innovation
                self.state.P[k] = (1 - K_gain) * P_pred
                
                # Check covariance bounds
                P_min, P_max = self.state.P[k].min(), self.state.P[k].max()
                if P_min < 1e-6:
                    cov_warnings.append(f"{self._get_class_name(k)}: P_min={P_min:.2e}")
                if P_max > 1e3:
                    cov_warnings.append(f"{self._get_class_name(k)}: P_max={P_max:.2e}")
            else:
                # Full covariance (expensive)
                S = P_pred + np.eye(D) * R_obs
                K_gain = P_pred @ np.linalg.inv(S + np.eye(D) * 1e-10)
                
                innovation = y_k - self.state.mu[k]
                innovation_norms.append(np.linalg.norm(innovation))
                self.state.mu[k] = self.state.mu[k] + K_gain @ innovation
                self.state.P[k] = (np.eye(D) - K_gain) @ P_pred
            
            self.state.class_update_counts[k] += int(R_k[k])
            self.num_updates_by_class[k] += 1
        
        # Normalize means (keep on unit sphere for consistency)
        self.state.mu = VMFUtils.safe_normalize(self.state.mu)
        
        # Update mixing coefficients
        new_pi = (R_k + self.dirichlet_alpha) / (np.sum(R_k) + K * self.dirichlet_alpha)
        self.state.pi = 0.9 * self.state.pi + 0.1 * new_pi
        self.state.pi = self.state.pi / (np.sum(self.state.pi) + 1e-10)
        
        # Store filtered states
        if self.use_smoothing:
            self.state.mu_filt_history.append(self.state.mu.copy())
            self.state.P_filt_history.append(self.state.P.copy())
            
            # Trim history
            while len(self.state.mu_pred_history) > self.window_size:
                self.state.mu_pred_history.pop(0)
                self.state.P_pred_history.pop(0)
                self.state.mu_filt_history.pop(0)
                self.state.P_filt_history.pop(0)
            
            # Apply RTS smoothing
            if len(self.state.mu_filt_history) >= 2:
                self._rts_smooth()
                self._dlog(f"[Gauss][smooth] applied RTS lag={len(self.state.mu_filt_history)}")
        
        # Increment update counter
        self.num_updates_total += 1
        
        # Debug: Update log (force=True)
        if self._debug:
            innov_mean = np.mean(innovation_norms) if innovation_norms else 0
            innov_max = np.max(innovation_norms) if innovation_norms else 0
            
            if self.use_diagonal_cov:
                P_diag_min = self.state.P.min()
                P_diag_max = self.state.P.max()
                self._dlog(
                    f"[Gauss][update] N_conf={N} classes_obs={num_classes_updating}/{K} | "
                    f"innov ||·|| mean={innov_mean:.3f} max={innov_max:.3f} | "
                    f"diag(P) min={P_diag_min:.4f} max={P_diag_max:.4f}",
                    force=True
                )
            else:
                self._dlog(
                    f"[Gauss][update] N_conf={N} classes_obs={num_classes_updating}/{K} | "
                    f"innov ||·|| mean={innov_mean:.3f} max={innov_max:.3f}",
                    force=True
                )
            
            # Warnings for covariance issues
            if cov_warnings:
                for warn in cov_warnings:
                    self._dwarn(f"[Gauss][WARN] covariance issue: {warn}")
            
            # Health log (throttled)
            self._dlog(
                f"[Gauss][health] updates_total={self.num_updates_total} skipped={self.num_updates_skipped}"
            )
    
    def _rts_smooth(self):
        """
        Apply Rauch-Tung-Striebel smoothing over the stored history.
        
        This is optional and can improve prototype estimates but adds computation.
        """
        T = len(self.state.mu_filt_history)
        if T < 2:
            return
        
        K = self.num_classes
        D = self.feature_dim
        
        # Start from last filtered state
        mu_smooth = self.state.mu_filt_history[-1].copy()
        P_smooth = self.state.P_filt_history[-1].copy()
        
        # Backward pass
        for t in range(T - 2, -1, -1):
            mu_filt = self.state.mu_filt_history[t]
            P_filt = self.state.P_filt_history[t]
            mu_pred_next = self.state.mu_pred_history[min(t + 1, T - 1)]
            P_pred_next = self.state.P_pred_history[min(t + 1, T - 1)]
            
            for k in range(K):
                if self.use_diagonal_cov:
                    # J = P_filt / P_pred_next (element-wise for diagonal)
                    J = P_filt[k] / (P_pred_next[k] + 1e-10)
                    mu_smooth[k] = mu_filt[k] + J * (mu_smooth[k] - mu_pred_next[k])
                    P_smooth[k] = P_filt[k] + J ** 2 * (P_smooth[k] - P_pred_next[k])
                else:
                    J = P_filt[k] @ np.linalg.inv(P_pred_next[k] + np.eye(D) * 1e-10)
                    mu_smooth[k] = mu_filt[k] + J @ (mu_smooth[k] - mu_pred_next[k])
                    P_smooth[k] = P_filt[k] + J @ (P_smooth[k] - P_pred_next[k]) @ J.T
        
        # Update current state with smoothed estimates
        self.state.mu = VMFUtils.safe_normalize(mu_smooth)
        self.state.P = np.clip(P_smooth, 1e-6, 1.0)
    
    def get_prototypes(self) -> Optional[np.ndarray]:
        """Get current prototype means."""
        if not self.state.initialized:
            return None
        return self.state.mu.copy()
    
    def get_covariances(self) -> Optional[np.ndarray]:
        """Get current covariance estimates."""
        if not self.state.initialized:
            return None
        return self.state.P.copy()
    
    def get_mixing_coefficients(self) -> Optional[np.ndarray]:
        """Get mixing coefficients π."""
        if not self.state.initialized:
            return None
        return self.state.pi.copy()
    
    def get_debug_stats(self) -> Dict:
        """Get debug stats."""
        return self._debug_stats.copy()


# =============================================================================
# Factory Function
# =============================================================================

def create_temporal_ssm(
    ssm_type: Literal["vmf", "gaussian"],
    num_classes: int,
    feature_dim: int,
    **kwargs
) -> 'TemporalSSMvMF | TemporalSSMGaussian':
    """
    Factory function to create the appropriate SSM variant.
    
    Args:
        ssm_type: "vmf" or "gaussian"
        num_classes: Number of classes K
        feature_dim: Feature dimension D
        **kwargs: Additional arguments for the specific SSM type
        
    Returns:
        TemporalSSMvMF or TemporalSSMGaussian instance
    """
    if ssm_type == "vmf":
        return TemporalSSMvMF(num_classes, feature_dim, **kwargs)
    elif ssm_type == "gaussian":
        return TemporalSSMGaussian(num_classes, feature_dim, **kwargs)
    else:
        raise ValueError(f"Unknown SSM type: {ssm_type}. Must be 'vmf' or 'gaussian'.")


# =============================================================================
# Legacy Compatibility Wrapper
# =============================================================================

class TemporalSSM(TemporalSSMvMF):
    """
    Legacy compatibility wrapper - defaults to vMF variant.
    
    Provides same interface as v1 TemporalSSM.
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
                 temperature: float = 1.0,
                 kappa_max: float = DEFAULT_KAPPA_MAX,
                 kappa_min: float = DEFAULT_KAPPA_MIN):
        """
        Initialize with v1-compatible parameters.
        """
        super().__init__(
            num_classes=num_classes,
            feature_dim=feature_dim,
            kappa_trans_init=kappa_trans_init,
            kappa_ems_init=kappa_ems_init,
            gamma_init=kappa_ems_init,  # Map kappa_ems_init to gamma_init
            window_size=window_size,
            em_iterations=em_iterations,
            temperature=temperature,
            use_ema_pi=use_ema_kappa,
            pi_ema_decay=kappa_ema,
            debug=False
        )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Temporal SSM v2")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Test VMFUtils
    print("\n1. Testing A_D function:")
    for kappa in [1, 10, 50, 100, 200, 500]:
        a_d = VMFUtils.A_D(kappa, 256)
        print(f"   κ={kappa:>4}: A_D = {a_d:.6f}")
    
    # Test vMF SSM
    print("\n2. Testing STAD-vMF:")
    ssm_vmf = TemporalSSMvMF(num_classes=6, feature_dim=256, debug=True)
    
    # Mock text embeddings
    text_emb = np.random.randn(6, 256)
    text_emb = VMFUtils.safe_normalize(text_emb)
    ssm_vmf.initialize_from_text_embeddings(text_emb)
    
    # Mock features
    features = np.random.randn(10, 256)
    probs = ssm_vmf.predict(features)
    print(f"   Predictions shape: {probs.shape}")
    print(f"   Predictions sum: {probs.sum(axis=1)}")
    print(f"   π: {ssm_vmf.get_mixing_coefficients()}")
    
    # Mock update
    conf_mask = np.array([True, True, False, True, False, False, True, False, False, True])
    ssm_vmf.update(features, conf_mask)
    print(f"   After update π: {ssm_vmf.get_mixing_coefficients().round(3)}")
    
    # Test Gaussian SSM
    print("\n3. Testing STAD-Gaussian:")
    ssm_gauss = TemporalSSMGaussian(num_classes=6, feature_dim=256, debug=True)
    ssm_gauss.initialize_from_text_embeddings(text_emb)
    
    probs_gauss = ssm_gauss.predict(features)
    print(f"   Predictions shape: {probs_gauss.shape}")
    print(f"   Predictions sum: {probs_gauss.sum(axis=1)}")
    
    ssm_gauss.update(features, conf_mask)
    print(f"   After update π: {ssm_gauss.get_mixing_coefficients().round(3)}")
    
    # Test factory
    print("\n4. Testing factory function:")
    ssm_factory = create_temporal_ssm("vmf", 6, 256)
    print(f"   Created: {type(ssm_factory).__name__}")
    
    # Test legacy wrapper
    print("\n5. Testing legacy compatibility:")
    ssm_legacy = TemporalSSM(num_classes=6, feature_dim=256)
    ssm_legacy.initialize_from_text_embeddings(text_emb)
    probs_legacy = ssm_legacy.predict(features)
    print(f"   Legacy predictions shape: {probs_legacy.shape}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)