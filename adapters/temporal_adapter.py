"""
Temporal TTA Adapter for VLM-based Object Detection - Corrected Version

Implements STAD (State-space Test-time Adaptation) for GroundingDINO.

Key design decisions:
1. Text embeddings serve as initial prototypes (W_0 in paper)
2. Query features from decoder are the observations (h_t,n)
3. Prototypes evolve over time using vMF state-space model
4. Prediction fuses VLM scores with prototype-based scores

Based on: "Temporal Test-Time Adaptation with State-Space Models"
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from PIL import Image
from .base_adapter import BaseAdapter
from .temporal_ssm import TemporalSSM, VMFUtils


class TemporalTTAAdapter(BaseAdapter):
    """
    Temporal Test-Time Adaptation for VLM-based object detection.
    
    Architecture mapping (Paper → GroundingDINO):
    - Source classifier W_0 → Text embeddings at class token positions
    - Feature representation h → Query features from decoder
    - Prediction W^T h → Query-prototype similarity (replaces text-query similarity)
    """
    
    def __init__(self, detector, config: dict):
        """
        Initialize the Temporal TTA adapter.
        
        Args:
            detector: Base VLM detector (GroundingDINO)
            config: Configuration dictionary
        """
        super().__init__(detector, config)
        
        # Get adaptation parameters
        params = config.get('adaptation', {}).get('params', {})
        
        # STAD parameters
        self.kappa_trans_init = params.get('kappa_trans', 50.0)
        self.kappa_ems_init = params.get('kappa_ems', 10.0)
        self.window_size = params.get('window_size', 5)
        self.em_iterations = params.get('em_iterations', 3)
        
        # Confidence threshold for SSM update
        self.tau_update = params.get('tau_update', 0.5)
        
        # Fusion mode: 'entropy', 'weighted', or 'prototype_only'
        self.fusion_mode = params.get('fusion_mode', 'entropy')
        self.fusion_weight = params.get('fusion_weight', 0.5)  # For 'weighted' mode
        
        # Class information
        self.num_classes = len(config['detector']['target_classes'])
        self.class_names = config['detector']['target_classes']
        
        # SSM - created after we know feature dimension
        self.ssm: Optional[TemporalSSM] = None
        self.feature_dim: Optional[int] = None
        
        # Text embeddings cache
        self._text_embeddings: Optional[np.ndarray] = None
        self._text_embeddings_classes: Optional[List[str]] = None
        
        # NMS threshold
        self.iou_threshold = config['detector'].get('iou_threshold', 0.3)
        
        # Statistics
        self.frame_count = 0
        self.total_detections = 0
    
    def adapt_and_detect(self, image: Image.Image, target_classes: List[str],
                         threshold: float = 0.10):
        """
        Run detection with temporal adaptation.
        
        Algorithm (following STAD Algorithm 1):
        1. Extract query features H_t and text embeddings (if first frame)
        2. If SSM not initialized: initialize from text embeddings
        3. Update SSM with observations
        4. Fuse VLM predictions with prototype-based predictions
        5. Filter and NMS
        
        Args:
            image: Input PIL Image
            target_classes: Class names to detect
            threshold: Detection confidence threshold
            
        Returns:
            DetectionResult
        """
        # Update class info if changed
        if self.class_names != target_classes:
            self.class_names = target_classes
            self.num_classes = len(target_classes)
            self._text_embeddings = None  # Reset cache
            self.ssm = None  # Reset SSM
        
        # Step 1: Get all queries with features AND text embeddings
        detection_data = self._detect_with_features_and_text(image, target_classes, threshold)
        
        if len(detection_data['boxes']) == 0:
            return self._to_detection_result(detection_data)
        
        # Step 2: Initialize SSM if needed
        if self.ssm is None:
            self._initialize_ssm(detection_data)
        
        # Step 3: Update SSM with high-confidence observations
        if self.ssm.state.initialized:
            features = detection_data['features']
            vlm_probs = detection_data['class_probs']
            vlm_scores = detection_data['scores']
            
            # Confidence mask for SSM update
            confidence_mask = vlm_scores >= self.tau_update
            
            # Update SSM (returns soft assignments)
            if confidence_mask.any():
                _ = self.ssm.update(features, confidence_mask)
            
            # Step 4: Fuse predictions
            adapted_result = self._fuse_predictions(detection_data)
        else:
            adapted_result = detection_data
        
        # Step 5: Filter by threshold
        mask = np.array(adapted_result['scores']) >= threshold
        if not np.any(mask):
            return self._to_detection_result({
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': [],
                'features': np.array([]),
                'class_probs': None
            })
        
        filtered = {
            'boxes': adapted_result['boxes'][mask],
            'scores': adapted_result['scores'][mask],
            'labels': [adapted_result['labels'][i] for i in np.where(mask)[0]],
            'features': adapted_result['features'][mask],
            'class_probs': adapted_result['class_probs'][mask] if adapted_result.get('class_probs') is not None else None
        }
        
        # Step 6: NMS
        final_result = self._apply_nms(filtered)
        
        self.frame_count += 1
        self.total_detections += len(final_result['boxes'])
        
        return self._to_detection_result(final_result)
    
    def _detect_with_features_and_text(self, image: Image.Image, 
                                        target_classes: List[str],
                                        threshold: float) -> Dict:
        """
        Get detection results with both query features and text embeddings.
        
        Returns dict with:
        - boxes, scores, labels: Standard detection outputs
        - features: Query features from decoder (N, D)
        - class_probs: VLM class probabilities (N, K)
        - text_embeddings: Text embeddings for each class (K, D)
        """
        # Check if we need to extract text embeddings
        need_text_embeddings = (
            self._text_embeddings is None or 
            self._text_embeddings_classes != target_classes
        )
        
        # Get detection result with features
        result = self.detector.detect_with_features(image, target_classes, threshold)
        
        # Convert to dict
        detection_data = {
            'boxes': np.array(result.boxes) if result.boxes else np.array([]),
            'scores': np.array(result.scores) if result.scores else np.array([]),
            'labels': result.labels if result.labels else [],
            'features': result.features if result.features is not None else np.array([]),
            'class_probs': result.class_probs if result.class_probs is not None else None
        }
        
        # Extract text embeddings if needed
        if need_text_embeddings and len(detection_data['features']) > 0:
            text_emb = self._extract_text_embeddings(target_classes)
            if text_emb is not None:
                self._text_embeddings = text_emb
                self._text_embeddings_classes = target_classes
        
        detection_data['text_embeddings'] = self._text_embeddings
        
        return detection_data
    
    def _extract_text_embeddings(self, target_classes: List[str]) -> Optional[np.ndarray]:
        """
        Extract text embeddings for each class from GroundingDINO.
        
        These serve as the initial prototypes W_0.
        
        Strategy: Use the text encoder to get embeddings at class token positions.
        """
        try:
            # Build text prompt same way as detection
            text_prompt = " . ".join(target_classes) + " ."
            
            # Access the model components
            if hasattr(self.detector, 'model') and hasattr(self.detector.model, 'model'):
                # HuggingFace pipeline wrapper
                model = self.detector.model.model
                tokenizer = self.detector.model.tokenizer
            elif hasattr(self.detector, 'model'):
                model = self.detector.model
                tokenizer = self.detector.tokenizer if hasattr(self.detector, 'tokenizer') else None
            else:
                print("Warning: Cannot access model internals for text embedding extraction")
                return None
            
            if tokenizer is None:
                print("Warning: Tokenizer not available")
                return None
            
            # Tokenize
            text_inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)
            text_inputs = {k: v.to(self.detector.device) for k, v in text_inputs.items()}
            
            # Get tokens for position mapping
            tokens = tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])
            
            # Find class token positions
            class_positions = []
            for class_name in target_classes:
                try:
                    pos = tokens.index(class_name.lower())
                    class_positions.append(pos)
                except ValueError:
                    # Try without lowercasing
                    try:
                        pos = tokens.index(class_name)
                        class_positions.append(pos)
                    except ValueError:
                        print(f"Warning: Token '{class_name}' not found, using position 0")
                        class_positions.append(0)
            
            # Get text encoder output
            with torch.no_grad():
                # For HuggingFace GroundingDINO
                if hasattr(model, 'get_text_features'):
                    text_features = model.get_text_features(**text_inputs)
                elif hasattr(model, 'text_encoder'):
                    text_outputs = model.text_encoder(**text_inputs)
                    text_features = text_outputs.last_hidden_state
                elif hasattr(model, 'bert'):
                    # Original GroundingDINO
                    text_outputs = model.bert(**text_inputs)
                    text_features = text_outputs.last_hidden_state
                else:
                    # Fallback: try forward pass and extract
                    print("Warning: Using fallback text embedding extraction")
                    return self._fallback_text_embeddings(target_classes)
            
            # Extract embeddings at class positions
            # text_features shape: (1, num_tokens, hidden_dim)
            text_features = text_features[0].cpu().numpy()  # (num_tokens, hidden_dim)
            
            # Get embedding for each class
            K = len(target_classes)
            D = text_features.shape[1]
            
            text_embeddings = np.zeros((K, D))
            for k, pos in enumerate(class_positions):
                if pos < text_features.shape[0]:
                    text_embeddings[k] = text_features[pos]
                else:
                    text_embeddings[k] = text_features[0]  # Use CLS token as fallback
            
            print(f"Extracted text embeddings: {K} classes, {D}D")
            return text_embeddings
            
        except Exception as e:
            print(f"Warning: Text embedding extraction failed: {e}")
            return self._fallback_text_embeddings(target_classes)
    
    def _fallback_text_embeddings(self, target_classes: List[str]) -> Optional[np.ndarray]:
        """
        Fallback: Initialize from first batch of detections using class probs.
        
        Returns None to signal that we should use feature-based initialization.
        """
        print("Using feature-based initialization (fallback)")
        return None
    
    def _initialize_ssm(self, detection_data: Dict):
        """Initialize the SSM with text embeddings or features."""
        features = detection_data['features']
        if features is None or len(features) == 0:
            return
        
        self.feature_dim = features.shape[1]
        
        # Create SSM
        self.ssm = TemporalSSM(
            num_classes=self.num_classes,
            feature_dim=self.feature_dim,
            kappa_trans_init=self.kappa_trans_init,
            kappa_ems_init=self.kappa_ems_init,
            window_size=self.window_size,
            em_iterations=self.em_iterations
        )
        
        # Prefer text embedding initialization
        text_emb = detection_data.get('text_embeddings')
        
        if text_emb is not None and text_emb.shape[1] == self.feature_dim:
            # Text embeddings match feature dimension - use directly
            self.ssm.initialize_from_text_embeddings(text_emb)
        elif text_emb is not None:
            # Dimension mismatch - need projection or use features
            print(f"Warning: Text embedding dim {text_emb.shape[1]} != feature dim {self.feature_dim}")
            print("Falling back to feature-based initialization")
            if detection_data.get('class_probs') is not None:
                self.ssm.initialize_from_features(features, detection_data['class_probs'])
            else:
                # Last resort: use text embeddings anyway with padding/truncation
                self._initialize_with_projected_text(text_emb)
        else:
            # No text embeddings - use features
            if detection_data.get('class_probs') is not None:
                self.ssm.initialize_from_features(features, detection_data['class_probs'])
            else:
                print("Warning: Cannot initialize SSM - no text embeddings or class probs")
    
    def _initialize_with_projected_text(self, text_emb: np.ndarray):
        """Project text embeddings to match feature dimension."""
        K, D_text = text_emb.shape
        D_feat = self.feature_dim
        
        if D_text > D_feat:
            # Truncate
            projected = text_emb[:, :D_feat]
        else:
            # Pad with zeros
            projected = np.zeros((K, D_feat))
            projected[:, :D_text] = text_emb
        
        self.ssm.initialize_from_text_embeddings(projected)
    
    def _fuse_predictions(self, detection_data: Dict) -> Dict:
        """
        Fuse VLM predictions with prototype-based predictions.
        
        Uses entropy-weighted fusion (Eq. 14 from BCA+ paper):
        p_fused = (w_vlm * p_vlm + w_proto * p_proto) / (w_vlm + w_proto)
        where w = exp(-entropy)
        """
        features = detection_data['features']
        vlm_probs = detection_data['class_probs']
        boxes = detection_data['boxes']
        
        if features is None or len(features) == 0:
            return detection_data
        
        # Get prototype-based predictions
        proto_probs = self.ssm.predict(features)  # (N, K)
        
        if vlm_probs is None:
            # No VLM probs - use prototype only
            final_probs = proto_probs
        elif self.fusion_mode == 'prototype_only':
            final_probs = proto_probs
        elif self.fusion_mode == 'vlm_only':
            final_probs = vlm_probs
        elif self.fusion_mode == 'weighted':
            # Simple weighted average
            w = self.fusion_weight
            final_probs = w * proto_probs + (1 - w) * vlm_probs
        else:  # 'entropy' mode (default)
            final_probs = self._entropy_fusion(vlm_probs, proto_probs)
        
        # Compute final scores and labels
        final_scores = np.max(final_probs, axis=1)
        final_label_indices = np.argmax(final_probs, axis=1)
        final_labels = [self.class_names[idx] for idx in final_label_indices]
        
        return {
            'boxes': boxes,
            'scores': final_scores,
            'labels': final_labels,
            'features': features,
            'class_probs': final_probs
        }
    
    def _entropy_fusion(self, vlm_probs: np.ndarray, proto_probs: np.ndarray) -> np.ndarray:
        """Entropy-weighted fusion: lower entropy = higher weight."""
        # Compute entropy
        eps = 1e-10
        H_vlm = -np.sum(vlm_probs * np.log(vlm_probs + eps), axis=1)
        H_proto = -np.sum(proto_probs * np.log(proto_probs + eps), axis=1)
        
        # Convert to weights
        w_vlm = np.exp(-H_vlm)[:, np.newaxis]
        w_proto = np.exp(-H_proto)[:, np.newaxis]
        
        # Weighted average
        fused = (w_vlm * vlm_probs + w_proto * proto_probs) / (w_vlm + w_proto + eps)
        
        return fused
    
    def _apply_nms(self, result: Dict, iou_threshold: float = None) -> Dict:
        """Apply Non-Maximum Suppression."""
        if iou_threshold is None:
            iou_threshold = self.iou_threshold
        
        if len(result['boxes']) == 0:
            return result
        
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        features = result['features']
        class_probs = result.get('class_probs')
        
        def compute_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - inter
            
            return inter / union if union > 0 else 0
        
        sorted_idx = np.argsort(scores)[::-1]
        keep = []
        
        while len(sorted_idx) > 0:
            current = sorted_idx[0]
            keep.append(current)
            
            if len(sorted_idx) == 1:
                break
            
            remaining = []
            for idx in sorted_idx[1:]:
                if compute_iou(boxes[current], boxes[idx]) < iou_threshold:
                    remaining.append(idx)
            
            sorted_idx = np.array(remaining)
        
        return {
            'boxes': boxes[keep],
            'scores': scores[keep],
            'labels': [labels[i] for i in keep],
            'features': features[keep],
            'class_probs': class_probs[keep] if class_probs is not None else None
        }
    
    def _to_detection_result(self, result: Dict):
        """Convert to DetectionResult."""
        from vlm_detector_system_new import DetectionResult
        return DetectionResult(
            boxes=result['boxes'].tolist() if len(result['boxes']) > 0 else [],
            scores=result['scores'].tolist() if len(result['scores']) > 0 else [],
            labels=result['labels'],
            image_path="",
            model_path=self.detector.model_path
        )
    
    # =========================================================================
    # Utility methods
    # =========================================================================
    
    def get_ssm_state(self) -> Optional[Dict]:
        """Get current SSM state summary."""
        if self.ssm is None:
            return None
        return self.ssm.get_state_summary()
    
    def get_prototypes(self) -> Optional[np.ndarray]:
        """Get current class prototypes."""
        if self.ssm is None:
            return None
        return self.ssm.get_prototypes()
    
    def get_text_embeddings(self) -> Optional[np.ndarray]:
        """Get cached text embeddings (initial prototypes)."""
        return self._text_embeddings
    
    def reset(self):
        """Reset adapter for new video sequence."""
        self.ssm = None
        self._text_embeddings = None
        self._text_embeddings_classes = None
        self.frame_count = 0
        self.total_detections = 0
        print("Temporal TTA adapter reset")