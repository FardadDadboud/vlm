import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Union, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import different model libraries
try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    OWLV2_AVAILABLE = True
except ImportError:
    OWLV2_AVAILABLE = False

try:
    from ultralytics import YOLO

    YOLO_WORLD_AVAILABLE = True
except ImportError:
    YOLO_WORLD_AVAILABLE = False

try:
    # import groundingdino.datasets.transforms as T
    # from groundingdino.models import build_model
    # from groundingdino.util import box_ops
    # from groundingdino.util.inference import load_model, load_image, predict, annotate
    from transformers import pipeline

    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False

try:
    from transformers import DetrImageProcessor, DetrForObjectDetection

    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False

try:
    from transformers import pipeline

    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False

GLIP_AVAILABLE = DEPTH_AVAILABLE
F_VLM_AVAILABLE = DEPTH_AVAILABLE
DETIC_AVAILABLE = DEPTH_AVAILABLE


@dataclass
class DetectionResult:
    boxes: List[List[float]]
    scores: List[float]
    labels: List[str]
    image_path: str = ""
    model_path: str = ""
    features: Optional[np.ndarray] = None           # Hybrid 262D
    class_probs: Optional[np.ndarray] = None
    text_embeddings: Optional[np.ndarray] = None    # Hybrid 262D
    raw_features: Optional[np.ndarray] = None       # Raw 256D (NEW)
    raw_text_embeddings: Optional[np.ndarray] = None  # Raw 256D (NEW)


class BaseDetector(ABC):
    """Abstract base class for all detectors"""

    @abstractmethod
    def detect(self, image: Image.Image, texts: List[str], threshold: float = 0.05) -> DetectionResult:
        pass

    @abstractmethod
    def load_model(self):
        pass

    def detect_with_sahi(self, image: Image.Image, texts: List[str], 
                     tile_size: int = 512, overlap_ratio: float = 0.25, 
                     threshold: float = 0.1) -> DetectionResult:
        """SAHI-based detection for small objects"""
        import math
        
        width, height = image.size
        step_w = int(tile_size * (1 - overlap_ratio))
        step_h = int(tile_size * (1 - overlap_ratio))
        
        all_boxes, all_scores, all_labels = [], [], []
        
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                x2 = min(x + tile_size, width)
                y2 = min(y + tile_size, height)
                
                tile = image.crop((x, y, x2, y2))
                result = self.detect(tile, texts, threshold)
                
                # Rescale boxes to full image coordinates
                for box, score, label in zip(result.boxes, result.scores, result.labels):
                    x1, y1, x2_tile, y2_tile = box
                    scaled_box = [x1 + x, y1 + y, x2_tile + x, y2_tile + y]
                    all_boxes.append(scaled_box)
                    all_scores.append(score)
                    all_labels.append(label)
        
        # Apply NMS to remove duplicates
        if all_boxes:
            keep_indices = self._nms_boxes(all_boxes, all_scores, 0.3)
            final_boxes = [all_boxes[i] for i in keep_indices]
            final_scores = [all_scores[i] for i in keep_indices]
            final_labels = [all_labels[i] for i in keep_indices]
        else:
            final_boxes, final_scores, final_labels = [], [], []
        
        return DetectionResult(final_boxes, final_scores, final_labels, "", "", self.model_path)
    
    def _nms_boxes(self, boxes: List[List[float]], scores: List[float], iou_threshold: float = 0.4) -> List[int]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes
        
        Args:
            boxes: List of boxes in [x1, y1, x2, y2] format
            scores: List of confidence scores
            iou_threshold: IoU threshold for suppression
            
        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []
        
        # Convert to numpy
        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        
        # Get coordinates
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        
        # Compute areas
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by score
        order = scores_np.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            # Pick the box with highest score
            i = order[0]
            keep.append(int(i))
            
            if len(order) == 1:
                break
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)
            
            # Keep boxes with IoU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep


class OWLv2Detector(BaseDetector):
    """OWLv2 detector implementation"""

    def __init__(self, model_path: str = "google/owlv2-base-patch16-ensemble", device: str = None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.load_model()

    def load_model(self):
        if not OWLV2_AVAILABLE:
            raise ImportError("OWLv2 not available. Install with: pip install transformers")

        print(f"Loading OWLv2 model: {self.model_path}")
        print(f"Using device: {self.device}")

        self.processor = Owlv2Processor.from_pretrained(self.model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        # Move model to device if not using device_map
        if self.device != "cuda" or not hasattr(self.model, 'hf_device_map'):
            self.model = self.model.to(self.device)

        print(f"✓ Model loaded on {self.device}")
        if self.device == "cuda":
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def detect(self, image: Image.Image, texts: List[str], threshold: float = 0.05) -> DetectionResult:
    
        # Store original size
        original_size = image.size
        
        # Preprocess image, resize to 1024x1024
        processed_image = image.resize((1024, 1024))
        
        # Process input
        inputs = self.processor(text=[texts], images=processed_image, return_tensors="pt")

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference with optimizations
        with torch.no_grad():
            if self.device == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        # Post-process results with PROCESSED image size
        target_sizes = torch.Tensor([[1024, 1024]])  # Use processed size
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )

        # Extract and scale results back to original size
        boxes = results[0]["boxes"].cpu().numpy().tolist()
        scores = results[0]["scores"].cpu().numpy().tolist()
        labels = results[0]["labels"] if "labels" in results[0] else [texts[0]] * len(boxes)

        # Scale boxes back to original image size
        scale_x = original_size[0] / 1024  # width scaling
        scale_y = original_size[1] / 1024  # height scaling
        
        scaled_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            scaled_box = [
                x1 * scale_x,
                y1 * scale_y, 
                x2 * scale_x,
                y2 * scale_y
            ]
            scaled_boxes.append(scaled_box)

        return DetectionResult(
            boxes=scaled_boxes,
            scores=scores,
            labels=labels,
            image_path="",
            model_path=self.model_path,
        )


class GroundingDINODetector(BaseDetector):
    """GroundingDINO detector - excellent for open vocabulary detection"""

    def __init__(self, model_path: str = "IDEA-Research/grounding-dino-tiny", device: str = None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

    def load_model(self):
        if not GROUNDING_DINO_AVAILABLE:
            raise ImportError("GroundingDINO not available. Install with: pip install groundingdino-py")

        print(f"Loading GroundingDINO: {self.model_path}")
        print(f"Using device: {self.device}")

        # Check if model_path is a local directory
        is_local_path = os.path.isdir(self.model_path) or os.path.isfile(self.model_path)
        
        # Check for offline mode via environment variable
        offline_mode = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1" or is_local_path
        
        if is_local_path:
            print(f"Using local model path: {self.model_path}")
        elif offline_mode:
            print("Offline mode enabled - will use cached models only")
        
        # Use transformers pipeline only
        device_id = 0 if self.device == "cuda" else -1
        
        # Build pipeline arguments
        pipeline_kwargs = {
            "task": "zero-shot-object-detection",
            "model": self.model_path,
            "device": device_id
        }
        
        # Force offline mode if using local path or environment variable is set
        if offline_mode:
            pipeline_kwargs["local_files_only"] = True
        
        self.model = pipeline(**pipeline_kwargs)
        print("✓ GroundingDINO (transformers) loaded successfully")

    def detect(self, image: Image.Image, texts: List[str], threshold: float = 0.05, iou_threshold: float = 0.3) -> DetectionResult:

        
        
        if not texts or not isinstance(texts, list): return DetectionResult([], [], [], "", self.model_path)
        
        # Convert texts to GroundingDINO format
        text_prompt = " . ".join(texts) + " ."

        try:
            if hasattr(self.model, 'cuda'):  # Custom GroundingDINO
                image_source, image_tensor = load_image(image)
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image_tensor,
                    caption=text_prompt,
                    box_threshold=threshold,
                    text_threshold=0.25,
                    device=self.device
                )

                # Convert to standard format
                boxes_list = boxes.cpu().numpy().tolist()
                scores_list = logits.cpu().numpy().tolist()
                labels_list = phrases

            else:  # Transformers - use manual model forward pass with offset-span aggregation
                model = self.model.model
                image_processor = self.model.image_processor
                tokenizer = self.model.tokenizer
                
                # Process image and text SEPARATELY (with offset mapping!)
                image_inputs = image_processor(images=image, return_tensors="pt")
                text_inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
                
                # Save offset_mapping and remove from model inputs
                offset_mapping = text_inputs.pop('offset_mapping')[0]  # (num_tokens, 2)
                
                # Combine inputs
                inputs = {
                    **{k: v.to(self.device) for k, v in image_inputs.items()},
                    **{k: v.to(self.device) for k, v in text_inputs.items()}
                }
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(**inputs)

                # Extract outputs
                logits = outputs.logits[0].cpu().numpy()  # (num_queries, num_text_tokens)
                pred_boxes = outputs.pred_boxes[0].cpu().numpy()  # (num_queries, 4)
                
                # ROBUST CLASS TOKEN EXTRACTION using offset-span aggregation
                class_token_indices_list = []
                
                for class_name in texts:
                    # Find character span of this class in the text prompt
                    start_char = text_prompt.find(class_name)
                    if start_char == -1:
                        print(f"Warning: Class '{class_name}' not found in prompt '{text_prompt}'")
                        class_token_indices_list.append([0])
                        continue
                    
                    end_char = start_char + len(class_name)
                    
                    # Find tokens whose offsets overlap with this class span
                    token_indices = []
                    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                        # Check if token overlaps with class span
                        if token_start < end_char and token_end > start_char and token_start != token_end:
                            token_indices.append(token_idx)
                    
                    if len(token_indices) == 0:
                        print(f"Warning: No tokens found for class '{class_name}', using fallback")
                        token_indices = [0]
                    
                    class_token_indices_list.append(token_indices)
                
                # Extract class-specific logits (aggregate multi-token classes with MAX)
                num_queries = logits.shape[0]
                num_classes = len(texts)
                class_logits = np.zeros((num_queries, num_classes))
                
                for class_idx, token_indices in enumerate(class_token_indices_list):
                    # Use MAX aggregation for multi-token classes
                    class_logits[:, class_idx] = logits[:, token_indices].max(axis=1)
                
                # Apply softmax to get probabilities
                class_probs = np.exp(class_logits - np.max(class_logits, axis=1, keepdims=True))
                class_probs = class_probs / (np.sum(class_probs, axis=1, keepdims=True) + 1e-8)
                
                # Get max score and label for each query
                all_scores = class_probs.max(axis=1)
                all_label_indices = class_probs.argmax(axis=1)
                all_labels = [texts[idx] for idx in all_label_indices]

                
                # Convert boxes to pixel coordinates
                image_w, image_h = image.size
                all_boxes_xyxy = np.zeros_like(pred_boxes)
                all_boxes_xyxy[:, 0] = (pred_boxes[:, 0] - pred_boxes[:, 2] / 2) * image_w
                all_boxes_xyxy[:, 1] = (pred_boxes[:, 1] - pred_boxes[:, 3] / 2) * image_h
                all_boxes_xyxy[:, 2] = (pred_boxes[:, 0] + pred_boxes[:, 2] / 2) * image_w
                all_boxes_xyxy[:, 3] = (pred_boxes[:, 1] + pred_boxes[:, 3] / 2) * image_h
                
                # Filter by threshold
                mask = all_scores >= threshold
                
                boxes_list = all_boxes_xyxy[mask].tolist()
                scores_list = all_scores[mask].tolist()
                labels_list = [all_labels[i] for i, m in enumerate(mask) if m]
                
                # Apply NMS to remove overlapping boxes
                if boxes_list:
                    keep_indices = self._nms_boxes(boxes_list, scores_list, iou_threshold=iou_threshold)
                    boxes_list = [boxes_list[i] for i in keep_indices]
                    scores_list = [scores_list[i] for i in keep_indices]
                    labels_list = [labels_list[i] for i in keep_indices]

        except Exception as e:
            print(f"GroundingDINO detection error: {e}")
            return DetectionResult([], [], [], "", "", self.model_path)

        return DetectionResult(
            boxes=boxes_list,
            scores=scores_list,
            labels=labels_list,
            image_path="",
            model_path=self.model_path
        )

    def detect_with_features(self, image: Image.Image, texts: List[str], threshold: float = 0.05, alpha: float = 0.7) -> DetectionResult:
        """
        Extended detection that returns internal features from the model.
        Returns DetectionResult with features and class_probs populated.
        """
        if not texts or not isinstance(texts, list):
            return DetectionResult([], [], [], "", self.model_path, None, 0.0, None, None)
        
        text_prompt = " . ".join(texts) + " ."
        
        try:
            if hasattr(self.model, 'cuda'):
                # Custom GroundingDINO - extract features from model internals
                return self._detect_custom_with_features(image, text_prompt, texts, threshold)
            else:
                # Transformers pipeline - extract features from underlying model
                return self._detect_transformers_with_features(image, text_prompt, texts, threshold, alpha)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Feature extraction error: {e} {traceback.format_exc()}")
            return DetectionResult([], [], [], "", self.model_path, None, 0.0, None, None)

    # def _detect_custom_with_features(self, image: Image.Image, text_prompt: str, texts: List[str], threshold: float) -> DetectionResult:
    #     """
    #     Extract features from custom GroundingDINO model.
    #     Access decoder query embeddings directly.
    #     """
    #     from groundingdino.util.inference import load_image
    #     from groundingdino.util import box_ops
    #     import groundingdino.datasets.transforms as T
        
    #     # Prepare image
    #     image_source, image_tensor = load_image(image)
    #     image_tensor = image_tensor.to(self.device)
        
    #     # Prepare text
    #     captions = [text_prompt]
        
    #     # Forward pass through model
    #     with torch.no_grad():
    #         outputs = self.model(image_tensor, captions=captions)
        
    #     # Extract components
    #     prediction_logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, num_classes)
    #     prediction_boxes = outputs["pred_boxes"][0]  # (num_queries, 4)
        
    #     # CRITICAL: Extract query features from decoder
    #     # These are the f_v_ij embeddings from the paper
    #     if "hs" in outputs:
    #         # hs contains hidden states from all decoder layers
    #         # Shape: (num_decoder_layers, batch_size, num_queries, hidden_dim)
    #         query_features = outputs["hs"][-1][0]  # Last layer, first batch: (num_queries, hidden_dim)
    #     elif "decoder_output" in outputs:
    #         query_features = outputs["decoder_output"][0]
    #     else:
    #         # Fallback: try to access decoder hidden states
    #         query_features = outputs.get("query_embed", None)
    #         if query_features is None:
    #             raise ValueError("Cannot extract query features from model outputs")
        
    #     query_features = query_features.cpu().numpy()  # (num_queries, hidden_dim)
        
    #     # Tokenize text to get phrase indices
    #     tokenized = self.model.tokenizer(text_prompt)
        
    #     # Filter predictions by threshold
    #     max_logits = prediction_logits.max(dim=1)[0]
    #     mask = max_logits > threshold
        
    #     if mask.sum() == 0:
    #         return DetectionResult([], [], [], "", self.model_path, None, 0.0, None, None)
        
    #     # Get filtered predictions
    #     logits = prediction_logits[mask]  # (N, num_classes)
    #     boxes = prediction_boxes[mask]  # (N, 4)
    #     features = query_features[mask.cpu().numpy()]  # (N, hidden_dim)
        
    #     # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
    #     image_h, image_w = image_source.shape[:2]
    #     boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.tensor([image_w, image_h, image_w, image_h])
    #     boxes = boxes.cpu().numpy()
        
    #     # Get class probabilities and labels
    #     class_probs = logits.cpu().numpy()  # (N, num_classes)
    #     scores = class_probs.max(axis=1)
    #     label_indices = class_probs.argmax(axis=1)
        
    #     # Map to text labels
    #     labels = []
    #     for idx in label_indices:
    #         if idx < len(texts):
    #             labels.append(texts[idx])
    #         else:
    #             labels.append("unknown")
        
    #     # Apply NMS to remove duplicates
    #     keep_indices = self._nms_boxes(boxes.tolist(), scores.tolist(), iou_threshold=0.5)
        
    #     # Convert to lists to match DetectionResult structure
    #     boxes_list = boxes[keep_indices].tolist()
    #     scores_list = scores[keep_indices].tolist()
    #     labels_list = [labels[i] for i in keep_indices]
    #     features_array = features[keep_indices]  # Keep as numpy array
    #     class_probs_array = class_probs[keep_indices]  # Keep as numpy array
        
    #     return DetectionResult(
    #         boxes=boxes_list,
    #         scores=scores_list,
    #         labels=labels_list,
    #         image_path="",
    #         model_path=self.model_path,
    #         depth_map=None,
    #         processing_time=0.0,
    #         features=features_array,  # These are the real f_v_ij embeddings
    #         class_probs=class_probs_array
    #     )

    def _detect_transformers_with_features(self, image: Image.Image, text_prompt: str, texts: List[str], threshold: float, alpha: float = 0.7) -> DetectionResult:
        """
        Extract ALL query features, scores, and text embeddings.
        Returns hybrid features for both queries and text prototypes.
        """
        model = self.model.model
        image_processor = self.model.image_processor
        tokenizer = self.model.tokenizer
        
        # Process image and text SEPARATELY (with offset mapping!)
        image_inputs = image_processor(images=image, return_tensors="pt")
        text_inputs = tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
        
        # Save offset_mapping and remove from model inputs
        offset_mapping = text_inputs.pop('offset_mapping')[0]  # (num_tokens, 2)
        
        # Combine inputs
        inputs = {
            **{k: v.to(self.device) for k, v in image_inputs.items()},
            **{k: v.to(self.device) for k, v in text_inputs.items()}
        }
        
        # Setup hook with kwargs enabled
        captured = {}
        
        def capture_pre_with_kwargs(module, args, kwargs):
            """Capture vision_hidden_state from class_embed kwargs"""
            if "vision_hidden_state" in kwargs:
                captured["vision_hs"] = kwargs["vision_hidden_state"].detach()
        
        # Register PRE-hook with kwargs enabled
        try:
            hook_target = model.class_embed[-1] if isinstance(model.class_embed, torch.nn.ModuleList) else model.class_embed
            handle = hook_target.register_forward_pre_hook(capture_pre_with_kwargs, with_kwargs=True)
            hook_registered = True
        except Exception as e:
            print(f"Warning: Could not register hook: {e}")
            handle = None
            hook_registered = False
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Remove hook
        if handle is not None:
            handle.remove()
        
        # Extract outputs
        logits = outputs.logits[0]  # (num_queries, max_text_tokens)
        pred_boxes = outputs.pred_boxes[0]  # (num_queries, 4)
        
        # Get query features from hook
        if hook_registered and "vision_hs" in captured:
            query_features = captured["vision_hs"]
            if query_features.dim() == 3:
                query_features = query_features[0]
            query_features_np = query_features.cpu().numpy()
        else:
            query_features = outputs.decoder_hidden_states[-1][0]
            query_features_np = query_features.cpu().numpy()
        
        # Handle inf/nan values in features
        query_features_np[np.isnan(query_features_np)] = 0
        query_features_np[np.isinf(query_features_np)] = 0
        
        # Extract text features for prototypes
        text_features = outputs.encoder_last_hidden_state_text[0].cpu().numpy()  # (num_tokens, 256)
        
        # Convert to numpy
        logits_np = logits.cpu().numpy()
        pred_boxes_np = pred_boxes.cpu().numpy()
        
        # ROBUST CLASS TOKEN EXTRACTION using offset-span aggregation
        class_token_indices_list = []
        num_classes = len(texts)
        
        for class_name in texts:
            start_char = text_prompt.find(class_name)
            if start_char == -1:
                print(f"Warning: Class '{class_name}' not found in prompt '{text_prompt}'")
                class_token_indices_list.append([0])
                continue
            
            end_char = start_char + len(class_name)
            
            token_indices = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < end_char and token_end > start_char and token_start != token_end:
                    token_indices.append(token_idx)
            
            if len(token_indices) == 0:
                print(f"Warning: No tokens found for class '{class_name}', using fallback")
                token_indices = [0]
            
            class_token_indices_list.append(token_indices)
        
        # Extract class-specific logits (aggregate multi-token classes with MAX)
        num_queries = logits_np.shape[0]
        class_logits = np.zeros((num_queries, num_classes))
        
        for class_idx, token_indices in enumerate(class_token_indices_list):
            class_logits[:, class_idx] = logits_np[:, token_indices].max(axis=1)
        
        # Apply softmax to get probabilities
        class_probs = np.exp(class_logits - np.max(class_logits, axis=1, keepdims=True))
        class_probs = class_probs / (np.sum(class_probs, axis=1, keepdims=True) + 1e-8)
        
        # Get max score and label for each query
        all_scores = class_probs.max(axis=1)
        all_label_indices = class_probs.argmax(axis=1)
        all_labels = [texts[idx] for idx in all_label_indices]
        
        # Convert boxes to pixel coordinates
        image_w, image_h = image.size
        all_boxes_xyxy = np.zeros_like(pred_boxes_np)
        all_boxes_xyxy[:, 0] = (pred_boxes_np[:, 0] - pred_boxes_np[:, 2] / 2) * image_w
        all_boxes_xyxy[:, 1] = (pred_boxes_np[:, 1] - pred_boxes_np[:, 3] / 2) * image_h
        all_boxes_xyxy[:, 2] = (pred_boxes_np[:, 0] + pred_boxes_np[:, 2] / 2) * image_w
        all_boxes_xyxy[:, 3] = (pred_boxes_np[:, 1] + pred_boxes_np[:, 3] / 2) * image_h
        
        # Store raw decoder features BEFORE hybrid mixing
        raw_decoder_norm = query_features_np / (np.linalg.norm(query_features_np, axis=1, keepdims=True) + 1e-8)

        # Create hybrid features for prediction
        if alpha > 0:
            semantic_norm = class_probs / (np.linalg.norm(class_probs, axis=1, keepdims=True) + 1e-8)
            hybrid_features = np.concatenate([
                (1 - alpha) * raw_decoder_norm,
                alpha * semantic_norm
            ], axis=1)
        else:
            hybrid_features = raw_decoder_norm

        # query_features_np is now hybrid (used for existing code paths)
        query_features_np = hybrid_features
        
        
        # CREATE HYBRID TEXT EMBEDDINGS: text (spatial) + one_hot (semantic)
        raw_text_embeddings = np.zeros((num_classes, text_features.shape[1]))
        for k, token_indices in enumerate(class_token_indices_list):
            raw_text_embeddings[k] = text_features[token_indices].mean(axis=0)
        
        # Store raw text embeddings
        raw_text_norm = raw_text_embeddings / (np.linalg.norm(raw_text_embeddings, axis=1, keepdims=True) + 1e-8)

        # Create hybrid text embeddings for prediction
        if alpha > 0:
            one_hot = np.eye(num_classes)
            one_hot_norm = one_hot / (np.linalg.norm(one_hot, axis=1, keepdims=True) + 1e-8)
            hybrid_text_embeddings = np.concatenate([
                (1 - alpha) * raw_text_norm,
                alpha * one_hot_norm
            ], axis=1)
        else:
            hybrid_text_embeddings = raw_text_norm

        # This is used for existing code paths
        text_embeddings = hybrid_text_embeddings
        
        return DetectionResult(
            boxes=all_boxes_xyxy.tolist(),
            scores=all_scores.tolist(),
            labels=all_labels,
            image_path="",
            model_path=self.model_path,
            features=query_features_np,
            class_probs=class_probs,
            text_embeddings=text_embeddings,  # NEW: hybrid prototypes
            raw_features=raw_decoder_norm,
            raw_text_embeddings=raw_text_norm
        )


class DETRDetector(BaseDetector):
    """DETR-based object detector"""

    def __init__(self, model_path: str = "facebook/detr-resnet-50", device: str = None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self.load_model()

    def load_model(self):
        if not DETR_AVAILABLE:
            raise ImportError("DETR not available. Install with: pip install transformers")

        print(f"Loading DETR: {self.model_path}")
        print(f"Using device: {self.device}")

        self.processor = DetrImageProcessor.from_pretrained(self.model_path)
        self.model = DetrForObjectDetection.from_pretrained(self.model_path)

        if self.device == "cuda":
            self.model = self.model.to(self.device)
            if hasattr(self.model, 'half'):
                self.model = self.model.half()

        print("✓ DETR loaded successfully")

    def detect(self, image: Image.Image, texts: List[str], threshold: float = 0.05) -> DetectionResult:
        # DETR has fixed COCO classes, so we need to map text queries to COCO classes
        coco_classes = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Process input
        inputs = self.processor(images=image, return_tensors="pt")
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]], device=outputs.logits.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        # Filter results based on text queries
        boxes = []
        scores = []
        labels = []

        for box, score, label_id in zip(results["boxes"], results["scores"], results["labels"]):
            label_name = coco_classes[label_id.item()]

            # Check if detected class matches any of our target texts
            for target_text in texts:
                if (target_text.lower() in label_name.lower() or
                        label_name.lower() in target_text.lower() or
                        (target_text.lower() in ['aircraft', 'airplane', 'plane'] and label_name == 'airplane') or
                        (target_text.lower() in ['bird'] and label_name == 'bird') or
                        (target_text.lower() in ['kite'] and label_name == 'kite')):
                    boxes.append(box.cpu().numpy().tolist())
                    scores.append(score.cpu().numpy().item())
                    labels.append(f"{label_name} ({target_text})")
                    break

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_path="",
            model_path=self.model_path
        )


class DepthEstimator:
    """VLM-based monocular depth estimation"""

    def __init__(self, model_path: str = "Intel/dpt-large", device: str = None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.load_model()

    def load_model(self):
        if not DEPTH_AVAILABLE:
            raise ImportError("Depth estimation not available. Install with: pip install transformers")

        print(f"Loading depth model: {self.model_path}")

        try:
            device_id = 0 if self.device == "cuda" else -1
            self.pipeline = pipeline(
                "depth-estimation",
                model=self.model_path,
                device=device_id
            )
            print("✓ Depth estimation model loaded")
        except Exception as e:
            print(f"⚠️  Depth model loading failed: {e}")
            # Fallback to simpler model
            try:
                self.pipeline = pipeline(
                    "depth-estimation",
                    model="Intel/dpt-hybrid-midas",
                    device=device_id
                )
                print("✓ Fallback depth model loaded")
            except:
                self.pipeline = None
                print("❌ No depth model available")

    def estimate_depth(self, image: Image.Image) -> np.ndarray:
        """Estimate depth map for an image"""
        if self.pipeline is None:
            return None

        try:
            result = self.pipeline(image)
            depth_map = np.array(result["depth"])
            return depth_map
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return None

    def estimate_object_distances(self, image: Image.Image, detection_result: DetectionResult) -> List[float]:
        """Estimate distances to detected objects"""
        depth_map = self.estimate_depth(image)
        if depth_map is None:
            return [0.0] * len(detection_result.boxes)

        distances = []
        for box in detection_result.boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Get depth in the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Ensure coordinates are within image bounds
            center_x = max(0, min(center_x, depth_map.shape[1] - 1))
            center_y = max(0, min(center_y, depth_map.shape[0] - 1))

            # Get relative depth (normalized)
            relative_depth = depth_map[center_y, center_x]
            distances.append(float(relative_depth))

        return distances


class YOLOWorldDetector(BaseDetector):
    """YOLO-World detector implementation"""

    def __init__(self, model_path: str = "yolov8s-world.pt", device: str = None):
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes_set = False
        self.last_texts = None
        self.load_model()

    def load_model(self):
        if not YOLO_WORLD_AVAILABLE:
            raise ImportError("YOLO-World not available. Install with: pip install ultralytics")

        print(f"Loading YOLO-World model: {self.model_path}")
        print(f"Using device: {self.device}")

        self.model = YOLO(self.model_path)

        # Move model to device but don't use FP16 initially
        if hasattr(self.model.model, 'to'):
            self.model.model = self.model.model.to(self.device)

        print(f"✓ YOLO-World loaded on {self.device}")

    def detect(self, image: Image.Image, texts: List[str], threshold: float = 0.05) -> DetectionResult:
        # Only set classes if they've changed (expensive operation)
        if not self.classes_set or self.last_texts != texts:
            try:
                # Temporarily set model to float32 for text encoding
                if self.device == "cuda" and hasattr(self.model.model, 'float'):
                    self.model.model.float()

                self.model.set_classes(texts)
                self.classes_set = True
                self.last_texts = texts.copy()

                print(f"✓ Set classes: {texts}")

            except Exception as e:
                print(f"⚠️  Error setting classes, retrying: {e}")
                # Fallback: reload model and try again
                self.load_model()
                self.model.set_classes(texts)
                self.classes_set = True
                self.last_texts = texts.copy()

        # Convert PIL to numpy array
        image_np = np.array(image)

        # Run detection - disable half precision to avoid mixed precision issues
        results = self.model(
            image_np,
            conf=threshold,
            verbose=False,
            device=self.device,
            half=False  # Disable FP16 to avoid CLIP model issues
        )

        # Extract results
        boxes = []
        scores = []
        labels = []

        if len(results) > 0 and results[0].boxes is not None:
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                boxes.append(box.cpu().numpy().tolist())
                scores.append(conf.cpu().numpy().item())
                labels.append(texts[int(cls.cpu().numpy().item())])

        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            image_path="",
            model_path=self.model_path
        )


class MultiModalDetector:
    """Main class that handles multiple detectors and input types"""

    def __init__(self, enable_depth=False):
        self.detectors = {}
        self.depth_estimator = None
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.supported_video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}

        if enable_depth:
            try:
                self.depth_estimator = DepthEstimator()
            except Exception as e:
                print(f"⚠️  Depth estimation disabled: {e}")

    def add_detector(self, name: str, detector: BaseDetector):
        """Add a detector to the system"""
        self.detectors[name] = detector
        print(f"Added detector: {name}")

    def get_available_models():
        """Get list of available models"""
        available = []

        if OWLV2_AVAILABLE:
            available.extend([
                "owlv2-base", "owlv2-large"
            ])

        if YOLO_WORLD_AVAILABLE:
            available.extend([
                "yolo-world-s", "yolo-world-m", "yolo-world-l"
            ])

        if GROUNDING_DINO_AVAILABLE:
            available.extend([
                "grounding-dino", "grounding-dino-tiny"
            ])

        if DETR_AVAILABLE:
            available.extend([
                "detr-resnet-50", "detr-resnet-101"
            ])

        if GLIP_AVAILABLE:
            available.append("glip")

        if F_VLM_AVAILABLE:
            available.append("f-vlm")

        if DETIC_AVAILABLE:
            available.append("detic")

        return available

    def detect_multiscale(self, image_path: str, texts: List[str], detector_name: str,
                     scales: List[float] = [0.8, 1.0, 1.2], threshold: float = 0.1):
        """Multi-scale inference for better small object detection"""
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        all_detections = []
        
        for scale in scales:
            # Resize image
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Detect on scaled image
            if hasattr(self.detectors[detector_name], 'detect_with_sahi'):
                result = self.detectors[detector_name].detect_with_sahi(scaled_image, texts, threshold=threshold)
            else:
                result = self.detectors[detector_name].detect(scaled_image, texts, threshold)
            
            # Rescale boxes back to original image size
            scale_factor = 1.0 / scale
            rescaled_boxes = []
            for box in result.boxes:
                x1, y1, x2, y2 = box
                rescaled_box = [x1 * scale_factor, y1 * scale_factor, 
                            x2 * scale_factor, y2 * scale_factor]
                rescaled_boxes.append(rescaled_box)
            
            result.boxes = rescaled_boxes
            all_detections.append(result)
        
        # Merge and apply NMS
        return self._merge_multiscale_results(all_detections)

    def detect_image(self, image_path: str, texts: List[str], detector_name: str,
                     threshold: float = 0.05, save_path: str = None, estimate_depth: bool = False) -> DetectionResult:
        """Detect objects in a single image"""
        if detector_name not in self.detectors:
            raise ValueError(f"Detector '{detector_name}' not found. Available: {list(self.detectors.keys())}")

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Run detection
        import time
        start_time = time.time()
        result = self.detectors[detector_name].detect(image, texts, threshold)
        result.image_path = image_path
        result.processing_time = time.time() - start_time

        # Add depth estimation if requested and available
        if estimate_depth and self.depth_estimator:
            print("Estimating depth...")
            depth_map = self.depth_estimator.estimate_depth(image)
            result.depth_map = depth_map

            if depth_map is not None and result.boxes:
                distances = self.depth_estimator.estimate_object_distances(image, result)
                # Add distance info to labels
                for i, distance in enumerate(distances):
                    if i < len(result.labels):
                        result.labels[i] += f" (depth: {distance:.2f})"

        # Save annotated image if requested
        if save_path:
            self._save_annotated_image(image, result, save_path)

        return result

    def detect_directory(self, directory_path: str, texts: List[str], detector_name: str,
                         threshold: float = 0.05, output_dir: str = None) -> List[DetectionResult]:
        """Detect objects in all images in a directory"""
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory '{directory_path}' does not exist")

        # Find all image files
        image_files = []
        for ext in self.supported_image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"No images found in {directory_path}")
            return []

        # Create output directory if specified
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        print(f"Processing {len(image_files)} images...")

        for image_file in tqdm(image_files):
            try:
                save_path = None
                if output_dir:
                    save_path = os.path.join(output_dir, f"annotated_{image_file.name}")

                result = self.detect_image(str(image_file), texts, detector_name, threshold, save_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        # Save summary results
        if output_dir:
            self._save_detection_summary(results, os.path.join(output_dir, "detection_summary.json"))

        return results

    def detect_video(self, video_path: str, texts: List[str], detector_name: str,
                     threshold: float = 0.05, output_path: str = None,
                     sample_rate: int = 1) -> List[DetectionResult]:
        """Detect objects in video frames"""
        if not os.path.exists(video_path):
            raise ValueError(f"Video '{video_path}' does not exist")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video '{video_path}'")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        results = []

        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        processed_frames = 0

        print(f"Processing video: {total_frames} frames at {fps} FPS")
        pbar = tqdm(total=total_frames // sample_rate)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame based on sample_rate
                if frame_count % sample_rate == 0:
                    # Convert frame to PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    # Run detection
                    result = self.detectors[detector_name].detect(pil_image, texts, threshold)
                    result.image_path = f"frame_{frame_count:06d}"
                    results.append(result)

                    # Annotate frame if saving video
                    if output_path:
                        annotated_frame = self._annotate_frame(frame, result)
                        out.write(annotated_frame)

                    processed_frames += 1
                    pbar.update(1)

                frame_count += 1

        finally:
            cap.release()
            if output_path:
                out.release()
            pbar.close()

        print(f"Processed {processed_frames} frames from video")
        return results

    def _save_annotated_image(self, image: Image.Image, result: DetectionResult, save_path: str):
        """Save image with detection annotations"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)

        for box, score, label in zip(result.boxes, result.scores, result.labels):
            x1, y1, x2, y2 = box

            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label and score
            ax.text(
                x1, y1 - 10,
                f'{label}: {score:.2f}',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=10
            )

        ax.set_title(f"Detection Results ({result.model_path})")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Annotate a video frame with detection results"""
        annotated_frame = frame.copy()

        for box, score, label in zip(result.boxes, result.scores, result.labels):
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Add label and score
            text = f'{label}: {score:.2f}'
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10),
                          (x1 + text_size[0], y1), (255, 255, 0), -1)
            cv2.putText(annotated_frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return annotated_frame

    def _save_detection_summary(self, results: List[DetectionResult], save_path: str):
        """Save detection results summary to JSON"""
        summary = {
            "total_images": len(results),
            "total_detections": sum(len(r.boxes) for r in results),
            "model_path": results[0].model_path if results else "unknown",
            "results": []
        }

        for result in results:
            summary["results"].append({
                "image_path": result.image_path,
                "detections": len(result.boxes),
                "objects": [
                    {
                        "label": label,
                        "confidence": score,
                        "bbox": box
                    }
                    for box, score, label in zip(result.boxes, result.scores, result.labels)
                ]
            })

        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Detection summary saved to {save_path}")


# Example usage and setup with GPU support
def check_gpu_status():
    """Check GPU availability and status"""
    print("=== GPU Status ===")
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")

        # Clear GPU cache
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared")
        return True
    else:
        print("❌ CUDA not available - using CPU")
        print("To enable GPU:")
        print("  1. Install CUDA toolkit")
        print(
            "  2. Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False


def main():
    """Example usage of the multi-modal detector system with all available models"""

    # Check GPU status first
    gpu_available = check_gpu_status()
    device = "cuda" if gpu_available else "cpu"

    # Initialize the detection system with depth estimation
    detector_system = MultiModalDetector(enable_depth=True)

    print(f"\n🔧 Loading all available detectors on {device}...")

    # Add OWLv2 detectors
    if OWLV2_AVAILABLE:
        try:
            print(f"Loading OWLv2...")
            owlv2_detector = OWLv2Detector("google/owlv2-base-patch16-ensemble", device=device)
            detector_system.add_detector("owlv2", owlv2_detector)
        except Exception as e:
            print(f"⚠️  OWLv2 failed: {e}")

    # Add YOLO-World detectors
    if YOLO_WORLD_AVAILABLE:
        try:
            print(f"Loading YOLO-World...")
            yolo_detector = YOLOWorldDetector("yolov8s-world.pt", device=device)
            detector_system.add_detector("yolo-world", yolo_detector)
        except Exception as e:
            print(f"⚠️  YOLO-World failed: {e}")

    # Add GroundingDINO detector
    if GROUNDING_DINO_AVAILABLE:
        try:
            print(f"Loading GroundingDINO...")
            grounding_detector = GroundingDINODetector(device=device)
            detector_system.add_detector("grounding-dino", grounding_detector)
        except Exception as e:
            print(f"⚠️  GroundingDINO failed: {e}")

    # Add DETR detector
    if DETR_AVAILABLE:
        try:
            print(f"Loading DETR...")
            detr_detector = DETRDetector("facebook/detr-resnet-50", device=device)
            detector_system.add_detector("detr", detr_detector)
        except Exception as e:
            print(f"⚠️  DETR failed: {e}")

    if not detector_system.detectors:
        print("❌ No detectors loaded! Install required packages:")
        print("  pip install transformers ultralytics")
        print("  pip install groundingdino-py  # for GroundingDINO")
        return

    print(f"\n✅ Loaded {len(detector_system.detectors)} detectors: {list(detector_system.detectors.keys())}")

    # Define drone/aerial objects to detect
    target_objects = [
        "drone", "UAV", "quadcopter", "unmanned aerial vehicle",
        "aircraft", "helicopter", "plane", "bird", "flying object",
        "multirotor", "rotorcraft", "aerial vehicle"
    ]

    # Performance monitoring
    import time

    # Example 1: Test all models on a single image
    image_path = "path/to/your/aerial/image.jpg"
    if os.path.exists(image_path):
        print(f"\n🔍 Testing all models on: {Path(image_path).name}")

        results_comparison = {}

        for detector_name in detector_system.detectors.keys():
            print(f"\n--- Testing {detector_name.upper()} ---")
            try:
                start_time = time.time()

                result = detector_system.detect_image(
                    image_path, target_objects, detector_name,
                    threshold=0.05,
                    save_path=f"result_{detector_name}.jpg",
                    estimate_depth=True  # Enable depth estimation
                )

                processing_time = time.time() - start_time

                print(f"✓ Found {len(result.boxes)} objects in {processing_time:.2f}s")

                results_comparison[detector_name] = {
                    'detections': len(result.boxes),
                    'time': processing_time,
                    'labels': result.labels[:3]  # Show first 3 detections
                }

                for i, (box, score, label) in enumerate(zip(result.boxes[:3], result.scores[:3], result.labels[:3])):
                    print(f"  {i + 1}. {label} (confidence: {score:.3f})")

                # GPU memory usage
                if gpu_available:
                    memory_used = torch.cuda.memory_allocated() / 1e9
                    print(f"  GPU Memory: {memory_used:.2f}GB")

            except Exception as e:
                print(f"✗ Error with {detector_name}: {e}")
                results_comparison[detector_name] = {'error': str(e)}

        # Model comparison summary
        print(f"\n{'=' * 60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Model':<20} {'Time (s)':<10} {'Objects':<10} {'Best For'}")
        print(f"{'-' * 60}")

        model_recommendations = {
            'owlv2': 'General purpose, good accuracy',
            'yolo-world': 'Fast inference, video processing',
            'grounding-dino': 'Best open-vocabulary detection',
            'detr': 'Standard objects, good baseline'
        }

        for model, stats in results_comparison.items():
            if 'error' not in stats:
                rec = model_recommendations.get(model, 'Good option')
                print(f"{model:<20} {stats['time']:<10.2f} {stats['detections']:<10} {rec}")
            else:
                print(f"{model:<20} {'ERROR':<10} {'N/A':<10} Install required packages")

    else:
        print(f"ℹ️  No test image found. Update image_path to test detection.")

    # Example 2: Recommend best model for different use cases
    print(f"\n💡 MODEL RECOMMENDATIONS:")
    print(f"  🎯 Best Detection: GroundingDINO (if installed)")
    print(f"  ⚡ Fastest: YOLO-World")
    print(f"  🎥 Video Processing: YOLO-World or OWLv2")
    print(f"  📏 With Depth: Any model + depth estimation")
    print(f"  🛸 Aerial/Drone: GroundingDINO or OWLv2")

    print(f"\n🚀 NEXT STEPS:")
    print(f"  1. Install missing packages for better models")
    print(f"  2. Test on your aerial footage")
    print(f"  3. Use the best performing model for batch processing")

    return detector_system


if __name__ == "__main__":
    main()