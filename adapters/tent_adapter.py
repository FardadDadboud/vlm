"""
TENT adapter (Wang et al., ICLR 2021): gradient-based TTA baseline.

Adapts a frozen GroundingDINO detector at test time by minimising the mean
entropy of per-query class predictions w.r.t. the model's LayerNorm gamma/beta
parameters only. Everything else stays frozen, matching the canonical TENT
recipe.

Used to give the paper a clean "gradient-based TTA" comparator alongside the
backpropagation-free baselines (BCA+, STAD) and TRUST. Reset-on-video-change
mirrors TRUST's per-stream protocol so the runtime / accuracy comparison is
apples-to-apples.

Notes:
  * Only GroundingDINO is supported here. The forward pass reproduces the
    relevant parts of GroundingDINODetector._detect_transformers_with_features,
    but with autograd enabled. OWLv2-TENT is out of scope for this comparator.
  * NMS / thresholding at the output mirror the vanilla GD path so the post-
    processing surface is shared with the rest of the comparison table.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from .base_adapter import BaseAdapter
from vlm_detector_system_new import DetectionResult


def _collect_layernorm_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Return all gamma/beta tensors from every LayerNorm in the model."""
    params: List[torch.nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    return params


def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean per-row entropy of a probability matrix."""
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1).mean()


class TENTAdapter(BaseAdapter):
    """
    Gradient-based TTA via entropy minimisation on LayerNorm gamma/beta.

    Per-frame procedure:
      1. Run the GD forward pass with grad enabled and softmax-aggregate the
         class-token logits into per-query class probabilities.
      2. Compute mean entropy across queries (no labels needed).
      3. Adam step over LayerNorm gamma/beta only.
      4. Use the same forward outputs (detached) to build the DetectionResult.

    Hyperparameters (overridable via config["adaptation"]["params"]):
      lr                    1e-3      Adam learning rate
      betas                 (0.9, 0.999)
      weight_decay          0.0
      grad_clip             1.0       max-norm clip on LN params; 0 disables
      use_amp               True      forward in autocast on CUDA
    """

    def __init__(self, detector, config: Dict):
        super().__init__(detector, config)
        self.iou_threshold = config["detector"].get("iou_threshold") or 0.3
        params = config.get("adaptation", {}).get("params", {})

        self.lr             = float(params.get("lr", 1e-3))
        self.betas          = tuple(params.get("betas", (0.9, 0.999)))
        self.weight_decay   = float(params.get("weight_decay", 0.0))
        self.grad_clip      = float(params.get("grad_clip", 1.0))
        self.use_amp        = bool(params.get("use_amp", True))

        # GroundingDINODetector.model is an HF pipeline; the actual nn.Module
        # is at .model. The image_processor / tokenizer also live on the pipe.
        self._pipe       = detector.model
        self._hf_model   = self._pipe.model
        self._image_proc = self._pipe.image_processor
        self._tokenizer  = self._pipe.tokenizer
        self._device     = next(self._hf_model.parameters()).device

        # Freeze everything; flip LayerNorm gamma/beta back on for adaptation.
        for p in self._hf_model.parameters():
            p.requires_grad_(False)
        self._adapt_params = _collect_layernorm_params(self._hf_model)
        for p in self._adapt_params:
            p.requires_grad_(True)
        if not self._adapt_params:
            raise RuntimeError(
                "TENTAdapter: no LayerNorm parameters found in GroundingDINO; "
                "cannot run entropy minimisation."
            )

        # Snapshot the originals so reset() restores the source model exactly.
        self._original_state = {
            id(p): p.detach().clone() for p in self._adapt_params
        }

        # train() lets dropout fire; LayerNorm has no running stats so its
        # behaviour is unaffected by mode. Both are standard for TENT.
        self._hf_model.train()

        self._optimizer = self._make_optimizer()

        self._frame_count = 0
        self._loss_total = 0.0
        self._loss_count = 0

    # ---- factory helpers ---------------------------------------------------

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self._adapt_params,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def reset(self):
        """Restore source LayerNorm values and reinitialise the optimizer.

        Called by the orchestrator on video boundaries (matches the
        reset_on_video_change protocol used for TRUST and the other adaptive
        baselines, so per-video drift is comparable).
        """
        with torch.no_grad():
            for p in self._adapt_params:
                p.copy_(self._original_state[id(p)])
        self._optimizer = self._make_optimizer()

    def get_stats(self) -> Dict:
        avg_loss = (self._loss_total / self._loss_count) if self._loss_count else 0.0
        return {
            "tent_frames": self._frame_count,
            "tent_avg_entropy_loss": avg_loss,
            "tent_lr": self.lr,
        }

    # ---- main hook ---------------------------------------------------------

    def adapt_and_detect(
        self, image: Image.Image, target_classes: List[str], threshold: float
    ) -> DetectionResult:
        if not target_classes:
            return DetectionResult([], [], [], "", self.detector.model_path)

        text_prompt = " . ".join(target_classes) + " ."

        # ---- inputs (mirrors _detect_transformers_with_features) -----------
        image_inputs = self._image_proc(images=image, return_tensors="pt")
        text_inputs = self._tokenizer(
            text_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        offset_mapping = text_inputs.pop("offset_mapping")[0]

        inputs = {
            **{k: v.to(self._device) for k, v in image_inputs.items()},
            **{k: v.to(self._device) for k, v in text_inputs.items()},
        }

        # ---- grad-enabled forward + TENT step ------------------------------
        self._optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            if self.use_amp and self._device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = self._hf_model(**inputs, return_dict=True)
            else:
                outputs = self._hf_model(**inputs, return_dict=True)

            logits = outputs.logits[0]      # (num_queries, max_text_tokens)
            pred_boxes = outputs.pred_boxes[0]  # (num_queries, 4) cxcywh in [0,1]

            class_token_indices = self._class_token_indices(text_prompt, target_classes, offset_mapping)

            # Aggregate multi-token classes with MAX over their token logits,
            # then softmax across classes -> simplex (matches the inference path).
            class_logits_cols = []
            for token_indices in class_token_indices:
                idx = torch.tensor(token_indices, device=logits.device, dtype=torch.long)
                class_logits_cols.append(logits.index_select(1, idx).amax(dim=1))
            class_logits = torch.stack(class_logits_cols, dim=1)
            class_probs = torch.softmax(class_logits.float(), dim=-1)

            loss = _entropy(class_probs)

        if torch.isfinite(loss):
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._adapt_params, self.grad_clip)
            self._optimizer.step()
            self._loss_total += float(loss.detach().cpu())
            self._loss_count += 1
        self._frame_count += 1

        # ---- build DetectionResult from the same forward outputs ----------
        with torch.no_grad():
            class_probs_np = class_probs.detach().float().cpu().numpy()
            pred_boxes_np = pred_boxes.detach().float().cpu().numpy()

        all_scores = class_probs_np.max(axis=1)
        all_label_idx = class_probs_np.argmax(axis=1)
        all_labels = [target_classes[i] for i in all_label_idx]

        # cxcywh in [0,1] -> xyxy in original image pixels
        image_w, image_h = image.size
        boxes_xyxy = np.zeros_like(pred_boxes_np)
        boxes_xyxy[:, 0] = (pred_boxes_np[:, 0] - pred_boxes_np[:, 2] / 2) * image_w
        boxes_xyxy[:, 1] = (pred_boxes_np[:, 1] - pred_boxes_np[:, 3] / 2) * image_h
        boxes_xyxy[:, 2] = (pred_boxes_np[:, 0] + pred_boxes_np[:, 2] / 2) * image_w
        boxes_xyxy[:, 3] = (pred_boxes_np[:, 1] + pred_boxes_np[:, 3] / 2) * image_h
        boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0.0, image_w)
        boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0.0, image_w)
        boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0.0, image_h)
        boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0.0, image_h)

        # Threshold + NMS, matching the GD vanilla path.
        keep = np.where(all_scores >= threshold)[0]
        if keep.size == 0:
            return DetectionResult([], [], [], "", self.detector.model_path)
        boxes_list = boxes_xyxy[keep].tolist()
        scores_list = all_scores[keep].tolist()
        labels_list = [all_labels[i] for i in keep]

        nms_keep = self.detector._nms_boxes(boxes_list, scores_list, self.iou_threshold)
        boxes_list = [boxes_list[i] for i in nms_keep]
        scores_list = [scores_list[i] for i in nms_keep]
        labels_list = [labels_list[i] for i in nms_keep]

        return DetectionResult(
            boxes=boxes_list,
            scores=scores_list,
            labels=labels_list,
            image_path="",
            model_path=self.detector.model_path,
        )

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _class_token_indices(
        text_prompt: str,
        classes: List[str],
        offset_mapping: torch.Tensor,
    ) -> List[List[int]]:
        """Map each class string to the tokenizer indices that span it.

        Mirrors the offset-span aggregation used by the inference path, so the
        TENT loss is computed on the exact same per-class scores the rest of
        the system reads.
        """
        out: List[List[int]] = []
        for class_name in classes:
            start_char = text_prompt.find(class_name)
            if start_char == -1:
                out.append([0])
                continue
            end_char = start_char + len(class_name)
            indices: List[int] = []
            for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
                if tok_start < end_char and tok_end > start_char and tok_start != tok_end:
                    indices.append(tok_idx)
            out.append(indices if indices else [0])
        return out
