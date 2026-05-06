"""Smoke test for OWLv2Detector.detect_with_features.

Loads OWLv2, runs detect_with_features() on one image (defaults to a SHIFT
sample under figs/, falls back to a synthetic 800x600 RGB), and prints sanity
checks on the DetectionResult fields TRUST consumes:

  - boxes are finite and within image bounds
  - class_probs sums to ~1 per query (simplex), no NaN/Inf
  - features and text_embeddings have matching dim, no NaN/Inf
  - shapes are mutually consistent

Run from code/VLM:
  python3 test_owlv2_features.py
  python3 test_owlv2_features.py --image /path/to/image.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_detector_system_new import OWLv2Detector

CLASSES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]


def _pick_image(arg_path: str | None) -> Image.Image:
    if arg_path:
        return Image.open(arg_path).convert("RGB")
    # Try a few SHIFT sample frames shipped under papers/ for convenience.
    candidates = [
        Path("../../papers/neurips2026-v3/figs/foggy_daytime_313_b7ad-7710_000013.jpg"),
        Path("../../papers/neurips2026-v3/figs/cloudy_daytime_471_c728-34ee_000071.jpg"),
        Path("../../papers/neurips2026-v3/figs/overcast_daytime_209_9c17-f275_000009.jpg"),
        Path("../../papers/neurips2026-v3/figs/rainy_daytime_046_3794-668b_000046.jpg"),
    ]
    for p in candidates:
        if p.exists():
            print(f"[smoke] using sample image: {p}")
            return Image.open(p).convert("RGB")
    print("[smoke] no SHIFT sample found, generating synthetic 800x600 image")
    return Image.fromarray(np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8))


def _check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="path to a test image")
    parser.add_argument("--alpha", type=float, default=0.7, help="hybrid feature alpha")
    args = parser.parse_args()

    image = _pick_image(args.image)
    image_w, image_h = image.size
    print(f"[smoke] image size: {image_w}x{image_h}")

    print("[smoke] loading OWLv2 ...")
    det = OWLv2Detector()
    print("[smoke] running detect_with_features ...")
    res = det.detect_with_features(image, CLASSES, threshold=0.05, alpha=args.alpha)

    print("[smoke] checks:")
    n = len(res.boxes)
    all_ok = True

    all_ok &= _check("non-empty result", n > 0, f"n_queries={n}")
    if n == 0:
        return 1

    boxes = np.asarray(res.boxes, dtype=np.float32)
    scores = np.asarray(res.scores, dtype=np.float32)
    cprobs = np.asarray(res.class_probs, dtype=np.float64)
    feats = np.asarray(res.features, dtype=np.float64)
    text_emb = np.asarray(res.text_embeddings, dtype=np.float64)
    raw_feats = np.asarray(res.raw_features, dtype=np.float64)
    raw_text = np.asarray(res.raw_text_embeddings, dtype=np.float64)

    all_ok &= _check("boxes shape", boxes.shape == (n, 4), f"shape={boxes.shape}")
    all_ok &= _check("class_probs shape", cprobs.shape == (n, len(CLASSES)),
                     f"shape={cprobs.shape}, expected=({n},{len(CLASSES)})")
    all_ok &= _check(
        "class_probs sums to 1",
        np.allclose(cprobs.sum(axis=1), 1.0, atol=1e-4),
        f"max|sum-1|={np.max(np.abs(cprobs.sum(axis=1)-1.0)):.2e}",
    )
    all_ok &= _check("class_probs no NaN/Inf",
                     np.isfinite(cprobs).all(), f"finite={np.isfinite(cprobs).all()}")
    all_ok &= _check("features no NaN/Inf",
                     np.isfinite(feats).all(), f"shape={feats.shape}")
    all_ok &= _check("raw_features no NaN/Inf",
                     np.isfinite(raw_feats).all(), f"shape={raw_feats.shape}")
    all_ok &= _check("text_embeddings shape",
                     text_emb.shape[0] == len(CLASSES) and text_emb.shape[1] == feats.shape[1],
                     f"text={text_emb.shape}, feat_dim={feats.shape[1]}")
    all_ok &= _check("raw_text_embeddings shape",
                     raw_text.shape[0] == len(CLASSES) and raw_text.shape[1] == raw_feats.shape[1],
                     f"raw_text={raw_text.shape}, raw_feat_dim={raw_feats.shape[1]}")
    all_ok &= _check(
        "boxes within image (allowing small overshoot)",
        ((boxes[:, 0] >= -2).all() and (boxes[:, 2] <= image_w + 2).all()
         and (boxes[:, 1] >= -2).all() and (boxes[:, 3] <= image_h + 2).all()),
        f"x_range=[{boxes[:,0].min():.1f},{boxes[:,2].max():.1f}], "
        f"y_range=[{boxes[:,1].min():.1f},{boxes[:,3].max():.1f}]",
    )
    all_ok &= _check("scores in [0,1]",
                     (scores >= 0).all() and (scores <= 1).all(),
                     f"min={scores.min():.4f}, max={scores.max():.4f}")
    all_ok &= _check(
        "score = max class prob (consistency)",
        np.allclose(scores, cprobs.max(axis=1), atol=1e-5),
        f"max diff={np.max(np.abs(scores - cprobs.max(axis=1))):.2e}",
    )

    print()
    print(f"[smoke] top-5 confident detections (post-softmax, pre-NMS):")
    top = np.argsort(-scores)[:5]
    for k in top:
        b = boxes[k]
        print(
            f"  q={k:5d}  score={scores[k]:.3f}  "
            f"label={res.labels[k]:<10s}  "
            f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]"
        )

    print()
    print(f"[smoke] feature dim: {feats.shape[1]} (raw={raw_feats.shape[1]}, "
          f"hybrid offset={feats.shape[1] - raw_feats.shape[1]})")
    print(f"[smoke] num candidate queries: {n}")

    print()
    if all_ok:
        print("[smoke] ALL CHECKS PASSED — OWLv2 detect_with_features is wired correctly.")
        return 0
    else:
        print("[smoke] SOME CHECKS FAILED — see above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
