"""Smoke test for TENTAdapter.

Loads GroundingDINO, wraps it with TENTAdapter, runs adapt_and_detect on one
image (defaults to a SHIFT sample under figs/, falls back to a synthetic image),
and validates:

  - the adapter constructs (LayerNorm params discovered, optimizer initialised)
  - the forward/backward step actually runs (loss is finite, optimizer steps OK)
  - the returned DetectionResult has the same shape conventions as the
    vanilla GD path (boxes within image, scores in [0,1], non-empty)
  - reset() restores LayerNorm gamma/beta to source values
  - per-frame stats are populated

Run from code/VLM:
  python3 test_tent_features.py
  python3 test_tent_features.py --image /path/to/image.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_detector_system_new import GroundingDINODetector
from adapters import create_adapter

CLASSES = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]


def _pick_image(arg_path):
    if arg_path:
        return Image.open(arg_path).convert("RGB")
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


def _check(name, ok, detail=""):
    mark = "OK " if ok else "FAIL"
    print(f"  [{mark}] {name}{(' — ' + detail) if detail else ''}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--n-frames", type=int, default=3,
                        help="how many times to call adapt_and_detect (TENT updates each call)")
    args = parser.parse_args()

    image = _pick_image(args.image)
    print(f"[smoke] image size: {image.size[0]}x{image.size[1]}")

    print("[smoke] loading GroundingDINO ...")
    det = GroundingDINODetector()

    config = {
        "detector": {"iou_threshold": 0.7},
        "adaptation": {
            "type": "tent",
            "params": {"lr": 1e-3, "grad_clip": 1.0, "use_amp": True},
        },
    }
    print("[smoke] wrapping with TENTAdapter ...")
    tent = create_adapter("tent", det, config)

    print("[smoke] checks:")
    all_ok = True
    n_ln = len(tent._adapt_params)
    all_ok &= _check("LayerNorm params discovered", n_ln > 0,
                     f"n_ln_params={n_ln}")

    # Snapshot one LN param so we can verify it changes after a step and
    # is restored after reset().
    probe = tent._adapt_params[0]
    pre_step = probe.detach().clone()

    print(f"[smoke] running adapt_and_detect x{args.n_frames} ...")
    last_res = None
    for i in range(args.n_frames):
        res = tent.adapt_and_detect(image, CLASSES, threshold=0.1)
        last_res = res
        stats = tent.get_stats()
        print(f"  frame {i+1}: n_dets={len(res.boxes):4d}  "
              f"avg_entropy_loss={stats['tent_avg_entropy_loss']:.4f}")

    n_dets = len(last_res.boxes)
    all_ok &= _check("adapt_and_detect returns non-empty", n_dets > 0,
                     f"n_dets={n_dets}")
    if n_dets > 0:
        boxes = np.asarray(last_res.boxes, dtype=np.float32)
        scores = np.asarray(last_res.scores, dtype=np.float32)
        W, H = image.size
        all_ok &= _check("scores in [0,1]",
                         (scores >= 0).all() and (scores <= 1).all(),
                         f"min={scores.min():.4f}, max={scores.max():.4f}")
        all_ok &= _check(
            "boxes within image",
            ((boxes[:, 0] >= 0).all() and (boxes[:, 2] <= W).all()
             and (boxes[:, 1] >= 0).all() and (boxes[:, 3] <= H).all()),
            f"x_range=[{boxes[:,0].min():.1f},{boxes[:,2].max():.1f}], "
            f"y_range=[{boxes[:,1].min():.1f},{boxes[:,3].max():.1f}]",
        )

    # LN params should have moved during adaptation.
    moved = float((probe - pre_step).abs().max())
    all_ok &= _check("LayerNorm gamma/beta updated by TENT", moved > 0,
                     f"max|delta|={moved:.2e}")

    # reset() should restore.
    tent.reset()
    diff_after_reset = float((probe - pre_step).abs().max())
    all_ok &= _check("reset() restores LayerNorm to source", diff_after_reset < 1e-6,
                     f"max|delta|={diff_after_reset:.2e}")

    stats = tent.get_stats()
    all_ok &= _check("stats populated",
                     stats["tent_frames"] == args.n_frames,
                     json.dumps(stats))

    print()
    if all_ok:
        print("[smoke] ALL CHECKS PASSED — TENT adapter is wired correctly.")
        return 0
    else:
        print("[smoke] SOME CHECKS FAILED — see above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
