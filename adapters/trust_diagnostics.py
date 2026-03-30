"""
TRUST Pipeline Diagnostic Logger
=================================
Comprehensive logging for reverse-engineering pipeline behavior.
Drop-in module — adapter calls these functions at key pipeline stages.

Usage:
    from trust_diagnostics import TrustDiagnostics
    diag = TrustDiagnostics(output_dir="./diagnostics", enabled=True)
    
    # In adapt_and_detect:
    diag.log_vlm_raw(scores, labels, class_probs, frame_idx, video_name)
    diag.log_cache_state(global_cache, frame_idx, video_name)
    diag.log_cache_adaptation(raw_probs, adapted_probs, labels, frame_idx)
    diag.log_cache_update(cache, features, boxes, probs, scores, frame_idx, image=None)
    diag.log_fusion_decision(p_init, p_global, p_track, fused, track, frame_idx)
    diag.log_track_stad(track_manager, frame_idx)
    
    # At end of video:
    diag.flush_video(video_name)
    
    # At end of evaluation:
    diag.generate_report()
"""

import numpy as np
import os
import json
from collections import defaultdict
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path


CLASS_NAMES = ['pedestrian', 'car', 'truck', 'bus', 'motorcycle', 'bicycle']


class TrustDiagnostics:
    """
    Comprehensive diagnostic logger for the TRUST pipeline.
    
    All logging is zero-cost when enabled=False.
    When enabled, collects per-frame statistics and writes summaries per video.
    """
    
    def __init__(self, output_dir: str = "./trust_diagnostics", 
                 enabled: bool = False,
                 save_crops: bool = False,
                 max_crops_per_class: int = 50,
                 log_every: int = 1):
        self.enabled = enabled
        self.save_crops = save_crops
        self.max_crops_per_class = max_crops_per_class
        self.log_every = log_every
        
        if not enabled:
            return
        
        print(f"Saving diagnostics to {output_dir}")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._reset_accumulators()
    
    def _reset_accumulators(self):
        """Reset all per-video accumulators."""
        # ── Log 1: VLM raw output statistics ──
        # Per-class score distributions (to determine rational tau1 per class)
        self.vlm_scores_by_class = defaultdict(list)   # class_name -> [scores]
        self.vlm_max_probs_by_class = defaultdict(list) # class_name -> [max_prob]
        self.vlm_total_detections = 0
        
        # ── Log 2: Cache composition over time ──
        # Snapshots of cache state at regular intervals
        self.cache_snapshots = []  # [{frame, M, class_counts, class_confidences}]
        
        # ── Log 3: Cache admission/rejection ──
        # What enters the cache vs what's rejected (and why)
        self.cache_admitted = defaultdict(int)    # class_name -> count admitted
        self.cache_rejected = defaultdict(int)    # class_name -> count rejected by tau1
        self.cache_rejected_scores = defaultdict(list)  # class_name -> [rejected scores]
        
        # ── Log 4: Cache adaptation effect ──
        # How the cache changes detections
        self.adaptation_class_changes = []  # [(frame, from_class, to_class, score)]
        self.adaptation_score_deltas = defaultdict(list)  # class_name -> [adapted-raw]
        self.adaptation_relabel_counts = defaultdict(lambda: defaultdict(int))  # from -> {to -> count}
        
        # ── Log 5: Fusion disagreement log ──
        self.fusion_disagreements = []  # [{frame, init_cls, global_cls, track_cls, fused_cls, ...}]
        self.fusion_agreement_count = 0
        self.fusion_total_count = 0
        
        # ── Log 6: Track STAD belief evolution ──
        # Per-track class belief history (π over time)
        self.track_belief_histories = {}  # track_id -> [(frame, pi_vector)]
        
        # ── Log 7: Per-frame summary ──
        self.frame_summaries = []
        
        # ── Crop storage ──
        self.crop_counts = defaultdict(int)

        self._timing_accum = defaultdict(list)
    
    # ================================================================
    # LOG 1: VLM Raw Output Analysis
    # ================================================================
    
    def log_vlm_raw(self, scores: np.ndarray, labels: list, 
                     class_probs: np.ndarray, frame_idx: int,
                     video_name: str = ""):
        """
        Log raw VLM detection output BEFORE any adaptation.
        This reveals the VLM's native per-class confidence distribution.
        
        Call this right after Stage 1 (VLM Detection) in adapt_and_detect.
        """
        if not self.enabled:
            return
        
        N = len(scores)
        self.vlm_total_detections += N
        
        for i in range(N):
            cls = labels[i] if i < len(labels) else "unknown"
            self.vlm_scores_by_class[cls].append(float(scores[i]))
            self.vlm_max_probs_by_class[cls].append(float(np.max(class_probs[i])))
    
    # ================================================================
    # LOG 2: Cache State Snapshots
    # ================================================================
    
    def log_cache_state(self, cache, frame_idx: int, video_name: str = ""):
        """
        Snapshot the current cache composition.
        
        Call this at the START of each frame (before any updates).
        Shows how the cache evolves over the video.
        """
        if not self.enabled or cache is None:
            return
        if frame_idx % self.log_every != 0:
            return
        
        M = cache.M
        if M == 0:
            self.cache_snapshots.append({
                'frame': frame_idx,
                'M': 0,
                'class_counts': {},
                'class_mean_conf': {},
                'batch_init_done': getattr(cache, 'batch_init_done', False)
            })
            return
        
        # Analyze cache entries by class
        class_counts = defaultdict(int)
        class_confidences = defaultdict(list)
        
        for j in range(M):
            top_class_idx = int(np.argmax(cache.V_cache[:, j]))
            top_class_name = CLASS_NAMES[top_class_idx] if top_class_idx < len(CLASS_NAMES) else f"cls{top_class_idx}"
            top_conf = float(np.max(cache.V_cache[:, j]))
            class_counts[top_class_name] += 1
            class_confidences[top_class_name].append(top_conf)
        
        snapshot = {
            'frame': frame_idx,
            'M': M,
            'class_counts': dict(class_counts),
            'class_mean_conf': {k: float(np.mean(v)) for k, v in class_confidences.items()},
            'batch_init_done': getattr(cache, 'batch_init_done', False)
        }
        
        # Add hit counts per entry if available
        if hasattr(cache, 'hits') and cache.hits is not None:
            snapshot['total_hits'] = int(np.sum(cache.hits[:M]))
            snapshot['max_hits'] = int(np.max(cache.hits[:M]))
        
        self.cache_snapshots.append(snapshot)
    
    # ================================================================
    # LOG 3: Cache Admission/Rejection
    # ================================================================
    
    def log_cache_update_attempt(self, probs: np.ndarray, scores: np.ndarray,
                                  tau1: float, tau1_per_class: Optional[dict],
                                  frame_idx: int):
        """
        Log which detections are admitted to / rejected from the cache.
        
        Call this inside update_cache, BEFORE the tau1 filtering.
        """
        if not self.enabled:
            return
        
        N = len(scores)
        for i in range(N):
            pred_class_idx = int(np.argmax(probs[i]))
            pred_class = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else f"cls{pred_class_idx}"
            
            # Determine effective threshold
            if tau1_per_class is not None:
                effective_tau1 = tau1_per_class.get(pred_class_idx, tau1)
            else:
                effective_tau1 = tau1
            
            if scores[i] >= effective_tau1:
                self.cache_admitted[pred_class] += 1
            else:
                self.cache_rejected[pred_class] += 1
                self.cache_rejected_scores[pred_class].append(float(scores[i]))
    
    # ================================================================
    # LOG 4: Cache Adaptation Effect
    # ================================================================
    
    def log_cache_adaptation(self, raw_probs: np.ndarray, adapted_probs: np.ndarray,
                              labels: list, frame_idx: int):
        """
        Log how the cache changes class probabilities.
        
        Call this after adapt_probs_batch, comparing raw vs adapted.
        Shows relabeling and score shifts per class.
        """
        if not self.enabled:
            return
        
        N = len(raw_probs)
        for i in range(N):
            raw_cls_idx = int(np.argmax(raw_probs[i]))
            adapted_cls_idx = int(np.argmax(adapted_probs[i]))
            
            raw_cls = CLASS_NAMES[raw_cls_idx] if raw_cls_idx < len(CLASS_NAMES) else f"cls{raw_cls_idx}"
            adapted_cls = CLASS_NAMES[adapted_cls_idx] if adapted_cls_idx < len(CLASS_NAMES) else f"cls{adapted_cls_idx}"
            
            # Score delta for the originally-predicted class
            raw_score = float(raw_probs[i, raw_cls_idx])
            adapted_score = float(adapted_probs[i, raw_cls_idx])
            self.adaptation_score_deltas[raw_cls].append(adapted_score - raw_score)
            
            # Class change
            if raw_cls_idx != adapted_cls_idx:
                self.adaptation_relabel_counts[raw_cls][adapted_cls] += 1
                self.adaptation_class_changes.append({
                    'frame': frame_idx,
                    'from': raw_cls,
                    'to': adapted_cls,
                    'raw_conf': float(np.max(raw_probs[i])),
                    'adapted_conf': float(np.max(adapted_probs[i]))
                })
    
    # ================================================================
    # LOG 4b: Visual Cache Debugging (crop storage)
    # ================================================================
    
    def log_cache_crop(self, image: np.ndarray, box: np.ndarray, 
                        pred_class: str, score: float, 
                        cache_entry_id: int, is_new: bool,
                        frame_idx: int, video_name: str = ""):
        """
        Save image crop of a detection that entered the cache.
        
        Call inside update_cache when a detection is admitted.
        Requires the image to be passed through from the adapter.
        """
        if not self.enabled or not self.save_crops:
            return
        if self.crop_counts[pred_class] >= self.max_crops_per_class:
            return
        
        try:
            x1, y1, x2, y2 = [int(c) for c in box]
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return
            
            crop = image[y1:y2, x1:x2]
            
            crop_dir = self.output_dir / "cache_crops" / video_name / pred_class
            crop_dir.mkdir(parents=True, exist_ok=True)
            
            action = "NEW" if is_new else "UPD"
            filename = f"f{frame_idx:06d}_{action}_e{cache_entry_id}_s{score:.2f}.jpg"
            
            import cv2
            cv2.imwrite(str(crop_dir / filename), crop)
            self.crop_counts[pred_class] += 1
            
        except Exception as e:
            pass  # Never let logging crash the pipeline
    
    # ================================================================
    # LOG 5: Fusion Disagreement Analysis
    # ================================================================
    
    def log_fusion_decision(self, p_init: np.ndarray, p_global: np.ndarray,
                             p_track: np.ndarray, fused: np.ndarray,
                             track_hits: int, track_id: int,
                             frame_idx: int):
        """
        Log the 3-source fusion decision for a tracked detection.
        
        Call inside the matched-track loop after _fuse_probs.
        """
        if not self.enabled:
            return
        
        self.fusion_total_count += 1
        
        init_cls = int(np.argmax(p_init))
        global_cls = int(np.argmax(p_global))
        track_cls = int(np.argmax(p_track))
        fused_cls = int(np.argmax(fused))
        
        all_agree = (init_cls == global_cls == track_cls)
        
        if all_agree:
            self.fusion_agreement_count += 1
            return  # Only log disagreements in detail
        
        # Compute entropies (same as _fuse_probs)
        eps = 1e-10
        H_init = float(-np.sum(p_init * np.log(p_init + eps)))
        H_global = float(-np.sum(p_global * np.log(p_global + eps)))
        H_track = float(-np.sum(p_track * np.log(p_track + eps)))
        
        def cn(idx):
            return CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"cls{idx}"
        
        record = {
            'frame': frame_idx,
            'track_id': track_id,
            'track_hits': track_hits,
            'init_cls': cn(init_cls),
            'global_cls': cn(global_cls),
            'track_cls': cn(track_cls),
            'fused_cls': cn(fused_cls),
            'init_conf': float(np.max(p_init)),
            'global_conf': float(np.max(p_global)),
            'track_conf': float(np.max(p_track)),
            'H_init': H_init,
            'H_global': H_global,
            'H_track': H_track,
            'cache_overrode_vlm': (global_cls != init_cls and fused_cls == global_cls),
            'track_overrode_cache': (track_cls != global_cls and fused_cls == track_cls),
        }
        
        self.fusion_disagreements.append(record)
    
    # ================================================================
    # LOG 6: Track STAD Belief Evolution
    # ================================================================
    
    def log_track_stad_beliefs(self, track_manager, frame_idx: int):
        """
        Snapshot the class belief (π) for all active tracks with STAD.
        
        Call at end of frame after all track updates.
        """
        if not self.enabled or track_manager is None:
            return
        if frame_idx % self.log_every != 0:
            return
        
        for track in track_manager.get_active_tracks():
            if track.class_stad is None:
                continue
            
            tid = track.track_id
            pi = track.get_class_probs()
            
            if tid not in self.track_belief_histories:
                self.track_belief_histories[tid] = []
            
            self.track_belief_histories[tid].append({
                'frame': frame_idx,
                'pi': pi.tolist(),
                'hits': track.hits,
                'top_class': CLASS_NAMES[int(np.argmax(pi))] if int(np.argmax(pi)) < len(CLASS_NAMES) else "?",
                'entropy': float(-np.sum(pi * np.log(pi + 1e-10)))
            })
    
    def log_frame_timing(self, timings: dict, frame_idx: int):
            """Accumulate per-frame timing breakdowns for computational cost reporting."""
            if not self.enabled:
                return
            if not hasattr(self, '_timing_accum'):
                self._timing_accum = defaultdict(list)
            for key, val in timings.items():
                self._timing_accum[key].append(float(val))

    def log_gpu_memory(self):
        """Snapshot GPU memory usage (if torch is available)."""
        if not self.enabled:
            return
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_memory = {
                    'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                    'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
                    'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                }
        except ImportError:
            pass

    def log_ram_usage(self):
        """Snapshot process RAM usage."""
        if not self.enabled:
            return
        try:
            import psutil
            process = psutil.Process()
            self._ram_mb = process.memory_info().rss / 1024**2
        except ImportError:
            import resource
            self._ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB→MB on Linux

    # ================================================================
    # Flush & Report
    # ================================================================
    
    def flush_video(self, video_name: str):
        """Write accumulated logs for this video and reset."""
        if not self.enabled:
            return

        
        video_dir = self.output_dir / video_name if video_name is not None else None
        if video_dir is not None:
            video_dir.mkdir(parents=True, exist_ok=True)
        
            # Capture resource usage BEFORE generating report
            self.log_gpu_memory()
            self.log_ram_usage()
            
            report = self._generate_video_report(video_name)
            
            # Add resource usage to report
            if hasattr(self, '_gpu_memory'):
                report['gpu_memory'] = self._gpu_memory
            if hasattr(self, '_ram_mb'):
                report['ram_usage_mb'] = self._ram_mb
            
            # Save JSON report
            with open(video_dir / f"diagnostics_{video_name}.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
            # Save human-readable summary
            with open(video_dir / f"summary_{video_name}.txt", 'w') as f:
                self._write_summary(f, report, video_name)
            
            # Save detailed logs
            if self.fusion_disagreements:
                with open(video_dir / f"fusion_disagreements_{video_name}.json", 'w') as f:
                    json.dump(self.fusion_disagreements[:1000], f, indent=2)
            
            if self.adaptation_class_changes:
                with open(video_dir / f"relabeling_events_{video_name}.json", 'w') as f:
                    json.dump(self.adaptation_class_changes[:1000], f, indent=2)
            
            # Save track belief histories (top 20 longest tracks)
            if self.track_belief_histories:
                sorted_tracks = sorted(self.track_belief_histories.items(), 
                                        key=lambda x: len(x[1]), reverse=True)[:20]
                beliefs = {str(tid): hist for tid, hist in sorted_tracks}
                with open(video_dir / f"track_beliefs_{video_name}.json", 'w') as f:
                    json.dump(beliefs, f, indent=2)
        
        self._reset_accumulators()
    
    def _generate_video_report(self, video_name: str) -> dict:
        """Generate structured report for one video."""
        report = {
            'video': video_name,
            'total_vlm_detections': self.vlm_total_detections,
        }
        
        # VLM score distributions per class
        vlm_stats = {}
        for cls in CLASS_NAMES:
            scores = self.vlm_scores_by_class.get(cls, [])
            if len(scores) > 0:
                arr = np.array(scores)
                vlm_stats[cls] = {
                    'count': len(scores),
                    'mean': float(np.mean(arr)),
                    'median': float(np.median(arr)),
                    'p25': float(np.percentile(arr, 25)),
                    'p75': float(np.percentile(arr, 75)),
                    'p90': float(np.percentile(arr, 90)),
                    'above_0.3': int((arr >= 0.3).sum()),
                    'above_0.5': int((arr >= 0.5).sum()),
                    'above_0.8': int((arr >= 0.8).sum()),
                    'rational_tau1': float(np.percentile(arr, 50)),  # median as suggestion
                }
        report['vlm_score_distributions'] = vlm_stats
        
        # Cache composition evolution
        if self.cache_snapshots:
            report['cache_evolution'] = {
                'snapshots': self.cache_snapshots[-10:],  # Last 10
                'final_state': self.cache_snapshots[-1] if self.cache_snapshots else None,
            }
        
        # Cache admission/rejection
        admission = {}
        for cls in CLASS_NAMES:
            admitted = self.cache_admitted.get(cls, 0)
            rejected = self.cache_rejected.get(cls, 0)
            total = admitted + rejected
            rej_scores = self.cache_rejected_scores.get(cls, [])
            admission[cls] = {
                'admitted': admitted,
                'rejected': rejected,
                'admission_rate': admitted / total if total > 0 else 0,
                'rejected_mean_score': float(np.mean(rej_scores)) if rej_scores else 0,
                'rejected_max_score': float(np.max(rej_scores)) if rej_scores else 0,
            }
        report['cache_admission'] = admission
        
        # Adaptation relabeling
        relabel_summary = {}
        total_relabeled = 0
        for from_cls, to_dict in self.adaptation_relabel_counts.items():
            for to_cls, count in to_dict.items():
                key = f"{from_cls} → {to_cls}"
                relabel_summary[key] = count
                total_relabeled += count
        report['adaptation_relabeling'] = {
            'total_relabeled': total_relabeled,
            'top_transitions': dict(sorted(relabel_summary.items(), key=lambda x: -x[1])[:10]),
        }
        
        # Adaptation score deltas per class
        score_delta_summary = {}
        for cls in CLASS_NAMES:
            deltas = self.adaptation_score_deltas.get(cls, [])
            if deltas:
                arr = np.array(deltas)
                score_delta_summary[cls] = {
                    'mean_delta': float(np.mean(arr)),
                    'median_delta': float(np.median(arr)),
                    'count_suppressed': int((arr < -0.05).sum()),  # Significantly reduced
                    'count_boosted': int((arr > 0.05).sum()),
                }
        report['adaptation_score_effects'] = score_delta_summary
        
        # Fusion statistics
        report['fusion'] = {
            'total_fused': self.fusion_total_count,
            'all_agree': self.fusion_agreement_count,
            'disagreements': len(self.fusion_disagreements),
            'agreement_rate': self.fusion_agreement_count / max(1, self.fusion_total_count),
            'cache_override_count': sum(1 for d in self.fusion_disagreements if d.get('cache_overrode_vlm')),
            'track_override_count': sum(1 for d in self.fusion_disagreements if d.get('track_overrode_cache')),
        }
        
        # Fusion disagreement breakdown by class
        if self.fusion_disagreements:
            override_by_class = defaultdict(int)
            for d in self.fusion_disagreements:
                if d.get('cache_overrode_vlm'):
                    override_by_class[f"{d['init_cls']}→{d['global_cls']}"] += 1
            report['fusion']['cache_override_breakdown'] = dict(
                sorted(override_by_class.items(), key=lambda x: -x[1])[:10]
            )

        # Computational footprint
        if hasattr(self, '_timing_accum') and self._timing_accum:
            timing_report = {}
            for key, vals in self._timing_accum.items():
                arr = np.array(vals)
                timing_report[key] = {
                    'mean_ms': float(np.mean(arr) * 1000),
                    'std_ms': float(np.std(arr) * 1000),
                    'total_s': float(np.sum(arr)),
                    'n_frames': len(vals),
                }
            report['computational_footprint'] = timing_report
        
        return report
    
    def _write_summary(self, f, report: dict, video_name: str):
        """Write human-readable summary."""
        f.write(f"{'='*80}\n")
        f.write(f"TRUST Diagnostics — {video_name}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Total VLM detections: {report['total_vlm_detections']}\n\n")
        
        # VLM score distributions
        f.write("── VLM Score Distributions (raw, pre-adaptation) ──\n")
        f.write(f"{'Class':<14s} {'Count':>7s} {'Mean':>7s} {'Med':>7s} {'P90':>7s} {'>0.3':>7s} {'>0.5':>7s} {'>0.8':>7s} {'Sug.tau1':>8s}\n")
        f.write("-" * 80 + "\n")
        for cls in CLASS_NAMES:
            s = report.get('vlm_score_distributions', {}).get(cls, {})
            if s:
                f.write(f"{cls:<14s} {s['count']:>7d} {s['mean']:>7.3f} {s['median']:>7.3f} "
                       f"{s['p90']:>7.3f} {s['above_0.3']:>7d} {s['above_0.5']:>7d} "
                       f"{s['above_0.8']:>7d} {s['rational_tau1']:>8.3f}\n")
        
        # Cache admission
        f.write("\n── Cache Admission vs Rejection ──\n")
        f.write(f"{'Class':<14s} {'Admitted':>9s} {'Rejected':>9s} {'Rate':>7s} {'Rej Mean':>9s} {'Rej Max':>8s}\n")
        f.write("-" * 65 + "\n")
        for cls in CLASS_NAMES:
            a = report.get('cache_admission', {}).get(cls, {})
            if a:
                f.write(f"{cls:<14s} {a['admitted']:>9d} {a['rejected']:>9d} "
                       f"{a['admission_rate']:>7.1%} {a['rejected_mean_score']:>9.3f} "
                       f"{a['rejected_max_score']:>8.3f}\n")
        
        # Cache composition
        if 'cache_evolution' in report and report['cache_evolution'].get('final_state'):
            final = report['cache_evolution']['final_state']
            f.write(f"\n── Final Cache State (frame {final['frame']}) ──\n")
            f.write(f"  Total entries: {final['M']}\n")
            for cls, count in sorted(final.get('class_counts', {}).items(), key=lambda x: -x[1]):
                conf = final.get('class_mean_conf', {}).get(cls, 0)
                f.write(f"  {cls:<14s}: {count:>3d} entries, mean conf={conf:.3f}\n")
        
        # Relabeling
        relab = report.get('adaptation_relabeling', {})
        if relab.get('total_relabeled', 0) > 0:
            f.write(f"\n── Cache Relabeling (total: {relab['total_relabeled']}) ──\n")
            for transition, count in relab.get('top_transitions', {}).items():
                f.write(f"  {transition}: {count:>6d}\n")
        
        # Fusion
        fus = report.get('fusion', {})
        f.write(f"\n── Fusion Statistics ──\n")
        f.write(f"  Total fused decisions: {fus.get('total_fused', 0)}\n")
        f.write(f"  All 3 sources agree: {fus.get('all_agree', 0)} ({fus.get('agreement_rate', 0):.1%})\n")
        f.write(f"  Cache overrode VLM: {fus.get('cache_override_count', 0)}\n")
        f.write(f"  Track overrode cache: {fus.get('track_override_count', 0)}\n")
        if 'cache_override_breakdown' in fus:
            f.write(f"  Top cache overrides:\n")
            for transition, count in fus['cache_override_breakdown'].items():
                f.write(f"    {transition}: {count}\n")

        # Computational footprint
        if 'computational_footprint' in report:
            f.write(f"\n── Computational Footprint ──\n")
            f.write(f"{'Stage':<20s} {'Mean(ms)':>10s} {'Std(ms)':>10s} {'Total(s)':>10s}\n")
            f.write("-" * 55 + "\n")
            for stage, stats in sorted(report['computational_footprint'].items(), 
                                        key=lambda x: -x[1]['mean_ms']):
                f.write(f"{stage:<20s} {stats['mean_ms']:>10.1f} {stats['std_ms']:>10.1f} "
                       f"{stats['total_s']:>10.2f}\n")
            total = report['computational_footprint'].get('total', {})
            if total:
                fps = total['n_frames'] / max(0.001, total['total_s'])
                f.write(f"\n  FPS: {fps:.2f} ({total['n_frames']} frames in {total['total_s']:.1f}s)\n")


def integrate_diagnostics_example():
    """
    Shows WHERE to add each logging call in global_instance_adapter.py.
    This is pseudo-code — adapt to your actual line numbers.
    """
    print("""
    # In __init__:
    from trust_diagnostics import TrustDiagnostics
    self.diag = TrustDiagnostics(
        output_dir=os.path.join(params.get('output_dir', '.'), 'diagnostics'),
        enabled=params.get('diagnostics', False),
        save_crops=params.get('diagnostic_crops', False),
    )
    
    # In adapt_and_detect, after Stage 1 (VLM Detection):
    self.diag.log_vlm_raw(scores, labels, class_probs, self.frame_count, self._current_video_name)
    
    # After Stage 3 (Global BCA+ Adaptation):
    self.diag.log_cache_state(self.global_cache, self.frame_count, self._current_video_name)
    self.diag.log_cache_adaptation(raw_vlm_probs, global_adapted_probs, labels, self.frame_count)
    
    # Inside Stage 7 (matched track loop), after _fuse_probs:
    self.diag.log_fusion_decision(p_init, p_global, p_track, fused_result, 
                                   track.hits, track.track_id, self.frame_count)
    
    # At end of frame:
    self.diag.log_track_stad_beliefs(self.track_manager, self.frame_count)
    
    # In reset() (between videos):
    self.diag.flush_video(self._current_video_name or f"video_{self.frame_count}")
    """)


if __name__ == "__main__":
    integrate_diagnostics_example()
