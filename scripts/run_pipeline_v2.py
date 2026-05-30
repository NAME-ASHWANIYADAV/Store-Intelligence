"""
Store Intelligence System v2.1 -- Config-Driven Pipeline Runner
FIXES from v2.0 run:
1. Added histogram-based appearance embeddings for deduplication (zero VRAM)
2. Increased merger spatial distance and relaxed threshold
3. Staff zones adjusted in YAML

Architecture (from 6 AI research sources):
1. Load YAML config -> camera roles, zones, ignore regions, tripwires
2. Per camera: YOLO detect -> spatial filter -> BoT-SORT track -> collect tracklets
3. Extract color histogram embeddings every 5th frame (DeepSeek interval)
4. Post-process: TrackletMerger deduplicates fragmented IDs
5. Generate events: line crossing (entry), zone dwell (floor), billing
6. Classify staff vs customer by zone-time ratio
7. Write JSONL events
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import StoreConfig, CameraConfig
from pipeline.ignore_filter import SpatialFilter
from events.line_crossing import DirectionalLineCrossing
from events.zone_engine import ZoneEngine
from events.staff_classifier import classify_person, compute_staff_probability
from tracking.tracklet_record import TrackletRecord
from tracking.tracklet_merger import TrackletMerger

# -- Paths --
CONFIG_PATH = PROJECT_ROOT / "configs" / "store_001.yaml"
DATA_DIR = Path(r"c:\Users\HP\OneDrive\Desktop\purplle\data\CCTV Footage")
OUTPUT_DIR = PROJECT_ROOT / "output" / "events_v2"
BOTSORT_YAML = PROJECT_ROOT / "botsort.yaml"
YOLO_WEIGHTS = "yolo11l.pt"
VID_STRIDE = 3
REID_SAMPLE_INTERVAL = 5  # Extract embedding every 5th detection (DeepSeek)


def compute_histogram_embedding(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Compute a lightweight color histogram embedding from a person crop.
    No extra model needed -- zero VRAM cost.

    Uses HSV color space + spatial layout for better appearance matching.
    Returns L2-normalized 192-dim vector.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_frame, w_frame = frame.shape[:2]

    # Clamp to frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_frame, x2), min(h_frame, y2)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return np.zeros(192, dtype=np.float32)

    crop = frame[y1:y2, x1:x2]
    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Split person into upper body (torso/shirt) and lower body (pants)
    mid_y = crop_hsv.shape[0] // 2
    upper = crop_hsv[:mid_y]
    lower = crop_hsv[mid_y:]
    full = crop_hsv

    features = []
    for region in [upper, lower, full]:
        # H channel: 32 bins (hue is most discriminative)
        h_hist = cv2.calcHist([region], [0], None, [32], [0, 180])
        # S channel: 16 bins
        s_hist = cv2.calcHist([region], [1], None, [16], [0, 256])
        # V channel: 16 bins
        v_hist = cv2.calcHist([region], [2], None, [16], [0, 256])
        features.extend([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])

    embedding = np.concatenate(features).astype(np.float32)  # 3 * (32+16+16) = 192

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-9:
        embedding /= norm

    return embedding


def load_yolo_model():
    """Load YOLO model with GPU."""
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    model.to("cuda")
    print(f"[OK] YOLO model loaded: {YOLO_WEIGHTS} on CUDA")
    return model


def process_camera(
    model,
    cam_config: CameraConfig,
    store_config: StoreConfig,
) -> dict:
    """
    Process a single camera's video through the full pipeline.
    Returns dict with: tracklets, events, stats
    """
    cam_id = cam_config.id
    video_path = DATA_DIR / cam_config.source
    print(f"\n{'='*60}")
    print(f"Processing {cam_id}: {cam_config.label} (role={cam_config.role})")
    print(f"  Source: {video_path}")
    print(f"{'='*60}")

    if not video_path.exists():
        print(f"  [ERR] Video not found: {video_path}")
        return {"tracklets": {}, "events": [], "stats": {}}

    # -- Open video --
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or store_config.fps_default
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"  Video: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")

    # -- Build spatial filter --
    spatial_filter = SpatialFilter(cam_config, w, h)

    # -- Build line crossing detector (entry cameras only) --
    line_detector = None
    if cam_config.role == "entry" and cam_config.tripwire:
        line_detector = DirectionalLineCrossing.from_config(cam_config.tripwire, w, h)
        print(f"  [OK] Tripwire line configured")

    # -- Build zone engine (floor/billing cameras) --
    zone_engine = None
    if cam_config.zones:
        zone_engine = ZoneEngine(cam_config.zones, w, h, fps)
        print(f"  [OK] Zone engine: {len(cam_config.zones)} zones")

    # -- PASS 1: Run YOLO + BoT-SORT and collect frames for embedding --
    print(f"  Running detection + tracking (vid_stride={VID_STRIDE})...")
    t0 = time.time()

    # We need to re-read frames for histogram embedding, so use 2-pass approach:
    # Pass 1: track and record bboxes per frame
    # Actually, model.track with stream gives us result per frame with .orig_img
    results = model.track(
        source=str(video_path),
        tracker=str(BOTSORT_YAML),
        persist=True,
        classes=[0],
        conf=cam_config.tracker_config.track_high_thresh,
        iou=0.45,
        vid_stride=VID_STRIDE,
        stream=True,
        verbose=False,
    )

    # -- Collect tracklets + generate online events + extract embeddings --
    tracklet_gallery: dict[int, TrackletRecord] = {}
    frame_counts_per_track: dict[int, int] = {}  # for embedding sampling
    all_events = []
    frame_idx = 0
    detection_count = 0
    embedding_count = 0

    for result in results:
        frame_idx += 1
        actual_frame = frame_idx * VID_STRIDE

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes
        if boxes.id is None:
            continue

        # Get raw detections as numpy
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        # Get the original frame for histogram embedding
        orig_frame = result.orig_img

        # -- Apply spatial filter --
        if spatial_filter.has_ignore_regions:
            keep_mask = np.ones(len(xyxy), dtype=bool)
            for i in range(len(xyxy)):
                foot_x = int((xyxy[i, 0] + xyxy[i, 2]) / 2)
                foot_y = int(xyxy[i, 3])
                foot_x = np.clip(foot_x, 0, w - 1)
                foot_y = np.clip(foot_y, 0, h - 1)
                if spatial_filter.mask[foot_y, foot_x] == 0:
                    keep_mask[i] = False

            xyxy = xyxy[keep_mask]
            track_ids = track_ids[keep_mask]
            confs = confs[keep_mask]

        detection_count += len(track_ids)

        # -- Process each tracked person --
        for i in range(len(track_ids)):
            tid = int(track_ids[i])
            x1, y1, x2, y2 = xyxy[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = x2 - x1
            bh = y2 - y1

            foot_x = cx
            foot_y = y2

            # -- Update tracklet gallery --
            if tid not in tracklet_gallery:
                tracklet_gallery[tid] = TrackletRecord(
                    track_id=tid,
                    camera_id=cam_id,
                    start_frame=actual_frame,
                    end_frame=actual_frame,
                )
                frame_counts_per_track[tid] = 0

            # Sample histogram embedding every N frames (DeepSeek optimization)
            embedding = None
            frame_counts_per_track[tid] += 1
            if frame_counts_per_track[tid] % REID_SAMPLE_INTERVAL == 1:
                if orig_frame is not None:
                    embedding = compute_histogram_embedding(
                        orig_frame, (x1, y1, x2, y2)
                    )
                    embedding_count += 1

            tracklet_gallery[tid].add_frame(
                actual_frame, cx, cy, bw, bh, embedding
            )

            # -- Line crossing (entry cameras) --
            if line_detector:
                event = line_detector.update(tid, foot_x, foot_y, actual_frame)
                if event:
                    event["camera_id"] = cam_id
                    event["camera_role"] = cam_config.role
                    event["timestamp"] = datetime.now(timezone.utc).isoformat()
                    all_events.append(event)

            # -- Zone events (floor/billing cameras) --
            if zone_engine:
                zone_events = zone_engine.update(tid, cx, cy, actual_frame)
                for ze in zone_events:
                    ze["camera_id"] = cam_id
                    ze["camera_role"] = cam_config.role
                    ze["timestamp"] = datetime.now(timezone.utc).isoformat()
                    all_events.append(ze)

    elapsed = time.time() - t0
    print(f"  [OK] Detection complete: {elapsed:.1f}s")
    print(f"    Raw tracks: {len(tracklet_gallery)}")
    print(f"    Detections processed: {detection_count}")
    print(f"    Embeddings extracted: {embedding_count}")
    if line_detector:
        counts = line_detector.get_counts()
        print(f"    Line crossing: {counts['entries']} entries, {counts['exits']} exits")
    print(f"    Online events: {len(all_events)}")

    # -- Post-process: Deduplicate tracklets --
    print(f"  Running tracklet deduplication...")

    # Compute staff probability before merging (DeepSeek)
    for tid, tl in tracklet_gallery.items():
        tl.staff_prob = compute_staff_probability(
            tl, cam_config.staff_zone, w, h
        )

    merger = TrackletMerger(
        fps=fps / VID_STRIDE,
        max_gap_sec=10.0,           # Increased from 8.0 -- person can be hidden longer
        max_spatial_dist_px=400.0,   # Increased from 250 -- 1920px wide frame
        match_threshold=0.55,        # Relaxed from 0.45 -- histogram is noisier than ReID
        alpha=0.50,                  # appearance weight (increased -- we now have embeddings!)
        beta=0.20,                   # temporal weight
        gamma=0.20,                  # spatial weight
        delta=0.10,                  # velocity weight
    )

    merged_gallery = merger.merge(tracklet_gallery)
    reduction = len(tracklet_gallery) - len(merged_gallery)
    print(f"    Before: {len(tracklet_gallery)} tracklets -> After: {len(merged_gallery)} "
          f"(merged {reduction})")

    # -- Classify staff vs customer --
    staff_count = 0
    customer_count = 0
    for tid, tl in merged_gallery.items():
        person_type = classify_person(tl, cam_config.staff_zone, w, h)
        if person_type == "staff":
            staff_count += 1
        else:
            customer_count += 1

    print(f"    Staff: {staff_count}, Customers: {customer_count}")

    stats = {
        "camera_id": cam_id,
        "camera_role": cam_config.role,
        "total_frames": total_frames,
        "raw_tracks": len(tracklet_gallery),
        "merged_tracks": len(merged_gallery),
        "staff": staff_count,
        "customers": customer_count,
        "events": len(all_events),
        "processing_time_sec": round(elapsed, 1),
    }
    if line_detector:
        stats.update(line_detector.get_counts())

    return {
        "tracklets": merged_gallery,
        "events": all_events,
        "stats": stats,
    }


def main():
    """Run the full pipeline on all configured cameras."""
    print("=" * 60)
    print("  STORE INTELLIGENCE SYSTEM v2.1")
    print("  With Histogram Embeddings + Improved Dedup")
    print("=" * 60)

    # -- Load config --
    config = StoreConfig.from_yaml(CONFIG_PATH)
    print(f"\n[OK] Loaded config: {config.store_id}")
    print(f"  Cameras: {len(config.cameras)} configured")
    for cam in config.cameras:
        print(f"    {cam.id}: {cam.label} (role={cam.role})")

    # -- Load model --
    model = load_yolo_model()

    # -- Process each camera --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_stats = []
    all_events = []

    total_t0 = time.time()

    for cam_config in config.cameras:
        result = process_camera(model, cam_config, config)
        all_stats.append(result["stats"])
        all_events.extend(result["events"])

    total_elapsed = time.time() - total_t0

    # -- Write events to JSONL --
    events_file = OUTPUT_DIR / f"{config.store_id}_v2.1.jsonl"
    with open(events_file, "w") as f:
        for event in all_events:
            event["store_id"] = config.store_id
            f.write(json.dumps(event, default=str) + "\n")
    print(f"\n[OK] Events written: {events_file} ({len(all_events)} events)")

    # -- Summary --
    print(f"\n{'='*60}")
    print(f"  PIPELINE v2.1 SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Camera':<12} {'Role':<8} {'Raw':<6} {'Merged':<8} {'Staff':<6} {'Cust':<6} {'Events':<8} {'Time':<8}")
    print("-" * 62)

    total_raw = 0
    total_merged = 0
    total_staff = 0
    total_cust = 0
    total_events = 0

    for s in all_stats:
        print(f"{s['camera_id']:<12} {s['camera_role']:<8} "
              f"{s['raw_tracks']:<6} {s['merged_tracks']:<8} "
              f"{s['staff']:<6} {s['customers']:<6} "
              f"{s['events']:<8} {s['processing_time_sec']:<8.1f}s")
        total_raw += s['raw_tracks']
        total_merged += s['merged_tracks']
        total_staff += s['staff']
        total_cust += s['customers']
        total_events += s['events']

    print("-" * 62)
    print(f"{'TOTAL':<12} {'':8} {total_raw:<6} {total_merged:<8} "
          f"{total_staff:<6} {total_cust:<6} {total_events:<8} {total_elapsed:<8.1f}s")

    if total_raw > 0:
        reduction_pct = (1 - total_merged / total_raw) * 100
        print(f"\n  Track dedup reduction: {reduction_pct:.1f}%")

    # -- v2.0 vs v2.1 comparison --
    print(f"\n  === v2.0 vs v2.1 COMPARISON ===")
    v2_stats = {
        "CAM_01": {"raw": 16, "merged": 16, "staff": 14, "cust": 2},
        "CAM_02": {"raw": 21, "merged": 20, "staff": 7, "cust": 13},
        "CAM_03": {"raw": 6, "merged": 6, "staff": 0, "cust": 6},
        "CAM_05": {"raw": 9, "merged": 8, "staff": 1, "cust": 7},
    }
    for s in all_stats:
        cid = s["camera_id"]
        if cid in v2_stats:
            old = v2_stats[cid]
            print(f"  {cid}: tracks {old['merged']}->{s['merged_tracks']}, "
                  f"staff {old['staff']}->{s['staff']}, "
                  f"cust {old['cust']}->{s['customers']}")

    print(f"\n[OK] Pipeline v2.1 complete!")


if __name__ == "__main__":
    main()
