"""
Store Intelligence System - Full Pipeline Runner
Processes all 5 CCTV cameras with YOLO11l + BoT-SORT on GPU.
Uses the proper EventEmitter API to generate JSONL events.
"""
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

from pipeline.detect import PersonDetector
from pipeline.zones import ZoneManager
from pipeline.event_emitter import EventEmitter

logger = structlog.get_logger("full_pipeline")

# ===== Configuration =====
DATA_DIR = Path(r"c:\Users\HP\OneDrive\Desktop\purplle\data")
VIDEO_DIR = DATA_DIR / "CCTV Footage"
LAYOUT_PATH = Path(__file__).parent.parent / "data" / "store_layout.json"
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "events"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STORE_ID = "STORE_BLR_001"

# Camera mapping — type determines what events to emit
CAMERAS = {
    "CAM 1.mp4": {"id": "CAM_ENTRY_01", "type": "entry"},
    "CAM 2.mp4": {"id": "CAM_FLOOR_01", "type": "floor"},
    "CAM 3.mp4": {"id": "CAM_FLOOR_02", "type": "floor"},
    "CAM 4.mp4": {"id": "CAM_ENTRY_02", "type": "entry"},
    "CAM 5.mp4": {"id": "CAM_BILLING_01", "type": "billing"},
}


def main():
    start_time = time.time()
    logger.info("pipeline_start", store_id=STORE_ID, cameras=len(CAMERAS))

    # Initialize
    detector = PersonDetector(
        model_path="yolo11l.pt",
        tracker_config="botsort.yaml",
        confidence=0.25,
        vid_stride=3,
    )
    zone_manager = ZoneManager(str(LAYOUT_PATH))
    emitter = EventEmitter(output_dir=str(OUTPUT_DIR))

    total_events = 0

    for video_name, cam_config in CAMERAS.items():
        video_path = VIDEO_DIR / video_name
        if not video_path.exists():
            logger.warning("video_not_found", path=str(video_path))
            continue

        camera_id = cam_config["id"]
        cam_type = cam_config["type"]
        logger.info("processing_camera", camera=camera_id, type=cam_type, file=video_name)

        # ===== Step 1: Detect + Track =====
        frame_results = detector.process_video(str(video_path), camera_id)
        tracks = detector.get_track_summary(frame_results)

        logger.info("detection_complete",
                     camera=camera_id,
                     frames=len(frame_results),
                     tracks=len(tracks))

        if not tracks:
            logger.warning("no_tracks_detected", camera=camera_id)
            continue

        # ===== Step 2: Classify staff (>60% presence = staff) =====
        total_frames = len(frame_results)
        staff_ids = set()
        for tid, summary in tracks.items():
            presence = summary["frame_count"] / max(total_frames, 1)
            if presence > 0.6:
                staff_ids.add(tid)

        logger.info("staff_classified",
                     camera=camera_id,
                     staff=len(staff_ids),
                     customers=len(tracks) - len(staff_ids))

        # ===== Step 3: Generate events per track =====
        fps = 25.0  # approximate
        for tid, summary in tracks.items():
            is_staff = tid in staff_ids
            visitor_id = f"V_{STORE_ID}_{camera_id}_{tid:04d}"
            avg_conf = sum(summary["confidences"]) / len(summary["confidences"])

            first_sec = (summary["first_frame"] * 3) / fps  # vid_stride=3
            last_sec = (summary["last_frame"] * 3) / fps

            # ENTRY event
            emitter.emit_entry(
                store_id=STORE_ID,
                camera_id=camera_id,
                visitor_id=visitor_id,
                timestamp_sec=first_sec,
                confidence=avg_conf,
                is_staff=is_staff,
            )

            # ZONE events for floor cameras
            if cam_type == "floor":
                visited_zones = set()
                for i, centroid in enumerate(summary["centroids"]):
                    zone = zone_manager.get_zone_for_point(STORE_ID, camera_id, (centroid[0], centroid[1]))
                    if zone and zone not in visited_zones:
                        visited_zones.add(zone)
                        zone_enter_sec = first_sec + (i * 3 / fps)

                        emitter.emit_zone_enter(
                            store_id=STORE_ID,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            zone_id=zone,
                            timestamp_sec=zone_enter_sec,
                            confidence=avg_conf,
                            is_staff=is_staff,
                        )

                # Dwell for each zone visited
                if visited_zones:
                    total_sec = last_sec - first_sec
                    dwell_per_zone_ms = int((total_sec * 1000) / max(len(visited_zones), 1))

                    for zone in visited_zones:
                        emitter.emit_zone_dwell(
                            store_id=STORE_ID,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            zone_id=zone,
                            timestamp_sec=first_sec + 15,
                            dwell_ms=dwell_per_zone_ms,
                            confidence=avg_conf,
                            is_staff=is_staff,
                        )

                        emitter.emit_zone_exit(
                            store_id=STORE_ID,
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            zone_id=zone,
                            timestamp_sec=last_sec - 5,
                            dwell_ms=dwell_per_zone_ms,
                            confidence=avg_conf,
                            is_staff=is_staff,
                        )

            # BILLING events for billing camera (customers only)
            if cam_type == "billing" and not is_staff:
                queue_depth = len(tracks) - len(staff_ids)
                emitter.emit_billing_queue_join(
                    store_id=STORE_ID,
                    camera_id=camera_id,
                    visitor_id=visitor_id,
                    timestamp_sec=first_sec + 5,
                    confidence=avg_conf,
                    queue_depth=queue_depth,
                )

            # EXIT event
            emitter.emit_exit(
                store_id=STORE_ID,
                camera_id=camera_id,
                visitor_id=visitor_id,
                timestamp_sec=last_sec,
                confidence=avg_conf,
                is_staff=is_staff,
            )

        # Write events for this camera
        count = emitter.write_events(STORE_ID)
        total_events += count
        logger.info("camera_events_written", camera=camera_id, events=count)

    # ===== Summary =====
    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — Store Intelligence System")
    print(f"{'='*60}")
    print(f"  Total events:     {total_events}")
    print(f"  Processing time:  {elapsed:.1f}s")
    print(f"  GPU:              NVIDIA GeForce GTX 1650")
    print(f"  Model:            YOLO11l + BoT-SORT (ReID)")
    print(f"  Output:           {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
