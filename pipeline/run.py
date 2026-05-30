"""
Store Intelligence System - Pipeline Orchestrator
Processes all CCTV clips → generates events → outputs JSONL per store.
"""

import os
import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set
from collections import defaultdict

import structlog

from pipeline.config import get_pipeline_config
from pipeline.detect import PersonDetector
from pipeline.zones import ZoneManager
from pipeline.direction import DirectionDetector
from pipeline.staff_detector import StaffDetector
from pipeline.event_emitter import EventEmitter

logger = structlog.get_logger("pipeline")


class PipelineOrchestrator:
    """
    Main orchestrator that processes all cameras for all stores.
    Order: entry cameras first (establish IDs), then floor, then billing.
    """

    def __init__(self):
        self.config = get_pipeline_config()

        # Initialize components
        self.detector = PersonDetector(
            model_path=self.config.yolo_model,
            tracker_config=self.config.tracker_config,
            confidence=self.config.detection_confidence,
            vid_stride=self.config.vid_stride,
        )

        self.zone_manager = ZoneManager(self.config.store_layout_path)

        self.staff_detector = StaffDetector(
            dwell_threshold=self.config.staff_dwell_threshold,
            xai_api_key=self.config.xai_api_key,
            xai_api_url=self.config.xai_api_url,
            xai_model=self.config.xai_model,
        )

        # Load POS transactions for conversion matching
        self.pos_transactions = self._load_pos_transactions()

        logger.info(
            "pipeline_initialized",
            stores=self.zone_manager.get_store_ids(),
            pos_transactions=len(self.pos_transactions),
        )

    def _load_pos_transactions(self) -> Dict[str, list]:
        """Load POS transactions grouped by store_id."""
        transactions: Dict[str, list] = defaultdict(list)
        pos_path = self.config.pos_transactions_path

        if not os.path.exists(pos_path):
            logger.warning("pos_file_not_found", path=pos_path)
            return transactions

        with open(pos_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                transactions[row["store_id"]].append({
                    "transaction_id": row["transaction_id"],
                    "timestamp": row["timestamp"],
                    "basket_value_inr": float(row["basket_value_inr"]),
                })

        logger.info("pos_loaded", total=sum(len(v) for v in transactions.values()))
        return dict(transactions)

    def run(self):
        """
        Process all stores and cameras, generating events.
        Camera processing order: entry → floor → billing.
        """
        store_ids = self.zone_manager.get_store_ids()
        logger.info("pipeline_starting", store_count=len(store_ids))

        for store_id in store_ids:
            logger.info("processing_store", store_id=store_id)
            self._process_store(store_id)

        logger.info("pipeline_complete", stores_processed=len(store_ids))

    def _process_store(self, store_id: str):
        """Process all cameras for a single store."""
        cameras = self.zone_manager.get_cameras_for_store(store_id)

        # Sort cameras: entry first, then floor, then billing
        camera_order = {"entry": 0, "floor": 1, "billing": 2}
        sorted_cameras = sorted(
            cameras.items(),
            key=lambda x: camera_order.get(x[1].get("type", "floor"), 1)
        )

        emitter = EventEmitter(
            output_dir=self.config.output_dir,
        )

        # Track visitor IDs across cameras for this store
        store_visitor_map: Dict[int, str] = {}  # track_id → visitor_id
        all_staff_tracks: Set[int] = set()
        visitor_counter = 0

        for camera_id, camera_config in sorted_cameras:
            video_file = camera_config.get("file", "")
            video_path = os.path.join(self.config.clips_dir, video_file)

            if not os.path.exists(video_path):
                logger.warning(
                    "video_not_found",
                    camera_id=camera_id,
                    video_path=video_path,
                )
                continue

            camera_type = camera_config.get("type", "floor")
            logger.info(
                "processing_camera",
                store_id=store_id,
                camera_id=camera_id,
                camera_type=camera_type,
                video=video_file,
            )

            # Run detection + tracking
            frame_results = self.detector.process_video(video_path, camera_id)

            if not frame_results:
                logger.warning("no_frames", camera_id=camera_id)
                continue

            # Build track summaries
            track_summary = self.detector.get_track_summary(frame_results)
            total_frames = len(frame_results)

            # Staff detection
            camera_staff = self.staff_detector.classify_tracks(
                track_summary, total_frames, video_path
            )
            all_staff_tracks.update(camera_staff)

            # Assign visitor IDs to tracks
            for track_id in track_summary:
                if track_id not in store_visitor_map:
                    visitor_counter += 1
                    store_visitor_map[track_id] = f"V_{store_id}_{visitor_counter:04d}"

            # Process based on camera type
            if camera_type == "entry":
                self._process_entry_camera(
                    store_id, camera_id, camera_config,
                    frame_results, track_summary,
                    store_visitor_map, all_staff_tracks, emitter,
                )
            elif camera_type == "floor":
                self._process_floor_camera(
                    store_id, camera_id,
                    frame_results, track_summary,
                    store_visitor_map, all_staff_tracks, emitter,
                )
            elif camera_type == "billing":
                self._process_billing_camera(
                    store_id, camera_id,
                    frame_results, track_summary,
                    store_visitor_map, all_staff_tracks, emitter,
                )

        # Write all events for this store
        event_count = emitter.write_events(store_id)
        logger.info(
            "store_complete",
            store_id=store_id,
            total_events=event_count,
            unique_visitors=visitor_counter,
            staff_count=len(all_staff_tracks),
        )

    def _process_entry_camera(
        self, store_id, camera_id, camera_config,
        frame_results, track_summary,
        visitor_map, staff_tracks, emitter,
    ):
        """Process entry camera: detect ENTRY/EXIT events using threshold line."""
        direction_detector = DirectionDetector()
        threshold_line = camera_config.get("threshold_line")

        if not threshold_line:
            logger.warning("no_threshold_line", camera_id=camera_id)
            # Fallback: first appearance = ENTRY, last = EXIT
            for track_id, summary in track_summary.items():
                visitor_id = visitor_map.get(track_id, f"V_unknown_{track_id}")
                is_staff = track_id in staff_tracks
                avg_conf = sum(summary["confidences"]) / len(summary["confidences"])

                emitter.emit_entry(
                    store_id, camera_id, visitor_id,
                    summary["first_frame"] * (self.config.vid_stride / 15.0),
                    avg_conf, is_staff,
                )
                emitter.emit_exit(
                    store_id, camera_id, visitor_id,
                    summary["last_frame"] * (self.config.vid_stride / 15.0),
                    avg_conf, is_staff,
                )
            return

        # Use direction detection for each frame
        for fr in frame_results:
            for det in fr["detections"]:
                track_id = det["track_id"]
                visitor_id = visitor_map.get(track_id, f"V_unknown_{track_id}")
                is_staff = track_id in staff_tracks

                direction = direction_detector.update(
                    track_id, det["centroid"], threshold_line
                )

                if direction == "ENTRY":
                    emitter.emit_entry(
                        store_id, camera_id, visitor_id,
                        fr["timestamp_sec"], det["confidence"], is_staff,
                    )
                elif direction == "EXIT":
                    emitter.emit_exit(
                        store_id, camera_id, visitor_id,
                        fr["timestamp_sec"], det["confidence"], is_staff,
                    )

    def _process_floor_camera(
        self, store_id, camera_id,
        frame_results, track_summary,
        visitor_map, staff_tracks, emitter,
    ):
        """Process floor camera: detect zone transitions and dwell events."""
        # Track current zone per person
        current_zone: Dict[int, str] = {}
        zone_enter_time: Dict[int, float] = {}  # track_id → enter timestamp
        last_dwell_report: Dict[int, float] = {}  # track_id → last dwell report time

        for fr in frame_results:
            for det in fr["detections"]:
                track_id = det["track_id"]
                visitor_id = visitor_map.get(track_id, f"V_unknown_{track_id}")
                is_staff = track_id in staff_tracks

                # Determine current zone
                zone = self.zone_manager.get_zone_for_point(
                    store_id, camera_id, det["centroid"]
                )

                prev_zone = current_zone.get(track_id)

                if zone and zone != prev_zone:
                    # Zone transition detected
                    if prev_zone:
                        # Emit ZONE_EXIT for previous zone
                        enter_t = zone_enter_time.get(track_id, fr["timestamp_sec"])
                        dwell_ms = int((fr["timestamp_sec"] - enter_t) * 1000)
                        emitter.emit_zone_exit(
                            store_id, camera_id, visitor_id,
                            prev_zone, fr["timestamp_sec"],
                            dwell_ms, det["confidence"], is_staff,
                        )

                    # Emit ZONE_ENTER for new zone
                    emitter.emit_zone_enter(
                        store_id, camera_id, visitor_id,
                        zone, fr["timestamp_sec"],
                        det["confidence"], is_staff,
                    )
                    current_zone[track_id] = zone
                    zone_enter_time[track_id] = fr["timestamp_sec"]
                    last_dwell_report[track_id] = fr["timestamp_sec"]

                elif zone and zone == prev_zone:
                    # Still in same zone — check if dwell report needed
                    enter_t = zone_enter_time.get(track_id, fr["timestamp_sec"])
                    last_report = last_dwell_report.get(track_id, fr["timestamp_sec"])
                    time_since_report = (fr["timestamp_sec"] - last_report) * 1000

                    if time_since_report >= self.config.dwell_report_interval_ms:
                        dwell_ms = int((fr["timestamp_sec"] - enter_t) * 1000)
                        emitter.emit_zone_dwell(
                            store_id, camera_id, visitor_id,
                            zone, fr["timestamp_sec"],
                            dwell_ms, det["confidence"], is_staff,
                        )
                        last_dwell_report[track_id] = fr["timestamp_sec"]

        # Emit final ZONE_EXIT for tracks still in zones at end of video
        for track_id, zone in current_zone.items():
            visitor_id = visitor_map.get(track_id, f"V_unknown_{track_id}")
            is_staff = track_id in staff_tracks
            summary = track_summary.get(track_id, {})
            enter_t = zone_enter_time.get(track_id, 0)
            last_ts = summary.get("last_frame", 0) * (self.config.vid_stride / 15.0)
            dwell_ms = int((last_ts - enter_t) * 1000)
            avg_conf = sum(summary.get("confidences", [0.5])) / max(len(summary.get("confidences", [1])), 1)

            emitter.emit_zone_exit(
                store_id, camera_id, visitor_id,
                zone, last_ts, max(dwell_ms, 0), avg_conf, is_staff,
            )

    def _process_billing_camera(
        self, store_id, camera_id,
        frame_results, track_summary,
        visitor_map, staff_tracks, emitter,
    ):
        """Process billing camera: detect queue join events."""
        queue_joined: Set[int] = set()  # Track IDs that already joined queue
        current_in_billing: Set[int] = set()

        for fr in frame_results:
            frame_people = set()
            for det in fr["detections"]:
                track_id = det["track_id"]
                visitor_id = visitor_map.get(track_id, f"V_unknown_{track_id}")
                is_staff = track_id in staff_tracks
                frame_people.add(track_id)

                if track_id not in queue_joined and not is_staff:
                    queue_joined.add(track_id)
                    current_in_billing.add(track_id)

                    emitter.emit_billing_queue_join(
                        store_id, camera_id, visitor_id,
                        fr["timestamp_sec"], det["confidence"],
                        queue_depth=len(current_in_billing),
                        is_staff=is_staff,
                    )

            # Remove people who left billing area
            left = current_in_billing - frame_people
            current_in_billing -= left


def main():
    """Main entry point for the detection pipeline."""
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    logger.info("pipeline_starting")
    orchestrator = PipelineOrchestrator()
    orchestrator.run()
    logger.info("pipeline_finished")


if __name__ == "__main__":
    main()
