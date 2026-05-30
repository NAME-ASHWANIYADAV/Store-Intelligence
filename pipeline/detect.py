"""
Store Intelligence System - Core Detection Module
YOLO11l + BoT-SORT (with native ReID) person detection and tracking.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import structlog

logger = structlog.get_logger("detection")


class PersonDetector:
    """
    YOLO11l-based person detector with BoT-SORT tracking.
    Uses native Ultralytics ReID for identity retention.
    """

    def __init__(self, model_path: str = "yolo11l.pt", tracker_config: str = "botsort.yaml",
                 confidence: float = 0.25, vid_stride: int = 3):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.confidence = confidence
        self.vid_stride = vid_stride
        self.person_class = 0  # COCO person class

        logger.info(
            "detector_initialized",
            model=model_path,
            tracker=tracker_config,
            confidence=confidence,
            vid_stride=vid_stride,
        )

    def process_video(self, video_path: str, camera_id: str) -> List[Dict]:
        """
        Process a video file and extract per-frame person detections with tracking IDs.

        Returns list of frame results:
        [
            {
                "frame_idx": int,
                "timestamp_sec": float,
                "detections": [
                    {
                        "track_id": int,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float,
                        "centroid": (cx, cy),
                    }
                ]
            }
        ]
        """
        video_path = str(video_path)
        logger.info("processing_video", video_path=video_path, camera_id=camera_id)

        # Get video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        logger.info("video_metadata", fps=fps, total_frames=total_frames,
                     duration_sec=round(total_frames / fps, 1))

        # Run YOLO tracking with BoT-SORT + native ReID
        results = self.model.track(
            source=video_path,
            tracker=self.tracker_config,
            persist=True,
            classes=[self.person_class],
            conf=self.confidence,
            vid_stride=self.vid_stride,
            stream=True,  # Memory efficient streaming
            verbose=False,
        )

        frame_results = []
        frame_idx = 0

        for result in results:
            detections = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()

                for bbox, track_id, conf in zip(boxes, track_ids, confidences):
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    detections.append({
                        "track_id": int(track_id),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(conf),
                        "centroid": (float(cx), float(cy)),
                    })

            timestamp_sec = (frame_idx * self.vid_stride) / fps

            frame_results.append({
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "detections": detections,
            })

            frame_idx += 1

        logger.info(
            "video_processed",
            camera_id=camera_id,
            total_frames_processed=len(frame_results),
            unique_tracks=len(set(
                d["track_id"]
                for fr in frame_results
                for d in fr["detections"]
            )),
        )

        return frame_results

    def get_track_summary(self, frame_results: List[Dict]) -> Dict[int, Dict]:
        """
        Build a summary of each track: first/last frame, total frames present, bboxes.
        """
        tracks = defaultdict(lambda: {
            "first_frame": float('inf'),
            "last_frame": 0,
            "frame_count": 0,
            "centroids": [],
            "confidences": [],
            "bboxes": [],
        })

        for fr in frame_results:
            for det in fr["detections"]:
                tid = det["track_id"]
                tracks[tid]["first_frame"] = min(tracks[tid]["first_frame"], fr["frame_idx"])
                tracks[tid]["last_frame"] = max(tracks[tid]["last_frame"], fr["frame_idx"])
                tracks[tid]["frame_count"] += 1
                tracks[tid]["centroids"].append(det["centroid"])
                tracks[tid]["confidences"].append(det["confidence"])
                tracks[tid]["bboxes"].append(det["bbox"])

        return dict(tracks)
