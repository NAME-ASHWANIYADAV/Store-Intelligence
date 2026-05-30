"""
Store Intelligence v2.0 — Staff Classifier
Classifies a tracklet as staff or customer based on zone-time ratio.

Sources:
- Architect Report: >30% time in staff_zone = staff
- DeepSeek: staff_prob prevents cross-merge in deduplication
"""

import cv2
import numpy as np
from typing import Optional
from tracking.tracklet_record import TrackletRecord
from config.models import StaffZone


def classify_person(
    tracklet: TrackletRecord,
    staff_zone: Optional[StaffZone],
    frame_w: int,
    frame_h: int,
) -> str:
    """
    Classify a tracklet as 'staff' or 'customer'.

    Logic: If person spent >threshold% of their tracked time
    inside the staff_zone polygon, they're staff.
    """
    if staff_zone is None:
        return "customer"

    total_frames = len(tracklet.positions)
    if total_frames == 0:
        return "unknown"

    px_poly = staff_zone.to_pixel_polygon(frame_w, frame_h)

    frames_in_zone = sum(
        1 for (_, cx, cy, _, _) in tracklet.positions
        if cv2.pointPolygonTest(px_poly, (float(cx), float(cy)), False) >= 0
    )

    ratio = frames_in_zone / total_frames
    return "staff" if ratio > staff_zone.threshold else "customer"


def compute_staff_probability(
    tracklet: TrackletRecord,
    staff_zone: Optional[StaffZone],
    frame_w: int,
    frame_h: int,
) -> float:
    """
    Compute staff probability (0.0-1.0) for use in dedup cross-merge prevention.
    Source: DeepSeek.
    """
    if staff_zone is None or len(tracklet.positions) == 0:
        return 0.0

    px_poly = staff_zone.to_pixel_polygon(frame_w, frame_h)

    frames_in_zone = sum(
        1 for (_, cx, cy, _, _) in tracklet.positions
        if cv2.pointPolygonTest(px_poly, (float(cx), float(cy)), False) >= 0
    )

    return frames_in_zone / len(tracklet.positions)
