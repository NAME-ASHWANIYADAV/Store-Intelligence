"""
Store Intelligence v2.0 — Zone Engine
Polygon containment + dwell timers for floor/billing cameras.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from config.models import Zone


class ZoneEngine:
    """
    Tracks person-zone interactions:
    - ZONE_ENTER when centroid enters a polygon
    - ZONE_DWELL when person stays beyond threshold
    """

    def __init__(self, zones: List[Zone], frame_w: int, frame_h: int, fps: float = 25.0):
        self.fps = fps
        self.zone_defs = []
        for z in zones:
            px_poly = z.to_pixel_polygon(frame_w, frame_h)
            self.zone_defs.append({
                "id": z.id,
                "label": z.label,
                "polygon": px_poly,
                "dwell_threshold_frames": int(z.dwell_threshold_sec * fps),
            })

        # Per-track state: {track_id: {zone_id: {"entered_frame": int, "dwell_emitted": bool}}}
        self._track_zones: Dict[int, Dict[str, Dict]] = {}

    def update(
        self,
        track_id: int,
        cx: float,
        cy: float,
        frame_idx: int,
    ) -> List[Dict]:
        """
        Check which zones this centroid is in and emit events.

        Returns list of event dicts (may be empty).
        """
        events = []

        if track_id not in self._track_zones:
            self._track_zones[track_id] = {}

        track_state = self._track_zones[track_id]

        for zdef in self.zone_defs:
            zone_id = zdef["id"]
            inside = cv2.pointPolygonTest(
                zdef["polygon"], (float(cx), float(cy)), False
            ) >= 0

            if inside:
                if zone_id not in track_state:
                    # ZONE_ENTER
                    track_state[zone_id] = {
                        "entered_frame": frame_idx,
                        "dwell_emitted": False,
                    }
                    events.append({
                        "event_type": "ZONE_ENTER",
                        "track_id": track_id,
                        "zone_id": zone_id,
                        "zone_label": zdef["label"],
                        "frame": frame_idx,
                        "cx": cx,
                        "cy": cy,
                    })
                else:
                    # Check dwell threshold
                    state = track_state[zone_id]
                    dwell_frames = frame_idx - state["entered_frame"]
                    if (dwell_frames >= zdef["dwell_threshold_frames"]
                            and not state["dwell_emitted"]
                            and zdef["dwell_threshold_frames"] > 0):
                        state["dwell_emitted"] = True
                        dwell_sec = dwell_frames / self.fps
                        events.append({
                            "event_type": "ZONE_DWELL",
                            "track_id": track_id,
                            "zone_id": zone_id,
                            "zone_label": zdef["label"],
                            "frame": frame_idx,
                            "dwell_sec": round(dwell_sec, 1),
                            "cx": cx,
                            "cy": cy,
                        })
            else:
                if zone_id in track_state:
                    # ZONE_EXIT
                    state = track_state[zone_id]
                    dwell_frames = frame_idx - state["entered_frame"]
                    dwell_sec = dwell_frames / self.fps
                    events.append({
                        "event_type": "ZONE_EXIT",
                        "track_id": track_id,
                        "zone_id": zone_id,
                        "zone_label": zdef["label"],
                        "frame": frame_idx,
                        "dwell_sec": round(dwell_sec, 1),
                        "cx": cx,
                        "cy": cy,
                    })
                    del track_state[zone_id]

        return events

    def get_final_dwells(self) -> List[Dict]:
        """Get dwell times for tracks still in zones (end of video)."""
        results = []
        for track_id, zones in self._track_zones.items():
            for zone_id, state in zones.items():
                zdef = next((z for z in self.zone_defs if z["id"] == zone_id), None)
                if zdef:
                    results.append({
                        "track_id": track_id,
                        "zone_id": zone_id,
                        "zone_label": zdef["label"],
                        "entered_frame": state["entered_frame"],
                    })
        return results

    def cleanup_track(self, track_id: int):
        self._track_zones.pop(track_id, None)
