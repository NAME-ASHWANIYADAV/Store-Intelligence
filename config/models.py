"""
Store Intelligence System v2.0 — Configuration Models
Dataclass-based config loader from YAML.
Supports: camera roles, ignore regions, tripwire lines, zones, staff zones.
Uses normalized coordinates (0.0-1.0) for resolution independence.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Literal, Dict
import numpy as np
import cv2
import yaml
from pathlib import Path

Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class IgnoreRegion:
    """Polygonal region where detections are dropped before tracking."""
    label: str
    polygon: Polygon  # normalized coords (0.0-1.0)

    def to_pixel_polygon(self, w: int, h: int) -> np.ndarray:
        """Convert normalized polygon to pixel coordinates."""
        return np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in self.polygon],
            dtype=np.int32
        )


@dataclass
class TripwireLine:
    """Directed line for entry/exit counting via cross-product FSM."""
    point_a: Point  # normalized (outside end)
    point_b: Point  # normalized (inside end)
    inside_direction: Literal["below", "above", "left", "right"] = "below"
    min_crossing_frames: int = 2
    cooldown_sec: float = 2.0
    min_pixel_distance: float = 50.0

    def to_pixel_points(self, w: int, h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        pa = (int(self.point_a[0] * w), int(self.point_a[1] * h))
        pb = (int(self.point_b[0] * w), int(self.point_b[1] * h))
        return pa, pb


@dataclass
class Zone:
    """Polygonal zone for dwell-time analytics."""
    id: str
    label: str
    polygon: Polygon  # normalized coords
    dwell_threshold_sec: float = 3.0

    def to_pixel_polygon(self, w: int, h: int) -> np.ndarray:
        return np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in self.polygon],
            dtype=np.int32
        )


@dataclass
class StaffZone:
    """Zone where >threshold% presence = staff classification."""
    polygon: Polygon
    threshold: float = 0.30

    def to_pixel_polygon(self, w: int, h: int) -> np.ndarray:
        return np.array(
            [[int(p[0] * w), int(p[1] * h)] for p in self.polygon],
            dtype=np.int32
        )


@dataclass
class TrackerConfig:
    """Per-camera tracker tuning parameters."""
    track_buffer: int = 90
    match_thresh: float = 0.8
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    reid_enabled: bool = True
    appearance_thresh: float = 0.25


@dataclass
class CameraConfig:
    """Complete configuration for a single camera."""
    id: str
    label: str
    source: str
    role: Literal["entry", "floor", "billing"]
    ignore_regions: List[IgnoreRegion] = field(default_factory=list)
    zones: List[Zone] = field(default_factory=list)
    tripwire: Optional[TripwireLine] = None
    staff_zone: Optional[StaffZone] = None
    tracker_config: TrackerConfig = field(default_factory=TrackerConfig)


@dataclass
class StoreConfig:
    """Top-level store configuration with all cameras."""
    store_id: str
    fps_default: float
    cameras: List[CameraConfig]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StoreConfig":
        """Load and parse YAML camera manifest."""
        with open(path) as f:
            data = yaml.safe_load(f)

        cameras = []
        for cam_data in data["cameras"]:
            # Parse ignore regions
            irs = [
                IgnoreRegion(label=r["label"], polygon=[tuple(p) for p in r["polygon"]])
                for r in cam_data.get("ignore_regions", [])
            ]

            # Parse zones
            zones = [
                Zone(
                    id=z["id"], label=z["label"],
                    polygon=[tuple(p) for p in z["polygon"]],
                    dwell_threshold_sec=z.get("dwell_threshold_sec", 3.0),
                )
                for z in cam_data.get("zones", [])
            ]

            # Parse tripwire
            tw_data = cam_data.get("tripwire")
            tw = None
            if tw_data:
                tw = TripwireLine(
                    point_a=tuple(tw_data["point_a"]),
                    point_b=tuple(tw_data["point_b"]),
                    inside_direction=tw_data.get("inside_direction", "below"),
                    min_crossing_frames=tw_data.get("min_crossing_frames", 2),
                    cooldown_sec=tw_data.get("cooldown_sec", 2.0),
                    min_pixel_distance=tw_data.get("min_pixel_distance", 50.0),
                )

            # Parse staff zone
            sz_data = cam_data.get("staff_zone")
            sz = None
            if sz_data:
                sz = StaffZone(
                    polygon=[tuple(p) for p in sz_data["polygon"]],
                    threshold=sz_data.get("threshold", 0.30),
                )

            # Parse tracker config
            tc_data = cam_data.get("tracker_config", {})
            tc = TrackerConfig(**tc_data)

            cameras.append(CameraConfig(
                id=cam_data["id"],
                label=cam_data["label"],
                source=cam_data["source"],
                role=cam_data["role"],
                ignore_regions=irs,
                zones=zones,
                tripwire=tw,
                staff_zone=sz,
                tracker_config=tc,
            ))

        return cls(
            store_id=data["store_id"],
            fps_default=data.get("fps_default", 25.0),
            cameras=cameras,
        )

    def get_camera(self, cam_id: str) -> Optional[CameraConfig]:
        for cam in self.cameras:
            if cam.id == cam_id:
                return cam
        return None

    def get_cameras_by_role(self, role: str) -> List[CameraConfig]:
        return [c for c in self.cameras if c.role == role]
