"""
Store Intelligence v2.0 — Spatial Ignore Region Filter
Drops detections whose foot-point falls inside ignore regions BEFORE tracking.
Uses binary mask rasterization for O(1) lookup per detection.

Sources: DeepSeek (foot-point), Gemini (binary mask), Architect (cv2.fillPoly)
"""

import numpy as np
import cv2
from typing import List
from config.models import CameraConfig


class SpatialFilter:
    """
    Pre-computes a binary mask from ignore region polygons.
    At runtime, checks each detection's foot-point against the mask.
    If foot-point falls in masked area (0), detection is dropped.
    """

    def __init__(self, cam_config: CameraConfig, frame_width: int, frame_height: int):
        self.cam_id = cam_config.id
        self.w = frame_width
        self.h = frame_height

        # Build binary mask: 255 = valid, 0 = ignore
        self.mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255

        for region in cam_config.ignore_regions:
            pts = region.to_pixel_polygon(frame_width, frame_height)
            cv2.fillPoly(self.mask, [pts], 0)

        n_ignored = np.sum(self.mask == 0)
        total = frame_width * frame_height
        pct = (n_ignored / total) * 100
        print(f"  [{self.cam_id}] SpatialFilter: {len(cam_config.ignore_regions)} ignore regions, "
              f"{pct:.1f}% of frame masked")

    def filter_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        Filter detections by checking foot-point (bottom-center of bbox).

        Args:
            detections: shape (N, 6+) — [x1, y1, x2, y2, conf, cls, ...]

        Returns:
            Filtered detections (only those with foot-point in valid area)
        """
        if len(detections) == 0 or len(self.mask) == 0:
            return detections

        # Compute foot-point: bottom-center of bbox (more accurate floor position)
        foot_x = ((detections[:, 0] + detections[:, 2]) / 2).astype(int)
        foot_y = detections[:, 3].astype(int)  # y2 = bottom edge

        # Clip to frame bounds
        foot_x = np.clip(foot_x, 0, self.w - 1)
        foot_y = np.clip(foot_y, 0, self.h - 1)

        # O(1) mask lookup per detection
        keep = self.mask[foot_y, foot_x] > 0

        n_dropped = np.sum(~keep)
        if n_dropped > 0:
            pass  # silent in production; uncomment for debug:
            # print(f"  [{self.cam_id}] Dropped {n_dropped}/{len(detections)} detections in ignore region")

        return detections[keep]

    def draw_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw ignore regions as semi-transparent red overlay for visual debug."""
        overlay = frame.copy()
        # Red tint on ignore regions
        red_mask = np.zeros_like(frame)
        red_mask[:, :, 2] = 255 - self.mask  # Red channel where mask is 0
        overlay = cv2.addWeighted(overlay, 0.7, red_mask, 0.3, 0)

        # Draw polygon outlines
        # (Need access to original polygons for this, so store them)
        return overlay

    @property
    def has_ignore_regions(self) -> bool:
        return np.any(self.mask == 0)
