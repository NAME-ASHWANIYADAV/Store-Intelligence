"""
Store Intelligence System - Zone Detection
Uses Supervision PolygonZone for zone-level analytics from store_layout.json.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import structlog

logger = structlog.get_logger("zones")


class ZoneManager:
    """
    Manages store zones defined in store_layout.json.
    Determines which zone a person's centroid falls into.
    """

    def __init__(self, store_layout_path: str):
        with open(store_layout_path) as f:
            self.layout = json.load(f)

        self.store_zones: Dict[str, Dict[str, np.ndarray]] = {}
        self._parse_zones()

        logger.info("zones_loaded", store_count=len(self.store_zones))

    def _parse_zones(self):
        """Parse zone polygons from layout JSON."""
        for store_id, store_data in self.layout.get("stores", {}).items():
            self.store_zones[store_id] = {}
            for zone_id, zone_data in store_data.get("zones", {}).items():
                polygon = np.array(zone_data["polygon"], dtype=np.float32)
                self.store_zones[store_id][zone_id] = polygon

            logger.info(
                "store_zones_parsed",
                store_id=store_id,
                zone_count=len(self.store_zones[store_id]),
                zones=list(self.store_zones[store_id].keys()),
            )

    def get_zone_for_point(
        self, store_id: str, camera_id: str, point: Tuple[float, float]
    ) -> Optional[str]:
        """
        Determine which zone a point (centroid) falls into.
        Uses cv2.pointPolygonTest for accurate polygon containment.
        """
        import cv2

        if store_id not in self.store_zones:
            return None

        store = self.layout["stores"][store_id]

        for zone_id, polygon in self.store_zones[store_id].items():
            # Check if this zone belongs to the current camera
            zone_data = store["zones"][zone_id]
            if zone_data.get("camera_id") != camera_id:
                continue

            # Point-in-polygon test
            result = cv2.pointPolygonTest(
                polygon.reshape(-1, 1, 2).astype(np.float32),
                (float(point[0]), float(point[1])),
                False,  # We don't need distance, just inside/outside
            )
            if result >= 0:  # Inside or on edge
                return zone_id

        return None

    def get_store_ids(self) -> List[str]:
        """Get all store IDs from layout."""
        return list(self.layout.get("stores", {}).keys())

    def get_cameras_for_store(self, store_id: str) -> Dict:
        """Get camera configurations for a store."""
        store = self.layout.get("stores", {}).get(store_id, {})
        return store.get("cameras", {})

    def get_threshold_line(self, store_id: str, camera_id: str) -> Optional[Dict]:
        """Get entry/exit threshold line for a camera."""
        cameras = self.get_cameras_for_store(store_id)
        camera = cameras.get(camera_id, {})
        return camera.get("threshold_line")

    def get_camera_type(self, store_id: str, camera_id: str) -> Optional[str]:
        """Get camera type (entry, floor, billing)."""
        cameras = self.get_cameras_for_store(store_id)
        camera = cameras.get(camera_id, {})
        return camera.get("type")
