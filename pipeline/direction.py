"""
Store Intelligence System - Entry/Exit Direction Detection
Determines if a person is entering or exiting based on centroid trajectory and threshold line.
"""

from typing import Dict, List, Tuple, Optional
from collections import deque

import structlog

logger = structlog.get_logger("direction")


class DirectionDetector:
    """
    Detects ENTRY/EXIT by tracking centroid trajectory across a threshold line.
    Uses a deque of recent centroids for smoothing (avoids jitter false positives).
    """

    def __init__(self):
        # Per-track centroid history: {track_id: deque([(cx, cy), ...])}
        self.track_history: Dict[int, deque] = {}
        self.history_size = 8  # Keep last 8 centroids for direction analysis
        self.min_frames_for_direction = 3  # Need at least 3 frames to determine direction

        # Track which side of line each track was last on
        self.track_side: Dict[int, Optional[str]] = {}  # "above" or "below"

        # Already triggered events
        self.triggered: Dict[int, str] = {}  # track_id -> "ENTRY" or "EXIT"

    def update(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        threshold_line: Dict,
    ) -> Optional[str]:
        """
        Update a track's position and check if it crossed the threshold line.

        Args:
            track_id: Unique tracking ID
            centroid: (cx, cy) current position
            threshold_line: {"start": [x1,y1], "end": [x2,y2], "in_direction": "top_to_bottom"}

        Returns:
            "ENTRY", "EXIT", or None
        """
        # Initialize history for new tracks
        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.history_size)
            self.track_side[track_id] = None

        self.track_history[track_id].append(centroid)

        # Need enough history for direction determination
        if len(self.track_history[track_id]) < self.min_frames_for_direction:
            return None

        # Don't trigger multiple times for the same track
        if track_id in self.triggered:
            return None

        # Determine which side of the line the centroid is on
        line_y = threshold_line["start"][1]  # Horizontal line Y coordinate
        current_side = "above" if centroid[1] < line_y else "below"

        # Check if previous side was recorded
        prev_side = self.track_side[track_id]
        self.track_side[track_id] = current_side

        if prev_side is None:
            return None

        # Detect crossing
        if prev_side != current_side:
            in_direction = threshold_line.get("in_direction", "top_to_bottom")

            # Verify direction consistency with recent history
            history = list(self.track_history[track_id])
            y_values = [p[1] for p in history]

            if in_direction == "top_to_bottom":
                # ENTRY: moving from above to below (top to bottom)
                if prev_side == "above" and current_side == "below":
                    # Verify: majority of recent y-values should be increasing
                    if self._is_consistent_direction(y_values, increasing=True):
                        self.triggered[track_id] = "ENTRY"
                        return "ENTRY"
                # EXIT: moving from below to above (bottom to top)
                elif prev_side == "below" and current_side == "above":
                    if self._is_consistent_direction(y_values, increasing=False):
                        self.triggered[track_id] = "EXIT"
                        return "EXIT"
            else:
                # Reversed direction
                if prev_side == "below" and current_side == "above":
                    if self._is_consistent_direction(y_values, increasing=False):
                        self.triggered[track_id] = "ENTRY"
                        return "ENTRY"
                elif prev_side == "above" and current_side == "below":
                    if self._is_consistent_direction(y_values, increasing=True):
                        self.triggered[track_id] = "EXIT"
                        return "EXIT"

        return None

    def _is_consistent_direction(self, y_values: List[float], increasing: bool) -> bool:
        """Check if the trajectory is consistently moving in one direction."""
        if len(y_values) < 3:
            return True

        # Count consistent direction changes
        consistent = 0
        total = 0
        for i in range(1, len(y_values)):
            diff = y_values[i] - y_values[i - 1]
            if abs(diff) > 2:  # Ignore tiny movements
                total += 1
                if (increasing and diff > 0) or (not increasing and diff < 0):
                    consistent += 1

        # At least 60% of movements should be in the expected direction
        return (consistent / total >= 0.6) if total > 0 else True

    def reset_track(self, track_id: int):
        """Reset tracking state for a track (e.g., for re-entry detection)."""
        self.track_history.pop(track_id, None)
        self.track_side.pop(track_id, None)
        self.triggered.pop(track_id, None)
