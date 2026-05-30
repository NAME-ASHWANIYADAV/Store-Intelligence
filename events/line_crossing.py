"""
Store Intelligence v2.0 — Directional Line Crossing Detector
Uses vector cross-product with multi-frame FSM + segment confinement.

Sources:
- Architect Report: FSM with multi-frame confirmation
- DeepSeek: Segment confinement check (point_within_segment)
- Gemini: Debounce registry with hysteresis
- ChatGPT: Cross-product sign math
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal
from config.models import TripwireLine

Side = Literal["INSIDE", "OUTSIDE", "UNKNOWN"]


@dataclass
class TrackCrossingState:
    """Per-track FSM state for line crossing detection."""
    track_id: int
    current_side: Side = "UNKNOWN"
    candidate_side: Optional[Side] = None
    candidate_frame_count: int = 0
    crossed: bool = False
    last_cross_frame: int = -9999
    last_cross_position: Tuple[float, float] = (0.0, 0.0)


class DirectionalLineCrossing:
    """
    Mathematically rigorous directional line crossing detector.

    Architecture:
    1. Cross-product determines which side of the line a centroid is on
    2. FSM requires N consecutive frames on the new side before confirming
    3. Segment confinement ensures crossing happens near the actual door
    4. Cooldown + distance hysteresis prevents jitter double-counting

    The line is defined as a directed vector from A to B.
    'inside_is_positive' determines which side of the line is "inside".
    """

    def __init__(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        inside_is_positive: bool = True,
        min_crossing_frames: int = 2,
        cooldown_frames: int = 50,
        min_rearm_distance: float = 50.0,
    ):
        self.ax, self.ay = float(point_a[0]), float(point_a[1])
        self.bx, self.by = float(point_b[0]), float(point_b[1])
        self.inside_is_positive = inside_is_positive
        self.min_crossing_frames = min_crossing_frames
        self.cooldown_frames = cooldown_frames
        self.min_rearm_distance = min_rearm_distance
        self._states: Dict[int, TrackCrossingState] = {}

        # Precompute line vector for segment confinement check
        self._line_vec = np.array([self.bx - self.ax, self.by - self.ay])
        self._line_len_sq = np.dot(self._line_vec, self._line_vec)

        # Counters
        self.entry_count = 0
        self.exit_count = 0

    @classmethod
    def from_config(cls, tripwire: TripwireLine, frame_w: int, frame_h: int) -> "DirectionalLineCrossing":
        """Build from TripwireLine config with pixel coordinate conversion."""
        pa_px, pb_px = tripwire.to_pixel_points(frame_w, frame_h)

        # Map inside_direction to cross-product sign convention
        inside_dir = tripwire.inside_direction
        # For a vertical line (top to bottom):
        #   "left" of the vector = positive cross_z
        #   "right" of the vector = negative cross_z
        # For a horizontal line (left to right):
        #   "below" = negative cross_z, "above" = positive cross_z
        inside_is_positive = inside_dir in ("left", "above")

        fps = 25  # approximate
        cooldown_frames = int(tripwire.cooldown_sec * fps)

        return cls(
            point_a=pa_px,
            point_b=pb_px,
            inside_is_positive=inside_is_positive,
            min_crossing_frames=tripwire.min_crossing_frames,
            cooldown_frames=cooldown_frames,
            min_rearm_distance=tripwire.min_pixel_distance,
        )

    def _cross_z(self, px: float, py: float) -> float:
        """
        2D cross product of vectors AB and AP.
        Determines which side of line AB the point P is on.

        cross_z > 0 → P is to the LEFT of vector AB
        cross_z < 0 → P is to the RIGHT of vector AB
        cross_z = 0 → P is exactly on the line
        """
        ab_x = self.bx - self.ax
        ab_y = self.by - self.ay
        ap_x = px - self.ax
        ap_y = py - self.ay
        return ab_x * ap_y - ab_y * ap_x

    def _classify_side(self, px: float, py: float) -> Side:
        """Classify which side of the line this point is on."""
        z = self._cross_z(px, py)
        if abs(z) < 1e-3:
            return "UNKNOWN"  # exactly on the line

        positive_side = z > 0
        if positive_side == self.inside_is_positive:
            return "INSIDE"
        return "OUTSIDE"

    def _point_within_segment(self, px: float, py: float) -> bool:
        """
        Check if the perpendicular projection of point P onto line AB
        falls between A and B (segment confinement).
        Prevents counting crossings far from the actual door.

        Source: DeepSeek — critical edge case others missed.
        """
        if self._line_len_sq < 1e-9:
            return False

        w = np.array([px - self.ax, py - self.ay])
        t = np.dot(w, self._line_vec) / self._line_len_sq

        # Allow some margin beyond segment endpoints (10%)
        return -0.1 <= t <= 1.1

    def _is_in_cooldown(self, state: TrackCrossingState, frame_idx: int, cx: float, cy: float) -> bool:
        """
        Check if this track is in cooldown (debounce).
        Track must either wait N frames OR move far enough from last crossing.
        Source: Gemini — hysteresis loop.
        """
        if not state.crossed:
            return False

        frames_since = frame_idx - state.last_cross_frame
        if frames_since < self.cooldown_frames:
            # Also check distance
            dx = cx - state.last_cross_position[0]
            dy = cy - state.last_cross_position[1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < self.min_rearm_distance:
                return True

        # Cooldown expired or moved far enough → re-arm
        state.crossed = False
        return False

    def update(
        self,
        track_id: int,
        cx: float,
        cy: float,
        frame_idx: int,
    ) -> Optional[Dict]:
        """
        Update FSM state for a tracked person's centroid.

        Returns:
            Dict with {track_id, event_type, direction, frame, cx, cy}
            or None if no crossing event.
        """
        new_side = self._classify_side(cx, cy)

        # First detection — just record initial side
        if track_id not in self._states:
            self._states[track_id] = TrackCrossingState(
                track_id=track_id,
                current_side=new_side if new_side != "UNKNOWN" else "OUTSIDE",
            )
            return None

        state = self._states[track_id]

        if new_side == "UNKNOWN":
            return None  # on the line — hold state

        if state.current_side == "UNKNOWN":
            state.current_side = new_side
            return None

        # Check cooldown/debounce
        if self._is_in_cooldown(state, frame_idx, cx, cy):
            return None

        # ── Side changed — begin multi-frame confirmation (FSM) ──────
        if new_side != state.current_side:
            if state.candidate_side == new_side:
                state.candidate_frame_count += 1
            else:
                # New candidate
                state.candidate_side = new_side
                state.candidate_frame_count = 1

            if state.candidate_frame_count >= self.min_crossing_frames:
                # ── Check segment confinement (DeepSeek) ──────────────
                if not self._point_within_segment(cx, cy):
                    # Crossing happened far from the door — reject
                    state.candidate_side = None
                    state.candidate_frame_count = 0
                    return None

                # ── CONFIRMED CROSSING ────────────────────────────────
                old_side = state.current_side
                state.current_side = new_side
                state.candidate_side = None
                state.candidate_frame_count = 0
                state.crossed = True
                state.last_cross_frame = frame_idx
                state.last_cross_position = (cx, cy)

                event_type = "ENTRY" if new_side == "INSIDE" else "EXIT"
                if event_type == "ENTRY":
                    self.entry_count += 1
                else:
                    self.exit_count += 1

                return {
                    "track_id": track_id,
                    "event_type": event_type,
                    "direction": f"{old_side}→{new_side}",
                    "frame": frame_idx,
                    "cx": cx,
                    "cy": cy,
                }
        else:
            # Back on same side — reset candidate (false alarm / jitter)
            state.candidate_side = None
            state.candidate_frame_count = 0

        return None

    def cleanup_track(self, track_id: int):
        """Remove state for a permanently lost track."""
        self._states.pop(track_id, None)

    def get_counts(self) -> Dict[str, int]:
        return {"entries": self.entry_count, "exits": self.exit_count}

    def draw_line(self, frame: np.ndarray) -> np.ndarray:
        """Draw tripwire line with direction arrow on frame."""
        import cv2

        ax, ay = int(self.ax), int(self.ay)
        bx, by = int(self.bx), int(self.by)

        # Draw the line (cyan)
        cv2.line(frame, (ax, ay), (bx, by), (255, 255, 0), 2)

        # Draw normal arrow indicating INSIDE direction
        mid_x, mid_y = (ax + bx) // 2, (ay + by) // 2
        ab = np.array([bx - ax, by - ay], dtype=float)
        # Perpendicular: rotate 90 degrees
        normal = np.array([-ab[1], ab[0]])
        if not self.inside_is_positive:
            normal = -normal
        normal_len = np.linalg.norm(normal)
        if normal_len > 0:
            normal = normal / normal_len * 40
        arrow_end = (int(mid_x + normal[0]), int(mid_y + normal[1]))
        cv2.arrowedLine(frame, (mid_x, mid_y), arrow_end, (0, 100, 255), 2)
        cv2.putText(frame, "INSIDE", arrow_end,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # Show counts
        cv2.putText(frame, f"IN: {self.entry_count}  OUT: {self.exit_count}",
                    (ax, ay - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
