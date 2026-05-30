"""
Store Intelligence v2.0 — Tracklet Record
Data structure for storing per-track trajectory + appearance data.

Sources: Architect Report (structure), Kimi (EMA feature bank)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class TrackletRecord:
    """
    Represents a single tracked trajectory fragment from BoT-SORT.
    Stores positions, embeddings, and computed properties.
    """
    track_id: int
    camera_id: str
    start_frame: int
    end_frame: int

    # Per-frame data: (frame_idx, cx, cy, w, h)
    positions: List[Tuple[int, float, float, float, float]] = field(default_factory=list)

    # ReID embeddings sampled every N frames (not every frame — DeepSeek)
    embeddings: List[np.ndarray] = field(default_factory=list)

    # Staff probability (DeepSeek: prevents cross-merge)
    staff_prob: float = 0.0

    # Cached mean embedding (invalidated on update)
    _mean_embedding: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def last_position(self) -> Tuple[float, float]:
        if self.positions:
            _, cx, cy, _, _ = self.positions[-1]
            return cx, cy
        return 0.0, 0.0

    @property
    def first_position(self) -> Tuple[float, float]:
        if self.positions:
            _, cx, cy, _, _ = self.positions[0]
            return cx, cy
        return 0.0, 0.0

    @property
    def velocity(self) -> Tuple[float, float]:
        """Average velocity vector from start to end."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        _, sx, sy, _, _ = self.positions[0]
        _, ex, ey, _, _ = self.positions[-1]
        dt = max(self.duration_frames, 1)
        return ((ex - sx) / dt, (ey - sy) / dt)

    @property
    def mean_embedding(self) -> Optional[np.ndarray]:
        """
        Mean-pooled, L2-normalized embedding across all samples.
        Source: Kimi (EMA concept), DeepSeek (mean after L2 norm)
        """
        if self._mean_embedding is None and self.embeddings:
            stack = np.stack(self.embeddings, axis=0)
            mean = stack.mean(axis=0)
            norm = np.linalg.norm(mean)
            if norm > 1e-9:
                self._mean_embedding = mean / norm
            else:
                self._mean_embedding = mean
        return self._mean_embedding

    def add_frame(
        self,
        frame_idx: int,
        cx: float,
        cy: float,
        w: float,
        h: float,
        embedding: Optional[np.ndarray] = None,
    ):
        """Add a frame observation."""
        self.positions.append((frame_idx, cx, cy, w, h))
        if embedding is not None:
            self.embeddings.append(embedding)
            self._mean_embedding = None  # invalidate cache
        self.end_frame = frame_idx

    def merge_from(self, other: "TrackletRecord"):
        """Merge another tracklet's data into this one (child → parent)."""
        self.positions.extend(other.positions)
        self.positions.sort(key=lambda x: x[0])  # sort by frame
        self.embeddings.extend(other.embeddings)
        self.start_frame = min(self.start_frame, other.start_frame)
        self.end_frame = max(self.end_frame, other.end_frame)
        self._mean_embedding = None  # invalidate
