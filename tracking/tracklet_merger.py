"""
Store Intelligence v2.0 — Tracklet Merger (Post-Hoc Deduplication)
Merges fragmented tracklets using ReID embeddings + spatio-temporal constraints.

Sources:
- Architect Report: TrackletMerger structure, cost fusion
- DeepSeek: Union-Find for transitive closure, Gaussian displacement cost
- Kimi: Velocity compatibility, EMA feature bank
- Gemini: Spatio-temporal gating
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from tracking.tracklet_record import TrackletRecord


class UnionFind:
    """
    Disjoint-set data structure for transitive tracklet merging.
    Source: DeepSeek — if A matches B, and B matches C, all three → same person.
    """

    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class TrackletMerger:
    """
    Post-hoc tracklet deduplication via:
    1. Spatio-temporal candidate filtering (gating)
    2. Fused cost: appearance (cosine) + spatial + temporal + velocity
    3. Hungarian bipartite matching
    4. Union-Find transitive closure

    Designed for retail where:
    - People disappear behind shelves for seconds
    - Same person gets 5-10 IDs from BoT-SORT
    - Staff and customers should NOT be cross-merged
    """

    def __init__(
        self,
        fps: float = 25.0,
        max_gap_sec: float = 8.0,
        max_spatial_dist_px: float = 250.0,
        match_threshold: float = 0.45,
        alpha: float = 0.45,  # appearance weight
        beta: float = 0.25,   # temporal weight
        gamma: float = 0.20,  # spatial weight
        delta: float = 0.10,  # velocity compatibility weight
    ):
        self.max_gap_frames = int(max_gap_sec * fps)
        self.max_spatial_dist_px = max_spatial_dist_px
        self.match_threshold = match_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def merge(self, gallery: Dict[int, TrackletRecord]) -> Dict[int, TrackletRecord]:
        """
        Main entry point. Returns a new gallery with fragmented
        tracklets merged into canonical IDs.
        """
        if len(gallery) < 2:
            return gallery

        tracklets = sorted(gallery.values(), key=lambda t: t.start_frame)

        # Build Union-Find merge map
        uf = self._build_merge_map(tracklets)

        # Apply merges
        return self._apply_merges(gallery, uf)

    def _build_merge_map(self, tracklets: List[TrackletRecord]) -> UnionFind:
        """
        Iteratively find mergeable pairs using Hungarian matching
        within temporal windows, then union them.

        Two modes:
        1. SEQUENTIAL: Track B starts after Track A ends (gap > 0)
           → classic re-id merge for re-appearing persons
        2. CONCURRENT: Track A and B overlap in time but are spatially close
           → handles clustering/splitting where tracker creates duplicate IDs
        """
        uf = UnionFind()
        n = len(tracklets)

        # ── MODE 1: Sequential merge (original logic) ────────────────
        for i in range(n):
            child = tracklets[i]

            candidates = []
            for j in range(i):
                parent = tracklets[j]
                gap = child.start_frame - parent.end_frame
                if gap <= 0 or gap > self.max_gap_frames:
                    continue

                # Spatial gating
                px, py = parent.last_position
                cx, cy = child.first_position
                dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                if dist > self.max_spatial_dist_px:
                    continue

                # Staff/customer cross-merge prevention
                if abs(parent.staff_prob - child.staff_prob) > 0.5:
                    continue

                candidates.append(parent)

            if not candidates:
                continue

            costs = [self._compute_cost(parent, child) for parent in candidates]
            min_idx = int(np.argmin(costs))
            if costs[min_idx] < self.match_threshold:
                uf.union(candidates[min_idx].track_id, child.track_id)

        # ── MODE 2: Concurrent/overlapping merge ─────────────────────
        # Handles: same person getting 2 IDs when people cluster together
        # If two tracks overlap in time AND are very close spatially AND
        # look similar (appearance), they're likely the same person
        for i in range(n):
            for j in range(i + 1, n):
                t1 = tracklets[i]
                t2 = tracklets[j]

                # Already merged?
                if uf.find(t1.track_id) == uf.find(t2.track_id):
                    continue

                # Check temporal overlap: do they share any frames?
                overlap_start = max(t1.start_frame, t2.start_frame)
                overlap_end = min(t1.end_frame, t2.end_frame)
                if overlap_end <= overlap_start:
                    continue  # No temporal overlap

                # One must be SHORT (< 30% of the other) — it's a fragment
                len1 = t1.end_frame - t1.start_frame + 1
                len2 = t2.end_frame - t2.start_frame + 1
                shorter = min(len1, len2)
                longer = max(len1, len2)
                if shorter > 0.5 * longer:
                    continue  # Both are long tracks — likely different people

                # Spatial proximity during overlap period
                # Use mean positions of both during the overlap window
                pos1 = [p for p in t1.positions if overlap_start <= p[0] <= overlap_end]
                pos2 = [p for p in t2.positions if overlap_start <= p[0] <= overlap_end]
                if not pos1 or not pos2:
                    continue

                mean_x1 = np.mean([p[1] for p in pos1])
                mean_y1 = np.mean([p[2] for p in pos1])
                mean_x2 = np.mean([p[1] for p in pos2])
                mean_y2 = np.mean([p[2] for p in pos2])
                dist = np.sqrt((mean_x1 - mean_x2)**2 + (mean_y1 - mean_y2)**2)

                # Must be very close — within 150px (tight for concurrent)
                if dist > 150.0:
                    continue

                # Appearance check
                if t1.mean_embedding is not None and t2.mean_embedding is not None:
                    cos_sim = float(np.dot(t1.mean_embedding, t2.mean_embedding))
                    if cos_sim < 0.3:  # too different looking
                        continue

                # Staff cross-merge prevention
                if abs(t1.staff_prob - t2.staff_prob) > 0.5:
                    continue

                # Merge the shorter into the longer
                uf.union(t1.track_id, t2.track_id)

        return uf

    def _compute_cost(self, parent: TrackletRecord, child: TrackletRecord) -> float:
        """
        Fused cost combining 4 factors:
        C = α·appearance + β·temporal + γ·spatial + δ·velocity

        Range: [0, 1]. Lower = more likely same person.
        """
        # ── Appearance cost (cosine distance) ────────────────────────
        if parent.mean_embedding is not None and child.mean_embedding is not None:
            cos_sim = float(np.dot(parent.mean_embedding, child.mean_embedding))
            app_cost = (1.0 - np.clip(cos_sim, -1.0, 1.0)) / 2.0  # → [0, 1]
        else:
            app_cost = 0.5  # neutral if no embeddings

        # ── Temporal cost (gap normalized by max gap) ────────────────
        gap = child.start_frame - parent.end_frame
        temp_cost = float(gap) / float(self.max_gap_frames)  # → [0, 1]

        # ── Spatial cost (Gaussian, DeepSeek-inspired) ───────────────
        px, py = parent.last_position
        cx, cy = child.first_position
        dist = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        sigma = self.max_spatial_dist_px * 0.4
        spat_cost = 1.0 - np.exp(-dist ** 2 / (2 * sigma ** 2))

        # ── Velocity compatibility (Kimi) ────────────────────────────
        vp = parent.velocity
        vc = child.velocity
        vel_cost = self._velocity_cost(vp, vc)

        return (self.alpha * app_cost +
                self.beta * temp_cost +
                self.gamma * spat_cost +
                self.delta * vel_cost)

    def _velocity_cost(
        self,
        v1: Tuple[float, float],
        v2: Tuple[float, float],
    ) -> float:
        """Velocity incompatibility score. 0 = same direction, 1 = opposite."""
        n1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2)
        n2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if n1 < 1e-6 or n2 < 1e-6:
            return 0.3  # neutral if nearly stationary

        cos_sim = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
        return (1.0 - cos_sim) / 2.0  # → [0, 1]

    def _apply_merges(
        self,
        gallery: Dict[int, TrackletRecord],
        uf: UnionFind,
    ) -> Dict[int, TrackletRecord]:
        """
        Merge tracklets that belong to the same Union-Find component.
        The canonical (root) tracklet absorbs all children.
        """
        # Group by canonical ID
        groups: Dict[int, List[int]] = {}
        for track_id in gallery:
            root = uf.find(track_id)
            groups.setdefault(root, []).append(track_id)

        merged: Dict[int, TrackletRecord] = {}
        for canonical_id, member_ids in groups.items():
            # Sort members by start frame
            members = sorted(
                [gallery[tid] for tid in member_ids],
                key=lambda t: t.start_frame,
            )

            # First member becomes the base
            base = TrackletRecord(
                track_id=canonical_id,
                camera_id=members[0].camera_id,
                start_frame=members[0].start_frame,
                end_frame=members[0].end_frame,
                positions=list(members[0].positions),
                embeddings=list(members[0].embeddings),
                staff_prob=members[0].staff_prob,
            )

            # Merge remaining members
            for m in members[1:]:
                base.merge_from(m)

            merged[canonical_id] = base

        return merged
