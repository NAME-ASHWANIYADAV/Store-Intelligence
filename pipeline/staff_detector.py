"""
Store Intelligence System - Staff Detection
Three-tier staff classification: dwell heuristic → VLM (Grok Vision) → fallback.
"""

import base64
import io
import cv2
import numpy as np
import requests
from typing import Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger("staff_detector")


class StaffDetector:
    """
    Three-tier staff detection:
    1. Dwell heuristic: person present in >60% of video frames → likely staff
    2. VLM (Grok Vision): crop + ask "uniform?" on sampled frames → majority vote
    3. Fallback: if no VLM available, use only dwell heuristic
    """

    def __init__(
        self,
        dwell_threshold: float = 0.6,
        xai_api_key: str = "",
        xai_api_url: str = "https://api.x.ai/v1/chat/completions",
        xai_model: str = "grok-2-vision-latest",
    ):
        self.dwell_threshold = dwell_threshold
        self.xai_api_key = xai_api_key
        self.xai_api_url = xai_api_url
        self.xai_model = xai_model
        self.vlm_available = bool(xai_api_key)

        logger.info(
            "staff_detector_initialized",
            dwell_threshold=dwell_threshold,
            vlm_available=self.vlm_available,
        )

    def detect_staff_by_dwell(
        self, track_summary: Dict[int, Dict], total_frames: int
    ) -> Set[int]:
        """
        Tier 1: Identify staff by dwell time.
        Staff typically stays for >60% of the video duration.
        """
        staff_tracks = set()

        for track_id, summary in track_summary.items():
            presence_ratio = summary["frame_count"] / max(total_frames, 1)
            if presence_ratio >= self.dwell_threshold:
                staff_tracks.add(track_id)
                logger.info(
                    "staff_detected_by_dwell",
                    track_id=track_id,
                    presence_ratio=round(presence_ratio, 3),
                )

        return staff_tracks

    def detect_staff_by_vlm(
        self,
        video_path: str,
        track_summary: Dict[int, Dict],
        candidate_tracks: Set[int],
        num_samples: int = 3,
    ) -> Set[int]:
        """
        Tier 2: Use Grok Vision to classify uniform/badge.
        Samples 3 frames per track at equal intervals, majority vote.
        """
        if not self.vlm_available:
            logger.warning("vlm_not_available", reason="no_api_key")
            return set()

        staff_tracks = set()
        cap = cv2.VideoCapture(video_path)

        for track_id in candidate_tracks:
            summary = track_summary.get(track_id)
            if not summary or not summary["bboxes"]:
                continue

            # Sample frames at equal intervals
            total_bboxes = len(summary["bboxes"])
            sample_indices = [
                int(i * total_bboxes / (num_samples + 1))
                for i in range(1, num_samples + 1)
            ]

            votes = []
            for idx in sample_indices:
                if idx >= len(summary["bboxes"]):
                    continue

                bbox = summary["bboxes"][idx]
                # We need to get the actual frame - use first_frame + idx * stride
                frame_num = summary["first_frame"] + idx
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Crop person from frame
                x1, y1, x2, y2 = [int(v) for v in bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Ask VLM
                is_staff = self._ask_vlm_is_staff(crop)
                votes.append(is_staff)

            # Majority vote
            if votes and sum(votes) > len(votes) / 2:
                staff_tracks.add(track_id)
                logger.info(
                    "staff_detected_by_vlm",
                    track_id=track_id,
                    votes=votes,
                )

        cap.release()
        return staff_tracks

    def _ask_vlm_is_staff(self, crop: np.ndarray) -> bool:
        """Ask Grok Vision API if the cropped person is wearing a uniform."""
        try:
            # Encode crop to base64 JPEG
            _, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            payload = {
                "model": self.xai_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Is this person wearing a retail store uniform, "
                                    "name badge, or employee attire? "
                                    "Answer ONLY 'YES' or 'NO'. Nothing else."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 5,
                "temperature": 0.1,
            }

            headers = {
                "Authorization": f"Bearer {self.xai_api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                self.xai_api_url,
                json=payload,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip().upper()
                return answer.startswith("YES")
            else:
                logger.warning(
                    "vlm_api_error",
                    status_code=response.status_code,
                    response=response.text[:200],
                )
                return False

        except Exception as e:
            logger.warning("vlm_request_failed", error=str(e))
            return False

    def classify_tracks(
        self,
        track_summary: Dict[int, Dict],
        total_frames: int,
        video_path: str = "",
    ) -> Set[int]:
        """
        Full three-tier staff classification.
        Returns set of track_ids classified as staff.
        """
        # Tier 1: Dwell heuristic (always runs)
        staff_by_dwell = self.detect_staff_by_dwell(track_summary, total_frames)

        # Find ambiguous tracks (present but not clear staff by dwell alone)
        ambiguous = set()
        for track_id, summary in track_summary.items():
            presence = summary["frame_count"] / max(total_frames, 1)
            if 0.4 <= presence < self.dwell_threshold:
                ambiguous.add(track_id)

        # Tier 2: VLM on ambiguous tracks (if available)
        staff_by_vlm = set()
        if ambiguous and self.vlm_available and video_path:
            staff_by_vlm = self.detect_staff_by_vlm(
                video_path, track_summary, ambiguous
            )

        all_staff = staff_by_dwell | staff_by_vlm

        logger.info(
            "staff_classification_complete",
            staff_by_dwell=len(staff_by_dwell),
            staff_by_vlm=len(staff_by_vlm),
            ambiguous_tracks=len(ambiguous),
            total_staff=len(all_staff),
        )

        return all_staff
