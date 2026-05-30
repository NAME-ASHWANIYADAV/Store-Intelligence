"""Quick test: run detection on CAM 4 (smallest file) to verify pipeline works."""
import structlog
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

from pipeline.detect import PersonDetector

detector = PersonDetector(
    model_path="yolo11l.pt",
    tracker_config="botsort.yaml",
    confidence=0.25,
    vid_stride=3,
)

video_path = r"c:\Users\HP\OneDrive\Desktop\purplle\data\CCTV Footage\CAM 4.mp4"
results = detector.process_video(video_path, "CAM_ENTRY_02")
tracks = detector.get_track_summary(results)

print(f"\n=== RESULTS ===")
print(f"Frames processed: {len(results)}")
print(f"Unique tracks: {len(tracks)}")
for tid, s in list(tracks.items())[:10]:
    avg_conf = sum(s["confidences"]) / len(s["confidences"])
    presence = s["frame_count"] / max(len(results), 1)
    print(f"  Track {tid}: {s['frame_count']} frames ({presence:.1%}), avg_conf={avg_conf:.3f}")
