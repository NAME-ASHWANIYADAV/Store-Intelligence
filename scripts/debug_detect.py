"""Debug: check what YOLO returns for CAM 4 — raw boxes vs tracked boxes."""
import sys
sys.path.insert(0, r"c:\Users\HP\OneDrive\Desktop\purplle\store-intelligence")

from ultralytics import YOLO

model = YOLO("yolo11l.pt")
video = r"c:\Users\HP\OneDrive\Desktop\purplle\data\CCTV Footage\CAM 4.mp4"

# Test 1: Simple predict (no tracking) — just check detections exist
print("=== TEST 1: Predict (no tracking) ===")
results = model.predict(source=video, classes=[0], conf=0.25, vid_stride=30, stream=True, verbose=False)
total_dets = 0
for i, r in enumerate(results):
    if r.boxes is not None and len(r.boxes) > 0:
        total_dets += len(r.boxes)
    if i >= 10:
        break
print(f"Detections in first 10 sampled frames: {total_dets}")

# Test 2: Track with built-in botsort (no custom config)
print("\n=== TEST 2: Track (built-in botsort) ===")
results2 = model.track(source=video, tracker="botsort.yaml", persist=True, classes=[0], conf=0.25, vid_stride=30, stream=True, verbose=False)
total_tracked = 0
track_ids_seen = set()
for i, r in enumerate(results2):
    if r.boxes is not None:
        has_id = r.boxes.id is not None
        n_boxes = len(r.boxes)
        if has_id:
            ids = r.boxes.id.cpu().numpy().astype(int).tolist()
            track_ids_seen.update(ids)
            total_tracked += len(ids)
        print(f"  Frame {i}: {n_boxes} boxes, has_id={has_id}, ids={ids if has_id else 'None'}")
    if i >= 10:
        break
print(f"Tracked detections: {total_tracked}, unique IDs: {track_ids_seen}")
