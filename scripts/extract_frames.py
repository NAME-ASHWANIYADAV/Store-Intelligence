"""
Quick visual analysis: extract 1 frame from each video to understand camera placement.
"""
import sys, os, cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(r"c:\Users\HP\OneDrive\Desktop\purplle\data\CCTV Footage")
OUT_DIR = Path(__file__).parent.parent / "output" / "frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for vid in sorted(DATA_DIR.glob("*.mp4")):
    cap = cv2.VideoCapture(str(vid))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Extract frame at 50% mark
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if ret:
        out_path = OUT_DIR / f"{vid.stem}_mid.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"{vid.name}: {w}x{h} @ {fps:.1f}fps, {total} frames ({total/fps:.0f}s) -> {out_path}")
    cap.release()
