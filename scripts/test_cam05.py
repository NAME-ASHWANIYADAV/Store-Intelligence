import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.models import StoreConfig
from tracking.reid_extractor import ReIDExtractor
from scripts.run_pipeline_v3 import load_yolo_model, process_camera, DATA_DIR, OUTPUT_DIR

CONFIG_PATH = PROJECT_ROOT / "configs" / "store_001.yaml"

def main():
    print("=" * 60)
    print("  STORE INTELLIGENCE v3.0 -- CAM_05 TEST")
    print("=" * 60)

    config = StoreConfig.from_yaml(CONFIG_PATH)
    
    # Filter config to only have CAM_05
    config.cameras = [cam for cam in config.cameras if cam.id == "CAM_05"]
    
    if not config.cameras:
        print("[ERR] CAM_05 not found in config")
        return

    model = load_yolo_model()
    reid = ReIDExtractor(device="cuda")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_t0 = time.time()
    
    cam_config = config.cameras[0]
    result = process_camera(model, reid, cam_config, config)
    
    stats = result["stats"]
    events = result["events"]

    total_elapsed = time.time() - total_t0

    # Write events
    events_file = OUTPUT_DIR / f"{config.store_id}_v3_CAM_05.jsonl"
    with open(events_file, "w") as f:
        for e in events:
            e["store_id"] = config.store_id
            f.write(json.dumps(e, default=str) + "\n")

    print(f"\n{'='*60}")
    print(f"  CAM_05 RESULTS")
    print(f"{'='*60}")
    print(f"Raw Tracks:    {stats['raw_tracks']}")
    print(f"Merged Tracks: {stats['merged_tracks']}")
    print(f"Staff:         {stats['staff']}")
    print(f"Customers:     {stats['customers']}")
    print(f"Events:        {stats['events']}")
    print(f"Total time:    {total_elapsed:.1f}s")
    
    if stats['raw_tracks'] > 0:
        reduction = (1 - stats['merged_tracks']/stats['raw_tracks'])*100
        print(f"Dedup Reduction: {reduction:.1f}%")

if __name__ == "__main__":
    main()
