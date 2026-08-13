from pathlib import Path
import json

import mujoco
from PIL import Image


root = Path(__file__).resolve().parent
metrics = json.loads((root / "outputs" / "metrics.json").read_text(encoding="utf-8"))
frames = sorted((root / "outputs" / "official_viewer_frames").glob("*.png"))
model = mujoco.MjModel.from_xml_path(str(root / "scene_seed123.xml"))

assert len(metrics) == 6, "Expected six method records"
assert len(frames) == 6, "Expected six official MuJoCo Viewer captures"
frame_sizes = {Image.open(frame).size for frame in frames}
assert len(frame_sizes) == 1, "Viewer captures must share one resolution"
for record in metrics:
    safe = record["method"].lower().replace(" ", "_").replace("-", "_")
    assert (root / "outputs" / f"{safe}_trajectories.json").is_file()

print(f"methods={[record['method'] for record in metrics]}")
print(f"official_viewer_frames={len(frames)}, resolution={frame_sizes.pop()}")
print(f"mujoco_bodies={model.nbody}")
print(f"metrics={metrics}")
