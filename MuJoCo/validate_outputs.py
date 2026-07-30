from pathlib import Path
import json

import mujoco
from PIL import Image


root = Path(__file__).resolve().parent
metrics = json.loads((root / "outputs" / "metrics.json").read_text(encoding="utf-8"))
frames = sorted((root / "outputs" / "final_frames").glob("*.png"))
composite = Image.open(root / "outputs" / "mujoco_six_trajectories_1x6.png")
model = mujoco.MjModel.from_xml_path(str(root / "scene_seed123.xml"))

assert len(metrics) == 6, "Expected six method records"
assert len(frames) == 6, "Expected six MuJoCo captures"
assert composite.width > composite.height * 4, "Composite is not a 1x6 layout"
assert (root / "outputs" / "mujoco_six_trajectories_1x6.pdf").is_file()
for record in metrics:
    safe = record["method"].lower().replace(" ", "_").replace("-", "_")
    assert (root / "outputs" / f"{safe}_trajectories.json").is_file()

print(f"methods={[record['method'] for record in metrics]}")
print(f"composite={composite.size}")
print(f"frames={len(frames)}")
print(f"mujoco_bodies={model.nbody}")
print(f"metrics={metrics}")
