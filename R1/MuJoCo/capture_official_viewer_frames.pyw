"""Capture final MuJoCo viewer frames for the manuscript on Windows.

This helper is intentionally a GUI script. It replays the trajectories saved by
``run_six_algorithms.py`` in the official passive MuJoCo viewer, draws the same
MuJoCo trail geometries used by the off-screen renderer, and captures the
viewer client area through the Windows desktop compositor. This avoids relying
on the GLFW off-screen context, which is unavailable in some headless shells.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import time
import traceback

from PIL import ImageGrab
import mujoco
import mujoco.viewer
import numpy as np

import run_six_algorithms as sim


DESTINATION = sim.OUTPUT / "official_viewer_frames"
LOG = DESTINATION / "capture.log"


def _find_mujoco_client_rect() -> tuple[int, int, int, int]:
    user32 = ctypes.windll.user32
    matches: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        if "mujoco" in title.value.lower():
            matches.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    if not matches:
        raise RuntimeError("No visible MuJoCo viewer window was found")
    hwnd = matches[-1]
    user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.45)
    client = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client)):
        raise ctypes.WinError()
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        raise ctypes.WinError()
    return (
        origin.x,
        origin.y,
        origin.x + client.right - client.left,
        origin.y + client.bottom - client.top,
    )


def _add_trails(scene, trajectories: list[np.ndarray]) -> None:
    for agent, trajectory in enumerate(trajectories):
        stride = max(1, len(trajectory) // 110)
        points = trajectory[::stride]
        if not np.allclose(points[-1], trajectory[-1]):
            points = np.vstack([points, trajectory[-1]])
        for start, end in zip(points[:-1], points[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.eye(3).reshape(-1),
                sim.AGENT_RGBA[agent].copy(),
            )
            p0 = np.array([start[0], start[1], 0.12], dtype=float)
            p1 = np.array([end[0], end[1], 0.12], dtype=float)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.105, p0, p1)
            scene.ngeom += 1


def capture(method: str, xml: str) -> None:
    trajectories = sim.load_saved_trajectories(method)
    if trajectories is None:
        raise FileNotFoundError(f"No saved MuJoCo trajectory for {method}")
    plant = sim.MuJoCoPlant(xml)
    for agent, trajectory in enumerate(trajectories):
        sim.set_playback_state(plant, agent, trajectory[-1], 0.0)
    mujoco.mj_forward(plant.model, plant.data)

    with mujoco.viewer.launch_passive(
        plant.model,
        plant.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        viewer.cam.distance = 56.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -90.0
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            _add_trails(viewer.user_scn, trajectories)
        viewer.sync()
        time.sleep(1.2)
        bbox = _find_mujoco_client_rect()
        ctypes.windll.user32.SetCursorPos(bbox[0] + 4, bbox[1] + 4)
        time.sleep(0.15)
        safe = method.lower().replace(" ", "_").replace("-", "_")
        image = ImageGrab.grab(bbox=bbox, all_screens=True)
        width, height = image.size
        if width > height:
            left = (width - height) // 2
            image = image.crop((left, 0, left + height, height))
        elif height > width:
            top = (height - width) // 2
            image = image.crop((0, top, width, top + width))
        image.save(DESTINATION / f"{safe}.png")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    obstacles = sim.generate_obstacles(123)
    xml = sim.make_scene_xml(obstacles)
    completed = []
    for method in sim.METHODS:
        capture(method, xml)
        completed.append(method)
    LOG.write_text("Captured: " + ", ".join(completed), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise
