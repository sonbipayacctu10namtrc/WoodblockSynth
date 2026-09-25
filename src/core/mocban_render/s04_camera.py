"""
mocban_render/s04_camera.py — Toán học camera: góc -> hướng, ma trận pose kiểu OpenGL.
"""
from __future__ import annotations

import math

import numpy as np

from .s03_shot import CameraSpec


# ----------------------------------------------------------------------------
# 4. Toán học camera
# ----------------------------------------------------------------------------

def _spherical_dir(elev_deg: float, azim_deg: float) -> np.ndarray:
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])



def look_at_pose(eye: np.ndarray, target: np.ndarray, roll_deg: float = 0.0) -> np.ndarray:
    """
    Ma trận pose 4x4 (OpenGL: camera nhìn theo -Z của chính nó).
    Hướng "lên" của ảnh = trục +Y thế giới (trục dọc của khối) xoay roll_deg quanh Z, chiếu vuông góc
    với hướng nhìn -> khối luôn đứng khi roll=0, bất kể elev/azim (không nhảy hướng ở elev ~80°).
    """
    f = target - eye; f /= np.linalg.norm(f)
    r = math.radians(roll_deg)
    up_hint = np.array([-math.sin(r), math.cos(r), 0.0])
    s = np.cross(f, up_hint)
    if np.linalg.norm(s) < 1e-6:  # nhìn dọc theo trục +Y (elev ~0): fallback +Z
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, 0] = s; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M



def camera_pose_from_spec(c: CameraSpec) -> np.ndarray:
    eye = np.array(c.look_at) + _spherical_dir(c.elev_deg, c.azim_deg) * c.dist_mm
    return look_at_pose(eye, np.array(c.look_at), c.roll_deg)
