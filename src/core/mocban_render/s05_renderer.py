"""
mocban_render/s05_renderer.py — Renderer pyrender: dựng cảnh, render ảnh màu + độ sâu có bóng đổ.
"""
from __future__ import annotations

import math

import numpy as np
import trimesh

# pyrender 0.1.45 (PyPI, 2020) còn dùng np.infty đã bị bỏ ở NumPy 2 -> shim trước khi import pyrender
if not hasattr(np, "infty"):
    np.infty = np.inf

from .s04_camera import _spherical_dir, camera_pose_from_spec, look_at_pose
from .s03_shot import ShotSpec


# ----------------------------------------------------------------------------
# 5. Renderer pyrender
# ----------------------------------------------------------------------------

class PyrenderBackend:
    def __init__(self, mesh: trimesh.Trimesh, width: int, height: int):
        import pyrender
        self.pr = pyrender
        self.width, self.height = width, height
        self.mesh_node_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        for prim in self.mesh_node_mesh.primitives:   # gỗ thấm mực: hơi bóng
            prim.material.roughnessFactor = 0.6
            prim.material.metallicFactor = 0.0
        self.r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        # mặt bàn phía dưới khối để có bóng đổ ra ngoài
        bw, bh = mesh.metadata["block_size_mm"][:2]
        table = trimesh.creation.box(extents=[bw * 4, bh * 4, 1.0])
        table.apply_translation([0, 0, -0.5])
        table.visual = trimesh.visual.ColorVisuals(table, face_colors=[120, 110, 100, 255])
        self.table = pyrender.Mesh.from_trimesh(table, smooth=False)

    def render(self, shot: ShotSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pr = self.pr
        table_col = shot.post.get("table_rgb", [0.5, 0.48, 0.45])
        scene = pr.Scene(ambient_light=np.array(shot.ambient + [1.0]), bg_color=table_col + [1.0])
        scene.add(self.mesh_node_mesh)
        self.table.primitives[0].material.baseColorFactor = np.array(table_col + [1.0])
        scene.add(self.table)
        c = shot.camera
        cam = pr.PerspectiveCamera(yfov=math.radians(c.yfov_deg), aspectRatio=c.width / c.height, znear=5.0, zfar=5000.0)
        cam_pose = camera_pose_from_spec(c)
        scene.add(cam, pose=cam_pose)
        for L in shot.lights:
            d = _spherical_dir(L.elev_deg, L.azim_deg)
            pose = look_at_pose(d * L.dist_mm, np.zeros(3))
            if L.kind == "directional":
                scene.add(pr.DirectionalLight(color=L.color, intensity=L.intensity), pose=pose)
            elif L.kind == "point":
                scene.add(pr.PointLight(color=L.color, intensity=L.intensity), pose=pose)
            else:
                scene.add(pr.SpotLight(color=L.color, intensity=L.intensity,
                                       innerConeAngle=0.3, outerConeAngle=0.6), pose=pose)
        flags = pr.RenderFlags.SHADOWS_DIRECTIONAL | pr.RenderFlags.SHADOWS_SPOT
        color, depth = self.r.render(scene, flags=flags)
        return color, depth, cam_pose

    def gl_info(self) -> dict:
        """GPU / driver OpenGL THẬT SỰ đang render. 'llvmpipe' / 'softpipe' / 'SwiftShader' = render bằng CPU
        (EGL không tới được driver GPU) -> chậm hàng chục lần dù máy có GPU."""
        from OpenGL.GL import glGetString, GL_RENDERER, GL_VENDOR, GL_VERSION
        self.r._platform.make_current()
        return {k: (glGetString(v) or b"").decode(errors="replace")
                for k, v in (("renderer", GL_RENDERER), ("vendor", GL_VENDOR), ("version", GL_VERSION))}

    def close(self):
        self.r.delete()
