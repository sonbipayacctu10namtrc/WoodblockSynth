"""
mocban_render/s01_scan.py — Nạp mesh scan, tìm texture rời, đưa scan về khung OBB chuẩn (prepare_scan).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image


# ----------------------------------------------------------------------------
# 1. Nạp mesh scan + chuẩn hoá khung (OBB)
# ----------------------------------------------------------------------------

MESH_EXT = (".glb", ".gltf", ".obj", ".stl", ".ply")
_IMG_EXT = (".jpg", ".jpeg", ".png")
_NOT_ALBEDO = ("_nm", "normal", "_disp", "_height", "_rough", "_metal", "_ao", "_spec", "_mask", "_bump")



def _has_texture_image(m) -> bool:
    """Mesh có ảnh texture thật? (trimesh gắn ảnh giữ chỗ 2x2 cho OBJ không kèm .mtl -> không tính)."""
    mat = getattr(m.visual, "material", None)
    img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
    return img is not None and min(img.size) >= 16



def _texture_search_dirs(mesh_path: Path, up: int = 1) -> list[Path]:
    """Thư mục của mesh + các thư mục con trực tiếp của `up` cấp tổ tiên (up=1: thư mục anh em, vd.
    <scan>/source/x.obj cạnh <scan>/textures/x_4K.jpg; up=2: thêm một cấp, vd. Kaggle giải nén zip lồng thành
    <scan>/source/<zip_long>/x.obj trong khi texture ở <scan>/textures/)."""
    dirs = [mesh_path.parent]
    anc = mesh_path.parent
    for _ in range(up):
        if anc.parent == anc:
            break
        anc = anc.parent
        dirs += [anc] + [d for d in anc.iterdir() if d.is_dir() and d not in dirs]
    return list(dict.fromkeys(dirs))



def find_texture(mesh_path: str | Path) -> Optional[Path]:
    """Ảnh texture màu cho mesh có UV nhưng không kèm material. Bỏ normal/roughness/..., chọn ảnh có tên chung
    tiền tố dài nhất với tên mesh (hoà -> file lớn nhất). None nếu không có ảnh, hoặc có nhiều ảnh mà không ảnh
    nào trùng tên (không đoán bừa: khi đó chỉ định trong manifest).
    Tìm gần trước (thư mục mesh + anh em); không thấy thì lên thêm 2 cấp nhưng khi đó BẮT BUỘC trùng tiền tố tên,
    vì càng lên cao càng dễ gặp texture của scan khác."""
    p = Path(mesh_path)
    stem = p.stem.lower()

    def pref(f):
        return len(os.path.commonprefix([stem, f.stem.lower()]))
    for up, strict in ((1, False), (3, True)):
        cands = list(dict.fromkeys(
            f for d in _texture_search_dirs(p, up) for f in d.iterdir()
            if f.is_file() and f.suffix.lower() in _IMG_EXT and not any(k in f.stem.lower() for k in _NOT_ALBEDO)))
        if not cands:
            continue
        best = max(cands, key=lambda f: (pref(f), f.stat().st_size))
        if pref(best) >= 4 or (len(cands) == 1 and not strict):
            return best
    return None



def resolve_texture(mesh_path: str | Path, texture: str | Path | None) -> Optional[Path]:
    """None -> không gắn texture. 'auto' / '' -> find_texture. Tên file hoặc đường dẫn không tồn tại -> tìm
    theo TÊN trong các thư mục cạnh mesh (để manifest viết ở máy này vẫn dùng được trên Kaggle)."""
    if texture is None:
        return None
    if texture in ("", "auto"):
        return find_texture(mesh_path)
    t = Path(texture)
    if t.exists():
        return t
    for d in _texture_search_dirs(Path(mesh_path), up=3):
        if (d / t.name).exists():
            return d / t.name
    raise FileNotFoundError(f"không thấy texture {texture} cho {mesh_path}")



def _proper_rotation(R: np.ndarray) -> np.ndarray:
    """Ép ma trận quay 3x3 về det = +1 (đảo dấu một trục) -> không lật gương."""
    R = np.array(R, dtype=float)
    if np.linalg.det(R) < 0:
        R[2] *= -1
    return R



def prepare_scan(src, target_extent_mm: float = 200.0, texture_path: str | Path | None = "auto") -> trimesh.Trimesh:
    """Nạp scan (đường dẫn hoặc Trimesh) và đưa về KHUNG OBB: tâm OBB ở gốc, cạnh OBB song song trục toạ độ,
    cạnh dài nhất = target_extent_mm. Khung chung để chấm điểm các mặt, không phụ thuộc scan đặt nghiêng.
    Giữ texture (UV); texture_path='auto' tự tìm ảnh rời khi mesh có UV nhưng không kèm material, None = không gắn.
    metadata['to_obb_frame'] = ma trận 4x4 từ toạ độ file -> khung OBB (để truy lại)."""
    if isinstance(src, trimesh.Trimesh):
        m = src.copy()
    else:
        m = trimesh.load(str(src), process=False)   # KHÔNG force="mesh": giữ TextureVisuals/UV
        if isinstance(m, trimesh.Scene):
            m = m.to_mesh()                          # áp phép biến đổi của từng node rồi gộp
        uv = getattr(m.visual, "uv", None)
        tex = resolve_texture(src, texture_path) if (uv is not None and not _has_texture_image(m)) else None
        if tex is not None:
            m.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.open(str(tex)).convert("RGB"))
        m.metadata["source_texture"] = str(tex) if tex is not None else None
    # có ảnh màu thật (nhúng sẵn hoặc vừa gắn)? False -> render hướng A sẽ ra màu xám
    m.metadata["has_texture"] = _has_texture_image(m)
    # mesh lộn trong ra ngoài (normal hướng vào trong) -> lật lại, vì chấm điểm dựa vào hướng normal
    o = m.bounding_box.centroid
    rel = m.triangles_center - o
    w = m.area_faces / np.maximum(np.linalg.norm(rel, axis=1), 1e-9)
    if (w * (m.face_normals * rel).sum(1)).sum() / max(m.area, 1e-9) < -0.2:
        m.invert()
    box = m.bounding_box_oriented.primitive.transform     # khung OBB -> toạ độ file
    R = _proper_rotation(box[:3, :3].T)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ box[:3, 3]
    m.apply_transform(M)
    s = target_extent_mm / m.bounding_box.extents.max()
    m.apply_scale(s)
    m.metadata["to_obb_frame"] = (np.diag([s, s, s, 1.0]) @ M).tolist()
    return m
