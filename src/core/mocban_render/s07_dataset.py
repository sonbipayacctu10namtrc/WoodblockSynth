"""
mocban_render/s07_dataset.py — Vòng lặp sinh dữ liệu, soát mặt khắc, ghép ảnh xem nhanh.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import trimesh

from .s02_face import detect_main_face, orient_to_face
from .s06_post import add_occluders, apply_post
from .s05_renderer import PyrenderBackend
from .s01_scan import prepare_scan
from .s03_shot import frontal_shot, sample_shot


# ----------------------------------------------------------------------------
# 7. Vòng lặp sinh dữ liệu
# ----------------------------------------------------------------------------

def render_dataset(
    backend,
    out_dir: str | Path,
    n: int,
    block_wh_mm: tuple[float, float],
    width: int = 1024,
    height: int = 768,
    preset: str = "mixed",
    seed: int = 0,
    name: str = "model",
    occluder_prob: float = 0.0,
    extra_meta: Optional[dict] = None,
) -> list[dict]:
    """
    Render n ảnh: mỗi ảnh lấy mẫu góc chụp/đèn (sample_shot) -> render -> hậu kỳ camera -> vật che 2D.
    Ghi <name>_XXXX.jpg + <name>_meta.json (pose camera, đèn, tham số hậu kỳ, vật che, % diện tích bị che).
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records = []
    for i in range(n):
        shot = sample_shot(rng, block_wh_mm, width, height, preset)
        color, depth, cam_pose = backend.render(shot)
        img = apply_post(color, shot.post, rng)
        img, occ, kinds = add_occluders(img, rng, occluder_prob)
        if kinds:  # vật che thêm sau JPEG của apply_post -> nén lại nhẹ cho đồng nhất
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        fn = f"{name}_{i:04d}.jpg"
        cv2.imwrite(str(out_dir / fn), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec = {"file": fn, "shot": asdict(shot), "cam_pose": cam_pose.tolist(), "occluders": kinds,
               "occluded_frac": round(float((occ >= 0.5).mean()), 4)}
        if extra_meta:
            rec.update(extra_meta)
        records.append(rec)
    with open(out_dir / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)
    return records



def face_audit_image(mesh_obb: trimesh.Trimesh, info: dict, width: int = 480, height: int = 360) -> np.ndarray:
    """Ảnh soát (RGB): nhìn thẳng từng mặt ứng viên kèm điểm chi tiết / độ phủ. Khung XANH = mặt được chọn và tin
    cậy, VÀNG = được chọn nhưng cần người xem, TÍM = chỉ định tay. mesh_obb: kết quả prepare_scan."""
    scores = info.get("scores") or {info["face"]: None}
    tiles = []
    for name, sc in scores.items():
        m = orient_to_face(mesh_obb, name)
        be = PyrenderBackend(m, width, height)
        img, _, _ = be.render(frontal_shot(m.metadata["block_size_mm"], width, height))
        be.close()
        img = np.ascontiguousarray(img)
        if name == info["face"]:
            col = (170, 60, 220) if not info.get("auto", True) else (0, 190, 0) if info["confident"] else (255, 185, 0)
            cv2.rectangle(img, (0, 0), (width - 1, height - 1), col, 10)
        txt = name if sc is None else f"{name}  detail {sc['detail_mm']:.3f}mm  cover {sc['coverage'] * 100:.0f}%"
        cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4)
        cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
        tiles.append(img)
    cols = min(len(tiles), 3)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])



def audit_scan(path: str | Path, target_extent_mm: float = 200.0, override: Optional[dict] = None,
               width: int = 480, height: int = 360, image: bool = True) -> tuple[dict, Optional[np.ndarray], Optional[str]]:
    """Soát một scan: prepare_scan + detect_main_face + áp chỉ định tay (dòng manifest: texture/face/rot90)
    + ảnh soát. Trả (info, ảnh RGB, đường dẫn texture rời hoặc None). info['auto_face'] = kết quả tự động."""
    ov = override or {}
    m = prepare_scan(path, target_extent_mm, ov.get("texture", "auto"))
    info, img = audit_prepared(m, ov, width, height, image)
    return info, img, m.metadata.get("source_texture")



def audit_prepared(m_obb: trimesh.Trimesh, override: Optional[dict] = None, width: int = 480, height: int = 360,
                   image: bool = True) -> tuple[dict, Optional[np.ndarray]]:
    """Như audit_scan nhưng nhận mesh ĐÃ nạp (kết quả prepare_scan) -> nạp scan một lần, dùng lại cho mọi bước.
    Trả (info, ảnh RGB hoặc None khi image=False)."""
    ov = override or {}
    info = detect_main_face(m_obb)
    info["auto_face"] = info["face"]
    if ov.get("face", "auto") not in (None, "", "auto"):
        same = "trùng" if ov["face"] == info["auto_face"] else "KHÁC"
        info.update(face=ov["face"], auto=False, confident=True,
                    reason=f"chỉ định qua manifest ({same} kết quả tự động {info['auto_face']})")
    info["rot90"] = int(ov.get("rot90", 0))
    info["has_texture"] = bool(m_obb.metadata.get("has_texture"))
    img = face_audit_image(m_obb, info, width, height) if image else None   # image=False: không cần pyrender/GPU
    return info, img



def contact_sheet(files: list[Path], cols: int = 4, thumb_w: int = 320) -> np.ndarray:
    thumbs = []
    for f in files:
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        thumbs.append(cv2.resize(im, (thumb_w, int(h * thumb_w / w))))
    th = max(t.shape[0] for t in thumbs)
    thumbs = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) for t in thumbs]
    while len(thumbs) % cols:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [np.hstack(thumbs[i:i + cols]) for i in range(0, len(thumbs), cols)]
    return np.vstack(rows)
