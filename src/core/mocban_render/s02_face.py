"""
mocban_render/s02_face.py — Tự xác định mặt khắc (detect_main_face), xoay mặt khắc lên +Z (orient_to_face), đọc manifest.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .s01_scan import prepare_scan


# ----------------------------------------------------------------------------
# 2. Tự xác định mặt khắc + manifest
# ----------------------------------------------------------------------------
# Giả định: mỗi khối chỉ có MỘT mặt mang thông tin (mặt khắc); các mặt còn lại là lưng / cạnh.
# Để dùng được cho các bộ scan khác:
#   - hướng tính theo hộp bao CÓ HƯỚNG (OBB) -> không phụ thuộc scan được đặt nghiêng thế nào trong file;
#   - mọi ngưỡng tính theo tỉ lệ cạnh dài của khối -> không phụ thuộc đơn vị (mm, m, tuỳ ý);
#   - quyết định bằng SO SÁNH TƯƠNG ĐỐI giữa các mặt, kèm độ tin cậy; mặt nào không chắc -> đánh dấu để soát;
#   - chỉ dùng phép quay thật (det = +1) -> mesh không bao giờ bị lật gương;
#   - luôn có đường chỉ định tay (tham số `face` / file manifest).
# Tên mặt: 3 trục OBB xếp theo độ dài giảm dần A (dài nhất), B, C (ngắn nhất = chiều dày); "C+" / "C-" là
# hai mặt vuông góc trục C. Tên cố định với cùng một file scan -> dùng được trong manifest.

FACE_THIN_RATIO = 0.6    # cạnh ngắn / cạnh giữa <= ngưỡng -> khối dẹt: chỉ xét 2 mặt lớn C+/C-, ngược lại xét cả 6
FACE_MIN_RATIO = 2.0     # điểm chi tiết mặt tốt nhất / mặt thứ nhì >= ngưỡng -> tin cậy (hiệu chỉnh khi có thêm scan)
FACE_MIN_COVERAGE = 0.5  # mặt được chọn phải có dữ liệu scan trên >= 50% diện tích
FACE_MISSING = 0.15      # mặt đối diện có dữ liệu < 15% diện tích -> scan chỉ quét một mặt



def _axis_names(ext) -> dict[int, str]:
    return {int(ax): "ABC"[i] for i, ax in enumerate(np.argsort(-np.asarray(ext), kind="stable"))}



def _face_axis(face: str, ext) -> tuple[int, int]:
    if len(face) != 2 or face[0] not in "ABC" or face[1] not in "+-":
        raise ValueError(f"tên mặt không hợp lệ: {face!r} (dạng 'C+', 'C-', 'A+', ...)")
    ax = next(a for a, n in _axis_names(ext).items() if n == face[0])
    return ax, (1 if face[1] == "+" else -1)



def _face_height(mesh: trimesh.Trimesh, ax: int, sgn: int, px: float) -> np.ndarray:
    """Z-buffer trực giao nhìn từ phía (ax, sgn) trong khung OBB: độ cao (mm) theo hướng nhìn, -inf = không có dữ liệu.
    Chỉ lấy tam giác quay mặt về phía người nhìn -> scan chỉ quét một mặt nhìn từ phía sau sẽ trống."""
    d = np.zeros(3); d[ax] = sgn
    sel = mesh.face_normals @ d > 0.05
    a1, a2 = [i for i in range(3) if i != ax]
    lo, ext = mesh.bounds[0], mesh.bounding_box.extents
    h = np.full((int(ext[a2] / px) + 1, int(ext[a1] / px) + 1), -np.inf, np.float32)
    if sel.any():
        P = np.vstack([mesh.vertices[np.unique(mesh.faces[sel])], mesh.triangles_center[sel]])
        ix = ((P[:, a1] - lo[a1]) / px).astype(int); iy = ((P[:, a2] - lo[a2]) / px).astype(int)
        np.maximum.at(h, (iy, ix), (sgn * P[:, ax]).astype(np.float32))
    return h



def _face_scores(h: np.ndarray, px: float, long_mm: float) -> dict:
    """detail_mm: trung vị |độ cao - bề mặt trơn (Gaussian σ = 1% cạnh dài)| trên lớp mặt ngoài -> mặt khắc có hoa
    văn/chữ phủ kín cho giá trị cao; mặt lưng phẳng (kể cả có vài hốc tay cầm lớn) cho giá trị thấp. Dùng TRUNG VỊ
    chứ không dùng độ lệch chuẩn: mép vài hốc sâu làm độ lệch chuẩn của mặt lưng vượt cả mặt khắc.
    coverage: phần diện tích mặt có bề mặt scan (sau khi lấp lỗ nhỏ giữa các điểm splat; mảnh vụn rời rạc
    không lấp kín được nên không được tính là có dữ liệu)."""
    raw = np.isfinite(h)
    k3 = np.ones((3, 3), np.uint8)
    # phép đóng: lấp lỗ nhỏ giữa các điểm splat mà không làm mảnh vụn phình to
    valid = cv2.erode(cv2.dilate(raw.astype(np.uint8), k3, iterations=3), k3, iterations=3).astype(bool)
    coverage = float(valid.mean())
    if raw.sum() < 100:
        return {"detail_mm": 0.0, "coverage": coverage}
    hf = np.where(raw, h, -1e9).astype(np.float32)
    for _ in range(3):                                   # gán độ cao cho các lỗ vừa lấp
        d = cv2.dilate(hf, k3)
        hf = np.where((hf < -1e8) & (d > -1e8), d, hf)
    valid &= hf > -1e8
    ref = np.percentile(hf[valid], 90)
    top = valid & (hf > ref - 0.075 * long_mm)          # lớp mặt ngoài: bỏ hốc sâu, vát cạnh
    er = max(1, int(0.025 * long_mm / px))              # co vào 2.5% cạnh dài, tránh mép
    top = cv2.erode(top.astype(np.uint8), np.ones((er, er), np.uint8)).astype(bool)
    if top.sum() < 100:
        return {"detail_mm": 0.0, "coverage": coverage}
    base = np.where(top, hf, np.median(hf[top])).astype(np.float32)
    res = base - cv2.GaussianBlur(base, (0, 0), 0.01 * long_mm / px)
    return {"detail_mm": float(np.median(np.abs(res[top]))), "coverage": coverage}



def detect_main_face(mesh: trimesh.Trimesh) -> dict:
    """Mesh ở khung OBB (prepare_scan) -> mặt khắc. Trả {face, auto, confident, reason, ratio, flat, px_mm, scores}.
    confident=False nghĩa là vẫn trả mặt đoán tốt nhất nhưng cần người soát (ảnh face_audit_image)."""
    ext = mesh.bounding_box.extents
    names = _axis_names(ext)
    long_mm = float(ext.max())
    e = np.sort(ext)
    flat = bool(e[0] / e[1] <= FACE_THIN_RATIO)
    axes = [a for a, n in names.items() if n == "C"] if flat else [0, 1, 2]
    # độ phân giải: 1/400 cạnh dài, nhưng không mịn hơn mật độ đỉnh của mesh (tránh lỗ khi splat)
    px = max(long_mm / 400, 0.7 * math.sqrt(2 * mesh.area / max(len(mesh.faces), 1)))
    scores = {}
    for ax in sorted(axes, key=lambda a: names[a]):
        for sgn in (1, -1):
            scores[names[ax] + ("+" if sgn > 0 else "-")] = _face_scores(_face_height(mesh, ax, sgn, px), px, long_mm)
    # chỉ xếp hạng các mặt có đủ dữ liệu: vài mảnh vụn (vd. mép cạnh thấy từ phía sau của scan hở) cho điểm chi
    # tiết cao giả tạo trên diện tích nhỏ
    def detail(k):
        return scores[k]["detail_mm"]
    elig = [k for k in scores if scores[k]["coverage"] >= FACE_MIN_COVERAGE]
    ratio = None
    if not elig:
        best = max(scores, key=lambda k: scores[k]["coverage"])
        confident, reason = False, f"không mặt nào có dữ liệu trên {FACE_MIN_COVERAGE:.0%} diện tích"
    else:
        best = max(elig, key=detail)
        opp = best[0] + ("-" if best[1] == "+" else "+")
        rest = [k for k in elig if k != best]
        if scores[opp]["coverage"] < FACE_MISSING:
            confident, reason = True, "scan chỉ quét một mặt (mặt đối diện không có dữ liệu)"
        elif not rest:
            confident, reason = False, "các mặt còn lại thiếu dữ liệu, không so sánh được"
        else:
            second = max(rest, key=detail)
            ratio = round(min(detail(best) / max(detail(second), 1e-9), 999.0), 2)
            confident = ratio >= FACE_MIN_RATIO
            reason = f"chi tiết gấp {ratio:.1f}x mặt thứ nhì ({second})" + ("" if confident else f", dưới ngưỡng {FACE_MIN_RATIO}")
    return {"face": best, "auto": True, "confident": confident, "reason": reason, "ratio": ratio,
            "flat": flat, "px_mm": round(px, 3),
            "scores": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in scores.items()}}



def orient_to_face(mesh: trimesh.Trimesh, face: str, rot90: int = 0) -> trimesh.Trimesh:
    """Bản sao mesh (khung OBB) xoay cho `face` hướng +Z, cạnh dài còn lại nằm dọc trục X, rồi xoay thêm
    rot90 x 90° quanh Z; đáy chạm z=0, tâm XY ở gốc. Chỉ phép quay thật (det = +1).
    Hướng 0° / 180° trong mặt phẳng KHÔNG suy ra được từ hình học -> chỉnh bằng rot90 nếu cần."""
    ext = mesh.bounding_box.extents
    ax, sgn = _face_axis(face, ext)
    z = np.zeros(3); z[ax] = sgn
    xa = max((a for a in range(3) if a != ax), key=lambda a: ext[a])
    x = np.zeros(3); x[xa] = 1.0
    R = np.eye(4); R[:3, :3] = np.stack([x, np.cross(z, x), z])     # hàng = trục mới; y = z × x -> det = +1
    k = int(rot90) % 4
    c, s = (1, 0, -1, 0)[k], (0, 1, 0, -1)[k]
    Rz = np.eye(4); Rz[:2, :2] = [[c, -s], [s, c]]
    out = mesh.copy()
    out.apply_transform(Rz @ R)
    t = np.eye(4); t[:3, 3] = [-out.bounding_box.centroid[0], -out.bounding_box.centroid[1], -out.bounds[0][2]]
    out.apply_transform(t)
    out.metadata = dict(mesh.metadata)
    out.metadata["block_size_mm"] = [float(v) for v in out.bounding_box.extents]
    if "to_obb_frame" in mesh.metadata:
        out.metadata["to_render_frame"] = (t @ Rz @ R @ np.array(mesh.metadata["to_obb_frame"])).tolist()
    return out



def load_external_mesh(path: str | Path, target_extent_mm: float = 200.0, texture_path: str | Path | None = "auto",
                       face: str = "auto", rot90: int = 0) -> trimesh.Trimesh:
    """Nạp GLB/OBJ/STL/PLY -> mesh sẵn sàng render: mặt khắc hướng +Z, đáy z=0, cạnh dài nhất = target.
    face='auto' tự xác định (xem detect_main_face), hoặc chỉ định 'C+' / 'C-' / ... (tên trong ảnh soát).
    metadata['face_info'] ghi mặt đã chọn, lý do, độ tin cậy, điểm từng mặt."""
    m = prepare_scan(path, target_extent_mm, texture_path)
    info = detect_main_face(m) if face in (None, "", "auto") else \
        {"face": face, "auto": False, "confident": True, "reason": "chỉ định (tham số / manifest)"}
    info["rot90"] = int(rot90)
    out = orient_to_face(m, info["face"], rot90)
    out.metadata["face_info"] = info
    return out



def read_manifest(path: str | Path) -> dict[str, dict]:
    """CSV cột mesh, texture, face, rot90 (cột khác bỏ qua). Khoá = TÊN file mesh (không kèm thư mục) để dùng
    được ở máy khác / Kaggle. face trống hoặc 'auto' -> tự xác định; texture trống -> tự tìm.
    File audit.csv do audit_scans.py sinh ra có đúng các cột này: sửa dòng confident=False rồi dùng lại."""
    import csv
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("mesh") or "").strip():
                continue
            out[Path(r["mesh"].strip()).name] = {
                "texture": (r.get("texture") or "").strip() or "auto",
                "face": (r.get("face") or "").strip() or "auto",
                "rot90": int(float(r.get("rot90") or 0)),
            }
    return out
