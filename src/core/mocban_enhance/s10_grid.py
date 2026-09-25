"""
mocban_enhance/s10_grid.py — Bảng phương pháp METHODS (m_*) và lưới so sánh enhance_grid.
"""
import cv2
import numpy as np

from .s07_ao import horizon_ao
from .s05_curvature import curvature
from .s04_depth import depth_map
from .s03_derivatives import normal_map_rgb, normals
from .s09_display import _lab, colorize, norm01
from .s06_msii import msii_combine, msii_volume
from .s08_shading import exaggerated_shading, lambert, light_dir, radiance_scaling


# ----------------------------------------------------------------------------
# 10. Lưới so sánh các phương pháp
# ----------------------------------------------------------------------------

def m_depth(h, px_mm, **kw):
    d = depth_map(h, px_mm, hp_sigma_px=kw.get("hp_sigma_px", 12.0))
    return {"raw": d["local"], "img": colorize(norm01(d["local"], sym=True), "coolwarm"),
            "label": "1 Depth (local, detrend)"}



def m_normal(h, px_mm, **kw):
    n = normals(h, px_mm, kw.get("sigma_px", 1.0), kw.get("gain", 0.35))
    return {"raw": n[..., 2], "img": normal_map_rgb(n),
            "label": "2 Normal map (gain %.2f)" % kw.get("gain", 0.35)}



def m_curvature(h, px_mm, **kw):
    """
    Shape index ĐIỀU BIÊN theo curvedness.

    S là một TỈ SỐ nên nó bão hoà về +-1 ngay cả khi độ cong nhỏ vô cùng: hiện S trần thì
    nền khoét (chỉ có nhiễu vết đục) cũng đỏ/xanh loang lổ và nuốt mất nét chữ. Koenderink
    vốn định nghĩa S đi kèm C: S nói HÌNH DẠNG gì, C nói MẠNH bao nhiêu. Nhân hai cái đưa
    vùng phẳng về 0 (trung tính) và giữ nguyên phân loại lồi/lõm ở nơi thực sự có độ cong.
    """
    c = curvature(h, px_mm, kw.get("sigma_px", 1.5))
    s = c["shape_index"] * norm01(c["curvedness"], pct=(2, 98))
    return {"raw": s, "img": colorize(norm01(s, vrange=(-1, 1)), "coolwarm"),
            "label": "3 Shape index x curvedness"}



def m_msii(h, px_mm, **kw):
    radii = kw.get("radii_px", (3, 6, 12, 24, 48))
    c = msii_combine(msii_volume(h, px_mm, radii), radii, px_mm)["combined"]
    return {"raw": c, "img": colorize(norm01(c, sym=True), "coolwarm"), "label": "4 MSII combined"}



def m_ao(h, px_mm, **kw):
    ao = horizon_ao(h, px_mm, kw.get("n_az", 16), kw.get("radius_px", 24), kw.get("n_steps", 12))
    return {"raw": ao, "img": colorize(ao, "gray"), "label": "5 Ambient occlusion"}



def m_exaggerated(h, px_mm, **kw):
    s = exaggerated_shading(h, px_mm, kw.get("ldir", light_dir()), kw.get("sigma0_px", 1.0),
                            kw.get("n_scales", 4), kw.get("gain", 0.4))
    return {"raw": s, "img": colorize(s, "gray"), "label": "6a Exaggerated shading"}



def m_radiance(h, px_mm, **kw):
    s = radiance_scaling(h, px_mm, kw.get("ldir", light_dir()), kw.get("alpha", 3.0),
                         kw.get("sigma_px", 1.5))
    return {"raw": s, "img": colorize(s, "gray"), "label": "6b Radiance scaling"}



def m_lambert(h, px_mm, **kw):
    s = lambert(h, px_mm, kw.get("ldir", light_dir()), kw.get("sigma_px", 1.0), kw.get("gain", 1.0))
    return {"raw": s, "img": colorize(s, "gray"), "label": "Lambert (control)"}



METHODS = {"depth": m_depth, "normal": m_normal, "curvature": m_curvature, "msii": m_msii,
           "ao": m_ao, "exaggerated": m_exaggerated, "radiance": m_radiance, "lambert": m_lambert}
GRID6 = ("depth", "normal", "curvature", "msii", "ao", "exaggerated")



def enhance_grid(h, px_mm, names=None, crop=None, cols=3, thumb_w=520, flip_mirror=True, **kw):
    """
    Chạy nhiều phương pháp rồi ghép thành một lưới RGB uint8 (ghép bằng numpy như
    mocban_render.contact_sheet: nhanh hơn matplotlib và giữ nguyên độ phân giải để zoom).

    crop=(y0,y1,x0,x1): cắt TRƯỚC khi tính. Nén cả tấm vào một ô lưới thường làm mất chi tiết
    nhỏ - bản crop mới là kết quả thật.
    flip_mirror: mộc bản khắc NGƯỢC nên lật ngang để chữ đọc xuôi; khối không có chữ -> False.
    """
    names = list(names or GRID6)
    hh = h[crop[0]:crop[1], crop[2]:crop[3]] if crop else h
    hh = np.ascontiguousarray(hh)

    tiles = []
    for nm in names:
        img = METHODS[nm](hh, px_mm, **kw)
        rgb = img["img"]
        if flip_mirror:
            rgb = rgb[:, ::-1]
        scale = float(thumb_w) / rgb.shape[1]
        rgb = cv2.resize(rgb, (thumb_w, max(int(round(rgb.shape[0] * scale)), 1)),
                         interpolation=cv2.INTER_AREA)
        tiles.append(_lab(rgb, img["label"]))

    rows, hgt = [], max(t.shape[0] for t in tiles)
    for i in range(0, len(tiles), cols):
        row = [np.pad(t, ((0, hgt - t.shape[0]), (0, 0), (0, 0))) for t in tiles[i:i + cols]]
        while len(row) < cols:
            row.append(np.zeros_like(row[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)
