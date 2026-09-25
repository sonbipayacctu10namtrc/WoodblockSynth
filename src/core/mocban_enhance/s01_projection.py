"""
mocban_enhance/s01_projection.py — Chiếu mesh 3D sang ảnh độ sâu 2D (trực giao, góc bất kỳ, rasterize tam giác).
"""
import math

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ----------------------------------------------------------------------------
# 1. Chiếu 3D sang 2D (trực giao, góc bất kỳ)
# ----------------------------------------------------------------------------

def _spherical_dir(elev_deg, azim_deg):
    """Hướng đơn vị trên mặt cầu - CÙNG công thức với mocban_render._spherical_dir."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])



def view_basis(elev_deg=90.0, azim_deg=0.0, roll_deg=0.0):
    """
    Hệ trục camera trực giao, CÙNG quy ước với mocban_render.look_at_pose: hướng "lên" của ảnh
    là +Y thế giới (trục dọc của khối) xoay roll quanh Z rồi trực giao hoá - nhờ vậy khối luôn
    đứng ở mọi elev/azim, không nhảy hướng ở elev ~ 80 độ.

    Trả (right, up, forward); forward là hướng NHÌN (từ camera vào khối).
    elev=90 cho right=+X, up=+Y, forward=-Z, tức nhìn thẳng từ trên xuống.
    """
    f = -_spherical_dir(elev_deg, azim_deg)          # camera ở ngoài nhìn vào
    r = math.radians(roll_deg)
    up_hint = np.array([-math.sin(r), math.cos(r), 0.0])
    s = np.cross(f, up_hint)
    if np.linalg.norm(s) < 1e-6:                     # nhìn dọc trục +Y -> fallback +Z
        s = np.cross(f, np.array([0.0, 0.0, 1.0]))
    s /= np.linalg.norm(s)
    return s, np.cross(s, f), f



def _raster_zbuffer(u, v, dep, F, W, H, max_cand=8_000_000):
    """
    Z-buffer bằng RASTERIZE tam giác, đúng tại TÂM pixel: pixel (r, c) có tâm (c + 0.5, r + 0.5) trong toạ độ
    pixel liên tục (u, v) nhận độ sâu, nội suy tuyến tính trên tam giác, của tam giác CAO NHẤT (gần camera nhất)
    chứa tâm đó. Xác định, không ngẫu nhiên: nơi có bề mặt phủ tâm pixel thì lớp bề mặt nằm dưới không bao giờ
    thắng được.

    Vector hoá theo nhóm: tam giác xếp theo số tâm pixel mà hộp bao của nó trải qua (k = 1, 2, 4, 8, ... theo mỗi
    chiều), mỗi nhóm thử k x k tâm ứng viên cho mọi tam giác cùng lúc, chia khối để giới hạn bộ nhớ.
    Tam giác chiếu thành đường thẳng (vách đứng nhìn ngang) không chứa tâm nào -> bỏ qua, đúng hình học.
    """
    buf = np.full(H * W, -np.inf)
    tu, tv, td = u[F], v[F], dep[F]                              # (n, 3)
    c_lo = np.ceil(tu.min(1) - 0.5).astype(np.int64)            # tâm c + 0.5 >= u_min
    c_hi = np.floor(tu.max(1) - 0.5).astype(np.int64)
    r_lo = np.ceil(tv.min(1) - 0.5).astype(np.int64)
    r_hi = np.floor(tv.max(1) - 0.5).astype(np.int64)
    span = np.maximum(c_hi - c_lo, r_hi - r_lo) + 1              # số tâm theo chiều dài hơn của hộp bao
    keep = (c_hi >= c_lo) & (r_hi >= r_lo)                       # hộp bao chứa ít nhất một tâm
    k_cls = np.where(keep, 2 ** np.ceil(np.log2(np.maximum(span, 1))).astype(np.int64), 0)
    for k in np.unique(k_cls[k_cls > 0]):
        idx = np.flatnonzero(k_cls == k)
        oc, orr = np.meshgrid(np.arange(k), np.arange(k))
        oc, orr = oc.ravel(), orr.ravel()
        step = max(1, int(max_cand // (k * k)))
        for s0 in range(0, len(idx), step):
            t = idx[s0:s0 + step]
            cc = c_lo[t, None] + oc[None, :]; rr = r_lo[t, None] + orr[None, :]
            pu, pv = cc + 0.5, rr + 0.5
            au, bu, cu = tu[t, 0, None], tu[t, 1, None], tu[t, 2, None]
            av, bv, cv = tv[t, 0, None], tv[t, 1, None], tv[t, 2, None]
            den = (bv - cv) * (au - cu) + (cu - bu) * (av - cv)
            ok_den = np.abs(den) > 1e-12
            den = np.where(ok_den, den, 1.0)
            l1 = ((bv - cv) * (pu - cu) + (cu - bu) * (pv - cv)) / den
            l2 = ((cv - av) * (pu - cu) + (au - cu) * (pv - cv)) / den
            l3 = 1.0 - l1 - l2
            eps = -1e-9
            ins = (ok_den & (l1 >= eps) & (l2 >= eps) & (l3 >= eps)
                   & (cc <= c_hi[t, None]) & (rr <= r_hi[t, None])
                   & (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H))
            if not ins.any():
                continue
            z = l1 * td[t, 0, None] + l2 * td[t, 1, None] + l3 * td[t, 2, None]
            np.maximum.at(buf, (rr * W + cc)[ins], z[ins])
    return buf.reshape(H, W)



def project_depth(mesh, elev_deg=90.0, azim_deg=0.0, roll_deg=0.0, px_mm=None, out_px=None, fill_holes=True):
    """
    PHÉP CHIẾU 3D -> 2D: chiếu trực giao mesh vào mặt phẳng ảnh và giữ bề mặt GẦN CAMERA NHẤT trên mỗi pixel
    (Z-buffer), theo góc nhìn bất kỳ.

    Rasterize tam giác (_raster_zbuffer), không splat điểm: độ sâu lấy ĐÚNG TẠI TÂM pixel từ tam giác cao nhất chứa
    tâm đó. Splat điểm (kể cả lấy mẫu ngẫu nhiên trên tam giác) có khe Poisson: ở pixel không nhận mẫu nào của mặt
    trên, một lớp bề mặt nằm DƯỚI (scan thường có lớp trong / mặt chồng) thắng Z-buffer -> hố giả sâu hàng chục mm.
    Rasterize mọi tam giác, KHÔNG lọc theo hướng pháp tuyến: scan hay có tam giác lật pháp tuyến ngay trên mặt ngoài,
    lọc sẽ để lọt lớp dưới; còn mặt sau / lớp trong bị che đúng bởi phép lấy max.

    px_mm: bước lưới ảnh. None (và out_px None) -> độ phân giải GỐC của scan = khoảng cách đỉnh trung bình trên phần
      bề mặt quay về camera; mịn hơn thế chỉ là nội suy trên tam giác phẳng, không thêm thông tin.
    out_px: nếu đặt (và px_mm None) -> px_mm = cạnh dài của ảnh / out_px.

    Trả dict:
      depth [mm] : độ cao so với điểm xa nhất, theo trục nhìn (lớn = gần camera)
      valid      : tâm pixel nằm trên bề mặt (lỗ kim <= 2 px của scan được lấp bằng phép đóng);
                   False = nền ngoài khối, lỗ thủng lớn của scan, hoặc bị CHE KHUẤT thật.
      px_mm      : bước lưới ảnh (đều theo cả hai trục vì mặt phẳng ảnh vuông góc trục nhìn)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces)
    s, u, f = view_basis(elev_deg, azim_deg, roll_deg)
    X, Y, D = V @ s, V @ u, V @ (-f)       # D: độ sâu dọc trục nhìn, lớn hơn = gần camera hơn
    if px_mm is None:
        if out_px:
            px_mm = max(X.max() - X.min(), Y.max() - Y.min()) / float(out_px)
        else:
            toward = np.asarray(mesh.face_normals) @ (-f)
            vis = toward > 0.0
            proj_area = float((np.asarray(mesh.area_faces)[vis] * toward[vis]).sum())
            px_mm = math.sqrt(max(proj_area, 1e-12) / max(len(np.unique(F[vis])) if vis.any() else len(V), 1))
    px_mm = float(px_mm)

    # N tâm pixel cách đều bước px_mm phủ (N-1)*px_mm -> round, KHÔNG ceil (ceil cộng dư 1 hàng vì sai số
    # dấu phẩy động).
    W = max(int(round((X.max() - X.min()) / px_mm)) + 1, 2)
    H = max(int(round((Y.max() - Y.min()) / px_mm)) + 1, 2)
    buf = _raster_zbuffer((X - X.min()) / px_mm, (Y.max() - Y) / px_mm, D, F, W, H)   # row 0 ở Y lớn nhất

    hit = np.isfinite(buf)
    k3 = np.ones((3, 3), np.uint8)                # phép đóng: lấp lỗ kim, không nở vùng
    valid = cv2.erode(cv2.dilate(hit.astype(np.uint8), k3), k3).astype(bool)
    if fill_holes and (~hit).any():
        _, (ri, ci) = distance_transform_edt(~hit, return_indices=True)
        buf = buf[ri, ci]
    else:
        buf = np.where(hit, buf, buf[hit].min() if hit.any() else 0.0)
    return {"depth": np.ascontiguousarray((buf - buf.min()).astype(np.float32)),
            "valid": valid, "px_mm": px_mm}
