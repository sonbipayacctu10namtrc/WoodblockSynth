"""
mocban_enhance/s06_msii.py — Phương pháp 4: bất biến tích phân đa tỉ lệ (MSII).
"""
import math

import cv2
import numpy as np

from .s07_ao import _base_maps, _shift


# ----------------------------------------------------------------------------
# 6. Bất biến tích phân đa tỉ lệ (MSII)
# ----------------------------------------------------------------------------

def _disk_kernel(r_px):
    r = max(int(round(r_px)), 1)
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = ((x * x + y * y) <= r * r).astype(np.float32)
    return k / float(k.sum())



def msii_volume(h, px_mm, radii_px=(3, 6, 12, 24, 48), exact=False, n_rho=6, n_phi=12):
    """
    Bất biến tích phân đa tỉ lệ (Mara & Kroemker), bản trường độ cao.

    Định nghĩa gốc: v_r(p) = Vol(B_r(p) giao Solid) / ((4/3)*pi*r^3). Mặt phẳng cho đúng
    v_r = 1/2. Khai triển Pottmann Vol = (2*pi/3)r^3 - (pi/4)*H_out*r^4 + O(r^5) với H_out
    lấy theo pháp tuyến NGOÀI của khối (lồi -> dương), quy về quy ước Monge của module này
    (H âm trên phần nổi, tức H = -H_out):

        v_r - 1/2 = (3/16) * H * r + O(r^2)            [đã kiểm chứng số: sai lệch 0.5%]

    tức v_r là ước lượng độ cong trung bình có tham số tỉ lệ. Đây là lý do DoG "cùng họ" -
    nhưng DoG thiếu mốc 1/2, thiếu chuẩn hoá 1/r và không bị chặn nên nổ ở vách đứng rồi
    nuốt hết dải percentile.

    Với trường độ cao, tích phân là chính xác:
        v_r = 1/2 + 3/(4*pi*r^3) * TichPhan_{rho<=r} clip(d, -c(rho), +c(rho)) dA
    với d = h(x,y) - h(p) và c(rho) = sqrt(r^2 - rho^2) là nửa chiều cao cột cầu.

    exact=False (mặc định): bỏ clip -> tích phân thành MỘT convolution đĩa:
        v_r ~ 1/2 + 3/(4*r_mm) * (trung_binh_dia_r(h) - h)
    Clip chỉ bất hoạt khi r_mm >~ 2*relief_mm (~7.8 px ở cấu hình chuẩn), nên r <= 6 px bão
    hoà và chỉ dùng định tính. Luôn kẹp [0,1] vì tích phân thật bảo đảm điều đó.

    exact=True: cầu phương cực n_rho x n_phi mẫu, có clip. Đắt hơn ~n lần nhưng đúng ở mọi r.

    Trả mảng HxWxR trong [0,1].
    """
    h = np.ascontiguousarray(np.asarray(h, dtype=np.float32))
    out = np.empty(h.shape + (len(radii_px),), dtype=np.float32)

    for j, r_px in enumerate(radii_px):
        r_mm = float(r_px) * float(px_mm)
        if not exact:
            mean_r = cv2.filter2D(h, -1, _disk_kernel(r_px), borderType=cv2.BORDER_REPLICATE)
            v = 0.5 + (3.0 / (4.0 * r_mm)) * (mean_r - h)
        else:
            acc = np.zeros_like(h)
            base_y, base_x = _base_maps(h.shape)
            mapx, mapy = np.empty_like(base_x), np.empty_like(base_y)
            d_rho, d_phi = r_px / n_rho, 2.0 * math.pi / n_phi
            for i in range(n_rho):
                rho_px = (i + 0.5) * d_rho
                c_mm = math.sqrt(max(r_mm * r_mm - (rho_px * px_mm) ** 2, 0.0))
                w = rho_px * d_rho * d_phi * (px_mm ** 2)   # rho*drho*dphi, đổi sang mm^2
                for m in range(n_phi):
                    phi = m * d_phi
                    hs = _shift(h, rho_px * math.cos(phi), rho_px * math.sin(phi),
                                mapx, mapy, base_x, base_y)
                    acc += np.clip(hs - h, -c_mm, c_mm) * w
            v = 0.5 + acc * (3.0 / (4.0 * math.pi * r_mm ** 3))
        out[..., j] = np.clip(v, 0.0, 1.0)
    return out



def msii_combine(V, radii_px, px_mm):
    """
    combined    : tổng (v_j - 1/2)/r_j  -> bản đồ đa tỉ lệ đưa vào lưới so sánh
    bands       : v_j - v_{j+1}, tách đặc trưng theo dải tỉ lệ
    scale_argmax: tỉ lệ có |v - 1/2| lớn nhất -> bản đồ "tỉ lệ đặc trưng" (colormap phân loại)
    """
    dev = V - 0.5
    r_mm = np.asarray(radii_px, dtype=np.float32) * float(px_mm)
    combined = (dev / r_mm[None, None, :]).sum(axis=-1)
    bands = dev[..., :-1] - dev[..., 1:]
    return {"combined": combined.astype(np.float32), "bands": bands.astype(np.float32),
            "scale_argmax": np.argmax(np.abs(dev), axis=-1).astype(np.int32)}
