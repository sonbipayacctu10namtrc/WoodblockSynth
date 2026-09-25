"""
mocban_enhance/s07_ao.py — Phương pháp 5: ambient occlusion (horizon mapping).
"""
import math

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 7. Ambient Occlusion (horizon mapping)
# ----------------------------------------------------------------------------

def _base_maps(shape):
    H, W = shape
    base_y, base_x = np.mgrid[0:H, 0:W]
    return base_y.astype(np.float32), base_x.astype(np.float32)



def _shift(h, dx_px, dy_px, mapx, mapy, base_x, base_y):
    """Lấy mẫu h tại offset (dx,dy) px. BORDER_REPLICATE, KHÔNG clip toạ độ."""
    np.add(base_x, np.float32(dx_px), out=mapx)
    np.add(base_y, np.float32(dy_px), out=mapy)
    return cv2.remap(h, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)



def horizon_tan(h, px_mm, az_rad, radius_px=24, n_steps=12, log_steps=True):
    """
    tan của góc chân trời dọc một phương vị:
        tan(theta_h) = max_{t in (0,R]} ( h(p + t*u) - h(p) ) / t,   t tính bằng mm, kẹp >= 0.
    Bước log vì phát hiện chân trời vô hướng tỉ lệ -> 12 bước log đủ thay 24 bước đều.
    """
    h = np.ascontiguousarray(np.asarray(h, dtype=np.float32))
    base_y, base_x = _base_maps(h.shape)
    mapx, mapy = np.empty_like(base_x), np.empty_like(base_y)
    ca, sa = math.cos(az_rad), math.sin(az_rad)

    if log_steps:
        ks = np.unique(np.round(np.geomspace(1.0, float(radius_px), n_steps)))
    else:
        ks = np.linspace(1.0, float(radius_px), n_steps)

    best = np.zeros_like(h)
    for k in ks:
        hs = _shift(h, k * ca, k * sa, mapx, mapy, base_x, base_y)
        np.maximum(best, (hs - h) / (float(k) * float(px_mm)), out=best)
    return best



def horizon_ao(h, px_mm, n_az=16, radius_px=24, n_steps=12, log_steps=True):
    """
    Ambient occlusion theo chân trời, tích phân bán cầu có trọng số cosine.

    Với mỗi phương vị, dải cực nhìn thấy là [0, pi/2 - theta_h] và
    tích_phan cos(psi)sin(psi) dpsi = (1/2)cos^2(theta_h), nên

        AO = trung_binh_phi [ cos^2(theta_h) ] = trung_binh_phi [ 1 / (1 + tan^2 theta_h) ]

    Không cần arctan. Kiểm chứng: mặt phẳng -> AO = 1; chân tường đứng -> AO = 0.5.
    1 = hở hoàn toàn, 0 = bị che hoàn toàn.

    Nâng cấp khả dĩ (không cài ở đây): GTAO (Jimenez 2016) có trọng số max(0, n.omega) cho
    mặt tiếp tuyến nghiêng; ở dữ liệu này chỉ đổi phần vách nét.
    """
    acc = None
    for i in range(n_az):
        tan_h = horizon_tan(h, px_mm, 2.0 * math.pi * i / n_az, radius_px, n_steps, log_steps)
        term = 1.0 / (1.0 + tan_h * tan_h)
        acc = term if acc is None else acc + term
    return (acc / float(n_az)).astype(np.float32)
