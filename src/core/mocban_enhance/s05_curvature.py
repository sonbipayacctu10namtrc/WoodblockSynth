"""
mocban_enhance/s05_curvature.py — Phương pháp 3: độ cong Monge chính xác.
"""
import math

import numpy as np

from .s03_derivatives import derivatives


# ----------------------------------------------------------------------------
# 5. Độ cong (Monge patch chính xác)
# ----------------------------------------------------------------------------

def curvature(h, px_mm, sigma_px=1.5):
    """
    Độ cong CHÍNH XÁC cho mặt Monge z = h(x,y). KHÔNG dùng xấp xỉ Laplacian.

        W = sqrt(1 + p^2 + q^2),   p = hx, q = hy, r = hxx, s = hxy, t = hyy
        K = (r*t - s^2) / W^4                                   [1/mm^2]
        H = ((1+q^2)*r - 2*p*q*s + (1+p^2)*t) / (2 * W^3)       [1/mm]
        k1, k2 = H +- sqrt(max(H^2 - K, 0))

    Vì sao không dùng Laplacian: H ~ (1/2)*lap(h) chỉ đúng khi độ dốc << 1. Ở đây độ dốc
    mép nét đạt ~2.4 nên W^3 ~ 17.6 (còn ~5.9 sau khi làm trơn 1 px). Laplacian thổi phồng
    |H| 6-18 lần ĐÚNG NGAY MÉP NÉT - nơi chứa toàn bộ tín hiệu - rồi bước chuẩn hoá
    percentile sẽ nghiền phần còn lại thành xám. Công thức đúng chỉ tốn thêm 3 convolution.

    DẤU: với n = (-p,-q,1)/W hướng lên, nét NỔI là cực đại địa phương nên H < 0.

    shape_index  S = (2/pi)*atan(H / sqrt(H^2 - K))  thuộc [-1,1], bất biến tỉ lệ.
    curvedness   C = sqrt(2*H^2 - K)                 độ lớn không dấu [1/mm].
    """
    d = derivatives(h, px_mm, sigma_px)
    p, q = d["hx"], d["hy"]
    r, s, t = d["hxx"], d["hxy"], d["hyy"]

    W2 = 1.0 + p * p + q * q
    W = np.sqrt(W2)
    K = (r * t - s * s) / (W2 * W2)
    H = ((1.0 + q * q) * r - 2.0 * p * q * s + (1.0 + p * p) * t) / (2.0 * W2 * W)

    disc = np.sqrt(np.maximum(H * H - K, 0.0))  # float32 làm H^2-K âm nhẹ ở vùng phẳng -> NaN
    k1, k2 = H + disc, H - disc
    shape_index = (2.0 / math.pi) * np.arctan(H / (disc + 1e-12))
    curvedness = np.sqrt(np.maximum(2.0 * H * H - K, 0.0))
    return {"H": H.astype(np.float32), "K": K.astype(np.float32),
            "k1": k1.astype(np.float32), "k2": k2.astype(np.float32),
            "shape_index": shape_index.astype(np.float32),
            "curvedness": curvedness.astype(np.float32)}



def normal_curvature_dir(h, px_mm, ldir, sigma_px=1.5):
    """
    Độ cong pháp tuyến dọc phương vị của đèn - thành phần "versatile" của Radiance Scaling.
    u = normalize(ldir_xy);  k_l = (ux^2*hxx + 2*ux*uy*hxy + uy^2*hyy) / W^3
    """
    d = derivatives(h, px_mm, sigma_px)
    u = np.asarray(ldir, dtype=np.float64)[:2]
    nrm = float(np.linalg.norm(u))
    ux, uy = (u / nrm) if nrm > 1e-9 else (1.0, 0.0)
    W2 = 1.0 + d["hx"] ** 2 + d["hy"] ** 2
    return ((ux * ux * d["hxx"] + 2.0 * ux * uy * d["hxy"] + uy * uy * d["hyy"])
            / (W2 * np.sqrt(W2))).astype(np.float32)
