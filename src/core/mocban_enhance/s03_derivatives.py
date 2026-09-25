"""
mocban_enhance/s03_derivatives.py — Đạo hàm Gauss (kernel ép đúng trên đa thức) và pháp tuyến.
"""
import math

import cv2
import numpy as np
from scipy.ndimage import correlate1d


# ----------------------------------------------------------------------------
# 3. Đạo hàm & pháp tuyến
# ----------------------------------------------------------------------------

def _dkernel(sigma_px, order, truncate=4.0):
    """
    Nhân Gauss đạo hàm 1-D, ÉP chính xác trên đa thức.

    KHÔNG dùng thẳng scipy gaussian_filter(order=2): scipy lấy mẫu đạo hàm giải tích của
    Gauss rồi CHỈ chuẩn hoá nhân bậc 0, nên nhân bậc 2 không tổng bằng 0. Phần dư (~1e-4)
    nhân với thành phần một chiều của trường độ cao rồi rò thẳng vào kết quả: đo trên bán cầu
    R = 46 mm sai số H lên tới 138% và K tới 467%. Nói cách khác toán tử đó KHÔNG bất biến
    tịnh tiến, trong khi đạo hàm buộc phải bất biến.

    Ở đây ép hai điều kiện, đúng bằng cấu trúc:
        sum(k) = 0                      -> triệt tiêu hằng số
        sum(k * i^order) / order! = 1   -> đúng chính xác trên đa thức bậc <= order
    """
    r = max(int(truncate * float(sigma_px) + 0.5), int(order) + 1)
    x = np.arange(-r, r + 1, dtype=np.float64)
    sig = float(sigma_px)
    g = np.exp(-x * x / (2.0 * sig * sig))
    g /= g.sum()
    if order == 0:
        return g
    k = (-x / sig ** 2) * g if order == 1 else ((x * x - sig * sig) / sig ** 4) * g
    k = k - k.mean()                                        # sum(k) = 0
    k = k / ((k * x ** order).sum() / math.factorial(order))  # đúng trên x^order
    return k



def _sepfilt(h, sigma_px, order_row, order_col):
    """Lọc tách được: bậc order_row dọc trục 0 (row), bậc order_col dọc trục 1 (col)."""
    out = correlate1d(h, _dkernel(sigma_px, order_row), axis=0, mode="nearest")
    return correlate1d(out, _dkernel(sigma_px, order_col), axis=1, mode="nearest")



def derivatives(h, px_mm, sigma_px=1.0):
    """
    Đạo hàm Gauss của trường độ cao. Nhân KHÔNG chia bước lưới nên phải chia px_mm bằng tay.

    Trả hx, hy (vô thứ nguyên) và hxx, hxy, hyy (1/mm), với x = col, y = row.
    """
    h = np.asarray(h, dtype=np.float64)
    s = float(px_mm)
    sig = float(sigma_px) if sigma_px and sigma_px > 0 else 1e-6
    f = lambda a, b: _sepfilt(h, sig, a, b).astype(np.float32)
    return {"hx": f(0, 1) / s, "hy": f(1, 0) / s,
            "hxx": f(0, 2) / (s * s), "hyy": f(2, 0) / (s * s), "hxy": f(1, 1) / (s * s)}



def normals(h, px_mm, sigma_px=1.0, gain=1.0):
    """
    Pháp tuyến đơn vị trong FRAME ẢNH: n = normalize(-gain*hx, -gain*hy, 1).

    gain < 1 nén độ dốc. Cần thiết ở đây: nét khắc cao 1.5 mm với mép chỉ ~1.6 px nên độ dốc
    nhảy 0 -> ~2.4, normal map thô sẽ ra nền phẳng một màu cộng viền 2 px cháy sáng.
    """
    d = derivatives(h, px_mm, sigma_px)
    g = float(gain)
    n = np.stack([-g * d["hx"], -g * d["hy"], np.ones_like(d["hx"])], axis=-1)
    return (n / np.linalg.norm(n, axis=-1, keepdims=True)).astype(np.float32)



def normal_map_rgb(n):
    """Pháp tuyến -> RGB uint8 theo quy ước (n+1)/2. Không kéo giãn percentile."""
    return np.clip((n + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)



def slope_aspect_hsv(h, px_mm, sigma_px=1.0):
    """
    Độ dốc/hướng dốc mã hoá HSV: hue = hướng dốc, value = độ lớn dốc.
    Trên nét khắc CJK đọc rõ hơn hẳn normal map RGB vì mắt phân tách nét theo hướng.
    """
    d = derivatives(h, px_mm, sigma_px)
    mag = np.hypot(d["hx"], d["hy"])
    ref = float(np.percentile(mag, 95)) or 1.0
    hue = (np.degrees(np.arctan2(d["hy"], d["hx"])) % 360.0) / 2.0        # OpenCV: 0..179
    val = np.tanh(mag / ref)
    hsv = np.stack([hue, np.full_like(hue, 255.0), val * 255.0], -1).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
