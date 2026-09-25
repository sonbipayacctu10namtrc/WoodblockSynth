"""
mocban_enhance/s02_height.py — Chuẩn bị trường độ cao (làm trơn) và thống kê độ sâu nét.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

if not hasattr(np, "infty"):
    np.infty = np.inf


# ----------------------------------------------------------------------------
# 2. Chuẩn bị trường độ cao
# ----------------------------------------------------------------------------

def prepare_height(h, px_mm, sigma0_px=1.0, detrend_sigma_px=None):
    """
    Chính quy hoá bắt buộc trước mọi đạo hàm bậc hai.

    Ảnh độ sâu của scan là mặt ghép từ các tam giác PHẲNG (liên tục C0 nhưng không C1) cộng nhiễu
    đo của máy scan / photogrammetry ở thang dưới mm. Đạo hàm bậc hai của mặt ghép tam giác là dãy
    xung trên cạnh tam giác -> bản đồ độ cong / MSII hiện lưới tam giác và hạt nhiễu. Đó là tính
    chất của dữ liệu scan, không phải lỗi cài đặt. Làm trơn sigma0 ~ 1 px xoá phần này mà gần như
    không đụng tới nét khắc (rộng nhiều px).

    detrend_sigma_px: nếu đặt, trừ đi nền thấp tần (khối gỗ cong / vênh).
    """
    h = np.ascontiguousarray(np.asarray(h, dtype=np.float32))
    if detrend_sigma_px:
        h = h - gaussian_filter(h, float(detrend_sigma_px), mode="nearest")
    if sigma0_px and sigma0_px > 0:
        h = gaussian_filter(h, float(sigma0_px), mode="nearest")
    return h



def relief_stats(h, px_mm):
    """Vài con số chi phối mọi lựa chọn tham số. In ra đầu notebook."""
    p1, p99 = np.percentile(h, [1, 99])
    relief_mm = float(p99 - p1)
    return {"shape": tuple(h.shape), "px_mm": float(px_mm), "relief_mm": relief_mm,
            "relief_px": relief_mm / float(px_mm),
            "h_min_mm": float(h.min()), "h_max_mm": float(h.max())}
