"""
mocban_enhance/s04_depth.py — Phương pháp 1: bản đồ độ sâu cục bộ.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

from .s09_display import _interior_percentile


# ----------------------------------------------------------------------------
# 4. Bản đồ độ sâu
# ----------------------------------------------------------------------------

def depth_map(h, px_mm, pct=(1, 99), hp_sigma_px=12.0):
    """
    Ba dạng của cùng một trường độ cao:
      mm      : thô, đơn vị vật lý (panel duy nhất có colorbar mm thật)
      stretch : kéo giãn percentile robust, [0,1]
      local   : h - G_sigma * h, khử nền thấp tần. Đây mới là bản lộ nét khắc khi tấm
                cong hoặc nứt, nên dùng bản này trong lưới so sánh.
    """
    h = np.asarray(h, dtype=np.float32)
    lo, hi = _interior_percentile(h, pct, margin_px=8)
    stretch = np.clip((h - lo) / max(hi - lo, 1e-9), 0, 1)
    local = h - gaussian_filter(h, float(hp_sigma_px), mode="nearest")
    return {"mm": h, "stretch": stretch.astype(np.float32), "local": local.astype(np.float32),
            "lo_mm": float(lo), "hi_mm": float(hi)}
