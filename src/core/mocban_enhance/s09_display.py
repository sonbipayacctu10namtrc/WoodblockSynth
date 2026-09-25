"""
mocban_enhance/s09_display.py — Chuẩn hoá hiển thị: đưa mọi kết quả về ảnh theo một chính sách duy nhất.
"""
import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 9. Chuẩn hoá hiển thị
# ----------------------------------------------------------------------------

def _interior(x, margin_px=8):
    m = int(margin_px)
    return x[m:-m, m:-m] if (m > 0 and x.shape[0] > 2 * m and x.shape[1] > 2 * m) else x



def _interior_percentile(x, pct, margin_px=8):
    xi = _interior(np.asarray(x, dtype=np.float32), margin_px)
    return (float(np.percentile(xi, pct[0])), float(np.percentile(xi, pct[1])))



def norm01(x, sym=False, pct=(2, 98), margin_px=8, vrange=None):
    """
    Chuẩn hoá về [0,1] theo một chính sách duy nhất cho cả notebook.

    sym=True: đối xứng quanh 0, thang a = percentile(|x|, pct[1]) -> 0 luôn rơi vào 0.5.
    vrange=(lo,hi): khoá thang thủ công. BẮT BUỘC dùng khi sweep tham số, nếu không mỗi panel
    tự chuẩn hoá và cả dãy sweep chẳng cho thấy gì.

    KHÔNG gọi hàm này cho đại lượng đã bị chặn sẵn (AO, MSII v_r, mọi shading): kéo giãn
    chúng sẽ phá mốc "1/2 = phẳng" / "1 = hở", vốn là toàn bộ nội dung khoa học của chúng.
    """
    x = np.asarray(x, dtype=np.float32)
    if vrange is not None:
        lo, hi = float(vrange[0]), float(vrange[1])
    elif sym:
        a = float(np.percentile(np.abs(_interior(x, margin_px)), pct[1])) or 1.0
        lo, hi = -a, a
    else:
        lo, hi = _interior_percentile(x, pct, margin_px)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)



def colorize(x01, cmap="gray"):
    """[0,1] -> RGB uint8 qua colormap matplotlib. Dùng coolwarm cho đại lượng có dấu."""
    x01 = np.clip(np.asarray(x01, dtype=np.float32), 0, 1)
    if cmap in (None, "gray", "grey"):
        g = (x01 * 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=-1)
    from matplotlib import colormaps
    return (colormaps[cmap](x01)[..., :3] * 255).astype(np.uint8)



def _lab(img, text):
    """Nhãn ASCII. cv2.putText dùng font Hershey nên KHÔNG vẽ được dấu tiếng Việt."""
    img = np.ascontiguousarray(img)
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img
