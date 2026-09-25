"""
mocban_enhance/s08_shading.py — Phương pháp 6: exaggerated shading, radiance scaling, Lambert đối chứng.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from .s05_curvature import curvature, normal_curvature_dir
from .s03_derivatives import normals
from .s09_display import _interior


# ----------------------------------------------------------------------------
# 8. Exaggerated Shading & Radiance Scaling
# ----------------------------------------------------------------------------

def light_dir(elev_deg=35.0, azim_deg=135.0):
    """Hướng đèn đơn vị trong frame ảnh (x = col, y = row, z hướng ra ngoài mặt)."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)],
                    dtype=np.float32)



def lambert(h, px_mm, ldir, sigma_px=1.0, gain=1.0):
    """Lambert thuần - ĐỐI CHỨNG. Không có nó thì không chứng minh được 6a/6b thắng."""
    n = normals(h, px_mm, sigma_px, gain)
    return np.clip((n * np.asarray(ldir, np.float32)[None, None, :]).sum(-1), 0, 1).astype(np.float32)



def exaggerated_shading(h, px_mm, ldir, sigma0_px=1.0, n_scales=4, gain=0.4, contrast=6.0):
    """
    Exaggerated Shading theo tinh thần Rusinkiewicz et al. 2006 (không phải port nguyên bản).

    Cơ chế gốc: dựng kim tự tháp pháp tuyến từ hình học làm trơn dần; ở mỗi tỉ lệ đèn được
    tham chiếu lại về gần phương tiếp tuyến của tỉ lệ THÔ HƠN, nên cái được tô bóng là phần
    CHI TIẾT mà tỉ lệ đó thêm vào so với tỉ lệ thô hơn, rồi gộp nhân qua các tỉ lệ:

        sigma_i = sigma0 * 2^i;   n_i từ G_{sigma_i} * h;   s_i = l . n_i
        S_i = clip( 0.5 + contrast * (s_i - s_{i+1}), 0, 1 )     <- chi tiết của tỉ lệ i
        S   = ( tich_i S_i ) ^ (1/L)

    Lấy HIỆU giữa hai tỉ lệ liên tiếp chính là hiện thực của bước tham chiếu lại đèn: nó bỏ
    đi thành phần bóng thô (vốn phụ thuộc hướng đèn và làm mất chi tiết ở góc chiếu thấp) và
    giữ lại đúng phần nổi/chìm ở tỉ lệ đang xét.

    LƯU Ý: bản trước dùng tỉ số S_i/(G*S_i) làm chuẩn hoá tương phản; nó dồn mọi giá trị về
    ~0.5 (đo được dải [0.44, 0.53] trên tấm thật) nên panel ra xám phẳng. Dạng hiệu + hệ số
    contrast cho dải gần kín [0,1].
    """
    ldir = np.asarray(ldir, dtype=np.float32)
    h32 = np.asarray(h, dtype=np.float32)
    shades = []
    for i in range(int(n_scales) + 1):
        sig = float(sigma0_px) * (2.0 ** i)
        n_i = normals(gaussian_filter(h32, sig, mode="nearest"), px_mm,
                      sigma_px=max(sig, 1e-6), gain=gain)
        shades.append((n_i * ldir[None, None, :]).sum(-1))

    prod = np.ones_like(h32)
    for i in range(int(n_scales)):
        prod *= np.clip(0.5 + float(contrast) * (shades[i] - shades[i + 1]), 0, 1)
    return np.power(np.maximum(prod, 0.0), 1.0 / float(n_scales)).astype(np.float32)



def radiance_scaling(h, px_mm, ldir, alpha=3.0, sigma_px=1.5, kappa="directional",
                     kref_pct=95, gain=1.0):
    """
    Radiance Scaling (Vergne et al., I3D 2010):  L'(p) = sigma(kappa_bar(p)) * L(p).

    Hàm scaling là hàm hữu tỉ (Moebius) xác định bởi ba tính chất:
        sigma(0) = 1,   sigma(+1) = alpha,   sigma(-1) = 1/alpha,   đơn điệu ở giữa
    tức lồi được làm sáng lên alpha lần, lõm bị tối đi alpha lần (đối xứng trên thang log):

        sigma(kb) = ((alpha+1) + (alpha-1)*kb) / ((alpha+1) - (alpha-1)*kb)

    (Ba tính chất trên là phần quyết định hành vi; đại số nguyên văn trong bài báo chưa được
    đối chiếu ở đây, nên markdown nên trích ba tính chất thay vì khẳng định công thức gốc.)

    Ánh xạ độ cong là BẮT BUỘC vì kappa thô không bị chặn:
        kappa_bar = (2/pi) * atan(kappa / kappa_ref),  kappa_ref = percentile(|kappa|, 95)
    Tự chuẩn theo percentile làm alpha thành núm ổn định giữa các tấm khác nhau.

    kappa="directional" dùng độ cong pháp tuyến dọc phương vị đèn (tạo hiệu ứng đặc trưng
    "gờ hướng về đèn sáng lên"); "mean" dùng độ cong trung bình H.
    """
    k = (normal_curvature_dir(h, px_mm, ldir, sigma_px) if kappa == "directional"
         else curvature(h, px_mm, sigma_px)["H"])
    kref = float(np.percentile(np.abs(_interior(k, 8)), kref_pct)) or 1.0
    kb = (2.0 / math.pi) * np.arctan(k / kref)
    a = float(alpha)
    sig = ((a + 1.0) + (a - 1.0) * kb) / ((a + 1.0) - (a - 1.0) * kb)
    return np.clip(sig * lambert(h, px_mm, ldir, sigma_px, gain), 0, 1).astype(np.float32)
