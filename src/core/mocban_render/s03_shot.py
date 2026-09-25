"""
mocban_render/s03_shot.py — Mô tả một lần chụp (CameraSpec, LightSpec, ShotSpec) và lấy mẫu ngẫu nhiên theo 4 kiểu chụp.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# 3. Lấy mẫu camera / ánh sáng
# ----------------------------------------------------------------------------

@dataclass
class CameraSpec:
    elev_deg: float      # 90 = nhìn thẳng từ trên xuống
    azim_deg: float      # vị trí camera quanh khối (hướng phối cảnh)
    roll_deg: float      # xoay trong mặt phẳng ảnh: 0 = trục +Y của khối hướng lên, 90/180 = khối bị xoay trong khung
    dist_mm: float
    yfov_deg: float
    look_at: list[float]
    width: int
    height: int



@dataclass
class LightSpec:
    kind: str            # 'directional' | 'point' | 'spot'
    elev_deg: float
    azim_deg: float
    intensity: float
    color: list[float]
    dist_mm: float = 600.0



@dataclass
class ShotSpec:
    camera: CameraSpec
    lights: list[LightSpec]
    ambient: list[float]
    post: dict = field(default_factory=dict)



def _kelvin_rgb(k: float) -> list[float]:
    """Xấp xỉ màu nguồn sáng theo nhiệt độ màu (Kelvin) -> RGB [0..1]."""
    t = k / 100.0
    r = 255 if t <= 66 else 329.7 * ((t - 60) ** -0.1332)
    g = 99.47 * math.log(t) - 161.1 if t <= 66 else 288.1 * ((t - 60) ** -0.0755)
    b = 255 if t >= 66 else (0 if t <= 19 else 138.5 * math.log(t - 10) - 305.0)
    return [float(np.clip(v / 255, 0, 1)) for v in (r, g, b)]



def sample_shot(rng: random.Random, block_wh_mm: tuple[float, float], width=1024, height=768,
                preset: str = "mixed") -> ShotSpec:
    """
    preset:
      'topdown'  : chụp gần thẳng, đèn khuếch tán (như chụp lưu trữ)
      'handheld' : góc nghiêng, đèn flash gần camera hoặc đèn phòng
      'raking'   : đèn xiên thấp -> bóng dài, relief nổi rõ
      'mixed'    : chọn ngẫu nhiên các preset trên
    """
    if preset == "mixed":
        preset = rng.choices(["topdown", "handheld", "raking", "closeup"], weights=[0.25, 0.25, 0.2, 0.3])[0]
    bw, bh = block_wh_mm
    diag = math.hypot(bw, bh)

    # elev: 90 = nhìn thẳng từ trên; azim ưu tiên 4 hướng trục (khối gần thẳng trong khung) + jitter
    azim = rng.uniform(0, 360)
    # hướng khối trong khung: 80% gần thẳng (jitter nhỏ), 20% xoay 90/180/270 như ảnh chụp vội
    u = rng.random()
    if u < 0.8:
        roll = rng.gauss(0, 4)
    else:
        roll = rng.choice([90, 180, 270]) + rng.gauss(0, 4)
    if preset == "topdown":
        elev = rng.uniform(78, 90); yfov = rng.uniform(30, 45); fill = rng.uniform(0.75, 0.95)
    elif preset == "handheld":
        elev = rng.uniform(45, 78); yfov = rng.uniform(40, 65); fill = rng.uniform(0.7, 1.0)
    elif preset == "raking":
        elev = rng.uniform(55, 88); yfov = rng.uniform(32, 50); fill = rng.uniform(0.75, 1.0)
    else:  # closeup: chỉ thấy một phần khối
        elev = rng.uniform(60, 90); yfov = rng.uniform(30, 45); fill = rng.uniform(2.0, 4.0)
    # khoảng cách sao cho đường chéo khối chiếm ~fill lần chiều cao khung
    dist = (diag / fill) / (2 * math.tan(math.radians(yfov / 2)))
    if preset == "closeup":
        look = [rng.uniform(-bw * 0.35, bw * 0.35), rng.uniform(-bh * 0.35, bh * 0.35), 0.0]
    else:
        look = [rng.gauss(0, bw * 0.04), rng.gauss(0, bh * 0.04), 0.0]
    cam = CameraSpec(elev, azim, roll, dist, yfov, look, width, height)

    lights: list[LightSpec] = []
    kelvin = rng.choice([3000, 4000, 5000, 5600, 6500])
    col = _kelvin_rgb(kelvin)
    if preset == "topdown":
        lights.append(LightSpec("directional", rng.uniform(50, 80), rng.uniform(0, 360), rng.uniform(2.0, 4.0), col))
        ambient = [rng.uniform(0.25, 0.5)] * 3
    elif preset == "handheld":
        if rng.random() < 0.5:  # flash gần camera
            lights.append(LightSpec("point", elev + rng.uniform(-8, 8), azim + rng.uniform(-15, 15),
                                    rng.uniform(4e5, 9e5), col, dist_mm=dist * 0.95))
        else:                   # đèn phòng
            lights.append(LightSpec("directional", rng.uniform(35, 75), rng.uniform(0, 360), rng.uniform(1.5, 3.5), col))
        if rng.random() < 0.4:  # thêm nguồn phụ màu khác (cửa sổ)
            lights.append(LightSpec("directional", rng.uniform(20, 60), rng.uniform(0, 360), rng.uniform(0.5, 1.5), _kelvin_rgb(6500)))
        ambient = [rng.uniform(0.1, 0.35)] * 3
    elif preset == "raking":
        lights.append(LightSpec("directional", rng.uniform(10, 30), rng.uniform(0, 360), rng.uniform(3.0, 6.0), col))
        ambient = [rng.uniform(0.12, 0.3)] * 3
    else:  # closeup: đèn bất kỳ, thường có 2 nguồn
        lights.append(LightSpec("directional", rng.uniform(25, 80), rng.uniform(0, 360), rng.uniform(2.0, 4.5), col))
        if rng.random() < 0.5:
            lights.append(LightSpec("directional", rng.uniform(20, 60), rng.uniform(0, 360), rng.uniform(0.5, 1.5), _kelvin_rgb(rng.choice([3000, 6500]))))
        ambient = [rng.uniform(0.15, 0.4)] * 3

    post = {
        "exposure": rng.uniform(-0.4, 0.5),
        "gamma": rng.uniform(0.9, 1.15),
        "noise_sigma": rng.uniform(0.0, 0.03),
        "blur_sigma": max(0.0, rng.gauss(0.3, 0.4)),
        "vignette": rng.uniform(0.0, 0.35),
        "jpeg_q": int(rng.uniform(55, 95)),
        "wb_shift": [rng.uniform(0.95, 1.05), 1.0, rng.uniform(0.95, 1.05)],
        # màu mặt bàn: giấy trắng (như ảnh lưu trữ), gỗ, vải xám, nền tối
        "table_rgb": rng.choice([[0.92, 0.91, 0.88], [0.55, 0.42, 0.30], [0.45, 0.45, 0.47], [0.15, 0.15, 0.15], [0.75, 0.72, 0.66]]),
    }
    return ShotSpec(cam, lights, ambient, post | {"preset": preset, "kelvin": kelvin})



def frontal_shot(block_size_mm, width: int, height: int, fill: float = 0.9, yfov_deg: float = 35.0) -> ShotSpec:
    """Nhìn thẳng mặt +Z (ảnh: +X sang phải, +Y lên), đèn xiên 40° cố định, không hậu kỳ -> để soát mặt chính."""
    bw, bh, bz = block_size_mm
    dist = (math.hypot(bw, bh) / fill) / (2 * math.tan(math.radians(yfov_deg / 2))) + bz
    return ShotSpec(CameraSpec(90.0, 0.0, 0.0, dist, yfov_deg, [0.0, 0.0, 0.0], width, height),
                    [LightSpec("directional", 40.0, 135.0, 3.5, [1.0, 1.0, 1.0])], [0.3, 0.3, 0.3],
                    {"table_rgb": [0.9, 0.9, 0.88]})
