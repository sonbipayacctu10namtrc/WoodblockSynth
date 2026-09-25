"""
mocban_render/s06_post.py — Hậu kỳ giả camera (apply_post) và vật che 2D (add_occluders).
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# 6. Hậu kỳ giả camera
# ----------------------------------------------------------------------------

def apply_post(rgb: np.ndarray, post: dict, rng: random.Random, min_mean: float = 0.14) -> np.ndarray:
    img = rgb.astype(np.float32) / 255.0
    exposure = post.get("exposure", 0.0)
    # auto-exposure tối thiểu: máy ảnh thật sẽ tự tăng ISO/phơi sáng, không cho ảnh gần đen hoàn toàn
    mean = float(img.mean()) * (2.0 ** exposure)
    if mean < min_mean:
        exposure += math.log2(min_mean / max(mean, 1e-4))
        post["exposure_applied"] = exposure
    img = img * (2.0 ** exposure)
    img = img * np.array(post.get("wb_shift", [1, 1, 1]), dtype=np.float32)[None, None, :]
    img = np.clip(img, 0, 1) ** post.get("gamma", 1.0)
    v = post.get("vignette", 0.0)
    if v > 0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        img *= (1 - v * np.clip(r, 0, 1.2) ** 2)[..., None]
    b = post.get("blur_sigma", 0.0)
    if b > 0.05:
        img = cv2.GaussianBlur(img, (0, 0), b)
    s = post.get("noise_sigma", 0.0)
    if s > 0:  # Poisson-Gaussian: vùng tối nhiễu mạnh hơn (ISO cao), thêm nhiễu màu theo kênh ở vùng tối
        nrng = np.random.default_rng(rng.randrange(1 << 30))
        lum = img.mean(-1, keepdims=True)
        sig = s * (0.6 + 1.2 * np.sqrt(np.clip(1 - lum, 0, 1)))
        img = img + nrng.normal(0, 1, img.shape).astype(np.float32) * sig
        img = img + nrng.normal(0, s * 0.4, (1, 1, 3)).astype(np.float32) * (lum < 0.3)
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    q = int(post.get("jpeg_q", 95))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)





# ----------------------------------------------------------------------------
# 6b. Che khuất bởi vật ngoài (tầng 2D, sau render)
# ----------------------------------------------------------------------------

def add_occluders(img: np.ndarray, rng: random.Random, prob: float = 0.5) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Thêm vật che thường gặp khi chụp mộc bản: ngón tay/bàn tay, thước, nhãn băng dính, lóa flash, bóng người chụp.
    Trả về (ảnh, mask che 0/1 HxW, danh sách loại đã thêm). Lóa/bóng không tính vào mask che (bề mặt vẫn thấy được một phần).
    """
    h, w = img.shape[:2]
    out = img.astype(np.float32) / 255
    occ = np.zeros((h, w), np.float32)
    kinds: list[str] = []
    if rng.random() >= prob:
        return img, occ, kinds
    n = rng.choice([1, 1, 2])
    for _ in range(n):
        k = rng.choices(["finger", "ruler", "tape", "glare", "shadow"], weights=[3, 2, 2, 2, 2])[0]
        kinds.append(k)
        layer = np.zeros((h, w), np.float32)
        if k == "finger":  # 1-2 ngón từ mép ảnh
            side = rng.choice(["l", "r", "b"])
            col = np.array(rng.choice([[0.85, 0.65, 0.5], [0.72, 0.5, 0.36], [0.55, 0.36, 0.25]]), np.float32)
            for j in range(rng.choice([1, 2])):
                fw = int(rng.uniform(0.05, 0.09) * w); fl = int(rng.uniform(0.2, 0.45) * h)
                if side == "b":
                    cx = int(rng.uniform(0.15, 0.85) * w) + j * int(fw * 1.3)
                    cv2.ellipse(layer, (cx, h), (fw // 2, fl), 0, 0, 360, 1.0, -1)
                else:
                    cy = int(rng.uniform(0.2, 0.8) * h) + j * int(fw * 1.3); cx = 0 if side == "l" else w
                    cv2.ellipse(layer, (cx, cy), (fl, fw // 2), 0, 0, 360, 1.0, -1)
            shade = cv2.GaussianBlur(layer, (0, 0), 6) * 0.35   # bóng mềm quanh ngón
        elif k == "ruler":  # thước vàng/trắng nằm dọc hoặc ngang gần mép
            t = int(rng.uniform(0.05, 0.09) * min(h, w))
            vertical = rng.random() < 0.5
            pos = int(rng.uniform(0.02, 0.15) * (w if vertical else h))
            if vertical:
                layer[:, pos:pos + t] = 1
            else:
                layer[pos:pos + t, :] = 1
            col = np.array(rng.choice([[0.95, 0.85, 0.2], [0.95, 0.95, 0.9]]), np.float32)
            shade = np.zeros_like(layer)
        elif k == "tape":  # nhãn băng dính xanh/trắng
            tw, th_ = int(rng.uniform(0.08, 0.18) * w), int(rng.uniform(0.03, 0.06) * h)
            x, y = int(rng.uniform(0, w - tw)), int(rng.uniform(0, h - th_))
            layer[y:y + th_, x:x + tw] = 1
            col = np.array(rng.choice([[0.3, 0.55, 0.9], [0.95, 0.95, 0.92]]), np.float32)
            shade = np.zeros_like(layer)
        elif k == "glare":  # lóa flash: đốm trắng bão hoà mềm (không tính che)
            cx, cy = int(rng.uniform(0.2, 0.8) * w), int(rng.uniform(0.2, 0.8) * h)
            r = int(rng.uniform(0.12, 0.3) * min(h, w))
            cv2.circle(layer, (cx, cy), r, 1.0, -1)
            g = cv2.GaussianBlur(layer, (0, 0), r * 0.5)[..., None] * rng.uniform(0.5, 0.9)
            out = np.clip(out + g, 0, 1)
            continue
        else:  # bóng người chụp: dải tối mềm (không tính che)
            x0 = int(rng.uniform(-0.2, 0.6) * w); x1 = x0 + int(rng.uniform(0.3, 0.7) * w)
            pts = np.array([[x0, 0], [x1, 0], [x1 + int(0.2 * w), h], [x0 + int(0.2 * w), h]], np.int32)
            cv2.fillPoly(layer, [pts], 1.0)
            d = cv2.GaussianBlur(layer, (0, 0), 25)[..., None] * rng.uniform(0.3, 0.6)
            out = out * (1 - d)
            continue
        a = layer[..., None]
        out = out * (1 - a) + col[None, None, :] * a
        out = out * (1 - shade[..., None])
        occ = np.maximum(occ, layer)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), occ, kinds
