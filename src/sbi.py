"""Self-Blended Images (Shiohara & Yamasaki, CVPR 2022), tu viet lai cho V-SBI (docs/plan_v3_faceswap.md muc 9,
docs/protocol.md muc 2026-11-15). Khong dung code goc (giay phep chi nghien cuu phi thuong mai, ban sua doi thuoc ve tac gia);
chi lay gia tri tham so tu bai bao va repo chinh thuc (doc 15/11):
  - bien doi nguon/dich (ap vao nguon hoac dich, moi ben 0,5): RGB shift +-20 (p 0,3); HSV +-0,3 (don vi uint8 nhu code goc);
    do sang/tuong phan +-0,1; mot trong hai: thu nho 2 hoac 4 lan (nearest) roi phong lai (linear), hoac lam net alpha 0,2-0,5, lightness 0,5-1,0;
  - nguon va mat na: affine ti le [0,95; 1/0,95] (doc lap ngang/doc), dich +-3% ngang, +-1,5% doc; bien dang dan hoi alpha 50, sigma 7;
  - mat na: bao loi diem moc (MediaPipe face_landmarker, 478 diem); xac suat 0,25 bo vung tran (tuong tu dung 68 diem thay 81 diem);
    doi kich thuoc mat na ngau nhien 192-256 moi chieu, lam mo lan 1 (kernel le 5-25, sigma 0), chuan hoa, gia tri <1 -> 0,
    lam mo lan 2 (kernel le 5-25, sigma 5-45), chuan hoa, doi ve kich thuoc anh; ti le tron r trong {0,25; 0,5; 0,75; 1; 1; 1}.
  - anh SBI = mask * nguon + (1 - mask) * dich, nhan gia.
Dung random cua Python (DataLoader seed rieng moi worker)."""
import random
import cv2
import numpy as np

BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]   # diem long may MediaPipe


def _rng(r):
    return np.random.default_rng(r.getrandbits(32))


def color_freq(img, r):
    x = img.astype(np.float32)
    if r.random() < 0.3:
        x += np.array([r.uniform(-20, 20) for _ in range(3)], np.float32)
    x = np.clip(x, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = np.mod(hsv[..., 0] + r.uniform(-0.3, 0.3), 180)
    hsv[..., 1:] += np.array([r.uniform(-0.3, 0.3), r.uniform(-0.3, 0.3)], np.float32)
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    x = x * (1 + r.uniform(-0.1, 0.1)) + r.uniform(-0.1, 0.1) * 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    h, w = x.shape[:2]
    if r.random() < 0.5:
        k = r.choice([2, 4])
        x = cv2.resize(cv2.resize(x, (w // k, h // k), interpolation=cv2.INTER_NEAREST), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        a, l = r.uniform(0.2, 0.5), r.uniform(0.5, 1.0)
        keep = np.zeros((3, 3), np.float32); keep[1, 1] = 1
        eff = -np.ones((3, 3), np.float32); eff[1, 1] = 8 + l
        x = np.clip(cv2.filter2D(x.astype(np.float32), -1, (1 - a) * keep + a * eff), 0, 255).astype(np.uint8)
    return x


def affine_elastic(src, mask, r):
    h, w = mask.shape
    sx, sy = r.uniform(0.95, 1 / 0.95), r.uniform(0.95, 1 / 0.95)
    tx, ty = r.uniform(-0.03, 0.03) * w, r.uniform(-0.015, 0.015) * h
    M = np.array([[sx, 0, (1 - sx) * w / 2 + tx], [0, sy, (1 - sy) * h / 2 + ty]], np.float32)
    src = cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    g = _rng(r)
    dx = cv2.GaussianBlur(g.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 7) * 50
    dy = cv2.GaussianBlur(g.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), 7) * 50
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + dx, gy + dy
    src = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.remap(mask, mx, my, cv2.INTER_LINEAR, borderValue=0)
    return src, mask


def hull_mask(lm, h, w, r):
    pts = lm
    if r.random() < 0.25:
        pts = lm[lm[:, 1] >= lm[BROWS, 1].min()]
    m = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1.0)
    return m


def blend_mask(mask, r):
    h, w = mask.shape
    m = cv2.resize(mask, (r.randint(192, 256), r.randint(192, 256)))
    k1, k2 = r.randrange(5, 26, 2), r.randrange(5, 26, 2)
    m = cv2.GaussianBlur(m, (k1, k1), 0)
    if m.max() <= 0:
        return None
    m = m / m.max(); m[m < 1] = 0
    m = cv2.GaussianBlur(m, (k2, k2), r.randint(5, 45))
    if m.max() <= 0:
        return None
    m = cv2.resize(m / m.max(), (w, h))
    return m * r.choice([0.25, 0.5, 0.75, 1, 1, 1])


def self_blend(img, lm, r=random):
    """img: RGB uint8 HxWx3; lm: (478,2) toa do pixel. Tra ve anh SBI uint8, hoac None neu mat na rong."""
    h, w = img.shape[:2]
    src, dst = img.copy(), img.copy()
    if r.random() < 0.5:
        src = color_freq(src, r)
    else:
        dst = color_freq(dst, r)
    mask = hull_mask(lm, h, w, r)
    src, mask = affine_elastic(src, mask, r)
    m = blend_mask(mask, r)
    if m is None:
        return None
    m = m[..., None]
    return np.clip(m * src + (1 - m) * dst, 0, 255).astype(np.uint8)
