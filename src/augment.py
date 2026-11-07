import io
import random
import numpy as np
from PIL import Image


def degrade(img, rng=random):
    """Mo phong nen manh: giam/phong do phan giai, JPEG, nhieu Gauss. rng co .random/.uniform/.randint/.getrandbits."""
    w, h = img.size
    if rng.random() < 0.5:
        s = rng.uniform(0.5, 1.0)
        small = img.resize((max(32, int(w * s)), max(32, int(h * s))), Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=rng.randint(20, 90))
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    if rng.random() < 0.5:
        sigma = rng.uniform(1.0, 6.0)
        arr = np.asarray(img).astype(np.float32)
        noise = np.random.default_rng(rng.getrandbits(32)).normal(0.0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return img


class RandomDegrade:
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img):
        return degrade(img, random) if random.random() < self.p else img


def social_degrade(img, rng=random):
    """Mo phong video mang xa hoi (docs/protocol.md, muc 2026-11-08): 1-3 bien doi ngau nhien."""
    from PIL import ImageDraw, ImageEnhance, ImageFilter, ImageFont
    import cv2
    w, h = img.size
    ops = rng.sample(['jpeg2', 'scale', 'smooth', 'text', 'color'], rng.randint(1, 3))
    for op in ops:
        if op == 'jpeg2':
            for _ in range(2):
                buf = io.BytesIO(); img.save(buf, 'JPEG', quality=rng.randint(30, 80)); buf.seek(0)
                img = Image.open(buf).convert('RGB')
        elif op == 'scale':
            s = rng.uniform(0.35, 0.7)
            img = img.resize((max(16, int(w * s)), max(16, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        elif op == 'smooth':
            if rng.random() < 0.5:
                d = rng.choice([5, 7, 9])
                img = Image.fromarray(cv2.bilateralFilter(np.asarray(img), d, 40, 40))
            else:
                img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
        elif op == 'text':
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default(size=rng.randint(14, 28))
            txt = ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz0123456789 #@') for _ in range(rng.randint(4, 14)))
            x, y = rng.randint(0, max(1, w - 60)), rng.randint(0, max(1, h - 30))
            fill = (255, 255, 255) if rng.random() < 0.5 else (255, 230, 0)
            draw.text((x, y), txt, font=font, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0))
        else:
            for enh in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
                img = enh(img).enhance(rng.uniform(0.8, 1.2))
    return img


class RandomSocialDegrade:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        return social_degrade(img, random) if random.random() < self.p else img
