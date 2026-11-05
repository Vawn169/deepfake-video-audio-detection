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
