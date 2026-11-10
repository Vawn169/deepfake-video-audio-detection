"""Tien xu ly audio dung chung cho huan luyen (audio_v2.py), danh gia (eval_de24.py) va API (api_core.py)."""
import numpy as np

SR, MAXLEN, FRAME = 16000, 64000, 320   # 16 kHz, 4 giay, khung 20 ms


def trim(w):
    """Cat khung dau va cuoi co nang luong thap hon khung lon nhat qua 40 dB (docs/protocol.md, muc 2026-11-08)."""
    n = len(w) // FRAME
    if n < 2:
        return w
    en = 10 * np.log10(np.mean(w[:n * FRAME].reshape(n, FRAME) ** 2, axis=1) + 1e-12)
    keep = np.where(en > en.max() - 40)[0]
    return w[keep[0] * FRAME:(keep[-1] + 1) * FRAME]
