"""Code cua giai doan nang cap v2: cat im lang, augmentation audio, SocialDegrade, khoa chong danh gia lai."""
import os, random, subprocess, sys
import numpy as np
import pytest
from PIL import Image
from conftest import ROOT

from audio_v2 import trim, FRAME
from augment import social_degrade


def test_trim_removes_leading_and_trailing_silence():
    rng = np.random.RandomState(0)
    speech = (0.3 * rng.randn(16000)).astype('float32')
    w = np.concatenate([np.zeros(8000, 'float32'), speech, 1e-4 * rng.randn(4000).astype('float32')])
    t = trim(w)
    assert abs(len(t) - len(speech)) <= 2 * FRAME
    assert np.array_equal(t[:FRAME], speech[:FRAME])  # 8000 mau im lang = 25 khung tron, nen cat dung dau tieng noi


def test_trim_keeps_short_or_loud_audio():
    w = (0.3 * np.random.RandomState(1).randn(400)).astype('float32')
    assert len(trim(w)) >= len(w) - FRAME


def test_social_degrade_deterministic_and_same_size():
    img = Image.fromarray((np.random.RandomState(2).rand(224, 224, 3) * 255).astype('uint8'))
    a = social_degrade(img.copy(), random.Random('x'))
    b = social_degrade(img.copy(), random.Random('x'))
    assert a.size == img.size
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.skipif(not os.path.isdir('/mnt/d/rirs_noises/RIRS_NOISES'), reason='thieu RIRS_NOISES')
def test_audio_augmenter_deterministic():
    from audio_v2 import Augmenter
    w = (0.1 * np.random.RandomState(3).randn(32000)).astype('float32')
    aug = Augmenter()
    for key in ('dev:a', 'dev:b', 'train:c'):
        x, y = aug(w.copy(), key), aug(w.copy(), key)
        assert np.array_equal(x, y) and x.dtype == np.float32 and len(x) > 0


@pytest.mark.parametrize('script,args', [
    ('src/audio_eval_itw.py', []),
    ('src/audio_v2.py', ['eval', '--config', 'a']),
    ('src/audio_v2.py', ['eval', '--config', 'b']),
])
def test_eval_scripts_refuse_second_run(script, args):
    ledger = os.path.join(ROOT, 'reports', 'eval_ledger.csv')
    if script == 'src/audio_eval_itw.py' and 'checkpoints_audio_s42/best_mlp.pt,In-the-Wild' not in open(ledger).read():
        pytest.skip('chua danh gia lan nao')
    if script == 'src/audio_v2.py' and f'checkpoints_audio_v2{args[-1]}_s42/best_mlp.pt,In-the-Wild' not in open(ledger).read():
        pytest.skip('chua danh gia lan nao')
    before = open(ledger).read()
    r = subprocess.run([sys.executable, os.path.join(ROOT, script), *args], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and 'Khong danh gia lai' in (r.stdout + r.stderr)
    assert open(ledger).read() == before
