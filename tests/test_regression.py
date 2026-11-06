"""Diem da ghi nhan truoc day phai giu nguyen (reports/NEXT_STEPS.md, reports/api_parity_audio.log).
Can dataset tren may, nen tu bo qua neu thieu file."""
import os
import pytest
from conftest import ROOT, needs_models

pytestmark = needs_models

FF = os.path.join(ROOT, 'data', 'ff')
VIDEO_CASES = [  # duong dan, diem API da ghi o Tuan 6
    ('original_sequences/youtube/c23/videos/862.mp4', 0.072),
    ('original_sequences/youtube/c23/videos/186.mp4', 0.0303),
    ('manipulated_sequences/FaceSwap/c23/videos/280_249.mp4', 0.9999),
]
LA_EVAL = os.path.join(ROOT, 'data', 'asvspoof2019_LA', 'LA', 'ASVspoof2019_LA_eval', 'flac')
AUDIO_CASES = [  # logit API da ghi o Tuan 8
    ('LA_E_5450657', -15.7081),
    ('LA_E_9497931', 9.6337),
    ('LA_E_8992933', 2.4978),
]


@pytest.mark.parametrize('rel,expected', VIDEO_CASES)
def test_known_video_scores(client, rel, expected):
    path = os.path.join(FF, rel)
    if not os.path.exists(path):
        pytest.skip(f'thieu {rel}')
    with open(path, 'rb') as f:
        r = client.post('/predict?heatmap=false', files={'file': (os.path.basename(path), f)}).json()
    assert r['status'] == 'ok'
    assert abs(r['score'] - expected) <= 0.005


@pytest.mark.parametrize('utt,expected', AUDIO_CASES)
def test_known_audio_logits(client, utt, expected):
    path = os.path.join(LA_EVAL, utt + '.flac')
    if not os.path.exists(path):
        pytest.skip(f'thieu {utt}')
    with open(path, 'rb') as f:
        r = client.post('/predict_audio', files={'file': (utt + '.flac', f)}).json()
    assert r['status'] == 'ok'
    assert abs(r['logit'] - expected) <= 0.01
