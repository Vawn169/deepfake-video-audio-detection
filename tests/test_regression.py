"""Diem da ghi nhan truoc day phai giu nguyen (reports/NEXT_STEPS.md, reports/api_parity_audio.log).
Can dataset tren may, nen tu bo qua neu thieu file."""
import os
import pytest
from conftest import ROOT, needs_models

pytestmark = needs_models

FF = os.path.join(ROOT, 'data', 'ff')
VIDEO_CASES = [  # diem API cua vb_s44 (16/11; aug_s44 cu: 0.072, 0.0303, 0.9999). 186, 280_249 trung diem offline;
    # 862: API loc landmark con 7/16 frame nen khac offline (0.691)
    ('original_sequences/youtube/c23/videos/862.mp4', 0.5309),
    ('original_sequences/youtube/c23/videos/186.mp4', 0.0017),
    ('manipulated_sequences/FaceSwap/c23/videos/280_249.mp4', 0.9976),
]
LA_EVAL = os.path.join(ROOT, 'data', 'asvspoof2019_LA', 'LA', 'ASVspoof2019_LA_eval', 'flac')
AUDIO_CASES = [  # logit cua checkpoint audio_v2c_s44, tinh offline float32 tren CPU (reports/api_parity_audio_v2c.log, 17/11; v2b_s43 cu: -7.7271, 18.4858, -16.1579)
    ('LA_E_5450657', -7.9807),
    ('LA_E_9497931', 30.7765),
    ('LA_E_8992933', -16.8093),
]


@pytest.mark.parametrize('rel,expected', VIDEO_CASES)
def test_known_video_scores(client, rel, expected):
    path = os.path.join(FF, rel)
    if not os.path.exists(path):
        pytest.skip(f'thieu {rel}')
    with open(path, 'rb') as f:
        r = client.post('/predict?heatmap=false', files={'file': (os.path.basename(path), f)}).json()
    assert r['status'] == 'ok' and r['verdict'] in ('high_suspicion', 'no_sign_detected')
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
