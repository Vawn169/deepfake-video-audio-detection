import os, sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

VIDEO_CKPT = os.path.join(ROOT, 'checkpoints_aug_s44', 'best_effnet_b0.pt')
AUDIO_CKPT = os.path.join(ROOT, 'checkpoints_audio_s43', 'best_mlp.pt')
MODEL_FILES = [os.path.join(ROOT, 'models', f) for f in
               ('blaze_face_short_range.tflite', 'face_landmarker.task', 'wav2vec2-base/model.safetensors')]
HAS_MODELS = all(os.path.exists(p) for p in [VIDEO_CKPT, AUDIO_CKPT] + MODEL_FILES)

needs_models = pytest.mark.skipif(not HAS_MODELS, reason='thieu checkpoint hoac file mo hinh (khong co trong repo cong khai)')


@pytest.fixture(scope='session')
def client():
    if not HAS_MODELS:
        pytest.skip('thieu checkpoint hoac file mo hinh')
    from fastapi.testclient import TestClient
    from api import app
    with TestClient(app) as c:  # "with" de chay su kien startup (nap mo hinh)
        yield c
