"""Hanh vi cua API voi file tu tao (khong can dataset). Can checkpoint va file mo hinh."""
import subprocess
import numpy as np, soundfile as sf
import pytest
from conftest import needs_models

pytestmark = needs_models


def _ffmpeg(*args):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True)


@pytest.fixture(scope='module')
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp('media')
    _ffmpeg('-f', 'lavfi', '-i', 'testsrc=size=320x240:rate=25', '-t', '2', '-pix_fmt', 'yuv420p', str(d / 'noface_noaudio.mp4'))
    rng = np.random.RandomState(0)
    t = np.arange(32000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.randn(len(t))).astype('float32')
    sf.write(d / 'mono16k.wav', tone, 16000)
    _ffmpeg('-i', str(d / 'mono16k.wav'), '-ac', '2', '-ar', '44100', str(d / 'stereo44k.wav'))
    (d / 'broken.wav').write_bytes(rng.bytes(5000))
    (d / 'audio.mp3').write_bytes(b'not really mp3')
    return d


def _post(client, url, path):
    with open(path, 'rb') as f:
        return client.post(url, files={'file': (path.name, f)})


def test_health(client):
    r = client.get('/health').json()
    assert r['status'] == 'ok'
    assert r['model_version'].startswith('effnet_b0-aug_s44')
    assert r['audio_model_version'].startswith('w2v2base-L2-mlp-audio_s43')


def test_video_without_face_is_not_scored(client, media):
    r = _post(client, '/predict?heatmap=false', media / 'noface_noaudio.mp4').json()
    assert r['status'] == 'no_reliable_face'
    assert r['score'] is None and r['label'] is None


def test_video_without_audio_track(client, media):
    r = _post(client, '/predict_audio', media / 'noface_noaudio.mp4').json()
    assert r['status'] == 'no_audio' and r['source'] == 'video'
    assert r['score'] is None


def test_audio_16k_mono_read_directly(client, media):
    r = _post(client, '/predict_audio', media / 'mono16k.wav').json()
    assert r['status'] == 'ok' and r['converted_by_ffmpeg'] is False
    assert r['label'] in ('spoof', 'bonafide') and 0.0 <= r['score'] <= 1.0
    assert r['label'] == ('spoof' if r['logit'] >= r['threshold_logit'] else 'bonafide')


def test_audio_other_rate_goes_through_ffmpeg(client, media):
    r = _post(client, '/predict_audio', media / 'stereo44k.wav').json()
    assert r['status'] == 'ok' and r['converted_by_ffmpeg'] is True


def test_unsupported_extension_rejected(client, media):
    assert _post(client, '/predict_audio', media / 'audio.mp3').status_code == 415


def test_broken_audio_rejected(client, media):
    assert _post(client, '/predict_audio', media / 'broken.wav').status_code == 422
