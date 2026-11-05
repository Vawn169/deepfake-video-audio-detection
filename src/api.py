import os, tempfile, threading
from fastapi import FastAPI, File, HTTPException, UploadFile
from api_core import AUDIO_EXT, VIDEO_EXT, AudioEngine, Engine

MAX_MB = 100
app = FastAPI(title='Deepfake face detector', version='0.2')
engine, audio_engine, lock = None, None, threading.Lock()


@app.on_event('startup')
def _load():
    global engine, audio_engine
    engine = Engine()
    audio_engine = AudioEngine()


def _save_upload(file, default_suffix):
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f'File qua {MAX_MB} MB')
    suffix = os.path.splitext(file.filename or '')[1].lower() or default_suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return tmp.name, suffix


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': engine.version, 'audio_model_version': audio_engine.version}


@app.post('/predict')
def predict(file: UploadFile = File(...), verify_face: bool = True, heatmap: bool = True):
    path, _ = _save_upload(file, '.mp4')
    try:
        with lock:  # model va detector khong an toan khi chay song song
            return engine.predict(path, verify_face=verify_face, heatmap=heatmap)
    finally:
        os.remove(path)


@app.post('/predict_audio')
def predict_audio(file: UploadFile = File(...)):
    path, ext = _save_upload(file, '')
    try:
        if ext not in AUDIO_EXT | VIDEO_EXT:
            raise HTTPException(415, f'Chi nhan {sorted(AUDIO_EXT | VIDEO_EXT)}')
        with lock:
            try:
                return audio_engine.predict(path, ext)
            except (ValueError, RuntimeError) as e:  # sf.info nem RuntimeError (LibsndfileError) khi file hong
                raise HTTPException(422, f'Khong doc duoc am thanh: {e}')
    finally:
        os.remove(path)
