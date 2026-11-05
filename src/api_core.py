import base64, os, subprocess, sys, time
import cv2, numpy as np, soundfile as sf, torch, torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from preprocess import make_detector, crop_face
from dataset import get_transforms
from model import build_model

CKPT = os.path.join(ROOT, 'checkpoints_aug_s44', 'best_effnet_b0.pt')
THRESH = 0.5951          # EER tren FF++ Val cua aug_s44 (reports/eval/checkpoints_aug_s44/summary.csv)
N_FRAMES, MARGIN, SIZE = 16, 0.3, 224
MIN_FRAMES = 4           # so frame co mat toi thieu de cham diem (lua chon cua nguoi viet, chua kiem chung)

AUDIO_CKPT = os.path.join(ROOT, 'checkpoints_audio_s43', 'best_mlp.pt')   # dev EER thap nhat (docs/protocol.md)
W2V_DIR = os.path.join(ROOT, 'models', 'wav2vec2-base')                   # chep tu cache HF, khong tai khi chay
SR, MAXLEN = 16000, 64000   # giong audio_features.py: 4 giay dau, khong pad
MIN_SAMPLES = 400           # wav2vec2 can toi thieu 400 mau (25 ms) de ra 1 frame
AUDIO_EXT = {'.wav', '.flac'}
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}


class Engine:
    def __init__(self, num_threads=4):
        torch.set_num_threads(num_threads)
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(ck['model_state_dict'])
        self.model.eval()
        self.version = f"effnet_b0-aug_s44-epoch{ck['epoch']}"
        self.tf = get_transforms(train=False)
        self.detector = make_detector(os.path.join(ROOT, 'models', 'blaze_face_short_range.tflite'))
        self.landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=os.path.join(ROOT, 'models', 'face_landmarker.task')),
            num_faces=1))

    def extract_frames(self, path):
        """Giong preprocess.process_video nhung giu trong bo nho; JPEG 95 de khop anh luc huan luyen."""
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return [], 0
        idxs = [int(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]
        out, read = [], 0
        for k, idx in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            read += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.detections:
                continue
            best = max(res.detections, key=lambda d: d.categories[0].score)
            face = crop_face(frame, best, MARGIN, SIZE)
            if face is None:
                continue
            _, buf = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out.append((k, cv2.imdecode(buf, cv2.IMREAD_COLOR)))
        cap.release()
        return out, read

    def has_face(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        return bool(r.face_landmarks)

    def _tensor(self, bgr):
        return self.tf(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))

    def gradcam(self, bgr):
        store = {}
        h = self.model.features[-1].register_forward_hook(lambda m, i, o: store.update(act=o))
        x = self._tensor(bgr).unsqueeze(0)
        with torch.enable_grad():
            logit = self.model(x).squeeze(1)
            act = store['act']
            grad = torch.autograd.grad(logit[0], act)[0][0]
        h.remove()
        w = grad.mean(dim=(1, 2))
        cam = torch.relu((w[:, None, None] * act[0].detach()).sum(0)).numpy()
        cam = cv2.resize(cam / (cam.max() + 1e-8), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.resize(bgr, (SIZE, SIZE)), 0.55, heat, 0.45, 0)
        _, buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buf.tobytes()).decode()

    def predict(self, path, verify_face=True, heatmap=True):
        t0 = time.time()
        frames, read = self.extract_frames(path)
        n_detected = len(frames)
        if verify_face:
            frames = [(k, f) for k, f in frames if self.has_face(f)]
        base = {'model_version': self.version, 'threshold': THRESH,
                'frames_read': read, 'frames_with_detection': n_detected, 'frames_used': len(frames)}
        if len(frames) < MIN_FRAMES:
            base.update(status='no_reliable_face', score=None, label=None,
                        note=f'Duoi {MIN_FRAMES} frame co khuon mat hop le, khong cham diem.',
                        processing_seconds=round(time.time() - t0, 3))
            return base
        x = torch.stack([self._tensor(f) for _, f in frames])
        with torch.no_grad():
            probs = torch.sigmoid(self.model(x).squeeze(1)).numpy()
        score = float(probs.mean())
        best = int(np.argmax(probs))
        base.update(status='ok', score=round(score, 4), label='fake' if score >= THRESH else 'real',
                    frame_scores=[{'frame': int(k), 'prob_fake': round(float(p), 4)} for (k, _), p in zip(frames, probs)],
                    best_frame=int(frames[best][0]),
                    note='Ket qua chi mang tinh tham khao.')
        if heatmap:
            base['heatmap_jpeg_base64'] = self.gradcam(frames[best][1])
        base['processing_seconds'] = round(time.time() - t0, 3)
        return base


class AudioEngine:
    def __init__(self):
        from transformers import Wav2Vec2Model
        c = torch.load(AUDIO_CKPT, map_location='cpu', weights_only=False)
        self.mlp = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))
        self.mlp.load_state_dict(c['state'])
        self.mlp.eval()
        self.layer, self.thresh = int(c['layer']), float(c['threshold'])   # nguong EER tren dev, tinh tren logit
        self.mean, self.std = c['mean'].astype(np.float32), c['std'].astype(np.float32)
        self.w2v = Wav2Vec2Model.from_pretrained(W2V_DIR).eval()
        self.version = f"w2v2base-L{self.layer}-mlp-audio_s43-epoch{c['epoch']}"

    @staticmethod
    def _ffmpeg(path):
        """Tach/chuyen ve mono 16 kHz float32. Tra None neu file khong co track am thanh."""
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index',
                                '-of', 'csv=p=0', path], capture_output=True, text=True)
        if probe.returncode != 0:
            raise ValueError('ffprobe khong doc duoc file')
        if not probe.stdout.strip():
            return None
        r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR),
                            '-t', str(MAXLEN / SR), '-f', 'f32le', 'pipe:1'], capture_output=True)
        if r.returncode != 0:
            raise ValueError('ffmpeg khong tach duoc am thanh')
        return np.frombuffer(r.stdout, dtype=np.float32)

    def load_wave(self, path, ext):
        """File wav/flac mono 16 kHz doc truc tiep nhu luc danh gia; con lai qua ffmpeg."""
        if ext in AUDIO_EXT:
            info = sf.info(path)
            if info.samplerate == SR and info.channels == 1:
                w, _ = sf.read(path, dtype='float32', frames=MAXLEN)
                return w, 'audio_file', False
            return self._ffmpeg(path), 'audio_file', True
        return self._ffmpeg(path), 'video', True

    def predict(self, path, ext):
        t0 = time.time()
        w, source, converted = self.load_wave(path, ext)
        base = {'model_version': self.version, 'threshold_logit': round(self.thresh, 4),
                'source': source, 'converted_by_ffmpeg': converted}
        if w is None or len(w) < MIN_SAMPLES:
            base.update(status='no_audio', score=None, label=None,
                        audio_seconds_used=0.0 if w is None else round(len(w) / SR, 3),
                        note='Khong co am thanh (hoac qua ngan), khong cham diem.',
                        processing_seconds=round(time.time() - t0, 3))
            return base
        w = w[:MAXLEN]
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
        with torch.inference_mode():
            hs = self.w2v(torch.from_numpy(w).unsqueeze(0), output_hidden_states=True).hidden_states
            x = (hs[self.layer][0].mean(0).numpy() - self.mean) / self.std
            logit = float(self.mlp(torch.from_numpy(x).unsqueeze(0)).squeeze())
        base.update(status='ok', score=round(float(1 / (1 + np.exp(-logit))), 4), logit=round(logit, 4),
                    label='spoof' if logit >= self.thresh else 'bonafide',
                    audio_seconds_used=round(len(w) / SR, 3),
                    note='Chi danh gia tren ASVspoof 2019 LA; am thanh ngoai mien nay chi mang tinh tham khao.',
                    processing_seconds=round(time.time() - t0, 3))
        return base
