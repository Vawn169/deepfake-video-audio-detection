"""Tinh truoc diem moc MediaPipe face_landmarker (478 diem, toa do pixel) cho moi anh mat THAT FF++ c23 train,
dung cho SBI (src/sbi.py). Anh khong tim duoc diem moc: khong tao SBI (giu la anh that).
  .venv/bin/python scripts/sbi_landmarks.py"""
import os, sys
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from dataset import FFPPFrameDataset

OUT = os.path.join(ROOT, 'data', 'processed', 'sbi_landmarks_train.npz')
os.chdir(ROOT)
ds = FFPPFrameDataset(split='train')
paths = [p for p, y, _ in ds.samples if y == 0]
det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=os.path.join(ROOT, 'models', 'face_landmarker.task')), num_faces=1))
keep, lms = [], []
for i, p in enumerate(paths):
    img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    if res.face_landmarks:
        keep.append(p); lms.append(np.array([[q.x * w, q.y * h] for q in res.face_landmarks[0]], np.float32))
    if i % 2000 == 0:
        print(f'{i}/{len(paths)}', flush=True)
np.savez_compressed(OUT, paths=np.array(keep), lms=np.stack(lms).astype(np.float16))
print(f'Co diem moc: {len(keep)}/{len(paths)} anh that train -> {OUT}')
