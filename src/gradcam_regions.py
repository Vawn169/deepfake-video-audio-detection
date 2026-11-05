import csv, os, sys
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

OUT = 'data/gradcam_out'
ROOT = 'data/processed/ffpp_c23/test'
LIPS = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
LEYE = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
REYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
NOSE = [168,129,358,98,327,2,94,1]

det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path='models/face_landmarker.task'), num_faces=1))
rows = list(csv.DictReader(open(f'{OUT}/summary.csv')))
cams = np.load(f'{OUT}/cams.npy').astype(np.float32)
assert len(rows) == len(cams)
os.makedirs(f'{OUT}/masks', exist_ok=True)

def hull(pts, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts.astype(np.int32)), 1)
    return m.astype(bool)

res, nolm = [], []
for i, (r, cam) in enumerate(zip(rows, cams)):
    vid = r['video_id']
    img = cv2.imread(os.path.join(ROOT, r['method'], vid, r['best_frame']))
    img = cv2.resize(img, (224, 224))
    out = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not out.face_landmarks:
        nolm.append(f"{r['method']}_{vid}"); continue
    pts = np.array([[p.x * 224, p.y * 224] for p in out.face_landmarks[0]])
    face = hull(pts, (224, 224))
    feat = np.zeros_like(face)
    for idx in (LIPS, LEYE, REYE, NOSE):
        feat |= hull(pts[idx], (224, 224))
    feat &= face
    regions = {'mat_mui_mieng': feat, 'mat_con_lai': face & ~feat, 'ngoai_mat': ~face}
    tot = cam.sum() + 1e-8
    d = {'method': r['method'], 'video_id': vid, 'label': r['label'], 'prob_mean': r['prob_mean']}
    for k, m in regions.items():
        d[f'nang_luong_{k}'] = round(float(cam[m].sum() / tot), 4)
        d[f'dien_tich_{k}'] = round(float(m.mean()), 4)
    res.append(d)
    if i % 5 == 0 or r['video_id'] in ('724', '507'):
        vis = img.copy(); vis[feat] = (0, 255, 0); vis[~face] = vis[~face] // 3
        cv2.imwrite(f"{OUT}/masks/{r['method']}_{vid}.jpg", vis)

with open(f'{OUT}/region_stats.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
print(f"Phan tich duoc {len(res)}/{len(rows)} anh. Khong thay landmark: {nolm}")
import pandas as pd
d = pd.DataFrame(res)
cols = [c for c in d.columns if c.startswith('nang_luong') or c.startswith('dien_tich')]
print(d.groupby('method')[cols].mean().round(3).to_string())
sai = d[((d.label == '0') & (d.prob_mean.astype(float) >= .5)) | ((d.label == '1') & (d.prob_mean.astype(float) < .5))]
print("\nCa sai:\n", sai.to_string())
