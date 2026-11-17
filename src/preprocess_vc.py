"""Cat mat cho V-C (docs/protocol.md muc 2026-11-17): MAVOS-DD train/validation, LAV-DF train/dev.
16 frame, MediaPipe, le 0,3, 224, JPEG 95 (nhu preprocess.py). Rieng LAV-DF train co modify_video: 16 frame lay deu
TRONG cac doan gia (fake_periods); con lai lay deu ca video. Ket qua data/processed/vc/<split>/<method>/<video_id>/f??.jpg,
log reports/preprocess_vc.csv (dinh dang nhu preprocess_vb.csv, doc bang FFPPFrameDataset). Chay lai duoc (bo qua video xong).
  .venv/bin/python src/preprocess_vc.py [--workers 4]"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import cv2
import mediapipe as mp
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import make_detector, crop_face

ROOT = Path(__file__).resolve().parent.parent
OUT, LOG = ROOT / 'data' / 'processed' / 'vc', ROOT / 'reports' / 'preprocess_vc.csv'
MAVOS, LAVDF = Path('/mnt/d/datasets/mavos_dd'), Path('/mnt/d/datasets/lavdf/extracted/LAV-DF')
N, MARGIN, SIZE = 16, 0.3, 224


def rows():
    out = []
    for split, fn in (('train', 'mavos_train_subset.csv'), ('val', 'mavos_validation_subset.csv')):
        for r in pd.read_csv(ROOT / 'configs' / fn).itertuples():
            vid = 'm_' + r.video_path[:-4].replace('/', '__')
            out.append((split, f'mavos-{r.generative_method}', int(r.video_fake), vid, str(MAVOS / r.video_path), None))
    for split, fn in (('train', 'lavdf_train_subset.csv'), ('val', 'lavdf_dev_subset.csv')):
        for r in pd.read_csv(ROOT / 'configs' / fn).itertuples():
            kind = {(False, False): 'real', (True, False): 'v', (False, True): 'a', (True, True): 'av'}[(r.modify_video, r.modify_audio)]
            per = json.loads(r.fake_periods) if (split == 'train' and r.modify_video) else None
            out.append((split, f'lavdf-{kind}', int(r.modify_video), 'l_' + r.file[:-4].replace('/', '_'), str(LAVDF / r.file), per))
    return out


def frame_idxs(total, fps, periods):
    if not periods:
        return [int(i * (total - 1) / (N - 1)) for i in range(N)]
    segs = [(max(0, int(a * fps)), min(total - 1, int(b * fps))) for a, b in periods]
    segs = [(a, b) for a, b in segs if b >= a]
    L = sum(b - a + 1 for a, b in segs)
    if L == 0:
        return [int(i * (total - 1) / (N - 1)) for i in range(N)]
    pos = [int(i * (L - 1) / (N - 1)) for i in range(N)]   # vi tri deu tren chuoi frame gia ghep lai
    idxs = []
    for p in pos:
        for a, b in segs:
            if p <= b - a:
                idxs.append(a + p); break
            p -= b - a + 1
    return idxs


det = None


def work(r):
    global det
    split, method, label, vid, path, per = r
    odir = OUT / split / method / vid
    if (odir / '_done').exists():
        return None
    if det is None:
        det = make_detector(str(ROOT / 'models' / 'blaze_face_short_range.tflite'))
    t0, read, wf, err = time.time(), 0, 0, ''
    try:
        cap = cv2.VideoCapture(path)
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            err = 'khong doc duoc so frame'
        else:
            odir.mkdir(parents=True, exist_ok=True)
            for k, idx in enumerate(frame_idxs(total, fps, per)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                read += 1
                res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                if not res.detections:
                    continue
                face = crop_face(frame, max(res.detections, key=lambda d: d.categories[0].score), MARGIN, SIZE)
                if face is None:
                    continue
                cv2.imwrite(str(odir / f'f{k:02d}.jpg'), face, [cv2.IMWRITE_JPEG_QUALITY, 95]); wf += 1
        cap.release()
    except Exception as e:
        err = repr(e)[:200]
    if not err:
        (odir / '_done').touch()
    return [split, method, label, vid, read, wf, f'{time.time() - t0:.2f}', err]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    rs = rows()
    if a.limit:
        rs = rs[::max(1, len(rs) // a.limit)][:a.limit]
    miss = [r[4] for r in rs if not os.path.exists(r[4])]
    if miss:
        sys.exit(f'Thieu {len(miss)} video, vd {miss[:3]}')
    new = not LOG.exists()
    fh = open(LOG, 'a', newline=''); w = csv.writer(fh)
    if new:
        w.writerow(['split', 'method', 'label', 'video_id', 'frames_read', 'frames_with_face', 'seconds', 'error'])
    t0, n = time.time(), 0
    with Pool(a.workers) as p:
        for res in p.imap_unordered(work, rs, chunksize=4):
            if res is None:
                continue
            w.writerow(res); fh.flush(); n += 1
            if n % 500 == 0:
                print(f'{n}/{len(rs)} | {n / (time.time() - t0):.2f} video/s', flush=True)
    print(f'Xong {n} video moi trong {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
