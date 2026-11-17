"""Test-C: doi mat kieu roop (docs/protocol.md, muc 2026-11-10 va 2026-11-15; docs/plan_v3_faceswap.md).
Viet lai loi doi mat cua s0md3v/roop (ban extension, scripts/swapper.py) cho tung frame video:
  - InsightFace 0.7.3 FaceAnalysis buffalo_l, det_size 640 (khong thay mat thi thu 320), lay mat ben trai nhat (face_index 0);
  - inswapper_128, INSwapper.get(paste_back=True) nhu ban goc; khong lam net mat, khong upscale;
  - kiem tra noi dung ifnude (diem > 0.7 nhu ban goc) tren 1 frame moi giay; video co frame bi gan co thi bo ca video.
Cap, anh mat nguon (25/50/75% thoi luong), ma hoa H.264 CRF 23: giong scripts/faceswap_gen.py.
Frame khong thay mat dich: giu nguyen frame (nhu ban goc). Video tao ra chi luu tren may, khong cong bo.
Chay bang moi truong rieng ~/faceswap/roop_full/.venv-roop (Python 3.12):
  LD_LIBRARY_PATH=... .venv-roop/bin/python scripts/roop_gen.py --split test --out /mnt/d/faceswap/vb"""
import argparse, csv, json, os, subprocess, time
import cv2
import numpy as np
import onnxruntime
import insightface
import ifnude.detector as ifd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF_REAL = os.path.join(ROOT, 'data', 'ff', 'original_sequences', 'youtube', 'c23', 'videos')
SWAP_MODEL = os.path.expanduser('~/faceswap/facefusion/.assets/models/inswapper_128.onnx')
PROV = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# ifnude tao lai phien ONNX moi lan goi detect(); giu mot phien (cung mo hinh, cung CPU) de khong nap lai 139 MB moi frame
_sess = onnxruntime.InferenceSession(ifd.model_path, providers=['CPUExecutionProvider'])
ifd.onnxruntime = type('ORT', (), {'InferenceSession': staticmethod(lambda *a, **k: _sess)})


def is_nsfw(bgr):
    return any(c['score'] > 0.7 for c in ifd.detect(bgr))


class Analyser:
    def __init__(self):
        self.apps = {}

    def face(self, img, det=640):
        if det not in self.apps:
            app = insightface.app.FaceAnalysis(name='buffalo_l', providers=PROV)
            app.prepare(ctx_id=0, det_size=(det, det))
            self.apps[det] = app
        f = self.apps[det].get(img)
        if not f and det > 320:
            return self.face(img, det // 2)
        return sorted(f, key=lambda x: x.bbox[0])[0] if f else None


def duration(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p], capture_output=True, text=True)
    return float(r.stdout.strip())


def source_face(an, vid, tmp):
    d = duration(vid); img = os.path.join(tmp, 'src_roop.jpg')
    for frac in (0.25, 0.5, 0.75):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{d * frac:.2f}', '-i', vid, '-frames:v', '1', '-q:v', '2', img], check=True)
        f = an.face(cv2.imread(img))
        if f is not None:
            return f, frac
    return None, None


def swap_video(an, swapper, src, target, part):
    cap = cv2.VideoCapture(target)
    fps = cap.get(cv2.CAP_PROP_FPS); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(int(round(fps)), 1)
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', f'{fps}',
                           '-i', '-', '-an', '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-pix_fmt', 'yuv420p', part],
                          stdin=subprocess.PIPE)
    n = noface = 0; flagged = False
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if n % step == 0 and is_nsfw(fr):
            flagged = True; break
        tf = an.face(fr)
        if tf is None:
            noface += 1
        else:
            fr = swapper.get(fr, tf, src, paste_back=True)
        ff.stdin.write(np.ascontiguousarray(fr).tobytes()); n += 1
    cap.release(); ff.stdin.close(); rc = ff.wait()
    return (rc == 0 and not flagged), n, noface, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    pairs = json.load(open(os.path.join(ROOT, 'configs', 'splits', f'{a.split}.json')))
    jobs = [(x, y) for p, q in pairs for x, y in ((p, q), (q, p))]
    if a.limit:
        jobs = jobs[:a.limit]
    odir = os.path.join(a.out, 'roop', a.split); tmp = os.path.join(a.out, 'tmp'); os.makedirs(odir, exist_ok=True); os.makedirs(tmp, exist_ok=True)
    an = Analyser(); swapper = insightface.model_zoo.get_model(SWAP_MODEL, providers=PROV)
    log = os.path.join(a.out, f'roop_{a.split}_log.csv'); new = not os.path.exists(log)
    with open(log, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['video', 'target', 'source', 'source_frac', 'ok', 'seconds', 'frames', 'note'])
        for t, s in jobs:
            name = f'{t}_{s}.mp4'; dst = os.path.join(odir, name)
            if os.path.exists(dst):
                continue
            t0 = time.time(); ok, frames, note = False, '', ''
            src, frac = source_face(an, os.path.join(FF_REAL, s + '.mp4'), tmp)
            if src is None:
                note = 'no source face'
            else:
                part = os.path.join(tmp, 'part_roop.mp4')
                ok, frames, noface, flagged = swap_video(an, swapper, src, os.path.join(FF_REAL, t + '.mp4'), part)
                note = 'nsfw flagged, skipped' if flagged else (f'no target face in {noface} frames' if noface else '')
                if ok:
                    os.replace(part, dst)
                elif os.path.exists(part):
                    os.remove(part)
            w.writerow([name, t, s, frac, int(ok), round(time.time() - t0, 1), frames, note]); fh.flush()
            print(f'roop {a.split} {name}: {"OK" if ok else "LOI"} {time.time() - t0:.1f}s {note}', flush=True)


if __name__ == '__main__':
    main()
