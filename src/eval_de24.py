"""Danh gia tren Deepfake-Eval-2024 theo docs/protocol.md (muc 2026-11-06, quy tac 1a; muc 2026-11-08).
Viet truoc khi co du lieu. Dau vao la manifest CSV: path,label,modality[,language]
  label: 1 = fake, 0 = real; modality: video | audio. Buoc tao manifest tu nhan goc them sau, khong doi cach tinh diem.
Video: 16 frame deu, MediaPipe, cat mat nhu preprocess.py. Chi so chinh: AUC tren video co >= 4 frame mat (kem ty le bao phu);
       chi so phu: AUC tren moi video, video khong du mat nhan diem 0,5.
Audio: 16 kHz mono (ffmpeg neu can), toi da 4 giay; checkpoint co trim=True thi cat im lang nhu audio_v2.py.
Fusion: tren video du mat co track am thanh, diem = trung binh xac suat gia cua checkpoint API (aug_s44, audio_s43), so voi nhan video.
Moi checkpoint danh gia mot lan cho moi ten tap (eval_ledger.csv), tru khi --no-ledger (chi de chay thu)."""
import argparse, csv, datetime, os, subprocess, sys
import cv2, numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn
import mediapipe as mp
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from preprocess import make_detector, crop_face
from dataset import get_transforms
from model import build_model
from audio_common import trim

N_FRAMES, MARGIN, SIZE, MIN_FACES, SR, MAXLEN = 16, 0.3, 224, 4, 16000, 64000
LEDGER = os.path.join(ROOT, 'reports', 'eval_ledger.csv')
VIDEO_CKPTS = [f'checkpoints_{p}s{s}/best_effnet_b0.pt' for p in ('', 'aug_') for s in (42, 43, 44)]
AUDIO_CKPTS = [f'checkpoints_audio_s{s}/best_mlp.pt' for s in (42, 43, 44)]
API_VIDEO, API_AUDIO = 'checkpoints_aug_s44/best_effnet_b0.pt', 'checkpoints_audio_s43/best_mlp.pt'


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr))); return float((fpr[i] + fnr[i]) / 2)


def face_crops(detector, path):
    """Giong preprocess.process_video, giu trong bo nho (JPEG 95 de khop anh luc train)."""
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    if total > 0:
        for idx in [int(i * (total - 1) / (N_FRAMES - 1)) for i in range(N_FRAMES)]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, frame = cap.read()
            if not ok:
                continue
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if not res.detections:
                continue
            face = crop_face(frame, max(res.detections, key=lambda d: d.categories[0].score), MARGIN, SIZE)
            if face is None:
                continue
            _, buf = cv2.imencode('.jpg', face, [cv2.IMWRITE_JPEG_QUALITY, 95])
            out.append(Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


def load_audio(path):
    """None neu khong co track am thanh. 16 kHz mono doc thang; con lai qua ffmpeg nhu API."""
    try:
        info = sf.info(path)
        if info.samplerate == SR and info.channels == 1:
            return sf.read(path, dtype='float32')[0]
    except Exception:
        pass
    pr = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index', '-of', 'csv=p=0', path],
                        capture_output=True, text=True)
    if pr.returncode != 0 or not pr.stdout.strip():
        return None
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR), '-f', 'f32le', 'pipe:1'], capture_output=True)
    w = np.frombuffer(r.stdout, np.float32).copy()
    return w if r.returncode == 0 and len(w) >= 400 else None


def w2v_layers(model, w, do_trim, dev):
    if do_trim:
        w = trim(w)
    w = w[:MAXLEN]; w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
        hs = model(torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(dev), output_hidden_states=True).hidden_states
    return np.stack([h[0].float().mean(0).cpu().numpy() for h in hs])


def mlp_logit(ck, feats):
    net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)); net.load_state_dict(ck['state']); net.eval()
    with torch.no_grad():
        return net(torch.from_numpy((feats[:, ck['layer']] - ck['mean']) / ck['std']).float()).squeeze(1).numpy()


def video_threshold(ckpt):
    p = os.path.join(ROOT, 'reports', 'eval', os.path.basename(os.path.dirname(ckpt)), 'summary.csv')
    if os.path.exists(p):
        return float(pd.read_csv(p)['val_threshold'].iloc[0])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--name', default='Deepfake-Eval-2024')
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'eval_de24'))
    ap.add_argument('--video_ckpts', nargs='*', default=VIDEO_CKPTS)
    ap.add_argument('--audio_ckpts', nargs='*', default=AUDIO_CKPTS)
    ap.add_argument('--no-ledger', action='store_true', help='chay thu, khong ghi eval_ledger.csv')
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    vids, auds = df[df.modality == 'video'].reset_index(drop=True), df[df.modality == 'audio'].reset_index(drop=True)
    names = {'v': f'{a.name} video', 'a': f'{a.name} audio', 'f': f'{a.name} fusion'}
    if not a.no_ledger:
        done = open(LEDGER).read()
        for ck in a.video_ckpts + a.audio_ckpts:
            for n in names.values():
                if f'{ck},{n}' in done:
                    sys.exit(f'{ck} da duoc danh gia tren {n} roi. Khong danh gia lai.')
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ledger, summary = [], []

    # --- video ---
    crops, n_faces = [], []
    if len(vids):
        det = make_detector(os.path.join(ROOT, 'models', 'blaze_face_short_range.tflite'))
        for p in vids.path:
            c = face_crops(det, p); crops.append(c); n_faces.append(len(c))
        vids['n_faces'] = n_faces
        ok = vids.n_faces >= MIN_FACES
        tf = get_transforms(train=False)
        for ck in a.video_ckpts:
            st = torch.load(os.path.join(ROOT, ck), map_location=dev, weights_only=False)
            m = build_model(pretrained=False).to(dev); m.load_state_dict(st['model_state_dict']); m.eval()
            scores = []
            for c in crops:
                if not c:
                    scores.append(np.nan); continue
                with torch.no_grad(), torch.autocast(dev, enabled=(dev == 'cuda')):
                    scores.append(float(torch.sigmoid(m(torch.stack([tf(i) for i in c]).to(dev)).squeeze(1)).float().mean()))
            s = np.array(scores); vids[ck] = s
            auc_main = float(roc_auc_score(vids.label[ok], s[ok]))
            auc_all = float(roc_auc_score(vids.label, np.where(ok, s, 0.5)))
            thr = video_threshold(ck)
            acc = float(((s[ok] >= thr).astype(int) == vids.label[ok]).mean()) if thr is not None else None
            summary.append({'part': 'video', 'ckpt': ck, 'n': int(ok.sum()), 'coverage': round(float(ok.mean()), 4),
                            'auc': round(auc_main, 4), 'auc_all_0.5': round(auc_all, 4), 'eer': round(eer(vids.label[ok], s[ok]), 4),
                            'acc_val_thr': None if acc is None else round(acc, 4)})
            ledger.append([now, ck, names['v'], f'{auc_main:.4f}'])
            print(f"video {ck}: AUC {auc_main:.4f} tren {ok.sum()}/{len(vids)} video du mat; AUC moi video (0,5) {auc_all:.4f}", flush=True)
        vids.to_csv(os.path.join(a.out, 'video_preds.csv'), index=False)

    # --- audio va fusion ---
    if len(auds) or len(vids):
        from transformers import Wav2Vec2Model
        w2v = Wav2Vec2Model.from_pretrained(os.path.join(ROOT, 'models', 'wav2vec2-base')).to(dev).eval()
        cks = {ck: torch.load(os.path.join(ROOT, ck), map_location='cpu', weights_only=False) for ck in set(a.audio_ckpts + [API_AUDIO])}

        def featurize(paths):
            raw, trm, has = [], [], []
            for p in paths:
                w = load_audio(p)
                has.append(w is not None)
                if w is not None:
                    raw.append(w2v_layers(w2v, w, False, dev)); trm.append(w2v_layers(w2v, w, True, dev))
            return np.array(raw), np.array(trm), np.array(has, bool)

        if len(auds):
            raw, trm, has = featurize(auds.path)
            y = auds.label.values[has]
            for ck in a.audio_ckpts:
                c = cks[ck]; s = mlp_logit(c, trm if c.get('trim') else raw)
                full = np.full(len(auds), np.nan); full[has] = s; auds[ck] = full
                e, auc = eer(y, s), float(roc_auc_score(y, s))
                pred = s >= c['threshold']
                fa = float((pred & (y == 0)).sum() / max((y == 0).sum(), 1)); miss = float(((~pred) & (y == 1)).sum() / max((y == 1).sum(), 1))
                row = {'part': 'audio', 'ckpt': ck, 'n': int(has.sum()), 'coverage': round(float(has.mean()), 4), 'auc': round(auc, 4),
                       'eer': round(e, 4), 'false_alarm_thr': round(fa, 4), 'miss_thr': round(miss, 4)}
                if 'language' in auds:
                    en = pd.Series(auds.language.values[has]).astype(str).str.lower().isin(['en', 'english']).values
                    for tag, mask in (('en', en), ('non_en', ~en)):
                        if len(set(y[mask])) == 2:
                            row[f'eer_{tag}'] = round(eer(y[mask], s[mask]), 4); row[f'n_{tag}'] = int(mask.sum())
                summary.append(row)
                ledger += [[now, ck, f"{names['a']} EER", f'{e:.4f}'], [now, ck, f"{names['a']} AUC", f'{auc:.4f}']]
                print(f'audio {ck}: EER {e*100:.2f}%  AUC {auc:.4f} tren {has.sum()}/{len(auds)} file', flush=True)
            auds.to_csv(os.path.join(a.out, 'audio_preds.csv'), index=False)

        if len(vids):
            raw, trm, has = featurize(vids.path)
            c = cks[API_AUDIO]; al = np.full(len(vids), np.nan); al[has] = mlp_logit(c, trm if c.get('trim') else raw)
            pa = 1 / (1 + np.exp(-al)); pv = vids[API_VIDEO].values if API_VIDEO in vids else np.full(len(vids), np.nan)
            both = (vids.n_faces >= MIN_FACES).values & has
            if both.sum() and len(set(vids.label[both])) == 2:
                fused = (pv[both] + pa[both]) / 2; y = vids.label.values[both]
                auc_f, auc_v, auc_a = (float(roc_auc_score(y, v)) for v in (fused, pv[both], pa[both]))
                summary.append({'part': 'fusion', 'ckpt': f'{API_VIDEO}+{API_AUDIO}', 'n': int(both.sum()),
                                'coverage': round(float(both.mean()), 4), 'auc': round(auc_f, 4), 'auc_video_only': round(auc_v, 4),
                                'auc_audio_only': round(auc_a, 4)})
                ledger.append([now, f'{API_VIDEO}+{API_AUDIO}', names['f'], f'{auc_f:.4f}'])
                print(f'fusion: AUC {auc_f:.4f} (chi video {auc_v:.4f}, chi audio {auc_a:.4f}) tren {both.sum()} video', flush=True)
            vids['audio_logit_api'] = al
            vids.to_csv(os.path.join(a.out, 'video_preds.csv'), index=False)

    pd.DataFrame(summary).to_csv(os.path.join(a.out, 'summary.csv'), index=False)
    if not a.no_ledger:
        with open(LEDGER, 'a', newline='') as fh:
            csv.writer(fh).writerows(ledger)
    print('Da ghi', os.path.join(a.out, 'summary.csv'), '' if not a.no_ledger else '(chay thu, khong ghi ledger)')


if __name__ == '__main__':
    main()
