"""Danh gia tren bo test thuc te R1-R4 (docs/protocol.md, muc 2026-11-09). Dung chung tien xu ly voi eval_de24.py.
Tap chi co mau that: ty le bao nham (khoang tin cay Wilson 95%) o nguong EER cua tung checkpoint, va o nguong bao nham 5% voi checkpoint API.
Tap co ca that va gia: them AUC, EER. Moi checkpoint danh gia mot lan cho moi tap (eval_ledger.csv), tru khi --no-ledger."""
import argparse, csv, datetime, math, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_de24 import (MIN_FACES, VIDEO_CKPTS, AUDIO_CKPTS, LEDGER, eer, face_crops, load_audio, w2v_layers, mlp_logit,
                       video_threshold, make_detector, get_transforms, build_model)
from api_core import THRESH_FPR5, AUDIO_THRESH_FPR5, MIN_SAMPLES
from audio_common import trim

VIDEO_ALL = VIDEO_CKPTS + [f'checkpoints_video_v2a_s{s}/best_effnet_b0.pt' for s in (42, 43, 44)]
AUDIO_ALL = AUDIO_CKPTS + [f'checkpoints_audio_v2{c}_s{s}/best_mlp.pt' for c in 'ab' for s in (42, 43, 44)]
API_VIDEO, API_AUDIO = 'checkpoints_aug_s44/best_effnet_b0.pt', 'checkpoints_audio_v2b_s43/best_mlp.pt'


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def rates(y, s, thr):
    out = {}
    for lab, key in ((0, 'false_alarm'), (1, 'detected')):
        m = y == lab
        if m.sum():
            k = int((s[m] >= thr).sum()); lo, hi = wilson(k, int(m.sum()))
            out.update({key: round(k / m.sum(), 4), f'{key}_ci_low': round(lo, 4), f'{key}_ci_high': round(hi, 4), f'n_{key}': int(m.sum())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'eval_realworld'))
    ap.add_argument('--no-ledger', action='store_true')
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    vids, auds = df[df.modality == 'video'].reset_index(drop=True), df[df.modality == 'audio'].reset_index(drop=True)
    names = {'video': f'{a.name} video', 'audio': f'{a.name} audio'}
    if not a.no_ledger:
        done = open(LEDGER).read()
        for ck in VIDEO_ALL * bool(len(vids)) + AUDIO_ALL * bool(len(auds)):
            for n in names.values():
                if f'{ck},{n}' in done:
                    sys.exit(f'{ck} da duoc danh gia tren {n} roi. Khong danh gia lai.')
    os.makedirs(a.out, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary, ledger = [], []

    if len(vids):
        det = make_detector(os.path.join(ROOT, 'models', 'blaze_face_short_range.tflite'))
        crops = [face_crops(det, p) for p in vids.path]
        vids['n_faces'] = [len(c) for c in crops]
        ok = (vids.n_faces >= MIN_FACES).values; y = vids.label.values
        tf = get_transforms(train=False)
        for ck in VIDEO_ALL:
            st = torch.load(os.path.join(ROOT, ck), map_location=dev, weights_only=False)
            m = build_model(pretrained=False).to(dev); m.load_state_dict(st['model_state_dict']); m.eval()
            s = []
            for c in crops:
                if not c:
                    s.append(np.nan); continue
                with torch.no_grad(), torch.autocast(dev, enabled=(dev == 'cuda')):
                    s.append(float(torch.sigmoid(m(torch.stack([tf(i) for i in c]).to(dev)).squeeze(1)).float().mean()))
            s = np.array(s); vids[ck] = s
            thr = video_threshold(ck)
            row = {'set': a.name, 'part': 'video', 'ckpt': ck, 'n': int(len(vids)), 'n_scored': int(ok.sum()),
                   'coverage': round(float(ok.mean()), 4), 'threshold': thr, **rates(y[ok], s[ok], thr)}
            if ck == API_VIDEO:
                row.update({f'fpr5_{k}': v for k, v in rates(y[ok], s[ok], THRESH_FPR5).items()})
            if len(set(y[ok])) == 2:
                row.update(auc=round(float(roc_auc_score(y[ok], s[ok])), 4), eer=round(eer(y[ok], s[ok]), 4))
            summary.append(row)
            ledger.append([now, ck, names['video'], f"{row.get('auc', row.get('false_alarm')):.4f}"])
            print(f"{a.name} video {ck}: du dieu kien {ok.sum()}/{len(vids)} | bao nham {row.get('false_alarm')} | AUC {row.get('auc')}", flush=True)
        vids.to_csv(os.path.join(a.out, f'{a.name}_video_preds.csv'), index=False)

    if len(auds):
        from transformers import Wav2Vec2Model
        w2v = Wav2Vec2Model.from_pretrained(os.path.join(ROOT, 'models', 'wav2vec2-base')).to(dev).eval()
        # Giong API: checkpoint cat im lang ma con duoi MIN_SAMPLES mau sau khi cat thi khong cham (inconclusive)
        raw, trm, has, has_t = [], [], [], []
        for p in auds.path:
            w = load_audio(p); has.append(w is not None); has_t.append(w is not None and len(trim(w)) >= MIN_SAMPLES)
            if w is not None:
                raw.append(w2v_layers(w2v, w, False, dev))
            if has_t[-1]:
                trm.append(w2v_layers(w2v, w, True, dev))
        raw, trm, has, has_t = np.array(raw), np.array(trm), np.array(has, bool), np.array(has_t, bool)
        print('khong cham sau khi cat im lang:', list(auds.path[has & ~has_t]), flush=True)
        for ck in AUDIO_ALL:
            c = torch.load(os.path.join(ROOT, ck), map_location='cpu', weights_only=False)
            hk = has_t if c.get('trim') else has; y = auds.label.values[hk]
            s = mlp_logit(c, trm if c.get('trim') else raw)
            full = np.full(len(auds), np.nan); full[hk] = s; auds[ck] = full
            row = {'set': a.name, 'part': 'audio', 'ckpt': ck, 'n': int(len(auds)), 'n_scored': int(hk.sum()),
                   'coverage': round(float(hk.mean()), 4), 'threshold': round(float(c['threshold']), 4), **rates(y, s, c['threshold'])}
            if ck == API_AUDIO:
                row.update({f'fpr5_{k}': v for k, v in rates(y, s, AUDIO_THRESH_FPR5).items()})
            if len(set(y)) == 2:
                row.update(auc=round(float(roc_auc_score(y, s)), 4), eer=round(eer(y, s), 4))
            summary.append(row)
            ledger.append([now, ck, names['audio'], f"{row.get('eer', row.get('false_alarm')):.4f}"])
            print(f"{a.name} audio {ck}: co tieng {hk.sum()}/{len(auds)} | bao nham {row.get('false_alarm')} | EER {row.get('eer')}", flush=True)
        auds.to_csv(os.path.join(a.out, f'{a.name}_audio_preds.csv'), index=False)

    pd.DataFrame(summary).to_csv(os.path.join(a.out, f'{a.name}_summary.csv'), index=False)
    if not a.no_ledger:
        with open(LEDGER, 'a', newline='') as fh:
            csv.writer(fh).writerows(ledger)


if __name__ == '__main__':
    main()
