"""Fusion voi checkpoint API hien tai (video vb_s44 + audio v2c_s44), docs/protocol.md muc 2026-11-17 (do lai fusion).
Tinh tu diem da luu (reports/eval_ood/<bo>/preds.csv cot vb_s44, reports/eval_ood_vc/<bo>/preds.csv cot v2c_s44),
cung cach nhu eval_ood.py (nhan chung, video du mat va co tieng; AUC trung binh xac suat; quy tac mot nhanh vuot nguong 5%).
Ledger: "<bo> fusion v2c" mot lan. Ket qua reports/eval_ood_vc/fusion_v2c.csv
  .venv/bin/python scripts/fusion_v2c.py"""
import csv, datetime, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src'))
from eval_ood import MIN_FACES, wilson
from eval_vb import fpr5
from eval_de24 import LEDGER
from api_core import AUDIO_THRESH_FPR5

V, A = 'checkpoints_video_vb_s44/best_effnet_b0.pt', 'checkpoints_audio_v2c_s44/best_mlp.pt'
done = open(LEDGER).read(); now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'); rows, led = [], []
thr_v = fpr5(V)
for name in ('MAVOS-DD', 'AV-Deepfake1M', 'LAV-DF', 'SocialDF'):
    if f'{name} fusion v2c' in done:
        sys.exit(f'{name} fusion v2c da co trong ledger')
    a = pd.read_csv(os.path.join(ROOT, 'reports', 'eval_ood', name, 'preds.csv'))
    b = pd.read_csv(os.path.join(ROOT, 'reports', 'eval_ood_vc', name, 'preds.csv'))
    assert list(a.path) == list(b.path), name
    pv, la = a['v:' + V].values, b['a:' + A].values
    both = (a.n_faces.values >= MIN_FACES) & ~np.isnan(pv) & ~np.isnan(la)
    y, pv, la = a.y.values[both], pv[both], la[both]; pa = 1 / (1 + np.exp(-la))
    r = dict(set=name, n=int(both.sum()), coverage=round(float(both.mean()), 4), thr_video=round(thr_v, 4), thr_audio=AUDIO_THRESH_FPR5,
             auc_mean=round(float(roc_auc_score(y, (pv + pa) / 2)), 4), auc_video_only=round(float(roc_auc_score(y, pv)), 4),
             auc_audio_only=round(float(roc_auc_score(y, la)), 4))
    for key, f in (('or_rule', (pv >= thr_v) | (la >= AUDIO_THRESH_FPR5)), ('video_only', pv >= thr_v), ('audio_only', la >= AUDIO_THRESH_FPR5)):
        for lab, k2 in ((1, 'detected'), (0, 'false_alarm')):
            k, n = int(f[y == lab].sum()), int((y == lab).sum()); lo, hi = wilson(k, n)
            r[f'{key}_{k2}'], r[f'{key}_{k2}_lo'], r[f'{key}_{k2}_hi'] = round(k / n, 4), round(lo, 4), round(hi, 4)
    rows.append(r); led.append([now, f'{V}+{A}', f'{name} fusion v2c', f"{r['auc_mean']:.4f}"])
    print(r, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, 'reports', 'eval_ood_vc', 'fusion_v2c.csv'), index=False)
with open(LEDGER, 'a', newline='') as fh:
    csv.writer(fh).writerows(led)
