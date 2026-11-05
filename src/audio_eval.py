import os, sys, csv, datetime, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, 'reports', 'eval_ledger.csv')
p = os.path.join(ROOT, 'data', 'audio_feats', 'eval.npz')
if not os.path.exists(p): sys.exit('Chua co eval.npz')
done = open(LEDGER).read()
d = np.load(p); X, y = d['X'].astype('float32'), d['y']
rows = []
for seed in (42, 43, 44):
    ck = f'checkpoints_audio_s{seed}/best_mlp.pt'
    if f'{ck},ASVspoof2019 LA eval' in done:
        sys.exit(f'{ck} da duoc danh gia eval roi. Khong danh gia lai.')
for seed in (42, 43, 44):
    ck = f'checkpoints_audio_s{seed}/best_mlp.pt'
    c = torch.load(os.path.join(ROOT, ck), map_location='cpu', weights_only=False)
    net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)); net.load_state_dict(c['state']); net.eval()
    with torch.no_grad():
        sc = net(torch.from_numpy((X[:, c['layer']] - c['mean']) / c['std'])).squeeze(1).numpy()
    fpr, tpr, thr = roc_curve(y, sc); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr))); eer = float((fpr[i] + fnr[i]) / 2)
    auc = float(roc_auc_score(y, sc))
    pred = sc >= c['threshold']   # nguong EER chon tren dev
    miss_spoof = float(((~pred) & (y == 1)).sum() / (y == 1).sum())
    false_alarm = float((pred & (y == 0)).sum() / (y == 0).sum())
    rows.append((seed, eer, auc, miss_spoof, false_alarm))
    print(f'seed {seed}: eval EER {eer*100:.2f}%  AUC {auc:.4f}  | nguong dev: bo sot spoof {miss_spoof*100:.2f}%, bao nham bonafide {false_alarm*100:.2f}%')
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open(LEDGER, 'a', newline='') as fh:
    w = csv.writer(fh)
    for seed, eer, auc, _, _ in rows:
        ck = f'checkpoints_audio_s{seed}/best_mlp.pt'
        w.writerow([now, ck, 'ASVspoof2019 LA eval EER', f'{eer:.4f}'])
        w.writerow([now, ck, 'ASVspoof2019 LA eval AUC', f'{auc:.4f}'])
with open(os.path.join(ROOT, 'reports', 'audio_eval_results.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['seed', 'eval_eer', 'eval_auc', 'miss_spoof_at_dev_thr', 'false_alarm_at_dev_thr']); w.writerows(rows)
e = np.array([r[1] for r in rows]); a = np.array([r[2] for r in rows])
print(f'TRUNG BINH 3 seed: EER {e.mean()*100:.2f}% +- {e.std(ddof=1)*100:.2f}% | AUC {a.mean():.4f} +- {a.std(ddof=1):.4f}')
