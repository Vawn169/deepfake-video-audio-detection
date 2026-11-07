"""Giai doan 1a: danh gia 3 checkpoint audio hien tai tren In-the-Wild (docs/protocol.md, muc mo lai 2026-11-06).
Tien xu ly giong audio_features.py: 16 kHz mono, toi da 4 giay dau, khong pad, khong cat im lang. Moi checkpoint chi danh gia mot lan."""
import csv, datetime, os, sys, time
import numpy as np, soundfile as sf, torch, torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score
from transformers import Wav2Vec2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITW = '/mnt/d/in_the_wild/release_in_the_wild'
FEAT = os.path.join(ROOT, 'data', 'audio_feats', 'itw.npz')
LEDGER = os.path.join(ROOT, 'reports', 'eval_ledger.csv')
NAME = 'In-the-Wild'
MAXLEN = 64000


class DS(torch.utils.data.Dataset):
    def __init__(self):
        rows = list(csv.DictReader(open(os.path.join(ITW, 'meta.csv'))))
        self.files = [r['file'] for r in rows]
        self.y = np.array([1 if r['label'] == 'spoof' else 0 for r in rows], dtype=np.int8)  # 1 = spoof
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        w, sr = sf.read(os.path.join(ITW, self.files[i]), dtype='float32')
        assert sr == 16000 and w.ndim == 1, (self.files[i], sr, w.shape)
        w = w[:MAXLEN]
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
        return torch.from_numpy(w)


def extract():
    ds = DS()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4, prefetch_factor=8)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Wav2Vec2Model.from_pretrained(os.path.join(ROOT, 'models', 'wav2vec2-base')).to(dev).eval()
    feats, t0 = [], time.time()
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
        for i, w in enumerate(dl):
            hs = m(w.unsqueeze(0).to(dev), output_hidden_states=True).hidden_states
            feats.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16))
            if (i + 1) % 5000 == 0:
                print(f'{i+1}/{len(ds)}  {(i+1)/(time.time()-t0):.1f} file/s', flush=True)
    np.savez(FEAT, X=np.stack(feats), y=ds.y, ids=np.array(ds.files))
    print(f'XONG trich dac trung: {len(ds)} file, {time.time()-t0:.1f}s, device={dev}', flush=True)


if __name__ == '__main__':
    done = open(LEDGER).read()
    cks = [f'checkpoints_audio_s{s}/best_mlp.pt' for s in (42, 43, 44)]
    for ck in cks:
        if f'{ck},{NAME}' in done:
            sys.exit(f'{ck} da duoc danh gia tren {NAME} roi. Khong danh gia lai.')
    if not os.path.exists(FEAT):
        extract()
    d = np.load(FEAT); X, y = d['X'].astype('float32'), d['y']
    rows = []
    for s, ck in zip((42, 43, 44), cks):
        c = torch.load(os.path.join(ROOT, ck), map_location='cpu', weights_only=False)
        net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)); net.load_state_dict(c['state']); net.eval()
        with torch.no_grad():
            sc = net(torch.from_numpy((X[:, c['layer']] - c['mean']) / c['std'])).squeeze(1).numpy()
        fpr, tpr, _ = roc_curve(y, sc); fnr = 1 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr))); eer = float((fpr[i] + fnr[i]) / 2)
        auc = float(roc_auc_score(y, sc))
        pred = sc >= c['threshold']   # nguong EER chon tren ASVspoof 2019 dev
        miss = float(((~pred) & (y == 1)).sum() / (y == 1).sum())
        fa = float((pred & (y == 0)).sum() / (y == 0).sum())
        rows.append((s, eer, auc, miss, fa))
        print(f'seed {s}: {NAME} EER {eer*100:.2f}%  AUC {auc:.4f} | nguong dev: bo sot spoof {miss*100:.2f}%, bao nham bona-fide {fa*100:.2f}%')
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LEDGER, 'a', newline='') as fh:
        w = csv.writer(fh)
        for (s, eer, auc, _, _), ck in zip(rows, cks):
            w.writerow([now, ck, f'{NAME} EER', f'{eer:.4f}'])
            w.writerow([now, ck, f'{NAME} AUC', f'{auc:.4f}'])
    with open(os.path.join(ROOT, 'reports', 'audio_itw_results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['seed', 'itw_eer', 'itw_auc', 'miss_spoof_at_dev_thr', 'false_alarm_at_dev_thr']); w.writerows(rows)
    e = np.array([r[1] for r in rows]); a = np.array([r[2] for r in rows])
    print(f'TRUNG BINH 3 seed: EER {e.mean()*100:.2f}% +- {e.std(ddof=1)*100:.2f}% | AUC {a.mean():.4f} +- {a.std(ddof=1):.4f}')
