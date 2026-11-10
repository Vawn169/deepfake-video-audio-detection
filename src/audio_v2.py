"""Giai doan 1b: audio v2 (docs/protocol.md, muc 2026-11-08).
  feats --set train|dev|eval|itw [--aug]   trich dac trung (cat im lang; --aug them augmentation co seed theo ten file)
  train --config a|b                        chon tang, train MLP 3 seed, chon epoch va nguong tren validation
  eval  --config a|b                        In-the-Wild va ASVspoof 2019 eval, moi checkpoint mot lan"""
import argparse, csv, datetime, glob, os, random, subprocess, sys, time
import numpy as np, soundfile as sf, torch, torch.nn as nn
from scipy.signal import fftconvolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LA = os.path.join(ROOT, 'data', 'asvspoof2019_LA', 'LA')
ITW = '/mnt/d/in_the_wild/release_in_the_wild'
RIRS = '/mnt/d/rirs_noises/RIRS_NOISES'
FD = '/mnt/d/audio_feats_v2'
LEDGER = os.path.join(ROOT, 'reports', 'eval_ledger.csv')
from audio_common import SR, MAXLEN, FRAME, trim  # dung chung voi API
EPOCHS, BATCH, LR = 20, 256, 1e-3   # giong audio_train.py, ghi trong protocol
PROTO = {'train': 'ASVspoof2019.LA.cm.train.trn.txt', 'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
         'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'}


def items(split):
    """(duong dan, id, nhan) voi nhan 1 = spoof."""
    if split == 'itw':
        rows = list(csv.DictReader(open(os.path.join(ITW, 'meta.csv'))))
        return [(os.path.join(ITW, r['file']), r['file'], int(r['label'] == 'spoof')) for r in rows]
    rows = [l.split() for l in open(os.path.join(LA, 'ASVspoof2019_LA_cm_protocols', PROTO[split]))]
    return [(os.path.join(LA, f'ASVspoof2019_LA_{split}', 'flac', r[1] + '.flac'), r[1], int(r[-1] == 'spoof')) for r in rows]


def _ffmpeg_roundtrip(w, codec, kbps):
    fmt = {'aac': 'adts', 'libmp3lame': 'mp3', 'libopus': 'ogg'}[codec]
    enc = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', 'pipe:0',
                          '-c:a', codec, '-b:a', f'{kbps}k', '-f', fmt, 'pipe:1'], input=w.astype(np.float32).tobytes(),
                         capture_output=True, check=True).stdout
    dec = subprocess.run(['ffmpeg', '-v', 'error', '-i', 'pipe:0', '-f', 'f32le', '-ac', '1', '-ar', str(SR), 'pipe:1'],
                         input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(dec, np.float32).copy()


class Augmenter:
    def __init__(self):
        self.rirs = sorted(glob.glob(os.path.join(RIRS, 'simulated_rirs', '*', '*', '*.wav')))
        self.noises = sorted(glob.glob(os.path.join(RIRS, 'pointsource_noises', '*.wav')))
        assert self.rirs and self.noises, 'chua co RIRS_NOISES'

    def __call__(self, w, key):
        rng = random.Random(key)
        if rng.random() < 0.4:
            rir, _ = sf.read(rng.choice(self.rirs), dtype='float32')
            rir = rir if rir.ndim == 1 else rir[:, 0]
            rms = np.sqrt(np.mean(w ** 2)) + 1e-9
            w = fftconvolve(w, rir)[:len(w)].astype(np.float32)
            w *= rms / (np.sqrt(np.mean(w ** 2)) + 1e-9)
        if rng.random() < 0.6:
            nz, _ = sf.read(rng.choice(self.noises), dtype='float32')
            nz = nz if nz.ndim == 1 else nz[:, 0]
            nz = np.tile(nz, int(np.ceil(len(w) / max(len(nz), 1))))
            off = rng.randint(0, len(nz) - len(w))
            nz = nz[off:off + len(w)]
            snr = rng.uniform(5, 20)
            p_s, p_n = np.mean(w ** 2) + 1e-12, np.mean(nz ** 2) + 1e-12
            w = (w + nz * np.sqrt(p_s / (p_n * 10 ** (snr / 10)))).astype(np.float32)
        if rng.random() < 0.2:
            w = _ffmpeg_band(w)
        if rng.random() < 0.5:
            codec, lo, hi = rng.choice([('aac', 32, 96), ('libmp3lame', 32, 128), ('libopus', 16, 48)])
            w = _ffmpeg_roundtrip(w, codec, rng.randint(lo, hi))
        return w


def _ffmpeg_band(w):
    out = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', 'pipe:0',
                          '-af', 'aresample=8000,aresample=16000', '-f', 'f32le', 'pipe:1'],
                         input=w.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
    return np.frombuffer(out, np.float32).copy()


class DS(torch.utils.data.Dataset):
    def __init__(self, split, aug):
        self.items, self.split, self.aug = items(split), split, Augmenter() if aug else None
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, key, _ = self.items[i]
        w, sr = sf.read(p, dtype='float32')
        assert sr == SR and w.ndim == 1, (p, sr)
        if self.aug:
            w = self.aug(w, f'{self.split}:{key}')
        w = trim(w)[:MAXLEN]
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
        return torch.from_numpy(w)


def feat_path(split, aug):
    return os.path.join(FD, f'{split}_{"aug" if aug else "trim"}.npz')


def extract(split, aug):
    from transformers import Wav2Vec2Model
    os.makedirs(FD, exist_ok=True)
    ds = DS(split, aug)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=8, prefetch_factor=8)
    m = Wav2Vec2Model.from_pretrained(os.path.join(ROOT, 'models', 'wav2vec2-base')).to('cuda').eval()
    feats, t0 = [], time.time()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        for i, w in enumerate(dl):
            hs = m(w.unsqueeze(0).to('cuda'), output_hidden_states=True).hidden_states
            feats.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16))
            if (i + 1) % 5000 == 0:
                print(f'{split} aug={aug}: {i+1}/{len(ds)}  {(i+1)/(time.time()-t0):.1f} file/s', flush=True)
    y = np.array([it[2] for it in ds.items], dtype=np.int8)
    np.savez(feat_path(split, aug), X=np.stack(feats), y=y, ids=np.array([it[1] for it in ds.items]))
    print(f'XONG {split} aug={aug}: {len(ds)} file, {time.time()-t0:.0f}s', flush=True)


def load(split, aug):
    d = np.load(feat_path(split, aug))
    return d['X'].astype('float32'), d['y']


def eer_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s); fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def train(config):
    Xa, ya = load('train', False)
    if config == 'b':
        Xb, yb = load('train', True)
        Xa, ya = np.concatenate([Xa, Xb]), np.concatenate([ya, yb])
    Xd, yd = load('dev', False)
    Xn, yn = load('dev', True)
    print(f'config {config}: train {Xa.shape}, dev {Xd.shape}, dev nhieu {Xn.shape}', flush=True)
    rows = []
    for l in range(Xa.shape[1]):
        m, s = Xa[:, l].mean(0), Xa[:, l].std(0) + 1e-6
        clf = LogisticRegression(class_weight='balanced', max_iter=300).fit((Xa[:, l] - m) / s, ya)
        e1, _ = eer_thr(yd, clf.decision_function((Xd[:, l] - m) / s))
        e2, _ = eer_thr(yn, clf.decision_function((Xn[:, l] - m) / s))
        rows.append((l, e1, e2, (e1 + e2) / 2)); print(f'tang {l:2d}: dev {e1*100:.2f}%  dev nhieu {e2*100:.2f}%', flush=True)
    with open(os.path.join(ROOT, 'reports', f'audio_v2{config}_layer_selection.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['layer', 'dev_eer', 'dev_aug_eer', 'mean']); w.writerows(rows)
    L = min(rows, key=lambda r: r[3])[0]; print('=> chon tang', L, flush=True)
    mean, std = Xa[:, L].mean(0), Xa[:, L].std(0) + 1e-6
    A, B = torch.from_numpy((Xa[:, L] - mean) / std), torch.from_numpy(ya.astype('float32'))
    D, N = torch.from_numpy((Xd[:, L] - mean) / std), torch.from_numpy((Xn[:, L] - mean) / std)
    pw = torch.tensor((ya == 0).sum() / (ya == 1).sum(), dtype=torch.float32)
    res = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed); np.random.seed(seed)
        net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        g = torch.Generator().manual_seed(seed)
        best = None
        for ep in range(1, EPOCHS + 1):
            net.train()
            perm = torch.randperm(len(A), generator=g)
            for i in range(0, len(A), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad(); lossf(net(A[idx]).squeeze(1), B[idx]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                sd, sn = net(D).squeeze(1).numpy(), net(N).squeeze(1).numpy()
            e1, _ = eer_thr(yd, sd); e2, _ = eer_thr(yn, sn)
            if best is None or (e1 + e2) / 2 < best[0]:
                _, thr = eer_thr(np.concatenate([yd, yn]), np.concatenate([sd, sn]))
                best = ((e1 + e2) / 2, thr, ep, e1, e2, {k: v.clone() for k, v in net.state_dict().items()})
        d = os.path.join(ROOT, f'checkpoints_audio_v2{config}_s{seed}'); os.makedirs(d, exist_ok=True)
        torch.save({'state': best[5], 'layer': L, 'mean': mean, 'std': std, 'threshold': best[1], 'epoch': best[2],
                    'trim': True, 'config': config}, os.path.join(d, 'best_mlp.pt'))
        res.append((seed, best[2], best[3], best[4], best[1]))
        print(f'seed {seed}: epoch {best[2]}, dev EER {best[3]*100:.2f}%, dev nhieu EER {best[4]*100:.2f}%, nguong {best[1]:.4f}', flush=True)
    with open(os.path.join(ROOT, 'reports', f'audio_v2{config}_dev_results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['seed', 'epoch', 'dev_eer', 'dev_aug_eer', 'threshold']); w.writerows(res)


def evaluate(config):
    sets = [('itw', 'In-the-Wild'), ('eval', 'ASVspoof2019 LA eval')]
    cks = [f'checkpoints_audio_v2{config}_s{s}/best_mlp.pt' for s in (42, 43, 44)]
    done = open(LEDGER).read()
    for ck in cks:
        for _, name in sets:
            if f'{ck},{name}' in done:
                sys.exit(f'{ck} da duoc danh gia tren {name} roi. Khong danh gia lai.')
    rows, ledger = [], []
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for split, name in sets:
        X, y = load(split, False)
        for s, ck in zip((42, 43, 44), cks):
            c = torch.load(os.path.join(ROOT, ck), map_location='cpu', weights_only=False)
            net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1)); net.load_state_dict(c['state']); net.eval()
            with torch.no_grad():
                sc = net(torch.from_numpy((X[:, c['layer']] - c['mean']) / c['std'])).squeeze(1).numpy()
            eer, _ = eer_thr(y, sc); auc = float(roc_auc_score(y, sc))
            pred = sc >= c['threshold']
            miss = float(((~pred) & (y == 1)).sum() / (y == 1).sum()); fa = float((pred & (y == 0)).sum() / (y == 0).sum())
            rows.append((name, s, eer, auc, miss, fa))
            ledger += [[now, ck, f'{name} EER', f'{eer:.4f}'], [now, ck, f'{name} AUC', f'{auc:.4f}']]
            print(f'{name} seed {s}: EER {eer*100:.2f}%  AUC {auc:.4f} | nguong val: bo sot spoof {miss*100:.2f}%, bao nham bona-fide {fa*100:.2f}%', flush=True)
        e = np.array([r[2] for r in rows if r[0] == name]); a = np.array([r[3] for r in rows if r[0] == name])
        print(f'{name} TRUNG BINH: EER {e.mean()*100:.2f}% +- {e.std(ddof=1)*100:.2f}% | AUC {a.mean():.4f} +- {a.std(ddof=1):.4f}', flush=True)
    with open(LEDGER, 'a', newline='') as fh:
        csv.writer(fh).writerows(ledger)
    with open(os.path.join(ROOT, 'reports', f'audio_v2{config}_eval_results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['dataset', 'seed', 'eer', 'auc', 'miss_spoof_at_val_thr', 'false_alarm_at_val_thr']); w.writerows(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['feats', 'train', 'eval'])
    ap.add_argument('--set', choices=['train', 'dev', 'eval', 'itw'])
    ap.add_argument('--aug', action='store_true')
    ap.add_argument('--config', choices=['a', 'b'])
    a = ap.parse_args()
    if a.cmd == 'feats':
        if os.path.exists(feat_path(a.set, a.aug)):
            sys.exit(f'da co {feat_path(a.set, a.aug)}')
        extract(a.set, a.aug)
    elif a.cmd == 'train':
        train(a.config)
    else:
        evaluate(a.config)
