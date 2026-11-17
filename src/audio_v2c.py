"""Audio v2c (docs/protocol.md muc 2026-11-17): nhu v2b (audio_v2.py config b) + tieng MAVOS-DD train (nhan audio_fake)
va LAV-DF train (nhan modify_audio). Video LAV-DF gia tieng: cua so 4 giay chua doan gia dau tien (giua cua so = giua doan);
con lai doc nhu luc test (cat im lang, 4 giay dau). Dac trung moi chi ban sach (khong augmentation).
  feats    trich dac trung MAVOS train/validation, LAV-DF train/dev -> /mnt/d/audio_feats_v2/vc_*.npz
  train    chon tang, train MLP 3 seed, chon epoch + nguong tren ASVspoof dev, dev nhieu, validation moi (gop)"""
import argparse, collections, csv, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_common import SR, MAXLEN
from audio_v2 import ROOT, FD, EPOCHS, BATCH, LR, load, eer_thr
from eval_de24 import load_audio, w2v_layers

SETS = {'mavos_train': ('mavos_train_subset.csv', '/mnt/d/datasets/mavos_dd'),
        'mavos_val': ('mavos_validation_subset.csv', '/mnt/d/datasets/mavos_dd'),
        'lavdf_train': ('lavdf_train_subset.csv', '/mnt/d/datasets/lavdf/extracted/LAV-DF'),
        'lavdf_dev': ('lavdf_dev_subset.csv', '/mnt/d/datasets/lavdf/extracted/LAV-DF')}


def items(name):
    fn, base = SETS[name]
    d = pd.read_csv(os.path.join(ROOT, 'configs', fn))
    if name.startswith('mavos'):
        return [(os.path.join(base, p), p, int(y), None) for p, y in zip(d.video_path, d.audio_fake)]
    out = []
    for f, y, per in zip(d.file, d.modify_audio, d.fake_periods):
        per = json.loads(per)
        out.append((os.path.join(base, f), f, int(y), per[0] if (y and per) else None))
    return out


def window(w, period):
    if period is None:
        return w, True
    mid = int((period[0] + period[1]) / 2 * SR)
    s = max(0, min(len(w) - MAXLEN, mid - MAXLEN // 2))
    return w[s:s + MAXLEN], False   # cua so da chon: khong cat im lang (de khong mat doan gia)


def feats(name):
    from transformers import Wav2Vec2Model
    its = items(name)
    m = Wav2Vec2Model.from_pretrained(os.path.join(ROOT, 'models', 'wav2vec2-base')).to('cuda').eval()
    X, y, ids, skip, t0 = [], [], [], [], time.time()

    def bounded(ex, k=32):
        q = collections.deque()
        for it in its:
            q.append((it, ex.submit(load_audio, it[0])))
            if len(q) >= k:
                yield q.popleft()
        while q:
            yield q.popleft()

    with ThreadPoolExecutor(6) as ex:
        for i, (it, fut) in enumerate(bounded(ex)):
            w = fut.result()
            if w is None:
                skip.append(it[1]); continue
            w, do_trim = window(w, it[3])
            X.append(w2v_layers(m, w, do_trim, 'cuda').astype(np.float16)); y.append(it[2]); ids.append(it[1])
            if (i + 1) % 1000 == 0:
                print(f'{name}: {i + 1}/{len(its)}  {(i + 1) / (time.time() - t0):.1f} file/s', flush=True)
    np.savez(os.path.join(FD, f'vc_{name}.npz'), X=np.stack(X), y=np.array(y, np.int8), ids=np.array(ids))
    print(f'XONG {name}: {len(X)} file, bo {len(skip)} khong co tieng {skip[:3]}, {time.time() - t0:.0f}s', flush=True)


def loadvc(name):
    d = np.load(os.path.join(FD, f'vc_{name}.npz'))
    return d['X'].astype('float32'), d['y']


def train():
    parts = [load('train', False), load('train', True), loadvc('mavos_train'), loadvc('lavdf_train')]
    Xa, ya = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    vals = [load('dev', False), load('dev', True)]
    nv = [loadvc('mavos_val'), loadvc('lavdf_dev')]
    vals.append((np.concatenate([p[0] for p in nv]), np.concatenate([p[1] for p in nv])))
    print(f'v2c: train {Xa.shape} (that {int((ya == 0).sum())}, gia {int(ya.sum())}); val {[v[0].shape[0] for v in vals]}', flush=True)
    rows = []
    for l in range(Xa.shape[1]):
        mu, sd = Xa[:, l].mean(0), Xa[:, l].std(0) + 1e-6
        clf = LogisticRegression(class_weight='balanced', max_iter=300).fit((Xa[:, l] - mu) / sd, ya)
        es = [eer_thr(yv, clf.decision_function((Xv[:, l] - mu) / sd))[0] for Xv, yv in vals]
        rows.append((l, *es, float(np.mean(es)))); print(f'tang {l:2d}: ' + '  '.join(f'{e * 100:.2f}%' for e in es), flush=True)
    with open(os.path.join(ROOT, 'reports', 'audio_v2c_layer_selection.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['layer', 'asv_dev_eer', 'asv_dev_aug_eer', 'new_val_eer', 'mean']); w.writerows(rows)
    L = min(rows, key=lambda r: r[-1])[0]; print('=> chon tang', L, flush=True)
    mean, std = Xa[:, L].mean(0), Xa[:, L].std(0) + 1e-6
    A, B = torch.from_numpy((Xa[:, L] - mean) / std), torch.from_numpy(ya.astype('float32'))
    V = [(torch.from_numpy((Xv[:, L] - mean) / std), yv) for Xv, yv in vals]
    pw = torch.tensor((ya == 0).sum() / (ya == 1).sum(), dtype=torch.float32)
    res = []
    for seed in (42, 43, 44):
        torch.manual_seed(seed); np.random.seed(seed)
        net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 1))
        opt = torch.optim.Adam(net.parameters(), lr=LR); lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        g = torch.Generator().manual_seed(seed); best = None
        for ep in range(1, EPOCHS + 1):
            net.train(); perm = torch.randperm(len(A), generator=g)
            for i in range(0, len(A), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad(); lossf(net(A[idx]).squeeze(1), B[idx]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                ss = [net(Xv).squeeze(1).numpy() for Xv, _ in V]
            es = [eer_thr(yv, s)[0] for (_, yv), s in zip(V, ss)]
            if best is None or np.mean(es) < best[0]:
                _, thr = eer_thr(np.concatenate([yv for _, yv in V]), np.concatenate(ss))
                best = (float(np.mean(es)), thr, ep, es, {k: v.clone() for k, v in net.state_dict().items()})
        d = os.path.join(ROOT, f'checkpoints_audio_v2c_s{seed}'); os.makedirs(d, exist_ok=True)
        torch.save({'state': best[4], 'layer': L, 'mean': mean, 'std': std, 'threshold': best[1], 'epoch': best[2],
                    'trim': True, 'config': 'c'}, os.path.join(d, 'best_mlp.pt'))
        res.append((seed, best[2], *best[3], best[1]))
        print(f'seed {seed}: epoch {best[2]}, EER ' + ' / '.join(f'{e * 100:.2f}%' for e in best[3]) + f', nguong {best[1]:.4f}', flush=True)
    with open(os.path.join(ROOT, 'reports', 'audio_v2c_dev_results.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['seed', 'epoch', 'asv_dev_eer', 'asv_dev_aug_eer', 'new_val_eer', 'threshold']); w.writerows(res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['feats', 'train']); ap.add_argument('--sets', nargs='+', default=list(SETS))
    a = ap.parse_args()
    if a.cmd == 'feats':
        for n in a.sets:
            feats(n)
    else:
        train()
