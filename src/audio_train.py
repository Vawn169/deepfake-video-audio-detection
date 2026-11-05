import os, sys, csv, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FD = os.path.join(ROOT, 'data', 'audio_feats')
EPOCHS, BATCH, LR = 20, 256, 1e-3   # co dinh, ghi trong protocol.md truoc khi chay

def eer_thr(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])   # y=1 la spoof

for f in ('train', 'dev'):
    if not os.path.exists(os.path.join(FD, f + '.npz')):
        sys.exit(f'Chua co {f}.npz, doi buoc trich dac trung xong roi chay lai.')
tr, dv = np.load(os.path.join(FD, 'train.npz')), np.load(os.path.join(FD, 'dev.npz'))
Xtr, ytr, Xdv, ydv = tr['X'].astype('float32'), tr['y'], dv['X'].astype('float32'), dv['y']
print('train', Xtr.shape, 'dev', Xdv.shape, flush=True)

# 1) Chon tang: logistic regression tren tung tang, chon EER thap nhat tren dev
rows = []
for l in range(Xtr.shape[1]):
    m, s = Xtr[:, l].mean(0), Xtr[:, l].std(0) + 1e-6
    clf = LogisticRegression(class_weight='balanced', max_iter=300)
    clf.fit((Xtr[:, l] - m) / s, ytr)
    sc = clf.decision_function((Xdv[:, l] - m) / s)
    e, _ = eer_thr(ydv, sc)
    rows.append((l, e)); print(f'tang {l:2d}: dev EER = {e*100:.2f}%', flush=True)
os.makedirs(os.path.join(ROOT, 'reports'), exist_ok=True)
with open(os.path.join(ROOT, 'reports', 'audio_layer_selection.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['layer', 'dev_eer']); w.writerows(rows)
L = min(rows, key=lambda r: r[1])[0]
print('=> chon tang', L, flush=True)

# 2) MLP 768 -> 256 -> 1, 3 seed, chon epoch theo dev EER
mean, std = Xtr[:, L].mean(0), Xtr[:, L].std(0) + 1e-6
A = torch.from_numpy((Xtr[:, L] - mean) / std); B = torch.from_numpy(ytr.astype('float32'))
D = torch.from_numpy((Xdv[:, L] - mean) / std)
pw = torch.tensor((ytr == 0).sum() / (ytr == 1).sum(), dtype=torch.float32)  # N_bonafide / N_spoof
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
        with torch.no_grad(): sc = net(D).squeeze(1).numpy()
        e, t = eer_thr(ydv, sc)
        if best is None or e < best[0]:
            best = (e, t, ep, roc_auc_score(ydv, sc), {k: v.clone() for k, v in net.state_dict().items()})
    d = os.path.join(ROOT, f'checkpoints_audio_s{seed}'); os.makedirs(d, exist_ok=True)
    torch.save({'state': best[4], 'layer': L, 'mean': mean, 'std': std, 'threshold': best[1], 'epoch': best[2]},
               os.path.join(d, 'best_mlp.pt'))
    print(f'seed {seed}: epoch {best[2]}, dev EER {best[0]*100:.2f}%, dev AUC {best[3]:.4f}, nguong {best[1]:.4f}', flush=True)
    res.append((seed, best[2], best[0], best[3], best[1]))
with open(os.path.join(ROOT, 'reports', 'audio_dev_results.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['seed', 'epoch', 'dev_eer', 'dev_auc', 'threshold']); w.writerows(res)
print('XONG. Chua dung eval.')
