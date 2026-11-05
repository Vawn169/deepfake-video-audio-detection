import argparse, time, os
import numpy as np, soundfile as sf, torch
from transformers import Wav2Vec2Model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'data', 'asvspoof2019_LA', 'LA')
PROTO = {'train': 'ASVspoof2019.LA.cm.train.trn.txt',
         'dev': 'ASVspoof2019.LA.cm.dev.trl.txt',
         'eval': 'ASVspoof2019.LA.cm.eval.trl.txt'}
MAXLEN = 64000  # 4 giay o 16 kHz, khong pad

class DS(torch.utils.data.Dataset):
    def __init__(self, split, limit):
        p = os.path.join(BASE, 'ASVspoof2019_LA_cm_protocols', PROTO[split])
        rows = [l.split() for l in open(p)]
        if limit: rows = rows[:limit]
        self.split = split
        self.ids = [r[1] for r in rows]
        self.y = np.array([1 if r[-1] == 'spoof' else 0 for r in rows], dtype=np.int8)  # 1 = spoof
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        w, sr = sf.read(os.path.join(BASE, f'ASVspoof2019_LA_{self.split}', 'flac', self.ids[i] + '.flac'), dtype='float32')
        assert sr == 16000, sr
        w = w[:MAXLEN]
        w = (w - w.mean()) / np.sqrt(w.var() + 1e-7)
        return torch.from_numpy(w)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=list(PROTO))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ds = DS(a.split, a.limit)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=4, prefetch_factor=8)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base').to(dev).eval()
    feats, t0 = [], time.time()
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == 'cuda')):
        for i, w in enumerate(dl):
            hs = m(w.unsqueeze(0).to(dev), output_hidden_states=True).hidden_states
            feats.append(torch.stack([h[0].float().mean(0) for h in hs]).cpu().numpy().astype(np.float16))
            if (i + 1) % 1000 == 0:
                print(f'{i+1}/{len(ds)}  {(i+1)/(time.time()-t0):.1f} file/s', flush=True)
    X = np.stack(feats)
    dt = time.time() - t0
    np.savez(a.out, X=X, y=ds.y, ids=np.array(ds.ids))
    print(f'XONG {a.split}: X={X.shape} thoi gian {dt:.1f}s ({len(ds)/dt:.1f} file/s) device={dev}')
