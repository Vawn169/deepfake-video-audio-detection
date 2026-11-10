import os, sys, time
import numpy as np, torch
sys.path.insert(0, 'src')
from api_core import AudioEngine, AUDIO_CKPT

# 10 file eval (5 bonafide, 5 spoof, seed 0); diem offline lay tu dac trung da luu, khong danh gia lai eval
c = torch.load(AUDIO_CKPT, map_location='cpu', weights_only=False)
# dac trung offline cung loai voi checkpoint API: ban cat im lang (audio_v2.py) hoac ban goc (audio_features.py)
z = np.load('/mnt/d/audio_feats_v2/eval_trim.npz' if c.get('trim') else 'data/audio_feats/eval.npz'); ids, y = list(z['ids']), z['y']
rng = np.random.RandomState(0)
pick = list(rng.choice(np.where(y == 0)[0], 5, replace=False)) + list(rng.choice(np.where(y == 1)[0], 5, replace=False))
eng, diffs = AudioEngine(), []
for j in pick:
    with torch.no_grad():
        x = torch.from_numpy((z['X'][j, c['layer']].astype('float32') - c['mean']) / c['std']).unsqueeze(0)
        off = float(eng.mlp(x))
    t = time.time()
    out = eng.predict(os.path.join('data/asvspoof2019_LA/LA/ASVspoof2019_LA_eval/flac', ids[j] + '.flac'), '.flac')
    d = abs(out['logit'] - off); diffs.append(d)
    off_label = 'spoof' if off >= c['threshold'] else 'bonafide'
    print(f"{ids[j]:<14} {'spoof' if y[j] else 'bonafide':<8} api_logit={out['logit']:.4f} offline_logit={off:.4f} "
          f"|d|={d:.4f} nhan_api={out['label']} nhan_offline={off_label} {time.time()-t:.2f}s")
print(f"max |d logit| = {max(diffs):.4f}  mean |d logit| = {sum(diffs)/len(diffs):.4f}")
