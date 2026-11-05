import csv, os, random, sys, time
import pandas as pd
sys.path.insert(0, 'src')
from api_core import Engine

rows = [r for r in csv.DictReader(open('configs/manifest_ffpp_c23.csv')) if r['split'] == 'test']
random.Random(1).shuffle(rows)
pred = pd.read_csv('reports/eval/checkpoints_aug_s44/predictions/ffpp_c23_test_preds.csv').set_index('video_id')['prob']
eng, diffs = Engine(), []
for r in rows:
    vid = f"{r['method']}_{r['video_id'].strip()}"
    if vid not in pred.index:
        continue
    t = time.time()
    out = eng.predict(os.path.join('data/ff', r['path']), verify_face=False, heatmap=False)
    if out['score'] is None:
        continue
    d = abs(out['score'] - pred[vid]); diffs.append(d)
    print(f"{vid:<22} api={out['score']:.4f} offline={pred[vid]:.4f} |d|={d:.4f} {time.time()-t:.2f}s")
    if len(diffs) == 10:
        break
print(f"max |d| = {max(diffs):.4f}  mean |d| = {sum(diffs)/len(diffs):.4f}")
