"""Tai du lieu train/validation cho V-C (docs/protocol.md muc 2026-11-17): tap con MAVOS-DD train + validation,
giai nen tap con LAV-DF train + dev tu LAV-DF.tar da co. Dung chung guard/retry voi download_ood.py.
  .venv/bin/python scripts/download_vc.py"""
import os, sys, tarfile, time
import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from download_ood import ROOT, MAVOS, LAVDF, free_gb, guard, retry


def ok(p, z):
    f = os.path.join(MAVOS, p)
    return os.path.exists(f) and os.path.getsize(f) == z


def mavos():
    s = pd.concat([pd.read_csv(os.path.join(ROOT, 'configs', f'mavos_{n}_subset.csv')) for n in ('train', 'validation')])
    todo = s[[not ok(p, z) for p, z in zip(s.video_path, s['size'])]]
    print(f'MAVOS train+val: can tai {len(todo)}/{len(s)} ({todo["size"].sum() / 1e9:.1f} GB), o D trong {free_gb():.1f} GB', flush=True)
    for i in range(0, len(todo), 400):
        b = todo.iloc[i:i + 400]
        guard(b['size'].sum() / 1e9)
        retry(lambda: snapshot_download('unibuc-cs/MAVOS-DD', repo_type='dataset', local_dir=MAVOS,
                                        allow_patterns=list(b.video_path), max_workers=8))
        print(f'  {min(i + 400, len(todo))}/{len(todo)} | o D trong {free_gb():.1f} GB', flush=True)
    bad = [p for p, z in zip(s.video_path, s['size']) if not ok(p, z)]
    print(f'MAVOS train+val xong: {len(s) - len(bad)}/{len(s)} dung kich thuoc; loi: {bad[:5]}', flush=True)
    if bad:
        sys.exit('DUNG: MAVOS thieu file')


def lavdf():
    out = os.path.join(LAVDF, 'extracted')
    want = set()
    for n in ('train', 'dev'):
        want |= {'LAV-DF/' + f for f in pd.read_csv(os.path.join(ROOT, 'configs', f'lavdf_{n}_subset.csv')).file}
    miss = {w for w in want if not os.path.exists(os.path.join(out, w))}
    print(f'LAV-DF train+dev: can giai nen {len(miss)}/{len(want)}', flush=True)
    if miss:
        guard(2)
        n = 0
        with tarfile.open(os.path.join(LAVDF, 'LAV-DF.tar')) as tf:
            for m in tf:   # doc tuan tu 25,6 GB
                if m.isfile() and m.name in miss:
                    tf.extract(m, out, filter='data'); n += 1
                    if n % 1000 == 0:
                        print(f'  giai nen {n}/{len(miss)}', flush=True)
    left = [w for w in want if not os.path.exists(os.path.join(out, w))]
    print(f'LAV-DF xong: {len(want) - len(left)}/{len(want)}; thieu vd {left[:3]}', flush=True)
    if left:
        sys.exit('DUNG: LAV-DF thieu file')


if __name__ == '__main__':
    print(time.strftime('%F %T'), 'bat dau', flush=True)
    lavdf()
    mavos()
    print(time.strftime('%F %T'), 'HOAN TAT', flush=True)
