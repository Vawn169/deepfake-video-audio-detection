"""Tai LAV-DF va tap con MAVOS-DD (docs/protocol.md, muc 2026-11-16). Chi tai, khong danh gia.
Dung khi o /mnt/d con < MIN_FREE GB; mat mang thi thu lai; kiem tra kich thuoc file sau khi tai.
  .venv/bin/python scripts/download_ood.py"""
import os, shutil, sys, tarfile, time
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAVOS, LAVDF = '/mnt/d/datasets/mavos_dd', '/mnt/d/datasets/lavdf'
MIN_FREE = 30   # GB


def free_gb():
    return shutil.disk_usage('/mnt/d').free / 1e9


def guard(need_gb=0):
    f = free_gb()
    if f - need_gb < MIN_FREE:
        sys.exit(f'DUNG: o D con {f:.1f} GB, can them {need_gb:.1f} GB, toi thieu {MIN_FREE} GB')


def retry(fn, tries=8):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f'  loi {type(e).__name__}: {str(e)[:150]} -> thu lai sau 60 s ({k + 1}/{tries})', flush=True)
            time.sleep(60)
    sys.exit('DUNG: loi lap lai qua nhieu lan')


def mavos():
    s = pd.read_csv(os.path.join(ROOT, 'configs', 'mavos_test_subset.csv'))
    todo = s[[not (os.path.exists(os.path.join(MAVOS, p)) and os.path.getsize(os.path.join(MAVOS, p)) == z)
              for p, z in zip(s.video_path, s['size'])]]
    print(f'MAVOS-DD: can tai {len(todo)}/{len(s)} video ({todo["size"].sum() / 1e9:.1f} GB), o D trong {free_gb():.1f} GB', flush=True)
    for i in range(0, len(todo), 400):
        b = todo.iloc[i:i + 400]
        guard(b['size'].sum() / 1e9)
        retry(lambda: snapshot_download('unibuc-cs/MAVOS-DD', repo_type='dataset', local_dir=MAVOS,
                                        allow_patterns=list(b.video_path), max_workers=8))
        print(f'  {min(i + 400, len(todo))}/{len(todo)} | o D trong {free_gb():.1f} GB', flush=True)
    bad = [p for p, z in zip(s.video_path, s['size'])
           if not os.path.exists(os.path.join(MAVOS, p)) or os.path.getsize(os.path.join(MAVOS, p)) != z]
    print(f'MAVOS-DD xong: {len(s) - len(bad)}/{len(s)} dung kich thuoc; loi: {bad[:5]}', flush=True)


def lavdf():
    tar = os.path.join(LAVDF, 'LAV-DF.tar')
    if not os.path.exists(tar):
        guard(25.6)
    retry(lambda: hf_hub_download('ControlNet/LAV-DF', 'LAV-DF.tar', repo_type='dataset', local_dir=LAVDF))
    print(f'LAV-DF.tar: {os.path.getsize(tar) / 1e9:.1f} GB | o D trong {free_gb():.1f} GB', flush=True)
    out = os.path.join(LAVDF, 'extracted')
    if os.path.exists(os.path.join(out, '.done')):
        return
    with tarfile.open(tar) as tf:
        n = 0
        for m in tf:   # doc tuan tu, chi giai nen metadata va phan test
            name = m.name.lower()
            if m.isfile() and ('/test/' in '/' + name or name.endswith(('.json', '.csv', '.txt', '.md'))):
                if n % 2000 == 0:
                    guard(1)
                tf.extract(m, out, filter='data'); n += 1
                if n % 5000 == 0:
                    print(f'  giai nen {n} file | o D trong {free_gb():.1f} GB', flush=True)
    open(os.path.join(out, '.done'), 'w').close()
    print(f'LAV-DF giai nen xong: {n} file | o D trong {free_gb():.1f} GB', flush=True)


if __name__ == '__main__':
    print(time.strftime('%F %T'), 'bat dau', flush=True)
    mavos()
    lavdf()
    print(time.strftime('%F %T'), 'HOAN TAT', flush=True)
