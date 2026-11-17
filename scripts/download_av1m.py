"""Tai AV-Deepfake1M val (20 phan zip), chi giai nen tap con configs/av1m_val_subset.csv, roi xoa cac phan zip
(docs/protocol.md, muc 2026-11-16). Doc cac phan zip nhu mot file noi lien, khong ghep ra dia.
  .venv/bin/python scripts/download_av1m.py"""
import glob, io, os, shutil, sys, time, zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = '/mnt/d/datasets/av1m'
MIN_FREE = 30   # GB
N_PARTS = 20


def free_gb():
    return shutil.disk_usage('/mnt/d').free / 1e9


def retry(fn, tries=8):
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f'  loi {type(e).__name__}: {str(e)[:150]} -> thu lai sau 60 s ({k + 1}/{tries})', flush=True)
            time.sleep(60)
    sys.exit('DUNG: loi lap lai qua nhieu lan')


class Concat(io.RawIOBase):
    """Cac phan .zip.001... doc nhu mot file (seek duoc)."""
    def __init__(self, paths):
        self.f = [open(p, 'rb') for p in paths]
        self.sz = [os.path.getsize(p) for p in paths]
        self.start = [sum(self.sz[:i]) for i in range(len(paths))]
        self.total, self.pos = sum(self.sz), 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def readinto(self, b):
        n, got = len(b), 0
        while got < n and self.pos < self.total:
            i = max(j for j in range(len(self.start)) if self.start[j] <= self.pos)
            f = self.f[i]; f.seek(self.pos - self.start[i])
            chunk = f.read(min(n - got, self.sz[i] - (self.pos - self.start[i])))
            b[got:got + len(chunk)] = chunk; got += len(chunk); self.pos += len(chunk)
        return got


def main():
    sub = pd.read_csv(os.path.join(ROOT, 'configs', 'av1m_val_subset.csv'))
    vdir = os.path.join(OUT, 'val_subset')
    if os.path.exists(os.path.join(vdir, '.done')):
        print('Da giai nen xong tu truoc'); return
    for i in range(1, N_PARTS + 1):
        name = f'val/val.zip.{i:03d}'
        if not os.path.exists(os.path.join(OUT, name)) and free_gb() - 1.1 < MIN_FREE:
            sys.exit(f'DUNG: o D con {free_gb():.1f} GB')
        retry(lambda: hf_hub_download('ControlNet/AV-Deepfake1M', name, repo_type='dataset', local_dir=OUT))
        print(f'  {name} | o D trong {free_gb():.1f} GB', flush=True)
    parts = sorted(glob.glob(os.path.join(OUT, 'val', 'val.zip.*')))
    assert len(parts) == N_PARTS, parts
    want = set(sub.file)
    with zipfile.ZipFile(io.BufferedReader(Concat(parts), 1 << 20)) as z:
        names = z.namelist()
        pick = {n: n.split('val/', 1)[-1] for n in names if n.split('val/', 1)[-1] in want}
        print(f'zip co {len(names)} muc; khop tap con {len(pick)}/{len(want)}', flush=True)
        if len(pick) != len(want):
            sys.exit(f'DUNG: thieu {len(want) - len(pick)} video trong zip, vd {sorted(want - set(pick.values()))[:3]}; giu nguyen cac phan zip')
        for k, (n, rel) in enumerate(pick.items()):
            dst = os.path.join(vdir, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
            with z.open(n) as src, open(dst, 'wb') as out:
                shutil.copyfileobj(src, out)
            if k % 1000 == 0:
                print(f'  giai nen {k}/{len(pick)}', flush=True)
    ok = sum(os.path.exists(os.path.join(vdir, f)) for f in want)
    if ok != len(want):
        sys.exit(f'DUNG: chi co {ok}/{len(want)} file sau giai nen; giu nguyen cac phan zip')
    open(os.path.join(vdir, '.done'), 'w').close()
    for p in parts:
        os.remove(p)
    print(f'AV-Deepfake1M xong: {ok} video; da xoa {len(parts)} phan zip | o D trong {free_gb():.1f} GB', flush=True)


if __name__ == '__main__':
    print(time.strftime('%F %T'), 'bat dau', flush=True)
    main()
    print(time.strftime('%F %T'), 'HOAN TAT', flush=True)
