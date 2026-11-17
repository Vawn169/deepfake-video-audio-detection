"""Ban nen cuoc goi cua cac tap test V-B (docs/protocol.md, muc 2026-11-15): cung cach nen video R2
(scripts/make_realworld_sets.py, CALL_V: toi da 640x360, 15 fps, H.264 300 kbps), bo tieng.
Dau vao: video gia test trong configs/manifest_vb.csv va 140 video that FF++ test (configs/splits/test.json).
Dau ra: /mnt/d/faceswap/vb_call/<cung duong dan tuong doi>, real/test/<id>.mp4. Chi luu tren may, khong cong bo.
  python3 scripts/make_call_vb.py"""
import csv, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from make_realworld_sets import CALL_V

VB, OUT = '/mnt/d/faceswap/vb', '/mnt/d/faceswap/vb_call'
FF_REAL = os.path.join(ROOT, 'data', 'ff', 'original_sequences', 'youtube', 'c23', 'videos')
WORKERS = 4


def ff(src, dst):
    if os.path.exists(dst):
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True); part = dst[:-4] + '.part.mp4'
    ok = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, *CALL_V, '-an', part]).returncode == 0
    if ok:
        os.replace(part, dst)
    return ok


jobs = [(os.path.join(VB, r['path']), os.path.join(OUT, r['path']))
        for r in csv.DictReader(open(os.path.join(ROOT, 'configs', 'manifest_vb.csv'))) if r['split'] == 'test']
ids = sorted({v for p in json.load(open(os.path.join(ROOT, 'configs', 'splits', 'test.json'))) for v in p})
jobs += [(os.path.join(FF_REAL, f'{v}.mp4'), os.path.join(OUT, 'real', 'test', f'{v}.mp4')) for v in ids]
with ThreadPoolExecutor(WORKERS) as ex:
    ok = list(ex.map(lambda j: ff(*j), jobs))
print(f'{sum(ok)}/{len(jobs)} video nen xong ->', OUT)
if not all(ok):
    print('LOI:', [j[0] for j, k in zip(jobs, ok) if not k]); sys.exit(1)
