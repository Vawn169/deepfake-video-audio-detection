"""Manifest V-B: moi video doi mat da tao (nhan 1) theo cong cu va split (docs/plan_v3_faceswap.md).
Chi liet ke file .mp4 da tao xong; cap loi (khong thay mat nguon) khong co file nen tu dong bi bo (protocol 2026-11-11).
  python3 scripts/make_manifest_vb.py --root /mnt/d/faceswap/vb --out configs/manifest_vb.csv"""
import argparse, csv, os

# thu muc dau ra cua scripts/faceswap_gen.py -> ten phuong phap trong manifest
METHODS = {'inswapper_128': 'inswapper', 'simswap_256': 'simswap', 'ghost_1_256': 'ghost',
           'hyperswap_1a_256': 'hyperswap', 'inswapper_128_gfpgan': 'inswapper_gfpgan', 'roop': 'roop'}

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rows = []
for d, m in METHODS.items():
    for s in ('train', 'val', 'test'):
        p = os.path.join(a.root, d, s)
        if os.path.isdir(p):
            rows += [(s, 1, m, f[:-4], f'{d}/{s}/{f}') for f in sorted(os.listdir(p)) if f.endswith('.mp4')]
with open(a.out, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['split', 'label', 'method', 'video_id', 'path'])
    w.writerows(rows)
print(len(rows), 'video ->', a.out)
