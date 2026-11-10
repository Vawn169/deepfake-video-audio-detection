"""Tao cac tap R1-R4 (docs/protocol.md, muc 2026-11-09) va manifest cua chung trong configs/.
R1: clip Commons da tai (scripts/collect_commons.py download). R2: R1 nen nhu cuoc goi. R3: Celeb-DF 200 nen nhu cuoc goi.
R4: 5.000 file In-the-Wild (seed 2026) nen Opus 16 kbps."""
import csv, os, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = '/mnt/d/realworld'
ITW = '/mnt/d/in_the_wild/release_in_the_wild'
CALL_V = ['-vf', "scale='min(640,iw)':'min(360,ih)':force_original_aspect_ratio=decrease,scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=15",
          '-c:v', 'libx264', '-b:v', '300k', '-maxrate', '300k', '-bufsize', '600k', '-pix_fmt', 'yuv420p']
CALL_A = ['-c:a', 'libopus', '-b:a', '16k', '-ac', '1', '-ar', '16000']


def ff(args):
    return subprocess.run(['ffmpeg', '-v', 'error', '-y', *args]).returncode == 0


def write(name, rows):
    p = os.path.join(ROOT, 'configs', f'realworld_{name}.manifest.csv')
    with open(p, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['path', 'label', 'modality', 'source']); w.writerows(rows)
    print(name, len(rows), '->', p)


def r1_r2():
    r1 = sorted(f for f in os.listdir(os.path.join(OUT, 'R1')) if f.endswith('.mp4'))
    os.makedirs(os.path.join(OUT, 'R2'), exist_ok=True)
    jobs = [(os.path.join(OUT, 'R1', f), os.path.join(OUT, 'R2', f)) for f in r1]
    with ThreadPoolExecutor(6) as ex:
        ok = list(ex.map(lambda j: os.path.exists(j[1]) or ff(['-i', j[0], *CALL_V, *CALL_A, j[1]]), jobs))
    rows = []
    for (src, dst), k in zip(jobs, ok):
        for mod in ('video', 'audio'):
            rows.append([src, 0, mod, 'commons'])
    write('R1', rows)
    write('R2', [[dst, 0, mod, 'commons_call'] for (src, dst), k in zip(jobs, ok) if k for mod in ('video', 'audio')])


def r3():
    os.makedirs(os.path.join(OUT, 'R3'), exist_ok=True)
    sub = list(csv.DictReader(open(os.path.join(ROOT, 'configs', 'celebdf_test_subset.csv'))))
    jobs = [(os.path.join(ROOT, 'data', 'celeb-df-v2', r['path']), os.path.join(OUT, 'R3', r['path'].replace('/', '__')), int(r['is_fake'])) for r in sub]
    with ThreadPoolExecutor(6) as ex:
        ok = list(ex.map(lambda j: os.path.exists(j[1]) or ff(['-i', j[0], *CALL_V, '-an', j[1]]), jobs))
    write('R3', [[dst, y, 'video', 'celebdf_call'] for (src, dst, y), k in zip(jobs, ok) if k])


def r4():
    os.makedirs(os.path.join(OUT, 'R4'), exist_ok=True)
    meta = sorted(csv.DictReader(open(os.path.join(ITW, 'meta.csv'))), key=lambda r: r['file'])
    pick = sorted(random.Random(2026).sample(meta, 5000), key=lambda r: r['file'])
    jobs = [(os.path.join(ITW, r['file']), os.path.join(OUT, 'R4', r['file'].replace('.wav', '.ogg')), int(r['label'] == 'spoof')) for r in pick]
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda j: os.path.exists(j[1]) or ff(['-i', j[0], *CALL_A, j[1]]), jobs))
    write('R4', [[dst, y, 'audio', 'itw_call'] for (src, dst, y), k in zip(jobs, ok) if k])


if __name__ == '__main__':
    for step in sys.argv[1:]:
        {'r1r2': r1_r2, 'r3': r3, 'r4': r4}[step]()
