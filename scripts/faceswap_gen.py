"""Tao video doi mat V-B bang FaceFusion (docs/protocol.md, muc 2026-11-10; docs/plan_v3_faceswap.md).
Cap theo danh sach chinh thuc FF++ (configs/splits/{train,val,test}.json), moi cap hai chieu: dich a, mat nguon b -> a_b.
Anh mat nguon: frame o 25% thoi luong video b (thu 50%, 75% neu khong co mat). Dau ra FaceFusion chat luong 100,
roi ma hoa lai H.264 CRF 23 bang ffmpeg cho moi cong cu giong nhau. Video tao ra chi luu tren may, khong cong bo.
  python3 scripts/faceswap_gen.py --model inswapper_128 --split train --limit 10 --out /mnt/d/faceswap/pilot"""
import argparse, csv, json, os, subprocess, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF_REAL = os.path.join(ROOT, 'data', 'ff', 'original_sequences', 'youtube', 'c23', 'videos')
FF_DIR = os.path.expanduser('~/faceswap/facefusion')
FF_PY = os.path.join(FF_DIR, '.venv-ff', 'bin', 'python')
NV = os.path.join(FF_DIR, '.venv-ff', 'lib', 'python3.12', 'site-packages', 'nvidia')
ENV = dict(os.environ, LD_LIBRARY_PATH=':'.join(sorted(os.path.join(NV, d, 'lib') for d in os.listdir(NV) if os.path.isdir(os.path.join(NV, d, 'lib')))))


def duration(p):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', p], capture_output=True, text=True)
    return float(r.stdout.strip())


def source_frame(vid, dst):
    """Frame o 25% (roi 50%, 75%) thoi luong; FaceFusion tu bao loi neu khong co mat."""
    d = duration(vid)
    for frac in (0.25, 0.5, 0.75):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{d * frac:.2f}', '-i', vid, '-frames:v', '1', '-q:v', '2', dst], check=True)
        yield frac


def run_ff(model, src_img, target, out_raw, extra, temp_dir):
    cmd = [FF_PY, 'facefusion.py', 'headless-run', '--processors', *extra.get('processors', ['face_swapper']),
           '--face-swapper-model', model, '-s', src_img, '-t', target, '-o', out_raw,
           '--execution-providers', 'cuda', '--output-video-quality', '100', '--log-level', 'error', '--temp-path', temp_dir, *extra.get('args', [])]
    r = subprocess.run(cmd, cwd=FF_DIR, env=ENV, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out_raw), (r.stdout + r.stderr)[-500:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='so video toi da (0 = tat ca)')
    ap.add_argument('--enhance', action='store_true', help='them face_enhancer gfpgan_1.4 (chi Test-A-enh)')
    a = ap.parse_args()
    pairs = json.load(open(os.path.join(ROOT, 'configs', 'splits', f'{a.split}.json')))
    jobs = [(x, y) for p, q in pairs for x, y in ((p, q), (q, p))]
    if a.limit:
        jobs = jobs[:a.limit]
    tag = a.model + ('_gfpgan' if a.enhance else '')
    odir = os.path.join(a.out, tag, a.split); tmp = os.path.join(a.out, 'tmp'); os.makedirs(odir, exist_ok=True); os.makedirs(tmp, exist_ok=True)
    extra = {'processors': ['face_swapper', 'face_enhancer'], 'args': ['--face-enhancer-model', 'gfpgan_1.4']} if a.enhance else {}
    log = os.path.join(a.out, f'{tag}_{a.split}_log.csv'); new = not os.path.exists(log)
    with open(log, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['video', 'target', 'source', 'source_frac', 'ok', 'seconds', 'frames', 'note'])
        for t, s in jobs:
            name = f'{t}_{s}.mp4'; dst = os.path.join(odir, name)
            if os.path.exists(dst):
                continue
            t0 = time.time(); src_img = os.path.join(tmp, 'src.jpg'); raw = os.path.join(tmp, 'raw.mp4'); ok, note, frac = False, '', None
            for frac in source_frame(os.path.join(FF_REAL, s + '.mp4'), src_img):
                ok, note = run_ff(a.model, src_img, os.path.join(FF_REAL, t + '.mp4'), raw, extra, '/tmp/faceswap_ff')   # tam tren o C: o D cham gap doi (do 10/11), chi ~1 GB moi luc
                if ok:
                    break
            frames = ''
            if ok:
                part = os.path.join(tmp, 'part.mp4')   # ghi file tam roi doi ten: dung giua chung khong de lai video do dang
                ok = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', raw, '-an', '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-pix_fmt', 'yuv420p', part]).returncode == 0
                if ok:
                    os.replace(part, dst)
                frames = subprocess.run(['ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v', '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', dst], capture_output=True, text=True).stdout.strip()
                os.remove(raw)
            w.writerow([name, t, s, frac, int(ok), round(time.time() - t0, 1), frames, '' if ok else note.replace('\n', ' ')]); fh.flush()
            print(f'{tag} {a.split} {name}: {"OK" if ok else "LOI"} {time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
