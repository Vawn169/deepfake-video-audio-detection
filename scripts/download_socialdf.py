"""Tai video SocialDF tu danh sach link Instagram cua tac gia (docs/protocol.md, muc 2026-11-16).
yt-dlp; neu co ~/.secrets/ig_cookies.txt thi dung cookie tai khoan phu nguoi lam tao (16/11), khong thi khong dang nhap;
tuan tu, nghi 4-8 s; link loi truoc do duoc thu lai; dong cuoi cua moi link trong log la ket qua cuoi; 25 link loi lien tiep thi dung (co the bi chan). Chay lai duoc (bo qua link da xong).
Ket qua: /mnt/d/datasets/socialdf/{fake,real}/<id>.mp4, log reports/socialdf_download.csv.
  .venv/bin/python scripts/download_socialdf.py"""
import csv, os, random, re, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = '/mnt/d/datasets/socialdf'
YT = os.path.expanduser('~/tools/gd/bin/yt-dlp')
LOG = os.path.join(ROOT, 'reports', 'socialdf_download.csv')
MIN_FREE = 30
COOKIES = os.path.expanduser('~/.secrets/ig_cookies.txt')


def links(fn):
    return sorted(set(re.findall(r'https://www\.instagram\.com/[^\s,"]+', open(os.path.join(D, fn), encoding='utf-8').read())))


def main():
    done = set()
    if os.path.exists(LOG):
        done = {r['url'] for r in csv.DictReader(open(LOG)) if r['status'] == 'ok'}
    todo = [(u, 'fake') for u in links('deepfake.csv')] + [(u, 'real') for u in links('real.csv')]
    todo = [t for t in todo if t[0] not in done]
    if os.path.exists(LOG):   # 17/11: link chua thu len truoc, link loi cu thu sau (tranh 25 loi lien tiep tu link da chet)
        tried = {r['url'] for r in csv.DictReader(open(LOG))}
        todo.sort(key=lambda t: t[0] in tried)
    ck = ['--cookies', COOKIES] if os.environ.get('SOCIALDF_COOKIES') == '1' and os.path.exists(COOKIES) else []   # 16/11: cookie bi HTTP 400 (1/13 thanh cong), mac dinh khong dung
    print(f'Can thu {len(todo)} link | cookie: {"co" if ck else "khong"}', flush=True)
    new = not os.path.exists(LOG)
    fh = open(LOG, 'a', newline=''); w = csv.writer(fh)
    if new:
        w.writerow(['url', 'label', 'status', 'file', 'error', 'time'])
    fails = 0
    for i, (u, lab) in enumerate(todo):
        if shutil.disk_usage('/mnt/d').free / 1e9 < MIN_FREE:
            sys.exit('DUNG: o D duoi 30 GB')
        out = os.path.join(D, lab); os.makedirs(out, exist_ok=True)
        try:
            p = subprocess.run([YT, *ck, '-q', '--no-warnings', '--no-playlist', '-f', 'mp4/best', '-o', os.path.join(out, '%(id)s.%(ext)s'),
                                '--print', 'after_move:filepath', u], capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:   # 17/11: 1 link treo lam dung ca script; ghi loi roi di tiep
            p = subprocess.CompletedProcess(u, 1, '', 'timeout 180 s')
        if p.returncode == 0 and p.stdout.strip():
            fails = 0
            w.writerow([u, lab, 'ok', os.path.relpath(p.stdout.strip().splitlines()[-1], D), '', time.strftime('%F %T')])
        else:
            fails += 1
            err = (p.stderr.strip().splitlines() or [''])[-1][:200]
            w.writerow([u, lab, 'error', '', err, time.strftime('%F %T')])
        fh.flush()
        if i % 50 == 0:
            print(f'{i}/{len(todo)} | loi lien tiep {fails}', flush=True)
        if fails >= 25:
            sys.exit('DUNG: 25 link loi lien tiep, co the bi Instagram chan; chay lai sau')
        time.sleep(random.uniform(4, 8))
    print('HOAN TAT', flush=True)


if __name__ == '__main__':
    main()
