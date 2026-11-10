"""Bo test thuc te R1 (docs/protocol.md, muc 2026-11-09): chon va tai video that co giay phep tu do tren Wikimedia Commons.
  list     -> configs/realworld_r1_candidates.csv (ung vien thoa dieu kien) va configs/realworld_r1.csv (150 file chon bang seed 2026)
  download -> tai tung file, cat 15 giay bat dau o 30% thoi luong vao /mnt/d/realworld/R1, xoa file goc"""
import csv, json, os, random, re, subprocess, sys, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = '/mnt/d/realworld'
UA = 'deepfake-thesis-research/0.2 (https://github.com/Vawn169/deepfake-video-audio-detection; student thesis)'
KEYWORDS = ['interview', 'vlog', 'lecture', 'speech', 'talk', 'conversation', 'testimony', 'presentation']
LICENSES = re.compile(r'^(cc0|cc[- ]by(-sa)?([- ][0-9.]+)?|public domain|pd)', re.I)
EXCLUDE = re.compile(r'deepfake|ai[- ]generated|synthetic|animation|cartoon|game', re.I)
CAND, SEL = os.path.join(ROOT, 'configs', 'realworld_r1_candidates.csv'), os.path.join(ROOT, 'configs', 'realworld_r1.csv')
FIELDS = ['title', 'url', 'license', 'artist', 'duration', 'width', 'height', 'size', 'mime']


def api(params):
    q = urllib.parse.urlencode({'format': 'json', 'formatversion': 2, **params})
    req = urllib.request.Request('https://commons.wikimedia.org/w/api.php?' + q, headers={'User-Agent': UA})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            print('loi api, thu lai:', e, flush=True); time.sleep(10 * (k + 1))
    raise RuntimeError('API that bai')


def strip(html):
    return re.sub(r'<[^>]+>', '', html or '').strip()


def list_candidates():
    seen = {}
    for kw in KEYWORDS:
        cont = {}
        while True:
            d = api({'action': 'query', 'generator': 'search', 'gsrnamespace': 6, 'gsrlimit': 50,
                     'gsrsearch': f'{kw} filetype:video', 'prop': 'imageinfo|categories', 'cllimit': 'max',
                     'iiprop': 'url|size|mime|extmetadata', 'iiextmetadatafilter': 'LicenseShortName|Artist', **cont})
            for p in d.get('query', {}).get('pages', []):
                ii = (p.get('imageinfo') or [{}])[0]
                meta = ii.get('extmetadata', {})
                lic = strip(meta.get('LicenseShortName', {}).get('value'))
                cats = ' '.join(c['title'] for c in p.get('categories', []))
                row = {'title': p['title'], 'url': ii.get('url'), 'license': lic, 'artist': strip(meta.get('Artist', {}).get('value'))[:200],
                       'duration': ii.get('duration') or 0, 'width': ii.get('width') or 0, 'height': ii.get('height') or 0,
                       'size': ii.get('size') or 0, 'mime': ii.get('mime', '')}
                ok = (row['mime'] in ('video/webm', 'application/ogg', 'video/ogg') and 20 <= float(row['duration']) <= 1800
                      and int(row['height']) >= 360 and int(row['size']) <= 200e6 and LICENSES.match(lic or '')
                      and not EXCLUDE.search(p['title'] + ' ' + cats))
                if ok:
                    seen[row['title']] = row
            if 'continue' not in d or len(seen) > 5000:
                break
            cont = d['continue']; time.sleep(1)
        print(f'{kw}: tong ung vien {len(seen)}', flush=True)
    rows = sorted(seen.values(), key=lambda r: r['title'])
    with open(CAND, 'w', newline='') as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    sel = sorted(random.Random(2026).sample(rows, min(150, len(rows))), key=lambda r: r['title'])
    with open(SEL, 'w', newline='') as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(sel)
    print(f'ung vien {len(rows)}, chon {len(sel)} -> {SEL}')


def download():
    os.makedirs(os.path.join(OUT, 'R1'), exist_ok=True); tmp = os.path.join(OUT, 'tmp'); os.makedirs(tmp, exist_ok=True)
    rows = list(csv.DictReader(open(SEL)))
    for i, r in enumerate(rows):
        name = f'r1_{i:03d}.mp4'; dst = os.path.join(OUT, 'R1', name)
        if os.path.exists(dst):
            continue
        src = os.path.join(tmp, 'src' + os.path.splitext(r['url'])[1])
        ok = False
        for k in range(8):   # bi chan (429) thi cho lau dan, khong bo qua file
            if subprocess.run(['curl', '-s', '-f', '-L', '-A', UA, '-o', src, r['url']]).returncode == 0:
                ok = True; break
            print(f'{name}: loi tai, cho {120 * (k + 1)} s', flush=True); time.sleep(120 * (k + 1))
        if not ok:
            print(f'{name}: TAI LOI {r["title"]}', flush=True); continue
        start = 0.3 * float(r['duration'])
        res = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.2f}', '-i', src, '-t', '15', '-c:v', 'libx264', '-crf', '18',
                              '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', dst])
        os.remove(src)
        print(f'{name}: {"OK" if res.returncode == 0 else "CAT LOI"} {r["title"]}', flush=True)
        time.sleep(20)


if __name__ == '__main__':
    {'list': list_candidates, 'download': download}[sys.argv[1]]()
