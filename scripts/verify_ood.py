"""Kiem tra du lieu da tai (MAVOS-DD, AV-Deepfake1M, LAV-DF): du so file, dung kich thuoc (neu biet), ffprobe doc duoc,
co luong hinh / tieng, giai ma duoc 1 frame. Khong chay mo hinh. Ket qua: reports/verify_ood.csv + tom tat.
  .venv/bin/python scripts/verify_ood.py"""
import glob, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = '/mnt/d/datasets'


def check(path):
    r = dict(exists=os.path.exists(path))
    if not r['exists']:
        return r
    r['bytes'] = os.path.getsize(path)
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=duration', '-of', 'json', path],
                       capture_output=True, text=True)
    try:
        j = json.loads(p.stdout or '{}')
        types = [s.get('codec_type') for s in j.get('streams', [])]
        r.update(probe_ok=p.returncode == 0, has_video='video' in types, has_audio='audio' in types,
                 duration=float(j.get('format', {}).get('duration', 'nan')))
    except Exception:
        r.update(probe_ok=False)
    f = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-frames:v', '1', '-f', 'null', '-'], capture_output=True)
    r['decode_ok'] = f.returncode == 0 and not f.stderr.strip()
    return r


def run(name, paths, expect_bytes=None):
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(check, paths))
    df = pd.DataFrame(res); df.insert(0, 'path', paths); df.insert(0, 'dataset', name)
    if expect_bytes is not None:
        df['size_ok'] = df.bytes.values == expect_bytes
    return df


def main():
    out = []
    m = pd.read_csv(os.path.join(ROOT, 'configs', 'mavos_test_subset.csv'))
    out.append(run('MAVOS-DD', [os.path.join(D, 'mavos_dd', p) for p in m.video_path], m['size'].values))
    a = pd.read_csv(os.path.join(ROOT, 'configs', 'av1m_val_subset.csv'))
    out.append(run('AV-Deepfake1M', [os.path.join(D, 'av1m', 'val_subset', p) for p in a.file]))
    lav = os.path.join(D, 'lavdf', 'extracted')
    metas = glob.glob(os.path.join(lav, '**', 'metadata*.json'), recursive=True)
    if metas:
        meta = json.load(open(metas[0])); base = os.path.dirname(metas[0])
        test = [x['file'] for x in meta if x.get('split') == 'test']
        print(f'LAV-DF metadata {metas[0]}: {len(meta)} muc, test {len(test)}')
        out.append(run('LAV-DF test', [os.path.join(base, f) for f in test]))
    else:
        print('LAV-DF: khong thay metadata*.json trong', lav)
    df = pd.concat(out, ignore_index=True)
    df.to_csv(os.path.join(ROOT, 'reports', 'verify_ood.csv'), index=False)
    cols = [c for c in ('exists', 'size_ok', 'probe_ok', 'has_video', 'has_audio', 'decode_ok') if c in df]
    s = df.groupby('dataset')[cols].agg(lambda x: f'{int(x.fillna(False).astype(bool).sum())}/{len(x)}')
    s['GB'] = (df.groupby('dataset').bytes.sum() / 1e9).round(2)
    s['duration_median_s'] = df.groupby('dataset').duration.median().round(2)
    print(s.to_string())


if __name__ == '__main__':
    main()
