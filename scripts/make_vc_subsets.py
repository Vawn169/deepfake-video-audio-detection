"""Tap con train/validation cho V-C + audio v2c (docs/plan_vc.md, docs/protocol.md muc 2026-11-17). Seed 42, sap truoc khi xao.
MAVOS-DD train 6.000: moi video giả tiếng (audio_fake) + 750 video giả hình mỗi phương pháp (memo lấy đủ phần giả tiếng) + video thật lấp cho đủ;
  bỏ mọi video train có cùng nguồn YouTube với tập test configs/mavos_test_subset.csv (chống rò rỉ).
MAVOS-DD validation 800: tương tự, 100 giả hình mỗi phương pháp + mọi giả tiếng (tối đa 200) + thật lấp cho đủ.
LAV-DF train 6.000 (1.500 mỗi ô modify_video × modify_audio), dev 800 (200 mỗi ô).
  .venv/bin/python scripts/make_vc_subsets.py"""
import json, os, re
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42


def yid(p):
    b = p.split('/')[-1]
    for k in ('.png_', '.jpg_'):
        if k in b:
            b = b.split(k, 1)[1]
    m = re.match(r'(.{11})_\d+_\d+', b)
    return m.group(1) if m else None


def shuf(d, key):
    return d.sort_values(key).sample(frac=1, random_state=SEED)


def mavos(v, n_total, per_method, max_afake):
    a = shuf(v[v.audio_fake], 'video_path').head(max_afake)
    parts = [a]
    for g, d in v[v.video_fake & ~v.video_path.isin(a.video_path)].groupby('generative_method'):
        have = (a.generative_method == g).sum()
        parts.append(shuf(d, 'video_path').head(max(0, per_method - have)))
    s = pd.concat(parts)
    real = shuf(v[~v.video_fake & ~v.audio_fake], 'video_path').head(n_total - len(s))
    return pd.concat([s, real]).sort_values('video_path')


def main():
    v = pd.read_csv('/mnt/d/datasets/mavos_dd/metadata_with_size.csv')
    v['yid'] = v.video_path.map(yid)
    test = pd.read_csv(os.path.join(ROOT, 'configs', 'mavos_test_subset.csv')).merge(v[['video_path', 'yid']], on='video_path')
    bad = set(test.yid.dropna())
    tr = v[(v.split == 'train') & ~v.yid.isin(bad)]
    va = v[(v.split == 'validation') & ~v.yid.isin(bad)]
    print(f'MAVOS train {(v.split == "train").sum()} -> sau khi bo trung nguon voi test {len(tr)}; validation {(v.split == "validation").sum()} -> {len(va)}')
    for name, d, n, per, amax in [('train', tr, 6000, 750, 1600), ('validation', va, 800, 100, 200)]:
        s = mavos(d, n, per, amax)
        s.drop(columns='yid').to_csv(os.path.join(ROOT, 'configs', f'mavos_{name}_subset.csv'), index=False)
        print(f'MAVOS {name}: {len(s)} video, {s["size"].sum() / 1e9:.1f} GB')
        print(pd.crosstab(s.generative_method, [s.video_fake, s.audio_fake]).to_string())
        print(s.language.value_counts().to_dict())
    m = pd.DataFrame(json.load(open('/mnt/d/datasets/lavdf/extracted/LAV-DF/metadata.min.json')))
    m['fake_seconds'] = m.fake_periods.map(lambda ps: round(sum(b - a for a, b in ps), 3))
    for split, out, per in [('train', 'lavdf_train_subset.csv', 1500), ('dev', 'lavdf_dev_subset.csv', 200)]:
        d = shuf(m[m.split == split], 'file')
        s = d.groupby(['modify_video', 'modify_audio']).head(per).sort_values('file')
        s['fake_periods'] = s.fake_periods.map(json.dumps)
        s[['file', 'modify_video', 'modify_audio', 'n_fakes', 'fake_periods', 'fake_seconds', 'duration']].to_csv(
            os.path.join(ROOT, 'configs', out), index=False)
        print(f'LAV-DF {split}: {len(s)} video; {s.groupby(["modify_video", "modify_audio"]).size().to_dict()}')


if __name__ == '__main__':
    main()
