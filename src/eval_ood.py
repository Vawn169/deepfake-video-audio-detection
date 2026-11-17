"""Danh gia ngoai mien MAVOS-DD, AV-Deepfake1M, LAV-DF, SocialDF (docs/protocol.md, muc 2026-11-16 "cach danh gia ngoai mien").
Viet truoc khi chay mo hinh tren bon bo nay.
Nhanh video: nhan hinh; nhanh audio: nhan tieng; fusion (checkpoint API): nhan chung. Nguong lay tu FF++ Val / ASVspoof dev.
  .venv/bin/python src/eval_ood.py --sets MAVOS-DD AV-Deepfake1M LAV-DF
  .venv/bin/python src/eval_ood.py --sets SocialDF
  --no-ledger --limit N --out <thu muc tam>: chi de chay thu"""
import argparse, csv, datetime, os, sys
import collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_de24 import MIN_FACES, LEDGER, eer, face_crops, load_audio, w2v_layers, mlp_logit, video_threshold, make_detector, get_transforms, build_model
from eval_realworld import wilson
from eval_vb import fpr5
from sklearn.metrics import roc_auc_score

D = '/mnt/d/datasets'
VFAM = {'baseline': 'checkpoints_s{}', 'aug': 'checkpoints_aug_s{}', 'V-A': 'checkpoints_video_v2a_s{}',
        'V-B': 'checkpoints_video_vb_s{}', 'V-SBI': 'checkpoints_video_vsbi_s{}'}
AFAM = {'audio cu': 'checkpoints_audio_s{}', 'audio v2a': 'checkpoints_audio_v2a_s{}', 'audio v2b': 'checkpoints_audio_v2b_s{}'}
VCK = {f'{d.format(s)}/best_effnet_b0.pt': f for f, d in VFAM.items() for s in (42, 43, 44)}
ACK = {f'{d.format(s)}/best_mlp.pt': f for f, d in AFAM.items() for s in (42, 43, 44)}
API_V, API_A, API_A_FPR5 = 'checkpoints_video_vb_s44/best_effnet_b0.pt', 'checkpoints_audio_v2b_s43/best_mlp.pt', 7.6671
# --families V-C "audio v2c" (protocol 2026-11-17): chi cham ho moi, ket qua reports/eval_ood_vc/, khong fusion (API van V-B/v2b)
NEW = {'V-C': 'checkpoints_video_vc_s{}', 'audio v2c': 'checkpoints_audio_v2c_s{}'}
if '--families' in sys.argv:
    _fs = sys.argv[sys.argv.index('--families') + 1:]
    _fs = _fs[:next((i for i, x in enumerate(_fs) if x.startswith('--')), len(_fs))]
    VCK = {f'{NEW[f].format(s)}/best_effnet_b0.pt': f for f in _fs if f.startswith('V-') for s in (42, 43, 44)}
    ACK = {f'{NEW[f].format(s)}/best_mlp.pt': f for f in _fs if f.startswith('audio') for s in (42, 43, 44)}


def manifest(name):
    """Cot: path, yv (hinh gia), ya (tieng gia), y (gia chung), group, language."""
    if name == 'MAVOS-DD':
        m = pd.read_csv(os.path.join(ROOT, 'configs', 'mavos_test_subset.csv'))
        return pd.DataFrame(dict(path=[os.path.join(D, 'mavos_dd', p) for p in m.video_path], yv=m.video_fake.astype(int),
                                 ya=m.audio_fake.astype(int), y=(m.label == 'fake').astype(int), group=m.generative_method,
                                 language=m.language, open_set=np.where(m.open_set_model, 'model_moi', 'model_cu') + '/' +
                                 np.where(m.open_set_language, 'ngon_ngu_moi', 'ngon_ngu_cu')))
    if name == 'AV-Deepfake1M':
        m = pd.read_csv(os.path.join(ROOT, 'configs', 'av1m_val_subset.csv'))
        return pd.DataFrame(dict(path=[os.path.join(D, 'av1m', 'val_subset', p) for p in m.file], yv=m.video_fake.astype(int),
                                 ya=m.audio_fake.astype(int), y=(m.modify_type != 'real').astype(int), group=m.modify_type))
    if name == 'LAV-DF':
        m = pd.read_csv(os.path.join(ROOT, 'configs', 'lavdf_test_subset.csv'))
        g = np.where(m.modify_video & m.modify_audio, 'both_modified', np.where(m.modify_video, 'visual_modified',
                     np.where(m.modify_audio, 'audio_modified', 'real')))
        return pd.DataFrame(dict(path=[os.path.join(D, 'lavdf', 'extracted', 'LAV-DF', f) for f in m.file], yv=m.modify_video.astype(int),
                                 ya=m.modify_audio.astype(int), y=(m.modify_video | m.modify_audio).astype(int), group=g))
    if name == 'SocialDF':
        m = pd.read_csv(os.path.join(ROOT, 'reports', 'socialdf_download.csv')).groupby('url').tail(1)
        m = m[m.status == 'ok']
        y = (m.label == 'fake').astype(int).values
        return pd.DataFrame(dict(path=[os.path.join(D, 'socialdf', f) for f in m.file], yv=y, ya=y, y=y, group=m.label.values))
    raise ValueError(name)


def rate(s, thr):
    k, n = int((s >= thr).sum()), len(s)
    lo, hi = wilson(k, n)
    return (round(k / n, 4) if n else np.nan), round(lo, 4), round(hi, 4)


def block(y, s, thr5, thre):
    r = dict(n_pos=int((y == 1).sum()), n_neg=int((y == 0).sum()))
    for lab, key in ((1, 'detected'), (0, 'false_alarm')):
        if (y == lab).sum():
            r[key], r[key + '_lo'], r[key + '_hi'] = rate(s[y == lab], thr5)
            if thre is not None:
                r[key + '_eerthr'] = rate(s[y == lab], thre)[0]
    if len(set(y)) == 2:
        r['auc'], r['eer'] = round(float(roc_auc_score(y, s)), 4), round(eer(y, s), 4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', nargs='+', required=True)
    ap.add_argument('--out', default=os.path.join(ROOT, 'reports', 'eval_ood'))
    ap.add_argument('--no-ledger', action='store_true'); ap.add_argument('--limit', type=int)
    ap.add_argument('--families', nargs='+')
    a = ap.parse_args()
    fusion = not a.families
    if a.families and a.out == os.path.join(ROOT, 'reports', 'eval_ood'):
        a.out = os.path.join(ROOT, 'reports', 'eval_ood_vc')
    done = pd.read_csv(LEDGER)
    if fusion:
        names = [f'{s} {k}' for s in a.sets for k in ('video', 'audio', 'fusion')]
        hit = done.dataset.isin(names)
    else:   # ho moi: kiem tra theo tung checkpoint
        hit = done.ckpt.isin(list(VCK) + list(ACK)) & done.dataset.str.split(' ').str[0].isin(a.sets)
    if not a.no_ledger and hit.any():
        sys.exit(f'Da danh gia roi:\n{done[hit]}')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    vm = {}
    for ck in VCK:
        st = torch.load(os.path.join(ROOT, ck), map_location=dev, weights_only=False)
        m = build_model(pretrained=False).to(dev); m.load_state_dict(st['model_state_dict']); m.eval(); vm[ck] = m
    ac = {ck: torch.load(os.path.join(ROOT, ck), map_location='cpu', weights_only=False) for ck in ACK}
    thr5 = {ck: fpr5(ck) for ck in VCK}; thre = {ck: video_threshold(ck) for ck in VCK}
    from transformers import Wav2Vec2Model
    w2v = Wav2Vec2Model.from_pretrained(os.path.join(ROOT, 'models', 'wav2vec2-base')).to(dev).eval()
    tf = get_transforms(train=False)
    import threading
    local = threading.local()

    def load(p):   # chay tren luong phu: cat mat + doc tieng
        if not hasattr(local, 'det'):
            local.det = make_detector(os.path.join(ROOT, 'models', 'blaze_face_short_range.tflite'))
        return face_crops(local.det, p), load_audio(p)

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for name in a.sets:
        df = manifest(name)
        if a.limit:
            df = df.groupby('group').head(a.limit).reset_index(drop=True)
        miss = [p for p in df.path if not os.path.exists(p)]
        if miss:
            sys.exit(f'{name}: thieu {len(miss)} file, vd {miss[:3]}')
        out = os.path.join(a.out, name); os.makedirs(out, exist_ok=True)
        print(f'{name}: {len(df)} video', flush=True)
        vs = {ck: [] for ck in VCK}; asc = {ck: [] for ck in ACK}; nf, has_a = [], []
        def bounded(ex, paths, k=16):   # 17/11: ex.map nap truoc ca tap -> het RAM; chi giu toi da k video cho cham
            paths, q = list(paths), collections.deque()
            for p in paths:
                q.append(ex.submit(load, p))
                if len(q) >= k:
                    yield q.popleft().result()
            while q:
                yield q.popleft().result()

        with ThreadPoolExecutor(4) as ex:
            for i, (crops, w) in enumerate(bounded(ex, df.path)):
                nf.append(len(crops))
                x = torch.stack([tf(im) for im in crops]).to(dev) if crops else None
                for ck, m in vm.items():
                    if x is None:
                        vs[ck].append(np.nan); continue
                    with torch.no_grad(), torch.autocast(dev, enabled=(dev == 'cuda')):
                        vs[ck].append(float(torch.sigmoid(m(x).squeeze(1)).float().mean()))
                has_a.append(w is not None)
                if w is not None:
                    raw, trm = w2v_layers(w2v, w, False, dev), w2v_layers(w2v, w, True, dev)
                    for ck, c in ac.items():
                        asc[ck].append(float(mlp_logit(c, (trm if c.get('trim') else raw)[None])[0]))
                else:
                    for ck in ac:
                        asc[ck].append(np.nan)
                if i % 500 == 0:
                    print(f'  {i}/{len(df)}', flush=True)
        df['n_faces'], df['has_audio'] = nf, has_a
        for ck in VCK:
            df['v:' + ck] = vs[ck]
        for ck in ACK:
            df['a:' + ck] = asc[ck]
        df.to_csv(os.path.join(out, 'preds.csv'), index=False)

        rows, ledger = [], []
        okv, oka = (df.n_faces >= MIN_FACES).values, df.has_audio.values
        for ck, fam in VCK.items():
            s, y = df['v:' + ck].values, df.yv.values
            base = dict(branch='video', ckpt=ck, family=fam, thr=round(thr5[ck], 4), coverage=round(float(okv.mean()), 4))
            r = block(y[okv], s[okv], thr5[ck], thre[ck]); rows.append({**base, 'table': 'all', 'subset': 'all', **r})
            ledger.append([now, ck, f'{name} video', f"{r.get('auc', float('nan')):.4f}"])
            neg = okv & (y == 0)
            for col in [c for c in ('group', 'language', 'open_set') if c in df]:
                for g in sorted(df[col].unique()):
                    pos = okv & (y == 1) & (df[col] == g).values
                    if pos.sum():   # gia cua nhom nay so voi moi am tinh
                        m_ = pos | neg
                        rows.append({**base, 'table': col, 'subset': g, **block(y[m_], s[m_], thr5[ck], thre[ck])})
                    negg = neg & (df[col] == g).values
                    if negg.sum() and not pos.sum():   # nhom chi co am tinh (vd video that / chi gia tieng)
                        rows.append({**base, 'table': col, 'subset': g, **block(y[negg], s[negg], thr5[ck], thre[ck])})
        for ck, fam in ACK.items():
            c = ac[ck]; s, y = df['a:' + ck].values, df.ya.values; t = float(c['threshold'])
            base = dict(branch='audio', ckpt=ck, family=fam, thr=round(t, 4), coverage=round(float(oka.mean()), 4))
            r = block(y[oka], s[oka], t, None); rows.append({**base, 'table': 'all', 'subset': 'all', **r})
            if ck == API_A:
                r5 = block(y[oka], s[oka], API_A_FPR5, None)
                rows.append({**base, 'thr': API_A_FPR5, 'table': 'all_fpr5', 'subset': 'all', **r5})
            ledger.append([now, ck, f'{name} audio', f"{r.get('auc', float('nan')):.4f}"])
            neg = oka & (y == 0)
            for col in [c for c in ('group', 'language') if c in df]:
                for g in sorted(df[col].unique()):
                    pos = oka & (y == 1) & (df[col] == g).values
                    m_ = (pos | neg) if pos.sum() else (neg & (df[col] == g).values)
                    if m_.sum():
                        rows.append({**base, 'table': col, 'subset': g, **block(y[m_], s[m_], t, None)})
        fr = {}
        if fusion:
            # fusion: checkpoint API, nhan chung
            both = okv & oka; y = df.y.values[both]
            pv = df['v:' + API_V].values[both]; la = df['a:' + API_A].values[both]; pa = 1 / (1 + np.exp(-la))
            fr = dict(branch='fusion', ckpt=f'{API_V}+{API_A}', family='API', coverage=round(float(both.mean()), 4), n=int(both.sum()))
            if len(set(y)) == 2:
                fr.update(auc_mean=round(float(roc_auc_score(y, (pv + pa) / 2)), 4), auc_video_only=round(float(roc_auc_score(y, pv)), 4),
                          auc_audio_only=round(float(roc_auc_score(y, la)), 4))
            flag = (pv >= thr5[API_V]) | (la >= API_A_FPR5)
            for key, s_ in (('or_rule', flag), ('video_only', pv >= thr5[API_V]), ('audio_only', la >= API_A_FPR5)):
                for lab, k2 in ((1, 'detected'), (0, 'false_alarm')):
                    if (y == lab).sum():
                        kk, nn = int(s_[y == lab].sum()), int((y == lab).sum()); lo, hi = wilson(kk, nn)
                        fr[f'{key}_{k2}'], fr[f'{key}_{k2}_lo'], fr[f'{key}_{k2}_hi'] = round(kk / nn, 4), round(lo, 4), round(hi, 4)
            rows.append({**fr, 'table': 'fusion', 'subset': 'all'})
            ledger.append([now, f'{API_V}+{API_A}', f'{name} fusion', f"{fr.get('auc_mean', float('nan')):.4f}"])

        res = pd.DataFrame(rows); res.to_csv(os.path.join(out, 'summary.csv'), index=False)
        cols = [c for c in ('auc', 'eer', 'detected', 'false_alarm') if c in res]
        agg = res[res.branch != 'fusion'].groupby(['branch', 'table', 'subset', 'family'], sort=False)[cols].agg(['mean', 'std', 'min', 'max']).round(4)
        agg.to_csv(os.path.join(out, 'summary_by_family.csv'))
        if not a.no_ledger:
            with open(LEDGER, 'a', newline='') as fh:
                csv.writer(fh).writerows(ledger)
        print(f'{name}: bao phu mat {okv.mean():.3f}, co tieng {oka.mean():.3f}')
        print(res[res.table == 'all'].groupby(['branch', 'family'], sort=False)[cols].mean().round(4).to_string())
        print(pd.DataFrame([fr]).T.to_string(), flush=True)


if __name__ == '__main__':
    main()
