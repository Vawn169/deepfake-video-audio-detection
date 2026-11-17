"""Danh gia V-B (docs/protocol.md, muc 2026-11-10 va 2026-11-15). Viet truoc khi danh gia bat ky checkpoint nao tren tap test V-B.
Tap (that = 140 video FF++ test goc; gia = configs/manifest_vb.csv phan test):
  Test-A = inswapper + simswap; Test-A-enh = inswapper_gfpgan; Test-B = ghost + hyperswap; Test-C = roop.
  Moi tap co hai ban: goc (/mnt/d/faceswap/vb) va nen cuoc goi (/mnt/d/faceswap/vb_call, scripts/make_call_vb.py).
  R1, R2 (chi video that): chi cham 3 checkpoint V-B (9 checkpoint cu da danh gia ngay 10/11).
Tien xu ly giong eval_realworld.py: 16 frame, MediaPipe, cat mat le 0,3, 224, JPEG 95; video duoi 4 frame mat khong cham, bao rieng.
Diem video = trung binh xac suat gia cua cac frame. Nguong bao nham 5% cua moi checkpoint = phan vi 95 diem video that FF++ Val
(reports/eval/<checkpoint>/predictions/ffpp_c23_val_preds.csv); nguong EER FF++ Val bao kem.
Chi so: AUC, EER, ty le phat hien va bao nham o nguong 5% (Wilson 95%). Them bang tung cong cu va bang tren cac cap ma moi cong cu
deu tao duoc va du mat. Moi checkpoint danh gia mot lan moi tap (eval_ledger.csv); co roi thi dung.
  .venv/bin/python src/eval_vb.py"""
import csv, datetime, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from eval_de24 import MIN_FACES, LEDGER, eer, face_crops, video_threshold, make_detector, get_transforms, build_model
from eval_realworld import wilson

VB, VB_CALL = '/mnt/d/faceswap/vb', '/mnt/d/faceswap/vb_call'
FF_REAL = os.path.join(ROOT, 'data', 'ff', 'original_sequences', 'youtube', 'c23', 'videos')
OUT = os.path.join(ROOT, 'reports', 'eval_vb')
FAMILY = {'baseline': 'checkpoints_s{}', 'aug': 'checkpoints_aug_s{}', 'V-A': 'checkpoints_video_v2a_s{}', 'V-B': 'checkpoints_video_vb_s{}'}
# --only V-SBI (protocol 2026-11-15): chi cham ho moi, ket qua rieng reports/eval_vb_V-SBI/
if '--only' in sys.argv:
    _f = sys.argv[sys.argv.index('--only') + 1]; FAMILY = {_f: {'V-SBI': 'checkpoints_video_vsbi_s{}', 'V-C': 'checkpoints_video_vc_s{}'}[_f]}
    OUT = OUT + '_' + _f
CKPTS = {f'{d.format(s)}/best_effnet_b0.pt': fam for fam, d in FAMILY.items() for s in (42, 43, 44)}
VB_CKPTS = [c for c, f in CKPTS.items() if f in ('V-B', 'V-SBI', 'V-C')]   # ho cham R1/R2
SETS = {'Test-A': ['inswapper', 'simswap'], 'Test-A-enh': ['inswapper_gfpgan'], 'Test-B': ['ghost', 'hyperswap'], 'Test-C': ['roop']}
VERS = {'goc': '', 'call': ' call'}


def fpr5(ck):
    d = pd.read_csv(os.path.join(ROOT, 'reports', 'eval', os.path.dirname(ck), 'predictions', 'ffpp_c23_val_preds.csv'))
    return float(np.percentile(d[d.label == 0].prob, 95))


def rate(s, thr):
    k, n = int((s >= thr).sum()), len(s)
    lo, hi = wilson(k, n)
    return (round(k / n, 4) if n else np.nan), round(lo, 4), round(hi, 4)


def videos():
    vb = pd.read_csv(os.path.join(ROOT, 'configs', 'manifest_vb.csv')); vb = vb[vb.split == 'test']
    ids = sorted({v for p in json.load(open(os.path.join(ROOT, 'configs', 'splits', 'test.json'))) for v in p})
    rows = []
    for ver, root in (('goc', VB), ('call', VB_CALL)):
        rows += [dict(ver=ver, group='real', video_id=v, label=0, path=os.path.join(FF_REAL, f'{v}.mp4') if ver == 'goc'
                      else os.path.join(VB_CALL, 'real', 'test', f'{v}.mp4')) for v in ids]
        rows += [dict(ver=ver, group=r.method, video_id=r.video_id, label=1, path=os.path.join(root, r.path)) for r in vb.itertuples()]
    for name in ('R1', 'R2'):
        m = pd.read_csv(os.path.join(ROOT, 'configs', f'realworld_{name}.manifest.csv'))
        rows += [dict(ver=name, group=name, video_id=os.path.basename(p), label=0, path=p) for p in m[m.modality == 'video'].path]
    return pd.DataFrame(rows)


def main():
    names = [f'{s}{v}' for s in SETS for v in VERS.values()]
    done = pd.read_csv(LEDGER)
    clash = done[(done.ckpt.isin(CKPTS) & done.dataset.isin(names)) | (done.ckpt.isin(VB_CKPTS) & done.dataset.isin(['R1 video', 'R2 video']))]
    if len(clash):
        sys.exit(f'Da danh gia roi, khong danh gia lai:\n{clash}')
    missing = [c for c in CKPTS if not os.path.exists(os.path.join(ROOT, 'reports', 'eval', os.path.dirname(c), 'predictions', 'ffpp_c23_val_preds.csv'))]
    if missing:
        sys.exit(f'Chua co du doan FF++ Val (chay src/evaluate.py truoc): {missing}')
    df = videos()
    miss = [p for p in df.path if not os.path.exists(p)]
    if miss:
        sys.exit(f'Thieu {len(miss)} video, vd {miss[:3]}')
    os.makedirs(OUT, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for ck in CKPTS:
        st = torch.load(os.path.join(ROOT, ck), map_location=dev, weights_only=False)
        m = build_model(pretrained=False).to(dev); m.load_state_dict(st['model_state_dict']); m.eval(); models[ck] = m
    thr5 = {ck: fpr5(ck) for ck in CKPTS}; thre = {ck: video_threshold(ck) for ck in CKPTS}
    print('Nguong bao nham 5%:', {c: round(t, 4) for c, t in thr5.items()}, flush=True)
    det = make_detector(os.path.join(ROOT, 'models', 'blaze_face_short_range.tflite')); tf = get_transforms(train=False)
    scores = {ck: [] for ck in CKPTS}; nf = []
    for i, r in enumerate(df.itertuples()):
        c = face_crops(det, r.path); nf.append(len(c))
        x = torch.stack([tf(im) for im in c]).to(dev) if c else None
        for ck, m in models.items():
            if x is None or (r.ver in ('R1', 'R2') and ck not in VB_CKPTS):
                scores[ck].append(np.nan); continue
            with torch.no_grad(), torch.autocast(dev, enabled=(dev == 'cuda')):
                scores[ck].append(float(torch.sigmoid(m(x).squeeze(1)).float().mean()))
        if i % 100 == 0:
            print(f'{i}/{len(df)} video', flush=True)
    df['n_faces'] = nf
    for ck in CKPTS:
        df[ck] = scores[ck]
    df.to_csv(os.path.join(OUT, 'preds.csv'), index=False)

    ok = df.n_faces >= MIN_FACES; rows, ledger = [], []
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def block(name, ver, real, fake, ck, kind):
        r_all, f_all = df[real], df[fake]; r, f = df[real & ok], df[fake & ok]
        row = dict(table=kind, set=name, ver=ver, ckpt=ck, family=CKPTS[ck], thr_fpr5=round(thr5[ck], 4), thr_eer=thre[ck],
                   n_real=len(r_all), n_real_scored=len(r), n_fake=len(f_all), n_fake_scored=len(f))
        if len(f):
            row['detected'], row['detected_lo'], row['detected_hi'] = rate(f[ck].values, thr5[ck])
            row['detected_eerthr'] = rate(f[ck].values, thre[ck])[0]
        if len(r):
            row['false_alarm'], row['false_alarm_lo'], row['false_alarm_hi'] = rate(r[ck].values, thr5[ck])
            row['false_alarm_eerthr'] = rate(r[ck].values, thre[ck])[0]
        if len(r) and len(f):
            y = np.r_[np.zeros(len(r)), np.ones(len(f))]; s = np.r_[r[ck].values, f[ck].values]
            row['auc'], row['eer'] = round(float(roc_auc_score(y, s)), 4), round(eer(y, s), 4)
        rows.append(row); return row

    for ver, suf in VERS.items():
        real = (df.ver == ver) & (df.group == 'real')
        for ck in CKPTS:
            for name, meths in SETS.items():
                row = block(name, ver, real, (df.ver == ver) & df.group.isin(meths), ck, 'set')
                ledger.append([now, ck, name + suf, f"{row.get('auc', float('nan')):.4f}"])
            for meth in sum(SETS.values(), []):
                block(meth, ver, real, (df.ver == ver) & (df.group == meth), ck, 'method')
        # cap chung: video_id co mat (du mat) o moi cong cu cua ban nay
        fk = df[(df.ver == ver) & (df.label == 1) & ok]
        meths = sum(SETS.values(), []); common = set.intersection(*[set(fk[fk.group == m].video_id) for m in meths])
        for ck in CKPTS:
            for meth in meths:
                block(f'{meth} (cap chung, {len(common)})', ver, real, (df.ver == ver) & (df.group == meth) & df.video_id.isin(common), ck, 'common')
    for name in ('R1', 'R2'):
        for ck in VB_CKPTS:
            row = block(name, name, (df.ver == name), np.zeros(len(df), bool), ck, 'realworld')
            ledger.append([now, ck, f'{name} video', f"{row['false_alarm']:.4f}"])

    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUT, 'summary.csv'), index=False)
    cols = [c for c in ('auc', 'eer', 'detected', 'false_alarm') if c in res]
    agg = res.groupby(['table', 'set', 'ver', 'family'], sort=False)[cols].agg(['mean', 'std', 'min', 'max']).round(4)
    agg.to_csv(os.path.join(OUT, 'summary_by_family.csv'))
    with open(LEDGER, 'a', newline='') as fh:
        csv.writer(fh).writerows(ledger)
    print('Khong du mat (<4 frame):', df[~ok].groupby(['ver', 'group']).size().to_dict())
    print(res[res.table.isin(['set', 'realworld'])].groupby(['set', 'ver', 'family'], sort=False)[cols].mean().round(4).to_string())


if __name__ == '__main__':
    main()
