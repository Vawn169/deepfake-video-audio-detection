import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FFPPFrameDataset, CelebDFFrameDataset
from model import build_model

LEDGER = 'reports/eval_ledger.csv'

# key -> (ten hien thi, ham tao dataset, file log can co)
REGISTRY = {
    'val': ('FF++ c23 Val', lambda: FFPPFrameDataset(split='val'), 'reports/preprocess_ffpp_c23.csv'),
    'test': ('FF++ c23 Test', lambda: FFPPFrameDataset(split='test'), 'reports/preprocess_ffpp_c23.csv'),
    'celebdf': ('Celeb-DF v2 (200)',
                lambda: CelebDFFrameDataset('reports/preprocess_celebdf200.csv', 'data/processed/celebdf'),
                'reports/preprocess_celebdf200.csv'),
    'crf30': ('Celeb-DF CRF30 (200)',
              lambda: CelebDFFrameDataset('reports/preprocess_celebdf_crf30.csv', 'data/processed/celebdf_crf30'),
              'reports/preprocess_celebdf_crf30.csv'),
    'c40': ('FF++ c40 Test',
            lambda: FFPPFrameDataset(csv_path='reports/preprocess_ffpp_c40.csv',
                                     root_dir='data/processed/ffpp_c40', split='test'),
            'reports/preprocess_ffpp_c40.csv'),
}


def compute_eer_and_thresh(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.absolute(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def bootstrap_auc_ci(y_true, y_score, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    aucs, n = [], len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def get_video_predictions(ds, model, device, batch_size=64, num_workers=4):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    vid_probs, vid_labels = {}, {}
    with torch.no_grad():
        for x, y, vids in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                probs = torch.sigmoid(model(x).squeeze(1)).float().cpu().numpy()
            for p, l, v in zip(probs, y.numpy(), vids):
                vid_probs.setdefault(v, []).append(float(p))
                vid_labels[v] = int(l)
    vids_sorted = sorted(vid_probs.keys())
    y_true = np.array([vid_labels[v] for v in vids_sorted])
    y_score = np.array([np.mean(vid_probs[v]) for v in vids_sorted])
    return vids_sorted, y_true, y_score


def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def already_evaluated(ckpt_path, name):
    if not os.path.exists(LEDGER):
        return 0
    df = pd.read_csv(LEDGER)
    return int(((df['ckpt'] == ckpt_path) & (df['dataset'] == name)).sum())


def append_csv(path, rows):
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        new_df = pd.concat([pd.read_csv(path), new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='duong dan best_effnet_b0.pt cua mot seed')
    ap.add_argument('--tag', default=None, help='ten thu muc ket qua (mac dinh: ten thu muc cua checkpoint)')
    ap.add_argument('--datasets', nargs='+', default=['val', 'test', 'celebdf', 'crf30'],
                    choices=list(REGISTRY.keys()))
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Chua tim thay {args.ckpt}")
    tag = args.tag or os.path.basename(os.path.dirname(os.path.abspath(args.ckpt)))
    out_dir = os.path.join('reports', 'eval', tag)
    os.makedirs(os.path.join(out_dir, 'predictions'), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Checkpoint {args.ckpt}: epoch {ckpt['epoch']} (Val AUC luc huan luyen = {ckpt['val_auc']:.4f})")

    # Val luon chay truoc de lay nguong EER (Val khong phai tap test).
    order = ['val'] + [k for k in args.datasets if k != 'val']
    gh, now = git_hash(), time.strftime('%Y-%m-%d %H:%M:%S')
    val_thresh = None
    summary_rows, ledger_rows = [], []

    print("\n--- KET QUA DANH GIA ---")
    for key in order:
        name, builder, req_csv = REGISTRY[key]
        if not os.path.exists(req_csv):
            print(f"BO QUA {name}: chua tim thay {req_csv}")
            continue
        if key != 'val':
            n_prev = already_evaluated(args.ckpt, name)
            if n_prev:
                print(f"!!! CANH BAO: {name} da duoc danh gia {n_prev} lan voi {args.ckpt}. "
                      f"Giao thuc: moi tap test chi danh gia mot lan voi cau hinh da chot.")

        ds = builder()
        vids, y_true, y_score = get_video_predictions(ds, model, device, args.batch_size, args.num_workers)
        auc = float(roc_auc_score(y_true, y_score))
        ci_low, ci_high = bootstrap_auc_ci(y_true, y_score)
        eer, eer_thresh = compute_eer_and_thresh(y_true, y_score)

        if key == 'val':
            val_thresh = eer_thresh
            print(f"--> Nguong EER chon tren Val: {val_thresh:.4f}")

        p05 = (y_score >= 0.5).astype(int)
        pvt = (y_score >= val_thresh).astype(int)
        acc_05, f1_05 = float(accuracy_score(y_true, p05)), float(f1_score(y_true, p05, zero_division=0))
        acc_vt, f1_vt = float(accuracy_score(y_true, pvt)), float(f1_score(y_true, pvt, zero_division=0))

        print(f"{name:<22} | Videos: {len(vids):<4} | AUC: {auc:.4f} [{ci_low:.4f}, {ci_high:.4f}] | "
              f"EER: {eer:.4f} | Acc@ValThr: {acc_vt:.4f} | F1@ValThr: {f1_vt:.4f} (Acc@0.5: {acc_05:.4f})")

        ledger_rows.append({'time': now, 'ckpt': args.ckpt, 'dataset': name, 'auc': round(auc, 4)})
        if key == 'val' and 'val' not in args.datasets:
            continue  # chi dung de lay nguong, khong ghi vao bang
        summary_rows.append({
            'dataset': name, 'num_videos': len(vids), 'auc': round(auc, 4),
            'auc_ci95_low': round(ci_low, 4), 'auc_ci95_high': round(ci_high, 4),
            'eer': round(eer, 4), 'val_threshold': round(val_thresh, 4),
            'acc_val_thresh': round(acc_vt, 4), 'f1_val_thresh': round(f1_vt, 4),
            'acc_05': round(acc_05, 4), 'f1_05': round(f1_05, 4),
            'ckpt': args.ckpt, 'ckpt_epoch': ckpt['epoch'], 'ckpt_val_auc': round(float(ckpt['val_auc']), 4),
            'git': gh, 'time': now,
        })
        safe = name.lower().replace(' ', '_').replace('+', 'p').replace('(', '').replace(')', '')
        pd.DataFrame({'video_id': vids, 'label': y_true, 'prob': y_score}).to_csv(
            os.path.join(out_dir, 'predictions', f'{safe}_preds.csv'), index=False)

    append_csv(os.path.join(out_dir, 'summary.csv'), summary_rows)
    append_csv(LEDGER, ledger_rows)
    print(f"\nDa ghi bang tong hop vao {out_dir}/summary.csv va nhat ky {LEDGER}")


if __name__ == '__main__':
    main()
