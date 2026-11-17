"""V-C: nhu V-B (src/train_vb.py) + MAVOS-DD train + LAV-DF train (docs/protocol.md muc 2026-11-17).
Chon checkpoint: trung binh AUC muc video cua FF++ Val, Val moi V-B, MAVOS validation, LAV-DF dev.
  .venv/bin/python src/train_vc.py --seed 42 --out_dir checkpoints_video_vc_s42"""
import os, sys, time, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FFPPFrameDataset, get_transforms
from model import build_model
from train import evaluate_video_level, seed_everything
from augment import RandomDegrade, RandomSocialDegrade
from train_vb import NORM, vb_dataset, merged

VC_CSV, VC_ROOT = 'reports/preprocess_vc.csv', 'data/processed/vc'


def vc_dataset(split, transform, prefix):
    ds = FFPPFrameDataset(csv_path=VC_CSV, root_dir=VC_ROOT, split=split, transform=transform)
    ds.samples = [s for s in ds.samples if s[2].startswith(prefix)]
    ds.num_videos = len({s[2] for s in ds.samples})
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aug_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(0.5),
                                 RandomDegrade(p=0.7), RandomSocialDegrade(p=0.5), transforms.ToTensor(), NORM])
    val_tf = get_transforms(train=False)
    train_ds = merged(merged(merged(FFPPFrameDataset(split='train', transform=aug_tf), vb_dataset('train', aug_tf)),
                             vc_dataset('train', aug_tf, 'mavos')), vc_dataset('train', aug_tf, 'lavdf'))
    val_ds = FFPPFrameDataset(split='val', transform=val_tf)
    val_real = FFPPFrameDataset(split='val', transform=val_tf)
    val_real.samples = [s for s in val_real.samples if s[1] == 0]
    val_real.num_videos = len({s[2] for s in val_real.samples})
    vals = {'ffpp': val_ds, 'vb': merged(val_real, vb_dataset('val', val_tf)),
            'mavos': vc_dataset('val', val_tf, 'mavos'), 'lavdf': vc_dataset('val', val_tf, 'lavdf')}

    labels_arr = np.array([s[1] for s in train_ds.samples])
    n_real, n_fake = float(np.sum(labels_arr == 0)), float(np.sum(labels_arr == 1))
    pos_weight_val = n_real / max(n_fake, 1.0)
    print(f"Train frames: {len(train_ds)} ({train_ds.num_videos} videos) | pos_weight: {pos_weight_val:.3f}")
    print('Val: ' + ', '.join(f'{k} {v.num_videos} videos' for k, v in vals.items()), flush=True)

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
                                   pin_memory=True, drop_last=sh)
    train_loader = mk(train_ds, True)
    val_loaders = {k: mk(v, False) for k, v in vals.items()}

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_score, history = -1.0, []
    print(f"\n--- V-C TRAIN seed {args.seed} tren {device} ({args.epochs} EPOCHS) ---", flush=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = criterion(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        scheduler.step()

        aucs = {k: evaluate_video_level(model, l, device, criterion)['auc'] for k, l in val_loaders.items()}
        score = float(np.mean(list(aucs.values())))
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'val_aucs': aucs,
                        'select_score': score, 'args': vars(args)}, os.path.join(args.out_dir, 'best_effnet_b0.pt'))
        el, tl = time.time() - t0, float(np.mean(losses))
        print(f"Epoch {ep:02d}/{args.epochs:02d} ({el:.1f}s) | Train Loss: {tl:.4f} | "
              + ' | '.join(f'{k} {v:.4f}' for k, v in aucs.items())
              + f" | Score: {score:.4f}{' [*BEST*]' if is_best else ''}", flush=True)
        history.append({'epoch': ep, 'seconds': round(el, 1), 'train_loss': round(tl, 4),
                        **{f'auc_{k}': round(v, 4) for k, v in aucs.items()}, 'select_score': round(score, 4), 'is_best': int(is_best)})
        pd.DataFrame(history).to_csv(f'reports/train_vc_history_s{args.seed}.csv', index=False)

    print(f"\nHoan tat. Best select score: {best_score:.4f} (luu tai {args.out_dir}/best_effnet_b0.pt)")


if __name__ == '__main__':
    main()
