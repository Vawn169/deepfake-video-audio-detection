import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FFPPFrameDataset, get_transforms
from model import build_model


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_video_level(model, loader, device, criterion=None):
    model.eval()
    vid_probs, vid_labels = {}, {}
    losses = []
    with torch.no_grad():
        for x, y, vids in loader:
            x = x.to(device, non_blocking=True)
            y_dev = y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x).squeeze(1)
                if criterion is not None:
                    losses.append(criterion(logits, y_dev).item())
                probs = torch.sigmoid(logits).cpu().numpy()
            for p, l, v in zip(probs, y.numpy(), vids):
                vid_probs.setdefault(v, []).append(float(p))
                vid_labels[v] = int(l)
    vids_sorted = sorted(vid_probs.keys())
    y_true = np.array([vid_labels[v] for v in vids_sorted])
    y_score = np.array([np.mean(vid_probs[v]) for v in vids_sorted])
    y_pred = (y_score >= 0.5).astype(int)
    return {
        'loss': float(np.mean(losses)) if losses else 0.0,
        'auc': float(roc_auc_score(y_true, y_score)),
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'num_videos': len(vids_sorted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Dang nap du lieu FF++ c23 (Train & Val)...")
    train_ds = FFPPFrameDataset(split='train', transform=get_transforms(train=True))
    val_ds = FFPPFrameDataset(split='val', transform=get_transforms(train=False))

    labels_arr = np.array([s[1] for s in train_ds.samples])
    n_real = float(np.sum(labels_arr == 0))
    n_fake = float(np.sum(labels_arr == 1))
    pos_weight_val = n_real / max(n_fake, 1.0)
    print(f"Train frames: {len(train_ds)} ({train_ds.num_videos} videos) | Real: {int(n_real)}, Fake: {int(n_fake)} | pos_weight: {pos_weight_val:.3f}")
    print(f"Val frames:   {len(val_ds)} ({val_ds.num_videos} videos)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_auc = 0.0
    history = []
    print(f"\n--- BAT DAU HUAN LUYEN TREN {device} ({args.epochs} EPOCHS) ---")
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        scheduler.step()
        vm = evaluate_video_level(model, val_loader, device, criterion)
        elapsed = time.time() - t0
        tr_loss = float(np.mean(train_losses))

        is_best = vm['auc'] > best_val_auc
        if is_best:
            best_val_auc = vm['auc']
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(),
                        'val_auc': best_val_auc, 'args': vars(args)},
                       os.path.join(args.out_dir, 'best_effnet_b0.pt'))

        mark = " [*BEST*]" if is_best else ""
        print(f"Epoch {ep:02d}/{args.epochs:02d} ({elapsed:.1f}s) | Train Loss: {tr_loss:.4f} | Val Loss: {vm['loss']:.4f} | Val AUC: {vm['auc']:.4f} | Val Acc: {vm['acc']:.4f} | Val F1: {vm['f1']:.4f}{mark}")
        history.append({'epoch': ep, 'train_loss': round(tr_loss, 4), 'val_loss': round(vm['loss'], 4),
                        'val_auc': round(vm['auc'], 4), 'val_acc': round(vm['acc'], 4),
                        'val_f1': round(vm['f1'], 4), 'seconds': round(elapsed, 1)})
        pd.DataFrame(history).to_csv('reports/train_baseline_history.csv', index=False)

    print(f"\nHoan tat. Best Val Video AUC: {best_val_auc:.4f} (luu tai {args.out_dir}/best_effnet_b0.pt)")


if __name__ == '__main__':
    main()
