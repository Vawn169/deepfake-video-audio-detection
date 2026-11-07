import os
import sys
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FFPPFrameDataset, get_transforms
from model import build_model
from train import evaluate_video_level, seed_everything
from augment import RandomDegrade, RandomSocialDegrade, social_degrade

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class DegradedValDataset(FFPPFrameDataset):
    """Val SocialDegrade CO DINH: moi anh co seed rieng theo duong dan (docs/protocol.md, muc 2026-11-08)."""
    def __getitem__(self, idx):
        img_path, label, vid_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        img = social_degrade(img, random.Random(img_path))
        return self.transform(img), torch.tensor(label, dtype=torch.float32), vid_id


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
    train_ds = FFPPFrameDataset(split='train', transform=aug_tf)
    val_ds = FFPPFrameDataset(split='val', transform=get_transforms(train=False))
    val_deg_ds = DegradedValDataset(split='val', transform=transforms.Compose([transforms.ToTensor(), NORM]))

    labels_arr = np.array([s[1] for s in train_ds.samples])
    n_real, n_fake = float(np.sum(labels_arr == 0)), float(np.sum(labels_arr == 1))
    pos_weight_val = n_real / max(n_fake, 1.0)
    print(f"Train frames: {len(train_ds)} ({train_ds.num_videos} videos) | pos_weight: {pos_weight_val:.3f}")
    print(f"Val: {val_ds.num_videos} videos | Val-degraded: {val_deg_ds.num_videos} videos")

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
                                   pin_memory=True, drop_last=sh)
    train_loader, val_loader, val_deg_loader = mk(train_ds, True), mk(val_ds, False), mk(val_deg_ds, False)

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_score, history = -1.0, []
    print(f"\n--- V-A (SOCIAL) TRAIN seed {args.seed} tren {device} ({args.epochs} EPOCHS) ---")
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

        vm = evaluate_video_level(model, val_loader, device, criterion)
        vd = evaluate_video_level(model, val_deg_loader, device, criterion)
        score = (vm['auc'] + vd['auc']) / 2.0
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'val_auc': vm['auc'],
                        'val_auc_deg': vd['auc'], 'select_score': score, 'args': vars(args)},
                       os.path.join(args.out_dir, 'best_effnet_b0.pt'))
        el, tl = time.time() - t0, float(np.mean(losses))
        print(f"Epoch {ep:02d}/{args.epochs:02d} ({el:.1f}s) | Train Loss: {tl:.4f} | Val AUC: {vm['auc']:.4f} | "
              f"ValDeg AUC: {vd['auc']:.4f} | Score: {score:.4f}{' [*BEST*]' if is_best else ''}")
        history.append({'epoch': ep, 'seconds': round(el, 1), 'train_loss': round(tl, 4),
                        'val_auc': round(vm['auc'], 4), 'val_deg_auc': round(vd['auc'], 4),
                        'select_score': round(score, 4), 'is_best': int(is_best)})
        pd.DataFrame(history).to_csv(f'reports/train_v2a_history_s{args.seed}.csv', index=False)

    print(f"\nHoan tat. Best select score: {best_score:.4f} (luu tai {args.out_dir}/best_effnet_b0.pt)")


if __name__ == '__main__':
    main()
