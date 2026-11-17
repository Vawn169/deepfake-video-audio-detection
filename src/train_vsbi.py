"""V-SBI (docs/plan_v3_faceswap.md muc 9; docs/protocol.md muc 2026-11-15): giong het V-B (src/train_vb.py: du lieu, augmentation,
sieu tham so, pos_weight, chon checkpoint), them SBI tao khi train: moi anh mat THAT trong tap train, xac suat 0,5, thanh anh SBI
nhan gia (src/sbi.py), truoc augmentation V-B. Anh that khong co diem moc thi giu nguyen. Val khong co SBI."""
import os, sys, time, random, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FFPPFrameDataset, get_transforms
from model import build_model
from train import evaluate_video_level, seed_everything
from augment import RandomDegrade, RandomSocialDegrade
from train_vb import NORM, vb_dataset, merged
from sbi import self_blend

LM = 'data/processed/sbi_landmarks_train.npz'
P_SBI = 0.5


class SBITrain(Dataset):
    def __init__(self, base, transform):
        d = np.load(LM)
        self.lm = {str(p): l for p, l in zip(d['paths'], d['lms'])}
        self.samples, self.num_videos, self.transform = base.samples, base.num_videos, transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y, v = self.samples[i]
        img = Image.open(p).convert('RGB')
        if y == 0 and p in self.lm and random.random() < P_SBI:
            s = self_blend(np.asarray(img), self.lm[p].astype(np.float32), random)
            if s is not None:
                img, y = Image.fromarray(s), 1.0
        return self.transform(img), torch.tensor(y, dtype=torch.float32), v


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
    base = merged(FFPPFrameDataset(split='train', transform=aug_tf), vb_dataset('train', aug_tf))
    train_ds = SBITrain(base, aug_tf)
    val_ds = FFPPFrameDataset(split='val', transform=val_tf)
    val_real = FFPPFrameDataset(split='val', transform=val_tf)
    val_real.samples = [s for s in val_real.samples if s[1] == 0]
    val_real.num_videos = len({s[2] for s in val_real.samples})
    val_new = merged(val_real, vb_dataset('val', val_tf))

    labels_arr = np.array([s[1] for s in train_ds.samples])   # nhan goc, giong V-B (khong tinh lai theo SBI)
    n_real, n_fake = float(np.sum(labels_arr == 0)), float(np.sum(labels_arr == 1))
    pos_weight_val = n_real / max(n_fake, 1.0)
    print(f"Train frames: {len(train_ds)} ({train_ds.num_videos} videos) | pos_weight: {pos_weight_val:.3f} | "
          f"anh that co diem moc: {sum(1 for s in train_ds.samples if s[1] == 0 and s[0] in train_ds.lm)}/{int(n_real)}")
    print(f"Val FF++: {val_ds.num_videos} videos | Val moi: {val_new.num_videos} videos")

    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh, num_workers=args.workers,
                                   pin_memory=True, drop_last=sh)
    train_loader, val_loader, val_new_loader = mk(train_ds, True), mk(val_ds, False), mk(val_new, False)

    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_score, history = -1.0, []
    print(f"\n--- V-SBI TRAIN seed {args.seed} tren {device} ({args.epochs} EPOCHS) ---")
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
        vn = evaluate_video_level(model, val_new_loader, device, criterion)
        score = (vm['auc'] + vn['auc']) / 2.0
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({'epoch': ep, 'model_state_dict': model.state_dict(), 'val_auc': vm['auc'],
                        'val_new_auc': vn['auc'], 'select_score': score, 'args': vars(args)},
                       os.path.join(args.out_dir, 'best_effnet_b0.pt'))
        el, tl = time.time() - t0, float(np.mean(losses))
        print(f"Epoch {ep:02d}/{args.epochs:02d} ({el:.1f}s) | Train Loss: {tl:.4f} | Val AUC: {vm['auc']:.4f} | "
              f"ValMoi AUC: {vn['auc']:.4f} | Score: {score:.4f}{' [*BEST*]' if is_best else ''}", flush=True)
        history.append({'epoch': ep, 'seconds': round(el, 1), 'train_loss': round(tl, 4),
                        'val_auc': round(vm['auc'], 4), 'val_new_auc': round(vn['auc'], 4),
                        'select_score': round(score, 4), 'is_best': int(is_best)})
        pd.DataFrame(history).to_csv(f'reports/train_vsbi_history_s{args.seed}.csv', index=False)

    print(f"\nHoan tat. Best select score: {best_score:.4f} (luu tai {args.out_dir}/best_effnet_b0.pt)")


if __name__ == '__main__':
    main()
