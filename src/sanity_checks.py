import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import FFPPFrameDataset, get_transforms
from model import build_model

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def eval_video_auc(model, loader, device):
    model.eval()
    vid_probs, vid_labels = {}, {}
    with torch.no_grad():
        for x, y, vids in loader:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(x).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
            for p, l, v in zip(probs, y.numpy(), vids):
                vid_probs.setdefault(v, []).append(float(p))
                vid_labels[v] = int(l)
    y_true = [vid_labels[v] for v in vid_probs]
    y_score = [np.mean(vid_probs[v]) for v in vid_probs]
    return roc_auc_score(y_true, y_score)

def run_overfit_check(device):
    print("\n=== 1. KIỂM TRA OVERFIT TRÊN 32 VIDEO (16 REAL, 16 FAKE) ===")
    seed_everything(42)
    ds = FFPPFrameDataset(split='train', max_videos=32, transform=get_transforms(train=False))
    print(f"Đã nạp thành công {ds.num_videos}/32 video ({len(ds)} frames) - Không thiếu video nào.")
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

    model = build_model(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    auc = 0.5
    for ep in range(1, 9):
        model.train()
        losses = []
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(x).squeeze(1)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        auc = eval_video_auc(model, loader, device)
        print(f"Epoch {ep}/8 | Loss: {np.mean(losses):.4f} | Train Video AUC: {auc:.4f}")
    return float(auc)

def run_shuffled_label_check(device, seeds=(42, 43, 44)):
    print("\n=== 2. KIỂM TRA XÁO NHÃN NGẪU NHIÊN QUA 3 SEED (42, 43, 44) ===")
    val_ds = FFPPFrameDataset(split='val', max_videos=60, shuffle_labels=False, seed=42)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    aucs = []
    for s in seeds:
        seed_everything(s)
        train_ds = FFPPFrameDataset(split='train', max_videos=100, shuffle_labels=True, seed=s)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)

        model = build_model(pretrained=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        crit = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        for ep in range(1, 5):
            model.train()
            for x, y, _ in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    loss = crit(model(x).squeeze(1), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
        val_auc = eval_video_auc(model, val_loader, device)
        aucs.append(float(val_auc))
        print(f"[Seed {s}] Sau 4 epoch học nhãn xáo trộn -> Val Video AUC: {val_auc:.4f}")

    mean_auc = float(np.mean(aucs))
    print(f"--> Trung bình Shuffled Label Val AUC (3 seeds): {mean_auc:.4f}")
    return aucs, mean_auc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang chạy Sanity Checks trên thiết bị: {device}")
    overfit_auc = run_overfit_check(device)
    shuffled_aucs, shuffled_mean = run_shuffled_label_check(device)
    os.makedirs('reports', exist_ok=True)
    with open('reports/sanity_check_results.json', 'w') as f:
        json.dump({
            'overfit_32videos_auc': round(overfit_auc, 4),
            'shuffled_label_val_aucs': [round(x, 4) for x in shuffled_aucs],
            'shuffled_label_val_auc_mean': round(shuffled_mean, 4)
        }, f, indent=2)
    print("\nĐã lưu kết quả vào reports/sanity_check_results.json!")
