import argparse
import csv
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import get_transforms
from model import build_model

METHODS = ['real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = build_model(pretrained=False).to(device)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    return m, ck


def gradcam_video(model, batch, device):
    """Tra ve (probs tung frame, frame tot nhat, cam 224x224 chuan hoa 0-1)."""
    store = {}
    hook = model.features[-1].register_forward_hook(lambda mod, i, o: store.update(act=o))
    x = batch.to(device)
    model.zero_grad(set_to_none=True)
    logits = model(x).squeeze(1)  # float32, khong AMP
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    best = int(np.argmax(probs))
    act = store['act']
    grad = torch.autograd.grad(logits[best], act)[0][best]  # (1280, 7, 7)
    hook.remove()
    w = grad.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * act[best].detach()).sum(0)).cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    return probs, best, cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints_aug_s44/best_effnet_b0.pt')
    ap.add_argument('--log', default='reports/preprocess_ffpp_c23.csv')
    ap.add_argument('--root', default='data/processed/ffpp_c23')
    ap.add_argument('--out', default='data/gradcam_out')
    ap.add_argument('--n_per_method', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ck = load_model(args.ckpt, device)
    print(f"Checkpoint {args.ckpt}: epoch {ck['epoch']}")
    tf = get_transforms(train=False)
    os.makedirs(args.out, exist_ok=True)

    rows = [r for r in csv.DictReader(open(args.log, newline=''))
            if r['split'] == 'test' and int(r['frames_with_face']) > 0]
    rng = random.Random(args.seed)
    picked = []
    for m in METHODS:
        g = sorted([r for r in rows if r['method'] == m], key=lambda r: r['video_id'])
        picked += rng.sample(g, min(args.n_per_method, len(g)))

    cams, summary = [], []
    for r in picked:
        vid = r['video_id'].strip()
        if '_' not in vid:
            vid = vid.zfill(3)
        vdir = os.path.join(args.root, 'test', r['method'], vid)
        files = sorted(f for f in os.listdir(vdir) if f.endswith('.jpg'))
        batch = torch.stack([tf(Image.open(os.path.join(vdir, f)).convert('RGB')) for f in files])
        probs, best, cam = gradcam_video(model, batch, device)

        bgr = cv2.imread(os.path.join(vdir, files[best]))
        bgr = cv2.resize(bgr, (224, 224))
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(bgr, 0.55, heat, 0.45, 0)
        cv2.imwrite(os.path.join(args.out, f"{r['method']}_{vid}.jpg"), np.hstack([bgr, overlay]))

        cams.append(cam.astype(np.float16))
        summary.append({'method': r['method'], 'video_id': vid, 'label': r['label'],
                        'best_frame': files[best], 'prob_best': round(float(probs[best]), 4),
                        'prob_mean': round(float(probs.mean()), 4)})
        print(f"{r['method']:<15} {vid:<8} label={r['label']} p_best={probs[best]:.3f} p_mean={probs.mean():.3f}")

    np.save(os.path.join(args.out, 'cams.npy'), np.stack(cams))
    with open(os.path.join(args.out, 'summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"Xong: {len(summary)} video, ket qua o {args.out}")


if __name__ == '__main__':
    main()
