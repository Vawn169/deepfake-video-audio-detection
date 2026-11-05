import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def get_transforms(train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class FFPPFrameDataset(Dataset):
    def __init__(self, csv_path='reports/preprocess_ffpp_c23.csv', root_dir='data/processed/ffpp_c23',
                 split='train', transform=None, max_videos=None, shuffle_labels=False, seed=42):
        df = pd.read_csv(csv_path, dtype={'video_id': str})
        df = df[(df['split'] == split) & (df['frames_with_face'] > 0)].copy()

        if max_videos is not None:
            half = max_videos // 2
            df_real = df[df['label'] == 0]
            df_fake = df[df['label'] == 1]
            df_real = df_real.sample(n=min(half, len(df_real)), random_state=seed)
            df_fake = df_fake.sample(n=min(half, len(df_fake)), random_state=seed)
            df = pd.concat([df_real, df_fake]).reset_index(drop=True)

        if shuffle_labels:
            df['label'] = df['label'].sample(frac=1.0, random_state=seed).values

        self.samples = []
        missing_videos = []
        for _, r in df.iterrows():
            vid_str = str(r['video_id']).strip()
            if '_' not in vid_str:
                vid_str = vid_str.zfill(3)
            vid_unique_id = f"{r['method']}_{vid_str}"
            vid_dir = os.path.join(root_dir, str(r['split']), str(r['method']), vid_str)
            imgs = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')))
            if not imgs:
                missing_videos.append(vid_dir)
                continue
            for img_path in imgs:
                self.samples.append((img_path, float(r['label']), vid_unique_id))

        if missing_videos:
            raise RuntimeError(
                f"Phát hiện {len(missing_videos)}/{len(df)} video có frames_with_face > 0 nhưng không tìm thấy ảnh! "
                f"Ví dụ: {missing_videos[:3]}"
            )

        self.num_videos = len(df)
        self.transform = transform or get_transforms(train=(split == 'train'))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, vid_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32), vid_id

class CelebDFFrameDataset(Dataset):
    def __init__(self, csv_path='reports/preprocess_celebdf200.csv', root_dir='data/processed/celebdf', transform=None):
        df = pd.read_csv(csv_path)
        df = df[df['frames_with_face'] > 0].copy()
        self.samples = []
        missing_videos = []
        for _, r in df.iterrows():
            sub = 'fake' if int(r['is_fake']) == 1 else 'real'
            folder = os.path.splitext(str(r['path']))[0].replace('/', '__')
            vid_dir = os.path.join(root_dir, sub, folder)
            imgs = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')))
            if not imgs:
                missing_videos.append(vid_dir)
                continue
            for img_path in imgs:
                self.samples.append((img_path, float(r['is_fake']), folder))

        if missing_videos:
            raise RuntimeError(
                f"Phát hiện {len(missing_videos)}/{len(df)} video Celeb-DF bị thiếu ảnh! Ví dụ: {missing_videos[:3]}"
            )

        self.num_videos = len(df)
        self.transform = transform or get_transforms(train=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, vid_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32), vid_id
