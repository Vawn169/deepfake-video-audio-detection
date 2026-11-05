import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def extract_features_ffpp(csv_path, root_dir):
    df = pd.read_csv(csv_path)
    df = df[df['frames_with_face'] > 0].copy()
    X, y, splits = [], [], []
    for _, r in df.iterrows():
        vid_dir = os.path.join(root_dir, r['split'], r['method'], str(r['video_id']).zfill(3))
        imgs = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')))
        if not imgs:
            continue
        sizes = [os.path.getsize(p) for p in imgs]
        arr = np.asarray(Image.open(imgs[0]), dtype=np.float32)
        X.append([
            float(r['frames_with_face']),
            float(np.mean(sizes)),
            float(np.std(sizes)),
            float(np.mean(arr)),
            float(np.std(arr))
        ])
        y.append(int(r['label']))
        splits.append(r['split'])
    return np.array(X), np.array(y), np.array(splits)

def extract_features_celebdf(csv_path, root_dir):
    df = pd.read_csv(csv_path)
    df = df[df['frames_with_face'] > 0].copy()
    X, y = [], []
    for _, r in df.iterrows():
        sub = 'fake' if int(r['is_fake']) == 1 else 'real'
        folder_name = os.path.splitext(r['path'])[0].replace('/', '__')
        vid_dir = os.path.join(root_dir, sub, folder_name)
        imgs = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')))
        if not imgs:
            continue
        sizes = [os.path.getsize(p) for p in imgs]
        arr = np.asarray(Image.open(imgs[0]), dtype=np.float32)
        X.append([
            float(r['frames_with_face']),
            float(np.mean(sizes)),
            float(np.std(sizes)),
            float(np.mean(arr)),
            float(np.std(arr))
        ])
        y.append(int(r['is_fake']))
    return np.array(X), np.array(y)

def main():
    print("Đang trích xuất siêu dữ liệu FF++ c23...")
    X_ff, y_ff, splits = extract_features_ffpp('reports/preprocess_ffpp_c23.csv', 'data/processed/ffpp_c23')
    X_train, y_train = X_ff[splits == 'train'], y_ff[splits == 'train']
    X_val, y_val = X_ff[splits == 'val'], y_ff[splits == 'val']
    X_test, y_test = X_ff[splits == 'test'], y_ff[splits == 'test']

    print("Đang trích xuất siêu dữ liệu Celeb-DF & Celeb-DF CRF30...")
    X_cd, y_cd = extract_features_celebdf('reports/preprocess_celebdf200.csv', 'data/processed/celebdf')
    X_cd30, y_cd30 = extract_features_celebdf('reports/preprocess_celebdf_crf30.csv', 'data/processed/celebdf_crf30')

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42).fit(X_train_s, y_train)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42).fit(X_train, y_train)

    datasets = [
        ('FF++ c23 Val', X_val, y_val),
        ('FF++ c23 Test', X_test, y_test),
        ('Celeb-DF v2 (200)', X_cd, y_cd),
        ('Celeb-DF CRF30 (200)', X_cd30, y_cd30),
    ]

    rows = []
    print("\n--- KẾT QUẢ KIỂM TRA METADATA SHORTCUT (AUC) ---")
    for name, X_eval, y_eval in datasets:
        X_eval_s = scaler.transform(X_eval)
        auc_lr = roc_auc_score(y_eval, lr.predict_proba(X_eval_s)[:, 1])
        auc_rf = roc_auc_score(y_eval, rf.predict_proba(X_eval)[:, 1])
        print(f"{name:<22} | Videos: {len(y_eval):<4} | LogReg AUC: {auc_lr:.4f} | RF AUC: {auc_rf:.4f}")
        rows.append({'dataset': name, 'num_videos': len(y_eval), 'logreg_auc': round(auc_lr, 4), 'rf_auc': round(auc_rf, 4)})

    pd.DataFrame(rows).to_csv('reports/metadata_shortcut_check.csv', index=False)
    print("\nĐã lưu kết quả vào reports/metadata_shortcut_check.csv")

if __name__ == '__main__':
    main()
