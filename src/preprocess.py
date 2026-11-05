import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def make_detector(model_path):
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceDetectorOptions(base_options=base, min_detection_confidence=0.5)
    return vision.FaceDetector.create_from_options(opts)


def crop_face(frame_bgr, det, margin, size):
    h, w = frame_bgr.shape[:2]
    bb = det.bounding_box
    cx = bb.origin_x + bb.width / 2
    cy = bb.origin_y + bb.height / 2
    side = max(bb.width, bb.height) * (1 + margin)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    return cv2.resize(frame_bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def process_video(detector, video_path, out_dir, n_frames, margin, size):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0, 0, "khong doc duoc so frame"
    idxs = [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    out_dir.mkdir(parents=True, exist_ok=True)
    read = with_face = 0
    for k, idx in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        read += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not res.detections:
            continue
        best = max(res.detections, key=lambda d: d.categories[0].score)
        face = crop_face(frame, best, margin, size)
        if face is None:
            continue
        cv2.imwrite(str(out_dir / f"f{k:02d}.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with_face += 1
    cap.release()
    return read, with_face, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--model", default="models/blaze_face_short_range.tflite")
    ap.add_argument("--limit", type=int, default=0, help="tong so video thu (chia deu that/gia)")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=224)
    a = ap.parse_args()
    assert a.frames >= 2

    root, out_root = Path(a.root).expanduser(), Path(a.out)
    rows = list(csv.DictReader(open(a.csv)))
    if a.limit:
        half = a.limit // 2
        real = [r for r in rows if r["is_fake"] == "0"][:half]
        fake = [r for r in rows if r["is_fake"] == "1"][:half]
        rows = real + fake

    detector = make_detector(a.model)
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    stats = []
    with open(a.log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "is_fake", "frames_read", "frames_with_face", "seconds", "error"])
        for r in rows:
            rel = r["path"]
            name = rel.split("/")[0] + "__" + Path(rel).stem
            sub = "fake" if r["is_fake"] == "1" else "real"
            t0 = time.time()
            try:
                read, wf, err = process_video(detector, root / rel, out_root / sub / name,
                                              a.frames, a.margin, a.size)
            except Exception as e:
                read, wf, err = 0, 0, repr(e)
            sec = time.time() - t0
            w.writerow([rel, r["is_fake"], read, wf, f"{sec:.2f}", err])
            stats.append((int(r["is_fake"]), wf, sec, err))
            print(f"{rel}: {wf}/{a.frames} frame co mat, {sec:.1f}s {err}")

    n = len(stats)
    print("\n=== TONG KET ===")
    print("so video:", n, "| giay/video trung binh:", round(sum(s[2] for s in stats) / n, 2))
    for lab, name in [(0, "that"), (1, "gia")]:
        grp = [s for s in stats if s[0] == lab]
        if grp:
            zero = sum(1 for s in grp if s[1] == 0)
            avg = sum(s[1] for s in grp) / len(grp)
            print(f"video {name}: {len(grp)} | khong tim thay mat nao: {zero} | frame co mat TB: {avg:.1f}")


if __name__ == "__main__":
    main()
