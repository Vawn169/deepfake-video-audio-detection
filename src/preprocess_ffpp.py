import argparse, csv, random, time
from pathlib import Path
from preprocess import make_detector, process_video  # dùng chung với Celeb-DF

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/ff")
ap.add_argument("--manifest", default="configs/manifest_ffpp_c23.csv")
ap.add_argument("--out", default="data/processed/ffpp_c23")
ap.add_argument("--log", default="reports/preprocess_ffpp_c23.csv")
ap.add_argument("--model", default="models/blaze_face_short_range.tflite")
ap.add_argument("--split", default=None)
ap.add_argument("--limit", type=int, default=0, help="tổng số video thử, chia đều 5 nhóm (seed 0)")
ap.add_argument("--frames", type=int, default=16)   # KHÔNG đổi: phải giống Celeb-DF
ap.add_argument("--margin", type=float, default=0.3)
ap.add_argument("--size", type=int, default=224)
a = ap.parse_args()

root, out_root = Path(a.root).expanduser(), Path(a.out)
rows = list(csv.DictReader(open(a.manifest)))
if a.split:
    rows = [r for r in rows if r["split"] == a.split]
if a.limit:
    rng, picked = random.Random(0), []
    for m in ["real", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        g = [r for r in rows if r["method"] == m]
        picked += rng.sample(g, min(a.limit // 5, len(g)))
    rows = picked

detector = make_detector(a.model)
Path(a.log).parent.mkdir(parents=True, exist_ok=True)
new = not Path(a.log).exists()
f = open(a.log, "a", newline="")
w = csv.writer(f)
if new:
    w.writerow(["split", "method", "label", "video_id", "frames_read", "frames_with_face", "seconds", "error"])

t_start, done = time.time(), 0
for i, r in enumerate(rows):
    odir = out_root / r["split"] / r["method"] / r["video_id"]
    if (odir / "_done").exists():
        continue
    t0 = time.time()
    try:
        read, wf, err = process_video(detector, root / r["path"], odir, a.frames, a.margin, a.size)
    except Exception as e:
        read, wf, err = 0, 0, repr(e)
    sec = time.time() - t0
    if not err:
        (odir / "_done").touch()
    w.writerow([r["split"], r["method"], r["label"], r["video_id"], read, wf, f"{sec:.2f}", err])
    f.flush()
    done += 1
    if done % 50 == 0:
        el = time.time() - t_start
        print(f"{i+1}/{len(rows)} | {el/done:.2f}s/video | còn ~{el/done*(len(rows)-i-1)/60:.0f} phút")
print("Xong")
