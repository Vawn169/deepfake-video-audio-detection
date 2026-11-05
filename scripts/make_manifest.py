import json, csv, sys
from pathlib import Path

ROOT = Path.home() / "deepfake-video"
DATA = ROOT / "data" / "ff"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]

rows, ids_by_split = [], {}
for split in ["train", "val", "test"]:
    pairs = json.load(open(ROOT / "configs/splits" / f"{split}.json"))
    ids_by_split[split] = {i for p in pairs for i in p}
    for a, b in pairs:
        for vid in (a, b):
            rows.append(dict(split=split, label=0, method="real",
                             video_id=vid,
                             path=f"original_sequences/youtube/c23/videos/{vid}.mp4"))
        for m in METHODS:
            for t, s in ((a, b), (b, a)):
                rows.append(dict(split=split, label=1, method=m,
                                 video_id=f"{t}_{s}",
                                 path=f"manipulated_sequences/{m}/c23/videos/{t}_{s}.mp4"))

# 1) mọi file trong manifest phải tồn tại
missing = [r["path"] for r in rows if not (DATA / r["path"]).exists()]
print("File thiếu:", len(missing), missing[:5])

# 2) rò rỉ: danh tính (cả target và source) giữa các split phải không giao nhau
leak = False
for x, y in [("train", "val"), ("train", "test"), ("val", "test")]:
    n = len(ids_by_split[x] & ids_by_split[y])
    print(f"Giao nhau {x}-{y}: {n}")
    leak |= n > 0
for r in rows:  # mỗi video giả chỉ dùng id thuộc đúng split của nó
    for i in r["video_id"].split("_"):
        if i not in ids_by_split[r["split"]]:
            leak = True
            print("RÒ RỈ:", r)
            break

# 3) thống kê
from collections import Counter
c = Counter((r["split"], "real" if r["label"] == 0 else "fake") for r in rows)
for k in sorted(c): print(k, c[k])

out = ROOT / "configs" / "manifest_ffpp_c23.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("Đã ghi", out, len(rows), "dòng")
sys.exit(1 if (leak or missing) else 0)
