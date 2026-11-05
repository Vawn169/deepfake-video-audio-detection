import csv, subprocess, sys
from pathlib import Path

SRC, DST = Path("data/celeb-df-v2"), Path("data/celeb-df-v2-crf30")
rows = list(csv.DictReader(open("configs/celebdf_test_subset.csv")))
fail = 0
for i, r in enumerate(rows):
    src, dst = SRC / r["path"], DST / r["path"]
    if dst.exists() and dst.stat().st_size > 0:
        continue
    if not src.exists():
        print("THIEU:", src); fail += 1; continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-crf", "30", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", str(dst)]
    if subprocess.run(cmd).returncode != 0:
        print("LOI:", src); fail += 1; dst.unlink(missing_ok=True)
    if (i + 1) % 20 == 0:
        print(f"{i+1}/{len(rows)}")
print("Xong. So video loi/thieu:", fail)
sys.exit(1 if fail else 0)
