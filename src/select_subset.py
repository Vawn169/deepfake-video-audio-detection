import argparse
import csv
import random
from collections import Counter
from pathlib import Path

REAL_DIRS = {"Celeb-real", "YouTube-real"}
FAKE_DIRS = {"Celeb-synthesis"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="configs/celebdf_test_subset.csv")
    ap.add_argument("--n-real", type=int, default=100)
    ap.add_argument("--n-fake", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = Path(a.root).expanduser()
    rows = []
    text = (root / "List_of_testing_videos.txt").read_text()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        label, rel = line.split(maxsplit=1)
        rows.append((int(label), rel.strip()))

    print("so dong trong danh sach test:", len(rows))
    print("dem theo (nhan, thu muc):", dict(Counter((l, r.split("/")[0]) for l, r in rows)))

    # Kiem tra quy uoc nhan: thu muc that phai la 1, thu muc gia phai la 0
    for label, rel in rows:
        d = rel.split("/")[0]
        if d in REAL_DIRS and label != 1:
            raise SystemExit(f"Nhan bat thuong: {label} {rel}")
        if d in FAKE_DIRS and label != 0:
            raise SystemExit(f"Nhan bat thuong: {label} {rel}")
        if d not in REAL_DIRS | FAKE_DIRS:
            raise SystemExit(f"Thu muc la: {rel}")

    missing = [rel for _, rel in rows if not (root / rel).is_file()]
    print("so file khong ton tai:", len(missing))
    if missing:
        raise SystemExit("Co file thieu, dung lai. Vi du: " + missing[0])

    real = sorted(rel for l, rel in rows if l == 1)
    fake = sorted(rel for l, rel in rows if l == 0)
    print("tong video that:", len(real), "| tong video gia:", len(fake))
    if len(real) < a.n_real or len(fake) < a.n_fake:
        raise SystemExit("Khong du video de lay mau")

    rng = random.Random(a.seed)
    pick_real = sorted(rng.sample(real, a.n_real))
    pick_fake = sorted(rng.sample(fake, a.n_fake))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "is_fake", "source_dir"])
        for rel in pick_real:
            w.writerow([rel, 1, 0, rel.split("/")[0]])
        for rel in pick_fake:
            w.writerow([rel, 0, 1, rel.split("/")[0]])

    print("da ghi", out, "voi seed", a.seed)
    print("thanh phan:", dict(Counter(r.split("/")[0] for r in pick_real + pick_fake)))


if __name__ == "__main__":
    main()
