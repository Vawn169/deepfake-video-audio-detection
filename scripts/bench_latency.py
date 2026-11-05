import argparse, csv, json, statistics as st, subprocess, sys, time
from pathlib import Path

def call(url, path, endpoint):
    t0 = time.perf_counter()
    r = subprocess.run(["curl", "-s", "-F", f"file=@{path}", f"{url}{endpoint}"],
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        d = {"status": "bad_response"}
    return d, wall

def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def summ(name, xs):
    return f"{name}: mean={st.mean(xs):.3f}  p50={pct(xs,50):.3f}  p95={pct(xs,95):.3f}  max={max(xs):.3f}  n={len(xs)}"

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--tag", default="run")
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--endpoint", default="/predict?heatmap=false")
ap.add_argument("videos", nargs="+")
a = ap.parse_args()

vids = [v for v in a.videos if Path(v).is_file()]
if not vids: sys.exit("Khong co video nao.")
for v in vids[:a.warmup]: call(a.url, v, a.endpoint)
print(f"Warm-up xong ({a.warmup} request, khong tinh).")

rows = []
for v in vids:
    d, w = call(a.url, v, a.endpoint)
    rows.append(dict(video=Path(v).name, status=d.get("status"), score=d.get("score"),
                     frames_with_detection=d.get("frames_with_detection"),
                     frames_used=d.get("frames_used"),
                     server_s=d.get("processing_seconds"), client_s=round(w, 3)))
    print(rows[-1])

ok = [r for r in rows if r["status"] == "ok"]
print(f"\n== TONG KET [{a.tag}] == thanh cong {len(ok)}/{len(rows)}")
if ok:
    print(summ("Server (processing_seconds)", [r["server_s"] for r in ok if r["server_s"] is not None]))
    print(summ("Client (tong thoi gian goi)", [r["client_s"] for r in ok]))
out = Path("reports") / f"latency_{a.tag}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("Da luu:", out)
