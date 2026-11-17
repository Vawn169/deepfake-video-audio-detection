#!/bin/bash
# V-C + audio v2c (docs/protocol.md 2026-11-17): cho tai xong -> cat mat -> dac trung audio MAVOS -> train audio v2c -> train V-C 3 seed.
# Buoc nao loi thi dung (khong chay lai).
cd /home/van/deepfake-video
G=/home/van/faceswap/guard.sh; PY=.venv/bin/python
while pgrep -f scripts/download_vc.py >/dev/null || pgrep -f "audio_v2c.py feats" >/dev/null; do sleep 30; done
grep -q "HOAN TAT" reports/download_vc.log || { echo "DUNG: download_vc chua HOAN TAT"; tail -3 reports/download_vc.log; exit 1; }
grep -q "XONG lavdf_dev" reports/audio_v2c_feats_lavdf.log || { echo "DUNG: dac trung audio LAV-DF chua xong"; exit 1; }
$G preprocess_vc $PY -u src/preprocess_vc.py --workers 4 || exit 1
$G audio_v2c_feats_mavos $PY -u src/audio_v2c.py feats --sets mavos_train mavos_val || exit 1
$G audio_v2c_train $PY -u src/audio_v2c.py train || exit 1
for s in 42 43 44; do
  $G train_vc_s$s $PY -u src/train_vc.py --seed $s --out_dir checkpoints_video_vc_s$s || exit 1
done
echo "$(date +%T) PIPELINE V-C XONG"
