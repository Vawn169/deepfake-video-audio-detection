#!/bin/bash
# Danh gia V-C + audio v2c (docs/protocol.md 2026-11-17). Moi buoc co canh nhiet (guard.sh); loi thi dung.
cd /home/van/deepfake-video
G=/home/van/faceswap/guard.sh; PY=.venv/bin/python
for s in 42 43 44; do
  $G eval_vc_old_s$s $PY -u src/evaluate.py --ckpt checkpoints_video_vc_s$s/best_effnet_b0.pt --datasets val test celebdf crf30 c40 || exit 1
done
$G eval_vc_vb $PY -u src/eval_vb.py --only V-C || exit 1
$G eval_v2c_old $PY -u -c "import sys; sys.path.insert(0,'src'); from audio_v2 import evaluate; evaluate('c')" || exit 1
$G eval_vc_ood $PY -u src/eval_ood.py --sets MAVOS-DD AV-Deepfake1M LAV-DF SocialDF --families V-C "audio v2c" || exit 1
echo "$(date +%T) DANH GIA V-C XONG"
