#!/bin/bash
# Chay 1 lenh co canh nhiet GPU: >=83C tam dung den <=70C; >85C hai lan lien: dung han. Loi: dung, khong chay lai.
#   guard.sh <ten> <lenh...>   (log reports/<ten>.log, nhiet /mnt/d/datasets/gpu_temp_<ten>.csv)
cd /home/van/deepfake-video
name=$1; shift
TL=/mnt/d/datasets/gpu_temp_$name.csv; LOG=reports/$name.log
temp() { nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1; }
stop() { echo "DA DUNG $name luc $(date +%T): $1"; exit 1; }
echo "$(date +%T) bat dau $name: $*"
"$@" >> $LOG 2>&1 & pid=$!; hot=0; pauses=0
while kill -0 $pid 2>/dev/null; do
  t=$(temp); [[ "$t" =~ ^[0-9]+$ ]] || { kill $pid; stop "khong doc duoc GPU"; }
  echo "$(date +%F_%T),$t" >> $TL
  if [ "$t" -gt 85 ]; then hot=$((hot+1)); else hot=0; fi
  [ $hot -ge 2 ] && { pkill -P $pid; kill $pid; stop "GPU ${t}C hai lan lien"; }
  if [ "$t" -ge 83 ]; then
    pkill -STOP -P $pid; kill -STOP $pid; pauses=$((pauses+1))
    while [ "$(temp)" -gt 70 ]; do sleep 15; done
    kill -CONT $pid; pkill -CONT -P $pid; continue
  fi
  sleep 20
done
wait $pid; rc=$?
[ $rc -ne 0 ] && { tail -5 $LOG; stop "loi (ma $rc), xem $LOG"; }
echo "$(date +%T) xong $name ($pauses lan tam dung lam mat). Max nhiet: $(cut -d, -f2 $TL | sort -n | tail -1)C"
