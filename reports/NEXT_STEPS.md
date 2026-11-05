# Trạng thái 2026-10-17 (cuối ngày)
Xong: huấn luyện 3 seed, sanity, kiểm tra siêu dữ liệu, ảnh ghép, đánh giá FF++ val/test và Celeb-DF 200.
Kết quả: FF++ Test AUC 0,9889 ± 0,0042; Celeb-DF 200 AUC 0,7807 ± 0,0215 (3 seed).
Tiếp theo: (3) Celeb-DF CRF30 bằng ffmpeg rồi evaluate --datasets crf30; (4) FF++ c40 chỉ test;
(5) tuần 5 augmentation nén, ghi "Sửa đổi" vào protocol trước khi train lại; Grad-CAM; API; audio.
Không chạy lại test với checkpoint đã đánh giá (xem reports/eval_ledger.csv).

# 2026-10-19: CRF30 xong (AUC TB 0,7176 ± 0,0183, commit 972b82d). FF++ c40 đã tải đủ 5x1000 video tại /mnt/d/ff_c40.
Tiếp theo: Prompt C0 + C3 (manifest c40 chỉ test 700 video, cắt mặt), xem báo cáo rồi C4 (thêm 'c40' vào REGISTRY, đánh giá mỗi seed một lần), C5 commit.
Sau đó: tuần 5 augmentation nén (bước train duy nhất, ~1 giờ cho 3 seed; ghi "Sửa đổi" vào docs/protocol.md trước, chọn theo validation).

# 2026-10-19/20: augmentation xong và đã đánh giá. AUC TB (baseline -> aug): c23 test 0,9889->0,9807; c40 0,7987->0,8714; Celeb-DF 0,7807->0,8373; CRF30 0,7176->0,7649.
Tiếp theo: Grad-CAM (checkpoint chính: chọn theo Val, ghi vào protocol), FastAPI + Docker + latency, audio (tùy chọn), báo cáo.

# Dừng giữa bước 8 (API). Đã xong: api_core.py, api.py, kiểm tra khớp điểm 10 video (max |d| 0,0044), server chạy, /health OK. Ngưỡng 0,5951 (aug_s44).
Còn lại: test ca real_862 qua API (kỳ vọng frames_used < frames_with_detection), thử 1 video thật và 1 giả, đo latency CPU, viết Dockerfile (image PyTorch CPU), rồi audio và báo cáo.

# 2026-10-26: Cổng 2 (Mức 1) đạt.
Docker: build OK (đã thêm libegl1 libgles2 vào Dockerfile vì MediaPipe cần libEGL), container chạy --cpus=4 --memory=4g, /health OK.
Parity trong container: real_862 0,072 (frames_used 7/16), real_186 0,0303, FaceSwap 280_249 0,9999, khớp ngoài container. heatmap=true OK (heatmap_jpeg_base64).
Latency (20 video: 10 thật + 10 FaceSwap, CPU): cpu_local: n=20 server mean=1.14s p50=0.98s p95=1.58s max=1.58s
Latency (trong container): docker_cpu: n=20 server mean=1.57s p50=1.31s p95=2.49s max=2.50s
Quyết định Mức 2 (audio): chờ quyết định ở bước 7.
Tiếp theo: kiểm tra điều kiện tải ASVspoof 2019 LA, rồi quyết định audio hay chuyển sang unit test, báo cáo, slide.

# 2026-11-01: Mức 2 (audio) có số liệu trên eval chính thức ASVspoof 2019 LA.
Wav2vec2-base đóng băng, tầng 2 (chọn theo dev), MLP 768-256-1, 3 seed. Eval: EER 1,57% ± 0,03%, AUC 0,9987 ± 0,0001. Dev EER 0,11-0,15%.
Ngưỡng EER chọn trên dev: bỏ sót spoof 6,3-7,2%, báo nhầm bonafide 0,4% trên eval (lệch dev-eval).
Hạn chế: kiểm tra đường tắt im lặng đầu file trên dev AUC 0,695 (ghi trong protocol trước khi xem eval); 3 seed dùng chung đặc trưng nên SD thấp.
Lỗi quy trình đã ghi nhận: siêu tham số MLP được commit sau khi chạy huấn luyện (dev), trước khi xem eval.
Tiếp theo: endpoint audio trong API (Tuần 8), unit test, báo cáo, slide. Sau Tuần 8 không chạy thí nghiệm mới.

# 2026-11-03: Tuần 8, endpoint audio trong API xong (giao thức ở docs/protocol.md, commit dac38cf).
POST /predict_audio: checkpoint audio_s43, ngưỡng logit 2,9935 (dev); nhận .wav/.flac hoặc video (ffmpeg tách tiếng mono 16 kHz); video không có tiếng trả no_audio; định dạng khác 415, file hỏng 422. /predict (video) giữ nguyên.
Kiểm tra trước khi viết code: hai snapshot wav2vec2-base trong cache HF giống hệt nhau (218 tensor); tính lại đặc trưng 3 file dev trên CPU khớp dev.npz (max |d| 0,0004). Trọng số chép vào models/wav2vec2-base (không đưa lên git), container chạy được khi không có mạng.
Parity audio (scripts/api_parity_audio.py, 10 file eval, 5 bonafide + 5 spoof, seed 0): max |d logit| 0,0049, mean 0,0028; nhãn API trùng offline cả 10 (2 spoof bị chấm bonafide ở cả hai: đúng hành vi của mô hình). Log: reports/api_parity_audio.log.
Trong container: kết quả trùng ngoài container (flac logit 17,2475; video ghép tiếng AAC 16,3516; wav stereo 44,1 kHz 17,1219; video không tiếng no_audio; real_000 c40 qua /predict 0,6733, offline 0,6783).
Image deepfake-api:cpu 3,84 GB (trước 2,64 GB). RAM container khoảng 0,7 GB.
Latency Docker CPU (--cpus=4 --memory=4g, bỏ 2 warm-up, n=20, server):
  video (cùng 20 video Tuần 6): mean=1.675s p50=1.327s p95=2.636s max=2.663s (Tuần 6: 1.57/1.31/2.49/2.50; mỗi bên đo một lần, không kết luận chậm hơn). Điểm 20 video trùng Tuần 6.
  audio flac (20 file eval, seed 1): mean=0.194s p50=0.199s p95=0.248s max=0.253s
  audio từ video (20 video trên ghép 20 flac đó, AAC): mean=0.314s p50=0.312s p95=0.368s max=0.374s
Lưu ý ổ đĩa: ổ ảo WSL (ext4.vhdx) nằm trên ổ C, 50,3 GB; ổ C còn khoảng 39 GB.
Tiếp theo: ghi mục đóng băng thí nghiệm vào protocol (cuối Tuần 8); Tuần 9: unit test, README, LICENSE, chạy lại repo trên máy sạch.
