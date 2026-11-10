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

# 2026-11-06: Giai đoạn 0 của kế hoạch nâng cấp v2.
Repo public (bản sạch, không lịch sử cũ, không ảnh mặt từ dataset): https://github.com/Vawn169/deepfake-video-audio-detection
Unit test (tests/, commit f93b001): 17 test. test_splits (không cần dữ liệu): danh tính FF++ train/val/test không giao nhau (720/140/140), manifest c23 5.000 dòng khớp split, c40 700 dòng chỉ test, Celeb-DF 200 (100/100, không trùng). test_api: không mặt, không tiếng, 415, 422, chuyển 44,1 kHz. test_regression: real_862 0,072, real_186 0,0303, FaceSwap_280_249 0,9999 (sai số 0,005); 3 logit audio (sai số 0,01).
requirements.txt Tuần 1 thiếu scikit-learn, transformers, soundfile, fastapi, uvicorn, python-multipart, safetensors, huggingface_hub và địa chỉ tải PyTorch cu128; đã tạo lại từ venv đang chạy, không đổi phiên bản nào (commit 374ce47). pytest, httpx ở requirements-dev.txt.
Kiểm tra máy sạch: clone repo public (b4f7f75) vào /mnt/d/clean_check, venv Python 3.14 mới, pip install -r requirements.txt -r requirements-dev.txt thành công sau 1.971 s; không có mô hình: 4 passed, 13 skipped (đúng thiết kế); nối mô hình và dữ liệu bằng symlink: 17 passed. Docker không build lại trong bước này (đã kiểm ở Tuần 8).
Quyền dữ liệu: Deepfake-Eval-2024 gửi form Hugging Face 2026-11-06, chờ duyệt tay. AV-Deepfake1M gửi EULA đã ký qua mail 2026-11-06; Deepfakes1m@gmail.com không tồn tại (lỗi 550), deepfakes1mchallenge@gmail.com không báo lỗi.
In-the-Wild: tải release_in_the_wild.zip 8.161.489.023 byte về /mnt/d/in_the_wild, zip kiểm tra CRC đạt, giải nén 31.779 wav; meta.csv: 19.963 bona-fide, 11.816 spoof (khớp README); 54 tên người nói trong meta.csv (bài báo ghi 58, chưa rõ lý do). Chưa chạy mô hình nào trên In-the-Wild.
Protocol: mục "mở lại thí nghiệm có kiểm soát" commit e88e267 trước khi giải nén dữ liệu mới.
Tiếp theo: Giai đoạn 1a audio trên In-the-Wild (chờ người làm đồng ý chạy), Deepfake-Eval-2024 khi được duyệt.

# 2026-11-07: Giai đoạn 1a audio trên In-the-Wild (script src/audio_eval_itw.py commit 9617eeb trước khi chạy; mỗi checkpoint một lần).
31.779 file (16 kHz mono, 19.963 bona-fide / 11.816 spoof), không loại file nào. Trích đặc trưng 572 s trên GPU.
EER 30,02% / 30,24% / 29,82% (s42/43/44), trung bình 30,03% ± 0,21%; AUC 0,7695 ± 0,0047. (ASVspoof 2019 LA eval: EER 1,57%.)
Ở ngưỡng chọn trên ASVspoof dev: báo nhầm bona-fide 54,38% / 65,35% / 60,92%, bỏ sót spoof 12,09% / 7,58% / 9,20%. Khớp với quan sát 05/11: giọng thật ngoài đời phần lớn bị chấm spoof.
Kết quả: reports/audio_itw_results.csv, reports/audio_itw_run.log, eval_ledger.csv. Đây là mốc để so với audio v2 (Giai đoạn 1b).

# 2026-11-08: Giai đoạn 1b audio v2 và Giai đoạn 2 video V-A (người làm ủy quyền chạy không duyệt từng mục; protocol commit ddd2e18 trước khi viết code).
Audio v2 (src/audio_v2.py, commit fcd5e4b trước khi chạy; đặc trưng ở /mnt/d/audio_feats_v2; RIRS_NOISES ở /mnt/d/rirs_noises):
  A cắt im lặng: tầng 4; In-the-Wild EER 21,83% ± 1,58%, AUC 0,8579 ± 0,0173; ASVspoof eval EER 8,89% ± 1,02%.
  B cắt im lặng + augmentation: tầng 4; In-the-Wild EER 22,56% ± 0,50%, AUC 0,8620 ± 0,0053; ASVspoof eval EER 10,93% ± 0,27%.
  Bản cũ: In-the-Wild 30,03%, ASVspoof eval 1,57%. Cắt im lặng là phần lớn cải thiện; augmentation không giảm thêm EER nhưng ổn định hơn và cân bằng lỗi hơn.
Video V-A (src/train_v2a.py, commit f3b4a3d trước khi train): điểm chọn Val 0,9767/0,9745/0,9735. Tập cũ (đã xem trước, chỉ báo cáo kèm): c23 0,9788 ± 0,0009; c40 0,8832 ± 0,0075; Celeb-DF 0,8422 ± 0,0154; CRF30 0,7929 ± 0,0263. Tập test chính (Deepfake-Eval-2024) chờ quyền.
src/eval_de24.py viết và chạy thử (không ghi ledger) trước khi có dữ liệu. Báo cáo Word cũ đã sửa số (a9956c3); bản nháp mới docs/BAO_CAO_v2_draft.md.
Test: 24 passed (thêm tests/test_v2.py).
Chưa đổi API: checkpoint audio cho API chỉ đổi sau khi chọn theo validation và ghi vào protocol (Giai đoạn 4).

# 2026-11-09 → 10: Ba việc sau khi bỏ DFDC (protocol 2b25e74, commit trước khi viết code).
1. API audio v2: checkpoint audio_v2b_s43 (chọn theo validation), cắt im lặng, đọc tối đa 60 s rồi cắt 4 s (commit 2725488). Parity: 0,0321 so với đặc trưng float16 (vượt 0,02), 0,0000 khi tính float32; ghi lệch vào protocol. Docker build lại (02bab22b48c6, 3,84 GB). Latency Docker audio v2 (cùng 20 file, 10/11): flac mean 0,151 s p95 0,226 s; từ video mean 0,275 s p95 0,338 s; RAM 584 MiB. Parity video đo lại 0,0044. Test 24 passed.
3. Verdict ba mức (high_suspicion / no_sign_detected / inconclusive), ngưỡng báo nhầm 5% trên validation: video 0,5961, audio logit 7,6671.
2. Bộ test thực tế (reports/eval_realworld/):
  R1 149 clip Commons thật (r1_114 file gốc lỗi CRC). Video API aug_s44 báo nhầm 10,8% [6,3–18,0], 111/149 đủ mặt. Audio API v2b_s43 42,4% [34,6–50,5] (ngưỡng 5%: 45,8%); bản cũ 66,2% trung bình.
  R2 (R1 nén cuộc gọi): video API 12,6% [7,7–20,1]; audio API 55,2% [47,1–63,2]; r1_037 không chấm ở checkpoint cắt im lặng (lỗi giải mã Opus).
  R3 Celeb-DF 200 cuộc gọi: AUC baseline 0,7816, aug 0,8319, V-A 0,8347; API ngưỡng 5%: báo nhầm 15,2%, phát hiện 57,0%.
  R4 5.000 In-the-Wild Opus: EER cũ 33,51%, v2a 25,95%, v2b 25,26%; API ngưỡng 5%: báo nhầm 42,0%, phát hiện 86,4%.
Tiếp theo: đồng bộ repo public, cập nhật slide/báo cáo Word, chờ Deepfake-Eval-2024 / DeepSpeak / AV-Deepfake1M.
