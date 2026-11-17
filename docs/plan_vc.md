# Kế hoạch nâng cấp V-C (17/11/2026) — bản nháp, chưa chạy

Mục tiêu: mô hình video (V-C) và audio (v2c) học thêm deepfake thế hệ mới (diffusion, tạo khẩu hình, sửa đoạn ngắn),
vẫn giữ tập test chưa thấy để đo trung thực. Người làm đã đọc điều khoản LAV-DF / MAVOS-DD và chịu trách nhiệm
về việc dùng phần train của hai bộ này (17/11). Không công bố video, frame, ảnh mặt; dữ liệu thô chỉ ở /mnt/d.

## 1. Số liệu làm căn cứ (đo được, 17/11)
- MAVOS-DD train: 21.370 video, 75,7 GB; phương pháp hình: echomimic 1.324, inswapper 5.154, liveportrait 1.213,
  memo 2.582, "real" 11.097; 6 ngôn ngữ (không có german, hindi). Validation: 3.895 video, 13,9 GB.
  Open-set của tác giả: test có hififace, roop, sonic và german, hindi — không có trong train.
- LAV-DF train: 78.703 video (21.254 thật, 19.271 chỉ giả hình, 19.088 chỉ giả tiếng, 19.090 giả cả hai);
  dev 31.501; đã nằm trong LAV-DF.tar trên ổ D (chỉ cần giải nén). Có fake_periods (thời điểm đoạn giả).
- Ổ D trống 74 GB (giữ tối thiểu 30 GB); ổ WSL trống 902 GB (nơi lưu ảnh mặt đã cắt).
- V-B: 78.172 frame train, 10 epoch, khoảng 150–210 giây/epoch (reports/train_vb_s44.log).
- Cắt mặt khi đánh giá: khoảng 0,8–2 video/giây (MAVOS chậm, AV1M nhanh).

## 2. Dữ liệu train mới (tập con cố định, seed 42, chốt trong protocol trước khi tải)
| Nguồn | Train | Validation (chọn checkpoint) |
|---|---|---|
| MAVOS-DD | khoảng 6.000 video (khoảng 21 GB — ước lượng theo tỷ lệ): cân bằng 4 phương pháp giả hình + video hình thật, cân bằng ngôn ngữ | khoảng 800 video từ split validation |
| LAV-DF | 6.000 video, 1.500 mỗi ô (modify_video × modify_audio) | 800 video từ split dev |
| Cũ | giữ nguyên dữ liệu V-B (FF++ + inswapper/simswap) | FF++ Val + Val mới V-B |

Điểm kỹ thuật quan trọng — đoạn giả ngắn: video LAV-DF giả chỉ có khoảng 0,7 giây bị sửa. Nếu lấy 16 frame đều cả video
và gán nhãn "giả" cho cả 16 frame thì hầu hết frame "giả" thực chất là thật (nhãn nhiễu). Cách xử lý: với video giả
có fake_periods, lấy frame bên trong đoạn giả; video thật lấy đều như cũ. Audio: lấy cửa sổ 4 giây chứa đoạn giả.

## 3. Huấn luyện
- V-C (video): cùng EfficientNet-B0, cùng tiền xử lý (MediaPipe, lề 0,3, 224) và augmentation như V-B, 10 epoch, 3 seed (42/43/44).
  Chọn checkpoint: trung bình AUC trên FF++ Val, Val mới V-B, MAVOS validation, LAV-DF dev (đặt trước).
  Ước lượng (suy luận, chưa đo): số frame train khoảng gấp đôi V-B, nên khoảng 1 giờ mỗi seed.
- Audio v2c: wav2vec2 + MLP như v2b, thêm tiếng MAVOS train (nhãn audio_fake) và LAV-DF train (nhãn modify_audio).
  Chọn epoch và ngưỡng trên ASVspoof dev + MAVOS validation + LAV-DF dev. MLP train nhanh; phần chậm là trích đặc trưng.
- Canh nhiệt như mọi lần (tạm dừng ≥83°C đến ≤70°C; >85°C hai lần thì dừng hẳn).

## 4. Đánh giá (một lần mỗi checkpoint, ghi ledger với tên "V-C" / "audio v2c")
| Tập | Vai trò sau nâng cấp |
|---|---|
| MAVOS-DD test, open-set (hififace, roop, sonic; german, hindi) | Ngoài miền — chỉ số chính |
| AV-Deepfake1M (6.000) | Ngoài miền — chỉ số chính |
| SocialDF (991) | Ngoài miền, ngoài đời thật — chỉ số chính |
| MAVOS-DD test, closed-set; LAV-DF test | Trong miền mới — báo riêng, không gọi là tổng quát hóa |
| Test-A/B/C, FF++ c23/c40, Celeb-DF, CRF30 | Kiểm tra có bị quên kiến thức cũ không |

So sánh với V-B trên cùng các tập (kết quả V-B đã có trong ledger). Ngưỡng: vẫn phân vị 95 FF++ Val để so được với bản cũ;
báo thêm AUC/EER (không phụ thuộc ngưỡng).

## 5. Phân tích bổ sung không cần train (tùy chọn, đặt trước)
Chấm theo đoạn: chia video thành các đoạn ngắn, lấy điểm cao nhất thay vì trung bình 16 frame; áp dụng cho V-B và V-C
trên AV1M / LAV-DF; báo là phân tích bổ sung.

## 6. Thứ tự và thời gian (ước lượng, chưa đo)
| Bước | Việc | Thời gian ước |
|---|---|---|
| 1 | Chốt tập con + ghi protocol V-C, commit (trước khi tải/train) | 1 giờ |
| 2 | Tải tập con MAVOS train + validation; giải nén tập con LAV-DF train/dev | 2–4 giờ (mạng) |
| 3 | Cắt mặt (theo fake_periods) + trích đặc trưng audio | 4–6 giờ |
| 4 | Train V-C 3 seed + audio v2c 3 seed | 4–5 giờ |
| 5 | Đánh giá tất cả tập ở mục 4 | 4–6 giờ |
| 6 | So sánh V-B / V-C, cập nhật báo cáo | nửa ngày |
Tổng: khoảng 2 ngày máy chạy (chủ yếu chạy qua đêm).

## 7. Rủi ro
- Có thể không cải thiện ngoài miền (đặc biệt SocialDF): vẫn báo cáo trung thực.
- Rò rỉ danh tính giữa train/test trong MAVOS-DD, LAV-DF: kiểm tra README/split của tác giả ở bước 1.
- Có thể quên kiến thức cũ: đã có bảng kiểm tra ở mục 4.
- RAM 9,7 GB: dùng nạp trước có giới hạn như eval_ood (sửa 17/11).
- API vẫn dùng V-B cho tới khi V-C được đánh giá xong và người làm quyết định.
