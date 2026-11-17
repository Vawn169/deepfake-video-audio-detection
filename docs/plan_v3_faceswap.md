# Kế hoạch V-B: train và test video trên deepfake tạo bằng công cụ đổi mặt đời mới (đã duyệt 10/11/2026)

Ngày soạn: 10/11/2026. Chưa tải, cài hay chạy gì. Khi được duyệt, phần quy tắc (mục 3–6) sẽ được chép vào docs/protocol.md và commit **trước** khi tạo dữ liệu.

## 1. Mục tiêu
- Bắt được video đổi mặt tạo bằng công cụ phổ biến hiện nay, kể cả khi đã nén như cuộc gọi.
- Mục tiêu đặt trước: phát hiện **≥ 80% video giả** ở ngưỡng báo nhầm 5% (ngưỡng chọn trên validation) trên tập test cùng loại công cụ (Test-A).
- Báo cáo trung thực kết quả trên công cụ chưa thấy lúc train (Test-B), dù đạt hay không.

## 2. Bốn công cụ và vai trò

| Công cụ | Mô hình | Vai trò | Giấy phép (đã đọc 10/11) |
|---|---|---|---|
| InsightFace inswapper_128 | inswapper_128 | **Train** (chạy qua FaceFusion) | Code MIT; mô hình: "contact contact@insightface.ai for licensing". Chưa rõ; người làm chịu trách nhiệm, không liên hệ InsightFace |
| SimSwap | simswap_256 | **Train** (chạy qua FaceFusion) | CC BY-NC 4.0, "academic and non-commercial use only" |
| FaceFusion | ghost_1_256 (Apache-2.0) và hyperswap_1a_256 (mặc định, "ResearchRAIL") | **Chỉ test** (Test-B: mô hình chưa thấy) | Code "OpenRAIL-AS": file LICENSE chỉ có tên, không tìm được toàn văn |
| roop | inswapper_128, pipeline riêng của roop | **Chỉ test** (Test-C: cùng mô hình, pipeline và cách ghép khác) | AGPL-3.0; repo đã lưu trữ 05/2026 |

Danh sách 14 mô hình đổi mặt đã kiểm tra trong code FaceFusion (10/11). Thêm Test-A-enh: inswapper + làm nét gfpgan_1.4 (Apache-2.0), chỉ test. FaceFusion cũng có sẵn inswapper và simswap, nên chỉ cần một môi trường cho cả hai mô hình train.

## 3. Dữ liệu nguồn và cách ghép cặp
- Chỉ dùng 1.000 video thật FF++ c23 (đã có quyền, điều khoản cho phép nghiên cứu phi thương mại và giáo dục).
- Ghép cặp theo **đúng danh sách cặp chính thức của FF++** (configs/splits: train 360, val 70, test 70 cặp; mỗi cặp làm cả hai chiều), nên danh tính train/val/test không giao nhau như cũ.
- Mỗi mô hình train tạo: train 720, val 140, test 140 video. Mỗi công cụ chỉ-test tạo: test 140 video.
- Mã hóa đầu ra H.264 CRF 23 (gần với c23). Không dùng bộ làm nét mặt (face enhancer) lúc train; một biến thể test có làm nét để đo riêng (chỉ khi giấy phép mô hình làm nét rõ ràng).
- Tiền xử lý giống hệt hiện tại: MediaPipe, 16 frame, 224, margin 0,3.

## 4. Huấn luyện (V-B)
- Như V-A (augmentation nén + SocialDegrade, 10 epoch, 3 seed 42/43/44), train set = FF++ c23 train (4 phương pháp cũ + thật) **cộng** fake inswapper và simswap của cặp train.
- Chọn checkpoint theo Val: trung bình AUC của FF++ Val và Val mới (inswapper + simswap), không nhìn tập test.
- Ngưỡng báo nhầm 5% chọn trên video thật của Val.

## 5. Đánh giá (mỗi checkpoint một lần, ghi eval_ledger.csv)
| Tập | Nội dung | Báo cáo |
|---|---|---|
| Test-A | inswapper + simswap trên 140 video test (danh tính chưa thấy) + 140 video thật test | AUC, phát hiện @5% |
| Test-B | Mô hình FaceFusion chưa dùng lúc train | AUC, phát hiện @5% |
| Test-C | roop | AUC, phát hiện @5% |
| Test-A/B/C nén cuộc gọi | Cùng cách nén R2 | như trên |
| R1, R2 | video thật Commons | tỷ lệ báo nhầm (không được tăng quá nhiều so với V-A) |
| FF++ test cũ, c40, Celeb-DF, CRF30, R3 | đã xem trước | chỉ báo kèm |

So sánh với baseline, aug, V-A trên cùng các tập mới (các checkpoint cũ cũng chỉ chạy một lần mỗi tập).
Hạn chế phải ghi: video thật trong Test-A/B/C là FF++ test đã xem trước; danh tính FF++ là người trên YouTube, không phải tình huống cuộc gọi thật.

## 6. Quy tắc an toàn và đạo đức
- Chỉ đổi mặt giữa các người trong FF++. Không dùng mặt người ngoài, người nổi tiếng, hay người quen chưa đồng ý bằng văn bản.
- Video và ảnh tạo ra chỉ lưu trên máy (/mnt/d), không công bố, không đưa lên GitHub, không chia sẻ.
- Không tắt bộ lọc nội dung của công cụ.
- Ghi vào báo cáo: công cụ, giấy phép, việc người làm chịu trách nhiệm dùng inswapper khi giấy phép chưa rõ. Nếu InsightFace từ chối: gỡ phần inswapper hoặc ghi là hạn chế.

## 7. Các bước và thời gian dự kiến (ước lượng, sẽ đo ở bước thử)
| Bước | Việc | Dự kiến |
|---|---|---|
| 0 | Đọc toàn văn OpenRAIL-AS, danh sách và giấy phép mô hình FaceFusion, weights SimSwap; gửi mail InsightFace; ghi protocol, commit | 11–12/11 |
| 1 | Cài FaceFusion và roop trong môi trường riêng (không đụng .venv hiện tại); thử 10 video: đo tốc độ, xem chất lượng | 12–14/11 |
| 2 | Tạo dữ liệu (khoảng 4.000 video), chạy nền | 14–20/11 |
| 3 | Tiền xử lý, train V-B 3 seed | 20–24/11 |
| 4 | Đánh giá, cập nhật báo cáo | 25–28/11 |

Rủi ro: GPU RTX 4060 8 GB (tốc độ tạo chưa biết, đo ở bước 1); công cụ có thể cần phiên bản Python khác (dùng môi trường riêng); ổ D còn 150 GB (dữ liệu tạo ra ước khoảng vài chục GB, đo ở bước 1).

## 8. Không làm trong kế hoạch này
Âm thanh (ASVspoof 5) là kế hoạch riêng. Video sinh hoàn toàn bằng mô hình khuếch tán: ngoài phạm vi.

## 9. Dự phòng: Self-Blended Images (ghi 11/11/2026, trước khi có kết quả V-B; CHƯA kích hoạt)
Nguồn: Shiohara & Yamasaki, "Detecting Deepfakes with Self-Blended Images", CVPR 2022. Ý tưởng: tạo ảnh giả từ chính ảnh thật bằng cách ghép mặt một người với bản biến đổi nhẹ của chính họ, để mô hình học dấu vết ghép mặt chung thay vì dấu vết riêng của từng công cụ. Không cần tạo thêm deepfake, chỉ dùng video thật FF++ train.

Điều kiện kích hoạt (chốt trước khi xem kết quả V-B):
- V-B (trung bình 3 seed) phát hiện dưới 80% video giả Test-A ở ngưỡng báo nhầm 5%, hoặc
- dưới 80% trên Test-B (ghost + hyperswap gộp).
Kể cả khi đạt điều kiện, chỉ làm khi người làm duyệt và còn thời gian trước khi đóng băng (khoảng 13/12).

Thiết kế (cố định từ bây giờ, không đổi theo kết quả V-B):
- Mô hình V-SBI = V-B (cùng dữ liệu, kiến trúc EfficientNet-B0, 16 frame, augmentation và siêu tham số V-B), thêm SBI tạo ngay khi train: mỗi ảnh mặt thật trong batch, với xác suất 0,5, được biến thành ảnh SBI nhãn giả.
- Cách tạo SBI (theo bài báo, tự viết lại; không dùng code gốc khi chưa đọc giấy phép): từ một ảnh mặt thật, tạo bản nguồn và bản đích bằng biến đổi màu, độ sáng, độ nét, co giãn và dịch nhẹ; mặt nạ là bao lồi các điểm mốc khuôn mặt (MediaPipe face_landmarker đã có trong models/), làm biến dạng và làm mờ mép ngẫu nhiên; trộn với tỉ lệ ngẫu nhiên. Các khoảng tham số lấy theo bài báo, ghi vào protocol trước khi train.
- Chọn checkpoint và ngưỡng như V-B. Train 3 seed 42/43/44.
- Đánh giá đúng các tập của mục 5, mỗi checkpoint một lần, ghi eval_ledger.csv.

Hạn chế phải ghi nếu dùng: các tập Test-A/B/C đã được xem qua kết quả V-B khi V-SBI chạy, nên V-SBI là thí nghiệm sau khi đã xem test (như V-A). Thiết kế được chốt trước ở mục này để giảm nguy cơ chỉnh theo kết quả, nhưng không loại bỏ hẳn.

Thời gian ước (chưa đo): viết và kiểm tra code SBI khoảng 2 ngày; train 3 seed chậm hơn V-B do tạo ảnh khi train.
