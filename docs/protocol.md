# Giao thức đánh giá (đóng băng ở tuần 1)

| Hạng mục | Quyết định cố định |
| --- | --- |
| Huấn luyện | FF++ c23: video thật và các phương pháp giả, theo split chính thức của bộ dữ liệu; lấy mẫu cân bằng thật/giả trong mỗi batch |
| Validation | Val split của FF++; dùng chọn epoch và ngưỡng quyết định |
| Test in-domain | Test split FF++ c23 |
| Cross-compression | FF++ c40 (test split) và bản nén lại H.264 CRF 30 của tập con Celeb-DF |
| Cross-dataset | Celeb-DF v2: tập con cố định 200 video (100 thật, 100 giả) lấy từ danh sách test chính thức bằng seed cố định; không bao giờ dùng để huấn luyện |
| Frame và video | 16 frame đều nhau mỗi video; điểm video = trung bình xác suất giả của các frame |
| Chỉ số | AUC cấp video là chính; accuracy và F1 tại ngưỡng cố định chọn từ validation; khoảng tin cậy bootstrap theo video |
| Audio | ASVspoof 2019 LA theo train/dev/eval chính thức; báo EER và AUC |
| Seed | Đo thời gian một lần huấn luyện ở tuần 3: đủ nhanh thì chạy 3 seed và báo trung bình ± độ lệch, nếu không thì 1 seed và nói rõ trong báo cáo |

## Quy tắc

- Video gốc của các tập test (FF++ test, tập con Celeb-DF) được giữ đến hết tuần 8 và bản nén được sinh ở tuần 2. Chỉ video gốc của tập train mới xóa sau khi xử lý xong.
- Không đổi ma trận sau khi đã thấy kết quả test.
- Kết quả cross-dataset thấp hơn in-domain là hiện tượng đã biết, không phải lỗi. Accuracy ở ngưỡng cố định sẽ thấp hơn AUC khi đổi miền, nên AUC là chỉ số chính.

## Quy tắc xử lý dữ liệu

- Celeb-DF v2 chỉ dùng làm tập test, cho nghiên cứu phi thương mại.
- Không đưa video, ảnh mặt đã cắt hay dữ liệu phái sinh từ Celeb-DF lên GitHub hoặc nơi công khai; thư mục data/ nằm trong .gitignore.
- Cần trích dẫn bài báo Celeb-DF (CVPR 2020) trong báo cáo.

- Tập con Celeb-DF: chọn bằng src/select_subset.py, seed 42, 100 thật (63 Celeb-real, 37 YouTube-real) và 100 giả (Celeb-synthesis) từ List_of_testing_videos.txt (518 video: 178 thật, 340 giả); danh sách ở configs/celebdf_test_subset.csv.

## Sửa đổi (ngày: 2026-10-17, trước khi huấn luyện baseline)
- Cân bằng lớp thật/giả bằng `pos_weight = N_real / N_fake` trong hàm mất mát (`BCEWithLogitsLoss`) kết hợp `shuffle=True` thay vì lấy mẫu cân bằng cứng trong từng batch. Lý do: tận dụng toàn bộ các khung hình của lớp đa số trong mỗi epoch mà không cần lặp lại mẫu lớp thiểu số, vẫn giữ nguyên mục đích cân bằng gradient. Chưa có kết quả test nào được xem.

## Sửa đổi: quy tắc báo cáo và chọn seed (2026-10-17, trước khi chạy test)
- Huấn luyện 3 seed (42, 43, 44), chọn checkpoint mỗi seed theo Val Video AUC.
- Báo cáo kết quả test là trung bình ± độ lệch chuẩn mẫu của cả 3 seed.
- Seed 44 (Val AUC cao nhất, 0,9901) là checkpoint chính cho Grad-CAM và API; chọn theo Val, không theo test.
- Ngưỡng phân loại là ngưỡng EER chọn trên FF++ Val của từng seed; AUC là chỉ số chính.
- Mỗi tập test được đánh giá một lần cho mỗi checkpoint; evaluate.py ghi nhật ký tại reports/eval_ledger.csv.

## Sửa đổi: augmentation nén (2026-10-19, trước khi huấn luyện, sau khi đã đo baseline)
- Baseline (checkpoints_s42/43/44) đã được đánh giá trên FF++ test, c40, Celeb-DF 200, CRF30; không đánh giá lại baseline.
- Một cấu hình duy nhất, không thử biến thể: RandomDegrade p=0,7 (giảm/phóng độ phân giải 0,5-1,0 với xác suất 0,5; JPEG chất lượng 20-90; nhiễu Gauss σ 1-6 với xác suất 0,5), chỉ áp dụng khi train. Mọi siêu tham số khác giữ như baseline (10 epoch, batch 64, lr 2e-4, pos_weight, seed 42/43/44).
- Chọn checkpoint theo trung bình AUC của FF++ Val sạch và FF++ Val làm nhiễu cố định (nhiễu tổng hợp, seed theo đường dẫn ảnh). Không dùng Celeb-DF, c40, CRF30 để chọn bất cứ thứ gì.
- Checkpoint mới ở checkpoints_aug_s42/43/44; mỗi tập test đánh giá một lần cho mỗi checkpoint mới. So sánh có/không augmentation theo trung bình ± SD ba seed. Kết quả báo cáo dù tốt hay xấu; không điều chỉnh augmentation sau khi xem điểm test.
- Mã: src/augment.py, src/train_aug.py.

## Sửa đổi: checkpoint chính cho Grad-CAM và API (2026-10-20, sau khi có kết quả aug)
- Checkpoint chính: checkpoints_aug_s44/best_effnet_b0.pt (score chọn theo FF++ Val sạch + Val nhiễu cao nhất trong 3 seed aug: 0,9752).
- Lý do chọn bản augmentation: trung bình 3 seed bền hơn với nén ở c40, Celeb-DF, CRF30 (xem reports/eval). Quyết định dựa trên Val và kết quả trung bình ba seed, không dựa vào điểm riêng của seed 44 trên các tập test.
- Báo cáo kết quả vẫn là trung bình ± SD của 3 seed; checkpoint chính chỉ dùng cho demo, Grad-CAM, API.

## Sửa đổi: giao thức nhánh audio, Mức 2 (2026-10-31, trước khi trích đặc trưng và huấn luyện)
- Dữ liệu: ASVspoof 2019 LA (Edinburgh DataShare, giấy phép ODC-By), chia train/dev/eval theo protocol chính thức ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt. Chỉ dùng file flac có trong protocol (bỏ qua 142 file thừa ở dev và 696 file thừa ở eval). Số mẫu: train 25.380 (2.580 bonafide, 22.800 spoof), dev 24.844 (2.548, 22.296), eval 71.237 (7.355, 63.882).
- Tiền xử lý: đọc flac 16 kHz, cắt tối đa 4 giây từ đầu file, không pad, mỗi file xử lý riêng.
- Đặc trưng: wav2vec2-base (facebook/wav2vec2-base) đóng băng, lấy trung bình theo thời gian của cả 13 tầng ẩn, lưu một lần ở dạng float16.
- Chọn tầng: logistic regression (class_weight cân bằng) trên từng tầng, chọn tầng có EER thấp nhất trên dev.
- Bộ phân loại: MLP nhỏ (768 -> 256 -> 1), BCEWithLogitsLoss với pos_weight = N_bonafide / N_spoof (bonafide là lớp thiểu số), 3 seed (42, 43, 44), chọn checkpoint mỗi seed theo EER trên dev. Một cấu hình duy nhất, không thử biến thể.
- Ngưỡng: ngưỡng EER chọn trên dev, giữ nguyên khi đánh giá eval.
- Chỉ số: EER là chính, kèm AUC. Báo trung bình ± độ lệch chuẩn mẫu của 3 seed. Không báo t-DCF.
- Tập eval đánh giá đúng một lần cho mỗi checkpoint, ghi vào reports/eval_ledger.csv. Không điều chỉnh gì sau khi xem điểm eval; kết quả báo cáo dù tốt hay xấu.
- Chưa có kết quả audio nào được xem tại thời điểm ghi mục này.

## Sửa đổi: siêu tham số MLP audio (ghi bổ sung 2026-11-01, SAU khi đã chạy train và xem kết quả dev; chưa xem eval)
- 20 epoch, batch 256, Adam lr 1e-3, không weight decay, không dropout. Chuẩn hóa đặc trưng theo trung bình và độ lệch chuẩn của train ở tầng đã chọn. Chọn tầng và epoch theo EER trên dev.
- Các giá trị này được cố định trong src/audio_train.py trước khi chạy và không đổi sau khi xem dev. Lỗi quy trình: mục này lẽ ra phải commit trước khi chạy huấn luyện.
- Kết quả dev (reports/audio_dev_results.csv): tầng 2, EER 0,11% đến 0,15%.

## Ghi nhận: kiểm tra đường tắt audio trên dev (2026-11-01, trước khi xem eval)
- Mô hình chỉ dùng độ dài file: AUC 0,525. Mô hình chỉ dùng độ dài đoạn im lặng đầu file: AUC 0,695 (4.000 file dev ngẫu nhiên, seed 0). Ngưỡng cảnh báo 0,70 do người làm tự đặt sau khi đã làm kiểm tra, nên kết quả này được ghi như một hạn chế, không coi là đạt.
- Không đổi cấu hình nào sau ghi nhận này. Báo cáo EER eval kèm cảnh báo rằng một phần có thể đến từ đường tắt im lặng đầu file.

## Sửa đổi: endpoint audio trong API (2026-11-02, trước khi viết code; không huấn luyện thêm)
- Checkpoint audio cho API: checkpoints_audio_s43/best_mlp.pt (dev EER thấp nhất trong 3 seed: 0,113%; chọn theo dev, không theo eval). Kết quả báo cáo vẫn là trung bình ± SD của 3 seed.
- Ngưỡng: ngưỡng EER trên dev lưu trong checkpoint (s43: 2,9935, tính trên logit, không phải xác suất). Nhãn spoof khi logit >= ngưỡng; điểm trả về là sigmoid(logit) = xác suất spoof, chỉ để tham khảo.
- Endpoint riêng (POST /predict_audio), trả điểm riêng của nhánh audio. /predict (video) giữ nguyên. Không gộp điểm hai nhánh (không làm fusion, Mức 3).
- Đầu vào: file .wav/.flac, hoặc video. Video được tách tiếng bằng ffmpeg thành mono 16 kHz. File âm thanh không phải 16 kHz hoặc nhiều kênh cũng được ffmpeg chuyển về mono 16 kHz. Video không có track âm thanh: trả status no_audio, score null, không kết luận.
- Tiền xử lý giống lúc đánh giá: tối đa 4 giây đầu, không pad, chuẩn hóa từng file về trung bình 0 và phương sai 1, wav2vec2-base đóng băng, tầng 2 trung bình theo thời gian, chuẩn hóa bằng mean/std của train lưu trong checkpoint, MLP.
- Trọng số wav2vec2-base được chép vào image lúc build từ cache đã dùng khi trích đặc trưng; không tải từ mạng khi chạy. Trước khi đóng gói, kiểm tra hai snapshot trong cache có trọng số giống nhau.
- Kiểm tra parity: điểm API so với điểm offline của audio_eval.py trên 10 file eval, sai lệch phải rất nhỏ. Đo lại latency CPU trong Docker (cả audio và video) theo cách đo video ở Tuần 6.
- Hạn chế: mô hình chỉ được đánh giá trên ASVspoof 2019 LA (giọng đọc tiếng Anh, 16 kHz, flac). Âm thanh tách từ video (nén AAC, tốc độ lấy mẫu khác, nhạc, tiếng ồn, 4 giây đầu có thể là im lặng hoặc nhạc mở đầu) nằm ngoài miền đánh giá; điểm trên video thực tế chưa được kiểm chứng và chỉ mang tính tham khảo. Cộng thêm hạn chế đường tắt im lặng đầu file đã ghi ở trên.

## Đóng băng thí nghiệm (cuối Tuần 8, 2026-11-03)
- Từ thời điểm này không huấn luyện thêm, không tạo checkpoint mới, không đánh giá lại bất kỳ tập test/eval nào (FF++ c23/c40, Celeb-DF 200, CRF30, ASVspoof 2019 LA eval). reports/eval_ledger.csv không được thêm dòng mới.
- Checkpoint cuối cùng: video checkpoints_s42/43/44 (baseline) và checkpoints_aug_s42/43/44 (augmentation); audio checkpoints_audio_s42/43/44. API dùng checkpoints_aug_s44 (video) và checkpoints_audio_s43 (audio).
- Được phép sau mốc này: unit test, README, LICENSE, sửa lỗi code, dọn repo, báo cáo, slide, demo. Mọi thay đổi trong src/ có ảnh hưởng tới API phải chạy lại scripts/api_parity.py và scripts/api_parity_audio.py; điểm phải giữ nguyên trong sai số đã ghi (video max |d| 0,0044; audio max |d logit| 0,0049).
- Đo latency lại (nếu cần cho demo) không tính là thí nghiệm mới, nhưng phải ghi rõ là lần đo khác.
- Các điểm lệch so với kế hoạch ban đầu, ghi trung thực trong báo cáo:
  - Cân bằng lớp bằng pos_weight thay vì lấy mẫu cân bằng trong batch (sửa đổi 2026-10-17, trước khi huấn luyện).
  - Augmentation thêm sau khi đã xem kết quả baseline (ghi trước khi huấn luyện lại, 2026-10-19).
  - Siêu tham số MLP audio commit sau khi chạy dev, trước khi xem eval (2026-11-01).
  - Video không tìm được mặt bị loại khỏi đánh giá: FF++ c40 test 699/700 (thiếu NeuralTextures_376_381), FF++ Val 694/700 (ngưỡng chọn trên 694 video).
  - Grad-CAM theo vùng: 48 video (kế hoạch 50).
  - MLflow và DVC không dùng; thay bằng git commit, docs/protocol.md, reports/eval_ledger.csv và các file trong reports/.
  - Mức 3 (fusion FakeAVCeleb) không làm: chưa có quyền dữ liệu.

## Sửa đổi: mở lại thí nghiệm có kiểm soát (2026-11-06, trước khi tải dữ liệu mới về giải nén và trước khi chạy bất kỳ mô hình nào trên dữ liệu mới)
- Lý do: mốc "Tuần 8" ở mục đóng băng (2026-11-03) tính theo tiến độ công việc; theo lịch thật dự án mới khoảng tuần 3–4 trên 10, còn khoảng 7 tuần. Thử trên dữ liệu ngoài đời (2026-11-05) cho thấy nhánh audio chấm mọi giọng thật ngoài đời là spoof và nhánh video báo nhầm clip mạng xã hội. Kế hoạch nâng cấp v2 dùng thời gian còn lại để đo và sửa các lỗi này.
- Mọi kết quả, checkpoint và mục đóng băng cũ giữ nguyên, không đánh giá lại các tập test cũ với checkpoint cũ. Mô hình mới lưu ở thư mục mới (checkpoints_audio_v2_*, checkpoints_video_v2_*).
- Tập test mới, chỉ dùng để test, không dùng để train, chọn mô hình, chọn tầng hay chọn ngưỡng:
  - In-the-Wild (Müller và cộng sự, 2022; Hugging Face mueller91/In-The-Wild, CC BY-SA 4.0): toàn bộ bộ dữ liệu, nhãn theo file meta đi kèm. Số câu phải khớp README (19.963 thật, 11.816 giả); nếu lệch thì ghi lại số thực tế trước khi chạy mô hình.
  - Deepfake-Eval-2024 (Hugging Face nuriachandra/Deepfake-Eval-2024, CC BY-SA 4.0, điều khoản chỉ cho phép dùng để đánh giá): phần video và phần audio, chỉ các mục có nhãn real/fake. Đang chờ duyệt quyền; chỉ dùng sau khi được duyệt.
- Giai đoạn 1a (đo mô hình hiện tại, không train):
  - Video: 6 checkpoint (checkpoints_s42/43/44, checkpoints_aug_s42/43/44) trên Deepfake-Eval-2024 video, mỗi checkpoint một lần, cùng tiền xử lý như cũ (16 frame, MediaPipe, ngưỡng EER từ FF++ Val). Chỉ số chính: AUC cấp video trên các video có ít nhất 4 frame có mặt, kèm tỷ lệ video đủ điều kiện. Chỉ số phụ: AUC trên toàn bộ video, video không đủ mặt nhận điểm 0,5.
  - Audio: 3 checkpoint (checkpoints_audio_s42/43/44) trên In-the-Wild và Deepfake-Eval-2024 audio, mỗi checkpoint một lần, cùng tiền xử lý như cũ (16 kHz mono, tối đa 4 giây đầu, không cắt im lặng). File không phải 16 kHz mono chuyển bằng ffmpeg như API. Chỉ số: EER và AUC; ngưỡng EER từ ASVspoof 2019 dev để báo tỷ lệ bỏ sót và báo nhầm. Không loại clip nào sau khi xem điểm; nếu metadata có ngôn ngữ thì báo thêm riêng tiếng Anh và không phải tiếng Anh.
  - Fusion không train: trên các video Deepfake-Eval-2024 có cả điểm video (đủ mặt) và điểm audio, điểm gộp = trung bình xác suất giả của checkpoint API (aug_s44 và audio_s43), so với nhãn video. Không chọn trọng số.
  - Mỗi lần đánh giá ghi vào reports/eval_ledger.csv; báo trung bình ± SD mẫu của 3 seed. Kết quả báo cáo dù tốt hay xấu.
- Giai đoạn 1b (audio v2) và 2 (video v2): cấu hình A/B/C và V-A/V-B theo kế hoạch v2. Siêu tham số chi tiết, tập validation và tiêu chí chọn checkpoint ghi trong một mục sửa đổi riêng, commit trước khi train.
- Bộ clip ngoài đời (clip đã được đồng ý của người làm và người thân, clip CC trên Wikimedia Commons): danh sách chốt trước, chỉ dùng để minh họa, không dùng để chọn hay tinh chỉnh.
- Không công khai mẫu từ Deepfake-Eval-2024 hay In-the-Wild trong repo, slide hay báo cáo. Dữ liệu lưu trên ổ D.
- Đóng băng lần hai dự kiến 2026-12-13.
