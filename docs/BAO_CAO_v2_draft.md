# Phát hiện Deepfake Video và Âm thanh — bản nháp báo cáo (cập nhật 17/11/2026, bản 2)

Bản nháp này thay cho phần kết quả của `reports/BAO_CAO_tong_hop_GVHD.docx` (bản chụp 28/10). Mỗi con số ghi kèm file nguồn trong repo. Phần đánh dấu **[chờ]** sẽ điền khi có kết quả.

## 1. Bài toán và phạm vi
- Mức 1 (video): EfficientNet-B0 trên ảnh mặt, huấn luyện FF++ c23, đánh giá chéo, Grad-CAM, API FastAPI chạy Docker CPU. Hoàn thành.
- Mức 1b (video, V-B, từ 10/11): train thêm trên video đổi mặt bằng công cụ đời mới (inswapper, simswap) và test trên 5 công cụ, bản gốc và bản nén như cuộc gọi. Hoàn thành; API video dùng V-B từ 16/11.
- Mức 1c (video, V-C, 17/11): train thêm phần train MAVOS-DD và LAV-DF. Hoàn thành; tốt hơn trong miền nhưng kém V-B trên nguồn mới, nên không dùng cho API (mục 4d).
- Mức 2 (audio): đặc trưng wav2vec2-base đóng băng + MLP nhỏ, huấn luyện ASVspoof 2019 LA; bản v2 cắt im lặng và augmentation; bản v2c (17/11) train thêm tiếng MAVOS-DD và LAV-DF. API audio dùng v2c từ 17/11.
- Mức 3 (fusion): bản không huấn luyện (trung bình xác suất hai nhánh, và quy tắc "một trong hai nhánh vượt ngưỡng"). Deepfake-Eval-2024 bị từ chối cấp quyền nên đo trên 4 bộ ngoài miền (mục 4e).
- Kết quả phát hiện chỉ mang tính tham khảo, không thay thế giám định.

## 2. Dữ liệu
| Bộ | Vai trò | Quy mô dùng | Nguồn / giấy phép |
|---|---|---|---|
| FaceForensics++ c23 | Train / Val / Test | 5.000 video theo split chính thức: train 3.600, val 700, test 700 (`configs/manifest_ffpp_c23.csv`) | Đơn xin quyền FF++ |
| FF++ c40 | Test nén mạnh | 700 video test | như trên |
| Celeb-DF v2 | Test chéo | 200 video cố định (100 thật, 100 giả), seed 42 (`configs/celebdf_test_subset.csv`) | Nghiên cứu phi thương mại |
| Celeb-DF CRF30 | Test chéo + nén | 200 video trên, nén lại H.264 CRF 30 | tự tạo |
| ASVspoof 2019 LA | Audio train / dev / eval | 25.380 / 24.844 / 71.237 file | Edinburgh DataShare, ODC-By |
| In-the-Wild | Test audio ngoài đời (mới) | 31.779 file: 19.963 thật, 11.816 giả | Hugging Face, CC BY-SA 4.0 |
| Deepfake-Eval-2024 | (dự kiến) Test video + audio 2024 | Bị từ chối trên Hugging Face; bỏ từ 16/11 | CC BY-SA 4.0 |
| MAVOS-DD (test) | Test ngoài miền | 6.800 video, 8 ngôn ngữ, 7 phương pháp hình (`configs/mavos_test_subset.csv`): 2.000 thật, 2.899 chỉ giả hình, 1.301 giả cả hai, 600 chỉ giả tiếng | CC BY-NC-SA 4.0 |
| MAVOS-DD (train / validation) | Train / chọn checkpoint V-C, v2c | 5.120 / 800 video; bỏ mọi video cùng video YouTube nguồn với tập test (`configs/mavos_train_subset.csv`, `mavos_validation_subset.csv`) | như trên |
| AV-Deepfake1M (val của tác giả) | Test ngoài miền | 6.000 video, 1.500 mỗi loại sửa (`configs/av1m_val_subset.csv`) | Chỉ nghiên cứu phi thương mại |
| LAV-DF (test) | Test | 6.000 video, 1.500 mỗi ô sửa hình × sửa tiếng (`configs/lavdf_test_subset.csv`) | CC BY-NC 4.0 |
| LAV-DF (train / dev) | Train / chọn checkpoint V-C, v2c | 6.000 / 800 video (`configs/lavdf_train_subset.csv`, `lavdf_dev_subset.csv`) | như trên |
| SocialDF | Test ngoài miền, deepfake lan truyền trên Instagram | 991 video tải được (535 giả, 456 thật) trên 1.556 link của tác giả | Chỉ nghiên cứu phi thương mại; không công bố |
| V-B (tự tạo, đổi mặt giữa các người trong FF++) | Train / Val / Test video | 2.487 video (`configs/manifest_vb.csv`): inswapper và simswap mỗi loại train 705, val 137, test 134; test thêm ghost 134, hyperswap 134, inswapper + GFPGAN 134, roop 133 | Mô hình đổi mặt chỉ cho nghiên cứu phi thương mại (inswapper, simswap); video chỉ lưu trên máy, không công bố |
| V-B nén cuộc gọi | Test | mọi video test V-B + 140 video thật FF++ test, nén ≤640x360, 15 fps, H.264 300 kbps (`scripts/make_call_vb.py`) | tự tạo |
| RIRS_NOISES | Tiếng vang, tiếng ồn cho augmentation audio | OpenSLR 28 | Apache 2.0 |

Phát hiện mặt (MediaPipe, 16 frame/video; `reports/preprocess_*.csv`): FF++ c23 có 35/5.000 video không tìm được mặt nào (6 thật, 29 giả), 71 video dưới 4 frame mặt; c40 có 1 video không có mặt (NeuralTextures_376_381, bị loại khỏi đánh giá → 699/700); Celeb-DF và CRF30 không có video nào mất mặt hoàn toàn. Ngưỡng video được chọn trên 694/700 video Val có mặt.

## 3. Giao thức
Toàn bộ quyết định và các lần sửa đổi có ngày nằm trong `docs/protocol.md`; mỗi lần đánh giá tập test ghi ở `reports/eval_ledger.csv`, script tự từ chối đánh giá lại.
Điểm lệch so với kế hoạch, ghi trung thực:
- Cân bằng lớp bằng pos_weight thay vì lấy mẫu cân bằng trong batch (17/10, trước khi train).
- Augmentation nén được thêm sau khi đã thấy kết quả baseline (ghi trước khi train lại, 19/10).
- Siêu tham số MLP audio commit sau khi chạy dev, trước khi xem eval (01/11).
- Grad-CAM theo vùng làm trên 48 video (kế hoạch 50).
- Không dùng MLflow/DVC; thay bằng git, protocol.md, eval_ledger.csv.
- Mở lại thí nghiệm sau mốc đóng băng 03/11 (mục 06/11), vì mốc "tuần" tính theo tiến độ chứ không theo lịch; kết quả cũ giữ nguyên, tập test mới được chốt trước khi giải nén.
- Các bước từ 08/11 (audio v2, V-A, script Deepfake-Eval-2024, báo cáo) do người làm ủy quyền chạy không duyệt từng mục; quy tắc vẫn commit trước khi train.
- V-B: train seed 43, 44 có tạm dừng tiến trình để làm mát GPU (không đổi dữ liệu hay tham số); lần chạy đầu seed 42, 43 bị dừng vì nhiệt, đã chạy lại từ đầu cùng seed (log giữ lại).
- Roop (Test-C) viết lại từ lõi của s0md3v/roop, chạy CUDA, bộ lọc nội dung kiểm tra 1 frame/giây thay vì mọi frame (ghi trong protocol 15/11).
- V-SBI là thí nghiệm sau khi đã xem kết quả test V-B (điều kiện kích hoạt viết trước trong kế hoạch); kết quả chỉ báo kèm.
- Deepfake-Eval-2024 bị từ chối; thay bằng MAVOS-DD, AV-Deepfake1M, LAV-DF, SocialDF (16/11). Tập test chính đặt trước cho V-A không đo được.
- SocialDF: tác giả chỉ công bố link Instagram; tải được 50,6% link giả và 91,6% link thật, nên tập không đại diện đầy đủ cho bài báo.
- V-C và v2c được thiết kế SAU khi thấy kết quả ngoài miền. Vì vậy chỉ các tập V-C không được train (MAVOS-DD phương pháp mới, AV-Deepfake1M, SocialDF) được dùng để nói về tổng quát hóa.
- Khi chạy đánh giá V-C, script bị dừng ở bước đầu vì checkpoint V-C lưu AUC theo khóa khác (`val_aucs`); sửa `evaluate.py` để đọc khóa mới, rồi chạy lại toàn bộ. Chưa có kết quả nào được ghi trước khi sửa.

## 4. Kết quả video (AUC cấp video, trung bình ± SD mẫu 3 seed)
Nguồn: `reports/eval/checkpoints_*/summary.csv`.

| Tập | Baseline | + augmentation nén | V-A (mạng xã hội) | V-B (thêm công cụ mới) | V-SBI | V-C (thêm MAVOS, LAV-DF) |
|---|---|---|---|---|---|---|
| FF++ c23 test | 0,9889 ± 0,0042 | 0,9807 ± 0,0015 | 0,9788 ± 0,0009 | 0,9683 ± 0,0017 | 0,9709 ± 0,0028 | 0,9523 ± 0,0020 |
| FF++ c40 test | 0,7987 ± 0,0163 | 0,8714 ± 0,0113 | 0,8832 ± 0,0075 | 0,8866 ± 0,0070 | 0,8730 ± 0,0041 | 0,8346 ± 0,0128 |
| Celeb-DF 200 | 0,7807 ± 0,0214 | 0,8373 ± 0,0111 | 0,8422 ± 0,0154 | 0,8257 ± 0,0203 | 0,8491 ± 0,0140 | 0,6821 ± 0,0159 |
| Celeb-DF CRF30 | 0,7176 ± 0,0183 | 0,7649 ± 0,0161 | 0,7929 ± 0,0263 | 0,7714 ± 0,0055 | 0,7764 ± 0,0175 | 0,6250 ± 0,0228 |
| Deepfake-Eval-2024 | bị từ chối quyền | | | | | |

Cột V-C: các tập cũ dùng để kiểm tra mô hình có quên không. V-C giảm ở cả 4 tập, rõ nhất ở Celeb-DF (−0,144) và CRF30 (−0,146), lớn hơn nhiều so với độ lệch giữa các seed. V-C từng seed: c23 0,9542/0,9524/0,9502; c40 0,8465/0,8361/0,8211; Celeb-DF 0,7002/0,6754/0,6707; CRF30 0,6491/0,6037/0,6221.

Cột V-B, V-SBI: các tập cũ chỉ báo kèm; tập test chính là mục 4b. V-B từng seed: c23 0,9694/0,9692/0,9663; c40 0,8794/0,8870/0,8933; Celeb-DF 0,8483/0,8195/0,8092; CRF30 0,7777/0,7679/0,7685. So với aug, V-B giảm ở FF++ c23 (−0,012, lớn hơn độ lệch giữa các seed); c40, Celeb-DF, CRF30 chênh lệch trong khoảng dao động giữa các seed.

Cột V-A: các tập này đã được xem trước khi thiết kế V-A, chỉ báo cáo kèm, không dùng để chọn; tập test chính của V-A là Deepfake-Eval-2024. Chênh lệch V-A so với aug (+0,012 c40, +0,005 Celeb-DF, +0,028 CRF30, −0,002 c23) nhỏ so với độ lệch giữa các seed, chưa đủ để kết luận V-A tốt hơn. V-A từng seed: c23 0,9779/0,9797/0,9787; c40 0,8875/0,8876/0,8745; Celeb-DF 0,8366/0,8596/0,8303; CRF30 0,8025/0,8130/0,7632. Chọn checkpoint theo trung bình AUC Val sạch và Val SocialDegrade: 0,9767 / 0,9745 / 0,9735.

Từng seed (baseline / aug): c23 0,9842/0,9902/0,9922 và 0,9822/0,9793/0,9805; c40 0,8001/0,8143/0,7818 và 0,8604/0,8708/0,8830; Celeb-DF 0,7754/0,8042/0,7624 và 0,8459/0,8413/0,8248; CRF30 0,7274/0,7288/0,6965 và 0,7694/0,7783/0,7470. Augmentation tăng AUC ở cả ba seed trên c40, Celeb-DF, CRF30 và giảm ở cả ba seed trên c23.

Ví dụ checkpoint API cũ aug_s44 (ngưỡng Val 0,5951): Celeb-DF AUC 0,8248 [KTC 95% bootstrap 0,7643–0,8786], accuracy 0,72; c40 AUC 0,8830 [0,8571–0,9062], accuracy 0,7496. Accuracy ở ngưỡng cố định thấp hơn AUC khi đổi miền, như đã dự kiến trong giao thức.

Mô hình: 4.008.829 tham số huấn luyện được, checkpoint 16,3 MB.

## 4b. Kết quả V-B: công cụ đổi mặt đời mới (protocol 10/11 và 15/11; chạy một lần 15/11; `reports/eval_vb/`)
Tỷ lệ phát hiện video giả ở ngưỡng báo nhầm 5% (phân vị 95 điểm video thật FF++ Val của từng checkpoint), trung bình 3 seed. Video thật là 140 video FF++ test (chấm được 139 bản gốc, 138 bản nén). Test-A dùng cùng công cụ với train (người khác); Test-B, Test-C, Test-A-enh là công cụ hoặc hậu xử lý không có khi train.

| Tập (bản gốc) | Video giả | Baseline | Aug | V-A | V-B | V-B AUC |
|---|---|---|---|---|---|---|
| Test-A: inswapper + simswap | 268 | 61,2% | 29,2% | 30,1% | **94,0%** (93,2–94,7) | 0,981 |
| Test-B: ghost + hyperswap | 268 | 44,5% | 22,2% | 28,3% | **78,7%** (76,3–80,8) | 0,941 |
| Test-C: roop | 133 | 75,5% | 34,9% | 32,6% | **85,6%** (83,3–88,6) | 0,952 |
| Test-A-enh: inswapper + GFPGAN | 134 | 23,0% | 10,4% | 10,6% | **48,0%** (46,2–50,8) | 0,839 |
| Báo nhầm video thật FF++ test | 140 | 7,7% | 4,6% | 5,3% | 6,2% (5,8–7,2) | — |

Trong ngoặc: seed thấp nhất – cao nhất. Bản nén như cuộc gọi, V-B: Test-A 93,7%, Test-B 79,2%, Test-C 85,5%, Test-A-enh 57,3%; báo nhầm video thật tăng lên 17,2% (16,7–18,1%).

- Mục tiêu đặt trước (V-B, Test-A bản gốc ≥ 80%) đạt: 94,0%, Wilson thấp nhất của 3 seed 89,6%.
- Mô hình cũ (aug, V-A) phát hiện kém hơn baseline trên Test-A, B, C (22–35% so với 44–76%). Nguyên nhân chưa kiểm chứng; một khả năng là augmentation nén làm mô hình bỏ qua dấu vết tần số cao mà các công cụ này để lại.
- Điểm yếu: công cụ chưa thấy khi train (Test-B 78,7%) thấp hơn Test-A; làm nét bằng GFPGAN làm tỷ lệ phát hiện còn khoảng một nửa; nén như cuộc gọi làm báo nhầm tăng gần gấp 3 (6,2% → 17,2%) dù ngưỡng không đổi.
- Roop: 133/140 cặp tạo được (7 cặp không thấy mặt nguồn). Bảng trên các cặp mà cả 6 công cụ đều tạo được (131 cặp) ở `reports/eval_vb/summary.csv` (table = common).

Checkpoint API vb_s44 (bản gốc, Wilson 95%): Test-A 94,7% [91,4–96,8], Test-B 80,8% [75,7–85,1], Test-C 84,8% [77,8–90,0], Test-A-enh 46,2% [37,9–54,7]; báo nhầm 7,2% [4,0–12,7]. Bản nén: báo nhầm 16,7% [11,4–23,8].

## 4c. V-SBI: thêm Self-Blended Images (kết quả âm tính; thí nghiệm sau khi đã xem test)
Kích hoạt vì Test-B dưới 80%. Tự viết lại SBI (`src/sbi.py`; code gốc có giấy phép chỉ phi thương mại và bản sửa đổi thuộc tác giả nên không dùng), tham số lấy từ repo gốc; mỗi ảnh mặt thật train có xác suất 0,5 thành ảnh SBI nhãn giả. Còn lại giống V-B. `reports/eval_vb_V-SBI/`.

| Bản gốc | V-B | V-SBI |
|---|---|---|
| Test-A | 94,0% | 94,9% |
| Test-B | 78,7% | 79,8% (75,9–83,5) |
| Test-C | 85,6% | 84,9% |
| Test-A-enh | 48,0% | 43,7% |
| Báo nhầm FF++ test | 6,2% | 5,5% |
| Báo nhầm bản nén cuộc gọi | 17,2% | 12,6% |
| Báo nhầm R1/R2 (ngưỡng 5%) | 9,3% / 9,6% | 15,3% / 15,3% |

Chênh lệch trên tập giả nằm trong dao động giữa các seed. V-SBI giảm báo nhầm khi nén cuộc gọi nhưng tăng báo nhầm trên clip thật ngoài đời (R1/R2). Không có cải thiện rõ, nên giữ V-B cho API.

## 4d. Bộ test ngoài miền: MAVOS-DD, AV-Deepfake1M, LAV-DF, SocialDF (protocol 16/11 và 17/11; mỗi checkpoint chạy một lần)
Nguồn: `reports/eval_ood/` (các họ cũ, 16–17/11) và `reports/eval_ood_vc/` (V-C, v2c, 17/11). AUC trung bình ± SD 3 seed. Nhánh video dùng nhãn hình, nhánh audio dùng nhãn tiếng; SocialDF chỉ có nhãn chung (nhánh audio chỉ báo kèm, không biết tiếng có bị giả không). Ngưỡng không chọn lại trên bộ mới. Video dưới 4 frame mặt không chấm (tỷ lệ chấm được: MAVOS 93,5%, AV-Deepfake1M và LAV-DF 100%, SocialDF 97,9%).

**Video (AUC)**

| Họ | MAVOS-DD | AV-Deepfake1M | LAV-DF | SocialDF |
|---|---|---|---|---|
| Baseline | 0,467 ± 0,035 | 0,533 ± 0,007 | 0,422 ± 0,018 | 0,589 ± 0,008 |
| Aug | 0,451 ± 0,015 | 0,547 ± 0,004 | 0,425 ± 0,006 | 0,590 ± 0,042 |
| V-A | 0,441 ± 0,014 | 0,544 ± 0,008 | 0,465 ± 0,017 | 0,594 ± 0,030 |
| V-B (API) | 0,512 ± 0,008 | **0,568 ± 0,006** | 0,449 ± 0,008 | **0,651 ± 0,017** |
| V-SBI | 0,562 ± 0,016 | 0,565 ± 0,004 | 0,448 ± 0,010 | 0,636 ± 0,031 |
| V-C | **0,885 ± 0,004** (trong miền một phần) | 0,527 ± 0,002 | **0,748 ± 0,007** (trong miền) | 0,507 ± 0,050 |

**Audio (AUC)**

| Họ | MAVOS-DD | AV-Deepfake1M | LAV-DF | SocialDF (nhãn chung) |
|---|---|---|---|---|
| Audio cũ | 0,388 ± 0,015 | 0,626 ± 0,006 | 0,622 ± 0,006 | 0,531 ± 0,012 |
| v2a | 0,534 ± 0,025 | 0,597 ± 0,006 | 0,575 ± 0,008 | 0,482 ± 0,007 |
| v2b | 0,604 ± 0,007 | 0,603 ± 0,006 | 0,601 ± 0,007 | 0,464 ± 0,008 |
| v2c (API từ 17/11) | **0,969 ± 0,004** (trong miền) | **0,773 ± 0,007** | **0,782 ± 0,001** (trong miền) | 0,419 ± 0,004 |

Tỷ lệ phát hiện / báo nhầm ở ngưỡng (video: báo nhầm 5% FF++ Val; audio: EER ASVspoof dev), trung bình 3 seed:
- V-B: MAVOS 22,3% / 11,7%; AV-Deepfake1M 7,3% / 5,3%; LAV-DF 26,5% / 34,5%; SocialDF 22,0% / 1,7%.
- V-C: MAVOS 74,5% / 12,9%; AV-Deepfake1M 1,7% / 1,9%; LAV-DF 6,2% / 3,0%; SocialDF 9,1% / 3,6%.
- v2b: MAVOS 68,5% / 53,5%; AV-Deepfake1M 56,8% / 42,4%; LAV-DF 59,1% / 44,7%; SocialDF 43,9% / 48,6%.
- v2c: MAVOS 64,0% / 0,3%; AV-Deepfake1M 3,4% / 0,2%; LAV-DF 47,2% / 1,1%; SocialDF 0,2% / 1,5%.

**MAVOS-DD theo phương pháp tạo hình** (AUC, V-B → V-C). Phương pháp có trong train V-C: echomimic 0,211 → 0,989; memo 0,308 → 0,975; inswapper 0,700 → 0,897; liveportrait 0,633 → 0,666. Phương pháp KHÔNG có trong train V-C (open-set của tác giả): sonic 0,306 → 0,971; hififace 0,646 → 0,871; roop 0,847 → 0,793. Theo nhóm open-set (phương pháp cũ/mới × ngôn ngữ cũ/mới): V-B 0,449 / 0,464 / 0,602 / 0,561; V-C 0,881 / 0,913 / 0,872 / 0,909.

**V-C trên bộ V-B** (`reports/eval_vb_V-C/`; phát hiện ở ngưỡng báo nhầm 5%, bản gốc; V-B trong ngoặc): Test-A 87,2% (94,0%); Test-B 70,4% (78,7%); Test-C 78,8% (85,6%); Test-A-enh 62,9% (48,0%); báo nhầm FF++ test 6,0% (6,2%). Bản nén cuộc gọi: báo nhầm 9,4% (V-B 17,2%).

**Audio v2c trên tập cũ** (`reports/audio_v2c_eval_results.csv`): In-the-Wild EER 20,00% ± 0,24% (v2b 22,56%), AUC 0,8777; ASVspoof 2019 eval EER 10,50% ± 0,02% (v2b 10,93%). Chọn tầng 4 như v2b. Validation (ASVspoof dev / dev nhiễu / MAVOS + LAV-DF): 1,85–2,01% / 7,55–8,04% / 3,45–3,82%.

Nhận xét:
- Các mô hình train trên FF++ và ASVspoof gần như không tổng quát sang deepfake thế hệ mới: video AUC 0,42–0,65, audio 0,39–0,63. Trên MAVOS và LAV-DF nhiều họ video có AUC dưới 0,5, tức xếp video thật là giả nhiều hơn video giả. Phương pháp tạo cả khuôn mặt (echomimic, memo, sonic) gần như không bị phát hiện (V-B AUC 0,21–0,31).
- **V-C:** thêm dữ liệu MAVOS và LAV-DF giúp mạnh trong miền, kể cả trên phương pháp MAVOS chưa thấy khi train (sonic 0,971, hififace 0,871). Nhưng các phương pháp này dùng chung video thật, ngôn ngữ và cách xử lý của MAVOS, nên chưa chứng minh được tổng quát hóa. Trên nguồn hoàn toàn mới, V-C kém V-B: AV-Deepfake1M 0,527 so với 0,568; SocialDF 0,507 so với 0,651; Celeb-DF 0,682 so với 0,826; roop trong MAVOS cũng giảm. Kết luận (suy luận, chưa kiểm chứng): V-C học thêm đặc điểm riêng của hai bộ dữ liệu, không học thêm dấu vết chung của deepfake. **Giữ V-B cho API.**
- LAV-DF dev chỉ đạt AUC 0,74–0,75 khi train V-C. Một khả năng (chưa kiểm chứng): đoạn sửa ngắn (trung vị 0,69 giây) nên 16 frame lấy đều cả video phần lớn rơi vào đoạn thật.
- **v2c:** tốt hơn v2b trên AV-Deepfake1M (nguồn không train, +0,17 AUC), In-the-Wild và ASVspoof eval, tức không quên kiến thức cũ. Ngoại lệ: SocialDF, nơi mọi họ audio đều dưới hoặc gần 0,5 (không biết tiếng có giả không). Ngưỡng EER của v2c rất dè dặt ngoài miền: trên AV-Deepfake1M chỉ phát hiện 3,4%. **API audio chuyển sang v2c.**
- AV-Deepfake1M và LAV-DF chỉ sửa đoạn rất ngắn (trung vị 0,36 và 0,69 giây), nên cách chấm trung bình 16 frame / 4 giây đầu là giới hạn thiết kế (đã ghi trước khi chạy).

## 4e. Fusion không huấn luyện (thay cho Deepfake-Eval-2024)
Checkpoint API lúc đo (video vb_s44, audio v2b_s43), nhãn chung (giả nếu hình hoặc tiếng giả), trên video đủ mặt và có tiếng. Chưa đo lại với v2c.

| Bộ | n | AUC trung bình xác suất | AUC chỉ video / chỉ audio | Quy tắc "một nhánh vượt ngưỡng 5%": phát hiện / báo nhầm |
|---|---|---|---|---|
| MAVOS-DD | 6.361 | 0,492 | 0,500 / 0,497 | 66,1% / 62,9% |
| AV-Deepfake1M | 6.000 | 0,609 | 0,603 / 0,574 | 56,9% / 44,5% |
| LAV-DF | 5.998 | 0,495 | 0,377 / 0,586 | 71,0% / 74,8% |
| SocialDF | 961 | 0,588 | 0,665 / 0,462 | 58,8% / 51,7% |

Fusion không tốt hơn nhánh tốt nhất ở bộ nào. Quy tắc "một nhánh vượt ngưỡng" tăng phát hiện nhưng báo nhầm 45–75%, chủ yếu do nhánh audio v2b (báo nhầm riêng 42–58%).

**Đo lại với cấu hình API hiện tại (video vb_s44 + audio v2c_s44, 17/11; `reports/eval_ood_vc/fusion_v2c.csv`)**, tính từ điểm đã lưu, cùng định nghĩa:

| Bộ | n | AUC trung bình xác suất | AUC chỉ video / chỉ audio | Quy tắc "một nhánh vượt ngưỡng 5%": phát hiện / báo nhầm |
|---|---|---|---|---|
| MAVOS-DD | 6.361 | 0,687 | 0,500 / 0,695 | 42,3% / 12,8% |
| AV-Deepfake1M | 6.000 | 0,648 | 0,603 / 0,711 | 10,4% / 4,7% |
| LAV-DF | 5.998 | 0,639 | 0,377 / 0,781 | 55,8% / 55,2% |
| SocialDF | 961 | 0,641 | 0,665 / 0,420 | 28,2% / 3,6% |

Với v2c, fusion tăng AUC ở cả 4 bộ so với bản v2b (+0,04 đến +0,20) và báo nhầm của quy tắc "một nhánh vượt ngưỡng" giảm mạnh (trừ LAV-DF, nơi báo nhầm đến từ nhánh video: 55,2%). Nhưng fusion vẫn không vượt nhánh tốt nhất ở bộ nào (MAVOS, AV-Deepfake1M, LAV-DF: audio riêng tốt hơn; SocialDF: video riêng tốt hơn). Lấy trung bình hai nhánh không trọng số kéo kết quả về giữa khi một nhánh gần mức ngẫu nhiên.

## 5. Kiểm tra chống lỗ hổng (checklist mục 4 của kế hoạch)
| Kiểm tra | Kết quả | Nguồn |
|---|---|---|
| Rò rỉ dữ liệu | Danh tính FF++ train/val/test giao nhau = 0 (720/140/140); kiểm tra tự động trong `tests/test_splits.py` | tests |
| Mặt thất bại | Xem mục 2 | preprocess_*.csv |
| Mô hình siêu dữ liệu | AUC 0,50–0,58 (logistic regression và random forest, 4 tập) | metadata_shortcut_check.csv |
| Overfit 32 video | AUC 1,0 | sanity_check_results.json |
| Xáo nhãn | Val AUC 0,4933 / 0,5378 / 0,6289 (TB 0,5533) | sanity_check_results.json |
| Ngưỡng | Chọn một lần trên Val, giữ nguyên khi test chéo | protocol.md |
| Tiền xử lý đồng nhất | API vs offline: video max \|d\| 0,0048 (10 video, vb_s44; bản aug_s44 0,0044); audio v2c_s44 max \|d logit\| 0,0000 so với offline float32 (0,0566 so với đặc trưng float16 đã lưu) (10 file) | api_parity*.py, api_parity_audio_v2c.log |
| Rò rỉ MAVOS train/test | Bỏ video train/validation cùng video YouTube nguồn với tập test (đọc được mã nguồn ở 57% tên file; video memo không kiểm được) | make_vc_subsets.py |
| Heatmap | Xem mục 7 | region_stats.csv |
| Tái lập | Repo public clone vào máy sạch, cài requirements, 17/17 test đạt (06/11); bộ test hiện tại 24 test | NEXT_STEPS.md |

## 6. Kết quả audio
Nguồn: `reports/audio_*_results.csv`.

| Tập | EER (TB ± SD, 3 seed) | AUC |
|---|---|---|
| ASVspoof 2019 LA dev | 0,15% / 0,11% / 0,12% | — |
| ASVspoof 2019 LA eval | 1,57% ± 0,03% | 0,9987 ± 0,0001 |
| In-the-Wild (ngoài đời) | 30,03% ± 0,21% | 0,7695 ± 0,0047 |
| Deepfake-Eval-2024 audio | bị từ chối quyền | |

Ở ngưỡng chọn trên ASVspoof dev, In-the-Wild bị báo nhầm giọng thật 54,4% / 65,3% / 60,9% và bỏ sót giọng giả 12,1% / 7,6% / 9,2%. Mô hình vẫn xếp hạng được một phần (AUC 0,77) nhưng ngưỡng không còn phù hợp khi đổi miền.

Chẩn đoán (05/11, 40+40 file dev, script chưa đưa vào repo): nén AAC 128/64 kbps và đổi tần số lấy mẫu không làm giọng thật bị đổi nhãn (0%); cắt im lặng đầu file làm 22,5% giọng thật chuyển sang spoof (giọng thật trong ASVspoof có trung vị 0,96 s im lặng đầu, giọng giả 0,35 s). Nguyên nhân chính được suy ra: toàn bộ giọng thật khi train đến từ một nguồn thu âm phòng thu tiếng Anh. Kiểm tra đường tắt trên dev trước khi xem eval: chỉ dùng độ dài im lặng đầu file đạt AUC 0,695.

Audio v2 (mục 08/11 trong protocol), đánh giá trên In-the-Wild và ASVspoof eval:
| Cấu hình | In-the-Wild EER | In-the-Wild AUC | ASVspoof eval EER |
|---|---|---|---|
| Hiện tại (không cắt im lặng) | 30,03% ± 0,21% | 0,7695 ± 0,0047 | 1,57% ± 0,03% |
| A: cắt im lặng | 21,83% ± 1,58% | 0,8579 ± 0,0173 | 8,89% ± 1,02% |
| B: A + tiếng vang, tiếng ồn, băng thoại, codec | 22,56% ± 0,50% | 0,8620 ± 0,0053 | 10,93% ± 0,27% |
| C (v2c, 17/11): B + tiếng MAVOS-DD train, LAV-DF train | 20,00% ± 0,24% | 0,8777 ± 0,0005 | 10,50% ± 0,02% |

Từng seed (42/43/44), In-the-Wild EER: A 21,60% / 23,51% / 20,38%; B 22,70% / 22,98% / 22,01%. ASVspoof eval EER: A 10,06% / 8,31% / 8,30%; B 11,13% / 11,02% / 10,63%. Cả A và B chọn tầng 4 của wav2vec2 (bản cũ chọn tầng 2). Validation: A dev 1,62–2,11%, dev nhiễu 22,3–25,4%; B dev 1,89–2,12%, dev nhiễu 7,6–7,7%.

Ở ngưỡng chọn trên validation, In-the-Wild: A bỏ sót giọng giả 36,8–58,9% và báo nhầm giọng thật 6,5–11,1%; B bỏ sót 19,2–23,8% và báo nhầm 21,5–25,2% (bản cũ: bỏ sót 7,6–12,1%, báo nhầm 54,4–65,3%).

Nhận xét:
- Chỉ cắt im lặng (A) đã giảm EER trên In-the-Wild từ 30,0% xuống 21,8%, trong khi EER trên ASVspoof eval tăng từ 1,57% lên 8,89%. Điều này ủng hộ giả thuyết rằng một phần kết quả 1,57% đến từ đường tắt im lặng đầu file của ASVspoof 2019, khớp với kiểm tra đường tắt ghi trước khi xem eval (AUC 0,695).
- Thêm augmentation (B) không giảm thêm EER trên In-the-Wild (22,6% so với 21,8%; chênh lệch nằm trong độ lệch giữa các seed), nhưng làm kết quả ổn định hơn giữa các seed (SD 0,50% so với 1,58%), làm dev nhiễu tốt hơn nhiều (EER khoảng 7,6% so với 22–25%), và làm hai loại lỗi ở ngưỡng cân bằng hơn.
- EER khoảng 22% vẫn cao: nhánh audio vẫn chưa đủ tin cậy cho giọng nói ngoài đời.
- Cấu hình C (v2c) thêm giọng thật và giả đa ngôn ngữ từ MAVOS-DD, LAV-DF: In-the-Wild EER giảm còn 20,0% (chênh lệch 2,6 điểm, lớn hơn SD giữa các seed), ASVspoof eval không xấu đi. Ở ngưỡng validation, In-the-Wild bỏ sót giọng giả 47,0–50,0%, báo nhầm giọng thật 6,5–6,9%: ngưỡng dè dặt hơn B. Kết quả ngoài miền ở mục 4d.
- In-the-Wild chỉ được dùng để test; tầng, epoch và ngưỡng đều chọn trên validation. 3 seed dùng chung đặc trưng nên SD có thể thấp hơn thực tế.

## 7. Grad-CAM
**Checkpoint API vb_s44 (17/11; `reports/gradcam_vb/`)**, cùng cách chọn video (seed 0, 10 video mỗi loại FF++ c23 test): phân tích được 49/50 ảnh (NeuralTextures_759_755 không tìm được landmark). Năng lượng trong mặt: Deepfakes 93,1%, FaceSwap 90,6%, NeuralTextures 85,6%, Face2Face 85,2%, real 74,3%. Trung bình: mắt-mũi-miệng 17,5% năng lượng trên 6,0% diện tích; phần còn lại của mặt 68,2% / 41,1%; ngoài mặt 14,3% / 52,9% (trung vị 11,8%). Kết luận giống bản aug_s44: mô hình tập trung trong mặt với video giả, real có nhiều năng lượng ngoài mặt hơn. Ở ngưỡng EER 0,6697 có 5 ca sai: 3 video thật bị báo nhầm (047: 0,931; 507: 0,802; 862: 0,693) và 2 NeuralTextures bị bỏ sót (759_755: 0,331; 142_138: 0,405). Ảnh heatmap chỉ lưu trên máy (`data/gradcam_out_vb/`), không công bố.

Bản cũ aug_s44: 48 video (Deepfakes 10, Face2Face 10, FaceSwap 10, NeuralTextures 9, real 9), ba vùng không chồng lấn tách bằng FaceLandmarker (`data/gradcam_out/region_stats.csv`). Năng lượng trong mặt: Deepfakes 95,4%, FaceSwap 93,8%, Face2Face 89,1%, NeuralTextures 82,9%, real 69,3%. Trung bình mọi video: mắt-mũi-miệng 19,5% năng lượng trên 6,0% diện tích; phần còn lại của mặt 67,0% / 41,4%; ngoài mặt 13,5% / 52,6% (trung vị 7,5%).
Ở ngưỡng 0,5951 có 2 ca sai, đều là NeuralTextures bị bỏ sót (420_257: 0,5416; 142_138: 0,4426). Xem tay 4 ảnh: hai ca NeuralTextures có heatmap trong mặt; hai video thật 924 và 705 có heatmap gần như toàn bộ ngoài mặt, nguyên nhân chưa xác định. Một khả năng (chưa kiểm chứng): Grad-CAM trên đầu ra "giả" khi điểm gần 0 cho bản đồ nhiễu. Mẫu real chỉ có 9 video nên không đủ kết luận chung.

## 8. API và triển khai
`POST /predict` (video: điểm, nhãn, điểm từng frame, heatmap; dưới 4 frame mặt thì không chấm) và `POST /predict_audio` (wav/flac hoặc video qua ffmpeg; video không tiếng trả no_audio). Docker CPU (`--cpus=4 --memory=4g`), image 3,84 GB, chạy không cần mạng.

**Từ 16/11 nhánh video của API dùng checkpoint V-B `checkpoints_video_vb_s44`** (điểm chọn Val cao nhất trong 3 seed: 0,9832), `model_version` = `effnet_b0-vb_s44-epoch6`. Ngưỡng nhãn = ngưỡng EER FF++ Val 0,6697; ngưỡng `high_suspicion` = ngưỡng báo nhầm 5% FF++ Val 0,7163. Kiểm tra sau khi chuyển: parity max |d| 0,0048; 24/24 test đạt; latency Docker (20 video, bỏ 2 warm-up, `reports/latency_docker_cpu_vb.csv`) trung bình 1,51 s, p50 1,22 s, p95 2,40 s; RAM 594 MiB. Các số trong đoạn dưới là của bản trước (aug_s44).

**Từ 17/11 nhánh audio của API dùng checkpoint v2c `checkpoints_audio_v2c_s44`** (trung bình 3 EER validation thấp nhất trong 3 seed: 4,31%; s42 4,39%, s43 4,57%), `audio_model_version` = `w2v2base-L4-mlp-audio_v2c_s44-epoch20`. Ngưỡng nhãn = ngưỡng EER trong checkpoint 6,8268; ngưỡng `high_suspicion` = phân vị 95 logit bona fide ASVspoof dev sạch + dev nhiễu = 7,4202 (cùng cách tính ra 7,6671 cho v2b_s43). Kiểm tra sau khi chuyển: parity so với offline float32 max |d logit| 0,0000; 24/24 test đạt; Docker build lại, một video kiểm tra cho cùng điểm như trước (0,9976). Latency Docker (20 file, bỏ 2 warm-up): flac trung bình 0,140 s (p95 0,199 s); tách từ video 0,256 s (p95 0,324 s; bộ video có tiếng ghép lại vì bộ 10/11 không còn); RAM khoảng 0,6 GB. Đoạn dưới là của các bản trước.

Latency trong Docker (n = 20, bỏ 2 warm-up, `reports/latency_docker_cpu_*.csv`): video trung bình 1,675 s (p95 2,636 s); audio file flac 0,194 s; audio tách từ video 0,314 s.
Từ 09/11 nhánh audio của API dùng checkpoint audio v2b_s43 (cắt im lặng, chọn theo validation). Mỗi kết quả có thêm `verdict` ba mức: `high_suspicion` (điểm ≥ ngưỡng báo nhầm 5% trên validation: video 0,5961 cho aug_s44, nay 0,7163 cho vb_s44; audio logit 7,6671), `no_sign_detected`, `inconclusive` (không chấm được). Parity audio: lệch 0,0321 so với đặc trưng offline float16, trùng hoàn toàn khi tính lại float32 (ghi trong protocol). Parity video max |d| 0,0044 (đo lại 10/11). Latency Docker audio v2 (cùng 20 file, 10/11): flac trung bình 0,151 s (p95 0,226 s); tách từ video 0,275 s (p95 0,338 s). RAM container khoảng 0,58 GB.

## 9. Thử trên dữ liệu ngoài đời (minh họa, không phải tỷ lệ lỗi)
Clip CC trên Wikimedia Commons và clip của người làm và người thân (đã đồng ý): 1 ảnh thật được chấm đúng; 2/3 video thật bị nhánh video báo nhầm là giả (TikTok 0,72; phỏng vấn UFC 0,93); 3/3 giọng thật bị nhánh audio chấm spoof. Một video nghi do AI tạo hoàn toàn được chấm real 0,11 dựa trên 4 frame mặt; mô hình không được thiết kế cho loại này.

## 9b. Bộ test thực tế R1–R4 (protocol 09/11; mỗi checkpoint chạy một lần; số liệu đo)
Tỷ lệ báo nhầm ở ngưỡng EER của từng checkpoint, trung bình ± SD 3 seed; checkpoint API kèm khoảng tin cậy Wilson 95%.

| Tập | Nội dung | Video (baseline / aug / V-A / V-B) | Audio (cũ / v2a / v2b) |
|---|---|---|---|
| R1 | 149 clip thật Commons; chấm được: video 111, audio 144 | báo nhầm 16,2% ± 3,3 / 10,5% ± 1,4 / 9,0% ± 4,1 / 12,6% ± 1,8. API aug_s44: 10,8% [6,3–18,0]; API vb_s44: 12,6% [7,7–20,1] | báo nhầm 66,2% ± 5,7 / 43,1% ± 5,4 / 43,5% ± 1,1. API v2b_s43: 42,4% [34,6–50,5] |
| R2 | R1 nén như cuộc gọi; chấm được: video 111, audio 144 (v2: 143) | báo nhầm 19,2% ± 8,1 / 10,8% ± 2,4 / 11,1% ± 6,1 / 12,9% ± 5,0. API aug_s44: 12,6% [7,7–20,1]; API vb_s44: 12,6% [7,7–20,1] | báo nhầm 75,0% ± 3,9 / 61,1% ± 2,8 / 56,2% ± 0,8. API: 55,2% [47,1–63,2] |
| R3 | Celeb-DF 200 nén như cuộc gọi (199 chấm được) | AUC 0,7816 / 0,8319 / 0,8347 | — |
| R4 | 5.000 In-the-Wild nén Opus 16 kbps | — | EER 33,5% / 26,0% / 25,3% (AUC v2b 0,8296) |

Ở ngưỡng báo nhầm 5% (chọn trên validation): R3 video báo nhầm 15,2% [9,4–23,5], phát hiện 57,0%; R4 audio báo nhầm 42,0% [40,3–43,8], phát hiện 86,4%; R1 audio 45,8%, R2 audio 55,9%. V-B ở ngưỡng 5%: R1 9,3% (8,1–10,8), R2 9,6% (8,1–10,8); API vb_s44 10,8% [6,3–18,0] cả hai tập. R1/R2 có 38/149 video dưới 4 frame mặt, không chấm. Ngưỡng chọn trên dữ liệu phòng thí nghiệm không giữ được 5% ngoài đời. Augmentation giúp video rõ rệt; audio v2 tốt hơn bản cũ nhưng vẫn báo nhầm gần một nửa giọng thật, nên kết quả audio chỉ nên là tham khảo.

**Audio v2c trên R4 (17/11, `reports/eval_realworld_v2c/R4_summary.csv`)**: EER 22,60% / 22,28% / 22,98% (TB 22,62%; v2b 25,3%), AUC TB 0,851 (v2b 0,830). API v2c_s44 ở ngưỡng báo nhầm 5%: báo nhầm 12,0% [10,9–13,2], phát hiện 59,2% [56,9–61,4] (v2b_s43: 42,0% / 86,4%). v2c xếp hạng tốt hơn và báo nhầm ít hơn nhiều, nhưng bỏ sót nhiều hơn ở ngưỡng cố định.

**Audio v2c trên R1/R2 (17/11, `reports/eval_realworld_v2c/`)**, báo nhầm ở ngưỡng EER từng checkpoint (s42 / s43 / s44): R1 4,2% / 4,2% / 3,5% (TB 3,9%; v2b 43,5%); R2 7,0% / 7,7% / 7,7% (TB 7,5%; v2b 56,2%). API v2c_s44 ở ngưỡng báo nhầm 5%: R1 3,5% [1,5–7,9] (144 clip chấm được); R2 7,0% [3,8–12,4] (143 clip). Báo nhầm giảm khoảng 10 lần so với v2b. R1/R2 chỉ có clip thật nên chỉ đo được báo nhầm; v2c đồng thời dè dặt hơn trên giả ngoài miền (mục 4d: AV-Deepfake1M phát hiện 3,4%, SocialDF 0,2%), nên kết quả này cho thấy v2c ít báo nhầm hơn, chưa cho thấy bắt giả ngoài đời tốt hơn.

## 10. Hạn chế
- Đánh giá chéo thấp hơn in-domain. Nhánh audio: In-the-Wild EER 30% (bản đầu), 22,6% (v2b), 20,0% (v2c, API hiện tại); trên clip thật Commons API v2c báo nhầm 3,5% (7,0% khi nén như cuộc gọi), nhưng ngưỡng dè dặt nên bỏ sót nhiều giả ngoài miền.
- Deepfake thế hệ mới (MAVOS-DD, AV-Deepfake1M, LAV-DF, SocialDF): mô hình video API (V-B) có AUC 0,45–0,65, gần mức ngẫu nhiên; video tạo cả khuôn mặt (echomimic, memo, sonic) gần như không bị phát hiện. Train thêm dữ liệu mới (V-C) chỉ giúp trong miền và làm kém đi trên nguồn mới.
- AV-Deepfake1M, LAV-DF chỉ sửa đoạn dưới 1 giây; cách chấm trung bình 16 frame và 4 giây tiếng đầu không phù hợp với kiểu sửa này.
- SocialDF chỉ tải được 991/1.556 link (giả 50,6%), nhãn từ dấu hiệu người đăng và bình luận; không biết tiếng có bị giả không.
- Fusion không huấn luyện không tốt hơn nhánh tốt nhất, kể cả khi đo lại với v2c (mục 4e).
- R1/R2 chỉ có 149 clip thật, không có giả; danh sách ứng viên nghiêng về vài từ khóa đầu; tính "thật" không kiểm chứng tuyệt đối.
- Video học thay và chỉnh mặt kiểu FF++ (2019) và 2 công cụ đổi mặt (inswapper, simswap); không nhắm tới video tạo hoàn toàn bằng mô hình khuếch tán.
- V-B: công cụ chưa thấy khi train chỉ phát hiện 78,7% (Test-B); làm nét bằng GFPGAN còn 48%; nén như cuộc gọi làm báo nhầm video thật lên 17%; FF++ c23 giảm 0,012 AUC so với aug.
- Video giả V-B đều là đổi mặt giữa các người trong FF++ (720 người train, 140 người test), nên chưa đo trên người, ánh sáng, máy quay khác.
- Giấy phép: mô hình inswapper và simswap ghi "Non-Commercial"; checkpoint V-B học từ video do hai mô hình này tạo, chưa xác nhận được có được dùng ngoài nghiên cứu hay không.
- 3 seed audio dùng chung đặc trưng nên SD thấp có thể đánh giá thấp phương sai.
- Grad-CAM mẫu nhỏ; hai video thật có heatmap ngoài mặt chưa giải thích được.

## 11. Hướng phát triển
Dữ liệu train audio đa dạng hơn (Common Voice, AV-Deepfake1M), fine-tune mô hình đa ngôn ngữ, bộ phát hiện riêng cho video sinh bằng mô hình khuếch tán (GVF, GenVideo, GenVidBench), fusion có học. Phát hiện theo đoạn thời gian (temporal localization) cho kiểu sửa ngắn như AV-Deepfake1M, LAV-DF, thay vì chấm trung bình cả video. Kiểm tra tổng quát hóa bằng cách để riêng cả một bộ dữ liệu khi train (leave-one-dataset-out), vì V-C cho thấy kết quả trong cùng bộ dữ liệu không đại diện cho nguồn mới. Mega-MMDF (1,2 triệu clip giả, 30 phương pháp; đọc điều khoản 17/11): chưa dùng vì không phát hành video thật (phải tự xin VoxCeleb2, MEAD, CelebV-HQ), dung lượng 973 GB với từng nhóm phải tải trọn bộ, và giấy phép cấm triển khai model train trên bộ này.
