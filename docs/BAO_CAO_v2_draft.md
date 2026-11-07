# Phát hiện Deepfake Video và Âm thanh — bản nháp báo cáo (cập nhật 08/11/2026)

Bản nháp này thay cho phần kết quả của `reports/BAO_CAO_tong_hop_GVHD.docx` (bản chụp 28/10). Mỗi con số ghi kèm file nguồn trong repo. Phần đánh dấu **[chờ]** sẽ điền khi có kết quả.

## 1. Bài toán và phạm vi
- Mức 1 (video): EfficientNet-B0 trên ảnh mặt, huấn luyện FF++ c23, đánh giá chéo, Grad-CAM, API FastAPI chạy Docker CPU. Hoàn thành.
- Mức 2 (audio): đặc trưng wav2vec2-base đóng băng + MLP nhỏ, huấn luyện ASVspoof 2019 LA. Hoàn thành; đang nâng cấp (audio v2).
- Mức 3 (fusion): bản không huấn luyện (trung bình xác suất hai nhánh) sẽ đo trên Deepfake-Eval-2024 khi được cấp quyền.
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
| Deepfake-Eval-2024 | Test video + audio 2024 (mới) | chờ duyệt quyền (gửi 06/11) | CC BY-SA 4.0, chỉ được dùng để đánh giá |
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

## 4. Kết quả video (AUC cấp video, trung bình ± SD mẫu 3 seed)
Nguồn: `reports/eval/checkpoints_*/summary.csv`.

| Tập | Baseline | + augmentation nén | V-A (mạng xã hội) |
|---|---|---|---|
| FF++ c23 test | 0,9889 ± 0,0042 | 0,9807 ± 0,0015 | 0,9788 ± 0,0009 |
| FF++ c40 test | 0,7987 ± 0,0163 | 0,8714 ± 0,0113 | 0,8832 ± 0,0075 |
| Celeb-DF 200 | 0,7807 ± 0,0214 | 0,8373 ± 0,0111 | 0,8422 ± 0,0154 |
| Celeb-DF CRF30 | 0,7176 ± 0,0183 | 0,7649 ± 0,0161 | 0,7929 ± 0,0263 |
| Deepfake-Eval-2024 | [chờ quyền] | [chờ quyền] | [chờ quyền] |

Cột V-A: các tập này đã được xem trước khi thiết kế V-A, chỉ báo cáo kèm, không dùng để chọn; tập test chính của V-A là Deepfake-Eval-2024. Chênh lệch V-A so với aug (+0,012 c40, +0,005 Celeb-DF, +0,028 CRF30, −0,002 c23) nhỏ so với độ lệch giữa các seed, chưa đủ để kết luận V-A tốt hơn. V-A từng seed: c23 0,9779/0,9797/0,9787; c40 0,8875/0,8876/0,8745; Celeb-DF 0,8366/0,8596/0,8303; CRF30 0,8025/0,8130/0,7632. Chọn checkpoint theo trung bình AUC Val sạch và Val SocialDegrade: 0,9767 / 0,9745 / 0,9735.

Từng seed (baseline / aug): c23 0,9842/0,9902/0,9922 và 0,9822/0,9793/0,9805; c40 0,8001/0,8143/0,7818 và 0,8604/0,8708/0,8830; Celeb-DF 0,7754/0,8042/0,7624 và 0,8459/0,8413/0,8248; CRF30 0,7274/0,7288/0,6965 và 0,7694/0,7783/0,7470. Augmentation tăng AUC ở cả ba seed trên c40, Celeb-DF, CRF30 và giảm ở cả ba seed trên c23.

Ví dụ checkpoint chính aug_s44 (ngưỡng Val 0,5951): Celeb-DF AUC 0,8248 [KTC 95% bootstrap 0,7643–0,8786], accuracy 0,72; c40 AUC 0,8830 [0,8571–0,9062], accuracy 0,7496. Accuracy ở ngưỡng cố định thấp hơn AUC khi đổi miền, như đã dự kiến trong giao thức.

Mô hình: 4.008.829 tham số huấn luyện được, checkpoint 16,3 MB.

## 5. Kiểm tra chống lỗ hổng (checklist mục 4 của kế hoạch)
| Kiểm tra | Kết quả | Nguồn |
|---|---|---|
| Rò rỉ dữ liệu | Danh tính FF++ train/val/test giao nhau = 0 (720/140/140); kiểm tra tự động trong `tests/test_splits.py` | tests |
| Mặt thất bại | Xem mục 2 | preprocess_*.csv |
| Mô hình siêu dữ liệu | AUC 0,50–0,58 (logistic regression và random forest, 4 tập) | metadata_shortcut_check.csv |
| Overfit 32 video | AUC 1,0 | sanity_check_results.json |
| Xáo nhãn | Val AUC 0,4933 / 0,5378 / 0,6289 (TB 0,5533) | sanity_check_results.json |
| Ngưỡng | Chọn một lần trên Val, giữ nguyên khi test chéo | protocol.md |
| Tiền xử lý đồng nhất | API vs offline: video max \|d\| 0,0044 (10 video); audio max \|d logit\| 0,0049 (10 file) | api_parity*.py |
| Heatmap | Xem mục 7 | region_stats.csv |
| Tái lập | Repo public clone vào máy sạch, cài requirements, 17/17 test đạt (06/11); bộ test hiện tại 24 test | NEXT_STEPS.md |

## 6. Kết quả audio
Nguồn: `reports/audio_*_results.csv`.

| Tập | EER (TB ± SD, 3 seed) | AUC |
|---|---|---|
| ASVspoof 2019 LA dev | 0,15% / 0,11% / 0,12% | — |
| ASVspoof 2019 LA eval | 1,57% ± 0,03% | 0,9987 ± 0,0001 |
| In-the-Wild (ngoài đời) | 30,03% ± 0,21% | 0,7695 ± 0,0047 |
| Deepfake-Eval-2024 audio | [chờ quyền] | |

Ở ngưỡng chọn trên ASVspoof dev, In-the-Wild bị báo nhầm giọng thật 54,4% / 65,3% / 60,9% và bỏ sót giọng giả 12,1% / 7,6% / 9,2%. Mô hình vẫn xếp hạng được một phần (AUC 0,77) nhưng ngưỡng không còn phù hợp khi đổi miền.

Chẩn đoán (05/11, 40+40 file dev, script chưa đưa vào repo): nén AAC 128/64 kbps và đổi tần số lấy mẫu không làm giọng thật bị đổi nhãn (0%); cắt im lặng đầu file làm 22,5% giọng thật chuyển sang spoof (giọng thật trong ASVspoof có trung vị 0,96 s im lặng đầu, giọng giả 0,35 s). Nguyên nhân chính được suy ra: toàn bộ giọng thật khi train đến từ một nguồn thu âm phòng thu tiếng Anh. Kiểm tra đường tắt trên dev trước khi xem eval: chỉ dùng độ dài im lặng đầu file đạt AUC 0,695.

Audio v2 (mục 08/11 trong protocol), đánh giá trên In-the-Wild và ASVspoof eval:
| Cấu hình | In-the-Wild EER | In-the-Wild AUC | ASVspoof eval EER |
|---|---|---|---|
| Hiện tại (không cắt im lặng) | 30,03% ± 0,21% | 0,7695 ± 0,0047 | 1,57% ± 0,03% |
| A: cắt im lặng | 21,83% ± 1,58% | 0,8579 ± 0,0173 | 8,89% ± 1,02% |
| B: A + tiếng vang, tiếng ồn, băng thoại, codec | 22,56% ± 0,50% | 0,8620 ± 0,0053 | 10,93% ± 0,27% |

Từng seed (42/43/44), In-the-Wild EER: A 21,60% / 23,51% / 20,38%; B 22,70% / 22,98% / 22,01%. ASVspoof eval EER: A 10,06% / 8,31% / 8,30%; B 11,13% / 11,02% / 10,63%. Cả A và B chọn tầng 4 của wav2vec2 (bản cũ chọn tầng 2). Validation: A dev 1,62–2,11%, dev nhiễu 22,3–25,4%; B dev 1,89–2,12%, dev nhiễu 7,6–7,7%.

Ở ngưỡng chọn trên validation, In-the-Wild: A bỏ sót giọng giả 36,8–58,9% và báo nhầm giọng thật 6,5–11,1%; B bỏ sót 19,2–23,8% và báo nhầm 21,5–25,2% (bản cũ: bỏ sót 7,6–12,1%, báo nhầm 54,4–65,3%).

Nhận xét:
- Chỉ cắt im lặng (A) đã giảm EER trên In-the-Wild từ 30,0% xuống 21,8%, trong khi EER trên ASVspoof eval tăng từ 1,57% lên 8,89%. Điều này ủng hộ giả thuyết rằng một phần kết quả 1,57% đến từ đường tắt im lặng đầu file của ASVspoof 2019, khớp với kiểm tra đường tắt ghi trước khi xem eval (AUC 0,695).
- Thêm augmentation (B) không giảm thêm EER trên In-the-Wild (22,6% so với 21,8%; chênh lệch nằm trong độ lệch giữa các seed), nhưng làm kết quả ổn định hơn giữa các seed (SD 0,50% so với 1,58%), làm dev nhiễu tốt hơn nhiều (EER khoảng 7,6% so với 22–25%), và làm hai loại lỗi ở ngưỡng cân bằng hơn.
- EER khoảng 22% vẫn cao: nhánh audio vẫn chưa đủ tin cậy cho giọng nói ngoài đời. Cấu hình C (thêm giọng thật đa dạng) chưa làm vì chưa có dữ liệu.
- In-the-Wild chỉ được dùng để test; tầng, epoch và ngưỡng đều chọn trên validation. 3 seed dùng chung đặc trưng nên SD có thể thấp hơn thực tế.

## 7. Grad-CAM
48 video (Deepfakes 10, Face2Face 10, FaceSwap 10, NeuralTextures 9, real 9), checkpoint aug_s44, ba vùng không chồng lấn tách bằng FaceLandmarker (`data/gradcam_out/region_stats.csv`). Năng lượng trong mặt: Deepfakes 95,4%, FaceSwap 93,8%, Face2Face 89,1%, NeuralTextures 82,9%, real 69,3%. Trung bình mọi video: mắt-mũi-miệng 19,5% năng lượng trên 6,0% diện tích; phần còn lại của mặt 67,0% / 41,4%; ngoài mặt 13,5% / 52,6% (trung vị 7,5%).
Ở ngưỡng 0,5951 có 2 ca sai, đều là NeuralTextures bị bỏ sót (420_257: 0,5416; 142_138: 0,4426). Xem tay 4 ảnh: hai ca NeuralTextures có heatmap trong mặt; hai video thật 924 và 705 có heatmap gần như toàn bộ ngoài mặt, nguyên nhân chưa xác định. Một khả năng (chưa kiểm chứng): Grad-CAM trên đầu ra "giả" khi điểm gần 0 cho bản đồ nhiễu. Mẫu real chỉ có 9 video nên không đủ kết luận chung.

## 8. API và triển khai
`POST /predict` (video: điểm, nhãn, điểm từng frame, heatmap; dưới 4 frame mặt thì không chấm) và `POST /predict_audio` (wav/flac hoặc video qua ffmpeg; video không tiếng trả no_audio). Docker CPU (`--cpus=4 --memory=4g`), image 3,84 GB, chạy không cần mạng.
Latency trong Docker (n = 20, bỏ 2 warm-up, `reports/latency_docker_cpu_*.csv`): video trung bình 1,675 s (p95 2,636 s); audio file flac 0,194 s; audio tách từ video 0,314 s.

## 9. Thử trên dữ liệu ngoài đời (minh họa, không phải tỷ lệ lỗi)
Clip CC trên Wikimedia Commons và clip của người làm và người thân (đã đồng ý): 1 ảnh thật được chấm đúng; 2/3 video thật bị nhánh video báo nhầm là giả (TikTok 0,72; phỏng vấn UFC 0,93); 3/3 giọng thật bị nhánh audio chấm spoof. Một video nghi do AI tạo hoàn toàn được chấm real 0,11 dựa trên 4 frame mặt; mô hình không được thiết kế cho loại này.

## 10. Hạn chế
- Đánh giá chéo thấp hơn in-domain; nhánh audio của API hiện tại gần như không dùng được trên giọng nói ngoài đời (In-the-Wild EER 30%); bản v2 giảm còn khoảng 22% nhưng vẫn cao.
- Video chỉ học thay và chỉnh mặt kiểu FF++ (2019); không nhắm tới video tạo hoàn toàn bằng mô hình khuếch tán.
- 3 seed audio dùng chung đặc trưng nên SD thấp có thể đánh giá thấp phương sai.
- Grad-CAM mẫu nhỏ; hai video thật có heatmap ngoài mặt chưa giải thích được.

## 11. Hướng phát triển
Dữ liệu train audio đa dạng hơn (Common Voice, AV-Deepfake1M), fine-tune mô hình đa ngôn ngữ, bộ phát hiện riêng cho video sinh bằng mô hình khuếch tán (GVF, GenVideo, GenVidBench), fusion có học.
