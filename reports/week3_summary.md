# Tuần 3: kết quả huấn luyện baseline (EfficientNet-B0, FF++ c23)

Cấu hình: 10 epoch, batch 64, AdamW lr 2e-4, cosine, AMP, pos_weight 0.25. Chọn checkpoint theo Val Video AUC.

| Seed | Best Val AUC | Epoch best | Val loss ở best | Val loss epoch 10 |
|---|---|---|---|---|
| 42 | 0.9887 | 2 | 0.1022 | 0.1355 |
| 43 | 0.9886 | 7 | 0.1524 | 0.1513 |
| 44 | 0.9901 | 8 | 0.1300 | 0.1529 |

Trung bình ± độ lệch chuẩn mẫu: **0.9891 ± 0.0008**

Ghi chú: Val AUC phẳng từ epoch 2 và chênh lệch giữa seed nhỏ hơn nhiễu của 694 video val. Val loss tăng trong khi train loss giảm (overfit, quá tự tin) nên accuracy/F1 ở ngưỡng cố định kém tin cậy hơn AUC. Kết quả này là trong miền FF++ c23, chưa chứng minh khái quát hóa.
