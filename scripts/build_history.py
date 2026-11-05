import re, csv, statistics as st
pat = re.compile(r"Epoch (\d+)/\d+ \(([\d.]+)s\) \| Train Loss: ([\d.]+) \| Val Loss: ([\d.]+) "
                 r"\| Val AUC: ([\d.]+) \| Val Acc: ([\d.]+) \| Val F1: ([\d.]+)( \[\*BEST\*\])?")
best = {}
for s in (42, 43, 44):
    rows = [pat.search(l) for l in open(f"reports/train_baseline_s{s}.log")]
    rows = [m for m in rows if m]
    assert len(rows) == 10, (s, len(rows))
    with open(f"reports/train_history_s{s}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "seconds", "train_loss", "val_loss", "val_auc", "val_acc", "val_f1", "is_best"])
        for m in rows:
            w.writerow([m[1], m[2], m[3], m[4], m[5], m[6], m[7], int(bool(m[8]))])
    b = max(rows, key=lambda m: float(m[5]))
    best[s] = (float(b[5]), int(b[1]), float(b[4]), float(rows[-1][4]))

aucs = [v[0] for v in best.values()]
lines = ["# Tuần 3: kết quả huấn luyện baseline (EfficientNet-B0, FF++ c23)", "",
         "Cấu hình: 10 epoch, batch 64, AdamW lr 2e-4, cosine, AMP, pos_weight 0.25. Chọn checkpoint theo Val Video AUC.", "",
         "| Seed | Best Val AUC | Epoch best | Val loss ở best | Val loss epoch 10 |", "|---|---|---|---|---|"]
for s, (a, e, vl, vl10) in best.items():
    lines.append(f"| {s} | {a:.4f} | {e} | {vl:.4f} | {vl10:.4f} |")
lines += ["", f"Trung bình ± độ lệch chuẩn mẫu: **{st.mean(aucs):.4f} ± {st.stdev(aucs):.4f}**", "",
          "Ghi chú: Val AUC phẳng từ epoch 2 và chênh lệch giữa seed nhỏ hơn nhiễu của 694 video val. "
          "Val loss tăng trong khi train loss giảm (overfit, quá tự tin) nên accuracy/F1 ở ngưỡng cố định kém tin cậy hơn AUC. "
          "Kết quả này là trong miền FF++ c23, chưa chứng minh khái quát hóa.", ""]
open("reports/week3_summary.md", "w").write("\n".join(lines))
print("\n".join(lines))
