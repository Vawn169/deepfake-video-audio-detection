import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs('reports/figures', exist_ok=True)

# 1. Chart AUC comparison
datasets = ['FF++ c23\n(In-domain)', 'FF++ c40\n(Nén mạnh)', 'Celeb-DF v2\n(Cross-dataset)', 'Celeb-DF CRF30\n(Cross + Nén)']
baseline_means = [0.9889, 0.7987, 0.7807, 0.7176]
baseline_stds  = [0.0042, 0.0163, 0.0215, 0.0183]

aug_means = [0.9807, 0.8714, 0.8373, 0.7649]
aug_stds  = [0.0015, 0.0113, 0.0111, 0.0161]

x = np.arange(len(datasets))
width = 0.35

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)

rects1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline (Không Augmentation)',
                color='#1B365D', capsize=5, edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, aug_means, width, yerr=aug_stds, label='Có Augmentation (RandomDegrade)',
                color='#2B6CB0', capsize=5, edgecolor='black', alpha=0.9)

ax.set_ylabel('AUC Cấp Video (Mean ± Std của 3 Seeds)', fontsize=11, fontweight='bold', color='#1B365D')
ax.set_title('So Sánh Hiệu Năng Mô Hình: Baseline vs. Augmentation Nén (3 Seeds: 42, 43, 44)',
             fontsize=12, fontweight='bold', pad=15, color='#1B365D')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10, fontweight='bold')
ax.set_ylim(0.60, 1.03)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 6), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
chart_auc_path = 'reports/figures/chart_auc_comparison.png'
plt.savefig(chart_auc_path)
plt.close()
print("Đã tạo:", chart_auc_path)

# 2. Chart Latency comparison
# cpu_local: mean=1.14s, p50=0.98s, p95=1.58s, max=1.58s
# docker_cpu: mean=1.57s, p50=1.31s, p95=2.49s, max=2.50s
metrics = ['Trung bình (Mean)', 'Trung vị (p50)', 'Phân vị 95% (p95)', 'Cực đại (Max)']
local_vals = [1.14, 0.98, 1.58, 1.58]
docker_vals = [1.57, 1.31, 2.49, 2.50]

x = np.arange(len(metrics))
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)

r1 = ax.bar(x - width/2, local_vals, width, label='CPU Local (WSL2 Host)',
            color='#2D3748', edgecolor='black', alpha=0.9)
r2 = ax.bar(x + width/2, docker_vals, width, label='Docker CPU (--cpus=4 --memory=4g)',
            color='#DD6B20', edgecolor='black', alpha=0.9)

ax.set_ylabel('Thời Gian Xử Lý (Giây / Video)', fontsize=11, fontweight='bold', color='#1B365D')
ax.set_title('Độ Trễ Suy Luận API: Môi Trường Local vs. Container Docker (N=20 Video)',
             fontsize=12, fontweight='bold', pad=15, color='#1B365D')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10, fontweight='bold')
ax.set_ylim(0, 3.0)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

def autolabel_lat(rects):
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.2f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel_lat(r1)
autolabel_lat(r2)

plt.tight_layout()
chart_lat_path = 'reports/figures/chart_latency.png'
plt.savefig(chart_lat_path)
plt.close()
print("Đã tạo:", chart_lat_path)
