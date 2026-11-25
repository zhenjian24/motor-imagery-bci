"""
生成三种方法的 9 被试结果对比图: figures/method_comparison.png
运行: python plot_method_comparison.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # Windows 中文字体
plt.rcParams["axes.unicode_minus"] = False

# ---- 实测结果 (4 类, T 会话内分层 10 折 CV, random_state=42) ----
subjects = [f"S{i}" for i in range(1, 10)] + ["平均"]
KAPPA = {
    "CSP+LDA 基线":   [0.657, 0.523, 0.792, 0.403, 0.148, 0.366, 0.727, 0.745, 0.574, 0.548],
    "FBCSP+互信息":   [0.773, 0.306, 0.838, 0.338, 0.579, 0.333, 0.801, 0.769, 0.718, 0.606],
    "黎曼几何":       [0.782, 0.310, 0.843, 0.468, 0.204, 0.366, 0.718, 0.810, 0.796, 0.588],
}
MEAN_ACC = {"CSP+LDA 基线": 0.661, "FBCSP+互信息": 0.704, "黎曼几何": 0.691}
COLORS = {"CSP+LDA 基线": "#888780", "FBCSP+互信息": "#1D9E75", "黎曼几何": "#185FA5"}
REF_KAPPA = 0.569  # FBCSP 公开基线

fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 1]})

# ---- 左图: 逐被试 kappa 分组柱状图 ----
ax = axes[0]
x = np.arange(len(subjects))
w = 0.26
for i, (name, vals) in enumerate(KAPPA.items()):
    ax.bar(x + (i - 1) * w, vals, w, label=name, color=COLORS[name])
ax.axhline(REF_KAPPA, ls="--", lw=1.2, color="#BA7517")
ax.text(len(subjects) - 1.05, REF_KAPPA + 0.015, f"公开基线 {REF_KAPPA}",
        color="#854F0B", fontsize=9, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(subjects)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Cohen's kappa")
ax.set_title("逐被试 kappa 对比")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

# ---- 右图: 9 被试平均 acc 与 kappa ----
ax = axes[1]
methods = list(KAPPA.keys())
acc = [MEAN_ACC[m] for m in methods]
kap = [v[-1] for v in KAPPA.values()]  # 最后一列是平均
xx = np.arange(3)
w2 = 0.35
ax.bar(xx - w2 / 2, acc, w2, label="平均 Acc", color="#85B7EB")
ax.bar(xx + w2 / 2, kap, w2, label="平均 Kappa", color="#EF9F27")
for i, (a, k) in enumerate(zip(acc, kap)):
    ax.text(i - w2 / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
    ax.text(i + w2 / 2, k + 0.015, f"{k:.3f}", ha="center", fontsize=9)
ax.set_xticks(xx)
ax.set_xticklabels(["CSP+LDA", "FBCSP+MI", "黎曼"], fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_title("9 被试平均指标")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("BCI IV-2a 四类运动想象: 三种方法 9 被试结果对比", fontsize=13)
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
out = os.path.join("figures", "method_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
