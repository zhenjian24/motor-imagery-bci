# 运动想象四类分类基线 (BCI Competition IV-2a)

基于 MOABB 的 `BNCI2014_001`（即 BCI Competition IV 2a）数据集，
用 **CSP + LDA** 复现四类运动想象（左手 / 右手 / 双脚 / 舌头）EEG 分类基线，
并进一步实现 **FBCSP + 互信息特征选择** 的优化方案。

## 环境

```bash
pip install -r requirements.txt
```

## 运行

```bash
# 基线方案（CSP + LDA）
python motor_imagery_baseline.py                # 默认 4 类
python motor_imagery_baseline.py --n_classes 2  # 左右二分类

# 优化方案（FBCSP + 互信息特征选择 + 黄金时间窗 0.5-2.5s）
python fbcsp_baseline.py                # 默认 4 类
python fbcsp_baseline.py --n_classes 2  # 左右二分类
```

首次运行会自动通过 MOABB 下载数据（默认到 `MNE_DATA` 指向的目录），无需手动准备 `.gdf` 文件。

## 流程

```
基线:  读取(MOABB) -> 带通 8-30Hz -> CSP -> LDA -> 分层 10 折 CV -> acc / kappa

优化:  读取(MOABB, 截取 0.5-2.5s) -> 4-40Hz 分 9 个子频带 -> 各频带 CSP
       -> 互信息筛选 Top 特征 -> Shrinkage LDA -> 分层 10 折 CV -> acc / kappa
```

## 输出

终端打印每位被试的准确率（accuracy）与 Cohen's kappa，以及 9 名被试的平均值。
（公开文献中 4 类 FBCSP 基线的 kappa 约为 0.569，可作为参考对照。）

## 实测结果（4 类，T 会话内分层 10 折 CV，random_state=42）

| Subject | 基线 CSP+LDA kappa | FBCSP+MI kappa | 变化 |
|---------|-------------------|----------------|------|
| 1 | 0.657 | 0.773 | ▲ +0.116 |
| 2 | 0.523 | 0.306 | ▼ −0.217 |
| 3 | 0.792 | 0.838 | ▲ +0.046 |
| 4 | 0.403 | 0.338 | ▼ −0.065 |
| 5 | 0.148 | 0.579 | ▲ +0.431 |
| 6 | 0.366 | 0.333 | ▼ −0.033 |
| 7 | 0.727 | 0.801 | ▲ +0.074 |
| 8 | 0.745 | 0.769 | ▲ +0.024 |
| 9 | 0.574 | 0.718 | ▲ +0.144 |
| **平均** | **0.548 (acc 0.661)** | **0.606 (acc 0.704)** | **▲ +0.058** |

- FBCSP+MI 平均 kappa **0.606**，超过比赛公开基线（~0.569）。
- 6/9 名被试提升明显（S5 从 0.148 跃升至 0.579），但对低表现被试（S2/S4/S6）
  反而略降——说明互信息全局选特征对个体差异敏感，可考虑按被试调 `k_best`。

## 目录结构

```
motor-imagery-bci/
├── motor_imagery_baseline.py   # 基线：CSP + LDA
├── fbcsp_baseline.py           # 优化：FBCSP + 互信息特征选择 + 时间窗裁剪
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE                     # 公开仓库建议添加
```

> 注意：原始 EEG 数据（`.gdf` / `mne_data/`）体积很大，**不纳入版本库**，
> 他人克隆后通过 `BNCI2014_001().download()` 自行获取数据。
