# 运动想象四类分类基线 (BCI Competition IV-2a)

基于 MOABB 的 `BNCI2014_001`（即 BCI Competition IV 2a）数据集，
用 **CSP + LDA** 复现四类运动想象（左手 / 右手 / 双脚 / 舌头）EEG 分类基线。

## 环境

```bash
pip install -r requirements.txt
```

## 运行

```bash
python motor_imagery_baseline.py                # 默认 4 类
python motor_imagery_baseline.py --n_classes 2  # 左右二分类
```

首次运行会自动通过 MOABB 下载数据（默认到 `MNE_DATA` 指向的目录），无需手动准备 `.gdf` 文件。

## 流程

```
读取(MOABB) -> 带通 8-30Hz -> CSP -> LDA -> 分层 10 折 CV -> acc / kappa
```

## 输出

终端打印每位被试的准确率（accuracy）与 Cohen's kappa，以及 9 名被试的平均值。
（公开文献中 4 类 FBCSP 基线的 kappa 约为 0.569，可作为参考对照。）

## 目录结构

```
motor-imagery-bci/
├── motor_imagery_baseline.py   # 主脚本
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE                     # 公开仓库建议添加
```

> 注意：原始 EEG 数据（`.gdf` / `mne_data/`）体积很大，**不纳入版本库**，
> 他人克隆后通过 `BNCI2014_001().download()` 自行获取数据。
