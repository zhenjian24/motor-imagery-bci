"""
BCI Competition IV-2a 优化方案: FBCSP + 互信息特征选择 + 黄金时间窗裁剪
======================================================================
1. 时间窗裁剪: 0.5s ~ 2.5s (避开视觉刺激 ERP 和后期疲劳)
2. 滤波器组: 4-40Hz 划分为 9 个子频带 (4Hz 带宽)
3. 空间滤波: 每个子频带分别进行 CSP 投影
4. 特征选择: 互信息 (Mutual Information) 筛选最具判别力的特征
5. 分类器: Shrinkage LDA (带正则化)

运行:
    python fbcsp_baseline.py            # 默认 4 类
    python fbcsp_baseline.py --n_classes 2   # 左右手二分类
"""
import argparse
import numpy as np
from scipy.signal import butter, filtfilt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from mne.decoding import CSP
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery, LeftRightImagery

# ----------------------------- 配置 -----------------------------
N_CLASSES = 4       # 4: 四分类; 2: 左右手二分类
T_MIN, T_MAX = 0.5, 2.5   # 黄金有效时间窗 (提示后 0.5s ~ 2.5s)
FS = 250.0          # BNCI2014_001 采样率为 250 Hz
N_SPLITS = 10
SUBJECTS = range(1, 10)
N_JOBS = 1          # Windows/Jupyter 下保持 1，避免多进程锁死

# 9 个子频带 (4Hz 步长，覆盖 4-40Hz)
SUB_BANDS = [
    (4, 8), (8, 12), (12, 16), (16, 20),
    (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)
]


class FilterBankCSP(BaseEstimator, TransformerMixin):
    """滤波器组 CSP 转换器，严格兼容 scikit-learn 流水线"""
    def __init__(self, bands=SUB_BANDS, sfreq=FS, n_csp=4):
        self.bands = bands
        self.sfreq = sfreq
        self.n_csp = n_csp
        self.csps = []

    def _bandpass_filter(self, X, low, high, order=3):
        """巴特沃斯双向无失真带通滤波"""
        nyq = 0.5 * self.sfreq
        b, a = butter(order, [low / nyq, high / nyq], btype="bandpass")
        # X 维度: (n_trials, n_channels, n_times)
        return filtfilt(b, a, X, axis=-1)

    def fit(self, X, y):
        self.csps = []
        for low, high in self.bands:
            X_filtered = self._bandpass_filter(X, low, high)
            csp = CSP(
                n_components=self.n_csp,
                reg="ledoit_wolf",
                log=True,
                norm_trace=False
            )
            csp.fit(X_filtered, y)
            self.csps.append(csp)
        return self

    def transform(self, X):
        features = []
        for (low, high), csp in zip(self.bands, self.csps):
            X_filtered = self._bandpass_filter(X, low, high)
            feat = csp.transform(X_filtered)
            features.append(feat)
        # 拼接所有子频带的 CSP 特征: (n_trials, n_bands * n_csp_components)
        return np.concatenate(features, axis=1)


def build_pipeline(n_classes, n_csp=4, k_best=16):
    """
    构建 FBCSP + 互信息特征选择 + Shrinkage LDA 流水线
    - k_best: 筛选出的最具判别力特征数 (4分类建议 16~24，2分类建议 8~12)
    """
    fbcsp = FilterBankCSP(bands=SUB_BANDS, sfreq=FS, n_csp=n_csp)
    selector = SelectKBest(
        score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
        k=k_best
    )
    lda = LDA(solver="eigen", shrinkage="auto")

    return Pipeline([
        ("fbcsp", fbcsp),
        ("selector", selector),
        ("lda", lda)
    ])


def evaluate_subject(X, y, n_classes, n_splits=N_SPLITS, n_jobs=N_JOBS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 动态参数匹配
    n_csp = 6 if n_classes == 4 else 4
    k_best = 20 if n_classes == 4 else 8

    clf = build_pipeline(n_classes=n_classes, n_csp=n_csp, k_best=k_best)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=n_jobs)

    acc = accuracy_score(y, y_pred)
    kap = cohen_kappa_score(y, y_pred)
    return acc, kap, y, y_pred


def load_subject_data(dataset, subject, n_classes=4):
    """
    加载数据，应用 0.5s ~ 2.5s 时间窗截取，并保留 4~40Hz 宽带用于后续分频
    """
    if n_classes == 2:
        paradigm = LeftRightImagery(
            fmin=4.0,
            fmax=40.0,
            tmin=T_MIN,
            tmax=T_MAX,
            resample=FS
        )
    else:
        paradigm = MotorImagery(
            n_classes=4,
            fmin=4.0,
            fmax=40.0,
            tmin=T_MIN,
            tmax=T_MAX,
            resample=FS,
            events=['left_hand', 'right_hand', 'feet', 'tongue']
        )

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    y = np.asarray(y)

    # 仅保留训练会话 (T session)
    if hasattr(meta, "columns") and "session" in meta.columns:
        sess = meta["session"].astype(str).str.lower()
        train_mask = sess.str.contains("train").values
        assert train_mask.any(), f"Subject {subject}: 未找到训练会话"
        X, y = X[train_mask], y[train_mask]

    X = np.nan_to_num(X.astype(np.float64))
    return X, y


def main():
    parser = argparse.ArgumentParser(description="BCI IV-2a FBCSP Baseline")
    parser.add_argument("--n_classes", type=int, default=N_CLASSES, choices=[2, 4])
    args, _ = parser.parse_known_args()

    dataset = BNCI2014_001()
    results = []

    print(f"=== 开始运行: 类别数={args.n_classes}, 裁剪窗=[{T_MIN}s, {T_MAX}s], 方案=FBCSP+MI ===")

    for s in SUBJECTS:
        X, y = load_subject_data(dataset, s, n_classes=args.n_classes)
        print(f"\n[Subject {s}] 样本数={X.shape[0]}, 通道={X.shape[1]}, 采样点={X.shape[2]}")

        acc, kap, y_true, y_pred = evaluate_subject(X, y, n_classes=args.n_classes)
        results.append((s, acc, kap))
        print(f"  Acc: {acc * 100:.2f}% | Kappa: {kap:.3f}")
        print(confusion_matrix(y_true, y_pred))

    mean_acc = np.mean([r[1] for r in results])
    mean_kap = np.mean([r[2] for r in results])

    ref_kappa = 0.569 if args.n_classes == 4 else 0.65

    print("\n" + "=" * 45)
    print("                 9 被试总览汇总")
    print("=" * 45)
    for s, acc, kap in results:
        print(f"  Subject {s}: Acc = {acc * 100:.2f}% | Kappa = {kap:.3f}")
    print("-" * 45)
    print(f"  平均准确率 (Mean Acc)  : {mean_acc * 100:.2f}%")
    print(f"  平均 Kappa 系数        : {mean_kap:.3f} (比赛冠军基线: ~{ref_kappa})")
    print("=" * 45)


if __name__ == "__main__":
    main()
