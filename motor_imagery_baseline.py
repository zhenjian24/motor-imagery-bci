"""
BCI Competition IV-2a 复现基线 (路线 A: MOABB 数据)
================================================================
读取(MOABB) -> 带通(8-30Hz) -> CSP -> LDA -> 分层 10 折 CV -> 指标

运行:
    python motor_imagery_baseline.py            # 默认 4 类
    python motor_imagery_baseline.py --n_classes 2   # 左右二分类
"""
import argparse
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from mne.decoding import CSP

# ----------------------------- 配置 -----------------------------
FILT_LO, FILT_HI = 8.0, 30.0
N_CSP = 8            # 4 类时每类 2 个成分 -> 8；2 类时常用 4~6
N_SPLITS = 10
SUBJECTS = range(1, 10)
N_JOBS = 1          # Windows 下建议 1，避免 multiprocessing 报错


def build_pipeline(n_csp=N_CSP):
    csp = CSP(n_components=n_csp, reg="ledoit_wolf", log=True, norm_trace=False)
    clf = Pipeline([
        ("csp", csp),
        ("sc", StandardScaler()),
        ("lda", LDA(solver="eigen", shrinkage="auto")),
    ])
    return clf


def evaluate_subject(X, y, n_csp=N_CSP, n_splits=N_SPLITS, n_jobs=N_JOBS):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = build_pipeline(n_csp)
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=n_jobs)
    acc = accuracy_score(y, y_pred)
    kap = cohen_kappa_score(y, y_pred)
    return acc, kap, y, y_pred


def load_subject_data(dataset, subject, n_classes=4):
    from moabb.paradigms import MotorImagery, LeftRightImagery
    if n_classes == 2:
        paradigm = LeftRightImagery(fmin=FILT_LO, fmax=FILT_HI)
    else:
        # MotorImagery 取全部 4 类（左手/右手/双脚/舌头）
        paradigm = MotorImagery(fmin=FILT_LO, fmax=FILT_HI)

    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])
    y = np.asarray(y)

    # 只保留训练会话（BNCI2014_001 的 T 会话；E 会话留作评测）
    if hasattr(meta, "columns") and "session" in meta.columns:
        sess = meta["session"].astype(str).str.lower()
        train_mask = sess.str.contains("train").values
        assert train_mask.any(), f"Subject {subject}: 未找到训练会话"
        if train_mask.any():
            X, y = X[train_mask], y[train_mask]

    X = np.nan_to_num(X.astype(np.float64))
    return X, y


def main():
    parser = argparse.ArgumentParser(description="BCI IV-2a CSP+LDA baseline")
    parser.add_argument("--n_classes", type=int, default=4, choices=[2, 4])
    args = parser.parse_args()

    from moabb.datasets import BNCI2014_001
    dataset = BNCI2014_001()
    dataset.download()  # 首次运行自动下载到 MNE_DATA 目录

    results = []
    for s in SUBJECTS:
        X, y = load_subject_data(dataset, s, n_classes=args.n_classes)
        print(f"\n[Subject {s}] X={X.shape} classes={np.unique(y)}")
        acc, kap, y_true, y_pred = evaluate_subject(X, y)
        results.append((s, acc, kap))
        print(f"  acc={acc:.3f}  kappa={kap:.3f}")
        print(confusion_matrix(y_true, y_pred))

    mean_acc = np.mean([r[1] for r in results])
    mean_kap = np.mean([r[2] for r in results])
    print("\n================ 9 被试汇总 ================")
    for s, acc, kap in results:
        print(f"  Subject {s}: acc={acc:.3f}  kappa={kap:.3f}")
    print(f"  平均 acc  = {mean_acc:.3f}")
    print(f"  平均 kappa= {mean_kap:.3f}  (4 类 FBCSP 公开值 ~0.569)")
    print("============================================")


if __name__ == "__main__":
    main()
