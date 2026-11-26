"""
BCI Competition IV-2a 深度学习方案: 纯 PyTorch 实现 (彻底脱离 braindecode)
================================================================================
核心策略:
1. 时间窗裁剪: 0.5s ~ 2.5s (消除 ERP 噪声)
2. 原生 PyTorch 模型: 内置 ShallowNet 与 EEGNet，杜绝版本兼容性报错
3. 跨被试预训练 + 微调: 8个被试混合预训练，目标被试 10 折 CV 微调
4. 数据增强: 同类试次片段重组 (S&R)

运行:
    python dl_baseline.py                # ShallowNet (默认)
    python dl_baseline.py --model eegnet # EEGNet
"""
import argparse
import warnings
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 屏蔽烦人的警告信息
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

# ----------------------------- 1. 全局配置 -----------------------------
N_CLASSES = 4
FILT_LO, FILT_HI = 4.0, 38.0
T_MIN, T_MAX = 0.5, 2.5
FS = 250.0
N_SPLITS = 10
SUBJECTS = range(1, 10)

# 训练超参数 (可根据显卡性能微调)
EPOCHS_PRETRAIN = 60
EPOCHS_FINETUNE = 40
BATCH_SIZE = 32
LR_PRETRAIN = 1e-3
LR_FINETUNE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------- 2. 纯 PyTorch 模型定义 -----------------------------
class ShallowNet(nn.Module):
    """
    经典浅层网络 (对应 ShallowFBCSPNet)
    原理: 时间卷积(滤波) -> 空间卷积(CSP投影) -> 平方(方差) -> 池化 -> 对数
    """
    def __init__(self, n_chans=22, n_times=501, n_classes=4):
        super().__init__()
        # 1. 时间卷积 (等效于带通滤波器)
        self.conv_time = nn.Conv2d(1, 40, (1, 25), bias=True)
        # 2. 空间卷积 (等效于 CSP 空间滤波器)
        self.conv_spat = nn.Conv2d(40, 40, (n_chans, 1), bias=False)
        self.bn = nn.BatchNorm2d(40)
        # 3. 池化层 (平滑)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop = nn.Dropout(0.5)

        # 动态推导全连接层输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            x = self.conv_time(dummy)
            x = self.conv_spat(x)
            x = self.pool(x)
            n_flat = x.numel()

        self.classifier = nn.Linear(n_flat, n_classes)

    def forward(self, x):
        # 输入维度: (Batch, Channels, Time) -> (Batch, 1, Channels, Time)
        x = x.unsqueeze(1)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = torch.square(x)  # 平方激活
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))  # 对数激活
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EEGNet(nn.Module):
    """
    通用 EEGNet 模型
    原理: 深度可分离卷积，极大减少参数量，泛化能力强
    """
    def __init__(self, n_chans=22, n_times=501, n_classes=4, F1=8, D=2, F2=16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)

        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            x = self.pool1(self.bn2(self.depthwise(self.bn1(self.conv1(dummy)))))
            x = self.pool2(self.bn3(self.separable(x)))
            n_flat = x.numel()

        self.classifier = nn.Linear(n_flat, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = torch.nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = torch.nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_model(model_name, n_chans, n_times, n_classes):
    if model_name == "shallow":
        model = ShallowNet(n_chans, n_times, n_classes)
    elif model_name == "eegnet":
        model = EEGNet(n_chans, n_times, n_classes)
    else:
        raise ValueError("模型名称错误")
    return model.to(DEVICE)


# ----------------------------- 3. 数据处理与增强 -----------------------------
def z_score_standardize(X):
    """逐试次、逐通道进行 Z-score 标准化 (替代 braindecode 的标准化)"""
    mean = np.mean(X, axis=2, keepdims=True)
    std = np.std(X, axis=2, keepdims=True)
    return (X - mean) / (std + 1e-6)


def load_subject_data(dataset, subject):
    paradigm = MotorImagery(
        n_classes=4,
        fmin=FILT_LO,
        fmax=FILT_HI,
        tmin=T_MIN,
        tmax=T_MAX,
        events=['left_hand', 'right_hand', 'feet', 'tongue']
    )
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject])

    # 【修复关键】将字符串标签转换为 0, 1, 2, 3 的整数类型
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np.asarray(y, dtype=np.int64)

    if hasattr(meta, "columns") and "session" in meta.columns:
        sess = meta["session"].astype(str).str.lower()
        train_mask = sess.str.contains("train").values
        X, y = X[train_mask], y[train_mask]

    X = np.nan_to_num(X.astype(np.float32))
    X = z_score_standardize(X)  # 自定义标准化
    return X, y


def snr_augment(X, y, n_segments=3, n_aug=3, random_state=42):
    """
    数据增强: 时域切片与重组 (S&R)
    将同一类的不同试次切碎后混合，生成新的"打乱相位但频率分布一致"的有效样本
    """
    rng = np.random.RandomState(random_state)
    X_aug, y_aug = [X], [y]
    n_times = X.shape[-1]
    seg_len = n_times // n_segments

    for cls in np.unique(y):
        X_cls = X[y == cls]
        n_cls = len(X_cls)
        if n_cls < 2:
            continue

        for _ in range(n_aug):
            new_trials = []
            for _ in range(n_cls):
                idx = rng.choice(n_cls, size=n_segments, replace=True)
                segs = []
                for i, seg_i in enumerate(idx):
                    start = i * seg_len
                    end = start + seg_len if i < n_segments - 1 else n_times
                    segs.append(X_cls[seg_i, :, start:end])
                new_trials.append(np.concatenate(segs, axis=-1))
            X_aug.append(np.stack(new_trials))
            y_aug.append(np.full(n_cls, cls))

    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)


# ----------------------------- 4. 训练引擎 -----------------------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        out = model(xb)
        preds.append(out.argmax(dim=1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def fit_model(model, X_train, y_train, X_val, y_val, epochs, lr, verbose=False):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                           torch.tensor(y_val, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 余弦退火学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        y_true, y_pred = evaluate(model, val_loader)
        acc = accuracy_score(y_true, y_pred)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (ep + 1) % 10 == 0:
            print(f"    Epoch {ep+1}/{epochs} | loss={loss:.4f} | val_acc={acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


# ----------------------------- 5. 主流水线 -----------------------------
def run_pretrain_finetune(all_data, target_subject, model_name="shallow"):
    # 1. 混合预训练 (除目标被试外)
    X_pretrain = np.concatenate([all_data[s][0] for s in all_data if s != target_subject])
    y_pretrain = np.concatenate([all_data[s][1] for s in all_data if s != target_subject])

    # 预训练增强扩充
    X_pretrain, y_pretrain = snr_augment(X_pretrain, y_pretrain, n_segments=3, n_aug=2)

    # 拆分验证集 (10%)
    n_val = int(0.1 * len(X_pretrain))
    idx = np.random.RandomState(42).permutation(len(X_pretrain))
    X_val_pre, y_val_pre = X_pretrain[idx[:n_val]], y_pretrain[idx[:n_val]]
    X_tr_pre, y_tr_pre = X_pretrain[idx[n_val:]], y_pretrain[idx[n_val:]]

    print(f"  [Pretrain] 混合 8 个被试预训练, 样本池大小={len(X_tr_pre)}")
    pretrained_model = build_model(model_name, X_pretrain.shape[1], X_pretrain.shape[2], N_CLASSES)

    pretrained_model, pre_acc = fit_model(
        pretrained_model, X_tr_pre, y_tr_pre, X_val_pre, y_val_pre,
        epochs=EPOCHS_PRETRAIN, lr=LR_PRETRAIN, verbose=False
    )

    # 2. 目标被试微调 (10折 CV)
    X_tgt, y_tgt = all_data[target_subject]
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    y_pred_all = np.zeros_like(y_tgt)

    for fold, (tr_idx, va_idx) in enumerate(cv.split(X_tgt, y_tgt)):
        X_tr, y_tr = X_tgt[tr_idx], y_tgt[tr_idx]
        X_va, y_va = X_tgt[va_idx], y_tgt[va_idx]

        # 仅在目标被试的训练折内部执行增强，绝对防止数据泄露
        X_tr, y_tr = snr_augment(X_tr, y_tr, n_segments=3, n_aug=4)

        # 复制预训练权重
        model = build_model(model_name, X_tgt.shape[1], X_tgt.shape[2], N_CLASSES)
        model.load_state_dict(pretrained_model.state_dict())

        # 冻结浅层卷积 (仅针对 ShallowNet)
        if model_name == "shallow":
            for name, param in model.named_parameters():
                if 'conv_time' in name or 'conv_spat' in name:
                    param.requires_grad = False

        model, _ = fit_model(model, X_tr, y_tr, X_va, y_va,
                             epochs=EPOCHS_FINETUNE, lr=LR_FINETUNE)

        _, y_pred = evaluate(model, DataLoader(
            TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long)),
            batch_size=BATCH_SIZE))
        y_pred_all[va_idx] = y_pred

    acc = accuracy_score(y_tgt, y_pred_all)
    kap = cohen_kappa_score(y_tgt, y_pred_all)
    return acc, kap, y_tgt, y_pred_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="shallow", choices=["shallow", "eegnet"])
    args, _ = parser.parse_known_args()

    print(f"=== 深度学习方案开始运行 | 模型: {args.model.upper()} | 设备: {DEVICE} ===")
    dataset = BNCI2014_001()

    # 缓存所有被试数据
    print("\n[第一阶段: 离线加载与缓存所有被试数据...]")
    all_data = {}
    for s in SUBJECTS:
        X, y = load_subject_data(dataset, s)
        all_data[s] = (X, y)
        print(f"  Subject {s}: X={X.shape}")

    results = []
    print("\n[第二阶段: 跨被试预训练 + 10折交叉验证微调 (LOO-CV)]")
    for s in SUBJECTS:
        print(f"\n================ [目标被试 Subject {s}] ================")
        acc, kap, y_true, y_pred = run_pretrain_finetune(all_data, s, model_name=args.model)
        results.append((s, acc, kap))
        print(f"  --> Acc: {acc*100:.2f}% | Kappa: {kap:.3f}")

    mean_acc = np.mean([r[1] for r in results])
    mean_kap = np.mean([r[2] for r in results])

    print("\n" + "=" * 50)
    print(f"         9 被试汇总 (模型: {args.model.upper()})")
    print("=" * 50)
    for s, acc, kap in results:
        print(f"  Subject {s}: Acc = {acc*100:.2f}% | Kappa = {kap:.3f}")
    print("-" * 50)
    print(f"  平均准确率: {mean_acc*100:.2f}%")
    print(f"  平均 Kappa: {mean_kap:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
