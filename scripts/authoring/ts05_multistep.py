#!/usr/bin/env python3
"""ch05 —— 序列到序列预测：单步 vs 多步 / 递归 / 直接多输出 / Seq2Seq + teacher forcing

本章主线（六条）：

1. **单步 vs 多步的评测口径**：`horizon=1` 的 MAE 与 `horizon=24` 的 MAE 不是一件事；
   本章的切分还刻意留出 95 个「跨在切分点上」的窗口不参与训练 —— 那就是 ch06 §6.4 的 `gap`
2. **递归多步（recursive）**：单步模型 + 把预测值喂回窗口，滚动 24 次；误差逐步累积
3. **直接多输出（direct）**：`nn.Linear(hidden, 24)` 一次吐出 24 维，没有累积但也没有反馈
4. **Seq2Seq（encoder-decoder）**：encoder 读完 72 小时 → decoder 逐步解码 24 步
5. **teacher forcing**：`tf=1.0 / 0.5 / 0.0` 三档实测「训练-推理不一致」（exposure bias）
6. **三类方法的 MAE–步长曲线**：连同两条朴素基线（persistence / seasonal naive）一起画

数据：`data/load_curve.csv` 的 **A 台区**（居民型，8760 h），`window=72` / `horizon=24`
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ts05_multistep.py

随机性：全章只在 `train_torch` 内部用随机数，且包在 `fork_rng + manual_seed(SEED)` 里。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch05_multistep"

# =========================================================================== #
# 1. 公共导入 / 数据加载 / 脚手架（两版都给，不挖空）                          #
# =========================================================================== #

IMPORTS = '''from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = Path("data")          # notebook 的 cwd = coding/05_timeseries
CSV = DATA / "load_curve.csv"

# 固定线程数：CPU 上矩阵乘法的规约顺序与线程数有关，锁死它结果才可复现
torch.set_num_threads(4)

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
np.set_printoptions(precision=4, suppress=True)

print("torch", torch.__version__, "| CUDA 可用？", torch.cuda.is_available(),
      "（本章数据小，CPU 足够）")
'''

SETUP = '''# ---- 超参：全部写死在 SETUP 里，后面每个 cell 都只引用它们 ----
WINDOW, HORIZON = 72, 24     # 输入看过去 3 天，预测未来 1 天
SPLIT_RATIO = 0.8            # 按时间切分：前 80% 训练
HIDDEN, HIDDEN_S2S = 64, 32  # MLP 隐层 / Seq2Seq 隐层
EPOCHS, EPOCHS_S2S = 60, 20  # MLP 训练轮数 / Seq2Seq 训练轮数
LR, BATCH = 1e-2, 256        # 学习率 / batch 大小
SEED = 0
TF_LIST = (1.0, 0.5, 0.0)    # 三档 teacher_forcing_ratio

raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)

print("原始行数 %d → 去重排序后 %d" % (len(raw), len(df)))
print("台区", sorted(df["台区编号"].unique()))
print("时间跨度 %s ~ %s" % (df["时间戳"].min(), df["时间戳"].max()))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容


def clean_series(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值（与 ch01 同一口径）。

    为什么要按小时分组：工业台区是「夜间低 + 日间高」的双峰分布，
    全局中位数做阈值会把整段日间峰误判成异常。
    """
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


class MLP1(nn.Module):
    """单步模型：吃 window 长度的窗口，吐 1 个数（下一小时）。递归多步就是反复调用它。"""

    def __init__(self, window, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(window, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x)


class MLP24(nn.Module):
    """直接多输出：最后一层直接是 nn.Linear(hidden, horizon)，一次吐出未来 24 小时。"""

    def __init__(self, window, hidden=64, horizon=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(window, hidden), nn.ReLU(),
                                 nn.Linear(hidden, horizon))

    def forward(self, x):
        return self.net(x)


class Seq2Seq(nn.Module):
    """encoder-decoder：encoder 读完整个输入窗口，把 (h, c) 交给 decoder。

    decoder 每一步吃「上一时刻的值」，输出「下一时刻的值」：
      - 训练时 tf（teacher_forcing_ratio）决定这一步吃**真值**还是吃**自己的预测**
        tf=1.0 全用真值 / tf=0.0 全用预测 / 0<tf<1 按比例随机混合
      - 推理时永远不传 y，于是自动退化成「把预测喂回去」的递归解码
    这就是「训练与推理不一致」（exposure bias）的来源，见 5.4.1。
    """

    def __init__(self, hidden=32, horizon=24):
        super().__init__()
        self.horizon = horizon
        self.enc = nn.LSTM(1, hidden, batch_first=True)
        self.dec = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, y=None, tf=1.0):
        if x.dim() == 2:
            x = x.unsqueeze(-1)                     # (B, T) → (B, T, 1)
        _, (h, c) = self.enc(x)
        inp = x[:, -1, :]                           # 解码的第 0 步吃「窗口最后一个真值」
        outs = []
        for t in range(self.horizon):
            o, (h, c) = self.dec(inp.unsqueeze(1), (h, c))
            p = self.head(o.squeeze(1))
            outs.append(p)
            if y is not None and tf > 0:
                keep = torch.rand(p.size(0), 1) < tf
                inp = torch.where(keep, y[:, t: t + 1], p.detach())
            else:
                inp = p.detach()
        return torch.cat(outs, dim=1)


def build(model_cls, *args, seed=SEED):
    """在固定种子下**实例化**模型。

    「权重初始化」本身就是一层随机数 —— 只在 `train_torch` 里固定种子是不够的：
    模型的初始权重是在 `train_torch` **之前**、用**全局随机源**抽出来的。
    不一起固定，跑第二遍数字就变了，`assert` 全部作废。

    顺带一个好处：三档 teacher forcing 的 Seq2Seq 会拿到**完全相同的初始权重**，
    差异只可能来自 `tf` 本身。
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return model_cls(*args)


def train_torch(model, X, Y, tf=None, epochs=60, lr=1e-2, batch=256, seed=0):
    """通用训练循环（CPU、小数据）。

    tf=None → 直接回归（模型输出 (B, horizon)）；tf=float → Seq2Seq 模式。
    另一处随机性（shuffle、teacher forcing 采样）也包在 fork_rng + manual_seed 里。
    """
    xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt = torch.tensor(np.asarray(Y), dtype=torch.float32)
    if yt.dim() == 1:
        yt = yt[:, None]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.MSELoss()
        n = len(xt)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for k in range(0, n, batch):
                idx = perm[k: k + batch]
                opt.zero_grad()
                out = model(xt[idx]) if tf is None else model(xt[idx], yt[idx], tf)
                loss = lossf(out, yt[idx])
                loss.backward()
                if tf is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
    return model


def mae_per_step(pred, true):
    """逐步 MAE：返回长度 = 预测步数的数组，第 k 个数是「第 k+1 步」的 MAE。"""
    return np.abs(np.asarray(pred, dtype=np.float64)
                  - np.asarray(true, dtype=np.float64)).mean(axis=0)
'''

# =========================================================================== #
# 2. 任务代码块                                                              #
# =========================================================================== #

T1_CODE = '''a_frame = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳").sort_index()
s_a = a_frame["负荷值"].asfreq("h")

# @@todo 清洗 A 台区序列，并取成 numpy 数组
# @@hint 脚手架已有 clean_series(s)，直接调用，再 .values.astype(np.float64)
s_clean = clean_series(s_a)
S = s_clean.values.astype(np.float64)
# @@end

SPLIT = int(SPLIT_RATIO * len(S))

# @@todo 缩放参数（均值 / 标准差）—— **只用训练段拟合**
# @@hint MU = float(S[:SPLIT].mean())；SD = float(S[:SPLIT].std())
# @@hint 用全量会悄悄把测试段的信息带进模型（ch06 §6.3 量化后果）
MU = float(S[:SPLIT].mean())
SD = float(S[:SPLIT].std())
# @@end

Z = (S - MU) / SD

print("整点 %d | 清洗前缺失 %d | 清洗后缺失 %d" % (len(S), int(s_a.isna().sum()), int(np.isnan(S).sum())))
print("训练段 MU %.10f / SD %.10f | 切分点 %d" % (MU, SD, SPLIT))
print("原始序列 min %.2f max %.2f（含哨兵 9999 与负值）" % (s_a.min(), s_a.max()))

# ---- 验收 ----
assert len(S) == 8760 and int(np.isnan(S).sum()) == 0
assert int(s_a.isna().sum()) == 70
assert abs(MU - 454.561801) < 1e-5, f"训练段均值实际是 {MU}"
assert abs(SD - 72.623128) < 1e-5, f"训练段标准差实际是 {SD}"
'''

T2_CODE = '''N = len(S) - WINDOW - HORIZON + 1
STARTS = np.arange(N)

# @@todo 按时间切分：训练窗口的**目标**要整体落在切分点之前，测试窗口的**输入**从切分点之后开始
# @@hint 训练：STARTS + WINDOW + HORIZON <= SPLIT；测试：STARTS >= SPLIT
# @@hint 夹在中间、跨过切分点的那一批窗口两边都不要 —— 这就是 ch06 §6.4 的 gap
tr_idx = STARTS[STARTS + WINDOW + HORIZON <= SPLIT]
te_idx = STARTS[STARTS >= SPLIT]
# @@end

# @@todo 把标准化后的序列窗化成 (N, WINDOW) 的输入和 (N, HORIZON) 的目标
# @@hint 列表推导 + np.stack；第 i 个样本：X[i] = Z[i : i+WINDOW]
# @@hint 目标是紧接窗口的 24 个点：Z[i+WINDOW : i+WINDOW+HORIZON]
x_all = np.stack([Z[i: i + WINDOW] for i in STARTS]).astype(np.float32)
y_all = np.stack([Z[i + WINDOW: i + WINDOW + HORIZON] for i in STARTS]).astype(np.float32)
# @@end

Xtr, Ytr = x_all[tr_idx], y_all[tr_idx]
Xte, Yte = x_all[te_idx], y_all[te_idx]
Xte_t = torch.tensor(Xte)
Yte_raw = np.stack([S[i + WINDOW: i + WINDOW + HORIZON] for i in te_idx])

print("样本总数 %d | 训练 %d | 测试 %d | 夹在切分点上的 gap 窗口 %d 个"
      % (N, len(tr_idx), len(te_idx), N - len(tr_idx) - len(te_idx)))
print("Xtr %s | Ytr %s | Xte %s | Yte_raw %s"
      % (Xtr.shape, Ytr.shape, Xte_t.shape, Yte_raw.shape))
print("测试段第一个预测起点 = 第 %d 个整点（切分点 %d）" % (int(te_idx[0]) + WINDOW, SPLIT))

# ---- 验收 ----
assert N == 8665
assert len(tr_idx) == 6913 and len(te_idx) == 1657
assert N - len(tr_idx) - len(te_idx) == 95, "跨在切分点上的窗口有 95 个"
assert Xtr.shape == (6913, 72) and Ytr.shape == (6913, 24) and Xte_t.shape == (1657, 72)
assert int(te_idx[0]) == SPLIT
'''

T3_CODE = '''# @@todo 建一个「只预测下一小时」的单步模型（权重初始化也要在固定种子下）
# @@hint MLP1(WINDOW, HIDDEN)：72 → 64 → 1；用 build(MLP1, WINDOW, HIDDEN) 实例化
model_1 = build(MLP1, WINDOW, HIDDEN)
# @@end

# @@todo 用训练窗口训练它 —— 目标只取 24 维里的第 1 维
# @@hint train_torch(模型, Xtr, Ytr[:, :1], epochs=EPOCHS, lr=LR, seed=SEED) 返回训练好的模型
model_1 = train_torch(model_1, Xtr, Ytr[:, :1], epochs=EPOCHS, lr=LR, seed=SEED)
# @@end

model_1.eval()
with torch.no_grad():
    step1 = model_1(Xte_t)[:, 0].numpy().astype(np.float64) * SD + MU
mae_1 = float(np.abs(step1 - Yte_raw[:, 0]).mean())
print("单步（horizon=1）测试 MAE = %.6f kW" % mae_1)

# ---- 验收 ----
assert abs(mae_1 - 14.544809) < 1e-5, f"单步 MAE 实际是 {mae_1}"
assert float(np.abs(step1 - Yte_raw[:, 0]).max()) < 120
'''

T4_CODE = '''model_1.eval()
rec = np.zeros((len(te_idx), HORIZON), dtype=np.float32)
win = Xte.copy()                       # 每个测试样本的初始输入窗口

with torch.no_grad():
    for t in range(HORIZON):
        # @@todo 用当前 72 小时窗口预测下一时刻，再把预测值接到窗口尾部滚动一步
        # @@hint 输入 torch.tensor(win) → 模型输出 (B, 1)，取 [:, 0] 得到 (B,)
        # @@hint 滚动：np.concatenate([win[:, 1:], p[:, None]], axis=1) —— 丢掉最老的点
        p = model_1(torch.tensor(win))[:, 0].numpy()
        win = np.concatenate([win[:, 1:], p[:, None]], axis=1)
        # @@end
        rec[:, t] = p

rec_raw = rec.astype(np.float64) * SD + MU

# @@todo 算逐步 MAE（长度 24 的数组）
# @@hint 脚手架已有 mae_per_step(pred, true)
mae_rec = mae_per_step(rec_raw, Yte_raw)
# @@end

print("递归多步 24 步平均 MAE = %.6f kW" % mae_rec.mean())
print("  第 1 步 %.6f → 第 12 步 %.6f → 第 24 步 %.6f"
      % (mae_rec[0], mae_rec[11], mae_rec[23]))
print("  逐步 MAE:", np.round(mae_rec, 4).tolist())

# ---- 验收 ----
assert abs(mae_rec[0] - mae_1) < 1e-9, "递归的第 1 步就是单步模型本身，必须逐位相等"
assert abs(mae_rec.mean() - 16.630607) < 1e-4, f"递归 24 步 MAE 实际是 {mae_rec.mean()}"
assert mae_rec[23] > mae_rec[0] > 0
'''

T5_CODE = '''# @@todo 建一个「一次吐出 24 维」的多输出模型
# @@hint MLP24(WINDOW, HIDDEN, HORIZON)：最后一层是 nn.Linear(hidden, HORIZON)
# @@hint 用 build(MLP24, WINDOW, HIDDEN, HORIZON) 实例化
model_24 = build(MLP24, WINDOW, HIDDEN, HORIZON)
# @@end

# @@todo 在同一批训练窗口上训练它 —— 这次目标直接用 Ytr（形状 (N, 24)），不要再切片
# @@hint train_torch(model_24, Xtr, Ytr, epochs=EPOCHS, lr=LR, seed=SEED)
model_24 = train_torch(model_24, Xtr, Ytr, epochs=EPOCHS, lr=LR, seed=SEED)
# @@end

model_24.eval()
with torch.no_grad():
    dir_raw = model_24(Xte_t).numpy().astype(np.float64) * SD + MU
mae_dir = mae_per_step(dir_raw, Yte_raw)
print("直接多输出 24 步平均 MAE = %.6f kW" % mae_dir.mean())
print("  第 1 步 %.6f → 第 24 步 %.6f" % (mae_dir[0], mae_dir[23]))
print("  逐步 MAE:", np.round(mae_dir, 4).tolist())

# ---- 验收 ----
assert abs(mae_dir.mean() - 16.696673) < 1e-4, f"直接多输出 MAE 实际是 {mae_dir.mean()}"
assert mae_dir[23] > mae_dir[0]
'''

T6_CODE = '''# @@todo 训练一个「全 teacher forcing」的 Seq2Seq（tf=1.0）
# @@hint 用 build(Seq2Seq, HIDDEN_S2S, HORIZON) 实例化
# @@hint train_torch(模型, Xtr, Ytr, tf=1.0, epochs=EPOCHS_S2S, lr=LR, seed=SEED)
model_tf1 = train_torch(build(Seq2Seq, HIDDEN_S2S, HORIZON), Xtr, Ytr,
                        tf=1.0, epochs=EPOCHS_S2S, lr=LR, seed=SEED)
# @@end

model_tf1.eval()
with torch.no_grad():
    tf1_raw = model_tf1(Xte_t).numpy().astype(np.float64) * SD + MU
mae_tf1 = mae_per_step(tf1_raw, Yte_raw)
print("Seq2Seq（全 teacher forcing，tf=1.0）24 步平均 MAE = %.6f kW" % mae_tf1.mean())
print("  第 1 步 %.6f → 第 24 步 %.6f" % (mae_tf1[0], mae_tf1[23]))

# ---- 验收 ----
assert abs(mae_tf1.mean() - 21.120460) < 1e-4, f"tf=1.0 的 MAE 实际是 {mae_tf1.mean()}"
assert tf1_raw.shape == (1657, 24)
'''

T7_CODE = '''tf_curves = {1.0: mae_tf1}          # tf=1.0 的那条曲线上一格已经算好了，直接复用
for tf in (0.5, 0.0):
    # @@todo 分别用「混合 TF」和「无 teacher forcing」各训练一个 Seq2Seq，算出各自的逐步 MAE
    # @@hint train_torch(build(Seq2Seq, HIDDEN_S2S, HORIZON), Xtr, Ytr, tf=tf,
    # @@hint            epochs=EPOCHS_S2S, lr=LR, seed=SEED)
    # @@hint 推理时不要传 y：m(Xte_t)；再 * SD + MU 还原成 kW
    m = train_torch(build(Seq2Seq, HIDDEN_S2S, HORIZON), Xtr, Ytr,
                    tf=tf, epochs=EPOCHS_S2S, lr=LR, seed=SEED)
    m.eval()
    pr = m(Xte_t).detach().numpy().astype(np.float64) * SD + MU
    tf_curves[tf] = mae_per_step(pr, Yte_raw)
    # @@end

print("teacher_forcing_ratio 对照（推理时一律不喂真值、只把预测喂回去）：")
for tf in TF_LIST:
    print("  tf=%.1f → 24 步平均 MAE %.6f kW（第 1 步 %.6f，第 24 步 %.6f）"
          % (tf, tf_curves[tf].mean(), tf_curves[tf][0], tf_curves[tf][23]))

# ---- 验收 ----
assert abs(tf_curves[0.5].mean() - 20.583487) < 1e-4
assert abs(tf_curves[0.0].mean() - 18.237022) < 1e-4
assert tf_curves[1.0].mean() > tf_curves[0.5].mean() > tf_curves[0.0].mean(), "tf 越大推理越差"
'''

T8_CODE = '''# @@todo 两条朴素基线：把「窗口最后一个值」重复 24 次 / 把「窗口最后 24 个值」原样搬过去
# @@hint 第 i 个测试样本的最后一个窗口点是 S[i + WINDOW - 1]
# @@hint np.repeat(列向量, HORIZON, axis=1)；seasonal 用窗口最后 HORIZON 个点
last_te = S[te_idx + WINDOW - 1]
pred_persist = np.repeat(last_te[:, None], HORIZON, axis=1)
prev24 = np.stack([S[i + WINDOW - HORIZON: i + WINDOW] for i in te_idx])
# @@end

mae_persist = mae_per_step(pred_persist, Yte_raw)
mae_season = mae_per_step(prev24, Yte_raw)
print("persistence（老值延续）24 步平均 MAE = %.6f kW" % mae_persist.mean())
print("seasonal naive（昨日同时刻）24 步平均 MAE = %.6f kW" % mae_season.mean())
print("seasonal 的逐步 MAE 几乎是平的:", np.round(mae_season, 4).tolist())

# ---- 验收 ----
assert abs(mae_persist.mean() - 60.671651) < 1e-4, f"persistence 实际是 {mae_persist.mean()}"
assert abs(mae_season.mean() - 21.741549) < 1e-4, f"seasonal 实际是 {mae_season.mean()}"
assert mae_season.max() - mae_season.min() < 0.2, "seasonal 的误差不随步长累积"
'''

T9_CODE = '''steps = np.arange(1, HORIZON + 1)

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(steps, mae_persist, ":", color="tab:gray", label="persistence（老值延续）")
ax.plot(steps, mae_season, "--", color="tab:green", label="seasonal naive（昨日同时刻）")
ax.plot(steps, mae_rec, "-o", ms=3.5, label="递归多步 recursive")
ax.plot(steps, mae_dir, "-s", ms=3.5, label="直接多输出 direct")
ax.plot(steps, tf_curves[0.5], "-^", ms=3.5, label="Seq2Seq（混合 TF=0.5）")
ax.axhline(mae_1, color="crimson", lw=0.9, ls="-.", label="单步 MAE（horizon=1）")
ax.set_xlabel("预测步长（小时）")
ax.set_ylabel("MAE（kW）")
ax.set_title("多步预测：MAE 随步长的变化（A 台区，window=72 / horizon=24）")
ax.set_xticks(np.arange(0, 25, 4))
ax.legend(loc="lower right", fontsize=9)

# @@todo 把 6 条曲线的「24 步平均 MAE」汇总成一个 dict，并按从好到坏打印
# @@hint 每条曲线都是长度 24 的数组，直接 .mean() 即可
# @@hint 键用 "persist" / "seasonal" / "recursive" / "direct" / "tf1.0" / "tf0.5" / "tf0.0"
summary = {
    "persist": float(mae_persist.mean()),
    "seasonal": float(mae_season.mean()),
    "recursive": float(mae_rec.mean()),
    "direct": float(mae_dir.mean()),
    "tf1.0": float(tf_curves[1.0].mean()),
    "tf0.5": float(tf_curves[0.5].mean()),
    "tf0.0": float(tf_curves[0.0].mean()),
}
# @@end

# @@todo 把 summary 按 MAE 从好到坏排序（排完是 [(方法名, MAE), ...]）
# @@hint sorted(summary.items(), key=lambda kv: kv[1])
ranking = sorted(summary.items(), key=lambda kv: kv[1])
# @@end

print("24 步平均 MAE 排行榜（越小越好）：")
for k, v in ranking:
    print("  %-10s %.6f kW" % (k, v))
plt.tight_layout()

# ---- 验收 ----
assert summary["recursive"] < summary["direct"] < summary["seasonal"]
assert summary["tf1.0"] > summary["tf0.5"] > summary["tf0.0"], "exposure bias 的单调性"
assert summary["seasonal"] < summary["persist"] / 2, "重复单个老值的误差是当日周期的两倍多"
'''

T10_CODE = '''# @@todo 量化两件事：① 递归相对单步的误差放大倍数 ② 第 24 步上「直接 − 递归」的差
# @@hint 放大倍数 = mae_rec.mean() / mae_1
# @@hint 第 24 步的差 = mae_dir[23] - mae_rec[23]
amp = float(mae_rec.mean() / mae_1)
gap24 = float(mae_dir[23] - mae_rec[23])
# @@end

# @@todo 挑出「递归与直接多输出差得最多」的那个测试样本
# @@hint 先逐样本算 24 步平均绝对误差：np.abs(rec_raw - Yte_raw).mean(axis=1)
# @@hint 再取两者之差的 argmax，得到样本下标 k
err_rec = np.abs(rec_raw - Yte_raw).mean(axis=1)
err_dir = np.abs(dir_raw - Yte_raw).mean(axis=1)
k = int(np.argmax(np.abs(err_dir - err_rec)))
# @@end

hor = np.arange(1, HORIZON + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
axes[0].plot(hor, Yte_raw[k], "-o", ms=3, color="black", label="真实")
axes[0].plot(hor, rec_raw[k], "--s", ms=3, label="递归")
axes[0].plot(hor, dir_raw[k], "--^", ms=3, label="直接多输出")
axes[0].set_xlabel("预测步长（小时）")
axes[0].set_ylabel("负荷 kW")
axes[0].set_title("第 %d 个测试样本的 24 小时预测" % k)
axes[0].legend(fontsize=9)
axes[1].plot(hor, np.abs(rec_raw[k] - Yte_raw[k]), "--s", ms=3, label="递归")
axes[1].plot(hor, np.abs(dir_raw[k] - Yte_raw[k]), "--^", ms=3, label="直接多输出")
axes[1].set_xlabel("预测步长（小时）")
axes[1].set_ylabel("绝对误差 kW")
axes[1].set_title("该样本的逐点绝对误差")
axes[1].legend(fontsize=9)
plt.tight_layout()

print("递归 24 步平均 MAE / 单步 MAE = %.4f（误差放大 %.1f%%）"
      % (amp, (amp - 1) * 100))
print("第 24 步：递归 %.6f | 直接 %.6f | 差 %+.6f" % (mae_rec[23], mae_dir[23], gap24))
print("递归逐步增幅：第 1 步 %.4f → 第 24 步 %.4f（%+.1f%%）"
      % (mae_rec[0], mae_rec[23], (mae_rec[23] / mae_rec[0] - 1) * 100))

# ---- 验收 ----
assert abs(amp - 1.1434) < 1e-4, f"放大倍数实际是 {amp}"
assert abs(gap24 - 0.271966) < 1e-5, f"第 24 步的差实际是 {gap24}"
assert amp > 1.0, "递归一定比单步差 —— 差别就是误差累积的代价"
'''

# =========================================================================== #
# 3. 讲解版（LESSON）                                                        #
# =========================================================================== #

LESSON = [
    md(
        """
# ch05 序列到序列预测（讲解版）

> **本节考点**：负荷 / 新能源功率预测里，「未来 24 小时」这类**多步预测**怎么做。
> 竞赛评分点对应**模型训练 40%** 与**模型性能评估 10%** —— 多步预测的口径选错，
> 报出来的 MAE 就是假的。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 模型训练 | **技能操作 40%** | 递归 / 直接多输出 / Seq2Seq 三种多步范式 + teacher forcing |
| 模型性能评估 | 技能操作 10% | 「单步 MAE」与「24 步 MAE」的口径差异、MAE–步长曲线 |
| 新能源功率 / 负荷预测 | 行业赛题 | 日前 24 小时负荷预测就是本章的标准场景 |

**与相邻章节的分工**：

| 章节 | 管什么 | 本章的边界 |
|---|---|---|
| `ch01` | 特征与切分（把脏 CSV 变成监督矩阵） | 本章直接用它的 `clean_series` 口径与「只按时间切分」纪律 |
| `ch03` / `ch04` | RNN / LSTM 的**内部机制**（BPTT、门控） | 本章把 LSTM 当黑盒用，只关心「怎么用它做多步」 |
| **`ch05`（本章）** | **多步的三种范式 + 训练-推理不一致** | — |
| `ch06` | 滑窗 Dataset / 防泄漏缩放 / `TimeSeriesSplit` | 本章切分处的 95 个 `gap` 窗口，ch06 会正式命名 |
| `ch07` | 误差指标（MAPE 零点陷阱）、可视化规范 | 本章只算 MAE，指标本身不展开 |

## 学习目标

1. 说清「单步 MAE」与「24 步 MAE」为什么**不能互相冒充**，并会自己构造合法的切分
2. 手写**递归多步**：单步模型 + 滚动喂回，并解释误差为什么逐步放大
3. 用**直接多输出**一次预测 24 步，并说清它「没有累积」的代价是什么
4. 搭出 encoder-decoder 版 **Seq2Seq**，并用 `teacher_forcing_ratio` 做三档对照
5. **当场解释 exposure bias**：为什么「全 teacher forcing」在推理时反而最差
6. 画出三类方法 + 两条朴素基线的 **MAE–步长曲线**，并据此说出该选哪种

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `nn.Linear(in, out)` | `out=24` | `Module` | 直接多输出的实现就是它 —— 最后一个维度的意义由你决定 |
| `nn.LSTM(input, hidden, batch_first=True)` | `batch_first` | `Module` | 返回 `(output, (h, c))`；`batch_first=True` 时形状是 `(B, T, H)` |
| `torch.random.fork_rng()` | — | 上下文管理器 | **局部**换随机种子，出了块自动还原，不污染后续 cell |
| `torch.manual_seed(k)` | — | `Generator` | 固定全局随机源；`fork_rng` + 它 = 可复现的最小组合 |
| `torch.randperm(n)` | — | `Tensor` | 随机排列，DataLoader 洗牌与手写 shuffle 都靠它 |
| `nn.MSELoss()` | `reduction` | `Module` | 训练用 MSE；**评估别用它**，MAE 更可解释 |
| `nn.utils.clip_grad_norm_(params, x)` | 阈值 | `float` | RNN 类模型必须做梯度裁剪，否则一个 batch 就能把权重打飞 |
| `torch.optim.lr_scheduler.CosineAnnealingLR` | `T_max` | `LRScheduler` | 余弦退火；RLR 从 `lr` 平滑降到 0，收敛更稳 |
| `torch.no_grad()` | — | 上下文管理器 | 推理时关掉自动求导：省内存、省时间 |
| `Tensor.detach()` | — | `Tensor` | 从计算图里摘出来（`no_grad` 块的替代写法） |
| `np.repeat(a, n, axis=1)` | `axis` | `ndarray` | 朴素基线「老值延续」一行就能写出来 |

## 本章的「真值锚点」

为了让 `assert` 与讲解里的数字完全一致，全章固定：

| 常量 | 值 | 含义 |
|---|---|---|
| `WINDOW` | **72** | 输入窗口 = 过去 3 天 |
| `HORIZON` | **24** | 预测未来 1 天 |
| `SPLIT_RATIO` | **0.8** | 前 80% 训练 |
| `EPOCHS` / `EPOCHS_S2S` | **60 / 20** | MLP / Seq2Seq 训练轮数 |
| `LR` / `BATCH` | **1e-2 / 256** | 学习率 / batch |
| `SEED` | **0** | 所有随机性都锚在它上面 |
| `TF_LIST` | **(1.0, 0.5, 0.0)** | teacher forcing 三档 |

> **本章不追求「训出最好的模型」。** 三种范式的代码规模刻意保持同量级，
> 结论才能归因到**范式差异**而不是「谁的调参更狠」。所有绝对值都只是这台数据上的参照。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md(
        """
## 5.1 单步 vs 多步：先统一「评测口径」

一份负荷数据的建模任务，最常见的两种问法：

| 问法 | 目标形状 | 典型说法 |
|---|---|---|
| **单步**（one-step-ahead） | `(N, 1)` | 「预测下一小时」`ŷ(t+1)` |
| **多步**（multi-step / 日前） | `(N, 24)` | 「预测未来 24 小时」`ŷ(t+1) … ŷ(t+24)` |

**这两种 MAE 不可比**：多步要把未来 24 个点**全算对**，而其中第 12 步离输入窗口
已经隔了 12 小时，日周期的相位早就转过去了。拿单步的 MAE 去汇报 24 步的活，
是时序比赛里最容易被一眼看穿的虚报。

本章的切分有**三块**，不是两块：

```
第 0 ─────────────────── 7008 ─────────── 8665 ──────── 8740
│      训练窗口（6913 个）  │  gap  │   测试窗口（1657 个）
│                          │  95 个 │
└── 目标整体落在切分点之前 ──┘        └── 输入从切分点之后开始 ──┘
```

中间那 **95 个**窗口「跨在切分点上」：它们的输入还在训练段，目标却已经伸进测试段。
如果把它们塞进训练集，模型就**见过测试期的目标值** —— 这属于白送的泄漏。
所以两块都不要，直接丢掉。（`ch06` §6.4 会把这个「丢掉」正式命名成
`TimeSeriesSplit(gap=...)` 的 `gap` 参数。）

**为什么要留 gap**：滑窗样本是**重叠**的。窗口长 96（72 输入 + 24 目标），
所以只要起点落在切分点前 96 个位置内，样本就会跨界。丢掉 95 个，换一个干净的切分。

### 5.1.1 难点深挖：`horizon=1` 的 MAE 拿去汇报 `horizon=24`，差多少？

**为什么难**：单步和 24 步的目标都是「一串负荷值」，形状不同但量纲相同，
于是 `MAE=14.4` 这个数字**看起来完全可以贴在 24 步的报告里**。它不会被任何代码拦住。

**错误示范**：

```python
model = MLP1(WINDOW, HIDDEN)
train_torch(model, Xtr, Ytr[:, :1], epochs=60, lr=1e-2, seed=0)
mae = ...                                     # 14.544809
report["未来 24 小时预测 MAE"] = mae           # ← 张冠李戴
```

**正误对照**（本章实测，A 台区，均为 kW）：

| 口径 | MAE |
|---|---|
| 单步 `horizon=1` | **14.544809** |
| 递归多步 24 步平均 | **16.630607**（+14.3%） |
| 递归多步第 24 步 | **18.099191**（+24.4%） |
| 直接多输出 24 步平均 | **16.696673**（+14.8%） |

所以「单步 14.54」与「日前 16.63」是两个数字，中间差着一整个**日内相位**。

**判定规则**：

> **汇报多步误差，分母必须是「步数」。**
> - 报「整体指标」→ 把 `(N, 24)` 的误差矩阵**摊平**再算 MAE（本章做法）
> - 报「第 k 步指标」→ 沿样本维取均值，得到**长度为 24 的曲线**（本章 5.5 的图）
> - 只报单个数就宣称「24 小时预测精度」→ **要么口径是摊平的，要么就是虚报**
>
> 还有一条同样重要的：**切分处必须留 gap**。95 个跨界窗口不丢掉，
> 训练集里就混进了测试期的目标值，指标会虚高而你看不出来。
"""
    ),
    code(T1_CODE),
    code(T2_CODE),
    md(
        """
### 5.1.2 两条朴素基线：先知道「不学」能拿到多少

在动手训模型之前，**必须先有两条傻瓜基线**，否则你不知道模型到底有没有干活：

```python
pred_persist = np.repeat(last_te[:, None], HORIZON, axis=1)   # 老值延续：把最后一个值抄 24 遍
prev24 = np.stack([S[i+WINDOW-HORIZON : i+WINDOW] for i in te_idx])   # 昨日同时刻：把窗口最后 24 个点原样搬过去
```

| 基线 | 24 步平均 MAE | 逐步误差的形状 |
|---|---|---|
| persistence（老值延续） | **60.671651** | **U 形**：第 1 步 30.1、第 18 步 81.3、第 24 步 21.8 |
| seasonal naive（昨日同时刻） | **21.741549** | **一条平线**（21.68 ~ 21.81） |

两条基线摆在一起，能一眼看出**日周期有多强**：

- 「老值延续」把**同一个数**抄 24 遍。早上 6 点抄了凌晨 5 点的低谷值去预测中午的高峰，
  误差自然炸到 80 kW；而抄到第 24 步时，又恰好抄在第二天的同一相位上 —— 所以 U 形两头低。
- 「昨日同时刻」**每个步长都用了正确的相位**，所以误差不随步长累积，
  曲线几乎是平的。**21.74 这个数才是本章真正的及格线** —— 任何模型赢不过它，
  就说明它连「昨天同一时刻」这个信息都没用上。
- 反过来说，本章的 Seq2Seq（最差的一档 21.12）也只是**刚刚赢过这条基线**，
  而递归 / 直接多输出赢了 23% —— **模型之间的差别，比"有没有模型"的差别小得多**。
"""
    ),
    code(T8_CODE),
    md(
        """
## 5.2 递归多步（recursive）：把预测喂回去

递归的思路最朴素：**我只训一个单步模型，然后把它自己当输入用**。

```python
p = model_1(torch.tensor(win))[:, 0].numpy()          # 用窗口预测 t+1
win = np.concatenate([win[:, 1:], p[:, None]], axis=1)  # 窗口左移一格，预测值接在右端
```

滚动 24 次，就得到未来 24 小时。三个必须注意的点：

1. **窗口必须真的左移**。`win[:, 1:]` 丢掉最老的一个点 —— 不丢的话窗口会越来越长，
   模型第二句就报形状错。这不是"顺手写的"，是滚动预测的定义。
2. **第 1 步就是单步模型本身**。所以 `mae_rec[0]` 必须与单步 MAE **逐位相等**，
   这是本章最硬的一个 `assert`：它同时验证了「递归没有在第一步引入额外的偏差」。
3. **推理阶段不需要随机性**：`fork_rng` 只包在训练里，滚动预测是纯前向。

### 5.2.1 难点深挖：误差为什么会「滚雪球」

**为什么难**：递归在代码上**完全正确**，报不出任何错。「误差累积」是个统计现象，
不是 bug —— 你只能从数字上认出来。

**机制**（一句话）：**第 t 步的输入里，有 t 个位置是自己预测的。**

```
第 1 步： 输入 = [a1 a2 ... a72]              ← 72 个全是真值
第 2 步： 输入 = [a2 ... a72, ŷ1]             ← 1 个预测值
第 3 步： 输入 = [a3 ... a72, ŷ1, ŷ2]         ← 2 个预测值
   ⋮
第 24 步：输入 = [a24 ... a72, ŷ1 ... ŷ23]    ← 23 个预测值
```

模型训练时**从没见过「输入里混着预测值」**这种数据分布 —— 这是**分布漂移**，
而不是"预测值不准确"这么简单。于是误差不是线性叠加，而是**复合**的。

**实测**（A 台区，逐步 MAE，kW）：

| 步长 | 1 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| 递归 | **14.545** | 15.347 | 16.098 | 16.737 | 17.327 | 17.689 | **18.099** |
| 相对第 1 步 | — | +5.5% | +10.7% | +15.1% | +19.1% | +21.6% | **+24.4%** |

`mae_rec.mean() / mae_1 = 1.1434` —— 递归的 24 步平均比单步差 **14.3%**。

值得留意的是：**这条曲线是单调递增且没有拐点的**。日周期把它"救"了一部分 ——
第 24 步恰好走到第二天的同一相位，所以增幅只有 24.4% 而不是爆炸；
但如果在没有强周期的序列上做，这条曲线会陡得多（试试 ch01 的残差那条序列）。

**判定规则**：

> **递归的适用范围：序列有强自相关 / 强周期，且 horizon 不长（≤ 24 步量级）。**
> - 看到 `mae_rec[k]` 单调递增、且 `mae_rec[0] == 单步 MAE` → 递归实现是对的
> - 看到 `mae_rec[0] != 单步 MAE` → 递归实现有 bug（通常是窗口没滚动，或没还原尺度）
> - 看到曲线在某个步长后**加速上翘** → 说明序列记忆太短，该换直接多输出或 Seq2Seq
"""
    ),
    code(T3_CODE),
    code(T4_CODE),
    md(
        """
## 5.3 直接多输出（direct）：一次吐出 24 维

递归的毛病来自「预测值进了输入」。那就干脆**让模型一次把 24 步全吐出来**：

```python
self.net = nn.Sequential(
    nn.Linear(window, hidden), nn.ReLU(),
    nn.Linear(hidden, horizon),        # ← 只改这一行：1 → 24
)
```

训练时目标从 `Ytr[:, :1]` 换成完整的 `Ytr`（形状 `(N, 24)`），
损失函数一个字都不用改 —— `nn.MSELoss` 会自动对所有 24 个维度求均值。

**三个"没有"**：

- **没有滚动**：24 个输出是**并行**产生的，互相之间不构成输入
- **没有累积**：第 24 步与第 1 步共享同一个输入窗口，误差不会传递
- **没有顺序**：`nn.Sequential` 根本不知道"24"是个时间维度，它只当 24 个回归目标

实测：直接多输出 24 步平均 **16.696673**，第 1 步 14.811790、第 24 步 18.371157。

### 5.3.1 难点深挖：直接多输出的「无累积」不是免费的

**为什么难**：教材一般只讲「直接多输出避免了误差累积」这个优点，
于是学生会以为它**严格优于**递归。实测恰恰相反 —— 本章数据上它**略差于**递归
（16.697 vs 16.631）。

**代价在三个地方**：

**① 参数预算被摊薄**。递归只有一个输出头（`Linear(64, 1)`），
全部容量用于"预测下一小时"这件事；直接多输出有 24 个输出头，
在同样的训练量下**每个步长分到的监督信号更少**。

**② 第 1 步就已经吃亏了**。注意逐步曲线：直接方法的第 1 步 14.811790
就已经**比递归的 14.544809 差 0.267 kW** —— 两者用的是同一个网络结构、同一批数据、
同一个 lr 和轮数，唯一区别就是输出头是 1 维还是 24 维。

**③ 无法利用"中间步已经预测对了"这件事**。递归在第 6 步时，输入里已经含有
自己对第 1~5 步的估计（哪怕有误差，也包含了"当前处于周期的哪个相位"这种信息）；
直接多输出是**一次盲赌 24 个数**。

**实测对照**（kW）：

| 指标 | 递归 | 直接多输出 | 差 |
|---|---|---|---|
| 24 步平均 MAE | **16.630607** | 16.696673 | **+0.066066** |
| 第 1 步 MAE | **14.544809** | 14.811790 | +0.267 |
| 第 24 步 MAE | **18.099191** | 18.371157 | **+0.271966** |

**判定规则**：

> **选择依据不是"谁更先进"，而是三条一起看**：
> | 你的情况 | 用哪个 |
> |---|---|
> | horizon 短（≤ 12）、要最少代码、能接受批量并行出数 | **直接多输出**（一次前向，推理最快） |
> | horizon 中等（≤ 24）、序列自相关强、想要单一模型反复复用 | **递归**（本章数据上的赢家） |
> | horizon 长 / 变长 / 多变量、要给未来已知协变量留接口 | **Seq2Seq**（见 5.4） |
>
> 反过来的推论同样重要：**"避免误差累积"这句优点，只有在 horizon 很长、
> 或序列自相关很弱时才真兑现。**
> 本章递归的累积量（+14.3%）摊到 24 步上每步只多 0.15 kW，
> 反而比直接多输出「摊薄参数」的损失（+0.066 kW 整体、第 1 步就 +0.27 kW）更划算。
> **这两种损失的量级，只有把逐步曲线画出来才看得见。**
"""
    ),
    code(T5_CODE),
    md(
        """
## 5.4 Seq2Seq（encoder-decoder）：把「预测」变成「解码」

Seq2Seq 把任务拆成两段：

```
encoder：读完整 72 小时   [x1 … x72]  ──LSTM──▶  (h, c)
decoder：从 (h, c) 出发，一步一个字地吐
         step 0: 输入 = x72      → 输出 ŷ1
         step 1: 输入 = 上一时刻的值 → 输出 ŷ2
         …
         step 23:                    → 输出 ŷ24
```

关键的那一行是 **decoder 的输入从哪来**：

```python
if y is not None and tf > 0:
    keep = torch.rand(p.size(0), 1) < tf
    inp = torch.where(keep, y[:, t:t+1], p.detach())   # tf 决定吃真值还是吃自己的预测
else:
    inp = p.detach()                                    # 推理：永远吃自己的预测
```

`tf` 就是 **teacher_forcing_ratio（教师强制比例）**：

| `tf` | 训练时 decoder 吃什么 | 直觉 |
|---|---|---|
| **1.0** | 每一步都吃**真值** | 「老师手把手」—— 训练最快，但学生从没练过自己走路 |
| **0.5** | 一半概率吃真值、一半吃自己的预测 | 折中 |
| **0.0** | 全程吃**自己的预测** | 「放养」—— 训练-推理一致，但早期梯度噪声大 |

### 5.4.1 难点深挖：全 teacher forcing 的模型，推理时最差（exposure bias）

**为什么难**：`tf=1.0` 的训练损失**最低、收敛最快**，看训练曲线的人会以为它最好。
但评估是在**推理模式**下做的，那里没有任何真值可喂 —— **两个阶段的输入分布不一致**。
这个现象叫 **exposure bias（曝光偏差）**：模型只"曝光"在真值输入下，
从没学过"输入里带着自己的误差"时该怎么办。

**错误示范 / 正误对照**（同一份数据、同一个网络、**连初始权重都一样**、只改 `tf`）：

```python
for tf in (1.0, 0.5, 0.0):
    m = train_torch(build(Seq2Seq(32, 24)), Xtr, Ytr, tf=tf, epochs=20, lr=1e-2, seed=0)
    mae = mae_per_step(m(Xte_t).numpy() * SD + MU, Yte_raw)
```

**实测结果**（kW，24 步平均）—— **严格单调**：

| `tf` | 训练时 decoder 吃什么 | 24 步平均 MAE | 第 1 步 | 第 24 步 |
|---|---|---|---|---|
| **1.0**（全真值） | 100% 真值 | **21.120460** | 16.944296 | 21.901755 |
| **0.5**（混合） | 一半真值 | **20.583487** | 16.605000 | 22.110400 |
| **0.0**（全预测） | 100% 自己的预测 | **18.237022** | 16.314900 | 19.843500 |

```
推理 MAE：  21.12  >  20.58  >  18.24
tf：         1.0   >   0.5   >   0.0
训练-推理一致性：  最差   →   中等   →   完全一致
```

**这张表是 exposure bias 最干净的证据**：`tf` 越大，训练时 decoder 看到的
输入分布与推理时差得越远，推理 MAE 就越大 —— 而且是**单调**的，
在第 1 步（16.94 / 16.61 / 16.31）就已经显现，不是"后面才滚雪球"。

**为什么"训练 loss 最低"骗了你**：`tf=1.0` 时 decoder 的每一步输入都是**干干净净的真值**，
它只需要学一个"平滑"的映射，MSE 自然降得最快。
但推理时输入里带着自己的误差，这一分布**在训练里从未出现过**。

**为什么实践中还是常用 `tf=0.5` 而不是 `tf=0.0`**：

`tf=0.0` 在本数据上赢了，但它有两个真实的代价：

1. **早期梯度噪声极大**。decoder 第 1 步的输入就是自己的输出，而权重还是随机的 →
   输入几乎是纯噪声。本章靠 `clip_grad_norm_(params, 1.0)` 才没发散；
   去掉裁剪，`tf=0.0` 常常直接炸成 NaN。（可以自己把裁剪注释掉试一次。）
2. **在长序列上把误差"喂"给自己**，训练初期容易陷进平凡解。
   教科书里的标准折中是 **scheduled sampling**：`tf` 从 **1.0 线性退火到 0.0**，
   既保留早期的稳定性，又让后期分布对齐。

**判定规则**：

> **train / inference 的输入分布必须对齐 —— 用一致性换精度，或者用退火换稳定。**
> - 用 teacher forcing（它确实能加速收敛）→ **`tf` 不能停在 1.0**；
>   要么取 **0.5**，要么 **从 1.0 退火到 0.0**
> - 只看训练 loss 选模型 = 一定会选到 `tf=1.0` 那个**推理最差**的
> - **永远用推理模式的指标做模型选择** —— 训练 loss 只是调试工具
> - 本章实测的三档排序 `1.0 > 0.5 > 0.0`，**不是"哪个超参最好"的普适结论**，
>   而是"训练-推理一致性有多重要"这一条规律的直接体现
>
> 顺带一句：本章 Seq2Seq 最好的一档（18.24）**仍然输给 MLP 系（16.63 / 16.70）**。
> 这不是"LSTM 不行"，而是**在单变量、强日周期、只有 72 步输入的数据上，
> 全连接网络对相位的刻画已经足够**，LSTM 的序列建模能力没有发挥空间。
> 序列建模的优势要在**多变量、变长、长依赖**的场景里才兑现。
"""
    ),
    code(T6_CODE),
    code(T7_CODE),
    md(
        """
## 5.5 MAE–步长曲线：把六条曲线放在一张图里

一个数字（24 步平均 MAE）会丢掉**全部形状信息**。把逐条逐步 MAE 画出来，
才能看清「误差是平的、单调爬升、还是 U 形」—— 而形状直接告诉你这条方法的**失效方式**：

| 曲线形状 | 说明什么 | 本章对应 |
|---|---|---|
| **一条平线** | 每个步长都用了正确的相位，误差不累积 | seasonal naive（21.68 ~ 21.81） |
| **U 形** | 只有一个"锚点"，离锚点越远越差；绕回同一相位又变好 | persistence（30.1 → 81.3 → 21.8） |
| **单调爬升** | 每一步都在用上一步的输出 → 累积 | 递归（14.54 → 18.10） |
| **先快后慢的爬升** | 参数被摊薄，且没有反馈可用 | 直接多输出（14.81 → 18.37） |
| **高位爬升** | 同样的累积，但起点就高（训练-推理不一致的代价） | Seq2Seq（16.31 → 19.84，tf=0.0） |

**排行榜（本章实测，A 台区，kW）**：

| 方法 | 24 步平均 MAE | 相对 seasonal 基线 |
|---|---|---|
| 递归多步 recursive | **16.630607** | **−23.5%** |
| 直接多输出 direct | 16.696673 | −23.2% |
| Seq2Seq（TF=0.0） | 18.237022 | −16.1% |
| Seq2Seq（TF=0.5） | 20.583487 | −5.3% |
| Seq2Seq（TF=1.0） | 21.120460 | −2.9% |
| **seasonal naive 基线** | **21.741549** | 0（及格线） |
| persistence 基线 | 60.671651 | +179.1% |

**这张表里最值得记住的是后三行**：

- 「老值延续」这种基线在任何数据上都能一行写出来，却比正经训练的 Seq2Seq 差 **2.9 倍**。
  **先把基线摆在报告里，再谈模型**，这是竞赛评分表里"数据可视化 / 完整性"那一栏的得分点。
- **seasonal naive 21.741549 是本章的及格线**。它只用了「昨天同一时刻」这一条信息。
- **全 teacher forcing（21.12）只比基线好 2.9%** —— 一个花了 20 轮训练、
  有 24 步解码循环的模型，赢一条一行代码的基线只赢这么点。
  这就是为什么 §5.4.1 那条"训练-推理不一致"必须当成一等公民来对待。

### 5.5.1 难点深挖：先看图，再选方法（形状即诊断）

**为什么难**：模型的比较表只给一个平均 MAE，你会自然而然地选平均最低的那个。
但**平均最低不等于你的业务目标最优** —— 日前调度往往更关心**拐点附近**的精度。

**诊断三问**（照着图问）：

1. **有没有哪条曲线的某个步长突然上翘？**
   本章没有，但在光伏 / 风电上很常见（日出日落、爬坡段）。
   上翘的步长就是"该加特征或改模型"的位置。
2. **模型曲线与 seasonal 基线的差，是均匀的还是在某些步长上为零？**
   如果第 1 步就和基线打平，说明模型只学会了"抄昨天的数"，没学到增量信息。
3. **累积是在哪一步开始加速的？**
   本章递归是严格单调、且增幅均匀（每步 +0.15 kW 左右）；
   一旦看到某步之后斜率突然变大，说明 horizon 已经超过序列的有效记忆长度。

**判定规则**：

> **模型选择的完整顺序是「基线 → 曲线形状 → 平均指标」，不是「平均指标」一步到位。**
> - 报告里**必须**同时给出：基线 + 至少两类模型的曲线 + 24 步平均
> - 「平均最低」的方法，在某些步长上可能是最差的 —— 图上看得见，表上看不见
> - 本章结论：**24 步以内的日前负荷预测，先用递归 / 直接多输出去打；
>   只有当你要做多变量、变长、或者要显式接入未来已知协变量时，才上 Seq2Seq**
"""
    ),
    code(T9_CODE),
    code(T10_CODE),
    md(
        """
## 5.6 本章小结

### 三类方法一句话对照（可直接背）

| | 递归 recursive | 直接多输出 direct | Seq2Seq |
|---|---|---|---|
| **推理方式** | 滚动 24 次前向 | 一次前向出 24 维 | 逐字解码 24 步 |
| **训练目标** | 只监督第 1 步（`Ytr[:, :1]`） | 一次监督全部 24 步 | 逐字监督（可混合 TF） |
| **误差累积** | **有**（+14.3%） | 无 | 有 |
| **训练-推理一致？** | 一致 | 一致 | **TF=1.0 时不一致** |
| **变长 horizon** | 天然支持（滚动次数随便改） | 需要改输出层 | 天然支持 |
| **多变量 / 未来协变量** | 要自己拼进窗口 | 要自己拼进窗口 | **encoder / decoder 两处都能接** |
| **本章 MAE** | **16.630607** | 16.696673 | 18.237022（TF=0.0） |
| **什么时候用** | 自相关强、horizon ≤ 24 | horizon 短、要最快推理 | horizon 长 / 变长 / 多变量 |

### 四个"必须背下来"的判定规则

| 场景 | 规则 |
|---|---|
| 汇报多步误差 | 分母是**步数**：`(N, 24)` 误差矩阵摊平算 MAE；要形状就画 24 个点的曲线 |
| 切分处要不要留空 | **必须留**。窗口长 96，就得丢掉跨过切分点的那 95 个样本 |
| teacher forcing 怎么设 | `tf` 不能停在 **1.0**；要么 **0.5**，要么从 1.0 退火到 0.0（scheduled sampling） |
| 用哪个模型 | 先摆基线（seasonal naive 21.74）；**用推理模式的指标**选模型，不看训练 loss |

### 四步流水线

```
① 定口径     WINDOW=72 / HORIZON=24 → 单步与多步的 MAE 分开报
② 干净切分   目标落在切分点之前 = 训练；输入从切分点之后起 = 测试；中间 95 个丢掉
③ 两条基线   persistence（老值延续）60.67 / seasonal naive（昨日同时刻）21.74
④ 三种范式   递归（16.63）→ 直接多输出（16.70）→ Seq2Seq（TF=0.0 时 18.24）
              最后画 MAE–步长曲线，按「基线 → 形状 → 平均」的顺序选模型
```

## 易错点清单

1. **`horizon=1` 的 MAE 不能冒充 24 步**。本章 14.544809 vs 16.630607，差 14.3%。
2. **切分处必须留 `gap`**。窗口长 96，就一定有 95 个样本跨在切分点上（本章实测 95）。
3. **递归的第一步必须与单步 MAE 逐位相等**（`< 1e-9`）。不等就是窗口没滚动或尺度没还原。
4. **递归滚动时一定要丢掉窗口最老的点**（`win[:, 1:]`），否则窗口会越来越长。
5. **「直接多输出避免了误差累积」不等于它更好**。本章它反而比递归差 0.066 kW。
6. **`tf=1.0` 训练 loss 最低，但推理时最差**（21.12 vs 20.58 / 18.24）—— **严格单调**。
7. **训练 loss 不是模型选择依据**。本章 `tf=1.0` 就是"训练最好、推理最差"的活样本。
8. **`tf=0.0` 要配梯度裁剪**。早期 decoder 输入是随机权重下的自噪声，不裁剪常常直接 NaN。
9. **RNN / Seq2Seq 必须做梯度裁剪**。`nn.utils.clip_grad_norm_(params, 1.0)`。
10. **不能只报一个平均 MAE**。同样的 16.6，逐步曲线可能是平的也可能是爬升的。
11. **朴素基线必须进报告**。`seasonal naive` 只要一行代码，却是本章所有模型的及格线。
12. **尺度还原别忘**。模型在 `Z = (S - MU) / SD` 上训练，`* SD + MU` 之后才是 kW。
13. **`MU` / `SD` 只能用训练段拟合**。用全量会把测试段的信息带进来（ch06 §6.3 量化）。
14. **权重初始化也是随机数**。只在训练循环里 `manual_seed` 不够，
    模型实例化也必须包在同一个 `fork_rng` 里（本章 `build()`），否则跑第二遍数字就变。
15. **推理时至少要 `.detach()` 或 `torch.no_grad()`**。否则会一路建图、白吃内存。
16. **Seq2Seq 在本章输给 MLP，是数据决定的**，不是实现 bug —— 单变量强周期数据上
    LSTM 的序列建模能力没有发挥空间。
"""
    ),
]

# =========================================================================== #
# 4. 练习 / 答案版（EXERCISE）                                                #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch05 序列到序列预测（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`data/load_curve.csv` 的 **A 台区**（8760 h），`window=72` / `horizon=24`。
>
> **每题都跟了 `# 提示：`**，照着提示能独立做完。
> 如果 assert 的数字对不上，**先回头看上一步的打印输出**，别急着改 assert —— 那些数字都是实跑出来的。
>
> ⚠️ 训练三个 Seq2Seq 大约要 1 分钟，耐心等。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 清洗 + 只用训练段拟合缩放参数 | 2 | §5.1 / §5.1.1 |
| 2 | 滑窗 + 三块切分（含 gap） | 2 | §5.1 |
| 3 | 单步模型 | 2 | §5.1.1 |
| 4 | 递归多步滚动 | 2 | §5.2.1 |
| 5 | 直接多输出 | 2 | §5.3 |
| 6 | Seq2Seq（全 teacher forcing） | 2 | §5.4 |
| 7 | teacher forcing 三档对照 | 3 | §5.4.1 |
| 8 | 两条朴素基线 | 2 | §5.1.2 |
| 9 | MAE–步长曲线 + 排行榜 | 1 | §5.5 |
| 10 | 误差累积量化 + 样本对照图 | 2 | §5.2.1 / §5.5.1 |

**合计 20 个挖空。**
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md("## 任务 1：清洗 A 台区 + 只用训练段拟合缩放参数（§5.1）"),
    code(T1_CODE),
    md("## 任务 2：滑窗与三块切分（训练 / gap / 测试）（§5.1）"),
    code(T2_CODE),
    md("## 任务 3：单步模型 —— 先看清「口径」的基准（§5.1.1）"),
    code(T3_CODE),
    md("## 任务 4：递归多步 —— 把预测喂回去滚动 24 次（§5.2.1）"),
    code(T4_CODE),
    md("## 任务 5：直接多输出 —— 一次吐出 24 维（§5.3）"),
    code(T5_CODE),
    md("## 任务 6：Seq2Seq（encoder-decoder）+ 全 teacher forcing（§5.4）"),
    code(T6_CODE),
    md("## 任务 7：teacher forcing 三档对照 —— exposure bias 实测（§5.4.1）"),
    code(T7_CODE),
    md("## 任务 8：两条朴素基线（§5.1.2）"),
    code(T8_CODE),
    md("## 任务 9：MAE–步长曲线 + 24 步平均 MAE 排行榜（§5.5）"),
    code(T9_CODE),
    md("## 任务 10：误差累积量化 + 单样本对照图（§5.2.1 / §5.5.1）"),
    code(T10_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 10 个代码块的 assert 全部通过
- [ ] 单步 MAE 是多少？递归 24 步平均是多少？两者差几个百分点？
- [ ] 切分处的 `gap` 有多少个样本？为什么必须丢掉？
- [ ] 递归滚动时窗口为什么要 `[:, 1:]`？不丢会怎样？
- [ ] 为什么 `mae_rec[0]` 必须严格等于单步 MAE？
- [ ] 「误差累积」的机制一句话怎么说？递归第 24 步的 MAE 相对第 1 步涨了多少？
- [ ] 直接多输出为什么「没有累积」，却还是输给了递归？
- [ ] `teacher_forcing_ratio` 三档里，哪一档推理 MAE 最差？这叫什么现象？三档的排序是单调的吗？
- [ ] `tf=0.0` 训练时最大的风险是什么？本章靠什么压住的？
- [ ] 两条朴素基线各是多少？哪一条是本章的"及格线"？
- [ ] 「U 形」的曲线对应哪条方法？它为什么两头低？
- [ ] 本章为什么用 MAE 而不是 MSE 做评估指标？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 切分点 `SPLIT` | 7008 |
| 训练 / gap / 测试窗口数 | 6913 / **95** / 1657 |
| 训练段 `MU` / `SD` | 454.561801 / 72.623128 |
| 单步 `horizon=1` MAE | **14.544809** |
| 递归 24 步平均 MAE | **16.630607** |
| 递归第 1 / 12 / 24 步 MAE | 14.5448 / 16.7367 / 18.0992 |
| 直接多输出 24 步平均 MAE | **16.696673** |
| 直接第 1 / 24 步 MAE | 14.8118 / 18.3712 |
| Seq2Seq `tf=1.0` / `0.5` / `0.0` | 21.120460 / 20.583487 / **18.237022** |
| persistence 基线 | 60.671651（U 形：30.1→81.3→21.8） |
| seasonal naive 基线 | **21.741549**（一条平线） |
| 递归放大倍数 `mae_rec.mean()/mae_1` | **1.1434** |
| 第 24 步「直接 − 递归」 | **+0.271966 kW** |

> **选模型的口诀**：先摆基线（21.74）→ 再看曲线形状 → 最后才比平均 MAE。
> **永远用推理模式的指标选模型**，不看训练 loss。
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
