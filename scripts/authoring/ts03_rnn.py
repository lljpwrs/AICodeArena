#!/usr/bin/env python3
"""ch03 —— RNN 手写：滑窗监督化 / numpy 前向与 torch 对账 / 手写 BPTT / 梯度消失实测

本章主线（六条）：

1. **把时序变成监督学习**：滑窗 → `(样本, 时间步, 特征)`，且必须讲清「标签是窗口后一格」
2. **手写 RNNCell 前向（numpy）**：`h_t = tanh(x_t W_x + h_{t-1} W_h + b_h)`，逐时间步展开，
   再把 numpy 权重搬进 `nn.RNN` **逐值对账**（double 下 max|diff| ~1e-16）
3. **手写 BPTT（numpy）**：自己推 `dh_{t-1} = da_t · W_h` 的递推，与 `torch.autograd` 对账
4. **torch 版回归器**：`nn.RNN` + `nn.Linear`，训练 + 测试 MAE，并与「朴素持续」基线对照
5. **长依赖梯度消失实测**：固定权重、只换序列长度（10/20/40/80/160），
   看「第 1 步梯度 / 第 T 步梯度」掉到 1e-12；再扫谱半径、扫输入幅度，说明
   `tanh' <= 1` 与 `W_h` 谱半径共同决定衰减率 —— 引出 ch04 的门控
6. **本章小结**：RNN 的三件事（参数共享 / 隐状态 / BPTT）+ 两条判据

数据：`data/load_curve.csv` 的 A 台区（居民型），2025 全年 8760 个整点。
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ts03_rnn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch03_rnn"

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

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "| MPS", torch.backends.mps.is_available(), "（本章全程 CPU，数据小）")
print("原始行数", len(raw), "| 列", list(raw.columns))
'''

SETUP = '''df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
print("去重排序后", len(df), "行")

# 本章只做主序列：A 台区（居民型，20 点单峰、周末更忙），2025 全年 8760 个整点
a_frame = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳").sort_index()
s_a = a_frame["负荷值"].asfreq("h").loc["2025-01-01":"2025-12-31 23:00"]
print("A 台区整点 %d | 缺失 %d" % (len(s_a), int(s_a.isna().sum())))
print(s_a.head(3))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
W = 24            # 滑窗长度：用过去 24 小时预测下 1 小时
HID = 12          # 隐藏单元数
EPOCHS = 15
BATCH = 128
SEED = 42


def clean_series(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值。"""
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


def make_windows(z, w):
    """一维序列 → 监督矩阵：X[i] = z[i : i+w]，y[i] = z[i+w]。

    **标签是窗口后面那一格**，不是窗口末格 —— 这是时序监督化的第一号坑。
    """
    x = np.lib.stride_tricks.sliding_window_view(z, w)[:-1]
    return x, z[w:]


def make_rnn(rho, hidden=16, seed=1):
    """造一个 W_h 谱半径被钉死在 rho 上的 torch RNN（3.5 的梯度实验专用）。"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = nn.RNN(1, hidden, batch_first=True).double()
    with torch.no_grad():
        w = m.weight_hh_l0
        m.weight_hh_l0.copy_(w * (rho / float(torch.linalg.eigvals(w).abs().max())))
    return m


def last_step_grads(model, length, seed=3, scale=1.0):
    """loss 只挂在**最后一步**的输出上，回传得到每个输入步的梯度幅度。

    为什么要"只挂最后一步"：若 loss 挂在所有步上，dL/dx_t 里会有
    dL/dh_t 这一项**本步**的贡献，它一个时间步都不用传，永远不衰减 --
    测出来的比值会是 0.3 这种"假的不衰减"。序列到一维的回归任务天然只有
    最后一步有 loss，本函数就是复现这个口径。
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        x = torch.randn(1, length, 1, dtype=torch.float64, requires_grad=True) * scale
    out, _ = model(x)
    g = torch.autograd.grad(out[:, -1, :].pow(2).sum(), x)[0]
    return g[0, :, 0].abs().detach().numpy()
'''

# =========================================================================== #
# 2. 任务代码块                                                              #
# =========================================================================== #

T1_CODE = '''# @@todo 清洗 A 台区序列（脚手架已有 clean_series）
s_a_clean = clean_series(s_a)
# @@end

SERIES = s_a_clean.values.astype("float64")
print("主序列 %d 点 | 均值 %.6f | 标准差 %.6f | min %.4f | max %.4f"
      % (len(SERIES), SERIES.mean(), SERIES.std(), SERIES.min(), SERIES.max()))

# @@todo 把一维序列切成监督矩阵 X / y（脚手架已有 make_windows）
# @@hint 先把形状与错位关系搭对，标准化留到下一步再做
X_all, y_all = make_windows(SERIES, W)
# @@end

# @@todo 手工验证「标签 = 窗口后一格」这条关系
# @@hint 窗口里的最后一格是 X_all[0, -1]，标签 y_all[0] 应该是它的**下一格**
last_in_win = X_all[0, -1]
first_label = y_all[0]
# @@end

print("X %s | y %s | 样本数 %d" % (X_all.shape, y_all.shape, len(y_all)))
print("X[0] 前 3 格 %s ... 窗口末格 %.4f" % (np.round(X_all[0, :3], 3), last_in_win))
print("标签 y[0] %.4f = SERIES[%d] %.4f（窗口末格是 SERIES[%d] = %.4f）"
      % (first_label, W, SERIES[W], W - 1, SERIES[W - 1]))

# ---- 验收 ----
assert X_all.shape == (8736, 24) and y_all.shape == (8736,), "8760 - 24 = 8736 个样本"
assert np.array_equal(X_all[0], SERIES[:W])
assert last_in_win == SERIES[W - 1] and first_label == SERIES[W]
assert abs(first_label - last_in_win) > 1e-9, "标签与窗口末格是两个不同的时刻"
'''

T2_CODE = '''# @@todo 先在**序列**级别按时间 8:2 切分
# @@hint 时序数据只有一种合法切法：按时间，绝不能随机抽
cut_pt = int(len(SERIES) * 0.8)
# @@end

# @@todo 标准化参数**只允许来自训练段**
# @@hint 训练段就是 SERIES[:cut_pt]；用全量均值/方差就是 ch01 说的第三种泄漏
MU = float(SERIES[:cut_pt].mean())
SD = float(SERIES[:cut_pt].std())
# @@end

Z = (SERIES - MU) / SD

# @@todo 在标准化后的序列上重新切窗，然后按样本切分
# @@hint 样本级切分点 = 序列级切分点 - 窗口长度（窗口要满 24 格才有一个标签）
Xz, yz = make_windows(Z, W)
n_train = cut_pt - W
Xtr, ytr = Xz[:n_train], yz[:n_train]
Xte, yte = Xz[n_train:], yz[n_train:]
# @@end

print("序列级切分点 %d（%s）| 样本级切分点 %d" % (cut_pt, s_a_clean.index[cut_pt], n_train))
print("训练 %s / 测试 %s" % (Xtr.shape, Xte.shape))
print("y 训练段 mean %.6f std %.6f | 测试段 mean %.6f std %.6f"
      % (ytr.mean(), ytr.std(), yte.mean(), yte.std()))

# ---- 验收 ----
assert len(SERIES) == 8760 and Xz.shape == (8736, 24)
assert n_train == 6984 and Xte.shape == (1752, 24)
assert abs(MU - 454.561801) < 1e-6 and abs(SD - 72.623128) < 1e-6
assert abs(float(ytr.mean()) - 0.001420) < 1e-6, "训练段标准化后均值几乎为 0"
assert abs(float(ytr.std()) - 1.000632) < 1e-6
assert abs(float(yte.mean()) - 0.388310) < 1e-6, "测试段是 10/20 以后的冬季，均值被抬到 +0.39"
'''

T3_CODE = '''# 用一组固定的小输入做对账：B=3 个样本、T=7 个时间步、D=2 维特征、H=5 个隐藏单元
T_DEMO, B_DEMO, D_DEMO, H_DEMO = 7, 3, 2, 5

with torch.random.fork_rng():
    torch.manual_seed(0)          # 随机性一律显式固定，且包进 fork_rng

    # @@todo 造固定输入，并建一个**双精度**的 nn.RNN 当参照
    # @@hint torch.randn(B, T, D, dtype=torch.float64) 造输入
    # @@hint .double() 是为了让逐值对账落到 1e-12 以下（float32 只有 1e-7 精度）
    x_demo = torch.randn(B_DEMO, T_DEMO, D_DEMO, dtype=torch.float64)
    ref = nn.RNN(D_DEMO, H_DEMO, batch_first=True).double()
    # @@end

out_ref, hn_ref = ref(x_demo)
print("nn.RNN 的 out %s | h_n %s（h_n 第 0 维 = 层数 x 方向数）"
      % (tuple(out_ref.shape), tuple(hn_ref.shape)))

# @@todo 把 torch 的权重与偏置搬进 numpy
# @@hint weight_ih_l0 形状 (H, D)，weight_hh_l0 形状 (H, H)
# @@hint **偏置有两个**（bias_ih_l0 与 bias_hh_l0），公式里的 b_h 是它们之和
Wx = ref.weight_ih_l0.detach().numpy().copy()
Wh = ref.weight_hh_l0.detach().numpy().copy()
bh = (ref.bias_ih_l0 + ref.bias_hh_l0).detach().numpy().copy()
xn = x_demo.numpy()
# @@end

# @@todo 初始化隐状态，并准备收集每一步的 h
# @@hint h 的形状是 (批大小, 隐藏维) —— 注意不是 (隐藏维,)
# @@hint 每一步的 h 都要收集，最后 stack 成 (B, T, H) 才能和 out_ref 比
h_np = np.zeros((B_DEMO, H_DEMO))
hs_np = []
# @@end

for t in range(T_DEMO):
    # @@todo 手写单步 RNN：h_t = tanh(x_t . W_x + h_{t-1} . W_h + b_h)
    # @@hint x_t 就是 xn[:, t, :]，形状 (B, D)；矩阵乘用 @
    # @@hint torch 的权重是 (out, in) 排布，所以转置用 .T
    h_np = np.tanh(xn[:, t, :] @ Wx.T + h_np @ Wh.T + bh)
    hs_np.append(h_np)
    # @@end

# torch 的 out 是 (B, T, H)，stack 出来是 (T, B, H)，所以要转置
hs_np = np.stack(hs_np).transpose(1, 0, 2)

d_fwd = float(np.abs(hs_np - out_ref.detach().numpy()).max())
d_hn = float(np.abs(hs_np[:, -1] - hn_ref.detach().numpy()[0]).max())
print("手写前向 vs nn.RNN：全部时间步 max|diff| = %.3e" % d_fwd)
print("手写最后一步 h_T vs nn.RNN 的 h_n：max|diff| = %.3e" % d_hn)
print("权重形状 W_x %s | W_h %s | b_h %s" % (Wx.shape, Wh.shape, bh.shape))

# ---- 验收 ----
assert hs_np.shape == (B_DEMO, T_DEMO, H_DEMO)
assert d_fwd < 1e-12, "双精度下手写前向必须与 nn.RNN 逐值一致（实测 ~1e-16）"
assert d_hn < 1e-12, "h_n 就是最后一个时间步的 h —— 不是另算出来的东西"
'''

T4_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(0)

    # @@todo 关掉 batch_first，看输出形状与输入方向怎么变
    # @@hint batch_first=False 时输入被解释成 (T, B, D)，所以要 transpose(0, 1)
    # @@hint 权重一模一样（load_state_dict 把 ref 的参数拷过来），结果应当只差一个转置
    ref_tf = nn.RNN(D_DEMO, H_DEMO, batch_first=False).double()
    ref_tf.load_state_dict(ref.state_dict())
    out_tf, hn_tf = ref_tf(x_demo.transpose(0, 1))
    # @@end

d_tf = float(np.abs(out_tf.transpose(0, 1).detach().numpy() - out_ref.detach().numpy()).max())
print("batch_first=False 的 out 形状 %s（batch_first=True 是 %s）"
      % (tuple(out_tf.shape), tuple(out_ref.shape)))
print("两者转置后 max|diff| = %.3e" % d_tf)

# @@todo 记下两件事：h_n 的第 0 维是什么，W_x 的形状是什么
# @@hint h_n 形状是 (num_layers * num_directions, B, H)
hn_shape_1layer = tuple(hn_ref.shape)
Wx_rows = tuple(Wx.shape)
# @@end

print("h_n 形状 %s（第 0 维 = 层数 x 方向数）| W_x 形状 %s = (H, D)" % (hn_shape_1layer, Wx_rows))

# ---- 验收 ----
assert tuple(out_tf.shape) == (T_DEMO, B_DEMO, H_DEMO)
assert d_tf < 1e-12, "batch_first 只改输入/输出的排布，不改数值"
assert hn_shape_1layer == (1, B_DEMO, H_DEMO), "单层单向时 h_n 第 0 维是 1"
assert Wx_rows == (H_DEMO, D_DEMO) and Wh.shape == (H_DEMO, H_DEMO)
'''

T5_CODE = '''# @@todo 先用 torch 的 autograd 算一遍「loss = 所有输出平方和」的梯度
# @@hint clone().requires_grad_(True)：留一份可求导的输入副本
# @@hint 求导前先 ref.zero_grad()，否则 .grad 会跨轮累加
x_grad = x_demo.clone().requires_grad_(True)
out_g, _ = ref(x_grad)
loss_ref = out_g.pow(2).sum()
ref.zero_grad()
loss_ref.backward()
# @@end

# @@todo numpy 侧算同一个 loss（必须完全一致，否则后面比的不是同一条曲线）
loss_np = float((hs_np ** 2).sum())
# @@end

# 反向要按时间**倒序**，所以递推变量初始化在循环外
d_out = 2 * hs_np                                                # d loss / d h_t = 2 h_t
h_prev = np.concatenate([np.zeros((B_DEMO, 1, H_DEMO)), hs_np[:, :-1, :]], axis=1)
dh_next = np.zeros((B_DEMO, H_DEMO))
dWx = np.zeros_like(Wx)
dWh = np.zeros_like(Wh)
db = np.zeros_like(bh)

for t in reversed(range(T_DEMO)):
    # @@todo 第 t 步总梯度 =「本步输出的贡献」+「来自下一步的 dh」
    # @@hint 漏掉 d_out[:, t, :] 这一项，梯度会直接退化成全 0（见 3.3.1）
    dh = d_out[:, t, :] + dh_next
    # @@end
    # @@todo 穿过 tanh：tanh'(a) = 1 - tanh(a)^2 = 1 - h_t^2
    da = dh * (1 - hs_np[:, t, :] ** 2)
    # @@end
    # @@todo 累计三个参数的梯度：x_t 与 h_{t-1} 各自乘在自己那一侧
    # @@hint da 形状 (B, H)，x_t 与 h_{t-1} 都是 (B, .)，所以写 da.T @ ...
    dWx = dWx + da.T @ xn[:, t, :]
    dWh = dWh + da.T @ h_prev[:, t, :]
    db = db + da.sum(axis=0)
    # @@end
    # @@todo 把梯度递给上一步：dh_{t-1} = da_t . W_h
    # @@hint 这里**不要**再乘一次 tanh' —— tanh' 是 a_{t-1} 那一层的事（见 3.3.1）
    dh_next = da @ Wh
    # @@end

dWx_ref = ref.weight_ih_l0.grad.numpy()
dWh_ref = ref.weight_hh_l0.grad.numpy()
db_ref = ref.bias_ih_l0.grad.numpy()

print("loss：手写 %.10f vs torch %.10f" % (loss_np, float(loss_ref.detach())))
print("dW_x max|diff| %.3e" % np.abs(dWx - dWx_ref).max())
print("dW_h max|diff| %.3e" % np.abs(dWh - dWh_ref).max())
print("db   max|diff| %.3e（对照的是 bias_ih_l0.grad）" % np.abs(db - db_ref).max())
print("梯度整体量级：|dW_x| %.4f | |dW_h| %.4f" % (np.linalg.norm(dWx), np.linalg.norm(dWh)))

# ---- 验收 ----
assert abs(loss_np - float(loss_ref.detach())) < 1e-9
assert np.abs(dWx - dWx_ref).max() < 1e-12, "手写 BPTT 的 dW_x 必须与 autograd 逐值一致"
assert np.abs(dWh - dWh_ref).max() < 1e-12, "dW_h 同理"
assert np.abs(db - db_ref).max() < 1e-12
assert float(np.abs(ref.bias_ih_l0.grad.numpy() - ref.bias_hh_l0.grad.numpy()).max()) < 1e-12, \\
    "b_ih 与 b_hh 在公式里是相加的，所以两者的梯度**完全相同**"
'''

T6_CODE = '''# ---- 错误示范 A：dh 只取 dh_next，漏掉「本步输出」的贡献 ----
dh_next_a = np.zeros((B_DEMO, H_DEMO))
dWx_a = np.zeros_like(Wx)
dWh_a = np.zeros_like(Wh)
for t in reversed(range(T_DEMO)):
    # @@todo 只保留 dh_next（这是错的），穿过 tanh 后累计梯度
    da = dh_next_a * (1 - hs_np[:, t, :] ** 2)
    dWx_a = dWx_a + da.T @ xn[:, t, :]
    dWh_a = dWh_a + da.T @ h_prev[:, t, :]
    # @@end
    dh_next_a = da @ Wh

# ---- 错误示范 B：把 tanh' 多乘了一次到 dh_{t-1} 上 ----
dh_next_b = np.zeros((B_DEMO, H_DEMO))
dWx_b = np.zeros_like(Wx)
dWh_b = np.zeros_like(Wh)
for t in reversed(range(T_DEMO)):
    # @@todo 本步处理是对的，但递推写成「da . W_h 之后再乘一次 (1 - h_t^2)」
    # @@hint 这格就是要复现这个 bug，别"顺手改对"
    dh = d_out[:, t, :] + dh_next_b
    da = dh * (1 - hs_np[:, t, :] ** 2)
    dWx_b = dWx_b + da.T @ xn[:, t, :]
    dWh_b = dWh_b + da.T @ h_prev[:, t, :]
    # @@end
    dh_next_b = (da @ Wh) * (1 - hs_np[:, t, :] ** 2)

rel_a = float(np.linalg.norm(dWx_a - dWx) / np.linalg.norm(dWx))
rel_b = float(np.linalg.norm(dWx_b - dWx) / np.linalg.norm(dWx))
rel_bh = float(np.linalg.norm(dWh_b - dWh) / np.linalg.norm(dWh))
print("A 漏掉 dh_out：|dW_x| = %.6e（正确 %.6f）相对误差 %.4f"
      % (np.linalg.norm(dWx_a), np.linalg.norm(dWx), rel_a))
print("B 多乘一次 tanh'：相对误差 dW_x %.4f | dW_h %.4f" % (rel_b, rel_bh))
print("B 的 dW_h 范数 %.6f vs 正确 %.6f（只差 %.2f%%，却已经错了）"
      % (np.linalg.norm(dWh_b), np.linalg.norm(dWh),
         100 * abs(np.linalg.norm(dWh_b) / np.linalg.norm(dWh) - 1)))

# ---- 验收 ----
assert np.linalg.norm(dWx_a) < 1e-30, "漏掉本步 dh_out 后，除最后一步外没有任何梯度来源"
assert rel_a > 0.999
assert 0.05 < rel_b < 0.20, "多乘一次 tanh' 只让梯度偏了约 9% —— 这种错不会报错，只会悄悄变差"
'''

T7_CODE = '''class RNNRegressor(nn.Module):
    """序列到一维：只取最后一步的隐状态，接一个线性头。"""

    def __init__(self, hidden=HID):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# @@todo 把 numpy 监督矩阵转成 torch 张量，特征维补成 1
# @@hint RNN 要的是 (B, T, D)；Xtr 是 (N, T)，所以再 unsqueeze(-1)
# @@hint 训练用 float32 就够了，不必 double
Xtr_t = torch.tensor(Xtr.astype("float32")).unsqueeze(-1)
ytr_t = torch.tensor(ytr.astype("float32")).unsqueeze(-1)
Xte_t = torch.tensor(Xte.astype("float32")).unsqueeze(-1)
# @@end

loss_hist = []
with torch.random.fork_rng():
    torch.manual_seed(SEED)
    model = RNNRegressor()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = nn.MSELoss()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), BATCH):
            idx = perm[i: i + BATCH]
            # @@todo 一个 batch 的四行：清零 → 前向算 loss → 反向 → 更新
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            # @@end
        loss_hist.append(float(loss.detach()))

model.eval()
with torch.no_grad():
    # @@todo 在测试段上前向，并把输出压平成一维
    # @@hint model(Xte_t) 形状 (N, T, 1)，用 .numpy().ravel() 变成 (N,)
    pred_te = model(Xte_t).numpy().ravel()
    # @@end

# @@todo 算三种 MAE —— 你的模型、朴素持续、训练段均值
# @@hint 朴素基线 = 直接拿窗口最后一格当预测（Xte[:, -1]）
# @@hint 另一个基线 = 拿训练段均值当预测（ytr.mean()）
mae = float(np.abs(pred_te - yte).mean())
mae_persist = float(np.abs(Xte[:, -1] - yte).mean())
mae_mean = float(np.abs(float(ytr.mean()) - yte).mean())
# @@end

print("训练 loss（每 epoch 最后一个 batch）%.6f → %.6f" % (loss_hist[0], loss_hist[-1]))
print("测试 MAE（标准化空间）%.6f | 换算回 kW %.6f" % (mae, mae * SD))
print("对照：朴素持续 %.6f | 训练段均值 %.6f" % (mae_persist, mae_mean))
print("参数个数 %d（24 个时间步共用同一份权重，所以参数量与序列长度无关）"
      % sum(p.numel() for p in model.parameters()))

# ---- 验收 ----
assert abs(mae - 0.259707) < 1e-4, "固定 seed 后结果可复现"
assert abs(mae_persist - 0.414592) < 1e-6 and abs(mae_mean - 0.733844) < 1e-6
assert mae < mae_persist, "RNN 必须打赢「上一小时」这个朴素基线"
'''

T8_CODE = '''N_SHOW = 168          # 一周 = 168 小时

# @@todo 取测试段开头一周，把标准化空间还原成 kW
# @@hint 反标准化：x * SD + MU
pred_kw = pred_te[:N_SHOW] * SD + MU
true_kw = yte[:N_SHOW] * SD + MU
# @@end

fig, axes = plt.subplots(2, 1, figsize=(12, 6.5))
axes[0].plot(np.arange(N_SHOW), true_kw, lw=1.8, label="真值")
axes[0].plot(np.arange(N_SHOW), pred_kw, lw=1.4, label="RNN 预测（测试段第 1 周）")
axes[0].set_ylabel("负荷 kW")
axes[0].set_xlabel("测试段小时序号")
axes[0].set_title("RNN 单步预测 vs 真值（测试段 MAE %.2f kW）" % (mae * SD))
axes[0].legend(loc="upper right")
axes[1].plot(np.arange(1, EPOCHS + 1), loss_hist, marker="o", color="crimson")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("MSE（标准化空间）")
axes[1].set_title("训练损失下降曲线")
plt.tight_layout()

# ---- 验收 ----
assert len(pred_kw) == N_SHOW == 168
assert abs(float(true_kw.mean()) - 438.223571) < 1e-3, \\
    "测试段第 1 周（10/20 起）均值 438.22 kW，比全年均值 460.20 略低"
assert loss_hist[-1] < loss_hist[0], "15 个 epoch 内损失必须下降"
'''

T9_CODE = '''RNN_LENS = [10, 20, 40, 80, 160]

# @@todo 造一个 W_h 谱半径被钉死在 0.9 的参照 RNN（不训练，只看梯度）
# @@hint 脚手架 make_rnn(rho)
rnn09 = make_rnn(0.9)
# @@end

print("谱半径实测 %.6f" % float(torch.linalg.eigvals(rnn09.weight_hh_l0.detach()).abs().max()))

ratio_len = {}
for T in RNN_LENS:
    # @@todo 取该长度下逐步的梯度幅度，算出「第 1 步 / 第 T 步」的比值
    # @@hint 脚手架 last_step_grads(model, length) 返回长度 T 的逐输入步梯度幅度
    g = last_step_grads(rnn09, T)
    ratio_len[T] = float(g[0] / g[-1])
    # @@end

for T in RNN_LENS:
    r = ratio_len[T]
    print("T=%3d  第1步/第T步 = %.3e   |  平均每步 x%.4f" % (T, r, r ** (1 / (T - 1))))

# ---- 验收 ----
assert all(ratio_len[a] > ratio_len[b] for a, b in zip(RNN_LENS[:-1], RNN_LENS[1:])), \\
    "序列越长，早期梯度越小"
assert ratio_len[10] > 0.05 and ratio_len[160] < 1e-11
assert abs(np.log10(ratio_len[160]) - (-12.017)) < 0.5, "T=160 时早期梯度已经掉了 12 个数量级"
'''

T10_CODE = '''RHOS = [0.5, 0.7, 0.9, 0.99, 1.2]

ratio_rho = {}
for rho in RHOS:
    # @@todo 造谱半径为 rho 的 RNN，在 T=160 上量「第 1 步 / 第 T 步」梯度比
    m = make_rnn(rho)
    g = last_step_grads(m, 160)
    ratio_rho[rho] = float(g[0] / g[-1])
    # @@end

for rho in RHOS:
    r = ratio_rho[rho]
    print("rho=%.2f  第1步/第T步 = %.3e   平均每步 x%.4f" % (rho, r, r ** (1 / 159)))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].semilogy(RNN_LENS, [ratio_len[T] for T in RNN_LENS], marker="o", color="tab:blue")
axes[0].set_xlabel("序列长度 T")
axes[0].set_ylabel("第 1 步 / 第 T 步 梯度比（对数轴）")
axes[0].set_title("梯度消失：长度每翻一倍，早期梯度掉几个量级")
axes[0].grid(alpha=0.3)
axes[1].semilogy(RHOS, [ratio_rho[r] for r in RHOS], marker="s", color="tab:red")
axes[1].set_xlabel("W_h 谱半径 rho")
axes[1].set_ylabel("第 1 步 / 第 T 步 梯度比（对数轴）")
axes[1].set_title("衰减率由 W_h 的谱半径决定（T=160）")
axes[1].grid(alpha=0.3)
plt.tight_layout()

# ---- 验收 ----
assert all(ratio_rho[a] < ratio_rho[b] for a, b in zip(RHOS[:-1], RHOS[1:])), "谱半径越大越不容易衰减"
assert ratio_rho[0.5] < 1e-40 and ratio_rho[0.99] > 1e-8
assert all((ratio_rho[r] ** (1 / 159)) < 1.0 for r in RHOS), \\
    "连 rho=1.2 也是衰减的 —— tanh 把它按住了，见 3.5.2"
'''

T11_CODE = '''# @@todo 实测「平均每步衰减因子」，再和理论上的谱半径对照（T=40）
# @@hint 因子 = 首末比的 1/(T-1) 次方
meas_factor = ratio_len[40] ** (1 / 39)
theory = 0.9
# @@end

print("rho=0.9 时：理论每步因子 = %.4f | 实测每步因子 = %.4f | 差 %.4f"
      % (theory, meas_factor, theory - meas_factor))
print("差的这部分就是 tanh' —— 链式法则里每穿过一次 tanh 就要乘一个 (1 - h^2) <= 1")

# rho=0.9，只把**输入幅度**放大，看衰减变快还是变慢（T=40）
ratio_scale = {}
for sc in (1.0, 5.0, 20.0):
    # @@todo 同口径量一次，只是把输入幅度乘上 sc
    # @@hint 脚手架 last_step_grads(model, length, scale=...)
    g_s = last_step_grads(rnn09, 40, scale=sc)
    ratio_scale[sc] = float(g_s[0] / g_s[-1])
    # @@end
for sc in (1.0, 5.0, 20.0):
    r = ratio_scale[sc]
    print("输入幅度 x%-5.1f  首末比 %.3e  平均每步 x%.4f" % (sc, r, r ** (1 / 39)))

# ---- 验收 ----
assert abs(meas_factor - 0.753637) < 1e-4, "实测因子比谱半径 0.9 小 0.146，就是 tanh' 的代价"
assert ratio_scale[1.0] > ratio_scale[5.0] > ratio_scale[20.0], \\
    "输入越大 → h 越饱和 → tanh' 越小 → 衰减越快"
assert ratio_scale[20.0] < 1e-13
'''

T12_CODE = '''# @@todo 把 rho 调到 1.2（谱半径 > 1，理论上该"放大"了）
# @@hint 只要改一个数字：make_rnn(1.2)
rnn12 = make_rnn(1.2)
# @@end

# 与上一格完全同口径，只换了 rho
ratio_scale12 = {}
for sc in (1.0, 5.0, 20.0):
    # @@todo 在 rho=1.2 的模型上再量一遍三个输入幅度
    g_s = last_step_grads(rnn12, 40, scale=sc)
    ratio_scale12[sc] = float(g_s[0] / g_s[-1])
    # @@end

for sc in (1.0, 5.0, 20.0):
    r = ratio_scale12[sc]
    print("rho=1.2 输入幅度 x%-5.1f  首末比 %.3e  平均每步 x%.4f" % (sc, r, r ** (1 / 39)))

print()
print("结论对照：")
print("  rho=0.9  x1 的每步因子 %.4f  →  x20 变成 %.4f"
      % (ratio_scale[1.0] ** (1 / 39), ratio_scale[20.0] ** (1 / 39)))
print("  rho=1.2  x1 的每步因子 %.4f  →  x20 变成 %.4f"
      % (ratio_scale12[1.0] ** (1 / 39), ratio_scale12[20.0] ** (1 / 39)))

# ---- 验收 ----
assert ratio_scale12[1.0] > ratio_scale[1.0], "谱半径越大，同样输入下越不容易衰减"
assert ratio_scale12[20.0] < 1e-8, "但只要输入一大，rho=1.2 也救不回来 —— 饱和"
assert (ratio_scale12[20.0] ** (1 / 39)) < 0.60
assert ratio_scale12[1.0] < 1.0, "谱半径 1.2 也没能让梯度放大（tanh 饱和封顶）"
'''

LESSON = [
    md(
        """
# ch03 RNN 手写（讲解版）

> **本节考点**：把「一个窗口」变成一个「时间步序列」，然后**亲手把它前向、反向各算一遍**。
> 手写不是为了替代 `nn.RNN`，而是为了在模型不收敛时，你能立刻判断出是**形状**错了、
> **偏置**错了、还是**梯度根本没传到**。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 模型训练 | **技能操作 40%** | `nn.RNN` 的输入排布、隐状态初值、损失口径 |
| 模型原理 | 理论 / 答辩 | 参数共享、隐状态、BPTT 递推式 |
| 新能源功率 / 负荷预测 | 行业赛题 | 滑窗监督化 + 序列到一维回归的完整链路 |

**与 `03_nlp/ch03` 的分工**：那边用 RNN 做**文本分类**（`out[:, -1, :]` 接分类头就完事）；
本章做**数值回归**，重点是**形状 / 逐值对账 / BPTT 手推**，以及时序特有的「滑窗 + 标签错位」。

## 学习目标

1. 把 8760 点负荷序列变成 `(样本, 时间步)` 的监督矩阵，并说清标签为什么是「窗口后一格」
2. 用 numpy 手写 RNN 单步前向，与 `nn.RNN` **逐值对到 1e-16**
3. 手推 BPTT 的 `dh_{t-1}` 递推，与 `torch.autograd` **逐值对到 1e-15**
4. 训练一个 `nn.RNN` 回归器，测试 MAE 0.259707（朴素基线 0.414592）
5. 用一次实验**亲眼看到**梯度消失：T=160 时早期梯度只剩 9.6e-13
6. 说清**两个旋钮**（`W_h` 谱半径、输入幅度 / tanh 饱和）怎样决定衰减率

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `np.lib.stride_tricks.sliding_window_view(z, w)` | `w` | `(N-w+1, w)` | 零拷贝滑窗；最后一行没有标签，要 `[:-1]` |
| `nn.RNN(in, hidden, batch_first=)` | `num_layers` / `bidirectional` | module | `batch_first=True` → 输入 `(B, T, D)` |
| `rnn.weight_ih_l0` / `weight_hh_l0` | — | `(H, D)` / `(H, H)` | **torch 是 (out, in) 排布**，numpy 里要 `.T` |
| `rnn.bias_ih_l0` / `bias_hh_l0` | — | `(H,)` | 公式里的 `b_h` 是这两个的**和** |
| `rnn(x)` | — | `(out, h_n)` | `out` 是每个时间步的 `h`；`h_n` 是最后一个（单层单向时第 0 维是 1） |
| `torch.autograd.grad(loss, x)` | `retain_graph` | tuple | 对**指定张量**求导，不污染 `.grad` |
| `torch.linalg.eigvals(W)` | — | complex | 谱半径 = `abs().max()`，决定梯度衰减率 |
| `torch.random.fork_rng()` | — | context | 局部固定随机性，**结果与执行顺序无关** |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md(
        """
## 3.1 把时序变成监督学习：滑窗 → (样本, 时间步, 特征)

统计方法（ARIMA 那一套）吃的是**一条序列**；神经网络吃的是**一堆样本**。
把两者接起来的一步就是**滑窗**：

```
序列:    s0  s1  s2  s3  ...  s23 | s24  s25  ...
样本 0: [s0  ...  s23]                  ↑ 标签 = s24
样本 1:      [s1  ...  s24]             ↑ 标签 = s25
```

一行代码就能零拷贝地切出来：

```python
X, y = make_windows(z, 24)
# X[i] = z[i : i+24]   ← (样本, 时间步)
# y[i] = z[i+24]       ← 窗口**后一格**
```

三个数必须能一口报出来：

| 量 | 值 | 怎么来的 |
|---|---|---|
| 主序列长度 | **8760** | 2025 全年小时级 |
| 样本数 | **8736** | 8760 − 24，滑窗把尾巴上凑不满一整窗的切掉 |
| `X` 的形状 | **(8736, 24)** | 暂时只有 1 个特征；进网络时才 `unsqueeze(-1)` 成 `(N, T, 1)` |

### 3.1.1 难点深挖：标签是「窗口后一格」，不是「窗口末格」

**为什么会写错**：多数人第一次写滑窗时会写成

```python
X = z.reshape(-1, 24)
y = X[:, -1]          # 拿窗口最后一格当标签
```

在**无监督 / 自编码**任务里这没错；但在**预测**任务里就是灾难：
`X[:, -1]`（即 `s23`）本来就在特征里，模型只要把它照抄出来，训练 loss 立刻归零 ——
你会看到一个"完美"的模型，然后在测试集上一塌糊涂。

**正确写法**：标签必须落在窗口**外面**的第一格。

```python
X, y = make_windows(z, 24)      # y[i] = z[i+24]
assert X[0, -1] == z[23]        # 窗口末格
assert y[0]      == z[24]       # 标签是它的下一格
```

一句话判据：

> **把 `X` 和 `y` 的时间范围拼起来看。**
> 若 `y` 的时刻 ≤ `X` 的最后一格 —— 泄漏；
> 若 `y` 的时刻 = `X` 的最后一格 + 1 —— 单步预测，正确。

同一个坑还有第二个变体：`sliding_window_view(z, 24)` 返回的是 **(N-23, 24)**，
最后一行 `[s_8736 ... s_8759]` 后面**没有标签**了，所以必须 `[:-1]`。
忘了这一步，`X` 和 `y` 长度会差 1，报错信息还很难懂
（`operands could not be broadcast together`）。
"""
    ),
    code(T1_CODE),
    md(
        """
## 3.2 手写 RNNCell 前向（numpy）

RNN 的全部秘密就一行公式：

```
h_t = tanh( x_t · W_x  +  h_{t-1} · W_h  +  b_h )
```

四个角色：

| 名字 | 形状 | 作用 |
|---|---|---|
| `W_x` | `(H, D)` | 输入 → 隐藏，**每个时间步共用同一份**（参数共享） |
| `W_h` | `(H, H)` | 上一步隐藏 → 本步隐藏，也就是"记忆"的通道 |
| `b_h` | `(H,)` | 偏置 |
| `h_t` | `(B, H)` | 本步的隐状态，**同时**就是这一步的输出 |

把 torch 的权重搬进 numpy 逐值对账，三个细节决定成败：

| 细节 | torch | numpy 写法 |
|---|---|---|
| 权重排布 | `(out, in)` | 必须 `.T` |
| 偏置有两个 | `bias_ih_l0` + `bias_hh_l0` | **先相加**才是 `b_h` |
| 隐状态初值 | 全 0（默认 `h_0=None`） | `np.zeros((B, H))` |

实测双精度下 `max|diff| = 1.39e-16` —— 这不是"差不多"，是**同一个函数**：
说明我们的公式、转置、偏置、初值全对了。

### 3.2.1 难点深挖：`batch_first` 只改排布，不改数值

`nn.RNN` 的 `batch_first` 默认是 **`False`**，即输入要被解释成 `(T, B, D)`；
而 sklearn 那套习惯让我们本能地写成 `(B, T, D)`。
忘了这个参数的表现是**不报错但全错**：批大小和时间步被静默互换，
RNN 会把"一个批里 3 个样本各自的 t=0"当成"一个样本的 3 个时间步"。

实测（同一份权重，`load_state_dict` 拷过去）：

| 设置 | 输入形状 | 输出形状 |
|---|---|---|
| `batch_first=True` | `(3, 7, 2)` | `(3, 7, 5)` |
| `batch_first=False` | `(7, 3, 2)` | `(7, 3, 5)` |

把 `False` 版的输出 `transpose(0, 1)` 回来，与 `True` 版 **max|diff| = 0**。
结论：**`batch_first` 是纯粹的排布约定，与数学无关** —— 但它决定了你的模型
是"在时间上展开"还是"在批上展开"。

还有一条更容易漏的：`h_n` 的形状是 `(num_layers * num_directions, B, H)`。
单层单向时第 0 维是 **1**（不是被 squeeze 掉的）。
所以 `h_n[0]` 才是 `(B, H)`；直接把 `h_n` 当成 `(B, H)` 去运算，
会得到 `(1, B, H)` 的广播结果 —— 又是一个不报错的错。
"""
    ),
    code(T3_CODE),
    code(T4_CODE),
    md(
        """
## 3.3 手写 BPTT（numpy 反向传播）

`torch.autograd` 能干这件事，但它不告诉你**梯度是怎么一步步传回去的**。
BPTT（Backpropagation Through Time）就是「把 RNN 沿时间展开成一个深网络，再普通地反向」。

### 3.3.1 难点深挖：`dh_{t-1}` 那一步的两个经典写法错误

设 `a_t = x_t W_x + h_{t-1} W_h + b_h`，`h_t = tanh(a_t)`，`loss = Σ_t h_t²`（本例）。

**前向**：`h_t = tanh(a_t)`，一步到底。

**反向**：从 `t = T-1` 倒着走到 `0`，每一步做四件事。

```
① 汇总本步的 dh：  dh_t = ∂loss/∂h_t（本步输出那一份）+ dh_next
② 穿过 tanh：      da_t = dh_t ⊙ (1 - h_t²)
③ 累计参数梯度：    dW_x += da_tᵀ x_t
                    dW_h += da_tᵀ h_{t-1}
                    db   += Σ_batch da_t
④ 递给上一步：      dh_{t-1} = da_t · W_h        ← 只有这一步叫 BPTT
```

**错误示范 A —— 漏掉第 ① 步的 `d_out`**：

```python
da = dh_next * (1 - h_t**2)      # 少了 + d_out[:, t, :]
```

后果是**梯度直接退化成全 0**（实测 `|dW_x| = 0.0`，相对误差 **1.0000**）。
道理：`dh_next` 从最后一步的 0 起步，第一次相乘就乘 0，后面全是 0 ——
loss 明明挂在 T 个输出上，梯度却只从最后一个往回传，而最后一步自己也只有 0。
这个版本**不会报错**，只会让模型完全学不动。

**错误示范 B —— 在 `dh_{t-1}` 上多乘一次 `tanh'`**：

```python
dh_next = (da @ W_h) * (1 - h_t**2)      # ← 多乘了一次
```

看起来"顺手把非线性导数补上"，实际上 `1 - h_t²` 是 `a_t` 那一层的导数，
下一步（`t-1`）会再乘一次属于它自己的。实测相对误差：
`dW_x` **8.91%**、`dW_h` **13.26%**。梯度范数只差百分之十几 ——
训练**照样能跑**，只是永远差一点。

> **判别口诀**：`tanh'` 只出现一次，位置固定在「`dh_t` → `da_t`」这一步。
> `dh_{t-1} = da_t · W_h` 后面**不再有任何导数因子** —— 因为 `W_h` 是线性的。

**③ 的一个附带结论**：因为公式里 `b_h = b_ih + b_hh`，所以
`∂loss/∂b_ih` 与 `∂loss/∂b_hh` **完全相同**（实测 max|diff| = 0）。
以后看到两个 bias 的梯度一模一样，不要以为是 bug。
"""
    ),
    code(T5_CODE),
    code(T6_CODE),
    md(
        """
## 3.4 torch 版回归器：`nn.RNN` + `nn.Linear`

把 handle 交给 torch，结构就是「RNN 扫一遍 → 取**最后一步**的 `h` → 线性头」：

```python
out, _ = self.rnn(x)           # (B, T, H)
return self.fc(out[:, -1, :])  # 只用最后一步
```

为什么只取最后一步？因为 `h_T` 按定义就是「看完整个窗口之后的总结」——
这正是序列到一维回归的标准范式。（文本分类同理，只是那边把 `h_T` 当句向量用。）

### 训练与评估结果

| 指标 | 标准化空间 | 换算回 kW | 说明 |
|---|---|---|---|
| 训练段均值基线 | 0.733844 | 53.29 | 什么都不学，用训练段均值硬猜 |
| **朴素持续**（拿窗口末格） | 0.414592 | 30.11 | 时序最强基线，**必须打赢它** |
| **RNN 单步预测** | **0.259707** | **18.86** | 15 epoch、12 个隐藏单元 |
| 相对朴素持续 | −37.4% | −11.25 kW | |

两个必须知道的细节：

1. **损失口径**：训练报的是「每个 epoch 最后一个 batch 的 MSE」，
   评估报的是「测试段 MAE」。两个数不可比 —— 永远别把训练 loss 当成绩报。
2. **测试段偏难**：切分点在 `2025-10-20 00:00`，测试段是**冬季**，
   `y` 的均值从训练段的 `0.0014` 抬到 `0.3883`（差 0.39 个标准差）。
   这是真实的分布漂移，也是为什么"测试 MAE 0.2597"没有训练 loss 看起来那么漂亮。

> **一条工程习惯**：报 MAE 时**永远同时报朴素基线**。
> 负荷预测里"上一小时的值"强得离谱（0.4146），
> 任何没打赢它的模型都还没有资格进答辩。
"""
    ),
    code(T7_CODE),
    code(T8_CODE),
    md(
        """
## 3.5 长依赖梯度消失实测

现在来量化 RNN 最著名的毛病。实验设计必须严格，否则测出来的东西是假的：

| 设计要点 | 做法 | 为什么 |
|---|---|---|
| 固定权重，只换长度 | `make_rnn(0.9)` 造一个谱半径钉死在 0.9 的模型 | 排除"训练过程"这个变量 |
| **loss 只挂最后一步** | `loss = out[:, -1, :].pow(2).sum()` | 见下面的坑 |
| 输入用同一段前缀 | 同一个 `torch.manual_seed(3)` | 不同长度看到的是同一段输入前 10 格 |
| 指标 | `第 1 步梯度 / 第 T 步梯度` | 越接近 0，越说明早期信息传不回来 |

> **测量口径的坑（第一版就踩了）**：如果写成 `loss = out.pow(2).sum()`（所有步都挂 loss），
> 那么 `∂L/∂x_t` 里含一项**本步自己的** `∂L/∂h_t`，它一个时间步都不用传，永远不衰减。
> 实测这样量出来的比值在 0.3 ~ 1.5 之间乱跳，**完全看不出梯度消失**。
> 真实的"序列 → 一维"任务天然只有最后一步有 loss，所以正确的口径是只挂最后一步。

### 实测：长度扫描（ρ = 0.9）

| T | 10 | 20 | 40 | 80 | **160** |
|---|---|---|---|---|---|
| 第1步/第T步 | 8.14e-02 | 4.81e-03 | 1.62e-05 | 9.13e-07 | **9.62e-13** |
| 平均每步因子 | 0.7567 | 0.7551 | 0.7536 | 0.8386 | 0.8403 |

T 从 10 涨到 160（16 倍），早期梯度掉了 **11 个数量级**。
换句话说：**第 160 步跟前的输入，对 loss 的影响已经在浮点噪声附近了。**

### 实测：谱半径扫描（T = 160）

| ρ(W_h) | 0.5 | 0.7 | 0.9 | 0.99 | 1.2 |
|---|---|---|---|---|---|
| 第1步/第T步 | 2.46e-52 | 2.45e-29 | 9.62e-13 | 4.41e-07 | 1.08e-03 |
| 平均每步因子 | 0.4736 | 0.6608 | 0.8403 | 0.9121 | 0.9580 |

**干净、单调**：谱半径每往上挪一点，衰减率就慢一大截。
这就是「衰减率由 `W_h` 的谱半径决定」的实测版本。

### 3.5.1 难点深挖：实测每步因子 0.7536，为什么比谱半径 0.9 小？

因为链式法则里**每穿过一次 `tanh` 就要乘一个 `tanh' = 1 - h²`**，而 `tanh' ≤ 1`。

```
每步因子 = ρ(W_h)  ×  tanh'        实测 0.7536 = 0.9 × 0.8374
```

所以 `W_h` 的谱半径只是**上界**，真正决定衰减的是「谱半径 × tanh 的平均斜率」。

**由此立刻得到一个反直觉的推论**：把输入放大（或让偏置变大）会让 `|h|` 变大，
`tanh` 进入**饱和区**，`tanh'` 趋近 0 —— 衰减反而**更快**。实测（ρ = 0.9，T = 40）：

| 输入幅度 | 平均每步因子 | 首末比 | &#124;h&#124; 均值 |
|---|---|---|---|
| ×1 | 0.7536 | 1.62e-05 | 0.196 |
| ×5 | 0.6041 | 2.90e-09 | 0.397 |
| ×20 | **0.4392** | **1.16e-14** | 0.665 |

### 3.5.2 难点深挖：把谱半径调到 1.2 也救不了

"衰减快就把 `W_h` 调大"是本能反应。实测（T = 40）：

| 输入幅度 | ρ = 0.9 每步因子 | ρ = 1.2 每步因子 |
|---|---|---|
| ×1 | 0.7536 | 0.9913 |
| ×5 | 0.6041 | 0.7948 |
| ×20 | 0.4392 | **0.5698** |

ρ = 1.2、小输入时每步因子确实接近 1（0.9913，几乎不衰减）；
但输入一大，**它也掉到 0.5698**。而且注意：**ρ = 1.2 全程都没有"爆炸"**
（首末比 1.08e-03 < 1），因为 `tanh` 的值域被锁在 `(−1, 1)`，把放大效应吃掉了。

> **所以 RNN 真正的病不是"梯度爆炸"，而是「记忆被两堵墙夹住」**：
> - 墙一（上限）：`tanh' ≤ 1`，每步至少衰减一点点；
> - 墙二（饱和）：想让记忆强一点就得把 `W_h` 调大、把 `h` 推大，
>   但 `h` 一大 `tanh'` 就趋近 0，衰减**更快**。
>
> 两堵墙之间没有可行解 —— **这就是必须引入门控（ch04）的唯一理由**。
> 门控要做的事：**给信息开一条不受 `tanh` 挤压的通道**（LSTM 的 cell state）。
"""
    ),
    code(T9_CODE),
    code(T10_CODE),
    code(T11_CODE),
    code(T12_CODE),
    md(
        """
## 本章小结

**一句话**：RNN = 一个对「时间」重复使用的全连接层，加上一个把时间连起来的隐状态。

### RNN 的三件事

```
① 参数共享   W_x / W_h 在所有时间步是同一份 → 序列多长，参数量都不变
② 隐状态     h_t 是「看完前 t 步之后的总结」，也是唯一的记忆载体
③ BPTT       把网络沿时间展开，反向时 dh_{t-1} = da_t · W_h，梯度按时间倒序递推
```

### 两条判据（背下来）

| 场景 | 判据 |
|---|---|
| 滑窗有没有泄漏 | 拼出 `X` 和 `y` 的**时间范围**：`y` 的时刻必须 = `X` 末格 + 1 |
| 梯度传不传得回来 | 量 `第 1 步梯度 / 第 T 步梯度`；**loss 必须只挂最后一步**，否则测不出来 |

### 一张对照表：手写 vs torch

| 部件 | numpy 手写 | torch 对应 |
|---|---|---|
| 单步前向 | `np.tanh(x @ Wx.T + h @ Wh.T + bh)` | `nn.RNNCell` / `nn.RNN` 内部 |
| 权重排布 | 需要手动 `.T` | `weight_ih_l0 (H, D)` 本来就是 `(out, in)` |
| 偏置 | 一个 `bh` | `bias_ih_l0 + bias_hh_l0` |
| 反向 | 手推 `dh_next = da @ Wh` | `loss.backward()` |
| 对账结果 | — | 前向 `1.39e-16`；反向 `1.78e-15` |

### 易错点清单

1. **标签是窗口后一格**，不是窗口末格。写成末格 = 把答案塞进特征。
2. **`sliding_window_view` 要 `[:-1]`**，否则 `X` / `y` 长度差 1。
3. **`batch_first` 默认 `False`**，输入被当成 `(T, B, D)`。忘了它不报错、只全错。
4. **公式里的 `b_h` 是两个偏置之和**（`bias_ih_l0 + bias_hh_l0`）。
5. **torch 权重是 `(out, in)`**，numpy 里手写要转置。
6. **`h_n` 形状是 `(层数×方向数, B, H)`**，单层单向时是 `(1, B, H)`。
7. **BPTT 里 `dh_t` 要加上本步输出那一项**，漏了梯度直接变全 0。
8. **`dh_next = da @ W_h` 后面不要再乘 `tanh'`** —— 多乘会让梯度偏 9%~13%，还不报错。
9. **`ref.zero_grad()` 要在 `backward()` 之前**，否则 `.grad` 会跨轮累加。
10. **梯度实验的 loss 必须只挂最后一步**，否则量出来的是"假的不衰减"。
11. **标准化参数只能来自训练段**（`SERIES[:cut_pt]`）。
12. **报 MAE 必须同时报朴素持续基线**（本章 0.414592），否则数字没有意义。
13. **`tanh' ≤ 1`**：谱半径只是衰减率的上界，实测每步因子 = ρ × tanh'。
14. **输入幅度越大衰减越快**（饱和），所以"把 `W_h` 调大"救不了 RNN。

**下一步**：ch04 给 RNN 加门。你会看到 LSTM 用一条**加法**的 cell state 高速公路，
把「每步至少衰减一点」这个限制绕开 —— 而代价是**四倍的参数量和一堆门顺序的坑**。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch03 RNN 手写（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`data/load_curve.csv` 的 A 台区（居民型），2025 全年 8760 个整点。
>
> **每题都跟了 `# 提示：`**，照着提示能独立做完。
> 如果 assert 的数字对不上，**先回头看上一步的输出**，别急着改 assert —— 那些数字都是实跑出来的。
>
> 建议顺序：先把 3.1 的滑窗搭对，再进 3.2 / 3.3 的手写对账。
> 手写对账这两个任务，**对不上就是有 bug**，别用 `abs() < 1` 糊过去。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 滑窗 → 监督矩阵（标签是窗口后一格） | 4 | 3.1 / 3.1.1 |
| 2 | 按时间切分 + 只用训练段标准化 | 7 | 3.1 |
| 3 | 手写 RNN 单步前向，与 `nn.RNN` 对账 | 10 | 3.2 |
| 4 | `batch_first` 与 `h_n` 的形状 | 5 | 3.2.1 |
| 5 | 手写 BPTT，与 `torch.autograd` 对账 | 12 | 3.3 |
| 6 | BPTT 的两个经典写法错误 | 7 | 3.3.1 |
| 7 | `nn.RNN` 回归器训练 + 基线对照 | 11 | 3.4 |
| 8 | 预测曲线 + 损失曲线 | 2 | 3.4 |
| 9 | 梯度消失：长度扫描 | 3 | 3.5 |
| 10 | 梯度消失：谱半径扫描 | 3 | 3.5 |
| 11 | 实测每步因子与 tanh 饱和 | 4 | 3.5.1 |
| 12 | 谱半径 1.2 也救不了 | 3 | 3.5.2 |
| **合计** | 12 个任务 / 32 个 `@@todo` 块 | **71** | |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md("## 任务 1：滑窗 → 监督矩阵（3.1）"),
    code(T1_CODE),
    md("## 任务 2：时间切分与训练段标准化（3.1）"),
    code(T2_CODE),
    md("## 任务 3：手写 RNN 单步前向并对账（3.2）"),
    code(T3_CODE),
    md("## 任务 4：`batch_first` 与 `h_n` 的形状（3.2.1）"),
    code(T4_CODE),
    md("## 任务 5：手写 BPTT 并对账（3.3）"),
    code(T5_CODE),
    md("## 任务 6：BPTT 的两个经典写法错误（3.3.1）"),
    code(T6_CODE),
    md("## 任务 7：`nn.RNN` 回归器训练 + 基线对照（3.4）"),
    code(T7_CODE),
    md("## 任务 8：预测曲线与损失曲线（3.4）"),
    code(T8_CODE),
    md("## 任务 9：梯度消失 —— 长度扫描（3.5）"),
    code(T9_CODE),
    md("## 任务 10：梯度消失 —— 谱半径扫描（3.5）"),
    code(T10_CODE),
    md("## 任务 11：实测每步因子与 tanh 饱和（3.5.1）"),
    code(T11_CODE),
    md("## 任务 12：谱半径 1.2 也救不了（3.5.2）"),
    code(T12_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 12 个代码块的 assert 全部通过
- [ ] 8736 这个样本数怎么来的？`X` 的形状为什么是 `(8736, 24)`？
- [ ] 标签为什么必须取「窗口后一格」？写成窗口末格会发生什么？
- [ ] 公式里的 `b_h` 对应 torch 的哪两个属性？
- [ ] torch 的 `weight_ih_l0` 是 `(H, D)` 还是 `(D, H)`？手写时要怎么处理？
- [ ] `batch_first=True` 与 `False` 的输出差在哪？数值上差多少？
- [ ] `h_n` 的形状是什么？单层单向时为什么不是 `(B, H)`？
- [ ] BPTT 里 `dh_t` 由哪两部分组成？漏掉其中一部分会怎样？
- [ ] `dh_{t-1} = da_t · W_h` 后面还要不要乘 `tanh'`？多乘会偏多少？
- [ ] 为什么 `bias_ih_l0` 与 `bias_hh_l0` 的梯度完全一样？
- [ ] 为什么梯度实验的 loss 必须只挂最后一步？
- [ ] T=160、ρ=0.9 时，第 1 步与第 T 步的梯度比是多少？
- [ ] 实测每步因子 0.7536 与谱半径 0.9 的差从哪来？
- [ ] 为什么把 `W_h` 谱半径调到 1.2 也解决不了梯度消失？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 主序列长度 / 样本数 | 8760 / 8736 |
| `X` 形状 | (8736, 24) |
| 训练段 MU / SD | 454.561801 / 72.623128 |
| 样本级切分点 / 测试样本数 | 6984 / 1752 |
| y 训练段 mean / 测试段 mean | 0.001420 / 0.388310 |
| 手写前向 vs `nn.RNN` max&#124;diff&#124; | 1.388e-16 |
| 手写 BPTT vs autograd max&#124;diff&#124; | dW_x 1.776e-15 / dW_h 8.882e-16 |
| 错误示范 A（漏 dh_out）相对误差 | 1.0000（梯度全 0） |
| 错误示范 B（多乘 tanh'）相对误差 | dW_x 0.0891 / dW_h 0.1326 |
| RNN 测试 MAE（标准化空间 / kW） | 0.259707 / 18.86 |
| 朴素持续 MAE / 训练段均值 MAE | 0.414592 / 0.733844 |
| T=160、ρ=0.9 首末梯度比 | 9.615e-13 |
| ρ 扫描（T=160）：0.5 / 0.7 / 0.9 / 0.99 / 1.2 | 2.46e-52 / 2.45e-29 / 9.62e-13 / 4.41e-07 / 1.08e-03 |
| ρ=0.9 实测每步因子（T=40） | 0.753637 |
| ρ=0.9 输入 ×1 / ×5 / ×20 每步因子 | 0.7536 / 0.6041 / 0.4392 |
| ρ=1.2 输入 ×1 / ×5 / ×20 每步因子 | 0.9913 / 0.7948 / 0.5698 |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
