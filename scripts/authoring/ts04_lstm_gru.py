#!/usr/bin/env python3
"""ch04 —— LSTM 与 GRU：门控动机 / 手写前向对账 / 双向与堆叠 / 梯度流对照

本章主线（七条）：

1. **门控动机**：RNN 的 `h` 只有**一条通道**，既当记忆又当输出 —— 用标量类比量出
   「40 步之后还剩多少」（RNN 0.747% vs 加法通路 66.90%）
2. **LSTM 公式推导**：`f / i / o / g` 四门 + cell state 的**加法高速公路**
3. **手写 LSTM 前向（numpy）** 与 `nn.LSTMCell` / `nn.LSTM` **逐值对账**
   （⚠️ PyTorch 的 gate 顺序是 **`i, f, g, o`**，不是 `f, i, o, g`，这是最大的坑）
4. **手写 GRU 前向（numpy）** 与 `nn.GRU` 对账，并踩一次「`n` 门的 `b_hh` 被 reset 门乘着走」
5. **双向与堆叠**：`bidirectional=True` / `num_layers>1` 的形状变化 + 训练对照
6. **梯度流对照**：同一长序列下 RNN vs LSTM 的梯度范数 —— 得到一个**负结果**
   （LSTM 默认初始化反而更差，因为 `f = sigmoid(0) = 0.5`）
7. **本章小结**：门控到底解决了什么，以及它没解决什么

数据：`data/load_curve.csv` 的 A 台区（居民型），2025 全年 8760 个整点（与 ch03 同口径）。
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ts04_lstm_gru.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch04_lstm_gru"

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
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "（本章全程 CPU）")
print("原始行数", len(raw), "| 列", list(raw.columns))
'''

SETUP = '''df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
print("去重排序后", len(df), "行")

# 与 ch03 完全同一份主序列：A 台区（居民型），2025 全年 8760 个整点
a_frame = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳").sort_index()
s_a = a_frame["负荷值"].asfreq("h").loc["2025-01-01":"2025-12-31 23:00"]
print("A 台区整点 %d | 缺失 %d" % (len(s_a), int(s_a.isna().sum())))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
W = 24            # 滑窗长度（与 ch03 一致）
HID = 12          # 隐藏单元数
EPOCHS = 15
BATCH = 128
SEED = 42


def clean_series(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值。"""
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


def make_windows(z, w):
    """一维序列 → 监督矩阵：X[i] = z[i : i+w]，y[i] = z[i+w]。"""
    x = np.lib.stride_tricks.sliding_window_view(z, w)[:-1]
    return x, z[w:]


def make_rnn(rho, hidden=16, seed=1):
    """造一个 W_h 谱半径被钉死在 rho 上的 torch RNN（与 ch03 的 3.5 同口径）。"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = nn.RNN(1, hidden, batch_first=True).double()
    with torch.no_grad():
        w = m.weight_hh_l0
        m.weight_hh_l0.copy_(w * (rho / float(torch.linalg.eigvals(w).abs().max())))
    return m


def make_lstm(f_bias, hidden=16, seed=1):
    """造一个 LSTM，并把**遗忘门**的偏置设成 f_bias（None = 保持默认）。

    bias_ih_l0 的第 2 段是遗忘门 —— 这就是 4.3.1 讲的 gate 顺序 i, f, g, o。
    PyTorch 默认把四门的偏置都初始化在 U(-1/sqrt(H), 1/sqrt(H))，
    所以 f = sigmoid(~0) ~= 0.5：**每过一步，记忆就被砍掉一半**。
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = nn.LSTM(1, hidden, batch_first=True).double()
    if f_bias is not None:
        with torch.no_grad():
            m.bias_ih_l0[hidden:2 * hidden] = f_bias
            m.bias_hh_l0[hidden:2 * hidden] = 0.0
    return m


def last_step_grads(model, length, seed=3):
    """loss 只挂在**最后一步**的输出上，回传得到每个输入步的梯度幅度。

    RNN / LSTM / GRU 都能用：`out, _ = model(x)` 对三者都成立
    （LSTM 的第二个返回值是 (h, c) 元组，用 `_` 接住即可）。
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        x = torch.randn(1, length, 1, dtype=torch.float64, requires_grad=True)
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
print("主序列 %d 点 | 均值 %.6f | 标准差 %.6f" % (len(SERIES), SERIES.mean(), SERIES.std()))

# @@todo 按时间 8:2 切分，标准化参数只取自训练段
# @@hint 和 ch03 完全同一套口径，保证两章的 MAE 可比
cut_pt = int(len(SERIES) * 0.8)
MU = float(SERIES[:cut_pt].mean())
SD = float(SERIES[:cut_pt].std())
# @@end

Z = (SERIES - MU) / SD

# @@todo 切窗 + 按样本切分 + 转成 torch 张量（特征维补成 1）
# @@hint 样本级切分点 = cut_pt - W；张量形状 (N, T, 1)
Xz, yz = make_windows(Z, W)
n_train = cut_pt - W
Xtr, ytr = Xz[:n_train], yz[:n_train]
Xte, yte = Xz[n_train:], yz[n_train:]
Xtr_t = torch.tensor(Xtr.astype("float32")).unsqueeze(-1)
ytr_t = torch.tensor(ytr.astype("float32")).unsqueeze(-1)
Xte_t = torch.tensor(Xte.astype("float32")).unsqueeze(-1)
# @@end

mae_persist = float(np.abs(Xte[:, -1] - yte).mean())
print("训练 %s / 测试 %s | MU %.6f SD %.6f" % (Xtr_t.shape, Xte_t.shape, MU, SD))
print("朴素持续基线（拿窗口末格当预测）MAE %.6f" % mae_persist)

# ---- 验收 ----
assert len(SERIES) == 8760 and n_train == 6984
assert abs(MU - 454.561801) < 1e-6 and abs(SD - 72.623128) < 1e-6
assert tuple(Xtr_t.shape) == (6984, 24, 1) and tuple(Xte_t.shape) == (1752, 24, 1)
assert abs(mae_persist - 0.414592) < 1e-6, "与 ch03 同一份数据、同一个基线"
'''

T2_CODE = '''STEPS = 40

# RNN 式记忆：唯一通道 h 每步都要穿过 tanh
pulse_rnn = [1.0]
for _ in range(STEPS):
    # @@todo 每步 c = tanh(0.9 * c)：0.9 是"衰减"，tanh 是"压缩"
    pulse_rnn.append(float(np.tanh(0.9 * pulse_rnn[-1])))
    # @@end

# LSTM 式记忆：cell state 走加法通路，f 可以学到 0.99、i * g 可以学到 0
pulse_lstm = [1.0]
for _ in range(STEPS):
    # @@todo 加法通路：c = f * c + i * g，取 f = 0.99、i * g = 0
    pulse_lstm.append(float(0.99 * pulse_lstm[-1] + 0.0))
    # @@end

print("第 1 步 t=0 注入一个脉冲 1.0，看它还能剩多少：")
print("  RNN 式（每步过 tanh）   t=1 %.6f  t=5 %.6f  t=40 %.10f"
      % (pulse_rnn[1], pulse_rnn[5], pulse_rnn[40]))
print("  LSTM 式（加法通路）     t=1 %.6f  t=5 %.6f  t=40 %.10f"
      % (pulse_lstm[1], pulse_lstm[5], pulse_lstm[40]))
print("  40 步后的保留率：RNN %.6f%% vs LSTM %.6f%%（差 %.0f 倍）"
      % (100 * pulse_rnn[40], 100 * pulse_lstm[40], pulse_lstm[40] / pulse_rnn[40]))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(np.arange(STEPS + 1), pulse_rnn, marker="o", ms=3, label="RNN 式：c=tanh(0.9c)")
ax.plot(np.arange(STEPS + 1), pulse_lstm, marker="s", ms=3, label="LSTM 式：c=0.99c+i*g")
ax.set_xlabel("时间步")
ax.set_ylabel("记忆强度（初始脉冲 1.0）")
ax.set_title("为什么需要加法通路：40 步之后还剩多少")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

# ---- 验收 ----
assert abs(pulse_rnn[40] - 0.0074665174) < 1e-9, "RNN 式记忆 40 步后只剩 0.75%"
assert abs(pulse_lstm[40] - 0.6689717586) < 1e-9, "0.99 的 40 次方 = 66.90%"
assert pulse_lstm[40] > 50 * pulse_rnn[40]
'''

T3_CODE = '''# 单步对账：B=2 个样本、D=3 维输入、H=4 个隐藏单元
B_L, D_L, H_L = 2, 3, 4

with torch.random.fork_rng():
    torch.manual_seed(5)
    x_step = torch.randn(B_L, D_L, dtype=torch.float64)
    h0 = torch.randn(B_L, H_L, dtype=torch.float64) * 0.5
    c0 = torch.randn(B_L, H_L, dtype=torch.float64) * 0.5
    cell = nn.LSTMCell(D_L, H_L).double()

h_ref, c_ref = cell(x_step, (h0, c0))

# @@todo 把 LSTMCell 的权重搬进 numpy（weight_ih / weight_hh，两个 bias 相加）
Wi_L = cell.weight_ih.detach().numpy().copy()
Wh_L = cell.weight_hh.detach().numpy().copy()
b_L = (cell.bias_ih + cell.bias_hh).detach().numpy().copy()
# @@end

# @@todo 一次算完四个门的预激活，再按 PyTorch 的顺序切成 i / f / g / o
# @@hint 顺序是 i, f, g, o —— 不是 f, i, o, g（见 4.3.1）
gates = x_step.numpy() @ Wi_L.T + h0.numpy() @ Wh_L.T + b_L
g_i, g_f, g_g, g_o = np.split(gates, 4, axis=1)
# @@end

# @@todo 激活：i / f / o 走 sigmoid，候选记忆 g 走 tanh
i_t = 1 / (1 + np.exp(-g_i))
f_t = 1 / (1 + np.exp(-g_f))
o_t = 1 / (1 + np.exp(-g_o))
g_t = np.tanh(g_g)
# @@end

# @@todo 两个状态方程：cell state 走加法，h 是 cell state 过 tanh 后再被 o 门筛一次
c_np = f_t * c0.numpy() + i_t * g_t
h_np = o_t * np.tanh(c_np)
# @@end

print("门的均值：i %.4f | f %.4f | g %+.4f | o %.4f" % (i_t.mean(), f_t.mean(), g_t.mean(), o_t.mean()))
print("LSTMCell 单步：h max|diff| %.3e | c max|diff| %.3e"
      % (np.abs(h_np - h_ref.detach().numpy()).max(), np.abs(c_np - c_ref.detach().numpy()).max()))

# ---- 验收 ----
assert np.abs(h_np - h_ref.detach().numpy()).max() < 1e-12
assert np.abs(c_np - c_ref.detach().numpy()).max() < 1e-12
assert abs(float(f_t.mean()) - 0.4730) < 1e-3, "默认初始化下遗忘门约 0.47 —— 每步丢掉一半记忆"
'''

T4_CODE = '''B, T_, D, H = 2, 6, 3, 5

with torch.random.fork_rng():
    torch.manual_seed(11)
    x_lstm = torch.randn(B, T_, D, dtype=torch.float64)
    lstm = nn.LSTM(D, H, batch_first=True).double()

out_l, (hn_l, cn_l) = lstm(x_lstm)
print("nn.LSTM 的 out %s | h_n %s | c_n %s"
      % (tuple(out_l.shape), tuple(hn_l.shape), tuple(cn_l.shape)))

# @@todo 把权重搬进 numpy
Wi = lstm.weight_ih_l0.detach().numpy().copy()
Wh = lstm.weight_hh_l0.detach().numpy().copy()
b = (lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().numpy().copy()
xn = x_lstm.numpy()
# @@end

# @@todo 初始化两个状态：LSTM 有 h 和 c **两条**独立通道（RNN 只有 h 一条）
h_np_l = np.zeros((B, H))
c_np_l = np.zeros((B, H))
hs_l, cs_l = [], []
# @@end

for t in range(T_):
    # @@todo 四门 + 两个状态方程，逐时间步展开
    g = xn[:, t, :] @ Wi.T + h_np_l @ Wh.T + b
    gi, gf, gg, go = np.split(g, 4, axis=1)
    i_t = 1 / (1 + np.exp(-gi))
    f_t = 1 / (1 + np.exp(-gf))
    o_t = 1 / (1 + np.exp(-go))
    g_t = np.tanh(gg)
    c_np_l = f_t * c_np_l + i_t * g_t
    h_np_l = o_t * np.tanh(c_np_l)
    # @@end
    hs_l.append(h_np_l)
    cs_l.append(c_np_l)

hs_l = np.stack(hs_l).transpose(1, 0, 2)
cs_l = np.stack(cs_l).transpose(1, 0, 2)

d_h = float(np.abs(hs_l - out_l.detach().numpy()).max())
d_c = float(np.abs(cs_l[:, -1] - cn_l.detach().numpy()[0]).max())
print("手写 LSTM 前向 vs nn.LSTM：全部时间步 max|diff| = %.3e" % d_h)
print("手写 c_T vs nn.LSTM 的 c_n：max|diff| = %.3e" % d_c)

# ---- 验收 ----
assert hs_l.shape == (B, T_, H) and cs_l.shape == (B, T_, H)
assert d_h < 1e-12, "双精度下手写 LSTM 前向必须与 nn.LSTM 逐值一致"
assert d_c < 1e-12
assert np.abs(hs_l[:, -1] - hn_l.detach().numpy()[0]).max() < 1e-12
'''

T5_CODE = '''# ---- 错误示范一：按 (f, i, o, g) 切门（听起来更符合公式书写顺序） ----
h_bad_l = np.zeros((B, H))
c_bad_l = np.zeros((B, H))
hs_bad = []
for t in range(T_):
    # @@todo 同样四条公式，但把 gate 顺序写成 (f, i, o, g)
    g = xn[:, t, :] @ Wi.T + h_bad_l @ Wh.T + b
    gf, gi, go, gg = np.split(g, 4, axis=1)
    i_t = 1 / (1 + np.exp(-gi))
    f_t = 1 / (1 + np.exp(-gf))
    o_t = 1 / (1 + np.exp(-go))
    g_t = np.tanh(gg)
    c_bad_l = f_t * c_bad_l + i_t * g_t
    h_bad_l = o_t * np.tanh(c_bad_l)
    # @@end
    hs_bad.append(h_bad_l)
hs_bad = np.stack(hs_bad).transpose(1, 0, 2)

# ---- 错误示范二：只把 i 与 f 换位，其余全对（最隐蔽的一种写法） ----
h_sw = np.zeros((B, H))
c_sw = np.zeros((B, H))
hs_sw = []
for t in range(T_):
    # @@todo 切门顺序是对的，但把 sigmoid 结果对调了：i 用了 f 的预激活，反之亦然
    g = xn[:, t, :] @ Wi.T + h_sw @ Wh.T + b
    gi, gf, gg, go = np.split(g, 4, axis=1)
    i_t = 1 / (1 + np.exp(-gf))
    f_t = 1 / (1 + np.exp(-gi))
    o_t = 1 / (1 + np.exp(-go))
    g_t = np.tanh(gg)
    c_sw = f_t * c_sw + i_t * g_t
    h_sw = o_t * np.tanh(c_sw)
    # @@end
    hs_sw.append(h_sw)
hs_sw = np.stack(hs_sw).transpose(1, 0, 2)

d_bad = float(np.abs(hs_bad - out_l.detach().numpy()).max())
d_sw = float(np.abs(hs_sw - out_l.detach().numpy()).max())
print("错误顺序 (f,i,o,g)：h max|diff| = %.3e" % d_bad)
print("只把 i / f 对调  ：h max|diff| = %.3e" % d_sw)

# ---- 验收 ----
assert d_bad > 0.3, "顺序错了，结果差得离谱，而且不会报任何错"
assert d_sw > 0.1, "只对调两个门的 sigmoid，一样是错的"
assert np.abs(hs_bad[:, -1] - hs_sw[:, -1]).max() > 1e-3, "两种错法错得还不一样"
'''

T6_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(12)
    x_gru = torch.randn(B, T_, D, dtype=torch.float64)
    gru = nn.GRU(D, H, batch_first=True).double()

out_g, hn_g = gru(x_gru)
print("nn.GRU 的 out %s | h_n %s（GRU 只有 h，没有 c）"
      % (tuple(out_g.shape), tuple(hn_g.shape)))

# @@todo 按 PyTorch 的 (r, z, n) 三段切开输入权重、循环权重，以及**两个**偏置
# @@hint np.split(张量, 3, axis=0)，偏置按 3 段切
Wir, Wiz, Win = np.split(gru.weight_ih_l0.detach().numpy(), 3, axis=0)
Whr, Whz, Whn = np.split(gru.weight_hh_l0.detach().numpy(), 3, axis=0)
bir, biz, bin_ = np.split(gru.bias_ih_l0.detach().numpy(), 3)
bhr, bhz, bhn = np.split(gru.bias_hh_l0.detach().numpy(), 3)
xng = x_gru.numpy()
# @@end

# @@todo 初始化隐状态（GRU 只有一条通道）
h_np_g = np.zeros((B, H))
hs_g = []
# @@end

for t in range(T_):
    # @@todo 三段公式：重置门 r（两个偏置都加）、更新门 z（同）、候选 n（b_hh 被 r 乘着）
    # @@hint n 门只直连 b_ih 那一段；b_hh 那一段在括号里，被 r_t 乘一遍（见 4.4.1）
    xt = xng[:, t, :]
    r_t = 1 / (1 + np.exp(-(xt @ Wir.T + bir + h_np_g @ Whr.T + bhr)))
    z_t = 1 / (1 + np.exp(-(xt @ Wiz.T + biz + h_np_g @ Whz.T + bhz)))
    n_t = np.tanh(xt @ Win.T + bin_ + r_t * (h_np_g @ Whn.T + bhn))
    h_np_g = (1 - z_t) * n_t + z_t * h_np_g
    # @@end
    hs_g.append(h_np_g)

hs_g = np.stack(hs_g).transpose(1, 0, 2)
d_gru = float(np.abs(hs_g - out_g.detach().numpy()).max())
print("手写 GRU 前向 vs nn.GRU：max|diff| = %.3e" % d_gru)
print("手写 h_T vs nn.GRU 的 h_n：max|diff| = %.3e"
      % np.abs(hs_g[:, -1] - hn_g.detach().numpy()[0]).max())

# ---- 验收 ----
assert d_gru < 1e-12, "双精度下手写 GRU 前向必须与 nn.GRU 逐值一致"
assert hs_g.shape == (B, T_, H)
'''

T7_CODE = '''# ---- 错误示范：以为"三个门都能直接加 b_ih + b_hh" ----
allb = (gru.bias_ih_l0 + gru.bias_hh_l0).detach().numpy()
h_bad_g = np.zeros((B, H))
hs_bad_g = []
for t in range(T_):
    # @@todo 用"三个门对称"的写法：r / z / n 都直接加 bi + bh 的对应段
    # @@hint 这是错的 —— 复现它，别改对
    xt = xng[:, t, :]
    r_t = 1 / (1 + np.exp(-(xt @ Wir.T + allb[:H] + h_bad_g @ Whr.T)))
    z_t = 1 / (1 + np.exp(-(xt @ Wiz.T + allb[H:2 * H] + h_bad_g @ Whz.T)))
    n_t = np.tanh(xt @ Win.T + allb[2 * H:] + h_bad_g @ Whn.T)
    h_bad_g = (1 - z_t) * n_t + z_t * h_bad_g
    # @@end
    hs_bad_g.append(h_bad_g)
hs_bad_g = np.stack(hs_bad_g).transpose(1, 0, 2)

d_bad_g = float(np.abs(hs_bad_g - out_g.detach().numpy()).max())
print("n 门的 b_hh 处理错：max|diff| = %.4f" % d_bad_g)
print("偏差达到输出量级的 %.1f%% —— 又是一个不报错、只是悄悄变差的写法"
      % (100 * d_bad_g / float(np.abs(out_g.detach().numpy()).max())))

# ---- 验收 ----
assert d_bad_g > 0.05, "n 门的 b_hh 必须被 r 门乘着走"
assert abs(d_bad_g - 0.1552) < 5e-3
'''

T8_CODE = '''SHAPE_CASES = [(1, False), (2, False), (1, True), (2, True)]
shape_l = {}
shape_g = {}
x_shape = torch.zeros(4, 10, 1, dtype=torch.float64)

for nl, bd in SHAPE_CASES:
    # @@todo 造 LSTM(num_layers=nl, bidirectional=bd)，记录 out / h / c 的形状
    # @@hint out 永远是 (B, T, H * 方向数)；h / c 是 (层数 * 方向数, B, H)
    m_l = nn.LSTM(1, 8, num_layers=nl, batch_first=True, bidirectional=bd).double()
    o_l, (h_l, c_l) = m_l(x_shape)
    shape_l[(nl, bd)] = (tuple(o_l.shape), tuple(h_l.shape), tuple(c_l.shape))
    # @@end

    # @@todo 同样的 GRU（注意 GRU 没有 c）
    m_g = nn.GRU(1, 8, num_layers=nl, batch_first=True, bidirectional=bd).double()
    o_g, h_g = m_g(x_shape)
    shape_g[(nl, bd)] = (tuple(o_g.shape), tuple(h_g.shape))
    # @@end

print("输入 (4, 10, 1)，隐藏 8：")
for nl, bd in SHAPE_CASES:
    print("  layers=%d bidir=%-5s | LSTM out %s h %s c %s | GRU out %s h %s"
          % (nl, bd, shape_l[(nl, bd)][0], shape_l[(nl, bd)][1], shape_l[(nl, bd)][2],
             shape_g[(nl, bd)][0], shape_g[(nl, bd)][1]))

# ---- 验收 ----
assert shape_l[(1, False)] == ((4, 10, 8), (1, 4, 8), (1, 4, 8))
assert shape_l[(2, True)] == ((4, 10, 16), (4, 4, 8), (4, 4, 8))
assert shape_g[(2, True)] == ((4, 10, 16), (4, 4, 8))
assert shape_l[(1, True)][1][0] == 2, "双向: 第 0 维 = 层数 x 2"
assert shape_l[(2, False)][0][2] == 8, "堆叠层数不改隐藏维，只改 h 的第 0 维"
'''

T9_CODE = '''class SeqRegressor(nn.Module):
    """只换循环层（RNN / LSTM / GRU），其余完全一致。"""

    def __init__(self, kind="lstm", hidden=HID, layers=1, bidir=False):
        super().__init__()
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.r = cls(1, hidden, num_layers=layers, batch_first=True, bidirectional=bidir)
        self.fc = nn.Linear(hidden * (2 if bidir else 1), 1)

    def forward(self, x):
        out, _ = self.r(x)
        return self.fc(out[:, -1, :])


def train_regressor(kind, **kw):
    """统一口径：同一个 seed、同一组超参，只换循环层。"""
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        model = SeqRegressor(kind, **kw)
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
        model.eval()
        with torch.no_grad():
            # @@todo 测试段前向，并把 (N, T, 1) 压平成 (N,)
            pred = model(Xte_t).numpy().ravel()
            # @@end
    # @@todo 返回测试段 MAE（标准化空间）
    mae_z = float(np.abs(pred - yte).mean())
    # @@end
    return mae_z


CFG = [("rnn", {}), ("lstm", {}), ("gru", {}), ("lstm", {"bidir": True})]
mae_table = {}
for kind, kw in CFG:
    # @@todo 按统一口径训练一个模型，把 MAE 记进 mae_table
    mae_table[(kind, tuple(sorted(kw.items())))] = train_regressor(kind, **kw)
    # @@end

print("朴素持续基线 MAE %.6f（%.2f kW）" % (mae_persist, mae_persist * SD))
for kind, kw in CFG:
    m = mae_table[(kind, tuple(sorted(kw.items())))]
    print("%-5s %-24s 测试 MAE %.6f（%.2f kW）  相对朴素 %+.1f%%"
          % (kind, kw, m, m * SD, 100 * (m / mae_persist - 1)))

# ---- 验收 ----
assert abs(mae_table[("rnn", ())] - 0.259707) < 1e-4
assert abs(mae_table[("lstm", ())] - 0.215303) < 1e-4
assert abs(mae_table[("gru", ())] - 0.223200) < 1e-4
assert abs(mae_table[("lstm", (("bidir", True),))] - 0.225713) < 1e-4
assert all(v < mae_persist for v in mae_table.values()), "四个模型都要打赢朴素基线"
'''

T10_CODE = '''T_LONG = 160          # 与 ch03 的 3.5 同口径：序列长度 160

# @@todo 同一口径量两个模型的「第 1 步 / 第 T 步」梯度比
# @@hint RNN 用 make_rnn(0.9)（谱半径 0.9）；LSTM 用 make_lstm(None)（PyTorch 默认初始化）
g_rnn = last_step_grads(make_rnn(0.9), T_LONG)
g_lstm = last_step_grads(make_lstm(None), T_LONG)
ratio_rnn = float(g_rnn[0] / g_rnn[-1])
ratio_lstm = float(g_lstm[0] / g_lstm[-1])
# @@end

print("T=%d：RNN(rho=0.9) 首末比 %.3e" % (T_LONG, ratio_rnn))
print("T=%d：LSTM(默认初始化) 首末比 %.3e" % (T_LONG, ratio_lstm))
print("差 %.1f 个数量级 —— 但方向是**反的**：门控默认配置下更差！"
      % (np.log10(ratio_rnn) - np.log10(ratio_lstm)))

# ---- 验收 ----
assert abs(np.log10(ratio_rnn) - (-12.017)) < 0.5, "与 ch03 的 9.615e-13 对得上"
assert ratio_lstm < ratio_rnn, "负结果：LSTM 默认初始化下梯度衰减比 RNN 更快"
assert np.log10(ratio_lstm) < -30
'''

T11_CODE = '''F_BIASES = [None, 1.0, 2.0, 3.0]

ratio_f = {}
for fb in F_BIASES:
    # @@todo 造一个遗忘门偏置为 fb 的 LSTM，量 T=160 上的首末梯度比
    # @@hint 脚手架 make_lstm(f_bias)，None 表示保持 PyTorch 默认初始化
    g_f = last_step_grads(make_lstm(fb), T_LONG)
    ratio_f[fb] = float(g_f[0] / g_f[-1])
    # @@end

for fb in F_BIASES:
    r = ratio_f[fb]
    print("遗忘门偏置 %-5s → 首末比 %.3e   平均每步 x%.6f"
          % (str(fb), r, r ** (1 / (T_LONG - 1))))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
labels = ["默认\\n(f=0.5)", "f 偏置=1\\n(f=0.73)", "f 偏置=2\\n(f=0.88)", "f 偏置=3\\n(f=0.95)"]
axes[0].bar(np.arange(4), [np.log10(ratio_f[fb]) for fb in F_BIASES], color="tab:green")
axes[0].axhline(np.log10(ratio_rnn), color="tab:blue", ls="--", lw=1.6,
                label="RNN(rho=0.9) 参照线")
axes[0].axhline(np.log10(ratio_lstm), color="tab:red", ls=":", lw=1.6,
                label="LSTM 默认 参照线")
axes[0].set_xticks(np.arange(4))
axes[0].set_xticklabels(labels)
axes[0].set_ylabel("log10(第 1 步 / 第 T 步 梯度比)")
axes[0].set_title("遗忘门越接近 1，梯度衰减越慢（T=160）")
axes[0].legend(fontsize=8)
axes[1].plot(np.arange(T_LONG), np.log10(last_step_grads(make_lstm(3.0), T_LONG)), lw=1.6,
             label="LSTM f 偏置=3")
axes[1].plot(np.arange(T_LONG), np.log10(last_step_grads(make_rnn(0.9), T_LONG)), lw=1.6,
             label="RNN rho=0.9")
axes[1].set_xlabel("输入步序号（0 = 最早）")
axes[1].set_ylabel("log10(梯度幅度)")
axes[1].set_title("逐步梯度曲线：LSTM 的尾部是一条斜线，RNN 是断崖")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()

# ---- 验收 ----
assert ratio_f[3.0] > ratio_f[2.0] > ratio_f[1.0] > ratio_f[None], \\
    "遗忘门偏置越大（f 越接近 1），越能保住远期梯度"
assert abs(np.log10(ratio_f[3.0]) - (-0.5223)) < 0.5
assert ratio_f[3.0] > 1e10 * ratio_lstm, "把 f 顶上去之后，比默认配置好了 10 个数量级以上"
'''

LESSON = [
    md(
        """
# ch04 LSTM 与 GRU（讲解版）

> **本节考点**：RNN 的记忆被 `tanh` 夹住了（ch03 的 3.5 已经量过），门控就是来拆这堵墙的。
> 本章要做三件事：**手推公式**、**逐值对账**、**亲眼看到门控到底管不管用**
> —— 最后一条的答案会让你意外。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 模型训练 | **技能操作 40%** | `nn.LSTM` / `nn.GRU` 的 gate 顺序、形状、双向与堆叠 |
| 模型原理 | 理论 / 答辩 | 四门公式、cell state 的加法通路、梯度流 |
| 新能源功率 / 负荷预测 | 行业赛题 | 与 ch03 同口径的 MAE 对照表 |

**与 `03_nlp/ch04` 的分工**：那边讲 Transformer / 自注意力；
本章停在门控 RNN，重点是**PyTorch 的 gate 顺序坑**与**梯度流的定量对照**。

## 学习目标

1. 说清 RNN 的 `h` 为什么"一条通道不够用"，并用一次标量实验量化它
2. 默写 LSTM 的四门公式与两个状态方程，并**记住 PyTorch 的顺序是 `i, f, g, o`**
3. 手写 LSTM / GRU 前向，与 `nn.LSTMCell` / `nn.LSTM` / `nn.GRU` **逐值对到 1e-16**
4. 说清 `bidirectional` / `num_layers` 对 `out` / `h` / `c` 三个形状的影响
5. 完成一次**梯度流对照实验**，并解释那个反直觉的负结果
6. 知道 LSTM 的"记忆开关"在哪：**遗忘门偏置**（`bias_ih_l0[H:2H]`）

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `nn.LSTMCell(in, hidden)` | — | module | 单步；权重 `weight_ih (4H, D)` / `weight_hh (4H, H)` |
| `nn.LSTM(in, hidden, batch_first=)` | `num_layers` / `bidirectional` | module | 返回 `(out, (h_n, c_n))` |
| `nn.GRU(in, hidden, batch_first=)` | 同上 | module | 返回 `(out, h_n)` —— **没有 c** |
| `weight_ih_l0` | — | `(4H, D)` / `(3H, D)` | 按 gate 顺序分段：LSTM `i,f,g,o`；GRU `r,z,n` |
| `bias_ih_l0` / `bias_hh_l0` | — | `(4H,)` / `(3H,)` | LSTM 四门**都**加两者；GRU 的 `n` 门不加直连的 `b_hh` |
| `out` | — | `(B, T, H*方向数)` | 每个时间步的隐状态（双向时**拼**在一起） |
| `h_n` / `c_n` | — | `(层数*方向数, B, H)` | 第 0 维不是批；双向时方向在最后一维拼 |
| `torch.linalg.eigvals(W)` | — | complex | 谱半径，ch03 用它钉住 `W_h` |

## 学习路径提示

本章的 10 个任务里，**任务 3 / 4 / 5 / 6 / 7 是硬骨头**（手写对账），
其余是"看得懂就行"的量化实验。建议先把 4.3 / 4.4 的公式抄一遍再动手写。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md(
        """
## 4.1 门控动机：`h` 只有一条通道

回顾 ch03 的结论：普通 RNN 的状态更新只有一条通路

```
h_t = tanh( x_t W_x + h_{t-1} W_h + b_h )
```

这条通路同时要干三件事：

1. **当记忆**（要长期不变地保存信息 → 需要 `W_h` 接近单位阵）
2. **当输出**（要随输入剧烈变化 → 需要 `W_h` 敏感）
3. **当非线性**（每次都要过 `tanh`，值域被压在 `(−1, 1)`，还会饱和）

三件事互相打架，**这就是同一个 `h` 一条通道的原罪**。

### 量化一下：40 步之后还剩多少

用一个**标量类比**把这件事量出来（`h` 的每个分量都是同一个方程）：

| 版本 | 递推 | 含义 |
|---|---|---|
| RNN 式 | `c = tanh(0.9 c)` | 每步都过 `tanh`（压缩 + 衰减一次做完） |
| LSTM 式 | `c = 0.99 c + i·g` | **加法通路**：只要门说"别丢"，`c` 就原样留着 |

实测（t=0 注入脉冲 1.0）：

| 时间步 | t=1 | t=5 | t=40 |
|---|---|---|---|
| RNN 式 | 0.716298 | 0.345338 | **0.0074665174** |
| LSTM 式 | 0.990000 | 0.950990 | **0.6689717586** |
| 保留率 | — | — | **0.747% vs 66.90%** |

**差约 90 倍**。而且注意 LSTM 这一版里没有任何"魔法"：
它只是把「记忆」放到了一个**不被 `tanh` 挤压**的通道上，
让 `f` 与 `i` 去决定「留多少、写多少」。

### 4.1.1 难点深挖：为什么"把 `W_h` 调大"永远救不了 RNN

ch03 的 3.5.2 已经量过：ρ = 1.2 时每步因子在小输入下能到 0.9913，
但输入一大就掉到 0.5698。原因是 `tanh` 的**饱和**：

```
h 大 → tanh'(a) = 1 - h² 小 → 梯度衰减快
h 小 → tanh' ≈ 1，但记忆本身也小，携带不了信息
```

这是**同一个变量同时承载"信息量"和"梯度通道"**导致的死结。

> **门控的本质，是把这两件事拆开**：
> - `h` 继续当**输出**（要能剧烈变化，所以留着 `tanh`）
> - 新开一条 `c` 当**记忆**（走加法，不受 `tanh` 挤压）
> - 用 `f` / `i` / `o` 三个门去**调节**两条通道之间的流量
>
> 所以 LSTM 的参数是 RNN 的 **4 倍**（`4H(H+D)` 对 `H(H+D)`）——
> 这不是浪费，是**买了另一条物理通道**。
"""
    ),
    code(T1_CODE),
    code(T2_CODE),
    md(
        """
## 4.2 LSTM 公式推导

LSTM 有**两个状态**（`h` 与 `c`）、**四个门**。先看公式：

```
f_t = σ( x_t W_if + h_{t-1} W_hf + b_f )     遗忘门：旧记忆留多少
i_t = σ( x_t W_ii + h_{t-1} W_hi + b_i )     输入门：新记忆写多少
g_t = tanh( x_t W_ig + h_{t-1} W_hg + b_g )  候选记忆：写什么内容
o_t = σ( x_t W_io + h_{t-1} W_ho + b_o )     输出门：这次往外露多少

c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t              ← cell state 的**加法高速公路**
h_t = o_t ⊙ tanh(c_t)                        ← h 只是 c 的一次"曝光"
```

四个门的分工，用一句话记：

| 门 | 激活 | 作用 | 值域 |
|---|---|---|---|
| `f` 遗忘 | sigmoid | `c_{t-1}` 保留比例 | 0~1 |
| `i` 输入 | sigmoid | 新内容写入门 | 0~1 |
| `g` 候选 | **tanh** | 新内容本身（可正可负） | −1~1 |
| `o` 输出 | sigmoid | `c_t` 对外曝光比例 | 0~1 |

**关键在 `c_t = f ⊙ c_{t-1} + i ⊙ g` 这一行**：
它是整个序列上唯一的**加法**运算，梯度沿它回传时只需要乘 `f`，
**不需要穿 `tanh`、不需要乘 `W_h`**。

> 这就是「加法高速公路」这个说法的全部含义。
> 本书 ch03 3.5.2 里那两堵墙（`tanh' ≤ 1` 与饱和），在 `c` 这条路上都不存在。

**参数排布**（PyTorch）：

| 属性 | 形状（单层） | 分段顺序 |
|---|---|---|
| `weight_ih_l0` | `(4H, D)` | **`i`, `f`, `g`, `o`** |
| `weight_hh_l0` | `(4H, H)` | 同上 |
| `bias_ih_l0` / `bias_hh_l0` | `(4H,)` | 同上 |
| 总偏置 | — | `b_ih + b_hh`（**四个门都加**） |

## 4.3 手写 LSTM 前向并对账

先做**单步**（和 `nn.LSTMCell` 对），再做**整条序列**（和 `nn.LSTM` 对）。
单步对了，整条序列就只是加一个 `for` 循环 —— 但如果 gate 顺序错了，
单步就会对不上，这正是我们想要的"早期失败"。

实测：

| 对账对象 | 量 | max\\|diff\\| |
|---|---|---|
| `nn.LSTMCell` | `h` | 5.55e-17 |
| `nn.LSTMCell` | `c` | 5.55e-17 |
| `nn.LSTM` | 全部时间步 `h` | 5.55e-17 |
| `nn.LSTM` | `c_n` | 1.11e-16 |

### 4.3.1 难点深挖：PyTorch 的 gate 顺序是 `i, f, g, o`

**为什么会写错**：几乎所有教材都按 `f, i, o, g`（或者 `i, f, o, g`）的顺序讲公式，
因为那个顺序更符合叙述逻辑（"先决定忘什么、再决定记什么"）。
但 PyTorch 的实现顺序是 **`i, f, g, o`** —— 而且**文档里只在一张表里提了一句**。

实测（同一份权重，只改切门顺序）：

| 切门顺序 | `h` 的 max\\|diff\\| |
|---|---|
| **`i, f, g, o`（正确）** | **5.55e-17** |
| `f, i, o, g`（把四段全错位） | **4.74e-01** |
| 只把 `i` 与 `f` 对调 | **1.23e-01** |

注意最后一行：**只对调两个门，其它全对**，结果照样错——而且**不报任何错**。
一个"看起来完全合理"的实现，会在训练时给你一条平缓下降但永远达不到最优的 loss 曲线。

```python
gi, gf, gg, go = np.split(gates, 4, axis=1)      # ← i, f, g, o，不能改
```

> **记忆口诀**：**i 在前，f 在第二，g 是候选，o 收尾**。
> 写代码前先用 `weight_ih_l0.shape == (4H, D)` 确认"四段"，再确认"段序"。
> 更稳的做法：与 `nn.LSTMCell` 对一次单步 —— 顺序错了立刻 5e-17 变 4e-01。

### 4.3.2 GRU 的 gate 顺序是 `r, z, n`

同一类坑，GRU 的顺序是 **reset, update, new**：

```
r_t = σ( x_t W_ir + b_ir + h_{t-1} W_hr + b_hr )
z_t = σ( x_t W_iz + b_iz + h_{t-1} W_hz + b_hz )
n_t = tanh( x_t W_in + b_in + r_t ⊙ ( h_{t-1} W_hn + b_hn ) )
h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

GRU 的设计取舍（与 LSTM 对照）：

| | LSTM | GRU |
|---|---|---|
| 状态通道 | `h` + `c`（两条） | `h`（一条，但用 `z` 在"新"与"旧"之间插值） |
| 门数量 | 4（`i f g o`） | 3（`r z n`） |
| 参数（单层，in=D，hid=H） | `4H(H+D)+4H` … 约 `4H²` | 约 `3H²` |
| gate 段序 | `i, f, g, o` | `r, z, n` |
| 额外偏置坑 | 无（四门都加 `b_ih+b_hh`） | **`n` 门只直连 `b_ih`**（见 4.4.1） |
"""
    ),
    code(T3_CODE),
    code(T4_CODE),
    code(T5_CODE),
    md(
        """
## 4.4 手写 GRU 前向

`h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}` —— 一个**线性插值**：
`z` 靠近 1 就保留旧状态，靠近 0 就接受新候选。

对账结果：`nn.GRU` 全部时间步 `h` 的 **max|diff| = 1.11e-16**。

### 4.4.1 难点深挖：`n` 门的 `b_hh` 被 reset 门乘着走

这一条比 gate 顺序更隐蔽。看公式的括号：

```
n_t = tanh( x_t W_in + b_in + r_t ⊙ ( h_{t-1} W_hn + b_hn ) )
                                            ↑ 这一项在括号里
```

也就是说：`b_hn` **不是**直接加进预激活的，它要先乘上 `r_t`。
而 `r` 与 `z` 两个门是**直接**加的（`b_ir + b_hr`、`b_iz + b_hz`）。

于是"三个门对称处理"这个最自然的写法就错了：

```python
b_all = bias_ih_l0 + bias_hh_l0
r = sigmoid(x @ Wir.T + b_all[:H]      + h @ Whr.T)   # ✓ 对的
z = sigmoid(x @ Wiz.T + b_all[H:2H]    + h @ Whz.T)   # ✓ 对的
n = tanh(   x @ Win.T + b_all[2H:]     + h @ Whn.T)   # ✗ 错了：b_hh 那一段应被 r 乘
```

实测 `max|diff| = 0.1552` —— 已经达到输出量级的 **27.6%**。
这种错法的表现是：**训练能跑、loss 会降、指标比 RNN 好一点但比正确的 GRU 差**。
如果你不写出"逐值对账"这一步，你永远不会知道自己的 GRU 是错的。

> **判别口诀**：谁在括号里，谁就要被乘。
> LSTM 四个门都是 `x W + h W + b_ih + b_hh`，全都"直连"；
> GRU 只有 `r` / `z` 是直连，`n` 的 `b_hh` 在括号里。
"""
    ),
    code(T6_CODE),
    code(T7_CODE),
    md(
        """
## 4.5 双向与堆叠

两个超参，改的**只有形状**，不改数学：

| 参数 | 效果 | 代价 |
|---|---|---|
| `bidirectional=True` | 反向再跑一遍，两个方向的 `h` **在最后一维拼起来** | 参数 ×2，**且不能用于在线预测**（要用到未来） |
| `num_layers=2` | 上一层的 `out` 当下一层的输入 | 参数 ×2，`h` / `c` 的第 0 维变成层数 |

### 形状速查（输入 `(4, 10, 1)`，hidden = 8）

| 配置 | `out` | `h_n` | `c_n` |
|---|---|---|---|
| `layers=1, bidir=False` | `(4, 10, 8)` | `(1, 4, 8)` | `(1, 4, 8)` |
| `layers=2, bidir=False` | `(4, 10, 8)` | `(2, 4, 8)` | `(2, 4, 8)` |
| `layers=1, bidir=True` | `(4, 10, **16**)` | `(2, 4, 8)` | `(2, 4, 8)` |
| `layers=2, bidir=True` | `(4, 10, **16**)` | `(4, 4, 8)` | `(4, 4, 8)` |

两条规律：

1. **`out` 的最后一维 = `H × 方向数`**（双向时变 2 倍），而 `h_n` / `c_n` 的最后一维**始终是 `H`**（方向在倒数第二维前面拼）
2. **`h_n` / `c_n` 的第 0 维 = `层数 × 方向数`**，**不是批大小**

所以接线性头时：

```python
out, _ = self.r(x)
self.fc(out[:, -1, :])       # 双向时 out[:, -1, :] 是 (B, 2H)，正好喂给 Linear(2H, 1)
```

### 训练对照（同一数据、同一 seed、同一超参，只换循环层）

| 模型 | 测试 MAE（标准化） | kW | 相对朴素基线 |
|---|---|---|---|
| 朴素持续 | 0.414592 | 30.11 | — |
| RNN | 0.259707 | 18.86 | −37.4% |
| **LSTM** | **0.215303** | **15.64** | −48.1% |
| GRU | 0.223200 | 16.21 | −46.2% |
| LSTM（双向） | 0.225713 | 16.39 | −45.6% |

**必须诚实地说**：门控带来的提升是**真实的，但不大**（0.2597 → 0.2153，约 17%）。
原因不神秘：

- 这一步任务只要求「看过去 24 小时，猜下一小时」。**24 步根本不算长依赖**，
  ch03 实测 T=24 时梯度衰减还很轻微；
- 日周期极强，`lag24` 本身就包了大部分信息；
- 双向在这里甚至**略差**（0.2257）—— 因为双向的后向分支看的是"未来"，而
  单步预测的未来信息对降低当前 MAE 帮助有限，反而多了一倍的参数量去拟合噪声。

> **结论**：门控不是免费的午餐。**先量一量你的序列到底有多长依赖**，
> 再决定要不要上 LSTM/GRU —— 短序列上两者的差距可能不到 5%。
"""
    ),
    code(T8_CODE),
    code(T9_CODE),
    md(
        """
## 4.6 梯度流对照：RNN vs LSTM

用 ch03 3.5 的同口径（`loss` 只挂最后一步、`T = 160`、`第 1 步 / 第 T 步`）：

| 模型 | 首末梯度比 | 平均每步因子 |
|---|---|---|
| RNN（ρ = 0.9） | 9.615e-13 | 0.8403 |
| **LSTM（PyTorch 默认初始化）** | **3.952e-34** | **0.6165** |

### 4.6.1 难点深挖：门控默认配置下**反而更差**（一个负结果）

这张表是本章最重要的发现，也是最反直觉的：

> **加了门 ≠ 梯度就能传得远。**

原因在 `f` 的初始化。PyTorch 把 LSTM 四个门的偏置都初始化成
`U(-1/√H, 1/√H)`，所以初始时刻

```
f = σ(b_f) ≈ σ(0) = 0.5
```

`c_t = f ⊙ c_{t-1} + ...` —— 每走一步，旧记忆就**被砍掉一半**。
`0.5^159` 是一个 48 位数，比 RNN 的 `0.84^159` 还小得多。

**门控给的是一条"可以调"的通道，不是一条"自动很快"的通道。**
真正决定它快慢的，是 `f` 学到的值。既然默认 `f = 0.5` 太狠，
工程上就有两个常规做法：

1. **把遗忘门的偏置直接设大**（"forget gate bias = 1~2"，2015 年之后的标准 trick）
2. 让网络在训练中学出来（但长序列上梯度太小，学不动 —— 鸡生蛋问题）

实测（`T = 160`，只改遗忘门偏置）：

| 遗忘门偏置 | `f = σ(bias)` | 首末梯度比 | 平均每步因子 |
|---|---|---|---|
| 默认（≈0） | 0.500 | 3.952e-34 | 0.6165 |
| 1.0 | 0.731 | 3.218e-12 | 0.8467 |
| 2.0 | 0.881 | 1.053e-03 | 0.9578 |
| **3.0** | **0.953** | **3.004e-01** | **0.9925** |

偏置设到 3 时，160 步之后梯度还剩 **30%** —— 而 RNN 只剩 `9.6e-13`，
**差了 11.5 个数量级**。

> **三条可以背下来的结论**：
> 1. **门控不自动解决梯度消失**。默认 LSTM 甚至比 RNN 更差（3.95e-34 vs 9.62e-13）。
> 2. **LSTM 的关键不是"有门"，而是"`c` 那条加法通路上没有 `tanh`"** ——
>    所以 `f` 可以一路学到接近 1，把记忆原样保留。RNN 的 `W_h` 做不到这件事，
>    因为它的每一次传递都要穿 `tanh`（ch03 3.5.1 的 0.9 → 0.7536）。
> 3. **实践建议**：长序列任务上，`nn.LSTM` 的遗忘门偏置先手动设成 **1.0** 起步，
>    不满意再往上试。这是零成本、几乎无副作用的改动。
>
> 代码上就是一行：
> ```python
> m.bias_ih_l0[H:2 * H] = 1.0     # H:2H 这一段 = 遗忘门（顺序 i, f, g, o）
> m.bias_hh_l0[H:2 * H] = 0.0
> ```
> ⚠️ 注意 `H:2H` 这个切片**同时**依赖"四段"和"段的顺序"两个知识 ——
> 写错了不会报错，只会让 `i` 门或 `g` 门的偏置被改掉。
"""
    ),
    code(T10_CODE),
    code(T11_CODE),
    md(
        """
## 本章小结

**一句话**：LSTM/GRU 做的事，是把 RNN 里"记忆"和"输出"这两件打架的事，
拆到**两条通道**上，再用门去调节它们之间的流量。

### 五件事，按重要性排

```
① c 的加法通路      c_t = f ⊙ c_{t-1} + i ⊙ g —— 全程唯一不穿 tanh 的路径
② 门控顺序坑        PyTorch: LSTM = i, f, g, o   GRU = r, z, n
③ 偏置的加与乘      LSTM 四门都加 b_ih+b_hh；GRU 的 n 门 b_hh 在括号里（被 r 乘）
④ 形状              out (B,T,H*方向数)；h/c (层数*方向数, B, H)
⑤ 遗忘门偏置        默认 f=σ(0)=0.5，长序列上要手动顶上去（1.0 起步）
```

### 结论对照表：RNN vs LSTM vs GRU

| 维度 | RNN | LSTM | GRU |
|---|---|---|---|
| 状态通道 | `h` | `h` + `c` | `h`（`z` 做插值） |
| 能否保住远期梯度 | 不能（`tanh'` 夹住） | **能**（`f` 可学到 ≈1） | 能（`z` 同理） |
| 默认配置下 T=160 首末比 | 9.615e-13 | **3.952e-34** | — |
| 调好之后 T=160 首末比 | 9.615e-13（没法调） | **3.004e-01**（`f` 偏置=3） | — |
| 参数（单层，in=D，hid=H） | `H(H+D)+H` | `4H(H+D)+4H` | `3H(H+D)+3H` |
| 本章任务测试 MAE | 0.259707 | **0.215303** | 0.223200 |
| 适合的序列长度 | 短（< 30） | 长 | 长（更省参数） |

### 易错点清单

1. **PyTorch 的 LSTM gate 顺序是 `i, f, g, o`**，写成 `f, i, o, g` 后 `h` 差 4.74e-01，且不报错。
2. **只对调 `i` 与 `f` 一样错**（差 1.23e-01）—— 错的不是"顺序"这个概念，是每一个具体位置。
3. **GRU 的段序是 `r, z, n`**（reset / update / new），不是 `z, r, n`。
4. **GRU 的 `n` 门不加直连的 `b_hh`**：它在括号里，要被 `r_t` 乘（写错差 0.1552）。
5. **LSTM 四个门都加 `b_ih + b_hh`**；别把 GRU 的习惯带过来。
6. **`out` 的最后一维 = `H × 方向数`**；`h_n` / `c_n` 的最后一维**永远是 `H`**。
7. **`h_n` / `c_n` 第 0 维 = `层数 × 方向数`**，不是批大小。
8. **GRU 没有 `c_n`**（只有一个返回的状态），拿 `out, (h, c) = gru(x)` 会解包失败。
9. **双向不能用于在线预测**：后向分支要看未来，实时场景里那些值还不存在。
10. **门控 ≠ 自动解决梯度消失**：LSTM 默认 `f = 0.5`，T=160 首末比 3.952e-34，比 RNN 更差。
11. **遗忘门偏置的切片是 `bias_ih_l0[H:2*H]`** —— 依赖"四段 + i,f,g,o 顺序"两个知识。
12. **门控的收益不总是大**：本章 24 步任务上 LSTM 只比 RNN 好 17%，双向甚至略差。
13. **对账用 double**：`float32` 只有 1e-7 精度，根本分不清"对"和"差一点"。
14. **随机性一律包 `fork_rng` + `manual_seed`**，否则两章的 MAE 表对不上。

**下一步**：`ch05` 会把循环层换成**卷积**（TCN）与**注意力**，
再往后就是 `final` 的多步预测综合题 —— 那里才是"长依赖"真正该出现的地方。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch04 LSTM 与 GRU（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`data/load_curve.csv` 的 A 台区（居民型），2025 全年 8760 个整点 —— 与 ch03 同口径。
>
> **本章最容易卡住的三个地方**，卡住时重点回看提示：
> 1. LSTM 的 gate 顺序（`i, f, g, o`）
> 2. GRU 的 `n` 门偏置（`b_hh` 在括号里）
> 3. `h_n` / `c_n` 的形状
>
> 手写对账的 assert 都是 **`< 1e-12`**（double 精度下的逐值一致）。
> **对不上就是有 bug**，别放宽阈值 —— 放宽之后你就学不到任何东西了。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 数据准备（与 ch03 同口径） | 11 | 4.1 |
| 2 | 记忆保留率：`tanh` 通路 vs 加法通路 | 2 | 4.1 |
| 3 | 手写 LSTM 单步，与 `nn.LSTMCell` 对账 | 11 | 4.2 / 4.3 |
| 4 | 手写 LSTM 全序列，与 `nn.LSTM` 对账 | 15 | 4.3 |
| 5 | gate 顺序写错的两种下场 | 16 | 4.3.1 |
| 6 | 手写 GRU 全序列，与 `nn.GRU` 对账 | 12 | 4.4 |
| 7 | GRU 的 `n` 门偏置坑 | 5 | 4.4.1 |
| 8 | `bidirectional` / `num_layers` 的形状 | 6 | 4.5 |
| 9 | 四个模型的 MAE 对照表 | 7 | 4.5 |
| 10 | 梯度流对照：RNN vs LSTM | 4 | 4.6 |
| 11 | 遗忘门偏置扫描 | 2 | 4.6.1 |
| **合计** | 11 个任务 / 26 个 `@@todo` 块 | **91** | |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md("## 任务 1：数据准备（4.1）"),
    code(T1_CODE),
    md("## 任务 2：记忆保留率 —— `tanh` 通路 vs 加法通路（4.1）"),
    code(T2_CODE),
    md("## 任务 3：手写 LSTM 单步并对账（4.2 / 4.3）"),
    code(T3_CODE),
    md("## 任务 4：手写 LSTM 全序列并对账（4.3）"),
    code(T4_CODE),
    md("## 任务 5：gate 顺序写错的两种下场（4.3.1）"),
    code(T5_CODE),
    md("## 任务 6：手写 GRU 全序列并对账（4.4）"),
    code(T6_CODE),
    md("## 任务 7：GRU 的 `n` 门偏置坑（4.4.1）"),
    code(T7_CODE),
    md("## 任务 8：`bidirectional` / `num_layers` 的形状（4.5）"),
    code(T8_CODE),
    md("## 任务 9：四个模型的 MAE 对照表（4.5）"),
    code(T9_CODE),
    md("## 任务 10：梯度流对照 —— RNN vs LSTM（4.6）"),
    code(T10_CODE),
    md("## 任务 11：遗忘门偏置扫描（4.6.1）"),
    code(T11_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 11 个代码块的 assert 全部通过
- [ ] RNN 的 `h` 一条通道为什么不够用？40 步后 RNN 式还剩多少、加法通路还剩多少？
- [ ] LSTM 的四个门分别叫什么？各自用什么激活函数？值域是多少？
- [ ] `c_t = f ⊙ c_{t-1} + i ⊙ g` 这一行为什么被称为"加法高速公路"？
- [ ] PyTorch 的 LSTM gate 顺序是什么？写错之后 `h` 会差多少？
- [ ] 只把 `i` 与 `f` 对调会怎样？为什么不报错？
- [ ] GRU 的段序是什么？`n` 门的 `b_hh` 为什么要被 `r` 乘？
- [ ] LSTM 和 GRU 的偏置处理有什么不同？
- [ ] `out` / `h_n` / `c_n` 三个形状分别是什么？双向和堆叠各改哪一维？
- [ ] 为什么双向 LSTM 不能用于在线预测？
- [ ] 本章的负结果是什么？为什么 LSTM 默认初始化下梯度比 RNN 衰减更快？
- [ ] 遗忘门偏置设成 3 之后，T=160 的首末梯度比是多少？
- [ ] 为什么"把遗忘门偏置顶上去"这件事零成本、几乎无副作用？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 序列长度 / 训练样本 / 测试样本 | 8760 / 6984 / 1752 |
| MU / SD | 454.561801 / 72.623128 |
| 朴素持续基线 MAE | 0.414592 |
| RNN 式记忆 40 步后（初始脉冲 1.0） | 0.0074665174（保留 0.747%） |
| 加法通路 40 步后 | 0.6689717586（保留 66.90%） |
| 默认遗忘门均值 `f` | 0.4730 |
| 手写单步 vs `nn.LSTMCell` max&#124;diff&#124;（h / c） | 5.55e-17 / 5.55e-17 |
| 手写全序列 vs `nn.LSTM` max&#124;diff&#124;（out / c_n） | 5.55e-17 / 1.11e-16 |
| 手写 GRU vs `nn.GRU` max&#124;diff&#124; | 1.11e-16 |
| 错误 gate 顺序 `(f,i,o,g)` 的 h 偏差 | 4.741e-01 |
| 只对调 `i` / `f` 的 h 偏差 | 1.227e-01 |
| GRU `n` 门偏置写错的偏差 | 0.1552 |
| 形状：`layers=2, bidir=True` 的 out / h / c | (4,10,16) / (4,4,8) / (4,4,8) |
| RNN / LSTM / GRU / LSTM 双向 测试 MAE | 0.259707 / 0.215303 / 0.223200 / 0.225713 |
| T=160 首末梯度比：RNN(ρ=0.9) / LSTM 默认 | 9.615e-13 / 3.952e-34 |
| 遗忘门偏置 1 / 2 / 3 的首末比 | 3.218e-12 / 1.053e-03 / 3.004e-01 |
| 遗忘门偏置 3 的平均每步因子 | 0.9925 |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
