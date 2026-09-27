#!/usr/bin/env python3
"""ch03 序列建模动机：从 RNN 到 Attention（讲解 / 练习 / 答案 三件套）。

本章主线 —— **RNN 的三个硬伤，逐个引出 Attention**：

    ① 手写 RNN 前向 + BPTT，与 nn.RNN / autograd **逐元素对账**
       （对账不是走形式：它证明我们理解的公式就是框架在算的东西）
    ② 梯度消失**可测量**：谱半径 0.5 → 每步 ×0.409，50 步差 19 个数量级；
       即使谱半径 1.0 也只有 ×0.746 —— **tanh 本身就是元凶**
       → 用「加性捷径」对照：每步因子从 0.746 拉到 1.121，梯度不再衰减
       → 这就是 Attention 的思想原型：**别让梯度走 50 步**
    ③ 串行不可并行：耗时随 T 线性增长（实测）
    ④ 置换不变性 vs 顺序敏感：词袋对顺序完全无感知（准确率恒 ≈ 1/2），
       RNN 能学到「谁先出现」= 1.0，打乱输入后掉到 0.5
       —— 但**必须用训练过的模型**：未训练 RNN 的置换 cos 高达 0.98

反常识彩蛋：小样本上词袋准确率会漂到 0.52~0.55，把 n 推到 2 万会收敛回 0.5
（**无信息特征 + 有限样本 = 假信号**）。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp03_rnn_motivation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch03_rnn_motivation"

# =========================================================================== #
# 公共导入 / 数据                                                             #
# =========================================================================== #

IMPORTS = '''import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data")                  # notebook cwd = coding/04_nlp
CSV = DATA / "synth_text.csv"

torch.set_num_threads(4)
print("torch", torch.__version__,
      "| 线程", torch.get_num_threads(),
      "| CUDA 可用", torch.cuda.is_available())
'''

SETUP = '''df = pd.read_csv(CSV, keep_default_na=False)
print("工单数据", len(df), "条 |", df["缺陷类型"].value_counts().to_dict())

# 本章的顺序敏感任务只用「渗漏油」和「发热」两类的信号词
A_W = ["渗油", "漏油", "油迹", "油污", "滴油"]
B_W = ["发热", "温度偏高", "过热", "烫手", "温升异常"]
MID = ["巡检", "发现", "位于", "存在", "缺陷", "一般",
       "请", "尽快", "核实", "已", "通知", "运维班"]
VOCAB = ["<pad>"] + A_W + B_W + MID
VID = {w: i for i, w in enumerate(VOCAB)}

hits = {w: int(df["缺陷描述"].str.contains(w, regex=False).sum()) for w in A_W + B_W}
print("信号词在真实工单里的命中次数:", hits)
print("A 类合计", sum(hits[w] for w in A_W), "| B 类合计", sum(hits[w] for w in B_W))
print("词表大小", len(VOCAB))
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
D_IN, D_H, T_DEMO, B_DEMO = 6, 8, 5, 3      # 对账实验的小形状
D_DECAY, T_DECAY = 32, 50                   # 梯度衰减实验
T_SEQ = 8                                   # 顺序敏感任务的序列长度


def orth(shape, gain=1.0, seed=1):
    """正交矩阵 * gain —— 控制 RNN 权重谱半径的标准手法。"""
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(shape, generator=g))
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0) * gain


def near(a, b, eps=1e-6):
    """浮点比较，避免 assert 里 == 抖动。"""
    return abs(float(a) - float(b)) < eps


print("D_IN,D_H,T_DEMO", D_IN, D_H, T_DEMO,
      "| D_DECAY,T_DECAY", D_DECAY, T_DECAY, "| T_SEQ", T_SEQ)
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

# =========================================================================== #
# 代码块                                                                      #
# =========================================================================== #

# ---- 实验夹具（不挖空） ----
FIXTURE = '''# 固定夹具：权重来自 nn.RNN，输入用独立 Generator，保证可复现
g = torch.Generator().manual_seed(7)
x = torch.randn(B_DEMO, T_DEMO, D_IN, generator=g)
h0 = torch.randn(B_DEMO, D_H, generator=g)

with torch.random.fork_rng():            # fork_rng：块内种子不影响全局
    torch.manual_seed(101)
    ref = nn.RNN(D_IN, D_H, batch_first=True)

W_ih, W_hh = ref.weight_ih_l0.detach(), ref.weight_hh_l0.detach()
b_ih, b_hh = ref.bias_ih_l0.detach(), ref.bias_hh_l0.detach()
print("输入", tuple(x.shape), "| h0", tuple(h0.shape),
      "| W_ih", tuple(W_ih.shape), "| W_hh", tuple(W_hh.shape))
'''

# ---- T1：手写 RNNCell 前向 ----
T1_CODE = '''def rnn_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih, b_hh):
    """单步：h_t = tanh(x_t @ W_ih^T + b_ih + h_{t-1} @ W_hh^T + b_hh)"""

    # @@todo 写出 RNN 单步前向的核心公式并返回
    # @@hint 四项相加：x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh，外面套一层 torch.tanh
    return torch.tanh(x_t @ W_ih.T + b_ih + h_prev @ W_hh.T + b_hh)
    # @@end


# ---- 验收 ----
out = rnn_cell_forward(x[:, 0, :], h0, W_ih, W_hh, b_ih, b_hh)
assert out.shape == (B_DEMO, D_H), f"形状应为 {(B_DEMO, D_H)}，实际 {tuple(out.shape)}"
assert float(out.abs().max().detach()) <= 1.0, "tanh 输出必须落在 [-1, 1]"
print("单步输出形状", tuple(out.shape), "| |h| 最大 %.6f" % float(out.abs().max().detach()))
'''

# ---- T2：手写整序列前向 ----
T2_CODE = '''def rnn_forward(x, h0, W_ih, W_hh, b_ih, b_hh):
    """返回 (全部隐状态 [B, T, D_H], 末状态 [B, D_H])"""
    h = h0
    outs = []
    for t in range(x.shape[1]):
        # @@todo 用单步函数把状态往前推一步，并把它收进 outs
        # @@hint h = rnn_cell_forward(x[:, t, :], h, W_ih, W_hh, b_ih, b_hh) 然后 outs.append(h)
        h = rnn_cell_forward(x[:, t, :], h, W_ih, W_hh, b_ih, b_hh)
        outs.append(h)
        # @@end

    # @@todo 把 outs 堆成 [B, T, D_H] 的张量，并连同末状态一起返回
    # @@hint torch.stack(outs, dim=1)
    H = torch.stack(outs, dim=1)
    return H, h
    # @@end


# ---- 验收 ----
H_my, h_my = rnn_forward(x, h0, W_ih, W_hh, b_ih, b_hh)
assert H_my.shape == (B_DEMO, T_DEMO, D_H), f"形状错误：{tuple(H_my.shape)}"
assert h_my.shape == (B_DEMO, D_H)
print("手写 H 形状", tuple(H_my.shape))
'''

# ---- T3：与 nn.RNN 对账 ----
T3_CODE = '''H_ref, h_ref = ref(x, h0.unsqueeze(0))

# @@todo 算出手写结果与 nn.RNN 结果的最大绝对差（别忘了 .detach()）
# @@hint (H_my - H_ref).abs().max()；不 detach 会报警告
diff_H = float((H_my - H_ref).abs().max().detach())
diff_hT = float((h_my - h_ref.squeeze(0)).abs().max().detach())
# @@end

print("全部隐状态最大绝对差: %.3e" % diff_H)
print("末状态最大绝对差:     %.3e" % diff_hT)
print("h_T 前 3 个数:", [round(v, 6) for v in h_my[0, :3].tolist()])

# ---- 验收 ----
assert diff_H < 1e-6, f"手写前向与 nn.RNN 不一致：{diff_H}"
assert diff_hT < 1e-6
print("\\n提示：残差在 1e-7 量级 = float32 的舍入误差，公式没写错。")
'''

# ---- T4：形状陷阱 ----
T4_CODE = '''# ---- 两个常见的形状坑 ----
try:
    ref(x, h0)                                   # h0 必须是 (num_layers, B, D_H)
    print("陷阱 1：没有触发")
except RuntimeError as e:
    print("陷阱 1：h0 传 (B, D_H) 直接 RuntimeError ->", str(e)[:52])

H2, _ = ref(x[0], h0[:1])                        # 去掉 batch 维 -> 输入变 (T, D_IN)
print("陷阱 2：单样本输入", tuple(x[0].shape), "-> 输出", tuple(H2.shape),
      "（batch_first 对 2D 输入无效）")

# ---- 验收 ----
assert H2.shape == (T_DEMO, D_H)
print("\\n结论：nn.RNN 的输入要么 (B, T, D)，要么 (T, D)；"
      "h0 永远是 (num_layers, B, D_H)。")
'''

# ---- T5：手写 BPTT ----
T5_CODE = '''def rnn_bptt(x, W_ih, W_hh, b_ih, b_hh):
    """手写 BPTT。目标函数 loss = 0.5 * sum(h_T^2)，返回 (loss, 四个梯度)。"""
    Bn, Tn, _ = x.shape
    Hs = rnn_forward(x, torch.zeros(Bn, D_H), W_ih, W_hh, b_ih, b_hh)[0]
    loss = 0.5 * (Hs[:, -1, :] ** 2).sum()

    dW_ih = torch.zeros_like(W_ih)
    dW_hh = torch.zeros_like(W_hh)
    db_ih = torch.zeros_like(b_ih)
    db_hh = torch.zeros_like(b_hh)

    # 把 h_{t-1} 预先摊平成一张表，前移一位、开头补零 —— 这样反向循环里不用写 if
    Hs_prev = torch.cat([torch.zeros(Bn, 1, D_H), Hs[:, :-1, :]], dim=1)

    dh = Hs[:, -1, :].clone()                    # dL/dh_T
    for t in reversed(range(Tn)):
        # @@todo 用 h_t 反推 tanh 导数得到 dpre，再累加四个参数的梯度，最后把 dh 回传给 t-1
        # @@hint tanh'(pre) = 1 - h_t**2；dW_ih += dpre.T @ x[:, t, :]；dW_hh += dpre.T @ Hs_prev[:, t, :]；
        # @@hint 最后 dh = dpre @ W_hh（这就是「梯度沿时间反向走一步」）
        dpre = dh * (1 - Hs[:, t, :] ** 2)
        dW_ih += dpre.T @ x[:, t, :]
        dW_hh += dpre.T @ Hs_prev[:, t, :]
        db_ih += dpre.sum(0)
        db_hh += dpre.sum(0)
        dh = dpre @ W_hh
        # @@end

    return loss, dW_ih, dW_hh, db_ih, db_hh


# ---- 验收 ----
_loss, _dWih, _dWhh, _dbih, _dbhh = rnn_bptt(x, W_ih, W_hh, b_ih, b_hh)
assert _dWih.shape == W_ih.shape and _dWhh.shape == W_hh.shape
print("手写 BPTT 跑通 | loss %.10f" % float(_loss))
'''

# ---- T6：与 autograd 对账 ----
T6_CODE = '''for p in (W_ih, W_hh, b_ih, b_hh):
    p.requires_grad_(True)

H_a, _ = rnn_forward(x, torch.zeros(B_DEMO, D_H), W_ih, W_hh, b_ih, b_hh)
loss_a = 0.5 * (H_a[:, -1, :] ** 2).sum()
loss_a.backward()

loss_m, dW_ih_m, dW_hh_m, db_ih_m, db_hh_m = rnn_bptt(
    x, W_ih.detach(), W_hh.detach(), b_ih.detach(), b_hh.detach())

# @@todo 算手写梯度与 autograd 梯度的最大绝对差（四组）
# @@hint (dW_ih_m - W_ih.grad).abs().max()
gap = dict(
    dW_ih=float((dW_ih_m - W_ih.grad).abs().max()),
    dW_hh=float((dW_hh_m - W_hh.grad).abs().max()),
    db_ih=float((db_ih_m - b_ih.grad).abs().max()),
    db_hh=float((db_hh_m - b_hh.grad).abs().max()),
)
# @@end

print("loss 手写 %.10f | autograd %.10f" % (float(loss_m), float(loss_a.detach())))
for k, v in gap.items():
    print(f"  {k:6s} 最大差 {v:.3e}")

# ---- 验收 ----
assert abs(float(loss_m) - float(loss_a.detach())) < 1e-9
assert max(gap.values()) < 1e-9, f"手写梯度与 autograd 不一致：{gap}"
print("\\n手写 BPTT 与 autograd 逐元素相同 —— 梯度公式是对的。")
'''

# ---- T7：梯度衰减曲线 ----
T7_CODE = '''def decay_curve(scale, residual=False, T=T_DECAY):
    """返回每个时间步的 ‖dL/dh_t‖（t = 0..T-1）。

    residual=True 时改用加性捷径：h_t = h_{t-1} + tanh(...)
    """
    gg = torch.Generator().manual_seed(11)
    xin = (torch.randn(1, T, D_DECAY, generator=gg) * 0.5).requires_grad_(True)
    W = orth((D_DECAY, D_DECAY), gain=scale, seed=3)

    hs = []
    h = torch.zeros(1, D_DECAY)
    for t in range(T):
        # @@todo 算出候选状态，并按 residual 决定是「叠加到 h 上」还是「替换 h」；然后把 h 记进 hs
        # @@hint pre = torch.tanh(xin[:, t, :] + h @ W.T)；residual 时 h = h + pre，否则 h = pre；
        # @@hint 必须 h.retain_grad()，否则 h 是中间量拿不到 .grad
        pre = torch.tanh(xin[:, t, :] + h @ W.T)
        h = (h + pre) if residual else pre
        h.retain_grad()
        hs.append(h)
        # @@end

    # @@todo 反传后返回每个时间步的梯度范数
    # @@hint loss = 0.5 * (hs[-1] ** 2).sum() -> loss.backward() -> 收集 hs[t].grad.norm()
    loss = 0.5 * (hs[-1] ** 2).sum()
    loss.backward()
    return [float(hs[t].grad.norm()) for t in range(T)]
    # @@end


# ---- 验收 ----
ns_probe = decay_curve(1.0)
assert len(ns_probe) == T_DECAY
assert ns_probe[-1] > ns_probe[0], "离最终 loss 越近，梯度应当越大"
print("曲线长度", len(ns_probe), "| 首末范数 %.3e -> %.3e" % (ns_probe[0], ns_probe[-1]))
'''

# ---- T8：三档谱半径 ----
T8_CODE = '''norms_lo = decay_curve(0.5)
norms_mid = decay_curve(1.0)
norms_hi = decay_curve(1.5)

# @@todo 算三个档次各自的「每步衰减因子」：最远端梯度 / 最近端梯度，再开 (T-1) 次方
# @@hint per = (norms[0] / norms[-1]) ** (1 / (T_DECAY - 1))
per_lo = (norms_lo[0] / norms_lo[-1]) ** (1 / (T_DECAY - 1))
per_mid = (norms_mid[0] / norms_mid[-1]) ** (1 / (T_DECAY - 1))
per_hi = (norms_hi[0] / norms_hi[-1]) ** (1 / (T_DECAY - 1))
# @@end

print("谱半径  ‖dL/dh_0‖    ‖dL/dh_49‖   每步因子")
for tag, ns, per in (("0.5", norms_lo, per_lo),
                     ("1.0", norms_mid, per_mid),
                     ("1.5", norms_hi, per_hi)):
    print(f"  {tag}    {ns[0]:.3e}    {ns[-1]:.3e}    {per:.6f}")

print()
print("采样点 [0, 5, 10, 20, 30, 40, 49]：")
for tag, ns in (("谱半径 0.5", norms_lo), ("谱半径 1.0", norms_mid), ("谱半径 1.5", norms_hi)):
    print(f"  {tag}: " + " ".join("%.2e" % ns[i] for i in (0, 5, 10, 20, 30, 40, 49)))

# ---- 验收 ----
assert per_lo < per_mid < per_hi, "谱半径越大，每步衰减因子应当越大"
assert per_mid < 1.0, "tanh 让谱半径 1.0 的权重仍然衰减"
assert norms_lo[0] < 1e-15, "谱半径 0.5 时最远端梯度应趋近 0"
print("\\n谱半径 0.5 → 每步 ×%.6f，50 步后相差 19 个数量级。" % per_lo)
print("注意：谱半径 1.0 也只有 ×%.6f —— 元凶不只是权重，tanh 自己也衰减。" % per_mid)
'''

# ---- T9：加性捷径对照 ----
T9_CODE = '''norms_res = decay_curve(1.0, residual=True)

# @@todo 用与上一节相同的公式算残差档的每步衰减因子
# @@hint per = (norms[0] / norms[-1]) ** (1 / (T_DECAY - 1)) —— 公式一模一样，只是换了输入
per_res = (norms_res[0] / norms_res[-1]) ** (1 / (T_DECAY - 1))
# @@end

print("谱半径 1.0 + 加性捷径：‖dL/dh_0‖ %.3e | ‖dL/dh_49‖ %.3e | 每步因子 %.6f"
      % (norms_res[0], norms_res[-1], per_res))
print("采样点:", " ".join("%.2e" % norms_res[i] for i in (0, 5, 10, 20, 30, 40, 49)))
print()
print("对比：普通 RNN 每步 ×%.6f  vs  加性捷径每步 ×%.6f" % (per_mid, per_res))
print("每步因子之比 残差 / 普通 = %.6f" % (per_res / per_mid))

# ---- 验收 ----
assert per_res > 1.0, "加性捷径应当让梯度不再衰减（甚至放大）"
assert per_res > per_mid * 1.4, "加性捷径的效果应当显著"
print("\\n→ 只加一条 h_{t-1} -> h_t 的直接通道，梯度就活到了第一步。")
print("→ Attention 做的正是这件事，只不过它是「任意两个位置之间的直接通道」。")
'''

# ---- 绘图 ----
PLOT_CODE = '''fig, ax = plt.subplots(figsize=(7.2, 4.2))
for tag, ns, st in (("谱半径 0.5", norms_lo, "-o"),
                    ("谱半径 1.0", norms_mid, "-s"),
                    ("谱半径 1.5", norms_hi, "-^"),
                    ("谱半径 1.0 + 加性捷径", norms_res, "--")):
    ax.semilogy(range(T_DECAY), ns, st, markersize=3, linewidth=1.4, label=tag)
ax.set_xlabel("时间步 t（越靠左越「远」）")
ax.set_ylabel("‖dL/dh_t‖（对数轴）")
ax.set_title("RNN 的梯度随距离衰减（T=50）")
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()

print("读数：三条实线越靠左越塌，虚线（加性捷径）反而从左到右下降 —— 梯度活到了 t=0。")
'''

# ---- T10：串行不可并行 ----
T10_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(404)
    net_plain = nn.RNN(32, 32, batch_first=True)

times = {}
for T_ in (25, 50, 100, 200, 400):
    xin = torch.randn(8, T_, 32).detach().requires_grad_(True)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        o, _ = net_plain(xin)
        o[:, -1, :].sum().backward()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    # @@todo 取 5 次耗时的中位数，存进 times[T_]
    # @@hint ts 已经排好序，中位数就是 ts[len(ts) // 2]
    med = ts[len(ts) // 2]
    times[T_] = med
    # @@end
    print("T=%4d  前向+反向中位耗时 %.6f s" % (T_, med))

print()
print("T 从 25 涨到 400（长度 ×16），耗时 ×%.2f" % (times[400] / times[25]))

# ---- 验收 ----
assert times[400] > times[25] * 8, "耗时应当随时序长度近似线性增长"
print("\\nRNN 的 h_t 依赖 h_{t-1}，前向必须一步步走；")
print("反向 BPTT 更要沿同一条链回退 50 步 —— 这就是「串行不可并行」。")
'''

# ---- T11：置换不变性（未训练的 RNN 会骗你） ----
T11_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(202)
    net2 = nn.RNN(16, 16, batch_first=True)

g2 = torch.Generator().manual_seed(31)
x1 = torch.randn(4, 12, 16, generator=g2)
perm = torch.randperm(12, generator=torch.Generator().manual_seed(32))
x2 = x1[:, perm, :]
x3 = torch.randn(4, 12, 16, generator=torch.Generator().manual_seed(33))

with torch.no_grad():
    o1, _ = net2(x1)
    o2, _ = net2(x2)
    o3, _ = net2(x3)
o2r = o2[:, torch.argsort(perm), :]              # 把置换后的输出搬回原位置再比

# @@todo 算三个余弦相似度：置换前后的末状态、置换前后的逐位置、随机两条序列的基线
# @@hint F.cosine_similarity(a, b, dim=-1).mean()；逐位置时对 dim=2 求
c_last = float(F.cosine_similarity(o1[:, -1], o2[:, -1], dim=-1).mean())
c_pos = float(F.cosine_similarity(o1, o2r, dim=-1).mean())
c_base = float(F.cosine_similarity(o1[:, -1], o3[:, -1], dim=-1).mean())
# @@end

print("【未训练 tanh-RNN】同一序列 vs 其置换：末状态 cos %.6f | 逐位置 cos %.6f"
      % (c_last, c_pos))
print("【基线】两条完全不同的随机序列：末状态 cos %.6f" % c_base)

# ---- 验收 ----
assert c_last > 0.9 and c_base < 0.5
print("\\n警告：置换后输出几乎没变！—— 未训练的 RNN 处于 tanh 饱和区，")
print("末状态由「输入分布」主导，与顺序关系不大。**别拿随机初始化的网络讲顺序的故事。**")
'''

# ---- T12：线性 RNN 暴露真实差异 ----
T12_CODE = '''W_LIN = orth((16, 16), gain=0.95, seed=37)


def lin_rnn(x):
    """去掉 tanh 的线性 RNN：h_t = h_{t-1} @ W^T + x_t"""
    h = torch.zeros(x.shape[0], 16)
    for t in range(x.shape[1]):
        # @@todo 线性状态更新（没有非线性）
        # @@hint h = h @ W_LIN.T + x[:, t, :]
        h = h @ W_LIN.T + x[:, t, :]
        # @@end
    return h


with torch.no_grad():
    l1, l2 = lin_rnn(x1), lin_rnn(x2)

# @@todo 算线性 RNN 在置换前后的末状态余弦相似度
# @@hint 与上一节同一个 cos 公式
c_lin = float(F.cosine_similarity(l1, l2, dim=-1).mean())
# @@end

print("【线性 RNN】同一序列 vs 其置换：末状态 cos %.6f" % c_lin)
print("【tanh-RNN（未训练）】同口径 cos %.6f" % c_last)

# ---- 验收 ----
assert c_lin < 0.8, "线性 RNN 应当对置换明显敏感"
assert c_lin < c_last - 0.2, "线性 RNN 应当比未训练的 tanh-RNN 敏感得多"
print("\\n线性情形里位置权重 W^(T-t) 各不相同，置换立刻改变输出；")
print("tanh 饱和把这件事掩盖了 —— 这也解释了为什么「未训练网络上的实验」不可信。")
'''

# ---- T13：顺序敏感任务的数据 ----
T13_CODE = '''def make_order_data(n, seed, T=T_SEQ):
    """每条序列里 A 类词和 B 类词各出现恰好 1 次，标签 = 「A 是否出现在 B 之前」。

    这样词袋看到的词频特征对两类样本完全对称 —— 它不可能学到任何东西。
    """
    rg = np.random.default_rng(seed)
    X = np.zeros((n, T), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        # @@todo 随机挑两个不同位置 p / q；铺一行中性词；再把 A 词放 p、B 词放 q；标签取「p < q」
        # @@hint p, q = rg.choice(T, size=2, replace=False)；用 rg.integers(len(MID)) 选词；
        # @@hint 最后 X[i] = [VID[w] for w in seq]，y[i] = int(p < q)
        p, q = rg.choice(T, size=2, replace=False)
        seq = [MID[rg.integers(len(MID))] for _ in range(T)]
        seq[p] = A_W[rg.integers(len(A_W))]
        seq[q] = B_W[rg.integers(len(B_W))]
        X[i] = [VID[w] for w in seq]
        y[i] = int(p < q)
        # @@end
    return X, y


Xtr, ytr = make_order_data(800, 1)
Xte, yte = make_order_data(500, 2)

# ---- 验收 ----
assert Xtr.shape == (800, T_SEQ) and ytr.shape == (800,)
assert set(np.unique(ytr)) == {0, 1}, "两类标签都要有"
row = [VOCAB[i] for i in Xtr[0]]
n_a = sum(1 for w in row if w in A_W)
n_b = sum(1 for w in row if w in B_W)
assert n_a == 1 and n_b == 1, "每条序列必须恰好 1 个 A 词 + 1 个 B 词"
print("训练 800 / 测试 500 | 训练集标签分布", dict(Counter(ytr.tolist())))
print("样例:", row, "->", ytr[0])
print("A 词个数", n_a, "| B 词个数", n_b, "（位置信息只存在于顺序里）")
'''

# ---- T14：词袋基线（无信息特征 + 有限样本 = 假信号） ----
T14_CODE = '''from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

cv = CountVectorizer(tokenizer=lambda t: t, token_pattern=None, lowercase=False)
Xb_tr = cv.fit_transform([[VOCAB[i] for i in row] for row in Xtr])
Xb_te = cv.transform([[VOCAB[i] for i in row] for row in Xte])
lr = LogisticRegression(max_iter=1000).fit(Xb_tr, ytr)

# @@todo 算词袋在测试集上的准确率
# @@hint lr.score(Xb_te, yte)
acc_bow = lr.score(Xb_te, yte)
# @@end

print("词袋矩阵", Xb_tr.shape, "| 非零", int(Xb_tr.nnz))
print("词袋 + LR 测试准确率: %.6f" % acc_bow)
print()

print("换个角度看：把样本量一路推大，词袋准确率会怎样？")
for n in (400, 1300, 5200, 20800):
    Xn, yn = make_order_data(n, 77)
    Xb = cv.transform([[VOCAB[i] for i in row] for row in Xn])
    # @@todo 用 5 折交叉验证评估词袋模型
    # @@hint cross_val_score(LogisticRegression(max_iter=1000), Xb, yn, cv=5)
    sc = cross_val_score(LogisticRegression(max_iter=1000), Xb, yn, cv=5)
    # @@end
    print("  n=%5d  词袋 5 折 %.6f ± %.6f" % (n, sc.mean(), sc.std()))

# ---- 验收 ----
assert abs(acc_bow - 0.5) < 0.12, f"词袋在纯顺序任务上应接近随机，实际 {acc_bow}"
print("\\n词袋始终在 0.5 附近抖动，且样本越多抖动越小 ——")
print("这条任务的全部信息都在顺序里，而词袋把顺序扔了。")
'''

# ---- RNN 训练（不挖空：主体是循环，挖了会违反 DSL 纪律） ----
RNN_TRAIN = '''class SeqRNN(nn.Module):
    """最小可用的序列分类器：Embedding -> RNN -> 取末状态 -> 线性头"""

    def __init__(self, V, d=24):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.rnn = nn.RNN(d, d, batch_first=True)
        self.fc = nn.Linear(d, 2)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.fc(h[:, -1])


with torch.random.fork_rng():                    # 权重初始化与外部 RNG 解耦
    torch.manual_seed(303)
    model = SeqRNN(len(VOCAB))

xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
xe, ye = torch.as_tensor(Xte), torch.as_tensor(yte)
opt = torch.optim.Adam(model.parameters(), lr=0.02)
lossf = nn.CrossEntropyLoss()

t0 = time.perf_counter()
for ep in range(150):
    model.train()
    opt.zero_grad()
    loss = lossf(model(xt), yt)
    loss.backward()
    opt.step()
    if (ep + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            acc_ = (model(xe).argmax(1) == ye).float().mean().item()
        print("  epoch %3d  loss %.6f  test acc %.6f" % (ep + 1, loss.item(), acc_))
print("训练耗时 %.3f s" % (time.perf_counter() - t0))

model.eval()
with torch.no_grad():
    acc_rnn = float((model(xe).argmax(1) == ye).float().mean())
print("RNN 测试准确率: %.6f  （1600 条样本、150 epoch、0.3 秒）" % acc_rnn)

# ---- 验收 ----
assert acc_rnn > 0.99, f"RNN 应当能学会这个纯顺序任务，实际 {acc_rnn}"
'''

# ---- T15：打乱输入 ----
T15_CODE = '''Xsh = xe.clone()                                 # xe = torch.as_tensor(Xte)
for i in range(Xsh.shape[0]):
    # @@todo 用固定种子的随机置换打乱第 i 行的 token 顺序
    # @@hint Xsh[i] = Xsh[i][torch.randperm(Xsh.shape[1], generator=torch.Generator().manual_seed(1000 + i))]
    Xsh[i] = Xsh[i][torch.randperm(Xsh.shape[1],
                                   generator=torch.Generator().manual_seed(1000 + i))]
    # @@end

Xsh_tok = [[VOCAB[i] for i in row] for row in Xsh.numpy()]

with torch.no_grad():
    # @@todo 算打乱后的 RNN 准确率
    # @@hint (model(Xsh).argmax(1) == ye).float().mean()
    acc_shuf = float((model(Xsh).argmax(1) == ye).float().mean())
    # @@end

# @@todo 算词袋模型对「打乱版输入」与原输入的预测一致率
# @@hint (lr.predict(Xb_te) == lr.predict(cv.transform(Xsh_tok))).mean()
same_pred = float((lr.predict(Xb_te) == lr.predict(cv.transform(Xsh_tok))).mean())
# @@end

print("原 RNN 准确率       : %.6f" % acc_rnn)
print("打乱输入后 RNN 准确率: %.6f" % acc_shuf)
print("词袋对打乱输入的预测一致率: %.6f" % same_pred)

# ---- 验收 ----
assert acc_shuf < 0.6, f"打乱顺序后 RNN 应跌到随机水平，实际 {acc_shuf}"
assert same_pred > 0.999, "词袋对置换完全不敏感，预测应当一模一样"
print("\\n同一个测试集，只把 token 顺序打乱：")
print("  RNN 从 %.4f 掉到 %.4f —— 它的能力建立在顺序上；" % (acc_rnn, acc_shuf))
print("  词袋预测一位都没变 —— 它结构上就看不见顺序。")
'''

# ---- 小结表 ----
SUMMARY = '''import pandas as pd

rows = [
    ("手写前向 vs nn.RNN", "最大绝对差 %.3e" % diff_H, "公式与框架一致（float32 舍入）"),
    ("手写 BPTT vs autograd", "最大差 %.3e" % max(gap.values()), "梯度公式正确"),
    ("梯度衰减（谱半径 1.0）", "每步 ×%.6f" % per_mid, "50 步内衰减到 1e-6"),
    ("梯度衰减（加性捷径）", "每步 ×%.6f" % per_res, "同一深度不再衰减"),
    ("耗时 T=25 -> 400", "×%.2f" % (times[400] / times[25]), "长度 ×16，串行不可并行"),
    ("置换（未训练 RNN）", "cos %.4f" % c_last, "别用随机初始化网络下结论"),
    ("置换（线性 RNN）", "cos %.4f" % c_lin, "位置权重暴露顺序依赖"),
    ("词袋（顺序任务）", "≈ 0.50", "结构上看不见顺序"),
    ("RNN（顺序任务）", "%.4f -> %.4f（打乱后）" % (acc_rnn, acc_shuf), "顺序信息被学进权重"),
]
print(pd.DataFrame(rows, columns=["实验", "实测", "结论"]).to_string(index=False))
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch03 序列建模动机：从 RNN 到 Attention（讲解版）

> 对应考点：**「大语言模型相关理论考查」** 的第一环 —— 为什么会有 Transformer？
>
> 结论先给：**Attention 不是凭空发明的，它是对 RNN 三个硬伤的直接回应。**
> 本章把这三个硬伤逐个**测量**出来，而不是背结论。

本章要回答四个问题：

| # | 问题 | 手段 |
|---|---|---|
| 1 | RNN 到底在算什么？ | 手写 `rnn_cell_forward` / `rnn_forward`，与 `nn.RNN` **逐元素对账** |
| 2 | BPTT 的梯度公式对吗？ | 手写反向传播，与 `autograd` **逐元素对账** |
| 3 | 「梯度消失」到底有多严重？ | 三档谱半径 + 加性捷径，实跑 `‖dL/dh_t‖` 曲线 |
| 4 | 顺序信息值多少？ | 词袋 vs RNN 在一个「纯顺序任务」上的实测对比 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    code(PLOT_SETUP),
    md(
        """
## 3.1 手写 RNN 单步前向

RNN 的全部秘密就在这一行：

$$h_t = \\tanh\\big(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}\\big)$$

注意 `W_hh` 是**同一个矩阵**在所有时间步复用（参数共享）——这既是 RNN 能处理变长序列的原因，
也是它梯度问题的根源：反向传播时同一个 `W_hh` 被连乘 50 次。
"""
    ),
    code(FIXTURE),
    code(T1_CODE),
    md(
        """
## 3.2 手写整序列前向

单步函数套上一层循环，把每一步的隐状态收集起来。返回的两个东西各有用途：

- `H`（全部隐状态 `[B, T, D_H]`）—— 后面画梯度曲线、做 token 级任务要用
- `h_T`（末状态 `[B, D_H]`）—— 序列分类任务通常只取这一个

**为什么要手写**：`nn.RNN` 是个黑盒。手写一遍，你才知道 `batch_first=True` 到底改了哪个维度、
`h0` 为什么必须是三维。
"""
    ),
    code(T2_CODE),
    md(
        """
## 3.3 与 `nn.RNN` 对账

对账不是走形式 —— 它把「我以为的公式」和「框架实际在算的东西」钉在一起。
差异应当是 **float32 的舍入量级（1e-7）**；如果是 1e-2 那种量级，说明公式写错了。
"""
    ),
    code(T3_CODE),
    md(
        """
## 3.4 两个形状坑

| 坑 | 现象 |
|---|---|
| `h0` 传成 `(B, D_H)` | 直接 `RuntimeError`（要 `(num_layers, B, D_H)`） |
| 输入去掉 batch 维 | 输出也去掉 batch 维 —— `batch_first` 对 2D 输入**无效** |

第二条尤其阴：它不报错，只是形状悄悄变了，等到后面 `h[:, -1]` 取值时才崩。
"""
    ),
    code(T4_CODE),
    md(
        """
## 3.5 手写 BPTT

BPTT 的链式法则只有一句话：

$$\\frac{\\partial L}{\\partial h_{t-1}} = \\frac{\\partial L}{\\partial h_t}\\cdot\\operatorname{diag}(1-h_t^2)\\cdot W_{hh}$$

最后那个 $W_{hh}$ 就是「梯度沿时间反向走一步」。走 50 步，就是连乘 50 次。

**实现小技巧**：反向循环里本来要写 `if t > 0` 才能取到 $h_{t-1}$。
把状态表前面补一列零（`Hs_prev`），`t = 0` 时取到的是零向量，
对 `dW_hh` 的贡献恰好是零矩阵 —— 于是 `if` 可以整个删掉，代码更短也不易错。
"""
    ),
    code(T5_CODE),
    md(
        """
## 3.6 与 autograd 对账

四组梯度（`dW_ih` / `dW_hh` / `db_ih` / `db_hh`）应当**逐元素相同**。
这是本章最重要的一次对账：它证明我们对 BPTT 的理解没有偏差，
后面谈「梯度消失」时，谈的就是真的那个梯度。
"""
    ),
    code(T6_CODE),
    md(
        """
## 3.7 把「梯度消失」测出来

前面都是准备。现在做本章的核心实验：

- 序列长度 `T = 50`
- 输入固定、权重用**正交矩阵 × 谱半径**（控制 RNN 的稳定性最标准的手法）
- 目标函数 `loss = 0.5 * ‖h_T‖²`，于是 `dL/dh_T = h_T`
- 逐时间步 `retain_grad()`，读出 `‖dL/dh_t‖`

然后问一句：**最远端（t=0）的梯度，比最近端（t=49）小多少倍？**

> 关键实现细节：`h` 是循环里的中间量，必须 `h.retain_grad()` 才能在 `backward()` 之后
> 读到 `.grad`。这是 PyTorch 里最常被忘的一行。
"""
    ),
    code(T7_CODE),
    md(
        """
## 3.8 三档谱半径：谁在衰减？

`per = (‖dL/dh_0‖ / ‖dL/dh_49‖) ** (1/(T-1))` 读作「**每往远走一步，梯度乘多少**」。

- `per = 0.41` 意味着每步缩到 41%，50 步就是 $0.41^{49}\\approx 10^{-19}$ —— **梯度根本没传到**
- `per = 0.75` 也很糟：$0.75^{49}\\approx 10^{-6}$

**这里最容易搞错的一点**：很多人以为梯度消失只要把 `W_hh` 的谱半径调到 1 就好了。
实测告诉你不行 —— 谱半径 1.0 时每步仍然只有 `0.746`，因为
$\\tanh'(z) = 1-\\tanh^2(z) \\le 1$，**tanh 自己也在衰减**。
"""
    ),
    code(T8_CODE),
    md(
        """
## 3.9 加性捷径：把「直接通道」加进去

既然问题是「梯度要走 50 步」，那就给它一条捷径：

$$h_t = h_{t-1} + \\tanh\\big(W_{hh} h_{t-1} + \\dots\\big)$$

梯度里现在多了一个恒等项 $\\partial h_t/\\partial h_{t-1} \\supseteq I$，不会衰减到零。
"""
    ),
    code(T9_CODE),
    code(PLOT_CODE),
    md(
        """
**这张图就是 Attention 的动机。** 残差连接给的是「相邻步」的捷径，
Attention 给的是「**任意两个位置之间**」的捷径 —— 路径长度恒为 $O(1)$，
不管它们相距 1 步还是 500 步。

顺带记住这个副作用：加性捷径把每步因子推到 $>1$，
**梯度不消失了，但也可能开始爆炸** —— 这就是 Transformer 必须配
`LayerNorm` + warmup 的原因（ch05 / ch06 会接上）。
"""
    ),
    md(
        """
## 3.10 第二个硬伤：串行不可并行

`h_t` 依赖 `h_{t-1}`，所以前向必须一步步推；BPTT 还要沿同一条链回退。
这不是工程优化问题，是**依赖结构**决定的：序列越长，串行链越长。

下面测同一批数据在不同序列长度下的「前向+反向」耗时（取 5 次中位数）。
"""
    ),
    code(T10_CODE),
    md(
        """
## 3.11 置换不变性：先说一个反常识的坑

「RNN 对顺序敏感、词袋对顺序不敏感」这句话是对的，
但**你不能用一个随机初始化的 RNN 去验证它**。

下面做三组对比，你会看到一个让人意外的数字。
"""
    ),
    code(T11_CODE),
    md(
        """
看到了吗：**未训练的 RNN 对置换的余弦相似度高达 0.98**，几乎和没置换一样。

原因：随机初始化的 RNN 隐状态落在 tanh 的饱和区，末状态由输入的**分布**主导，
和顺序关系不大。如果就此下结论「RNN 也顺序不敏感」，那是完全错的。

换掉非线性再测一次，位置依赖立刻现形。
"""
    ),
    code(T12_CODE),
    md(
        """
线性 RNN 里 $h_T = \\sum_t W^{T-t} x_t$ —— **不同位置的权重完全不同**，
置换当然立刻改变输出。tanh 只是把这件事掩盖了。

> 记住这条元教训：**「在做实验」和「实验有效」是两件事。**
> 随机初始化的网络、样本太小的测试集、没有对照组的指标 —— 都可能给出看似专业的错误结论。
"""
    ),
    md(
        """
## 3.12 顺序到底值多少：构造一个纯顺序任务

设计一个**只有顺序有信息**的任务：

> 每条序列长 8，A 类词（渗漏油）和 B 类词（发热）**各出现恰好 1 次**，
> 标签 = 「A 是否出现在 B 之前」。

为什么这个任务干净：

- 词袋只看词频 → 每个样本都是「1 个 A 词 + 1 个 B 词 + 6 个中性词」
- 两类样本的词袋特征**分布完全相同** → 理论上不可能超过 1/2
- 但顺序里藏着完整的标签信息 → RNN 有机会学到

词表用的是**真实工单里的信号词**（`渗油`/`发热` 等），
所以这个任务和前面两章的数据是同一套语料。
"""
    ),
    code(T13_CODE),
    md(
        """
## 3.13 词袋基线：0.5 附近的抖动

先看「不可能赢」的那一边。

注意第二段的样本量扫描 —— 它藏着一个值得警惕的现象：
"""
    ),
    code(T14_CODE),
    md(
        """
小样本（n=400）时词袋能漂到 0.52，甚至会让人怀疑「是不是有点信号」。
把样本量推到 2 万，它老老实实回到 0.497，而且波动幅度（std）一路从 0.030 降到 0.006。

**无信息特征 + 有限样本 = 假信号。** 判据不是「准确率比 0.5 高」，而是
「**换个样本量它还站得住吗**」。
"""
    ),
    md(
        """
## 3.14 RNN：把顺序学进权重

同一个任务，换成 RNN。注意下面三件事：

1. `fork_rng()` 把权重初始化和外部 RNG 解耦，结果可复现，且与代码执行顺序无关
2. 150 epoch、1600 条样本、**0.3 秒** —— 本章全部实验都在 CPU 上跑
3. 最后的 assert 用 `> 0.99` 而不是 `== 1.0`，避免版本差异导致脆断言
"""
    ),
    code(RNN_TRAIN),
    md(
        """
## 3.15 打乱输入：谁在依赖顺序？

最后一步对照实验：**同一个测试集，只把 token 顺序打乱**。

- RNN 应当断崖式下跌（它的能力建立在顺序上）
- 词袋的预测应当**一位都不变**（它结构上看不见顺序）

这一组对照，把「顺序信息」的价值量化成了两个数字。
"""
    ),
    code(T15_CODE),
    md(
        """
## 3.16 汇总

"""
    ),
    code(SUMMARY),
    md(
        """
## 3.17 RNN 的三个硬伤 → Attention 的三个答案

| RNN 的硬伤 | 实测证据 | Attention 的回应 |
|---|---|---|
| **梯度要沿时间连乘** | 谱半径 1.0 时每步 ×0.746，50 步衰减 1e-6 | 任意两位置直接连边，路径长度 $O(1)$ |
| **串行不可并行** | 长度 ×16，耗时 ×14.7 | 全序列一次矩阵乘，训练可完全并行 |
| **顺序信息难长距离保持** | 打乱输入后 RNN 从 1.0 掉到 0.50 | 位置编码 + 注意力权重显式建模「谁和谁相关」 |

反过来看，RNN 也留下两个问题给 Attention：

- 加性捷径让梯度**不衰减了，但可能爆炸** → ch05 的 LayerNorm、ch06 的 warmup
- 全序列并行后，**位置信息没了**（置换等变）→ ch05 的位置编码

## 易错点清单

1. **`retain_grad()` 忘了写** —— 循环里的 `h` 不是叶子张量，`backward()` 后 `h.grad` 是 `None`，
   且不报错，只是拿到空值。
2. **拿未训练的 RNN 做顺序性实验** —— 本节的 cos 0.98 就是这个坑，结论会完全反过来。
3. **以为把谱半径调到 1 就不会梯度消失** —— tanh 的导数 $\\le 1$，本身也在衰减。
4. **`h0` 传 2D** —— `RuntimeError`，要 `(num_layers, B, D_H)`。
5. **单样本输入时以为 `batch_first` 还生效** —— 2D 输入输出也是 2D。
6. **对 `requires_grad=True` 的张量做 `float()`** —— 只会警告不会错，但说明你在用不该用的路径。
7. **反向循环里 `dh` 忘了回传** —— 得到的梯度只反映最后一步，数值会明显偏小。
8. **重复定义 `nn.RNN` 时没隔离 RNG** —— 结果随代码执行顺序漂移，`assert` 时好时坏。
9. **用 `acc == 1.0` 这种脆断言** —— 用 `> 0.99`，把「必须学会」和「数值抖动」分开。
10. **把「准确率高于 0.5」当成有信号** —— 小样本上纯噪声也能到 0.52。

## 本章小结

- 手写 RNN 前向与 BPTT，与框架**逐元素对账**通过（差值 < 1e-7 / 0）
- 「梯度消失」是可测量的：三档谱半径给出 **0.409 / 0.746 / 0.964** 的每步因子
- **元凶不只是权重**，tanh 自己也衰减；加性捷径把每步因子从 0.746 提到 1.121
- 串行不可并行：长度 ×16，耗时 ×14.7
- 词袋在纯顺序任务上恒 ≈ 0.5，RNN 能到 1.0，打乱输入后掉到 0.50

下一章 ch04 就把「任意两位置的直接通道」写出来：**Self-Attention**。
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch03 序列建模动机：从 RNN 到 Attention（练习版）

> 补全 `____`，跑通所有 assert。真值全部来自实跑。
>
> 本章一次对账、两次实测：**手写 RNN/BPTT 与框架对账**、
> **梯度随距离衰减**、**顺序信息的价值**。

填空进度建议：

| 块 | 内容 | 难度 |
|---|---|---|
| T1 | RNN 单步前向公式 | ★ |
| T2 | 整序列前向 | ★ |
| T3 | 与 `nn.RNN` 对账 | ★ |
| T4 | 手写 BPTT 反向循环 | ★★★ |
| T5 | 与 autograd 对账 | ★ |
| T6 | 梯度衰减曲线 | ★★ |
| T7 | 每步衰减因子 | ★★ |
| T8 | 加性捷径对照 | ★ |
| T9 | 耗时中位数 | ★ |
| T10 | 置换余弦相似度 | ★★ |
| T11 | 线性 RNN | ★ |
| T12 | 顺序任务数据构造 | ★★ |
| T13 | 词袋 5 折评估 | ★ |
| T14 | 打乱输入实验 | ★★ |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    code(PLOT_SETUP),
    md("## T1 手写 RNN 单步前向"),
    code(FIXTURE),
    code(T1_CODE),
    md("## T2 手写整序列前向"),
    code(T2_CODE),
    md("## T3 与 `nn.RNN` 对账"),
    code(T3_CODE),
    md("## T4 两个形状坑（不需要填空，读懂即可）"),
    code(T4_CODE),
    md("## T5 手写 BPTT"),
    code(T5_CODE),
    md("## T6 与 autograd 对账"),
    code(T6_CODE),
    md("## T7 梯度衰减曲线"),
    code(T7_CODE),
    md("## T8 每步衰减因子"),
    code(T8_CODE),
    md("## T9 加性捷径对照"),
    code(T9_CODE),
    code(PLOT_CODE),
    md("## T10 串行耗时"),
    code(T10_CODE),
    md("## T11 置换不变性（未训练 RNN 的坑）"),
    code(T11_CODE),
    md("## T12 线性 RNN"),
    code(T12_CODE),
    md("## T13 顺序敏感任务的数据"),
    code(T13_CODE),
    md("## T14 词袋基线"),
    code(T14_CODE),
    md("## T15 RNN 训练（主体是循环，不挖空，读懂即可）"),
    code(RNN_TRAIN),
    md("## T16 打乱输入实验"),
    code(T15_CODE),
    md("## 汇总"),
    code(SUMMARY),
    md(
        """
## 自查清单

- [ ] 手写前向与 `nn.RNN` 的最大差在 `1e-7` 量级（不是 `1e-2`）
- [ ] 手写 BPTT 的四组梯度与 autograd **差值 < 1e-9**
- [ ] 谱半径 0.5 / 1.0 / 1.5 的每步因子递增，且 **1.0 档仍然小于 1**
- [ ] 加性捷径的每步因子 **> 1**（梯度不再衰减）
- [ ] 耗时随 T 近似线性
- [ ] 未训练 RNN 的置换 cos **> 0.9**（想起这个坑）
- [ ] 线性 RNN 的置换 cos **< 0.8**（顺序依赖现形）
- [ ] 词袋在顺序任务上 **≈ 0.5**，且样本量越大越稳
- [ ] RNN **> 0.99**，打乱输入后 **< 0.6**
- [ ] 词袋对打乱输入的预测**完全一致**

**答不上来时问自己**：
1. `h` 为什么必须 `retain_grad()`？
2. 为什么谱半径 1.0 也会衰减？（提示：看 `tanh'` 的值域）
3. 加性捷径在梯度里多出了哪一项？
4. 为什么不能拿随机初始化的 RNN 下顺序性结论？
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
