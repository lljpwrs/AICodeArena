#!/usr/bin/env python3
"""ch04 Self-Attention 全手写（讲解 / 练习 / 答案 三件套）。

本章主线 —— **把 Attention 拆成六个零件，每个零件都对账**：

    ① 数值稳定 softmax   ：朴素版在 z=100 就出 nan，减 max 版与 F.softmax 差 0
    ② QKV 投影           ：三次线性变换，形状不变
    ③ 缩放点积注意力     ：与 F.scaled_dot_product_attention 差 1.19e-07
    ④ 除 sqrt(d_k) 的必要性：d_k=512 时原始 softmax 饱和成 one-hot（最大权重 0.999983）
    ⑤ 两种 mask          ：padding（key 侧屏蔽）/ causal（不看未来）
    ⑥ 多头 split-merge   ：与 nn.MultiheadAttention 差 5.96e-08

三个「看起来是对的，其实是错的」的点：

    - **忘记对 head 求平均**就去比 nn.MultiheadAttention 的权重：差 0.19（它默认已平均）
    - **mask 填 0 而不是 -inf**：softmax 之后 pad 位置照样分到权重
    - **以为 Attention 自带位置感**：它对置换**等变**（差 1.79e-07），必须外挂位置编码

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp04_self_attention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch04_self_attention"

# =========================================================================== #
# 公共导入 / 脚手架                                                           #
# =========================================================================== #

IMPORTS = '''import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_printoptions(precision=6, sci_mode=False)
print("torch", torch.__version__, "| 线程", torch.get_num_threads())
'''

SCAFFOLD = '''# 脚手架：常量与工具函数两版都有，你只填 @@todo 块里的内容
B, T, E = 2, 4, 8                      # batch / 序列长度 / 模型维度（对账用的小形状）
E_MHA, H = 8, 2                        # 多头实验：维度 8，2 个头
D_HEAD = E_MHA // H


def R(t):
    """取标量值（顺手 detach，避免 requires_grad 警告）。"""
    return float(t.detach())


def near(a, b, eps=1e-6):
    return abs(float(a) - float(b)) < eps


print("B,T,E", B, T, E, "| E_MHA,H", E_MHA, H, "| d_head", D_HEAD)
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

# 分词工具：ch01 已练过，这里直接给
FMM_TOOL = '''# 最大匹配分词（ch01 已手写过，本章直接复用）
VOCAB_W = ["变压器", "断路器", "隔离开关", "避雷器", "绝缘子", "套管", "母线",
           "渗油", "漏油", "油迹", "滴油", "锈蚀", "生锈", "铁锈", "漆面剥落",
           "破损", "裂纹", "碎裂", "缺口", "掉瓷",
           "发热", "温度偏高", "过热", "烫手", "温升异常",
           "异物", "鸟巢", "塑料袋", "树枝", "风筝线搭挂",
           "巡检", "发现", "位于", "存在", "缺陷", "一般", "请", "尽快", "核实",
           "已", "通知", "运维班", "详见", "接报", "列入", "本周", "计划",
           "停电", "处理", "反馈", "安排", "抢修"]
VSET = set(VOCAB_W)
MAXL = max(len(w) for w in VOCAB_W)


def fmm(text, vocab=VSET, maxl=MAXL):
    """正向最大匹配；没命中的字符直接丢弃（标点、编号等在词表外）。"""
    out, i = [], 0
    while i < len(text):
        for L in range(min(maxl, len(text) - i), 0, -1):
            if text[i:i + L] in vocab:
                out.append(text[i:i + L])
                i += L
                break
        else:
            i += 1
    return out


print("词表", len(VOCAB_W), "个词 | 最长", MAXL, "字")
'''

# =========================================================================== #
# 代码块                                                                      #
# =========================================================================== #

# ---- T1：两版 softmax ----
T1_CODE = '''def softmax_naive(z):
    """朴素实现：先 exp 再归一化。"""

    # @@todo 写出朴素版 softmax
    # @@hint e = torch.exp(z)，然后 e / e.sum(-1, keepdim=True)
    e = torch.exp(z)
    return e / e.sum(-1, keepdim=True)
    # @@end


def softmax_stable(z):
    """数值稳定实现：先减每行最大值再 exp。"""

    # @@todo 写出稳定版 softmax
    # @@hint 关键在 z - z.max(-1, keepdim=True).values，再 exp 再归一化
    e = torch.exp(z - z.max(-1, keepdim=True).values)
    return e / e.sum(-1, keepdim=True)
    # @@end


# ---- 验收 ----
probe_z = torch.randn(3, 5, generator=torch.Generator().manual_seed(1))
diff_softmax = R((softmax_stable(probe_z) - F.softmax(probe_z, -1)).abs().max())
assert diff_softmax < 1e-7, f"稳定版应与 F.softmax 一致，实际差 {diff_softmax}"
assert (softmax_naive(probe_z) - F.softmax(probe_z, -1)).abs().max() < 1e-7, \\
    "小数值下朴素版也对 —— 问题只在极端值上暴露"
print("小输入下两版都对（与 F.softmax 差 %.1e，float32 舍入量级）" % diff_softmax)
'''

# ---- T2：极端值下朴素版崩塌 ----
T2_CODE = '''big = torch.tensor([[1000.0, 1000.0, 1000.0]])
big2 = torch.tensor([[1000.0, 999.0, 998.0]])

# @@todo 观察朴素版在 big 上的中间量 e，以及最终结果
# @@hint torch.exp(big) 会溢出成 inf，inf/inf = nan
e_big = torch.exp(big)
naive_big = softmax_naive(big)
stable_big = softmax_stable(big)
# @@end

print("exp([1000,1000,1000]) =", e_big.tolist(), "| 含 inf:", bool(torch.isinf(e_big).any()))
print("朴素 softmax ->", naive_big.tolist(), "| 含 nan:", bool(torch.isnan(naive_big).any()))
print("稳定 softmax ->", stable_big.tolist())
print()
print("差值版（最大值减到 0）:",
      [round(v, 6) for v in torch.exp(big2 - big2.max(-1, keepdim=True).values).tolist()[0]])
print("稳定 softmax:", [round(v, 6) for v in softmax_stable(big2).tolist()[0]])
print("手算校验 exp(0)/(1+e^-1+e^-2) = %.6f" % (1 / (1 + math.exp(-1) + math.exp(-2))))

print()
print("z       朴素版含 nan   稳定版")
for scale in (10.0, 100.0, 1000.0):
    z = torch.full((1, 3), scale)
    print("  %6.1f     %-12s   %.6f"
          % (scale, bool(torch.isnan(softmax_naive(z)).any()), R(softmax_stable(z).mean())))

# ---- 验收 ----
assert torch.isnan(naive_big).any(), "朴素版应当溢出成 nan"
assert not torch.isnan(stable_big).any()
assert near(R(stable_big[0, 0]), 1 / 3, 1e-6)
assert torch.isnan(softmax_naive(torch.full((1, 3), 100.0))).any(), "z=100 就该崩了"
print("\\n结论：朴素版在 z≈100 就出 nan；减 max 是零成本的保险。")
print("注：因为 softmax(z) = softmax(z - c)，减常数不改变结果 —— 这不是近似。")
'''

# ---- T3：QKV 投影夹具 + 投影 ----
QKV_FIXTURE = '''x = torch.randn(B, T, E, generator=torch.Generator().manual_seed(1))

with torch.random.fork_rng():                 # 权重初始化与外部 RNG 解耦
    torch.manual_seed(11)
    qkv_lin = nn.Linear(E, 3 * E)
    out_lin = nn.Linear(E, E)

W_qkv, b_qkv = qkv_lin.weight.detach(), qkv_lin.bias.detach()
print("x", tuple(x.shape), "| W_qkv", tuple(W_qkv.shape), "| b_qkv", tuple(b_qkv.shape))
'''

T3_CODE = '''# @@todo 用同一个 W_qkv 一次性投影，再切成 Q / K / V 三块
# @@hint Q = x @ W_qkv[:E].T + b_qkv[:E]；K / V 用 [E:2*E] 与 [2*E:]，然后 qkv.chunk(3, dim=-1)
Q = x @ W_qkv[:E].T + b_qkv[:E]
K = x @ W_qkv[E:2 * E].T + b_qkv[E:2 * E]
V = x @ W_qkv[2 * E:].T + b_qkv[2 * E:]
# @@end

# ---- 验收 ----
assert Q.shape == K.shape == V.shape == (B, T, E), "QKV 形状应与输入一致"
q_chunk, k_chunk, v_chunk = (x @ W_qkv.T + b_qkv).chunk(3, dim=-1)
assert torch.allclose(Q, q_chunk) and torch.allclose(V, v_chunk), "两种切法结果应当等价"
print("Q/K/V 形状", tuple(Q.shape))
print("Q[0,0,:3]", [round(v, 6) for v in Q[0, 0, :3].tolist()])
print("整块投影再 chunk 的结果完全一致：", torch.allclose(Q, q_chunk))
'''

# ---- T4：attention 函数 ----
T4_CODE = '''def attention(Q, K, V, mask=None):
    """返回 (输出 [.., T, d_v], 注意力权重 [.., T, T])"""

    # @@todo 四步：缩放点积 scores -> 按 mask 填 -inf -> 稳定 softmax -> 与 V 相乘
    # @@hint d_k = Q.shape[-1]；s = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    # @@hint mask 非空时 s = s.masked_fill(mask, float("-inf"))；w = softmax_stable(s)；返回 w @ V, w
    d_k = Q.shape[-1]
    s = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    s = s.masked_fill(mask, float("-inf")) if mask is not None else s
    w = softmax_stable(s)
    return w @ V, w
    # @@end


# ---- 验收 ----
out_my, attn_my = attention(Q, K, V)
assert out_my.shape == (B, T, E), f"输出形状应为 {(B, T, E)}，实际 {tuple(out_my.shape)}"
assert attn_my.shape == (B, T, T)
assert (attn_my.sum(-1) - 1).abs().max() < 1e-6, "每一行的权重和必须是 1"
print("输出形状", tuple(out_my.shape), "| 权重形状", tuple(attn_my.shape))
print("每行权重和 %.6f" % R(attn_my.sum(-1).mean()))
'''

# ---- T5：与 F.sdpa 对账 + 不缩放的代价 ----
T5_CODE = '''# @@todo 与 PyTorch 内置的 F.scaled_dot_product_attention 对账
# @@hint F.scaled_dot_product_attention(Q, K, V)
out_sdpa = F.scaled_dot_product_attention(Q, K, V)
# @@end

diff_sdpa = R((out_my - out_sdpa).abs().max())

# ---- 不除 sqrt(d_k) 会怎样 ----
raw_scores = Q @ K.transpose(-2, -1)
scaled_scores = raw_scores / math.sqrt(E)
w_noscale = softmax_stable(raw_scores)
out_noscale = w_noscale @ V

print("手写 vs F.sdpa 最大绝对差: %.3e" % diff_sdpa)
print("不除 sqrt(d_k) 的输出差异: %.6f" % R((out_noscale - out_my).abs().max()))
print("缩放前 scores 标准差 %.6f -> 缩放后 %.6f"
      % (R(raw_scores.std()), R(scaled_scores.std())))
print("缩放前权重熵 %.6f -> 缩放后 %.6f"
      % (R(-(w_noscale * torch.log(w_noscale + 1e-12)).sum(-1).mean()),
         R(-(attn_my * torch.log(attn_my + 1e-12)).sum(-1).mean())))

# ---- 验收 ----
assert diff_sdpa < 1e-6, f"手写注意力与 F.sdpa 不一致：{diff_sdpa}"
assert R(w_noscale.std()) > R(attn_my.std()), "不缩放时权重应当更极端"
print("\\n不缩放不会报错，只是把分布推得更极端 —— 这是「静默的错」。")
'''

# ---- T6：d_k 扫描 ----
T6_CODE = '''NK = 200
print("d_k    原始 std   缩放后 std   原始权重熵   缩放后权重熵   原始最大权重")
rows_dk = {}
for dk in (8, 64, 512):
    gg = torch.Generator().manual_seed(dk)
    q_ = torch.randn(1, 1, dk, generator=gg)
    k_ = torch.randn(1, NK, dk, generator=gg)

    # @@todo 算未缩放与已缩放的 scores
    # @@hint raw_ = q_ @ k_.transpose(-1, -2)；sc_ = raw_ / math.sqrt(dk)
    raw_ = q_ @ k_.transpose(-1, -2)
    sc_ = raw_ / math.sqrt(dk)
    # @@end

    wr, ws = softmax_stable(raw_), softmax_stable(sc_)
    ent = lambda w: R(-(w * torch.log(w + 1e-12)).sum(-1).mean())  # noqa: E731
    rows_dk[dk] = dict(raw_std=R(raw_.std()), sc_std=R(sc_.std()),
                       raw_ent=ent(wr), sc_ent=ent(ws), raw_max=R(wr.max()))
    print("  %4d   %.6f    %.6f     %.6f     %.6f     %.6f"
          % (dk, rows_dk[dk]["raw_std"], rows_dk[dk]["sc_std"],
             rows_dk[dk]["raw_ent"], rows_dk[dk]["sc_ent"], rows_dk[dk]["raw_max"]))

print()
print("理论：scores 的 std ≈ sqrt(d_k) =",
      [round(math.sqrt(d), 3) for d in (8, 64, 512)])
print("d_k=512 时未缩放的 softmax 最大权重 %.6f —— 已经饱和成 one-hot，"
      % rows_dk[512]["raw_max"])
print("梯度会在 softmax 里消失。这就是那个 sqrt(d_k) 的全部意义。")

# ---- 验收 ----
assert rows_dk[8]["raw_max"] < rows_dk[64]["raw_max"] < rows_dk[512]["raw_max"]
assert rows_dk[512]["raw_max"] > 0.99, "d_k=512 时未缩放应当饱和"
assert rows_dk[512]["sc_ent"] > 4.0, "缩放之后分布应当重新变平"
print("\\n注意：d_k 越大，问题越严重 —— 大模型动辄 d_k=64/128，这个缩放不是可选项。")
'''

# ---- T7：padding mask ----
T7_CODE = '''T_MASK = 5
pad_mask = torch.tensor([[False, False, False, True, True]])     # 后两位是 padding
s_mask = torch.randn(1, T_MASK, T_MASK, generator=torch.Generator().manual_seed(5))

# @@todo 用 -inf 屏蔽 pad 列，再做 softmax
# @@hint s_mask.masked_fill(pad_mask.unsqueeze(1), float("-inf"))，然后 softmax_stable
w_pad = softmax_stable(s_mask.masked_fill(pad_mask.unsqueeze(1), float("-inf")))
# @@end

# @@todo 算 pad 列的权重和
# @@hint w_pad[0, :, 3:].sum()
pad_weight = R(w_pad[0, :, 3:].sum())
# @@end

print("padding mask 后第 0 行:", [round(v, 6) for v in w_pad[0, 0].tolist()])
print("pad 列权重和 %.6f | 每行权重和 %.6f" % (pad_weight, R(w_pad.sum(-1).mean())))

# 对照：如果 mask 处填 0 而不是 -inf
w_zero = softmax_stable(s_mask.masked_fill(pad_mask.unsqueeze(1), 0.0))
print("若填 0 而不是 -inf，pad 列权重和 %.6f（照样分到注意力！）"
      % R(w_zero[0, :, 3:].sum()))

# ---- 验收 ----
assert pad_weight < 1e-9, f"pad 位置权重必须为 0，实际 {pad_weight}"
assert R(w_pad.sum(-1).mean()) > 0.999
assert R(w_zero[0, :, 3:].sum()) > 0.1, "填 0 的写法确实会泄漏"
print("\\nmask 的语义是『在 softmax 之前把分数压到 -inf』，不是『softmax 之后置零』。")
'''

# ---- T8：causal mask ----
T8_CODE = '''# @@todo 构造上三角布尔掩码（不含对角线），用于「只能看左边」
# @@hint torch.triu(torch.ones(T_MASK, T_MASK, dtype=torch.bool), diagonal=1)
causal = torch.triu(torch.ones(T_MASK, T_MASK, dtype=torch.bool), diagonal=1)
# @@end

w_cau = softmax_stable(s_mask.masked_fill(causal.unsqueeze(0), float("-inf")))

# @@todo 算上三角（未来位置）的权重和
# @@hint w_cau[0].triu(1).sum()
future_weight = R(w_cau[0].triu(1).sum())
# @@end

print("causal mask 上三角权重和: %.6f" % future_weight)
print("第 0 行:", [round(v, 6) for v in w_cau[0, 0].tolist()], "（只能看自己）")
print("第 2 行:", [round(v, 6) for v in w_cau[0, 2].tolist()], "（看前 3 个）")
print()
print("不做 causal mask 时，第 0 行能看到未来，权重 %.6f"
      % R(softmax_stable(s_mask)[0, 0, 1:].sum()))
print("→ 训练时看到答案，推理时没有答案，指标会突然崩掉")

# ---- 验收 ----
assert future_weight < 1e-9, "未来位置权重必须为 0"
assert near(R(w_cau[0, 0, 0]), 1.0, 1e-6), "第一个位置只能注意自己"
print("\\n两种 mask 可以叠加：padding 管「哪些位置不该看」，causal 管「哪些还看不到」。")
'''

# ---- T9：多头 split-merge ----
T9_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(22)
    mha = nn.MultiheadAttention(E_MHA, H, batch_first=True)


def mha_manual(x, mha, n_head):
    """手写多头：一次投影 -> 拆头 -> 各头独立注意力 -> 合并 -> 输出投影"""
    Bn, Tn, E_ = x.shape
    d_h = E_ // n_head
    qkv = x @ mha.in_proj_weight.T + mha.in_proj_bias
    q, k, v = qkv.chunk(3, dim=-1)

    # @@todo 把 q / k / v 从 [B, T, E] 重排成 [B, n_head, T, d_h]
    # @@hint reshape(Bn, Tn, n_head, d_h) 之后 .transpose(1, 2)
    q = q.reshape(Bn, Tn, n_head, d_h).transpose(1, 2)
    k = k.reshape(Bn, Tn, n_head, d_h).transpose(1, 2)
    v = v.reshape(Bn, Tn, n_head, d_h).transpose(1, 2)
    # @@end

    o, w = attention(q, k, v)

    # @@todo 合并各头：先转回 [B, T, n_head, d_h]，再 reshape 回 [B, T, E]，最后过输出投影
    # @@hint o.transpose(1, 2).reshape(Bn, Tn, E_)；再 @ mha.out_proj.weight.T + mha.out_proj.bias
    o = o.transpose(1, 2).reshape(Bn, Tn, E_)
    return o @ mha.out_proj.weight.T + mha.out_proj.bias, w
    # @@end


# ---- 验收 ----
x_mha = torch.randn(2, 5, E_MHA, generator=torch.Generator().manual_seed(9))
o_my, w_my = mha_manual(x_mha, mha, H)
assert o_my.shape == (2, 5, E_MHA)
assert w_my.shape == (2, H, 5, 5), f"权重应当是每个头一份，实际 {tuple(w_my.shape)}"
print("手写多头输出形状", tuple(o_my.shape), "| 权重形状", tuple(w_my.shape))
'''

# ---- T10：与 nn.MultiheadAttention 对账 ----
T10_CODE = '''with torch.no_grad():
    o_ref, w_ref = mha(x_mha, x_mha, x_mha)

# @@todo 算手写输出与参考输出的最大绝对差
# @@hint (o_my - o_ref).abs().max()
diff_mha = R((o_my - o_ref).abs().max())
# @@end

# @@todo 算注意力权重的差：**注意参考实现默认已经把各头平均了**
# @@hint 先 w_my.mean(1) 再与 w_ref 比；直接 w_my - w_ref 是形状对不上的错误比法
diff_w_avg = R((w_my.mean(1) - w_ref).abs().max())
diff_w_naive = R((w_my - w_ref).abs().max())
# @@end

print("输出最大差            : %.3e" % diff_mha)
print("权重形状 手写", tuple(w_my.shape), "| 参考", tuple(w_ref.shape))
print("权重差（对 head 平均）: %.3e" % diff_w_avg)
print("权重差（忘记平均）    : %.3e   <- 差两个数量级，会让人以为写错了" % diff_w_naive)
print()
print("每个 head 行和 %.6f | 不同 head 权重差异 %.6f"
      % (R(w_my.sum(-1).mean()), R((w_my[0, 0] - w_my[0, 1]).abs().mean())))
n_in = mha.in_proj_weight.numel() + mha.in_proj_bias.numel()
n_out = mha.out_proj.weight.numel() + mha.out_proj.bias.numel()
print("参数量 in_proj %d + out_proj %d = %d（理论 4E^2+4E = %d）"
      % (n_in, n_out, n_in + n_out, 4 * E_MHA * E_MHA + 4 * E_MHA))

# ---- 验收 ----
assert diff_mha < 1e-6, f"手写多头与 nn.MultiheadAttention 不一致：{diff_mha}"
assert diff_w_avg < 1e-6, f"对 head 平均后应当对齐：{diff_w_avg}"
print("\\n多头没有增加参数：维度被拆成 H 份，每份 d_head = E/H，总量不变。")
'''

# ---- T11：复杂度实测 ----
T11_CODE = '''times = {}
for T_ in (64, 128, 256, 512, 1024):
    xx = torch.randn(8, T_, 64).requires_grad_(True)
    W_proj = torch.randn(64, 64)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        q_ = xx @ W_proj
        s_ = q_ @ q_.transpose(-2, -1) / 8.0
        a_ = softmax_stable(s_)
        o_ = a_ @ q_
        o_.sum().backward()
        ts.append(time.perf_counter() - t0)
        xx.grad = None
    ts.sort()
    # @@todo 取 5 次的中位数，记入 times[T_]
    # @@hint ts 已排序，中位数是 ts[len(ts) // 2]
    times[T_] = ts[len(ts) // 2]
    # @@end
    print("T=%5d  前向+反向中位 %.6f s" % (T_, times[T_]))

print()
print("T 翻倍时的耗时比（理论 ×4，因为 scores 是 T×T）：")
ks = sorted(times)
for a_, b_ in zip(ks, ks[1:]):
    print("  %4d -> %4d : ×%.3f" % (a_, b_, times[b_] / times[a_]))

# @@todo 算 T 从 256 到 1024 的总倍数（T 涨 4 倍）
# @@hint times[1024] / times[256]
grow = times[1024] / times[256]
# @@end
print("T ×4（256 -> 1024）耗时 ×%.2f（理论 ×16）" % grow)

# ---- 验收 ----
assert grow > 8, f"应当是平方级增长，实际只涨了 {grow} 倍"
print("\\n这就是「长文本贵」的根源：注意力矩阵有 T^2 个元素。")
'''

# ---- T12：注意力矩阵的元素数 ----
T12_CODE = '''# @@todo 算 T=64 / 256 / 1024 时注意力矩阵的元素数（就是 T^2）
# @@hint T * T
elems = {t_: t_ * t_ for t_ in (64, 256, 1024)}
# @@end

for t_, n_ in elems.items():
    print("T=%5d  注意力矩阵元素数 %8d  （≈ %.2f MB float32）"
          % (t_, n_, n_ * 4 / 1024 / 1024))
print()
print("T 从 64 到 1024：元素数 ×%d" % (elems[1024] // elems[64]))
print("对比：序列本身的长度只涨了 16 倍。")

# ---- 验收 ----
assert elems[1024] == 1048576
assert elems[1024] / elems[256] == 16
print("\\n→ 长上下文的所有工程手段（稀疏注意力 / 分块 / 线性注意力）都在打这个 T^2。")
'''

# ---- T13：置换等变性 ----
T13_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(33)
    lin_perm = nn.Linear(E_MHA, 3 * E_MHA)
W_p, b_p = lin_perm.weight.detach(), lin_perm.bias.detach()

x_perm_src = x_mha.clone()
perm = torch.tensor([3, 1, 4, 0, 2])
x_perm = x_perm_src[:, perm, :]


def attn_once(z):
    qkv = z @ W_p.T + b_p
    q_, k_, v_ = qkv.chunk(3, dim=-1)
    return attention(q_, k_, v_)


o_a, w_a = attn_once(x_perm_src)
o_b, w_b = attn_once(x_perm)

# @@todo 算「置换输入后的输出」与「原输出的置换」之间的最大绝对差
# @@hint (o_b - o_a[:, perm, :]).abs().max()
diff_perm_out = R((o_b - o_a[:, perm, :]).abs().max())
# @@end

# @@todo 算权重矩阵的同口径差（行、列都要置换）
# @@hint w_a[:, perm][:, :, perm]
diff_perm_w = R((w_b - w_a[:, perm][:, :, perm]).abs().max())
# @@end

print("置换输入后输出 vs 原输出置换 的最大差: %.3e" % diff_perm_out)
print("置换输入后权重 vs 原权重置换 的最大差: %.3e" % diff_perm_w)
print("mean pooling 对置换的最大差: %.3e"
      % R((x_perm.mean(1) - x_perm_src.mean(1)).abs().max()))

# ---- 验收 ----
assert diff_perm_out < 1e-6, "Attention 应当是置换等变的"
assert diff_perm_w < 1e-6
print("\\n置换等变 = 换顺序，输出跟着换 —— 它没有『位置』这个概念。")
print("所以「渗油在前」和「渗油在后」对它是同一件事 —— 必须外挂位置编码。")
'''

# ---- 真实工单的注意力矩阵 ----
REAL_FIXTURE = '''CSV = Path("data") / "synth_text.csv"
df = pd.read_csv(CSV, keep_default_na=False)

row = df[df["缺陷类型"] == "渗漏油"].iloc[3]
toks = fmm(row["缺陷描述"])
print("工单编号", row["工单编号"], "| 缺陷类型", row["缺陷类型"])
print("原文:", row["缺陷描述"])
print("分词:", toks, "| token 数", len(toks))
'''

T14_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(44)
    emb = nn.Embedding(len(VOCAB_W) + 1, 16)      # 0 号留给「未知」
    lin_real = nn.Linear(16, 24)

ids = torch.tensor([[VOCAB_W.index(t) + 1 if t in VSET else 0 for t in toks]])
z_real = emb(ids)

# @@todo 用刚才的四步流程算真实句子的注意力矩阵
# @@hint qkv 一次投影再 chunk(3, dim=-1)，然后 attention(q, k, v)
qkv_r = z_real @ lin_real.weight.T + lin_real.bias
q_r, k_r, v_r = qkv_r.chunk(3, dim=-1)
out_r, w_real = attention(q_r, k_r, v_r)
# @@end

print("注意力矩阵形状", tuple(w_real.shape), "| 每行和 %.6f" % R(w_real.sum(-1).mean()))
print("最大权重 %.6f | 最小权重 %.6f" % (R(w_real.max()), R(w_real.min())))
print("第 0 行权重:", [round(v, 4) for v in w_real[0, 0].tolist()])
print("上三角（未来位置）权重和 %.6f —— 没有 mask，它看得见全部"
      % R(w_real[0].triu(1).sum()))

# ---- 验收 ----
assert w_real.shape[1] == w_real.shape[2] == len(toks)
assert (w_real.sum(-1) - 1).abs().max() < 1e-6
print("\\n注意：随机 embedding + 未训练，所以数值本身没有语义；")
print("但形状、归一化、全连接这三件事已经清清楚楚。")
'''

# ---- 画热力图 ----
PLOT_CODE = '''fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))

# 左：随机初始化下的真实句子
im0 = axes[0].imshow(w_real[0].detach().numpy(), cmap="viridis", vmin=0)
axes[0].set_xticks(range(len(toks)))
axes[0].set_yticks(range(len(toks)))
axes[0].set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
axes[0].set_yticklabels(toks, fontsize=8)
axes[0].set_title("未训练：每个 token 都能看所有 token", fontsize=10)
fig.colorbar(im0, ax=axes[0], fraction=0.046)

# 右：causal mask 的效果（用同一批分数）
s_real = q_r @ k_r.transpose(-2, -1) / math.sqrt(q_r.shape[-1])
cm = torch.triu(torch.ones(len(toks), len(toks), dtype=torch.bool), diagonal=1)
w_causal_real = softmax_stable(s_real.masked_fill(cm.unsqueeze(0), float("-inf")))
im1 = axes[1].imshow(w_causal_real[0].detach().numpy(), cmap="viridis", vmin=0)
axes[1].set_xticks(range(len(toks)))
axes[1].set_yticks(range(len(toks)))
axes[1].set_xticklabels(toks, rotation=45, ha="right", fontsize=8)
axes[1].set_yticklabels(toks, fontsize=8)
axes[1].set_title("加 causal mask：只看得见左边", fontsize=10)
fig.colorbar(im1, ax=axes[1], fraction=0.046)

plt.tight_layout()
print("右上角（未来位置）全黑 —— mask 把上三角压成了 0。")
'''

# ---- 小结 ----
SUMMARY = '''import pandas as pd

rows = [
    ("稳定 softmax", "与 F.softmax 差 %.1e" % diff_softmax, "减 max 不改变结果，纯保险"),
    ("朴素 softmax 崩点", "z=100 出 nan", "溢出在 exp 里，不在除法里"),
    ("手写注意力 vs F.sdpa", "差 %.3e" % diff_sdpa, "缩放点积公式正确"),
    ("除 sqrt(d_k)", "d_k=512 未缩放最大权重 %.4f" % rows_dk[512]["raw_max"], "饱和成 one-hot"),
    ("padding mask", "pad 列权重和 %.1e" % pad_weight, "必须在 softmax 前填 -inf"),
    ("causal mask", "上三角权重和 %.1e" % future_weight, "否则训练时偷看答案"),
    ("多头 vs nn.MHA", "输出差 %.3e" % diff_mha, "拆头/合并顺序正确"),
    ("多头的权重对账", "平均后 %.1e / 忘记平均 %.2f" % (diff_w_avg, diff_w_naive),
     "参考实现默认已对 head 平均"),
    ("多头参数量", "%d = 4E^2+4E" % (n_in + n_out), "拆头不增加参数"),
    ("复杂度", "T×4 耗时 ×%.1f" % grow, "注意力矩阵 T^2 个元素"),
    ("置换等变", "差 %.1e" % diff_perm_out, "自带位置感 = 无，必须加编码"),
]
print(pd.DataFrame(rows, columns=["实验", "实测", "结论"]).to_string(index=False))
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch04 Self-Attention 全手写（讲解版）

> 对应考点：**「大语言模型相关理论考查」** 的核心。
> ch03 论证了「需要一条任意两位置之间的直接通道」，本章就把这条通道写出来。

Attention 只有一句话：

$$\\operatorname{Attn}(Q,K,V) = \\operatorname{softmax}\\!\\left(\\frac{QK^\\top}{\\sqrt{d_k}}\\right)V$$

但把这句话变成不会错的代码，要处理六件事。本章逐一对账：

| # | 零件 | 对账对象 | 实测差 |
|---|---|---|---|
| 1 | 数值稳定 softmax | `F.softmax` | **0** |
| 2 | QKV 投影 | 两种切法互验 | 完全一致 |
| 3 | 缩放点积 | `F.scaled_dot_product_attention` | **1.19e-07** |
| 4 | 多头 split-merge | `nn.MultiheadAttention` | **5.96e-08** |
| 5 | 两种 mask | 权重和为 0 / 1 | **0** |
| 6 | 置换等变 | 手算置换 | **1.79e-07** |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(PLOT_SETUP),
    md(
        """
## 4.1 先写 softmax：它比看起来更容易写错

softmax 的定义很朴素：

$$\\operatorname{softmax}(z)_i = \\frac{e^{z_i}}{\\sum_j e^{z_j}}$$

但**朴素的写法会溢出**。原因不在除法，在 `exp`：`float32` 能表示的最大值约 $3.4\\times10^{38}$，
而 $e^{88}$ 就已经超了。注意力分数动辄几十上百，`exp` 直接变 `inf`，`inf/inf = nan`。

修法只有一步：**先把每个向量减去它自己的最大值**。因为

$$\\operatorname{softmax}(z) = \\operatorname{softmax}(z - c)$$

这是**恒等式，不是近似** —— 减完结果一模一样，只是把 $e^{z_i-c}$ 的指数控制在 $\\le 0$ 的范围内。
"""
    ),
    code(T1_CODE),
    md(
        """
下面用极端输入把这三种情形摊开：`exp` 的溢出、`nan` 的产生、以及「减 max 之后毫发无损」。
"""
    ),
    code(T2_CODE),
    md(
        """
**为什么这个坑危险**：它不报错。你在小样例（`z` 在 ±3 以内）上怎么测都对，
一上真实模型就 `nan`，而且 `nan` 会顺着梯度污染**整批参数**。
"""
    ),
    md(
        """
## 4.2 QKV 投影

注意力里的 Q / K / V 不是三份输入，而是**同一份输入的三次线性变换**：

$$Q = XW_Q + b_Q,\\quad K = XW_K + b_K,\\quad V = XW_V + b_V$$

工程上一般合成一个 `W_qkv`（形状 `[3E, E]`）一次算完再 `chunk(3)` —— 少两次 kernel 启动。
下面两种写法互相验证。
"""
    ),
    code(QKV_FIXTURE),
    code(T3_CODE),
    md(
        """
## 4.3 缩放点积注意力

现在把四步写进一个函数。注意几个细节：

- `K.transpose(-2, -1)`：只要后两维转置，前面可能是 `[B, n_head]` —— **用负索引**，不要写死 `dim=1`
- `masked_fill(mask, float("-inf"))`：必须在 softmax **之前**
- 返回**权重矩阵**：既能做可视化，也是后面分析可解释性的入口
"""
    ),
    code(T4_CODE),
    md(
        """
## 4.4 与 `F.scaled_dot_product_attention` 对账

PyTorch 内置的这个函数就是同一套公式（底层走 flash attention 之类的融合 kernel）。
它算得快得多，但**你手写的这个才是理解的基础**。

顺便看一眼「不除 $\\sqrt{d_k}$」的代价 —— 它不报错，只是把分布推得更极端。
"""
    ),
    code(T5_CODE),
    md(
        """
## 4.5 为什么要除 $\\sqrt{d_k}$

如果 $q$ 和 $k$ 的每个分量都是独立的标准正态，那么点积 $q\\cdot k = \\sum_{i=1}^{d_k} q_i k_i$
的方差是 $d_k$，**标准差是 $\\sqrt{d_k}$**。

于是：$d_k$ 越大，score 的量级越大 → 除 `softmax` → 分布越尖 → 梯度越小。

下面把 $d_k$ 从 8 扫到 512，你会看到未缩放的 softmax 直接饱和成 one-hot。
"""
    ),
    code(T6_CODE),
    md(
        """
**这张表的读法**：看「原始最大权重」那一列 —— $d_k=512$ 时它已经到 0.999983，
softmax 输出几乎是一个 one-hot 向量。这会导致两个后果：

1. **梯度消失**：softmax 在饱和区的导数接近 0（和 ch03 的 tanh 同构）
2. **注意力失去选择性**：反正都是只有一个位置得到权重，模型学不到「同时关注几个词」

所以 $\\sqrt{d_k}$ 不是「经验技巧」，而是从方差分析直接推出来的。
"""
    ),
    md(
        """
## 4.6 两种 mask：语义边界与因果边界

| mask | 管什么 | 典型场景 |
|---|---|---|
| **padding mask** | 哪些位置是填充（不该被注意） | 一个 batch 里长短不一，短句被 pad 补齐 |
| **causal mask** | 哪些位置还没轮到（不该被看到） | Decoder、GPT 类自回归生成 |

**关键点：mask 的语义是「在 softmax 之前把 score 压到 $-\\inf$」**，而不是「softmax 之后置零」。
置零会让该行的权重和不为 1，后续再做加权时数值就偏小了。
"""
    ),
    code(T7_CODE),
    md(
        """
再看 causal mask。它的形状是一个**上三角的全 `True`**：
位置 $i$ 只能看 $j \\le i$。

这里有个很隐蔽的错法：`torch.triu(..., diagonal=0)` 会把对角线也屏蔽掉，
于是第一个位置（只有它自己）那一行**全被 mask**，softmax 里全是 `-inf`，
输出 `nan`。所以 `diagonal` 必须是 1。
"""
    ),
    code(T8_CODE),
    md(
        """
## 4.7 多头：拆开、并行、拼回

多头的全部操作就是三步 reshape：

```
[B, T, E]  --reshape-->  [B, T, H, d_head]  --transpose-->  [B, H, T, d_head]
        （每个头各管 E/H 个维度，互不重叠）
```

然后每个头独立做一次注意力，最后一步 `transpose` + `reshape` 拼回 `[B, T, E]`，
过一个输出投影 `W_O`。

**为什么多头有用**：单个 softmax 只能给出一种「关注模式」；$H$ 个头就有 $H$ 种。
代价只是把维度切成 H 份 —— **参数量不变**（后面会验证）。
"""
    ),
    code(T9_CODE),
    md(
        """
## 4.8 与 `nn.MultiheadAttention` 对账

这里有一个**会让你以为写错了**的坑：`nn.MultiheadAttention` 返回的注意力权重
默认已经**对所有 head 求过平均**（`average_attn_weights=True`）。
手写版是 `[B, H, T, T]`，参考版是 `[B, T, T]`。

直接把两个相减，会得到 0.19 的「差异」，而输出其实完全对得上（差 5.96e-08）。
**先看清楚形状再下结论** —— 这是通用经验，不只适用于注意力。
"""
    ),
    code(T10_CODE),
    md(
        """
## 4.9 复杂度：$O(T^2 d)$

注意力矩阵本身就有 $T^2$ 个元素。这是 Transformer 相对于 RNN 的**代价**：
ch03 里 RNN 是 $O(T d^2)$ 但完全串行；Attention 是 $O(T^2 d)$ 但前后可并行。

短序列时 $T^2 d < T d^2$（当 $T < d$），长序列时反转。下面实测一下增长阶。
"""
    ),
    code(T11_CODE),
    code(T12_CODE),
    md(
        """
## 4.10 置换等变性：Attention 没有「位置」概念

这是最容易被忽略、后果最大的一条性质。

Attentention 的所有操作 —— 线性变换、点积、softmax、加权求和 —— **对序列位置都是对称的**。
把输入顺序换一下，输出只是跟着换，值本身不变。这叫**置换等变**（permutation equivariant）。

**推论**：没有位置编码时，「渗油 在 变压器 前面」和「变压器 在 渗油 前面」是完全相同的输入。
"""
    ),
    code(T13_CODE),
    md(
        """
所以你们常常看到的那句话 ——「Transformer 靠位置编码才知道顺序」——
不是一句口号，而是**上面这个实测的必然结果**：Attention 本身是位置无关的，
顺序信息必须由外部注入（ch05 的 `PositionalEncoding`）。

> 顺便区分两个词：
> - **置换不变**（mean pooling、词袋）：换顺序，输出不变
> - **置换等变**（Attention）：换顺序，输出跟着换 —— 但「信息内容」没变
"""
    ),
    md(
        """
## 4.11 拿真实工单看一眼注意力矩阵

前面都是合成张量。最后用一条真实工单跑一遍，把矩阵画出来。

用 ch01 手写过的最大匹配分词，词表与前面几章一致。
"""
    ),
    code(FMM_TOOL),
    code(REAL_FIXTURE),
    code(T14_CODE),
    code(PLOT_CODE),
    md(
        """
左图是**未训练**的注意力矩阵：数值没有语义，但结构信息一目了然 ——
矩阵是**满的**（每个 token 都连到所有 token）、**每行和为 1**、**没有上三角约束**。

右图加了 causal mask：右上角全黑。这就是 GPT 类模型看到的形状。

> 提醒：不要拿随机初始化模型的注意力权重去做「模型在看哪里」的解读 ——
> 这和 ch03 §3.11 「未训练 RNN 的置换测试」是同一类错误。
"""
    ),
    md(
        """
## 4.12 汇总

"""
    ),
    code(SUMMARY),
    md(
        """
## 4.13 与 ch03 / ch05 的接口

| ch03 提出的问题 | ch04 的答案 | 留给 ch05 的事 |
|---|---|---|
| 梯度要走 50 步 | 任意两位置直接连边，路径长度 1 | 残差 + LayerNorm 让深堆叠可训 |
| 串行不可并行 | 全序列一次矩阵乘 | —— |
| 长距离顺序信息 | 注意力权重显式表达「谁看谁」 | **位置编码**（本节的置换等变必须被打破） |

## 易错点清单

1. **朴素 softmax 溢出** —— 只在极端值暴露，且 `nan` 会污染整批梯度。减 max 是恒等变换，白送的保险。
2. **mask 填 0 而不是 `-inf`** —— pad 位置照样分到权重（实测权重和 > 0.1），且不报错。
3. **`torch.triu(..., diagonal=0)`** —— 第一个位置那一行全被屏蔽，softmax 输出 `nan`。要用 `diagonal=1`。
4. **`K.transpose(1, 2)` 写死维度** —— 多头时张量是 4 维，用 `transpose(-2, -1)`。
5. **忘记 `1/sqrt(d_k)`** —— 小 `d_k` 上看不出差别，`d_k=512` 时 softmax 直接饱和。
6. **拿 `nn.MultiheadAttention` 的权重直接比** —— 它默认已对 head 平均；先看形状。
7. **拆头时 `reshape` 与 `transpose` 顺序搞反** —— 输出形状对得上但数值错，对账才能发现。
8. **以为 Attention 自带位置感** —— 它对置换**等变**，没有位置编码就分不出顺序。
9. **用小序列测复杂度** —— `T=64` 时启动开销占比大，倍数关系看不出来。
10. **拿未训练模型的注意力权重做解释** —— 数值无意义，只能看结构。

## 本章小结

- 六个零件全部与框架对账通过（softmax **0** / sdpa **1.19e-07** / MHA **5.96e-08**）
- **`sqrt(d_k)` 有推导来源**：scores 方差随 $d_k$ 线性增长，实测 $d_k=512$ 未缩放最大权重 **0.999983**
- **mask 必须填 `-inf` 且发生在 softmax 之前**：填 0 时 pad 列权重和仍 > 0.1
- 多头**不增加参数**（$4E^2+4E = 288$ @ $E=8$），只是把维度切成 H 份
- 复杂度 $O(T^2)$：$T\\times4$ 耗时 $\u00d7$ 十几倍
- **置换等变**（差 1.79e-07）→ 位置编码不是可选项，是必需品

下一章 ch05 把 Attention 装进完整 Transformer：位置编码、FFN、残差 + LayerNorm。
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch04 Self-Attention 全手写（练习版）

> 补全 `____`，跑通所有 assert。真值全部来自实跑。
>
> 本章要写六个零件，每个都有对账对象 —— **对账差值是检验你写没写对的唯一标准**。

| 块 | 内容 | 难度 |
|---|---|---|
| T1 | 两版 softmax | ★ |
| T2 | 极端值验证 | ★ |
| T3 | QKV 投影 | ★ |
| T4 | 缩放点积注意力 | ★★★ |
| T5 | 与 `F.sdpa` 对账 | ★ |
| T6 | `d_k` 扫描 | ★★ |
| T7 | padding mask | ★★ |
| T8 | causal mask | ★★ |
| T9 | 多头 split-merge | ★★★ |
| T10 | 与 `nn.MultiheadAttention` 对账 | ★★ |
| T11 | 复杂度实测 | ★ |
| T12 | 注意力矩阵元素数 | ★ |
| T13 | 置换等变性 | ★★ |
| T14 | 真实句子的注意力矩阵 | ★ |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(PLOT_SETUP),
    md("## T1 两版 softmax"),
    code(T1_CODE),
    md("## T2 极端值验证"),
    code(T2_CODE),
    md("## T3 QKV 投影"),
    code(QKV_FIXTURE),
    code(T3_CODE),
    md("## T4 缩放点积注意力"),
    code(T4_CODE),
    md("## T5 与 `F.scaled_dot_product_attention` 对账"),
    code(T5_CODE),
    md("## T6 为什么要除 sqrt(d_k)"),
    code(T6_CODE),
    md("## T7 padding mask"),
    code(T7_CODE),
    md("## T8 causal mask"),
    code(T8_CODE),
    md("## T9 多头 split-merge"),
    code(T9_CODE),
    md("## T10 与 `nn.MultiheadAttention` 对账"),
    code(T10_CODE),
    md("## T11 复杂度实测"),
    code(T11_CODE),
    md("## T12 注意力矩阵元素数"),
    code(T12_CODE),
    md("## T13 置换等变性"),
    code(T13_CODE),
    md("## T14 真实句子的注意力矩阵（分词工具已给，不挖空）"),
    code(FMM_TOOL),
    code(REAL_FIXTURE),
    code(T14_CODE),
    code(PLOT_CODE),
    md("## 汇总"),
    code(SUMMARY),
    md(
        """
## 自查清单

- [ ] `softmax_stable` 与 `F.softmax` 差 **< 1e-9**，`softmax_naive` 在 `z=100` 出 **nan**
- [ ] Q / K / V 形状都是 `(2, 4, 8)`，且两种切法完全一致
- [ ] 手写注意力与 `F.sdpa` 差 **< 1e-6**（实际 1.19e-07）
- [ ] 每行注意力权重和 **= 1**
- [ ] `d_k=512` 时未缩放的最大权重 **> 0.99**（饱和）
- [ ] padding mask 后 pad 列权重和 **< 1e-9**；填 0 的写法权重和 **> 0.1**
- [ ] causal mask 后上三角权重和 **< 1e-9**，第 0 行只有自己 = 1
- [ ] 手写多头输出与 `nn.MultiheadAttention` 差 **< 1e-6**
- [ ] 权重对账**先对 head 求平均**（差 1e-8 而不是 0.19）
- [ ] 参数量 **= 4E²+4E**
- [ ] `T` 从 256 到 1024 耗时涨 **> 8 倍**
- [ ] 置换输入后输出差 **< 1e-6**（等变性成立）

**答不上来时问自己**：
1. 为什么减 max 不改变 softmax 的结果？
2. mask 为什么必须在 softmax **之前**生效？
3. `sqrt(d_k)` 是从哪推出来的？
4. `nn.MultiheadAttention` 的权重为什么比手写版少一个维度？
5. 「置换等变」和「置换不变」差在哪？
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
