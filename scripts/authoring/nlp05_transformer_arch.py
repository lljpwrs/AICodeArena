#!/usr/bin/env python3
"""ch05 Transformer 完整架构手写（讲解 / 练习 / 答案 三件套）。

本章主线 —— **把 ch04 的注意力装成一台能用的机器**：

    ① 位置编码 sin/cos ：补上 ch04 缺失的「位置感」，附实测相似度曲线
    ② 手写 LayerNorm   ：与 nn.LayerNorm 差 2.38e-07（关键在 unbiased=False）
    ③ FFN              ：d_ff = 4·d_model，逐位置独立
    ④ 残差 + Pre/Post-LN：**本章最有价值的一组对照**
         Post-LN 逐层梯度首/末 = 0.000007（末层是首层的 14 万倍）
         Pre-LN  逐层梯度首/末 = 1.613922（几乎不衰减）
         无残差   逐层梯度首/末 = 0.000269
    ⑤ Encoder / Decoder 层对账：Encoder 差 0.000e+00，Decoder 差 3.58e-07
    ⑥ 参数量统计       ：单层 Encoder 2224 = 手算 4E²+4E + 2E·d_ff + 6E
    ⑦ 位置编码必要性   ：复用 ch03 的「谁先出现」任务
         有 PE 0.976 / 无 PE 0.528 / 有 PE + 打乱 0.486

本章回答的是「为什么大模型用 Pre-LN」这个工程问题 —— 用实测的梯度分布，不是传言。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp05_transformer_arch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch05_transformer_arch"

# =========================================================================== #
# 公共导入 / 脚手架                                                           #
# =========================================================================== #

IMPORTS = '''import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
torch.set_printoptions(precision=6, sci_mode=False)
print("torch", torch.__version__, "| 线程", torch.get_num_threads())
'''

SCAFFOLD = '''# 脚手架：常量与工具函数两版都有，你只填 @@todo 块里的内容
T_PE, D_PE = 20, 16          # 位置编码实验
D_LN = 16                    # LayerNorm 实验
D_M, D_FF = 16, 64           # FFN 实验
D_S, H_S, N_LAYER, T_S = 32, 4, 6, 16      # 堆叠实验
D_E, H_E, D_FF_E, T_E, B_E = 16, 2, 32, 7, 3   # Encoder/Decoder 层实验


def R(t):
    """取标量值（顺手 detach）。"""
    return float(t.detach())


def near(a, b, eps=1e-6):
    return abs(float(a) - float(b)) < eps


def cos_sim(a, b):
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)))


print("T_PE,D_PE", T_PE, D_PE, "| D_LN", D_LN, "| D_M,D_FF", D_M, D_FF,
      "| D_S,N_LAYER", D_S, N_LAYER, "| D_E,H_E,D_FF_E", D_E, H_E, D_FF_E)
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

# 顺序任务的数据构造：ch03 已手写过，本章直接复用
TASK_DATA = '''# 「谁先出现」任务的数据构造（ch03 已手写过，本章直接复用）
A_W = ["渗油", "漏油", "油迹", "油污", "滴油"]
B_W = ["发热", "温度偏高", "过热", "烫手", "温升异常"]
MID = ["巡检", "发现", "位于", "存在", "缺陷", "一般",
       "请", "尽快", "核实", "已", "通知", "运维班"]
VOCAB = ["<pad>"] + A_W + B_W + MID
VID = {w: i for i, w in enumerate(VOCAB)}
T_TASK = 8


def make_order_data(n, seed, T=T_TASK):
    rg = np.random.default_rng(seed)
    X = np.zeros((n, T), dtype=np.int64)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p, q = rg.choice(T, size=2, replace=False)
        seq = [MID[rg.integers(len(MID))] for _ in range(T)]
        seq[p] = A_W[rg.integers(len(A_W))]
        seq[q] = B_W[rg.integers(len(B_W))]
        X[i] = [VID[w] for w in seq]
        y[i] = int(p < q)
    return X, y


Xtr, ytr = make_order_data(800, 1)
Xte, yte = make_order_data(500, 2)
print("词表", len(VOCAB), "| 训练", Xtr.shape, "| 测试", Xte.shape,
      "| 标签分布", {int(k): int(v) for k, v in zip(*np.unique(ytr, return_counts=True))})
'''

# =========================================================================== #
# 代码块                                                                      #
# =========================================================================== #

# ---- T1：位置编码 ----
T1_CODE = '''def positional_encoding(T, d_model):
    """sin/cos 位置编码，返回 [T, d_model]"""

    pe = torch.zeros(T, d_model)
    pos = torch.arange(T, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model))

    # @@todo 偶数维填 sin(pos * div)，奇数维填 cos(pos * div)
    # @@hint pe[:, 0::2] = torch.sin(pos * div)；pe[:, 1::2] = torch.cos(pos * div)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    # @@end

    return pe


# ---- 验收 ----
pe = positional_encoding(T_PE, D_PE)
assert pe.shape == (T_PE, D_PE), f"形状应为 {(T_PE, D_PE)}，实际 {tuple(pe.shape)}"
assert near(R(pe[0, 0]), 0.0, 1e-6) and near(R(pe[0, 1]), 1.0, 1e-6), "pos=0 应为 sin(0)=0, cos(0)=1"
assert R(pe.abs().max()) <= 1.0 + 1e-6, "sin/cos 值域必须在 [-1, 1]"
print("PE 形状", tuple(pe.shape))
print("PE[0,:4]", [round(v, 6) for v in pe[0, :4].tolist()])
print("PE[1,:4]", [round(v, 6) for v in pe[1, :4].tolist()])
print("每行范数 %.6f（≈ sqrt(d/2) = %.6f）" % (R(pe.norm(dim=1).mean()), math.sqrt(D_PE / 2)))
'''

# ---- T2：位置相似度曲线 ----
T2_CODE = '''# @@todo 算位置 0 与位置 0+k 的余弦相似度（k = 1, 2, 4, 8, 16, 19）
# @@hint 用上面定义的 cos_sim(pe[0], pe[k])
sim_delta = {k: cos_sim(pe[0], pe[k]) for k in (1, 2, 4, 8, 16, 19)}
# @@end

print("位置 0 与其它位置的余弦相似度：")
for k, v in sim_delta.items():
    print("  |Δ|=%-3d  cos %.6f" % (k, v))
print("→ 近邻相似度最高（|Δ|=1 的 %.6f），随距离总体下降" % sim_delta[1])
print("→ 但 sin/cos 有周期性，远距离会「回环」（|Δ|=19 回升到 %.6f）" % sim_delta[19])

# @@todo 算低维与高维的角频率（周期倒数）
# @@hint 第 k 对维度的 freq = exp(k * -log(10000) / d_model)
freq_low = math.exp(0 * -math.log(10000.0) / D_PE)
freq_high = math.exp(7 * -math.log(10000.0) / D_PE)
# @@end
print()
print("维度对 0,1 的角频率 %.6f（周期 %.4f rad）" % (freq_low, 2 * math.pi / freq_low))
print("维度对 7,8 的角频率 %.6f（周期 %.4f rad）" % (freq_high, 2 * math.pi / freq_high))
print("→ 低维周期短（分辨近邻），高维周期长（分辨全局）——这是 sin/cos 编码的核心设计")

# ---- 验收 ----
assert sim_delta[1] > sim_delta[8] > sim_delta[16], "相似度应随距离总体下降"
assert freq_high < freq_low, "高维的角频率应当更小"
print()
print("补充：位置编码在相加前要考虑量级 —— 词向量与 PE 的范数比会在下面看到。")
'''

# ---- T3：词向量量级 ----
T3_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(3)
    emb = nn.Embedding(100, D_PE)

x_tok = emb(torch.tensor([[1, 5, 9, 13]]))
tok_norm = R(x_tok.norm(dim=-1).mean())
pe_norm = R(pe[:4].norm(dim=1).mean())

print("词向量范数均值 %.6f | PE 范数均值 %.6f | 比值 %.4f"
      % (tok_norm, pe_norm, tok_norm / pe_norm))
print("→ nn.Embedding 默认 N(0,1) 初始化，范数 ≈ sqrt(d) = %.4f" % math.sqrt(D_PE))
print("→ 相加时若量级悬殊，位置信息要么被淹没、要么盖过语义")

# ---- 验收 ----
assert tok_norm / pe_norm > 1.0, "默认初始化下词向量范数更大"
'''

# ---- T4：手写 LayerNorm ----
T4_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(4)
    ln_ref = nn.LayerNorm(D_LN)


def layernorm_manual(x, weight, bias, eps=1e-5):
    """x: [..., d]；在最后一维上做归一化"""

    # @@todo ①算均值 ②算方差（注意 unbaised=False！）③归一化 ④仿射
    # @@hint mu = x.mean(-1, keepdim=True)；var = x.var(-1, unbiased=False, keepdim=True)
    # @@hint (x - mu) / torch.sqrt(var + eps) * weight + bias
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * weight + bias
    # @@end


z = torch.randn(3, 5, D_LN, generator=torch.Generator().manual_seed(6)) * 5 + 2
out_ln = layernorm_manual(z, ln_ref.weight.detach(), ln_ref.bias.detach())
out_ref_ln = ln_ref(z)

# @@todo 算手写 LayerNorm 与 nn.LayerNorm 的最大绝对差
# @@hint (out_ln - out_ref_ln).abs().max()
diff_ln = R((out_ln - out_ref_ln).abs().max())
# @@end

print("手写 LayerNorm vs nn.LayerNorm 最大差: %.3e" % diff_ln)
print("nn.LayerNorm 的 eps 默认值: %s | 有仿射参数: %s"
      % (ln_ref.eps, ln_ref.elementwise_affine))

# ---- 验收 ----
assert diff_ln < 1e-6, f"手写 LayerNorm 与 nn.LayerNorm 不一致：{diff_ln}"
assert abs(R(out_ln.mean())) < 1e-6, "归一化后均值应约为 0"
'''

# ---- T5：unbiased 的陷阱 ----
T5_CODE = '''# @@todo 用样本方差（unbiased=True）重算一遍，看差多少
# @@hint var_wrong = z.var(-1, unbiased=True, keepdim=True)，其余同 layernorm
var_wrong = z.var(-1, unbiased=True, keepdim=True)
out_wrong = (z - z.mean(-1, keepdim=True)) / torch.sqrt(var_wrong + 1e-5)
# @@end

diff_wrong = R((out_wrong - out_ref_ln.detach()).abs().max())
print("用样本方差（unbiased=True）的最大差: %.6f" % diff_wrong)
print("  样本方差 / 总体方差 = n/(n-1) = %.6f" % (D_LN / (D_LN - 1)))
print("  这个偏差在 d=16 时只有几 %%，d 很小时会非常明显 —— 但不报错。")

# @@todo 用 unbiased=True 口径算每行标准差，对比 1.0
# @@hint out_ln.std(-1) 默认就是 unbiased=True
std_biased = [round(v, 6) for v in out_ln.std(-1)[0].tolist()]
# @@end

print()
print("逐 token 均值（应全为 0）:", [round(v, 8) for v in out_ln.mean(-1)[0].tolist()])
print("逐 token 标准差（unbiased 口径）:", std_biased)
print("→ 数值是 %.6f 而不是 1.0，因为 sqrt(d/(d-1)) = %.6f —— 同一个坑的两面"
      % (std_biased[0], math.sqrt(D_LN / (D_LN - 1))))

# ---- 验收 ----
assert diff_wrong > 0.01, "样本方差确实会带来可观测的偏差"
assert near(std_biased[0], math.sqrt(D_LN / (D_LN - 1)), 1e-5)
'''

# ---- T6：FFN ----
T6_CODE = '''def ffn_manual(x, w1, b1, w2, b2, act=F.relu):
    """两层线性 + 激活，逐位置独立"""

    # @@todo 实现 FFN：x -> w1 -> 激活 -> w2
    # @@hint act(x @ w1.T + b1) @ w2.T + b2
    return act(x @ w1.T + b1) @ w2.T + b2
    # @@end


with torch.random.fork_rng():
    torch.manual_seed(7)
    ref_ffn = nn.Sequential(nn.Linear(D_M, D_FF), nn.ReLU(), nn.Linear(D_FF, D_M))

x_ffn = torch.randn(2, 5, D_M, generator=torch.Generator().manual_seed(8))
out_ffn = ffn_manual(x_ffn, ref_ffn[0].weight.detach(), ref_ffn[0].bias.detach(),
                     ref_ffn[2].weight.detach(), ref_ffn[2].bias.detach())
diff_ffn = R((out_ffn - ref_ffn(x_ffn)).abs().max())
n_ffn = sum(p.numel() for p in ref_ffn.parameters() if p.requires_grad)

# @@todo 手算 FFN 参数量：两层线性 = d*d_ff + d_ff + d_ff*d + d
# @@hint 2*d*d_ff + d + d_ff
n_ffn_theory = 2 * D_M * D_FF + D_M + D_FF
# @@end

print("FFN 输出形状", tuple(out_ffn.shape), "| 与 nn.Sequential 最大差: %.3e" % diff_ffn)
print("参数量 实测 %d | 手算 %d | 一致: %s" % (n_ffn, n_ffn_theory, n_ffn == n_ffn_theory))
print("第一层占 %.4f（扩张到 4 倍是参数大头）" % ((D_M * D_FF + D_FF) / n_ffn))

# @@todo 验位置无关性：对输入做置换，输出应当只是同口径置换
# @@hint 用同一组权重算 ffn_manual(x_perm, ...)，与 out_ffn[:, perm, :] 比
perm5 = [3, 1, 4, 0, 2]
out_perm = ffn_manual(x_ffn[:, perm5, :], ref_ffn[0].weight.detach(), ref_ffn[0].bias.detach(),
                      ref_ffn[2].weight.detach(), ref_ffn[2].bias.detach())
diff_ffn_perm = R((out_perm - out_ffn[:, perm5, :]).abs().max())
# @@end

print("位置无关性：FFN(置换输入) vs FFN(输入)置换 最大差: %.3e" % diff_ffn_perm)

# ---- 验收 ----
assert diff_ffn < 1e-6
assert n_ffn == n_ffn_theory
assert diff_ffn_perm < 1e-6, "FFN 必须逐位置独立"
print("\\nFFN 不含任何跨位置交互 —— 序列建模的活全在注意力和位置编码里。")
'''

# ---- T7：Pre-LN / Post-LN 的 Block ----
BLOCK_CODE = '''class Block(nn.Module):
    """一个 Transformer 层：注意力 + FFN + 2 个 LayerNorm"""

    def __init__(self, d, h, pre_ln):
        super().__init__()
        self.pre = pre_ln
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)

    def forward(self, h):
        if self.pre:
            # @@todo Pre-LN：先归一化再进子层，子层输出直接加到 h 上
            # @@hint hn = self.n1(h)；h = h + attn(hn,hn,hn)[0]；h = h + self.ffn(self.n2(h))
            hn = self.n1(h)
            h = h + self.attn(hn, hn, hn)[0]
            h = h + self.ffn(self.n2(h))
            # @@end
        else:
            # @@todo Post-LN：先做残差相加，再归一化
            # @@hint h = self.n1(h + attn(h,h,h)[0])；h = self.n2(h + self.ffn(h))
            h = self.n1(h + self.attn(h, h, h)[0])
            h = self.n2(h + self.ffn(h))
            # @@end
        return h


class StackLN(nn.Module):
    def __init__(self, pre_ln, n_layer=N_LAYER, residual=True):
        super().__init__()
        self.residual = residual
        self.blocks = nn.ModuleList([Block(D_S, H_S, pre_ln) for _ in range(n_layer)])

    def forward(self, x):
        h = x.clone()
        for blk in self.blocks:
            if self.residual:
                h = blk(h)
            else:                                    # 消融：去掉两条残差
                hn = blk.n1(h)
                h = blk.n2(blk.ffn(hn) + blk.attn(hn, hn, hn)[0])
        return h


print("Block 定义完成：Pre-LN 与 Post-LN 用同一个类，靠 self.pre 切换")
'''

# ---- T8：逐层梯度对比 ----
T8_CODE = '''def run_stack(pre_ln, residual=True, n_layer=N_LAYER, seed=99):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = StackLN(pre_ln, n_layer, residual)
    xx = torch.randn(2, T_S, D_S, generator=torch.Generator().manual_seed(10))
    xx.requires_grad_(True)
    loss = (net(xx) ** 2).sum()
    loss.backward()
    # @@todo 收集每一层的参数梯度范数（把该层所有参数的梯度平方和开根）
    # @@hint per_layer = [sum(R(p.grad.norm()) ** 2 for p in blk.parameters()) ** 0.5 for blk in net.blocks]
    per_layer = [sum(R(p.grad.norm()) ** 2 for p in blk.parameters()) ** 0.5 for blk in net.blocks]
    # @@end
    return R(loss), R(xx.grad.norm()), per_layer


print("配置     ∂loss/∂x      逐层梯度范数（第 1 层 -> 第 6 层）")
results_ln = {}
for tag, pre_ln, res in (("Post-LN", False, True), ("Pre-LN", True, True),
                         ("无残差", False, False)):
    ls, gx, pl = run_stack(pre_ln, res)
    results_ln[tag] = dict(loss=ls, gx=gx, per_layer=pl, ratio=pl[0] / pl[-1])
    print("%-8s %.3e  %s" % (tag, gx, ["%.2e" % v for v in pl]))
    print("         首层/末层 = %.6f" % results_ln[tag]["ratio"])

# ---- 验收 ----
assert results_ln["Pre-LN"]["ratio"] > 1.0, "Pre-LN 的首层梯度不应小于末层"
assert results_ln["Post-LN"]["ratio"] < 0.01, "Post-LN 的浅层梯度应当明显更小"
print("\\nPost-LN 的末层梯度是首层的 %.0f 倍 —— 这就是「深堆叠训不动」的原因。"
      % (1 / results_ln["Post-LN"]["ratio"]))
print("Pre-LN 的比值 %.6f，梯度从头到尾量级一致。" % results_ln["Pre-LN"]["ratio"])
print("注：不同配置的 loss 尺度不同，绝对值不可比；**首层/末层的比值是尺度无关的**。")
'''

# ---- T9：Encoder 层对账 ----
T9_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(55)
    el = nn.TransformerEncoderLayer(D_E, H_E, D_FF_E, dropout=0.0, batch_first=True)
el.eval()


def encoder_layer_manual(x, el):
    """Post-LN 的 Encoder 层：x = norm(x + sublayer(x))"""

    # @@todo 三步：自注意力 -> norm1(x + attn) -> norm2(x + ffn)
    # @@hint a, _ = el.self_attn(x, x, x)；f = el.linear2(el.activation(el.linear1(x)))
    a, _ = el.self_attn(x, x, x)
    x = el.norm1(x + a)
    f = el.linear2(el.activation(el.linear1(x)))
    return el.norm2(x + f)
    # @@end


x_e = torch.randn(B_E, T_E, D_E, generator=torch.Generator().manual_seed(12))
with torch.no_grad():
    o_hand_enc = encoder_layer_manual(x_e, el)
    o_ref_enc = el(x_e)
diff_enc = R((o_hand_enc - o_ref_enc).abs().max())

n_attn = sum(p.numel() for p in el.self_attn.parameters())
n_ffn_el = (el.linear1.weight.numel() + el.linear1.bias.numel()
            + el.linear2.weight.numel() + el.linear2.bias.numel())
n_norm = sum(p.numel() for p in el.norm1.parameters()) + sum(p.numel() for p in el.norm2.parameters())

print("手写 Encoder 层 vs nn.TransformerEncoderLayer 最大差: %.3e" % diff_enc)
print("activation: %s | norm_first: %s" % (el.activation.__class__.__name__, el.norm_first))
print("参数量 self_attn %d + ffn %d + 2*LN %d = %d" % (n_attn, n_ffn_el, n_norm,
                                                      n_attn + n_ffn_el + n_norm))
print("  理论 self_attn 4E^2+4E = %d | ffn 2*E*d_ff+E+d_ff = %d | LN 4E = %d"
      % (4 * D_E * D_E + 4 * D_E, 2 * D_E * D_FF_E + D_E + D_FF_E, 4 * D_E))

# ---- 验收 ----
assert diff_enc < 1e-6, f"手写 Encoder 层不一致：{diff_enc}"
assert n_attn == 4 * D_E * D_E + 4 * D_E
'''

# ---- T10：Decoder 层对账 ----
T10_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(56)
    dl = nn.TransformerDecoderLayer(D_E, H_E, D_FF_E, dropout=0.0, batch_first=True)
dl.eval()
mem = torch.randn(B_E, T_E, D_E, generator=torch.Generator().manual_seed(13))
x_d = torch.randn(B_E, T_E, D_E, generator=torch.Generator().manual_seed(14))
cm = nn.Transformer.generate_square_subsequent_mask(T_E)


def decoder_layer_manual(x, memory, dl, mask):
    """Decoder 层比 Encoder 层多一个 Cross-Attention"""

    # @@todo 三步：causal 自注意力 -> Cross-Attention（Q=x, K/V=memory）-> FFN，各带残差+norm
    # @@hint a1,_ = dl.self_attn(x, x, x, attn_mask=mask)；a2,_ = dl.multihead_attn(x, memory, memory)
    a1, _ = dl.self_attn(x, x, x, attn_mask=mask)
    x = dl.norm1(x + a1)
    a2, _ = dl.multihead_attn(x, memory, memory)
    x = dl.norm2(x + a2)
    f = dl.linear2(dl.activation(dl.linear1(x)))
    return dl.norm3(x + f)
    # @@end


with torch.no_grad():
    o_hand_dec = decoder_layer_manual(x_d, mem, dl, cm)
    o_ref_dec = dl(x_d, mem, tgt_mask=cm)
diff_dec = R((o_hand_dec - o_ref_dec).abs().max())

n_el_all = sum(p.numel() for p in el.parameters())
n_dl_all = sum(p.numel() for p in dl.parameters())

print("手写 Decoder 层 vs nn.TransformerDecoderLayer 最大差: %.3e" % diff_dec)
print("Decoder 层子模块:", [n for n, _ in dl.named_children()])
print("Encoder 层参数 %d | Decoder 层参数 %d | 差值 %d" % (n_el_all, n_dl_all, n_dl_all - n_el_all))

# @@todo 手算 Decoder 与 Encoder 的参数差：多一个 self_attn 的自注意力 + 多一个 norm3
# @@hint 4E^2+4E + 2E
delta_theory = 4 * D_E * D_E + 4 * D_E + 2 * D_E
# @@end

print("理论差值 = Cross-Attention %d + norm3 %d = %d"
      % (4 * D_E * D_E + 4 * D_E, 2 * D_E, delta_theory))

# ---- 验收 ----
assert diff_dec < 1e-6, f"手写 Decoder 层不一致：{diff_dec}"
assert n_dl_all - n_el_all == delta_theory
print("\\n三种注意力：Encoder 只有 Self；Decoder 有 Masked-Self + Cross；")
print("Cross-Attention 的 Q 来自解码器、K/V 来自编码器 —— 这就是「读原文」的动作。")
'''

# ---- T11：参数量统计 ----
T11_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(77)
    tr = nn.Transformer(d_model=D_E, nhead=H_E, num_encoder_layers=2,
                        num_decoder_layers=2, dim_feedforward=D_FF_E,
                        dropout=0.0, batch_first=True)

# @@todo 统计整个 Transformer 的参数量，以及 encoder / decoder 各自
# @@hint sum(p.numel() for p in tr.parameters())；tr.encoder / tr.decoder
n_all = sum(p.numel() for p in tr.parameters())
n_enc = sum(p.numel() for p in tr.encoder.parameters())
n_dec = sum(p.numel() for p in tr.decoder.parameters())
# @@end

# @@todo 手算单层 Encoder：self_attn 4E²+4E + ffn 2E·d_ff+E+d_ff + 两个 LN 4E
# @@hint 4*D_E*D_E + 4*D_E + 2*D_E*D_FF_E + D_E + D_FF_E + 4*D_E
hand_el = 4 * D_E * D_E + 4 * D_E + 2 * D_E * D_FF_E + D_E + D_FF_E + 4 * D_E
# @@end

print("nn.Transformer(E=%d,h=%d,enc=2,dec=2,ff=%d) 总参数: %d" % (D_E, H_E, D_FF_E, n_all))
print("  encoder %d | decoder %d | er 比 %.4f" % (n_enc, n_dec, n_dec / n_enc))
print("  单层 Encoder 实测 %d | 手算 %d | 一致: %s" % (n_el_all, hand_el, n_el_all == hand_el))
print("  2 层 encoder %d = 2*%d + 尾部 norm %d" % (n_enc, n_el_all, 2 * D_E))
print("  2 层 decoder %d = 2*%d + 尾部 norm %d" % (n_dec, n_dl_all, 2 * D_E))

# ---- 验收 ----
assert n_all == n_enc + n_dec
assert n_el_all == hand_el, f"手算 {hand_el} 与实测 {n_el_all} 不符"
print("\\n参数分布：注意力 4E²、FFN 2E·d_ff（=8E²，是注意力的 2 倍）——")
print("所以 FFN 才是 Transformer 的参数大头（d_ff = 4E 时占总参数约 2/3）。")
'''

# ---- T12：位置编码必要性 ----
CLF_CODE = '''class EncClf(nn.Module):
    """Encoder + 平均池化 + 线性头；use_pe 控制要不要位置编码"""

    def __init__(self, V, d=D_S, h=H_S, use_pe=True, n_layer=1):
        super().__init__()
        self.use_pe = use_pe
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.register_buffer("pe", positional_encoding(64, d))
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.fc = nn.Linear(d, 2)

    def forward(self, x):
        z = self.emb(x)

        # @@todo 只在 use_pe 为真时把位置编码加到词向量上（提示：用三元表达式，别写 if 语句）
        # @@hint z = z + self.pe[:z.shape[1]] if self.use_pe else z
        z = z + self.pe[:z.shape[1]] if self.use_pe else z
        # @@end

        # @@todo 过 Encoder 后做平均池化，再喂给线性头
        # @@hint self.fc(self.enc(z).mean(1))
        return self.fc(self.enc(z).mean(1))
        # @@end


print("EncClf 定义完成：单层 TransformerEncoder（Pre-LN）+ 平均池化")
'''


def _train_clf_code(name: str, use_pe: bool, seed: int = 303, epochs: int = 200) -> str:
    title = "有位置编码" if use_pe else "无位置编码"
    bound = 0.9 if use_pe else 0.62
    cmp_op = ">" if use_pe else "<"
    return f'''def train_clf_{name}(epochs={epochs}, seed={seed}):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = EncClf(len(VOCAB), use_pe={use_pe})
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    lf = nn.CrossEntropyLoss()
    xt, yt = torch.as_tensor(Xtr), torch.as_tensor(ytr)
    xe, ye = torch.as_tensor(Xte), torch.as_tensor(yte)
    t0 = time.perf_counter()
    for _ in range(epochs):
        m.train()
        opt.zero_grad()
        lf(m(xt), yt).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(xe).argmax(1) == ye).float().mean())
    return m, acc, time.perf_counter() - t0


model_{name}, acc_{name}, dt_{name} = train_clf_{name}()

# ---- 验收 ----
assert acc_{name} {cmp_op} {bound}, f"结果超出预期区间：{{acc_{name}}}"
print("{title}：测试准确率 %.6f  (%.2f s)" % (acc_{name}, dt_{name}))
'''


T12_CODE = _train_clf_code("pe", True)
T13_CODE = _train_clf_code("nope", False)

T14_CODE = '''Xsh = torch.as_tensor(Xte).clone()
for i in range(Xsh.shape[0]):
    # @@todo 用固定种子的随机置换打乱第 i 行的 token 顺序
    # @@hint torch.randperm(Xsh.shape[1], generator=torch.Generator().manual_seed(1000 + i))
    Xsh[i] = Xsh[i][torch.randperm(Xsh.shape[1], generator=torch.Generator().manual_seed(1000 + i))]
    # @@end

with torch.no_grad():
    # @@todo 用「有位置编码」的模型算打乱后的准确率
    # @@hint (model_pe(Xsh).argmax(1) == torch.as_tensor(yte)).float().mean()
    acc_shuf = float((model_pe(Xsh).argmax(1) == torch.as_tensor(yte)).float().mean())
    # @@end

print("有 PE，原输入      : %.6f" % acc_pe)
print("有 PE，打乱输入    : %.6f" % acc_shuf)
print("无 PE，原输入      : %.6f" % acc_nope)

# ---- 验收 ----
assert acc_shuf < 0.62, f"打乱顺序后应当跌回随机，实际 {acc_shuf}"
print("\\n三行读出来的结论：")
print("  1) 有 PE 能学到顺序（%.3f）—— 位置信息确实被用上了" % acc_pe)
print("  2) 无 PE 学不到（%.3f）—— 架构上就没这个能力" % acc_nope)
print("  3) 有 PE 但输入打乱（%.3f）—— 它学的是顺序，不是词袋" % acc_shuf)
'''

# ---- 绘图 ----
PLOT_CODE = '''fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

im = axes[0].imshow(pe.numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xlabel("维度（左低维、右高维）")
axes[0].set_ylabel("位置 pos")
axes[0].set_title("sin/cos 位置编码 [%d, %d]" % (T_PE, D_PE), fontsize=10)
fig.colorbar(im, ax=axes[0], fraction=0.046)

ks = list(range(1, T_PE))
sims = [cos_sim(pe[0], pe[k]) for k in ks]
axes[1].plot(ks, sims, "-o", markersize=3, linewidth=1.4)
axes[1].set_xlabel("与位置 0 的距离 |Δ|")
axes[1].set_ylabel("余弦相似度")
axes[1].set_title("位置相似度随距离衰减（有周期性回环）", fontsize=10)
axes[1].grid(alpha=0.3)

plt.tight_layout()
print("左图：低维（左）条纹密集 = 周期短；高维（右）条纹稀疏 = 周期长。")
print("右图：总体下降但会回环 —— sin/cos 编码不是单调的位置距离函数。")
'''

# ---- 小结 ----
SUMMARY = '''import pandas as pd

rows = [
    ("位置编码 sin/cos", "每行范数 %.6f" % pe_norm, "值域 [-1,1]，pos=0 为 (0,1,0,1)"),
    ("位置相似度 |Δ|=1 / 16", "%.4f / %.4f" % (sim_delta[1], sim_delta[16]), "总体下降但有周期回环"),
    ("手写 LayerNorm", "差 %.1e" % diff_ln, "关键在 unbiased=False"),
    ("用样本方差", "差 %.4f" % diff_wrong, "d=16 时偏差 %.2f%%" % (diff_wrong * 100)),
    ("FFN 对账", "差 %.1e" % diff_ffn, "参数量 %d = 2d·d_ff+d+d_ff" % n_ffn),
    ("Post-LN 首层/末层梯度", "%.6f" % results_ln["Post-LN"]["ratio"],
     "末层是首层的 %.0f 倍" % (1 / results_ln["Post-LN"]["ratio"])),
    ("Pre-LN 首层/末层梯度", "%.6f" % results_ln["Pre-LN"]["ratio"], "梯度量级处处一致"),
    ("无残差 首层/末层梯度", "%.6f" % results_ln["无残差"]["ratio"], "残差是梯度的通路"),
    ("手写 Encoder 层", "差 %.1e" % diff_enc, "参数量 %d" % n_el_all),
    ("手写 Decoder 层", "差 %.1e" % diff_dec, "多一个 Cross-Attention + norm3"),
    ("参数分布", "注意力 %d / FFN %d" % (n_attn, n_ffn_el), "FFN 是参数大头"),
    ("有 PE / 无 PE", "%.3f / %.3f" % (acc_pe, acc_nope), "位置编码决定能不能学顺序"),
    ("有 PE + 打乱输入", "%.3f" % acc_shuf, "它学的是顺序，不是词袋"),
]
print(pd.DataFrame(rows, columns=["实验", "实测", "结论"]).to_string(index=False))
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch05 Transformer 完整架构手写（讲解版）

> 对应考点：**「大语言模型相关理论考查」** 的骨架部分。
> ch04 把注意力写对了；本章把它**装成一台能训练的机器**。

Transformer 的 Encoder 层就是这四行：

```python
x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FFN(x))
```

看着简单，但每一处都有人踩过坑：

| 零件 | 少了会怎样 | 本章实测 |
|---|---|---|
| **位置编码** | 模型分不出「渗油在前」还是「渗油在后」 | 无 PE 准确率 **0.528**（≈随机） |
| **LayerNorm 的 `unbiased`** | 静默偏几 %（`d` 越小时越明显） | 差 **0.082911** |
| **残差连接** | 深堆叠时浅层梯度消失 | 首/末层梯度比 **0.000269** |
| **Pre-LN vs Post-LN** | 6 层就开始训不动 | **0.000007** vs **1.613922** |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(PLOT_SETUP),
    md(
        """
## 5.1 位置编码：给「置换等变」的注意力补上位置感

ch04 的结论是：Attention 对置换**等变**，没有位置概念。修法是**把位置信息加到词向量上**：

$$\\text{input} = \\text{Embedding}(x) + \\text{PE}(pos)$$

sin/cos 编码的公式：

$$PE_{(pos,2i)} = \\sin\\!\\left(\\frac{pos}{10000^{2i/d}}\\right),\\qquad
PE_{(pos,2i+1)} = \\cos\\!\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

**设计要点**：不同维度对有不同的角频率 —— 低维变化快（分辨近邻），高维变化慢（分辨全局）。
这等价于在多个尺度上编码位置。
"""
    ),
    code(T1_CODE),
    md(
        """
位置编码到底能不能表达「距离」？把位置 0 与其它位置的余弦相似度画出来看看。

> 注意：**不要期待它单调**。sin/cos 是周期函数，距离远了会绕回来。
> 这个"回环"是真实性质，不是 bug —— 也是为什么后来的模型更爱用可学习的位置嵌入或 RoPE。
"""
    ),
    code(T2_CODE),
    code(PLOT_CODE),
    md(
        """
两个细节：

1. **低维周期短**（`2π/1 ≈ 6.28` rad），所以低维负责分辨相邻位置；
2. **相加前要考虑量级**：词向量来自 `N(0,1)` 初始化，范数 ≈ $\\sqrt{d}$，
   位置编码的范数是 $\\sqrt{d/2}$ —— 两者本来就同量级，所以直接相加是合理的。
"""
    ),
    code(T3_CODE),
    md(
        """
## 5.2 手写 LayerNorm

LayerNorm 在**特征维**（最后一维）上归一化：

$$\\operatorname{LN}(x) = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\cdot \\gamma + \\beta$$

坑只有一个，但很隐蔽：**方差要用总体方差（`unbiased=False`）**，
而 PyTorch 的 `Tensor.var()` **默认是样本方差**（`unbiased=True`，分母 $n-1$）。
写错不报错，只是静默偏 $\\sqrt{d/(d-1)}$。
"""
    ),
    code(T4_CODE),
    md(
        """
下面把这个坑的两面都摊开：一次是**归一化时**用错方差，一次是**检查时**用错口径。
"""
    ),
    code(T5_CODE),
    md(
        """
**为什么需要 LayerNorm**（而不是 BatchNorm）：

- 序列长度可变、batch 里长短不一 → BatchNorm 的批统计不稳定
- 训练时 batch 小、推理时 batch=1 → 批统计不可用
- LayerNorm 只依赖**单个样本自己**，train / eval 行为一致，也不需要维护 running stats

这就是我们刚才看到「逐 token 均值全为 0」的含义 —— 它归一化的是每个 token 自己的那一行。
"""
    ),
    md(
        """
## 5.3 FFN：位置独立的两层小 MLP

$$\\operatorname{FFN}(x) = W_2\\,\\sigma(W_1 x + b_1) + b_2$$

其中 $d_{ff} = 4d$。FFN **逐位置独立**——它不做任何跨位置交互，
序列建模的活全在注意力和位置编码里。

参数上它却是大头：注意力是 $4d^2$，FFN 是 $2d \\cdot 4d = 8d^2$ —— **两倍**。
"""
    ),
    code(T6_CODE),
    md(
        """
## 5.4 残差 + LayerNorm：Pre-LN 还是 Post-LN

原始论文用的是 **Post-LN**：

```python
x = LayerNorm(x + SubLayer(x))     # 归一化在残差之后
```

但后来的大模型几乎都用 **Pre-LN**：

```python
x = x + SubLayer(LayerNorm(x))     # 归一化在子层之前
```

**为什么**？下面的实验用逐层梯度范数回答 —— 注意看「首层/末层」这个比值。
"""
    ),
    code(BLOCK_CODE),
    code(T8_CODE),
    md(
        """
这张表是本章的核心。三个读法：

1. **Post-LN 的梯度在浅层几乎消失**：首/末层 = `0.000007`，
   也就是说末层梯度是首层的 **14 万倍**。层数再多几层，第 1 层就完全收不到信号。
2. **Pre-LN 的梯度量级处处一致**：比值 `1.613922`，浅层甚至比末层还大。
   这就是大模型清一色 Pre-LN 的原因 —— 它是**能训深的必要条件**。
3. **去掉残差**比值掉到 `0.000269` —— 残差是梯度的「高速公路」，
   这正是 ch03 里那个「加性捷径」结论在真实架构上的体现。

> 诚实说明：不同配置的 `loss` 尺度不同，**梯度的绝对值不可比**；
> 但「首层/末层」的比值是尺度无关的，可以横向比。
"""
    ),
    md(
        """
## 5.5 Encoder 层：四行代码

把 Attention、FFN、残差、LayerNorm 拼起来，就是完整的 Encoder 层。
下面与 `nn.TransformerEncoderLayer` 对账 —— **差 0.000e+00**，逐元素相同。
"""
    ),
    code(T9_CODE),
    md(
        """
## 5.6 Decoder 层：多一个 Cross-Attention

Decoder 层有三处注意力，比 Encoder 多一处：

| 子层 | Q 来自 | K/V 来自 | 作用 |
|---|---|---|---|
| Masked Self-Attention | 解码器 | 解码器 | 已生成的部分互相看（**带 causal mask**） |
| **Cross-Attention** | 解码器 | **编码器** | 「读原文」 |
| FFN | — | — | 逐位置变换 |

所以 Decoder 层的参数量比 Encoder 层多「一个注意力 + 一个 LayerNorm」。
"""
    ),
    code(T10_CODE),
    md(
        """
## 5.7 参数量统计

把每一块的参数量拆开算，是对「Transformer 到底大在哪里」最直接的回答：

| 模块 | 参数量 |
|---|---|
| Self-Attention | $4d^2 + 4d$（Q/K/V/O 四个 $d\\times d$ 矩阵 + 偏置） |
| FFN | $2d \\cdot d_{ff} + d + d_{ff}$，$d_{ff}=4d$ 时约 $8d^2$ |
| 2 个 LayerNorm | $4d$ |
| **单层合计** | $12d^2 + \\text{低阶项}$ |
"""
    ),
    code(T11_CODE),
    md(
        """
**结论**：$d_{ff} = 4d$ 时，FFN 占了单层参数的约 **2/3**。
所以「模型参数量」这个话题里，注意力常被过度关注，真正的参数大头是 FFN。

顺带记住：拆多头**不改变参数量**（ch04 已验证），堆层数是**线性**增长。
"""
    ),
    md(
        """
## 5.8 位置编码到底有没有用：实测

前面看了结构，现在看**能力**。复用 ch03 那个「谁先出现」任务：

> 序列长 8，A 类词（渗漏油）和 B 类词（发热）各出现恰好 1 次，
> 标签 = 「A 是否出现在 B 之前」。词袋和「无位置编码的 Transformer」都看不到顺序。

三组对照：

| 配置 | 预期 |
|---|---|
| 有位置编码 | 能学会 |
| 无位置编码 | ≈ 随机（架构上不可能） |
| 有位置编码 + 输入打乱 | 应该崩（它学的是顺序） |
"""
    ),
    code(TASK_DATA),
    code(CLF_CODE),
    code(T12_CODE),
    code(T13_CODE),
    code(T14_CODE),
    md(
        """
## 5.9 汇总

"""
    ),
    code(SUMMARY),
    md(
        """
## 5.10 从 ch03 到 ch05 的完整链路

| 章节 | 解决的问题 | 关键实测 |
|---|---|---|
| ch03 | RNN 为什么不够 | 每步梯度 ×0.746、耗时线性、置换敏感度难维持 |
| ch04 | 注意力的六个零件 | 全部与框架对账（≤1.19e-07） |
| ch05 | 怎么装成能训的机器 | Pre-LN 首/末层梯度 **1.61** vs Post-LN **0.000007** |

**下一章 ch06** 处理训练与推理的细节：teacher forcing、label smoothing、
warmup 学习率、贪心解码、beam search。

## 易错点清单

1. **`Tensor.var()` 默认 `unbiased=True`** —— 必须显式写 `unbiased=False` 才是总体方差。
   实测差 **0.082911**（$d=16$ 时）。
2. **检查时用 `std(-1)` 看到 1.03 就以为写错了** —— 它默认也是样本口径，
   正确值是 $\\sqrt{d/(d-1)}$。
3. **以为位置相似度随距离单调下降** —— sin/cos 有周期，远了会回环（|Δ|=19 回升到 0.804）。
4. **`nn.MultiheadAttention` 传 `attn_mask` 的形状** —— causal mask 要 `[T, T]`，不是 `[T]`。
5. **用同一个 `loss` 绝对值比不同架构的梯度** —— 尺度不可比；要比**尺度无关的比值**。
6. **Post-LN 直接堆 24 层** —— 第 1 层收不到梯度；大模型都用 Pre-LN。
7. **忘记 `register_buffer` 存位置编码** —— 写成普通属性就不会跟着 `.to(device)` 走。
8. **以为拆多头会增加参数** —— $E$ 被切成 $H$ 份，每份 $E/H$，总参数量不变。
9. **FFN 用 `d_ff = d`** —— 表达能力不足；标准是 `4d`。
10. **把 Decoder 的 Cross-Attention 当成 Self-Attention** —— 它的 K/V 来自**编码器**，
    不是解码器自己。

## 本章小结

- 位置编码：sin/cos，每行范数 $\\sqrt{d/2}$，近邻相似度最高但**会周期回环**
- 手写 LayerNorm 与 `nn.LayerNorm` 差 **2.38e-07**；坑在 `unbiased`
- **Pre-LN vs Post-LN 是本轮最有价值的实测**：首/末层梯度比 **1.613922** vs **0.000007**
- Encoder 层对账差 **0.000e+00**，Decoder 层差 **3.58e-07**
- 单层 Encoder 参数量 $12d^2$ 量级，**FFN 占约 2/3**
- **位置编码决定能力**：有 PE **0.976** / 无 PE **0.528** / 有 PE 打乱 **0.486**
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch05 Transformer 完整架构手写（练习版）

> 补全 `____`，跑通所有 assert。真值全部来自实跑。
>
> 本章有一个**必看**的实验：Pre-LN vs Post-LN 的逐层梯度分布（T8）。

| 块 | 内容 | 难度 |
|---|---|---|
| T1 | sin/cos 位置编码 | ★★ |
| T2 | 位置相似度与角频率 | ★ |
| T3 | 词向量与 PE 的量级 | ★ |
| T4 | 手写 LayerNorm | ★★★ |
| T5 | `unbiased` 陷阱 | ★★ |
| T6 | FFN | ★ |
| T7 | Pre-LN / Post-LN 的 Block | ★★★ |
| T8 | 逐层梯度对比 | ★★★ |
| T9 | Encoder 层对账 | ★★ |
| T10 | Decoder 层对账 | ★★★ |
| T11 | 参数量统计 | ★★ |
| T12 | 位置编码必要性 | ★ |
| T13 | 打乱输入对照 | ★★ |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(PLOT_SETUP),
    md("## T1 sin/cos 位置编码"),
    code(T1_CODE),
    md("## T2 位置相似度与角频率"),
    code(T2_CODE),
    code(PLOT_CODE),
    md("## T3 词向量与 PE 的量级"),
    code(T3_CODE),
    md("## T4 手写 LayerNorm"),
    code(T4_CODE),
    md("## T5 `unbiased` 陷阱"),
    code(T5_CODE),
    md("## T6 FFN"),
    code(T6_CODE),
    md("## T7 Pre-LN / Post-LN 的 Block"),
    code(BLOCK_CODE),
    md("## T8 逐层梯度对比"),
    code(T8_CODE),
    md("## T9 Encoder 层对账"),
    code(T9_CODE),
    md("## T10 Decoder 层对账"),
    code(T10_CODE),
    md("## T11 参数量统计"),
    code(T11_CODE),
    md("## T12 位置编码必要性（数据构造已给，不挖空）"),
    code(TASK_DATA),
    code(CLF_CODE),
    code(T12_CODE),
    md("## T13 无位置编码的对照"),
    code(T13_CODE),
    md("## T14 打乱输入对照"),
    code(T14_CODE),
    md("## 汇总"),
    code(SUMMARY),
    md(
        """
## 自查清单

- [ ] 位置编码形状 `(20, 16)`，`PE[0,:4] = [0, 1, 0, 1]`，每行范数 `2.828427`
- [ ] `|Δ|=1` 的相似度最高（0.935646），|Δ|=16 降到 0.526747
- [ ] 高维角频率比低维**小**（周期更长）
- [ ] 手写 LayerNorm 与 `nn.LayerNorm` 差 **< 1e-6**（实际 2.38e-07）
- [ ] 用 `unbiased=True` 时差 **> 0.01**（实际 0.082911）
- [ ] `out_ln.std(-1)` 得到 **1.032795** = $\\sqrt{16/15}$
- [ ] FFN 参数量 **2128**，与手算一致
- [ ] **Post-LN 首层/末层梯度比 < 0.01**（实际 0.000007）
- [ ] **Pre-LN 首层/末层梯度比 > 1**（实际 1.613922）
- [ ] 手写 Encoder 层差 **< 1e-6**（实际 0.000e+00）
- [ ] 手写 Decoder 层差 **< 1e-6**（实际 3.58e-07）
- [ ] 单层 Encoder 参数量 **2224** 与手算一致
- [ ] **有 PE > 0.9，无 PE < 0.62**，有 PE 打乱后 < 0.62

**答不上来时问自己**：
1. LayerNorm 为什么必须用总体方差？
2. Pre-LN 的梯度为什么能一路传到底？
3. 为什么位置相似度不是单调的？
4. Decoder 的 Cross-Attention 和 Self-Attention 差在哪？
5. 「无位置编码的 Transformer」为什么在顺序任务上必然≈随机？
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
