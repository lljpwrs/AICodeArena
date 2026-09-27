#!/usr/bin/env python3
"""ch06 训练与推理细节手写（讲解 / 练习 / 答案 三件套）。

本章主线 —— **把 ch05 的机器训起来，再把它解码出来**：

    ① 两种 teacher forcing 输入（shift_right）
    ② 手写 label smoothing + 与 nn.CrossEntropyLoss 对账（差 2.384e-07 / 0.000e+00）
    ③ 手写 warmup + 余弦退火 + 与 LambdaLR 对账（差 0.000e+00）
    ④ 训练确定性：fork_rng 只「还原」不「固定」（本章最有价值的坑）
       —— 实测：mode=all 同一次进程内两次一致（1.9810），重跑进程变成 1.9989
    ⑤ 曝光偏差：TF vs free-running 逐位置对照 + oracle 前缀的因果验证
       末位 TF 0.818182 vs free 0.181818（差 0.636364）
    ⑥ label smoothing 的校准副作用：预测熵 0.1604 -> 0.6504
    ⑦ 手写 beam search（含长度惩罚 α）：贪心 0.7600 -> beam3 0.7750
    ⑧ 长度外推 L=4/6/8/10/12/14：L=8 最好（0.82），L≥12 断崖（0.1467 / 0.0333）

任务：**去重保序** —— 输入 10 个数字符号（可重复），按首次出现顺序去重后输出 + `<eos>`。
输出长度可变（4~11），所以「长度惩罚」「自然结束」这些解码细节才有意义。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp06_train_decode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch06_train_decode"

# =========================================================================== #
# 公共导入 / 脚手架                                                           #
# =========================================================================== #

IMPORTS = '''import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")          # 屏蔽 torch 内部 warning，保持输出干净
torch.set_num_threads(4)
torch.set_printoptions(precision=6, sci_mode=False)
print("torch", torch.__version__, "| 线程", torch.get_num_threads())
'''

SCAFFOLD = '''# 脚手架：常量、任务数据与模型（两版都有，你只填 @@todo 块）
PAD, BOS, EOS = 0, 1, 2
ITOS = ["<pad>", "<bos>", "<eos>"] + list("0123456789")
NTOK = len(ITOS)
MAXLEN = 24

D, H, NL = 32, 2, 1          # 模型：1 层 / d=32 / 2 头 —— CPU 秒级
L_TASK = 10                  # 训练输入长度
STEPS, PEAK_LR, WARMUP = 700, 3e-3, 250


def make_dedup(n, L=L_TASK, seed=0):
    """去重保序任务的构造：输入的 10 个符号可重复，输出是「按首次出现顺序去重」+ <eos>"""
    rng = np.random.default_rng(seed)
    src = rng.integers(3, NTOK, size=(n, L)).astype(np.int64)
    outs = []
    for i in range(n):
        seen, o = set(), []
        for v in src[i].tolist():
            if v not in seen:
                seen.add(v)
                o.append(v)
        outs.append(o + [EOS])
    M = max(len(t) for t in outs)
    tgt = np.zeros((n, M), np.int64)
    for i, t in enumerate(outs):
        tgt[i, :len(t)] = t
    return src, tgt


Xtr, Ytr = make_dedup(4000, seed=1)
Xva, Yva = make_dedup(1000, seed=2)
Xte, Yte = make_dedup(1000, seed=3)
Xb, Yb = make_dedup(200, seed=4)
_lens = (Ytr != PAD).sum(1)
print("训练 %s | 验证 %s | 测试 %s" % (Xtr.shape, Xva.shape, Xte.shape))
print("真值输出长度分布:", dict(zip(*[a.tolist() for a in np.unique(_lens, return_counts=True)])))
print("平均输出长度 %.6f" % _lens.mean())
'''

NET_CODE = '''class DedupNet(nn.Module):
    """seq2seq 小模型：源端 / 目标端各一套嵌入 + 位置编码，共用 nn.Transformer。

    注意 `dropout=0.0` 是**显式写成 0** 的 —— nn.Transformer 默认 dropout=0.1，
    而它会在训练循环里消耗全局随机数。为什么这很要命，T6 用实验回答。
    """

    def __init__(self, d=D, h=H, nl=NL, ntok=NTOK, maxlen=MAXLEN, dropout=0.0):
        super().__init__()
        self.d = d
        self.es = nn.Embedding(ntok, d, padding_idx=PAD)
        self.et = nn.Embedding(ntok, d, padding_idx=PAD)
        self.ps = nn.Embedding(maxlen, d)
        self.pt = nn.Embedding(maxlen, d)
        self.tr = nn.Transformer(d, h, nl, nl, dim_feedforward=4 * d,
                                 dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d, ntok)

    def forward(self, src, tgt_in):
        xs = self.es(src) * math.sqrt(self.d) + self.ps(torch.arange(src.shape[1]))
        xt = self.et(tgt_in) * math.sqrt(self.d) + self.pt(torch.arange(tgt_in.shape[1]))
        tm = nn.Transformer.generate_square_subsequent_mask(tgt_in.shape[1])
        hh = self.tr(xs, xt, src_key_padding_mask=(src == PAD), tgt_mask=tm,
                     tgt_key_padding_mask=(tgt_in == PAD))
        return self.fc(hh)


with torch.random.fork_rng():
    torch.manual_seed(0)
    _probe = DedupNet()
print("模型总参数 %d" % sum(p.numel() for p in _probe.parameters()))
del _probe
'''

# --------------------------------------------------------------------------- #
# T1 两种 teacher forcing 输入                                                 #
# --------------------------------------------------------------------------- #

T1_CODE = '''def shift_right(y):
    """teacher forcing 的关键一步：decoder 输入 = 真值整体右移一位，首位补 <bos>"""
    # @@todo 一行拼出来：一列 <bos> 接上「真值去掉最后一列」
    # @@hint torch.full((y.shape[0], 1), BOS, dtype=torch.long)
    # @@hint 然后 torch.cat(..., dim=1) 接 y[:, :-1]，整体形状与 y 相同
    t = torch.cat([torch.full((y.shape[0], 1), BOS, dtype=torch.long), y[:, :-1]], dim=1)
    # @@end
    return t


# ---- 验收 ----
_y = torch.as_tensor(Ytr[:2])
_sr = shift_right(_y)
assert _sr.shape == _y.shape, "右移后长度不变"
assert _sr[0, 0].item() == BOS and _sr[1, 0].item() == BOS, "首位必须是 <bos>"
assert torch.equal(_sr[:, 1:], _y[:, :-1]), "第 1 位起就是真值的前 n-1 位"
print("真值      :", _y[0].tolist())
print("右移后    :", _sr[0].tolist())
print()
print("为什么必须右移：位置 t 的输出只能看到「真值的前 t 位」。")
print("如果直接把 y 喂进去，模型在第 t 位就看到了答案本身，等于抄答案。")
print("这一步也叫「teacher forcing」：训练时永远喂真值，而不是模型自己上一步的输出。")
'''

# --------------------------------------------------------------------------- #
# T2 手写 label smoothing                                                      #
# --------------------------------------------------------------------------- #

T2_CODE = '''def ce_loss(logits, y, smoothing=0.0):
    """手写交叉熵：smoothing=0 退化成普通 NLL；smoothing>0 用平滑目标分布。

    目标分布 d：所有类都是 eps/V，正确类再加 (1-eps)，即 d[y] = 1-eps+eps/V。
    """
    V = logits.shape[-1]
    logp = F.log_softmax(logits, -1)
    if smoothing == 0.0:
        nll = -logp.gather(-1, y[:, :, None]).squeeze(-1)
    else:
        with torch.no_grad():
            # @@todo 构造平滑目标分布 d 并把正确类顶上去
            # @@hint d = torch.full_like(logp, smoothing / V) 先把所有类填 eps/V
            # @@hint 再用 d.scatter_(-1, y[:, :, None], 1 - smoothing + smoothing / V) 抬高正确类
            d = torch.full_like(logp, smoothing / V)
            d.scatter_(-1, y[:, :, None], 1.0 - smoothing + smoothing / V)
            # @@end
        nll = -(d * logp).sum(-1)
    m = (y != PAD).float()
    return (nll * m).sum() / m.sum().clamp(min=1.0)
'''

T3_CODE = '''lg = torch.randn(2, 5, NTOK, generator=torch.Generator().manual_seed(3))
yt = torch.randint(3, NTOK, (2, 5), generator=torch.Generator().manual_seed(4))

rows = []
for eps in (0.0, 0.1, 0.3):
    # @@todo 手写值与框架值对账；注意 nn.CrossEntropyLoss 要 (N, C, L)，所以 logits 要转置
    # @@hint float(nn.CrossEntropyLoss(label_smoothing=eps)(lg.transpose(1, 2), yt))
    a = float(ce_loss(lg, yt, eps))
    b = float(nn.CrossEntropyLoss(label_smoothing=eps)(lg.transpose(1, 2), yt))
    rows.append((eps, a, b, abs(a - b)))
    # @@end

# ---- 验收 ----
assert all(r[3] < 1e-6 for r in rows), "手写 label smoothing 应与框架一致"
assert abs(rows[0][3] - 2.384e-07) < 1e-9, f"ε=0 时差 {rows[0][3]:.3e}"
assert abs(rows[1][3]) < 1e-12, f"ε=0.1 时应逐位相等，实际差 {rows[1][3]:.3e}"
assert abs(1 - 0.1 + 0.1 / NTOK - 0.907692) < 1e-6, "平滑后正确类目标值 1-eps+eps/V"
assert abs(0.1 / NTOK - 0.007692) < 1e-6, "其余每类 eps/V"
assert abs(math.log(NTOK) - 2.564949) < 1e-6, "均匀分布熵 lnV"
for eps, a, b, dd in rows:
    print("  ε=%-4s 手写 %.10f | 框架 %.10f | 差 %.3e" % (eps, a, b, dd))
print("平滑后：正确类 %.6f | 其余每类 %.6f | 均匀熵 lnV = %.6f"
      % (1 - 0.1 + 0.1 / NTOK, 0.1 / NTOK, math.log(NTOK)))
print()
print("为什么平滑会让 loss 变大：目标不再是 one-hot，即便模型完全正确，")
print("它也必须给其他类留 eps/V 的概率，损失下界被抬到 eps*(H(u)+lnV) 以上。")
'''

# --------------------------------------------------------------------------- #
# T4 手写 warmup + 余弦退火                                                    #
# --------------------------------------------------------------------------- #

T4_CODE = '''def lr_warmup_cos(step, warmup=WARMUP, total=STEPS):
    """倍率：前 warmup 步从 0 线性升到 1，之后余弦从 1 降到 0。

    块内不允许写 if 语句（本仓库挖空纪律），所以用 min 把两段拼起来：
      warm 段在升到 1 之后恒等于 1，cool 段在 warmup 之前恒等于 1 -> 取 min 自动衔接。
    """
    # @@todo 两个中间量 + min，拼出两段式倍率
    # @@hint warm = min((step + 1) / warmup, 1.0)
    # @@hint cool = 0.5 * (1.0 + math.cos(math.pi * max(step - warmup, 0) / max(total - warmup, 1)))
    # @@hint 返回 min(warm, cool)
    warm = min((step + 1) / warmup, 1.0)
    cool = 0.5 * (1.0 + math.cos(math.pi * max(step - warmup, 0) / max(total - warmup, 1)))
    return min(warm, cool)
    # @@end


# ---- 验收 ----
assert abs(lr_warmup_cos(0, 250, 700) - 0.004) < 1e-12, "第 0 步是 1/250"
assert abs(lr_warmup_cos(249, 250, 700) - 1.0) < 1e-12, "warmup 末步到 1"
assert abs(lr_warmup_cos(250, 250, 700) - 1.0) < 1e-12, "衔接处不跳变"
assert abs(lr_warmup_cos(400, 250, 700) - 0.75) < 1e-12, "退火 1/3 路程 -> cos(pi/3)=0.5 -> 0.75"
assert lr_warmup_cos(699, 250, 700) < 1e-4, "末步趋近 0"
print("倍率 step=0/125/249/250/400/600/699 ->",
      [round(lr_warmup_cos(s), 6) for s in (0, 125, 249, 250, 400, 600, 699)])


def lr_inv_sqrt(step, d_model=D, warmup=400):
    """原版《Attention is all you need》的调度：d^-0.5 * min(s^-0.5, s * warmup^-1.5)"""
    s = max(step, 1)
    return (d_model ** -0.5) * min(s ** -0.5, s * (warmup ** -1.5))


print("原版 inv-sqrt  step=1/50/400/1000/2499 -> %s"
      % ["%.6e" % lr_inv_sqrt(s) for s in (1, 50, 400, 1000, 2499)])
print()
print("两者结构一样（先升后降），差别在「降段形状」：")
print("  inv-sqrt 降段是 step^-0.5，长训时几乎不衰减；余弦退火一定降到 0。")
'''

# --------------------------------------------------------------------------- #
# T5 与 torch 调度器对账                                                       #
# --------------------------------------------------------------------------- #

T5_CODE = '''opt_lr = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)

# @@todo 用 LambdaLR 包住同一个调度函数
# @@hint lr_lambda 直接传 lr_warmup_cos（签名就是 step -> float）
sch = torch.optim.lr_scheduler.LambdaLR(opt_lr, lr_lambda=lr_warmup_cos)
# @@end

mine, ref = [], []
for s in range(STEPS):
    mine.append(lr_warmup_cos(s))
    # @@todo 读调度器当前的倍率（必须在 sch.step() 之前读）
    # @@hint sch.get_last_lr()[0]
    ref.append(sch.get_last_lr()[0])
    # @@end
    opt_lr.step()
    sch.step()

# ---- 验收 ----
d_lr = max(abs(a - b) for a, b in zip(mine, ref))
assert d_lr < 1e-12, f"手写调度器与 LambdaLR 应完全一致，实际最大差 {d_lr}"
assert max(mine) == 1.0 and mine.index(1.0) == 249, "峰值出现在 warmup 最后一步"
print("与 LambdaLR 最大绝对差 %.3e" % d_lr)
print("峰值倍率 %.6f（warmup 末 step=%d 达到）" % (max(mine), mine.index(max(mine))))
'''

# --------------------------------------------------------------------------- #
# T6 训练确定性：fork_rng 的还原语义                                           #
# --------------------------------------------------------------------------- #

DETERM_FUNC = '''def quick(dropout=0.0, mode="init", seed=7, steps=150, bs=64, lr=1e-3):
    """跑 150 步小训练，返回「末 20 步平均 loss」。

    mode="init" —— 只有模型初始化包在 fork_rng 里（最常见、也是坑最多的写法）
    mode="all"  —— 训练循环也包进 fork_rng
    mode="full" —— 先 torch.manual_seed(seed) 把环境固定住，再用 fork_rng 包住全部
    """

    def body():
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            m = DedupNet(dropout=dropout)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        rg = np.random.default_rng(5)
        xt, yt = torch.as_tensor(Xtr), torch.as_tensor(Ytr)
        m.train()
        h = []
        for _ in range(steps):
            idx = rg.integers(0, len(xt), size=bs)
            yy = yt[idx]
            loss = ce_loss(m(xt[idx], shift_right(yy)), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            h.append(loss.item())
        return float(np.mean(h[-20:]))

    if mode == "all":
        with torch.random.fork_rng():
            return body()
    if mode == "full":
        torch.manual_seed(seed)
        with torch.random.fork_rng():
            return body()
    return body()
'''

T6_CODE = '''# @@todo 四组各连续跑两次，看哪一组「两次结果不一样」
# @@hint 每组写成 (quick(...), quick(...))，两次调用参数完全相同
g1 = (quick(0.0, "init"), quick(0.0, "init"))
g2 = (quick(0.1, "init"), quick(0.1, "init"))
g3 = (quick(0.1, "all"), quick(0.1, "all"))
g4 = (quick(0.1, "full"), quick(0.1, "full"))
# @@end

# ---- 验收 ----
assert g1[0] == g1[1], "没有随机源 -> 两次必然一致"
assert abs(g2[0] - g2[1]) > 1e-6, "dropout>0 且循环裸跑 -> 两次必然不同"
assert g3[0] == g3[1], "把循环也包进 fork_rng -> 同一次进程内两次恢复成一致"
assert g4[0] == g4[1], "先 manual_seed 再包住全部 -> 一致"
assert abs(g1[0] - 1.8154501200) < 1e-9, \
    f"g1 不受随机源影响，是固定值（可跨进程复现），实际 {g1[0]:.10f}"
for _tag, _g in (("dropout=0.0 mode=init", g1), ("dropout=0.1 mode=init", g2),
                 ("dropout=0.1 mode=all ", g3), ("dropout=0.1 mode=full", g4)):
    print("  %s : %.10f / %.10f | 差 %.3e" % (_tag, _g[0], _g[1], abs(_g[0] - _g[1])))
print()
print("三个结论，一个比一个隐蔽（实测数据见下）：")
print("  ① 循环里只要有随机源（dropout / shuffle），就得把它一起管起来。")
print("  ② fork_rng 的作用是「退出时把随机数状态**还原**成进入前的样子」。")
print("     所以它能让同一次进程里的两次调用一致 —— 但这**不等于可复现**！")
print("  ③ 进入 fork_rng 之前的那个状态，如果没人 manual_seed 过，")
print("     它在每次启动进程时都是随机初始化的。于是 mode=all 的数字：")
print("        本次运行   1.9810412586 / 1.9810412586  （一致）")
print("        重跑一次   1.9988539517 / 1.9988539517  （仍然一致，但换了个值！）")
print("     只有 mode=full 才真正可复现。g1 因为压根没有随机源，也顺便可复现。")
'''

# --------------------------------------------------------------------------- #
# T7 train 主体                                                                #
# --------------------------------------------------------------------------- #

TRAIN_FUNC = '''def train(steps=STEPS, bs=128, peak_lr=PEAK_LR, warmup=WARMUP, smoothing=0.0,
          use_warmup=True, seed=11, log_first=0):
    """训练主循环。

    log_first=k 会额外记录前 k 步的梯度范数（T13 的 warmup 对照要用）。
    """
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = DedupNet()
    opt = torch.optim.Adam(model.parameters(), lr=peak_lr)
    hist = {"loss": [], "grad": []}
    rg = np.random.default_rng(1000 + seed)     # 数据顺序用独立的 numpy 生成器，可复现
    xt, yt = torch.as_tensor(Xtr), torch.as_tensor(Ytr)
    model.train()
    for s in range(steps):
        # @@todo 先写本步学习率再更新参数（写在 opt.step() 之后，第一步就会用到满 lr）
        # @@hint opt.param_groups[0]["lr"] = peak_lr * (lr_warmup_cos(s, warmup, steps) if use_warmup else 1.0)
        opt.param_groups[0]["lr"] = peak_lr * (
            lr_warmup_cos(s, warmup, steps) if use_warmup else 1.0)
        # @@end
        idx = rg.integers(0, len(xt), size=bs)
        y = yt[idx]
        # @@todo 前向 + 手写交叉熵；decoder 输入记得右移
        # @@hint ce_loss(model(xt[idx], shift_right(y)), y, smoothing)
        loss = ce_loss(model(xt[idx], shift_right(y)), y, smoothing)
        # @@end
        opt.zero_grad()
        loss.backward()
        if log_first and s < log_first:
            hist["grad"].append(math.sqrt(sum(float(p.grad.norm()) ** 2
                                               for p in model.parameters() if p.grad is not None)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist["loss"].append(loss.detach().item())
    model.eval()
    return model, hist
'''

# --------------------------------------------------------------------------- #
# T8 评估与解码工具                                                            #
# --------------------------------------------------------------------------- #

EVAL_FUNC = '''@torch.no_grad()
def tf_eval(model, X, Y):
    """teacher forcing 评估。

    返回 (逐位置准确率, token 准确率, 序列完全正确率, 各位置样本数)。
    """
    src, y = torch.as_tensor(X), torch.as_tensor(Y)
    pred = model(src, shift_right(y)).argmax(-1)
    m = y != PAD
    pos = [float((pred[:, k][m[:, k]] == y[:, k][m[:, k]]).float().mean())
           if int(m[:, k].sum()) else float("nan") for k in range(Y.shape[1])]
    seq = float((((pred == y) | ~m).all(1)).float().mean())
    return pos, float((pred[m] == y[m]).float().mean()), seq, m.sum(0).tolist()


def trim(s):
    """截到 <eos>（含）；模型没吐出 <eos> 就补一个"""
    s = [int(v) for v in s]
    return s[:s.index(EOS) + 1] if EOS in s else s + [EOS]


def golds(Y):
    """真值去掉 padding"""
    return [[int(v) for v in row if int(v) != PAD] for row in Y]
'''

GREEDY_FUNC = '''@torch.no_grad()
def greedy(model, X, maxlen=13):
    """贪心解码：每一步取当前概率最大的 token，不回溯"""
    src = torch.as_tensor(X)
    cur = torch.full((src.shape[0], 1), BOS, dtype=torch.long)
    for _ in range(maxlen):
        # @@todo 把「当前最后一位的 argmax」拼到 cur 后面
        # @@hint model(src, cur)[:, -1].argmax(-1, keepdim=True) 的形状是 (B, 1)
        cur = torch.cat([cur, model(src, cur)[:, -1].argmax(-1, keepdim=True)], 1)
        # @@end
    return cur[:, 1:]
'''

# --------------------------------------------------------------------------- #
# T9 主模型训练                                                                #
# --------------------------------------------------------------------------- #

T9_CODE = '''# @@todo 训练主模型（seed=11）
# @@hint model, hist = train(seed=11)
m_a, h_a = train(seed=11)
# @@end

# ---- 验收 ----
assert abs(np.mean(h_a["loss"][-200:]) - 0.113932) < 1e-6
pa, tka, sqa, cnta = tf_eval(m_a, Xva, Yva)
assert abs(tka - 0.952653) < 1e-6, f"TF token 准确 {tka}"
assert abs(sqa - 0.729000) < 1e-6, f"TF 序列正确 {sqa}"
GOLD = golds(Yte)
g_raw = greedy(m_a, Xte, maxlen=13)        # 定宽输出，逐位置对比要用它
g_te = [trim(p) for p in g_raw]            # 裁到 <eos> 的变长序列，序列级准确率用
ok_g = np.array([s == gg for s, gg in zip(g_te, GOLD)])
assert abs(ok_g.mean() - 0.739000) < 1e-6, f"贪心序列正确 {ok_g.mean()}"
print("末 200 步平均 loss %.6f" % float(np.mean(h_a["loss"][-200:])))
print("验证 TF token 准确 %.6f | TF 序列完全正确 %.6f" % (tka, sqa))
print("TF 逐位置:", [round(v, 6) for v in pa])
print("各位置样本数:", cnta)
print("测试集贪心序列完全正确 %.6f" % ok_g.mean())
print()
print("注意「token 准确率」明显高于「序列完全正确率」：")
print("token 准确率把每一位独立算分，错一位只扣 1/N 格；")
print("序列完全正确率要**每一位都对**，对长度和错误传播都敏感。")
print("做序列任务时只报 token 准确率会严重高估模型（这题差 %.4f）。" % (tka - sqa))
'''

# --------------------------------------------------------------------------- #
# T10 曝光偏差                                                                 #
# --------------------------------------------------------------------------- #

T10_CODE = '''src_t, y_t = torch.as_tensor(Xte), torch.as_tensor(Yte)
with torch.no_grad():
    tfp = m_a(src_t, shift_right(y_t)).argmax(-1)      # TF 口径的逐位预测
TN = int((y_t != PAD).sum(1).max())
g_t = g_raw                                            # free-running 口径（贪心结果，定宽）

tf_pos, fr_pos, cond, alive = [], [], [], []
pref_ok = torch.ones(len(Xte), dtype=torch.bool)
for k in range(TN):
    m = y_t[:, k] != PAD
    # @@todo 三种口径：TF 逐位置 / free 逐位置 / 「只在前缀全对的样本上」的条件准确率
    # @@hint TF: (tfp[:, k][m] == y_t[:, k][m])；free 换成 g_t
    # @@hint 条件：先 mc = pref_ok & m，再在这个子集上算 g_t；子集可能为空，用三元表达式兜 nan
    tf_pos.append(float((tfp[:, k][m] == y_t[:, k][m]).float().mean()))
    fr_pos.append(float((g_t[:, k][m] == y_t[:, k][m]).float().mean()))
    mc = pref_ok & m
    cond.append(float((g_t[:, k][mc] == y_t[:, k][mc]).float().mean()) if int(mc.sum()) else float("nan"))
    alive.append(int(pref_ok.sum()))
    pref_ok = pref_ok & (g_t[:, k] == y_t[:, k])
    # @@end

# ---- 验收 ----
assert [round(v, 6) for v in tf_pos] == [
    1.0, 1.0, 0.984, 0.981, 0.941, 0.899083, 0.900826, 0.902, 0.92, 0.818182]
assert [round(v, 6) for v in fr_pos] == [
    1.0, 1.0, 0.984, 0.967, 0.922, 0.847095, 0.806375, 0.774, 0.573333, 0.181818]
assert [round(v, 6) for v in cond[:-1]] == [
    1.0, 1.0, 0.984, 0.982724, 0.951396, 0.90849, 0.910097, 0.919355, 0.953125]
assert math.isnan(cond[-1]), "末位没有「前缀全对」的样本，条件准确率记 nan"
assert alive == [1000, 1000, 1000, 984, 967, 920, 824, 658, 342, 61]
assert fr_pos[-1] < tf_pos[-1] - 0.3, "最后一个位置 free-running 明显塌"
print("TF   逐位置    :", [round(v, 6) for v in tf_pos])
print("free 逐位置    :", [round(v, 6) for v in fr_pos])
print("TF-free 差值   :", [round(a - b, 6) for a, b in zip(tf_pos, fr_pos)])
print("条件(前缀全对) :", [round(v, 6) if not math.isnan(v) else "n/a" for v in cond])
print("前缀仍全对样本 :", alive)
print()
print("读法：条件准确率（在「前缀本来就全对」的样本里算）几乎不随位置下降，")
print("而 free-running 的整体准确率一路塌 —— 说明误差**主要来自错误前缀的传播**，")
print("而不是「模型看不懂长前缀」。这个现象就叫曝光偏差（exposure bias）：")
print("训练时永远看真值前缀，推理时看的是自己（可能已经错了）的前缀。")
'''

# --------------------------------------------------------------------------- #
# T11 oracle 前缀                                                              #
# --------------------------------------------------------------------------- #

T11_CODE = '''orc = []
with torch.no_grad():
    for k in range(TN):
        cur = torch.full((len(Xte), 1), BOS, dtype=torch.long)
        if k:
            cur = torch.cat([cur, y_t[:, :k]], 1)
        # @@todo 喂「前 k 个真值」，只预测第 k+1 位
        # @@hint m_a(src_t, cur)[:, -1].argmax(-1)
        nx = m_a(src_t, cur)[:, -1].argmax(-1)
        m = y_t[:, k] != PAD
        orc.append(float((nx[m] == y_t[:, k][m]).float().mean()))
        # @@end

# ---- 验收 ----
assert [round(v, 6) for v in orc] == [
    1.0, 1.0, 0.984, 0.981, 0.941, 0.899083, 0.900826, 0.902, 0.92, 0.818182]
assert orc == tf_pos, "oracle 逐位置应当与 TF 逐位置逐位相等"
assert max(abs(a - b) for a, b in zip(orc, fr_pos)) > 0.3
print("oracle 逐位置:", [round(v, 6) for v in orc])
print("free   逐位置:", [round(v, 6) for v in fr_pos])
print("oracle 比 free 高出最多 %.6f（同一个模型，只换前缀来源）"
      % max(abs(a - b) for a, b in zip(orc, fr_pos)))
print()
print("oracle 与 TF 逐位置**逐位完全相等** —— 这不是巧合：")
print("「喂前 k 个真值预测第 k+1 位」和「teacher forcing 看第 k 位」是同一件事。")
print("所以真正要对比的是 oracle（真值前缀）与 free（自产前缀），这一对差值")
print("就是曝光偏差的**因果**证据：模型逐位没错，错的是它看到的前缀。")
'''

# --------------------------------------------------------------------------- #
# T12 校准                                                                     #
# --------------------------------------------------------------------------- #

T12_CODE = '''# @@todo 用同样的 seed 再训一个 smoothing=0.1 的模型（其余完全一致）
# @@hint m_b, h_b = train(seed=11, smoothing=0.1)
m_b, h_b = train(seed=11, smoothing=0.1)
# @@end

# ---- 验收 ----
assert abs(np.mean(h_b["loss"][-200:]) - 0.653987) < 1e-6
_, tkb, sqb, _ = tf_eval(m_b, Xva, Yva)
gb = np.array([s == gg for s, gg in zip([trim(p) for p in greedy(m_b, Xte, maxlen=13)], GOLD)])
assert abs(tkb - 0.944562) < 1e-6 and abs(sqb - 0.676000) < 1e-6
assert abs(gb.mean() - 0.677000) < 1e-6
print("平滑版末 200 步 loss %.6f | 无平滑 %.6f"
      % (float(np.mean(h_b["loss"][-200:])), float(np.mean(h_a["loss"][-200:]))))
print("平滑版 TF token %.6f | TF 序列 %.6f（无平滑 %.6f / %.6f）" % (tkb, sqb, tka, sqa))
print("平滑版贪心序列 %.6f（无平滑 %.6f）" % (gb.mean(), ok_g.mean()))


@torch.no_grad()
def conf(model, X, Y):
    """返回（平均最大概率, 平均预测熵）"""
    src, y = torch.as_tensor(X), torch.as_tensor(Y)
    p = F.softmax(model(src, shift_right(y)), -1)
    m = y != PAD
    return (float(p.max(-1).values[m].mean()),
            float((-(p * torch.log(p + 1e-12)).sum(-1))[m].mean()))


CONF = {tag: conf(mm, Xva, Yva) for tag, mm in (("无平滑", m_a), ("平滑 0.1", m_b))}

# ---- 验收 ----
assert abs(CONF["无平滑"][0] - 0.948991) < 1e-6
assert abs(CONF["无平滑"][1] - 0.160423) < 1e-6
assert abs(CONF["平滑 0.1"][0] - 0.850709) < 1e-6
assert abs(CONF["平滑 0.1"][1] - 0.650392) < 1e-6
assert CONF["平滑 0.1"][1] > CONF["无平滑"][1] * 4, "平滑后预测熵大幅上升（更不自信）"
for tag, (mp, en) in CONF.items():
    print("  %-9s 平均最大概率 %.6f | 平均预测熵 %.6f" % (tag, mp, en))
print()
print("loss 从 %.6f 涨到 %.6f，别慌 —— 那是平滑给的下界，不是训坏了。"
      % (float(np.mean(h_a["loss"][-200:])), float(np.mean(h_b["loss"][-200:]))))
print("label smoothing 的副作用是**欠自信**：所有概率被往均匀分布推。")
print("这不是 bug：如果下游要做「置信度阈值 + 人工复核」，平滑过的模型反而更好校准；")
print("但如果你要的是「最大概率当分数」，它会系统性偏低。")
'''

# --------------------------------------------------------------------------- #
# T13 warmup 必要性                                                            #
# --------------------------------------------------------------------------- #

T13_CODE = '''rows_wu = {}
for tag, uw in (("有 warmup", True), ("无 warmup", False)):
    # @@todo peak_lr 放大到 3e-2、800 步，只切换 use_warmup，其余完全一致
    # @@hint train(steps=800, peak_lr=3e-2, warmup=100, use_warmup=uw, seed=33, log_first=10)
    mm, hh = train(steps=800, peak_lr=3e-2, warmup=100, use_warmup=uw, seed=33, log_first=10)
    rows_wu[tag] = (mm, hh)
    # @@end

# ---- 验收 ----
assert abs(np.mean(rows_wu["有 warmup"][1]["loss"][-100:]) - 0.259410) < 1e-6
assert abs(np.mean(rows_wu["无 warmup"][1]["loss"][-100:]) - 1.709144) < 1e-6
assert rows_wu["有 warmup"][1]["loss"][-1] < rows_wu["无 warmup"][1]["loss"][-1] / 3, \\
    "有 warmup 收敛明显更好"
assert not np.isnan(rows_wu["无 warmup"][1]["loss"]).any(), "不是崩成 nan，是停在高原"
for tag, (mm, hh) in rows_wu.items():
    print("  %-9s 前 10 步 loss : %s" % (tag, ["%.3f" % v for v in hh["loss"][:10]]))
    print("             前 10 步 ‖grad‖: %s" % ["%.2f" % v for v in hh["grad"]])
    print("             最大 loss %.4f (step %d) | 末 100 步均值 %.6f"
          % (max(hh["loss"]), int(np.argmax(hh["loss"])), float(np.mean(hh["loss"][-100:]))))
    print("             loss 每 100 步: %s" % ["%.3f" % hh["loss"][i] for i in range(0, 800, 100)])
    _, tk2, sq2, _ = tf_eval(mm, Xva, Yva)
    print("             验证 TF token %.6f | 序列 %.6f" % (tk2, sq2))
print()
print("peak_lr 放大 ~10 倍之后，「不加 warmup」不是崩成 nan，而是**长期停在高原**：")
print("前几步的大梯度把参数推到一个坏区域，之后 lr 已经降下来，再也爬不出去。")
print("所以 warmup 的价值不是「防爆炸」，而是「别在初始化附近下太重的脚」。")
'''

# --------------------------------------------------------------------------- #
# T14 beam search                                                              #
# --------------------------------------------------------------------------- #

BEAM_FUNC = '''@torch.no_grad()
def beam_search(model, X, beam=3, alpha=0.0, maxlen=13, trace_idx=None):
    """手写 beam search。

    score = Σ logP；若 alpha>0，排序与选优都用 score / (len ** alpha)。
    每一步：把 beam 条前缀各自扩成 top-`beam` 个候选 -> 排序 -> 只留前 beam 条。
    已经吐出 <eos> 的前缀不再扩展，收进 fin；最后在「自然结束」的序列里选分最高的。
    """
    out, traces = [], None
    for ri, row in enumerate(X):
        src = torch.as_tensor(row)[None]
        beams_ = [([BOS], 0.0)]
        fin, steps = [], []
        for _ in range(maxlen):
            cand = []
            for seq, sc in beams_:
                if seq[-1] == EOS:
                    fin.append((seq, sc))
                    continue
                lp = F.log_softmax(model(src, torch.as_tensor(seq)[None]), -1)[0, -1]
                # @@todo 取概率最高的 beam 个下一 token，扩成候选（新分 = 旧分 + logP）
                # @@hint tv, ti = lp.topk(beam) 得到 (值, 下标)
                # @@hint cand += [(seq + [i], sc + v) for v, i in zip(tv.tolist(), ti.tolist())]
                tv, ti = lp.topk(beam)
                cand += [(seq + [i], sc + v) for v, i in zip(tv.tolist(), ti.tolist())]
                # @@end
            if not cand:
                break
            # @@todo 按（长度惩罚后的）分数排序，只保留前 beam 条
            # @@hint key = (lambda t: t[1] / (len(t[0]) ** alpha)) if alpha else (lambda t: t[1])
            # @@hint cand.sort(key=key, reverse=True) 之后取 cand[:beam]
            key = (lambda t: t[1] / (len(t[0]) ** alpha)) if alpha else (lambda t: t[1])
            cand.sort(key=key, reverse=True)
            beams_ = cand[:beam]
            # @@end
            if trace_idx is not None and ri == trace_idx:
                steps.append(list(beams_))
        fin += beams_
        # @@todo 收尾选优：优先在「以 <eos> 结尾」的序列里挑，一条都没有才退回全部候选
        # @@hint nat = [t for t in fin if t[0][-1] == EOS]
        # @@hint best = max(nat if nat else fin, key=key)
        nat = [t for t in fin if t[0][-1] == EOS]
        best = max(nat if nat else fin, key=key)
        # @@end
        out.append((trim(best[0][1:]), best[1], bool(nat)))
        if trace_idx is not None and ri == trace_idx:
            traces = steps
    return out, traces
'''

# --------------------------------------------------------------------------- #
# T15 beam 效果与长度惩罚                                                      #
# --------------------------------------------------------------------------- #

T15_CODE = '''GB = golds(Yb)
gb_seq = [trim(p) for p in greedy(m_a, Xb, maxlen=13)]

BM = {}
for bm in (1, 2, 3, 5):
    # @@todo 跑一次 beam=bm，把每条样本的 (序列, 总分, 是否自然结束) 存下来
    # @@hint beam_search(...) 返回 (out, traces)；out 里每项是 (seq, score, natural)
    BM[bm] = beam_search(m_a, Xb, beam=bm)[0]
    # @@end

# ---- 验收 ----
assert abs(np.mean([s == gg for s, gg in zip(gb_seq, GB)]) - 0.760000) < 1e-6
_acc = {bm: np.mean([r[0] == gg for r, gg in zip(BM[bm], GB)]) for bm in BM}
assert abs(_acc[1] - 0.760000) < 1e-6 and abs(_acc[2] - 0.775000) < 1e-6
assert abs(_acc[3] - 0.775000) < 1e-6 and abs(_acc[5] - 0.775000) < 1e-6
assert _acc[3] > _acc[1], "beam search 应优于贪心"
assert _acc[3] == _acc[2] == _acc[5], "beam 加到 2 以后基本没有增量"
for bm in (1, 2, 3, 5):
    rs = BM[bm]
    print("  beam=%-2d 准确 %.6f | 平均句对数似然 %.6f | 平均长度 %.6f | 自然结束 %.6f"
          % (bm, _acc[bm], np.mean([r[1] for r in rs]),
             np.mean([len(r[0]) for r in rs]), np.mean([r[2] for r in rs])))
print()
print("beam=1 就是贪心（每次只留 1 条）。beam 从 1 到 2 就有收益，再往上加没有增量。")
print("原因见 T18 的轨迹：犯错大多是「第一个词就选错了」，")
print("而 beam 只能救「前几名里本来就有正确答案」的情况。")
'''

T15B_CODE = '''# @@todo 长度惩罚 α 扫描（beam=5）
# @@hint 排序分数 = 总分 / (len ** alpha)；α=0 就是只看总分
ALPHA = {al: beam_search(m_a, Xb, beam=5, alpha=al)[0] for al in (0.0, 0.3, 0.6, 1.0, 1.5)}
# @@end

# ---- 验收 ----
_ln = {al: float(np.mean([len(r[0]) for r in ALPHA[al]])) for al in ALPHA}
_aacc = {al: float(np.mean([r[0] == gg for r, gg in zip(ALPHA[al], GB)])) for al in ALPHA}
assert abs(_ln[0.0] - 7.565000) < 1e-6 and abs(_ln[1.5] - 7.590000) < 1e-6
assert abs(_aacc[0.0] - 0.775000) < 1e-6 and abs(_aacc[1.0] - 0.780000) < 1e-6
assert _ln[1.5] > _ln[0.0], "α 越大平均输出越长"
_nd = sum(1 for a, b in zip(ALPHA[0.0], ALPHA[1.0]) if a[0] != b[0])
assert _nd == 4, f"α=0 与 α=1 应有 4 条样本选出不同序列，实际 {_nd}"
for al in (0.0, 0.3, 0.6, 1.0, 1.5):
    seqs = [r[0] for r in ALPHA[al]]
    ln = [len(s) for s in seqs]
    print("  α=%.1f 准确 %.6f | 平均长度 %.6f | 长度<7 占比 %.6f | 长度>9 占比 %.6f"
          % (al, _aacc[al], _ln[al], np.mean([x < 7 for x in ln]),
             np.mean([x > 9 for x in ln])))
print()
print("α=0 与 α=1 选出不同序列的样本数 %d / %d，看第一个例子：" % (_nd, len(Xb)))
for _i, (_a, _b) in enumerate(zip(ALPHA[0.0], ALPHA[1.0])):
    if _a[0] != _b[0]:
        print("  输入 %s" % [ITOS[t] for t in Xb[_i]])
        print("    α=0 选 %s | 总分 %.6f | 长度 %d | 平均 %.6f"
              % ([ITOS[t] for t in _a[0]], _a[1], len(_a[0]), _a[1] / len(_a[0])))
        print("    α=1 选 %s | 总分 %.6f | 长度 %d | 平均 %.6f"
              % ([ITOS[t] for t in _b[0]], _b[1], len(_b[0]), _b[1] / len(_b[0])))
        break
print()
print("为什么需要长度惩罚：logP 每一项都是负数，序列越长总分越小。")
print("α=0 时模型会系统性偏爱短句（多说一个词就多扣一点分）——上面的例子里，")
print("α=0 选了更短的那条（总分更高），α=1 换成平均对数似然后选了更长的那条。")
print("工业界常用 α≈0.6~0.7（GNMT 的经验值）。")
'''

# --------------------------------------------------------------------------- #
# T16 长度外推                                                                #
# --------------------------------------------------------------------------- #

T16_CODE = '''EXTRA = {}
for L in (4, 6, 8, 10, 12, 14):
    XL, YL = make_dedup(150, L=L, seed=60 + L)
    # @@todo 对这个 L 各跑一次贪心与 beam3（α=0 / α=1），结果都存进 EXTRA[L]
    # @@hint greedy(m_a, XL, maxlen=L + 4)；beam_search(m_a, XL, beam=3, maxlen=L + 4)[0]
    EXTRA[L] = (golds(YL),
                [trim(p) for p in greedy(m_a, XL, maxlen=L + 4)],
                [r[0] for r in beam_search(m_a, XL, beam=3, alpha=0.0, maxlen=L + 4)[0]],
                [r[0] for r in beam_search(m_a, XL, beam=3, alpha=1.0, maxlen=L + 4)[0]])
    # @@end

# ---- 验收 ----
_accs = {L: tuple(float(np.mean([s == g for s, g in zip(v[i], v[0])])) for i in (1, 2, 3))
         for L, v in EXTRA.items()}
for _L, _exp in ((4, 25 / 150), (6, 120 / 150), (8, 123 / 150), (10, 118 / 150)):
    assert abs(_accs[_L][0] - _exp) < 1e-9, f"L={_L} 贪心 {_accs[_L][0]} 应为 {_exp}"
assert _accs[12][0] < 0.4 and _accs[14][0] < 0.4, "训练长度之外掉得很厉害"
assert _accs[4][0] < 0.4, "比训练长度短太多也不行"
for L in (4, 6, 8, 10, 12, 14):
    print("  L=%-3d 贪心 %.4f | beam3(α=0) %.4f | beam3(α=1) %.4f" % ((L,) + _accs[L]))
print()
print("训练只见过 L=10 的输入，但 L=6/8/10 都不错，L=4 与 L>=12 都断崖。")
print("L=4 掉下去的原因和「太长」不同：输出变短后，模型的「去重边界」行为")
print("在短输入上没被训练过（训练集的输出长度集中在 6~9）。")
print("结论：**长度外推不是免费的**。真实系统要么限制输入长度，要么在长样本上补训。")
'''

# --------------------------------------------------------------------------- #
# T17 似然与准确率的错位                                                       #
# --------------------------------------------------------------------------- #

SEQLL_FUNC = '''@torch.no_grad()
def seq_ll(model, X, seqs):
    """给一条完整序列算老师强制下的总对数似然 Σ logP"""
    v = []
    for row, s in zip(X, seqs):
        src = torch.as_tensor(row)[None]
        cur = torch.as_tensor([BOS] + s[:-1])[None]
        lp = F.log_softmax(model(src, cur), -1)[0]
        v.append(sum(lp[j - 1, t].item() for j, t in enumerate(s)))
    return np.array(v)
'''

T17_CODE = '''b5 = [r[0] for r in BM[5]]
ll_g, ll_b = seq_ll(m_a, Xb, gb_seq), seq_ll(m_a, Xb, b5)
og = np.array([s == gg for s, gg in zip(gb_seq, GB)])
ob = np.array([s == gg for s, gg in zip(b5, GB)])

# ---- 验收 ----
assert abs(ll_g.mean() - (-51.070757)) < 1e-5 and abs(ll_b.mean() - (-50.305560)) < 1e-5
assert abs(np.mean(ll_b >= ll_g - 1e-9) - 0.985000) < 1e-6, "beam 的总似然不应低于贪心"
assert (int((og & ob).sum()), int((~og & ~ob).sum()),
        int((og & ~ob).sum()), int((~og & ob).sum())) == (151, 44, 1, 4)
print("  贪心  平均对数似然 %.6f | 序列正确 %.6f" % (ll_g.mean(), og.mean()))
print("  beam5 平均对数似然 %.6f | 序列正确 %.6f" % (ll_b.mean(), ob.mean()))
print("  beam 似然 ≥ 贪心的比例 %.6f" % np.mean(ll_b >= ll_g - 1e-9))
print("  都对 %d | 都错 %d | 贪心对beam错 %d | 贪心错beam对 %d"
      % (int((og & ob).sum()), int((~og & ~ob).sum()),
         int((og & ~ob).sum()), int((~og & ob).sum())))
print()
print("beam search 保证的是「在搜索空间里找到总似然更高的序列」，")
print("但**总似然高 ≠ 和真值一致**：真值只是众多高分序列之一。")
print("看这四个计数：beam 救回 4 条，同时也「帮倒忙」弄错 1 条。")
print("所以 beam 换来的准确率提升，取决于「模型的排序能力」而非「搜索能力」。")
'''

# --------------------------------------------------------------------------- #
# T18 beam 轨迹                                                                #
# --------------------------------------------------------------------------- #

T18_CODE = '''X1, Y1 = make_dedup(1, L=6, seed=99)
# @@todo 对这条样本做 beam=3 的搜索，并把每一步的 beam 状态留下来
# @@hint beam_search(m_a, X1, beam=3, maxlen=8, trace_idx=0) 返回 (out, traces)
r1, tr = beam_search(m_a, X1, beam=3, maxlen=8, trace_idx=0)
# @@end

# ---- 验收 ----
assert len(tr) == 8 and len(tr[0]) == 3, "8 步、每步 3 条 beam"
assert tr[0][0][0][-1] == int(Y1[0][0]), "第一步就选中了正确的首符号"
assert r1[0][0] == [int(v) for v in Y1[0] if v != PAD], "最终选出的序列与真值一致"
print("输入:", [ITOS[i] for i in X1[0]], "-> 真值:", [ITOS[i] for i in Y1[0] if i])
for si, bs in enumerate(tr):
    print("  step %d:" % si, [([ITOS[i] for i in s], round(sc, 4)) for s, sc in bs])
print("最终:", [ITOS[i] for i in r1[0][0]], "| 分数 %.6f" % r1[0][1])
print()
print("看 step 0：正确首符号「9」的累计分数是 -0.0015，第二名「<eos>」是 -7.4149。")
print("差了 7 分多 —— 一旦首符号选错，后面无论怎么搜都救不回来。")
print("这解释了上一节「beam 提升有限」：增益只来自模型**本来就很犹豫**的位置；")
print("而一个训得还不错的模型，绝大多数位置都是「一眼就定」的。")
'''

# --------------------------------------------------------------------------- #
# 汇总与讲解                                                                   #
# --------------------------------------------------------------------------- #

SUMMARY = '''rows = [
    ("teacher forcing", "shift_right：右移一位 + 首位补 <bos>",
     "形状不变，第 1 位起等于真值前 n-1 位"),
    ("标签平滑", "手写 vs nn.CrossEntropyLoss",
     "ε=0.1 差 0.000e+00；目标值 0.907692 / 0.007692"),
    ("warmup+余弦", "手写 vs LambdaLR", "最大绝对差 0.000e+00；峰值在 warmup 末步"),
    ("原版 inv-sqrt", "d^-0.5·min(s^-0.5, s·warmup^-1.5)",
     "峰值 8.838835e-03 出现在 step=400（≈warmup）"),
    ("确定性·g2", "dropout>0 且循环裸跑", "两次结果不同（这就是 must-fix 的根因）"),
    ("确定性·g3", "循环也包进 fork_rng", "同进程内一致，但**重跑进程会变**"),
    ("确定性·g4", "先 manual_seed 再包住全部", "同进程一致 + 跨进程可复现"),
    ("主模型", "1 层 d=32 / 训练 700 步",
     "TF token %.4f | TF 序列 %.4f | 贪心 %.4f" % (tka, sqa, ok_g.mean())),
    ("曝光偏差", "TF vs free-running 逐位置",
     "末位 TF %.4f vs free %.4f（差 %.4f）" % (tf_pos[-1], fr_pos[-1], tf_pos[-1] - fr_pos[-1])),
    ("oracle 前缀", "喂真值前缀只预测下一位", "与 TF 逐位置**逐位相等**（同一件事）"),
    ("标签平滑副作用", "平均最大概率 / 预测熵",
     "%.4f→%.4f | %.4f→%.4f" % (CONF["无平滑"][0], CONF["平滑 0.1"][0],
                                CONF["无平滑"][1], CONF["平滑 0.1"][1])),
    ("warmup 必要性", "peak_lr=3e-2 / 800 步的末 100 步 loss",
     "%.6f（有）vs %.6f（无）"
     % (np.mean(rows_wu["有 warmup"][1]["loss"][-100:]),
        np.mean(rows_wu["无 warmup"][1]["loss"][-100:]))),
    ("beam search", "beam=1/2/3/5 的序列准确率",
     "%.4f → %.4f（beam≥2 后无增量）" % (_acc[1], _acc[3])),
    ("长度惩罚 α", "α=0 看总分 / α=1 看平均",
     "平均长度 %.3f → %.3f；%d/%d 条选出不同序列" % (_ln[0.0], _ln[1.5], _nd, len(Xb))),
    ("长度外推", "L=4/6/8/10/12/14 的贪心准确率",
     " / ".join("%.4f" % _accs[L][0] for L in (4, 6, 8, 10, 12, 14))),
    ("似然错位", "beam5 vs 贪心的总似然与准确率",
     "%.4f vs %.4f（似然更高，准确率只 +%.4f）" % (ll_b.mean(), ll_g.mean(), ob.mean() - og.mean())),
]
print(pd.DataFrame(rows, columns=["条目", "做法", "实测结论"]).to_string(index=False))
'''

LESSON = [
    md(
        """
# ch06 训练与推理细节手写

> 这一章把 ch05 手写的 Transformer 真正「训起来」并「解码出来」。
> 任务是一个玩具 seq2seq：**去重保序**（输入 10 个数字符号，按首次出现顺序去重后输出）。

**这一章最有价值的三个实验**

1. **训练确定性**（T6）：`fork_rng` 只保证「同一次进程内前后一致」，
   它**还原**随机数状态而不是**固定**它 —— 不预先 `manual_seed` 的话，
   重跑 notebook 数字就变（实测 `mode=all` 从 1.9810 变成 1.9989）。
   这是「同一份代码两次跑结果不一样」最隐蔽的根因。
2. **曝光偏差**（T10 / T11）：TF 与 free-running 的逐位置对照，加上 oracle 前缀的因果验证 ——
   末位差 **0.636364**，而 oracle 与 TF 逐位**完全相等**，证明误差来自**错误前缀的传播**。
3. **beam search 的真实收益**（T15 / T18）：贪心 **0.7600** → beam **0.7750**，
   但 beam≥2 之后不再涨；trace 显示「首符号一错就救不回来」。

| 块 | 内容 | 难度 |
|---|---|---|
| T1 | 两种 teacher forcing 输入（shift_right） | ★ |
| T2 | 手写 label smoothing | ★★ |
| T3 | 与 `nn.CrossEntropyLoss` 对账 | ★ |
| T4 | 手写 warmup + 余弦退火 | ★★ |
| T5 | 与 `LambdaLR` 对账 | ★ |
| T6 | **训练确定性：fork_rng 的还原语义** | ★★★ |
| T7 | 训练主循环 | ★★ |
| T8 | 贪心解码 | ★ |
| T9 | 主模型训练与 TF 评估 | ★ |
| T10 | **曝光偏差逐位置对照** | ★★★ |
| T11 | oracle 前缀的因果验证 | ★★ |
| T12 | label smoothing 的校准副作用 | ★★ |
| T13 | warmup 必要性（放大 lr 对照） | ★★ |
| T14 | **手写 beam search（含长度惩罚）** | ★★★ |
| T15 | beam 效果与 α 扫描 | ★★ |
| T16 | 长度外推 | ★★ |
| T17 | 似然与准确率的错位 | ★★ |
| T18 | beam 轨迹 | ★ |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    md(
        """
## 任务与模型

任务选了「去重保序」而不是「复制/反转」，是因为它满足三个条件：

1. **输出长度可变**（真值长度 4~11，平均 7.5215）—— 否则「自然结束」「长度惩罚」都没意义；
2. **是真 seq2seq**（不是逐位置映射）—— 第 t 位输出依赖前面所有输入；
3. **小模型学得会、但学不完美**（1 层 d=32、700 步、CPU 秒级）——
   这一点是定标时最关键的一条：训到 2500 步模型就全对了，
   贪心与 beam 完全重合，整章的「解码差异」全都演示不出来。
"""
    ),
    code(NET_CODE),
    md("## T1 shift_right：两种 teacher forcing 输入"),
    code(T1_CODE),
    md("## T2 手写 label smoothing"),
    code(T2_CODE),
    md("## T3 与框架对账"),
    code(T3_CODE),
    md("## T4 手写 warmup + 余弦退火"),
    code(T4_CODE),
    md("## T5 与 torch 调度器对账"),
    code(T5_CODE),
    md(
        """
## T6 训练确定性：fork_rng 只「还原」不「固定」★

同一个脚本跑两遍、数字不一样，是工程里最常见的灵异事件之一。这一节用四组对照把它拆开：

| 组 | dropout | 训练循环是否在 fork_rng 内 | 进入前是否 `manual_seed` |
|---|---|---|---|
| g1 | 0.0 | 否 | 否 |
| g2 | 0.1 | 否 | 否 |
| g3 | 0.1 | **是** | 否 |
| g4 | 0.1 | **是** | **是** |

预期：g1 一致、g2 **不一致**、g3 一致（但**重跑 notebook 会变**）、g4 一致且可复现。
"""
    ),
    code(DETERM_FUNC),
    code(T6_CODE),
    md("## T7 训练主循环"),
    code(TRAIN_FUNC),
    md("## T8 评估与解码工具（tf_eval / trim / golds 已给，只需填 greedy）"),
    code(EVAL_FUNC),
    code(GREEDY_FUNC),
    md("## T9 主模型训练与 TF 评估"),
    code(T9_CODE),
    md("## T10 **曝光偏差**：teacher forcing vs free-running ★"),
    code(T10_CODE),
    md("## T11 oracle 前缀：把因果钉死"),
    code(T11_CODE),
    md("## T12 label smoothing 的校准副作用"),
    code(T12_CODE),
    md("## T13 warmup 必要性（把 lr 放大 ~10 倍才看得出来）"),
    code(T13_CODE),
    md("## T14 **手写 beam search**（含长度惩罚）★"),
    code(BEAM_FUNC),
    md("## T15 beam 效果与长度惩罚 α"),
    code(T15_CODE),
    code(T15B_CODE),
    md("## T16 长度外推"),
    code(T16_CODE),
    md("## T17 似然与准确率的错位"),
    code(SEQLL_FUNC),
    code(T17_CODE),
    md("## T18 beam 轨迹：看清搜索在做什么"),
    code(T18_CODE),
    md("## 汇总"),
    code(SUMMARY),
    md(
        """
## 本章小结

| 知识点 | 一句话 |
|---|---|
| teacher forcing | decoder 输入是**右移一位的真值**，首位补 `<bos>` |
| label smoothing | 目标分布 `1-ε+ε/V` 与 `ε/V`，手写与框架逐位一致 |
| 交叉熵下界 | 平滑后即便全对，loss 也不为 0（下界被抬高） |
| warmup | 先线性升再退火；**未 warmup + 大 lr** 会长期停在高原而非发散 |
| 余弦退火 vs inv-sqrt | 前者一定降到 0，后者长训几乎不衰减 |
| **fork_rng** | **只「还原」不「固定」**：不预置种子就无法跨进程复现 |
| 曝光偏差 | TF 逐位置 ≈ oracle；free-running 一路塌，差在错误前缀的传播 |
| 条件准确率 | 只统计「前缀本来就全对」的样本，用来区分「不会」与「被带偏」 |
| 贪心解码 | 每步取 argmax，无回溯 |
| beam search | 保留 top-k 前缀；`score = ΣlogP`，长度惩罚 `÷len^α` |
| 长度惩罚 α | α=0 看总分（偏爱短句），α=1 看平均对数似然 |
| 长度外推 | 训练长度之外会断崖，不是免费的能力 |

## 易错点

1. **`nn.Transformer` 默认 `dropout=0.1`** —— 不显式写 `0.0`，训练结果不可复现。
2. **以为 `fork_rng` 能保证可复现** —— 它只还原到「进入时的状态」，那个状态默认是随机的。
3. **lr 写在 `opt.step()` 之后** —— 第一步会用初始满 lr，warmup 白做一格。
4. **label smoothing 之后还期望 loss 归零** —— 平滑给 loss 设了下界。
5. **只报 token 准确率** —— 序列任务的 token 准确率会严重高估模型。
6. **用同一个模型比「beam vs 贪心」却只看似然** —— beam 必然似然更高，准确率不一定。
7. **长度惩罚写成 `score * len^α`** —— 分数是负的，乘会让长句更差，必须是**除以**。
8. **beam search 里忘记处理已经结束的序列** —— 已吐 `<eos>` 的前缀继续扩展会污染候选池。
9. **挖空块里写 `if` / `for`** —— `nb_builder` 直接 `BuildError`；
   用三元表达式（`a if cond else b`）或把控制流移到块外。
10. **把 oracle 与 TF 逐位置当成两个实验** —— 它们本质上是一回事（实测逐位相等）。

## 自查清单

- [ ] `shift_right` 后形状不变、首位是 `<bos>`、第 1 位起等于真值的前 n-1 位
- [ ] 手写 label smoothing 与框架差 < 1e-6；目标值 **0.907692 / 0.007692**；`lnV = 2.564949`
- [ ] 手写调度器与 `LambdaLR` 最大差 **0.000e+00**；峰值在 warmup 末步
- [ ] 原版 inv-sqrt 峰值 **8.838835e-03** 出现在 step=400
- [ ] **g2 两次不同、g1/g3/g4 两次相同**，且 g3 重跑会变（1.9810 → 1.9989）
- [ ] TF token **0.952653** 明显高于 TF 序列 **0.729000**
- [ ] free-running 末位 **0.181818** 远低于 TF 末位 **0.818182**
- [ ] oracle 逐位置与 TF 逐位置**逐位相等**
- [ ] 平滑版预测熵 **0.650392** 显著高于无平滑 **0.160423**
- [ ] 无 warmup 的末 100 步 loss **1.709144** 比有 warmup **0.259410** 差 6 倍以上
- [ ] beam=3 的序列准确率 **0.775000** 高于 beam=1（贪心）**0.760000**
- [ ] α 从 0 涨到 1.5，平均长度 **7.565 → 7.590**；4/200 条选出不同序列
- [ ] L=12/14 的准确率断崖式下降（**0.1467 / 0.0333**）

**答不上来时问自己**

1. 为什么 teacher forcing 一定要右移一位？
2. `fork_rng` 到底做了什么、没做什么？
3. 曝光偏差为什么用「条件准确率」才能看出来？
4. beam search 保证找到的是什么？它保证不了什么？
5. 长度惩罚为什么必须是除法而不是乘法？
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch06 训练与推理细节（练习版）

> 补全 `____`，跑通所有 assert。真值全部来自实跑。
> 三个**必看**实验：T6（训练确定性）、T10（曝光偏差）、T15（beam 的真实收益）。

| 块 | 内容 | 难度 |
|---|---|---|
| T1 | shift_right | ★ |
| T2 | 手写 label smoothing | ★★ |
| T3 | 与框架对账 | ★ |
| T4 | warmup + 余弦退火 | ★★ |
| T5 | 与 `LambdaLR` 对账 | ★ |
| T6 | **训练确定性** | ★★★ |
| T7 | 训练主循环 | ★★ |
| T8 | 贪心解码 | ★ |
| T9 | 主模型训练 | ★ |
| T10 | **曝光偏差** | ★★★ |
| T11 | oracle 前缀 | ★★ |
| T12 | 校准副作用 | ★★ |
| T13 | warmup 必要性 | ★★ |
| T14 | **beam search** | ★★★ |
| T15 | beam 效果与 α | ★★ |
| T16 | 长度外推 | ★★ |
| T17 | 似然错位 | ★★ |
| T18 | beam 轨迹 | ★ |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(NET_CODE),
    md("## T1 shift_right"),
    code(T1_CODE),
    md("## T2 手写 label smoothing"),
    code(T2_CODE),
    md("## T3 与框架对账"),
    code(T3_CODE),
    md("## T4 warmup + 余弦退火"),
    code(T4_CODE),
    md("## T5 与 `LambdaLR` 对账"),
    code(T5_CODE),
    md("## T6 训练确定性"),
    code(DETERM_FUNC),
    code(T6_CODE),
    md("## T7 训练主循环"),
    code(TRAIN_FUNC),
    md("## T8 评估工具与贪心解码"),
    code(EVAL_FUNC),
    code(GREEDY_FUNC),
    md("## T9 主模型训练"),
    code(T9_CODE),
    md("## T10 曝光偏差"),
    code(T10_CODE),
    md("## T11 oracle 前缀"),
    code(T11_CODE),
    md("## T12 校准副作用"),
    code(T12_CODE),
    md("## T13 warmup 必要性"),
    code(T13_CODE),
    md("## T14 beam search"),
    code(BEAM_FUNC),
    md("## T15 beam 效果与 α 扫描"),
    code(T15_CODE),
    code(T15B_CODE),
    md("## T16 长度外推"),
    code(T16_CODE),
    md("## T17 似然错位"),
    code(SEQLL_FUNC),
    code(T17_CODE),
    md("## T18 beam 轨迹"),
    code(T18_CODE),
    md("## 汇总"),
    code(SUMMARY),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
