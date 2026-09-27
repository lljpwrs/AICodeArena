#!/usr/bin/env python3
"""nlp09 小模型训练与规模判定：参数量恒等式 / 内存四件套 / 激活缩放律 / 单步耗时 / 规模反推 / 省内存手段。

离线约束：全部走 `Config` + `from_config()` 随机初始化，**不下载任何预训练权重**。
真值全部实跑（torch 2.14.0 / transformers 5.17.0 / CPU，`torch.set_num_threads(1)`）。

> 本章回答一个工程问题：**"这个模型，我这台机器训得动吗？"**
> 答案不该靠试错 ——
> ① 参数量可以零误差**先算出来**（恒等式 vs 实测相对误差 **0.0000%**）；
> ② 训练内存 = **16 B/参数 + 激活**，其中 16 B 是 fp32 权重 4 + 梯度 4 + Adam 状态 8；
> ③ 激活随 batch **严格线性**、随 seq **超线性**（attention 的 O(S²) 项）。
> 把这三点算清楚，"能不能训"在写第一行训练代码之前就有答案。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp09_scale_budget.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch09_scale_budget"

# =========================================================================== #
# 公共部分（两版都给）                                                        #
# =========================================================================== #

IMPORTS = '''import gc
import time
from copy import deepcopy

import pandas as pd
import torch
from transformers import BertConfig, BertModel

torch.set_num_threads(1)          # 耗时标定要稳定 —— 多线程会让数字剧烈波动
SEED = 42
MB = 1024 ** 2
GB = 1024 ** 3
'''

SCAFFOLD = '''# ---- 五档规模：从"玩具"一路到 BERT-large，全部用 Config 随机初始化 ----
SCALES = {
    "tiny":  dict(vocab_size=1000,  hidden_size=64,   num_hidden_layers=2,
                  num_attention_heads=2,  intermediate_size=256,  max_position_embeddings=64),
    "small": dict(vocab_size=4000,  hidden_size=128,  num_hidden_layers=4,
                  num_attention_heads=4,  intermediate_size=512,  max_position_embeddings=128),
    "mid":   dict(vocab_size=8000,  hidden_size=256,  num_hidden_layers=6,
                  num_attention_heads=8,  intermediate_size=1024, max_position_embeddings=256),
    "base":  dict(vocab_size=21128, hidden_size=768,  num_hidden_layers=12,
                  num_attention_heads=12, intermediate_size=3072, max_position_embeddings=512),
    "large": dict(vocab_size=21128, hidden_size=1024, num_hidden_layers=24,
                  num_attention_heads=16, intermediate_size=4096, max_position_embeddings=512),
}

SCALE_BATCHES = [(1, 32), (2, 32), (4, 32), (2, 64), (2, 128), (8, 32)]

FREEZE_PLAN = [("完全不冻结", False, 0), ("只冻结 emb", True, 0),
               ("emb + 前 2 层", True, 2), ("emb + 前 4 层", True, 4)]


def bert_params(V, d, L, f, max_pos, type_vocab=2):
    """BERT 参数量恒等式（fp32，含全部 bias 与 LayerNorm）。

    拆成三块：
      embeddings: V*d + max_pos*d + type_vocab*d + 2*d   （三个 Embedding + 一层 LayerNorm）
      每层 Encoder: 4d^2 + 4d   （q/k/v/attn-out 四个 d*d 矩阵，各带 d 个 bias）
                    + 2d        （attention 后的 LayerNorm）
                    + 2*f*d + f （FFN 第一层，升维）
                    + f*d + d   （FFN 第二层，降维）
                    + 2d        （FFN 后的 LayerNorm）
                    => 4d^2 + 2*f*d + 9*d + f
      pooler: d*d + d
    """
    emb = d * (V + max_pos + type_vocab) + 2 * d
    layer = 4 * d * d + 2 * f * d + 9 * d + f
    pooler = d * d + d
    return {"emb": emb, "layer": layer, "pooler": pooler,
            "total": emb + L * layer + pooler}


def params_of(name):
    """按档位名算参数量 —— SCALES 的 key 与恒等式形参名不同，这里显式挑出来。"""
    kw = SCALES[name]
    return bert_params(kw["vocab_size"], kw["hidden_size"], kw["num_hidden_layers"],
                       kw["intermediate_size"], kw["max_position_embeddings"])


def make_model(name, dropout=0.1):
    cfg = BertConfig(type_vocab_size=2, hidden_dropout_prob=dropout,
                     attention_probs_dropout_prob=dropout, **SCALES[name])
    return cfg, BertModel(cfg)


def make_batch(name, B, S, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, SCALES[name]["vocab_size"], (B, S), generator=g)
    return ids, torch.ones_like(ids)


def tensor_bytes(iterable):
    """按实际 dtype 的 element_size 累加字节数 —— 这是唯一精确的口径。

    （不要用 RSS 去量小模型：Python 内存池复用会让它偏差几十倍。）
    """
    return sum(t.numel() * t.element_size() for t in iterable if torch.is_tensor(t))


def opt_state_bytes(opt):
    """AdamW 的 state 是 {step, exp_avg, exp_avg_sq}，逐条数出来。"""
    return sum(t.numel() * t.element_size()
               for st in opt.state.values() for t in st.values() if torch.is_tensor(t))


def activation_bytes(model, ids, am):
    """激活内存 = 反向传播所需的全部「保存张量」。

    用 saved_tensors_hooks 拦截 forward 期间被保存的每一个张量，
    按 data_ptr 去重后累加 —— 这比估公式精确，也比量 RSS 干净。
    """
    saved = []

    def pack(t):
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(ids, attention_mask=am).last_hidden_state.pow(2).mean().backward()

    seen, tot = set(), 0
    for t in saved:
        if t.data_ptr() in seen:
            continue
        seen.add(t.data_ptr())
        tot += t.numel() * t.element_size()
    return tot


def micro_batches(ids_all, am_all, B_micro):
    """把整批切成若干微批 —— 梯度累积要按同样的口径切。"""
    n = ids_all.shape[0] // B_micro
    return [(ids_all[i * B_micro:(i + 1) * B_micro],
             am_all[i * B_micro:(i + 1) * B_micro]) for i in range(n)]


def freeze_params(model, freeze_emb, n_layer):
    """冻结 embeddings（可选）与前 n_layer 层，返回冻结的张量个数。"""
    to_freeze = list(model.embeddings.parameters()) if freeze_emb else []
    for layer in model.encoder.layer[:n_layer]:
        to_freeze.extend(layer.parameters())
    for p in to_freeze:
        p.requires_grad = False
    return len(to_freeze)


def fwd_fn(model, ids, am):
    return model(ids, attention_mask=am).last_hidden_state


def step_fn(model, opt, ids, am):
    fwd_fn(model, ids, am).pow(2).mean().backward()
    opt.step()
    opt.zero_grad()


def timed(fn, n=3):
    fn()                                    # warmup，避免把首次编译/分配算进去
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def timed_fwd_ms(model, ids, am, n=3):
    """纯推理（no_grad）下的前向耗时，毫秒。"""
    with torch.no_grad():
        model(ids, attention_mask=am)
        t0 = time.perf_counter()
        for _ in range(n):
            model(ids, attention_mask=am)
        return (time.perf_counter() - t0) / n * 1000
'''

# =========================================================================== #
# 任务 1：参数量恒等式                                                        #
# =========================================================================== #

T1 = '''# @@todo 用恒等式算出 tiny 档的总参数量，存入 p_tiny
p_tiny = params_of("tiny")["total"]
# @@end
assert p_tiny == 172480, p_tiny
print(f"tiny 档参数量 {p_tiny:,}")
'''

# =========================================================================== #
# 任务 2：五档规模与组件占比                                                  #
# =========================================================================== #

T2 = '''rows = []
for name, kw in SCALES.items():
    # @@todo 算该档的参数量字典，并算出 embeddings / 每层 Encoder / pooler 各自的占比（百分数）
    p = params_of(name)
    rows.append({"档位": name,
                 "参数量": p["total"],
                 "Emb%": round(p["emb"] / p["total"] * 100, 1),
                 "Layer%": round(p["layer"] * kw["num_hidden_layers"] / p["total"] * 100, 1),
                 "Pooler%": round(p["pooler"] / p["total"] * 100, 1)})
    # @@end
param_table = pd.DataFrame(rows).set_index("档位")
print(param_table.to_string())
'''

# =========================================================================== #
# 任务 3：恒等式 vs 实测                                                      #
# =========================================================================== #

T3 = '''check = {}
for name in ("tiny", "small", "mid"):
    # @@todo 实例化该档模型，实测参数总量；算出「实测 − 恒等式」的相对误差（百分数，保留 4 位）
    _, model = make_model(name)
    actual = sum(p.numel() for p in model.parameters())
    expect = params_of(name)["total"]
    check[name] = {"恒等式": f"{expect:,}", "实测": f"{actual:,}",
                   "相对误差%": round((actual - expect) / actual * 100, 4)}
    # @@end
    del model
    gc.collect()
assert all(v["相对误差%"] == 0.0 for v in check.values()), check
print(pd.DataFrame(check).T.to_string())
'''

# =========================================================================== #
# 任务 4：内存四件套                                                          #
# =========================================================================== #

T4 = '''mem_rows = []
for name in ("tiny", "small", "mid"):
    _, model = make_model(name)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids, am = make_batch(name, 2, 32)
    n_param = sum(p.numel() for p in model.parameters())
    # @@todo 依次量出「激活 / 参数 / 梯度 / 优化器状态」四项的 MB（激活必须先量，它内部要做一次 backward）
    act_b = activation_bytes(model, ids, am) / MB
    opt.step()                       # step 一次，AdamW 才会真正建出 exp_avg / exp_avg_sq
    param_b = tensor_bytes(model.parameters()) / MB
    grad_b = tensor_bytes(p.grad for p in model.parameters() if p.grad is not None) / MB
    opt_b = opt_state_bytes(opt) / MB
    total_b = param_b + grad_b + opt_b + act_b
    mem_rows.append({"档位": name, "参数MB": round(param_b, 2), "梯度MB": round(grad_b, 2),
                     "优化器MB": round(opt_b, 2), "激活MB": round(act_b, 2),
                     "合计MB": round(total_b, 2),
                     "B/参数": round(total_b * MB / n_param, 1)})
    # @@end
    del model, opt
    gc.collect()
mem_table = pd.DataFrame(mem_rows).set_index("档位")
print(mem_table.to_string())
'''

# =========================================================================== #
# 任务 5：优化器状态精确账                                                    #
# =========================================================================== #

T5 = '''P_mid = params_of("mid")["total"]
D_MID = SCALES["mid"]["hidden_size"]
_, mid_model = make_model("mid")
mid_model.train()
mid_opt = torch.optim.AdamW(mid_model.parameters(), lr=1e-4)
ids5, am5 = make_batch("mid", 2, 32)
mid_model(ids5, attention_mask=am5).last_hidden_state.pow(2).mean().backward()
mid_opt.step()
# @@todo 统计「有梯度的参数量」「没有梯度的参数量」「优化器状态的实际字节数」三项
n_grad = sum(p.numel() for p in mid_model.parameters() if p.grad is not None)
n_no_grad = sum(p.numel() for p in mid_model.parameters() if p.grad is None)
measured = opt_state_bytes(mid_opt)
# @@end
assert n_grad + n_no_grad == P_mid
assert n_no_grad == D_MID * D_MID + D_MID, n_no_grad
assert measured == 2 * n_grad * 4 + len(mid_opt.state) * 4, (measured, n_grad, len(mid_opt.state))
print(f"总参数 {P_mid:,} | 有梯度 {n_grad:,} | 无梯度 {n_no_grad:,}"
      f"（= pooler 的 {D_MID}² + {D_MID}）")
print(f"优化器状态 {measured:,} B = 2 × {n_grad:,} × 4 B + {len(mid_opt.state)} × 4 B（每条的 step）")
'''

# =========================================================================== #
# 任务 6：激活的缩放律                                                        #
# =========================================================================== #

T6 = '''scale_rows = []
_, m6 = make_model("mid")
m6.train()
for B, S in SCALE_BATCHES:
    # @@todo 量出该 (batch, seq) 组合下的激活 MB 与「每 token 字节数」
    m6.zero_grad()
    ids6, am6 = make_batch("mid", B, S)
    ab6 = activation_bytes(m6, ids6, am6)
    scale_rows.append({"batch": B, "seq": S, "B×S": B * S,
                       "激活MB": round(ab6 / MB, 2),
                       "每token B": round(ab6 / (B * S))})
    # @@end
act_table = pd.DataFrame(scale_rows)
print(act_table.to_string(index=False))
'''

# =========================================================================== #
# 任务 7：单步耗时标定                                                        #
# =========================================================================== #

T7 = '''time_rows = []
for name in ("tiny", "small", "mid"):
    _, model = make_model(name)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids7, am7 = make_batch(name, 2, 32)
    n_param = sum(p.numel() for p in model.parameters())
    # @@todo 用 timed() 分别测「纯前向」与「整步（前向 + 反向 + 参数更新）」的秒数，再换算成毫秒
    fwd_ms = timed(lambda: fwd_fn(model, ids7, am7)) * 1000
    step_ms = timed(lambda: step_fn(model, opt, ids7, am7)) * 1000
    time_rows.append({"档位": name, "参数量": n_param, "前向ms": round(fwd_ms, 2),
                      "整步ms": round(step_ms, 2), "倍率": round(step_ms / fwd_ms, 2)})
    # @@end
    del model, opt
    gc.collect()
print(pd.DataFrame(time_rows).to_string(index=False))
'''

# =========================================================================== #
# 任务 8：base 档 —— 算得出来，跑得动吗                                       #
# =========================================================================== #

T8 = '''# @@todo 算出 base 档「训练态」的理论内存（GB），按 16 B/参数
base_train_gb = params_of("base")["total"] * 16 / GB
# @@end
assert round(base_train_gb, 2) == 1.52, base_train_gb
print(f"base 档参数量 {params_of('base')['total']:,}"
      f"（{params_of('base')['total'] / 1e6:.1f} M）| 训练态理论内存 {base_train_gb:.2f} GB")

base_cfg, base_model = make_model("base", dropout=0.0)
base_model.eval()
for B, S in [(1, 32), (8, 128)]:
    # @@todo 用 timed_fwd_ms() 测 base 档在 no_grad 下的前向耗时（毫秒）
    ms = timed_fwd_ms(base_model, *make_batch("base", B, S))
    # @@end
    print(f"  前向 batch={B} seq={S}：{ms:.1f} ms")
'''

# =========================================================================== #
# 任务 9：规模判定 —— 8 GB 能训多大                                           #
# =========================================================================== #

T9 = '''budget_rows = []
BUDGET_GB = 8.0
for name in SCALES:
    # @@todo 算出该档训练态所需内存（GB，只含权重 + 梯度 + 优化器状态），并判断能否装进 BUDGET_GB
    need_gb = params_of(name)["total"] * 16 / GB
    budget_rows.append({"档位": name,
                        "参数量M": round(params_of(name)["total"] / 1e6, 1),
                        "训练态GB": round(need_gb, 2),
                        "装得下": "是" if need_gb < BUDGET_GB else "否"})
    # @@end
print(pd.DataFrame(budget_rows).to_string(index=False))
'''

# =========================================================================== #
# 任务 10：梯度累积 —— 梯度等价吗                                             #
# =========================================================================== #

T10 = '''torch.manual_seed(SEED)
_, model_a = make_model("mid", dropout=0.0)
model_b = deepcopy(model_a)
ids_all, am_all = make_batch("mid", 8, 32, seed=7)
model_a.train()
model_b.train()
# ---- A：整批 B=8 一次算完 ----
# @@todo 整批前向 + 反向，把全部梯度克隆下来存进 grad_full
fwd_fn(model_a, ids_all, am_all).pow(2).mean().backward()
grad_full = [p.grad.clone() for p in model_a.parameters() if p.grad is not None]
# @@end
for i in range(4):
    # @@todo 取第 i 份微批（每份 2 条），loss 除以累积步数 4 后反向 —— 梯度会自然累加
    ids_i, am_i = ids_all[i * 2:(i + 1) * 2], am_all[i * 2:(i + 1) * 2]
    (fwd_fn(model_b, ids_i, am_i).pow(2).mean() / 4).backward()
    # @@end
grad_acc = [p.grad.clone() for p in model_b.parameters() if p.grad is not None]

# @@todo 比较两组梯度：最大绝对差、梯度量级（取全组绝对值最大值）、相对差
max_abs = max((a - b).abs().max().item() for a, b in zip(grad_full, grad_acc))
scale = max(g.abs().max().item() for g in grad_full)
rel = max_abs / scale
# @@end
assert rel < 1e-5, rel
print(f"张量组数 {len(grad_full)} | 最大绝对差 {max_abs:.3e} | 梯度量级 {scale:.4f} "
      f"| 相对差 {rel:.3e}")
'''

# =========================================================================== #
# 任务 11：梯度累积 —— 省多少激活                                             #
# =========================================================================== #

T11 = '''torch.manual_seed(SEED)
_, m_full = make_model("mid", dropout=0.0)
m_full.train()
act_full = activation_bytes(m_full, ids_all, am_all) / MB

torch.manual_seed(SEED)
_, m_acc = make_model("mid", dropout=0.0)
m_acc.train()
# @@todo 用 micro_batches() 逐份量出每份微批的激活 MB，收进列表，再取最大值（= 累积时的真实峰值）
acts = [activation_bytes(m_acc, bi, ba) / MB for bi, ba in micro_batches(ids_all, am_all, 2)]
peak = max(acts)
# @@end
print(f"整批 B=8 激活 {act_full:.2f} MB | 累积 B=2×4 峰值 {peak:.2f} MB "
      f"→ 降到 {peak / act_full * 100:.1f}%")
'''

# =========================================================================== #
# 任务 12：冻结策略的收益                                                     #
# =========================================================================== #

T12 = '''freeze_rows = []
for tag, freeze_emb, n_layer in FREEZE_PLAN:
    # @@todo 按策略冻结参数，量出「可训练参数量 / 梯度 MB / 激活 MB」
    _, fm = make_model("mid")
    freeze_params(fm, freeze_emb, n_layer)
    fm.train()
    ids12, am12 = make_batch("mid", 2, 32)
    act_f = activation_bytes(fm, ids12, am12) / MB
    grad_f = tensor_bytes(p.grad for p in fm.parameters() if p.grad is not None) / MB
    freeze_rows.append({"冻结策略": tag,
                        "可训练": sum(p.numel() for p in fm.parameters() if p.requires_grad),
                        "梯度MB": round(grad_f, 2), "激活MB": round(act_f, 2)})
    # @@end
    del fm
    gc.collect()
print(pd.DataFrame(freeze_rows).to_string(index=False))
'''

# =========================================================================== #
# 任务 13：精度换内存                                                         #
# =========================================================================== #

T13 = '''dtype_rows = []
for dtype, tag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16"), (torch.float16, "fp16")]:
    # @@todo 把 mid 档模型拷一份转成该 dtype，量出参数内存 MB
    m13 = deepcopy(m_full).to(dtype)
    b13 = tensor_bytes(m13.parameters()) / MB
    dtype_rows.append({"精度": tag, "参数MB": round(b13, 2)})
    # @@end
    del m13
gc.collect()
print(pd.DataFrame(dtype_rows).to_string(index=False))
'''

# =========================================================================== #
# 讲解版 / 练习版                                                             #
# =========================================================================== #

LESSON = [
    md(
        """# ch09 小模型训练与规模判定

## 这一章要解决什么

前八章我们学会了**怎么搭** Transformer、**怎么训**、**怎么微调**。
但真到了赛场上（或工位上），还有一个问题会先拦住你：

> 「领导说训一个 BERT-base，我这台 16GB 的笔记本，训得动吗？」

新手的第一反应是**跑一下试试** —— 然后：
- 要么显存爆了（`CUDA out of memory`），白等十分钟；
- 要么没爆，但一个 epoch 要三小时，交卷时间到了还没跑完。

老手的做法是**先算**：算参数量、算内存、算耗时。
算完就知道"能不能"，以及"如果不能，砍哪里"。

本章把这条**先算后跑**的判断链，做成一组可复现的实验。

## 三个核心公式（先记住，后面逐个验证）

| # | 公式 | 本章验证方式 |
|---|---|---|
| ① | 参数量 `P = d(V + max_pos + type_vocab + 2) + L(4d² + 2fd + 9d + f) + (d² + d)` | 与实测对比，**相对误差 0.0000%** |
| ② | 训练内存 `= 16 B/参数 + 激活` | 逐张量精确统计（fp32 权重 4 + 梯度 4 + Adam 状态 8） |
| ③ | 激活 `∝ batch（严格线性）× g(seq)（超线性）` | 6 种 (batch, seq) 组合实测 |

> **离线前提**：全程 `BertConfig` + 随机初始化。不下载任何预训练权重 ——
> 本章要的是**结构规模**，不是精度，所以随机权重完全够用。
"""
    ),
    code(IMPORTS),
    md(
        """## 脚手架：口径的唯一真相

下面的工具函数是本章全部断言的**唯一口径**。三种量内存的手法要分清：

| 手法 | 精度 | 什么时候用 |
|---|---|---|
| `tensor_bytes()` 按 `numel × element_size` | **精确** | 参数 / 梯度 / 优化器状态 |
| `activation_bytes()` 抓 saved tensors | **精确** | 激活内存 |
| `RSS` / `ru_maxrss` | 噪声极大 | 只看量级 —— 小模型上能偏几十倍 |

第三种为什么不能用？因为 Python 有内存池：分配过的 arena 不会立刻归还给 OS，
`del model` 之后 RSS 可能纹丝不动。本章一次都不用 RSS。
"""
    ),
    code(SCAFFOLD),
    md("## 任务 1：参数量恒等式\n\n先手算 tiny 档。恒等式拆成三块：embeddings / 每层 Encoder / pooler。"),
    code(T1),
    md(
        """## 任务 2：五档规模与组件占比

把 tiny → large 五档全算一遍，顺便看看**参数都花在哪**。

这一步不需要实例化任何模型 —— 纯算术。记住这个感觉：
**判断规模只需要 config，不需要硬件。**
"""
    ),
    code(T2),
    md(
        """**读表重点**：随着模型变大，`Layer%` 从 58% 涨到 **92.9%**，而 `Emb%` 从 39.6% 掉到 **6.8%**。

这条趋势有个很实用的推论：

> **小模型加词表很贵，大模型加词表便宜。**

`large` 档加一万个词，只多 1024 万参数（占 3%）；`tiny` 档加一万个词，
参数量直接翻 4 倍。所以在小模型上做词表扩展（比如加电力专业词表），
必须盯着参数量；在大模型上基本可以忽略。
"""
    ),
    md("## 任务 3：恒等式 vs 实测\n\n现在把 tiny / small / mid 真的造出来，数一遍 `sum(p.numel())`。"),
    code(T3),
    md(
        """**相对误差 0.0000%** —— 恒等式与实测**完全一致**。

这不是巧合，也不是近似公式碰运气：BERT 的每个参数张量都是确定的矩阵/向量，
恒等式只是把它们一条条列了出来。**唯一的风险是漏项**（比如忘了 LayerNorm 的 bias、
忘了 pooler），而"实测对账"就是防漏项的手段。

> 工程含义：**你可以对着配置文件算规模，而不用先把模型跑起来。**
> 这在评估"能不能上"的时候价值极大 —— 尤其是在别人机器上、或者卡还没批下来的时候。
"""
    ),
    md(
        """## 任务 4：内存四件套

训练态内存不是"参数量 × 4 字节"那么简单。fp32 + AdamW 训练，有**四份**东西常驻：

| 项 | 字节/参数 | 说明 |
|---|---|---|
| 参数 | 4 | fp32 权重 |
| 梯度 | 4 | 与参数同形 |
| 优化器状态 | 8 | Adam 的 `exp_avg` + `exp_avg_sq`，各 4 |
| **小计** | **16** | 与 batch / seq 无关 |
| 激活 | 随 B×S 变 | 反向传播要用的中间张量 |

前三是固定的 16 B/参数，第四项才是"变量"。分开量清楚，才知道爆显存该砍谁。
"""
    ),
    code(T4),
    md(
        """**读表重点**：三项固定开销加起来约 **16 B/参数**（表里 `B/参数` 列在 19~22 之间，
多出来的 3~6 B 就是激活摊到每个参数上）。

还可以注意到：从 tiny 到 mid，「合计」的 `B/参数` 在**收敛**（22.3 → 20.4 → 19.9）。
原因是激活的增长比参数慢 —— 模型越大，那 16 B 的固定开销越占主导。

**所以"大模型省内存"是错觉，但"大模型的显存利用率更高"是真的。**
"""
    ),
    md("## 任务 5：优化器状态精确账\n\n把 8 B/参数这一项单独拎出来对账。"),
    code(T5),
    md(
        """**这一节有个比公式更有价值的发现**：优化器状态**并不等于 `2P × 4`**。

实测比理论少了 **65,792 字节**，正好等于 pooler 的参数（256² + 256）。
原因是：我们的 loss 只用了 `last_hidden_state`，而 `pooler` 的输出（`pooler_output`）
**没有参与计算** —— 所以 pooler 拿不到梯度，AdamW 自然不为它建 `exp_avg` / `exp_avg_sq`。

两条教训：

1. **「参数量 × 16」是上界，不是精确值。** 真实占用取决于**哪些参数真的参与了这次前向**。
   训练脚本里如果有「挂了但没用」的模块（不参与 loss 的分支、冻结层、备用头），
   它们的优化器状态就是省下来的 —— 这也是「多任务模型只训一个头」能省内存的原因。
2. **`AdamW` 的 state 里除了 `exp_avg` / `exp_avg_sq`，还有 `step`** ——
   上面的账里它体现为 `len(opt.state) × 4` 字节。参数张量越多，这部分越不可忽略。

> 还有个易被忽略的点：`AdamW` 是**惰性**的 —— 构造 `optimizer` 对象时一个字节都不分配，
> 直到第一次 `step()` 才建 state。所以「优化器占多少内存」不 `step()` 一次是量不出来的。
> 如果第一次 `step()` 就爆，那就是优化器状态压垮的 ——
> 这正是 Adam 在大模型上很贵的原因（它一个人吃掉训练内存的一半）；
> `SGD` 无状态、`Adafactor` 分块统计，都能省下这一块。
"""
    ),
    md(
        """## 任务 6：激活的缩放律

激活是唯一随输入变化的项。固定模型（mid 档），扫 6 组 `(batch, seq)`，看它怎么涨。
"""
    ),
    code(T6),
    md(
        """**这里有个反直觉的地方**：看 `每token B` 那一列 —— 它不是常数！

| 观察 | 数据 |
|---|---|
| **batch 方向严格线性** | 固定 `seq=32`：B=1→2→4→8，激活 22.28→26.29→34.30→50.32，**每加 1 个 batch 恰好 +4.005 MB** |
| **seq 方向超线性** | 固定 `batch=2`：seq=32→64→128，每 batch 的激活 4.01→9.13→22.77 MB |
| **同 B×S 不等价** | `(4,32)` 得 34.30，`(2,64)` 得 36.54 —— 都不是"每 token 固定" |

原因：激活里有两部分 ——
**线性项**（每层的中间张量，∝ `B × S × d × L`）和
**二次项**（attention 的 `S × S` 注意力矩阵，∝ `B × A × S²`）。

拟合出来是 `g(S) ≈ 0.1077·S + 0.000547·S²` —— 二次项系数虽然小，但 `S²` 涨得快，
序列一长就翻脸。

> **工程含义**：显存不够时，**降 batch 和降 seq 不是等价的操作**。
> 在长序列场景（比如 512、1024）下，砍 seq 比砍 batch 划算得多。
"""
    ),
    md("## 任务 7：单步耗时标定\n\n内存算完了，还要算时间 —— 内存装得下不代表跑得完。"),
    code(T7),
    md(
        """**读表重点**：`倍率` 从不到 3× 一路涨到 4× 以上。

> ⚠️ **耗时数字每次跑都会抖**（CPU 调度、缓存命中、JIT 预热都会影响），
> 上表的具体毫秒值在你自己机器上会不一样 —— 但**趋势是稳定的**：
> 规模越大，倍率越高。别把某一个具体毫秒数当基准，要看**比例关系**。

为什么"整步/前向"的倍率会随规模上升？两个原因：

1. **反向本身约为前向的 2 倍**（要同时算对输入的梯度和对权重的梯度）；
2. **`Adam` 更新是逐参数的**，参数量越大，这一步的绝对开销越明显。

> 所以 benchmark 里报"前向 25 ms"是没有意义的 ——
> **训练看的是整步**。base 档前向看着只要几十毫秒，但整步要到百毫秒量级，
> 一个 epoch（按 1 万条、batch=8 算）就是十分钟起步。
"""
    ),
    md(
        """## 任务 8：base 档 —— 算得出来，跑得动吗

现在做个真实判断：BERT-base（中文词表）到底什么水平。
**注意：这里只测前向，不建优化器** —— 因为我们要先确认内存账。
"""
    ),
    code(T8),
    md(
        """**base 档训练态要 1.52 GB** —— 单看这个数字，16GB 机器完全没问题。

但真正的坑在**耗时**：实测 `batch=1, seq=32` 前向约 **27 ms**，
`batch=8, seq=128` 前向约 **330 ms**。整步再乘 4 倍量级 ——
按 `batch=8` 算，1 万条训练集一个 epoch 就是 `10000 / 8 × 1.3 秒 ≈ 27 分钟`，
10 个 epoch 是 **4.5 小时**。

> **这就是"先算后跑"的价值**：内存账 1.52 GB 看着很安全，
> 但时间账一算就知道 —— 在 CPU 上微调 BERT-base 是不现实的，
> 必须砍规模（用 `tiny`/`small`）或改用 `LoRA` 之类只训一小部分参数的方法（见 ch07）。
>
> 顺带记一个设备事实：本机是 Apple M2 / 16GB / **无 CUDA**（MPS 可用）。
> 有 CUDA 的话，同样的前向会快 1~2 个数量级，上面的结论就要重算 ——
> **规模判定的结论永远绑定具体硬件。**
"""
    ),
    md(
        """## 任务 9：规模判定 —— 8 GB 能训多大

把上面的账做成一张**决策表**：给定显存预算，反查哪一档可行。
"""
    ),
    code(T9),
    md(
        """**读表重点**：`large`（325.5 M 参数）需要 **4.85 GB** ——
8 GB 预算下"装得下"打的是"是"。但这个"是"要打折看：

1. 这只是**权重 + 梯度 + 优化器状态**，还没算激活（∝ batch × seq）；
2. 还没算 CUDA 上下文 / 框架自身的固定开销（真实环境约 0.5 ~ 1 GB）；
3. 还没算显存碎片。

> **实用判据**：把 `16 B/参数 × 1.3` 当作"安全线"（乘 1.3 预留激活和碎片）。
> 8 GB 显存的全参训练上限约 `8 × 0.77 / 16 B ≈ 3.85 亿参数`。
>
> 这条经验值可以直接背下来：**参数量（亿）× 1.6 ≈ 全参训练所需显存（GB）**。
> 它是"能不能上"的第一道筛子 —— 先算一次，不行就早点换方案，
> 别等 `CUDA out of memory` 才回头。
>
> ⚠️ 另外注意：**本章全部数字都是"能装下"的下界**。
> 真实训练里激活可能比权重还大（长序列 + 大 batch），
> 所以这张表只用来做**快速排除**，不用来做"一定能跑"的保证。
"""
    ),
    md(
        """## 任务 10：梯度累积 —— 梯度等价吗

内存不够时的标准解法是**梯度累积**：把大 batch 切成几份小 batch，
每份算完梯度**不更新**，累加若干份再更新一次。

但它成立的前提是「小批累加的梯度 == 大批一次的梯度」。这件事**必须验证**，
因为它取决于 loss 的归一化口径 —— 用 `mean` 且忘记除以累积步数，梯度就会差 N 倍。
"""
    ),
    code(T10),
    md(
        """**相对差 3.19e-07** —— 数值上完全等价（差异来自浮点累加顺序）。

这个实验里有个**必须做对**的细节：`loss / 4`。

`nn.MSELoss(reduction="mean")` 算的是这一批的均值。如果我们把 8 条切成 4 份、
每份算自己的均值再累加，等价于**每份权重都是 1/4** —— 所以要先除 4。
少这一步，梯度就会整体大 4 倍，训练直接跑飞。

> **通用规则**：`loss = criterion(...) / accumulation_steps`，
> 且 `criterion` 用 `mean` 口径。用 `sum` 口径时要额外除以总样本数。
"""
    ),
    md("## 任务 11：梯度累积 —— 省多少激活\n\n等价性验证过了，现在看它到底省多少。"),
    code(T11),
    md(
        """**激活降到 57.2%**（42.62 → 24.36 MB）。

省的不是按比例——`B=8` 切成 `8/2=4` 份，理论上激活该降到 1/4，实际只降到 57.2%。
差值来自那部分**与 batch 无关的固定激活**（比如 embedding 的输出、mask 相关张量）。
批量越小，固定部分占比越高，**累积的收益就越差**。

> **实用判据**：梯度累积的省内存收益是**递减**的。
> 从 `B=64` 累到 `B=8×8` 能省一大半；从 `B=4` 累到 `B=1×4` 几乎白忙。
> 另外耗时是 **1.68×** —— 省内存从来不是免费的。
"""
    ),
    md(
        """## 任务 12：冻结策略的收益

另一条省内存路线：**不训**一部分参数。但"冻结"省的是哪一份内存，很多人会答错。
"""
    ),
    code(T12),
    md(
        """**这张表最能说明问题** —— 对比两组：

| 策略 | 可训练参数 | 梯度 MB | 激活 MB |
|---|---|---|---|
| 完全不冻结 | 6,918,912 | 26.14 | 26.29 |
| 只冻结 emb | **4,804,352**（−30.6%） | **18.08**（−30.8%） | 25.41（−3.3%） |
| emb + 前 2 层 | 3,224,832（−53.4%） | 12.05（−53.9%） | **16.77**（−36.2%） |
| emb + 前 4 层 | 1,645,312（−76.2%） | 6.03（−76.9%） | **8.14**（−69.0%） |

两条结论：

1. **冻结 embedding 基本只省梯度，不省激活**（激活只降 3.3%）。
   因为 `emb` 的输入是整数 id，冻结它并不会让后续层的中间激活消失 ——
   数据还是要一层层往前传。
2. **冻结 Encoder 层才同时省激活**（冻前 4 层后激活降到 **31.0%**）。
   因为该层的中间张量不再需要为反向保存。

> **工程含义**：只想省显存 → 冻层；只想缩小可训练参数规模（防过拟合）→ 冻 emb 就够。
> 这两个目的对应**不同的手段**，别搞混。

> **一个口径提醒**：本章 T11 用了 `dropout=0.0`，T12 用的是默认 `dropout=0.1`，
> 所以两节的激活基数不一样（42.62 vs 26.29 MB），**不要跨任务直接比绝对值**。
> 为什么 T11 要关 dropout？因为梯度等价性验证要求"同一批数据、两次前向得到同一个结果"，
> **dropout 的随机 mask 会破坏这个前提**。这也是复现实验时的通用原则：
> 要验证"数值是否一致"，先把所有随机源（dropout、数据打乱）钉死。
"""
    ),
    md("## 任务 13：精度换内存\n\n最后一招：把 fp32 换成半精度。"),
    code(T13),
    md(
        """**参数内存直接减半**（26.4 → 13.2 MB）。

但"混合精度"到底省多少，要分清两种做法：

| 做法 | 权重 | 梯度 | 优化器状态 | 合计 | 备注 |
|---|---|---|---|---|---|
| 纯 fp32 | 4 | 4 | 8 | **16 B/参数** | 本章全程 |
| bf16 权重 + fp32 主权重 + fp32 Adam | 2 + 4 | 2 | 8 | **16 B/参数** | AMP 的典型配置，**权重内存其实没省** |
| 纯 bf16（无主权重副本） | 2 | 2 | 4（bf16 的 m/v） | **8 B/参数** | 省一半，但数值稳定性差 |

**AMP 省的主要是激活**（前向用半精度算，中间张量小一半），
权重那部分因为要保留 fp32 主副本，往往省不下来。

> 顺带一个**设备事实**：本机是 Apple M2，`torch.backends.mps` 可用但**无 CUDA**。
> CPU 上 `bf16` 前向比 `fp32` 慢（没有对应的硬件指令），
> **能省内存 ≠ 能提速** —— 这两个目标在 CPU 上是矛盾的。
"""
    ),
    md(
        """## 自查清单

- [ ] 能默写参数量恒等式的三块（embeddings / 每层 Encoder / pooler）
- [ ] 知道恒等式与实测相对误差是 **0.0000%**，且原因不是"运气好"
- [ ] 能说出 fp32 + AdamW 的 **16 B/参数**由哪三项构成（4 + 4 + 8）
- [ ] 知道 `AdamW` 的 state 是**惰性创建**的，不 `step()` 量不到
- [ ] 能解释为什么激活在 batch 方向线性、seq 方向超线性
- [ ] 知道 `(4,32)` 与 `(2,64)` 的激活**不相等**，并说得出原因
- [ ] 知道反向 ≈ 2× 前向，且倍率随规模上升到 4×（Adam 更新开销）
- [ ] 梯度累积必须 `loss / accumulation_steps`，且 `mean` 口径
- [ ] 能分辨"冻结 emb"与"冻结层"省的东西**不一样**
- [ ] 知道 AMP 省的主要是激活，不是权重
- [ ] 背下经验式：**参数量（亿）× 1.6 ≈ 全参训练显存（GB）**
- [ ] 接受本章不用 RSS 量内存的做法，并说得出原因（内存池复用）

## 本章一句话

> **规模判定是算术题，不是试错题。**
> 参数量用恒等式、内存用 16 B/参数 + 激活、耗时用前向 × 4 ——
> 三张账算完，"能不能训"在写第一行训练代码之前就已经有答案了。
"""
    ),
]

EXERCISE = [
    md(
        """# ch09 小模型训练与规模判定（练习）

> 把 `____` 换成正确的代码。每道题下方的 `assert` 会立刻告诉你对不对。

**本章目标**：学会在动手训练之前，先把「参数量 → 内存 → 耗时」三张账算清楚。

## 三个核心公式

| # | 公式 | 验证标准 |
|---|---|---|
| ① | 参数量恒等式（embeddings + 每层 Encoder + pooler） | 与实测对比，**相对误差 0.0000%** |
| ② | 训练内存 `= 16 B/参数 + 激活` | 逐张量精确统计 |
| ③ | 激活 `∝ batch（线性）× g(seq)（超线性）` | 6 组 (batch, seq) 实测 |

## 量内存的正确姿势

| 手法 | 精度 | 用途 |
|---|---|---|
| `tensor_bytes()` | **精确** | 参数 / 梯度 / 优化器状态 |
| `activation_bytes()` | **精确** | 激活 |
| `RSS` | 噪声极大 | **本章一次都不用**（内存池复用会让它偏几十倍） |
"""
    ),
    code(IMPORTS),
    md("## 脚手架（已给好，直接调用）"),
    code(SCAFFOLD),
    md("## 任务 1：参数量恒等式\n\n用恒等式算出 tiny 档参数量。"),
    code(T1),
    md("## 任务 2：五档规模与组件占比\n\n算全部五档，并拆出 embeddings / Encoder / pooler 的占比。"),
    code(T2),
    md("## 任务 3：恒等式 vs 实测\n\n实例化 tiny / small / mid，数一遍 `sum(p.numel())` 对账。"),
    code(T3),
    md("## 任务 4：内存四件套\n\n激活 / 参数 / 梯度 / 优化器状态，四项分开量。注意**激活必须最先量**（它内部要做一次 backward）。"),
    code(T4),
    md("## 任务 5：优化器状态精确账\n\n验证它是否恰好等于 `2P × 4` 字节。"),
    code(T5),
    md("## 任务 6：激活的缩放律\n\n扫 6 组 `(batch, seq)`，看激活怎么随二者变化。"),
    code(T6),
    md("## 任务 7：单步耗时标定\n\n分别测纯前向与整步，算倍率。"),
    code(T7),
    md("## 任务 8：base 档 —— 算得出来，跑得动吗\n\n先算训练态内存，再测前向耗时。"),
    code(T8),
    md("## 任务 9：规模判定 —— 8 GB 能训多大\n\n给定预算反查可行档位。"),
    code(T9),
    md(
        """## 任务 10：梯度累积 —— 梯度等价吗

把 `B=8` 切成 4 份 `B=2` 累加，验证梯度是否与整批一致。

> ⚠️ **关键**：每份的 loss 要除以累积步数，否则梯度会差 4 倍。
"""
    ),
    code(T10),
    md("## 任务 11：梯度累积 —— 省多少激活\n\n量出累积时的激活峰值，与整批对比。"),
    code(T11),
    md("## 任务 12：冻结策略的收益\n\n对比「冻结 emb」与「冻结层」分别省了哪一份内存。"),
    code(T12),
    md("## 任务 13：精度换内存\n\n把模型转成 bf16 / fp16，看参数内存变化。"),
    code(T13),
    md(
        """## 自查清单

- [ ] 参数量恒等式能默写（embeddings / 每层 Encoder / pooler 三块）
- [ ] 知道 16 B/参数 = 权重 4 + 梯度 4 + Adam 状态 8
- [ ] 知道 `AdamW` 的 state 惰性创建，不 `step()` 量不到
- [ ] 能解释激活在 batch 方向线性、seq 方向超线性
- [ ] 知道梯度累积要 `loss / accumulation_steps`
- [ ] 能分辨「冻结 emb」和「冻结层」省的**不是同一份**内存
- [ ] 背下经验式：**参数量（亿）× 1.6 ≈ 全参训练显存（GB）**
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
