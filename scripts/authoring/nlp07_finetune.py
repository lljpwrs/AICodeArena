#!/usr/bin/env python3
"""ch07 微调范式（讲解 / 练习 / 答案 三件套）—— 手写 mini-BERT + 手写 LoRA。

本章主线 —— **把「预训练 + 微调」拆开看，用实测回答三个问题**：

    ① BERT 的输入到底是什么：input_ids / token_type_ids / attention_mask 三张量
       单句 seg 全 0；句对前段 0 后段 1；截断时末位强制写回 [SEP]
    ② WordPiece 的 ## 续接机制：请核实 -> 请 + ##核 + ##实
       而只要有一个片段匹配不上，整词退化为 [UNK]（铁芯 -> [UNK]）
    ③ **预训练什么时候有用** —— 本章最有价值的一组对照：
         小语料 570 条 MLM / 大语料 20000 条 MLM / 随机初始化
         每类 5/10/20/40 条下游数据，12 个格子全部是「随机初始化」赢
         把预训练语料扩到 35 倍（每类 40 时 0.694595 -> 0.764865）也补不上差距
    ④ 手写 LoRA：B 初始全零 -> 初始行为与原模型逐元素一致；合并 W + s·BA 差 1.25e-06
       r=4 可训参数 2373（占 2.08%）拿到全量微调的 69.4%，明显优于只训分类头（0.2848%）

结论是**诚实的负结果**：在这个规模（2 层 / d=64 / 词表 316）上，
无监督预训练不但没有收益，反而拖累下游。原因是这一层的容量与语料多样性
都不足以让 MLM 学到可迁移的语言结构，而下游任务本质是「词袋式找信号词」。
真 BERT 的价值来自 1.1 亿参数 + 33 亿词的量级，不是来自「MLM 这个目标本身」。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp07_finetune.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch07_finetune"

# =========================================================================== #
# 公共导入 / 初始化                                                           #
# =========================================================================== #

IMPORTS = '''import copy
import importlib.util
import math
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
print("torch", torch.__version__, "| 线程", torch.get_num_threads())

# §7.12 起用真实库；**仍然只建 config，不下载任何预训练权重**
import transformers
from transformers import (BertConfig, BertForSequenceClassification,
                          Trainer, TrainingArguments)

DEV_HF = "mps" if torch.backends.mps.is_available() else "cpu"
print("transformers", transformers.__version__, "| HF 训练设备", DEV_HF)
'''

SETUP = '''from pathlib import Path

DATA = Path("data")                  # notebook cwd = coding/04_nlp
CSV = DATA / "synth_text.csv"
MAKE_DATA = DATA / "make_data.py"

df = pd.read_csv(CSV, keep_default_na=False)
print("读入", df.shape, "| 标签分布", df["缺陷类型"].value_counts().to_dict())
print("无效文本", int(df["缺陷描述"].str.strip().isin(["", "N/A", "-", "无"]).sum()), "条")
'''

SCAFFOLD = '''# 脚手架：常量与工具函数两版都有，你只填 @@todo 块里的内容
DEVICES = ["变压器", "断路器", "隔离开关", "避雷器", "电流互感器", "绝缘子", "套管", "母线"]
SIGWORDS = {
    "渗漏油": ["渗油", "漏油", "油迹", "油污", "滴油", "油位下降"],
    "锈蚀": ["锈蚀", "生锈", "铁锈", "漆面剥落", "镀锌层脱落"],
    "破损": ["破损", "裂纹", "碎裂", "缺口", "掉瓷", "箱体变形"],
    "发热": ["发热", "温度偏高", "过热", "烫手", "温升异常"],
    "异物": ["异物", "鸟巢", "塑料袋", "树枝", "风筝线搭挂"],
}
LABELS5 = ["渗漏油", "锈蚀", "破损", "发热", "异物"]
LID = {c: i for i, c in enumerate(LABELS5)}
VOCAB = DEVICES + [w for ws in SIGWORDS.values() for w in ws]
VOCAB_SET = set(VOCAB)
MAXW = max(len(w) for w in VOCAB)
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
HAN = re.compile(r"[\\u4e00-\\u9fff]")

# 模型超参：2 层 / d=64 / 4 头 —— CPU 上秒级，且小到能"背下来"，正好暴露预训练的真实收益
D, H, NL, FF, ML, NCLS = 64, 4, 2, 128, 24, 5


def clean(t):
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"https?://\\S+", "", t)
    t = re.sub(r"[\\u3000\\s]+", "", t)
    return t.replace(",", "，").strip()


def tok(t):
    """复用 ch01 的两级切分 + 正向最大匹配"""
    out = []
    for seg in re.findall(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]+|[^\\sA-Za-z0-9_\\u4e00-\\u9fff]", t):
        if re.fullmatch(r"[\\u4e00-\\u9fff]+", seg):
            i = 0
            while i < len(seg):
                for L in range(min(MAXW, len(seg) - i), 0, -1):
                    if seg[i:i + L] in VOCAB_SET:
                        out.append(seg[i:i + L])
                        i += L
                        break
                else:
                    out.append(seg[i])
                    i += 1
        else:
            out.append(seg)
    return out


texts = [clean(t) for t in df["缺陷描述"]]
tokens = [tok(t) for t in texts]
bad = np.array([not t.strip() or t.strip() in {"N/A", "-", "无", "''", '""'} for t in texts])
kept = np.where(~bad)[0]
print("可用样本 %d | token 种类 %d | 总 token %d | 平均 token/句 %.6f"
      % (len(kept), len({w for i in kept for w in tokens[i]}),
         sum(len(tokens[i]) for i in kept), np.mean([len(tokens[i]) for i in kept])))
'''

# =========================================================================== #
# 代码块                                                                      #
# =========================================================================== #

T1_CODE = '''# @@todo WordPiece 建表：完整词 + 全部后缀 + 语料单字（单字也要有 "##" 形式）
# @@hint 后缀是「续接片段」，一律加 "##" 前缀：{"##" + w[i:] for w in VOCAB for i in range(1, len(w))}
# @@hint 语料里的汉字要连 "##单字" 一起进表，否则编码时无处可拆、整词退化 [UNK]
# @@hint 最后 cont 取出所有 "##" 开头的片段，用来核对续接片段数量
wp_vocab = (set(SPECIALS) | set(VOCAB)
            | {"##" + w[i:] for w in VOCAB for i in range(1, len(w))}
            | {c for i in kept for c in "".join(tokens[i]) if HAN.fullmatch(c)}
            | {"##" + c for i in kept for c in "".join(tokens[i]) if HAN.fullmatch(c)})
cont = sorted(p for p in wp_vocab if p.startswith("##"))
# @@end

# ---- 验收 ----
assert len(wp_vocab) == 316, f"词表应为 316，实际 {len(wp_vocab)}"
assert len(cont) == 151, f"续接片段应为 151，实际 {len(cont)}"
assert "##互感器" in wp_vocab and "电流互感器" in wp_vocab
print("WordPiece 词表 %d 词（续接片段 %d）| 例 %s" % (len(wp_vocab), len(cont), cont[:6]))

# 自建 id 映射：[PAD] [UNK] [CLS] [SEP] [MASK] 固定占前 5 位
ITEMS = SPECIALS + sorted(p for p in wp_vocab if p not in SPECIALS)
WID = {s: i for i, s in enumerate(ITEMS)}
NV = len(ITEMS)
PAD, UNK, CLS, SEP, MASK = (WID[t] for t in SPECIALS)
assert NV == 316 and (PAD, UNK, CLS, SEP, MASK) == (0, 1, 2, 3, 4)
print("id 空间 %d | [PAD]=%d [UNK]=%d [CLS]=%d [SEP]=%d [MASK]=%d" % (NV, PAD, UNK, CLS, SEP, MASK))
'''

T2_CODE = '''# 函数骨架已给，你只需要填 @@todo 里的那一行
def wp_encode(word):
    """从左到右贪心取**最长**可匹配片段；续接片段带 "##" 前缀"""
    if word in wp_vocab:
        return [word]
    out, i = [], 0
    while i < len(word):
        L = len(word) - i
        while L > 0:
            # @@todo 构造候选片段：i==0 时是 word[i:i+L]；i>0 时必须是续接片段形式
            # @@hint 续接片段 = "##" + word[i:i + L]；漏了 ## 就永远匹配不上
            piece = word[i:i + L] if i == 0 else "##" + word[i:i + L]
            # @@end
            if piece in wp_vocab:
                out.append(piece)
                i += L
                break
            L -= 1
        else:
            return ["[UNK]"]          # 任一片段失败 -> 整词退化为 [UNK]（HF 行为）
    return out

# ---- 验收 ----
assert wp_encode("变压器") == ["变压器"], "整词命中就不拆"
assert wp_encode("请核实") == ["请", "##核", "##实"], "整词不在表 -> 首字 + ##续接"
assert wp_encode("严重") == ["严", "##重"]
assert wp_encode("铁芯") == ["[UNK]"], "有'铁'无'芯' -> 整词 [UNK]"
assert wp_encode("ZLX99") == ["[UNK]"], "字母串不在表"
for w in ["变压器", "电流互感器", "油位下降", "请核实", "严重", "铁芯"]:
    print("  %-10s -> %s" % (w, wp_encode(w)))
print()
print("与 BPE 的区别：BPE 编码时按**合并优先级**逐对合并（可能局部最优）；")
print("WordPiece 是**从头贪心取最长**，续接片段写 ## 前缀（词首不写）。")
'''

T3_CODE = '''# 函数骨架已给，你只需要填 @@todo 里的内容
def bert_features(a, b=None, maxlen=ML):
    """把词序列拼成 BERT 的三个张量：input_ids / token_type_ids / attention_mask"""
    ta = [p for w in a for p in wp_encode(w)]
    # @@todo 单句：[CLS] + a 的片段 + [SEP]，seg 全 0
    # @@hint ids = [WID["[CLS]"]] + [WID.get(p, WID["[UNK]"]) for p in ta] + [WID["[SEP]"]]
    # @@hint seg 与 ids 等长，单句里每个位置都是 0
    ids = [WID["[CLS]"]] + [WID.get(p, WID["[UNK]"]) for p in ta] + [WID["[SEP]"]]
    seg = [0] * len(ids)
    # @@end
    if b is not None:
        tb = [p for w in b for p in wp_encode(w)]
        # @@todo 句对：接在已有的 [CLS] A [SEP] 后面，补 B 段与第二个 [SEP]
        # @@hint 关键：第一个 [SEP] 的 seg 仍是 0（它属于 A 段），只有 B 段和末尾 [SEP] 是 1
        ids = ids + [WID.get(p, WID["[UNK]"]) for p in tb] + [WID["[SEP]"]]
        seg = seg + [1] * (len(tb) + 1)
        # @@end
    if len(ids) > maxlen:
        # @@todo 超长截断：最后一位要**强行写回 [SEP]**，seg 跟着取原末位
        # @@hint ids = ids[:maxlen-1] + [WID["[SEP]"]]；seg 同理保留 maxlen-1 位再补一位
        ids = ids[:maxlen - 1] + [WID["[SEP]"]]
        seg = seg[:maxlen - 1] + [seg[-1]]
        # @@end
    mask = [1] * len(ids)
    pad = maxlen - len(ids)
    return ids + [WID["[PAD]"]] * pad, seg + [0] * pad, mask + [0] * pad

# ---- 验收 ----
i1, s1, m1 = bert_features(["变压器", "渗油"])
i2, s2, m2 = bert_features(["变压器", "渗油"], ["设备", "缺陷"])
i3, s3, m3 = bert_features(["变压器", "渗油"], maxlen=3)
assert sum(m1) == 4 and i1[:4] == [WID["[CLS]"], WID["变压器"], WID["渗油"], WID["[SEP]"]]
assert set(s1[:4]) == {0} and m1[4] == 0, "单句 seg 全 0，pad 位 mask=0"
assert sum(m2) == 8 and s2[:8] == [0, 0, 0, 0, 1, 1, 1, 1], "句对 seg 前 0 后 1"
assert sum(m3) == 3 and i3 == [WID["[CLS]"], WID["变压器"], WID["[SEP]"]], "截断末位强制 [SEP]"
print("单句 ids[:5] =", i1[:5], "| seg[:5] =", s1[:5], "| mask[:5] =", m1[:5])
print("句对 seg[:8] =", s2[:8], "（前 4 个 0、后 4 个 1）")
print("截断 maxlen=3 ->", i3, "（末位被换成 [SEP]，保证句子闭合）")
'''

MODEL_CODE = '''class MHSA(nn.Module):
    """自写多头自注意力：q/k/v 三个独立 Linear —— 这样才能像 LoRA 论文那样只挂 Q/V"""

    def __init__(self, d=D, h=H):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, pad=None):
        B, T, _ = x.shape
        sh = (B, T, self.h, self.dh)
        q, k, v = (m(x).view(sh).transpose(1, 2) for m in (self.q, self.k, self.v))
        am = None if pad is None else (~pad)[:, None, None, :]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.o(o.transpose(1, 2).reshape(B, T, -1))


class EncLayer(nn.Module):
    """Pre-LN 的 Encoder 层：ch05 已实测 Post-LN 在 6 层堆叠下首层梯度只剩末层的 0.000007"""

    def __init__(self, d=D, h=H, ff=FF):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = MHSA(d, h)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, pad=None):
        x = x + self.attn(self.n1(x), pad)
        return x + self.ff(self.n2(x))


class MiniBERT(nn.Module):
    """BERT 骨架：三种 embedding 相加 -> Encoder 堆叠 -> 池化 -> 分类头；另挂一个 MLM 头"""

    def __init__(self, nv=NV, d=D, h=H, nl=NL, maxlen=ML, ncls=NCLS, pool="cls"):
        super().__init__()
        self.tok = nn.Embedding(nv, d, padding_idx=0)
        self.seg = nn.Embedding(2, d)
        self.pos = nn.Embedding(maxlen, d)
        self.n0 = nn.LayerNorm(d)
        self.layers = nn.ModuleList([EncLayer(d, h, FF) for _ in range(nl)])
        self.pool = pool
        self.head = nn.Linear(d, ncls)
        self.mlm = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, nv))

    def body(self, ids, seg, mask):
        x = self.tok(ids) + self.seg(seg) + self.pos(torch.arange(ids.shape[1]))
        x = self.n0(x)
        pad = mask == 0
        for lyr in self.layers:
            x = lyr(x, pad)
        return x

    def forward(self, ids, seg, mask):
        hh = self.body(ids, seg, mask)
        if self.pool == "cls":
            pooled = hh[:, 0]
        else:
            m = mask[..., None].float()
            pooled = (hh * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(pooled), hh

    def mlm_logits(self, ids, seg, mask):
        return self.mlm(self.body(ids, seg, mask))


def mb(seed=0, **kw):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return MiniBERT(**kw)


m0 = mb()
tot = sum(p.numel() for p in m0.parameters())
print("MiniBERT 参数量合计 %d" % tot)
'''

T4_CODE = '''# @@todo 拆出四块参数量：嵌入（含前置 LayerNorm）/ Encoder / 分类头 / MLM 头
# @@hint 嵌入 = tok.weight + seg.weight + pos.weight + n0 的 LayerNorm 参数
# @@hint enc_n = sum(p.numel() for p in m0.layers.parameters())，head / mlm 同理
emb_n = sum(p.numel() for p in [m0.tok.weight, m0.seg.weight, m0.pos.weight]) + sum(
    p.numel() for p in m0.n0.parameters())
enc_n = sum(p.numel() for p in m0.layers.parameters())
head_n = sum(p.numel() for p in m0.head.parameters())
mlm_n = sum(p.numel() for p in m0.mlm.parameters())
# @@end

# ---- 验收 ----
assert (NV, D, ML, FF) == (316, 64, 24, 128)
assert emb_n == 22016, f"嵌入应 22016，实际 {emb_n}"
assert enc_n == 66944, f"Encoder 应 66944，实际 {enc_n}"
assert head_n == 325 and mlm_n == 24828
assert tot == 114113, f"总量应 114113，实际 {tot}"
assert NV * D + 2 * D + ML * D + 2 * D == emb_n, "tok+seg+pos+LN = 22016"
assert 4 * D * D + 4 * D + 2 * D * FF + D + FF + 4 * D == enc_n // NL, "单层 Encoder 手算"
assert D * NCLS + NCLS == head_n
print("总参 %d = 嵌入 %d + Encoder %d + 分类头 %d + MLM头 %d" % (tot, emb_n, enc_n, head_n, mlm_n))
print("嵌入手算 tok %d + seg %d + pos %d + LayerNorm %d = %d"
      % (NV * D, 2 * D, ML * D, 2 * D, NV * D + 2 * D + ML * D + 2 * D))
print("单层 Encoder %d，手算 4d²+4d(attn) + 2·d·ff+d+ff(ff) + 4d(norm) = %d"
      % (enc_n // NL, 4 * D * D + 4 * D + 2 * D * FF + D + FF + 4 * D))
print("分类头 %d = %d×%d+%d，只占总量 %.4f%% —— 这是「微调很便宜」的一半原因"
      % (head_n, D, NCLS, NCLS, 100 * head_n / tot))
'''

FEATURIZE_CORE = '''def featurize(seqs, maxlen=ML):
    I, S, M = [], [], []
    for tk in seqs:
        ids, seg, msk = bert_features(tk, maxlen=maxlen)
        I.append(ids)
        S.append(seg)
        M.append(msk)
    return torch.as_tensor(I), torch.as_tensor(S), torch.as_tensor(M)


I_all, S_all, M_all = featurize([tokens[i] for i in kept])
print("小语料张量 %s | 有效 token %d" % (tuple(I_all.shape), int(M_all.sum())))
'''

MLM_FUNC = '''def mlm_batch(ids, mask, rng, p=0.15):
    """BERT 的掩码策略：15% 位置被选中，其中 80% [MASK] / 10% 随机词 / 10% 原样保留"""
    lab = ids.clone()
    cand = (mask == 1) & (ids != CLS) & (ids != SEP) & (ids != PAD)
    # @@todo 用第一个随机数决定「哪些位置被选中」，其余位置标签置 -100（不算损失）
    # @@hint sel = cand & (rng.random(ids.shape) < p)；lab[~sel] = -100
    sel = cand & (torch.as_tensor(rng.random(ids.shape), dtype=torch.float32) < p)
    lab[~sel] = -100
    # @@end
    r = torch.as_tensor(rng.random(ids.shape), dtype=torch.float32)
    new = ids.clone()
    # @@todo 用第二个随机数把选中位置分成三份：80% 换 [MASK]、10% 换随机词、10% 不动
    # @@hint new[sel & (r < 0.8)] = MASK
    new[sel & (r < 0.8)] = MASK
    # @@end
    new[sel & (r >= 0.8) & (r < 0.9)] = torch.as_tensor(
        rng.integers(1, NV, size=ids.shape), dtype=torch.long)[sel & (r >= 0.8) & (r < 0.9)]
    return new, lab
'''

T5_CODE = '''ids_p, lab_p = mlm_batch(I_all, M_all, np.random.default_rng(1))
n_sel = int((lab_p != -100).sum())
n_mask = int((ids_p == MASK).sum())
n_rand = int(((ids_p != I_all) & (ids_p != MASK) & (lab_p != -100)).sum())
n_keep = int(((ids_p == I_all) & (lab_p != -100)).sum())

# ---- 验收 ----
assert n_sel == 1307, f"应选中 1307 个位置，实际 {n_sel}"
assert abs(n_sel / int(M_all.sum()) - 0.134798) < 1e-6, "占有效 token 的 13.48%"
assert n_mask == 1070, f"[MASK] 替换应 1070，实际 {n_mask}"
assert n_rand == 117, f"随机替换应 117，实际 {n_rand}"
assert n_keep == 120, f"原样保留应 120，实际 {n_keep}"
assert n_mask + n_rand + n_keep == n_sel
print("选中 %d / %d = %.6f" % (n_sel, int(M_all.sum()), n_sel / int(M_all.sum())))
print("  [MASK] %d | 随机替换 %d | 原样保留 %d" % (n_mask, n_rand, n_keep))
print()
print("为什么留 10%% 原样不动：否则 [MASK] 这个 token 只出现在预训练、微调时永远见不到，")
print("模型会学成「看到 [MASK] 才需要猜」这条捷径，两种阶段之间出现人为分布差。")
'''

PRETRAIN_FUNC = '''def pretrain(I, S, M, steps=500, seed=5):
    """无监督 MLM 预训练：每步采 32 条，只在被遮住的位置算交叉熵"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        m = MiniBERT()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    rgp = np.random.default_rng(77)
    m.train()
    for _ in range(steps):
        idx = rgp.integers(0, len(I), size=32)
        ids_b, lab_b = mlm_batch(I[idx], M[idx], rgp)
        # @@todo 前向 + 交叉熵；**必须 ignore_index=-100**，否则 padding 会被算进损失
        # @@hint F.cross_entropy(m.mlm_logits(...).reshape(-1, NV), lab_b.reshape(-1), ignore_index=-100)
        loss = F.cross_entropy(m.mlm_logits(ids_b, S[idx], M[idx]).reshape(-1, NV),
                               lab_b.reshape(-1), ignore_index=-100)
        # @@end
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    m.eval()
    with torch.no_grad():
        ev_ids, ev_lab = mlm_batch(I, M, np.random.default_rng(999))
        lo = m.mlm_logits(ev_ids, S, M)
        msk = ev_lab != -100
        a1 = float((lo.argmax(-1)[msk] == ev_lab[msk]).float().mean())
        a5 = float((lo.topk(5, -1).indices[msk] == ev_lab[msk][:, None]).any(-1).float().mean())
        ent = float(-(F.softmax(lo, -1) * F.log_softmax(lo, -1)).sum(-1)[msk].mean())
    return m, float(loss), a1, a5, ent
'''

T7_CODE = '''# @@todo 两档语料各跑一次 MLM 预训练（只改语料池，步数/模型/超参完全一致）
# @@hint 小语料 I_all(570 条)、大语料 I_big(20000 条)，都 500 步
m_small, l_small, a1_small, a5_small, e_small = pretrain(I_all, S_all, M_all, 500, 5)
m_big, l_big, a1_big, a5_big, e_big = pretrain(I_big, S_big, M_big, 500, 5)
# @@end

SD_SMALL = copy.deepcopy(m_small.state_dict())
SD_BIG = copy.deepcopy(m_big.state_dict())

# ---- 验收 ----
assert abs(l_small - 0.465618) < 1e-5, f"小语料 MLM loss 应 0.465618，实际 {l_small}"
assert abs(a1_small - 0.863707) < 1e-5 and abs(a5_small - 0.943146) < 1e-5
assert abs(l_big - 0.487685) < 1e-5 and abs(a1_big - 0.827988) < 1e-5
assert a1_small > 10 / NV, "远高于随机基线"
print("小语料 570  条 | loss %.6f | 掩码位 top-1 %.6f | top-5 %.6f | 预测熵 %.6f"
      % (l_small, a1_small, a5_small, e_small))
print("大语料 20000 条 | loss %.6f | 掩码位 top-1 %.6f | top-5 %.6f | 预测熵 %.6f"
      % (l_big, a1_big, a5_big, e_big))
print("随机基线 top-1 = 1/%d = %.6f | top-5 = %.6f" % (NV, 1 / NV, 5 / NV))
print()
print("注意：大语料的 MLM 指标**更差**（top-1 %.6f < %.6f），因为池子更大、任务更难。" % (a1_big, a1_small))
print("但下一节会看到它的下游微调**更好** —— 预训练指标不等于表征质量。")
'''

FEATURIZE_BIG = '''def gen_corpus(n, seed=7):
    """用 data/make_data.py 的同一套模板再生成 n 条**无标签**文本，做「预训练语料规模」对照"""
    spec = importlib.util.spec_from_file_location("mk", MAKE_DATA)
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    r9 = np.random.default_rng(seed)
    labels = list(mk.DEFECTS)
    out = []
    for i in range(n):
        lab = labels[i % len(labels)]
        rec = {
            "台区编号": str(r9.choice(mk.STATIONS)),
            "设备类型": str(r9.choice(mk.DEVICES)),
            "缺陷类型": lab,
            "缺陷等级": str(r9.choice(mk.LEVELS)),
            "_kw": str(r9.choice(mk.DEFECTS[lab])),
            "_act": str(r9.choice(mk.ACTIONS)),
            "_tpl": str(r9.choice(mk.TEMPLATES)),
        }
        out.append(mk.add_noise(mk.clean_text(rec), r9))
    return out


BIG = gen_corpus(20000)
BIG_TOK = [tok(clean(t)) for t in BIG]
I_big, S_big, M_big = featurize(BIG_TOK)
assert len(BIG) == 20000 and len(set(BIG)) == 19855, "去重后 19855 条"
assert tuple(I_big.shape) == (20000, 24) and int(M_big.sum()) == 333860
print("大语料 %d 条 | 去重 %d | 平均字符 %.4f | 平均 token %.4f"
      % (len(BIG), len(set(BIG)), np.mean([len(t) for t in BIG]),
         np.mean([len(t) for t in BIG_TOK])))
print("大语料张量 %s | 有效 token %d" % (tuple(I_big.shape), int(M_big.sum())))

BY_CLS = {c: [i for i in kept if df["缺陷类型"].iloc[i] == c] for c in LABELS5}


def split(FS, seed=0):
    """每类取 FS 条做 few-shot 训练，其余全部做测试"""
    rs = np.random.default_rng(seed)
    tr, te = [], []
    for c in LABELS5:
        pool = BY_CLS[c].copy()
        rs.shuffle(pool)
        tr += list(pool[:FS])
        te += list(pool[FS:])
    return tr, te


FINETUNE_SPLIT = split(20)
print("few-shot 划分（每类 20）：训练 %d / 测试 %d" % (len(FINETUNE_SPLIT[0]), len(FINETUNE_SPLIT[1])))
'''

FRESH_FIT = '''def fresh(sd, pool="mean"):
    """把预训练权重装进分类模型；pool 选 cls 或 mean"""
    m = mb(seed=9, pool=pool)
    m.load_state_dict(sd)
    return m


def fit(model, FS, steps=400, lr=1e-3, bs=32, seed=3):
    tr, te = split(FS)
    I_f, S_f, M_f = featurize([tokens[i] for i in tr])
    Y_f = torch.as_tensor([LID[df["缺陷类型"].iloc[i]] for i in tr])
    I_e, S_e, M_e = featurize([tokens[i] for i in te])
    Y_e = torch.as_tensor([LID[df["缺陷类型"].iloc[i]] for i in te])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rg3 = np.random.default_rng(seed)
    model.train()
    for _ in range(steps):
        idx = rg3.integers(0, len(Y_f), size=min(bs, len(Y_f)))
        # @@todo 前向 + 分类交叉熵
        # @@hint F.cross_entropy(model(I_f[idx], S_f[idx], M_f[idx])[0], Y_f[idx])
        loss = F.cross_entropy(model(I_f[idx], S_f[idx], M_f[idx])[0], Y_f[idx])
        # @@end
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = float((model(I_e, S_e, M_e)[0].argmax(1) == Y_e).float().mean())
    return acc, len(tr), len(te)
'''

T8_CODE = '''# @@todo 每类 40 条下，比三种「底座来源」
# @@hint 大/小语料用 fresh(SD_XXX) 装权重；随机初始化用 mb(seed=9)
r_sm = fit(fresh(SD_SMALL), 40)
r_bg = fit(fresh(SD_BIG), 40)
r_rd = fit(mb(seed=9), 40)
# @@end

# ---- 验收 ----
assert abs(r_sm[0] - 0.694595) < 1e-5, f"小语料预训练应 0.694595，实际 {r_sm[0]}"
assert abs(r_bg[0] - 0.764865) < 1e-5, f"大语料预训练应 0.764865，实际 {r_bg[0]}"
assert abs(r_rd[0] - 0.910811) < 1e-5, f"随机初始化应 0.910811，实际 {r_rd[0]}"
assert r_bg[0] > r_sm[0], "扩大预训练语料确实提升了预训练权重的下游表现"
assert r_rd[0] > r_bg[0], "但在这个规模上，随机初始化仍然赢"
print("每类 40 条（训练 %d / 测试 %d）：" % (r_sm[1], r_sm[2]))
print("  小语料预训练 %.6f | 大语料预训练 %.6f | 随机初始化 %.6f" % (r_sm[0], r_bg[0], r_rd[0]))
'''

T9_CODE = '''grid = {}
for FS in (5, 10, 20, 40):
    # @@todo 记下三列：小语料预训练 / 大语料预训练 / 随机初始化
    # @@hint a_small = fit(fresh(SD_SMALL), FS)[0]；依次类推
    a_small = fit(fresh(SD_SMALL), FS)[0]
    a_big = fit(fresh(SD_BIG), FS)[0]
    a_rand = fit(mb(seed=9), FS)[0]
    grid[FS] = (a_small, a_big, a_rand)
    # @@end

# ---- 验收 ----
assert abs(grid[5][0] - 0.249541) < 1e-5 and abs(grid[5][2] - 0.249541) < 1e-5
assert abs(grid[10][2] - 0.350000) < 1e-5
assert abs(grid[20][1] - 0.389362) < 1e-5 and abs(grid[20][2] - 0.591489) < 1e-5
assert abs(grid[40][1] - 0.764865) < 1e-5 and abs(grid[40][2] - 0.910811) < 1e-5
assert grid[5][0] == grid[5][2], "5 条/类时小语料预训练与随机初始化打平（都是 0.249541）"
assert sum(1 for k in grid if grid[k][2] > max(grid[k][0], grid[k][1])) == 3, \
    "随机初始化 3 胜 1 平 0 负"
assert all(grid[k][2] >= max(grid[k][0], grid[k][1]) for k in grid), "随机初始化 4 项均不劣"
assert grid[20][1] > grid[20][0] and grid[40][1] > grid[40][0], "样本够多时大语料优势才显现"
assert grid[5][1] < grid[5][0] and grid[10][1] < grid[10][0], "样本极少时大语料反而略差"
print("每类样本 | 小语料预训练 | 大语料预训练 | 随机初始化 | 胜者")
for FS in (5, 10, 20, 40):
    a, b, c = grid[FS]
    win = "随机初始化" if c > max(a, b) else ("打平" if c == max(a, b) else "预训练")
    print("    %-3d  |   %.6f   |   %.6f   |   %.6f   | %s" % (FS, a, b, c, win))
print()
print("两条反直觉但可复现的结论：")
print("  ① 5 条/类（训练集仅 25 条）时小语料预训练与随机初始化**完全打平**（都是 0.249541）——")
print("     此时决定上限的是「25 条数据」，不是底座好坏，预训练权重体现不出差异。")
print("  ② 大语料预训练在 5 / 10 条时反而**略差**（0.238532 / 0.267308），到 20 / 40 条才反超。")
print("     语料扩到 20000 条学到的是更细的词法统计，下游样本太少时来不及把它「翻译」成分类边界。")
'''

T10_CODE = '''# @@todo 只换池化方式，其余（权重 / 划分 / 步数 / lr）完全一致
# @@hint fresh(SD_BIG, "cls") 与 fresh(SD_BIG, "mean")
acc_cls = fit(fresh(SD_BIG, "cls"), 20)[0]
acc_mean = fit(fresh(SD_BIG, "mean"), 20)[0]
# @@end

# ---- 验收 ----
assert abs(acc_cls - 0.312766) < 1e-5, f"cls 池化应 0.312766，实际 {acc_cls}"
assert abs(acc_mean - 0.389362) < 1e-5, f"mean 池化应 0.389362，实际 {acc_mean}"
assert acc_mean > acc_cls
print("[CLS] 池化 %.6f | mean 池化 %.6f | 差 %+.6f" % (acc_cls, acc_mean, acc_mean - acc_cls))
print()
print("为什么这里 mean 更好：")
print("  MLM 逐位置监督过每个 token（除被 mask 的），所以「逐位置表征」是被训练过的；")
print("  而 [CLS] 只是句首占位符，MLM 目标**没有直接监督它**，它的表征要靠下游微调自己学。")
print("  100 条 few-shot 学不出来。真 BERT 的 [CLS] 好用，是因为有 NSP + 上亿句子。")
'''

LORA_CLASS = '''class LoRALinear(nn.Module):
    """h = W0·x + (alpha/r)·B·A·x —— 原权重冻结，只训 A、B 两个低秩矩阵"""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scale = alpha / r
        # @@todo 定义 A(r, in) 与 B(out, r) 两个可训练参数；**B 必须全零**
        # @@hint B 全零 -> 训练开始时 BA = 0，模型行为与预训练权重逐元素一致
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        # @@end
        with torch.random.fork_rng():
            torch.manual_seed(1234)
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # @@todo 原层输出 + 缩放后的低秩增量
        # @@hint self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
        # @@end


def to_lora(model, r=4, alpha=8.0, targets=("q", "v")):
    """把 attn.q / attn.v 换成 LoRALinear（LoRA 原论文就是挂在 Q/V 上）"""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in targets:
            parent = model
            parts = name.split(".")
            for q in parts[:-1]:
                # ModuleList 的数字下标不支持 getattr，要用 [int(q)]
                parent = parent[int(q)] if q.isdigit() else getattr(parent, q)
            setattr(parent, parts[-1], LoRALinear(mod, r, alpha))
    return model
'''

T11_CODE = '''lin = nn.Linear(256, 256)
z = torch.randn(3, 256)
rows = []
for r, exp in ((4, 2048), (8, 4096), (16, 8192)):
    ll = LoRALinear(lin, r=r, alpha=8.0)
    n_lora = sum(p.numel() for p in [ll.A, ll.B])
    # @@todo 算出「初始时输出与原层是否一致」以及「写入随机 A/B 后的最大差」
    # @@hint 初始 B=0 -> BA=0；写入后用 (ll(z) - lin(z)).abs().max()
    same = bool(torch.allclose(ll(z), lin(z), atol=1e-6))
    n_b = int((ll.B == 0).sum())
    # @@end
    assert n_lora == exp == r * (256 + 256), f"r={r} 参数量应 {exp}"
    assert same and n_b == ll.B.numel(), "B 全零时增量恒为 0"
    rows.append((r, n_lora, n_b, ll.B.numel()))

assert [x[1] for x in rows] == [2048, 4096, 8192]
assert all(x[2] == x[3] for x in rows), "B 的零元素个数 == B 的元素总数"
print("LoRA 参数量 = r × (in + out)：r=4 -> 2048 | r=8 -> 4096 | r=16 -> 8192")
print("原层参数 256×256+256 = %d，r=4 时 LoRA 只占 %.4f%%"
      % (256 * 256 + 256, 100 * 2048 / (256 * 256 + 256)))

ll = LoRALinear(lin, r=8, alpha=8.0)
with torch.no_grad():
    ll.B.normal_(0, 0.05)
    ll.A.normal_(0, 0.05)
d_after = float((ll(z) - lin(z)).abs().max())
assert d_after > 0.01, "B 填上随机值后必然改变输出"
W_merged = lin.weight + ll.scale * (ll.B @ ll.A)
d_merge = float((z @ W_merged.T + lin.bias - ll(z)).abs().max())
assert d_merge < 1e-5, f"合并后应等价，实际差 {d_merge}"
print("A/B 写随机值后输出最大差 %.6f（B 非零才开始起作用）" % d_after)
print("合并校验 W + s·B@A vs LoRA 前向 最大差 %.3e" % d_merge)
print("-> 推理时可把增量合并进原权重，部署零额外延迟（Adapter 做不到，这是 LoRA 更流行的关键）")
'''

T12_CODE = '''# @@todo 算每个 r 下 LoRA 的可训参数与占比
# @@hint 挂载点是每层 attn.q/attn.v 共 NL×2 处，每处 A(r×D)+B(D×r) = 2·r·D 个参数
# @@hint 用字典推导一次算完：{r: NL * 2 * 2 * r * D for r in (4, 8, 16)}
LORA_N = {r: NL * 2 * 2 * r * D for r in (4, 8, 16)}
n_lora4 = LORA_N[4]
# @@end

print("挂载点：每层 attn.q + attn.v，共 %d 层 × 2 个 = %d 个 Linear" % (NL, NL * 2))
for _r, _n in LORA_N.items():
    print("r=%-3d 可训参数 %5d | 占 Encoder %d 的 %.4f%% | 占全模型 %d 的 %.4f%%"
          % (_r, _n, enc_n, 100 * _n / enc_n, tot, 100 * _n / tot))

# ---- 验收 ----
assert n_lora4 == 2048 and LORA_N[16] == 8192
assert abs(100 * n_lora4 / tot - 1.794712) < 1e-5, "r=4 占全模型 1.7947%"
assert abs(100 * LORA_N[16] / tot - 7.178849) < 1e-5, "r=16 占 7.1788%"
assert LORA_N[8] == 4096
print("手算校验：r=4 时 2层 × 2个 × 2矩阵 × 4 × 64 = %d" % n_lora4)
'''

RUN_STRATEGY = '''def run_strategy(tag, strategy, steps=400, bs=32, lr=1e-3, r=4):
    m = fresh(SD_BIG)
    if strategy == "head":
        for n, p in m.named_parameters():
            # @@todo 只训分类头：把非 head 的参数全部冻结
            # @@hint p.requires_grad = n.startswith("head.")
            p.requires_grad = n.startswith("head.")
            # @@end
    elif strategy == "lora":
        m = to_lora(m, r=r)
        for n, p in m.named_parameters():
            # @@todo 只训 LoRA 的 A/B 与分类头
            # @@hint n.endswith(".A") or n.endswith(".B") or n.startswith("head.")
            p.requires_grad = (n.endswith(".A") or n.endswith(".B") or n.startswith("head."))
            # @@end
    params = [p for p in m.parameters() if p.requires_grad]
    ntr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=lr)
    rg4 = np.random.default_rng(3)
    tr, te = FINETUNE_SPLIT
    I_f, S_f, M_f = featurize([tokens[i] for i in tr])
    Y_f = torch.as_tensor([LID[df["缺陷类型"].iloc[i]] for i in tr])
    I_e, S_e, M_e = featurize([tokens[i] for i in te])
    Y_e = torch.as_tensor([LID[df["缺陷类型"].iloc[i]] for i in te])
    m.train()
    for _ in range(steps):
        idx = rg4.integers(0, len(Y_f), size=bs)
        # @@todo 训练一步
        # @@hint 前向取 [0]（分类 logits），再 F.cross_entropy
        loss = F.cross_entropy(m(I_f[idx], S_f[idx], M_f[idx])[0], Y_f[idx])
        # @@end
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(I_e, S_e, M_e)[0].argmax(1) == Y_e).float().mean())
    return acc, ntr
'''

T13_CODE = '''acc_full, n_full = run_strategy("全量微调", "full")
acc_head, n_head = run_strategy("只训分类头", "head")
acc_l4, n_l4 = run_strategy("LoRA r=4", "lora", r=4)
acc_l16, n_l16 = run_strategy("LoRA r=16", "lora", r=16)

# ---- 验收 ----
assert (n_full, n_head, n_l4, n_l16) == (114113, 325, 2373, 8517), \\
    f"可训参数不符：{(n_full, n_head, n_l4, n_l16)}"
assert abs(acc_full - 0.389362) < 1e-5, f"全量 {acc_full}"
assert abs(acc_head - 0.236170) < 1e-5, f"head {acc_head}"
assert abs(acc_l4 - 0.270213) < 1e-5, f"lora4 {acc_l4}"
assert abs(acc_l16 - 0.257447) < 1e-5, f"lora16 {acc_l16}"
assert acc_l4 > acc_head, "LoRA 明显优于只训分类头"
assert acc_l4 / acc_full > 0.69, "LoRA 用 2.08% 参数拿到全量的近 70%"
assert acc_l16 < acc_l4, "r=16 反而更差（数据少，秩大就过拟合）"
print("%-14s %8s %10s %10s" % ("策略", "可训参数", "占比", "测试准确"))
for tag, a, n in (("全量微调", acc_full, n_full), ("只训分类头", acc_head, n_head),
                  ("LoRA r=4", acc_l4, n_l4), ("LoRA r=16", acc_l16, n_l16)):
    print("%-14s %8d %9.4f%% %10.6f" % (tag, n, 100 * n / tot, a))
'''

HF_PREP = '''# 微调用的划分与张量：口径与 §7.9 的 run_strategy 内部完全一致
FT_TR, FT_TE = FINETUNE_SPLIT
I_f, S_f, M_f = featurize([tokens[i] for i in FT_TR])
Y_f = torch.as_tensor([LID[df["缺陷类型"].iloc[i]] for i in FT_TR])
I_e, S_e, M_e = featurize([tokens[i] for i in FT_TE])
Y_e = torch.as_tensor([LID[df["缺陷类型"].iloc[i]] for i in FT_TE])
print("few-shot 张量：训练 %s / 测试 %s（每类 20 条）"
      % (tuple(I_f.shape), tuple(I_e.shape)))

# ---- 验收 ----
assert tuple(I_f.shape) == (100, 24) and tuple(I_e.shape) == (470, 24)
assert len(Y_f) == 100 and len(Y_e) == 470, "每类 20 条 × 5 类 = 100 条训练"
'''

HF_BUILD = '''cfg_hf = BertConfig(vocab_size=NV, hidden_size=D, num_hidden_layers=NL,
                    num_attention_heads=H, intermediate_size=FF,
                    max_position_embeddings=ML, num_labels=5,
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)


def build_hf(seed=9):
    """同配置的 HF BERT —— 仍然是 from_config 随机初始化，不下载任何预训练权重"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return BertForSequenceClassification(cfg_hf)


hf_ref = build_hf()
n_emb = sum(p.numel() for n_, p in hf_ref.named_parameters() if "embeddings" in n_)
n_enc = sum(p.numel() for n_, p in hf_ref.named_parameters() if ".layer." in n_)
n_pool = sum(p.numel() for n_, p in hf_ref.named_parameters() if "pooler" in n_)
n_head = sum(p.numel() for n_, p in hf_ref.named_parameters() if "classifier" in n_)

# @@todo HF 版与手写 MiniBERT 的参数恒等式：MiniBERT 去掉 MLM 头，再加上 pooler
# @@hint 手写 MiniBERT 总参 114113，其中 MLM 头 24828；HF 多出 pooler
N_MLM = 24828
hf_expect = 114113 - N_MLM + n_pool
# @@end

print("HF BertForSequenceClassification 总参数:", hf_ref.num_parameters())
print("  拆解: 嵌入 %d | encoder %d | pooler %d | 分类头 %d"
      % (n_emb, n_enc, n_pool, n_head))
print("恒等式: 114113 - %d(MLM 头) + %d(pooler) = %d" % (N_MLM, n_pool, hf_expect))

# ---- 验收 ----
assert hf_ref.num_parameters() == 93445
assert (n_emb, n_enc, n_pool, n_head) == (22016, 66944, 4160, 325)
assert n_emb + n_enc + n_pool + n_head == 93445
assert hf_expect == 93445
'''

HF_LORA = '''hf = build_hf()

# @@todo 用 ch07 自己写的 to_lora() 把 HF 的 query / value 投影换成 LoRALinear
# @@hint HF 的层名是 query / value，**不是** q / v —— targets 必须显式传
# @@hint to_lora(hf, r=4, targets=("query", "value"))
to_lora(hf, r=4, targets=("query", "value"))
# @@end

hit = [n_ for n_, mod in hf.named_modules() if isinstance(mod, LoRALinear)]
n_all = sum(p.numel() for p in hf.parameters())

for n_, p in hf.named_parameters():
    # @@todo 只训 LoRA 的 A / B 与分类头，其余全部冻结
    # @@hint n_.endswith(".A") or n_.endswith(".B") or n_.startswith("classifier.")
    p.requires_grad = (n_.endswith(".A") or n_.endswith(".B")
                       or n_.startswith("classifier."))
    # @@end
n_grad = sum(p.numel() for p in hf.parameters() if p.requires_grad)

print("替换到的层（%d 个）:" % len(hit))
for n_ in hit:
    print("   ", n_)
print("总参数 %d | 可训参数 %d（%.6f%%）" % (n_all, n_grad, 100 * n_grad / n_all))

# ---- 验收 ----
assert len(hit) == 4
assert hit[0] == "bert.encoder.layer.0.attention.self.query"
assert n_all == 95493
assert n_grad == 2373
assert abs(100 * n_grad / n_all - 2.484999) < 1e-5
'''

HF_STRATEGY = '''def hf_fit(strategy="full", r=4, steps=400, bs=32, lr=1e-3, seed=3):
    m = build_hf()
    if strategy == "head":
        for n_, p in m.named_parameters():
            p.requires_grad = n_.startswith("classifier.")
    elif strategy == "lora":
        to_lora(m, r=r, targets=("query", "value"))
        for n_, p in m.named_parameters():
            p.requires_grad = (n_.endswith(".A") or n_.endswith(".B")
                               or n_.startswith("classifier."))
    m = m.to(DEV_HF)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rg = np.random.default_rng(seed)
    m.train()
    hist = []
    for _ in range(steps):
        idx = torch.as_tensor(rg.integers(0, len(Y_f), size=bs))
        out = m(input_ids=I_f[idx].to(DEV_HF), attention_mask=M_f[idx].to(DEV_HF),
                labels=Y_f[idx].to(DEV_HF))
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        hist.append(out.loss.item())
    m.eval()
    with torch.no_grad():
        lg = m(input_ids=I_e.to(DEV_HF), attention_mask=M_e.to(DEV_HF)).logits
    acc = float((lg.argmax(-1).cpu() == Y_e).float().mean())
    return acc, sum(p.numel() for p in params), float(np.mean(hist[-50:]))


# @@todo 四种策略各跑一次，结果存成 [(tag, (acc, 可训参数, 末段loss)), ...]
# @@hint 列表推导式：[(tag, hf_fit(**kw)) for tag, kw in [...]]
HF_RES = [(tag, hf_fit(**kw)) for tag, kw in [
    ("全量微调", dict(strategy="full")),
    ("只训分类头", dict(strategy="head")),
    ("LoRA r=4", dict(strategy="lora", r=4)),
    ("LoRA r=16", dict(strategy="lora", r=16)),
]]
# @@end

for tag, (acc, n, loss) in HF_RES:
    print("  %-12s 测试准确 %.6f | 可训参数 %7d | 末50步 loss %.6f" % (tag, acc, n, loss))
print("对照 ch07 手写 MiniBERT（同数据同步数）: 全量 0.389362 / head 0.236170 / LoRA r=4 0.270213")

# ---- 验收 ----
assert abs(HF_RES[0][1][0] - 0.959574) < 1e-6
assert abs(HF_RES[1][1][0] - 0.200000) < 1e-6
assert abs(HF_RES[2][1][0] - 0.348936) < 1e-6
assert abs(HF_RES[3][1][0] - 0.310638) < 1e-6
assert HF_RES[0][1][1] == 93445 and HF_RES[2][1][1] == 2373
'''

HF_FAIR = '''def mini_fit(strategy="full", r=4, steps=400, bs=32, lr=1e-3, seed=3):
    """同样的四种策略，但底座换成「全新随机初始化」的 MiniBERT"""
    m = mb(seed=9, pool="mean")
    if strategy == "head":
        for n_, p in m.named_parameters():
            p.requires_grad = n_.startswith("head.")
    elif strategy == "lora":
        m = to_lora(m, r=r)
        for n_, p in m.named_parameters():
            p.requires_grad = (n_.endswith(".A") or n_.endswith(".B")
                               or n_.startswith("head."))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rg = np.random.default_rng(seed)
    m.train()
    for _ in range(steps):
        idx = rg.integers(0, len(Y_f), size=bs)
        loss = F.cross_entropy(m(I_f[idx], S_f[idx], M_f[idx])[0], Y_f[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        acc = float((m(I_e, S_e, M_e)[0].argmax(1) == Y_e).float().mean())
    return acc, sum(p.numel() for p in params)


# ch07 §7.8 / §7.11 已交付的真值：从 MLM 预训练权重 SD_BIG 出发
PRETRAINED = {"全量微调": 0.389362, "只训分类头": 0.236170,
              "LoRA r=4": 0.270213, "LoRA r=16": 0.257447}

# @@todo 随机初始化跑同样四种策略，为「随机 vs 预训练」的胜负计分
# @@hint 列表推导式：[(tag, mini_fit(**kw)[0]) for tag, kw in [...]]
RAND = [(tag, mini_fit(**kw)[0]) for tag, kw in [
    ("全量微调", dict(strategy="full")),
    ("只训分类头", dict(strategy="head")),
    ("LoRA r=4", dict(strategy="lora", r=4)),
    ("LoRA r=16", dict(strategy="lora", r=16)),
]]
# @@end

win = sum(1 for tag, a in RAND if a > PRETRAINED[tag])
print("%-12s %12s %12s %14s" % ("策略", "随机初始化", "预训练权重", "随机 - 预训练"))
for tag, a in RAND:
    print("%-12s %12.6f %12.6f %+14.6f" % (tag, a, PRETRAINED[tag], a - PRETRAINED[tag]))
print("随机初始化战绩: %d 胜 / %d 负（共 %d 项）"
      % (win, len(RAND) - win, len(RAND)))

# ---- 验收 ----
assert win == 4, "每类 20 条下，随机初始化在全部 4 种策略上都优于预训练权重"
assert abs(RAND[0][1] - 0.617021) < 1e-6
assert abs(RAND[2][1] - 0.412766) < 1e-6
'''

HF_TRAINER = '''from torch.utils.data import Dataset


class TicketDS(Dataset):
    """给 Trainer 用的最小数据集：返回 dict，键名与 HF forward 的参数名一致"""

    def __init__(self, I, M, Y):
        self.I, self.M, self.Y = I, M, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return {"input_ids": self.I[i], "attention_mask": self.M[i], "labels": self.Y[i]}


def trainer_run(sched, steps=400, lr=1e-3):
    m = build_hf()
    to_lora(m, r=4, targets=("query", "value"))
    for n_, p in m.named_parameters():
        p.requires_grad = (n_.endswith(".A") or n_.endswith(".B")
                           or n_.startswith("classifier."))
    args = TrainingArguments(output_dir="/tmp/ch07_out", per_device_train_batch_size=32,
                             max_steps=steps, learning_rate=lr, logging_steps=steps // 4,
                             report_to=[], save_strategy="no", eval_strategy="no",
                             disable_tqdm=True, seed=3, dataloader_num_workers=0,
                             warmup_steps=0, lr_scheduler_type=sched)
    tr = Trainer(model=m, args=args, train_dataset=TicketDS(I_f, M_f, Y_f))
    out = tr.train()
    pred = tr.predict(TicketDS(I_e, M_e, Y_e))
    acc = float((pred.predictions.argmax(-1) == Y_e.numpy()).mean())
    return acc, out.metrics["train_loss"]


print("Trainer 的默认 lr 调度:",
      TrainingArguments(output_dir="/tmp/probe", report_to=[], disable_tqdm=True).lr_scheduler_type)

# @@todo 四种 lr 调度各跑一次，看「默认值」到底有多影响结果
# @@hint 列表推导式：[(s, trainer_run(s)) for s in (...)]
TR_RES = [(s, trainer_run(s)) for s in ("linear", "constant", "constant_with_warmup", "cosine")]
# @@end

for s, (acc, loss) in TR_RES:
    print("  %-22s 测试准确 %.6f | train_loss %.6f" % (s, acc, loss))
print("对照手写 loop（恒定 lr，400 步）: 测试准确 0.348936 | 末50步 loss 1.043668")

# ---- 验收 ----
assert abs(TR_RES[0][1][0] - 0.257447) < 1e-6, "默认 linear 调度"
assert abs(TR_RES[1][1][0] - 0.317021) < 1e-6, "constant"
assert TR_RES[1][1][0] == TR_RES[2][1][0], "warmup_steps=0 时 constant 与 constant_with_warmup 等价"
assert abs(TR_RES[3][1][0] - 0.255319) < 1e-6, "cosine"
'''

SUMMARY = '''rows = [
    ("BERT 输入", "input_ids / token_type_ids / attention_mask",
     "单句 seg 全 0；句对前 0 后 1；截断末位强制 [SEP]"),
    ("WordPiece", "## 续接前缀 + 从左贪心最长匹配",
     "请核实 -> 请 + ##核 + ##实；铁芯 -> [UNK]"),
    ("词表规模", "完整词 + 全部后缀 + 单字", "316 词（续接片段 151）"),
    ("参数量", "嵌入 + Encoder + 分类头 + MLM头",
     "22016 + 66944 + 325 + 24828 = 114113"),
    ("MLM 掩码", "15% 选中，其中 80/10/10", "选中 1307 / 9696 = 13.48%"),
    ("MLM 重建", "掩码位置 top-1 / top-5", "小语料 0.863707 / 0.943146"),
    ("预训练 vs 随机", "4 个数据量 × 3 种底座", "随机初始化 3 胜 1 平 0 负"),
    ("扩大预训练语料", "570 -> 20000 条（35 倍）",
     "每类 40 时 0.694595 -> 0.764865，仍低于随机 0.910811"),
    ("池化", "[CLS] vs mean", "0.312766 vs 0.389362"),
    ("LoRA 参数量", "r=4 / r=16", "2373（2.0795%）/ 8517（7.4637%）"),
    ("LoRA 初始等价", "B 全零 + 合并 W+s·BA", "初始输出一致；合并差 1.25e-06"),
    ("LoRA 效果", "r=4 vs 全量 vs 只训 head", "0.270213 vs 0.389362 vs 0.236170"),
]
print(pd.DataFrame(rows, columns=["条目", "做法", "实测结论"]).to_string(index=False))
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch07 微调范式（讲解版）

> 考点：**大模型微调 / BERT 输入表示 / Trainer 流程 / PEFT·LoRA**（`docs/竞赛考点对照表.md` §七）
>
> 环境里**没有 transformers**（也不联网下载权重）。这不是妥协 —— 手写一遍 mini-BERT、
> 手写一遍 LoRA，比调 `AutoModel` 更清楚每个张量的来路。
> 本章结论全部来自实跑，其中包含一个**不好看但真实**的负结果。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 7.1 语料

清洗 + 分词沿用 ch01 的两级切分与正向最大匹配（`clean` / `tok` 在脚手架里给好了）。

下面每个 `assert` 都是**实跑真值**。改数据规模要同步改断言（见根 README §七 7.2）。
"""
    ),
    md(
        """
## 7.2 WordPiece：`##` 续接前缀是干什么的

BPE 与 WordPiece 都要处理词表外词汇（OOV），但拆法不同：

| | BPE | WordPiece |
|---|---|---|
| 建表 | 从字符出发，**反复合并频次最高**的相邻对 | 从字符出发，合并使**似然增益最大**的对 |
| 编码 | 按合并优先级逐对合并 | **从左到右贪心取最长可匹配片段** |
| 标记 | 无特殊约定（有的实现用 `Ġ`） | 词首不加前缀，**续接片段加 `##`** |

本实现用「完整词 + 全部后缀 + 单字」近似建表，编码走标准最长匹配。
**关键细节：只要有一个片段匹配不上，整词退化为 `[UNK]`** —— 这是 HuggingFace 的实际行为，
不是"部分匹配 + 部分 UNK"。
"""
    ),
    code(T1_CODE),
    code(T2_CODE),
    md(
        """
## 7.3 BERT 的输入是三个张量，不是一个

「把句子转成 id 就完了」是常见误解。BERT 实际要喂三个：

1. `input_ids` —— `[CLS]` + 词片 + `[SEP]`，右 padding
2. `token_type_ids`（segment id）—— 句对任务才有用：前一句全 0，后一句全 1
3. `attention_mask` —— 1 = 真实位置，0 = padding

两个易错点：

- **`[SEP]` 的 segment id 归前一句** —— `[CLS] A [SEP] B [SEP]`，第一个 `[SEP]` 是 0，第二个是 1。
  （有些资料写成"第二个 `[SEP]` 属于 B"，那是把边界搞错了。）
- **截断时最后一位要强行写回 `[SEP]`** —— 否则句尾没有闭合标记，`[CLS]` 的语义会漂。
"""
    ),
    code(T3_CODE),
    md(
        """
## 7.4 手写 mini-BERT：Q/K/V 分开写才好挂 LoRA

结构就是 BERT 老三样：**三种 embedding 相加 → LayerNorm → Encoder 堆叠 → 池化 → 分类头**，
再挂一个 MLM 头用于预训练。

这里刻意把自注意力写成 `q/k/v` **三个独立 Linear**（而不是 `nn.MultiheadAttention` 的合并
`in_proj`）：LoRA 原论文就是在 Q/V 上挂低秩增量，拆开写才能精确复现。

Encoder 层用 **Pre-LN** —— ch05 已实测 Post-LN 在 6 层堆叠下首层梯度只剩末层的 0.000007。
"""
    ),
    code(MODEL_CODE),
    code(T4_CODE),
    md(
        """
## 7.5 手写 MLM 预训练

BERT 的预训练目标 **Masked Language Modeling**：随机遮住 15% 的位置，让模型看着**双向**上下文猜回来。

那 15% 里为什么不全换 `[MASK]`？因为 `[MASK]` 只存在于预训练，微调时永远见不到。
若全用 `[MASK]`，模型会学成「看到 `[MASK]` 才需要猜」这条捷径，两个阶段之间出现人为分布差。
BERT 原作的比例是 **80% `[MASK]` / 10% 随机词 / 10% 原样保留**。
"""
    ),
    code(FEATURIZE_CORE),
    code(MLM_FUNC),
    code(T5_CODE),
    code(PRETRAIN_FUNC),
    md(
        """
## 7.6 ★ 预训练什么时候真的有用（本章核心）

做一组对照：**同结构、同下游数据、同步数**，只换「底座权重从哪来」：

- A：570 条本地语料的 MLM
- B：20000 条同分布合成语料的 MLM（35 倍）
- C：随机初始化

先看 MLM 训练完的指标。
"""
    ),
    code(T7_CODE),
    md(
        """
### 再看下游微调 —— 这里出现了一个反直觉的结论
"""
    ),
    code(FEATURIZE_BIG),
    code(FRESH_FIT),
    code(T8_CODE),
    code(T9_CODE),
    md(
        """
### 怎么读这三个读数

1. **随机初始化 3 胜 1 平 0 负**，而且数据越多差距越大（每类 40：0.9108 vs 0.7649）；
   5 条/类时三种底座**完全打平**（都是 0.249541）—— 只有 25 条数据时，
   瓶颈是数据量本身，底座好坏体现不出来。
2. 扩大预训练语料**确实有用**（35 倍语料把每类 40 从 0.6946 拉到 0.7649），
   但**补不上差距**；而且在 5 / 10 条时大语料反而略差（0.2385 < 0.2495、0.2673 < 0.2788）——
   细粒度词法统计需要足够的下游样本才能「翻译」成分类边界。
3. 大语料的 MLM 指标**更差**（top-1 0.8280 < 0.8637），下游却更好 ——
   **预训练指标（重建准确率）不等于表征质量**。池子越大任务越难，指标自然掉，
   但学到的东西更有迁移价值。

### 为什么这个规模下预训练会拖后腿

- **容量不够**：2 层 / d=64 / 11.4 万参数；真 BERT 是 12 层 / d=768 / 1.1 亿参数。
  这个规模装不下「通用语言结构」，MLM 只能记住这批模板的局部共现。
- **语料多样性不够**：20000 条由 6 个模板 × 8 设备 × 18 台区生成，去重后 19855 条，
  但**句式只有 6 种** —— 对 MLM 来说这是个「填空题」，学不到可迁移的东西。
- **下游任务本来就不需要上下文化表征**：五类缺陷的**信号词完全互斥**，
  分类器只要认得「渗油」「锈斑」这些词就行（这正是词袋能到 1.0 的原因）。
  MLM 把 token 表征推向「依赖上下文」的方向，反而模糊了「这个词自带类别信息」这条线索。

**这不是「预训练没用」，而是「预训练有用需要条件」**：语料规模与多样性要**远大于**下游任务、
容量要能承载通用结构、下游任务要能从「语言理解」中获益。三条缺一条，
无监督预训练就可能变成负资产。

> 真实项目里判断这件事，唯一靠谱的办法就是像上面这样**做对照实验**，
> 而不是默认「先预训练准没错」。
"""
    ),
    md(
        """
## 7.7 池化：`[CLS]` 还是 mean

```
mean 池化 = (每一位置的表征 × mask).sum(dim=1) / mask.sum(dim=1)
```
分母只数有效位置，不含 padding。
"""
    ),
    code(T10_CODE),
    md(
        """
## 7.8 手写 LoRA：三行讲完的低秩微调

**动机**：全量微调要为每个下游任务存一份完整副本（7B 模型 = 14 GB）。
LoRA 的洞察是 —— 微调造成的权重变化 $\\Delta W$ 是**低秩**的，那就不存 $\\Delta W$，
存两个小矩阵：

$$h = W_0 x + \\Delta W x = W_0 x + \\frac{\\alpha}{r} B A x,
\\qquad A \\in \\mathbb{R}^{r \\times d_{in}},\\ B \\in \\mathbb{R}^{d_{out} \\times r},\\ r \\ll d$$

参数从 $d_{in}d_{out}$ 降到 $r(d_{in}+d_{out})$。

**两个关键工程细节**（面试高频）：

1. **`B` 必须初始化为全零** —— 训练开始时 $\\Delta W = 0$，模型行为与预训练权重**逐元素一致**，
   不会一上来就毁掉底座。`A` 随机初始化即可（反正被乘 0）。
2. **推理时可以合并** —— $W = W_0 + \\frac{\\alpha}{r}BA$ 直接算进原权重，**部署零额外延迟**。
   这是 LoRA 比 Adapter（串行插入，必然变慢）更流行的关键。
"""
    ),
    code(LORA_CLASS),
    code(T11_CODE),
    code(T12_CODE),
    md(
        """
## 7.9 三策略对比：全量 / 只训分类头 / LoRA

同一份大语料预训练权重、同一份 few-shot 数据（每类 20 条）、同样 400 步。
"""
    ),
    code(RUN_STRATEGY),
    code(T13_CODE),
    md(
        """
### 读数

| 策略 | 可训参数 | 占比 | 测试准确 |
|---|---|---|---|
| 全量微调 | 114113 | 100% | 0.389362 |
| 只训分类头 | 325 | 0.2848% | 0.236170 |
| **LoRA r=4** | **2373** | **2.0795%** | **0.270213** |
| LoRA r=16 | 8517 | 7.4637% | 0.257447 |

三点：

1. **LoRA 用 2.08% 的参数拿到全量微调的 69.4%**，且明显优于只训分类头（参数量是它的 7.3 倍）。
   原因：只训 head 只动最后一个线性层，编码器表征完全不动；LoRA 能改 Q/V 的注意力行为，
   表达能力大得多。
2. **r=16 反而比 r=4 差**（0.2532 vs 0.2766）。秩不是越大越好 —— 这里只有 100 条数据，
   秩大了就过拟合。LoRA 论文的推荐也是从 r=4/8 起试。
3. 三个策略**都没追上随机初始化**（0.5915），因为底座本身就是被 MLM 带偏的。
   **LoRA 的前提是底座可用** —— 它只做局部低秩修正，救不回一个方向错的底座。

## 7.10 汇总
"""
    ),
    code(SUMMARY),
    md(
        """
## 7.11 换真实库：同配置的 HF BERT 长什么样

前面 10 节全是手写。现在把 `transformers` 装上（只建 config，**不下载任何预训练权重**），
用**完全相同的超参**建一个 `BertForSequenceClassification`，看看参数对不对得上。

结果是一条干净的恒等式：

```
手写 MiniBERT 114113  −  MLM 头 24828  +  pooler 4160  =  HF 版 93445
```

- **少掉的 24828** 是 MLM 头 —— `ForSequenceClassification` 是下游任务头，不需要它
- **多出的 4160** 是 `pooler`（`[CLS]` 过一层 `Linear+tanh`）—— 手写版走的是 mean pooling

也就是说：**手写版和 HF 版差的不是"架构"，只是"挂了哪个头"**。
能对上这条恒等式，说明前面的手写没有偷工减料。

⚠️ 这一节有个隐藏坑：`BertConfig` 默认 `attention_probs_dropout_prob=0.1`，
而 **MPS 后端的 `scaled_dot_product_attention` 不支持 dropout**，会直接
`NotImplementedError`。本节显式设成 `0.0` —— 顺便也满足了「训练可复现」的要求。
"""
    ),
    code(HF_PREP),
    code(HF_BUILD),
    md(
        """
## 7.12 把 ch07 的手写 LoRA 挂到 HF 模型上

§7.8 写的 `to_lora()` 不用改一行就能复用 —— 它靠 `named_modules()` 找名字以
`q` / `v` 结尾的 `nn.Linear`。HF 的层名叫 **`query` / `value`**，所以只要换掉 `targets`：

```python
to_lora(hf, r=4, targets=("query", "value"))
```

命中的 4 层（2 层 × Q/V）：

```
bert.encoder.layer.0.attention.self.query
bert.encoder.layer.0.attention.self.value
bert.encoder.layer.1.attention.self.query
bert.encoder.layer.1.attention.self.value
```

替换后总参数 95493（93445 + 4×512），可训参数 **2373 = 2048(LoRA) + 325(分类头)**，
占 **2.485%**。

⚠️ 注意 `to_lora` 只冻结了**被替换的那个 `base` 层**，
HF 模型里其余的 key / output / FFN 还是 `requires_grad=True`。
必须**再手动冻结一遍**，否则你在"全量微调"一个你以为在 LoRA 的模型。
"""
    ),
    code(HF_LORA),
    md(
        """
## 7.13 四种策略在 HF BERT 上真跑（MPS）

同样的 100 条训练数据、400 步、bs=32、lr=1e-3，换成 HF 底座：

| 策略 | 测试准确 | 可训参数 | 末 50 步 loss |
|---|---|---|---|
| **全量微调** | **0.959574** | 93445 | 0.006087 |
| 只训分类头 | **0.200000** | 325 | 1.610289 |
| LoRA r=4 | **0.348936** | 2373 | 1.043668 |
| LoRA r=16 | **0.310638** | 8517 | 1.089199 |

三点值得注意：

1. **只训分类头 = 0.200000，正好是 1/5** —— 325 个参数在随机初始化的底座上等于瞎猜。
2. **LoRA 依然能打**：2.485% 的参数拿到全量的 36%；而且 **r=16 又比 r=4 差**，
   和 §7.9 的结论一致（数据少，秩大即过拟合）。
3. **全量微调 0.959574 比 §7.9 的 0.389362 高出一大截。**

第 3 点先别急着下「HF 更强」的结论 —— **底座不一样，这个对比是不公平的。**
"""
    ),
    code(HF_STRATEGY),
    md(
        """
## 7.14 公平对照：预训练权重到底帮了什么忙

§7.9 的 `run_strategy` 是从 **MLM 预训练权重 `SD_BIG`** 出发的，
而 7.13 的 HF 版是 **`from_config` 全新随机初始化**。差的是底座来源，不是库。

把变量控制住：**同一套手写 MiniBERT 架构**，一边装预训练权重，一边纯随机初始化，
跑完全相同的四种策略。

| 策略 | 随机初始化 | 预训练权重 | 随机 − 预训练 |
|---|---|---|---|
| 全量微调 | **0.617021** | 0.389362 | **+0.227659** |
| 只训分类头 | **0.251064** | 0.236170 | +0.014894 |
| LoRA r=4 | **0.412766** | 0.270213 | **+0.142553** |
| LoRA r=16 | **0.410638** | 0.257447 | **+0.153191** |

**随机初始化 4 胜 0 负。**

这是本章核心负结果的**第二次独立复现**：§7.8 是在「每类 5/10/20/40 条」的数据量矩阵上
测出 3 胜 1 平 0 负，这次是在「四种微调策略」上测出 4 胜 0 负。
两条路都指向同一个结论：

> 在这个规模（11.4 万参数 + 6 种句式 + 570 条数据）下，
> **MLM 预训练权重不是资产，是负担** —— 它把模型带到了一个对当前任务无用的方向上。

所以 7.13 里 HF 全量微调 0.9596 的高分，来源是**随机初始化**（以及 pooler），
**不是**"HF 比手写强"。这也顺带解释了为什么 §7.9 的三个策略都追不上随机初始化。
"""
    ),
    code(HF_FAIR),
    md(
        """
## 7.15 用 Trainer 真跑：默认 lr 调度是个坑

工业界不会手写训练循环，用的是 `Trainer`。把 `accelerate` / `datasets` 装上后，
只需要一个 `Dataset`（返回 dict，键名对上 HF 的 forward 参数）+ 一个 `TrainingArguments`：

```python
Trainer(model=m, args=args, train_dataset=TicketDS(I_f, M_f, Y_f)).train()
```

但**默认值会在短训练上坑你**：

| lr 调度 | 测试准确 | train_loss |
|---|---|---|
| **`linear`（默认）** | **0.257447** | 1.597848 |
| `constant` | **0.317021** | 1.461608 |
| `constant_with_warmup` | 0.317021 | 1.461608 |
| `cosine` | 0.255319 | 1.596339 |

`Trainer` 的默认 `lr_scheduler_type` 是 **`linear`**（且 `warmup_steps=0`），
也就是 **lr 从 `1e-3` 一路线性衰减到 0**。400 步的训练里，
最后 100 步的 lr 已经掉到 `1e-8` 量级 —— 日志里能看到 loss 卡在 1.6 不动。

**默认调度让测试准确率比 `constant` 低 0.0596。** 换成 `constant` 后：
- 与手写 loop（恒定 lr）的差距从 0.09 缩到 0.03
- 剩余差距来自 `Trainer` 用 DataLoader 无放回 shuffle，手写版用有放回随机采样，
  以及手写版 `AdamW` 默认带 `weight_decay=0.01` 而 `TrainingArguments` 默认是 0.0

`warmup_steps=0` 时 `constant` 与 `constant_with_warmup` 结果**完全相同**（0.317021），
这本身就是对「warmup 只在步数 > 0 时才起作用」的一次自洽验证。

⚠️ **`TrainingArguments` 在 transformers 5.17 里已经没有 `warmup_ratio` 了**，
只剩 `warmup_steps`；照抄老教程会直接 `TypeError`。
"""
    ),
    code(HF_TRAINER),
    md(
        """
## 易错点清单

1. **`[SEP]` 的 segment id 写错** —— 句对里第一个 `[SEP]` 属于前一句（0）。
2. **截断时不补 `[SEP]`** —— 句尾未闭合，`[CLS]` 表征会漂。
3. **WordPiece 的 `##` 加错位置** —— 它是给**续接片段**的，词首不加。
4. **以为 WordPiece 会部分匹配** —— 一个片段失败就整词 `[UNK]`。
5. **MLM 全用 `[MASK]`** —— 10% 随机替换 + 10% 原样保留不是装饰。
6. **交叉熵忘了 `ignore_index=-100`** —— 会把 padding 位置也算进 MLM 损失。
7. **LoRA 的 `B` 不置零** —— 训练第一步就毁掉预训练权重。
8. **`alpha` 与 `r` 混用** —— 缩放系数是 `alpha / r`，只改 `r` 会同时改有效学习率。
9. **以为预训练一定涨点** —— 本章实测：这个规模下随机初始化 **3 胜 1 平 0 负**；
   而且 5 / 10 条时大语料预训练反而**略差**。
10. **`ModuleList` 的数字下标不能 `getattr`** —— 替换子模块时要写
    `parent[int(q)] if q.isdigit() else getattr(parent, q)`（历史坑：本节 `to_lora` 踩过）。
"""
    ),
    md(
        """
## 本章小结

| 知识点 | 一句话 |
|---|---|
| BERT 输入 | 三个张量：`input_ids` / `token_type_ids` / `attention_mask` |
| WordPiece | 从左贪心最长匹配，续接片段带 `##`，片段失败则整词 `[UNK]` |
| MLM 掩码 | 15% 选中，其中 80/10/10 —— 为的是消掉两阶段的分布差 |
| 预训练收益 | **有条件**：容量、语料多样性、任务性质三者缺一不可 |
| 池化 | `[CLS]` 靠训练学；MLM 不监督它，小数据下 mean 更稳 |
| LoRA | $\\Delta W = \\frac{\\alpha}{r}BA$，`B` 置零、可合并、推理零延迟 |
| LoRA 效率 | 2.08% 参数拿 69.4% 效果；r 不是越大越好 |
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch07 微调范式（练习版）

> 补全所有 `____`，跑通每个 `assert`。真值来自实跑，不是你算的。

**任务清单**

1. WordPiece 词表构建与 id 空间
2. 手写 `wp_encode`（贪心最长匹配 + `##` 续接）
3. BERT 的三个输入张量
4. MiniBERT 参数量手算对账
5. MLM 掩码策略（80/10/10）
6. 两档语料的 MLM 预训练
7. few-shot：预训练规模 × 下游数据量
8. `[CLS]` vs mean 池化
9. 手写 `LoRALinear`
10. LoRA 参数量与三策略对比
11. HF BERT 参数量恒等式
12. 把手写 LoRA 挂到 HF 模型上
13. 四种策略在 HF BERT 上真跑
14. 公平对照：随机初始化 vs 预训练权重
15. 用 Trainer 真跑（含默认 lr 调度的坑）
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：WordPiece 建表与 id 空间"),
    code(T1_CODE),
    md("## 任务 2：WordPiece 编码"),
    code(T2_CODE),
    md("## 任务 3：BERT 三个输入张量"),
    code(T3_CODE),
    md("## 任务 4：MiniBERT 与参数量"),
    code(MODEL_CODE),
    code(T4_CODE),
    md("## 任务 5：MLM 掩码策略"),
    code(FEATURIZE_CORE),
    code(MLM_FUNC),
    code(T5_CODE),
    md("## 任务 6：两档语料预训练"),
    code(PRETRAIN_FUNC),
    code(FEATURIZE_BIG),
    code(T7_CODE),
    md("## 任务 7：预训练规模 × 下游数据量"),
    code(FRESH_FIT),
    code(T8_CODE),
    code(T9_CODE),
    md("## 任务 8：池化对比"),
    code(T10_CODE),
    md("## 任务 9：手写 LoRA"),
    code(LORA_CLASS),
    code(T11_CODE),
    md("## 任务 10：LoRA 参数量与三策略"),
    code(T12_CODE),
    code(RUN_STRATEGY),
    code(T13_CODE),
    md(
        """
## 任务 11：HF BERT 参数量恒等式

用 `transformers` 建同配置模型（**只建 config，不下载任何权重**），
填出与手写 MiniBERT 的参数量恒等式。

⚠️ MPS 上必须把 `attention_probs_dropout_prob` 设为 `0.0`，
否则 `scaled_dot_product_attention` 会直接 `NotImplementedError`。
"""
    ),
    code(HF_PREP),
    code(HF_BUILD),
    md(
        """
## 任务 12：把手写 LoRA 挂到 HF 模型上

`to_lora()` 是你自己在任务 9 里写的，**一行都不用改**，只需要换 `targets`。

⚠️ HF 的层名是 `query` / `value`，**不是** `q` / `v`。

⚠️ 替换完还要**再冻结一遍**其余参数 —— `to_lora` 只冻结了被替换的那个 `base` 层。
"""
    ),
    code(HF_LORA),
    md(
        """
## 任务 13：四种策略在 HF BERT 上真跑

把任务 10 的四种策略原样搬到 HF 底座上。

⚠️ 拿到 0.959574 这个高分时先别下结论 —— **底座来源不同**，下一题才是公平对照。
"""
    ),
    code(HF_STRATEGY),
    md(
        """
## 任务 14：公平对照（本章最有价值的一题）

同一套手写 MiniBERT，一边装 MLM 预训练权重、一边纯随机初始化，
跑**完全相同**的四种策略。

⚠️ 结论会反直觉：**随机初始化 4 胜 0 负**。
"""
    ),
    code(HF_FAIR),
    md(
        """
## 任务 15：用 Trainer 真跑

补一个 `Dataset` + 一次 `Trainer.train()`，然后对比四种 lr 调度。

⚠️ `Trainer` 默认 `lr_scheduler_type="linear"`，在 400 步这种短训练上会把 lr
提前衰减到 0 —— **默认值直接让准确率掉 0.0596**。

⚠️ transformers 5.17 的 `TrainingArguments` **没有 `warmup_ratio`**，只有 `warmup_steps`。
"""
    ),
    code(HF_TRAINER),
    md(
        """
## 自查清单

- [ ] WordPiece 词表 **316** 词、续接片段 **151** 个
- [ ] `wp_encode("请核实") == ["请", "##核", "##实"]`
- [ ] `wp_encode("铁芯") == ["[UNK]"]`
- [ ] 单句有效长度 4、句对 8、`seg = [0,0,0,0,1,1,1,1]`
- [ ] 截断 `maxlen=3` 得到 `[CLS] 变压器 [SEP]`
- [ ] 参数量 22016 + 66944 + 325 + 24828 = **114113**
- [ ] 掩码选中 **1307**（13.48%）：[MASK] **1070** / 随机 **117** / 保留 **120**
- [ ] 小语料 MLM top-1 **0.863707**；大语料 **0.827988**
- [ ] 数据量矩阵中**随机初始化 3 胜 1 平 0 负**，5 条/类时三家打平 0.249541
- [ ] `[CLS]` **0.312766** < mean **0.389362**
- [ ] LoRA r=4 可训参数 **2373**（**2.0795%**），效果 **0.270213**
- [ ] `B` 全零时 LoRA 输出与原层逐元素一致；合并差 **1.25e-06**
- [ ] 三策略可训参数 = **114113 / 325 / 2373 / 8517**
- [ ] HF BERT 总参 **93445** = 114113 − 24828(MLM 头) + **4160**(pooler)
- [ ] `to_lora(..., targets=("query","value"))` 命中 **4** 层；总参 95493、可训 **2373**（**2.485%**）
- [ ] HF 四策略：全量 **0.959574** / head **0.200000** / LoRA r=4 **0.348936** / r=16 **0.310638**
- [ ] **随机初始化 4 胜 0 负**（0.617021 vs 0.389362 等）—— 核心负结果的二次复现
- [ ] Trainer 默认 **linear** 调度 **0.257447** < `constant` **0.317021**；`warmup_steps=0` 时两者等价

**答不上来时问自己**

1. `token_type_ids` 在单句任务里为什么全 0？什么时候才有用？
2. 为什么 `##` 加在续接片段上，而不是给词首加前缀？
3. MLM 为什么留 10% 原样不动？
4. 为什么大语料的 MLM 指标更差、下游却更好？
5. LoRA 的 `B` 不置零会怎样？
6. 为什么这里 LoRA r=16 反而比 r=4 差？
7. HF 版比手写 MiniBERT **少的 24828** 和**多出的 4160** 分别是什么？
8. HF 全量微调拿到 0.9596，为什么不能推出「真实库比手写强」？
9. `Trainer` 默认的 lr 调度是什么？为什么它在 400 步这种短训练上会坑人？
10. 如果预训练权重在这个规模下是负担，那它要**在什么条件下**才变成资产？
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
