#!/usr/bin/env python3
"""nlp final_01 综合题 —— 工单文本分类全流程（清洗 → 分词 → 词表 → 基线 → mini-Transformer → 评估 → 归因）。

把 ch01~ch07 串成一条**可交付的流水线**，并刻意做两件最容易被跳过的事：

    ① **先建一个最笨的基线**：手写词袋 + 逻辑回归，555 个可训参数
       它拿到测试准确 **1.000000**，把 44741 参数的 mini-Transformer（**0.982353**）压在下面
    ② **对差异做归因，而不是对差异做解释**
       3 条错分**全部**落在同一个信号词「异物」上 —— 这个词在 400 条训练集里只出现 **9 次**
       （其它信号词 14~21 次），它的词嵌入严重欠训练
       词袋给这一列一个 **1.9069** 的权重（强判据直接投票），模型把它摊进 48 维嵌入 + 注意力里，就漏检了

最后一节是本章的「刹车」：换 seed / 换步数，同一个模型的测试准确在
**0.976471 ~ 1.000000** 之间摆动，幅度 **0.023529** —— 比「模型 vs 词袋」的差距
（0.017647）还大。所以本章的结论**不是**「Transformer 不如词袋」，而是：

    在这个任务上两者不可区分，那就选便宜的那个；170 条测试集上的一次实验不构成证据。

真值全部实跑（torch 2.14.0 CPU / numpy 2.5.3 / sklearn 1.9.1），断网可跑、不下载任何权重。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp_final_01_ticket.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp" / "final"
NAME = "final_01_ticket"

# =========================================================================== #
# 公共导入 / 脚手架                                                            #
# =========================================================================== #

IMPORTS = '''import math
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
print("torch", torch.__version__, "| 线程", torch.get_num_threads())
print("依赖只有 标准库 + numpy + torch + sklearn —— 断网可跑，不下载任何预训练权重")
'''

SETUP = '''DATA = Path("../data")          # notebook cwd = coding/04_nlp/final
df = pd.read_csv(DATA / "synth_text.csv", keep_default_na=False)
print("读入", df.shape)
print("列名", list(df.columns))
print("缺陷类型分布", df["缺陷类型"].value_counts().to_dict())
'''

SCAFFOLD = '''# 脚手架：常量与两个工具函数两版都给好，你只填 @@todo 块里的内容
DEVICES = ["变压器", "断路器", "隔离开关", "避雷器", "电流互感器", "绝缘子", "套管", "母线"]
SIGWORDS = {
    "渗漏油": ["渗油", "漏油", "油迹", "油污", "滴油", "油位下降"],
    "锈蚀": ["锈蚀", "生锈", "铁锈", "漆面剥落", "镀锌层脱落"],
    "破损": ["破损", "裂纹", "碎裂", "缺口", "掉瓷", "箱体变形"],
    "发热": ["发热", "温度偏高", "过热", "烫手", "温升异常"],
    "异物": ["异物", "鸟巢", "塑料袋", "树枝", "风筝线搭挂"],
}
LABELS = ["渗漏油", "锈蚀", "破损", "发热", "异物"]
LID = {c: i for i, c in enumerate(LABELS)}
VOCAB = DEVICES + [w for ws in SIGWORDS.values() for w in ws]
VOCAB_SET = set(VOCAB)
MAXW = max(len(w) for w in VOCAB)
BAD = {"", "N/A", "-", "无", "''", '""'}


def clean(t):
    """ch01 的清洗：去 HTML 标签 / URL / 全部空白，半角逗号统一成全角。"""
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"https?://\\S+", "", t)
    t = re.sub(r"[\\u3000\\s]+", "", t)
    return t.replace(",", "，").strip()


def tok(t):
    """ch01 的两级切分 + 正向最大匹配（词典 = 8 个设备名 + 28 个信号词）。"""
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
'''

# --------------------------------------------------------------------------- #
# 任务 1：数据体检                                                             #
# --------------------------------------------------------------------------- #

T1_CODE = '''texts = [clean(t) for t in df["缺陷描述"]]

# @@todo 标出「清洗后仍为空」或「清洗后只剩无效占位符」的样本
# @@hint 先把文本 strip()，再看它是不是落在 BAD 集合里（BAD 里已经有空串）
# @@hint 结果要能直接做布尔索引，所以用 np.array 把一个布尔列表包起来
bad = np.array([t.strip() in BAD for t in texts])
# @@end

# ---- 验收 ----
assert len(texts) == 600, f"原始 {len(texts)} 条"
assert int(bad.sum()) == 30, f"无效文本应为 30 条，实际 {int(bad.sum())}"
keep = np.where(~bad)[0]
assert keep.size == 570, f"可用样本 {keep.size} 条"
Y = np.array([LID[c] for c in df["缺陷类型"]])
print("清洗后无效 %d 条 | 可用 %d 条" % (int(bad.sum()), keep.size))
print("无效样本长这样：", texts[570:575])
print("-> 先过滤再统计：30 条脏样本会污染长度分布、词表和 loss")
'''

# --------------------------------------------------------------------------- #
# 任务 2：分词与长度分布                                                       #
# --------------------------------------------------------------------------- #

T2_CODE = '''tokens = [tok(t) for t in texts]

# @@todo 统计「可用样本」的 token 长度数组（第 k 个元素 = 第 k 条可用样本的 token 数）
# @@hint 对 keep 里的下标取 len(tokens[i])
# @@hint 后面要算 min / max / mean，包成 np.array 最方便
n_tok = np.array([len(tokens[i]) for i in keep])
# @@end

# ---- 验收 ----
assert n_tok.size == 570
assert int(n_tok.min()) == 9 and int(n_tok.max()) == 22, f"长度 {n_tok.min()}~{n_tok.max()}"
assert abs(float(n_tok.mean()) - 15.010526) < 1e-5, f"平均 {n_tok.mean()}"
print("平均 token 数 %.6f | 最短 %d | 最长 %d" % (n_tok.mean(), n_tok.min(), n_tok.max()))
print("最长句的 token 数（前 10）:", sorted(n_tok.tolist(), reverse=True)[:10])
print("-> 最长 22，这就是后面定长 padding 取 24 的依据（留 2 格余量）")
'''

# --------------------------------------------------------------------------- #
# 任务 3：分层切分                                                             #
# --------------------------------------------------------------------------- #

T3_CODE = '''def split_idx(seed=7, frac=0.3):
    """分层切分：每个类别内部打乱后，按 frac 抽进测试集。"""
    rs = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(len(LABELS)):
        pool = keep[Y[keep] == c].copy()
        rs.shuffle(pool)
        # @@todo 算出这一类的测试集条数 n，把前 n 条并入 te，其余并入 tr
        # @@hint n = int(len(pool) * frac)
        # @@hint 用「列表 = 列表 + 列表」的写法（不要用 +=，挖空后更清楚）
        # @@hint 先用 .tolist() 把 numpy 数组转成普通列表
        n = int(len(pool) * frac)
        te = te + pool[:n].tolist()
        tr = tr + pool[n:].tolist()
        # @@end
    return np.array(sorted(tr)), np.array(sorted(te))


TR, TE = split_idx()

# ---- 验收 ----
assert (len(TR), len(TE)) == (400, 170), f"{len(TR)}/{len(TE)}"
assert all(int((Y[TE] == c).sum()) == 34 for c in range(5)), "测试集应当每类恰好 34 条"
assert set(TR.tolist()) & set(TE.tolist()) == set(), "训练集与测试集不能有交集"
print("训练 %d 条 | 测试 %d 条" % (len(TR), len(TE)))
print("测试集每类条数", {LABELS[c]: int((Y[TE] == c).sum()) for c in range(5)})
print("-> 直接随机切分会让某些类在测试集里只剩十来条；分层切分把每类都锁在 34 条")
'''

# --------------------------------------------------------------------------- #
# 任务 4：只用训练集建词表                                                     #
# --------------------------------------------------------------------------- #

T4_CODE = '''tr_tok = {int(i): tokens[int(i)] for i in TR}

# @@todo 统计训练集里每个 token 的出现次数
# @@hint Counter 能直接吃生成器：Counter(w for i in TR for w in tr_tok[int(i)])
cnt = Counter(w for i in TR for w in tr_tok[int(i)])
# @@end

# @@todo 排出词表：先按频次降序，频次相同的按字典序（保证每次结果完全一致）
# @@hint sorted(cnt, key=lambda w: (-cnt[w], w))
vocab = sorted(cnt, key=lambda w: (-cnt[w], w))
# @@end

WID = {w: i + 1 for i, w in enumerate(vocab)}     # 0 号留给 <pad>
NV = len(WID) + 1
TEST_NEW = {w for i in TE for w in tokens[int(i)]} - set(vocab)

# ---- 验收 ----
assert len(vocab) == 110, f"词表 {len(vocab)} 词"
assert len(TEST_NEW) == 0, f"测试集出现训练集没见过的词 {len(TEST_NEW)} 个"
assert NV == 111 and WID[vocab[0]] == 1 and WID[vocab[-1]] == 110
print("训练集词表 %d 词 | 只在测试集出现的词 %d 个" % (len(vocab), len(TEST_NEW)))
print("高频 token（前 12）：", vocab[:12])
print("低频 token（末 5）：", vocab[-5:])
print("-> 排最前的是逗号和模板词（请 / 发 / 现 / 核 / 实），真正的信号词全在尾部")
print("   记住这一点：信号词是「低频但高判别性」，这正是线性模型的主场")
print("-> 词表必须只用训练集建。拿全量建词表 = 偷看测试集，是最常见的泄漏形态")
'''

# --------------------------------------------------------------------------- #
# 任务 5：张量化                                                               #
# --------------------------------------------------------------------------- #

FEATURIZE = '''ML = 24          # 最长 22 个 token，留 2 格余量


def encode(tks):
    """定长编码 + attention mask：padding 填 0（<pad>），真实位置 mask = 1。"""
    ids = [WID.get(w, 0) for w in tks][:ML]
    m = [1] * len(ids)
    return ids + [0] * (ML - len(ids)), m + [0] * (ML - len(ids))


def featurize(idx):
    I, M = [], []
    for i in idx:
        a, b = encode(tokens[int(i)])
        I.append(a)
        M.append(b)
    return torch.as_tensor(I), torch.as_tensor(M, dtype=torch.float32)
'''

T5_CODE = '''# @@todo 把训练集和测试集都张量化：输入 id、注意力掩码、标签
# @@hint 输入调 featurize(TR) / featurize(TE)
# @@hint 标签用 torch.as_tensor(Y[TR]) / torch.as_tensor(Y[TE])
I_tr, M_tr = featurize(TR)
I_te, M_te = featurize(TE)
y_tr = torch.as_tensor(Y[TR])
y_te = torch.as_tensor(Y[TE])
# @@end

# ---- 验收 ----
assert tuple(I_tr.shape) == (400, ML) and tuple(I_te.shape) == (170, ML)
assert tuple(M_tr.shape) == (400, ML) and M_tr.dtype == torch.float32
assert int(M_tr.sum()) == 6001 and int(M_te.sum()) == 2555
assert int((M_tr.sum(1) == ML).sum()) == 0, "训练集不该有样本被截断"
assert int(I_tr.max()) <= len(vocab) and int(I_tr.min()) == 0 and int(I_te.min()) == 0
print("张量 shapes:", tuple(I_tr.shape), tuple(I_te.shape))
print("有效 token 数：训练 %d | 测试 %d" % (int(M_tr.sum()), int(M_te.sum())))
print("被截断的样本数（训练）%d" % int((M_tr.sum(1) == ML).sum()))
print("-> 定长 padding 的代价是浪费算力；attention mask 的作用就是把这部分不算进去")
'''

# --------------------------------------------------------------------------- #
# 任务 6：基线 —— 手写词袋 + 逻辑回归                                          #
# --------------------------------------------------------------------------- #

BOW_HELPER = '''# ---- 基线：手写词袋 + 逻辑回归 --------------------------------------------- #
# 关键：分词口径与正文完全一致。用 CountVectorizer 自带的 tokenizer 会换一套切词规则，
# 那样比出来的差距是「分词差异」而不是「模型差异」，属于不公平对照。
VJ = {w: j for j, w in enumerate(vocab)}


def bow_row(tks, n):
    """把一个样本的 token 列表变成词频行向量。"""
    row = np.zeros(n, np.float32)
    for w in tks:
        row[VJ[w]] += 1.0
    return row
'''

T6_CODE = '''# @@todo 用 bow_row 拼出训练 / 测试的词袋矩阵（行 = 样本，列 = 词表）
# @@hint 列表推导 + np.stack：np.stack([bow_row(tokens[int(i)], len(vocab)) for i in TR])
Xb_tr = np.stack([bow_row(tokens[int(i)], len(vocab)) for i in TR])
Xb_te = np.stack([bow_row(tokens[int(i)], len(vocab)) for i in TE])
# @@end

# @@todo 训练逻辑回归（max_iter 调大一点保证收敛）
# @@hint LogisticRegression(max_iter=3000).fit(Xb_tr, Y[TR])
clf = LogisticRegression(max_iter=3000).fit(Xb_tr, Y[TR])
# @@end

base_pred = clf.predict(Xb_te)
base_acc = float((base_pred == Y[TE]).mean())
base_n = Xb_tr.shape[1] * len(LABELS) + len(LABELS)

# ---- 验收 ----
assert tuple(Xb_tr.shape) == (400, 110) and tuple(Xb_te.shape) == (170, 110)
assert int((Xb_tr > 0).sum()) == 5526 and int((Xb_te > 0).sum()) == 2364
assert abs(base_acc - 1.000000) < 1e-9, f"词袋基线 {base_acc}"
assert base_n == 555, f"可训参数 {base_n}"
print("词袋矩阵", Xb_tr.shape, "| 训练非零 %d | 测试非零 %d"
      % (int((Xb_tr > 0).sum()), int((Xb_te > 0).sum())))
print("词袋 + 逻辑回归：测试准确 %.6f | 可训参数 %d" % (base_acc, base_n))
print("-> %d 个参数的线性模型已经满分。请记住这个数，它是后面所有比较的底线。" % base_n)
'''

# --------------------------------------------------------------------------- #
# 任务 7：mini-Transformer                                                     #
# --------------------------------------------------------------------------- #

MODEL = '''# ---- mini-Transformer（编码器，沿用 ch04 / ch05 的写法）--------------------- #
class MHA(nn.Module):
    """多头自注意力（batch_first：x 形状 (B, T, d)）。"""

    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q, self.k, self.v = (nn.Linear(d, d) for _ in range(3))
        self.o = nn.Linear(d, d)

    def forward(self, x, mask):
        B, T, _ = x.shape
        q, k, v = (p(x).view(B, T, self.h, self.dh).transpose(1, 2)
                   for p in (self.q, self.k, self.v))
        s = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        # @@todo 把 padding 位置的注意力分数压到 -1e9，softmax 之后它们的权重就约等于 0
        # @@hint mask 形状是 (B, T)，要广播成 (B, 1, 1, T)；mask == 0 的位置才是 padding
        # @@hint 用 Tensor.masked_fill
        s = s.masked_fill(mask[:, None, None, :] == 0, -1e9)
        # @@end
        a = s.softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, -1))


class EncLayer(nn.Module):
    """Pre-LN 的编码器层：自注意力 + 前馈，各自带残差。"""

    def __init__(self, d, h, ff):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MHA(d, h)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask):
        # @@todo Pre-LN 残差：先归一化再进子层，子层输出加回原输入
        # @@hint 第一行 x = x + self.att(self.n1(x), mask)
        # @@hint 第二行 x = x + self.ff(self.n2(x))
        x = x + self.att(self.n1(x), mask)
        x = x + self.ff(self.n2(x))
        # @@end
        return x


class TicketNet(nn.Module):
    """工单分类网络：词嵌入 + 位置嵌入 + N 层编码器 + 掩码均值池化 + 线性分类头。"""

    def __init__(self, d=48, h=4, nl=2, ff=96, nv=NV, ncls=len(LABELS), ml=ML, dp=0.0):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(nv, d, padding_idx=0)
        self.pos = nn.Embedding(ml, d)
        self.layers = nn.ModuleList([EncLayer(d, h, ff) for _ in range(nl)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, ncls)
        self.drop = nn.Dropout(dp)

    def forward(self, ids, mask):
        # @@todo 词嵌入先乘 sqrt(d) 再叠加位置嵌入（位置 = 0..T-1）
        # @@hint self.emb(ids) * math.sqrt(self.d) + self.pos(torch.arange(ids.shape[1]))
        # @@hint 乘 sqrt(d) 是为了让词嵌入和位置嵌入量级相当，也是原论文的做法
        x = self.emb(ids) * math.sqrt(self.d) + self.pos(torch.arange(ids.shape[1]))
        # @@end
        for ly in self.layers:
            x = ly(x, mask)
        x = self.norm(x)
        m = mask[:, :, None]
        # @@todo 掩码均值池化：只对非 padding 位置求平均
        # @@hint (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        # @@hint 分母一定要 clamp，否则遇到全 padding 的样本会除零
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        # @@end
        return self.head(self.drop(pooled)), pooled
'''

T7_CODE = '''with torch.random.fork_rng():
    torch.manual_seed(5)
    net0 = TicketNet()

# @@todo 把参数拆成四块：词嵌入 / 位置嵌入 / 全部编码器层 / 归一化 + 分类头
# @@hint 词嵌入 net0.emb.weight.numel()，位置嵌入 net0.pos.weight.numel()
# @@hint 编码器层 = net0.layers 里所有层 .parameters() 的总和（两层都要算上）
# @@hint 归一化 net0.norm + 分类头 net0.head，各取 .parameters() 再 .numel() 求和
N_EMB = net0.emb.weight.numel()
N_POS = net0.pos.weight.numel()
N_ENC = sum(p.numel() for ly in net0.layers for p in ly.parameters())
N_TAIL = sum(p.numel() for p in net0.norm.parameters()) + sum(p.numel() for p in net0.head.parameters())
# @@end
N_ALL = sum(p.numel() for p in net0.parameters())

# ---- 验收 ----
assert (N_EMB, N_POS, N_ENC, N_TAIL) == (5328, 1152, 37920, 341), \\
    f"参数拆解 {(N_EMB, N_POS, N_ENC, N_TAIL)}"
assert N_EMB + N_POS + N_ENC + N_TAIL == N_ALL == 44741
assert N_ENC // 2 == 18960, "单层编码器 18960 个参数"
print("%-12s %8s" % ("模块", "参数量"))
for _tag, _v in (("词嵌入", N_EMB), ("位置嵌入", N_POS), ("2 层编码器", N_ENC),
                 ("LN+分类头", N_TAIL), ("合计", N_ALL)):
    print("%-12s %8d" % (_tag, _v))
print("-> 总参数是词袋基线（%d）的 %.1f 倍，但 token 只有 %d 个，参数却多了两个数量级"
      % (base_n, N_ALL / base_n, len(vocab)))
'''

# --------------------------------------------------------------------------- #
# 任务 8：训练循环（warmup + 余弦退火 + label smoothing）                      #
# --------------------------------------------------------------------------- #

TRAIN_FN = '''def lr_warmup_cos(step, warmup, total):
    """线性 warmup + 余弦退火，返回 [0, 1] 之间的学习率缩放系数。"""
    # @@todo 先线性升到 1.0，再余弦降到 0，取两者较小值
    # @@hint warm = min((step + 1) / warmup, 1.0)
    # @@hint cool = 0.5 * (1.0 + math.cos(math.pi * max(step - warmup, 0) / max(total - warmup, 1)))
    warm = min((step + 1) / warmup, 1.0)
    cool = 0.5 * (1.0 + math.cos(math.pi * max(step - warmup, 0) / max(total - warmup, 1)))
    # @@end
    return min(warm, cool)


def ce_smooth(logits, y, eps=0.0):
    """手写交叉熵：eps > 0 时做 label smoothing（目标类只给 1-eps 的信任）。"""
    V = logits.shape[-1]
    lp = F.log_softmax(logits, -1)
    if eps == 0.0:
        return -lp.gather(-1, y[:, None]).squeeze(-1).mean()
    # @@todo 先构造平滑后的目标分布 d，再算 -(d * lp).sum(-1).mean()
    # @@hint 每个类先均分 eps / V，正确类再加 1 - eps
    # @@hint d = torch.full_like(lp, eps / V)，然后 d.scatter_(-1, y[:, None], 1.0 - eps + eps / V)
    d = torch.full_like(lp, eps / V)
    d.scatter_(-1, y[:, None], 1.0 - eps + eps / V)
    # @@end
    return -(d * lp).sum(-1).mean()


def train_eval(seed=5, steps=400, bs=32, lr=3e-3, warmup=80, eps=0.1):
    """训练并评估一个 mini-Transformer，返回 (模型, loss 曲线, 训练准确, 测试准确, 测试 logits)。"""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = TicketNet()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    rg = np.random.default_rng(seed)
    net.train()
    hist = []
    for s in range(steps):
        # @@todo 读出当前步的学习率缩放系数
        # @@hint lr_warmup_cos(s, warmup, steps)
        # @@hint 必须在 opt.step() 之前读出来，否则每一步用的都是上一步写进去的 lr
        scale = lr_warmup_cos(s, warmup, steps)
        # @@end
        for g in opt.param_groups:
            g["lr"] = lr * scale
        idx = torch.as_tensor(rg.integers(0, len(y_tr), size=bs))
        # @@todo 前向 + label smoothing 交叉熵（只取分类 logits，第二个返回值用不上）
        # @@hint ce_smooth(net(I_tr[idx], M_tr[idx])[0], y_tr[idx], eps)
        loss = ce_smooth(net(I_tr[idx], M_tr[idx])[0], y_tr[idx], eps)
        # @@end
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        hist.append(loss.item())
    net.eval()
    with torch.no_grad():
        tr_logits = net(I_tr, M_tr)[0]
        te_logits = net(I_te, M_te)[0]
    # @@todo 训练 / 测试准确率：argmax(1) 后与标签比较，取 float().mean()
    tr_acc = float((tr_logits.argmax(1) == y_tr).float().mean())
    te_acc = float((te_logits.argmax(1) == y_te).float().mean())
    # @@end
    return net, hist, tr_acc, te_acc, te_logits
'''

T8_CODE = '''# @@todo 验证调度：warmup 内单调升、峰值处等于 1、之后余弦降到 0 附近
# @@hint 取 s = 0 / 40 / 79 / 80 / 239 / 399 六个点
SCHED = [lr_warmup_cos(s, 80, 400) for s in (0, 40, 79, 80, 239, 399)]
# @@end

CE0 = float(ce_smooth(torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]]), torch.tensor([0]), 0.0))
CE2 = float(ce_smooth(torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]]), torch.tensor([0]), 0.2))

# ---- 验收 ----
assert SCHED[0] < SCHED[1] < SCHED[2], "warmup 阶段必须单调升"
assert abs(SCHED[2] - 1.0) < 1e-6 and abs(SCHED[3] - 1.0) < 1e-6, "warmup 结束处应恰为 1"
assert SCHED[4] > SCHED[5] > 0.0, "之后余弦单调降到 0 附近"
assert abs(CE0 - 0.432653) < 1e-5, f"logits=[2,0,0,0,0] 上的普通 CE {CE0}"
assert abs(CE2 - 0.752653) < 1e-5, f"加平滑后的 CE {CE2}"
assert CE2 > CE0, "平滑会把正确类的目标概率压到 1-eps，损失必然变大"
print("lr 缩放（s = 0/40/79/80/239/399）：", [round(v, 6) for v in SCHED])
print("logits=[2,0,0,0,0] 标签=0：eps=0 -> %.10f | eps=0.2 -> %.10f" % (CE0, CE2))
print("-> label smoothing 让模型不敢把某个类打到 1.0，是抑制过拟合最便宜的正则")
'''

# --------------------------------------------------------------------------- #
# 任务 9：训练并报告                                                           #
# --------------------------------------------------------------------------- #

T9_CODE = '''t0 = time.time()
net, hist, tr_acc, te_acc, te_logits = train_eval()
elapsed = time.time() - t0
last50 = float(np.mean(hist[-50:]))

# ---- 验收 ----
assert len(hist) == 400
assert abs(last50 - 0.393147) < 1e-5, f"末 50 步平均 loss {last50}"
assert abs(tr_acc - 1.000000) < 1e-6, f"训练准确 {tr_acc}"
assert abs(te_acc - 0.982353) < 1e-6, f"测试准确 {te_acc}"
assert te_acc < base_acc, "本配置下 mini-Transformer 仍然没追上 555 参数的词袋基线"
print("训练 %d 步，用时 %.2f s" % (len(hist), elapsed))
print("loss：首步 %.4f -> 末 50 步均值 %.6f" % (hist[0], last50))
print("训练准确 %.6f | 测试准确 %.6f | 参数 %d"
      % (tr_acc, te_acc, sum(p.numel() for p in net.parameters())))
print("对照基线：词袋 %d 参数 -> %.6f" % (base_n, base_acc))
print("-> 训练集已经 100% 背下来了，测试集却掉下去：这是过拟合的教科书画面")
'''

# --------------------------------------------------------------------------- #
# 任务 10：逐类指标                                                            #
# --------------------------------------------------------------------------- #

PRF_FN = '''def prf(y_true, y_pred):
    """逐类算 TP / FP / FN 与查准率 / 查全率 / F1。"""
    ys, ps = np.asarray(y_true), np.asarray(y_pred)
    rows = []
    for c in range(len(LABELS)):
        tp = int(((ys == c) & (ps == c)).sum())
        fp = int(((ys != c) & (ps == c)).sum())
        fn = int(((ys == c) & (ps != c)).sum())
        # @@todo 用 tp / fp / fn 算出查准率、查全率、F1（分母为 0 时取 0.0）
        # @@hint 查准率 = tp / (tp + fp)，查全率 = tp / (tp + fn)
        # @@hint F1 = 2 * P * R / (P + R)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        # @@end
        rows.append((LABELS[c], tp, fp, fn, prec, rec, f1))
    return rows
'''

T10_CODE = '''pred = te_logits.argmax(1).numpy()
rows = prf(y_te.numpy(), pred)
macro_f1 = float(np.mean([r[6] for r in rows]))
base_rows = prf(Y[TE], base_pred)
base_macro = float(np.mean([r[6] for r in base_rows]))

# ---- 验收 ----
assert abs(macro_f1 - 0.982074) < 1e-5, f"macro-F1 {macro_f1}"
assert rows[4][2] == 0 and rows[4][3] == 3, "异类：FP=0 但 FN=3"
assert abs(rows[4][6] - 0.9538) < 1e-4, f"异类 F1 {rows[4][6]}"
assert abs(rows[3][6] - 1.0) < 1e-9, "发热类全对"
assert abs(base_macro - 1.0) < 1e-6, f"词袋 macro-F1 {base_macro}"
print("%-6s %4s %4s %4s %9s %9s %9s" % ("类别", "TP", "FP", "FN", "P", "R", "F1"))
for _tag, _tp, _fp, _fn, _p, _r, _f in rows:
    print("%-6s %4d %4d %4d %9.4f %9.4f %9.4f" % (_tag, _tp, _fp, _fn, _p, _r, _f))
print("mini-Transformer macro-F1 %.6f | 词袋 macro-F1 %.6f" % (macro_f1, base_macro))
print("-> 4 个类 F1 都在 0.98 以上，唯一塌陷的是「异物」：查准率满分、查全率 0.9118")
print("-> FP=0 说明模型判它是异物时从没错过；FN=3 说明它不敢判 —— 这是畏缩，不是混淆")
'''

# --------------------------------------------------------------------------- #
# 任务 11：混淆矩阵与错误分析                                                  #
# --------------------------------------------------------------------------- #

T11_CODE = '''probs = F.softmax(te_logits, -1)
conf = probs.max(-1).values.numpy()
yt = y_te.numpy()
ok = pred == yt
cm = np.zeros((len(LABELS), len(LABELS)), int)

# 混淆矩阵一定要用「真值行、预测列」；转置过来会得到完全不同的故事
for a_, b_ in zip(yt, pred):
    # @@todo 混淆矩阵自增：真值 a_ 被判成预测 b_，记一格
    # @@hint 下标顺序是 cm[真值, 预测]
    # @@hint 写成 cm[a_, b_] = cm[a_, b_] + 1（用 += 挖空后看不出赋值目标）
    cm[a_, b_] = cm[a_, b_] + 1
    # @@end
wrong = np.where(~ok)[0]

# ---- 验收 ----
assert int(cm.sum()) == 170 and int(np.trace(cm)) == 167
assert len(wrong) == 3, f"错分 {len(wrong)} 条"
assert abs(float(conf[ok].mean()) - 0.894818) < 1e-6, f"正确样本平均置信 {conf[ok].mean()}"
assert abs(float(conf[~ok].mean()) - 0.288754) < 1e-6, f"错误样本平均置信 {conf[~ok].mean()}"
assert int((conf[wrong] > 0.9).sum()) == 0, "所有错分的置信度都不高 —— 模型自己也不确定"
print("混淆矩阵（行 = 真值，列 = 预测）")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())
print("错分 %d 条（%.4f）| 正确样本平均置信 %.6f | 错误样本 %.6f"
      % (len(wrong), len(wrong) / len(TE), conf[ok].mean(), conf[~ok].mean()))
print("错分里置信度 > 0.9 的有 %d 条" % int((conf[wrong] > 0.9).sum()))
print("=== 错分样例 ===")
for _w in wrong:
    _i = int(TE[_w])
    print("  [%s] 真值 %s -> 预测 %s（置信 %.4f）设备 %s"
          % (df["工单编号"][_i], LABELS[yt[_w]], LABELS[pred[_w]], conf[_w], df["设备类型"][_i]))
    print("      原文：%s" % texts[_i])
print("-> 置信度 0.24~0.33 就是「四选一乱猜」的水平：模型明确知道自己不会")
'''

# --------------------------------------------------------------------------- #
# 任务 12：归因 —— 长尾信号词                                                  #
# --------------------------------------------------------------------------- #

T12_CODE = '''SIG_TR = {kw: sum(1 for i in TR for w in tr_tok[int(i)] if w == kw) for kw in SIGWORDS["异物"]}

# @@todo 找出测试集里含「异物」这个 token 的样本（返回长度 170 的布尔数组）
# @@hint 用 tokens（全场分词结果）判断 "异物" in tokens[int(i)]
has_wu = np.array(["异物" in tokens[int(i)] for i in TE])
# @@end

# @@todo 在「含异物」的这些样本上，分别数一下词袋和模型各答对几条
# @@hint 布尔索引后比较，再 .sum() 取整
# @@hint 词袋用 base_pred 与 Y[TE]，模型用 pred 与 yt
bow_ok = int((base_pred[has_wu] == Y[TE][has_wu]).sum())
net_ok = int((pred[has_wu] == yt[has_wu]).sum())
# @@end

coef_wu = clf.coef_[:, VJ["异物"]]

# ---- 验收 ----
assert SIG_TR["异物"] == 9, f"「异物」在训练集出现 {SIG_TR['异物']} 次"
assert min(SIG_TR.values()) == 9 and max(SIG_TR.values()) == 21, "「异物」是最冷门的信号词"
assert int(has_wu.sum()) == 4, f"测试集含「异物」的样本 {int(has_wu.sum())} 条"
assert (bow_ok, net_ok) == (4, 1), f"词袋对 {bow_ok} / 模型对 {net_ok}"
assert int(coef_wu.argmax()) == 4, "词袋里「异物」这一列的系数峰值应落在「异物」类"
assert abs(float(coef_wu.max()) - 1.9069) < 1e-3, f"系数峰值 {coef_wu.max()}"
print("「异物」在训练集出现 %d 次（本类其它信号词 %d ~ %d 次）"
      % (SIG_TR["异物"], min(SIG_TR.values()), max(SIG_TR.values())))
print("各信号词训练集出现次数：", SIG_TR)
print("测试集含「异物」token 的样本 %d 条 | 词袋答对 %d 条 | 模型答对 %d 条"
      % (int(has_wu.sum()), bow_ok, net_ok))
print("词袋里「异物」这一列对 5 个类的系数：", np.round(coef_wu, 4).tolist())
print("-> 词袋给这一列一个 %.4f 的权重：它只为「异物」类投票，出现即判「异物」" % coef_wu.max())
print("-> 模型把「异物」的 9 次出现摊进 48 维嵌入 + 2 层注意力里，欠训练，于是漏检")
print("-> 这不是「Transformer 不行」，而是「低频强判据 + 小数据」这个组合的固有短板")
'''

# --------------------------------------------------------------------------- #
# 任务 13：消融 —— 池化方式                                                    #
# --------------------------------------------------------------------------- #

POOL_ABL = '''def train_pool(pool, steps=400, seed=5, lr=3e-3, warmup=80, eps=0.1):
    """只换池化方式的对照实验 —— 其余超参完全一致。"""

    class Net(TicketNet):
        def forward(self, ids, mask):
            x = self.emb(ids) * math.sqrt(self.d) + self.pos(torch.arange(ids.shape[1]))
            for ly in self.layers:
                x = ly(x, mask)
            x = self.norm(x)
            m = mask[:, :, None]
            if pool == "cls":
                h = x[:, 0]
            elif pool == "max":
                h = x.masked_fill(m == 0, -1e9).max(1).values
            else:
                h = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
            return self.head(self.drop(h)), h

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    rg = np.random.default_rng(seed)
    net.train()
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * lr_warmup_cos(s, warmup, steps)
        idx = torch.as_tensor(rg.integers(0, len(y_tr), size=32))
        loss = ce_smooth(net(I_tr[idx], M_tr[idx])[0], y_tr[idx], eps)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    with torch.no_grad():
        lg = net(I_te, M_te)[0]
    return float((lg.argmax(1) == y_te).float().mean()), lg
'''

T13_CODE = '''# @@todo 三种池化各训一次，把测试准确率收进字典
# @@hint 字典推导：{pl: train_pool(pl)[0] for pl in ("mean", "max", "cls")}
POOL_ACC = {pl: train_pool(pl)[0] for pl in ("mean", "max", "cls")}
# @@end

# ---- 验收 ----
assert abs(POOL_ACC["mean"] - 0.982353) < 1e-6, POOL_ACC
assert abs(POOL_ACC["max"] - 0.994118) < 1e-6, POOL_ACC
assert abs(POOL_ACC["cls"] - 0.894118) < 1e-6, POOL_ACC
assert POOL_ACC["max"] > POOL_ACC["mean"] > POOL_ACC["cls"]
print("池化方式对比（只改池化，其余完全一致）")
for _pl in ("mean", "max", "cls"):
    print("  %-5s 测试准确 %.6f" % (_pl, POOL_ACC[_pl]))
print("-> [CLS] 池化塌到 0.894：这里没有预训练，序列首位那个位置的表示基本是随机的")
print("-> max 池化略好（0.994118）：它不会被均值稀释，但差距同样落在噪声范围内（见下一节）")
'''

# --------------------------------------------------------------------------- #
# 任务 14：灵敏度 —— 别用 170 条测试集下结论                                   #
# --------------------------------------------------------------------------- #

T14_CODE = '''# @@todo 固定 seed=5，只改训练步数：200 / 400 / 800
# @@hint train_eval(steps=t)[3] 就是测试准确率
STEP_ACC = [train_eval(steps=t)[3] for t in (200, 400, 800)]
# @@end

# @@todo 固定 400 步，只改随机种子：1 / 2 / 3 / 11
# @@hint train_eval(seed=s)[3]
SEED_ACC = [train_eval(seed=s)[3] for s in (1, 2, 3, 11)]
# @@end

# ---- 验收 ----
assert [round(v, 6) for v in STEP_ACC] == [0.976471, 0.982353, 1.000000], STEP_ACC
assert [round(v, 6) for v in SEED_ACC] == [0.988235, 1.000000, 1.000000, 0.994118], SEED_ACC
_spread = max(max(STEP_ACC), max(SEED_ACC)) - min(min(STEP_ACC), min(SEED_ACC))
assert abs(_spread - 0.023529) < 1e-6, f"波动幅度 {_spread}"
print("只换步数（seed=5）：", [round(v, 6) for v in STEP_ACC])
print("只换种子（400 步）：", [round(v, 6) for v in SEED_ACC])
print("同一个模型、同一份数据，只换 seed / 步数，测试准确在 %.6f ~ %.6f 之间摆动"
      % (min(min(STEP_ACC), min(SEED_ACC)), max(max(STEP_ACC), max(SEED_ACC))))
print("摆动幅度 %.6f —— 比「模型 vs 词袋」的差距（%.6f）还大" % (_spread, base_acc - te_acc))
print("-> 170 条测试集上，1 条样本价值 0.005882。差 3 条就是「结论」，差 1 条就是「噪声」")
print("-> 要下结论必须多种子重复 + 报区间；单次实验只能用来发现 bug，不能用来定优劣")
'''

# --------------------------------------------------------------------------- #
# 汇总                                                                         #
# --------------------------------------------------------------------------- #

SUMMARY = '''rows = [
    ("数据体检", "清洗 600 条", "无效 30 条 / 可用 570 条"),
    ("分词", "两级切分 + 正向最大匹配", "平均 15.010526 token，最长 22"),
    ("切分", "分层抽样 seed=7 frac=0.3", "训练 400 / 测试 170（每类 34）"),
    ("词表", "只用训练集建", "110 词；测试集新词 0"),
    ("张量化", "定长 24 + attention mask", "(400,24)/(170,24)，有效 6001/2555，截断 0"),
    ("基线", "手写词袋 + 逻辑回归", "555 参数 -> 测试准确 1.000000"),
    ("mini-Transformer", "d48 h4 2层 ff96 400步", "44741 参数 -> 测试准确 0.982353"),
    ("逐类指标", "macro-F1 / 逐类 P·R·F1", "0.982074；异物 F1 0.9538（FP=0，FN=3）"),
    ("错误分析", "3 条错分", "平均置信 0.288754；置信 > 0.9 的有 0 条"),
    ("归因", "「异物」token", "训练集仅出现 9 次（其它 14~21）-> 词嵌入欠训练"),
    ("池化消融", "mean / max / cls", "0.982353 / 0.994118 / 0.894118"),
    ("灵敏度", "换 seed / 换步数", "0.976471 ~ 1.000000，摆动 0.023529"),
]
print(pd.DataFrame(rows, columns=["环节", "做法", "实测结论"]).to_string(index=False))
print()
print("结论：这个任务上「555 参数的线性模型」与「44741 参数的 Transformer」不可区分，")
print("      选便宜的。大模型的价值在需要上下文消歧 / 长依赖 / 多义性的场景，不在词袋能解的任务上。")
'''

CHECKLIST = """
## 自查清单

- [ ] 清洗：600 条 → 无效 **30** / 可用 **570**
- [ ] 分词：平均 **15.010526** token，最短 9 / 最长 **22**
- [ ] 分层切分：训练 **400** / 测试 **170**，每类 **34** 条
- [ ] 词表 **110** 词（`NV=111`），测试集新词 **0**
- [ ] 张量 `(400,24)` / `(170,24)`，有效 token **6001** / **2555**，训练集截断 **0**
- [ ] 词袋矩阵 `(400,110)` / `(170,110)`，非零 **5526** / **2364**
- [ ] 词袋 + 逻辑回归：**555** 参数 → 测试准确 **1.000000**
- [ ] 参数拆解 **5328 / 1152 / 37920 / 341 = 44741**（单层编码器 18960）
- [ ] lr 调度峰值 1.0，末步 ≈ 0；`logits=[2,0,0,0,0]` 上 CE：eps=0 → **0.432653**，eps=0.2 → **0.752653**
- [ ] mini-Transformer：末 50 步 loss **0.393147**，训练准确 **1.0**，测试准确 **0.982353**
- [ ] macro-F1 **0.982074**；异类 FP=0 / FN=3 / F1 **0.9538**；发热类 F1 **1.0**
- [ ] 混淆矩阵对角 **167**/170；错分 **3** 条
- [ ] 正确样本平均置信 **0.894818**，错误样本 **0.288754**，错分里 >0.9 的 **0** 条
- [ ] 「异物」训练集出现 **9** 次；测试集含该 token **4** 条，词袋对 **4**、模型对 **1**
- [ ] 词袋中「异物」列对「异物」类系数 **1.9069**（5 类中最高）
- [ ] 池化消融：mean **0.982353** / max **0.994118** / cls **0.894118**
- [ ] 灵敏度：步数 **0.976471 / 0.982353 / 1.000000**；种子 **0.988235 / 1.0 / 1.0 / 0.994118**，摆动 **0.023529**

**答不上来时问自己**

1. 为什么词表只能用训练集建？拿全量建会发生什么？
2. 为什么必须做分层切分？随机切分在 5 类均分的小数据集上会出什么问题？
3. 词袋基线为什么必须复用同一套分词？换成 `CountVectorizer` 默认 tokenizer 会怎样？
4. 为什么 `opt.param_groups[0]["lr"]` 必须在 `opt.step()` 之前写？
5. 为什么本任务的 `[CLS]` 池化会塌到 0.894？
6. 「异物」错分的真正原因是什么？换 max 池化解决了吗？
7. 为什么说「0.982353 vs 1.000000」这个差距不构成证据？
8. 如果要让这个流水线上线，你会先做哪三件事？
"""

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# final_01 综合题：工单文本分类全流程（讲解版）

> 把 `ch01` ~ `ch07` 串成一条**可交付的流水线**：
> 清洗 → 手写分词 → 分层切分 → 只用训练集建词表 → 张量化 →
> **手写词袋基线** → 手写 mini-Transformer → 逐类指标 → 混淆矩阵 → 错误分析 →
> **归因** → 消融 → 灵敏度 → 报告
>
> 本题的重点不是把准确率刷高，而是回答一个工程问题：
> **「这个任务到底需不需要 Transformer？」**
> 答案来自实跑，而且不好看 —— 但它是真的。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 1. 数据体检：先过滤，再统计

工单文本是**脏数据重灾区**：HTML 标签、URL、汉字间随机空格、半角逗号、重复标点，
还有 30 条「空 / `N/A` / `-` / `无`」的无效占位。

顺序很重要：**先过滤再统计**。留着 30 条空文本会让长度分布、词表、loss 全部失真。
"""
    ),
    code(T1_CODE),
    md(
        """
## 2. 手写分词与长度分布

分词沿用 `ch01` 的两级切分（字母数字 / 汉字 / 其它符号）+ 正向最大匹配，
词典 = 8 个设备名 + 28 个信号词。

长度分布直接决定后面定长 padding 取多少：最长 22，取 `ML = 24`。
"""
    ),
    code(T2_CODE),
    md(
        """
## 3. 分层切分，不是随机切分

5 类各 120 条。**随机切分**在 570 条上会让某个类在测试集里只剩十几条，
两三条样本就能让该类指标从 0.9 飙到 1.0 —— 波动盖过信号。

**分层切分**（stratified split）在每个类内部打乱后按比例抽，把每类锁死在 34 条。
"""
    ),
    code(T3_CODE),
    md(
        """
## 4. 词表必须只用训练集建

这是**数据泄漏**最隐蔽的一种：拿全量（含测试集）建词表，
等于告诉模型「测试集会用到哪些词」，指标会虚高。

本题实测测试集新词 = 0（合成数据词表可控），所以这一层没有额外收益，
但流程上必须先斩断 —— 真实数据里新词率常在 1%~5%。

顺带注意词表的**顺序**：高频 token 全是逗号和模板词，真正的信号词全在尾部。
「低频但高判别性」正是线性模型最擅长吃的信号。
"""
    ),
    code(T4_CODE),
    md(
        """
## 5. 张量化：定长 padding + attention mask

`encode` 做两件事：截断到 `ML`、右侧补 0；同时产出 `mask`（真实位置 1，padding 0）。

后文的 `attention mask` 与 `mean pooling` 都依赖它 —— **padding 位置绝不能参与计算**。
"""
    ),
    code(FEATURIZE),
    code(T5_CODE),
    md(
        """
## 6. 先建一个最笨的基线

这一步 90% 的人会跳过，然后花三天把 Transformer 调到一个「看起来不错」的数字 ——
却不知道一个 555 参数的线性模型早就满分了。

**公平对照的前提**：词袋必须复用正文同一套分词口径。
用 `CountVectorizer` 自带的 tokenizer 会换一套切词规则，
那比出来的是「分词差异」而不是「模型差异」。
"""
    ),
    code(BOW_HELPER),
    code(T6_CODE),
    md(
        """
## 7. mini-Transformer（编码器）

结构复刻 `ch04` / `ch05`：位置嵌入、Pre-LN 残差、多头注意力（带 padding mask）、
掩码均值池化、线性分类头。

参数量 **44741**，是词袋基线的 80 倍 —— 而词表只有 110 个 token、训练集只有 400 条。
"""
    ),
    code(MODEL),
    code(T7_CODE),
    md(
        """
## 8. 训练循环：warmup + 余弦退火 + label smoothing

三处容易写出隐性 bug：

1. **`lr` 必须在 `opt.step()` 之前写入**。写在之后，第一步会用满 `lr`，
   而 warmup 的意义恰恰是让第一步很小。
2. **label smoothing 是手写的**：目标类只给 `1 - eps` 的信任，其余 `eps / V`。
   实测 `logits = [2, 0, 0, 0, 0]`、标签 0：`eps = 0` 时 CE = **0.432653**，
   `eps = 0.2` 时升到 **0.752653** —— 平滑主动给模型「留余地」，这正是它防过拟合的机制。
3. warmup 与余弦的合成用 `min(warm, cool)`，因为块内不能写 `if`。
"""
    ),
    code(TRAIN_FN),
    code(T8_CODE),
    md(
        """
## 9. 训练并报告

训练集 100% 背下来，测试集 0.982353 —— 典型过拟合。
**并且仍然低于 555 参数的词袋基线。**
"""
    ),
    code(T9_CODE),
    md(
        """
## 10. 只看准确率是不够的：逐类 P / R / F1

准确率会被多数类掩盖。逐类看才暴露出真相：
**4 个类 F1 ≥ 0.985，唯独「异物」塌陷到 0.9538，而且 FP = 0、FN = 3。**

FP = 0 说明模型判「异物」时从没错过，FN = 3 说明它不敢判 —— 这是**畏缩**，不是混淆。
"""
    ),
    code(PRF_FN),
    code(T10_CODE),
    md(
        """
## 11. 混淆矩阵与错误分析

3 条错分全是「异物」→ 分别错成「渗漏油 / 锈蚀 / 破损」。
更有价值的是**置信度**：正确样本平均 0.8948，错误样本平均 **0.2888**，
错分里置信度 > 0.9 的 **0 条**。

0.24~0.33 就是「四选一乱猜」的水平 —— **模型明确知道自己不会**。
这类样本最适合进主动学习（active learning）的待标注队列。
"""
    ),
    code(T11_CODE),
    md(
        """
## 12. 归因：不是「模型不行」，是「这个词太冷」

把 3 条错分放在一起看，共同点立刻跳出来 —— 它们**全都用了字面信号词「异物」**，
而其它 31 条「异物」样本用的是「塑料袋 / 树枝 / 鸟巢 / 风筝线搭挂」。

统计一下训练集的信号词频次：「异物」只出现 **9 次**（其它 14~21 次）。
它的词嵌入只被更新过 9 次，48 维向量基本没训开。

对照词袋：这一列在 5 个类上的系数是 `[-0.46, -0.46, -0.53, -0.46, 1.9069]` ——
**一个 1.9069 的大权重，出现即投票**。9 次出现足够线性模型把这个判据学死。

这就是「低频强判据 + 小数据」下线性模型的天然优势，也是 Transformer 的固有短板：
它把证据摊进嵌入 + 注意力 + 层归一化里，样本不够就摊不平。
"""
    ),
    code(T12_CODE),
    md(
        """
## 13. 消融：池化方式

错误分析给出的第一直觉是「均值池化把单个信号词稀释了，换 max 试试」。

实测：`max` 确实好一点（0.994118 vs 0.982353），但 `[CLS]` 池化塌到 0.894118 ——
因为没有预训练，序列首位那个位置的表示基本是随机的。

**注意 `max` 的收益同样落在噪声范围内**（下一节会证明），
所以正确的说法是「池化是一个可调杠杆」，而不是「max 修好了这个 bug」。
"""
    ),
    code(POOL_ABL),
    code(T13_CODE),
    md(
        """
## 14. 灵敏度：本章的「刹车」

最后一节必须做，否则前面所有比较都不成立。

固定数据、固定模型，只换 seed 或步数：
- 步数 200 / 400 / 800 → **0.976471 / 0.982353 / 1.000000**
- 种子 1 / 2 / 3 / 11 → **0.988235 / 1.000000 / 1.000000 / 0.994118**

摆动幅度 **0.023529**，**比「模型 vs 词袋」的差距（0.017647）还大**。

170 条测试集上，1 条样本价值 `1/170 = 0.005882`。差 3 条就是「结论」，差 1 条就是「噪声」。

**所以本题的结论不是「Transformer 不如词袋」**，而是：
> 在这个任务上两者统计上不可区分，那就选便宜的（555 vs 44741 参数，训练 3 秒 vs 0.02 秒）。

大模型的价值在需要上下文消歧、长依赖、多义性的场景 —— 不在词袋能解的任务上。
"""
    ),
    code(T14_CODE),
    md(
        """
## 15. 汇报：把结论压成一张表

交付物不是「准确率 0.9824」这个数字，而是**一张能解释每个数字从哪来的表**。
"""
    ),
    code(SUMMARY),
    md(CHECKLIST),
]

# =========================================================================== #
# 练习 / 答案版（练习版由本列表自动派生）                                        #
# =========================================================================== #

EXERCISE = [
    md(
        """
# final_01 综合题：工单文本分类全流程（练习版）

> 补全所有 `____`，跑通每个 `assert`。真值全部来自实跑，不是你算的。
> 本题是一条完整流水线，**顺序不能变** —— 后面每一步都依赖前面洗干净的数据。

**任务清单**

1. 数据体检：清洗 + 过滤无效文本
2. 手写分词与长度分布
3. 分层切分（不是随机切分）
4. 只用训练集建词表
5. 张量化：定长 padding + attention mask
6. 手写词袋 + 逻辑回归基线
7. mini-Transformer（注意力 mask / Pre-LN 残差 / 掩码池化）
8. 训练循环：warmup + 余弦退火 + label smoothing
9. 训练并报告
10. 逐类 TP / FP / FN 与 macro-F1
11. 混淆矩阵与错误分析
12. 归因：长尾信号词
13. 消融：池化方式
14. 灵敏度：换 seed / 换步数
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：数据体检（清洗 + 过滤）"),
    code(T1_CODE),
    md("## 任务 2：分词与长度分布"),
    code(T2_CODE),
    md("## 任务 3：分层切分"),
    code(T3_CODE),
    md("## 任务 4：只用训练集建词表"),
    code(T4_CODE),
    md("## 任务 5：张量化（定长 padding + mask）"),
    code(FEATURIZE),
    code(T5_CODE),
    md("## 任务 6：手写词袋基线"),
    code(BOW_HELPER),
    code(T6_CODE),
    md("## 任务 7：mini-Transformer"),
    code(MODEL),
    code(T7_CODE),
    md("## 任务 8：训练循环（warmup + 余弦 + label smoothing）"),
    code(TRAIN_FN),
    code(T8_CODE),
    md("## 任务 9：训练并报告"),
    code(T9_CODE),
    md("## 任务 10：逐类指标与 macro-F1"),
    code(PRF_FN),
    code(T10_CODE),
    md("## 任务 11：混淆矩阵与错误分析"),
    code(T11_CODE),
    md("## 任务 12：归因 —— 长尾信号词"),
    code(T12_CODE),
    md("## 任务 13：消融 —— 池化方式"),
    code(POOL_ABL),
    code(T13_CODE),
    md("## 任务 14：灵敏度 —— 换 seed / 换步数"),
    code(T14_CODE),
    md(
        """
## 汇总
"""
    ),
    code(SUMMARY),
    md(CHECKLIST),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
