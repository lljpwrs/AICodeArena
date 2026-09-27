#!/usr/bin/env python3
"""nlp02 文本表示：词袋 / 手写 TF-IDF（对齐 sklearn 口径）/ n-gram（特征 + 语言模型）/ 共现+SVD 词向量 / 手写 BPE。

数据：`data/synth_text.csv`（承接 ch01 的清洗与分词管道，本章自包含）。
真值全部实跑（pandas 3.x + scikit-learn 1.9.1，无 jieba / gensim / transformers）。

> 本章五条主线：
> ① **TF-IDF 有三种口径** —— 教科书公式 `tf*ln(N/df)` 与 sklearn 的结果最大差
>    **0.322036**；按 sklearn 的平滑口径自己实现，差 **0.0000000000**。
> ② **计算"经典算法"在新库上要重新对齐** —— 不是公式背错了，是默认参数不同。
> ③ **小语料训词向量是没用的** —— 570 条工单的共现矩阵做 SVD，「渗油」的最近邻
>    竟是台区编号；PPMI 加权也没救回来（负结果，本章如实呈现）。
> ④ **表示方法不能创造信息** —— 预测「缺陷类型」三种表示全是 1.000000，
>    而把标签随机打乱后掉到 **0.194737**（≈1/5 瞎猜）。
> ⑤ **n-gram 的天花板可以被量化**（§2.8.1 ~ §2.8.4 新增）：
>    手写 bigram 与 sklearn **逐格对账 6292 = nnz**；
>    未平滑的 bigram 语言模型困惑度 = `inf`（测试集 **18.1818%** 的 bigram 训练时没见过）；
>    Lidstone 的 δ 是 **U 形曲线**，最优 **δ = 0.15 → PP 21.626352**，
>    教科书的「加一平滑」（δ = 1.0）比它差 **25%**；
>    n 从 1 扫到 4，未见率 **0% → 18.18% → 61.07% → 82.88%**，
>    且 4-gram 的训练集种类（1132）**比 3-gram（1250）还少** —— 稀疏让组合数本身收缩。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp02_representation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch02_representation"

# =========================================================================== #
# 公共部分（两版都给）                                                        #
# =========================================================================== #

IMPORTS = '''import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

DATA = Path("data")
CSV = DATA / "synth_text.csv"
'''

SETUP = '''df = pd.read_csv(CSV, keep_default_na=False)

# ---- 词表 / 停用词 / 正则（沿用 ch01）----
VOCAB = [
    "变压器", "断路器", "隔离开关", "避雷器", "电流互感器", "绝缘子", "套管", "母线",
    "渗油", "漏油", "油迹", "油污", "滴油", "油位下降",
    "锈蚀", "生锈", "铁锈", "漆面剥落", "镀锌层脱落",
    "破损", "裂纹", "碎裂", "缺口", "掉瓷", "箱体变形",
    "发热", "温度偏高", "过热", "烫手", "温升异常", "过热保护",
    "异物", "鸟巢", "塑料袋", "树枝", "风筝线搭挂",
    "发现", "巡检", "位于", "存在", "缺陷", "一般", "严重", "危急",
    "接报", "详见", "保护", "动作",
    "请安排抢修", "请尽快核实", "已通知运维班", "请列入本周计划",
    "需停电处理", "请核实后反馈",
    "研究", "研究生", "生命", "命", "起源",
]
VOCAB_SET = set(VOCAB)
MAX_WORD_LEN = max(len(w) for w in VOCAB)
SEG_RE = re.compile(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]+|[^\\sA-Za-z0-9_\\u4e00-\\u9fff]")
HAN_RE = re.compile(r"[\\u4e00-\\u9fff]+")
STOPWORDS = {
    "的", "了", "在", "是", "和", "与", "及", "或", "也", "就", "都",
    "，", "。", "！", "？", "、", "（", "）", "【", "】", "：", "；", " ",
}
# 无效占位符：清洗救不了它们，只能做「有效性过滤」（ch01 §1.3 埋的伏笔）
BAD_TEXTS = {"", "N/A", "-", "无"}


def fmm(seg: str) -> list[str]:
    """正向最大匹配（ch01 手写过）。"""
    out, i = [], 0
    while i < len(seg):
        length = min(MAX_WORD_LEN, len(seg) - i)
        while length > 1 and seg[i:i + length] not in VOCAB_SET:
            length -= 1
        out.append(seg[i:i + length])
        i += length
    return out


def tokenize(text: str) -> list[str]:
    """两级切分：汉字段走最大匹配，其余整体入列。"""
    out = []
    for seg in SEG_RE.findall(text):
        out.extend(fmm(seg) if HAN_RE.fullmatch(seg) else [seg])
    return out


def clean(text: str) -> str:
    """五步清洗（ch01 手写过）。"""
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"https?://\\S+", "", s)
    s = s.replace(",", "，")
    s = re.sub(r"([，。！？])\\1+", r"\\1", s)
    s = re.sub(r"(?<=[\\u4e00-\\u9fff])\\s+(?=[\\u4e00-\\u9fff])", "", s)
    return re.sub(r"\\s+", " ", s).strip()


# ---- 有效性过滤：600 -> 570（这一步在 ch01 只是提了一句，本章必须做）----
texts = [clean(t) for t in df["缺陷描述"]]
keep = [i for i, t in enumerate(texts) if t.strip() not in BAD_TEXTS]
labels = df["缺陷类型"].to_numpy()[keep]
docs = [
    [w for w in tokenize(texts[i]) if w not in STOPWORDS and w.strip()]
    for i in keep
]

vocab = sorted({w for d in docs for w in d})
idx = {w: j for j, w in enumerate(vocab)}

print("原始工单", len(texts), "-> 有效", len(docs), "（剔掉无效占位符）")
print("词表大小", len(vocab), "| 平均每篇 token 数 %.4f" % (sum(len(d) for d in docs) / len(docs)))
print("文档数", len(docs), "| 标签分布", dict(Counter(map(str, labels))))
'''

# =========================================================================== #
# 各任务的代码块                                                              #
# =========================================================================== #

BOW = '''pairs = [(i, idx[w]) for i, d in enumerate(docs) for w in d]
cbow = np.zeros((len(docs), len(vocab)), dtype=np.int32)

for r, c in pairs:
    # @@todo 把 (行, 列) 对应位置累加 1
    # @@hint cbow[r, c] += 1
    cbow[r, c] += 1
    # @@end

print("词袋矩阵形状:", cbow.shape)
print("非零元素:", int((cbow > 0).sum()), "/", cbow.size)
print("第 0 篇的非零词:", [(vocab[j], int(cbow[0, j])) for j in np.nonzero(cbow[0])[0]])

# ---- 验收 ----
assert cbow.shape == (570, 70)
assert int(cbow.sum()) == 3431
'''

BOW_SK = '''cv = CountVectorizer(tokenizer=lambda t: t, lowercase=False, token_pattern=None)
M_cv = cv.fit_transform(docs)

print("sklearn 形状:", M_cv.shape, "| 词表大小:", len(cv.vocabulary_))
print("与手写词表是否完全一致:", sorted(cv.vocabulary_) == vocab)

# ---- 验收 ----
assert M_cv.shape == cbow.shape
assert sorted(cv.vocabulary_) == vocab, "词表必须一字不差"
assert int(M_cv.sum()) == 3431
'''

SPARSITY = '''n_cells = cbow.size
n_nonzero = int((cbow > 0).sum())

# @@todo 算稀疏度：零元素占比
# @@hint 1 - 非零 / 总数
sparsity = 1 - n_nonzero / n_cells
# @@end

print("矩阵元素 %d 个，非零 %d 个" % (n_cells, n_nonzero))
print("稀疏度 %.6f" % sparsity)
print("如果词表涨到 5 万（真实中文词表量级），同样 570 篇文档的矩阵会到 %.1f 万个元素"
      % (570 * 50000 / 10000))

# ---- 验收 ----
assert abs(sparsity - 0.914010) < 1e-6
'''

TFIDF_PLAIN = '''def tfidf_plain(docs, vocab):
    """教科书版 TF-IDF：tf * ln(N / df)，不做归一化。"""
    N = len(docs)
    doc_freq = Counter()
    for d in docs:
        for w in set(d):
            doc_freq[w] += 1

    M = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        cnt = Counter(d)
        tf = {w: c / len(d) for w, c in cnt.items()}
        for w, t in tf.items():
            # @@todo 把 tf * ln(N / df) 写进第 i 行、第 idx[w] 列
            M[i, idx[w]] = t * math.log(N / doc_freq[w])
            # @@end
    return M


M_plain = tfidf_plain(docs, vocab)
print("教科书版形状:", M_plain.shape, "| 最大值 %.6f" % M_plain.max())

# ---- 验收 ----
assert M_plain.shape == (570, 70)
'''

TFIDF_SK = '''def tfidf_sklearn_like(docs, vocab):
    """对齐 sklearn 默认参数：smooth_idf=True + L2 归一化。"""
    N = len(docs)
    doc_freq = Counter()
    for d in docs:
        for w in set(d):
            doc_freq[w] += 1

    # @@todo sklearn 的平滑 idf：ln((1 + N) / (1 + df)) + 1
    # @@hint 注意分子分母都要 +1，最后再 +1
    idf = {w: math.log((1 + N) / (1 + doc_freq[w])) + 1.0 for w in vocab}
    # @@end

    M = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        cnt = Counter(d)
        cols = [idx[w] for w in cnt]
        # @@todo 先填 tf * idf，再对整行做 L2 归一化
        M[i, cols] = [cnt[w] * idf[w] for w in cnt]
        M[i] /= np.sqrt((M[i] ** 2).sum())
        # @@end
    return M


M_sk = tfidf_sklearn_like(docs, vocab)
print("sklearn 口径版形状:", M_sk.shape, "| 行范数示例:",
      [round(float(np.linalg.norm(M_sk[i])), 6) for i in range(3)])

# ---- 验收 ----
assert abs(float(np.linalg.norm(M_sk[0])) - 1.0) < 1e-9, "每行应被 L2 归一化"
'''

TFIDF_CMP = '''sk = TfidfVectorizer(tokenizer=lambda t: t, lowercase=False, token_pattern=None)
M_ref = sk.fit_transform(docs).toarray()
# sklearn 内部词表顺序与我们的 vocab 不同，先对齐列
M_ref = M_ref[:, [sk.vocabulary_[w] for w in vocab]]

doc_freq_sy = sum(1 for d in docs if "渗油" in d)
diff_plain = float(np.abs(M_plain - M_ref).max())
diff_sk = float(np.abs(M_sk - M_ref).max())

print("教科书版  vs sklearn 最大绝对差: %.6f" % diff_plain)
print("sklearn口径 vs sklearn 最大绝对差: %.10f" % diff_sk)
print()
print("「渗油」的 df=%d，idf：教科书 %.6f | sklearn %.6f"
      % (doc_freq_sy, math.log(len(docs) / doc_freq_sy),
         math.log((1 + len(docs)) / (1 + doc_freq_sy)) + 1))

# ---- 验收 ----
assert abs(diff_plain - 0.322036) < 1e-5, "教科书公式与 sklearn 差得很远"
assert diff_sk < 1e-9, "对齐口径后必须几乎完全一致"
'''

NGRAM = '''cv2 = CountVectorizer(tokenizer=lambda t: t, lowercase=False, token_pattern=None,
                      ngram_range=(1, 2))
M_bigram = cv2.fit_transform(docs)
feat2 = list(cv2.get_feature_names_out())

print("unigram+bigram 特征数:", M_bigram.shape[1], "（unigram 只有 %d）" % len(vocab))
print("非零元素:", M_bigram.nnz)
print("bigram 示例:", [f for f in feat2 if " " in f][:5])

# ---- 验收 ----
assert M_bigram.shape == (570, 910)
assert len([f for f in feat2 if " " in f]) == 840
'''

NGRAM_HAND = '''uni_pairs = [(i, idx[w]) for i, d in enumerate(docs) for w in d]
bg_pairs = [(i, (a, b)) for i, d in enumerate(docs) for a, b in zip(d, d[1:])]
bg_keys = sorted({t for _, t in bg_pairs})
bg_pos = {t: j for j, t in enumerate(bg_keys)}

# @@todo 手写 bigram 的「非零格子」集合：去重后的 (文档号, 列号)
# @@hint 集合推导式：{(i, bg_pos[t]) for i, t in bg_pairs}
# @@hint 口径要和 unigram 的 set(uni_pairs) 对齐，两者相加才等于 sklearn 的 nnz
bg_cells = {(i, bg_pos[t]) for i, t in bg_pairs}
# @@end
uni_cells = set(uni_pairs)

print("手写 bigram 种类:", len(bg_keys), "| 出现次数（含重复）:", len(bg_pairs))
print("手写非零格子: unigram %d + bigram %d = %d"
      % (len(uni_cells), len(bg_cells), len(uni_cells) + len(bg_cells)))

M2_cv = CountVectorizer(tokenizer=lambda t: t, lowercase=False, token_pattern=None,
                        ngram_range=(1, 2))
M2 = M2_cv.fit_transform(docs)
feat2b = list(M2_cv.get_feature_names_out())

# @@todo 从 sklearn 的特征名里筛出 bigram（名字形如「词A 词B」，中间一个空格）
# @@hint 列表推导式，条件是 " " in f
sk_bg = [f for f in feat2b if " " in f]
# @@end

print("sklearn 矩阵:", M2.shape, "| bigram 列:", len(sk_bg), "| nnz:", M2.nnz)
print("bigram 列集合与手写一致:", set(sk_bg) == {"%s %s" % (a, b) for a, b in bg_keys})
print("sklearn 前 3 列（注意这里就是它的输出顺序）:", feat2b[:3])

# ---- 验收 ----
assert len(bg_keys) == 840
assert len(bg_pairs) == 2861
assert len(uni_cells) == 3431
assert len(bg_cells) == 2861
assert len(uni_cells) + len(bg_cells) == 6292 == M2.nnz
assert set(sk_bg) == {"%s %s" % (a, b) for a, b in bg_keys}
assert feat2b[:3] == ["STATION_A_01", "STATION_A_01 一般", "STATION_A_01 严重"]
'''

CHAR_NGRAM = '''char_docs = [[c for c in texts[i] if HAN_RE.fullmatch(c)] for i in keep]
char_vocab = sorted({c for d in char_docs for c in d})

# @@todo 字级 bigram 的「非零格子」集合：相邻两字之间**加一个空格**再拼
# @@hint 集合推导式：{(i, a + " " + b) for i, d in enumerate(char_docs) for a, b in zip(d, d[1:])}
# @@hint 这个空格是 sklearn 拼 n-gram 特征名的约定，漏了就对不上账
c_bg = {(i, a + " " + b) for i, d in enumerate(char_docs) for a, b in zip(d, d[1:])}
# @@end
c_uni = {(i, c) for i, d in enumerate(char_docs) for c in d}

print("字级：单字 %d 种 / 总字数 %d | bigram 出现 %d 次 / 去重格子 %d"
      % (len(char_vocab), sum(len(d) for d in char_docs),
         sum(len(d) - 1 for d in char_docs), len(c_bg)))

cvv = CountVectorizer(tokenizer=lambda t: t, lowercase=False, token_pattern=None,
                      ngram_range=(1, 2))
Mc = cvv.fit_transform(char_docs)
fc = list(cvv.get_feature_names_out())

# @@todo 按特征名长度筛出 unigram 列与 bigram 列
# @@hint 单字列 len(f) == 1；bigram 名单字之间带空格，所以 len(f) == 3
fc_uni = [f for f in fc if len(f) == 1]
fc_bg = [f for f in fc if len(f) == 3]
# @@end

print("sklearn 字级:", Mc.shape, "| unigram 列 %d | bigram 列 %d | nnz %d"
      % (len(fc_uni), len(fc_bg), Mc.nnz))
print("字级 bigram 示例（注意空格）:", fc_bg[:4])
print("手写 vs sklearn：unigram 列数一致 %s / bigram 列数一致 %s"
      % (len(c_uni) == len(fc_uni), len(c_bg) == len(fc_bg)))

cw = LogisticRegression(max_iter=1000)
cc = LogisticRegression(max_iter=1000)
s_word = cross_val_score(cw, M_bigram, labels, cv=5)
s_char = cross_val_score(cc, Mc, labels, cv=5)
print("分类 5 折：词级 %.6f | 字级 %.6f" % (s_word.mean(), s_char.mean()))

# ---- 验收 ----
assert len(char_vocab) == 125
assert sum(len(d) for d in char_docs) == 8578
assert len(c_uni) == 8440 and len(c_bg) == 7988
assert len(c_uni) + len(c_bg) == 16428 == Mc.nnz
assert Mc.shape == (570, 620)
assert len(fc_uni) == 125 and len(fc_bg) == 495
assert abs(s_word.mean() - 1.000000) < 1e-6
assert abs(s_char.mean() - 1.000000) < 1e-6
'''

NGRAM_LM = '''N_TR = 456
tr_docs, te_docs = docs[:N_TR], docs[N_TR:]
c1 = Counter(w for d in tr_docs for w in d)

# @@todo 训练集的 bigram 计数：把每篇的相邻对丢进 Counter
# @@hint Counter(t for d in tr_docs for t in zip(d, d[1:]))
# @@hint zip(d, d[1:]) 产出的就是 (w_i, w_{i+1}) 这些相邻对
c2 = Counter(t for d in tr_docs for t in zip(d, d[1:]))
# @@end
V = len(c1)

c2_te = Counter(t for d in te_docs for t in zip(d, d[1:]))
tot_te = sum(c2_te.values())

# @@todo 测试集里「训练集从未见过」的 bigram 出现次数
# @@hint sum(v for k, v in c2_te.items() if k not in c2)
miss = sum(v for k, v in c2_te.items() if k not in c2)
# @@end

print("训练集: unigram %d 种 / %d 个 token | bigram %d 种"
      % (len(c1), sum(c1.values()), len(c2)))
print("测试集: bigram 共 %d 个 | 其中训练集未见 %d (%.4f%%)"
      % (tot_te, miss, 100 * miss / tot_te))


def perplexity(test_docs, prob):
    """困惑度 PP = exp( -1/N * Σ ln P(w_i | w_{i-1}) )。任一项概率为 0 则整体 inf。"""
    log_sum, n, zeros = 0.0, 0, 0
    for d in test_docs:
        for a, b in zip(d, d[1:]):
            p = prob(a, b)
            n = n + 1
            log_sum = log_sum + (math.log(p) if p > 0 else 0.0)
            zeros = zeros + (1 if p <= 0 else 0)
    return (float("inf") if zeros else math.exp(-log_sum / n)), zeros


def mle(a, b):
    # @@todo 最大似然条件概率 P(b|a) = c(a,b) / c(a)
    # @@hint 分子 c2.get((a, b), 0)；分母 c1[a]
    return c2.get((a, b), 0) / c1[a]
    # @@end


def lidstone(a, b, delta):
    # @@todo Lidstone 平滑：分子加 δ，分母加 δ·V（V 是训练集词表大小）
    # @@hint 分母写成 c1[a] + delta * V，分子写成 c2.get((a, b), 0) + delta
    return (c2.get((a, b), 0) + delta) / (c1[a] + delta * V)
    # @@end


print("")
print("均匀分布参照：PP = V =", V)
pp0, z0 = perplexity(te_docs, mle)
print("MLE 未平滑   测试 PP = %s（%d 个零概率）"
      % ("inf" if pp0 == float("inf") else "%.6f" % pp0, z0))

print("--- δ 扫描：困惑度先降后升（U 形）---")
scan = {}
for delta in (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.02, 0.01, 0.001):
    pp, _ = perplexity(te_docs, lambda a, b, delta=delta: lidstone(a, b, delta))
    scan[delta] = pp
    print("  δ=%-6s 测试 PP = %12.6f" % (delta, pp))
best_delta = min(scan, key=scan.get)
print("=> 最优 δ = %s，PP = %.6f" % (best_delta, scan[best_delta]))

pp_tr, _ = perplexity(tr_docs, mle)
print("过拟合的量化：MLE 在训练集 PP = %.6f，到测试集却是 inf" % pp_tr)

# ---- 验收 ----
assert len(c1) == 70 and len(c2) == 752
assert sum(c1.values()) == 2756
assert tot_te == 561 and miss == 102
assert abs(100 * miss / tot_te - 18.1818) < 1e-3
assert pp0 == float("inf") and z0 == 102
assert abs(scan[1.0] - 26.993018) < 1e-5
assert abs(scan[0.15] - 21.626352) < 1e-5
assert abs(scan[0.001] - 43.319158) < 1e-5
assert best_delta == 0.15
assert abs(pp_tr - 9.070455) < 1e-5
'''

NGRAM_SPARSE = '''print("n 越大越稀疏（训练 %d 条 / 测试 %d 条）：" % (len(tr_docs), len(te_docs)))
print("%-8s %10s %10s %10s %10s" % ("n", "训练种类", "测试个数", "未见", "未见率"))
for n in (1, 2, 3, 4):
    ctr = Counter(t for d in tr_docs for t in zip(*(d[i:] for i in range(n))))
    cte = Counter(t for d in te_docs for t in zip(*(d[i:] for i in range(n))))
    # @@todo 统计测试集里「训练集没有」的 n-gram 出现次数
    # @@hint sum(v for k, v in cte.items() if k not in ctr)
    miss_n = sum(v for k, v in cte.items() if k not in ctr)
    # @@end
    total = sum(cte.values())
    print("%-8s %10d %10d %10d %9.4f%%"
          % ("%d-gram" % n, len(ctr), total, miss_n, 100 * miss_n / total))

# ---- 验收 ----
assert len(Counter(t for d in tr_docs for t in zip(*(d[i:] for i in range(1))))) == 70
assert len(Counter(t for d in tr_docs for t in zip(*(d[i:] for i in range(2))))) == 752
assert len(Counter(t for d in tr_docs for t in zip(*(d[i:] for i in range(3))))) == 1250
assert len(Counter(t for d in tr_docs for t in zip(*(d[i:] for i in range(4))))) == 1132
'''

COOC = '''WINDOW = 5
co = np.zeros((len(vocab), len(vocab)), dtype=np.float64)

for d in docs:
    ids = [idx[w] for w in d]
    for i, a in enumerate(ids):
        lo, hi = max(0, i - WINDOW), min(len(ids), i + WINDOW + 1)
        for j in range(lo, hi):
            step = 0 if j == i else 1
            # @@todo 把 step 累加进共现矩阵的 (a, ids[j]) 位置
            co[a, ids[j]] += step
            # @@end

print("共现矩阵形状:", co.shape, "| 非零:", int((co > 0).sum()), "| 总计数:", int(co.sum()))

# ---- 验收 ----
assert co.shape == (70, 70)
assert int(co.sum()) == 17348
assert int((co > 0).sum()) == 3256
'''

PPMI = '''total = co.sum()
row_sum = co.sum(axis=1, keepdims=True)
col_sum = co.sum(axis=0, keepdims=True)

# @@todo 算 PMI，并把负值截断成 0（得到 PPMI）
# @@hint np.log((co * total) / (row_sum * col_sum)) 然后 np.maximum(pmi, 0)
pmi = np.log((co * total) / (row_sum * col_sum + 1e-12) + 1e-12)
ppmi = np.maximum(pmi, 0)
# @@end

print("PPMI 非零:", int((ppmi > 0).sum()), "| 最大值 %.6f" % ppmi.max())

# ---- 验收 ----
assert int((ppmi > 0).sum()) == 2160
assert abs(float(ppmi.max()) - 1.931072) < 1e-5
'''

SVD_EMB = '''svd_raw = TruncatedSVD(n_components=10, random_state=42)
emb_raw = svd_raw.fit_transform(co)
svd = TruncatedSVD(n_components=10, random_state=42)
emb = svd.fit_transform(ppmi)

print("词向量形状:", emb.shape)
print("原始计数版 解释方差比合计: %.6f" % svd_raw.explained_variance_ratio_.sum())
print("PPMI 版    解释方差比合计: %.6f" % svd.explained_variance_ratio_.sum())


def topn(word, matrix, k=3):
    v = matrix[idx[word]]
    # @@todo 算该词向量与所有词向量的余弦相似度
    # @@hint matrix @ v 除以两个范数之积（分母加 1e-12 防零）
    sims = matrix @ v / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(v) + 1e-12)
    # @@end
    order2 = np.argsort(-sims)
    return [(vocab[j], round(float(sims[j]), 6)) for j in order2 if j != idx[word]][:k]


for w in ["变压器", "渗油", "绝缘子", "发热"]:
    print(f"  「{w}」原始计数:", topn(w, emb_raw))
    print(f"  「{w}」PPMI   :", topn(w, emb))

# ---- 验收 ----
assert emb.shape == (70, 10)
assert abs(svd.explained_variance_ratio_.sum() - 0.488376) < 1e-5
assert topn("变压器", emb_raw, 1)[0][0] == "隔离开关"
'''

BPE_CORPUS = '''BPE_CORPUS = ["变压器渗油", "变压器发热", "绝缘子破损", "绝缘子渗油", "断路器发热", "变压器"]
print("BPE 训练语料（6 条，故意很小，方便逐步观察）:", BPE_CORPUS)
'''

BPE_PAIRS = '''def count_pairs(words):
    """统计相邻符号对的加权频次。words: {符号元组: 频次}。"""
    pairs = Counter()
    for syms, freq in words.items():
        for i in range(len(syms) - 1):
            # @@todo 给这一对符号加上该词的频次
            pairs[(syms[i], syms[i + 1])] += freq
            # @@end
    return pairs


def merge_pair(words, pair):
    """把每个词里出现的 pair 合并成一个符号。"""
    merged = {}
    for syms, freq in words.items():
        out, i = [], 0
        while i < len(syms):
            take = 2 if tuple(syms[i:i + 2]) == pair else 1
            # @@todo 按 take 个符号拼成一个：2 个直接拼接，1 个原样
            out.append("".join(syms[i:i + take]))
            i += take
            # @@end
        merged[tuple(out)] = freq
    return merged
'''

BPE_TRAIN = '''def bpe_train(corpus, n_merges):
    """从零训练 BPE：反复找出最高频的相邻符号对并合并。"""
    words = {}
    for w in corpus:
        words[tuple(w)] = words.get(tuple(w), 0) + 1

    merges = []
    for _ in range(n_merges):
        pairs = count_pairs(words)
        if not pairs:
            break
        # @@todo 挑出频次最高的符号对记为一次合并，并用它与语料做合并
        best = max(pairs, key=pairs.get)
        merges.append(best)
        words = merge_pair(words, best)
        # @@end
    return merges, words


merges, final_words = bpe_train(BPE_CORPUS, 8)
print("合并序列（按发生顺序）:")
for step, pair in enumerate(merges, 1):
    print(f"  {step}. {pair[0]!r} + {pair[1]!r} -> {(pair[0] + pair[1])!r}")
print()
print("合并后每个词被切成:", {"/".join(k): v for k, v in final_words.items()})

# ---- 验收 ----
assert merges[0] == ("变", "压")
assert ("变压", "器") in merges, "「器」应跟着「变压」被合并出来"
assert len(merges) == 8
'''

BPE_ENCODE = '''def bpe_encode(word, merges):
    """按学到的合并规则，把一个新词切成子词。"""
    syms = tuple(word)
    for pair in merges:
        out, i = [], 0
        while i < len(syms):
            take = 2 if tuple(syms[i:i + 2]) == pair else 1
            out.append("".join(syms[i:i + take]))
            i += take
        # @@todo 把这一轮合并后的符号序列作为下一轮的输入
        syms = tuple(out)
        # @@end
    return list(syms)


for w in ["变压器渗油", "绝缘子发热", "母线生锈"]:
    print(f"  {w} -> {bpe_encode(w, merges)}")

print()
print("注意最后一个：语料里没出现过「母线」「生锈」，所以只能逐字切开。")
print("子词的价值就在这里 —— 没见过的词也能用已知符号拼出来，不会 OOV。")

# ---- 验收 ----
assert bpe_encode("变压器渗油", merges) == ["变压器渗油"]
assert bpe_encode("绝缘子发热", merges) == ["绝缘子", "发热"]
assert bpe_encode("母线生锈", merges) == ["母", "线", "生", "锈"]
'''

DOWNSTREAM = '''y = labels
sc_bow = cross_val_score(LogisticRegression(max_iter=1000), M_cv, y, cv=5)
sc_tfidf = cross_val_score(LogisticRegression(max_iter=1000), M_sk, y, cv=5)
sc_bigram = cross_val_score(LogisticRegression(max_iter=1000), M_bigram, y, cv=5)

print("预测「缺陷类型」（5 类，文本里确实有信号）:")
print("  词袋       mean %.6f  std %.6f" % (sc_bow.mean(), sc_bow.std()))
print("  TF-IDF     mean %.6f  std %.6f" % (sc_tfidf.mean(), sc_tfidf.std()))
print("  词袋 1-2gram mean %.6f  std %.6f" % (sc_bigram.mean(), sc_bigram.std()))

y_shuffled = np.random.default_rng(0).permutation(y)
sc_shuf = cross_val_score(LogisticRegression(max_iter=1000), M_cv, y_shuffled, cv=5)
print()
print("对照：把标签随机打乱后（文本里就没有这个信息了）:")
print("  词袋       mean %.6f   （≈ 1/5 = 0.2 的水平）" % sc_shuf.mean())

# ---- 验收 ----
assert sc_bow.mean() > 0.99, "信号词互不重叠，线性分类器应该满分"
assert sc_shuf.mean() < 0.35, "随机标签下准确率必须掉回瞎猜水平"
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch02 文本表示：从词袋到子词（讲解版）

> **本节考点**：「文本 / 工单分类」的技术选型环节（`docs/竞赛考点对照表.md` §五），
> 也是理解 ch04 Self-Attention 的必要前情 —— **没有好的表示，注意力也无从谈起**。

数据：`data/synth_text.csv`，承接 ch01 的清洗与分词管道（本章自包含，不依赖上一章的执行状态）。

**本章四条主线**

| # | 结论 | 数字 |
|---|---|---|
| ① | 教科书 TF-IDF 与 sklearn 差得很远 | 最大绝对差 **0.322036** |
| ② | 对齐口径后自己实现能完全一致 | 差 **0.0000000000** |
| ③ | 570 条语料训词向量是没用的 | 「渗油」最近邻是台区编号 |
| ④ | 表示方法不能创造信息 | 满分 → 打乱标签掉到 **0.194737** |
"""
    ),
    md(
        """
## 学习目标 & API 速查

**学习目标**

1. 能手写词袋矩阵，并说清「词表 = 特征维度」意味着什么
2. 能手写 TF-IDF，并**对齐 sklearn 的默认口径**（这是最容易翻车的地方）
3. 能把 n-gram 扩成特征，并知道特征数会怎么涨
4. 能用共现矩阵 + SVD 得到词向量，并**判断它在这份数据上有没有用**
5. 能手写 BPE 的训练与编码，理解子词为什么能缓解 OOV

**API 速查表**

| 操作 | API | 说明 |
|---|---|---|
| 词袋 | `CountVectorizer(tokenizer=..., token_pattern=None)` | 传 `tokenizer` 时**必须**关掉 `token_pattern` |
| TF-IDF | `TfidfVectorizer(norm="l2", smooth_idf=True)` | 默认就带平滑 + L2 |
| n-gram | `ngram_range=(1, 2)` | 上界包含 |
| 降维 | `TruncatedSVD(n_components=k)` | 作用在稀疏矩阵上，不同于 PCA |
| 交叉验证 | `cross_val_score(est, X, y, cv=5)` | |
| 词表顺序 | `vectorizer.vocabulary_` | **每次 fit 都可能不同**，对账前要先对齐列 |
"""
    ),
    code(IMPORTS),
    md(
        """
## 2.1 先做有效性过滤：600 → 570

ch01 留下一个伏笔：**清洗不等于有效性过滤**。
`"N/A"`、`"-"`、`"无"` 这些占位符洗完还是它们自己，但它们不是文本。

不做这一步的代价很具体：`"N/A"` 会被分词器切成 `N`、`/`、`A` 三个 token，
`"-"` 变成 `-`，于是词表里凭空多出四个纯噪声特征。
"""
    ),
    code(SETUP),
    md(
        """
## 2.2 手写词袋（Bag of Words）

词袋的全部信息就是**每个词出现了几次**，词序完全丢弃。

实现上用「先收集所有 (行, 列) 坐标，再统一累加」的写法，
能把两层循环压到一行 —— 也顺便避开了「挖空块里不能写 `for`」的限制。
"""
    ),
    code(BOW),
    md(
        """
## 2.3 对账：手写 vs sklearn

自己写的词袋必须和 `CountVectorizer` 对得上，否则后面的 TF-IDF 就没法信。

**一个坑**：`CountVectorizer` 在传了自定义 `tokenizer` 之后，
必须同时设 `token_pattern=None`，否则它会用默认的正则**再切一遍**，
把已经切好的 token 又打散（单字符 token 会被默认正则直接丢掉）。
"""
    ),
    code(BOW_SK),
    md(
        """
## 2.4 难点深挖：稀疏度不是"性能问题"，是"方法选择问题"

**为什么难**：`570 × 70` 看着很小，没人会关心稀疏度。但真实中文词表是 5 万量级，
同样的文档数就是 `570 × 50000 = 2850 万` 个格子，而**非零的只有几千个**。

**判定规则**：

> 稀疏度 > 0.99 时，**任何"把矩阵当稠密数组算"的做法都会爆**。
> 这也是为什么词袋只能配线性模型（逻辑回归 / 线性 SVM），
> 而深度模型必须先把文本**嵌入**成低维稠密向量（ch04 的 Attention 就是干这个的）。
"""
    ),
    code(SPARSITY),
    md(
        """
## 2.5 手写 TF-IDF（教科书版）

TF-IDF 的直觉：**一个词在当前文档里出现得多（TF 高）、在整个语料里出现得少（IDF 高），
它就越能代表这篇文档**。

教科书公式是 `tf × ln(N / df)`。先照这个写一版。
"""
    ),
    code(TFIDF_PLAIN),
    md(
        """
## 2.6 难点深挖：TF-IDF 有三种口径，你背的是哪一种？

**为什么难**：TF-IDF 没有唯一标准定义。光"IDF 怎么算"就有三种常见写法，
而 sklearn 用的那一种**和大多数教材不一样**。

| 口径 | IDF 公式 | 出处 |
|---|---|---|
| 教科书 | `ln(N / df)` | 大多数教材、维基百科 |
| sklearn 默认 | `ln((1+N) / (1+df)) + 1` | `smooth_idf=True`（**默认开**） |
| 平滑版 | `ln(N / (1+df)) + 1` | 部分实现 |

**效果**：

- 「渗油」的 IDF：教科书 `3.210142`，sklearn `4.169335`
- 两个矩阵最大绝对差：**0.322036**

**判定规则**：

> **自己实现 TF-IDF 时，先查下游库的默认参数，再决定用哪个公式。**
> 竞赛里如果要求"与 sklearn 结果一致"，就必须用平滑口径 + L2 归一化；
> 如果只要求"能分类"，三种口径的效果差别通常很小 —— 但**不能拿教科书公式去对账 sklearn**。
"""
    ),
    code(TFIDF_SK),
    md(
        """
## 2.7 对账：0.322036 与 0.0000000000

同一个数据、同一个模型，**只因为 IDF 公式不同**，结果就差这么多。
对齐口径后误差降到 0（浮点意义上完全一致）。

**一个容易忽略的细节**：`TfidfVectorizer` 的词表顺序和你的 `vocab` 不一样，
对账前必须按 `sk.vocabulary_` 重新排一遍列，否则比较的是错位的矩阵。
"""
    ),
    code(TFIDF_CMP),
    md(
        """
## 2.8 n-gram：花小钱买词序

把相邻的 2 个 token 当一个特征（`("变压器", "渗油")`），
就能把"谁挨着谁"这个信息捡回来一部分。

代价非常直观：unigram 只有 **70** 个特征，加上 bigram 变成 **910** 个
（其中 840 个是 bigram）。**特征数涨了 13 倍，而样本还是 570 条。**
"""
    ),
    code(NGRAM),
    md(
        """
### 2.8.1 手写 n-gram：和 sklearn 逐格对账

前面用的是 `CountVectorizer(ngram_range=(1, 2))`。现在自己实现一遍，
**口径必须完全一致**：unigram **3431** 个非零格子、bigram **2861** 个，
合计 **6292** —— 和 sklearn 的 `nnz` 一分不差。

> 小巧合：词级 unigram 的**出现次数**也正好是 3431，与去重格子数相等。
> 这说明每个词在单篇工单里最多出现一次 —— 模板合成数据的特征，真实语料不会这样。

⚠️ 两个容易踩的点：

1. **sklearn 的输出顺序不是「先 unigram 后 bigram」**，而是**按列名字典序**排的。
   所以 `feat[:70]` 并不是那 70 个 unigram：前 3 列直接就是
   `STATION_A_01` / `STATION_A_01 一般` / `STATION_A_01 严重`。
   要挑 bigram 只能按「名字里有空格」筛。
2. n-gram 的特征名是**用空格把 token 拼起来**的（`"变压器 渗油"`），不是元组。
   对账时得用同样的拼法，否则集合永远对不上。
"""
    ),
    code(NGRAM_HAND),
    md(
        """
### 2.8.2 字级 n-gram：换一个切分粒度

词级 n-gram 有个前提 —— **词得切对**。如果分词器把「变压器」切成「变压 / 器」，
那 bigram `变压器 渗油` 就永远统计不到。

**字级 n-gram** 直接绕过分词：拿汉字当 token。同一份数据两种粒度：

| 粒度 | unigram | bigram | 合计（矩阵列数） |
|---|---|---|---|
| 词级 | 70 | 840 | **910** |
| 字级 | 125 | 495 | **620** |

字级的特征数反而**更少**（620 < 910）—— 汉字只有 125 种，而词的组合空间大得多。

⚠️ 对账时的坑：**sklearn 的字级 bigram 特征名是 `"一 般"`（两字中间夹一个空格，共 3 个字符）**，
不是 `"一般"`。用 `len(f) == 2` 去筛会得到 **0 个** bigram，
然后误判成「sklearn 不支持字级 n-gram」。

**一个诚实的结论**：本节两种表示的 5 折准确率**都是 1.000000**，
OOV 也都是 **0** —— 因为这份数据用的是**固定 70 词白名单**切分，
加上模板生成，压根不会出现未登录词。所以「字级能解决 OOV」这个常见说法
**在这份数据上验证不了**：字级在这里唯一的可见好处是**维度更低**（620 vs 910）。
真实中文语料上字级的价值，要等 ch08 用 jieba 来做才有对照。
"""
    ),
    code(CHAR_NGRAM),
    md(
        """
### 2.8.3 n-gram 语言模型：从计数到困惑度

前面都是把 n-gram 当**特征**喂给分类器。n-gram 的本职其实是**语言模型**：
给定前文，预测下一个词。

最大似然（MLE）条件概率，直接数比例：

```
P(w_i | w_{i-1}) = c(w_{i-1}, w_i) / c(w_{i-1})
```

按 456 / 114 切开（两边标签分布都均匀），跑出来的第一组数字就很难看：

- 训练集 bigram **752** 种，测试集 bigram **561** 个
- 其中**训练集从未见过的有 102 个（18.1818%）**
- 这 102 个让 MLE 的困惑度直接 = `inf` —— **一个零概率就能毁掉整个指标**

**困惑度（perplexity）** 是语言模型的标准指标：

```
PP = exp( -(1/N) * Σ ln P(w_i | w_{i-1}) )
```

直觉上，它约等于「模型在每个位置上平均在多少个词之间犹豫」，**越小越好**。

**参照系很重要**：均匀分布 `P = 1/V` 时 `PP = V = 70`。
所以只要 PP < 70，模型就确实学到了东西（比瞎猜强）。

**平滑（Lidstone）** 把零概率抬起来 —— 分子加 δ，分母加 δ·V：

```
P(w_i | w_{i-1}) = ( c(w_{i-1}, w_i) + δ ) / ( c(w_{i-1}) + δ·V )
```

δ 扫描的结果是标准的 **U 形**：

| δ | 1.0 | 0.5 | 0.3 | 0.2 | **0.15** | 0.1 | 0.05 | 0.01 | 0.001 |
|---|---|---|---|---|---|---|---|---|---|
| 测试 PP | 26.993018 | 23.604461 | 22.216765 | 21.706757 | **21.626352** | 21.864783 | 23.088274 | 28.965477 | 43.319158 |

**最优 δ = 0.15** —— 不是越小越好，也不是越大越好：

- δ 太大（1.0）→ **平滑过度**，把真实分布也一起抹平了 → PP 26.99
- δ 太小（0.001）→ **平滑不足**，几乎等于没平滑 → PP 43.32
- δ = 1.0 就是教科书上的「加一平滑」，在本数据上**比最优值差 25%**

最后一句是过拟合的量化：同一个 MLE 模型在**训练集**上 PP = **9.070455**，
换到测试集就是 `inf`。**训练集上的低困惑度什么也证明不了。**
"""
    ),
    code(NGRAM_LM),
    md(
        """
### 2.8.4 n-gram 的核心矛盾：上下文越长越准，但越稀疏

把 n 从 1 扫到 4，测试集的未见率一路飙升：

| n | 训练集种类 | 测试个数 | 未见 | **未见率** |
|---|---|---|---|---|
| 1-gram | 70 | 675 | 0 | **0.0000%** |
| 2-gram | 752 | 561 | 102 | **18.1818%** |
| 3-gram | 1250 | 447 | 273 | **61.0738%** |
| 4-gram | 1132 | 333 | 276 | **82.8829%** |

n = 4 时，测试集里 **82.88%** 的 4-gram 训练时压根没见过 —— 只能靠平滑瞎猜。

**一个反直觉的细节**：4-gram 的训练集种类（**1132**）**比 3-gram（1250）还少**。
按理说组合数应该随 n 单调增长才对。真实原因是：上下文越长，
能连续凑出 4 个 token 的片段越少，**数据稀疏已经开始让「可能的组合」本身收缩了**。
这是「再往上加 n 也没用」的直接证据，也是后面要上 Transformer 的动机之一。
"""
    ),
    code(NGRAM_SPARSE),
    md(
        """
## 2.9 共现矩阵 → SVD 词向量

词袋丢掉了"词和词的关系"。补法之一是**共现矩阵**：
统计每个词在窗口内和谁一起出现过。

`共现矩阵 (70×70) → SVD 降到 10 维`，就得到每个词的稠密向量 ——
这就是 Word2Vec 之前的经典做法（LSA / LSI 一脉）。
"""
    ),
    code(COOC),
    md(
        """
## 2.10 PPMI 加权

原始共现计数会被高频词主导（"发现"出现 193 次，它和谁都共现得多）。
标准补救是 **PPMI**：把共现次数转成「点互信息」，再把负值截成 0，
衡量的是"这两个词一起出现，**超出随机水平多少**"。
"""
    ),
    code(PPMI),
    code(SVD_EMB),
    md(
        """
## 2.11 难点深挖：小语料训词向量 = 没用（一个诚实的负结果）

**为什么难**：所有教程都会给你看 "king - man + woman ≈ queen" 的漂亮例子，
于是很容易以为"跑个 SVD 就有词向量了"。

**实际结果**（本书实测）：

| 词 | 原始计数最近邻 | PPMI 最近邻 |
|---|---|---|
| 变压器 | 隔离开关 / 断路器 / 避雷器 ✅ | 请安排抢修 / 隔离开关 |
| 渗油 | **STATION_C_02** / 油位下降 / 掉瓷 ❌ | 箱体变形 / 掉瓷 / 漏油 |
| 绝缘子 | 避雷器 / 套管 / 断路器 ✅ | **STATION_F_02** / 电流互感器 |
| 发热 | **STATION_E_03** / 碎裂 / 破损 ❌ | 树枝 / 锈蚀 / 铁锈 |

**结论**：

> ① 词向量需要**百万级词次**的语料。570 条工单（约 3.5 千 token）远远不够，
>    这时候学出来的向量基本是噪声。
> ② PPMI **不是万能药** —— 它放大了低频词的偶然共现，在小语料上反而更差。
> ③ 本章的价值在于**把机制跑通**：知道词向量是怎么从共现矩阵里长出来的，
>    比调包得到一堆漂亮的相似词更重要。
>
> 真实项目里请直接用预训练词向量 / BERT 嵌入（ch07 讨论），
> 不要用几百条业务文本自己训。
"""
    ),
    md(
        """
## 2.12 手写 BPE：让模型自己学出"词"

上面两种表示都要先分词（依赖词表）。**子词（subword）**是另一条路：
不问"词是什么"，而是从字符出发，**反复把最高频的相邻符号对合并**，
最后得到一族"高频片段"。

这就是 BPE（Byte Pair Encoding），GPT 系列 / BERT 用的都是这套思路的变体。

本章用一个 6 条文本的极小语料，方便逐步观察合并过程。
"""
    ),
    code(BPE_CORPUS),
    md(
        """
### 2.12.1 两个基本操作：统计相邻对、合并相邻对

- `count_pairs`：遍历每个词，统计所有相邻符号对的**加权频次**（权重是词频）
- `merge_pair`：把每个词里出现的目标对压成一个符号

`merge_pair` 里用 `take = 2 if ... else 1` 的技巧，把 `if` 挡在挖空块之外 ——
效果与写 `if/else` 完全相同。
"""
    ),
    code(BPE_PAIRS),
    md(
        """
### 2.12.2 训练：贪心地合并 N 次

每轮做三件事：**统计 → 选最高频对 → 合并**。

注意这是**贪心**算法：每轮只看当前最优点，不回溯。
所以合并顺序一旦定下来，后面就只能在这基础上继续 —— 这也是 BPE 简单又有效的原因。
"""
    ),
    code(BPE_TRAIN),
    md(
        """
看合并序列，BPE **自己学出了「变压器」「渗油」「发热」**这些真词 ——
没人告诉它中文的词边界在哪，它只是照着"高频相邻"这条规则走。

### 2.12.3 编码：用学到的规则切新词

训练只跑一次，产出一串 `merges`；之后所有文本都用这串规则切分。

关键看第三个例子 `母线生锈`：这两个词在训练语料里**从没出现过**，
但因为 `母`/`线`/`生`/`锈` 这些单字在语料里出现过，它仍然能被切出来。
**这就是子词相对词表的根本优势：不会 OOV（未登录词）。**
"""
    ),
    code(BPE_ENCODE),
    md(
        """
## 2.13 难点深挖：表示方法不能创造信息

把三种表示都拿去做同一个分类任务：

| 表示 | 特征数 | 5 折准确率 |
|---|---|---|
| 词袋 | 70 | **1.000000** |
| TF-IDF | 70 | **1.000000** |
| 词袋 + 1-2gram | 910 | **1.000000** |

**全是满分**。为什么？因为本数据的五类缺陷用了**互不重叠的信号词**
（渗油 / 锈蚀 / 破损 / 发热 / 异物），线性分类器只要找到那个词就能判对。

**判定规则**：

> 当数据"太干净"时，**表示方法的优劣被淹没** —— 这不是说表示不重要，
> 而是说明「用什么表示」这个问题只有在**任务有难度**时才有意义。
>
> 更要紧的是下面这个对照：把标签随机打乱后，同一套词袋的准确率掉到 **0.194737**
> （≈ 1/5，纯粹瞎猜）。**表示方法能把文本里已有的信息提出来，但提不出不存在的信息。**
>
> 所以竞赛里遇到"模型怎么都上不去"的情况，第一件事不是换模型，是回去看数据与标签。
"""
    ),
    code(DOWNSTREAM),
    md(
        """
## 易错点清单

1. **传了 `tokenizer` 却没设 `token_pattern=None`** —— 默认正则会再切一遍，
   单字符 token 被直接丢弃，词表对不上。
2. **拿教科书 TF-IDF 公式对账 sklearn** —— 差 0.32 不是 bug，是口径不同。
3. **对账时忘了对齐列顺序** —— `vocabulary_` 每次 fit 都可能不同，必须按它重排。
4. **`TfidfVectorizer` 有 `norm="l2"` 默认值** —— 自己实现时忘了归一化，行范数就不是 1。
5. **在稠密数组上做共现矩阵** —— 词表 5 万时 `co` 是 25 亿个格子，必须用稀疏结构。
6. **以为 PPMI 一定比原始计数好** —— 小语料上反而更差。
7. **用几百条业务文本训词向量** —— 得到的相似词基本是噪声。
8. **BPE 合并是贪心的** —— 顺序不可回溯；`merges` 必须**训练与推理用同一串**，否则切分不一致。
9. **以为 n-gram 越大越好** —— 特征数按 V^n 涨，570 条样本撑不起 3-gram。
10. **忽略有效性过滤** —— `N/A` 会变成 `N` `/` `A` 三个噪声特征。

## 本章小结

| 步骤 | 关键判断 | 本章数字 |
|---|---|---|
| 有效性过滤 | 清洗 ≠ 可用 | 600 → **570** |
| 词袋 | 词表 = 特征维度 | (570, 70)，非零 3431 |
| 稀疏度 | 决定能不能用深度模型 | **0.914010** |
| TF-IDF | 口径必须对齐 | 差 **0.322036** → **0.0000000000** |
| n-gram | 特征数爆炸 | 70 → **910** |
| 共现 + SVD | 小语料没用（负结果） | (70, 10)，方差比 0.488376 |
| BPE | 子词解决 OOV | 8 次合并学出「变压器」 |
| 下游验证 | 表示不能创造信息 | 满分 → 打乱标签 **0.194737** |

**下一章**：从 RNN 到 Attention —— 为什么"顺序处理"注定要被"并行加权"取代。
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch02 文本表示（练习版）

> 补全 `____`，跑通所有 assert。真值全部实跑（sklearn 1.9.1）。
>
> `VOCAB` / `docs` / `vocab` / `idx` / 清洗与分词函数都已给好；
> 挖空块内保持**单行**，控制流都在块外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    md("## 任务 1：手写词袋矩阵\n\n坐标收集好了，你只需要把它累加进矩阵。"),
    code(BOW),
    md("## 任务 2：与 `CountVectorizer` 对账\n\n注意 `token_pattern=None`。"),
    code(BOW_SK),
    md("## 任务 3：稀疏度\n\n零元素占比。"),
    code(SPARSITY),
    md("## 任务 4：教科书版 TF-IDF\n\n`tf × ln(N/df)`，先不归一化。"),
    code(TFIDF_PLAIN),
    md(
        """
## 任务 5：对齐 sklearn 口径

两处：**平滑 idf** 的公式，以及**整行 L2 归一化**。
这一题做完，误差应该从 0.32 掉到 0。
"""
    ),
    code(TFIDF_SK),
    code(TFIDF_CMP),
    md("## 任务 6：n-gram 特征（sklearn 版）\n\n`ngram_range` 会把特征数从 70 抬到 910。"),
    code(NGRAM),
    md(
        """
## 任务 7：手写 n-gram 并逐格对账

两处要补：手写的 bigram「非零格子」集合，以及从 sklearn 特征名里筛出 bigram。

⚠️ **别以为 `feat[:70]` 就是那 70 个 unigram** —— sklearn 的输出是**按列名字典序**排的，
前 3 列直接就是台区编号那几项。筛 bigram 只能靠「名字里有空格」。
"""
    ),
    code(NGRAM_HAND),
    md(
        """
## 任务 8：字级 n-gram

两个坑：

1. 拼特征名时**两个汉字之间要加一个空格**（这是 sklearn 拼 n-gram 的约定）。
2. 筛 bigram 列用 `len(f) == 3`（2 个字 + 1 个空格），**不是 `len(f) == 2`** ——
   用错会得到 **0 个**，然后误以为 sklearn 不支持字级 n-gram。
"""
    ),
    code(CHAR_NGRAM),
    md(
        """
## 任务 9：n-gram 语言模型与困惑度

三处要补：① 训练集的 bigram 计数；② 测试集里「训练集没见过」的 bigram 条数；
③ MLE 条件概率与 Lidstone 平滑两个函数。

困惑度的公式已经写在 `perplexity()` 里，你只需要补概率。

⚠️ **δ 不是越小越好**：扫描结果是 **U 形**，最优值在 `0.15` 附近；
教科书上的「加一平滑」（δ=1.0）在这里比最优值差 **25%**。
"""
    ),
    code(NGRAM_LM),
    md(
        """
## 任务 10：n 越大越稀疏

统计测试集里「训练集没有」的 n-gram 出现次数，看未见率随 n 怎么变。

⚠️ 4-gram 的训练集种类会**比 3-gram 少** —— 不是统计错了，是稀疏让组合数本身收缩了。
"""
    ),
    code(NGRAM_SPARSE),
    md(
        """
## 任务 11：共现矩阵

窗口内的词对累加；**别把自己和自己算进去**（`step` 已经帮你算好了）。
"""
    ),
    code(COOC),
    md("## 任务 12：PPMI\n\nPMI 后把负值截成 0。"),
    code(PPMI),
    md("## 任务 13：词向量与余弦最近邻\n\n先降维，再补 `topn` 的相似度计算。"),
    code(SVD_EMB),
    md("## 任务 14：BPE 的两个基本操作"),
    code(BPE_CORPUS),
    code(BPE_PAIRS),
    md("## 任务 15：BPE 训练（贪心合并）"),
    code(BPE_TRAIN),
    md("## 任务 16：BPE 编码\n\n把每一轮的合并结果传给下一轮。"),
    code(BPE_ENCODE),
    md(
        """
## 任务 17：表示 × 分类器（含负对照）

三种表示都是满分；把标签随机打乱后掉到 0.19 —— 这才是关键结论。
"""
    ),
    code(DOWNSTREAM),
    md(
        """
## 自查清单

- [ ] `CountVectorizer` 传了 `tokenizer` 就设了 `token_pattern=None`
- [ ] 词袋非零 3431、稀疏度 0.914010
- [ ] 教科书 TF-IDF 与 sklearn 差 **0.322036**
- [ ] 对齐口径后差 **0.0000000000**，且行范数为 1
- [ ] `ngram_range=(1, 2)` → 910 个特征
- [ ] 共现矩阵对角线上不计入自身
- [ ] PPMI 负值截成 0
- [ ] BPE 第一对合并是 `("变", "压")`
- [ ] 未登录词 `母线生锈` 也能被切成子词（不 OOV）
- [ ] 能解释"随机标签下准确率为什么掉到 0.19"
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
