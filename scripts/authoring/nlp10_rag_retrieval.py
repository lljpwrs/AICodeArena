#!/usr/bin/env python3
"""nlp10 RAG 检索增强：清洗深度 / 手写 BM25 / 稀疏-稠密-融合 / 评测口径 / 误差归因 / 定向改进 / 拒答与 prompt 拼装。

离线约束：不调用任何大模型 API，生成端用**模板式拼装**演示"增强"这一步。
真值全部实跑（Python 3.13 / jieba 0.42.1 / sklearn 1.9.1 / pandas 3.0.6，`env -u PYTHONPATH`）。

> 本章回答一个工程问题：**"检索不好，到底该改哪里？"**
> 四条实测结论（按收益从大到小）：
> ① 数据清洗深度：MRR 0.5659 → **0.6111**（+0.045）、Hit@5 0.7069 → **0.7759**（+0.069）；
> ② 误差归因驱动的定向改进（设备元数据硬过滤 + 缺陷词加权）：MRR → **0.9345**、Hit@5 → **0.9655**；
> ③ 换检索算法 / 融合 / 重排：MRR 只动 **±0.02**；
> ④ 索引里剔无效占位符：几乎不影响 top-5（它们本来就排不上来）。
> 换句话说：**先修数据，再看误差归因，最后才调算法。**

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp10_rag_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch10_rag_retrieval"

# =========================================================================== #
# 公共部分（两版都给）                                                        #
# =========================================================================== #

IMPORTS = '''import math
import re
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

jieba.setLogLevel(60)          # 关掉 jieba 的日志，输出才干净
jieba.initialize()
np.random.seed(42)
SEED = 42
'''

SCAFFOLD = '''RAW = pd.read_csv("data/synth_text.csv", keep_default_na=False)

# ---- 清洗用的正则（与 ch01 同一套口径）----
TAG_RE = re.compile(r"<[^>]+>")                       # HTML 标签
STATION_RE = re.compile(r"STATION_[A-Z]_\\d{2}")       # 台区编号（st1~st3 的语义噪声）
URL_RE = re.compile(r"https?://\\S+|10\\.0\\.0\\.1/\\S*")   # 附件 URL
PUNCT = set("，,。.、；;：:（）()<>《》“”‘’！!？?、·-—[]{}【】 　/")
STOP = {"请", "的", "了", "和", "与", "及", "或", "位于", "存在",
        "发现", "详见", "一般", "严重", "危急"}
BAD_PLACEHOLDERS = {"N/A", "-", "无"}                  # 无效占位符（不是真文本）

# ---- 领域词表：误差归因之后要加权的那两族词 ----
DEV_WORDS = {"变压器", "断路器", "隔离开关", "避雷器", "电流互感器", "绝缘子", "套管",
             "母线", "互感器", "开关", "隔离"}
TYP_WORDS = {"渗油", "漏油", "油迹", "油污", "滴油", "锈蚀", "生锈", "铁锈", "破损",
             "裂纹", "碎裂", "缺口", "异物", "鸟巢", "塑料袋", "树枝", "发热",
             "过热", "烫手", "温度", "偏高", "温升"}


def clean_basic(text):
    """A 口径：只去 HTML 标签与台区编号，空白折叠成一个空格。"""
    return re.sub(r"\\s+", " ", STATION_RE.sub(" ", TAG_RE.sub(" ", text))).strip()


def clean_full(text):
    """B 口径：再去 URL，并且**去掉全部空白** —— 顺手修掉"字间空格"这类 OCR 噪声。"""
    return re.sub(r"\\s+", "", STATION_RE.sub(" ", URL_RE.sub(" ", TAG_RE.sub(" ", text))))


def tok(text):
    """分词 + 去标点 + 去停用词。"""
    return [w for w in jieba.lcut(text) if w.strip() and w not in PUNCT and w not in STOP]


def tokenize_all(texts):
    return [tok(t) for t in texts]


def load_corpus():
    """从 CSV 到"可检索语料"：清洗 → 丢空文本 → 拼出分组键。"""
    df = RAW.copy()
    df["text"] = df["缺陷描述"].map(clean_full)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    df["group"] = df["设备类型"] + "|" + df["缺陷类型"]
    return df


def stratified_split(df, frac=0.1, seed=42):
    """按 `设备类型|缺陷类型` 分层抽查询 —— 每组抽 round(n × frac) 条，至少 1 条。

    为什么要分层：如果随机抽，小分组可能一条都不进查询集（或全被抽走），
    指标就变成"大分组说了算"。分层保证每个分组都有自己的查询。
    """
    rng = np.random.default_rng(seed)
    picked = []
    for _, sub in df.groupby("group"):
        picked.extend(rng.choice(sub.index.to_numpy(),
                                 size=max(1, round(len(sub) * frac)), replace=False).tolist())
    picked = sorted(int(i) for i in picked)
    return df.drop(index=picked).reset_index(drop=True), df.loc[picked].reset_index(drop=True)


def bm25_score(q_tokens, tf_doc, idf, dl, avgdl, k1=1.5, b=0.75):
    """单文档 BM25 分数 —— 本章公式的**唯一真相**。

    三个部分相乘再求和：
        idf(w)                稀有词更值钱（概率型 idf）
        f·(k1+1)/(f + k1·…)   tf 饱和：一个词出现 1 次和 100 次不该差 100 倍
        分母里的 dl/avgdl     长文档天然更容易撞词，要按长度罚一下
    """
    s = 0.0
    for w in q_tokens:
        f = tf_doc.get(w, 0)
        if f:
            s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
    return s


class BM25:
    """手写 BM25：全库 idf + 逐文档 tf + 长度归一化。"""

    def __init__(self, doc_tokens, k1=1.5, b=0.75):
        self.docs = doc_tokens
        self.k1, self.b = k1, b
        self.N = len(doc_tokens)
        self.avgdl = sum(len(d) for d in doc_tokens) / self.N
        self.tf = [Counter(d) for d in doc_tokens]
        dfreq = Counter()
        for d in doc_tokens:
            dfreq.update(set(d))          # 注意：这里是 set(d)，统计的是"含该词的文档数"
        self.dfreq = dfreq
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in dfreq.items()}

    def score(self, q_tokens, i):
        return bm25_score(q_tokens, self.tf[i], self.idf,
                          len(self.docs[i]), self.avgdl, self.k1, self.b)

    def rank(self, q_tokens):
        return order_desc([self.score(q_tokens, i) for i in range(self.N)])


def weighted_score(q_tokens, doc_idx, boost_words, boost=2.0):
    """在 BM25 基础上给指定词族加权（误差归因之后才用得上的武器）。"""
    tf_doc = BM.tf[doc_idx]
    dl = len(BM.docs[doc_idx])
    s = 0.0
    for w in q_tokens:
        f = tf_doc.get(w, 0)
        if f:
            w_boost = boost if w in boost_words else 1.0
            s += w_boost * BM.idf[w] * f * (BM.k1 + 1) / (
                f + BM.k1 * (1 - BM.b + BM.b * dl / BM.avgdl))
    return s


def order_desc(scores):
    """按分数降序给出文档下标。"""
    return np.argsort(np.asarray(scores))[::-1].tolist()


def l2_normalize(matrix):
    """行向量 L2 归一化 —— 稠密向量算余弦相似度前必做（0 范数行保持 0）。"""
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm > 0)


def rrf_fuse(rank_lists, k=60, top=100):
    """Reciprocal Rank Fusion：只看名次、不看分数，所以天然免疫"两路分数量纲不同"。"""
    agg = {}
    for ranks in rank_lists:
        for pos, doc in enumerate(ranks[:top], 1):
            agg[doc] = agg.get(doc, 0.0) + 1.0 / (k + pos)
    return sorted(agg, key=lambda doc: -agg[doc])


def relevant_sets(docs, queries):
    """正样本 = 与查询同属「设备类型|缺陷类型」组的全部文档（**多正样本**）。"""
    return [set(docs.index[docs["group"] == g]) for g in queries["group"]]


def recall_at(ranks, rel_sets, k):
    return float(np.mean([len(set(r[:k]) & rel) / len(rel) for r, rel in zip(ranks, rel_sets)]))


def hit_at(ranks, rel_sets, k):
    return float(np.mean([1.0 if set(r[:k]) & rel else 0.0 for r, rel in zip(ranks, rel_sets)]))


def reciprocal_rank(ranks, rel_sets):
    return float(np.mean([next((1 / pos for pos, d in enumerate(r, 1) if d in rel), 0.0)
                          for r, rel in zip(ranks, rel_sets)]))


def ndcg_at(ranks, rel_sets, k=10):
    """nDCG@k：越靠前命中越值钱；分母是"理想排序"的 DCG。"""
    gains = []
    for r, rel in zip(ranks, rel_sets):
        dcg = sum(1 / math.log2(pos + 1) for pos, d in enumerate(r[:k], 1) if d in rel)
        idcg = sum(1 / math.log2(pos + 1) for pos in range(1, min(k, len(rel)) + 1))
        gains.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(gains))


def eval_stats(ranks, rel_sets):
    """统一口径的五个指标 —— 全部 round 到 4 位，便于对账。"""
    return {"R@5": round(recall_at(ranks, rel_sets, 5), 4),
            "P@5": round(float(np.mean([len(set(r[:5]) & rel) / 5
                                        for r, rel in zip(ranks, rel_sets)])), 4),
            "Hit@5": round(hit_at(ranks, rel_sets, 5), 4),
            "MRR": round(reciprocal_rank(ranks, rel_sets), 4),
            "nDCG@10": round(ndcg_at(ranks, rel_sets, 10), 4)}


def quadrant(doc_idx, q_idx, _=None):
    """把「文档 doc_idx 对查询 q_idx 而言」归到四个象限之一（误差归因用）。"""
    if doc_idx in REL[q_idx]:
        return "①同设备+同缺陷"
    if DOCS.iloc[doc_idx]["设备类型"] == QUERIES.iloc[q_idx]["设备类型"]:
        return "②同设备+异缺陷"
    if DOCS.iloc[doc_idx]["缺陷类型"] == QUERIES.iloc[q_idx]["缺陷类型"]:
        return "③异设备+同缺陷"
    return "④都不沾"


# ---- 全章统一的语料与索引（后面所有任务都在这套口径上比）----
DOCS, QUERIES = stratified_split(load_corpus())
DOC_TOKENS = tokenize_all(DOCS["text"])
QUERY_TOKENS = tokenize_all(QUERIES["text"])
REL = relevant_sets(DOCS, QUERIES)
BM = BM25(DOC_TOKENS)

VEC = TfidfVectorizer(tokenizer=str.split, token_pattern=None, sublinear_tf=True)
XD = VEC.fit_transform([" ".join(d) for d in DOC_TOKENS])      # 文档 TF-IDF 矩阵
XQ = VEC.transform([" ".join(q) for q in QUERY_TOKENS])        # 查询 TF-IDF 矩阵
SIM = (XQ @ XD.T).toarray()                                    # 稀疏点积 = 余弦相似度
'''

# =========================================================================== #
# 任务 1：清洗深度 —— 同一批行，两种洗法                                      #
# =========================================================================== #

T1 = '''clean_rows = []
for tag, fn in [("A 只去 HTML+台区", clean_basic), ("B +URL +去空格", clean_full)]:
    dtk = tokenize_all(DOCS["缺陷描述"].map(fn))
    qtk = tokenize_all(QUERIES["缺陷描述"].map(fn))
    engine = BM25(dtk)
    # @@todo 算出该清洗口径下的「词表大小」「平均文档长度」，并把 eval_stats 的检索指标一起收进表里
    clean_rows.append({"口径": tag,
                       "词表": len({w for d in dtk for w in d}),
                       "平均文档长度": round(sum(len(d) for d in dtk) / len(dtk), 3),
                       **eval_stats([engine.rank(q) for q in qtk], REL)})
    # @@end
clean_table = pd.DataFrame(clean_rows).set_index("口径")
print(clean_table.to_string())
# @@todo 算出 B 口径相对 A 口径的指标增量（先丢掉「平均文档长度」与「词表」两列，再 round 到 4 位）
gain = (clean_table.loc["B +URL +去空格"] - clean_table.loc["A 只去 HTML+台区"]).drop(["词表", "平均文档长度"])
clean_gain = {k: round(float(v), 4) for k, v in gain.items()}
# @@end
assert clean_table.loc["A 只去 HTML+台区", "词表"] == 226
assert clean_table.loc["B +URL +去空格", "词表"] == 70
assert round(clean_table.loc["B +URL +去空格", "平均文档长度"], 3) == 5.575
assert clean_table.loc["A 只去 HTML+台区", "MRR"] == 0.5659
assert clean_table.loc["B +URL +去空格", "MRR"] == 0.6111
print("B − A：", clean_gain)
'''

# =========================================================================== #
# 任务 2：无效占位符 —— 平均文档长度被谁拉低了                                #
# =========================================================================== #

T2 = '''bad_mask = DOCS["text"].str.strip().isin(BAD_PLACEHOLDERS).to_numpy()
# @@todo 统计索引里「无效占位符」的条数，以及剔掉它们前后两版平均文档长度
n_bad = int(bad_mask.sum())
avg_before = sum(len(d) for d in DOC_TOKENS) / len(DOC_TOKENS)
kept_tokens = [d for d, flag in zip(DOC_TOKENS, bad_mask) if not flag]
avg_after = sum(len(d) for d in kept_tokens) / len(kept_tokens)
# @@end
assert n_bad == 17
assert round(avg_before, 3) == 5.575
assert round(avg_after, 3) == 5.729
raw_bad = RAW["缺陷描述"].str.strip()
print("原始 CSV 里的无效文本：空白 %d 条 | N/A %d 条 | '-' %d 条 | '无' %d 条"
      % ((raw_bad == "").sum(), (raw_bad == "N/A").sum(), (raw_bad == "-").sum(), (raw_bad == "无").sum()))
print("进入索引的无效占位符：", DOCS.loc[bad_mask, "text"].value_counts().to_dict())
print("零 token 文档数：", sum(1 for d in DOC_TOKENS if len(d) == 0))
print(f"平均文档长度 {avg_before:.3f} → {avg_after:.3f}（+{(avg_after / avg_before - 1) * 100:.1f}%）")
'''

# =========================================================================== #
# 任务 3：手写 BM25 第一步 —— 概率型 idf                                      #
# =========================================================================== #

T3 = '''dfreq = Counter()
for d in DOC_TOKENS:
    dfreq.update(set(d))
# @@todo 按概率型 idf 公式 log(1 + (N − n + 0.5) / (n + 0.5)) 算出 idf 字典，并找出 idf 最高、最低的两个词
idf_ratio = {w: (len(DOC_TOKENS) - n + 0.5) / (n + 0.5) for w, n in dfreq.items()}
idf_manual = {w: math.log(1.0 + v) for w, v in idf_ratio.items()}
hi_word = max(idf_manual, key=lambda w: idf_manual[w])
lo_word = min(idf_manual, key=lambda w: idf_manual[w])
# @@end
assert len(idf_manual) == 70
assert all(abs(idf_manual[w] - BM.idf[w]) < 1e-12 for w in idf_manual)
assert dfreq["核实"] == 151 and dfreq["N"] == 5
assert lo_word == "核实" and hi_word in {"A", "N"}
assert round(idf_manual[hi_word], 6) == 4.570014
print(f"词表 {len(idf_manual)} 个词 | N = {len(DOC_TOKENS)} 篇文档")
print(f"idf 最高的词：{sorted([w for w in idf_manual if idf_manual[w] == idf_manual[hi_word]])}"
      f" = {idf_manual[hi_word]:.6f}（df={dfreq[hi_word]}）"
      f" | max() 选中了 {hi_word!r}，这取决于「谁先被遇到」")
print(f"idf 最低：{lo_word} = {idf_manual[lo_word]:.6f}（df={dfreq[lo_word]}）")
'''

# =========================================================================== #
# 任务 4：手写 BM25 第二步 —— 给一条查询的一条文档打分                        #
# =========================================================================== #

T4 = '''i0 = 0
top_doc_0 = BM.rank(QUERY_TOKENS[i0])[0]
dl0 = len(DOC_TOKENS[top_doc_0])
tf0 = BM.tf[top_doc_0]
words0 = sorted(set(QUERY_TOKENS[i0]) & set(tf0))
# @@todo 手工复算 top_doc_0 对第 0 条查询的 BM25 分数（idf × tf 饱和 × 长度归一化，逐词相加）
denom0 = {w: tf0[w] + BM.k1 * (1 - BM.b + BM.b * dl0 / BM.avgdl) for w in words0}
contrib = [BM.idf[w] * tf0[w] * (BM.k1 + 1) / denom0[w] for w in words0]
score_manual = sum(contrib)
# @@end
assert abs(score_manual - BM.score(QUERY_TOKENS[i0], top_doc_0)) < 1e-9
assert round(score_manual, 4) == 10.7924
assert top_doc_0 == 14 and DOCS.iloc[top_doc_0]["工单编号"] == "WO_0016"
print("查询：", QUERIES.iloc[i0]["text"], "| 真实分组：", QUERIES.iloc[i0]["group"])
print(f"榜首文档 {DOCS.iloc[top_doc_0]['工单编号']}（{DOCS.iloc[top_doc_0]['group']}）"
      f" 得分 {score_manual:.4f} | 文档长度 {dl0} | 全库平均长度 {BM.avgdl:.3f}")
print("逐词贡献：", {w: round(c, 3) for w, c in zip(words0, contrib)})
'''

# =========================================================================== #
# 任务 5：分层切分与正样本规模                                                #
# =========================================================================== #

T5 = '''# @@todo 统计每条查询的正样本数（同分组的文档条数）的 min / 中位 / max / 平均
sizes = np.array([len(r) for r in REL])
size_min, size_med, size_max, size_avg = int(sizes.min()), int(np.median(sizes)), int(sizes.max()), round(float(sizes.mean()), 2)
# @@end
assert (len(DOCS), len(QUERIES)) == (530, 58)
assert DOCS["group"].nunique() == 40
assert (size_min, size_med, size_max) == (6, 13, 23)
assert size_avg == 14.09
print(f"语料 {len(DOCS)} 篇文档 / {len(QUERIES)} 条查询 / {DOCS['group'].nunique()} 个分组")
print(f"每条查询的正样本数：min {size_min} | 中位 {size_med} | max {size_max} | 平均 {size_avg}")
print("正样本 ≥ 5 的查询占比：%.4f" % float((sizes >= 5).mean()))
'''

# =========================================================================== #
# 任务 6：四条检索路线                                                        #
# =========================================================================== #

T6 = '''SVD = TruncatedSVD(n_components=64, random_state=42)
# @@todo 造出四条路线的完整排序：BM25 / TF-IDF 稀疏向量 / SVD(64) 稠密向量 / RRF 融合
bm_rank = [BM.rank(q) for q in QUERY_TOKENS]
vec_rank = [order_desc(row) for row in SIM]
zd = l2_normalize(SVD.fit_transform(XD))
zq = l2_normalize(SVD.transform(XQ))
svd_rank = [order_desc(row) for row in zq @ zd.T]
rrf_rank = [rrf_fuse([bm_rank[i], vec_rank[i]]) for i in range(len(QUERIES))]
# @@end
assert BM.N == 530 and len(bm_rank) == 58 and len(bm_rank[0]) == 530
route_table = pd.DataFrame([{"方法": tag, **eval_stats(rk, REL)}
                            for tag, rk in [("BM25", bm_rank), ("TF-IDF 向量", vec_rank),
                                            ("SVD 向量(64)", svd_rank), ("RRF 融合", rrf_rank)]]).set_index("方法")
print(route_table.to_string())
'''

# =========================================================================== #
# 任务 7：指标拆解 —— R@5 为什么这么低                                       #
# =========================================================================== #

T7 = '''cover_rows = []
for k in (1, 3, 5, 10, 20):
    # @@todo 算出实测 R@k / Hit@k，以及 R@k 的「理论上限」（正样本被截断到 k 条时的最大 Recall）
    cover_rows.append({"k": k,
                       "R@k": round(recall_at(bm_rank, REL, k), 4),
                       "上限": round(float(np.mean([min(k, len(r)) / len(r) for r in REL])), 4),
                       "Hit@k": round(hit_at(bm_rank, REL, k), 4)})
    # @@end
cover_table = pd.DataFrame(cover_rows).set_index("k")
cover_table["占上限%"] = (cover_table["R@k"] / cover_table["上限"] * 100).round(1)
print(cover_table.to_string())
'''

# =========================================================================== #
# 任务 8：SVD 降到多少维才不亏                                                #
# =========================================================================== #

T8 = '''var_rows = []
for nc in (16, 32, 64):
    # @@todo 用 nc 个主成分拟合 SVD，取出解释方差比之和（保留 6 位小数）
    svd = TruncatedSVD(n_components=nc, random_state=42).fit(XD)
    var_rows.append({"维度": nc, "解释方差比": round(float(svd.explained_variance_ratio_.sum()), 6)})
    # @@end
var_table = pd.DataFrame(var_rows).set_index("维度")
print(var_table.to_string())
assert var_table.loc[64, "解释方差比"] == 1.0
assert var_table.loc[16, "解释方差比"] == 0.588147
'''

# =========================================================================== #
# 任务 9：BM25 的 k1 / b 扫描                                                 #
# =========================================================================== #

T9 = '''sweep_rows = []
for k1 in (0.9, 1.5, 2.0):
    for b in (0.0, 0.5, 0.75, 1.0):
        # @@todo 用该 (k1, b) 重建 BM25、给全部查询排序，把 Hit@5 与 MRR 收进表里
        eng = BM25(DOC_TOKENS, k1, b)
        st = eval_stats([eng.rank(q) for q in QUERY_TOKENS], REL)
        sweep_rows.append({"k1": k1, "b": b, "Hit@5": st["Hit@5"], "MRR": st["MRR"]})
        # @@end
sweep_table = pd.DataFrame(sweep_rows).set_index(["k1", "b"])
print(sweep_table.to_string())
assert sweep_table["Hit@5"].max() == 0.8103
assert sweep_table["MRR"].max() == 0.6611
'''

# =========================================================================== #
# 任务 10：融合与重排的收益                                                  #
# =========================================================================== #

T10 = '''k_rows = []
for k_rrf in (10, 30, 60, 100):
    # @@todo 用该 k 融合 BM25 与 TF-IDF 两路排序，算出 Hit@5 与 MRR
    st = eval_stats([rrf_fuse([bm_rank[i], vec_rank[i]], k_rrf) for i in range(len(QUERIES))], REL)
    k_rows.append({"RRF k": k_rrf, "Hit@5": st["Hit@5"], "MRR": st["MRR"]})
    # @@end
print(pd.DataFrame(k_rows).set_index("RRF k").to_string())

rerank_rows = []
for w_vec in (0.0, 0.3, 0.5, 0.7, 1.0):
    reranked = []
    for i in range(len(QUERIES)):
        cand = bm_rank[i][:50]
        b_max = max(BM.score(QUERY_TOKENS[i], d) for d in cand) or 1.0
        v_max = max(SIM[i][d] for d in cand) or 1.0
        # @@todo 把两路分数各自归一化到 [0,1] 后按向量权重 w_vec 加权，重排 top-50 候选，再接上第 50 名之后的原序
        key = lambda d: -(w_vec * SIM[i][d] / v_max + (1 - w_vec) * BM.score(QUERY_TOKENS[i], d) / b_max)
        reranked.append(sorted(cand, key=key) + bm_rank[i][50:])
        # @@end
    rerank_rows.append({"向量权重": w_vec, **eval_stats(reranked, REL)})
rerank_table = pd.DataFrame(rerank_rows).set_index("向量权重")
print(rerank_table.to_string())
assert rerank_table["MRR"].max() == 0.6321
assert rerank_table.loc[0.0, "MRR"] == 0.6111
'''

# =========================================================================== #
# 任务 11：误差归因 —— top-5 里的错，错在哪一类                               #
# =========================================================================== #

T11 = '''attr_rows = []
for tag, ranks in [("BM25", bm_rank), ("TF-IDF 向量", vec_rank)]:
    labels = []
    for i, r in enumerate(ranks):
        # @@todo 用 quadrant() 把该查询 top-5 里的每个文档翻译成象限标签，追加进 labels
        labels.extend(quadrant(d, i) for d in r[:5])
        # @@end
    cnt = Counter(labels)
    tot = sum(cnt.values())
    attr_rows.append({"方法": tag, **{k: round(v / tot * 100, 1) for k, v in sorted(cnt.items())}})
attr_table = pd.DataFrame(attr_rows).set_index("方法")
print(attr_table.to_string())
assert attr_table.loc["BM25", "③异设备+同缺陷"] == 40.0
assert attr_table.loc["BM25", "④都不沾"] == 1.7
'''

# =========================================================================== #
# 任务 12：定向改进 —— 拿误差归因换指标                                       #
# =========================================================================== #

T12 = '''improved_rows = [{"策略": "基线 BM25", **eval_stats(bm_rank, REL)}]

dev_filter_rank = []
for i in range(len(QUERIES)):
    dev = QUERIES.iloc[i]["设备类型"]
    # @@todo 把同设备类型的文档按原 BM25 名次提到前面，异设备文档整体接在后段
    same_dev = [j for j in bm_rank[i] if DOCS.iloc[j]["设备类型"] == dev]
    other_dev = [j for j in bm_rank[i] if DOCS.iloc[j]["设备类型"] != dev]
    dev_filter_rank.append(same_dev + other_dev)
    # @@end
improved_rows.append({"策略": "设备元数据硬过滤", **eval_stats(dev_filter_rank, REL)})

two_stage_rank = []
for i in range(len(QUERIES)):
    dev = QUERIES.iloc[i]["设备类型"]
    cand = [j for j in range(len(DOCS)) if DOCS.iloc[j]["设备类型"] == dev]
    # @@todo 在同设备候选上按「缺陷词 ×2」重新打分排序，其余文档接在后段
    scores = [weighted_score(QUERY_TOKENS[i], j, TYP_WORDS, 2.0) for j in cand]
    ordered = [cand[p] for p in np.argsort(scores)[::-1]]
    rest = [j for j in range(len(DOCS)) if j not in set(ordered)]
    two_stage_rank.append(ordered + rest)
    # @@end
improved_rows.append({"策略": "硬过滤 + 缺陷词 ×2", **eval_stats(two_stage_rank, REL)})
improved_table = pd.DataFrame(improved_rows).set_index("策略")
print(improved_table.to_string())
assert improved_table.loc["硬过滤 + 缺陷词 ×2", "MRR"] == 0.9345
assert improved_table.loc["硬过滤 + 缺陷词 ×2", "Hit@5"] == 0.9655
'''

# =========================================================================== #
# 任务 13：拒答 —— 检索不到就别硬答                                            #
# =========================================================================== #

T13 = '''GENERIC = ["今天长沙天气怎么样", "帮我订一张去北京的机票", "这道数学题怎么解",
           "食堂中午几点开饭", "如何申请年假", "推荐一部好看的电影"]
TOPICAL = ["变压器的工作原理是什么", "电力设备的保养周期是多久", "绝缘子的国家标准是什么",
           "如何填写工单编号", "断路器的作用有哪些", "母线电压等级怎么选"]

hit_top = [BM.score(QUERY_TOKENS[i], bm_rank[i][0]) for i in range(len(QUERIES))]
gen_top = []
for text in GENERIC:
    # @@todo 给这条「完全无关」的查询排序，取榜首分数追加进 gen_top
    q_tokens = tok(text)
    gen_top.append(BM.score(q_tokens, BM.rank(q_tokens)[0]))
    # @@end
topical_top = []
for text in TOPICAL:
    # @@todo 给这条「含语料词但意图不同」的查询排序，取榜首分数追加进 topical_top
    q_tokens = tok(text)
    topical_top.append(BM.score(q_tokens, BM.rank(q_tokens)[0]))
    # @@end

score_rows = []
for tag, group in [("真命中查询", hit_top), ("完全无关", gen_top), ("含语料词但意图不同", topical_top)]:
    score_rows.append({"查询类型": tag, "min": round(min(group), 3),
                       "中位": round(float(np.median(group)), 3), "max": round(max(group), 3)})
print(pd.DataFrame(score_rows).set_index("查询类型").to_string())

thr_rows = []
for thr in (1.0, 2.0, 3.0, 4.0):
    # @@todo 统计该阈值下「真命中被保留 / 无关查询被拒 / 沾边查询被拒」的条数
    thr_rows.append({"阈值": thr,
                     "真命中保留": f"{sum(1 for s in hit_top if s >= thr)}/{len(hit_top)}",
                     "无关拒绝": f"{sum(1 for s in gen_top if s < thr)}/{len(gen_top)}",
                     "沾边拒绝": f"{sum(1 for s in topical_top if s < thr)}/{len(topical_top)}"})
    # @@end
print(pd.DataFrame(thr_rows).set_index("阈值").to_string())
assert max(gen_top) == 0.0
assert round(float(np.median(hit_top)), 3) == 10.805
assert sum(1 for s in topical_top if s < 4.0) == 6
assert sum(1 for s in topical_top if s < 3.0) == 4
'''

# =========================================================================== #
# 任务 14：prompt 拼装与引用                                                  #
# =========================================================================== #

T14 = '''i_p = 0
picked = bm_rank[i_p][:3]
# @@todo 拼出带编号引用的 prompt：参考资料每条形如「[1] 工单 WO_0016｜设备：绝缘子｜缺陷：渗漏油｜等级：危急」，
#      再附上该条的描述、用户问题、以及「只依据资料作答 + 标注引用编号 + 资料不足说无法确定」的作答要求
ref_lines = [f"[{pos}] 工单 {DOCS.iloc[d]['工单编号']}｜设备：{DOCS.iloc[d]['设备类型']}｜缺陷：{DOCS.iloc[d]['缺陷类型']}｜等级：{DOCS.iloc[d]['缺陷等级']}" for pos, d in enumerate(picked, 1)]
desc_lines = [f"    描述：{DOCS.iloc[d]['text']}" for d in picked]
ref_blocks = [f"{a}\\n{b}" for a, b in zip(ref_lines, desc_lines)]
prompt = "\\n".join(["【参考资料】", *ref_blocks, f"【问题】{QUERIES.iloc[i_p]['text']}", "【要求】仅依据以上参考资料作答，并标注引用编号（如 [1]）；资料不足则回答「无法确定」。"])
# @@end
assert prompt.startswith("【参考资料】")
assert "[1]" in prompt and "[3]" in prompt
assert "WO_0016" in prompt
assert "无法确定" in prompt
print(prompt)
print("\\n--- 检索到的 top-3 与真实分组的关系 ---")
for pos, d in enumerate(picked, 1):
    print(f"  [{pos}] {DOCS.iloc[d]['group']} | 与查询同分组：{d in REL[i_p]}")
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """# ch10 RAG 检索增强

## 这一章要解决什么

前九章我们把文本**表示**了（ch02）、把 Transformer **搭**起来了（ch04/05）、把模型**微调**了（ch07），
还学会了在动手前把**规模账**算清楚（ch09）。

但真到了业务场景里，最常见的问题不是"模型不够大"，而是：

> 「现场班组写了一句缺陷描述，系统能不能从 600 条历史工单里，
> 找出**同类型**的案例给他参考？」

这就是 **RAG**（Retrieval-Augmented Generation，检索增强生成）。
它的第一段 —— **检索（Retrieval）** —— 决定了后面所有环节的上限：
检索回来的资料不对，再强的模型也只能胡编。

而检索恰恰是**不需要 GPU、不需要 API、完全可度量**的一段。
这是本章选它做主线的原因。

## RAG 的三段式，以及本章的离线边界

| 段 | 做什么 | 本章怎么处理 |
|---|---|---|
| **R** Retrieve | 从知识库里找最相关的 N 条 | **本章主体**：BM25 / 向量 / 融合 / 重排 / 拒答 |
| **A** Augment | 把资料拼进 prompt（带编号引用） | 任务 14：模板式拼装，完全可复现 |
| **G** Generate | 调模型生成答案 | **不调任何 API** —— 换成"引用式作答"的模板输出 |

> **为什么要砍掉 G？** 因为 G 的评价需要人或另一个模型，不可复现也不可对账。
> 而 R 有**标准答案**（正样本），有**标准指标**（Recall / MRR / nDCG），
> 可以做得跟 ch09 的规模账一样硬：**先算、再跑、再解释**。

## 本章的四条实测结论（先看结论，再走过程）

| # | 动作 | MRR 变化 | Hit@5 变化 |
|---|---|---|---|
| ① | **加深清洗**（去 URL、去字间空格） | 0.5659 → **0.6111**（+0.045） | 0.7069 → **0.7759**（+0.069） |
| ② | **误差归因后的定向改进**（设备过滤 + 缺陷词加权） | 0.6111 → **0.9345**（+0.323） | 0.7759 → **0.9655**（+0.190） |
| ③ | 换算法 / 融合 / 重排 | ±0.02 | ±0.03 |
| ④ | 剔掉索引里的无效占位符 | −0.004 ~ +0.02 | +0.017 |

**一句话**：**先修数据，再看误差归因，最后才轮到调算法。**
这跟很多人的直觉顺序正好相反 —— 直觉是"效果不好就换个新模型"。

## 本章地图

| 任务 | 内容 | 收获 |
|---|---|---|
| 1~2 | 清洗深度实验 + 无效占位符 | 数据质量的杠杆有多大 |
| 3~4 | 手写 BM25（idf + 打分） | 公式拆开看得见 |
| 5 | 分层切分与正样本规模 | 评测口径先行 |
| 6 | 四条路线对比 | 稀疏 / 稠密 / 融合 |
| 7 | 指标拆解（R@k 上限） | **R@5 低 ≠ 检索差** |
| 8~10 | SVD 维度 / 参数扫描 / 融合重排 | 调参的收益有多大 |
| 11 | 误差归因四象限 | 从"指标不好"到"错在哪" |
| 12 | 定向改进 | 归因换指标（本章高潮） |
| 13 | 拒答阈值 | 检索不到就别硬答 |
| 14 | prompt 拼装与引用 | RAG 的 A 段 |
"""
    ),
    code(IMPORTS),
    md(
        """## 脚手架：口径的唯一真相

这一章所有任务都在**同一套语料、同一套评测口径**上比较，否则数字没法比。
所以清洗函数、分词器、BM25 类、四个指标、正样本定义全部放在脚手架里。

三件事要先记住：

1. **语料是同一批**：`DOCS` / `QUERIES` 由 `clean_full` + 分层切分确定，后面不再改动
   （除了任务 1 故意换清洗函数做对照）。
2. **正样本是"分组"**：与查询同属 `设备类型|缺陷类型` 组的**全部**文档。
   所以这是**多正样本**检索 —— 这一点会在任务 7 让 `R@5` 看起来很惨。
3. **指标全部 round 到 4 位**，方便断言对账。

> 顺带记住 `RRF`（Reciprocal Rank Fusion）为什么好用：
> 它只用**名次**不用分数，所以 BM25 的 0~20 分和 TF-IDF 的 0~1 分可以直接融合，
> 不存在"要不要先归一化"的扯皮。
"""
    ),
    code(SCAFFOLD),
    md(
        """## 任务 1：清洗深度 —— 同一批行，两种洗法

**实验设计比结果更重要。** 这个对比必须是**受控**的：同一批文档行、同一批查询行，
只有"清洗函数"这一个变量在变。

> 反例（本章不采用）：如果先按 A 口径切分、再按 B 口径切分，
> 两次的查询集就不一样了，指标差多少根本说不清。
> **换语料必须固定评测集，这是评测纪律。**
"""
    ),
    code(T1),
    md(
        """**读表重点**：词表从 **226 掉到 70**（砍掉 69%），平均文档长度 6.725 → 5.575，指标**全面提升**。

| 指标 | A 口径 | B 口径 | 增量 |
|---|---|---|---|
| R@5 | 0.0795 | **0.1013** | +0.0218 |
| P@5 | 0.2276 | **0.2897** | +0.0621 |
| Hit@5 | 0.7069 | **0.7759** | +0.0690 |
| MRR | 0.5659 | **0.6111** | +0.0452 |
| nDCG@10 | 0.2122 | **0.2750** | +0.0628 |

**为什么清洗能让指标变好？** 因为噪声词在"稀释"信号词：

- `http` / `10.0` / `wo` 这些 URL 碎片进了词表，它们**每个文档都有**（82 条带 URL），
  idf 很低，但它们挤占了 `tf` 的位置，把 BM25 的长度归一化项撑大；
- 更糟的是**字间空格**（`断路 器`、`核 实后 反 馈`）：`tok()` 会把它们切成
  `断路` + `器` 这种**碎片词**，真正的信号词 `断路器` 反而匹配不上；
- 词表从 226 缩到 70，说明有 **156 个词是噪声**（碎片词、URL 片段），
  它们既不能召回，又拉高了 `dfreq`，把真实的 idf 结构搅乱。

> **工程含义**：在这次实验里，**把清洗做深一点的收益（MRR +0.045）
> 比换一个检索算法（±0.02）大得多**。
> 很多人一上手就纠结"用 BM25 还是向量"，其实先该问的是"我的语料洗干净了吗"。
>
> 但也要注意：这 156 个噪声词里有 156 − 70 = 86 个是**只在 A 口径里存在**的碎片；
> 剩下的差异来自 `dfreq` 的变化。**数字对得上账，结论才站得住。**
"""
    ),
    md(
        """## 任务 2：无效占位符 —— 平均文档长度被谁拉低了

清洗做完了，还有一种"脏"是**清洗洗不掉**的：`N/A`、`-`、`无` 这类**无效占位符**。
它们不是噪声，而是"这条记录压根没有描述"。

这类数据在 ch01 讲过（有效性过滤），但那是**行级**的视角。
放到检索里，它有一个很隐蔽的副作用：**污染 `avgdl`**。
"""
    ),
    code(T2),
    md(
        """**读表重点**：

- 原始 CSV 里 30 条无效文本（空白 12 / `N/A` 6 / `-` 6 / `无` 6），
  空白那 12 条被 `len(text) > 0` 拦掉了，**剩下 18 条混进了语料** ——
  其中 17 条落进索引、1 条被抽成了查询（`N/A`）。
- 索引里 **6 条 `-` 文档的 token 数是 0**（`-` 是标点，被 `tok()` 过滤干净），
  另外 11 条是 `N/A`（→ `N`、`A`）和 `无`（→ `无`）。
- 剔掉这 17 条之后，平均文档长度 **5.575 → 5.729**（+2.8%）。

**为什么要盯着 `avgdl`？** 因为 BM25 的长度归一化项是 `dl / avgdl`：

```
分母 = f + k1 · (1 − b + b · dl / avgdl)
```

`avgdl` 变小 → 每篇文档的 `dl / avgdl` 都变大 → **全库的分数被整体压低**（相当于把 `b` 调大了）。
这不是"某几条文档受影响"，而是**所有文档的分数都偏移了**。

> **实测结论（别过度反应）**：虽然 `avgdl` 变了 2.8%，但剔除这些文档后
> top-5 指标基本没动（Hit@5 0.7759 → 0.7931，MRR 0.6111 → 0.6073）。
> 原因很直接：**零 token 的文档得分恒为 0，本来就排不到前十**。
>
> 所以两条结论要分开记：
> ① **索引侧的脏数据**主要影响 `avgdl` 这类全局统计量，对 top-k 影响有限；
> ② **查询侧的脏数据**（一条 `N/A` 查询）会直接让一次检索**彻底失效** ——
> 这才是任务 13 要处理的"拒答"问题。
"""
    ),
    md(
        """## 任务 3：手写 BM25 第一步 —— 概率型 idf

BG: BM25 常被当成"一个能调的库"，但它的公式只有三块，
先自己算一遍，以后调参才知道在调什么。

idf 用的是**概率型（probabilistic）**版本，不是教科书里那个 `log(N / n)`：

```
idf(w) = log(1 + (N − n + 0.5) / (n + 0.5))
```

那个 `+1` 和两个 `0.5` 不是装饰：
- `+0.5` 是**平滑**，保证 `n = N`（词在所有文档出现）时 idf 仍为正；
- 外层 `log(1 + ·)` 保证 idf **恒 ≥ 0** —— 而 `log(N/n)` 在 `n > N/2` 时会变负，
  负 idf 意味着"词越常见分数越低到负无穷"，会反过来把排序搞乱。
"""
    ),
    code(T3),
    md(
        """**读表重点**：idf 最高的是 **`N` / `A`**（都是 4.570014，df = 5），最低的是 **`核实`**（1.254176，df = 151）。

这里有**两条**值得记的：

1. **最高 idf 的两个词居然是 `N` 和 `A`** —— 它们来自 `N/A` 占位符，
   不是任何领域术语。idf 只认统计，不认语义：
   **一个没意义的词只要够稀有，就能拿到高 idf**。
   这也是为什么词表清洗（把 `N/A` 这类占位符处理掉）比调 idf 参数更重要。
2. **`核实` 的 df = 151**（151 篇文档里有它），占了全语料的 28%，
   所以 idf 只有 1.25。而 `缺口`（df = 13）能拿到 3.67 —— **差了近 3 倍**。
   这个差距就是 BM25 能工作的核心：**稀有词 = 判别力**。

> **对账习惯**：算完之后 `assert` 一下与脚手架 `BM.idf` 完全一致（浮点误差 < 1e-12）。
> 自己推的公式必须与线上口径**逐位对账**，否则"实现了两套 idf"这种坑迟早会踩。

> ⚠️ **一个真实的复现性陷阱**：`N` 和 `A` 的 idf **完全相等**（4.570014），
> 而 `max(dict, key=...)` 在并列时返回的是**"先被遇到的那个"**。
> `dfreq` 是 `dfreq.update(set(d))` 累加出来的，**`set` 的迭代顺序受字符串哈希随机化
> （`PYTHONHASHSEED`）影响** —— 所以同一个脚本、同一份数据，**换个进程跑，
> `max()` 选出来的可能是 `N`，也可能是 `A`**。
>
> 所以这段断言写的是 `hi_word in {"A", "N"}` 而不是 `hi_word == "N"`。
> **凡是"取最大/最小"的地方，都要问一句：并列时怎么办？**
> 生产代码里应该定一个 secondary key（比如按词本身排序），
> 否则"同样的输入，两次跑出不同的 top-1"会成为排查噩梦。
"""
    ),
    md("## 任务 4：手写 BM25 第二步 —— 给一条文档打分\n\n公式的另外两块：**tf 饱和** 与 **长度归一化**。"),
    code(T4),
    md(
        """**读表重点**：榜首 `WO_0016` 得分 **10.7924**，手工复算与 `BM.score()` 完全一致。

逐词贡献那一行很说明问题（查询 `绝缘子渗油,请核实后反馈,请安排抢修`，
命中榜首文档 `绝缘子渗油,请安排抢修`，共 4 个词）：

| 词 | idf | 对该文档的贡献 | 占比 |
|---|---|---|---|
| `渗油` | 3.2067（df 21） | **3.674** | 34.0% |
| `绝缘子` | 2.2584（df 55） | 2.587 | 24.0% |
| `抢修` | 1.9775（df 73） | 2.266 | 21.0% |
| `安排` | 1.9775（df 73） | 2.266 | 21.0% |

两点：

1. **idf 排序 ≠ 直觉排序**。缺陷词 `渗油` 的 df 只有 21（最稀有），拿到最高 idf 3.21；
   而动作词 `抢修` / `安排` 的 df 是 73 —— **它们才是"模板词"**（5 个 ACTION 模板轮流出现）。
   但注意：`核实`（df 151）和 `反馈`（df 102）在**这条文档里没出现**，所以贡献是 0 ——
   它们是"**别的文档**里的噪声"，会拉高别的文档的分数。
2. **模板词的占比并不小**：`抢修` + `安排` 两条合起来贡献 **42%** 的分数。
   也就是说，**一次检索里有四成分数是"所有同类工单都会有的套话"** ——
   这就是任务 11/12 要处理的噪声源。

**再看 `tf` 饱和**（`f·(k1+1) / (f + k1·…)`）：

| f（词频） | 系数 | 相对 f=1 |
|---|---|---|
| 1 | (k1+1)/(1+k1) = 1.00 | 1× |
| 2 | 2(k1+1)/(2+k1) = 1.43 | 1.43× |
| 5 | 5(k1+1)/(5+k1) = 1.85 | 1.85× |
| 100 | 100(k1+1)/(100+k1) = 2.44 | 2.44× |

**出现 100 次只比出现 1 次强 2.44 倍** —— 这就是"饱和"。
它是为了防止"关键词堆砌"（老式 SEO 的做法）把排序刷爆。
`k1` 控制饱和速度（越大越慢饱和），`b` 控制长度惩罚强度 —— 任务 9 会扫一遍。

> ⚠️ **一个易踩的坑**：如果要打印"逐词贡献"，
> 必须让**词表和贡献值来自同一个有序列表**（代码里的 `words0`）。
> 如果一边用 `sorted(set(...))` 一边用 `set(...)`，
> 两边的顺序不一致，打印出来的就是**错位的对照表** ——
> 会得出"模板词 idf 最高"这种完全相反的结论。
> 自己第一次跑就踩了这个坑，靠 `sum(contrib)` 对得上 10.7924 才发现。
"""
    ),
    md(
        """## 任务 5：分层切分与正样本规模

检索评测的第一步不是"跑模型"，是**把评测集定下来**。

本章的做法：按 `设备类型|缺陷类型` **分层**，每组抽 10% 当查询，其余进索引。
"""
    ),
    code(T5),
    md(
        """**读表重点**：530 篇文档 / 58 条查询 / 40 个分组；
每条查询的正样本数 **min 6、中位 13、max 23、平均 14.09**。

**为什么必须分层？** 如果全局随机抽 10%，58 条查询里可能一条都抽不到某些小分组
（`母线|锈蚀` 全语料只有 7 条），那这些小分组等于**没被评测** ——
指标只反映大分组的表现。分层抽样保证**每个分组都有自己的查询**。

**为什么要固定这个切分？** 因为后面每个任务都在比"同一个指标在改了什么之后变成多少"。
切分一变，所有历史数字作废 —— 这就是任务 1 说的评测纪律。

> **一个重要的事实**：这里的正样本数是 **6~23 条**，不是 1 条。
> 这会让 `R@5` 这个指标**看起来很惨**（天花板只有 5/13 ≈ 0.38），
> 下一节任务 7 专门拆这件事。
"""
    ),
    md(
        """## 任务 6：四条检索路线

现在把四条路线摆在一起。这是本章唯一一次"换算法"的实验 ——
注意看它带来的收益，和任务 1 的清洗收益比一比。
"""
    ),
    code(T6),
    md(
        """**读表重点**：

| 方法 | R@5 | P@5 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| BM25 | 0.1013 | 0.2897 | 0.7759 | 0.6111 | 0.2750 |
| **TF-IDF 向量** | **0.1135** | **0.3207** | **0.8448** | **0.6309** | **0.2973** |
| SVD 向量(64) | 0.1135 | 0.3207 | 0.8448 | 0.6223 | 0.2954 |
| RRF 融合 | 0.1072 | 0.3069 | 0.8103 | 0.6173 | 0.2829 |

三条观察，每条都能直接用：

1. **换算法的收益很小**：BM25（0.6111）vs TF-IDF（0.6309），MRR 只差 **0.0198**；
   而任务 1 的清洗带来 **+0.0452**。**清洗的收益是换算法的 2 倍多。**
2. **SVD(64) 与 TF-IDF 的 R@5 / P@5 / Hit@5 完全相同**，只有 MRR / nDCG 略降。
   原因是任务 8 要量的事实：这门语料的**词表只有 70 个词**，
   降到 64 维几乎是**无损压缩**，所以稠密向量等于白折腾了一遍（还多花了算力）。
   > **推论**：向量检索的优势要在**大词表**（真实语料几万到几十万词）上才体现出来；
   > 小词表上，稀疏的精确匹配反而更稳。
3. **RRF 融合反而比单路差**（0.6173 < 0.6309）。原因是两路高度相关
   （同一个 TF-IDF 词空间），融合没有带来新信息，只带来"两个排序互相拉平"的副作用。
   > **能用 RRF 的前提是两路**互补**（比如一路词法 + 一路语义 embedding），
   > 而不是同一套特征的两种算法。
"""
    ),
    md(
        """## 任务 7：指标拆解 —— R@5 为什么这么低

现在解释上一节那个"看起来很惨"的 `R@5 = 0.1013`。
这不是检索差，是**多正样本**下的**指标天花板**问题。
"""
    ),
    code(T7),
    md(
        """**读表重点**：

| k | R@k | 上限 | 占上限 | Hit@k |
|---|---|---|---|---|
| 1 | 0.0328 | 0.0760 | 43.2% | 0.4655 |
| 3 | 0.0766 | 0.2279 | 33.6% | 0.7241 |
| 5 | 0.1013 | 0.3798 | 26.7% | 0.7759 |
| 10 | 0.1635 | 0.7324 | 22.3% | 0.9138 |
| 20 | 0.2332 | 0.9955 | 23.4% | 0.9138 |

怎么理解这张表：

- **上限怎么来的**：每组的正样本中位数是 13 条，`R@5` 最多只能召回 5 条 →
  上限就是 `5/13 ≈ 0.38`。**R@5 = 0.1013 意味着"达到了上限的 26.7%"**，
  而不是"只对了 10%"。
- **同一件事，`Hit@5 = 0.7759`**：58 条查询里 45 条的前 5 名里**至少有一条**正确答案。
  这就是"R@5 低 ≠ 检索差"的直接证据。
- **越往后越难提升**：`Hit@10 = Hit@20 = 0.9138` —— 排名 10 以后**再也没有新命中**了。
  换句话说，**召回瓶颈在"排到第 10 名之前"**，再往后看没用。

> **实用判据**：
> - 正样本只有 1 条（大多数问答场景）→ 看 `MRR` / `Hit@1`；
> - 正样本多（案例推荐、文档检索）→ 看 `Hit@k` / `nDCG@k`，`R@k` 只作参考；
> - 报指标时**必须同时报上限**，否则别人会以为你的系统很差。
"""
    ),
    md("## 任务 8：SVD 降到多少维才不亏\n\n上一节发现 SVD(64) 与 TF-IDF 指标几乎一样。为什么？量一下解释方差。"),
    code(T8),
    md(
        """**读表重点**：16 维只能解释 **0.588147** 的方差，32 维到 **0.865843**，64 维 **1.000000**。

**64 维就是满秩** —— 因为词表只有 70 个词，`TruncatedSVD(64)` 把整个矩阵压完了，
一个信息都没丢。所以它的 R@5 / P@5 / Hit@5 与原始 TF-IDF **逐位相同**。

`n_components` 的经验取法：**取到解释方差比 ≥ 0.9 就够**（这里 32 维已到 0.87，40 维左右可到 0.9）。
再往上加维度，收益递减、算力线性增长。

> **反过来说**：如果这是一份真实语料（词表 5 万 +），64 维只能解释百分之几的方差，
> 那时降维就真的会掉指标 —— 这时候该上的不是 SVD，而是**预训练句向量**。
> **方法是跟着数据规模走的，不是跟着"最新论文"走的。**
"""
    ),
    md("## 任务 9：BM25 的 k1 / b 扫描\n\n两个参数各管一件事：`k1` 管 tf 饱和速度，`b` 管长度惩罚强度。"),
    code(T9),
    md(
        """**读表重点**（12 组）：

| 观察 | 数据 |
|---|---|
| `b` 越大 → `MRR` 越高 | `b` 从 0.0 → 1.0，MRR 逐级上升（`k1=2.0` 时 0.5730 → **0.6611**） |
| `b = 0.5` 时 `Hit@5` 最好 | 出现 **0.8103**（`k1=0.9` 和 `k1=1.5` 都是） |
| `k1` 影响很小 | 同一个 `b` 下，`k1` 从 0.9 变到 2.0，MRR 只动 0.001~0.03 |

**为什么 `b` 的影响这么大？** 因为这门语料的文档长度**差异悬殊**：
无效占位符文档长度是 0~1，正常文档 5~8 个词，而查询本身也有长有短。
长度惩罚越强（`b` → 1），短文档越占优势 —— 而短文档恰好更可能是"精准命中"。

**为什么 `Hit@5` 和 `MRR` 的最优点不在同一处？** 因为两者关心的东西不同：
- `Hit@5` 只问"有没有命中"，排第 1 还是第 5 无所谓 → 更喜欢**广度**；
- `MRR` 关心"命中得多靠前" → 更喜欢**精度**。

> **结论**：`k1 = 1.5, b = 0.75` 这个"默认值"在这份数据上**不是最优的**
> （`b = 1.0` 的 MRR 是 0.6611，比默认的 0.6111 高 0.05）。
> 但它也不是"错"的 —— 如果没有时间扫参，默认值是个安全的起点；
> **如果有 12 组网格的时间，就该扫**，因为收益与"换算法"量级相当，且成本几乎为零。
>
> ⚠️ **但注意：这就是任务 6 说的"调算法"路线，天花板在 +0.05 量级。
> 别把调参当成救命稻草 —— 任务 11/12 那条路能给你 +0.32。**
"""
    ),
    md("## 任务 10：融合与重排的收益\n\n把任务 6/9 的“调算法”路线走到底：融合权重、重排权重都扫一遍。"),
    code(T10),
    md(
        """**读表重点**：

**RRF 的 `k` 完全不影响结果**（10 / 30 / 60 / 100 四项指标一模一样）。
原因是 RRF 的分数是 `1/(k + pos)`：`k` 只改变**同一个名次之间的相对权重**，
但在名次差固定的情况下（两路排序高度一致），它**不改变合并后的顺序**。
> **推论**：RRF 的 `k` 是"不敏感的旋钮"，调它基本是浪费时间；
> 真正决定融合效果的是**两路排序的互补性**（任务 6 已经看到了：不互补就白融）。

**重排（对 BM25 的 top-50 用向量分加权）**：

| 向量权重 w | Hit@5 | MRR |
|---|---|---|
| 0.0（纯 BM25） | 0.7759 | 0.6111 |
| 0.3 | 0.8103 | 0.6243 |
| 0.5 | 0.8448 | 0.6197 |
| **0.7** | **0.8448** | **0.6321** ← 最高 |
| 1.0（纯向量） | 0.8448 | 0.6309 |

三点：
1. **重排确实有效**，但**上限很低**：MRR 从 0.6111 提到 **0.6321**（+0.021）。
2. **最优不是极值点**（`w = 0.7` 而不是 1.0）——
   两路各有价值，混合才最好；这也是"两阶段检索"（召回 → 重排）的经典形态。
3. **两路分数必须先归一化**（代码里的 `b_max` / `v_max`）。
   否则 BM25 的 0~20 分会完全压死向量的 0~1 分，加权就变成"只留 BM25"。

> **把账合起来看**：本章"调算法"这条路 —— 换算法（+0.02）+ 扫参（+0.05）+ 融合重排（+0.02）——
> 全部加起来大约 **+0.09 MRR**，但复杂度翻了几倍（四套排序 + 一个重排模型 + 一堆超参）。
> 相比之下，任务 12 换来的 **+0.32** 只用了 20 行代码。
> **这就是"先修数据、再看归因"的价值。**
"""
    ),
    md(
        """## 任务 11：误差归因 —— top-5 里的错，错在哪一类

到这里指标已经调不动了（±0.02 量级）。**该换个问法**：

> 不要问"怎么把 MRR 从 0.61 提到 0.7"，
> 要问"**排进 top-5 的那些错文档，到底是什么东西**"。

把每个 top-5 结果按"设备"和"缺陷"两个维度分四类：
"""
    ),
    code(T11),
    md(
        """**读表重点**：

| 成分 | BM25 | TF-IDF 向量 |
|---|---|---|
| ① 同设备 + 同缺陷（正确命中） | 29.0% | 32.1% |
| ② 同设备 + 异缺陷 | 29.3% | 20.0% |
| ③ **异设备 + 同缺陷** | **40.0%** | **46.6%** |
| ④ 都不沾（纯噪声） | 1.7% | 1.4% |

**这张表直接给出了改进方向**：

1. **最大的一类错误是 ③「异设备+同缺陷」（40%）** —— 也就是说，
   检索系统**抓住了缺陷词，却跑偏了设备**。
   为什么？回看任务 4 的逐词贡献：缺陷词（`渗油`）idf 最高，权重最大；
   而设备词（`绝缘子`）虽然也稀有，但**查询里往往只有 3~4 个字提到设备**，
   一旦别的设备下有一堆同样的缺陷词反复出现，分数就会被反超。

2. **④「都不沾」只有 1.7%** ——
   说明检索**没有在瞎排**，噪音很低。这是个好消息：
   剩下的 98.3% 错误都是"**相近但不完全对**"，不是"完全无关"。

3. **TF-IDF 向量的 ③ 更高（46.6%）** ——
   因为它做了 `sublinear_tf` 和 L2 归一化，**更弱化高频词差异**，
   于是"缺陷词占主导"的问题更严重。

> **归因的正确姿势**：不要看总量指标，要**看错误的结构**。
> 只看 MRR = 0.61，你的改进方向可能是"换个模型"；
> 看了这张表，你会立刻想到"**按设备过滤**" —— 因为 40% 的错误是设备跑偏。
>
> 这就是任务 12 的起点。**这一步不需要任何新技术，只需要把结果拆开看。**
"""
    ),
    md(
        """## 任务 12：定向改进 —— 拿误差归因换指标

归因说了两件事：
- 40% 的错误是**设备跑偏** → 用**元数据硬过滤**（只让同设备的文档进榜）；
- 缺陷词是主要信号，而模板词（`核实`/`反馈`/`安排`/`抢修`）是噪声 → **给缺陷词加权**。

两步走：先各自试，再合起来。
"""
    ),
    code(T12),
    md(
        """**读表重点**（本章最有价值的一张表）：

| 策略 | R@5 | P@5 | Hit@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| 基线 BM25 | 0.1013 | 0.2897 | 0.7759 | 0.6111 | 0.2750 |
| 设备元数据硬过滤 | 0.1476 | 0.4207 | 0.9138 | 0.7867 | 0.4246 |
| **硬过滤 + 缺陷词 ×2** | **0.2066** | **0.5897** | **0.9655** | **0.9345** | **0.5129** |

**MRR 0.6111 → 0.9345（+0.323）**，`Hit@5` 0.7759 → **0.9655**。
这比本章所有"调算法"手段加起来（+0.09）还高 3 倍多，代码只多 20 行。

为什么这两招这么有效？

1. **元数据硬过滤** = 把"设备跑偏"这类错误**从候选的最前面整片挪走**。
   代码里没删文档，而是把「同设备」的文档按原名次提到前面、「异设备」的整体接在后段 ——
   效果上等于**把 87.5% 的候选（8 个设备里的另外 7 个）压到榜尾**。
   这不是"调分数"，是**改候选集** —— 而检索系统的上限由候选集决定。
   任务 11 的 40% 错误是设备跑偏 → 挪走这批候选，错误自然消失。
   > 注意这里**没有算"设备识别"这一步的代价** —— 实际系统里设备类型是从
   > 工单结构化字段来的（本章数据里它本来就是一列）。
   > **有结构化元数据就一定要用** —— 这是最便宜的"外部知识"。
2. **缺陷词加权 ×2** = 把任务 4 看到的"信号词 vs 模板词"的权重差**显式放大**。
   模板词（`核实`/`反馈`/`安排`/`抢修`）几乎每篇都有，加权后也不会有影响；
   而缺陷词（`渗油`/`裂纹`/`鸟巢`）能把正确分组顶上去。
3. **两者是乘法关系，不是加法**：硬过滤管"设备对不对"，
   加权管"缺陷像不像"。两个维度各自独立，所以叠加起来接近"把两个瓶颈都消除"。

> ⚠️ **两个必须说的限制**：
> ① 这份数据是**合成**的，每组的设备-缺陷组合几乎一一对应（8 设备 × 5 缺陷 = 40 组，
> 每组 7~25 条），所以"按设备过滤"的收益被放大了。
> 真实工单里"同一设备同一缺陷"的样本更稀疏，收益会小一些，但方向不变。
> ② 这里的"设备类型"是从查询里读出来的**已知字段**，
> 不是从文本猜出来的。如果设备类型也要从文本里抽，
> 就得先做一个实体识别（NER），那是有成本的 —— **但一次性成本，长期收益**。
>
> **方法论的收获**：改进的**顺序**应该是
> **修数据（任务 1）→ 定口径（任务 5/7）→ 看归因（任务 11）→ 定向改进（任务 12）→ 最后才是调算法（任务 6/9/10）**。
> 这个顺序在电力、金融、法律这类"有结构化元数据 + 有专业词表"的场景里普遍成立。
"""
    ),
    md(
        """## 任务 13：拒答 —— 检索不到就别硬答

RAG 最危险的失败模式不是"答错"，而是**"检索不到还硬答"** ——
模型拿到 5 条无关资料，照样会编出一个看起来很专业的答案。

解法是**拒答（abstention）**：先看检索分数的**绝对水平**，太低就直接回"无法确定"。

问题在于：**阈值定多少？** 定高了会误杀真查询，定低了拦不住瞎答。
"""
    ),
    code(T13),
    md(
        """**读表重点**：

| 查询类型 | min | 中位 | max |
|---|---|---|---|
| 真命中查询（58 条） | 4.507 | 10.805 | 17.523 |
| 完全无关（6 条） | 0.000 | 0.000 | **0.000** |
| 含语料词但意图不同（6 条） | 0.000 | 1.371 | 3.174 |

| 阈值 | 真命中保留 | 无关拒绝 | 沾边拒绝 |
|---|---|---|---|
| 1.0 | 58/58 | 6/6 | 3/6 |
| 2.0 | 58/58 | 6/6 | 3/6 |
| 3.0 | 58/58 | 6/6 | 4/6 |
| **4.0** | **58/58** | **6/6** | **6/6** |

**三条结论**：

1. **完全无关的查询得分恰好是 0.000** —— 因为它们的词**一个都不在语料里**，
   BM25 逐词求和，没有词命中就是 0。这是 BM25 的一个白送的好处：
   **零分本身就是信号**。（向量检索没有这个性质：稀疏的相似度也有正值，
   所以"能不能拒答"这件事，词法检索反而比稠密向量好做。）
2. **阈值 4.0 是这份数据上的完美分割点**：真命中最低 4.507，
   沾边最高 3.174 —— **两个区间不重叠**，中间有一条干净的缝隙。
3. **但"沾边"这一类是最难的**：`变压器的工作原理是什么` 这种查询，
   `变压器` 是语料词（得分 > 0），但**意图是问原理，不是找工单**。
   阈值 1~3 都拦不住它（只能拦 3~4 条）。**这说明"分数阈值"只能处理词法层面的无关，
   处理不了意图层面的不匹配** —— 后者要靠意图分类或领域路由。

> **工程做法**：
> - 阈值**不要拍脑袋**，要像上图一样，**画出"真命中最低分"和"干扰最高分"之间的缝隙**，
>   取值落在缝隙里；
> - 留一点余量（取 4.0 而不是 3.5），因为测试集只有 6 条干扰查询，
>   **样本太少，边界不稳定**；
> - 更稳的做法是**两级拒答**：分数阈值（词法层）+ 意图路由（语义层）。
"""
    ),
    md(
        """## 任务 14：prompt 拼装与引用

检索做完，进入 RAG 的 **A（Augment）** 段：把 top-k 结果拼进 prompt。

这一步看着简单，但**拼得好不好直接决定 G 段的可用性**。
"""
    ),
    code(T14),
    md(
        """**读表重点**：拼出来的 prompt 长这样（真实输出）：

```
【参考资料】
[1] 工单 WO_0016｜设备：绝缘子｜缺陷：渗漏油｜等级：危急
    描述：【危急】绝缘子存在渗油，请安排抢修
[2] 工单 WO_0003｜设备：绝缘子｜缺陷：破损｜等级：危急
    描述：绝缘子碎裂，请核实后反馈，请安排抢修
[3] 工单 WO_0010｜设备：绝缘子｜缺陷：异物｜等级：严重
    描述：绝缘子塑料袋,请核实后反馈,请安排抢修
【问题】绝缘子渗油,请核实后反馈,请安排抢修
【要求】仅依据以上参考资料作答，并标注引用编号（如 [1]）；资料不足则回答「无法确定」。
```

四个**必备**要素（缺一个都会出事）：

| 要素 | 为什么必须有 |
|---|---|
| **编号引用**（`[1]`） | 让答案可以**回溯核查**。没有编号，模型说的话无法验证 —— 在电力/金融场景这是硬要求 |
| **结构化字段**（设备/缺陷/等级） | 拼进上下文的不止正文，还有元数据；模型才能做"同一设备"的判断 |
| **问题放最后** | 长上下文里，**结尾位置的注意力最强**，问题放最后能显著提高"回答了问题"的概率 |
| **资料不足就说无法确定** | 这是任务 13 的"拒答"在 prompt 层的兜底。**必须显式写进去** |
| **不要写"请综合你的知识作答"** | 那会让模型绕过资料自由发挥 —— 与 RAG 的初衷相反 |

> **注意 top-3 里只有 `[1]` 是正确答案**（真实分组 `绝缘子|渗漏油`），
> `[2]`/`[3]` 是"同设备异缺陷"（任务 11 的象限 ②）。
> 这说明**检索不完美时，prompt 的措辞就是最后一道防线** ——
> 有了"只依据资料 + 标注引用"，模型至少不会把 `[2]` 的破损当成渗漏油来答。
>
> 顺带一个细节：`[2]` 和 `[3]` 的 BM25 分数**并列**（都是 9.648）。
> 并列时 `argsort` 的顺序由下标决定 —— **并列是排序不稳定的常见来源**，
> 生产环境里应该定一个 secondary key（比如工单时间倒序）保证确定性。
"""
    ),
    md(
        """## 自查清单

- [ ] 知道"换语料必须固定评测集"，并能说出为什么任务 1 的对比是受控的
- [ ] 能背出 BM25 的三块（概率型 idf / tf 饱和 / 长度归一化）
- [ ] 知道 idf 公式里的 `+1` 和两个 `0.5` 各是干什么的（非负 + 平滑）
- [ ] 能解释 "`N` / `A` 拿到最高 idf" 说明 **idf 只认统计不认语义**
- [ ] 知道 `avgdl` 被脏数据污染会让**全库**分数整体偏移
- [ ] 能说出"索引侧脏数据 vs 查询侧脏数据"的影响差异
- [ ] 知道 `R@5 = 0.10` 在 13 条正样本下**不是差**，并会算 `R@k` 上限
- [ ] 能解释为什么 SVD(64) 与 TF-IDF 指标相同（词表只有 70）
- [ ] 知道 `b` 在这份语料上比 `k1` 敏感得多，且 `Hit@k` 与 `MRR` 的最优点不同
- [ ] 知道 RRF 的 `k` 基本不敏感，融合要看两路**互补性**
- [ ] 能画出误差归因四象限，并指出最大错误是「异设备+同缺陷」
- [ ] 能说出任务 12 两招各自的原理，以及"硬过滤 = 改候选集"这个关键点
- [ ] 知道 BM25 零分是天然的拒答信号（向量检索没有这个性质）
- [ ] 能说出 prompt 拼装的 5 个必备要素
- [ ] 能复述改进顺序：**修数据 → 定口径 → 看归因 → 定向改进 → 最后调算法**

## 本章一句话

> **检索效果不好，先别换算法。**
> 先把数据洗干净（+0.045），再把评测口径定死（多正样本要报上限），
> 然后把 top-k 的结果拆开看错在哪（40% 是设备跑偏），
> 最后用**元数据过滤 + 关键词加权**定向修理（+0.32）——
> 这一套下来，比"换个新模型"便宜、有效，而且**能解释为什么有效**。
"""
    ),
]

# =========================================================================== #
# 练习版                                                                      #
# =========================================================================== #

EXERCISE = [
    md(
        """# ch10 RAG 检索增强（练习）

> 把 `____` 换成正确的代码。每道题下方的 `assert` 会立刻告诉你对不对。

**本章目标**：学会把"检索效果不好"这个模糊的问题，
拆成**可度量、可归因、可定向改进**的一串实验。

## 四条实测结论

| # | 动作 | MRR 变化 |
|---|---|---|
| ① | 加深清洗（去 URL、去字间空格） | 0.5659 → **0.6111** |
| ② | 误差归因后的定向改进（设备过滤 + 缺陷词加权） | 0.6111 → **0.9345** |
| ③ | 换算法 / 融合 / 重排 | ±0.02 |
| ④ | 剔掉索引里的无效占位符 | 约 ±0.01 |

**改进顺序**：修数据 → 定口径 → 看归因 → 定向改进 → 最后才调算法。

## 本章地图

| 任务 | 内容 |
|---|---|
| 1~2 | 清洗深度实验 / 无效占位符与 `avgdl` |
| 3~4 | 手写 BM25（idf + 单条打分） |
| 5 | 分层切分与正样本规模 |
| 6~10 | 四条路线 / 指标拆解 / SVD 维度 / 参数扫描 / 融合重排 |
| 11~12 | 误差归因四象限 → 定向改进 |
| 13~14 | 拒答阈值 / prompt 拼装与引用 |
"""
    ),
    code(IMPORTS),
    md("## 脚手架（已给好，直接调用）\n\n清洗函数、分词器、BM25 类、四个指标、正样本定义都在这里 —— 全章统一口径。"),
    code(SCAFFOLD),
    md("## 任务 1：清洗深度 —— 同一批行，两种洗法\n\n注意：**同一批文档行**，只换清洗函数。这是受控实验。"),
    code(T1),
    md("## 任务 2：无效占位符 —— 平均文档长度被谁拉低了\n\n统计索引里的无效占位符条数，以及剔掉前后的 `avgdl`。"),
    code(T2),
    md("## 任务 3：手写 BM25 第一步 —— 概率型 idf\n\n按 `log(1 + (N − n + 0.5)/(n + 0.5))` 算出 idf 字典，并与脚手架对账。"),
    code(T3),
    md("## 任务 4：手写 BM25 第二步 —— 给一条查询的一条文档打分\n\nidf × tf 饱和 × 长度归一化，逐词相加。"),
    code(T4),
    md("## 任务 5：分层切分与正样本规模\n\n统计每条查询的正样本数分布。"),
    code(T5),
    md("## 任务 6：四条检索路线\n\n造出 BM25 / TF-IDF / SVD(64) / RRF 四套完整排序。"),
    code(T6),
    md("## 任务 7：指标拆解 —— R@5 为什么这么低\n\n同时算实测 `R@k` 与理论上限，理解“多正样本”的指标含义。"),
    code(T7),
    md("## 任务 8：SVD 降到多少维才不亏\n\n扫 16 / 32 / 64 维的解释方差比。"),
    code(T8),
    md("## 任务 9：BM25 的 k1 / b 扫描\n\n12 组网格，看 `Hit@5` 和 `MRR` 的最优点是否一致。"),
    code(T9),
    md("## 任务 10：融合与重排的收益\n\nRRF 的 `k` + 重排的向量权重，各扫一遍。**注意两路分数要先归一化。**"),
    code(T10),
    md("## 任务 11：误差归因 —— top-5 里的错，错在哪一类\n\n用四象限拆开 top-5，找出最大的一类错误。"),
    code(T11),
    md(
        """## 任务 12：定向改进 —— 拿误差归因换指标

按任务 11 的归因做两件事：

1. **设备元数据硬过滤**：只让同设备类型的文档进榜（应对「异设备+同缺陷」）；
2. **在硬过滤的基础上给缺陷词 ×2 加权**（放大信号词、压制模板词）。

> ⚠️ 两阶段叠加才是最优解，单做一步收益差很多。
"""
    ),
    code(T12),
    md("## 任务 13：拒答 —— 检索不到就别硬答\n\n先看三类查询的榜首分数分布，再找阈值缝隙。"),
    code(T13),
    md("## 任务 14：prompt 拼装与引用\n\n拼出带编号引用、含结构化字段、以问题收尾、并写明“资料不足说无法确定”的 prompt。"),
    code(T14),
    md(
        """## 自查清单

- [ ] 能说出"换语料必须固定评测集"，并解释任务 1 为什么是受控实验
- [ ] 能背出 BM25 的三块（idf / tf 饱和 / 长度归一化）
- [ ] 知道 idf 的 `+1` 是干什么的（保证非负）
- [ ] 知道 `avgdl` 污染影响的是**全库**分数
- [ ] 会算 `R@k` 的上限，并知道多正样本下 `R@5` 低不代表检索差
- [ ] 能解释 SVD(64) 为什么和 TF-IDF 指标相同
- [ ] 能画出误差归因四象限，说出最大错误是哪一类
- [ ] 知道任务 12 的两招为什么有效（改候选集 + 放大信号词）
- [ ] 知道 BM25 零分是天然的拒答信号
- [ ] 能复述改进顺序：**修数据 → 定口径 → 看归因 → 定向改进 → 调算法**
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
