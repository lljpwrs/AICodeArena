#!/usr/bin/env python3
"""nlp08 jieba 中文分词实战：三模式 / 前缀词典 + DAG + DP 手写复刻 / HMM / 用户词典 / 词性 / 关键词。

数据：`data/synth_text.csv`（600 条电力设备缺陷工单，由 `data/make_data.py` 生成）。
真值全部实跑（jieba 0.42.1，`jieba.dt` 词表 498113 条 / total 60101967）。

> 本章三条主线：
> ① **jieba 不是黑盒** —— 它的核心就是一个 498113 条的前缀词典 + DAG + 对数概率 DP，
>    手写完只有 30 行；但**最后的「连续英数字符合并」一步不做，570 条一条都对不上**；
> ② **词典 + HMM 各有各的错** —— HMM 开出「主变」这种未登录词（186/570 条切分因此改变），
>    但也会把「3号主变温升异常」切成错误的「号主 / 变温」；用户词典才是可控的修法；
> ③ **切分方式对下游没有影响** —— 手写 / jieba / jieba+用户词典三种切法喂给同一分类器，
>    5 折准确率**全是 1.000000**（负结果：信号词互斥的数据集上，分词不是瓶颈）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp08_jieba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch08_jieba"

# =========================================================================== #
# 公共部分（两版都给）                                                        #
# =========================================================================== #

IMPORTS = '''import math
import re
import tempfile
import time
from collections import Counter
from pathlib import Path

import jieba
import jieba.analyse
import jieba.posseg as pseg
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

jieba.setLogLevel(60)          # 关掉 "Building prefix dict from ..." 的日志
DATA = Path("data")
CSV = DATA / "synth_text.csv"
'''

SETUP = '''df = pd.read_csv(CSV, keep_default_na=False)
jieba.initialize()             # 惰性初始化：不先调一次，jieba.dt.total 会是 0

# ---- 以下词表 / 清洗 / 分词函数与 ch01 完全一致（本章要做对照，必须同口径）----
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


def clean(text: str) -> str:
    """ch01 的五步清洗管道（去 HTML/URL、标点归一、去汉字间空格）。"""
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"https?://\\S+", "", s)
    s = s.replace(",", "，")
    s = re.sub(r"([，。！？])\\1+", r"\\1", s)
    s = re.sub(r"(?<=[\\u4e00-\\u9fff])\\s+(?=[\\u4e00-\\u9fff])", "", s)
    return re.sub(r"\\s+", " ", s).strip()


def fmm(seg: str) -> list[str]:
    """正向最大匹配（ch01 §1.6）。"""
    out, i = [], 0
    while i < len(seg):
        length = min(MAX_WORD_LEN, len(seg) - i)
        while length > 1 and seg[i:i + length] not in VOCAB_SET:
            length -= 1
        out.append(seg[i:i + length])
        i += length
    return out


def bmm(seg: str) -> list[str]:
    """逆向最大匹配（ch01 §1.7）。"""
    out, i = [], len(seg)
    while i > 0:
        length = min(MAX_WORD_LEN, i)
        while length > 1 and seg[i - length:i] not in VOCAB_SET:
            length -= 1
        out.insert(0, seg[i - length:i])
        i -= length
    return out


def tokenize(text: str) -> list[str]:
    """两级切分：纯汉字段走 fmm，英文/数字/标点整体入列（ch01 §1.8）。"""
    out = []
    for seg in SEG_RE.findall(text):
        out.extend(fmm(seg) if HAN_RE.fullmatch(seg) else [seg])
    return out


desc_clean = df["缺陷描述"].map(clean)
BAD = {"", "N/A", "-", "无", "''", '""'}
keep_idx = [i for i, t in enumerate(desc_clean) if t.strip() not in BAD]
texts = [desc_clean[i] for i in keep_idx]
labels = df["缺陷类型"].to_numpy()[keep_idx]

print("原始 %d 条 | 清洗后可用 %d 条" % (len(df), len(texts)))
print("手写词表 %d 个词 | 最长 %d 字" % (len(VOCAB), MAX_WORD_LEN))
print("首条：", texts[0])
'''

# =========================================================================== #
# 各任务的代码块                                                              #
# =========================================================================== #

TRIANGLE = '''CASES = ["研究生命起源", "变压器渗油", "电流互感器发热", "3号主变渗漏油", "隔离开关破损"]

print("%-10s %-20s %-20s %s" % ("输入", "正向最大匹配", "逆向最大匹配", "jieba"))
for t in CASES:
    # @@todo 同一句话分别用「正向最大匹配 / 逆向最大匹配 / jieba」切一遍
    # @@hint fmm(t) / bmm(t) / jieba.lcut(t) —— 都返回 list，用 "/" 拼起来看
    a, b, c = fmm(t), bmm(t), jieba.lcut(t)
    # @@end
    print("%-10s %-20s %-20s %s" % (t, "/".join(a), "/".join(b), "/".join(c)))

print()
print("三种切法两两都不完全一致 —— 它们是三套不同的「什么算一个词」的定义。")

# ---- 验收 ----
assert fmm("研究生命起源") == ["研究生", "命", "起源"], "正向贪心：先把「研究生」吃掉"
assert bmm("研究生命起源") == ["研究", "生命", "起源"], "逆向贪心：从右往左吃到「生命」"
assert jieba.lcut("研究生命起源") == ["研究", "生命", "起源"]
assert fmm("电流互感器发热") == ["电流互感器", "发热"], "手写词表里有「电流互感器」这个长词"
assert jieba.lcut("电流互感器发热") == ["电流", "互感器", "发热"], "jieba 词典里没有「电流互感器」"
assert jieba.lcut("隔离开关破损") == ["隔离", "开关", "破损"], "jieba 词典里也没有「隔离开关」"
'''

MODES = '''MODE_CASES = ["主变压器渗漏油严重", "3号主变温升异常"]

for s in MODE_CASES:
    # @@todo jieba 三种模式各切一遍
    # @@hint jieba.lcut(s) / jieba.lcut(s, cut_all=True) / jieba.lcut_for_search(s)
    precise, full, search = jieba.lcut(s), jieba.lcut(s, cut_all=True), jieba.lcut_for_search(s)
    # @@end
    print(s)
    print("  精确模式   %d 个：" % len(precise), "/".join(precise))
    print("  全模式     %d 个：" % len(full), "/".join(full))
    print("  搜索引擎   %d 个：" % len(search), "/".join(search))
    print()

# ---- 验收 ----
assert jieba.lcut("主变压器渗漏油严重") == ["主", "变压器", "渗漏", "油", "严重"]
assert jieba.lcut("主变压器渗漏油严重", cut_all=True) == ["主", "变压", "变压器", "渗漏", "漏油", "严重"]
assert jieba.lcut_for_search("主变压器渗漏油严重") == ["主", "变压", "变压器", "渗漏", "油", "严重"]
assert len(jieba.lcut("主变压器渗漏油严重", cut_all=True)) == 6, "全模式会输出所有可能词，含重叠"
assert len(jieba.lcut("主变压器渗漏油严重")) == 5, "精确模式保证「切出来的词首尾相接、正好拼回原句」"
assert jieba.lcut("3号主变温升异常") == ["3", "号主", "变温", "升", "异常"], "jieba 也会切错：HMM 把「号主/变温」拼出来了"
'''

PREFIX_DICT = '''FREQ = jieba.dt.FREQ          # 词 + 所有前缀；纯前缀条目的 freq 人为记 0
TOTAL = jieba.dt.total        # DP 的归一化分母
LOG_TOTAL = math.log(TOTAL)

# @@todo 词典规模：条目总数 / 其中「freq==0 的纯前缀」条目数 / 所有词频之和
# @@hint len(FREQ) / sum(1 for v in FREQ.values() if not v) / sum(FREQ.values())
n_all = len(FREQ)
n_prefix_only = sum(1 for v in FREQ.values() if not v)
freq_sum = sum(FREQ.values())
# @@end

print("FREQ 条目 %d | 纯前缀 %d | 真词 %d" % (n_all, n_prefix_only, n_all - n_prefix_only))
print("sum(FREQ.values()) = %d" % freq_sum)
print("jieba.dt.total     = %d   <- 分母必须是这个" % TOTAL)
print("两者差 %d —— 所以别自己重算 total（差这一点点，DP 的口径就和 jieba 不一样了）" % (TOTAL - freq_sum))
print()
print("%-14s %-8s %s" % ("片段", "in FREQ", "freq"))
for w in ["研", "研究", "研究生", "研究生命", "主", "主变", "温升", "发现", "AT", "STATION_E_02"]:
    print("%-14s %-8s %s" % (w, w in FREQ, FREQ.get(w)))

# ---- 验收 ----
assert n_all == 498113 and n_prefix_only == 149068
assert freq_sum == 60101964 and TOTAL == 60101967
assert FREQ["研究"] == 35029 and FREQ["研究生"] == 1816 and FREQ["命"] == 11603
assert FREQ["温升"] == 0, "「温升」在 FREQ 里但 freq=0 —— 它只是前缀索引（某个以「温升」开头的词的前缀），不是词"
assert "主变" not in FREQ, "「主变」连前缀都不是 —— 所以默认切分只能切出「主/变」"
assert "STATION_E_02" not in FREQ
'''

DAG_FN = '''def make_dag(s: str) -> dict:
    """构造 DAG：dag[k] = 以 k 为起点、所有「落在词典里」的终点。"""
    dag = {}
    N = len(s)
    for k in range(N):
        ends, i, frag = [], k, s[k]
        while i < N and frag in FREQ:
            # @@todo 只把「真词」的终点收进 ends（freq==0 的前缀不收），再把片段右扩一格
            # @@hint 挖空块里不能写 if —— 用条件表达式：`ends.append(i) if FREQ[frag] else None`
            # @@hint 两层判断别混：`frag in FREQ` 决定「继续扩」，`FREQ[frag]` 决定「算不算词」
            ends.append(i) if FREQ[frag] else None
            i = i + 1
            frag = s[k:i + 1]
            # @@end
        if not ends:
            ends.append(k)      # 整段都不在词典里：单字成词
        dag[k] = ends
    return dag
'''

DAG_DEMO = '''DEMO = "研究生命起源"

# @@todo 拿到 DEMO 的 DAG
# @@hint make_dag(DEMO) -> {起点: [终点...]}
dag_demo = make_dag(DEMO)
# @@end

for k in sorted(dag_demo):
    cand = [DEMO[k:x + 1] for x in dag_demo[k]]
    print("起点 %d「%s」-> 终点 %-12s 候选词 %s" % (k, DEMO[k], dag_demo[k], cand))

print()
print("注意起点 0：候选是「研 / 研究 / 研究生」——DAG 只给候选，不负责选哪个。")

# ---- 验收 ----
assert dag_demo == {0: [0, 1, 2], 1: [1], 2: [2, 3], 3: [3], 4: [4, 5], 5: [5]}
assert [DEMO[0:x + 1] for x in dag_demo[0]] == ["研", "研究", "研究生"]
assert [DEMO[2:x + 1] for x in dag_demo[2]] == ["生", "生命"]
'''

DP_HELPER = '''def wlog(word: str) -> float:
    """单个词的对数概率权重：log(freq) - log(total)。

    未登录词 freq 为 0 → `FREQ.get(word) or 1` 兜成 1，否则 math.log(0) 直接 ValueError。
    """
    return math.log(FREQ.get(word) or 1) - LOG_TOTAL
'''

DP_FN = '''def dp_route(s: str, dag: dict) -> dict:
    """从后往前 DP：route[k] = (从 k 到句尾的最优「对数概率之和」, 这一步切在哪)。"""
    N = len(s)
    route = {N: (0.0, 0)}          # 句尾哨兵：后面没词了，分值为 0
    for idx in range(N - 1, -1, -1):
        # @@todo 枚举 idx 的所有出边，每条算「本词权重 + 后缀最优值」，取最大的那一格
        # @@hint cand = [(wlog(s[idx:x + 1]) + route[x + 1][0], x) for x in dag[idx]] 后 route[idx] = max(cand)
        # @@hint 平局时 max 会取更大的 x（更长的词）—— 与 jieba 的 tie-break 一致
        cand = [(wlog(s[idx:x + 1]) + route[x + 1][0], x) for x in dag[idx]]
        route[idx] = max(cand)
        # @@end
    return route
'''

DP_DEMO = '''def cut_dp(s: str) -> list[str]:
    """把 DP 的最优路径还原成切分结果。"""
    dag = make_dag(s)
    route = dp_route(s, dag)
    out, i, N = [], 0, len(s)
    while i < N:
        # @@todo 取 route[i] 记下的最优切点 j，把 s[i:j+1] 收进 out，指针跳到 j+1
        # @@hint j = route[i][1] / out.append(s[i:j + 1]) / i = j + 1
        j = route[i][1]
        out.append(s[i:j + 1])
        i = j + 1
        # @@end
    return out


for s in ["研究生命起源", "电流互感器发热", "3号主变渗漏油"]:
    print("%-10s -> %s" % (s, "/".join(cut_dp(s))))

print()
print("「研究生命起源」的 route（起点 -> (分值, 切点)）：")
route_demo = dp_route("研究生命起源", make_dag("研究生命起源"))
for k in range(7):
    print("  %d %s" % (k, route_demo[k]))

print()
print("为什么 DP 能切对？它在所有合法路径里挑「对数概率之和最大」的那条：")
print("  「研/究/生/命/起/源」每个字都是低频，6 个 log 加起来很小；")
print("  「研究/生命/起源」三个高频词，3 个 log 加起来明显更大。")

# ---- 验收 ----
assert cut_dp("研究生命起源") == ["研究", "生命", "起源"]
assert cut_dp("电流互感器发热") == ["电流", "互感器", "发热"]
assert cut_dp("3号主变渗漏油") == ["3", "号", "主", "变", "渗漏", "油"], "纯 DP 拿 HMM 的未登录词没办法"
'''

CUT_ENG = '''RE_ENG = re.compile("[a-zA-Z0-9]")


def merge_eng(pieces: list[str]) -> list[str]:
    """把切分结果里「连续的单个英数字符」合并成一串。

    jieba 的 `__cut_DAG_NO_HMM` 在三模式之外还多做这一步：DP 只认汉字词，
    英文/数字必然是单字符 token，连着的要拼回去，否则 `STATION_E_02` 就成 12 个 token。
    """
    out, buf = [], ""
    for w in pieces:
        if RE_ENG.match(w) and len(w) == 1:
            buf += w
        else:
            if buf:
                out.append(buf)
                buf = ""
            out.append(w)
    if buf:
        out.append(buf)
    return out


def cut_dp_eng(s: str) -> list[str]:
    # @@todo 先用 cut_dp 拿纯 DP 结果，再交给 merge_eng 收尾
    # @@hint merge_eng(cut_dp(s))
    merged = merge_eng(cut_dp(s))
    # @@end
    return merged


demo_s = "STATION_E_02绝缘子渗油"
print("裸 DP    :", "/".join(cut_dp(demo_s)))
print("+英数合并 :", "/".join(cut_dp_eng(demo_s)))
print("jieba    :", "/".join(jieba.lcut(demo_s, HMM=False)))

# ---- 验收 ----
assert cut_dp(demo_s)[:3] == ["S", "T", "A"], "裸 DP 把编号逐字符切碎"
assert cut_dp_eng(demo_s)[0] == "STATION"
assert cut_dp_eng(demo_s) == jieba.lcut(demo_s, HMM=False)
'''

REPLICATE = '''# @@todo 统计「裸 DP」与「DP + 英数合并」各自与 jieba(HMM=False) 全句一致的条数
# @@hint sum(1 for t in texts if cut_dp(t) == jieba.lcut(t, HMM=False))，再换成 cut_dp_eng
n_naive = sum(1 for t in texts if cut_dp(t) == jieba.lcut(t, HMM=False))
n_eng = sum(1 for t in texts if cut_dp_eng(t) == jieba.lcut(t, HMM=False))
# @@end

print("裸 DP          vs jieba(HMM=False)：%d / %d = %.6f"
      % (n_naive, len(texts), n_naive / len(texts)))
print("DP + 英数合并   vs jieba(HMM=False)：%d / %d = %.6f"
      % (n_eng, len(texts), n_eng / len(texts)))
print()
print("首条：", texts[0])
print("  裸 DP    :", "/".join(cut_dp(texts[0])))
print("  DP+合并  :", "/".join(cut_dp_eng(texts[0])))
print("  jieba    :", "/".join(jieba.lcut(texts[0], HMM=False)))

# ---- 验收 ----
assert n_naive == 0, "少了「英数字符合并」这一步，570 条一条都对不上"
assert n_eng == len(texts) == 570, "补上这一步，就 100% 复刻了 jieba 的词典分词"
'''

HMM_DEMO = '''# @@todo 统计「默认(HMM=True) 与 HMM=False 结果不同」的句子数
# @@hint sum(1 for t in texts if jieba.lcut(t) != jieba.lcut(t, HMM=False))
n_hmm = sum(1 for t in texts if jieba.lcut(t) != jieba.lcut(t, HMM=False))
# @@end

print("HMM 开关造成差异：%d / %d = %.6f" % (n_hmm, len(texts), n_hmm / len(texts)))
print()
for t in ["3号主变渗漏油", "主变压器渗漏油严重", "未登录词阿巴阿巴"]:
    print("  %-14s HMM=True  %-24s HMM=False %s"
          % (t, "/".join(jieba.lcut(t)), "/".join(jieba.lcut(t, HMM=False))))
print()
print("HMM 在这批工单上**净收益是正的**（主变 / 渗漏油 这类电力词词典里都没有），")
print("但它也会犯错 —— 见 §8.4「3号主变温升异常」被切成「号主 / 变温」。")

# ---- 验收 ----
assert n_hmm == 186
assert jieba.lcut("3号主变渗漏油") == ["3", "号", "主变", "渗漏", "油"], "HMM 把未登录词「主变」拼出来了"
assert jieba.lcut("3号主变渗漏油", HMM=False) == ["3", "号", "主", "变", "渗漏", "油"]
'''

USERDICT_SRC = '''USER_DICT = Path(tempfile.gettempdir()) / "nlp08_user_dict.txt"

# @@todo 按 jieba 用户词典格式写一份电力术语表：每行「词 词频 [词性]」
# @@hint 词频给大一点（100）好压过默认切分；行之间用 \\n 分隔
USER_DICT.write_text(
    "主变 100 n\\n渗漏油 100 n\\n母线巡检 100 n\\n3号主变 100 n\\n温升异常 100 n\\n",
    encoding="utf-8",
)
# @@end

print("用户词典：", USER_DICT)
print(USER_DICT.read_text(encoding="utf-8"))

# ---- 验收 ----
rows_ud = [ln.split() for ln in USER_DICT.read_text(encoding="utf-8").splitlines() if ln.strip()]
assert len(rows_ud) == 5
assert all(len(r) >= 2 for r in rows_ud), "每行至少要有「词 + 词频」"
'''

USERDICT_DEMO = '''SENT1 = "母线巡检时发现异物，位于3号主变，严重渗漏油"
SENT2 = "3号主变温升异常"

# @@todo 建一个独立 Tokenizer 并加载用户词典（不要污染全局，后面还要用默认分词对照）
# @@hint jieba.Tokenizer() -> tk.initialize() -> tk.load_userdict(str(USER_DICT))
tk = jieba.Tokenizer()
tk.initialize()
tk.load_userdict(str(USER_DICT))
# @@end

for name, cut in [("全局默认", jieba.lcut), ("独立实例 + 用户词典", tk.lcut)]:
    print("%s：" % name)
    print("  ", "/".join(cut(SENT1)))
    print("  ", "/".join(cut(SENT2)))

# ---- 验收 ----
assert jieba.lcut(SENT1) == ["母线", "巡检", "时", "发现", "异物", "，",
                             "位于", "3", "号", "主变", "，", "严重", "渗漏", "油"]
assert tk.lcut(SENT1) == ["母线巡检", "时", "发现", "异物", "，",
                          "位于", "3号主变", "，", "严重", "渗漏油"]
assert jieba.lcut(SENT2) == ["3", "号主", "变温", "升", "异常"], "默认分词在 OOV 上切错了"
assert tk.lcut(SENT2) == ["3号主变", "温升异常"], "用户词典把错切成对"
assert jieba.lcut(SENT2) != tk.lcut(SENT2), "独立实例生效，且没有污染全局"
'''

POSSEG = '''SAMPLE = "母线巡检时发现严重渗漏油，位于3号主变，请尽快核实"

# @@todo 词性标注 + 全语料词性分布统计
# @@hint tagged = [(w, f) for w, f in pseg.cut(SAMPLE)]
# @@hint pos_all = Counter(f for t in texts for w, f in pseg.cut(t))
tagged = [(w, f) for w, f in pseg.cut(SAMPLE)]
pos_all = Counter(f for t in texts for w, f in pseg.cut(t))
# @@end

print("词性标注：")
print("  " + " | ".join("%s/%s" % (w, f) for w, f in tagged))
print()
print("全语料词性种类 %d 种 | Top 8：%s" % (len(pos_all), pos_all.most_common(8)))
print("  n(名词)=%d  v(动词)=%d  x(标点)=%d  eng(英文)=%d  m(数词)=%d"
      % (pos_all["n"], pos_all["v"], pos_all["x"], pos_all["eng"], pos_all["m"]))

# ---- 验收 ----
tag_map = dict(tagged)
assert tag_map["严重"] == "a" and tag_map["位于"] == "v" and tag_map["尽快"] == "d"
assert len(pos_all) == 13
assert pos_all.most_common(1) == [("x", 2800)], "标点/分隔符是词性分布的第一名 —— 所以停用词表必须包含标点"
assert pos_all["v"] == 2081 and pos_all["n"] == 1372 and pos_all["eng"] == 1140
'''

KEYWORD = '''# @@todo TF-IDF 与 TextRank 各抽 topK=5 关键词
# @@hint jieba.analyse.extract_tags(SAMPLE, topK=5) / jieba.analyse.textrank(SAMPLE, topK=5)
kw_tfidf = jieba.analyse.extract_tags(SAMPLE, topK=5)
kw_tr = jieba.analyse.textrank(SAMPLE, topK=5)
# @@end

print("TF-IDF  :", kw_tfidf)
print("TextRank:", kw_tr)
print("交集    :", sorted(set(kw_tfidf) & set(kw_tr)))
print()
print("两者都用到「没被用户词典修正过的分词」，所以关键词里依然是「主变 / 渗漏」这种半截词。")

# ---- 验收 ----
assert kw_tfidf == ["主变", "母线", "渗漏", "巡检", "核实"]
assert kw_tr == ["渗漏", "巡检", "发现", "主变", "位于"]
assert set(kw_tfidf) & set(kw_tr) == {"主变", "巡检", "渗漏"}
'''

COMPARE = '''hand_tok = [[w for w in tokenize(t) if w.strip()] for t in texts]

t0 = time.time()
jb_tok = [jieba.lcut(t) for t in texts]
jb_sec = time.time() - t0

# @@todo 两边分别统计：词表大小 / 平均 token 数 / 总 token 数；再数「逐句完全一致」条数
# @@hint len({w for d in docs for w in d})；float(np.mean([len(d) for d in docs]))；sum(len(d) for d in docs)
hand_vocab = len({w for d in hand_tok for w in d})
jb_vocab = len({w for d in jb_tok for w in d})
hand_avg = float(np.mean([len(d) for d in hand_tok]))
jb_avg = float(np.mean([len(d) for d in jb_tok]))
hand_total = sum(len(d) for d in hand_tok)
jb_total = sum(len(d) for d in jb_tok)
n_same = sum(1 for a, b in zip(hand_tok, jb_tok) if a == b)
# @@end

print("手写  词表 %3d | 平均 %9.6f | 总 %5d" % (hand_vocab, hand_avg, hand_total))
print("jieba 词表 %3d | 平均 %9.6f | 总 %5d" % (jb_vocab, jb_avg, jb_total))
print("jieba 切完 %d 条耗时 %.3fs" % (len(texts), jb_sec))
print()
only_hand = sorted({w for d in hand_tok for w in d} - {w for d in jb_tok for w in d})
only_jb = sorted({w for d in jb_tok for w in d} - {w for d in hand_tok for w in d})
print("只在手写出现 %2d 个：" % len(only_hand), only_hand[:5], "...")
print("只在 jieba 出现 %2d 个：" % len(only_jb), only_jb[:5], "...")
print()
print("逐句完全一致：%d / %d = %.6f" % (n_same, len(hand_tok), n_same / len(hand_tok)))

# ---- 验收 ----
assert hand_vocab == 77 and jb_vocab == 94
assert abs(hand_avg - 8.536842) < 1e-6 and abs(jb_avg - 15.768421) < 1e-6
assert hand_total == 4866 and jb_total == 8988
assert len(only_hand) == 34 and len(only_jb) == 51
assert n_same == 0, "两种切法一句都对不上：手写把整句短语当词，jieba 把编号切散"
assert jb_sec < 5.0
'''

DIFF_SHOW = '''print("逐条对照（前 3 条）：")
for i in range(3):
    print()
    print(texts[i])
    print("  手写 :", "/".join(hand_tok[i]))
    print("  jieba:", "/".join(jb_tok[i]))

print()
print("两边各有各的错：")
print("  · 手写版把「请核实后反馈 / 请安排抢修」当成一个词 —— 那是词表里故意塞的长词；")
print("  · jieba 把台区编号 STATION_E_02 切成 STATION / _ / E / _ / 02，还把「镀锌层脱落」切成「镀锌」「层」「脱落」。")
print("  · 手写版在这批数据上反而更「像人」——因为它见过这个数据集，jieba 的通用词典没见过。")
print()
print("但反过来看 §8.9：只要给 jieba 一份用户词典，它立刻就能学会「母线巡检 / 3号主变 / 渗漏油」。")
'''

DOWNSTREAM = '''def bow_cv5(docs):
    """词袋 + 逻辑回归 + 5 折交叉验证，返回 (词表大小, 特征维度, 平均准确率)。"""
    cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False, token_pattern=None)
    M = cv.fit_transform(docs)
    scores = cross_val_score(LogisticRegression(max_iter=1000), M, labels, cv=5)
    return len(cv.vocabulary_), M.shape[1], float(scores.mean())


# @@todo 三种切分（手写 / jieba 默认 / jieba+用户词典）各喂一遍同一个分类器
# @@hint bow_cv5(hand_tok) / bow_cv5(jb_tok) / bow_cv5([tk.lcut(t) for t in texts])
jd_tok = [tk.lcut(t) for t in texts]
rows_ds = [("手写最大匹配", bow_cv5(hand_tok)),
           ("jieba 默认", bow_cv5(jb_tok)),
           ("jieba + 用户词典", bow_cv5(jd_tok))]
# @@end

print("三种切分 × 同一分类器（词袋 + 逻辑回归，5 折）：")
for tag, (n_vocab, n_feat, score) in rows_ds:
    print("  %-18s 词表 %3d | 特征 %3d | 5 折 %.6f" % (tag, n_vocab, n_feat, score))
print()
print("三种切法的词表大小差 22%（77 vs 94），准确率却一模一样。")
print("原因在 §2.3：这份数据的五类信号词**互不重叠**，分词方式动不了「渗油 / 锈蚀 / 碎裂」这些锚点。")

# ---- 验收 ----
assert [r[1][2] for r in rows_ds] == [1.0, 1.0, 1.0], "三种切法打平 —— 又一个负结果"
assert [r[1][1] for r in rows_ds] == [77, 94, 94]
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch08 jieba 中文分词实战（讲解版）

> **本节考点**：`docs/竞赛考点对照表.md` 里「数据准备及处理 10%」在中文文本侧的落点，
> 也是 ch01「手写最大匹配」的**工业对照版** ——
> 竞赛里给你一份中文语料，大概率不允许你手写分词器，`jieba` 是标准答案。

数据沿用 ch01 的 `data/synth_text.csv`（600 条电力设备缺陷工单，五类各 120 条，清洗后可用 **570** 条）。

**本章主线**：ch01 教了「规则分词怎么实现」，本章回答「工程上到底该怎么切」。

| 问题 | 本章结论 |
|---|---|
| jieba 是不是黑盒？ | **不是** —— 前缀词典 + DAG + 对数概率 DP，手写 30 行；补上「英数符合并」后 **570/570** 与它完全一致 |
| 词典 + HMM 够用吗？ | **不够** —— HMM 改变了 **186/570（32.63%）** 条切分，也会把「3号主变温升异常」切成「号主 / 变温」 |
| 分词方式影响下游吗？ | **这份数据上不影响** —— 手写 / jieba / jieba+用户词典三种切法，5 折准确率**全是 1.000000** |

> ⚠️ 与 ch01 ~ ch07 不同，**本章需要 `jieba` 0.42.1**（已装）。
> 但**仍然断网可跑**：jieba 的词典和 IDF 表都是随包分发的本地文件，不联网。
"""
    ),
    md(
        """
## 学习目标 & API 速查

**学习目标**

1. 能说清 jieba 三种模式（精确 / 全模式 / 搜索引擎）**分别输出什么**、用在哪
2. 能从 `jieba.dt.FREQ` 反推出「前缀词典」的结构，并解释 `freq==0` 的条目是什么
3. 能**手写** DAG 构造 + 对数概率 DP，并解释为什么 DP 比最大匹配切得对
4. 知道 `HMM=True/False` 到底改了什么，能举出它**改对了**和**改错了**的例子
5. 会用 `load_userdict` 加专业术语，并知道为什么要用**独立 `Tokenizer` 实例**
6. 会用 `posseg` 做词性标注、`analyse` 做 TF-IDF / TextRank 关键词提取
7. 能设计「切分方式 → 下游指标」的对照实验，并接受「没有差异」这个结果

**API 速查表**

| 操作 | API | 返回 | 一句话说明 |
|---|---|---|---|
| 精确分词 | `jieba.lcut(s)` | list | 默认，**首尾相接正好拼回原句** |
| 全模式 | `jieba.lcut(s, cut_all=True)` | list | 输出所有可能词，**互相重叠、拼不回去**，只用于词典挖掘 |
| 搜索引擎模式 | `jieba.lcut_for_search(s)` | list | 精确模式 + 对长词再切细，给索引/召回用 |
| 关 HMM | `jieba.lcut(s, HMM=False)` | list | 纯词典 + DP，结果**完全可复现** |
| 初始化 | `jieba.initialize()` | None | 惰性触发；**不调它 `jieba.dt.total` 是 0** |
| 词频表 | `jieba.dt.FREQ` | dict | 词 + 所有前缀；纯前缀 freq 为 **0** |
| 词频总和 | `jieba.dt.total` | int | DP 的归一化分母（**≠ `sum(FREQ.values())`**） |
| 加载词典 | `jieba.load_userdict(path)` | None | 每行 `词 词频 [词性]`；**污染全局** |
| 独立实例 | `jieba.Tokenizer()` | Tokenizer | `.initialize()` / `.load_userdict()` / `.lcut()` |
| 词性标注 | `jieba.posseg.cut(s)` | 生成 `(word, flag)` | `n` 名词 / `v` 动词 / `a` 形容词 / `x` 标点 / `eng` 英文 |
| TF-IDF 关键词 | `jieba.analyse.extract_tags(s, topK=)` | list | 依赖内置 `idf.txt` |
| TextRank 关键词 | `jieba.analyse.textrank(s, topK=)` | list | 需要词性过滤（默认留 `nsvn` 等实词） |
"""
    ),
    code(IMPORTS),
    md(
        """
## 8.1 数据与基线：把 ch01 的清洗和分词原样搬过来

要做「手写 vs jieba」的对照，**两边必须同口径**：同一份 CSV、同一个 `clean`、同一份词表。
所以本节把 ch01 的 `clean` / `fmm` / `bmm` / `tokenize` 原样复制过来（不 import，notebook 之间不互相依赖）。

`jieba.setLogLevel(60)` 是必须的 —— 否则首次调用会往 stderr 打一长串
`Building prefix dict from the default dictionary ... Loading model cost 0.5 seconds.`，
在 notebook 里刷屏，也污染校验脚本的输出比对。
"""
    ),
    code(SETUP),
    md(
        """
## 8.2 三方对照：手写 FMM / 手写 BMM / jieba

同一句话，三种切法。**先别看结论，自己在心里切一遍**，再看数字。

读结果时注意三件事：

1. `研究生命起源` 上，正向切出 `研究生/命/起源`（错），逆向切出 `研究/生命/起源`（对），
   jieba 与逆向一致 —— 这就是 ch01 §1.7 说的「交叉歧义」。
2. `电流互感器发热` 上，**手写反而对了**（词表里有 `电流互感器` 这个长词），
   jieba 切成 `电流/互感器/发热` —— 它的通用词典里没有这个电力复合词。
3. `隔离开关破损` 上，jieba 切成 `隔离/开关/破损` —— 同样是词典覆盖问题。

**这三条一起看才能得出正确结论**：不存在「谁更好」，只存在「谁的词典更贴这份数据」。
"""
    ),
    code(TRIANGLE),
    md(
        """
## 8.3 jieba 三种模式：全模式不是「更细的精确模式」

**为什么难**：三种模式名字很像，很多人以为「全模式 = 精确模式 + 更细的切分」。

**实际关系**：

| 模式 | 性质 | 典型用途 |
|---|---|---|
| 精确模式 | **是一个合法切分**（无重叠、拼回原文） | 下游模型输入 |
| 全模式 | **不是切分**，是所有词典命中的集合（允许重叠） | 挖词典、查召回 |
| 搜索引擎模式 | 精确模式 + 对长词补充细粒度切分 | 倒排索引、搜索召回 |

看 `主变压器渗漏油严重` 的全模式输出：`主/变压/变压器/渗漏/漏油/严重` ——
`变压` 和 `变压器` 同时出现、`渗漏` 与 `漏油` 也重叠。
如果你把全模式结果直接喂给词袋，**同一个字会被统计多次**，特征就废了。

**判定规则**：

> 只要下游是「把 token 当特征」的模型（词袋 / TF-IDF / 序列模型），**只能用精确模式**。
> 全模式只用于「扫一遍语料看有哪些候选词」，搜索引擎模式只用于检索。

**再看一个反例**：`3号主变温升异常` → 精确模式给出 `3 / 号主 / 变温 / 升 / 异常`。
`号主`、`变温` 都是错切 —— 这是 HMM 在未登录词上的翻车（§8.8 细讲）。
"""
    ),
    code(MODES),
    md(
        """
## 8.4 原理一：前缀词典 —— `jieba.dt.FREQ` 到底是什么

**为什么难**：`jieba.dt.FREQ` 是 49.8 万条的字典，比一个正常中文词表（10 万级）**大了近 5 倍**。
多出来的是**前缀**。

jieba 的加载流程（`gen_pfdict`）会做两件事：

1. 把词典文件里每个词按词频写进 `FREQ`
2. **同时**把每个词的**所有非空前缀**也写进 `FREQ`，词频记为 **0**

为什么？因为要支持「在 O(1) 内判断某个片段还可能是更长词的开头」。
构造 DAG 时的循环条件是 `frag in FREQ` —— 如果前缀不在表里，就得提前 break，
否则会一路扫到句尾做大量无意义匹配。

本章实测：`FREQ` **498113** 条，其中 **149068** 条是纯前缀（freq=0）。

**两个必须记住的结论**：

1. **`freq == 0` 的条目不是词**。比如 `温升` 在 `FREQ` 里但 freq=0
   （它是「温升XX」某个更长词的前缀），所以 DAG 不能收它。
2. **`jieba.dt.total != sum(FREQ.values())`**！实测 `60101967` vs `60101964`，**差 3**。
   别自己拿 `sum()` 当分母，DP 的口径会与 jieba 偏掉。
"""
    ),
    code(PREFIX_DICT),
    md(
        """
## 8.5 原理二：DAG —— 把「所有可能的词」列出来

DAG（有向无环图）在这里的含义非常朴素：

> 对句子的每个位置 `k`，列出所有「以 k 开头、且整段都在词典里」的终点 `x`。

于是 `dag[k] = [x1, x2, ...]`，每个 `(k, x)` 就是一条边，对应一个候选词 `s[k:x+1]`。

实现上有个小技巧（和 ch01 的 FMM 同一个套路）：把「往右扩一格」写成 `while` 骨架，
`@@todo` 里只留判定与收尾 —— 因为挖空块里**不允许出现控制流**。

**关键细节**：`if FREQ[frag]` 和 `frag in FREQ` 是**两个不同的判断**。
- `frag in FREQ` 决定「要不要继续往右扩」
- `FREQ[frag]` 决定「这一格算不算词、要不要收进 ends」

写反任何一边都不会报错，只是切分结果全错。
"""
    ),
    code(DAG_FN),
    code(DAG_DEMO),
    md(
        """
## 8.6 原理三：动态规划 —— 在所有路径里挑一条最优的

DAG 只给候选，不给答案。选哪条边，交给 DP。

**打分函数**就是朴素的语言模型概率：

$$P(w) = \\frac{\\text{freq}(w)}{\\text{total}}, \\qquad
\\text{score} = \\sum_i \\log P(w_i)$$

取对数的好处：连乘变连加，**不溢出**，且能用「从后往前递推」求解。

**递推式**（`route[k]` = 从 k 到句尾的最优分值）：

```
route[N] = 0
route[k] = max over x in dag[k] of ( log P(s[k:x+1]) + route[x+1] )
```

这就是一个**最短/最长路径**问题，DAG 上没有环，所以一次倒序扫描就够（O(n·L)）。

**tie-break 也重要**：`max` 在分值相同时会比较元组的第二项，于是会选**更大的 x（更长的词）**
—— 这恰好和 jieba 的实现一致（它也用 `max(generator)`）。

**为什么 DP 比最大匹配强**：最大匹配是贪心，只看「当前这一步能取多长」；
DP 看的是**整条路径的总分**。`研究生命起源` 上 DP 能切对，正是因为
`log P(研究) + log P(生命) + log P(起源)` 严格大于
`log P(研究生) + log P(命) + log P(起源)` —— 贪心看不到这个全局比较。
"""
    ),
    code(DP_HELPER),
    code(DP_FN),
    code(DP_DEMO),
    md(
        """
## 8.7 难点深挖：复刻 jieba 的最后一步，藏在英数字符里

到这里，你的手写实现「看起来」已经和 jieba 一样了。跑一遍全语料：

**裸 DP 与 `jieba.lcut(t, HMM=False)` 的一致条数是 `0 / 570`。**

一条都对不上。这不是小事 —— 说明你的理解里缺了一层。

缺的是 **jieba 在 DP 之后还有一个后处理**：`__cut_DAG_NO_HMM` 会把**连续的单个英数字符**
（`[a-zA-Z0-9]`）攒进一个 buffer，遇到非英数单字符才一起吐出来。

```python
if re_eng.match(l_word) and len(l_word) == 1:
    buf += l_word            # 攒着
else:
    if buf: yield buf        # 先冲掉
    yield l_word
```

为什么会有这一步？因为 DP 只在**词典**里找词，而 `STATION`、`E`、`02` 这些
在中文词典里必然都是单字符 → DP 会把它们**逐字符**切开。
后处理把它们拼回去，`STATION_E_02` 才不会被切碎。

**补上这一步，一致率从 `0/570` 直接跳到 `570/570 = 1.000000`。**

**判定规则**：

> 当一个「复刻版」与「原版」在某项指标上出现 **0% 或 100%** 这种极端值时，
> 说明差的是**某个被跳过的步骤**，不是参数没调好。
> 先去找「原版有哪些后处理/兜底逻辑」，而不是去调你的算法。

> 📌 顺带一提：这道工序恰好是 ch01 里**手写版做对、jieba 做错**的地方 ——
> 手写版用 `SEG_RE` 先做粗切分，把 `STATION_E_02` 整段保护起来；
> jieba 是切完再粘回去。**同一个目标的两种实现路径。**
"""
    ),
    code(CUT_ENG),
    code(REPLICATE),
    md(
        """
## 8.8 HMM：未登录词的「猜词器」

到 §8.7 为止，我们复刻的是 `HMM=False` 的 jieba。**默认的 jieba 还多做一件事**：HMM。

机制很简单：词典 DP 切完之后，对于**连续的单字碎片**，用 HMM（Viterbi）再猜一遍，
判断这些单字该不该合并成词。jieba 的 HMM 是用 BMES 四状态 + 已标注语料训出来的。

**实测影响**：`HMM=True` 与 `HMM=False` 在 **186 / 570 = 32.631579%** 的工单上给出不同结果。
这个比例远高于一般人的直觉（"词典够大就行了"）。

**它对的时候**：

| 输入 | HMM=True | HMM=False |
|---|---|---|
| `3号主变渗漏油` | `3 / 号 / 主变 / 渗漏 / 油` | `3 / 号 / 主 / 变 / 渗漏 / 油` |

`主变` 是电力行业缩写，jieba 词典里没有（连前缀都不是），HMM 靠上下文把它拼了出来。

**它错的时候**：

| 输入 | 输出 |
|---|---|
| `3号主变温升异常` | `3 / 号主 / 变温 / 升 / 异常` ← 两个词都错了 |

**判定规则**：

> HMM 是**统计猜词**，没有领域知识。它在「碎片相邻且高频共现」时表现好，
> 在「数字 + 缩写的短片段」上容易把不该连的连起来。
> **工程做法是别去调 HMM**（`HMM=False` 更可控），而是用用户词典补领域词 —— 见下一节。
"""
    ),
    code(HMM_DEMO),
    md(
        """
## 8.9 自定义词典：把「猜」换成「告诉它」

`load_userdict(path)` 的格式是每行：

```
词 词频 [词性代码]
```

- **词频**：不是真实统计值，是**优先级**。给足够大（如 100）就能压过默认切分。
- **词性**：可选。给了之后 `posseg` 会用这个标签。

本章要找的电力术语：`主变` / `渗漏油` / `母线巡检` / `3号主变` / `温升异常`。

### 一个必须注意的坑：`load_userdict` 污染全局

`jieba.load_userdict()` 改的是**模块级单例** `jieba.dt`。一旦调用，
**后面所有 `jieba.lcut` 都变了** —— 你要做「加词典前 vs 后」的对照就做不成了。

正确做法是用**独立实例**：

```python
tk = jieba.Tokenizer()
tk.initialize()
tk.load_userdict("user.dict")
tk.lcut("...")        # 用了用户词典
jieba.lcut("...")     # 全局仍然是默认词典
```

*（如果非要动全局单例，另一个办法是 `jieba.dt.FREQ` 手动加词，
但那是改内部结构，不推荐。）*

**效果对照**：

| 输入 | 默认 | 用户词典 |
|---|---|---|
| `母线巡检时发现异物，位于3号主变，严重渗漏油` | `母线/巡检/时/发现/异物/，/位于/3/号/主变/，/严重/渗漏/油` | `母线巡检/时/发现/异物/，/位于/3号主变/，/严重/渗漏油` |
| `3号主变温升异常` | `3/号主/变温/升/异常`（错） | `3号主变/温升异常`（对） |

用户词典不是「加几个词」这么简单 —— 它是把**领域先验**注入分词器的唯一正规途径。
"""
    ),
    code(USERDICT_SRC),
    code(USERDICT_DEMO),
    md(
        """
## 8.10 词性标注（`jieba.posseg`）

`posseg.cut()` 返回 `(word, flag)` 的生成器。常用 flag：

| flag | 含义 | flag | 含义 |
|---|---|---|---|
| `n` | 名词 | `a` | 形容词 |
| `v` | 动词 | `d` | 副词 |
| `m` | 数词 | `x` | 标点 / 非语素字 |
| `eng` | 英文串 | `f` | 方位词 |

**两个实用观察**：

1. **词性分布可以用来验证停用词表**。本章实测全语料 **13** 种 flag，
   第一名是 `x`（**2800** 次）—— 也就是标点。
   所以 ch01 §1.9 强调「停用词表里必须有标点」，这里给出了量化依据。
2. **`eng` 有 1140 次** —— 全部来自 `STATION_X_0N` 这类编号。
   如果做「只保留实词」的预处理，`eng` 和 `m` 要不要留，取决于任务：
   分类任务里台区编号是噪声（不该带标签信息），可以丢；
   但要按台区聚合统计时，它就是关键字段。

> ⚠️ `posseg` 会**重新跑一遍切分**（它内部用的是自己的切分 + 标注模型），
> 结果可能与 `jieba.lcut` 不是逐字一致。要「先修正再标注」，
> 应当用 `posseg.cut` 之前先 `load_userdict`（或在 `posseg` 上也挂独立词典）。
"""
    ),
    code(POSSEG),
    md(
        """
## 8.11 关键词提取：TF-IDF 与 TextRank

jieba 自带两套无监督关键词抽取，都**建立在分词结果之上**：

| | `analyse.extract_tags` | `analyse.textrank` |
|---|---|---|
| 依据 | 词频 × 逆文档频率（内置 `idf.txt`） | 词共现图 + PageRank |
| 需要语料 | 不需要（IDF 是内置的通用值） | 不需要（在**当前文本内**建图） |
| 可调参数 | `topK` / `withWeight` / `allowPOS` | `topK` / `withWeight` / `allowPOS` |
| 特点 | 偏向**稀有且重要的词** | 偏向**在文中反复共现**的词 |

本章实测（`母线巡检时发现严重渗漏油，位于3号主变，请尽快核实`）：

- TF-IDF：`['主变', '母线', '渗漏', '巡检', '核实']`
- TextRank：`['渗漏', '巡检', '发现', '主变', '位于']`
- 交集：`主变 / 巡检 / 渗漏`（3 个）

**两个坑**：

1. **关键词质量的上限就是分词质量**。上面两个结果里出现的是 `主变`、`渗漏`（半截词），
   而不是 `3号主变`、`渗漏油` —— 因为默认分词就没切出完整术语。
   **要提关键词，先修用户词典。**
2. **单句上的 TextRank 意义有限**。它靠共现图，一句话里词少、图稀疏，
   pageRank 退化得很快。TextRank 的常规用法是**整篇文档**或**整个语料**。

**TF-IDF 的 `extract_tags` 打的是单文档分**：它把当前字符串当成一篇文档，
`tf` 来自本文，`idf` 来自内置表。想做语料级的关键词，应当用 ch02 手写的 TF-IDF
（可控、可对账），或者对全语料统计后再 `extract_tags`。
"""
    ),
    code(KEYWORD),
    md(
        """
## 8.12 量化对账：手写 vs jieba（570 条）

前面都是小句子，容易被个例误导。这里把 570 条工单全部过一遍。

**预期你会看到的三件事**：

1. **jieba 的词表更大**（94 vs 77），**平均 token 数几乎翻倍**（15.77 vs 8.54）。
   原因：手写词的词表里有 `请核实后反馈`、`请安排抢修` 这类整句短语，
   一刀就切掉一长串；jieba 必须按通用词切，而且把 `STATION_E_02` 切成 5 个 token。
2. **逐句完全一致：0 / 570**。注意这不是「谁错」—— 两份词表本来就不一样。
3. **速度**：jieba 切 570 条约 **0.02s**，比手写 Python `fmm` 还快（它有前缀词典剪枝 + C 加速的 DP）。

**判读结论**：在「统一词表」的封闭场景里，手写词表**更紧凑**；
在「开放场景、没有词表」时，jieba 是唯一可用的选择。
"""
    ),
    code(COMPARE),
    code(DIFF_SHOW),
    md(
        """
## 8.13 下游检验：切分方式真的影响分类吗

这是本章最重要的对照实验。同一份 570 条数据、同一份标签、同一个分类器
（词袋 + 逻辑回归，5 折交叉验证），只改分词方式：

| 切分方式 | 词表 | 特征维度 | 5 折准确率 |
|---|---|---|---|
| 手写最大匹配 | 77 | 77 | **1.000000** |
| jieba 默认 | 94 | 94 | **1.000000** |
| jieba + 用户词典 | 94 | 94 | **1.000000** |

**三家打平，全是满分。**

这是一个**负结果**，而且是本章最有价值的一条结论。它说明：

> 分词不是瓶颈 —— 数据集的**五类信号词互不重叠**（见 README §2.3），
> 所以无论怎么切，「渗油 / 锈蚀 / 碎裂 / 发热 / 异物」这些锚点总能被切出来，
> 词袋 + 线性分类器就足够。

**这也解释了为什么「中文分词」在竞赛里通常不是得分点**：

> 不是因为它不重要，而是因为**评测数据的信号词足够显著**时，分词方式的差异被淹没了。
> 真正会拉开差距的是：**特征设计**（ch02）、**模型结构**（ch04~ch06）、
> **微调策略**（ch07）—— 以及在**噪声更大的真实数据**上，分词才开始起作用。

**反向验证**：如果这份数据的信号词是「重叠」的（比如「油」同时出现在渗漏油和润滑故障里），
那么 `渗漏/油` 与 `渗漏油` 的差别就会影响结果。**要判断分词重不重要，先看信号词是否唯一指向标签。**
"""
    ),
    code(DOWNSTREAM),
    md(
        """
## 易错点清单

1. **忘了 `jieba.initialize()`** —— `jieba.dt.FREQ` 会是空 dict、`jieba.dt.total` 是 **0**，
   后面 `math.log(TOTAL)` 直接 `ValueError: math domain error`。
2. **以为 `FREQ` 里的都是词** —— 49.8 万条里有 **14.9 万**条是 freq=0 的纯前缀
   （如 `温升`）。DAG 用 `if FREQ[frag]` 过滤，漏了这层过滤就会切出一堆假词。
3. **把 `frag in FREQ` 和 `FREQ[frag]` 写成一个判断** —— 前者管「继续扩」，后者管「算不算词」，
   合起来写不报错但结果全错。
4. **自己用 `sum(FREQ.values())` 当分母** —— 实测与 `jieba.dt.total` **差 3**。
5. **以为复刻完 DP 就完了** —— 裸 DP 与 `jieba.lcut(HMM=False)` 的一致率是 **0/570**，
   缺的是「连续英数字符合并」这一步；补上就是 **570/570**。
6. **用全模式的结果喂模型** —— `cut_all=True` 输出的词**互相重叠**，
   同一个字会被统计多次，特征维度与权重全乱。
7. **把 `lcut_for_search` 当精确模式用** —— 它会对 `变压器` 额外吐出 `变压`，
   下游词表凭空变大（本章实测 5 → 6 个 token）。
8. **`load_userdict` 直接调全局** —— 污染 `jieba.dt`，之后所有 `lcut` 都变了；
   要做前后对照必须用 `jieba.Tokenizer()` 独立实例。
9. **以为 `HMM=True` 总是更好** —— 它把 `3号主变温升异常` 切成 `号主 / 变温`。
   OOV 场景该用用户词典，不是靠 HMM 猜。
10. **用户词典里词频写 0 或太小** —— 词频是优先级，压不过默认切分就白写。
11. **指望 `posseg` 与 `lcut` 结果逐字一致** —— `posseg` 内部会重新切分，
    两者可以不同；要一致就得对 `posseg` 也挂同一份用户词典。
12. **以为「更好的分词」一定能涨点** —— 本章实测三种切法下游 **1.000000 / 1.000000 / 1.000000**。
    先确认信号词是否唯一指向标签，再决定要不要在分词上花时间。

## 本章小结

| 环节 | 关键判断 | 本章数字 |
|---|---|---|
| 三方对照 | 不存在「谁更好」，只存在「谁的词典更贴数据」 | `研究生命起源` 三方三种结果 |
| 三种模式 | 只有精确模式是合法切分 | 5 / 6 / 6 个 token |
| 前缀词典 | `freq==0` 的是前缀，不是词 | 498113 条，其中 149068 条纯前缀 |
| DAG | `in FREQ` 管扩、`FREQ[...]` 管收 | `研究生命起源` 的 DAG 6 个起点 10 条边 |
| DP | 全场路径最优，不是贪心 | `研究/生命/起源` |
| **英数符合并** | **不做这步 0/570，做了 570/570** | **0.000000 → 1.000000** |
| HMM | 会改 **32.63%** 的句子，有对有错 | 186 / 570 |
| 用户词典 | 独立实例，别污染全局 | `3/号主/变温/升/异常` → `3号主变/温升异常` |
| 词性 | 标点是第一名 → 停用词必须含标点 | `x` 2800 / 13 种 |
| 关键词 | 上限由分词决定 | 交集 3 个，全是半截词 |
| 下游 | **切分方式不影响这批数据的准确率** | `1.000000 / 1.000000 / 1.000000` |

**本章一句话**：**jieba = 前缀词典 + DAG + 对数概率 DP + 英数符合并 + HMM**，
五步全都能手写复刻（`570/570` 对账通过）；但**复刻它不是目的** ——
目的是知道每一步在什么情况下会出错，以及**什么时候这些差别根本不影响下游指标**。

**下一章**：把 `transformers` 真正用起来 —— `Config` 建模型、`Trainer` 走工业流程，
并回答一个更实际的问题：**给定参数量的模型，我这台机器到底训不训得动。**
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch08 jieba 中文分词实战（练习版）

> 按提示补全 `____`，跑通所有 assert。真值全部来自实跑（jieba 0.42.1）。
>
> 本章依赖 `jieba`（已装）；**不下载任何模型**，断网可跑。
> `VOCAB` / `clean` / `fmm` / `bmm` / `tokenize` 已给好（与 ch01 同口径），
> 你只需要补 jieba 相关的部分。挖空块内每条语句保持**单行**，控制流都在块外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    md(
        """
## 任务 1：三方对照（FMM / BMM / jieba）

同一句话切三遍。注意 `电流互感器发热` 这一行 —— **手写对了、jieba 错了**，
因为 jieba 的通用词典里没有这个电力复合词。
"""
    ),
    code(TRIANGLE),
    md(
        """
## 任务 2：jieba 三种模式

精确模式是**合法切分**（首尾相接拼回原句），全模式**不是**（词互相重叠）。
"""
    ),
    code(MODES),
    md(
        """
## 任务 3：前缀词典的真实结构

`jieba.dt.FREQ` 把「词」和「词的所有前缀」放在同一张表里，**纯前缀记 freq=0**。
另外注意最后一行：`sum(FREQ.values())` 和 `jieba.dt.total` **不相等**。
"""
    ),
    code(PREFIX_DICT),
    md(
        """
## 任务 4：手写 DAG 构造

`while i < N and frag in FREQ` 已经在帮你往右扩了，你只需要**判定 + 收尾**。
关键：`freq==0` 的前缀不能收，但**还得继续往后扩**。
"""
    ),
    code(DAG_FN),
    code(DAG_DEMO),
    md(
        """
## 任务 5：对数概率 DP

从后往前扫，每次在 `dag[idx]` 的所有出边里取「本词权重 + 后缀最优值」最大的一格。
"""
    ),
    code(DP_HELPER),
    code(DP_FN),
    code(DP_DEMO),
    md(
        """
## 任务 6：复刻的最后一步 —— 英数字符合并

`merge_eng` 已经给好了（有控制流，不能放挖空块里）。
你只需要把 `cut_dp` 的结果交给它。
"""
    ),
    code(CUT_ENG),
    md(
        """
## 任务 7：全语料对账

这一步会给你本章最震撼的数字：`0/570` → `570/570`。
"""
    ),
    code(REPLICATE),
    md("## 任务 8：HMM 开关（未登录词猜词）"),
    code(HMM_DEMO),
    md(
        """
## 任务 9：用户词典（电力术语）

按 `词 词频 [词性]` 格式写一份词典文件。词频是**优先级**，要给得够大。
"""
    ),
    code(USERDICT_SRC),
    md(
        """
## 任务 10：用独立 Tokenizer 加载词典

**不要**用 `jieba.load_userdict` 直接改全局 —— 后面还要用默认分词做对照。
"""
    ),
    code(USERDICT_DEMO),
    md("## 任务 11：词性标注 + 语料级词性分布"),
    code(POSSEG),
    md("## 任务 12：TF-IDF / TextRank 关键词"),
    code(KEYWORD),
    md(
        """
## 任务 13：570 条量化对账

统计两边的词表大小、平均 token 数、总 token 数，以及「逐句完全一致」的条数。
"""
    ),
    code(COMPARE),
    code(DIFF_SHOW),
    md(
        """
## 任务 14：下游检验（切分方式 × 分类器）

三种切法喂同一个分类器。**答案会让你失望：全部打平。**
但请想清楚原因，这是本章最有价值的一条结论。
"""
    ),
    code(DOWNSTREAM),
    md(
        """
## 自查清单

- [ ] 调用了 `jieba.initialize()`（不调的话 `dt.total` 是 0）
- [ ] 能说出 `freq==0` 的条目是什么（前缀，不是词）
- [ ] DAG 里 `frag in FREQ` 管「继续扩」，`FREQ[frag]` 管「算不算词」
- [ ] 知道 `jieba.dt.total != sum(FREQ.values())`
- [ ] 裸 DP 与 `jieba(HMM=False)` 一致率是 0/570
- [ ] 补上英数符合并后一致率是 570/570
- [ ] 全模式与搜索引擎模式的区别说得清（重叠 vs 补充细粒度）
- [ ] 用 `jieba.Tokenizer()` 而不是 `load_userdict` 改全局
- [ ] 用户词典里的词频给得足够大
- [ ] 举得出 HMM **改对**和**改错**各一个例子
- [ ] 知道全语料词性的第一名是 `x`（标点）
- [ ] 接受「三种切法下游 5 折准确率全是 1.000000」这个结果，并能解释原因
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
