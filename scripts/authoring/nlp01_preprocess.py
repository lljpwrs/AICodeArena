#!/usr/bin/env python3
"""nlp01 文本预处理：口径陷阱 / 正则清洗 / 手写最大匹配分词 / 停用词 / n-gram / 词干。

数据：`data/synth_text.csv`（600 条电力设备缺陷工单，由 `data/make_data.py` 生成）。
真值全部实跑（pandas 3.x + 纯标准库，无第三方 NLP 依赖 —— 分词是自己写的）。

> 本章三条主线：
> ① **读取口径决定数字** —— 同一份 CSV，`read_csv` 默认把 `""` 与 `N/A` 认成 NaN，
>    「缺失 12 条」和「缺失 0 条」说的是同一批数据；
> ② **最大匹配的两个方向会给出不同切分** —— 「研究生命起源」正向切出
>    `研究生/命/起源`，逆向切出 `研究/生命/起源`；但在 600 条真实工单上两者
>    **0 条分歧**（歧义只存在于人为构造的词对里）；
> ③ **词干 ≠ 词形还原** —— `running` 的简化词干是 `runn` 而不是 `run`。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/nlp01_preprocess.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"
NAME = "ch01_preprocess"

# =========================================================================== #
# 公共部分（两版都给）                                                        #
# =========================================================================== #

IMPORTS = '''import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data")
CSV = DATA / "synth_text.csv"
'''

SETUP = '''# 文本数据一定要关掉默认的 NA 解析：否则 "" 和 "N/A" 会被静默吃掉（见 §1.2）
df = pd.read_csv(CSV, keep_default_na=False)

print("样本数", len(df), "| 列数", df.shape[1])
print("列名：", list(df.columns))
print("标签分布：", df["缺陷类型"].value_counts().to_dict())
print()
print(df[["工单编号", "设备类型", "缺陷类型", "缺陷描述"]].head(4).to_string(index=False))
'''

VOCAB_SRC = '''VOCAB = [
    # 设备
    "变压器", "断路器", "隔离开关", "避雷器", "电流互感器", "绝缘子", "套管", "母线",
    # 缺陷信号词（五类互不重叠）
    "渗油", "漏油", "油迹", "油污", "滴油", "油位下降",
    "锈蚀", "生锈", "铁锈", "漆面剥落", "镀锌层脱落",
    "破损", "裂纹", "碎裂", "缺口", "掉瓷", "箱体变形",
    "发热", "温度偏高", "过热", "烫手", "温升异常", "过热保护",
    "异物", "鸟巢", "塑料袋", "树枝", "风筝线搭挂",
    # 动作与描述
    "发现", "巡检", "位于", "存在", "缺陷", "一般", "严重", "危急",
    "接报", "详见", "保护", "动作",
    # 高频短语（故意把整句当词，看最大匹配怎么处理长词）
    "请安排抢修", "请尽快核实", "已通知运维班", "请列入本周计划",
    "需停电处理", "请核实后反馈",
    # 教学用词：专门用来演示「正向 ≠ 逆向」
    "研究", "研究生", "生命", "命", "起源",
]
'''

SCAFFOLD = '''# 脚手架：词表、停用词、正则常量（两版都有，你只填 @@todo 块）
VOCAB_SET = set(VOCAB)
MAX_WORD_LEN = max(len(w) for w in VOCAB)

# 粗切分：连续 ASCII 字母数字下划线 / 连续汉字 / 单个其他字符
SEG_RE = re.compile(r"[A-Za-z0-9_]+|[\\u4e00-\\u9fff]+|[^\\sA-Za-z0-9_\\u4e00-\\u9fff]")
# 注意这里的 + ：没有 + 就只能匹配单个汉字，而 fullmatch 要求整串匹配（见 §1.8）
HAN_RE = re.compile(r"[\\u4e00-\\u9fff]+")

STOPWORDS = {
    "的", "了", "在", "是", "和", "与", "及", "或", "也", "就", "都",
    "，", "。", "！", "？", "、", "（", "）", "【", "】", "：", "；", " ",
}

print("词表", len(VOCAB), "个词 | 最长", MAX_WORD_LEN, "字 | 停用词", len(STOPWORDS), "个")
'''

# =========================================================================== #
# 各任务的代码块                                                              #
# =========================================================================== #

OVERVIEW = '''desc = df["缺陷描述"]

print("字符数统计（清洗前）：")
print(desc.str.len().describe().to_string())
print()
print("长度前 3 的样本：")
for s in desc.str.len().nlargest(3).index:
    print(" ", repr(desc[s])[:90], "...")
'''

NA_TRAP = '''# 同一份 CSV 读两遍：一遍用默认设置，一遍关掉 NA 解析
raw = pd.read_csv(CSV)
safe = pd.read_csv(CSV, keep_default_na=False)

# @@todo 分别统计两版「缺陷描述」的缺失值个数
# @@hint Series.isna().sum()
n_raw = int(raw["缺陷描述"].isna().sum())
n_safe = int(safe["缺陷描述"].isna().sum())
# @@end

print("默认读取的缺失值：", n_raw)
print("关闭 NA 解析的缺失值：", n_safe)
print("默认读取的 dtype：", raw["缺陷描述"].dtype)

# ---- 验收 ----
assert n_raw == 12, f"默认读取应吃掉 12 条，实际 {n_raw}"
assert n_safe == 0, f"关掉 NA 解析后不应有缺失，实际 {n_safe}"
'''

INSPECT = '''desc = df["缺陷描述"]

# @@todo 统计「空 / 纯空白」的条数
# @@hint 先 Series.str.strip() 再和 "" 比较
n_blank = int((desc.str.strip() == "").sum())
# @@end

# @@todo 统计含 HTML 标签的条数
# @@hint Series.str.contains("<p>", regex=False)
n_html = int(desc.str.contains("<p>", regex=False).sum())
# @@end

n_url = int(desc.str.contains("http", regex=False).sum())
n_half = int(desc.str.contains(",", regex=False).sum())
n_bang = int(desc.str.contains("！！", regex=False).sum())
n_ws = int(((desc.str.contains("  ", regex=False)) | (desc != desc.str.strip())).sum())
n_bad = int(desc.str.strip().isin(["", "N/A", "-", "无"]).sum())

print(f"空/纯空白     {n_blank}")
print(f"含 HTML      {n_html}")
print(f"含 URL       {n_url}")
print(f"含半角逗号     {n_half}")
print(f"含重复标点     {n_bang}")
print(f"含多余/首尾空白  {n_ws}")
print(f"无效占位符     {n_bad}")
print(f"去重后         {desc.nunique()}")

# ---- 验收 ----
assert n_blank == 12, f"空/纯空白应为 12，实际 {n_blank}"
assert n_html == 107, f"含 HTML 应为 107，实际 {n_html}"
assert n_url == 82 and n_half == 131 and n_bang == 41
assert n_ws == 180 and n_bad == 30 and desc.nunique() == 575
'''

CLEAN_FN = '''def clean(text: str) -> str:
    """把一条工单描述洗成干净的中文句子。"""
    # @@todo 去掉 HTML 标签，再去掉 URL
    # @@hint re.sub(r"<[^>]+>", "", text)；URL 用 r"https?://\\S+"
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"https?://\\S+", "", s)
    # @@end

    # @@todo 半角逗号统一成全角；把连续重复的句末标点压成一个
    # @@hint str.replace(",", "，")；re.sub(r"([，。！？])\\1+", r"\\1", s)
    s = s.replace(",", "，")
    s = re.sub(r"([，。！？])\\1+", r"\\1", s)
    # @@end

    # @@todo 去掉汉字之间的空格（OCR / 复制粘贴噪声），再把连续空白压成一个并收尾
    # @@hint re.sub(r"(?<=[\\u4e00-\\u9fff])\\s+(?=[\\u4e00-\\u9fff])", "", s) 后接 re.sub(r"\\s+", " ", s).strip()
    s = re.sub(r"(?<=[\\u4e00-\\u9fff])\\s+(?=[\\u4e00-\\u9fff])", "", s)
    s = re.sub(r"\\s+", " ", s).strip()
    # @@end
    return s
'''

CLEAN_APPLY = '''desc_clean = df["缺陷描述"].map(clean)

print("清洗后平均字符数：", round(float(desc_clean.str.len().mean()), 6))
print("清洗前平均字符数：", round(float(df["缺陷描述"].str.len().mean()), 6))
print()
print("样例对照：")
i = 0
print("  原文：", repr(df["缺陷描述"][i]))
print("  清洗：", repr(desc_clean[i]))

# ---- 验收 ----
assert int((desc_clean.str.strip() == "").sum()) == 12
assert int(desc_clean.str.contains("<", regex=False).sum()) == 0
assert int(desc_clean.str.contains("http", regex=False).sum()) == 0
assert int(desc_clean.str.contains(",", regex=False).sum()) == 0
assert abs(float(desc_clean.str.len().mean()) - 28.513333) < 1e-5
'''

GREEDY = '''greedy_demo = "<p>文本</p><p>更多</p>"

# @@todo 分别用贪婪和非贪婪写法去掉所有标签
# @@hint r"<.*>" 会一路吃到最后一个 ">"；r"<.*?>" 只吃到最近的 ">"
greedy = re.sub(r"<.*>", "", greedy_demo)
lazy = re.sub(r"<.*?>", "", greedy_demo)
# @@end

print("贪婪  <.*>  ->", repr(greedy))
print("非贪婪 <.*?> ->", repr(lazy))

# ---- 验收 ----
assert greedy == "", "贪婪会连中间的正文字符一起吃掉"
assert lazy == "文本更多", "非贪婪才只去标签、留下正文"
'''

FMM = '''def fmm(seg: str) -> list[str]:
    """正向最大匹配：从左往右扫，每次取词表里能匹配上的最长词。"""
    out, i = [], 0
    while i < len(seg):
        length = min(MAX_WORD_LEN, len(seg) - i)
        while length > 1 and seg[i:i + length] not in VOCAB_SET:
            length -= 1
        # @@todo 把匹配到的这一段并入结果，并把指针右移它的长度
        # @@hint out.append(...)；i += length
        out.append(seg[i:i + length])
        i += length
        # @@end
    return out
'''

FMM_DEMO = '''print("变压器渗油      ->", "/".join(fmm("变压器渗油")))
print("巡检发现渗油     ->", "/".join(fmm("巡检发现渗油")))
print("研究生命起源     ->", "/".join(fmm("研究生命起源")))

# ---- 验收 ----
assert fmm("变压器渗油") == ["变压器", "渗油"]
assert fmm("巡检发现渗油") == ["巡检", "发现", "渗油"]
'''

BMM = '''def bmm(seg: str) -> list[str]:
    """逆向最大匹配：从右往左扫，每次取词表里能匹配上的最长词。"""
    out, i = [], len(seg)
    while i > 0:
        length = min(MAX_WORD_LEN, i)
        while length > 1 and seg[i - length:i] not in VOCAB_SET:
            length -= 1
        # @@todo 从右侧取词插到结果最前面，并把指针左移它的长度
        # @@hint out.insert(0, ...)；i -= length
        out.insert(0, seg[i - length:i])
        i -= length
        # @@end
    return out
'''

BMM_DEMO = '''fmm_ans = fmm("研究生命起源")
bmm_ans = bmm("研究生命起源")

print("正向：", "/".join(fmm_ans))
print("逆向：", "/".join(bmm_ans))
print("词数：", len(fmm_ans), "vs", len(bmm_ans))

both_diff = sum(1 for s in desc_clean if fmm(s) != bmm(s))
print("600 条工单里两方向不一致的条数：", both_diff)

# ---- 验收 ----
assert bmm("研究生命起源") == ["研究", "生命", "起源"]
assert fmm("研究生命起源") != bmm("研究生命起源")
assert both_diff == 0, "真实工单里两种方向其实并不分歧"
'''

TOKENIZE = '''def tokenize(text: str) -> list[str]:
    """先粗切分，汉字段走最大匹配，其余（英文 / 数字 / 标点）整体入列。"""
    out = []
    for seg in SEG_RE.findall(text):
        # @@todo 判断这一段是不是纯汉字：是则走 fmm，否则整段作为一个 token
        # @@hint HAN_RE.fullmatch(seg) 作三目条件
        pieces = fmm(seg) if HAN_RE.fullmatch(seg) else [seg]
        out.extend(pieces)
        # @@end
    return out
'''

TOKENIZE_DEMO = '''sample = desc_clean[0]
print("样例：", sample)
print("分词：", tokenize(sample))
print()
print("如果跳过粗切分、直接对整句做最大匹配：")
print(" ", fmm(sample)[:12], "...")

# ---- 验收 ----
assert tokenize(sample) == [
    "STATION_E_02", "绝缘子", "渗油", "，", "请核实后反馈", "，", "请安排抢修",
]
assert len(fmm(sample)) > len(tokenize(sample)), "裸 FMM 会把编号逐字符切碎"
'''

STOP = '''tokens = tokenize(desc_clean[0])

# @@todo 过滤掉停用词，以及纯空白的 token
# @@hint 列表推导 + `w not in STOPWORDS and w.strip()`
kept = [w for w in tokens if w not in STOPWORDS and w.strip()]
# @@end

print("分词结果：", tokens)
print("去停用词：", kept)
print("token 数：", len(tokens), "->", len(kept))

# ---- 验收 ----
assert len(tokens) == 7 and len(kept) == 5
assert "，" not in kept
'''

STOP_ALL = '''all_tokens = [
    w
    for s in desc_clean
    for w in tokenize(s)
    if w not in STOPWORDS and w.strip()
]
counter = Counter(all_tokens)

print("全语料 token 总数：", len(all_tokens))
print("词表大小（去停用词后）：", len(counter))
print("Top 10：", counter.most_common(10))

# ---- 验收 ----
assert len(all_tokens) == 3461
assert len(counter) == 75
'''

NGRAM = '''def ngrams(seq, n):
    """返回所有长度为 n 的连续子序列（元组形式）。"""
    # @@todo 用列表推导扫描所有合法起点
    # @@hint range(len(seq) - n + 1) 配合 seq[i:i + n]
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    # @@end
'''

NGRAM_DEMO = '''base = kept
print("序列：", base)
print("1-gram：", ngrams(base, 1))
print("2-gram：", ngrams(base, 2))
print("3-gram：", ngrams(base, 3))
print()
print("字级 2-gram「电力设备」：", ngrams(list("电力设备"), 2))

# ---- 验收 ----
assert len(ngrams(base, 1)) == 5
assert len(ngrams(base, 2)) == 4
assert len(ngrams(base, 3)) == 3
assert ngrams(list("电力设备"), 2) == [("电", "力"), ("力", "设"), ("设", "备")]
'''

STEM = '''SUFFIXES = ("ational", "ization", "iveness", "fulness", "ousness", "ing", "ed", "ly", "es", "s")


def stem(word: str) -> str:
    """简化版 Porter 词干：命中后缀且剩余长度 >= 3 就切掉。"""
    w = word.lower()
    hit = next((s for s in SUFFIXES if w.endswith(s) and len(w) - len(s) >= 3), None)
    # @@todo 命中后缀就把后缀切掉，否则原样返回
    # @@hint w[: -len(hit)] if hit else w
    return w[: -len(hit)] if hit else w
    # @@end
'''

STEM_DEMO = '''words = ["running", "studies", "organization", "cars", "boxes", "quickly", "flying"]
for w in words:
    print(f"  {w:14s} -> {stem(w)}")

print()
print("注意：running -> runn（不是 run）。词干只负责砍后缀，不保证结果是真词；")
print("      running -> run 属于词形还原（lemmatization），要靠词表或词性标注。")

# ---- 验收 ----
assert stem("running") == "runn", "简化词干砍掉 ing 后得到 runn，不会自动补成 run"
assert stem("organization") == "organ"
assert stem("cars") == "car"
assert stem("quickly") == "quick"
assert stem("flying") == "fly"
'''

# =========================================================================== #
# 讲解版                                                                      #
# =========================================================================== #

LESSON = [
    md(
        """
# ch01 文本预处理（讲解版）

> **本节考点**：`docs/竞赛考点对照表.md` 里「文本 / 工单分类」的第一步，
> 也对应技能操作「数据准备及处理 10%」在文本侧的落点：
> **把一堆脏文本洗成可以喂给模型的形式**。

数据：`data/synth_text.csv` —— 600 条电力设备缺陷工单，五类标签各 120 条，
由 `data/make_data.py` 用固定种子合成（**不下载任何公开语料**，断网可跑）。

**本章主线**：预处理不是「调个包」，而是**一连串会静默改变结果的判断**。
三个最容易翻车的地方，本章都会用实跑数字讲清楚：

| 坑 | 现象 |
|---|---|
| 读取口径 | 同一份 CSV，「缺失 12 条」和「缺失 0 条」都对 |
| 分词方向 | 「研究生命起源」正向 / 逆向切分不同，但真实工单 0 条分歧 |
| 词干 vs 词形还原 | `running` 的词干是 `runn`，不是 `run` |
"""
    ),
    md(
        """
## 学习目标 & API 速查

**学习目标**

1. 能用正则把 HTML / URL / 空白 / 半角标点 / 重复标点洗掉，并知道每一步为什么必要
2. 能手写**正向 / 逆向最大匹配**分词器，并解释两者的差异边界
3. 能处理「中文里夹英文编号」这种真实场景，不被逐字符切碎
4. 能区分**停用词过滤**、**词干提取**、**词形还原**三件事
5. 能构造词级 / 字级 n-gram 特征

**API 速查表**

| 操作 | API | 返回 | 一句话说明 |
|---|---|---|---|
| 读文本 CSV | `pd.read_csv(path, keep_default_na=False)` | DataFrame | **文本列必须加这个参数**，否则空串变 NaN |
| 缺失统计 | `Series.isna().sum()` | int | |
| 正则替换 | `re.sub(pat, repl, s)` | str | `.*` 贪婪、`.*?` 非贪婪 |
| 查找全部 | `re.findall(pat, s)` | list | 用于粗切分 |
| 整串匹配 | `re.fullmatch(pat, s)` | Match/None | 判断「是不是纯汉字段」 |
| 长度 | `Series.str.len()` | Series | 字符数（非字节） |
| 去空白 | `Series.str.strip()` / `re.sub(r"\\s+", " ", s)` | 同型 | |
| 计数 | `collections.Counter(list)` | dict-like | `most_common(n)` 取 Top-N |
| 词表化 | `set(vocab)` | set | 分词时 `in` 查询 **O(1)**，这点很关键 |
"""
    ),
    code(IMPORTS),
    md(
        """
## 1.1 数据长什么样

先看形状、列名、标签分布和几条真样本。**任何预处理之前都要先做这一步** ——
不看数据就写清洗规则，等于闭着眼睛改。
"""
    ),
    code(SETUP),
    code(OVERVIEW),
    md(
        """
## 1.2 难点深挖：`""` 到底是不是缺失值？

**为什么难**：`pd.read_csv` 有一张默认的「NA 词表」，里面包含 `""`、`"N/A"`、`"NA"`、
`"NULL"`、`"None"`、`"nan"` 等。所以**空字符串不会原样进来，它会变成 `NaN`**。
对数值列这是好事，对文本列这是灾难 —— 一条「正文为空」的工单，
和一条「本来就没填」的工单，被合并成了同一种东西。

**错误示范**：直接用默认设置读。

**判定规则**：

> 只要有一列是**文本**（尤其是要做「空文本过滤」「长度统计」「分词」的列），
> 就加 `keep_default_na=False`。
> 需要真实缺失语义的数值列，反而应该保留默认行为。
> **一份数据里两种需求可以并存，但要你自己指定 `na_values`，别靠默认。**
"""
    ),
    code(NA_TRAP),
    md(
        """
## 1.3 先把「脏」数清楚

清洗之前必须**量化**：多少条是空的、多少条带 HTML、多少条混了半角标点。
否则你无法验证清洗是否真的生效 —— 只能「感觉干净了」。

下面这张就是本章的**真值锚点**，后面每一步清洗都要对着它验收。
"""
    ),
    code(INSPECT),
    md(
        """
## 1.4 正则清洗五步管道

| 步 | 目标 | 正则 | 为什么 |
|---|---|---|---|
| 1 | 去 HTML | `<[^>]+>` | 工单系统导出的富文本标签 |
| 2 | 去 URL | `https?://\\S+` | 链接对分类是纯噪声 |
| 3 | 半角→全角逗号 | `,` → `，` | 中英标点混用会让词表无谓膨胀一倍 |
| 4 | 压重复标点 | `([，。！？])\\1+` | `，，，` / `！！！` 是情绪噪声 |
| 5 | 去汉字间空格 | `(?<=[一-龥])\\s+(?=[一-龥])` | OCR / 复制粘贴会把「渗油」拆成「渗 油」 |

第 5 步的写法值得单独说：它用了**零宽断言**（`(?<=...)` 和 `(?=...)`），
只匹配「夹在两个汉字中间的空格」，不会碰 `STATION_E_02` 里的下划线。
"""
    ),
    code(CLEAN_FN),
    code(CLEAN_APPLY),
    md(
        """
## 1.5 难点深挖：`<.*>` 与 `<.*?>` 差在哪

**为什么难**：两个正则都能「去掉标签」，在小样例上结果完全一样，
所以没人会去查它们的区别 —— 直到遇到一行里有多个标签。

**错误示范**：用贪婪写法处理 `"<p>文本</p><p>更多</p>"`。

**判定规则**：

> **`*` / `+` 默认贪婪，会一路吃到最后一个可能的位置。**
> 想「只吃到最近的结束符」，就在量词后面加 `?`（`*?` / `+?`）。
> 处理成对的标记（HTML 标签、括号、引号）时**默认就该用非贪婪**，
> 除非你明确知道要跨标记匹配。
"""
    ),
    code(GREEDY),
    md(
        """
## 1.6 手写分词：正向最大匹配（FMM）

中文没有空格，所以第一步是**切词**。最大匹配是最经典的无监督分词法：

> 从左往右，在每个位置**尽量取词表里最长的词**；取不到就单字成词。

实现上有个小技巧能避开「块内不能写控制流」的限制：
把「缩短候选长度」写成内层 `while` 骨架，块里只剩两行 —— **收词 + 移指针**。

`while length > 1 and seg[i:i+length] not in VOCAB_SET` 的意思是：
只要还没缩到 1 个字、又不在词表里，就把候选缩短一格。
缩到 1 之后无条件收下 —— 效果等价于「找不到就单字成词」。
"""
    ),
    code(VOCAB_SRC),
    code(SCAFFOLD),
    code(FMM),
    code(FMM_DEMO),
    md(
        """
## 1.7 难点深挖：正向与逆向，谁对？

**为什么难**：两种扫法看起来只是方向不同，很多人以为结果应该一样。

**错误示范**：拿「研究生命起源」跑两个方向。

**判定规则**：

> 正向贪心会先把左边的长词吃掉，导致右边剩下碎片；
> 逆向贪心则先把右边的长词吃掉。
> **只有存在「交叉歧义」（一个词跨越另一个词的边界）时，两者才分歧。**
> 在 600 条规则生成的工单上，两者**一条都不分歧** —— 这恰好说明
> 靠人工规则的分词上限在哪，也是后来统计分词（HMM / CRF / BERT 分词）登场的原因。

补一个真实例子：`"断路器过热保护动作"` 两个方向都切出
`断路器 / 过热保护 / 动作` —— 因为「过热保护」是个完整长词，没有交叉。
**「不发生歧义」比「发生歧义」更常见。**
"""
    ),
    code(BMM),
    code(BMM_DEMO),
    md(
        """
## 1.8 中文里的英文编号：先粗切分再分词

真实工单里一定夹着 `STATION_E_02`、`WO_0001` 这类编号。
如果把它们直接丢给最大匹配，会**逐字符切碎**成 `S / T / A / T / I / O / N ...`，
词表里凭空多出一堆单字母，分类器会被这些噪声维度干扰。

标准做法是**两级切分**：

1. 先用正则把文本粗切成「连续 ASCII 字母数字下划线」/「连续汉字」/「其他单字符」三类段
2. 只有**纯汉字段**才走最大匹配；其余段整体作为一个 token

`re.fullmatch` 用来判断「这一段是不是纯汉字」—— 注意是 `fullmatch` 不是 `match`，
`match` 只要求**开头**匹配，`"a汉字"` 也会被判成汉字段。
"""
    ),
    code(TOKENIZE),
    code(TOKENIZE_DEMO),
    md(
        """
## 1.9 停用词过滤

停用词表看起来最没技术含量，但它有两个容易被忽略的点：

1. **标点也要进停用词表**。中文标点（`，。！？`）不是「词」，但分词器会把它切出来，
   不过滤就会进词表。
2. **过滤后的词表大小才是真正的特征维度**。这一步的数字要记下来，
   它是 ch02 词袋的输入规模。
"""
    ),
    code(STOP),
    code(STOP_ALL),
    md(
        """
## 1.10 n-gram：把词序信息捡回来

词袋模型会丢掉词序 —— 「电流互感器发热」和「发热电流互感器」在词袋里一模一样。
n-gram 是最便宜的补救手段：把**连续 n 个 token** 当成一个整体特征。

- **词级 n-gram**：`("变压器", "渗油")` —— 捕捉词组搭配
- **字级 n-gram**：`("电", "力")` —— 中文里常用于**绕过分词**（干脆不切词）

代价：特征数会爆炸（词表 V 个词，2-gram 理论上限 V² 个）。
"""
    ),
    code(NGRAM),
    code(NGRAM_DEMO),
    md(
        """
## 1.11 难点深挖：词干提取 ≠ 词形还原

**为什么难**：中文教材经常把两者混着讲，但它们在英文里是**两个完全不同的任务**。

| | 词干提取 stemming | 词形还原 lemmatization |
|---|---|---|
| 做法 | 砍后缀（纯规则） | 查词表 / 词性分析 |
| 输出 | **可能不是真词** | **一定是真词** |
| `running` | `runn` | `run` |
| `studies` | `studi` | `study` |
| 成本 | 极低 | 需要词典 + 词性 |

**判定规则**：

> 只要**下游是统计模型**（TF-IDF、词袋、线性分类器），用词干就够了 ——
> `runn` 和 `run` 是不是真词不重要，重要的是**同一个词的不同形态落到同一个桶里**。
> 只有当输出**要给人看**（检索展示、可读报告）时，才值得上词形还原。

本章的 `stem` 是 Porter 的**简化版**（只保留了最常用的后缀表），
与标准 Porter 在个别词上会有差异 —— 这是刻意的，用来说明
「词干规则本身就是一族近似算法，没有唯一正确答案」。
"""
    ),
    code(STEM),
    code(STEM_DEMO),
    md(
        """
## 易错点清单

1. **文本列不加 `keep_default_na=False`** —— `""` 和 `"N/A"` 会被静默吃掉，
   后面所有「空文本过滤」「长度统计」的数字全是错的。
2. **`Series.astype(str)` 不会把 NaN 变成 `"nan"`** —— pandas 3.x 的 `str` dtype
   会保留 NA，`map(clean)` 直接抛 `TypeError: expected string or bytes-like object`。
   正确做法是从读取源头就关掉 NA 解析。
3. **`re.match` vs `re.fullmatch`** —— 前者只匹配开头，`"a汉字"` 会被误判成纯汉字段。
4. **`fullmatch` 配了单字符 pattern** —— `re.compile(r"[\u4e00-\u9fff]")` 只能匹配**一个**汉字，
   而 `fullmatch` 要求整串匹配，于是 `"绝缘子渗油"` 判定失败、整段被当成一个 token。
   pattern 一定要写成 `[\u4e00-\u9fff]+`（**这个错不报错，只是分词结果全错**，本章实际踩过）。
5. **贪婪 `.*` 吃掉正文** —— 成对标记一律用非贪婪 `.*?`。
6. **忘记对英文/数字预切分** —— `STATION_E_02` 被逐字符切碎，词表被单字母污染。
7. **词表用 `list` 而不是 `set`** —— `x in list` 是 O(n)，600 条 × 每句 20 次查询
   就是几万次线性扫描。`in set` 是 O(1)。
8. **`i += length` 写成 `i += 1`** —— 分词器会退化成逐字切分，
   且**不报错**，只是结果全错。
9. **停用词表漏掉标点** —— 词表里会出现 `，`、`【`、`】` 这类无意义 token。
10. **把词干当词形还原用** —— `running -> runn` 给模型用没问题，展示给人看就成了笑话。

## 本章小结

| 步骤 | 关键判断 | 本章数字 |
|---|---|---|
| 读取 | 文本列关掉 NA 解析 | 缺失 12 → 0 |
| 体检 | 先量化再清洗 | HTML 107 / URL 82 / 半角逗号 131 / 重复标点 41 |
| 清洗 | 五步管道 + 零宽断言 | 平均字符 34.986667 → 28.513333 |
| 分词 | FMM / BMM + 英文预切分 | 词表 59，最长 7 字 |
| 过滤 | 停用词去标点 | token 3461 → 词表 75 |
| 特征 | n-gram | 首条 1/2/3-gram = 5 / 4 / 3 |

**下一章**：把这 78 维词表喂给词袋和 TF-IDF，并手写 BPE 看子词怎么切。
"""
    ),
]

# =========================================================================== #
# 练习 / 答案版                                                               #
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch01 文本预处理（练习版）

> 按提示补全 `____`，跑通所有 assert。真值全部来自实跑（pandas 3.x，六条主线）。
>
> `VOCAB` / `STOPWORDS` / `SEG_RE` / `HAN_RE` / `MAX_WORD_LEN` 已给好；
> 挖空块内每条语句保持**单行**，控制流（`while` / `for` / `if`）都在块外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    md(
        """
## 任务 1：读取口径（`keep_default_na`）

同一份 CSV 读两遍，看缺失值个数差多少，并想清楚**哪个数字才是你想要的**。
"""
    ),
    code(NA_TRAP),
    md(
        """
## 任务 2：脏数据体检

先数清楚，再动手洗。这两行是后面所有清洗的验收基准。
"""
    ),
    code(INSPECT),
    md(
        """
## 任务 3：正则清洗五步管道

按注释分三段补全：去 HTML/URL、标点归一、去汉字间空格。
第 3 段的零宽断言写法是关键 —— 它必须**不碰** `STATION_E_02` 里的下划线。
"""
    ),
    code(CLEAN_FN),
    code(CLEAN_APPLY),
    md(
        """
## 任务 4：贪婪 vs 非贪婪

两个正则都能「去标签」，但一个会把正文也吃掉。
"""
    ),
    code(GREEDY),
    md(
        """
## 任务 5：正向最大匹配

内层 `while` 已经在帮你缩短候选长度了，你只需要**收词 + 移指针**。
注意指针要右移「匹配长度」，不是 1。
"""
    ),
    code(VOCAB_SRC),
    code(SCAFFOLD),
    code(FMM),
    code(FMM_DEMO),
    md(
        """
## 任务 6：逆向最大匹配 + 方向对比

从右往左扫，注意收词是 `insert(0, ...)`（保持从左到右的输出顺序）。

最后一个 assert 会告诉你一个反直觉的事实：**真实工单里两个方向不分歧**。
"""
    ),
    code(BMM),
    code(BMM_DEMO),
    md(
        """
## 任务 7：中文夹英文编号的两级切分

判断「是不是纯汉字段」要用 `fullmatch`，不是 `match`。
"""
    ),
    code(TOKENIZE),
    code(TOKENIZE_DEMO),
    md("## 任务 8：停用词过滤（别忘了标点）"),
    code(STOP),
    code(STOP_ALL),
    md("## 任务 9：n-gram"),
    code(NGRAM),
    code(NGRAM_DEMO),
    md(
        """
## 任务 10：简化版词干提取

命中后缀就切掉，但**要留够 3 个字符** —— 否则 `"is"` 这种短词会被砍成空串。
"""
    ),
    code(STEM),
    code(STEM_DEMO),
    md(
        """
## 自查清单

- [ ] 文本列读取关掉了默认 NA 解析
- [ ] 清洗后 `<` / `http` / 半角逗号 / `！！` 四种残留全部为 0
- [ ] 平均字符数从 34.986667 降到 28.513333
- [ ] 正向最大匹配：指针右移的是**匹配长度**
- [ ] 逆向最大匹配：收词用 `insert(0, ...)`
- [ ] 粗切分用 `fullmatch` 判断纯汉字段
- [ ] 停用词表里**包含中文标点**
- [ ] n-gram 的起点范围是 `len(seq) - n + 1`
- [ ] 词干只剩 3 个字符时不再继续砍
- [ ] 能说出「研究生命起源」为什么两个方向结果不同
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
