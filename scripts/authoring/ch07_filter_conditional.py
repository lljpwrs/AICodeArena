"""ch07 —— 筛选、索引与条件逻辑：where 与 mask 的 NaN 差异

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch07_filter_conditional.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch07_filter_conditional"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("data")
dev = pd.read_csv(DATA / "device_defects.csv")

SENTINEL_CUT = 1000      # 负荷值 >= 该值视为哨兵/尖峰
REPLACE_WITH = 888       # 演示 where / mask 的替换值
RISK_CHOICES = ["高", "中"]

print("pandas", pd.__version__)
print("dev", dev.shape)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch07 筛选、索引与条件逻辑：两个"看起来一样"的 API 其实不一样

> 方向：数据预处理 ｜ 竞赛对应：**数据清洗（每项 2 分）+ 特征构造**
> —— 筛选是清洗和特征工程之间的那道闸门：条件写错，后面全错，而且**不报错**。

本章只解决一件事：**让你写出的每个条件都精确等于你脑子里想的那个条件。**
听起来不难，但本章会用实测数据证明：

- `where` 和 `mask` 在 `NaN` 上的行为**不同**，差 21 行
- `query` 有四种失效场景，其中一种必须换 `engine`
- `filter` 筛的是**列**，不筛行——这是最高频的概念错用
- `query` 的**硬性约束只有两条**：引用外部变量必须 `@`，特殊列名必须反引号
  （另外会有个"老教程说法在 3.0 上已失效"的实测澄清）
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据清洗 | 每项 2 分 | 按条件定位待处理记录，不改动范围外的数据 |
| 数据探索 | 每项 2 分 | 分组计数（`isin` / `between` / 多条件组合） |
| 特征构造 | 影响建模分 | `np.select` 分档造标签列 |
| 结果交付 | 交卷格式分 | 条件写错不报错，只在结果里错——必须用 `assert` 自证 |

**为什么单独成章**：条件逻辑的错误**不会抛异常**。
`dev[dev["等级"] == "严重" | dev["等级"] == "危急"]` 这种写法在 Python 里
会因为优先级问题静默变成别的东西（或直接报 `TypeError`），
你只有在最后发现"样本数对不上"时才知道错了——那时候已经很难回溯。
"""
    ),
    md(
        """
## 二、API 速查表

| API | 作用对象 | 返回 | 一句话说明 |
|---|---|---|---|
| `df["列"] == 值` | 行 | `Series[bool]` | 最基础的掩码，注意 `NaN` 参与比较恒为 `False` |
| `df[mask]` | 行 | `DataFrame` | 按布尔掩码筛行（也可以用列名列表筛列） |
| `df.loc[行条件, 列名]` | 行列 | `DataFrame`/`Series` | **条件筛行 + 指定列**，标签索引 |
| `df.iloc[行号, 列号]` | 行列 | `DataFrame`/`Series` | 位置索引 |
| `df.query("表达式")` | 行 | `DataFrame` | 字符串表达式，最贴近"人怎么念" |
| `df.filter(like=/regex=/items=)` | **列** | `DataFrame` | 按**列名**筛选列，不筛行 |
| `s.isin([...])` | 元素 | `Series[bool]` | 属于集合，等价于一串 `or` |
| `s.between(a, b)` | 元素 | `Series[bool]` | 闭区间，等价于 `(s>=a)&(s<=b)` |
| `s.where(cond, other)` | 元素 | `Series` | **cond 为假**时换成 `other` |
| `s.mask(cond, other)` | 元素 | `Series` | **cond 为真**时换成 `other` |
| `np.where(cond, a, b)` | 元素 | `ndarray` | 二元分支，比 `apply` 快得多 |
| `np.select(conds, choices, default)` | 元素 | `ndarray` | 多分支分档，**替代多层 if-else** |
| `df.loc[条件, "列"] = 值` | — | `None` | 条件赋值**唯一正确**写法 |

### `where` / `mask` 一句话记忆

| | 命中条件时 | 不命中时 |
|---|---|---|
| `where(cond, other)` | 保留原值 | 换成 `other` |
| `mask(cond, other)` | 换成 `other` | 保留原值 |

**但两者对"原本就是 `NaN`"的位置处理不同**，见 §3.5 —— 这是本章最值钱的一节。
"""
    ),
    code(HEADER),
    md(
        """
---

## 三、逐节讲解

### 3.1 三种筛行方式的选型规则

`df[mask]` / `df.loc[mask, cols]` / `df.query("...")` 三者结果等价，
**选哪个是"可读性"问题，不是"正确性"问题**。但有一个硬边界：

| 需求 | 用什么 |
|---|---|
| 只筛行、要全部列 | `df[mask]` |
| 筛行 + 只取几列 | `df.loc[mask, [列]]` ← **推荐**，一步到位 |
| 条件多、要读起来像自然语言 | `df.query("...")` |
| 按行号 / 列号取 | `df.iloc[...]` |
| 按列名筛列 | `df.filter(like=...)` |

**选型对效率的影响**：`df[mask]` 之后再 `[[列]]` 是**两次**索引（多一次拷贝），
`df.loc[mask, [列]]` 是**一次**。数据量大时差距明显；竞赛小数据感知不到，
但**写上习惯**能省掉后面所有"为什么这么慢"的排查。
"""
    ),
    code(
        """mask_crit = dev["缺陷等级"] == "危急"
print("① 基础掩码 dev['缺陷等级'] == '危急'：")
print("   命中行数:", int(mask_crit.sum()), "/", len(dev))
print("   掩码 dtype:", mask_crit.dtype)

print("\\n② 三种写法结果是否完全一致：")
a = dev[mask_crit & (dev["设备类型"] == "变压器")]
b = dev.loc[mask_crit & (dev["设备类型"] == "变压器")]
c = dev.query("缺陷等级 == '危急' and 设备类型 == '变压器'")
print("   df[mask]:", a.shape, "| loc[mask]:", b.shape, "| query:", c.shape)
print("   三者 equals:", a.equals(b), b.equals(c))

print("\\n③ 筛行 + 取列，一步 vs 两步：")
one_step = dev.loc[mask_crit, ["记录ID", "缺陷等级", "负荷值"]]
two_step = dev[mask_crit][["记录ID", "缺陷等级", "负荷值"]]
print("   一步 .loc[mask, cols]:", one_step.shape)
print("   两步 [mask][cols]    :", two_step.shape)
print("   结果相同:", one_step.equals(two_step), "（但两步多一次中间拷贝）")

print("\\n④ iloc 按位置取：")
print("   前 3 行前 3 列:")
print(dev.iloc[:3, :3].to_string(index=False))"""
    ),
    md(
        """
> **`NaN` 参与比较永远为 `False`**——这一点必须在心里默念：
> ```python
> s = pd.Series([1.0, np.nan])
> (s > 0).tolist()      # [True, False]   ← NaN 位置是 False，不是 NaN
> (~(s > 0)).tolist()   # [False, True]   ← 取反后 NaN 位置变 True
> ```
> 所以 `~mask` **不是**"选出满足条件的反例"，而是"选出所有不满足条件的，包括缺失"。
> `between` / `isin` 同理，它们返回的是纯 `bool`，不会传播 `NaN`。

---

### 3.2 布尔索引五件套：`&` `|` `~` `isin` `between`

五个运算符的组合就是**筛选的全部表达力**。写法上只有两条纪律，
但违反任何一条都会出问题。
"""
    ),
    code(
        """crit = dev["缺陷等级"] == "危急"
sev = dev["缺陷等级"] == "严重"

print("① & 且：")
both = crit & (dev["设备类型"] == "变压器")
print("   危急 且 变压器:", int(both.sum()))

print("\\n② | 或（注意每个条件都要括号）：")
print("   危急 或 严重 -> isin 写法:", int(dev["缺陷等级"].isin(["严重", "危急"]).sum()))
print("   ~(一般) 反选写法  :", int((~(dev["缺陷等级"] == "一般")).sum()))

print("\\n③ ~ 取反会连缺失一起选进来：")
probe = pd.Series([1.0, np.nan, 3.0])
print("   (probe > 0)      :", (probe > 0).tolist())
print("   ~(probe > 0)     :", (~(probe > 0)).tolist(), " ← NaN 位置被判为 True")
print("   只想要非缺失的反例:", ((probe <= 0) & probe.notna()).tolist())

print("\\n④ between 是闭区间：")
bet = dev["负荷值"].between(100, 300)
print("   between(100, 300) 命中:", int(bet.sum()))
print("   等价写法 (>=100) & (<=300):", int(((dev["负荷值"] >= 100) & (dev["负荷值"] <= 300)).sum()))
print("   与 query 链式比较一致:",
      bool(dev.query("100 < 负荷值 < 300").index.equals(dev.loc[bet].index)))

print("\\n⑤ contains / startswith（务必给 na=False）：")
print("   处理状态含'处理':", int(dev["处理状态"].str.contains("处理", na=False).sum()))
print("   不给 na 会得到:", dev["处理状态"].str.contains("处理").dtype,
      "（对象型，含 NaN 时布尔索引直接报错）")

print("\\n⑥ 括号陷阱三种写法的真实表现（这是本节最值钱的一张表）：")
print("   写法 A：两个条件都加括号（正确）")
good = (dev["缺陷等级"] == "危急") & (dev["设备类型"] == "变压器")
print("      (dev['缺陷等级']=='危急') & (dev['设备类型']=='变压器')")
print("      执行成功, 命中 =", int(good.sum()), " ← 正确结果")

print("\\n   写法 B：只给第一个加括号（静默错误！）")
half = (dev["缺陷等级"] == "危急") & dev["设备类型"] == "变压器"
print("      (dev['缺陷等级']=='危急') & dev['设备类型'] == '变压器'")
print("      执行成功, 命中 =", int(half.sum()), " ← 不报错，但结果全 False")
print("      原因：& 优先级高于 ==，实际算的是 ((A) & 设备类型) == '变压器'")
print("      中间量 (A) & 设备类型 命中", int(((dev["缺陷等级"] == "危急") & dev["设备类型"]).sum()),
      "个，再与字符串 '变压器' 比较 -> 全部 False")

print("\\n   写法 C：完全不写括号（会报错，反而安全）")
try:
    dev["缺陷等级"] == "危急" & dev["设备类型"] == "变压器"
    print("      执行成功")
except TypeError as exc:
    print("      ->", type(exc).__name__, ":", str(exc)[:60])

print("\\n   写法 D：用 and（会报错，也安全）")
try:
    good and dev["设备类型"].eq("变压器")
    print("      执行成功")
except ValueError as exc:
    print("      ->", type(exc).__name__, ":", str(exc)[:60])"""
    ),
    md(
        """
> **两条纪律**（第 ⑥ 步给了实测证据）：
>
> | 写法 | 结果 | 危险等级 |
> |---|---|---|
> | `(A) & (B)` | 命中 5 ✅ | 正确 |
> | `(A) & B == 值` | 命中 **0**，**不报错** | 🔴 **最危险**——静默给出空结果 |
> | `A == 值 & B == 值`（完全不写括号） | `TypeError` | 🟢 安全，至少报错 |
> | `A and B` | `ValueError: truth value is ambiguous` | 🟢 安全，至少报错 |
>
> 结论：**报错不可怕，"命中 0 条"才可怕**——你会以为数据里真没有。
> 所以纪律是：**每个条件单独加括号**，写完之后立刻 `mask.sum()` 数一遍。
>
> 原理：`&` 的优先级**高于** `==`。`(A) & B == "变压器"` 被解析为
> `((A) & B) == "变压器"`——先做 `布尔 & 字符串`，再拿结果跟字符串比，
> 于是整列变成 `False`。

---

### 3.3 难点深挖：`query` 的两条硬约束

**为什么难**：`query` 用字符串写条件，可读性最好，但**字符串里的东西不走 Python 的作用域**，
所以有一整套"看起来该工作但不工作"的写法。而且网上教程大量基于 pandas 1.x/2.x，
有几条流传很广的"必须这么写"在 3.0 上已经不成立——**照抄会写出多余参数，不抄又怕踩坑**，
所以本节全部结论都现场实测一遍。
"""
    )
]
LESSON += [
    code(
        '''print("① 普通用法：像念句子一样写条件")
print("   负荷值 > 500 且 缺陷等级 == '危急':",
      dev.query("负荷值 > 500 and 缺陷等级 == '危急'").shape)

print("\\n② in 列表：")
print("   设备类型 in ['变压器', '绝缘子']:", dev.query("设备类型 in ['变压器', '绝缘子']").shape)

print("\\n③ 引用外部变量必须加 @，不加就报 UndefinedVariableError：")
thr = 500
print("   query('负荷值 > @thr'):", dev.query("负荷值 > @thr").shape)
try:
    dev.query("负荷值 > thr")
    print("   不加 @ 居然成功")
except Exception as exc:
    print("   不加 @ ->", type(exc).__name__, ":", str(exc)[:70])

print("\\n④ 列名含空格 / 是保留字时用反引号：")
odd = dev.rename(columns={"负荷值": "负荷 值"})
print("   反引号包裹:", odd.query("`负荷 值` > 500").shape)
odd2 = dev.rename(columns={"负荷值": "class"})
print("   保留字 class:", odd2.query("`class` > 500").shape)

print("\\n⑤ 涉及 .str 方法：本机实测默认引擎已支持（与老教程的说法不同）")
q_default = dev.query("台区编号.str.startswith('STATION_A')")
q_python = dev.query("台区编号.str.startswith('STATION_A')", engine="python")
print("   默认引擎      :", q_default.shape)
print("   engine=python :", q_python.shape)
print("   两者结果一致  :", q_default.index.equals(q_python.index))
print("   -> 基于 pandas 1.x/2.x 的教程写『.str 必须换 engine=python』，")
print("      在 3.0 上已不成立；竞赛机器版本不确定，写上也不亏（只是慢一点）。")

print("\\n⑥ 默认引擎实测还能处理这些写法：")
print("   缺陷类型.str.len() > 2 ->", dev.query("缺陷类型.str.len() > 2").shape)
print("   负荷值.notna()        ->", dev.query("负荷值.notna()").shape)
print("   abs(负荷值) > 500     ->", dev.query("abs(负荷值) > 500").shape)
print("   index < 100           ->", dev.query("index < 100").shape)'''
    ),
    md(
        """
> **判定规则**：
>
> | 情况 | 要不要特殊处理 |
> |---|---|
> | 纯数值 / 类别比较（`>` `==` `in`） | 直接用默认引擎 |
> | 涉及 `.str` / `.notna()` 等方法调用 | 本机 3.0 实测**可直接用**；想稳妥写 `engine="python"` |
> | 引用局部变量 | **必须加 `@`**，否则 `UndefinedVariableError` |
> | 列名含空格 / 是保留字（`class` `if`） | **必须反引号包裹** |
> | 条件里引用 3 个以上外部变量 | 干脆别用 `query`，改回掩码更好 review |
>
> 一句话：`query` 的**硬性约束只有 `@` 和反引号两条**，其余是可读性取舍。
> 遇到报错先看错误类型——`UndefinedVariableError` 就是缺 `@`。
> 不要因为"教程说要换引擎"就无脑加参数，也不要因为本机能用就以为到处能用。

---

### 3.4 `filter`：筛的是列，不是行

这是最高频的概念错用。`filter` 的 `like` / `regex` / `items` 全部作用在**列名**上。
名字起得像筛行，所以很多人第一次用它筛数据时会得到"整列没变"的结果，
而且**不报错**。
"""
    ),
    code(
        """print("① 按列名模糊筛列：")
print("   filter(like='设备')   ->", list(dev.filter(like="设备").columns))
print("   filter(regex='^缺陷') ->", list(dev.filter(regex="^缺陷").columns))
print("   filter(items=[...])   ->", list(dev.filter(items=["记录ID", "负荷值"]).columns))

print("\\n② 它们的对象是列，行数一行不少：")
print("   原始行数:", len(dev), "| filter 后行数:", len(dev.filter(like="设备")))

print("\\n③ 想筛行还得用掩码：")
print("   dev[dev['设备类型'] == '变压器'] ->",
      dev[dev["设备类型"] == "变压器"].shape, " ← 行数才变了")

print("\\n④ filter 的正确使用场景：一次性把「数值列」捞出来算统计量")
num_block = dev.filter(items=dev.select_dtypes("number").columns.tolist())
print("   取到的列:", list(num_block.columns))
print(num_block.describe().round(2).to_string())"""
    ),
    md(
        """
> **记法**：`filter` 里的 `like`/`regex` 修饰的是**列名这个字符串**。
> 想筛行 → `[]` / `loc` / `query`；想筛列 → `filter` / `select_dtypes` / 列名列表。

---

### 3.5 难点深挖：`where` 与 `mask` 对 `NaN` 的不同处理

**为什么难**：几乎所有教程都只讲"`where` 保留满足条件的、`mask` 替换满足条件的"，
然后给一个**没有缺失值的例子**。一旦列里有 `NaN`，两者就分道扬镳了，
而且**都不报错**。
"""
    ),
    code(
        """load = dev["负荷值"]
print("原始列缺失数:", int(load.isna().sum()), "（这 21 个是真正的数据缺失）")

print("\\n① 不给 other 时，两者结果一样：")
w_none = load.where(load < SENTINEL_CUT)
m_none = load.mask(load >= SENTINEL_CUT)
print("   where(load < 1000) 后 NaN 数:", int(w_none.isna().sum()))
print("   mask(load >= 1000) 后 NaN 数:", int(m_none.isna().sum()))
print("   -> 都是 28（21 原有缺失 + 7 个 >=1000 的哨兵）")

print("\\n② 给了 other 之后，差异出现了：")
w_other = load.where(load < SENTINEL_CUT, REPLACE_WITH)
m_other = load.mask(load >= SENTINEL_CUT, REPLACE_WITH)
n_w = int(w_other.eq(REPLACE_WITH).sum())
n_m = int(m_other.eq(REPLACE_WITH).sum())
print(f"   where(..., 888) 命中 888 的个数: {n_w:3d}   ← 连原有 21 个 NaN 一起换了")
print(f"   mask(..., 888)  命中 888 的个数: {n_m:3d}   ← 只换了 7 个哨兵，NaN 保留")
print(f"   差值 = {n_w - n_m} = 原有缺失数 {int(load.isna().sum())}")

print("\\n③ 为什么：")
print("   where 的判定是『cond 为假 -> 换 other』，而 NaN < 1000 的结果是 False")
print("   -> NaN 位置被判为『不满足条件』，于是被 other 覆盖")
print("   mask 的判定是『cond 为真 -> 换 other』，而 NaN >= 1000 也是 False")
print("   -> NaN 位置不满足替换条件，原样保留")

print("\\n④ 想用 where 但保住 NaN：把 notna() 一起写进条件")
w_safe = load.where((load < SENTINEL_CUT) | load.isna(), REPLACE_WITH)
print("   命中 888 的个数:", int(w_safe.eq(REPLACE_WITH).sum()), " ← 与 mask 一致了")
print("   缺失仍为:", int(w_safe.isna().sum()))"""
    ),
    md(
        """
> **判定规则**（背下来）：
>
> | 我要做的事 | 用哪个 |
> |---|---|
> | 把"满足条件的那批"统一改成某值 | **`mask`** —— 不会碰到缺失值，最安全 |
> | 把"不满足条件的那批"统一改成某值，且**明确想连缺失一起改** | `where` |
> | 把"不满足条件的"改掉，但**想保住缺失** | `where((cond) \\| s.isna(), v)` |
>
> **默认用 `mask`**。因为"我要处理的那批记录"通常是有明确定义的（比如 `load >= 1000`），
> 而"其余全部"里混着缺失值，把它们一起改掉几乎总是错的。

---

### 3.6 多条件分档：`np.select` 与 `np.where`

竞赛里几乎必考的一项：**用连续值造一个有序类别列**（风险等级、负荷区间、告警级别）。
标准做法是 `np.select`，不是 `apply` + `if-else`。
"""
    ),
    code(
        """print("① 二元分支用 np.where（最快）：")
high_load = np.where(dev["负荷值"] > 200, "高负荷", "低负荷")
print("   ", pd.Series(high_load).value_counts().to_dict())

print("\\n② 多分支用 np.select —— 顺序即优先级，先命中先返回：")
conds = [dev["缺陷等级"].eq("危急"), dev["缺陷等级"].eq("严重")]
risk = np.select(conds, RISK_CHOICES, default="低")
print("   ", pd.Series(risk).value_counts().to_dict())
print("   合计:", len(risk), "= 行数", len(dev), "| 无遗漏:",
      (pd.Series(risk) != "").all())

print("\\n③ 为什么不用 apply：")
import time
big = pd.concat([dev] * 400, ignore_index=True)
t0 = time.perf_counter()
_r1 = np.select([big["缺陷等级"].eq("危急"), big["缺陷等级"].eq("严重")], RISK_CHOICES, default="低")
t1 = time.perf_counter()
_r2 = big["缺陷等级"].apply(lambda v: "高" if v == "危急" else ("中" if v == "严重" else "低"))
t2 = time.perf_counter()
print(f"   行数 {len(big)}")
print(f"   np.select : {t1 - t0:.4f}s")
print(f"   apply     : {t2 - t1:.4f}s   （慢 {(t2 - t1) / (t1 - t0):.0f} 倍）")

print("\\n④ 写回 DataFrame 变成特征列：")
demo = dev.assign(风险等级=risk)
print(demo["风险等级"].value_counts().to_string())
print("   交叉验证（等级 x 风险）:")
print(pd.crosstab(demo["缺陷等级"], demo["风险等级"]).to_string())"""
    ),
    md(
        """
> **优先级顺序陷阱**：`np.select` 按列表顺序**首个命中即返回**。
> 如果先写 `负荷值 > 100`，再写 `负荷值 > 500`，那 `> 500` 永远不会生效。
> **永远把最严格的条件写在最前面。**

---

### 3.7 条件赋值：只有 `loc` 一种正确写法

第 3.1 节讲过 `df[mask]["col"] = v` 在 pandas 3.0 下静默失效。
这里给出**唯一正确写法**和一个自查技巧。
"""
    ),
    code(
        """d = dev.copy()

# 正确：loc 一次性完成定位与赋值
d.loc[d["负荷值"] > 500, "复核标记"] = "高负荷待复核"
print("① loc 条件赋值命中:", int((d["复核标记"] == "高负荷待复核").sum()))

# 其余行补默认值（先整列赋默认，再覆盖，顺序不能反）
d["复核标记"] = d["复核标记"].fillna("常规")
print("② 补默认值后分布:", d["复核标记"].value_counts().to_dict())

# 错误示范：链式赋值（静默失效）
d2 = dev.copy()
d2[d2["负荷值"] > 500]["备注"] = "xxx"
print("\\n③ 链式赋值后 '备注' 有 'xxx' 的行数:",
      int((d2["备注"] == "xxx").sum()), " ← 0，一行都没改上")

# 自查技巧：赋值后立刻数一遍命中数
before = int((d["负荷值"] > 500).sum())
after = int((d["复核标记"] == "高负荷待复核").sum())
print("\\n④ 自查：条件命中数", before, "== 赋值命中数", after, "->", before == after)"""
    ),
    md(
        """
> **自查纪律**：条件赋值之后，**立刻断言"条件命中数 == 赋值命中数"**。
> 这是成本最低、收益最高的一条自检——它能抓住所有静默失效。

---

### 3.8 组合筛选实战：把条件堆成一个清单

真实任务的筛选条件从来不是一条。下面用 3 个条件组合出「高优先检修清单」，
并顺便演示**怎么验证自己没筛错**。
"""
    ),
    code(
        """urgent = dev.loc[
    dev["缺陷等级"].isin(["严重", "危急"])
    & dev["设备类型"].isin(["变压器", "断路器"])
    & dev["处理状态"].str.contains("处理", na=False),
    ["记录ID", "台区编号", "设备类型", "缺陷等级", "负荷值", "处理状态"],
]

print("① 高优先检修清单:", urgent.shape)
print(urgent.head(5).to_string(index=False))

print("\\n② 逐条件验证（每一步都要对数）：")
c1 = dev["缺陷等级"].isin(["严重", "危急"])
c2 = dev["设备类型"].isin(["变压器", "断路器"])
c3 = dev["处理状态"].str.contains("处理", na=False)
print("   条件1 等级 ∈ {严重,危急}:", int(c1.sum()))
print("   条件2 类型 ∈ {变压器,断路器}:", int(c2.sum()))
print("   条件3 处理状态含'处理':", int(c3.sum()))
print("   三条件交集:", int((c1 & c2 & c3).sum()), " ← 必须等于清单行数", len(urgent))

print("\\n③ 清单内部结构：")
print("   按等级:", urgent["缺陷等级"].value_counts().to_dict())
print("   按类型:", urgent["设备类型"].value_counts().to_dict())
print("\\n④ 清单里负荷值最高的 3 条：")
print(urgent.nlargest(3, "负荷值")[["记录ID", "设备类型", "缺陷等级", "负荷值"]].to_string(index=False))"""
    ),
    md(
        """
---

## 四、易错点清单

1. **只给一半条件加括号** → `(A) & B == 值` 静默返回全 `False`（因为 `&` 优先级高于 `==`）；
   完全不写括号则抛 `TypeError`。**报错不可怕，"命中 0 条"才可怕**
2. 用 `and` / `or` 连接 Series 条件 → `ValueError: truth value is ambiguous`
3. `~mask` 会把 `NaN` 一起选进来（`NaN` 参与比较恒为 `False`，取反即 `True`）
4. `str.contains` 不给 `na=False` → 布尔索引时抛 `ValueError`
5. **`filter` 用来筛行** → 行数一行不少，列却被砍了，且不报错
6. `query` 引用局部变量忘记加 `@` → `UndefinedVariableError`
7. 列名含空格、或是 `class` / `if` 这类保留字时**不加反引号** → 查询直接失败
8. `np.select` 条件顺序写反 → 宽条件在前，窄条件永远命中不了
9. **`where(cond, v)` 会把原有 `NaN` 一起替换**（实测差 21 行）
10. `df[mask]["列"] = 值` 链式赋值静默失效，一行都改不上
11. 条件赋值不写默认值 → 未命中行是 `NaN`，而不是"默认类别"
12. 筛完不数一遍 → 竞赛按步骤给分，**写了验证才算**

---

## 五、本章小结

| 需求 | 正确写法 |
|---|---|
| 筛行（全部列） | `df[mask]` |
| 筛行 + 取列（推荐） | `df.loc[mask, [列名]]` |
| 条件多、可读性优先 | `df.query("...")`（有方法调用就加 `engine="python"`） |
| 按列名筛列 | `df.filter(like=/regex=)` 或 `df.select_dtypes()` |
| 属于集合 | `s.isin([...])` |
| 区间 | `s.between(a, b)`（闭区间） |
| 把命中的行改掉 | `s.mask(cond, v)` ← **默认选它** |
| 把未命中的行改掉（保住缺失） | `s.where((cond) \\| s.isna(), v)` |
| 多档分类 | `np.select(conds, choices, default=)`，**窄条件在前** |
| 条件赋值 | `df.loc[cond, "列"] = v`，**赋完立刻对数** |
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch07 练习 —— 筛选、索引与条件逻辑

数据：`data/device_defects.csv`（270 行 x 13 列）

**每题都有 `assert` 验收块**。跑通即正确，跑不通的报错信息本身就说清了哪里错。
**不要翻答案版。**

| 题号 | 考什么 |
|---|---|
| TODO(1) | 单条件掩码 |
| TODO(2) | 多条件 `&`（括号不能省） |
| TODO(3) | `isin` |
| TODO(4) | `between` |
| TODO(5) | `~` 取反与 `NaN` 的关系 |
| TODO(6) | `query` 多条件 |
| TODO(7) | `query` 引用局部变量（`@`） |
| TODO(8) | `query` + `.str` 方法调用（并澄清"必须换引擎"的旧说法） |
| TODO(9) | `filter` 筛列（不是筛行） |
| TODO(10) | **`where` 与 `mask` 的 `NaN` 差异** |
| TODO(11) | `loc` 条件赋值 |
| TODO(12) | `np.select` 多档分类 |
| 综合 ① | 三条件组合清单 + 图表 |
| 综合 ② | 条件筛选自查报告 |
"""
    ),
    code(HEADER),
    md(
        """
---

## 基础题
"""
    ),
    code(
        """# @@todo(1) 构造「缺陷等级 == 危急」的布尔掩码，存入 crit_mask
# @@hint Series 与字符串比较，得到 Series[bool]
crit_mask = dev["缺陷等级"] == "危急"
# @@end
# ---- 验收 ----
assert crit_mask.dtype == bool, f"掩码 dtype 应为 bool，实际 {crit_mask.dtype}"
assert int(crit_mask.sum()) == 31, f"危急应有 31 条，实际 {int(crit_mask.sum())}"
assert len(crit_mask) == 270, "掩码长度必须等于行数"
print("TODO(1) 通过 -> 危急", int(crit_mask.sum()), "条")"""
    ),
    code(
        """# @@todo(2) 多条件：危急 且 变压器，存入 crit_trans_mask（两个条件都要括号）
# @@hint (dev["缺陷等级"] == "危急") & (dev["设备类型"] == "变压器")
crit_trans_mask = (dev["缺陷等级"] == "危急") & (dev["设备类型"] == "变压器")
# @@end
# ---- 验收 ----
assert int(crit_trans_mask.sum()) == 5, f"危急且变压器应有 5 条，实际 {int(crit_trans_mask.sum())}"
# 证明括号不能省：只给一半加括号 -> 静默得到 0 命中（不报错）
half = (dev["缺陷等级"] == "危急") & dev["设备类型"] == "变压器"
assert int(half.sum()) == 0, f"半括号写法应静默得到 0 命中，实际 {int(half.sum())}"
assert int(crit_trans_mask.sum()) == 5, "正确写法仍应是 5"
print("   半括号写法命中 0 条且不报错 -> 这就是为什么要给每个条件都加括号")
print("TODO(2) 通过 -> 危急且变压器", int(crit_trans_mask.sum()), "条")"""
    ),
    code(
        """# @@todo(3) 用 isin 选出「严重」或「危急」的行，掩码存入 sev_crit_mask
# @@hint Series.isin([...])
sev_crit_mask = dev["缺陷等级"].isin(["严重", "危急"])
# @@end
# ---- 验收 ----
assert int(sev_crit_mask.sum()) == 106, f"应为 106 条，实际 {int(sev_crit_mask.sum())}"
assert sev_crit_mask.dtype == bool
assert sev_crit_mask.equals(dev["缺陷等级"].isin({"严重", "危急"})), "list 与 set 结果应一致"
print("TODO(3) 通过 -> 严重或危急", int(sev_crit_mask.sum()), "条")"""
    ),
    code(
        """# @@todo(4) 用 between 选出负荷值在 [100, 300] 的行，掩码存入 load_mid_mask
# @@hint Series.between(下限, 上限)   —— 默认闭区间
load_mid_mask = dev["负荷值"].between(100, 300)
# @@end
# ---- 验收 ----
assert int(load_mid_mask.sum()) == 56, f"应为 56 条，实际 {int(load_mid_mask.sum())}"
manual = (dev["负荷值"] >= 100) & (dev["负荷值"] <= 300)
assert bool((load_mid_mask == manual).all()), "between 应等价于 (>=下限) & (<=上限)"
assert load_mid_mask.dtype == bool, "between 返回纯 bool，不传播 NaN"
print("TODO(4) 通过 -> 负荷值在 [100,300] 的有", int(load_mid_mask.sum()), "条")"""
    ),
    code(
        """# @@todo(5) 用取反找出「非一般」的行；同时统计取反后有多少个「原本是缺失」的位置被带入
# @@hint ~mask ；再和 dev["缺陷等级"].notna() 求交，看取反版比"正常反选"多算了几个
not_normal_mask = ~(dev["缺陷等级"] == "一般")
# @@end
# ---- 验收 ----
assert int(not_normal_mask.sum()) == 106, f"非一般应为 106（270 - 一般 164），实际 {int(not_normal_mask.sum())}"
strict = (dev["缺陷等级"] != "一般") & dev["缺陷等级"].notna()
assert int(strict.sum()) == int(not_normal_mask.sum()), "本章无缺失时两者应完全相等"
assert dev["缺陷等级"].value_counts().to_dict() == {"一般": 164, "严重": 75, "危急": 31}, \
    "先核对三类基准数"
missing_lvl = int(dev["缺陷等级"].isna().sum())
assert missing_lvl == 0, "本章数据里缺陷等级没有缺失，取反与严格版必然相等"
assert bool((not_normal_mask == strict).all())
print("TODO(5) 通过 -> 非一般", int(not_normal_mask.sum()), "条（本章无缺失，取反版与严格版一致）")"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(6) 用 query 选出「负荷值 > 500 且 缺陷等级 == '危急'」的行，存入 q_crit_high
# @@hint DataFrame.query("条件 and 条件")
q_crit_high = dev.query("负荷值 > 500 and 缺陷等级 == '危急'")
# @@end
# ---- 验收 ----
assert q_crit_high.shape[0] == 5, f"应为 5 行，实际 {q_crit_high.shape[0]}"
assert q_crit_high.shape[1] == 13, "query 不改变列数"
manual = dev[(dev["负荷值"] > 500) & (dev["缺陷等级"] == "危急")]
assert q_crit_high.index.equals(manual.index), "query 与掩码结果应完全一致"
print("TODO(6) 通过 -> query 命中", q_crit_high.shape[0], "行")"""
    ),
    code(
        """THR = 500

# @@todo(7) 用 query 引用上面的局部变量 THR（必须加 @），存入 q_var
# @@hint query("负荷值 > @THR")
q_var = dev.query("负荷值 > @THR")
# @@end
# ---- 验收 ----
assert q_var.shape[0] == 45, f"应为 45 行，实际 {q_var.shape[0]}"
assert q_var["负荷值"].min() > THR, "结果里不该有小于等于阈值的行"
try:
    dev.query("负荷值 > THR")
    raise AssertionError("不加 @ 应该报 UndefinedVariableError")
except AssertionError:
    raise
except Exception:
    pass
print("TODO(7) 通过 -> @ 引用局部变量命中", q_var.shape[0], "行")"""
    ),
    code(
        """# @@todo(8) 用 query 选出 STATION_A 开头的台区（条件里含 .str 方法调用）
# @@hint dev.query("台区编号.str.startswith('STATION_A')")；本机 3.0 默认引擎已支持
q_station_a = dev.query("台区编号.str.startswith('STATION_A')")
# @@end
# ---- 验收 ----
assert q_station_a.shape[0] == 44, f"应为 44 行，实际 {q_station_a.shape[0]}"
assert q_station_a.shape[1] == 13, "query 不改变列数"
assert q_station_a["台区编号"].str.startswith("STATION_A").all()
manual = dev[dev["台区编号"].str.startswith("STATION_A")]
assert q_station_a.index.equals(manual.index), "query 与掩码结果应完全一致"

# 澄清一个流传很广但已失效的说法：.str 是否必须换 engine="python"？
q_python = dev.query("台区编号.str.startswith('STATION_A')", engine="python")
assert q_python.index.equals(q_station_a.index), "两种引擎结果一致 -> 3.0 默认引擎已支持 .str"
print("TODO(8) 通过 -> 命中", q_station_a.shape[0], "行；默认引擎与 engine='python' 结果一致")"""
    ),
    code(
        """# @@todo(9) 用 filter 按列名筛列：取出名字里含「设备」的列
# @@hint DataFrame.filter(like="设备")
dev_columns = dev.filter(like="设备")
# @@end
# ---- 验收 ----
assert list(dev_columns.columns) == ["设备类型", "设备编号"], f"列不对：{list(dev_columns.columns)}"
assert len(dev_columns) == 270, "filter 只筛列，行数一行不少"
assert dev_columns.shape == (270, 2), f"形状应为 (270, 2)，实际 {dev_columns.shape}"

# 对照组：想筛出「变压器」的行，必须用掩码而不是 filter
rows_by_mask = dev[dev["设备类型"] == "变压器"]
assert len(rows_by_mask) == 55, "掩码才是筛行的正确工具"
print("TODO(9) 通过 -> filter 得到", list(dev_columns.columns), "，行数仍为", len(dev_columns))"""
    ),
    code(
        """CUT = 1000
FILL = 888

# @@todo(10) 分别用 where 与 mask 把「负荷值 >= 1000」的哨兵换成 FILL，存入 w_filled 与 m_filled
# @@hint Series.where(cond, other) 保留满足 cond 的；Series.mask(cond, other) 替换满足 cond 的
w_filled = dev["负荷值"].where(dev["负荷值"] < CUT, FILL)
m_filled = dev["负荷值"].mask(dev["负荷值"] >= CUT, FILL)
# @@end
# ---- 验收 ----
n_w = int(w_filled.eq(FILL).sum())
n_m = int(m_filled.eq(FILL).sum())
assert n_w == 28, f"where 命中 FILL 应为 28（7 哨兵 + 21 原有缺失），实际 {n_w}"
assert n_m == 7, f"mask 命中 FILL 应为 7（只换哨兵），实际 {n_m}"
assert n_w - n_m == 21, "差值必须等于原有缺失数"
assert int(w_filled.isna().sum()) == 0, "where 给了 other 后不该再有 NaN"
assert int(m_filled.isna().sum()) == 21, "mask 必须保住原有 21 个缺失"

# 想让 where 也保住缺失：把 notna 写进条件
w_safe = dev["负荷值"].where((dev["负荷值"] < CUT) | dev["负荷值"].isna(), FILL)
assert int(w_safe.eq(FILL).sum()) == n_m, "加上 isna() 条件后 where 应与 mask 一致"
print(f"TODO(10) 通过 -> where 换掉 {n_w} 个（含 21 个缺失），mask 只换 {n_m} 个")"""
    ),
    code(
        """# @@todo(11) 用 loc 做条件赋值：负荷值 > 500 的行，给「复核标记」列写「高负荷待复核」
# @@hint 先 df.loc[cond, "列"] = 值，再把未命中的用 fillna 补「常规」
flagged = dev.copy()
flagged.loc[flagged["负荷值"] > 500, "复核标记"] = "高负荷待复核"
flagged["复核标记"] = flagged["复核标记"].fillna("常规")
# @@end
# ---- 验收 ----
n_flag = int((flagged["复核标记"] == "高负荷待复核").sum())
n_cond = int((flagged["负荷值"] > 500).sum())
assert n_flag == n_cond == 45, f"条件命中 {n_cond} 与赋值命中 {n_flag} 必须相等且为 45"
assert flagged["复核标记"].isna().sum() == 0, "补默认值后不该有缺失"
assert flagged["复核标记"].value_counts().to_dict() == {"常规": 225, "高负荷待复核": 45}

# 反例：链式赋值一行都改不上
chained = dev.copy()
chained[chained["负荷值"] > 500]["备注"] = "xxx"
assert int((chained["备注"] == "xxx").sum()) == 0, "链式赋值应该静默失效"
print("TODO(11) 通过 -> loc 条件赋值命中", n_flag, "行；链式赋值命中 0 行")"""
    ),
    code(
        """# @@todo(12) 用 np.select 造「风险等级」列：危急->高、严重->中、其余->低，存入 risk
# @@hint np.select(条件列表, 取值列表, default="低")  —— 注意条件顺序：窄的在前
risk_conds = [dev["缺陷等级"].eq("危急"), dev["缺陷等级"].eq("严重")]
risk = np.select(risk_conds, RISK_CHOICES, default="低")
# @@end
# ---- 验收 ----
assert isinstance(risk, np.ndarray), f"np.select 返回 ndarray，实际 {type(risk).__name__}"
assert len(risk) == 270, "长度必须等于行数"
counts = pd.Series(risk).value_counts().to_dict()
assert counts == {"低": 164, "中": 75, "高": 31}, f"分布不对：{counts}"
assert pd.Series(risk).isna().sum() == 0, "default 保证不产生缺失"

# 反证：条件顺序写反会出错
reversed_risk = np.select([dev["负荷值"] > 100, dev["负荷值"] > 500], ["中", "高"], default="低")
assert int(pd.Series(reversed_risk).eq("高").sum()) == 0, \
    "宽条件在前时，窄条件永远命中不了（这就是顺序陷阱）"
print("TODO(12) 通过 -> 风险分布", counts)"""
    ),
    md(
        """
---

## 综合题

### 综合 ① 高优先检修清单

从 270 条缺陷记录里筛出需要**优先安排检修**的记录，条件三条（缺一不可）：

1. `缺陷等级` ∈ {严重, 危急}
2. `设备类型` ∈ {变压器, 断路器}
3. `处理状态` 含「处理」二字

产出 `urgent`（保留 `记录ID`/`台区编号`/`设备类型`/`缺陷等级`/`负荷值`/`处理状态` 六列），
并画一张「按缺陷等级 × 设备类型」的清点柱状图。

**这一步的关键不是筛出来，而是证明自己没筛错**：三个条件各自的命中数、
交集数、清单行数必须能对上。
"""
    ),
    code(
        """# @@todo 综合①：三条件组合筛出 urgent，并画出分组柱状图，图对象存入 ax
# @@hint df.loc[cond1 & cond2 & cond3, [列...]] ；再 groupby + unstack + plot.bar(ax=ax)
cond_level = dev["缺陷等级"].isin(["严重", "危急"])
cond_type = dev["设备类型"].isin(["变压器", "断路器"])
cond_status = dev["处理状态"].str.contains("处理", na=False)

urgent = dev.loc[
    cond_level & cond_type & cond_status,
    ["记录ID", "台区编号", "设备类型", "缺陷等级", "负荷值", "处理状态"],
]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
urgent.groupby(["缺陷等级", "设备类型"]).size().unstack(fill_value=0).plot.bar(ax=ax)
ax.set_title("高优先检修清单：缺陷等级 x 设备类型")
ax.set_xlabel("缺陷等级")
ax.set_ylabel("条数")
plt.tight_layout()
plt.show()
# @@end
# ---- 验收 ----
assert len(urgent) == 38, f"清单应有 38 行，实际 {len(urgent)}"
assert list(urgent.columns) == ["记录ID", "台区编号", "设备类型", "缺陷等级", "负荷值", "处理状态"]
# 三个条件逐一自查
assert int(cond_level.sum()) == 106
assert int(cond_type.sum()) == 96, f"变压器55+断路器41 应为 96，实际 {int(cond_type.sum())}"
assert int(cond_status.sum()) == 257
assert int((cond_level & cond_type & cond_status).sum()) == len(urgent), \
    "三条件交集数必须等于清单行数"
# 清单内部结构
assert urgent["缺陷等级"].value_counts().to_dict() == {"严重": 30, "危急": 8}
assert urgent["设备类型"].value_counts().to_dict() == {"断路器": 20, "变压器": 18}
# 清单里不能有不符合条件的行（反向抽查）
assert urgent["缺陷等级"].isin(["严重", "危急"]).all()
assert urgent["设备类型"].isin(["变压器", "断路器"]).all()
assert urgent["处理状态"].str.contains("处理", na=False).all()
# 图
assert ax is not None and len(ax.patches) >= 4, f"柱状图柱子数不对：{len(ax.patches)}"
print("综合① 通过 -> 清单", urgent.shape)
print(urgent["缺陷等级"].value_counts().to_string())"""
    ),
    md(
        """
### 综合 ② 条件筛选自查报告

写一个自查函数的核心逻辑：给定掩码，输出它的质量指标。
竞赛里"数据清洗"那几分的交出物之一就是**这种能自证的过程记录**。
"""
    ),
    code(
        """# @@todo 综合②：产出自查指标存入 audit 字典
# @@hint 用 crit_mask / cond_level 现算：命中数 / 占比 / 与"严格反选"的差异
mask_audit = crit_mask
n_hit = int(mask_audit.sum())
n_total = len(mask_audit)
hit_ratio = round(n_hit / n_total, 4)
complement_hit = int((~mask_audit).sum())
complement_nan_extra = int((~mask_audit & crit_mask.isna()).sum())
audit = {
    "命中数": n_hit,
    "总行数": n_total,
    "命中占比": hit_ratio,
    "反选命中数": complement_hit,
    "取反额外带入的缺失数": complement_nan_extra,
}
# @@end
# ---- 验收 ----
expect = {"命中数", "总行数", "命中占比", "反选命中数", "取反额外带入的缺失数"}
assert set(audit) == expect, f"键不对。多 {set(audit) - expect}，缺 {expect - set(audit)}"
assert audit["命中数"] == 31 and audit["总行数"] == 270
assert abs(audit["命中占比"] - 0.1148) < 0.0005, f"占比应约 0.1148，实际 {audit['命中占比']}"
assert audit["反选命中数"] == 239, f"反选应为 239，实际 {audit['反选命中数']}"
assert audit["命中数"] + audit["反选命中数"] == audit["总行数"], "命中 + 反选必须等于总数"
assert audit["取反额外带入的缺失数"] == 0, "本章缺陷等级无缺失，取反不会多带入缺失"
print("综合② 通过 -> 条件自查报告：")
for key, value in audit.items():
    print(f"   {key:20s} = {value!r}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（五题都能答上来才算过）：

1. `dev[mask]["列"] = v` 为什么不生效？正确写法是什么？怎么自查？
2. `~mask` 和"严格反选"差在哪？什么情况下两者结果不同？
3. `where(cond, 888)` 和 `mask(~cond, 888)` 等价吗？为什么实测差 21 行？
4. `filter(like="设备")` 为什么筛不出一行数据？它到底在筛什么？
5. `np.select` 的条件列表为什么必须"窄条件在前"？给个会翻车的反例。

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
