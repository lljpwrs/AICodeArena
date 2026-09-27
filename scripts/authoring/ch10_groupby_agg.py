"""ch10 —— 分组聚合：agg 的四种写法与三个默认值陷阱

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch10_groupby_agg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch10_groupby_agg"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("data")
dev = pd.read_csv(DATA / "device_defects.csv")

GROUP_KEY = "设备类型"
VALUE_COL = "负荷值"

print("pandas", pd.__version__)
print("dev", dev.shape)
print("分组基准：", dev.groupby(GROUP_KEY).size().to_dict())'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch10 分组聚合：`agg` 的四种写法和三个默认值陷阱

> 方向：数据预处理 ｜ 竞赛对应：**数据挖掘（分组统计）+ 数据探索（分布概览）**
> —— 分组聚合是"从行级数据里抽特征"的标准手段，
> 也是把「明细表」变成「一张能看的汇总表」的唯一办法。

本章的核心不是"会用 `groupby`"，而是三件更硬的事：

1. `agg` 有**四种写法**，输出的列结构各不相同——选错就得返工重写
2. `size()` 和 `count()` **差 21 行**，这两个最容易被当成同一个东西
3. `groupby` 有**三个默认值**会在你不知情时改变结果：`dropna` / `observed` / `as_index`
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据挖掘 · 分组统计 | 20 分模块内的核心动作 | 按维度聚合指标、交叉汇总 |
| 数据探索 · 分布概览 | 每项 2 分 | 分组后的均值/中位数/最大最小 |
| 特征工程 | 影响建模分 | 组内均值/标准差/排名广播回每一行 |
| 结果交付 | 提交格式分 | `as_index=False` 拿到可直接写出的表 |

**为什么单独成章**：竞赛里的"数据挖掘"环节几乎都是从
「按台区/按设备类型/按时间维度汇总」开始的。
而 `groupby` 是 pandas 里**参数默认值最多、且默认值会改结果**的 API——
本章会把三个默认值逐一实测出来。
"""
    ),
    md(
        """
## 二、API 速查表

| API | 返回 | 一句话说明 |
|---|---|---|
| `df.groupby(键)` | `DataFrameGroupBy` | 只分组，不算（惰性对象） |
| `gb.size()` | `Series` | 每组**行数**（含缺失，不排除） |
| `gb[列].count()` | `Series` | 每组该列的**非缺失**个数 |
| `gb[列].mean()/sum()/median()/std()` | `Series` | 单函数快捷聚合 |
| `gb[列].agg(["mean","max"])` | `DataFrame` | 多函数，列名 = 函数名 |
| `gb.agg({"列":["mean"],"列2":"sum"})` | `DataFrame` | 多列多函数，**列名变 MultiIndex** |
| `gb.agg(别名=("列","函数"))` | `DataFrame` | **命名聚合**（推荐，列名直接可用） |
| `gb.transform("mean")` | `Series` | 组统计量**广播回原长度**（见 ch11） |
| `gb.rank(pct=True)` | `Series` | 组内排名（0~1） |
| `gb.head(n)` | `DataFrame` | 每组前 n 行（配合 `sort_values` 做组内 top N） |
| `gb.ngroups` | `int` | 组数 |
| `df.pivot_table(index=,columns=,values=,aggfunc=)` | `DataFrame` | 交叉汇总（等价于"多维 groupby + unstack"） |
| `pd.crosstab(行,列)` | `DataFrame` | 交叉**计数**（只数条数） |

### `agg` 四种写法对照（最重要的表）

| 写法 | 代码 | 输出列名 | 什么时候用 |
|---|---|---|---|
| ① 单函数 | `gb["负荷值"].mean()` | `Series`，index=组 | 只要一个指标 |
| ② 函数列表 | `gb["负荷值"].agg(["mean","max"])` | 函数名（单层） | 同一列多个指标 |
| ③ 字典 | `gb.agg({"负荷值":["mean"],"湿度":"mean"})` | **MultiIndex 元组** | 多列多指标（竞赛少用） |
| ④ 命名聚合 | `gb.agg(均值=("负荷值","mean"))` | **自定义名（单层）** | ⭐ **默认选它** |

> ④ 为什么最好：`as_index=False` + 命名聚合能直接得到一张
> **列名正常、可以 `to_csv` 直接交卷**的表。② ③ 的列名要么是函数名，
> 要么是元组，后续还得改列名——多一步就多一个出错点。
"""
    ),
    code(HEADER),
    md(
        """
---

## 三、逐节讲解

### 3.1 心智模型：split → apply → combine

`groupby` 干三件事，记住这三步，所有参数都能对上号：

| 阶段 | 做什么 | 对应参数 |
|---|---|---|
| **split** 分组 | 按键把行切成若干块 | `dropna`（缺失键算不算一组） / `observed`（空类别算不算一组） |
| **apply** 计算 | 在每块上跑聚合函数 | `agg` / `transform` / `apply` |
| **combine** 合并 | 把结果拼回一张表 | `as_index`（组键变索引还是变列） / `sort` |

**"缺一块行不行"是判断参数归属的最好办法**：凡是影响"行被分到哪块"的，都是 split 阶段；
凡是影响"结果表长什么样"的，都是 combine 阶段。
"""
    ),
    code(
        """gb = dev.groupby(GROUP_KEY)
print("① groupby 是惰性的——只分组，不算：")
print("   对象类型:", type(gb).__name__)
print("   组数 ngroups:", gb.ngroups)
print("   组键:", list(gb.groups.keys()))

print("\\n② 各组行数 size()：")
size_g = gb.size()
print(size_g.to_string())
print("   合计:", int(size_g.sum()), "= 行数", len(dev))

print("\\n③ count() 与 size() 的差别（这里就出问题了）：")
count_g = gb[VALUE_COL].count()
cmp_df = pd.DataFrame({"size": size_g, "count": count_g})
cmp_df["差（=缺失数）"] = cmp_df["size"] - cmp_df["count"]
print(cmp_df.to_string())
print("\\n   size 合计:", int(size_g.sum()))
print("   count 合计:", int(count_g.sum()), " ← 少", int(size_g.sum() - count_g.sum()), "个，正是负荷值的缺失数")
print("   -> size() 数的是行，count() 数的是值。'多少台设备'用 size，'多少条有效负荷记录'用 count")"""
    ),
    md(
        """
> **判定规则**：
> - 问"**有多少条记录**" → `size()`
> - 问"**这个指标有多少个有效观测**" → `count()`
>
> 这两个之所以危险，是因为**数据没有缺失时它们完全相等**，
> 于是你会在小样本上验证通过、在正式数据上差 21 行。
> 写 `agg` 时同理：`("负荷值", "size")` 和 `("负荷值", "count")` 是两个不同的指标。

---

### 3.2 单函数聚合：五个最常用的

`mean` / `sum` / `median` / `std` / `max` 都可以直接挂。但要记住一件事：
**`mean` 会被异常值污染**——本章数据里负荷值含 7 个 `>= 1000` 的哨兵码，
所以分组均值会明显偏高。所以下面每一组都同时输出 `mean` 和 `median`。
"""
    )
]
LESSON += [
    code(
        """print("① 分组均值（先看这个，但别全信）：")
mean_g = dev.groupby(GROUP_KEY)[VALUE_COL].mean()
print(mean_g.round(4).to_string())

print("\\n② 均值 vs 中位数：哨兵值把均值拉高了")
med_g = dev.groupby(GROUP_KEY)[VALUE_COL].median()
cmp2 = pd.DataFrame({"mean": mean_g, "median": med_g})
cmp2["mean/median"] = (cmp2["mean"] / cmp2["median"]).round(2)
print(cmp2.round(2).to_string())
print("\\n   变压器组：mean = %.2f，median = %.2f，相差 %.1f 倍"
      % (mean_g["变压器"], med_g["变压器"], mean_g["变压器"] / med_g["变压器"]))
print("   原因：该组里有 9999 这种哨兵码，把均值拽上去了")

print("\\n③ 其他常用聚合：")
print("   合计:", dev.groupby(GROUP_KEY)[VALUE_COL].sum().round(2).to_dict())
print("   标准差:", dev.groupby(GROUP_KEY)[VALUE_COL].std().round(2).to_dict())
print("   最小值:", dev.groupby(GROUP_KEY)[VALUE_COL].min().to_dict(), " ← 3 个组的最小值是 -1 哨兵")
print("   湿度均值:", dev.groupby(GROUP_KEY)["湿度"].mean().round(2).to_dict())"""
    ),
    md(
        """
> **一个能立刻提升分析质量的习惯**：任何"分组求均值"的地方，
> **旁边都放一个中位数**。两者差距大 → 组内有异常值 → 先回去清洗。
> 本章数据里变压器组 `mean/median = 1.89`，这就是明确的清洗信号。

---

### 3.3 `agg` 的四种写法与它们的输出结构

这一节是本章的核心。四种写法的**输出结构完全不同**，选错不会报错，
但你后面 `to_csv` 或 `merge` 的时候才发现列名是一堆元组。
"""
    ),
    code(
        """gv = dev.groupby(GROUP_KEY)[VALUE_COL]

print("① 单函数 -> Series（index 是组键）")
s1 = gv.mean()
print("   type:", type(s1).__name__, "| index.name:", s1.index.name)

print("\\n② 函数列表 -> DataFrame，列名是函数名（单层）")
s2 = gv.agg(["count", "mean", "median", "std", "max"])
print("   shape:", s2.shape, "| columns:", list(s2.columns))
print(s2.round(2).to_string())

print("\\n③ 字典 -> DataFrame，但列名是 MultiIndex 元组（最容易踩的坑）")
s3 = dev.groupby(GROUP_KEY).agg({"负荷值": ["mean", "max"], "湿度": "mean"})
print("   shape:", s3.shape)
print("   columns 类型:", type(s3.columns).__name__)
print("   columns 内容:", list(s3.columns))
print("   -> 想把它压成单层列名，要么 s3.columns = [...] 手工改，要么直接用命名聚合")

print("\\n④ 命名聚合 -> 列名完全由你定（推荐）")
s4 = dev.groupby(GROUP_KEY).agg(
    样本数=("负荷值", "size"),
    有效负荷数=("负荷值", "count"),
    平均负荷=("负荷值", "mean"),
    中位负荷=("负荷值", "median"),
    最大负荷=("负荷值", "max"),
    危急数=("缺陷等级", lambda s: int((s == "危急").sum())),
)
print("   shape:", s4.shape, "| columns:", list(s4.columns))
print(s4.round(2).to_string())"""
    ),
    md(
        """
> **命名聚合里可以用 lambda**，这让它比字典写法强得多：任何"自定义指标"
> （危急率、超标条数、去重计数……）都能塞进同一张表。
>
> **注意**：`agg` 里的 lambda 收到的是**每个组的子 Series**，
> 所以要写 `lambda s: int((s == "危急").sum())`——拿到的是值，不是行。

---

### 3.4 难点深挖：`as_index` —— 结果表能不能直接交卷

**为什么难**：`groupby` 默认把组键变成**索引**（`as_index=True`）。
这让 `df["设备类型"]` 这种写法失效（因为它是 index 不是列），
于是你要么 `reset_index()`，要么一开始就设 `as_index=False`。
**两个选择都没错，但混用会让代码里到处是 `reset_index`。**
"""
    ),
    code(
        """print("① 默认 as_index=True：组键是索引，不是列")
d1 = dev.groupby(GROUP_KEY).agg(平均负荷=("负荷值", "mean"))
print("   columns:", list(d1.columns))
print("   index.name:", d1.index.name)
print("   能写 d1['设备类型'] 吗:", "设备类型" in d1.columns, " ← 不在列里")

print("\\n② as_index=False：组键变回普通列")
d2 = dev.groupby(GROUP_KEY, as_index=False).agg(
    样本数=("负荷值", "size"),
    平均负荷=("负荷值", "mean"),
)
print("   columns:", list(d2.columns))
print(d2.round(2).to_string(index=False))

print("\\n③ 两种写法数值上完全一致（只是组键位置不同）：")
d3 = d1.reset_index()
print("   形状:", d3.shape, "vs", d2.shape)
print("   平均负荷数值一致:", bool(np.allclose(d3["平均负荷"].to_numpy(), d2["平均负荷"].to_numpy())))
print("   组键顺序一致:", d3[GROUP_KEY].tolist() == d2[GROUP_KEY].tolist())

print("\\n④ 建议：")
print("   要交卷 / 要 to_csv / 要 merge -> 一律 as_index=False + 命名聚合")
print("   只做中间计算、还要 .loc[组名] 取数 -> 用默认的 as_index=True 更方便")"""
    ),
    md(
        """
> **判定规则**：
> - 结果是**最终产出**（要写文件、要 join、要画图）→ `as_index=False`
> - 结果是**中间步骤**（还要 `.loc["变压器"]` 取某个组）→ 保留索引更方便
>
> 一条纪律：**一个 notebook 里选定一种风格就别改**。混用会出现
> "有时候 `df['设备类型']` 能用、有时候不能用"的困惑，而这类困惑
> 在限时竞赛里是最浪费时间的。

---

### 3.5 难点深挖：`groupby` 的三个默认值（都会改结果）

`groupby` 最反直觉的地方在于：**有三个默认值会静默改变你的结果**，
而且它们的默认值在 pandas 版本间还变过。

| 参数 | 本机 3.0 默认 | 作用 | 踩坑表现 |
|---|---|---|---|
| `dropna` | `True` | 键是缺失的行**整组丢掉** | 组数莫名少一组，且**不报警** |
| `observed` | **`True`**（2.x 是 `False`） | 类别键只保留"实际出现过的"类别 | 老教程教你设 `True`，3.0 已默认 `True` |
| `as_index` | `True` | 组键变索引 | `df["组键"]` 报 KeyError |
"""
    ),
    code(
        """print("① dropna 默认 True：键缺失的整组被丢掉")
d = dev.copy()
d.loc[0:4, "缺陷等级"] = np.nan          # 人为制造 5 行缺失键
print("   默认 dropna=True :", d.groupby("缺陷等级").size().to_dict())
print("   dropna=False     :", d.groupby("缺陷等级", dropna=False).size().to_dict())
print("   -> 少了 nan 那一组，而且不报错。'各组加起来等于总行数'这个自检能抓住它")
print("   自检: 默认分组求和 =", int(d.groupby("缺陷等级").size().sum()),
      " vs 总行数 =", len(d))

print("\\n② observed 的默认值在 3.0 变了（这条会让老教程把你带偏）")
full_dtype = pd.CategoricalDtype(categories=dev["设备类型"].astype("category").cat.categories)
sub = dev[dev["设备类型"].isin(["变压器", "绝缘子"])]
key = sub["设备类型"].astype(full_dtype)
print("   子集里只有 2 类，但 key 的 categories 有", len(full_dtype.categories), "类")
print("   不传 observed :", sub.groupby(key).size().to_dict(), " ← 2 组")
print("   observed=True :", sub.groupby(key, observed=True).size().to_dict(), " ← 与上面一致")
obs_false = sub.groupby(key, observed=False).size()
print("   observed=False:", obs_false.to_dict())
print("   -> 组数变成", len(obs_false), "，多出 3 个计数为 0 的空组（笛卡尔积）")
print("   实测默认值：", "True（3.0 起）" if sub.groupby(key).size().shape[0] == 2 else "False")

print("\\n③ as_index 影响的是形状，不是数值")
print("   见上一节，这里只提醒：它不会改数值，只会改组键的位置")

print("\\n④ 三条自检（写进每个分组任务里）")
print("   1) gb.size().sum() == len(df)                 -> 抓 dropna")
print("   2) len(gb.size()) == df[键].nunique()          -> 抓 observed 空组")
print("   3) '组键' in result.columns                    -> 抓 as_index")"""
    ),
    md(
        """
> **判定规则**（背下来）：
>
> | 场景 | 怎么写 |
> |---|---|
> | 键**确定没有缺失** | 用默认，不用管 |
> | 键**可能有缺失**且缺失本身有含义（"未评级"） | `dropna=False` |
> | 键是 `category` 且只统计实际出现的类别 | 用默认（3.0 已是 `observed=True`） |
> | 要把**所有类别**（含未出现的）都列出来 | `observed=False`（做完整报表时才需要） |
> | 结果要交卷 | `as_index=False` |
>
> **每次分组后立刻跑那三条自检**，比事后 debug 便宜一百倍。

---

### 3.6 多键分组与 `unstack`

按两个维度分组会得到 MultiIndex；`unstack` 把最内层搬到列上，
就变成一张**交叉表**——这是竞赛里"一次性看清楚"的标准图形。
"""
    )
]
LESSON += [
    code(
        """print("① 两个键分组 -> MultiIndex")
gm = dev.groupby(["台区编号", "设备类型"])[VALUE_COL].mean()
print("   组数:", len(gm), "| index.names:", gm.index.names)
print("   前 4 行:")
print(gm.head(4).round(2).to_string())

print("\\n② unstack 把最内层键搬到列上 -> 交叉表")
cross = gm.unstack()
print("   形状:", cross.shape, "（18 台区 x 5 类型）")
print(cross.round(1).iloc[:4, :].to_string())
print("\\n   空单元格（NaN）的含义：那个台区没有那种设备 -> 这是真·缺失，不是数据错误")

print("\\n③ 台区填满类型：")
print("   每个台区有几种设备:", cross.notna().sum(axis=1).value_counts().sort_index().to_dict())

print("\\n④ as_index=False + 多键，直接得到长表")
gl = dev.groupby(["台区编号", "设备类型"], as_index=False)[VALUE_COL].mean()
print("   columns:", list(gl.columns), "| shape:", gl.shape)

print("\\n⑤ 交叉计数用 crosstab 更直接")
ct = pd.crosstab(dev["设备类型"], dev["缺陷等级"])
print("   shape:", ct.shape)
print(ct.to_string())
print("\\n   行合计:", ct.sum(axis=1).to_dict())"""
    ),
    md(
        """
> **`unstack()` 之后的 `NaN` 是什么**：是"这个组合不存在"，不是"数据缺失"。
> 两者必须分开对待——**填 0 还是留 `NaN` 取决于业务**：
> - "该台区没有这种设备" → 填 0 会误导（0 台和没有是两回事），留 `NaN` 更好
> - "该台区这种设备的条数" → 填 0 才对（真的 0 条）

---

### 3.7 组内选取：`sort_values` + `groupby().head(n)`

"每个组取最大的 N 条"是竞赛高频题。标准套路是**先全局排序，再分组取头**。
"""
    ),
    code(
        """print("① 每组取负荷值最高的 1 条：")
top1 = dev.sort_values(VALUE_COL, ascending=False).groupby(GROUP_KEY).head(1)
print(top1[["记录ID", GROUP_KEY, VALUE_COL, "缺陷等级"]].to_string(index=False))

print("\\n② 注意：三个组的 top1 都是 9999 —— 哨兵码污染了极值选取")
print("   -> 取 top N 之前必须先把哨兵处理掉（回看 ch05）")

print("\\n③ 清洗后再取：")
dev_clean = dev[dev[VALUE_COL].between(0, 1000)]
top1_clean = dev_clean.sort_values(VALUE_COL, ascending=False).groupby(GROUP_KEY).head(1)
print(top1_clean[["记录ID", GROUP_KEY, VALUE_COL]].to_string(index=False))

print("\\n④ 每组取前 2 条（按负荷值）：")
top2 = dev_clean.sort_values(VALUE_COL, ascending=False).groupby(GROUP_KEY).head(2)
print("   形状:", top2.shape, "= 5 组 x 2 条")
print("   组内排名 rank(pct=True) 前 5 行:",
      dev.groupby(GROUP_KEY)[VALUE_COL].rank(pct=True).head(5).round(3).tolist())

print("\\n⑤ 每组最小 1 条（升序排序后取 head）：")
bot1 = dev_clean.sort_values(VALUE_COL).groupby(GROUP_KEY).head(1)
print(bot1[["记录ID", GROUP_KEY, VALUE_COL]].to_string(index=False))"""
    ),
    md(
        """
> **顺序不能反**：`groupby().head(n)` 是"取每组**原始顺序**里的前 n 行"，
> **它不会自动排序**。所以一定是 `sort_values(...)` 在前。
> 反过来的写法（`groupby().apply(lambda d: d.nlargest(n, col))`）结果对但慢很多，
> 且返回结构会变（见 ch11）。

---

### 3.8 `pivot_table` 与 `groupby` 的等价关系

`pivot_table` 不是新东西，它就是"多维 `groupby` + `unstack`"的语法糖。
用数据验证这个等价关系，你以后就知道什么时候用哪个。
"""
    ),
    code(
        """print("① pivot_table 写法：")
pt = dev.pivot_table(index=GROUP_KEY, columns="缺陷等级", values=VALUE_COL, aggfunc="mean")
print("   shape:", pt.shape)
print(pt.round(1).to_string())

print("\\n② 等价的 groupby 写法：")
gt = dev.groupby([GROUP_KEY, "缺陷等级"])[VALUE_COL].mean().unstack()
print("   两者形状相同:", pt.shape == gt.shape)
print("   逐格数值一致:", bool(np.allclose(pt.to_numpy(), gt.to_numpy(), equal_nan=True)))

print("\\n③ 抽一格手工验证：变压器 x 危急")
manual = dev[(dev[GROUP_KEY] == "变压器") & (dev["缺陷等级"] == "危急")][VALUE_COL].mean()
print("   pivot_table:", round(pt.loc["变压器", "危急"], 4))
print("   手写筛选  :", round(manual, 4))
print("   一致:", abs(pt.loc["变压器", "危急"] - manual) < 1e-9)

print("\\n④ aggfunc='count' 时是计数表：")
print(dev.pivot_table(index=GROUP_KEY, columns="缺陷等级", values="记录ID", aggfunc="count").to_string())
print("   与 crosstab 一致?",
      dev.pivot_table(index=GROUP_KEY, columns="缺陷等级", values="记录ID", aggfunc="count").equals(
          pd.crosstab(dev[GROUP_KEY], dev["缺陷等级"])))"""
    ),
    md(
        """
> **选型规则**：
> - 结果要**继续参与计算**（还要 `merge`、还要加减） → 用 `groupby`（能拿到标准列名）
> - 结果是**给人看的报表**，两个维度交叉 → 用 `pivot_table`（可读性更好）
> - 只数**条数** → 用 `pd.crosstab`（最简洁）
>
> `pivot_table` 的默认 `aggfunc` 是 `"mean"`——**这是它最大的暗坑**：
> 忘记写 `aggfunc` 时它会静默求均值，而你以为是计数。

---

### 3.9 `transform`：把组统计量广播回每一行（ch11 预告）

`agg` 把 270 行压成 5 行；`transform` **保持 270 行**，
把每组的统计量贴回该组每一行。这是特征工程里最常用的一招：
"这条记录的负荷值，相对于它所在设备类型是高还是低？"
"""
    ),
    code(
        """print("① agg 会压缩行数，transform 不会：")
print("   agg 长度      :", len(dev.groupby(GROUP_KEY)[VALUE_COL].mean()))
print("   transform 长度:", len(dev.groupby(GROUP_KEY)[VALUE_COL].transform("mean")))

print("\\n② transform 的值就是该行所属组的统计量：")
dev_feat = dev.copy()
dev_feat["组均值"] = dev_feat.groupby(GROUP_KEY)[VALUE_COL].transform("mean")
dev_feat["组标准差"] = dev_feat.groupby(GROUP_KEY)[VALUE_COL].transform("std")
print(dev_feat[["记录ID", GROUP_KEY, VALUE_COL, "组均值"]].head(5).round(2).to_string(index=False))
print("\\n   第 0 行属于", dev_feat.loc[0, GROUP_KEY], "，组均值 =", round(dev_feat.loc[0, "组均值"], 4))

print("\\n③ 组内 z-score：这条记录相对同类型设备偏了多少个标准差")
dev_feat["组内z"] = (dev_feat[VALUE_COL] - dev_feat["组均值"]) / dev_feat["组标准差"]
print("   前 3 行 z:", dev_feat["组内z"].head(3).round(3).tolist())

print("\\n④ 组内百分位排名：")
dev_feat["组内排名"] = dev_feat.groupby(GROUP_KEY)[VALUE_COL].rank(pct=True)
print("   前 5 行:", dev_feat["组内排名"].head(5).round(3).tolist())

print("\\n⑤ 组内占比：这条记录的负荷占本组总负荷的比例")
dev_feat["组内占比"] = dev_feat[VALUE_COL] / dev_feat.groupby(GROUP_KEY)[VALUE_COL].transform("sum")
print("   各组组内占比之和:", dev_feat.groupby(GROUP_KEY)["组内占比"].sum().round(6).to_dict())
print("   -> 每组都加起来等于 1.0，这就是 transform 广播正确的证明")"""
    ),
    md(
        """
---

## 四、易错点清单

1. **`size()` 当成 `count()`** → 有缺失时差 21 行，且无缺失时完全相等（发现不了）
2. `agg` 用字典写法 → 列名变成 MultiIndex 元组，`to_csv` 出来是一堆括号
3. `agg` 后想用 `df["组键"]` → 报 `KeyError`（默认 `as_index=True`，组键在索引上）
4. 分组后**不做"各组之和 == 总行数"自检** → 漏掉 `dropna` 丢的那一组
5. `dropna` 默认 `True` → 键缺失的行整组消失，**不报警**
6. `observed` 在 3.0 默认已是 `True` → 照抄老教程显式传 `True` 是多余（但无害）；
   真要看空类别得显式传 `False`
7. `unstack()` 后的 `NaN` 被一律填 0 → "该组合不存在"和"计数为 0"被混淆
8. **`groupby().head(n)` 不排序** → 拿到的是原始顺序前 n 行，不是最大 n 行
9. 取组内 top N 前不清哨兵 → 三个组的 top1 全是 9999（实测）
10. 只看 `mean` 不看 `median` → 异常值污染看不出来（变压器组 mean/median = 1.89）
11. `pivot_table` 忘记 `aggfunc` → 默认求**均值**，你以为在做计数
12. `transform` 与 `agg` 混用 → 一个保持长度、一个压缩长度，写错必炸

---

## 五、本章小结

| 需求 | 正确写法 |
|---|---|
| 每组有多少条记录 | `gb.size()` |
| 每组某列有多少有效值 | `gb[列].count()` |
| 一列多指标 | `gb[列].agg(["mean","max"])` |
| 多列多指标（**推荐**） | `gb.agg(均值=("负荷值","mean"), 危急数=("缺陷等级", lambda s: ...))` |
| 结果要交卷 | 加 `as_index=False`，得到正常列名的表 |
| 键可能有缺失 | `dropna=False`（否则整组消失） |
| 要看未出现的类别 | `observed=False`（3.0 默认已是 `True`） |
| 二维交叉表 | `pivot_table(index=,columns=,values=,aggfunc=)` 或 `crosstab` |
| 组内 top N | `df.sort_values(列, ascending=False).groupby(键).head(n)` |
| 组统计量广播回每行 | `gb[列].transform("mean")` |
| 三条自检 | `size().sum()==len(df)`、`组数==nunique()`、`'组键' in result.columns` |
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch10 练习 —— 分组聚合

数据：`data/device_defects.csv`（270 行 x 13 列）

**每题都有 `assert` 验收块**。跑通即正确，跑不通的报错信息本身就说清了哪里错。
**不要翻答案版。**

| 题号 | 考什么 |
|---|---|
| TODO(1) | `size()` vs `count()`（差 21 行） |
| TODO(2) | 单函数聚合 + 均值/中位数对比 |
| TODO(3) | `agg` 函数列表 |
| TODO(4) | `agg` 字典（列名变 MultiIndex） |
| TODO(5) | **命名聚合**（含条件计数） |
| TODO(6) | `as_index=False` |
| TODO(7) | 多键分组 + `unstack` |
| TODO(8) | `dropna=False` |
| TODO(9) | `observed` 的真实默认值 |
| TODO(10) | 组内 top N（先排序再 head） |
| TODO(11) | `pivot_table` 与手写筛选取值对账 |
| TODO(12) | `transform` 组内 z-score |
| 综合 ① | 分组汇总表 + 柱状图 |
| 综合 ② | 分组统计的三条自检 |
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
        """gb = dev.groupby(GROUP_KEY)

# @@todo(1) 分别算出每组「行数」和「负荷值的有效个数」，存入 size_g 与 count_g
# @@hint gb.size() ；gb[VALUE_COL].count()
size_g = gb.size()
count_g = gb[VALUE_COL].count()
# @@end
# ---- 验收 ----
assert size_g.to_dict() == {"互感器": 58, "变压器": 55, "断路器": 41, "绝缘子": 57, "避雷器": 59}
assert count_g.to_dict() == {"互感器": 55, "变压器": 49, "断路器": 37, "绝缘子": 53, "避雷器": 55}
assert int(size_g.sum()) == 270, "size 之和必须等于总行数"
assert int(count_g.sum()) == 249, f"count 之和应为 249（270-21），实际 {int(count_g.sum())}"
assert int(size_g.sum() - count_g.sum()) == 21, "差值必须等于负荷值缺失数"
print("TODO(1) 通过 -> size 合计", int(size_g.sum()), "| count 合计", int(count_g.sum()),
      "| 差", int(size_g.sum() - count_g.sum()))"""
    ),
    code(
        """# @@todo(2) 算出每组负荷值的均值与中位数，存入 mean_g 与 median_g
# @@hint dev.groupby(GROUP_KEY)[VALUE_COL].mean() ；... .median()
mean_g = dev.groupby(GROUP_KEY)[VALUE_COL].mean()
median_g = dev.groupby(GROUP_KEY)[VALUE_COL].median()
# @@end
# ---- 验收 ----
assert abs(mean_g["变压器"] - 986.9231) < 1e-3, f"变压器组均值应为 986.9231，实际 {mean_g['变压器']:.4f}"
assert abs(median_g["变压器"] - 521.08) < 0.01, f"变压器组中位数应为 521.08，实际 {median_g['变压器']:.2f}"
ratio = mean_g["变压器"] / median_g["变压器"]
assert ratio > 1.5, f"均值/中位数应明显大于 1（实际 {ratio:.2f}），否则说明没看出异常值污染"
assert len(mean_g) == 5 and mean_g.index.name == GROUP_KEY
print(f"TODO(2) 通过 -> 变压器组 mean={mean_g['变压器']:.2f} median={median_g['变压器']:.2f} "
      f"比值={ratio:.2f}（哨兵污染信号）")"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(3) 用函数列表一次算出每组负荷值的 count/mean/median/std/max，存入 multi_stats
# @@hint dev.groupby(GROUP_KEY)[VALUE_COL].agg(["count","mean","median","std","max"])
multi_stats = dev.groupby(GROUP_KEY)[VALUE_COL].agg(["count", "mean", "median", "std", "max"])
# @@end
# ---- 验收 ----
assert multi_stats.shape == (5, 5), f"形状应为 (5, 5)，实际 {multi_stats.shape}"
assert list(multi_stats.columns) == ["count", "mean", "median", "std", "max"]
assert multi_stats.columns.nlevels == 1, "函数列表写法的列名应是单层"
assert multi_stats.loc["绝缘子", "count"] == 53
assert abs(multi_stats.loc["避雷器", "max"] - 103.81) < 0.01
print("TODO(3) 通过 -> 函数列表聚合", multi_stats.shape)
print(multi_stats.round(2).to_string())"""
    ),
    code(
        """# @@todo(4) 用字典写法：负荷值算 mean 与 max，湿度算 mean，存入 dict_stats
# @@hint dev.groupby(GROUP_KEY).agg({"负荷值": ["mean","max"], "湿度": "mean"})
dict_stats = dev.groupby(GROUP_KEY).agg({"负荷值": ["mean", "max"], "湿度": "mean"})
# @@end
# ---- 验收 ----
assert dict_stats.shape == (5, 3), f"形状应为 (5, 3)，实际 {dict_stats.shape}"
assert dict_stats.columns.nlevels == 2, "字典写法的列名应是 MultiIndex（这就是它的坑）"
assert ("负荷值", "mean") in dict_stats.columns
assert ("湿度", "mean") in dict_stats.columns
assert abs(dict_stats[("负荷值", "mean")]["变压器"] - 986.9231) < 1e-3
# 验证压平成单层列名的正确做法
flat = dict_stats.copy()
flat.columns = ["_".join(col) for col in flat.columns]
assert list(flat.columns) == ["负荷值_mean", "负荷值_max", "湿度_mean"]
print("TODO(4) 通过 -> 字典聚合", dict_stats.shape, "（列名是元组，需手动压平）")"""
    ),
    code(
        """# @@todo(5) 用命名聚合产出一张指标表 summary，包含 5 个指标：
#         样本数(size)、有效数(count)、平均负荷(mean)、中位负荷(median)、危急数(条件计数)
# @@hint gb.agg(样本数=("负荷值","size"), ..., 危急数=("缺陷等级", lambda s: int((s=="危急").sum())))
summary = dev.groupby(GROUP_KEY).agg(
    样本数=("负荷值", "size"),
    有效数=("负荷值", "count"),
    平均负荷=("负荷值", "mean"),
    中位负荷=("负荷值", "median"),
    危急数=("缺陷等级", lambda s: int((s == "危急").sum())),
)
# @@end
# ---- 验收 ----
assert list(summary.columns) == ["样本数", "有效数", "平均负荷", "中位负荷", "危急数"], \\
    f"列名不对：{list(summary.columns)}"
assert summary.columns.nlevels == 1, "命名聚合的列名应是单层"
assert summary.loc["互感器", "样本数"] == 58 and summary.loc["互感器", "有效数"] == 55
assert summary["危急数"].to_dict() == {"互感器": 5, "变压器": 5, "断路器": 4, "绝缘子": 4, "避雷器": 13}
assert abs(summary.loc["变压器", "平均负荷"] - 986.9231) < 1e-3
print("TODO(5) 通过 -> 命名聚合得到", list(summary.columns))
print(summary.round(2).to_string())"""
    ),
    code(
        """# @@todo(6) 用 as_index=False 重新产出 summary2，要求「设备类型」是普通列
# @@hint dev.groupby(GROUP_KEY, as_index=False).agg(样本数=("负荷值","size"), 平均负荷=("负荷值","mean"))
summary2 = dev.groupby(GROUP_KEY, as_index=False).agg(
    样本数=("负荷值", "size"),
    平均负荷=("负荷值", "mean"),
)
# @@end
# ---- 验收 ----
assert list(summary2.columns) == [GROUP_KEY, "样本数", "平均负荷"], f"列名不对：{list(summary2.columns)}"
assert summary2.shape == (5, 3), f"形状应为 (5, 3)，实际 {summary2.shape}"
assert summary2[GROUP_KEY].tolist()[0] in {"互感器", "变压器", "断路器", "绝缘子", "避雷器"}
assert not isinstance(summary2.index, pd.MultiIndex) and summary2.index.tolist() == [0, 1, 2, 3, 4], \\
    "as_index=False 后应是普通 RangeIndex"
print("TODO(6) 通过 -> 组键回到列里：", list(summary2.columns))"""
    ),
    code(
        """# @@todo(7) 按「台区编号 + 设备类型」双键分组求负荷均值，再 unstack 成交叉表，存入 cross
# @@hint dev.groupby(["台区编号", GROUP_KEY])[VALUE_COL].mean().unstack()
cross = dev.groupby(["台区编号", GROUP_KEY])[VALUE_COL].mean().unstack()
# @@end
# ---- 验收 ----
assert cross.shape == (18, 5), f"交叉表应为 (18, 5)，实际 {cross.shape}"
assert list(cross.columns) == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert list(cross.index.names) == ["台区编号"]
# 抽一格与手写筛选取值对账
cell = dev[(dev["台区编号"] == "STATION_A_01") & (dev[GROUP_KEY] == "断路器")][VALUE_COL].mean()
assert abs(cross.loc["STATION_A_01", "断路器"] - cell) < 1e-9, "交叉表取值与手写筛选不一致"
# STATION_A_01 没有互感器与变压器 -> 应为 NaN（该组合不存在）
assert pd.isna(cross.loc["STATION_A_01", "互感器"]), "不存在的组合应留 NaN，不是 0"
assert pd.isna(cross.loc["STATION_A_01", "变压器"])
print("TODO(7) 通过 -> 交叉表", cross.shape, "| A_01 行:", cross.loc["STATION_A_01"].round(1).to_dict())"""
    ),
    code(
        """# @@todo(8) 人为把前 5 行的缺陷等级置为缺失，然后分别用默认与 dropna=False 分组计数
#         存入 size_default 与 size_keepna
# @@hint 先 d = dev.copy() 并置 NaN；再 d.groupby("缺陷等级").size() 与 d.groupby("缺陷等级", dropna=False).size()
d_na = dev.copy()
d_na.loc[0:4, "缺陷等级"] = np.nan
size_default = d_na.groupby("缺陷等级").size()
size_keepna = d_na.groupby("缺陷等级", dropna=False).size()
# @@end
# ---- 验收 ----
assert size_default.to_dict() == {"一般": 161, "严重": 73, "危急": 31}, f"默认结果不对：{size_default.to_dict()}"
assert len(size_default) == 3
assert len(size_keepna) == 4, f"dropna=False 应有 4 组，实际 {len(size_keepna)}"
assert int(size_keepna.sum()) == 270, "dropna=False 的各组之和等于总行数"
assert int(size_default.sum()) == 265, f"默认丢掉了 5 行，实际求和 {int(size_default.sum())}"
assert int(size_keepna.sum() - size_default.sum()) == 5, "差值应正好是被丢掉的那 5 行"
print("TODO(8) 通过 -> 默认 3 组求和", int(size_default.sum()),
      "| dropna=False 4 组求和", int(size_keepna.sum()))"""
    ),
    code(
        """# @@todo(9) 验证 observed 的真实默认值：在「只有 2 类实际出现」的子集上分组
#         分别产出默认、observed=True、observed=False 三种结果
# @@hint 先构造 full_dtype 并 astype，再 groupby(key).size() / groupby(key, observed=True).size() / (..., observed=False).size()
key_full = sub_dev[GROUP_KEY].astype(full_dtype)
size_obs_default = sub_dev.groupby(key_full).size()
size_obs_true = sub_dev.groupby(key_full, observed=True).size()
size_obs_false = sub_dev.groupby(key_full, observed=False).size()
# @@end
# ---- 验收 ----
assert len(sub_dev) == 112, f"子集应有 112 行（变压器 55 + 绝缘子 57），实际 {len(sub_dev)}"
assert len(full_dtype.categories) == 5, "全量 category 应有 5 个类别"
assert len(size_obs_default) == 2, f"默认应为 2 组，实际 {len(size_obs_default)}"
assert len(size_obs_true) == 2
assert len(size_obs_false) == 5, f"observed=False 应产生 5 组（含 3 个空组），实际 {len(size_obs_false)}"
assert size_obs_default.to_dict() == size_obs_true.to_dict(), "默认值等价于 observed=True -> 说明 3.0 起默认已改为 True"
assert size_obs_false.to_dict() == {"互感器": 0, "变压器": 55, "断路器": 0, "绝缘子": 57, "避雷器": 0}
print("TODO(9) 通过 -> 默认", len(size_obs_default), "组 == observed=True；",
      "observed=False 才产生", len(size_obs_false), "组（3 个空组）")"""
    ),
    code(
        """# @@todo(10) 每组取负荷值最大的 1 条记录，存入 top1（先全局降序排序，再分组 head）
# @@hint dev.sort_values(VALUE_COL, ascending=False).groupby(GROUP_KEY).head(1)
top1 = dev.sort_values(VALUE_COL, ascending=False).groupby(GROUP_KEY).head(1)
# @@end
# ---- 验收 ----
assert top1.shape[0] == 5, f"每组 1 条，共 5 条，实际 {top1.shape[0]}"
assert len(top1[GROUP_KEY].unique()) == 5, "5 个组都要有代表"
worst = top1.set_index(GROUP_KEY)[VALUE_COL].to_dict()
assert worst["变压器"] == 9999.0 and worst["互感器"] == 9999.0 and worst["断路器"] == 9999.0, \\
    "未清洗时三个组的 top1 都是哨兵值 9999"
assert top1.set_index(GROUP_KEY)["记录ID"].to_dict() == {
    "互感器": 91, "变压器": 16, "断路器": 112, "绝缘子": 190, "避雷器": 191,
}
# 清洗掉哨兵后重取，极值才真实
clean_sorted = dev[dev[VALUE_COL].between(0, 1000)].sort_values(VALUE_COL, ascending=False)
top1_clean = clean_sorted.groupby(GROUP_KEY).head(1)
assert top1_clean.set_index(GROUP_KEY)[VALUE_COL].to_dict()["绝缘子"] == 118.01
print("TODO(10) 通过 -> 未清洗 top1:", worst,
      "\\n   清洗后 top1:", top1_clean.set_index(GROUP_KEY)[VALUE_COL].to_dict())"""
    ),
    md(
        """
---

## 进阶题（续）
"""
    ),
    code(
        """# @@todo(11) 用 pivot_table 做「设备类型 x 缺陷等级」的负荷均值交叉表，存入 pt
#         再单独取出「变压器 x 危急」这一格的值，存入 cell_manual（用手写筛选取均值）
# @@hint dev.pivot_table(index=GROUP_KEY, columns="缺陷等级", values=VALUE_COL, aggfunc="mean")
pt = dev.pivot_table(index=GROUP_KEY, columns="缺陷等级", values=VALUE_COL, aggfunc="mean")
cell_manual = dev[(dev[GROUP_KEY] == "变压器") & (dev["缺陷等级"] == "危急")][VALUE_COL].mean()
# @@end
# ---- 验收 ----
assert pt.shape == (5, 3), f"形状应为 (5, 3)，实际 {pt.shape}"
assert list(pt.columns) == ["一般", "严重", "危急"]
assert abs(pt.loc["变压器", "危急"] - 528.394) < 1e-3, f"该格应为 528.394，实际 {pt.loc['变压器', '危急']:.4f}"
assert abs(cell_manual - 528.394) < 1e-3
assert abs(pt.loc["变压器", "危急"] - cell_manual) < 1e-9, "pivot_table 与手写筛选必须一致"
# 与 groupby + unstack 写法完全等价
gt = dev.groupby([GROUP_KEY, "缺陷等级"])[VALUE_COL].mean().unstack()
assert pt.shape == gt.shape and bool(np.allclose(pt.to_numpy(), gt.to_numpy(), equal_nan=True)), \\
    "pivot_table 应等价于 groupby + unstack"
# 默认 aggfunc 是 mean：不写 aggfunc 时结果应与上面一致
pt_default = dev.pivot_table(index=GROUP_KEY, columns="缺陷等级", values=VALUE_COL)
assert bool(np.allclose(pt_default.to_numpy(), pt.to_numpy(), equal_nan=True)), \\
    "不写 aggfunc 时默认就是求均值（这就是那个暗坑）"
print("TODO(11) 通过 -> pivot_table", pt.shape, "| 变压器-危急 =", round(cell_manual, 4))"""
    ),
    code(
        """# @@todo(12) 用 transform 造三列特征：组均值、组内 z-score、组内百分位排名
#         存入 feat（一个含 记录ID / 设备类型 / 负荷值 / 组均值 / 组内z / 组内排名 的 DataFrame）
# @@hint feat = dev[["记录ID", GROUP_KEY, VALUE_COL]].copy() ；再依次赋三列，
#       均值和 std 用 .transform("mean") / .transform("std")，排名用 .rank(pct=True)
feat = dev[["记录ID", GROUP_KEY, VALUE_COL]].copy()
feat["组均值"] = feat.groupby(GROUP_KEY)[VALUE_COL].transform("mean")
feat["组内z"] = (feat[VALUE_COL] - feat["组均值"]) / feat.groupby(GROUP_KEY)[VALUE_COL].transform("std")
feat["组内排名"] = feat.groupby(GROUP_KEY)[VALUE_COL].rank(pct=True)
# @@end
# ---- 验收 ----
assert feat.shape == (270, 6), f"形状应为 (270, 6)，实际 {feat.shape}"
assert len(feat) == len(dev), "transform 保持行数不变，这是它与 agg 的根本区别"
assert list(feat.columns) == ["记录ID", GROUP_KEY, VALUE_COL, "组均值", "组内z", "组内排名"]
# 组均值应等于对应的组统计量
assert abs(feat.loc[0, "组均值"] - 355.4531) < 1e-3, f"第 0 行组均值应为 355.4531，实际 {feat.loc[0, '组均值']:.4f}"
for grp, sub in feat.groupby(GROUP_KEY):
    assert sub["组均值"].nunique() == 1, f"{grp} 组的组均值应该只有一个值"
# z-score 前 3 行（与讲解一致）
assert np.allclose(feat["组内z"].head(3).round(3).tolist(), [-0.179, -0.221, 0.218], atol=1e-3)
# 排名在 (0, 1] 区间
assert feat["组内排名"].min() > 0 and feat["组内排名"].max() == 1.0
# 组内排名的均值应约为 0.5
assert abs(feat.groupby(GROUP_KEY)["组内排名"].mean().mean() - 0.5) < 0.02
print("TODO(12) 通过 -> 造出 3 个组内特征，z 前 3 行:",
      feat["组内z"].head(3).round(3).tolist())"""
    ),
    md(
        """
---

## 综合题

### 综合 ① 分组汇总表 + 柱状图

产出一张**可以直接交卷**的分组汇总表，并且画出「各设备类型的平均负荷 vs 中位负荷」对比柱状图。

两个要求：

1. 表要满足：`设备类型` 是普通列、行数 = 组数、含 5 个指标
2. 图要能一眼看出"均值被异常值拉高"——这是本章的核心结论
"""
    ),
    code(
        """# @@todo 综合①：产出 report（as_index=False 的汇总表）与柱状图 ax
# @@hint groupby(GROUP_KEY, as_index=False).agg(...) ；再 report.set_index(GROUP_KEY)[["平均负荷","中位负荷"]].plot.bar(ax=ax)
report = dev.groupby(GROUP_KEY, as_index=False).agg(
    样本数=("负荷值", "size"),
    有效数=("负荷值", "count"),
    平均负荷=("负荷值", "mean"),
    中位负荷=("负荷值", "median"),
    危急数=("缺陷等级", lambda s: int((s == "危急").sum())),
)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
report.set_index(GROUP_KEY)[["平均负荷", "中位负荷"]].plot.bar(ax=ax)
ax.set_title("各设备类型的平均负荷 vs 中位负荷（均值被哨兵值拉高）")
ax.set_xlabel("设备类型")
ax.set_ylabel("负荷值")
plt.tight_layout()
plt.show()
# @@end
# ---- 验收 ----
assert list(report.columns) == [GROUP_KEY, "样本数", "有效数", "平均负荷", "中位负荷", "危急数"], \\
    f"列名不对：{list(report.columns)}"
assert report.shape == (5, 6), f"形状应为 (5, 6)，实际 {report.shape}"
assert report[GROUP_KEY].notna().all(), "设备类型必须是普通列"
# 自检 1：各组样本数之和 == 总行数
assert int(report["样本数"].sum()) == 270, "分组漏了行（检查 dropna）"
# 自检 2：组数 == 键的 nunique
assert len(report) == dev[GROUP_KEY].nunique(), "组数与 nunique 不一致（检查 observed）"
# 自检 3：有效数之和 == 非缺失数
assert int(report["有效数"].sum()) == int(dev[VALUE_COL].notna().sum()) == 249
# 均值/中位数比：被哨兵污染的组比值明显 > 1，没被污染的组比值 ≈ 1
ratio = report["平均负荷"] / report["中位负荷"]
assert int((ratio > 1).sum()) >= 4, f"多数组应满足 均值 > 中位数，实际只有 {int((ratio > 1).sum())} 组"
assert ratio[report[GROUP_KEY] == "变压器"].iloc[0] > 1.5, "变压器组污染最重，比值应 > 1.5"
assert ratio[report[GROUP_KEY] == "避雷器"].iloc[0] < 1.05, \
    "避雷器组最大值仅 103.81、没有哨兵，均值应≈中位数（比值≈1）"
assert abs(report.loc[report[GROUP_KEY] == "变压器", "平均负荷"].iloc[0] - 986.9231) < 1e-3
# 图
assert ax is not None and len(ax.patches) >= 8, f"柱子数不对：{len(ax.patches)}"
print("综合① 通过 ->")
print(report.round(2).to_string(index=False))"""
    ),
    md(
        """
### 综合 ② 分组统计的三条自检

把本章反复强调的三条自检写成一个可复用的检查结果：给定 `groupby` 结果与原始表，
输出三个布尔量。竞赛里这三行能省掉大量 debug 时间。
"""
    ),
    code(
        """# @@todo 综合②：对 report 与 gb 做三条自检，产出 checks 字典
# @@hint 三个键：行数守恒 / 组数一致 / 组键在列；再补一个「哨兵是否污染极值」的键
n_rows_ok = int(report["样本数"].sum()) == len(dev)
n_group_ok = len(report) == dev[GROUP_KEY].nunique()
key_is_col = GROUP_KEY in report.columns
sentinel_pollutes = bool(
    dev.sort_values(VALUE_COL, ascending=False).groupby(GROUP_KEY).head(1)[VALUE_COL].ge(1000).any()
)
checks = {
    "行数守恒": n_rows_ok,
    "组数一致": n_group_ok,
    "组键在列": key_is_col,
    "哨兵污染极值": sentinel_pollutes,
}
# @@end
# ---- 验收 ----
expect = {"行数守恒", "组数一致", "组键在列", "哨兵污染极值"}
assert set(checks) == expect, f"键不对。多 {set(checks) - expect}，缺 {expect - set(checks)}"
assert checks["行数守恒"] is True
assert checks["组数一致"] is True
assert checks["组键在列"] is True
assert checks["哨兵污染极值"] is True, "未清洗时组内 top1 应被 9999 污染，这正是要检测出来的"
assert all(isinstance(v, bool) for v in checks.values()), "四项都应是布尔量"
print("综合② 通过 -> 分组统计自检：")
for key, value in checks.items():
    print(f"   {key:10s} = {value}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（五题都能答上来才算过）：

1. `size()` 和 `count()` 差在哪？什么情况下两者相等、以至于你发现不了写错？
2. `agg` 四种写法里，只有一种能直接拿到可用的列名，是哪一种？为什么？
3. `groupby` 的 `dropna` 默认 `True` 会造成什么后果？写下那条能抓住它的自检。
4. 本机 pandas 3.0 的 `observed` 默认值是 `True` 还是 `False`？老教程为什么教反了？
5. `groupby().head(1)` 能不能拿到"负荷值最大的那条"？不能的话正确写法是什么？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习需要的脚手架（放在 HEADER 之后的独立 cell，练习版同样保留）
# =========================================================================== #

SCAFFOLD_SUB = """# 为 TODO(9) 准备：一个只有 2 类实际出现、但 category 记住全部 5 类的子集
full_dtype = pd.CategoricalDtype(categories=dev[GROUP_KEY].astype("category").cat.categories)
sub_dev = dev[dev[GROUP_KEY].isin(["变压器", "绝缘子"])]
print("full_dtype.categories:", list(full_dtype.categories))
print("sub_dev:", sub_dev.shape, "实际出现的类别:", list(sub_dev[GROUP_KEY].unique()))"""


def _inject_scaffold(cells: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """在第 3 个 cell（HEADER 之后、第一个题目之前）插入脚手架 cell。"""
    out = list(cells)
    out.insert(3, code(SCAFFOLD_SUB))
    return out


if __name__ == "__main__":
    stats = build(
        OUT,
        NAME,
        lesson=LESSON,
        exercise=_inject_scaffold(EXERCISE),
    )
    report([stats])
    print(f"输出目录：{OUT}")
