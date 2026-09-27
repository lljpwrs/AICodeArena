"""ch15 —— 多级索引：`MultiIndex` / `xs` / `IndexSlice` / `swaplevel`

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch15_multiindex.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch15_multiindex"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

dev = pd.read_csv(DATA / "device_defects.csv")

KEY = "台区编号"
CAT = "设备类型"
VAL = "负荷值"
IDX = pd.IndexSlice

print("pandas", pd.__version__)
print("dev", dev.shape, "| 台区", dev[KEY].nunique(), "| 类型", dev[CAT].nunique())'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


PAIR = ("STATION_A_01", "断路器")     # 有数据的组合（A_01×变压器 的负荷值全缺）
BAD_PAIR = ("STATION_A_01", "变压器")  # 全缺组合：均值 NaN、条数 0

mi = dev.set_index([KEY, CAT]).sort_index()          # 排好序的多级索引 (270, 11)
unsorted_mi = dev.set_index([KEY, CAT])              # 原始行序（未 lexsorted）
m = dev.groupby([KEY, CAT])[VAL].mean()              # groupby 两键 → MultiIndex Series

print("mi", mi.shape, "| 已排序:", mi.index.is_monotonic_increasing)
print("unsorted_mi 已排序:", unsorted_mi.index.is_monotonic_increasing)
print("m", m.shape)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch15 多级索引：`MultiIndex` / `xs` / `IndexSlice` / `swaplevel`

> 方向：数据预处理 ｜ 竞赛对应：**数据探索**（多维度交叉查询）+ **数据准备及处理**
> （`groupby` 多键聚合的产物就是 MultiIndex，绕不开）。

多级索引的坑集中在**取数**环节，全部是"要么报错、要么拿到不想要的东西"：

1. **切片前必须 lexsorted**——`set_index` 出来的原始顺序直接切片报 `UnsortedIndexError`
2. **`loc` 元组与 `IndexSlice` 语义不同**——一个是"精确点"，一个是"逐层切片"
3. **`xs` 截面会自动丢掉被取的那一层**——`drop_level` 默认 True
4. **`swaplevel` 之后不再有序**——要重新 `sort_index` 才能继续切片
5. **全缺组合的均值是 `NaN`、条数是 0**——不识别它，报表会整行填空

> 本章的 `assert` 全部可以在方向 README 的「ch15 专项真值」对上。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据探索 | 每项 2 分 | 多级交叉查询（台区 × 设备类型） |
| 数据准备及处理 | 占技能操作 10% | `groupby` 多键聚合、`set_index`、`xs` 截面 |

**学习目标**：拿到 `groupby([A, B])` 的产物，能用 `IndexSlice` / `xs` 取出任意
截面，并能解释为什么切片前必须排序。
"""
    ),
    md(
        """
## 二、API 速查表

| 方法 / 类 | 关键参数 | 返回 | 一句话说明 |
|---|---|---|---|
| `set_index([a, b])` | `drop` | DataFrame | 两列变两级索引 |
| `groupby([a, b])` | `as_index` | Series/DataFrame | 聚合产物自带 MultiIndex |
| `pd.IndexSlice` | — | 切片工具 | `loc[idx[:, "x"], :]` 逐层切 |
| `xs` | `level` `drop_level` | DataFrame/Series | **截面**：固定某层取值 |
| `swaplevel` | `i` `j` | DataFrame | 交换两层顺序（**不再有序**） |
| `droplevel(n)` | — | DataFrame | 直接扔掉某层 |
| `reset_index()` | — | DataFrame | 多级还原为普通列 |
| `.index.unique(n)` | — | Index | 取第 n 层的唯一值 |
| `.index.get_level_values(n)` | — | Index | 取第 n 层的逐行值 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 `groupby` 两键 → MultiIndex 是怎么来的

两个聚合键就是两级索引。`as_index=False` 则把键还原成普通列——
**要交叉查询就留 MultiIndex，要拼接报表就 `as_index=False` 或 `reset_index`。**
"""
    ),
    code(
        """print("m 的索引类型:", type(m.index).__name__, "| nlevels:", m.index.nlevels,
      "| names:", m.index.names)
print("len:", len(m), "= 台区组合数（18 台区 × 5 类型中实际出现的 88 个）")
print("第 0 层唯一值数:", len(m.index.unique(0)), "| 第 1 层:", len(m.index.unique(1)))
print("取一个点:", round(float(m.loc[PAIR]), 4))

m2 = dev.groupby([KEY, CAT], as_index=False)[VAL].mean()
print("as_index=False:", m2.shape, m2.columns.tolist())

assert type(m.index).__name__ == "MultiIndex" and m.index.nlevels == 2
assert m.index.names == [KEY, CAT]
assert len(m) == 88
assert round(float(m.loc[PAIR]), 4) == 171.24"""
    ),
    md(
        """
## 3.2 `set_index` 造多级索引

`set_index([KEY, CAT])` 把两列变成两级索引，行数不变（270）；
**原始行序不保证 lexsorted**——下一节就是它的报错现场。
"""
    ),
    code(
        """print("mi:", mi.shape, "| nlevels:", mi.index.nlevels)
print("第 0 层唯一:", len(mi.index.unique(0)), "| 第 1 层唯一:", len(mi.index.unique(1)))
print("sort 后已 lexsorted:", mi.index.is_monotonic_increasing)
print("未排序版:", unsorted_mi.index.is_monotonic_increasing, "← set_index 不排序")

assert mi.shape == (270, 11)
assert mi.index.nlevels == 2
assert bool(mi.index.is_monotonic_increasing)
assert not bool(unsorted_mi.index.is_monotonic_increasing)"""
    ),
    md(
        """
### 3.3 难点深挖：`UnsortedIndexError`——MultiIndex 切片前必须 lexsorted

**为什么难**：`set_index` 的产物保持**原始行序**，而 MultiIndex 的**范围切片**
（`:` 或元组区间）要求索引按字典序排好。没排序 → 直接
`UnsortedIndexError`；**单点 `loc` 不受影响**（只发 `PerformanceWarning`），
这种"点行、片不行"的分裂最迷惑人。

**错误示范**（`unsorted_mi` 是原始行序）：

```python
unsorted_mi.loc[(slice("STATION_A_01"), slice("断路器", "变压器")), :]
# UnsortedIndexError: 'MultiIndex slicing requires the index to be lexsorted:
#                      slicing on levels [0], positions [0]'
```

**正误对照**：

| 操作 | 未排序 | `sort_index()` 后 |
|---|---|---|
| 单点 `loc[(A_01, 断路器)]` | ✅ 3 行（带 PerformanceWarning） | ✅ 3 行 |
| 范围切片 `loc[idx[A:"C"], :]` | ❌ `UnsortedIndexError` | ✅ |

**判定规则**：**`set_index` 之后只要打算做范围切片，先 `sort_index()`；
把 `set_index(...).sort_index()` 当成一个原子动作背下来。**
"""
    ),
    code(
        """status1, msg1 = probe(
    lambda: unsorted_mi.loc[(slice("STATION_A_01"), slice("断路器", "变压器")), :])
print("未排序范围切片 →", status1, "|", str(msg1)[:60])

status2, msg2 = probe(lambda: unsorted_mi.loc[PAIR])
print("未排序单点 loc →", status2, "| 行数", len(msg2) if status2 == "ok" else msg2)

status3, msg3 = probe(lambda: unsorted_mi.loc[PAIR, VAL])
print("未排序单点 loc 取列 →", status3, "| 均值", round(float(msg3.mean()), 4))

assert status1 == "err" and msg1 == "UnsortedIndexError"
assert status2 == "ok" and len(msg2) == 3, "单点 loc 不需要排序"
assert round(float(unsorted_mi.loc[PAIR, VAL].mean()), 4) == 171.24"""
    ),
    md(
        """
## 3.4 `IndexSlice`：逐层切片的三板斧

`loc` 里每个位置要么是"该层的键"，要么是 `:`（全选）。多层混着切就必须
`pd.IndexSlice`（别名 `idx`）包起来。
"""
    ),
    code(
        """sub1 = mi.loc[IDX["STATION_A_01", :], :]
sub2 = mi.loc[IDX[:, "变压器"], :]
sub3 = mi.loc[IDX["STATION_A_01":"STATION_C_01", ["变压器", "断路器"]], VAL]

print("A_01 的全部行:", sub1.shape)
print("全部台区 × 变压器:", sub2.shape)
print("A~C 台区 × (变压器,断路器) 的负荷值:", sub3.shape,
      "| 均值:", round(float(sub3.mean()), 4))

assert sub1.shape == (8, 11)
assert sub2.shape == (55, 11)
assert len(sub3) == 36
assert round(float(sub3.mean()), 4) == 359.347"""
    ),
    md(
        """
## 3.5 `xs`：截面取数

`xs` 是"固定某层取值"的专用语法：**返回结果自动丢掉被固定的那一层**
（`drop_level=True` 默认）。双键 `xs` 直接给最细粒度。
"""
    ),
    code(
        """x1 = mi.xs("STATION_A_01", level=0)
x2 = mi.xs("变压器", level=1)
x4 = mi.xs("变压器", level=1, drop_level=False)
x3 = mi.xs(PAIR)

print("xs level=0:", x1.shape, "| 剩余索引名:", x1.index.names)
print("xs level=1:", x2.shape, "| 剩余索引名:", x2.index.name)
print("drop_level=False:", x4.index.names, "| 形状不变:", x4.shape)
print("xs 双键:", type(x3).__name__, "| 行数:", len(x3),
      "| 负荷值均值:", round(float(x3[VAL].mean()), 4))

assert x1.shape == (8, 11) and x1.index.names == [CAT]
assert x2.shape == (55, 11) and x2.index.name == KEY
assert x4.shape == (55, 11) and x4.index.names == [KEY, CAT]
assert len(x3) == 3 and round(float(x3[VAL].mean()), 4) == 171.24"""
    ),
    md(
        """
## 3.6 `swaplevel`：换层之后必须重新排序

换层是纯重排标签，**不排序**；换完还想切片就得再 `sort_index()`。
换层的意义：让"设备类型"变第 0 层后，就能按类型先切。
"""
    ),
    code(
        """sw = mi.swaplevel()
sw2 = sw.sort_index()

print("swap 后索引名:", sw.index.names, "| 已排序:", sw.index.is_monotonic_increasing)
print("swap+sort:", sw2.index.is_monotonic_increasing)
print("swap 后按第 0 层取变压器:", sw2.loc["变压器"].shape,
      "≈ mi.xs('变压器', level=1)", mi.xs("变压器", level=1).shape)

assert sw.index.names == [CAT, KEY]
assert not bool(sw.index.is_monotonic_increasing)
assert bool(sw2.index.is_monotonic_increasing)
assert len(sw2.loc["变压器"]) == 55"""
    ),
    md(
        """
## 3.7 交付口径：`droplevel` 与 `reset_index`

MultiIndex 是**工作形态**，交付报表前一般要还原：扔掉某层用 `droplevel`，
全部还原成列用 `reset_index`（练习版验收表、写入 Excel 前的标准动作）。
"""
    ),
    code(
        """d1 = mi.droplevel(1)
r = mi.reset_index()

print("droplevel(1):", d1.shape, "| 索引名:", d1.index.name,
      "| 唯一台区:", d1.index.nunique())
print("reset_index:", r.shape, "| 前两列:", r.columns.tolist()[:2])

assert d1.shape == (270, 11) and d1.index.name == KEY
assert d1.index.nunique() == 18
assert r.shape == (270, 13)
assert r.columns.tolist()[:2] == [KEY, CAT]"""
    ),
    md(
        """
### 3.8 难点深挖：全缺组合——均值 NaN、条数 0

**为什么难**：`STATION_A_01 × 变压器` 只有 1 行，且负荷值全缺。
聚合结果：均值 `NaN`、条数 `0`——**行还在、值全空**。不识别这种组合，
交叉报表会出现一整行"看起来该有值却是空"的记录。

**错误示范**：

```python
g.loc[("STATION_A_01", "变压器"), "均值"]   # nan  ← 不是没有这行，是值全缺
```

**正误对照**：

| 检查 | 结果 | 含义 |
|---|---|---|
| `g.loc[BAD_PAIR, "条数"]` | **0** | 组合存在但无有效值 |
| `BAD_PAIR in g.index` | True | 行本身在（88 组合里） |
| `组合 not in m.index` | — | ch14 的另一类：组合**根本不存在**（2 个） |

**判定规则**：**交叉报表先数三类格子——有值 / 组合在但全缺（均值 NaN + 条数 0）/
组合不存在（索引里就没有）。** 与 ch14 的"宽表 NaN 两张面孔"是同一件事的
两个视角。
"""
    ),
    code(
        """g = dev.groupby([KEY, CAT]).agg(均值=(VAL, "mean"), 条数=(VAL, "count"))

print("g:", g.shape, g.columns.tolist())
print("全缺组合:", BAD_PAIR, "| 均值:", float(g.loc[BAD_PAIR, "均值"]),
      "| 条数:", int(g.loc[BAD_PAIR, "条数"]))
print("组合在索引里:", BAD_PAIR in g.index)
print("正常组合:", PAIR, "| 均值:", round(float(g.loc[PAIR, "均值"]), 4),
      "| 条数:", int(g.loc[PAIR, "条数"]))

assert g.shape == (88, 2) and g.columns.tolist() == ["均值", "条数"]
assert bool(np.isnan(g.loc[BAD_PAIR, "均值"]))
assert int(g.loc[BAD_PAIR, "条数"]) == 0
assert BAD_PAIR in g.index
assert round(float(g.loc[PAIR, "均值"]), 4) == 171.24 and int(g.loc[PAIR, "条数"]) == 3"""
    ),
    md(
        """
## 3.9 综合题：聚合 → unstack 交叉表（衔接 ch14）

MultiIndex 的终点经常是 `unstack` 成宽表——多级索引 Series 一转就是
"台区 × 设备类型" 交叉表，与 ch14 的 `pivot_table` 结果对得上。
"""
    ),
    code(
        """w = m.unstack(CAT)

print("unstack 后:", w.shape, "| 列:", w.columns.tolist())
print("与 ch14 的 pivot_table 对齐（非空格子均值应一致）:",
      round(float(w.loc["STATION_A_01", "断路器"]), 4))

assert w.shape == (18, 5)
assert w.columns.tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert round(float(w.loc["STATION_A_01", "断路器"]), 4) == 171.24"""
    ),
    md(
        """
---

## 四、易错点清单

1. `set_index` 不排序——范围切片前必须 `sort_index()`，否则 `UnsortedIndexError`。
2. 单点 `loc` 未排序也能用（只警告）——"点行片不行"最容易让人误判索引是好的。
3. `loc` 元组是精确点，`IndexSlice` 才能逐层切；混用层级切片必须包 `idx`。
4. `xs` 默认 `drop_level=True`——取截面后少一层是**设计如此**。
5. `swaplevel` 后不再有序，切片前要重新 `sort_index()`。
6. 全缺组合：行在索引里、均值 NaN、条数 0；组合不存在则索引里根本没有。
7. `as_index=False` 与 `reset_index()` 都能把键还原成列，交付前二选一。
8. `unstack` 的列顺序是码位序（互感器在前），与业务顺序无关。

## 五、本章小结

- MultiIndex 两个来源：`groupby([a, b])` 与 `set_index([a, b])`，行为一致。
- 取数三板斧：单点 `loc[(a, b)]`、逐层 `loc[idx[:, b], :]`、截面 `xs(b, level=)`。
- 一切范围切片的前提是 lexsorted——`set_index(...).sort_index()` 是原子动作。
- 交付前 `reset_index()`；交叉报表前先分清"有值 / 全缺 / 不存在"三类格子。

### 复盘提问

1. 为什么 `set_index` 之后直接切片会报 `UnsortedIndexError`，而单点 `loc` 不会？
2. `xs` 与 `loc[idx[:, "变压器"], :]` 的结果有什么差别？
3. `swaplevel` 之后必须做什么才能继续切片？
4. 全缺组合和"组合不存在"怎么区分？各对应 ch14 的哪个概念？
5. 交付 Excel 报表前，多级索引一般怎么处理？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch15 练习：多级索引

与讲解版逐 Cell 对应，`assert` 验收保留。卡住回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 数据探索 | 多级交叉查询 |
| 数据准备及处理 | `groupby` 多键聚合 + `xs` 截面 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：groupby 两键的产物（对应讲解 §3.1）"),
    code(
        """# @@todo(1) groupby 两键算负荷均值已给出（m）；请打印它的 nlevels、names、
#            并用 .loc 取 PAIR 点的值存入 v_pair
# @@hint m.index.nlevels ；m.loc[PAIR]
print("nlevels:", m.index.nlevels, "| names:", m.index.names)
v_pair = m.loc[PAIR]
# @@end

print("PAIR 均值:", round(float(v_pair), 4))
assert m.index.nlevels == 2 and m.index.names == [KEY, CAT]
assert len(m) == 88
assert round(float(v_pair), 4) == 171.24"""
    ),
    md("## 题 2：as_index=False 的平面表（对应讲解 §3.1）"),
    code(
        """# @@todo(2) groupby 两键、as_index=False、负荷均值，存入 m2
m2 = dev.groupby([KEY, CAT], as_index=False)[VAL].mean()
# @@end

print(m2.shape, m2.columns.tolist())
assert m2.shape == (88, 3)
assert m2.columns.tolist() == [KEY, CAT, VAL]"""
    ),
    md("## 题 3：UnsortedIndexError 现场（对应讲解 §3.3）"),
    code(
        """# @@todo(3) 用 probe 测未排序的 unsorted_mi 做范围切片（结果存入 status, msg）
# @@hint unsorted_mi.loc[(slice("STATION_A_01"), slice("断路器", "变压器")), :]
status, msg = probe(
    lambda: unsorted_mi.loc[(slice("STATION_A_01"), slice("断路器", "变压器")), :])
# @@end

# @@todo(4) 未排序单点 loc[PAIR] 取负荷值列，均值存入 mean_unsorted
mean_unsorted = unsorted_mi.loc[PAIR, VAL]
# @@end

print("未排序范围切片 →", status)
print("未排序单点均值:", round(float(mean_unsorted.mean()), 4))
assert status == "err" and msg == "UnsortedIndexError"
assert round(float(mean_unsorted.mean()), 4) == 171.24"""
    ),
    md("## 题 4：IndexSlice 三板斧（对应讲解 §3.4）"),
    code(
        """# @@todo(5) 用 IDX 取「全部台区 × 变压器」的行，存入 sub2
# @@hint mi.loc[IDX[:, "变压器"], :]
sub2 = mi.loc[IDX[:, "变压器"], :]
# @@end

# @@todo(6) 取 A_01~C_01 台区、(变压器, 断路器) 两种类型的负荷值，存入 sub3
# @@hint IDX["STATION_A_01":"STATION_C_01", ["变压器", "断路器"]]
sub3 = mi.loc[IDX["STATION_A_01":"STATION_C_01", ["变压器", "断路器"]], VAL]
# @@end

print("全部×变压器:", sub2.shape, "| A~C×两类:", sub3.shape)
assert sub2.shape == (55, 11)
assert len(sub3) == 36
assert round(float(sub3.mean()), 4) == 359.347"""
    ),
    md("## 题 5：xs 截面（对应讲解 §3.5）"),
    code(
        """# @@todo(7) xs 取第 1 层「变压器」的截面（默认 drop_level），存入 x2；
#            再取 drop_level=False 版本存入 x4
# @@hint mi.xs("变压器", level=1) ；drop_level 参数
x2 = mi.xs("变压器", level=1)
x4 = mi.xs("变压器", level=1, drop_level=False)
# @@end

print("x2:", x2.shape, x2.index.name, "| x4:", x4.shape, x4.index.names)
assert x2.shape == (55, 11) and x2.index.name == KEY
assert x4.shape == (55, 11) and x4.index.names == [KEY, CAT]"""
    ),
    md("## 题 6：swaplevel 与重排（对应讲解 §3.6）"),
    code(
        """# @@todo(8) 对 mi 换层存入 sw；再排序存入 sw2；按新第 0 层取「变压器」存入 by_cat
sw = mi.swaplevel()
sw2 = sw.sort_index()
by_cat = sw2.loc["变压器"]
# @@end

print("swap names:", sw.index.names, "| 已排序:", sw.index.is_monotonic_increasing)
print("排序后取变压器:", by_cat.shape)
assert sw.index.names == [CAT, KEY]
assert not bool(sw.index.is_monotonic_increasing)
assert bool(sw2.index.is_monotonic_increasing)
assert len(by_cat) == 55"""
    ),
    md("## 题 7：交付口径（对应讲解 §3.7）"),
    code(
        """# @@todo(9) 对 mi 丢掉第 1 层存入 d1；全部还原成列存入 r
d1 = mi.droplevel(1)
r = mi.reset_index()
# @@end

print("droplevel:", d1.shape, d1.index.name, "| reset:", r.shape)
assert d1.shape == (270, 11) and d1.index.name == KEY
assert r.shape == (270, 13) and r.columns.tolist()[:2] == [KEY, CAT]"""
    ),
    md("## 题 8（综合）：全缺组合识别 + unstack（对应讲解 §3.8-3.9）"),
    code(
        """# @@todo(10) 两键命名聚合（均值/条数）存入 g；判断 BAD_PAIR 的均值是否为 NaN
#            存入 bad_is_nan、条数存入 bad_count；最后 m.unstack(CAT) 存入 w
# @@hint g.agg(均值=(VAL, "mean"), 条数=(VAL, "count")) ；pd.isna(...)
g = dev.groupby([KEY, CAT]).agg(均值=(VAL, "mean"), 条数=(VAL, "count"))
bad_is_nan = bool(pd.isna(g.loc[BAD_PAIR, "均值"]))
bad_count = int(g.loc[BAD_PAIR, "条数"])
w = m.unstack(CAT)
# @@end

print("g:", g.shape, "| 全缺组合 NaN:", bad_is_nan, "| 条数:", bad_count)
print("unstack:", w.shape)
assert g.shape == (88, 2) and g.columns.tolist() == ["均值", "条数"]
assert bool(bad_is_nan) and int(bad_count) == 0
assert w.shape == (18, 5)
assert round(float(w.loc["STATION_A_01", "断路器"]), 4) == 171.24"""
    ),
    md(
        """
---

## 综合自查

1. 题 3 里"单点能用、切片报错"的根因是什么？
2. 题 5 的 x2 和 x4 差在哪一层？什么时候要 `drop_level=False`？
3. 题 6 换层后为什么必须重新排序？
4. 题 8 的全缺组合与"组合不存在"怎么区分？

全部答得上来，进入 ch16（时间序列）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
