"""ch09 —— 分箱与离散化：`cut` / `qcut` / `Categorical` / 分位分箱

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch09_bin_discretize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch09_bin_discretize"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

dev = pd.read_csv(DATA / "device_defects.csv")
load = pd.read_csv(DATA / "load_curve.csv")

VAL = "负荷值"
KEY = "台区编号"
CAT = "设备类型"
s = dev[VAL]

print("pandas", pd.__version__)
print("dev", dev.shape, "| 负荷值 NaN", int(s.isna().sum()),
      "| min", s.min(), "| max", s.max())'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    """
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


# 演示 right 语义用的合成序列：正好踩在边界 100 / 300 / 500 上
DEMO = pd.Series([100, 200, 300, 400, 500], name="演示")
EDGES = [-10, 100, 300, 600, 10000]
LABELS = ["低", "中", "高", "极端"]

print("DEMO", DEMO.tolist(), "| 边界", EDGES)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch09 分箱与离散化：`cut` / `qcut` / `Categorical`

> 方向：数据预处理 ｜ 竞赛对应：**数据探索**（分布分层看）+ **数据准备及处理**
> （连续变量离散化是特征工程常规操作）+ **数据清洗**（按档位定位异常）。

分箱的核心矛盾只有一个：**边界归谁、越界去哪**。这一章的五个坑全部围绕它：

1. **`cut` 默认 `right=True` 左开右闭**——正好压线的值归右边那箱
2. **越界值不报错，静默变 `NaN`**——`20000` 进不了任何箱
3. **等宽分箱被离群值毁掉**——`9999` 一来，5 个箱空了 3 个
4. **`qcut` 遇到重复分位数**——不加 `duplicates="drop"` 直接报错
5. **分箱结果是 ordered Categorical**——能比大小、NaN 比较恒为 False

> 本章的 `assert` 全部可以在方向 README 的「ch09 专项真值」对上。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据探索 | 每项 2 分 | `value_counts(bins=)` 看分布、分箱计数表 |
| 数据准备及处理 | 占技能操作 10% | `cut` / `qcut` 离散化、`groupby(分箱)` 分层统计 |
| 数据清洗 | 每项 2 分 | 按档位（高负荷 / 极端值）圈出待复核行 |

**学习目标**：给定一列连续值，能 30 秒内选出 cut / qcut / 自定义边界三种方案之一，
说清边界值归哪箱、NaN 去哪、越界值去哪。
"""
    ),
    md(
        """
## 二、API 速查表

| 方法 | 关键参数 | 返回 | 一句话说明 |
|---|---|---|---|
| `pd.cut` | `bins` `labels` `right` `include_lowest` | Categorical | **等宽**或自定义边界分箱 |
| `pd.qcut` | `q` `labels` `duplicates` | Categorical | **分位**分箱，每箱计数近似相等 |
| `value_counts(bins=n)` | — | Series | 直接分箱计数，区间索引 |
| `retbins=True` | — | (Categorical, ndarray) | 顺便拿回实际边界 |
| `.cat.codes` | — | Series | 类别 → 整数编码 |
| `.cat.ordered` | — | bool | 分箱结果恒为 True |
| `groupby(分箱列)` | `observed` | DataFrameGroupBy | 按档位分层统计 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 `pd.cut(s, 5)`：等宽分箱

给整数 `bins=5`，pandas 取 `min` 到 `max` 均分 5 段。本数据 min=-1、max=9999，
段宽约 2000——结果 5 个箱空了 2 个：**离群值把等宽箱撑爆了**。
"""
    ),
    code(
        """b5 = pd.cut(s, 5)

print("箱数:", len(b5.cat.categories))
for iv in b5.cat.categories:
    print("  ", iv, "| 计数", int((b5 == iv).sum()))
print("总计数:", b5.value_counts().sort_index().tolist())
print("NaN 保留:", int(b5.isna().sum()), "← 原 NaN 原样穿透")
print("cat.ordered:", b5.cat.ordered, "| codes 前 5:", b5.cat.codes[:5].tolist())

assert len(b5.cat.categories) == 5
assert b5.value_counts().sort_index().tolist() == [242, 0, 3, 0, 4], \\
    "离群值撑爆等宽箱：两箱全空"
assert int(b5.isna().sum()) == 21, "缺失值分箱后仍是缺失"
assert b5.cat.ordered is True"""
    ),
    md(
        """
### 3.2 难点深挖：`right` 语义与越界值——边界归谁、越界去哪

**为什么难**：`cut` 的区间是**数学写法 `(a, b]`**——左开右闭。恰好等于边界的值
归**右边**那箱；而**落在所有箱之外的值不报错**，静默变成 `NaN`。

**错误示范**（贴实际输出，`DEMO = [100, 200, 300, 400, 500]`，边界 `[100, 300, 500]`）：

```python
pd.cut(DEMO, bins=[100, 300, 500], labels=["a", "b"]).tolist()
# [nan, 'a', 'a', 'b', 'b']   ← 100 本以为在最左箱，实际是 NaN！
```

**正误对照**：

| 写法 | 输出 | 语义 |
|---|---|---|
| `right=True`（默认） | `[nan, 'a', 'a', 'b', 'b']` | `(100,300]` `(300,500]`，**左端点 100 出界** |
| `right=False` | `['a', 'a', 'b', 'b', nan]` | `[100,300)` `[300,500)`，**右端点 500 出界** |

**判定规则**：**先想清楚业务上"压线算谁"（负荷 300 算不算高负荷？），再选
`right`；边界值必须被覆盖时，把边界外扩一点，或用 `include_lowest=True` 保住
最小值。** 记住：`cut` 对越界值**永远静默给 NaN**，绝不报错。
"""
    ),
    code(
        """r = pd.cut(DEMO, bins=[100, 300, 500], labels=["a", "b"])
rl = pd.cut(DEMO, bins=[100, 300, 500], labels=["a", "b"], right=False)
print("right=True :", r.tolist())
print("right=False:", rl.tolist())

out = pd.cut(pd.Series([50, 20000]), bins=[100, 300, 500])
print("50 和 20000 都在箱外 →", out.tolist(), "← 全是 NaN，不报错")

# include_lowest：把最小值并进第一箱（边界从 (a,b] 变成 [a,b]）
b5 = pd.cut(s, 5)
b5l = pd.cut(s, 5, include_lowest=True)
print("默认第一箱:", b5.cat.categories[0], "| include_lowest:", b5l.cat.categories[0])
print("两版计数相同:", b5.value_counts().tolist() == b5l.value_counts().tolist(),
      "（本数据 min=-1 远离边界，恰好没差别）")

assert r.tolist() == [np.nan, "a", "a", "b", "b"]
assert rl.tolist() == ["a", "a", "b", "b", np.nan]
assert bool(out.isna().all()), "越界值静默变 NaN"
assert str(b5.cat.categories[0]) == "(-11.0, 1999.0]"
assert str(b5l.cat.categories[0]) == "(-11.001, 1999.0]", \\
    "include_lowest 把下界外扩 0.1% 并改为闭区间\""""
    ),
    md(
        """
## 3.3 自定义边界 + `labels`：业务说了算的档位

等宽箱被 9999 毁掉之后，正确做法是**业务定边界**：`[-10, 100, 300, 600, 10000]`
切出 低 / 中 / 高 / 极端 四档。`labels` 直接给业务名，`labels=False` 给整数编码。
"""
    ),
    code(
        """bin_col = pd.cut(s, bins=EDGES, labels=LABELS)

print("计数:", bin_col.value_counts().to_dict())
print("NaN:", int(bin_col.isna().sum()))
print("整数编码版:", pd.cut(s, bins=EDGES, labels=False).value_counts().sort_index().to_dict())

# retbins 顺便拿回实际边界（整数 bins 时很有用）
b5, edges = pd.cut(s, 5, retbins=True)
print("bins=5 的实际边界:", np.round(edges, 3).tolist())

assert bin_col.value_counts().to_dict() == {"低": 141, "中": 56, "高": 45, "极端": 7}
assert int(bin_col.isna().sum()) == 21
assert np.round(edges, 3).tolist()[:2] == [-11.0, 1999.0]"""
    ),
    md(
        """
### 3.4 难点深挖：`qcut` 分位分箱与 `duplicates`

**为什么难**：`qcut` 按分位数切箱，**每箱计数近似相等**——这是它相对 `cut` 的
核心价值（对偏态分布稳健）。但两个坑：

1. 分位数边界**重复**时（大量相同值堆积），不加 `duplicates="drop"` 直接
   `ValueError`；
2. NaN 不参与分位计算但**原样穿透**到结果。

**正误对照**（本数据 qcut(4)，249 个非缺失分 4 箱）：

| 写法 | 结果 |
|---|---|
| `pd.qcut(s, 4)` | 4 箱计数 **[63, 62, 62, 62]**，边界 48.31 / 87.34 / 192.34 |
| `pd.qcut(s, 10, duplicates="drop")` | 10 箱，计数 **[25,25,25,25,25,24,25,26,24,25]** |
| 遇到大量重复值不写 `duplicates` | `ValueError: Bin edges must be unique` |

**判定规则**：**看分布选工具——近似均匀 / 离群值小 → `cut` 等宽；
偏态 / 有离群值 → `qcut`（或业务定边界）；边界重复报错就 `duplicates="drop"`
并检查箱数是否变少。** 口诀：**cut 均宽，qcut 均人。**
"""
    ),
    code(
        """q4 = pd.qcut(s, 4)
q10 = pd.qcut(s, 10, duplicates="drop")

print("qcut(4) 边界:", [round(iv.right, 2) for iv in q4.cat.categories])
print("qcut(4) 计数:", q4.value_counts().sort_index().tolist(),
      "| NaN 穿透:", int(q4.isna().sum()))
print("qcut(10, drop) 箱数:", len(q10.cat.categories),
      "| 计数:", q10.value_counts().sort_index().tolist())

status, msg = probe(pd.qcut, pd.Series([1, 1, 1, 1, 1, 1, 2, 3]), 4)
print("重复分位数不 drop →", status, "|", msg)

assert q4.value_counts().sort_index().tolist() == [63, 62, 62, 62]
assert [round(iv.right, 2) for iv in q4.cat.categories] == [48.31, 87.34, 192.34, 9999.0]
assert int(q4.isna().sum()) == 21
assert len(q10.cat.categories) == 10
assert q10.value_counts().sort_index().tolist() == [25, 25, 25, 25, 25, 24, 25, 26, 24, 25]
assert status == "err" and msg == "ValueError"
"""
    ),
    md(
        """
## 3.5 `value_counts(bins=n)`：一行看分布

不用先 `cut` 再数，`value_counts` 自带分箱参数，返回**区间索引**的 Series。
区间写法与 `cut` 相同（左开右闭），计数近似——适合探索阶段的快速体检。
"""
    ),
    code(
        """vc = s.value_counts(bins=5)

print("索引类型:", type(vc.index).__name__)
print("计数（按区间排序）:", vc.sort_index().tolist())
print("-1 落在:", [str(i) for i in vc.index if i.left <= -1 <= i.right])
print("9999 落在:", [str(i) for i in vc.index if i.left <= 9999 <= i.right])
print("注意：value_counts 默认按计数降序，索引本身不单调:", vc.index.is_monotonic_increasing)

assert type(vc.index).__name__ == "IntervalIndex"
assert vc.sort_index().tolist() == [242, 0, 3, 0, 4], "与 cut(s, 5) 计数一致"
assert vc.sort_index().index.is_monotonic_increasing, "排序后区间单调"
"""
    ),
    md(
        """
## 3.6 `groupby(分箱列)`：按档位分层统计

把分箱结果作为列，再 `groupby`——分层统计的标配组合。
`observed=False` 会把**空箱也列出来**（pandas 3.0 默认 `observed=True`），
本数据四档都有值，两种写法结果一致；但**类别多、数据稀时差异巨大**。
"""
    ),
    code(
        """dev["负荷档"] = pd.cut(s, bins=EDGES, labels=LABELS)

g = dev.groupby("负荷档", observed=False)[VAL].agg(["count", "mean"])
print(g.round(2))
print("\\n各档占比(%):", (dev["负荷档"].value_counts(normalize=True).sort_index() * 100)
      .round(2).to_dict())
print("observed=True 温度均值:", dev.groupby("负荷档", observed=True)["温度"]
      .mean().round(3).to_dict())

assert g["count"].tolist() == [141, 56, 45, 7]
assert [round(v, 2) for v in g["mean"]] == [52.09, 157.83, 519.16, 7856.57]
assert round(dev["负荷档"].value_counts(normalize=True)["低"], 4) == 0.5663"""
    ),
    md(
        """
### 3.7 难点深挖：分箱结果是 ordered Categorical——能比大小，NaN 比较恒 False

**为什么难**：`cut` 的输出不是字符串列，是 **ordered Categorical**。三个后果：

1. 可以直接比较：`负荷档 > "中"`（按 labels 顺序！不是码位序）；
2. **NaN 参与比较返回 False**——过滤"高负荷"时不会误伤缺失行，但想找
   "高负荷**或**缺失"得显式写；
3. `.value_counts()` 默认**只数出现过的类别**，想看空箱要 `observed=False`
   或 `dropna=False`（视方法而定）。

**错误示范**：

```python
(dev["负荷档"] > "中").sum()          # 52 —— 缺失行不会混进来，结果干净
dev.loc[dev["负荷档"] > "中", VAL].isna().sum()   # 0 —— NaN 行被静默排除
```

**正误对照**：

| 需求 | 写法 | 结果 |
|---|---|---|
| 高负荷行（不含缺失） | `dev["负荷档"] > "中"` | 52 行 |
| 高负荷或缺失行 | `(dev["负荷档"] > "中") \\| dev["负荷档"].isna()` | 52 + 21 = 73 行 |

**判定规则**：**用分箱列过滤前先问一句"NaN 行算不算"——算就显式 `| isna()`，
不算就放心用比较（NaN 恒 False，天然安全）。**
"""
    ),
    code(
        """hi = dev["负荷档"] > "中"
hi_or_na = hi | dev["负荷档"].isna()
print("档 > '中':", int(hi.sum()), "行 | 其中含 NaN 负荷值的:", int(dev.loc[hi, VAL].isna().sum()))
print("高或缺失:", int(hi_or_na.sum()), "行 = 52 + 21")

print("档列 dtype:", dev["负荷档"].dtype)
print("codes 与档位:", dev["负荷档"].cat.codes.groupby(dev["负荷档"], observed=True).first().to_dict())

assert int(hi.sum()) == 52
assert int(hi_or_na.sum()) == 73
assert int(dev.loc[hi, VAL].isna().sum()) == 0, "NaN 比较恒 False，天然过滤"
assert str(dev["负荷档"].dtype) == "category\""""
    ),
    md(
        """
## 3.8 综合题：负荷曲线的四分位档与台区分档

时序数据也常用分箱：`load_curve` 按天聚合出 90 个日均负荷，`qcut` 切 Q1~Q4
四分位档；每台区的平均负荷切 小 / 中 / 大 三档。
"""
    ),
    code(
        """lc = load.copy()
lc["时间戳"] = pd.to_datetime(lc["时间戳"])
daily = lc.groupby(lc["时间戳"].dt.date)[VAL].mean()
print("日均负荷 n:", len(daily), "| min", round(daily.min(), 2),
      "| max", round(daily.max(), 2))

q_label = pd.qcut(daily, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("四分位档计数:", q_label.value_counts().to_dict())

by_station = lc.groupby(KEY)[VAL].mean()
print("台区数:", len(by_station), "| 均值范围",
      round(by_station.min(), 2), "~", round(by_station.max(), 2))
print("台区分档:", pd.qcut(by_station, 3, labels=["小", "中", "大"]).value_counts().to_dict())

assert len(daily) == 90
assert round(daily.min(), 2) == 424.68 and round(daily.max(), 2) == 778.21
assert q_label.value_counts().to_dict() == {"Q1": 23, "Q2": 22, "Q3": 22, "Q4": 23}
assert pd.qcut(by_station, 3, labels=["小", "中", "大"]).value_counts().to_dict() \\
    == {"小": 1, "中": 1, "大": 1}"""
    ),
    md(
        """
---

## 四、易错点清单

1. `cut` 默认 `right=True` 左开右闭——压线值归右箱，`right=False` 反转。
2. 越界值**静默变 NaN**，不报错；`include_lowest=True` 只救最小值。
3. 等宽分箱会被离群值撑爆（本数据 5 箱空 2 箱）——有离群值改用 `qcut` 或业务边界。
4. `qcut` 分位边界重复直接 `ValueError`，用 `duplicates="drop"` 并检查箱数。
5. NaN 分箱后仍是 NaN（cut / qcut 都一样），计数时别漏。
6. `value_counts(bins=n)` 与 `cut(s, n)` 的计数一致，但前者区间下界带 0.1% 外扩。
7. 分箱列是 ordered Categorical：`>` `<` 按 labels 顺序比较，NaN 比较恒 False。
8. `groupby(分箱列)` 记得想 `observed`——空箱要不要列出来。
9. `labels=False` 给整数编码（可喂给模型），`retbins=True` 拿回实际边界。
10. "cut 均宽，qcut 均人"——先看分布再选工具。

## 五、本章小结

- 边界语义：`(a, b]` 左开右闭，`right=False` 变 `[a, b)`。
- 越界与缺失：越界 → NaN（静默），缺失 → NaN（穿透）。
- 选型：均匀/干净 → `cut`；偏态/有离群值 → `qcut`；业务有标准 → 自定义边界 + labels。
- 分箱列的比较过滤对 NaN 天然安全，但"缺失也算高负荷"的需求要显式补 `| isna()`。

### 复盘提问

1. `pd.cut([100,200,300], [100,300])` 的输出是什么？为什么 100 变 NaN？
2. 什么时候等宽分箱完全失效？失效的数字特征是什么？
3. `qcut` 报 "Bin edges must be unique" 怎么救？救完要检查什么？
4. 分箱列和普通字符串列比较过滤的差异是什么？
5. `value_counts(bins=5)` 和 `cut(s, 5)` 的区间下界有什么细微差别？

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
# ch09 练习：分箱与离散化

与讲解版逐 Cell 对应，`assert` 验收保留。卡住回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 数据探索 | `value_counts(bins=)` 快速体检 |
| 数据准备及处理 | `cut` / `qcut` 离散化 + 分层统计 |
| 数据清洗 | 按档位圈异常行 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：等宽分箱体检（对应讲解 §3.1）"),
    code(
        """# @@todo(1) 对负荷值 s 做 5 箱等宽分箱，结果存入 b5
# @@hint pd.cut(s, 5)
b5 = pd.cut(s, 5)
# @@end

for iv in b5.cat.categories:
    print("  ", iv, "| 计数", int((b5 == iv).sum()))
print("NaN:", int(b5.isna().sum()), "| ordered:", b5.cat.ordered)

assert len(b5.cat.categories) == 5
assert b5.value_counts().sort_index().tolist() == [242, 0, 3, 0, 4]
assert int(b5.isna().sum()) == 21"""
    ),
    md(
        "**为什么这样做**：两箱计数为 0 不是 bug——是 9999 这个离群值把等宽箱"
        "撑爆的直接证据，也是下一题换自定义边界的理由。"
    ),
    md("## 题 2：right 语义与越界（对应讲解 §3.2）"),
    code(
        """# @@todo(2) 对 DEMO 以边界 [100, 300, 500]、labels=["a","b"] 分箱（默认 right），存入 r
r = pd.cut(DEMO, bins=[100, 300, 500], labels=["a", "b"])
# @@end

# @@todo(3) 同样的边界但 right=False，存入 rl
rl = pd.cut(DEMO, bins=[100, 300, 500], labels=["a", "b"], right=False)
# @@end

print("right=True :", r.tolist())
print("right=False:", rl.tolist())
out = pd.cut(pd.Series([50, 20000]), bins=[100, 300, 500])
print("箱外值:", out.tolist())

assert r.tolist() == [np.nan, "a", "a", "b", "b"]
assert rl.tolist() == ["a", "a", "b", "b", np.nan]
assert bool(out.isna().all())"""
    ),
    md("## 题 3：业务边界四档 + 整数编码（对应讲解 §3.3）"),
    code(
        """# @@todo(4) 以 EDGES 边界、LABELS 标签对 s 分箱，存入 bin_col；计数打印
# @@hint pd.cut(s, bins=EDGES, labels=LABELS)
bin_col = pd.cut(s, bins=EDGES, labels=LABELS)
# @@end

# @@todo(5) 再来一版 labels=False 的整数编码，存入 bin_code
bin_code = pd.cut(s, bins=EDGES, labels=False)
# @@end

print("计数:", bin_col.value_counts().to_dict())
print("编码分布:", bin_code.value_counts().sort_index().to_dict())

assert bin_col.value_counts().to_dict() == {"低": 141, "中": 56, "高": 45, "极端": 7}
assert int(bin_col.isna().sum()) == 21
assert sorted(bin_code.dropna().unique().tolist()) == [0, 1, 2, 3]"""
    ),
    md("## 题 4：qcut 分位分箱（对应讲解 §3.4）"),
    code(
        """# @@todo(6) 对 s 做 4 分位分箱存入 q4；再做 10 分位 duplicates=\"drop\" 存入 q10
# @@hint pd.qcut(s, 4) ；pd.qcut(s, 10, duplicates="drop")
q4 = pd.qcut(s, 4)
q10 = pd.qcut(s, 10, duplicates="drop")
# @@end

print("qcut(4) 边界:", [round(iv.right, 2) for iv in q4.cat.categories])
print("qcut(4) 计数:", q4.value_counts().sort_index().tolist())
print("qcut(10) 箱数:", len(q10.cat.categories))
status, msg = probe(pd.qcut, pd.Series([1, 1, 1, 1, 1, 1, 2, 3]), 4)
print("重复分位 →", status)

assert q4.value_counts().sort_index().tolist() == [63, 62, 62, 62]
assert [round(iv.right, 2) for iv in q4.cat.categories] == [48.31, 87.34, 192.34, 9999.0]
assert int(q4.isna().sum()) == 21
assert len(q10.cat.categories) == 10
assert status == "err\""""
    ),
    md("## 题 5：分箱计数体检（对应讲解 §3.5）"),
    code(
        """# @@todo(7) 用 value_counts 的分箱参数直接对 s 做 5 箱计数，存入 vc
# @@hint s.value_counts(bins=5)
vc = s.value_counts(bins=5)
# @@end

print("索引类型:", type(vc.index).__name__)
print("计数:", vc.sort_index().tolist())

assert type(vc.index).__name__ == "IntervalIndex"
assert vc.sort_index().tolist() == [242, 0, 3, 0, 4]"""
    ),
    md("## 题 6：分档分层统计 + 档位过滤（对应讲解 §3.6-3.7）"),
    code(
        """# @@todo(8) 把题 3 的 bin_col 存为 dev["负荷档"]，再按负荷档分组统计
#            负荷值的 count/mean，存入 g（observed=False）
dev["负荷档"] = pd.cut(s, bins=EDGES, labels=LABELS)
g = dev.groupby("负荷档", observed=False)[VAL].agg(["count", "mean"])
# @@end

# @@todo(9) 高负荷行数（档 > \"中\"）存入 n_hi；高或缺失行数存入 n_hi_na
# @@hint (dev["负荷档"] > "中") ；缺失要显式补 | dev["负荷档"].isna()
n_hi = int((dev["负荷档"] > "中").sum())
n_hi_na = int(((dev["负荷档"] > "中") | dev["负荷档"].isna()).sum())
# @@end

print(g.round(2))
print("高负荷:", n_hi, "行 | 高或缺失:", n_hi_na, "行")
assert g["count"].tolist() == [141, 56, 45, 7]
assert [round(v, 2) for v in g["mean"]] == [52.09, 157.83, 519.16, 7856.57]
assert n_hi == 52 and n_hi_na == 73"""
    ),
    md("## 题 7（综合）：日均负荷四分位档（对应讲解 §3.8）"),
    code(
        """# @@todo(10) load_curve 按天聚合负荷均值存入 daily（先把时间戳 to_datetime）；
#            再 qcut 成 Q1~Q4 四档存入 q_label
# @@hint lc["时间戳"].dt.date ；pd.qcut(daily, 4, labels=["Q1","Q2","Q3","Q4"])
lc = load.copy()
lc["时间戳"] = pd.to_datetime(lc["时间戳"])
daily = lc.groupby(lc["时间戳"].dt.date)[VAL].mean()
q_label = pd.qcut(daily, 4, labels=["Q1", "Q2", "Q3", "Q4"])
# @@end

print("日均负荷 n:", len(daily), "| min", round(daily.min(), 2),
      "| max", round(daily.max(), 2))
print("四分位档计数:", q_label.value_counts().to_dict())

assert len(daily) == 90
assert round(daily.min(), 2) == 424.68 and round(daily.max(), 2) == 778.21
assert q_label.value_counts().to_dict() == {"Q1": 23, "Q2": 22, "Q3": 22, "Q4": 23}"""
    ),
    md(
        """
---

## 综合自查

1. 题 1 两箱为 0 说明什么？数字证据是什么？
2. 题 2 的 100 和 500 分别在哪种语义下出界？
3. 题 4 为什么 qcut 对偏态数据更稳？
4. 题 6 里 NaN 行是怎么被比较过滤天然排除的？想保留它们怎么写？

全部答得上来，进入 ch15（多级索引）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
