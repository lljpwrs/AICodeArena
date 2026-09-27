"""ch14 —— 重塑与透视：`stack` / `unstack` / `pivot` / `pivot_table` / `melt` / `explode`

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch14_reshape_pivot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch14_reshape_pivot"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

dev = pd.read_csv(DATA / "device_defects.csv")

ROW_KEY = "台区编号"
COL_KEY = "设备类型"
LIST_KEY = "缺陷等级"
VALUE_COL = "负荷值"

# 长表（Series 形态）：每个 (台区编号, 设备类型) 组合一行 —— 88 个组合
long = dev.groupby([ROW_KEY, COL_KEY])[VALUE_COL].mean()

# 宽表：把 设备类型 展开成列 —— (18, 5)
wide = long.unstack(COL_KEY)

# 对照用的 DataFrame 形态长表：unstack 后列名会是两级的（详见讲解 §3.2）
long_df = (
    dev.groupby([ROW_KEY, COL_KEY], as_index=False)[VALUE_COL]
    .mean()
    .set_index([ROW_KEY, COL_KEY])
)

print("pandas", pd.__version__)
print("dev", dev.shape, "| long", long.shape, "| wide", wide.shape)
print("wide 列名:", list(wide.columns), "| 列轴名字:", wide.columns.name)'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    """
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def grid_to_long(frame, row_key, col_key, value_name):
    """宽 -> 长：把列名收成一列（列名来源放进 col_key）。"""
    return (
        frame.reset_index()
        .melt(id_vars=row_key, var_name=col_key, value_name=value_name)
    )


def nan_audit(frame):
    """宽表的 NaN 有多少个，以及每行/每列各有多少。"""
    return {
        "总数": int(frame.isna().sum().sum()),
        "按行": frame.isna().sum(axis=1).to_dict(),
    }


print("long 组合数:", len(long), "| 宽表满格数:", wide.shape[0] * wide.shape[1])'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch14 重塑与透视：`stack` / `unstack` / `pivot` / `melt` / `explode`

> 方向：数据预处理 ｜ 竞赛对应：**数据探索**（报表/交叉表）+ **数据准备及处理**（宽表拼装）
> —— 竞赛最后几乎一定要求"输出一张按 X 分组、按 Y 展开的汇总表"，
> 这就是本章的全部内容。

本章的四个核心陷阱：

1. **`pivot` 的 `NaN` 有两张面孔**：组合不存在、和值本身缺失，长得一模一样
2. **重复索引会让 `pivot` / `unstack` 直接报错**，而 `pivot_table` 不会（它聚合）
3. **`stack` 在 pandas 3.0 换了实现**：`dropna=` 参数报错、`Series.stack()` 被移除
4. **`explode` 只处理"已经是列表"的列**——`str.split` 没切开的话它什么也不做
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据探索 | 每项 2 分 | `crosstab` 交叉表、`pivot_table` 汇总表 |
| 数据准备及处理 | 占技能操作 10% | 长宽互转、宽表对齐 |
| 数据清洗 | 每项 2 分 | 多值字段拆分（`explode`） |
| 结果交付 | 提交格式错 = 白做 | 表头层级压平、索引复位、列序整理 |

**为什么单独成章**：重塑操作的输出**形状**由"数据里实际出现了哪些组合"决定，
而不是由你指定的行列决定。所以 `pivot` 出来的表**缺格子**是常态，
而"缺格子"填 `0` 还是留 `NaN` 是业务判断——
填错了，后面所有均值/计数都跟着错，而且**没有任何报错**。
"""
    ),
    md(
        """
## 二、API 速查表

### 2.1 六个入口的分工

| 入口 | 方向 | 重复索引 | 本质 |
|---|---|---|---|
| `df.pivot(index, columns, values)` | 长 → 宽 | ❌ **直接报错** | 纯**位置搬运**，不做任何计算 |
| `df.pivot_table(index, columns, values, aggfunc)` | 长 → 宽 | ✅ 自动聚合 | 分组 + 聚合 + 展开（= `groupby`+`unstack`） |
| `pd.crosstab(index, columns)` | 长 → 宽 | ✅ 默认 `count` | 频次交叉表（专为"计数"而生） |
| `df.stack()` | 宽 → 长 | — | 列 → 索引（**列名变成索引一层**） |
| `df.unstack(level)` | 长 → 宽 | ❌ 报错 | 索引 → 列（**默认拆最后一层**，注意两级列名） |
| `df.melt(id_vars, var_name, value_name)` | 宽 → 长 | — | 指定"谁不动"，其余全变两列 |

### 2.2 关键参数

| 参数 | 所属 | 作用 |
|---|---|---|
| `aggfunc` | `pivot_table` | 聚合函数；**默认 `"mean"`** |
| `fill_value` | `pivot_table` / `unstack` | 把空缺格子填成指定值 |
| `margins=True` | `pivot_table` / `crosstab` | 加"合计"行列 |
| `normalize` | `crosstab` | `"index"` 行归一 / `"columns"` 列归一 / `True` 全表归一 |
| `values` + `aggfunc` | `crosstab` | 从"计数"改成"对某列求聚合" |
| `level` | `unstack` / `stack` | 拆哪一层（`unstack` 默认最内层） |
| `ignore_index` | `explode` | 展开后重排索引 |

### 2.3 一句话选型

```text
要「计数」交叉表            → pd.crosstab
要「聚合值」汇总表          → pivot_table（有重复就靠它）
数据已经无重复、只想搬家    → pivot
不想记参数、从宽表拆列      → melt
列名其实是一个维度          → stack / unstack
一个格子里塞了多个值        → explode
```
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 三、逐节讲解

### 3.1 心智模型：长表（long）与宽表（wide）

```text
长表 long（一行一个组合）              宽表 wide（列名是一个维度）
台区编号        设备类型   负荷值         设备类型    互感器  变压器  断路器  绝缘子  避雷器
STATION_A_01   互感器     335.04       台区编号
STATION_A_01   变压器     1311.66      STATION_A_01  335.04  1311.66  ...     ...     ...
STATION_A_01   断路器     180.18      STATION_A_02  ...
...                                    ...
```

- **长表**是数据库/明细的形态，适合继续 `groupby`
- **宽表**是报表/交付的形态，适合给人看、给 Excel 用

**关键数量关系**（本数据实测）：

```text
dev（明细）   270 行
long（组合）   88 行   ← groupby 后的组合数
wide（宽表）   18 行 × 5 列 = 90 格
               90 - 88 = 2 个格子"理论上存在但数据里没有"
```

记住 **90 vs 88 vs 270** 这三个数字，本章所有坑都从它们的差异里出来。
"""
    ),
    code(
        """print("dev 明细行数 :", len(dev))
print("long 组合数  :", len(long), "（= groupby 后的行数）")
print("wide 形状    :", wide.shape, "= 满格", wide.shape[0] * wide.shape[1], "格")
print("理论组合     :", dev[ROW_KEY].nunique(), "x", dev[COL_KEY].nunique(),
      "=", dev[ROW_KEY].nunique() * dev[COL_KEY].nunique())

print("\\n三者关系:")
print("  270 行明细  → groupby  → 88 个组合")
print("  90 个理论格子 - 88 个有数据的组合 =", 90 - 88, "个「组合不存在」的格子")
print("  实测 wide 的 NaN 数:", int(wide.isna().sum().sum()))

print("\\nwide 的 NaN 分布（按行）:")
print(wide.isna().sum(axis=1).loc[lambda s: s > 0].to_string())"""
    ),
    md(
        """
### 3.2 三者分工：`pivot` / `pivot_table` / `crosstab`

| 需求 | 选哪个 | 一个例子 |
|---|---|---|
| "每个设备类型有几种缺陷等级"（计数） | `pd.crosstab` | 最省事，默认就是 `count` |
| "每个设备类型各等级的平均负荷" | `pivot_table` | 有重复行也能算 |
| "已经 groupby 好了，只想把列名展开" | `pivot` | 最快，但要求无重复 |

`crosstab` 的两种用法（计数 vs 聚合）：
"""
    ),
    code(
        """count_table = pd.crosstab(dev[COL_KEY], dev[LIST_KEY])
print("① crosstab 默认 = 计数")
print(count_table.to_string())
print("\\n  总计 =", int(count_table.to_numpy().sum()), "= dev 行数", len(dev))
print("  行合计:", count_table.sum(axis=1).to_dict())

print("\\n② crosstab + margins=True 加合计")
with_margins = pd.crosstab(dev[COL_KEY], dev[LIST_KEY], margins=True)
print(with_margins.to_string())
print("\\n  右下角 =", int(with_margins.loc["All", "All"]), "（必须等于总行数）")

print("\\n③ crosstab + normalize='index' 行归一（看占比）")
print(pd.crosstab(dev[COL_KEY], dev[LIST_KEY], normalize="index").round(4).to_string())

print("\\n④ crosstab + values + aggfunc 变成聚合表")
print(pd.crosstab(dev[COL_KEY], dev[LIST_KEY], values=dev[VALUE_COL],
                  aggfunc="mean").round(3).to_string())"""
    ),
    md(
        """
**`pivot_table` 与 `groupby + unstack` 的等价关系**（值得背下来）：

```python
dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
# 完全等价于
dev.groupby([ROW_KEY, COL_KEY])[VALUE_COL].mean().unstack(COL_KEY)
```

`pivot_table` 只是**默认补了 `aggfunc="mean"`**、并且帮你把 `NaN` 空洞显式画出来。
本数据两种写法结果都是 `(18, 5)`：
"""
    ),
    code(
        """via_pivot_table = dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
via_groupby = dev.groupby([ROW_KEY, COL_KEY])[VALUE_COL].mean().unstack(COL_KEY)

print("pivot_table     :", via_pivot_table.shape, "| NaN:", int(via_pivot_table.isna().sum().sum()))
print("groupby+unstack :", via_groupby.shape, "| NaN:", int(via_groupby.isna().sum().sum()))
print("列名是否一致:", list(via_pivot_table.columns) == list(via_groupby.columns))

# 对账用「差值最大值」写法 —— 比布尔掩码更省事，且 NaN 会自动跳过
max_diff = (via_pivot_table - via_groupby).abs().max().max()
print("逐格最大差值:", float(max_diff), "| < 1e-9:", bool(float(max_diff) < 1e-9))"""
    ),
    md(
        """
#### 顺带一个必踩的坑：`DataFrame.unstack` 会产生**两级列名**

`pivot_table` 的列名是干净的（`互感器 / 变压器 / ...`）。
但如果长表是 **`DataFrame` 形态**（多列聚合），
`unstack()` 会把**原来那一列的名字也带进列轴**：

```text
long_df 是一个 (88, 1) 的 DataFrame，列名 = ['负荷值']
long_df.unstack(1).columns
# → [('负荷值', '互感器'), ('负荷值', '变压器'), ...]   ⚠️ 两级！
```

后果：`wide["互感器"]` 直接 **`KeyError`**；写进 Excel 会有两行表头。

**修法有两种**，都比事后 `droplevel` 干净：

| 做法 | 写法 | 结果列名层级 |
|---|---|---|
| ✅ 推荐：长表用 **Series** 形态 | `dev.groupby([...])[col].mean().unstack(col)` | **1 级** |
| ⚠️ 长表已是 DataFrame | 事后压平：`w.columns = w.columns.droplevel(0)` | 1 级（手动） |
| ❌ 不管它 | 直接交付 | 2 级 → `KeyError` + Excel 双行表头 |

实测三种形态：
"""
    ),
    code(
        """df_cols = long_df.unstack(1).columns
s_cols = long.unstack(COL_KEY).columns

print("① DataFrame 形态长表 unstack 后的列名:")
print("  ", list(df_cols)[:3], "...")
print("   列轴层级数:", df_cols.nlevels, "| 列轴名字:", df_cols.names)
print("   取 w['互感器'] →",
      probe(lambda: long_df.unstack(1)["互感器"])[1], "KeyError ⚠️")

print("\\n② 手动压平（droplevel）:")
flat = long_df.unstack(1)
flat.columns = flat.columns.droplevel(0)
print("   列轴层级数:", flat.columns.nlevels, "| 列名:", list(flat.columns))
print("   取 w['互感器'] →", round(float(flat["互感器"].iloc[0]), 4), "✅")

print("\\n③ Series 形态长表（推荐）:")
print("   列轴层级数:", s_cols.nlevels, "| 列名:", list(s_cols),
      "| 列轴名字:", s_cols.name)

print("\\n→ 结论：`groupby(...)[单列].mean()` 得到 Series，")
print("  unstack 后列名天然是干净的；用 `as_index=False` 或 [[多列]] 变 DataFrame 就会踩坑。")"""
    ),
    md(
        """
---

### 3.3 难点深挖①：`pivot` 的 `NaN` 有**两张脸**

**为什么难**：宽表里的空格子只有一种外观（`NaN`），但有两种成因：

| 成因 | 含义 | 该填什么 |
|---|---|---|
| **组合不存在** | 这个台区根本没有这种设备 | 填 `0`（"没有"）或留 `NaN` |
| **组合存在但值缺失** | 有这种设备，但所有记录的负荷值都缺 | **不能填 `0`**——填 0 会被当成"负荷为 0" |

本数据两种各占几个？用 `fill_value` 一测就知道——**真的全为 `NaN` 的格子是不会被填的**：

```python
wide.unstack(1, fill_value=-1)   # 只填"组合不存在"的格子
```

先看看 `wide` 里那 3 个 `NaN`：
"""
    ),
    code(
        """print("wide 的 NaN 总数:", int(wide.isna().sum().sum()))
print("\\n逐格定位 NaN:")
flat = wide.isna().stack()
nan_cells = flat[flat].index.tolist()
for station, device in nan_cells:
    print(f"  {station} × {device}")

print("\\n它们各自的成因:")
for station, device in nan_cells:
    subset = dev[(dev[ROW_KEY] == station) & (dev[COL_KEY] == device)]
    print(f"  {station} × {device}: 明细有 {len(subset)} 行，"
          f"负荷值缺失 {int(subset[VALUE_COL].isna().sum())} 个 "
          f"→ {'组合不存在' if len(subset) == 0 else '组合存在但值全缺'}")

print("\\n用分组结果直观看:")
in_long = long.reset_index()
print("  long 里 (A_01, 变压器) 的行:", in_long[(in_long[ROW_KEY] == 'STATION_A_01') & (in_long[COL_KEY] == '变压器')].to_dict("records"))
print("  long 里 (A_01, 互感器) 的行:", in_long[(in_long[ROW_KEY] == 'STATION_A_01') & (in_long[COL_KEY] == '互感器')].to_dict("records"))"""
    ),
    md(
        """
现在用 `fill_value` 把两种 `NaN` 分开：

```python
wide_filled = long.unstack(1, fill_value=-1)
# -1 的格子 = 组合不存在（本来就没有数据）
# 仍然是 NaN 的格子 = 组合存在、但值本身全是缺失
```

**正误对照**：

| 表 | 总格数 | `NaN` 数 | `-1` 数 | 解读 |
|---|---|---|---|---|
| `unstack(1)` | 90 | **3** | — | 两种成因混在一起 |
| `unstack(1, fill_value=-1)` | 90 | **1** | **2** | **2 个组合不存在 + 1 个值全缺** |

**3 = 2 + 1** —— 这就是本章最值钱的一条实测结论。
"""
    ),
    code(
        """wide_raw = long.unstack(1)
wide_filled = long.unstack(1, fill_value=-1)

print(f"{'版本':<28}{'形状':>10}{'NaN 数':>9}{'-1 格数':>10}")
print("-" * 58)
print(f"{'unstack(1)':<28}{str(wide_raw.shape):>10}{int(wide_raw.isna().sum().sum()):>9}{'—':>10}")
print(f"{'unstack(1, fill_value=-1)':<28}{str(wide_filled.shape):>10}"
      f"{int(wide_filled.isna().sum().sum()):>9}{int((wide_filled == -1).sum().sum()):>10}")

print("\\n→ 3 个 NaN 里其实只有 1 个是「值真的缺」，另外 2 个是「组合不存在」")
print("  填 0 之前必须分清：把「值缺失」填成 0，等于告诉模型这里负荷是 0")

only_missing = wide_filled.isna().stack()
print("\\n真正「值全缺」的格子:", [tuple(i) for i in only_missing[only_missing].index.tolist()])
missing_combo = (wide_filled == -1).stack()
print("「组合不存在」的格子结构:", len(missing_combo[missing_combo].index.tolist()), "个（填 -1 即可识别）")"""
    ),
    md(
        """
> **判定规则**：
> - 宽表**要交卷给人看** → 用 `fill_value=0` 填掉"组合不存在"的格子，
>   但必须先确认**没有"值本身缺失"的格子**（否则会被一起填成 0）
> - 宽表**要喂模型** → 建议留 `NaN`，让模型自己的缺失策略去处理；
>   这时候 `fill_value` 只用来**做诊断**，不要用来交付
> - **判断方法**：`fill_value=x` 之后再数一遍 `NaN`。
>   还剩 `NaN` 的格子就是"值本身缺失"，填任何常数都是错
"""
    ),
    md(
        """
### 3.4 难点深挖②：重复索引让 `pivot` / `unstack` **直接报错**

**为什么难**：`pivot` 和 `unstack` 都要求"每个 (行键, 列键) 组合**唯一**"。
只要有重复，它们**不做聚合、直接抛异常**：

```text
ValueError: Index contains duplicate entries, cannot reshape
```

而 `pivot_table` 完全没这个问题——因为它会先 `groupby` 再展开。
很多人第一次遇到这个报错会以为"数据坏了"，其实只是**用错了 API**。

本数据 `dev` 的 `(台区编号, 设备类型)` 组合有大量重复（270 行 → 88 个组合）：
"""
    ),
    code(
        """dup_index = dev.set_index([ROW_KEY, COL_KEY])[[VALUE_COL]]
print("dev 设为双层索引后:", dup_index.shape,
      "| 有重复:", dup_index.index.has_duplicates,
      f"| 唯一组合 {dup_index.index.nunique()} 个")

print("\\n三种写法对比:")
for label, fn in [
    ("df.pivot(...)", lambda: dev.pivot(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)),
    ("dup_index.unstack(1)", lambda: dup_index.unstack(1)),
    ("df.pivot_table(...)", lambda: dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)),
]:
    status, payload = probe(fn)
    if status == "ok":
        print(f"  {label:<24} OK    shape={payload.shape}")
    else:
        print(f"  {label:<24} {payload}: Index contains duplicate entries, cannot reshape")

print("\\n→ pivot / unstack 是「纯搬运」，遇到重复无解；pivot_table 会聚合，所以能过")"""
    ),
    md(
        """
**正确做法**：先自己决定"重复行怎么合并"，再 `pivot`。

```python
# ✅ 方案 A（最省事）：直接用 pivot_table，用 aggfunc 明确怎么合并
dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL, aggfunc="mean")

# ✅ 方案 B：先 groupby 聚合，此时组合已唯一，再用 pivot
(dev.groupby([ROW_KEY, COL_KEY], as_index=False)[VALUE_COL].mean()
    .pivot(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL))

# ❌ 只 drop_duplicates 后 pivot：随机丢掉重复行，结果是"随便挑了一条"
```

**方案 B 有个副产品值得注意**：`groupby` 之后组合数从 `270` 降到 `88`，
但那 88 个组合里可能有**值本身是 `NaN`** 的（本数据就有 1 个：
`STATION_A_01 × 变压器`）。这就是 §3.3 里那个"填不掉"的格子。
"""
    ),
    code(
        """plan_a = dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL, aggfunc="mean")
plan_b = (
    dev.groupby([ROW_KEY, COL_KEY], as_index=False)[VALUE_COL]
    .mean()
    .pivot(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
)
print("方案 A（pivot_table）     :", plan_a.shape, "| NaN:", int(plan_a.isna().sum().sum()))
print("方案 B（groupby → pivot） :", plan_b.shape, "| NaN:", int(plan_b.isna().sum().sum()))
max_diff = (plan_a - plan_b).abs().max().max()
print("两方案逐格最大差值:", float(max_diff), "| < 1e-9:", bool(float(max_diff) < 1e-9))

print("\\n注意 B 里那个填不掉的格子: (STATION_A_01, 变压器) =",
      plan_b.loc["STATION_A_01", "变压器"], "← groupby 均值，本来就是 NaN")
print("  对应明细行数:", len(dev[(dev[ROW_KEY] == 'STATION_A_01') & (dev[COL_KEY] == '变压器')]),
      "| 其中负荷值缺失:", int(dev[(dev[ROW_KEY] == 'STATION_A_01')
                            & (dev[COL_KEY] == '变压器')][VALUE_COL].isna().sum()))"""
    ),
    md(
        """
---

### 3.5 难点深挖③：`stack` / `unstack` 的往返**不守恒**（且 pandas 3.0 改了实现）

**为什么难**：直觉上 `unstack()` 之后 `stack()` 应该回到原样。
但 pandas 3.0 的 `stack` 换了实现，行为和老版本**不一样**，而且
`dropna` 直接变成了**会报错的参数**。

**先说两个 3.0 的硬变更**：

| 老写法 | pandas 3.0 结果 |
|---|---|
| `df.stack(dropna=False)` | **`ValueError`**：`dropna must be unspecified as the new implementation does not introduce rows of NA values` |
| `series.stack()` | **`AttributeError`**：`Series` 没有 `stack` 方法（旧版有） |
| `df.stack(future_stack=True)` | ✅ 仍可用（新实现本来就是它的语义） |

**再看好消息**：新实现**不再丢 `NaN` 行**，所以 `unstack → stack` 往返
**格数守恒**了：
"""
    ),
    code(
        """print("wide 形状:", wide.shape, "→ 每格一个值，理论上应是",
      wide.shape[0] * wide.shape[1], "行")
round_trip = wide.stack()
print("wide.stack() 形状:", round_trip.shape, "| NaN 数:", int(round_trip.isna().sum().sum()))
print("→", wide.shape[0] * wide.shape[1], "格 = ", len(round_trip), "行，格数守恒 ✅（含 3 个 NaN 格）")
print("  索引层级:", round_trip.index.names, "← 原来的列名变成了索引的一层")

print("\\n两个 3.0 变更实测:")
print("  ① df.stack(dropna=False) →",
      probe(wide.stack, dropna=False)[1],
      "（新实现不允许传 dropna）")
print("  ② Series.stack() 是否存在 →", hasattr(wide['互感器'], "stack"),
      "（旧版有，3.0 已移除）")
print("  ③ stack(future_stack=True) →", probe(wide.stack, future_stack=True)[0], "，形状",
      wide.stack(future_stack=True).shape)

print("\\nunstack 默认拆【最后一层】:  ")
two_level = dev.groupby([ROW_KEY, COL_KEY])[[VALUE_COL]].mean()
print("  原始索引层级:", two_level.index.names)
print("  unstack()   →", two_level.unstack().shape, "索引", two_level.unstack().index.names)
print("  unstack(1)  →", two_level.unstack(1).shape, "（与默认相同）")
print("  unstack(0)  →", two_level.unstack(0).shape, "索引", two_level.unstack(0).index.names)"""
    ),
    md(
        """
**`stack` / `unstack` 的索引方向**（记这张图就够了）：

```text
        unstack(1)                    ┌── 列名变成索引一层
宽表  ───────────▶  长表（Series）   │
        stack()    ◀───────────       └── 索引一层变回列名

df.stack()      :  列名 → 索引最内层
df.unstack()    :  索引最内层 → 列名（默认 level=-1）
df.unstack(0)   :  索引最外层 → 列名
```

> **判定规则**：
> - 只想"把列名收起来" → `stack()`；想"把某一层摊开" → `unstack(level)`
> - `unstack` 一定要**显式写 level**（`unstack(0)` / `unstack(1)`），
>   默认值 `-1` 在多层索引下很容易不是你想要的
> - 老代码里出现 `stack(dropna=)` → 直接删掉该参数
> - 老代码里出现 `series.stack()` → 改成 `series.to_frame().stack()`
"""
    ),
    md(
        """
### 3.6 `melt`：宽 → 长（`stack` 的"显式版"）

`melt` 和 `unstack` 都能宽变长，区别在于：

| | `melt` | `unstack` / `stack` |
|---|---|---|
| 身份列 | 用 `id_vars` **显式指定** | 靠索引 |
| 列名 | 变成普通列（`var_name`） | 变成索引一层 |
| 结果索引 | `RangeIndex` | 可能是多层索引 |
| 列名来自 | 任意列 | 只能是索引层 |

**交付场景优先用 `melt`**——因为它产出的是"平铺的普通表"，
索引干净、列名可控，不用再 `reset_index`。
"""
    ),
    code(
        """wide_tbl = wide.copy()
wide_tbl.columns.name = COL_KEY        # 给列轴起个名字，melt 出来的变量列会更清楚
long_back = wide_tbl.reset_index().melt(
    id_vars=ROW_KEY, var_name=COL_KEY, value_name="平均负荷"
)

print("wide :", wide_tbl.shape)
print("melt :", long_back.shape, "| 列:", list(long_back.columns))
print("  索引类型:", type(long_back.index).__name__,
      "| 索引层级:", long_back.index.names)
print("  NaN 数:", int(long_back["平均负荷"].isna().sum()), "（与宽表的 3 个空格一一对应）")

print("\\n往返对账:")
print("  18 x 5 =", 18 * 5, "→ melt 后", len(long_back), "行 ✅")
print("  melt 的值求和 =", round(float(long_back["平均负荷"].sum()), 4))
print("  wide 的值求和 =", round(float(wide_tbl.to_numpy().sum()), 4))
print("  两者一致:", abs(float(long_back["平均负荷"].sum())
                      - float(np.nansum(wide_tbl.to_numpy()))) < 1e-9)

print("\\n靠坐标回查一个格子（对账的正确方式）:")
probe_cell = long_back[(long_back[ROW_KEY] == "STATION_A_02")
                       & (long_back[COL_KEY] == "断路器")]
print(probe_cell.to_string(index=False))
print("  wide 里同格 =", round(float(wide_tbl.loc["STATION_A_02", "断路器"]), 4))"""
    ),
    md(
        """
### 3.7 难点深挖④：`explode` 只处理"已经是列表"的列

**为什么难**：一个格子里塞了多个值（`"渗油/异物"`）时，
直觉是"直接 `explode` 一下"——但 `explode` **不会帮你切分字符串**。
它只把**已经是 list/tuple/Series** 的元素摊开。
如果传进去的是字符串，`explode` 会**原样返回**（因为字符串本身是可迭代的，
但 pandas 会把它当标量处理），不报错、不生效。

**正确顺序**：先 `str.split(...)` 变成列表，再 `explode`。

```python
df["缺陷类型"].str.split("/").explode()      # ✅ 先切后炸
df["缺陷类型"].explode()                     # ❌ 字符串没被切开，等于什么也没做
```
"""
    ),
    code(
        """split_demo = pd.DataFrame(
    {
        "编号": [1, 2, 3, 4],
        "缺陷类型": ["渗油/异物", "放电,发热", "无", "发热/放电/无"],
    }
)
print("原始:")
print(split_demo.to_string(index=False))

print("\\n❌ 直接 explode（字符串没被切）:")
wrong = split_demo.explode("缺陷类型")
print("  行数:", len(wrong), "（还是 4 行，看起来「没生效」）")

print("\\n✅ 先 str.split 再 explode:")
split_demo["拆分"] = split_demo["缺陷类型"].str.split("/")
right = split_demo.explode("缺陷类型")
print("  行数:", len(right), "| 索引:", right.index.tolist())
print(right[["编号", "缺陷类型"]].to_string(index=False))

print("\\n细节 1：split 切不开时返回单元素列表（不是 NaN）")
print("  '无'.split('/') →", pd.Series(["无"]).str.split("/").tolist())
print("\\n细节 2：多个分隔符要用正则")
print("  '放电,发热'.split('/') →", pd.Series(["放电,发热"]).str.split("/").tolist())
print("  '放电,发热'.split('[/,]', regex=True) →",
      pd.Series(["放电,发热"]).str.split(r"[/,]", regex=True).tolist())

print("\\n细节 3：索引会重复（一个源行炸出多行）")
print("  explode 后索引:", right.index.tolist(), "| 有重复:", right.index.has_duplicates)
print("  加 ignore_index=True →",
      split_demo.explode("缺陷类型", ignore_index=True).index.tolist())"""
    ),
    md(
        """
**`explode` 的四个细节**：

| 细节 | 现象 | 处理 |
|---|---|---|
| 没先 `split` | 行数不变，**静默无效** | 必须先 `.str.split(pattern)` |
| 切不开（无分隔符） | 得到**单元素列表**，展开后仍是 1 行 | 正常行为，不会产生 `NaN` |
| 多个分隔符 | 只按第一个字符切 | `regex=True` + 字符类 |
| 索引重复 | 一个源行炸出多行 → 索引重复 | `ignore_index=True` |

`NaN` 在 `str.split` 后是什么？——也是 `NaN`，`explode` 会把它保留成一行 `NaN`。
"""
    ),
    code(
        """with_nan = pd.DataFrame({"编号": [1, 2], "缺陷类型": ["渗油", None]})
exploded_nan = with_nan["缺陷类型"].str.split("/").explode()
print("含 NaN 的列 split 后:", with_nan["缺陷类型"].str.split("/").tolist())
print("explode 后:", exploded_nan.tolist(), "| 行数:", len(exploded_nan))
print("→ NaN 保留为一行 NaN（行数不丢，这一点对行数对账很重要）")

print("\\n多列同时 explode（两列长度必须一致）:")
multi = pd.DataFrame({
    "编号": [1, 2],
    "类型": [["A", "B"], ["C"]],
    "分值": [[1, 2], [3]],
})
print(multi.explode(["类型", "分值"]).to_string(index=False))"""
    ),
    md(
        """
---

### 3.8 综合：长 ↔ 宽 的闭环（带行数对账）

一份可交付的重塑流程，必须能**原路返回**并核对行数/列数/合计三个量：

```text
明细 dev (270, 13)
   │ groupby([台区编号, 设备类型])[负荷值].mean()   ← 单列 = Series
   ▼
long (88,)     ← 88 个组合（其中 1 个组合的值为 NaN）
   │ unstack(设备类型)
   ▼
wide (18, 5) = 90 格（含 3 个 NaN：2 个组合不存在 + 1 个值全缺）
   │ melt(id_vars=[台区编号])
   ▼
long_back (90, 3)  ← 90 行 = 满格数
   │ dropna(subset=["平均负荷"])
   ▼
87 行  ← 90 - 3 个空格子 = 88 个组合 - 1 个「值全缺」的组合
```

**注意最后一行的 `87 ≠ 88`**：这里有两个不同的"长表行数"概念：

| 口径 | 行数 | 含义 |
|---|---|---|
| `long`（groupby 结果） | **88** | 出现过的组合数（含 1 个值全缺的） |
| `long.dropna()` | **87** | **有值**的组合数 |
| `long_back`（melt 结果） | **90** | 满格数（18 × 5） |
| `long_back.dropna()` | **87** | 有值的格子数 ✅ 与 `long.dropna()` 对齐 |

要"回到组合数 88"，**不能靠 `dropna`**——必须用"组合是否存在"这个信息
（也就是 §3.3 里 `fill_value` 诊断出来的那 2 个 `-1` 格子）来筛。

这就是重塑闭环里最容易错的一步：**"长 → 宽 → 长" 不回到原来的行数**，
而且要区分"组合数"和"有值格子数"两个口径。
"""
    ),
    code(
        """print(f"{'阶段':<24}{'形状':>12}{'NaN 数':>9}")
print("-" * 48)
print(f"{'dev（明细）':<24}{str(dev.shape):>12}{int(dev.isna().sum().sum()):>9}")
print(f"{'long（组合）':<24}{str(long.shape):>12}{int(long.isna().sum().sum()):>9}")
print(f"{'wide（宽表）':<24}{str(wide.shape):>12}{int(wide.isna().sum().sum()):>9}")
print(f"{'long_back（melt）':<24}{str(long_back.shape):>12}{int(long_back['平均负荷'].isna().sum()):>9}")

print("\\n四个口径的行数对账:")
print("  long 组合数            :", len(long))
print("  long 里有值的组合数    :", int(long.notna().sum()))
print("  long_back 满格行数     :", len(long_back))
print("  long_back dropna 后    :", len(long_back.dropna(subset=["平均负荷"])))

print("\\n关键: 90 - 87 =", len(long_back) - len(long_back.dropna(subset=["平均负荷"])),
      "= 3 个空格子")
print("      88 - 87 =", len(long) - int(long.notna().sum()),
      "= 1 个「组合存在但值全缺」的组合")

back_to_long = long_back.dropna(subset=["平均负荷"]).set_index([ROW_KEY, COL_KEY])
print("\\n值层面能否回到原始 long（两边都只看有值的）：")
orig = long.dropna()
print("  索引集合一致:", set(back_to_long.index) == set(orig.index))
print("  行数:", len(back_to_long), "vs", len(orig))
print("  值完全一致:",
      bool(np.allclose(back_to_long["平均负荷"].sort_index(), orig.sort_index())))"""
    ),
    md(
        """
---

## 四、易错点清单

1. **`pivot` 遇到重复索引直接 `ValueError`** → 不是数据坏了，是选错 API，该用 `pivot_table`
2. **把宽表两种 `NaN` 当成一回事** → "组合不存在" vs "值本身缺失"，填 `0` 前必须分清
3. **`fill_value` 填不掉"值本身缺失"的格子** → 填完还剩 `NaN`，此时填任何常数都是错
4. **`pivot_table` 的 `aggfunc` 默认是 `"mean"`** → 想计数必须显式写 `"count"` 或 `"size"`
5. **`pivot_table` 的 `size` 与 `count` 结果不同** → `count` 不含缺失，`size` 含
6. **`unstack` 默认拆最后一层** → 多层索引下很容易不是你想要的那层，务必显式写 `level`
7. **`stack(dropna=False)` 在 pandas 3.0 报 `ValueError`** → 新实现不允许传该参数
8. **`Series.stack()` 在 pandas 3.0 已移除** → 改 `series.to_frame().stack()`
9. **"长 → 宽 → 长" 行数不回到原值** → `melt` 带出空格子，`90 ≠ 88`，要显式 `dropna`
10. **`explode` 前忘了 `str.split`** → 静默无效，行数不变
11. **`explode` 后索引重复** → 加 `ignore_index=True`
12. **多分隔符只写了一个字符** → 要用 `regex=True` + 字符类
13. **`crosstab` 的 `normalize` 方向写反** → `"index"` 是行归一，`"columns"` 是列归一
14. **`DataFrame.unstack` 会出两级列名** → `wide["互感器"]` 直接 `KeyError`，交付前必须 `droplevel(0)`
15. **`pivot_table` 多值聚合也会出两级列名** → 交卷前统一压平，Excel 里才好看

---

## 五、本章小结

| 我想…… | 用哪个 | 关键参数 |
|---|---|---|
| 频次交叉表 | `pd.crosstab` | `margins=True` / `normalize=` |
| 聚合汇总表（有重复行） | `df.pivot_table(...)` | `aggfunc`（默认 `mean`）/ `fill_value` |
| 已经聚合好、只展开列 | `df.pivot(...)` | 要求组合唯一 |
| 宽 → 长（交付友好） | `df.melt(...)` | `id_vars` / `var_name` / `value_name` |
| 列名是个维度 | `df.stack()` | 3.0 不要传 `dropna` |
| 索引层摊成列 | `df.unstack(level)` | **显式写 `level`** |
| 一格多值拆开 | `s.str.split(...).explode()` | `ignore_index=True` |

**一句话总结**：重塑的本质是**"格子的定义"**。
长表的格子是"一行"，宽表的格子是"(行键, 列键) 坐标"。
所以每次重塑之后要问三个数：**格数对不对、`NaN` 从哪来、能不能原路返回**。
三个数都对上了，重塑就是安全的。
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch14 练习 —— 重塑与透视

> 数据：`device_defects.csv`（270 x 13，18 台区 × 5 设备类型）
> 已备好：`long`（88 个组合的双层索引）、`wide`（18 x 5 宽表）

**做题规则**：从上往下依次填，每个 `____` 处跑通再往下。
验收块 `assert` 不要改，它是你自己对账用的。

| 题号 | 考点 |
|---|---|
| TODO(1) | `crosstab` 计数表与 `margins` |
| TODO(2) | `crosstab` 的 `normalize` 与 `values+aggfunc` |
| TODO(3) | `pivot_table` 与 `groupby+unstack` 的等价 |
| TODO(4) | 重复索引下 `pivot` / `unstack` 报错 |
| TODO(5) | 宽表 `NaN` 的两张面孔（`fill_value` 诊断） |
| TODO(6) | `unstack` 的 `level` 方向 + 两级列名压平 |
| TODO(7) | `stack` 的往返与 3.0 的两处变更 |
| TODO(8) | `melt` 宽 → 长 |
| TODO(9) | `explode` 的正确顺序 |
| TODO(10) | `explode` 的多分隔符与索引 |
| 综合① | 长 ↔ 宽 闭环 + 行数对账 |
| 综合② | 可交付交叉报表 |

共 12 个挖空块。
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 基础题
"""
    ),
    code(
        """# @@todo(1) 用 crosstab 造「设备类型 x 缺陷等级」计数表存入 ct；
#           再带 margins=True 存入 ct_margin；两者总和分别存入 n_ct / n_margin
# @@hint pd.crosstab(dev[COL_KEY], dev[LIST_KEY]) ；加 margins=True
#        总和用 ct.to_numpy().sum()
ct = pd.crosstab(dev[COL_KEY], dev[LIST_KEY])
ct_margin = pd.crosstab(dev[COL_KEY], dev[LIST_KEY], margins=True)
n_ct = int(ct.to_numpy().sum())
n_margin = int(ct_margin.to_numpy().sum())
# @@end
# ---- 验收 ----
assert ct.shape == (5, 3), f"计数表应为 (5, 3)，实际 {ct.shape}"
assert n_ct == 270, f"计数表总和应等于 dev 行数 270，实际 {n_ct}"
assert ct_margin.shape == (6, 4), f"带 margins 应为 (6, 4)，实际 {ct_margin.shape}"
assert n_margin == n_ct + int(ct.sum(axis=1).sum()) + int(ct.sum(axis=0).sum()) + 270, \\
    "margins 表的总和 = 原表 + 行合计 + 列合计 + 总计"
assert int(ct_margin.loc["All", "All"]) == 270, "右下角总计应为 270"
assert ct.sum(axis=1).to_dict() == {"互感器": 58, "变压器": 55, "断路器": 41, "绝缘子": 57, "避雷器": 59}, \\
    f"行合计不对：{ct.sum(axis=1).to_dict()}"
assert ct.sum(axis=0).to_dict() == {"一般": 164, "严重": 75, "危急": 31}, \\
    f"列合计不对：{ct.sum(axis=0).to_dict()}"
print("TODO(1) 通过 ->", ct.shape, "总和", n_ct, "| margins", ct_margin.shape)"""
    ),
    code(
        """# @@todo(2) 行归一表存入 ct_norm（每个设备类型的等级占比）；
#           再把 values 设为负荷值、aggfunc 用 mean，存入 ct_mean
# @@hint pd.crosstab(dev[COL_KEY], dev[LIST_KEY], normalize="index")
#        pd.crosstab(dev[COL_KEY], dev[LIST_KEY], values=dev[VALUE_COL], aggfunc="mean")
ct_norm = pd.crosstab(dev[COL_KEY], dev[LIST_KEY], normalize="index")
ct_mean = pd.crosstab(dev[COL_KEY], dev[LIST_KEY], values=dev[VALUE_COL], aggfunc="mean")
# @@end
# ---- 验收 ----
assert ct_norm.shape == (5, 3), f"归一表形状应为 (5, 3)，实际 {ct_norm.shape}"
assert np.allclose(ct_norm.sum(axis=1).to_numpy(), 1.0), "行归一的每行之和应为 1"
assert int((ct_norm > 0).to_numpy().sum()) == 15, "本数据 5x3 共 15 格都有记录，归一后全部非零"
assert abs(float(ct_norm.loc["避雷器", "危急"]) - 13 / 59) < 1e-9, \\
    "避雷器×危急 的占比应为 13/59"
assert int(ct_mean.isna().sum().sum()) == 0, "这份数据每个组合都有值，不该有 NaN"
assert abs(float(ct_mean.loc["变压器", "一般"]) - 1311.664) < 0.01, \\
    f"变压器×一般的均值应为 1311.664，实际 {float(ct_mean.loc['变压器', '一般']):.3f}"
assert abs(float(ct_mean.loc["避雷器", "一般"]) - 39.039) < 0.01, "避雷器×一般应为 39.039"
print("TODO(2) 通过 -> 归一表行和全 1 | 聚合表 NaN 0 | 变压器×一般 =",
      round(float(ct_mean.loc["变压器", "一般"]), 3))"""
    ),
    code(
        """# @@todo(3) 验证 pivot_table 与 groupby+unstack 的等价：
#           pivot_table 结果存入 via_pt（index=ROW_KEY, columns=COL_KEY, values=负荷值），
#           groupby+unstack 结果存入 via_gb；两者形状存入 shape_pt / shape_gb
# @@hint dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
#        dev.groupby([ROW_KEY, COL_KEY])[VALUE_COL].mean().unstack(COL_KEY)
via_pt = dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
via_gb = dev.groupby([ROW_KEY, COL_KEY])[VALUE_COL].mean().unstack(COL_KEY)
shape_pt = via_pt.shape
shape_gb = via_gb.shape
# @@end
# ---- 验收 ----
assert shape_pt == (18, 5), f"pivot_table 应为 (18, 5)，实际 {shape_pt}"
assert shape_gb == (18, 5), f"groupby+unstack 应为 (18, 5)，实际 {shape_gb}"
assert list(via_pt.columns) == list(via_gb.columns), "两种写法的列名顺序应一致"
assert int(via_pt.isna().sum().sum()) == 3, f"两种写法都应有 3 个 NaN，实际 {int(via_pt.isna().sum().sum())}"
max_diff = float((via_pt - via_gb).abs().max().max())
assert max_diff < 1e-9, f"非空格子数值应完全一致，实际最大差值 {max_diff}"
assert via_gb.columns.nlevels == 1, "Series.unstack 的列名是单级的"
print("TODO(3) 通过 -> pivot_table 与 groupby+unstack 完全等价，NaN 均为 3")"""
    ),
    code(
        """# @@todo(4) 用 probe 探测重复索引下的报错：
#           dev.pivot(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL) → probe_pivot
#           dev.set_index([ROW_KEY,COL_KEY])[[VALUE_COL]].unstack(1) → probe_unstack
#           dev.pivot_table(...) 的形状 → shape_pt_ok
# @@hint probe(dev.pivot, index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
probe_pivot = probe(dev.pivot, index=ROW_KEY, columns=COL_KEY, values=VALUE_COL)
probe_unstack = probe(
    lambda: dev.set_index([ROW_KEY, COL_KEY])[[VALUE_COL]].unstack(1)
)
shape_pt_ok = dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL).shape
# @@end
# ---- 验收 ----
assert probe_pivot[0] == "err" and probe_pivot[1] == "ValueError", \\
    f"pivot 遇重复索引应抛 ValueError，实际 {probe_pivot}"
assert probe_unstack[0] == "err" and probe_unstack[1] == "ValueError", \\
    f"unstack 遇重复索引应抛 ValueError，实际 {probe_unstack}"
assert shape_pt_ok == (18, 5), "pivot_table 能聚合，应正常返回"
dup = dev.set_index([ROW_KEY, COL_KEY])
assert dup.index.has_duplicates is True, "270 行只对应 88 个唯一组合"
assert dup.index.nunique() == 88, f"唯一组合应为 88，实际 {dup.index.nunique()}"
print("TODO(4) 通过 -> pivot / unstack 均抛", probe_pivot[1], "| pivot_table 正常", shape_pt_ok)"""
    ),
    code(
        """# @@todo(5) 诊断宽表 NaN 的两张面孔：
#           wide_raw = long.unstack(1)（NaN 总数存入 n_raw）
#           wide_filled = long.unstack(1, fill_value=-1)（剩余 NaN 存入 n_left，-1 格数存入 n_filled）
#           再定位「值本身缺失」的格子，把它的元组存入 cell_missing
# @@hint long.unstack(1, fill_value=-1) ；真正全缺的格子填 -1 之后仍是 NaN
wide_raw = long.unstack(1)
n_raw = int(wide_raw.isna().sum().sum())
wide_filled = long.unstack(1, fill_value=-1)
n_left = int(wide_filled.isna().sum().sum())
n_filled = int((wide_filled == -1).sum().sum())
still_nan = wide_filled.isna().stack()
cell_missing = tuple(still_nan[still_nan].index[0])
# @@end
# ---- 验收 ----
assert n_raw == 3, f"原始宽表应有 3 个 NaN，实际 {n_raw}"
assert n_filled == 2, f"「组合不存在」的格子应为 2 个，实际 {n_filled}"
assert n_left == 1, f"「值本身缺失」的格子应为 1 个，实际 {n_left}"
assert n_filled + n_left == n_raw, "3 = 2（组合不存在）+ 1（值本身缺失）"
assert cell_missing == ("STATION_A_01", "变压器"), f"该格子应为 (STATION_A_01, 变压器)，实际 {cell_missing}"
# 该组合确实存在，但负荷值全缺
subset = dev[(dev[ROW_KEY] == "STATION_A_01") & (dev[COL_KEY] == "变压器")]
assert len(subset) > 0, "该组合在明细里确实存在"
assert int(subset[VALUE_COL].isna().sum()) == len(subset), "该组合的负荷值应全为缺失"
print("TODO(5) 通过 -> 3 = 2（组合不存在）+ 1（值本身缺失）")
print("   填不掉的格子:", cell_missing, "| 该组合明细", len(subset), "行，负荷值全缺")"""
    ),
    code(
        """# @@todo(6) 验证 unstack 的 level 方向：
#           unstack() 默认结果形状存入 shape_default
#           unstack(1) 形状存入 shape_lv1，unstack(0) 形状存入 shape_lv0
#           unstack(0) 的索引层级名存入 idx_lv0
# @@hint long.unstack() ；long.unstack(1) ；long.unstack(0) ；.index.names
shape_default = long.unstack().shape
shape_lv1 = long.unstack(1).shape
res_lv0 = long.unstack(0)
shape_lv0 = res_lv0.shape
idx_lv0 = list(res_lv0.index.names)
# @@end
# ---- 验收 ----
assert shape_default == (18, 5), f"默认（拆最后一层）应为 (18, 5)，实际 {shape_default}"
assert shape_lv1 == (18, 5), "unstack(1) 应与默认相同"
assert shape_lv0 == (5, 18), f"unstack(0) 应把台区摊成列，实际 {shape_lv0}"
assert idx_lv0 == [COL_KEY], f"unstack(0) 的索引应是设备类型，实际 {idx_lv0}"
assert shape_lv0 == (shape_lv1[1], shape_lv1[0]), "两个方向的形状应互为转置"
assert long.index.names == [ROW_KEY, COL_KEY], f"原始索引层级：{long.index.names}"
# 附带验证：DataFrame 形态长表 unstack 出两级列名，Series 形态不会
assert long_df.unstack(1).columns.nlevels == 2, "DataFrame.unstack 应产生两级列名"
assert long.unstack(COL_KEY).columns.nlevels == 1, "Series.unstack 的列名是单级的"
assert wide.columns.name == COL_KEY, "Series.unstack 会给列轴起上原名"
assert probe(lambda: long_df.unstack(1)["互感器"])[1] == "KeyError", \\
    "两级列名下用单层键取列会 KeyError —— 所以长表要用 Series 形态"
print("TODO(6) 通过 -> 默认", shape_default, "| unstack(1)", shape_lv1,
      "| unstack(0)", shape_lv0, "索引", idx_lv0)
print("   DataFrame 形态列名层级 ->", long_df.unstack(1).columns.nlevels,
      "| Series 形态 ->", long.unstack(COL_KEY).columns.nlevels)"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(7) stack 的往返与 3.0 两处变更：
#           wide.stack() 的形状存入 shape_stack，索引层级存入 idx_stack，NaN 数存入 n_stack_nan
#           dropna=False 的探测结果存入 probe_dropna
#           Series 是否还有 stack 方法存入 has_series_stack
# @@hint wide.stack() ；probe(wide.stack, dropna=False) ；hasattr(wide[COL_KEY], "stack")
stacked = wide.stack()
shape_stack = stacked.shape
idx_stack = list(stacked.index.names)
n_stack_nan = int(stacked.isna().sum().sum())
probe_dropna = probe(wide.stack, dropna=False)
has_series_stack = hasattr(wide["互感器"], "stack")
# @@end
# ---- 验收 ----
assert shape_stack == (90,), f"stack 后应为 90 行（18 x 5 格），实际 {shape_stack}"
assert idx_stack == [ROW_KEY, COL_KEY], f"stack 后索引层级应为 [台区编号, 设备类型]，实际 {idx_stack}"
assert n_stack_nan == 3, f"3.0 的新 stack 不丢 NaN，应保留 3 个，实际 {n_stack_nan}"
assert probe_dropna[0] == "err", "pandas 3.0 的 stack 不接受 dropna 参数"
assert probe_dropna[1] == "ValueError", f"应抛 ValueError，实际 {probe_dropna[1]}"
assert has_series_stack is False, "pandas 3.0 已移除 Series.stack"
assert probe(wide.stack, future_stack=True)[0] == "ok", "future_stack=True 仍可用"
print("TODO(7) 通过 -> stack", shape_stack, "索引", idx_stack, "| NaN", n_stack_nan,
      "| dropna 报", probe_dropna[1], "| Series.stack 存在:", has_series_stack)"""
    ),
    code(
        """# @@todo(8) 用 melt 把 wide 变回长表：
#           先把列轴命名为 COL_KEY，再 reset_index 后 melt(id_vars=ROW_KEY,
#           var_name=COL_KEY, value_name="平均负荷")，结果存入 long_back
#           形状存入 shape_melt，NaN 数存入 n_melt_nan
# @@hint wide.columns.name = COL_KEY ；wide.reset_index().melt(...)
wide_tbl = wide.copy()
wide_tbl.columns.name = COL_KEY
long_back = wide_tbl.reset_index().melt(
    id_vars=ROW_KEY, var_name=COL_KEY, value_name="平均负荷"
)
shape_melt = long_back.shape
n_melt_nan = int(long_back["平均负荷"].isna().sum())
# @@end
# ---- 验收 ----
assert shape_melt == (90, 3), f"melt 后应为 (90, 3)（满格数），实际 {shape_melt}"
assert list(long_back.columns) == [ROW_KEY, COL_KEY, "平均负荷"], \\
    f"列名应为 [台区编号, 设备类型, 平均负荷]，实际 {list(long_back.columns)}"
assert isinstance(long_back.index, pd.RangeIndex), "melt 产出 RangeIndex，索引干净"
assert n_melt_nan == 3, f"melt 带出 3 个空格子，实际 {n_melt_nan}"
assert abs(float(long_back["平均负荷"].sum()) - float(np.nansum(wide.to_numpy()))) < 1e-9, \\
    "melt 后求和应与宽表 nan 求和一致"
# 关键：90 ≠ 88，必须显式 dropna 才能回到组合数
assert len(long_back) == 90 and len(long) == 88, "满格 90 vs 组合 88，两者不相等"
print("TODO(8) 通过 -> melt", shape_melt, "| NaN", n_melt_nan,
      "| 90 格 vs 88 组合（长→宽→长 行数不守恒）")"""
    ),
    code(
        """# @@todo(9) explode 的正确顺序：
#           造 split_demo = {编号:[1,2,3,4], 缺陷类型:["渗油/异物","放电,发热","无","发热/放电/无"]}
#           直接 explode 的行数存入 n_wrong
#           先 str.split("/") 再 explode 的结果存入 right，行数存入 n_right
# @@hint split_demo.explode("缺陷类型") ；split_demo["缺陷类型"].str.split("/").explode()
split_demo = pd.DataFrame(
    {
        "编号": [1, 2, 3, 4],
        "缺陷类型": ["渗油/异物", "放电,发热", "无", "发热/放电/无"],
    }
)
n_wrong = len(split_demo.explode("缺陷类型"))
right = split_demo["缺陷类型"].str.split("/").explode()
n_right = len(right)
# @@end
# ---- 验收 ----
assert n_wrong == 4, f"不先 split 时 explode 静默无效，仍是 4 行，实际 {n_wrong}"
assert n_right == 7, f"正确拆分后应为 7 行，实际 {n_right}"
assert right.tolist() == ["渗油", "异物", "放电,发热", "无", "发热", "放电", "无"], \\
    f"拆分结果不对：{right.tolist()}"
assert right.index.tolist() == [0, 0, 1, 2, 3, 3, 3], f"索引应为重复索引，实际 {right.index.tolist()}"
assert right.index.has_duplicates, "explode 后索引必然重复"
print("TODO(9) 通过 -> 直接 explode", n_wrong, "行（无效）| 先 split 再 explode",
      n_right, "行 | 索引", right.index.tolist())"""
    ),
    code(
        """# @@todo(10) 多分隔符与 ignore_index：
#           '放电,发热' 用 split('/') 与 split('[/,]', regex=True) 的结果分别存入 s_slash / s_regex
#           split_demo.explode("缺陷类型", ignore_index=True) 的索引存入 idx_ignore
#           含 NaN 的列 split 后再 explode 的行数存入 n_nan_case
# @@hint pd.Series(["放电,发热"]).str.split("/") ；str.split(r"[/,]", regex=True)
#        pd.DataFrame({"编号":[1,2],"缺陷类型":["渗油",None]})["缺陷类型"].str.split("/").explode()
s_slash = pd.Series(["放电,发热"]).str.split("/").tolist()
s_regex = pd.Series(["放电,发热"]).str.split(r"[/,]", regex=True).tolist()
idx_ignore = (
    split_demo.assign(缺陷类型=split_demo["缺陷类型"].str.split("/"))
    .explode("缺陷类型", ignore_index=True)
    .index.tolist()
)
nan_case = pd.DataFrame({"编号": [1, 2], "缺陷类型": ["渗油", None]})
n_nan_case = len(nan_case["缺陷类型"].str.split("/").explode())
# @@end
# ---- 验收 ----
assert s_slash == [["放电,发热"]], f"单字符分隔符切不开，实际 {s_slash}"
assert s_regex == [["放电", "发热"]], f"正则可切多个分隔符，实际 {s_regex}"
assert idx_ignore == [0, 1, 2, 3, 4, 5, 6], f"ignore_index 应重排为 0..6，实际 {idx_ignore}"
assert n_nan_case == 2, f"NaN 会保留为一行，行数应为 2，实际 {n_nan_case}"
assert pd.Series(["无"]).str.split("/").tolist() == [["无"]], "无分隔符时返回单元素列表"
print("TODO(10) 通过 -> 正则切分", s_regex, "| ignore_index", idx_ignore,
      "| NaN 行数", n_nan_case)"""
    ),
    md(
        """
---

## 综合题

### 综合 ① 长 ↔ 宽 闭环与行数对账

要求证明：`dev → long → wide → long_back` 的每一步都能说清"行数为什么是这个数"。
"""
    ),
    code(
        """# @@todo 综合①：产出闭环的四个阶段数字，存入 closure 字典：
#          dev 行数 / long 组合数 / wide 满格数 / long_back 行数
#          以及「dropna 后回到 long 的行数」→ 键名 "回到组合数"
# @@hint 满格数 = wide.shape[0] * wide.shape[1] ；dropna(subset=["平均负荷"])
closure = {
    "dev 行数": len(dev),
    "long 组合数": len(long),
    "long 有值组合数": int(long.notna().sum()),
    "wide 满格数": wide.shape[0] * wide.shape[1],
    "long_back 行数": len(long_back),
    "有值格子数": len(long_back.dropna(subset=["平均负荷"])),
}
# @@end
# ---- 验收 ----
assert closure["dev 行数"] == 270, f"明细应为 270 行，实际 {closure['dev 行数']}"
assert closure["long 组合数"] == 88, f"组合应为 88，实际 {closure['long 组合数']}"
assert closure["wide 满格数"] == 90, f"满格数应为 90，实际 {closure['wide 满格数']}"
assert closure["long_back 行数"] == 90, f"melt 应带出满格 90 行，实际 {closure['long_back 行数']}"
assert closure["long 有值组合数"] == 87, f"88 个组合里有 1 个值全缺，实际 {closure['long 有值组合数']}"
assert closure["有值格子数"] == 87, f"90 格减 3 个空格应为 87，实际 {closure['有值格子数']}"
assert closure["wide 满格数"] - closure["long 组合数"] == 2, "90 - 88 = 2 个组合不存在的格子"
assert closure["long 组合数"] - closure["long 有值组合数"] == 1, "88 - 87 = 1 个值全缺的组合"
assert closure["long_back 行数"] != closure["long 组合数"], \\
    "关键结论：长→宽→长 行数【不守恒】，90（满格）≠ 88（组合）"
# 值层面能否原路返回（两边都只看有值的）
back = long_back.dropna(subset=["平均负荷"]).set_index([ROW_KEY, COL_KEY])
assert set(back.index) == set(long.dropna().index), "有值组合的索引集合应一致"
assert bool(np.allclose(back["平均负荷"].sort_index(), long.dropna().sort_index())), "值应完全一致"
print("综合① 通过 ->", closure)
print("   关键：90（满格）≠ 88（组合）≠ 87（有值），reshape 闭环三个口径都要对账")"""
    ),
    md(
        """
### 综合 ② 一张可交付的交叉报表

要求：以「台区编号 × 设备类型」为轴，值为平均负荷；
用 `assign` 加一列"台区平均"，缺格填 0，最后按"台区平均"降序排列。
"""
    ),
    code(
        """# @@todo 综合②：产出报表 report：
#           ① 对 dev 做 pivot_table(index=ROW_KEY, columns=COL_KEY, values=负荷值,
#              aggfunc="mean", fill_value=0)
#           ② 加一列 "台区平均" = 该行 5 个设备类型的均值
#           ③ 按 "台区平均" 降序排序
#          此外把「填 0 的格子数」存入 n_zero、行数存入 n_rows、列数存入 n_cols
# @@hint pt = dev.pivot_table(index=ROW_KEY, columns=COL_KEY, values=VALUE_COL,
#                             aggfunc="mean", fill_value=0)
#        report = pt.assign(台区平均=pt.mean(axis=1)).sort_values("台区平均", ascending=False)
pt = dev.pivot_table(
    index=ROW_KEY, columns=COL_KEY, values=VALUE_COL, aggfunc="mean", fill_value=0
)
report = pt.assign(台区平均=pt.mean(axis=1)).sort_values("台区平均", ascending=False)
n_zero = int((report[list(pt.columns)] == 0).sum().sum())
n_rows, n_cols = report.shape
# @@end
# ---- 验收 ----
assert n_rows == 18, f"报表应有 18 行（每个台区一行），实际 {n_rows}"
assert n_cols == 6, f"报表应有 6 列（5 设备类型 + 台区平均），实际 {n_cols}"
assert "台区平均" in report.columns, "应有 台区平均 列"
assert list(report.columns[-1:]) == ["台区平均"], f"台区平均应在最后一列，实际 {list(report.columns)}"
assert set(report.columns[:-1]) == set(pt.columns), "前 5 列应恰好是 5 个设备类型"
# 填 0 的格子：2 个组合不存在 + 1 个值全缺（也被填成 0）→ 共 3 个
assert n_zero == 3, f"填 0 的格子应为 3 个，实际 {n_zero}"
assert report["台区平均"].is_monotonic_decreasing, "应按台区平均降序"
assert np.allclose(report[list(pt.columns)].mean(axis=1).to_numpy(),
                   report["台区平均"].to_numpy()), "台区平均应等于 5 个设备类型的均值"
assert int(report[list(pt.columns)].isna().sum().sum()) == 0, "fill_value=0 后不应有 NaN"
print("综合② 通过 -> 报表", report.shape, "| 填 0 格子", n_zero, "个")
print("   注意：那 1 个「值全缺」的格子也被填成了 0 —— 交付前应在报告里说明")
print(report.round(2).head(5).to_string())"""
    ),
    md(
        """
> ⚠️ **这道题的隐藏结论**：`fill_value=0` 把 **3** 个格子都填成了 `0`，
> 但其中只有 **2** 个是"组合不存在"（填 0 合理），
> 剩下 **1** 个是"`STATION_A_01 × 变压器` 有设备但负荷值全缺"（填 0 是把缺失当 0）。
>
> 竞赛里这属于"结果表能跑但语义有瑕疵"。
> 规范做法是：**先 `fill_value` 诊断出的那一格单独处理**（置 `NaN` 并在报告中说明），
> 其余 2 格再填 0。
"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（六题都能答上来才算过）：

1. 宽表里的 `NaN` 有哪两种成因？用什么参数能区分它们？
2. 为什么 `pivot` 会因重复索引报错，而 `pivot_table` 不会？
3. `pivot_table` 与 `groupby + unstack` 是什么关系？`aggfunc` 默认值是什么？
4. `stack` / `unstack` 在 pandas 3.0 有哪两处破坏性变更？
5. `pd.crosstab(index, columns, values=..., aggfunc=...)` 和纯计数版有什么区别？
6. `explode` 前为什么必须先 `str.split`？多分隔符怎么写？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
