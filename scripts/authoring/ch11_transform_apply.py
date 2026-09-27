"""ch11 —— 变换与函数应用：groupby.apply 的返回形状陷阱

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch11_transform_apply.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch11_transform_apply"

HEADER = '''import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")
dev = pd.read_csv(DATA / "device_defects.csv")

GROUP_KEY = "设备类型"
VALUE_COL = "负荷值"
NUM_COLS = ["负荷值", "温度", "湿度"]

LEVEL_SCORE = {"一般": 1, "严重": 2, "危急": 3}

# 脚手架函数：写在 @@todo 之外，练习版原样保留
def add_relay_flag(df, col=VALUE_COL, thr=500):
    """给 DataFrame 打一个高负荷标记列。"""
    out = df.copy()
    out["高负荷"] = out[col] > thr
    return out


def row_range(row):
    """一行内数值列的极差（nanmax - nanmin）。"""
    values = row[NUM_COLS].to_numpy(dtype="float64")
    return float(np.nanmax(values) - np.nanmin(values))


def group_span(frame):
    """一个分组内部的负荷值极差。"""
    return float(frame[VALUE_COL].max() - frame[VALUE_COL].min())


def drop_sentinel(df, lo=0, hi=1000, col=VALUE_COL):
    """丢掉负荷值落在 [lo, hi] 之外的哨兵行。"""
    return df[df[col].between(lo, hi)]


def add_ratio(df, col=VALUE_COL, key=GROUP_KEY):
    """加一列：该记录在本分组内的负荷占比。"""
    out = df.copy()
    out["组内占比"] = out[col] / out.groupby(key)[col].transform("sum")
    return out


def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    """
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


print("pandas", pd.__version__)
print("dev", dev.shape)'''

SCAFFOLD = '''gb = dev.groupby(GROUP_KEY)
big = pd.concat([dev] * 400, ignore_index=True)
print("gb:", type(gb).__name__, "| big:", big.shape)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch11 变换与函数应用：`groupby.apply` 的返回形状陷阱

> 方向：数据预处理 ｜ 竞赛对应：**特征工程 + 数据清洗**
> —— `apply` 是"标准 API 覆盖不到时"的通用出口，但它同时是
> **最慢、最容易返回意料外结构**的一族方法。

本章的核心不是"会用 `apply`"，而是三件更硬的事：

1. **先选层级，再选 API**：元素级 / 行级 / 组级，层级选错性能差 100 倍且代码更脏
2. `groupby.apply` 的**返回形状由"第一个组的返回值"决定**——同一段代码
   换个数据就可能从 `Series` 变成 `DataFrame`，甚至多出一级索引
3. `map` 和 `replace` 在"未命中"时的行为**完全不同**，一个变 `NaN` 一个保留原值

还有一条版本情报：`include_groups=True` 在 pandas 3.0 **已被移除**，
传它直接报 `ValueError`——老代码搬过来会当场崩。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 特征工程 | 影响建模分 | 元素映射、行内计算、组内统计广播 |
| 数据清洗 | 每项 2 分 | 用向量化替代逐行遍历，保住执行时间 |
| 结果交付 | 运行时限 | 大表上 `apply` 可能直接超时，必须知道替代方案 |
| 代码质量 | — | 同名操作有多种 API 时选对的那个 |

**为什么单独成章**：竞赛实操有时间限制。用 `iterrows` 处理 10 万行会比
向量化慢上千倍——**算法对不对之外，还有一个"跑不跑得完"的问题**。
"""
    ),
    md(
        """
## 二、API 速查表

| API | 作用层级 | 保持长度 | 一句话说明 |
|---|---|---|---|
| `s.replace(dict)` | 元素 | ✅ | 替换指定值，**未命中保留原值** |
| `s.map(dict)` | 元素 | ✅ | 按字典查表，**未命中变 `NaN`** |
| `s.map(func)` | 元素 | ✅ | 逐元素函数（比 `apply` 略快） |
| `s.apply(func)` | 元素 | ✅ | 逐元素函数 |
| `df.map(func)` | 元素（全表） | ✅ | 逐单元格（**取代已移除的 `applymap`**） |
| `df.apply(func, axis=0)` | 列 | ✅ | 每个**列**一个结果 |
| `df.apply(func, axis=1)` | 行 | ✅ | 每个**行**一个结果（慢，慎用） |
| `gb.transform(func)` | 组 | ✅ | 组统计量广播回原长度（**首选**） |
| `gb.agg(func)` | 组 | ❌ 压缩 | 每组一个或多个标量 |
| `gb.apply(func)` | 组 | ⚠️ 不定 | 每组返回任意结构（**形状不可预测**） |
| `df.pipe(func, **kwargs)` | 表 | ✅ | 把 `df` 当前链式的对象传给函数，便于写管道 |

### 层级决策表（先看这张，再选 API）

| 我要做的事 | 层级 | 首选 | 备用 |
|---|---|---|---|
| 等级 → 分值（查表） | 元素 | `map(dict)` | `replace(dict)` |
| 多分支分档 | 元素 | `np.select` | `map` + `fillna` |
| 一行内跨列算极差 | 行 | **拆成两列算出**（`max(axis=1)`） | `apply(axis=1)` |
| 每列算一个统计量 | 列 | `df.apply(func, axis=0)` 或直接聚合 | — |
| 组均值贴回每行 | 组 | **`transform`** | `gb.apply` + 手工对齐 |
| 每组输出若干行 | 组 | `sort_values` + `gb.head(n)` | `gb.apply(lambda d: d.nlargest(...))` |
| 每组输出自定义结构 | 组 | `gb.apply` | — |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 三、逐节讲解

### 3.1 元素级：`map` 与 `replace` 的分工

这两个几乎总被混用，但在"遇到不在字典里的值时"行为完全相反：
`map` 把它变成 `NaN`，`replace` 原样保留。**选错的后果很不一样**：

- 用 `replace` 做等级映射 → 没列出的等级会**原样留着字符串**，后面 `astype(int)` 报错
- 用 `map` 做同义合并 → 没列出的类别**全变 `NaN`**，`value_counts` 里少一大半
"""
    ),
    code(
        """print("① map(dict)：查表翻译，未命中 -> NaN")
level_score = dev["缺陷等级"].map(LEVEL_SCORE)
print("   结果分布:", level_score.value_counts(dropna=False).to_dict())

print("\\n② 未命中会静默变 NaN（这是本节第一个坑）：")
partial = dev["缺陷等级"].map({"一般": 1})
print("   只给『一般』映射表，结果 NaN 数:", int(partial.isna().sum()),
      "= 总行数 - 一般条数 =", len(dev) - int((dev["缺陷等级"] == "一般").sum()))
print("   你以为漏了几行，其实是漏了 106 行")

print("\\n③ replace(dict)：替换指定值，未命中保留原值")
partial_r = dev["缺陷等级"].replace({"一般": 1})
print("   结果里的取值:", sorted(partial_r.astype(str).unique().tolist()))
print("   未命中保留原样 -> 所以它适合做『同义合并』，不适合做『等级映射』")

print("\\n④ 两者对照表：")
probe = pd.Series(["一般", "严重", "危急"])
print("   原值      :", probe.tolist())
print("   map(2条)  :", probe.map({"一般": 1, "严重": 2}).tolist())
print("   replace(2条):", probe.replace({"一般": 1, "严重": 2}).tolist())

print("\\n⑤ map(func)：需要判断逻辑时用它")
score_func = dev["缺陷等级"].map(lambda v: {"一般": 1, "严重": 2, "危急": 3}.get(v, 0))
print("   map(func) 分布:", score_func.value_counts().to_dict())
print("   与 map(dict) 结果一致:", bool(score_func.equals(level_score)))"""
    ),
    md(
        """
> **判定规则**：
>
> | 目的 | 用什么 | 理由 |
> |---|---|---|
> | 同义合并（"渗油"→"渗漏油"） | `replace(dict)` | 未列出的类别必须原样保留 |
> | 等级映射（"危急"→3） | `map(dict)` | 未命中变 `NaN` 是**好事**，能被 `isna()` 抓到 |
> | 需要兜底逻辑 | `map(func)` + `.get(v, 默认值)` | 显式处理未命中 |
>
> **一条保险**：无论用哪个，做完都跑一遍
> `s.isna().sum()` 和 `s.dtype`——`map` 出意外时会静默产生一大片 `NaN`。

---

### 3.2 行级与列级：`DataFrame.apply(axis=)`

`axis` 决定"一次喂给函数的是列还是行"：

| `axis` | 函数收到 | 返回结构 | 典型用途 |
|---|---|---|---|
| `0`（默认） | 每一**列**（`Series`，index 是行号） | `Series`（index 是列名） | 逐列统计 |
| `1` | 每一**行**（`Series`，index 是列名） | `Series`（index 是行号） | 逐行求和/条件判断 |

`axis=1` 是**最容易写慢**的地方：它对每一行都构造一个 `Series` 对象，
10 万行就会构造 10 万个对象。
"""
    ),
    code(
        """num = dev[NUM_COLS]

print("① axis=0：按列算极差（每列一个数）")
col_range = num.apply(lambda s: s.max() - s.min())
print(col_range.round(3).to_dict())
print("   -> 负荷值极差 10000.0 = 9999 - (-1)，两个哨兵码贡献的")

print("\\n② axis=1：按行算极差（每行一个数）")
row_range_series = num.apply(row_range, axis=1)
print("   前 3 行:", row_range_series.head(3).round(2).tolist())
print("   长度:", len(row_range_series), "= 行数")

print("\\n③ raw=True：把 ndarray 直接喂给函数，省掉构造 Series 的开销")
col_range_raw = num.apply(lambda a: np.nanmax(a) - np.nanmin(a), axis=0, raw=True)
print("   结果与不传 raw 一致:", bool(np.allclose(col_range_raw.to_numpy(), col_range.to_numpy())))

print("\\n④ 同样的事，向量化写法比 axis=1 快得多：")
t0 = time.perf_counter()
_r = num.apply(row_range, axis=1)
t1 = time.perf_counter()
fast = num.max(axis=1) - num.min(axis=1)
t2 = time.perf_counter()
print(f"   apply(axis=1): {t1 - t0:.4f}s")
print(f"   向量化 max-min: {t2 - t1:.4f}s   （快 {(t1 - t0) / max(t2 - t1, 1e-9):.0f} 倍）")
print("   结果一致:", bool(np.allclose(fast.to_numpy(), _r.to_numpy(), equal_nan=True)))

print("\\n⑤ 结论：axis=1 只在该行逻辑『无法用列运算表达』时才用")
print("   能用 num.max(axis=1) / np.where / np.select 表达的，一律不要 axis=1")"""
    ),
    md(
        """
> **判定规则**：写 `apply(axis=1)` 之前先问自己一句：
> **"这个行内计算能不能用列运算写出来？"**
> - 行内求和 → `df[列们].sum(axis=1)` ✅
> - 行内极差 → `df[列们].max(axis=1) - df[列们].min(axis=1)` ✅
> - 行内多分支 → `np.select([...], [...])` ✅
> - 行内要调外部 API / 走复杂分支 → 才用 `apply(axis=1)`
>
> 上面第 ④ 步是实测：**同结果、差十几倍**。

---

### 3.3 难点深挖：`groupby.apply` 的返回形状不可预测

**为什么难**：`gb.apply(func)` 会先拿**第一个组**试一下，看 `func` 返回什么，
然后决定整个结果的组装方式。于是"同一段代码"在不同数据上可能得到三种结构：

| `func` 每组返回 | `apply` 结果 | 索引 |
|---|---|---|
| 标量（如 `float`） | `Series` | `[组键]` |
| `Series` | `DataFrame` | `[组键]` |
| `DataFrame` | `DataFrame` | `[组键, 原索引]` ← **多一级** |

第三种最坑：多出来的一级索引如果不 `reset_index` 掉，
后面 `merge` / `to_csv` 全乱。
"""
    ),
    code(
        """print("① 每组返回标量 -> Series")
r_scalar = gb.apply(lambda d: d[VALUE_COL].mean(), include_groups=False)
print("   type:", type(r_scalar).__name__, "| 长度:", len(r_scalar), "| index.name:", r_scalar.index.name)

print("\\n② 每组返回 Series -> DataFrame（一列一行）")
r_series = gb.apply(lambda d: d[[VALUE_COL, "湿度"]].mean(), include_groups=False)
print("   type:", type(r_series).__name__, "| shape:", r_series.shape, "| index.names:", r_series.index.names)
print(r_series.round(2).to_string())

print("\\n③ 每组返回 DataFrame -> DataFrame，但索引多出一级")
r_frame = gb.apply(lambda d: d.nlargest(2, VALUE_COL), include_groups=False)
print("   type:", type(r_frame).__name__, "| shape:", r_frame.shape)
print("   index.names:", r_frame.index.names, " ← 多了一级 None（原行索引）")
print("   前 4 行:")
print(r_frame[[VALUE_COL, "缺陷等级"]].head(4).to_string())

print("\\n④ 处理多级索引：reset_index 掉多余的一级")
flat = r_frame.reset_index(level=1, drop=True).reset_index()
print("   处理后的列:", list(flat.columns)[:4], "...")
print("   index.names:", flat.index.names)
print("   行数仍是:", len(flat))"""
    ),
    md(
        """
> **判定规则**：`gb.apply` 只在这两种情况用：
> 1. `func` 每组返回**一个标量或一个 `Series`**（结构可预测）
> 2. `func` 每组返回**若干行**，且你已经准备好 `reset_index`
>
> **能用 `agg` / `transform` / `sort+head` 表达的，一律不用 `apply`。**
>
> 另外一条硬情报：`include_groups=True` 在 pandas 3.0 **已被移除**，
> 传它当场报错——见下一节。

---

### 3.4 难点深挖：`include_groups` 在 pandas 3.0 已被移除

**为什么难**：老代码里到处是 `gb.apply(func, include_groups=False)`，
新代码里很多人直接从老代码抄 `include_groups=True`。
在 3.0 上它不仅没用，还会**直接抛错**。

关键行为变化：**分组键不再传进 `func` 收到的子表里**。
如果你的函数依赖分组键那一列，迁移时会报 `KeyError`。
"""
    ),
    code(
        """print("① include_groups=False：仍可用（3.0 的默认行为）")
try:
    ok = gb.apply(lambda d: d[VALUE_COL].mean(), include_groups=False)
    print("   结果长度:", len(ok))
except Exception as exc:
    print("   ->", type(exc).__name__, ":", str(exc)[:60])

print("\\n② include_groups=True：3.0 直接报错")
try:
    gb.apply(lambda d: d[VALUE_COL].mean(), include_groups=True)
    print("   居然成功")
except ValueError as exc:
    print("   ->", type(exc).__name__, ":", str(exc)[:70])

print("\\n③ 关键行为：分组键不在 func 收到的子表里")
sample = next(iter(gb))[1]
print("   第一个组的列:", list(sample.columns))
print("   含分组键 '设备类型' 吗:", GROUP_KEY in sample.columns, " ← 不含")
try:
    gb.apply(lambda d: d[GROUP_KEY].iloc[0], include_groups=False)
    print("   访问分组键成功")
except KeyError as exc:
    print("   -> KeyError:", str(exc)[:50], " ← 老代码迁移时的典型报错")

print("\\n④ 正解：需要分组键就直接用 apply 的 index（它就是组键）")
with_key = gb.apply(lambda d: d[VALUE_COL].max(), include_groups=False)
print("   组键在 index 上:", with_key.index.tolist())
print("   想变成列:", with_key.reset_index(name="最大负荷").to_dict("records")[:2])"""
    ),
    md(
        """
> **判定规则**：
> - 写**新代码**：直接 `gb.apply(func)`，不要传 `include_groups`
> - 迁移**老代码**：把 `include_groups=True` 删掉，并检查 `func` 里有没有
>   `d[分组键]` 这种写法——有就改成用 `apply` 结果的 index
> - 注释里也不要写 `include_groups=True`，否则未来自己抄同样会崩

---

### 3.5 `DataFrame.map` 取代已移除的 `applymap`

pandas 2.1 引入 `DataFrame.map`（逐单元格），并在 3.0 **移除 `applymap`**。
`applymap` 现在连 `hasattr` 都是 `False`。
"""
    ),
    code(
        """print("① 两个方法的存活状态：")
print("   DataFrame.map  存在:", hasattr(dev, "map"))
print("   DataFrame.applymap 存在:", hasattr(dev, "applymap"), " ← 已移除")

print("\\n② DataFrame.map：逐单元格，返回同形状的表")
rounded = dev[NUM_COLS].map(lambda v: round(v, 1) if pd.notna(v) else v)
print("   形状:", rounded.shape, "（与输入一致）")
print("   前 3 行负荷值:", rounded[VALUE_COL].head(3).tolist())

print("\\n③ 与 Series.map 的对照：")
print("   Series.map  只作用于一个 Series")
print("   DataFrame.map 作用于一整张表的所有单元格")
print("\\n④ 警告：DataFrame.map 遇到非数值列会整体崩")
mixed = dev[["负荷值", "缺陷等级"]]
try:
    mixed.map(lambda v: round(v, 1))
    print("   没报错")
except TypeError as exc:
    print("   ->", type(exc).__name__, ":", str(exc)[:70])
print("   正解：先 select_dtypes('number') 再 map")"""
    ),
    md(
        """
> **判定规则**：`DataFrame.map` 只对**纯数值块**用；
> 混类型表会抛 `TypeError`。老代码里的 `applymap` 全部改成 `map`。

---

### 3.6 性能阶梯：向量化 > `map`/`np.select` > `apply` > `iterrows`

这一节的数字是本章最该记住的。用 10.8 万行做同一件事，看四个层级的差距。
"""
    ),
    code(
        """n = len(big)
print("测试规模:", n, "行")

t0 = time.perf_counter()
a_map = big["缺陷等级"].map(LEVEL_SCORE)
t1 = time.perf_counter()
a_apply = big["缺陷等级"].apply(lambda v: LEVEL_SCORE[v])
t2 = time.perf_counter()
a_np = np.select(
    [big["缺陷等级"].eq("危急"), big["缺陷等级"].eq("严重")],
    [3, 2],
    default=1,
)
t3 = time.perf_counter()

rows = []
for _, r in big.iloc[:20000].iterrows():
    rows.append(LEVEL_SCORE[r["缺陷等级"]])
t4 = time.perf_counter()

print("\\n① 四层级的耗时（同一件事：等级 -> 分值）：")
print(f"   Series.map(dict)   : {t1 - t0:.4f}s")
print(f"   Series.apply(func) : {t2 - t1:.4f}s   （{ (t2-t1)/(t1-t0):.1f} 倍）")
print(f"   np.select          : {t3 - t2:.4f}s   （{ (t3-t2)/(t1-t0):.1f} 倍）")
print(f"   iterrows 2 万行    : {t4 - t3:.4f}s   （{(t4-t3)/(t1-t0):.1f} 倍，且只跑了 1/5 行数）")

print("\\n② 结果一致性（三条路径必须完全一致）：")
print("   map == apply:", bool(a_map.equals(pd.Series(a_apply, index=big.index))))
print("   map == np.select:", bool((a_map.to_numpy() == a_np).all()))

print("\\n③ 折算到 10.8 万行：iterrows 大约需要",
      f"{(t4 - t3) / 20000 * n:.1f}s", "—— 竞赛里这就叫超时")"""
    ),
    md(
        """
> **性能阶梯（从快到慢）**：
> 1. **纯向量化**：`np.where` / `np.select` / 列算术
> 2. **`map` / `replace`（字典）**：走 C 层哈希查表
> 3. **`Series.apply`**：每元素一次 Python 函数调用
> 4. **`DataFrame.apply(axis=1)`**：每行构造一个 `Series`
> 5. **`iterrows` / `iteritems`**：每行构造一个 `Series` 且是显式 Python 循环
>
> **纪律**：`iterrows` 在竞赛里**永远不要写**。要写循环就写 `np.select`。

---

### 3.7 `pipe`：把清洗步骤串成一条管道

`df.pipe(func, **kw)` 做的事很简单：把 `df` 作为第一个参数传给 `func`。
但它的价值在于**让代码读起来像流水线**，而且每个 `func` 都能**单独测试**。
"""
    ),
    code(
        """print("① 不用 pipe：变量名一层层叠")
step1 = add_relay_flag(dev, thr=500)
print("   add_relay_flag 命中:", int(step1["高负荷"].sum()))

print("\\n② 用 pipe：读起来就是流水线")
piped = dev.pipe(add_relay_flag, thr=500)
print("   结果一致:", bool(piped["高负荷"].equals(step1["高负荷"])))

print("\\n③ pipe 可以串多步（每步一个可测试的小函数）")
def drop_sentinel(df, lo=0, hi=1000, col=VALUE_COL):
    return df[df[col].between(lo, hi)]

def add_ratio(df, col=VALUE_COL, key=GROUP_KEY):
    out = df.copy()
    out["组内占比"] = out[col] / out.groupby(key)[col].transform("sum")
    return out

chain = (
    dev.pipe(drop_sentinel, lo=0, hi=1000)
    .pipe(add_relay_flag, thr=500)
    .pipe(add_ratio)
)
print("   行数:", len(chain), "（清掉哨兵与负值后）")
print("   新增列:", [c for c in chain.columns if c not in dev.columns])
print("   高负荷条数:", int(chain["高负荷"].sum()))
print("   各组组内占比之和:", chain.groupby(GROUP_KEY)["组内占比"].sum().round(6).to_dict())

print("\\n④ pipe 的另一个好处：可以传 lambda，快速试一步")
quick = dev.pipe(lambda d: d[d[VALUE_COL] > 500])
print("   命中:", len(quick), "（与 dev[dev['负荷值']>500] 一致:",
      len(quick) == int((dev[VALUE_COL] > 500).sum()), "）")"""
    ),
    md(
        """
---

## 四、易错点清单

1. **用 `map` 做同义合并** → 未列出的类别全变 `NaN`（本章实测：106 行消失）
2. **用 `replace` 做等级映射** → 未列出的等级原样保留字符串，后面 `astype(int)` 报错
3. `map` 之后不查 `isna().sum()` → 静默产生大片 `NaN`
4. `df.apply(axis=1)` 处理大表 → 每行构造一个 `Series`，慢十倍以上
5. `axis` 记反（以为 `axis=1` 是按列） → 结果形状像但语义完全错
6. **`gb.apply` 的返回形状由第一个组决定** → 换数据就从 `Series` 变 `DataFrame`
7. `gb.apply` 返回 `DataFrame` 时**多出一级原索引** → 不 `reset_index` 后面全乱
8. **传 `include_groups=True`** → pandas 3.0 直接 `ValueError`
9. `func` 里访问 `d[分组键]` → `KeyError`（分组键不再传入子表）
10. `DataFrame.applymap` → 3.0 已移除，改用 `DataFrame.map`
11. `DataFrame.map` 用在混类型表上 → `TypeError`
12. 写 `iterrows` → 慢到超时；改用 `np.select` / 列运算
13. `pipe` 里的函数**必须返回 DataFrame** → 忘了 `return` 会得到 `None`

---

## 五、本章小结

| 需求 | 正确写法 | 不要用 |
|---|---|---|
| 值 → 值的查表 | `map(dict)` | `apply(lambda)` |
| 同义合并 | `replace(dict)` | `map(dict)` |
| 多分支分档 | `np.select` | `apply` + `if-else` |
| 逐单元格变换 | `DataFrame.map` | ~~`applymap`~~（已移除） |
| 逐列统计 | `df.apply(func, axis=0)` | `axis=1` |
| 行内计算 | 列运算 / `np.where` | `apply(axis=1)` |
| 组统计量广播回每行 | **`gb.transform`** | `gb.apply` |
| 每组取若干行 | `sort_values` + `gb.head(n)` | `gb.apply(nlargest)` |
| 每组自定义结构 | `gb.apply(func)`，**记得 `reset_index`** | — |
| 串清洗步骤 | `df.pipe(func1).pipe(func2)` | 变量名层层叠 |
| 性能底线 | `map` / `np.select` / 列运算 | `iterrows`（永不使用） |
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch11 练习 —— 变换与函数应用

数据：`data/device_defects.csv`（270 行 x 13 列）

**每题都有 `assert` 验收块**。跑通即正确，跑不通的报错信息本身就说清了哪里错。
**不要翻答案版。**

| 题号 | 考什么 |
|---|---|
| TODO(1) | `map(dict)` 等级映射 |
| TODO(2) | `map` 未命中的静默后果 |
| TODO(3) | `replace(dict)` 与 `map` 的差别 |
| TODO(4) | `df.apply(axis=0)` 逐列统计 |
| TODO(5) | `df.apply(axis=1)` 逐行计算 |
| TODO(6) | 向量化替代 `axis=1` |
| TODO(7) | `gb.apply` 返回 `Series` 的形状 |
| TODO(8) | `gb.apply` 返回 `DataFrame` 的**多级索引** |
| TODO(9) | `include_groups` 在 3.0 的真实行为 |
| TODO(10) | `DataFrame.map` 取代 `applymap` |
| 综合 ① | `transform` 造组内特征（对照 `agg`） |
| 综合 ② | `pipe` 串一条清洗流水线 |
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
        """# @@todo(1) 用 map(dict) 把「缺陷等级」映射成分值，存入 score
# @@hint Series.map(LEVEL_SCORE)
score = dev["缺陷等级"].map(LEVEL_SCORE)
# @@end
# ---- 验收 ----
assert score.isna().sum() == 0, "三个等级都在映射表里，不该有未命中"
assert score.value_counts().to_dict() == {1: 164, 2: 75, 3: 31}
assert str(score.dtype) == "int64", f"dtype 应为 int64，实际 {score.dtype}"
assert len(score) == len(dev), "元素级映射保持长度不变"
print("TODO(1) 通过 -> 分值分布", score.value_counts().to_dict())"""
    ),
    code(
        """# @@todo(2) 故意用一张不完整的映射表，观察未命中变成什么，统计 NaN 个数存入 n_miss
# @@hint dev["缺陷等级"].map({"一般": 1}) 后取 .isna().sum()
partial = dev["缺陷等级"].map({"一般": 1})
n_miss = int(partial.isna().sum())
# @@end
# ---- 验收 ----
assert n_miss == 106, f"未命中应为 106（270 - 一般 164），实际 {n_miss}"
assert n_miss == len(dev) - int((dev["缺陷等级"] == "一般").sum()), "未命中数 = 总行数 - 命中行数"
# 对照：同样不完整的 replace 不会产生 NaN
partial_r = dev["缺陷等级"].replace({"一般": 1})
assert int(partial_r.isna().sum()) == 0, "replace 未命中会保留原值，不产生 NaN"
assert partial_r.nunique() == 3, "replace 后仍有 1 / 严重 / 危急 三种取值"
print("TODO(2) 通过 -> map 未命中产生", n_miss, "个 NaN；replace 未命中产生 0 个")"""
    ),
    code(
        """# @@todo(3) 用 replace 把「缺陷类型」的同义写法合并（渗油/漏油 -> 渗漏油，异物搭挂 -> 异物）
#         先去空白，再合并，结果存入 defect_merged
# @@hint dev["缺陷类型"].str.strip().replace({...})
defect_merged = dev["缺陷类型"].str.strip().replace({
    "渗油": "渗漏油",
    "漏油": "渗漏油",
    "异物搭挂": "异物",
})
# @@end
# ---- 验收 ----
assert defect_merged.nunique() == 6, f"合并后应为 6 类，实际 {defect_merged.nunique()}"
assert int(defect_merged.isna().sum()) == 0, "replace 不该产生缺失（变出 NaN 说明用错成 map 了）"
assert defect_merged.value_counts().to_dict() == {
    "渗漏油": 61, "发热": 51, "放电痕迹": 44, "异物": 40, "破损": 38, "锈蚀": 36,
}
print("TODO(3) 通过 -> 合并为", defect_merged.nunique(), "类:", defect_merged.value_counts().to_dict())"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(4) 用 df.apply(axis=0) 算出三个数值列的极差，存入 col_range
# @@hint dev[NUM_COLS].apply(lambda s: s.max() - s.min())
col_range = dev[NUM_COLS].apply(lambda s: s.max() - s.min())
# @@end
# ---- 验收 ----
assert isinstance(col_range, pd.Series), f"axis=0 应返回 Series，实际 {type(col_range).__name__}"
assert list(col_range.index) == NUM_COLS, "index 应是列名"
assert abs(col_range["负荷值"] - 10000.0) < 1e-6, f"负荷值极差应为 10000.0（9999-(-1)），实际 {col_range['负荷值']}"
assert abs(col_range["温度"] - 45.3) < 1e-6, f"温度极差应为 45.3，实际 {col_range['温度']}"
assert abs(col_range["湿度"] - 155.0) < 1e-6, f"湿度极差应为 155.0，实际 {col_range['湿度']}"
print("TODO(4) 通过 -> 逐列极差", col_range.round(3).to_dict())"""
    ),
    code(
        """# @@todo(5) 用 df.apply(axis=1) 算出每行三个数值列的极差，存入 row_range_series
# @@hint dev[NUM_COLS].apply(row_range, axis=1)   —— row_range 已在脚手架里定义
row_range_series = dev[NUM_COLS].apply(row_range, axis=1)
# @@end
# ---- 验收 ----
assert isinstance(row_range_series, pd.Series), "axis=1 应返回 Series"
assert len(row_range_series) == 270, f"长度应等于行数，实际 {len(row_range_series)}"
assert list(row_range_series.index) == list(dev.index), "index 应是行索引"
assert np.allclose(row_range_series.head(3).round(2).tolist(), [61.16, 180.99, 34.61], atol=0.01), \\
    f"前 3 行应为 [61.16, 180.99, 34.61]，实际 {row_range_series.head(3).round(2).tolist()}"
assert row_range_series.notna().all(), "极差计算不应产生缺失"
print("TODO(5) 通过 -> 前 3 行行内极差:", row_range_series.head(3).round(2).tolist())"""
    ),
    code(
        """# @@todo(6) 用向量化写法算同样的行内极差，存入 row_range_fast，并与 apply(axis=1) 结果对齐
# @@hint dev[NUM_COLS].max(axis=1) - dev[NUM_COLS].min(axis=1)
num_block = dev[NUM_COLS]
row_range_fast = num_block.max(axis=1) - num_block.min(axis=1)
# @@end
# ---- 验收 ----
assert len(row_range_fast) == 270
assert bool(np.allclose(row_range_fast.to_numpy(), row_range_series.to_numpy(), equal_nan=True)), \\
    "向量化结果必须与 apply(axis=1) 完全一致"
# 实测耗时对比（只打印，不做断言——计时在不同机器上会漂）
t0 = time.perf_counter()
_ = num_block.apply(row_range, axis=1)
t1 = time.perf_counter()
_ = num_block.max(axis=1) - num_block.min(axis=1)
t2 = time.perf_counter()
print(f"TODO(6) 通过 -> 结果一致；apply(axis=1) {t1 - t0:.4f}s vs 向量化 {t2 - t1:.4f}s"
      f"（快 {(t1 - t0) / max(t2 - t1, 1e-9):.0f} 倍）")"""
    ),
    code(
        """# @@todo(7) 用 gb.apply 让每组返回一个 Series（负荷值与湿度的均值），存入 r_series
# @@hint gb.apply(lambda d: d[[VALUE_COL, "湿度"]].mean(), include_groups=False)
r_series = gb.apply(lambda d: d[[VALUE_COL, "湿度"]].mean(), include_groups=False)
# @@end
# ---- 验收 ----
assert isinstance(r_series, pd.DataFrame), f"返回 Series 时 apply 结果应是 DataFrame，实际 {type(r_series).__name__}"
assert r_series.shape == (5, 2), f"形状应为 (5, 2)，实际 {r_series.shape}"
assert r_series.index.names == [GROUP_KEY], f"index 名应为 [{GROUP_KEY}]，实际 {r_series.index.names}"
assert list(r_series.columns) == [VALUE_COL, "湿度"]
assert abs(r_series.loc["变压器", VALUE_COL] - 986.9231) < 1e-3
print("TODO(7) 通过 ->", r_series.shape)
print(r_series.round(2).to_string())"""
    ),
    code(
        """# @@todo(8) 用 gb.apply 让每组返回「负荷值最大的 2 行」，存入 r_frame
# @@hint gb.apply(lambda d: d.nlargest(2, VALUE_COL), include_groups=False)
r_frame = gb.apply(lambda d: d.nlargest(2, VALUE_COL), include_groups=False)
# @@end
# ---- 验收 ----
assert isinstance(r_frame, pd.DataFrame), "应返回 DataFrame"
assert r_frame.shape == (10, 12), f"形状应为 (10, 12)（5 组 x 2 行，12 列＝原 13 列去掉分组键），实际 {r_frame.shape}"
assert r_frame.index.nlevels == 2, f"索引应有 2 级，实际 {r_frame.index.nlevels}"
assert r_frame.index.names == [GROUP_KEY, None], f"索引名应为 [{GROUP_KEY}, None]，实际 {r_frame.index.names}"
assert GROUP_KEY not in r_frame.columns, "分组键在索引上，不在列里"
# 处理掉多余的一级原索引
flat = r_frame.reset_index(level=1, drop=True).reset_index()
assert flat.shape[0] == 10, "行数不能变"
assert flat.shape[1] == 13, f"reset_index 后应恢复成 13 列，实际 {flat.shape[1]}"
assert GROUP_KEY in flat.columns, "reset_index 后分组键回到列里"
assert flat.index.tolist() == list(range(10)), "应得到干净的 RangeIndex"
print("TODO(8) 通过 -> r_frame", r_frame.shape, "索引", r_frame.index.names)
print(flat[[GROUP_KEY, VALUE_COL, "缺陷等级"]].to_string(index=False))"""
    ),
    code(
        """# @@todo(9) 用实验验证 include_groups 在 pandas 3.0 的行为：
#         include_groups=False 的返回值存入 ok_result；
#         用脚手架里的 probe 探测 include_groups=True，结果存入 probe_true
# @@hint ok_result = gb.apply(lambda d: d[VALUE_COL].mean(), include_groups=False)
# @@hint probe_true = probe(lambda **kw: gb.apply(lambda d: d[VALUE_COL].mean(), **kw), include_groups=True)
ok_result = gb.apply(lambda d: d[VALUE_COL].mean(), include_groups=False)
probe_true = probe(
    lambda **kw: gb.apply(lambda d: d[VALUE_COL].mean(), **kw), include_groups=True
)
# @@end
# ---- 验收 ----
assert ok_result is not None, "include_groups=False 应该可用"
assert len(ok_result) == 5, f"应有 5 组，实际 {len(ok_result)}"
assert probe_true[0] == "err", "include_groups=True 在 pandas 3.0 应直接报错"
assert probe_true[1] == "ValueError", f"include_groups=True 应抛 ValueError，实际 {probe_true[1]}"
# 关键行为：include_groups=False 时 func 收到的子表里没有分组键
has_key_in_apply = bool(gb.apply(lambda d: GROUP_KEY in d.columns, include_groups=False).any())
assert has_key_in_apply is False, "include_groups=False 下子表不该含分组键"
# 对照：直接 for/next 迭代 GroupBy 拿到的子表**仍然含**分组键（这一步不受 include_groups 影响）
sub = next(iter(gb))[1]
assert GROUP_KEY in sub.columns, "直接迭代 GroupBy 的子表仍保留分组键（与 apply 不同！）"
assert len(sub.columns) == 13, f"直接迭代应为 13 列，实际 {len(sub.columns)}"
r_frame_cols = gb.apply(lambda d: d.nlargest(1, VALUE_COL), include_groups=False).columns
assert len(r_frame_cols) == 12, f"apply(include_groups=False) 的子表应为 12 列，实际 {len(r_frame_cols)}"
print("TODO(9) 通过 -> include_groups=False 可用；=True 抛", probe_true[1])
print("   直接迭代列数:", len(sub.columns), "| apply(include_groups=False) 列数:", len(r_frame_cols))"""
    ),
    code(
        """# @@todo(10) 用 DataFrame.map 把三个数值列四舍五入到 1 位小数，存入 rounded
#         并记录 applymap 是否还存在（布尔量存入 has_applymap）
# @@hint dev[NUM_COLS].map(lambda v: round(v, 1) if pd.notna(v) else v) ；hasattr(dev, "applymap")
rounded = dev[NUM_COLS].map(lambda v: round(v, 1) if pd.notna(v) else v)
has_applymap = hasattr(dev, "applymap")
# @@end
# ---- 验收 ----
assert has_applymap is False, "applymap 在 pandas 3.0 已被移除"
assert hasattr(dev, "map") is True, "DataFrame.map 存在"
assert rounded.shape == (270, 3), f"形状应保持 (270, 3)，实际 {rounded.shape}"
assert rounded[VALUE_COL].head(2).tolist() == [90.0, 180.1], \\
    f"前 2 个负荷值应为 [90.0, 180.1]，实际 {rounded[VALUE_COL].head(2).tolist()}"
assert int(rounded[VALUE_COL].isna().sum()) == 21, "缺失必须原样保留（不能被 round 成 0）"
# 混类型表会崩
try:
    dev[["负荷值", "缺陷等级"]].map(lambda v: round(v, 1))
    raise AssertionError("混类型表上 DataFrame.map 应该抛 TypeError")
except AssertionError:
    raise
except TypeError:
    pass
print("TODO(10) 通过 -> applymap 存在:", has_applymap, "| 数值块 map 后前 2 值:",
      rounded[VALUE_COL].head(2).tolist())"""
    ),
    md(
        """
---

## 综合题

### 综合 ① `transform` 造组内特征（并对照 `agg`）

用 `transform` 造三列组内特征，同时用实验证明 **`agg` 会压缩行数、`transform` 不会**。

三列特征：组均值、组内 z-score、组内百分位排名。
"""
    ),
    code(
        """# @@todo 综合①：造出 feat（含 组均值 / 组内z / 组内排名）与 agg_cmp（每组均值，长度为组数）
# @@hint feat = dev[[GROUP_KEY, VALUE_COL]].copy()；三列分别用 transform("mean") / transform("std") / rank(pct=True)；
#       agg_cmp = dev.groupby(GROUP_KEY)[VALUE_COL].mean()
feat = dev[[GROUP_KEY, VALUE_COL]].copy()
feat["组均值"] = feat.groupby(GROUP_KEY)[VALUE_COL].transform("mean")
feat["组内z"] = (feat[VALUE_COL] - feat["组均值"]) / feat.groupby(GROUP_KEY)[VALUE_COL].transform("std")
feat["组内排名"] = feat.groupby(GROUP_KEY)[VALUE_COL].rank(pct=True)
agg_cmp = dev.groupby(GROUP_KEY)[VALUE_COL].mean()
# @@end
# ---- 验收 ----
assert feat.shape == (270, 5), f"feat 形状应为 (270, 5)，实际 {feat.shape}"
assert len(feat) == len(dev), "transform 保持行数"
assert len(agg_cmp) == 5, "agg 压缩成 5 行"
assert len(feat) != len(agg_cmp), "这就是 transform 与 agg 的根本区别"
assert list(feat.columns) == [GROUP_KEY, VALUE_COL, "组均值", "组内z", "组内排名"]
# 组均值：同一组内应只有一个值，且等于 agg 的结果
for grp, sub in feat.groupby(GROUP_KEY):
    assert sub["组均值"].nunique() == 1, f"{grp} 组内组均值应唯一"
    assert abs(sub["组均值"].iloc[0] - agg_cmp[grp]) < 1e-9, f"{grp} 组 transform 值与 agg 值不一致"
# z-score 与讲解一致
assert np.allclose(feat["组内z"].head(3).round(3).tolist(), [-0.179, -0.221, 0.218], atol=1e-3)
# 组内排名是百分位，落在 (0, 1]
assert feat["组内排名"].min() > 0 and feat["组内排名"].max() == 1.0
# 每组的排名均值应接近 0.5
assert abs(feat.groupby(GROUP_KEY)["组内排名"].mean().mean() - 0.5) < 0.02
print("综合① 通过 -> feat", feat.shape, "vs agg_cmp", agg_cmp.shape)
print(feat.head(3).round(3).to_string(index=False))"""
    ),
    md(
        """
### 综合 ② `pipe` 串一条清洗流水线

用 `pipe` 把三步清洗串起来，全程不出现中间变量：

1. `drop_sentinel`：去掉负荷值不在 `[0, 1000]` 的行（清哨兵与负值）
2. `add_relay_flag`：打高负荷标记（阈值 500）
3. `add_ratio`：算每条记录在本设备类型内的负荷占比

这三个函数**已写在首格脚手架里**（`pipe` 考的是「串」而不是「写函数」），
你只需要把它们按顺序接到 `dev` 后面，并产出报告字典 `pipeline`。

要求最后一步的占比满足"各组之和 == 1"，这是流水线造特征正确的证明。
"""
    ),
    code(
        """# @@todo 综合②：用 pipe 串出 chain —— 先丢哨兵（0~1000），再打高负荷标记（阈值 500），最后加组内占比
# @@hint chain = dev.pipe(drop_sentinel, lo=0, hi=1000).pipe(add_relay_flag, thr=500).pipe(add_ratio)
# pipeline = {"原始行数": len(dev), "清洗后行数": len(chain),
#             "清掉行数": len(dev) - len(chain), "高负荷条数": int(chain["高负荷"].sum())}
chain = (
    dev.pipe(drop_sentinel, lo=0, hi=1000)
    .pipe(add_relay_flag, thr=500)
    .pipe(add_ratio)
)
pipeline = {
    "原始行数": len(dev),
    "清洗后行数": len(chain),
    "清掉行数": len(dev) - len(chain),
    "高负荷条数": int(chain["高负荷"].sum()),
}
# @@end
# ---- 验收 ----
assert set(pipeline) == {"原始行数", "清洗后行数", "清掉行数", "高负荷条数"}
assert pipeline["原始行数"] == 270
assert pipeline["清掉行数"] == 32, (
    f"应清掉 32 行（7 个 >=1000 + 4 个负值 + 21 个缺失），实际 {pipeline['清掉行数']}"
)
assert pipeline["清洗后行数"] == 238
assert pipeline["高负荷条数"] == int((chain[VALUE_COL] > 500).sum())
assert pipeline["高负荷条数"] == 38, f"清洗后负荷值>500 的应为 38 条，实际 {pipeline['高负荷条数']}"
# 新增列
new_cols = [c for c in chain.columns if c not in dev.columns]
assert sorted(new_cols) == ["组内占比", "高负荷"], f"新增列不对：{new_cols}"
# 流水线正确性证明：各组组内占比之和 == 1
ratio_sum = chain.groupby(GROUP_KEY)["组内占比"].sum()
assert np.allclose(ratio_sum.to_numpy(), 1.0, atol=1e-9), f"各组占比之和应为 1，实际 {ratio_sum.to_dict()}"
assert len(chain) < len(dev), "清洗应该减少行数"
print("综合② 通过 -> 流水线报告：")
for key, value in pipeline.items():
    print(f"   {key:10s} = {value}")
print("   各组组内占比之和:", ratio_sum.round(6).to_dict())"""
    ),
    md(
        """
> **⚠️ 这道题里藏着一个静默陷阱**：你算出的「清掉行数」是 **32**，不是 11。
>
> 32 = 7 个 `>= 1000` 的哨兵 + 4 个负值 + **21 个缺失值**。
> `Series.between(0, 1000)` 对 `NaN` 返回 `False`，所以缺失值会被**一起丢掉**，
> 而且不报错、不警告。
>
> - 如果「丢弃缺失」是你想要的 → 正好，一行搞定
> - 如果「缺失要保留、只清哨兵」→ 得写成
>   `df[df[col].isna() | df[col].between(lo, hi)]`
>
> 竞赛里这类"顺手多丢了几行"极难被发现——**直到最后对不上分组统计的样本数**。
> 记住规则：**任何基于比较的过滤（`between` / `>` / `==` / `isin`）都会静默丢掉 `NaN`。**
"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（六题都能答上来才算过）：

1. `map(dict)` 和 `replace(dict)` 在"遇到字典里没有的值"时分别怎么做？各自适合什么场景？
2. `df.apply(axis=1)` 在这份 270 行数据上比向量化慢多少？为什么 10 万行就不可接受了？
3. `gb.apply` 返回 `Series` / `DataFrame` 时，结果的形状和索引分别是什么？多出来的一级怎么处理？
4. `include_groups=True` 在 pandas 3.0 会发生什么？为什么"直接 `for` 迭代"拿到的子表**仍然含**分组键？
5. 你有一个"等级 → 分值"的需求，写出三种实现，并按速度排序。
6. 用 `between` 做过滤时，缺失值去哪了？想保留它该怎么写？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
