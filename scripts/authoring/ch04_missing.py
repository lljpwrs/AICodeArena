"""ch04 —— 缺失值：先量化，再决定填还是删

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch04_missing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch04_missing"

HEADER = """import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("data")
dev = pd.read_csv(DATA / "device_defects.csv")

print("pandas", pd.__version__)
print("读入 device_defects.csv ->", dev.shape)
print("全表缺失单元格 ->", int(dev.isna().sum().sum()))"""


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch04 缺失值：先量化，再决定填还是删

> 方向：数据预处理 ｜ 竞赛对应：**数据清洗第 1 项（2 分）**
> —— 缺失值处理是评分表里**第一个出现的清洗动作**，也是唯一一个
> 「做错方向就全盘皆输」的清洗动作（填错策略比不填更糟）。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据清洗 · 缺失值 | 2 分 | 检测 → 量化 → 判定机制 → 选择策略 → 执行 → 留痕 |
| 数据探索 · 缺失量 | 2 分 | `isna().sum()` / 缺失率 / 缺失行占比 |
| 数据可视化 · 缺失图 | 2 分 | 缺失热力图 + 逐列缺失率条形图 |
| 后续建模 | 影响全部 | 缺失没处理干净，`sklearn` 直接报错或悄悄算出错的均值 |

**这一章的纪律：先量化，再动手。**

竞赛阅卷是**按步骤给分**的。「我直接把 NaN 填了 0」这种写法，
即使结果看起来能提交，也会丢掉「检测 + 判定 + 留痕」这几步的分。

> 标准动作链（背下来）：
> `isna().sum()` → 缺失率 → 缺失行占比 → **判定机制** → 选策略 → 执行 → 记录前后对比
"""
    ),
    md(
        """
## 二、API 速查表

| API | 返回 | 一句话说明 |
|---|---|---|
| `df.isna()` / `df.notna()` | `DataFrame[bool]` | 缺失掩码（`isnull` / `notnull` 是别名，行为一致） |
| `df.isna().sum()` | `Series` | **逐列**缺失计数（最常用的一行） |
| `df.isna().sum().sum()` | `int` | 全表缺失单元格总数 |
| `df.isna().mean()` | `Series` | 逐列**缺失率**（0~1），乘 100 就是百分比 |
| `df.isna().any(axis=1)` | `Series[bool]` | 该行是否有任意缺失（`axis=1` 代表"横着看"） |
| `df.notna().all(axis=1)` | `Series[bool]` | 该行是否**完全没有**缺失 |
| `df.isna().sum(axis=1)` | `Series[int]` | 每行缺了几列 |
| `df.dropna()` | `DataFrame` | 默认 `how="any"`：**只要有一个缺失就删整行** |
| `df.dropna(how="all")` | `DataFrame` | 全行都缺才删 |
| `df.dropna(subset=[列])` | `DataFrame` | **只按指定列**判断是否删除 |
| `df.dropna(thresh=n)` | `DataFrame` | 非空值**少于 n 个**就删（保留"至少 n 个非空"的行） |
| `df.dropna(axis=1)` | `DataFrame` | 删列（按列缺失） |
| `df.fillna(值)` | `DataFrame` | 常量填充 |
| `df.fillna(df["列"].mean())` | `DataFrame` | 用统计量填充 |
| `df.ffill()` / `df.bfill()` | `DataFrame` | 前向 / 后向填充（**pandas 3.0 起 `fillna(method=)` 已被移除**） |
| `df.fillna(v, limit=n)` | `DataFrame` | 最多连续填 n 个 |
| `df.interpolate(method=)` | `DataFrame` | 插值：`linear` / `time` / `nearest` / `polynomial` |
| `df.groupby(键)["列"].transform("median")` | `Series` | **分组统计量**，与原始行一一对应，可直接喂给 `fillna` |
| `df["列"].isna().astype(int)` | `Series[int]` | 缺失指示列（把"缺没缺"变成一个特征） |

> 已移除提醒：`df.fillna(method="ffill")`、`df.fillna(downcast=...)`、
> `df.dropna(add_indicator=True)` 在 pandas 3.0 里**全部报错**。见易错点 §1。
"""
    ),
    md(
        """
---

## 三、逐节讲解

### 3.0 环境准备
"""
    ),
    code(HEADER),
    md(
        """
### 3.1 先分清三种缺失机制（决定你能不能填）

| 机制 | 全称 | 含义 | 能不能填 |
|---|---|---|---|
| **MCAR** | 完全随机缺失 | 缺失与任何变量无关，纯随机 | ✅ 可以填（均值/中位数/模型） |
| **MAR** | 随机缺失 | 缺失**与其他已观测变量**有关 | ✅ 可以填（**必须按相关变量分组填**） |
| **MNAR** | 非随机缺失 | 缺失**与被缺失的值本身**有关 | ❌ 不能随便填（例如"负荷越高越测不到"） |

**怎么判断**：看缺失是不是集中在某个子群体里。

我们的数据里两种机制**都有**：

- `负荷值` 的缺失是 **MCAR** —— 用 `rng.choice` 在整个表上随机挑行挖掉
- `温度` 的缺失是 **MAR** —— 只在 `STATION_C_*` / `STATION_D_*` 这 6 个台区上挖

MAR 如果按 MCAR 处理（直接填全局均值），就等于把 C/D 台区的温度硬拉到全表水平，
**这是竞赛里最常见的失分点**。下面用代码实证这一点。
"""
    ),
    code(
        """# 先看两份缺失的分布
miss = dev.isna().sum()
print("逐列缺失（只列非零）：")
print(miss[miss > 0].sort_values(ascending=False).to_string())
print("\\n全表缺失单元格：", int(miss.sum()), "| 涉及列数：", int((miss > 0).sum()))

# 判定 MAR：把「台区是否属于 C/D」当成分组变量
is_cd = dev["台区编号"].str.startswith(("STATION_C_", "STATION_D_"))
print("\\n判定 温度 缺失是否为 MAR —— 按台区分组看缺失率：")
tmp = dev.assign(_C_D台区=is_cd)
rate_by_group = tmp.groupby("_C_D台区")["温度"].apply(lambda s: s.isna().mean())
print((rate_by_group * 100).round(4).to_string(), " (单位 %)")

print("\\n温度缺失涉及的台区：")
print(sorted(dev.loc[dev["温度"].isna(), "台区编号"].unique()))

print("\\n对照：负荷值 的缺失率在不同台区上应该差不多（MCAR 的特征）")
print((dev.groupby("台区编号")["负荷值"].apply(lambda s: s.isna().mean()) * 100)
      .round(2).sort_values(ascending=False).head(6).to_string(), " (单位 %)")

print("\\n结论：温度缺失率在 C/D 台区是 %.1f%%，其他台区是 %.1f%% —— 典型的 MAR。"
      % (rate_by_group[True] * 100, rate_by_group[False] * 100))
print("      负荷值缺失在各台区上都在 8%% 附近抖动 —— 典型的 MCAR。")"""
    ),
    md(
        """
> **MAR 的处理原则**：`温度` 的缺失只能**按台区或供电所分组**去填，
> 绝对不能拿全表均值填。下面的 3.8 节会给出分组填充的正确写法。
"""
    ),
    md(
        """
### 3.2 缺失检测：`isna` / `notna`

一行代码记住：

```python
df.isna()          # 缺失 -> True
df.notna()         # 非缺失 -> True
df.isnull()        # isna 的别名，行为完全一致
df.isna().sum()    # 逐列缺失计数
```

> pandas 3.0 里已经把 `NA`（`pd.NA`）、`None`、`np.nan` 统一处理。
> 但**字符串 `"—"` / `"N/A"` / `"无"` 不会被当成缺失** —— 那是读取阶段要解决的（ch02 / ch06）。
"""
    ),
    code(
        """# 掩码本身长什么样
print("isna() 前 5 行 x 前 6 列：")
print(dev.isna().iloc[:5, :6].to_string())

print("\\nnotna().all(axis=1) 转成布尔：前 8 行是否完整")
print(dev.notna().all(axis=1).head(8).to_list())

print("\\nisnull 与 isna 结果是否一致：",
      dev.isnull().to_numpy().tolist() == dev.isna().to_numpy().tolist())

# 缺失掩码可以直接拿去索引，取出所有「备注缺失」的行
print("\\n备注缺失的行数：", len(dev[dev["备注"].isna()]))
print("备注缺失的前 3 行 记录ID：", dev.loc[dev["备注"].isna(), "记录ID"].head(3).tolist())"""
    ),
    md(
        """
### 3.3 缺失量化：三个维度都要报

竞赛的「数据探索」小项要的是**结论**，不是"我看到了 NaN"。三个维度：

| 维度 | 代码 | 报什么 |
|---|---|---|
| 按**列** | `df.isna().sum()` | 哪几列缺、各缺多少、缺了多少比例 |
| 按**行** | `df.isna().any(axis=1).sum()` | 有多少行受影响、有多少行是完整的 |
| 按**单元格** | `df.isna().sum().sum()` | 全表总量，做前后对比的基准线 |

**`axis` 参数是这里的唯一难点**，记住一句话：

```
axis=0（默认）-> 沿着「行」方向压缩 -> 得到「每列」的结果
axis=1        -> 沿着「列」方向压缩 -> 得到「每行」的结果
```
"""
    ),
    code(
        """# ① 按列
col_miss = dev.isna().sum()
col_rate = dev.isna().mean() * 100
summary = pd.DataFrame({"缺失数": col_miss, "缺失率%": col_rate.round(4)})
print("① 按列（只列有缺失的）：")
print(summary[summary["缺失数"] > 0].sort_values("缺失数", ascending=False).to_string())

# ② 按行
rows_any = dev.isna().any(axis=1)
rows_all_na = dev.isna().all(axis=1)
print("\\n② 按行：")
print("  含任意缺失的行数 :", int(rows_any.sum()), f"({rows_any.mean() * 100:.2f}%)")
print("  完全完整的行数   :", int(dev.notna().all(axis=1).sum()))
print("  整行全空的极端行 :", int(rows_all_na.sum()), " <- 0 说明没有空行")

# 每行缺了几列
per_row = dev.isna().sum(axis=1)
print("\\n  每行缺失列数的分布：")
print(per_row.value_counts().sort_index().to_string())

# ③ 按单元格
print("\\n③ 全表缺失单元格：", int(dev.isna().sum().sum()))"""
    ),
    md(
        """
### 3.4 缺失可视化：两张图（竞赛「可视化」独立计分）

**不依赖 `missingno`**（环境里没装，且竞赛环境不一定允许联网安装）。
用 matplotlib 自绘两张图就够拿分：

1. **缺失热力图** —— 横轴是列、纵轴是行，黑色代表缺失。一眼看出缺失是"散点"还是"成块"
2. **逐列缺失率条形图** —— 一眼看出哪一列最严重

> **图是给阅卷人看的**：一张热力图能省掉 200 字描述。
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# ---- 图 1：缺失热力图 ----
ax = axes[0]
ax.imshow(dev.isna().to_numpy(), aspect="auto", cmap="gray_r", interpolation="nearest")
ax.set_title("缺失热力图（黑色 = 缺失）", fontsize=12)
ax.set_xlabel("列序号")
ax.set_ylabel("行序号")
ax.set_xticks(range(len(dev.columns)))
ax.set_xticklabels(dev.columns, rotation=90, fontsize=8)

# ---- 图 2：逐列缺失率 ----
ax = axes[1]
rate = (dev.isna().mean() * 100).sort_values(ascending=True)
rate = rate[rate > 0]
ax.barh(rate.index, rate.values, color="#c0392b")
for i, (name, val) in enumerate(rate.items()):
    ax.text(val + 1.2, i, f"{val:.2f}%", va="center", fontsize=9)
ax.set_title("逐列缺失率", fontsize=12)
ax.set_xlabel("缺失率 (%)")
ax.set_xlim(0, 100)

plt.tight_layout()
plt.show()

print("从热力图能读出两件事：")
print("  1) 备注列几乎整列都是黑的 -> 缺失率 72.6%%，不是'零星缺失'，是'基本没有'")
print("  2) 温度列的黑色集中在纵轴某一段 -> 那就是 C/D 台区所在的行段，MAR 的直观证据")"""
    ),
    md(
        """
> **怎么读这张热力图**（竞赛时对着它下结论）：
>
> | 图形特征 | 含义 | 该怎么做 |
> |---|---|---|
> | 黑色均匀散布 | MCAR | 可用统计量填充 |
> | 黑色集中成块 | MAR | **必须分组填充** |
> | 整列几乎全黑 | 该列基本无信息 | 考虑直接删列，而不是填 |
> | 黑色集中在少数几行 | 整条记录有问题 | 考虑删行 |
"""
    ),
    md(
        """
### 3.5 判定缺失机制：用「分组缺失率」实证

不靠猜，靠算。核心动作：**把缺失掩码当成一个 0/1 变量，去和别的列做分组统计。**

```python
df.assign(缺失=df["列"].isna()).groupby("分组键")["缺失"].mean()
```
"""
    ),
    code(
        """# 把「是否缺失」当做一个特征，看它在不同分组上的均值（= 缺失率）
def missing_rate_by(frame: pd.DataFrame, col: str, by: str) -> pd.Series:
    \"\"\"按 by 分组，算 col 的缺失率（%）。\"\"\"
    flag = frame[col].isna()
    out = frame.assign(_miss=flag).groupby(by)["_miss"].mean() * 100
    return out.round(2).sort_values(ascending=False)


print("温度 缺失率 by 台区（只列缺失非零的）：")
r = missing_rate_by(dev, "温度", "台区编号")
print(r[r > 0].to_string(), " (单位 %)")

print("\\n温度 缺失率 by 设备类型（应当都很低且分散 -> 与设备类型无关）：")
print(missing_rate_by(dev, "温度", "设备类型").to_string(), " (单位 %)")

print("\\n负荷值 缺失率 by 台区（应当均匀 -> MCAR）：")
r2 = missing_rate_by(dev, "负荷值", "台区编号")
print(r2.to_string(), " (单位 %)")
print("\\n台区间的极差：%.2f 个百分点" % (r2.max() - r2.min()))

# 还可以用卡方/相关快速佐证：缺失标志与台区是否 C/D 强相关
tmp = dev.assign(_miss=dev["温度"].isna(),
                 _cd=dev["台区编号"].str.startswith(("STATION_C_", "STATION_D_")))
print("\\n温度缺失标志 与 是否C/D台区 的相关系数：",
      round(tmp["_miss"].astype(int).corr(tmp["_cd"].astype(int)), 4))
print("负荷值缺失标志 与 是否C/D台区 的相关系数：",
      round(tmp.assign(_m2=dev["负荷值"].isna())["_m2"].astype(int).corr(tmp["_cd"].astype(int)), 4))"""
    ),
    md(
        """
> **结论怎么写进报告**（照抄这个句式）：
>
> > 「温度」缺失 23 个（8.52%），缺失全部集中在 STATION_C_* / STATION_D_* 共 6 个台区，
> > 该子群缺失率 23.71%，其他台区为 0.00%，与「台区」强相关 → 判定为 **MAR**，
> > 按台区分组用中位数填充。
> >
> > 「负荷值」缺失 21 个（7.78%），各台区缺失率极差仅 X 个百分点 → 判定为 **MCAR**，
> > 用全局中位数填充。
"""
    ),
    md(
        """
### 3.6 `dropna`：删行的四种口径

删除是最"安全"但最"浪费"的策略。竞赛里只有在**缺失率极低**或**该列是标签**时才应该删。

| 写法 | 语义 | 结果 |
|---|---|---|
| `dropna()` | 任缺一列就删（`how="any"`） | 最激进 |
| `dropna(how="all")` | 全行都缺才删 | 最保守 |
| `dropna(subset=["列A"])` | **只看列 A** 缺不缺 | 定点删除 |
| `dropna(subset=["列A", "列B"])` | A、B 都不缺才留 | 部分列全要求 |
| `dropna(thresh=n)` | 非空值 ≥ n 才留 | 按"信息量"过滤 |
| `dropna(axis=1)` | **删列** | 按列缺失率删列 |

**关键理解 `thresh`**：`thresh=n` 的意思是"这一行**至少有 n 个非空值**才保留"。
表有 13 列，`thresh=12` 就是"最多允许缺 1 列"。
"""
    ),
    code(
        """print("原始形状                       :", dev.shape)
print("dropna()  默认 how='any'      :", dev.dropna().shape, " <- 几乎删光，13 列里任何一列缺就删")
print("dropna(how='all')             :", dev.dropna(how="all").shape, " <- 没有空行，所以一行没删")
print("dropna(subset=['负荷值'])     :", dev.dropna(subset=["负荷值"]).shape)
print("dropna(subset=['负荷值','温度']):", dev.dropna(subset=["负荷值", "温度"]).shape, " <- 两个都不缺才留")
print("dropna(subset=['备注'])       :", dev.dropna(subset=["备注"]).shape, " <- 备注太稀疏，删完只剩 %d 行" % len(dev.dropna(subset=["备注"])))

print("\\nthresh 口径（13 列，thresh=n 表示至少 n 个非空）：")
for t in (11, 12, 13):
    print(f"  thresh={t:2d} -> {dev.dropna(thresh=t).shape}")

# 按列删：某列缺失率超过 60% 就删掉这一列
miss_rate = dev.isna().mean()
drop_cols = list(miss_rate[miss_rate > 0.6].index)
print("\\n缺失率 > 60%% 的列:", drop_cols)
print("dropna(axis=1, thresh=int(0.4*len(dev))) 结果形状:", dev.dropna(axis=1, thresh=int(0.4 * len(dev))).shape)"""
    ),
    md(
        """
> **`dropna()` 的坑**：它默认是 `axis=0`（删行）、`how="any"`（最激进）。
> 在 13 列的表上直接 `df.dropna()` 会只剩 59 行 —— 数据丢 78%，几乎必然失分。
> **正确姿势是先看缺失热力图，再定点 `subset` 删除。**
"""
    ),
    md(
        """
### 3.7 `fillna`：五种填充策略

从"最粗糙"到"最贴近"排序：

| 策略 | 代码 | 适用 | 风险 |
|---|---|---|---|
| 常量 | `fillna(0)` / `fillna("未知")` | 类别列缺失代表"未知" | 数值列填 0 会拉偏均值 |
| 全局统计量 | `fillna(df["列"].median())` | MCAR 数值列 | 忽略组间差异（MAR 下会错） |
| 组内统计量 | `fillna(groupby(...).transform("median"))` | **MAR** | 需要选对分组键 |
| 前向 / 后向 | `ffill()` / `bfill()` | **有序数据**（时序、流水） | 无序数据上毫无意义 |
| 插值 | `interpolate()` | 时序、连续变化的量 | 连续大片缺失段会插出"假的平滑" |

**均值 vs 中位数**：这个数据里 `负荷值` 的均值是 379.68、中位数是 87.34 ——
因为掺了 `9999` 哨兵值。**被异常值污染的数据必须用中位数**，否则你填进去的是污染值。
"""
    ),
    code(
        """load = dev["负荷值"]

print("原始：count=%d, mean=%.4f, median=%.4f, std=%.4f"
      % (load.count(), load.mean(), load.median(), load.std()))

# ① 常量填充 —— 数值列填 0 是最糟的选择之一
fill_0 = load.fillna(0)
print("\\n① fillna(0)          -> 缺失 %d, mean=%.4f  (均值被拉低)"
      % (fill_0.isna().sum(), fill_0.mean()))

# ② 均值填充 —— 被哨兵值污染
fill_mean = load.fillna(load.mean())
print("② fillna(mean)       -> 缺失 %d, mean=%.4f  (等于用污染值填)"
      % (fill_mean.isna().sum(), fill_mean.mean()))

# ③ 中位数填充 —— MCAR 数值列的默认选择
fill_med = load.fillna(load.median())
print("③ fillna(median)     -> 缺失 %d, mean=%.4f  (抗异常值，推荐)"
      % (fill_med.isna().sum(), fill_med.mean()))

# ④ limit 限制连续填充个数
small = pd.Series([1.0, np.nan, np.nan, np.nan, 5.0])
print("\\n④ 对一个 [1, NaN, NaN, NaN, 5] 演示 limit：")
print("   fillna(0)            :", list(small.fillna(0)))
print("   fillna(0, limit=1)   :", list(small.fillna(0, limit=1)), " <- 只填前 1 个")
print("   ffill()              :", list(small.ffill()))
print("   bfill()              :", list(small.bfill()))

# ⑤ 类别列用众数或固定类别
status = dev["处理状态"]
print("\\n⑤ 类别列填充：")
print("   众数 =", status.mode()[0])
print("   fillna(众数) 后缺失:", int(status.fillna(status.mode()[0]).isna().sum()))

# ---- 关键：pandas 3.0 里 fillna(method=) 已经没了 ----
try:
    load.fillna(method="ffill")
except TypeError as exc:
    print("\\n⚠ pandas 3.0 起 fillna(method=...) 已移除 ->", exc)
    print("   替代写法：Series.ffill() / Series.bfill()")"""
    ),
    md(
        """
### 3.8 分组填充：MAR 场景的唯一正确姿势

这是**本章最重要的一个坑**。先看错误写法：

```python
med_by_type = dev.groupby("设备类型")["负荷值"].median()   # Series，索引是「设备类型」
dev["负荷值"].fillna(med_by_type)                          # ❌ 一个都没填上！
```

**为什么错**：`fillna` 是按**索引对齐**的。`dev["负荷值"]` 的索引是 `0..269`（行号），
而 `med_by_type` 的索引是 `["互感器", "变压器", ...]`（设备类型）。两套索引**完全没有交集**，
所以 pandas 认为"这些标签都没有对应的填充值"，结果是**静默地什么都没填**。

**正确写法有两种**：

```python
# 方案 A：transform —— 直接把分组统计量"广播"回每一行，索引天然对齐
dev["负荷值"].fillna(dev.groupby("设备类型")["负荷值"].transform("median"))

# 方案 B：map —— 用分组键把统计量映射成"每行一个值"
dev["负荷值"].fillna(dev["设备类型"].map(med_by_type))
```

> **这个坑在竞赛里极容易踩**，而且不报错、不警告，只是"没填上"。
> 判据：填完立刻 `isna().sum()` 复查一次，数字没变成 0 就是写错了。
"""
    ),
    code(
        """med_by_type = dev.groupby("设备类型")["负荷值"].median()
print("按设备类型的中位数：")
print(med_by_type.round(2).to_string())

# ❌ 错误写法
wrong = load.fillna(med_by_type)
print("\\n❌ fillna(med_by_type) 后残余缺失 :", int(wrong.isna().sum()), " <- 完全没填上，且不报错")

# ✅ 方案 A：transform
right_a = load.fillna(dev.groupby("设备类型")["负荷值"].transform("median"))
print("✅ 方案 A transform 后残余缺失   :", int(right_a.isna().sum()))

# ✅ 方案 B：map
right_b = load.fillna(dev["设备类型"].map(med_by_type))
print("✅ 方案 B map 后残余缺失         :", int(right_b.isna().sum()))

print("\\n两种正确写法结果是否一致:", bool(np.allclose(right_a.to_numpy(), right_b.to_numpy())))
print("填充后 mean = %.4f（全局中位数填是 %.4f，分组填是 %.4f）"
      % (right_a.mean(), fill_med.mean(), right_a.mean()))

# 验证填充值确实按组生效：变压器组的缺失行应该被填成变压器的中位数
mask = load.isna() & (dev["设备类型"] == "变压器")
print("\\n变压器组原本缺失的 %d 行，填充值 = %s"
      % (int(mask.sum()), sorted(right_a[mask].unique().tolist())))
print("变压器组中位数 =", round(med_by_type["变压器"], 2), " -> 对上了")"""
    ),
    md(
        """
### 3.9 `interpolate`：时序数据的正解

**前向填充（`ffill`）和线性插值的区别，是时序题的高频考点。**

拿我们负荷曲线里那段「连续 6 小时缺失」来看：

- `ffill`：把缺失段全部填成**缺失前最后一个值** → 变成一段水平直线，**抹掉了日周期**
- `interpolate(method="linear")`：按索引等距**线性过渡** → 保住了下降趋势

时序数据有强周期的（负荷、功率、温度），**优先用插值，不要用 ffill**。
"""
    ),
    code(
        """curve = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"])
print("load_curve.csv ->", curve.shape, "| 负荷值缺失:", int(curve["负荷值"].isna().sum()))

# 取一个台区单独看（连续缺失段在 index 500~505）
a01 = curve.loc[curve["台区编号"] == "STATION_A_01"].reset_index(drop=True)
print("STATION_A_01 ->", a01.shape, "| 缺失:", int(a01["负荷值"].isna().sum()))

seg = a01.loc[494:509, ["时间戳", "负荷值"]].copy()
seg["ffill 填"] = a01["负荷值"].ffill().loc[494:509].round(2)
seg["线性插值"] = a01["负荷值"].interpolate(method="linear").loc[494:509].round(2)
print("\\n连续缺失段（20:00 ~ 次日 01:00）的两种填法对比：")
print(seg.to_string(index=False))

print("\\n整段统计对比：")
print("  原始       mean=%.2f  std=%.2f" % (a01["负荷值"].mean(), a01["负荷值"].std()))
print("  ffill 后   mean=%.2f  std=%.2f" % (a01["负荷值"].ffill().mean(), a01["负荷值"].ffill().std()))
print("  linear 后  mean=%.2f  std=%.2f"
      % (a01["负荷值"].interpolate().mean(), a01["负荷值"].interpolate().std()))

# 画出对比图：ffill 会把缺失段压成一条水平线
plt.figure(figsize=(13, 4))
w = a01.loc[480:530]
plt.plot(w["时间戳"], w["负荷值"], "o-", label="原始", color="#2c3e50", markersize=4)
plt.plot(w["时间戳"], a01["负荷值"].ffill().loc[480:530], "--",
         label="ffill 前向填充", color="#e67e22", linewidth=1.6)
plt.plot(w["时间戳"], a01["负荷值"].interpolate().loc[480:530], "-",
         label="linear 线性插值", color="#c0392b", linewidth=1.6)
plt.title("连续缺失段的两种填法：ffill 抹平了日周期，插值保住了趋势")
plt.xlabel("时间")
plt.ylabel("负荷值")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# interpolate 的其他 method（都要 index 严格递增/等距才靠谱）
demo = pd.Series([1.0, np.nan, np.nan, 4.0])
for m in ("linear", "nearest", "zero"):
    print("  method=%-8s ->" % m, list(demo.interpolate(method=m)))

print("\\n插值后统一复查：")
print("  单台区 linear 后残余缺失 :", int(a01["负荷值"].interpolate().isna().sum()))
print("  整表按台区分组插值后残余 :",
      int(curve.assign(负荷值=curve.groupby("台区编号")["负荷值"].transform(
          lambda s: s.interpolate()))["负荷值"].isna().sum()))"""
    ),
    md(
        """
> **`interpolate(method="time")` 的使用前提**：索引必须是 `DatetimeIndex`，
> 且时间间隔可以不等距——它会**按时间差加权**插值。等距小时数据用 `linear` 即可。
>
> `method="pad"` / `"ffill"` / `"bfill"` 在 pandas 3.0 里**已经不能传给 `interpolate`** 了
> （直接 `ValueError`），要前向填就用 `.ffill()`。
"""
    ),
    md(
        """
### 3.10 缺失指示列：把「缺没缺」变成特征

有时候缺失本身就是信息（MNAR 场景：**测不到**可能意味着**超标**）。
这时不要填，而是**加一列 0/1 标记**，把"是否缺失"当特征喂给模型。

```python
df["列_缺失"] = df["列"].isna().astype(int)
```

> pandas 3.0 里 `dropna(add_indicator=True)` 和 `fillna(downcast=...)` **都已移除**，
> 指示列必须自己加。
"""
    ),
    code(
        """work = dev.copy()
work["温度_缺失"] = work["温度"].isna().astype(int)

print("指示列统计：")
print("  温度_缺失 == 1 的行数 :", int(work["温度_缺失"].sum()), " <- 与温度缺失数一致")
print("  该指示列的均值        :", round(work["温度_缺失"].mean(), 4), " <- 就是缺失率")
print("  dtype                :", work["温度_缺失"].dtype)

# 指示列与既有变量的关系（用于判断是不是 MNAR）
print("\\n按「温度_缺失」分组的其他列均值：")
print(work.groupby("温度_缺失")[["负荷值", "湿度"]].mean().round(2).to_string())
print("\\n如果两组的 负荷值 均值差异很大 -> 缺失不是随机的（MNAR 信号），不能简单填。")"""
    ),
    md(
        """
### 3.11 决策清单：填还是删？

```
缺失值处理决策树（照着走，每步都留痕）

  1. 量化：isna().sum() / 缺失率 / 缺失行占比
       |
  2. 该列是【标签 / 主键】吗？
       ├─ 是 -> 直接 dropna(subset=[该列])，标签缺失的行对训练无意义
       └─ 否 -> 继续
       |
  3. 缺失率有多高？
       ├─ > 80%  -> 考虑直接删列（信息量太低），并在报告里写明理由
       ├─ 60~80% -> 加「缺失指示列」，原列不填或填「未知」
       └─ < 60%  -> 继续
       |
  4. 判定机制（看分组缺失率 / 热力图）
       ├─ MCAR -> 中位数（数值）/ 众数或"未知"（类别）
       ├─ MAR  -> 按相关变量分组，组内中位数（必须 transform 或 map！）
       └─ MNAR -> 不填，加缺失指示列，把问题交给模型
       |
  5. 数据类型
       ├─ 时序/有序 -> ffill/bfill（短缺失）或 interpolate（保趋势）
       └─ 无序       -> 统计量填充
       |
  6. 复查：填完立刻 isna().sum()，确认归零；记录填充前后的 mean/std 变化
```

**第 6 步是竞赛最容易被忽略、又最容易被加分的一步**：填充前后各算一次统计量，
写进报告，阅卷人一眼就能看出你**知道自己在干什么**。
"""
    ),
    md(
        """
---

## 四、易错点清单

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | 用 `fillna(method="ffill")` | pandas 3.0 已移除该参数，改用 `df.ffill()` / `df.bfill()` |
| 2 | 用 `dropna(add_indicator=True)` | 已移除，自己加 `df["列_缺失"] = df["列"].isna().astype(int)` |
| 3 | 用 `fillna(0, downcast="infer")` | 已移除该参数 |
| 4 | **`fillna(groupby_median_series)` 静默失败** | 索引对不上，一个都填不上。必须 `transform` 或 `map` |
| 5 | 无脑 `df.dropna()` | 默认 `how="any"`，13 列的表会删掉 78% 的行。先 `subset` 定点删 |
| 6 | 用**均值**填被异常值污染的列 | 先清异常值（ch05）或直接用**中位数** |
| 7 | MAR 缺失用全局统计量填 | 会抹平组间差异，必须分组填 |
| 8 | 类别列填数值 | 类别列只能用众数或"未知"这类**合法类别** |
| 9 | 时序缺失用 `ffill` | 长缺失段会被压成水平线，抹掉周期。用 `interpolate` |
| 10 | `interpolate(method="pad")` | pandas 3.0 会 `ValueError`，前向填用 `.ffill()` |
| 11 | 填充后不复查 | 一定再跑一次 `isna().sum()`，归零才算成功 |
| 12 | 把 `"—"` / `"N/A"` 当缺失 | 字符串占位符 `isna()` **识别不了**，要在读取阶段用 `na_values`（ch02） |
| 13 | `isna().sum()` 与 `isna().sum(axis=1)` 混淆 | 前者逐列，后者逐行 |
| 14 | 未在报告中体现缺失处理过程 | 竞赛按步骤给分：检测 + 判定 + 策略 + 复查，**写了才算** |

---

## 五、本章小结

```
缺失值处理四步法（每一步都是独立得分点）：

  ① 检测   isna().sum()  /  isna().mean()  /  any(axis=1).sum()
  ② 判定   分组缺失率 -> MCAR / MAR / MNAR；热力图看"散点"还是"成块"
  ③ 处理   dropna(subset=)  或  fillna(统计量)  /  ffill  /  interpolate
           MAR 必须分组：groupby().transform("median")  <- 别用 fillna(聚合Series)
  ④ 复查   isna().sum() 归零 + 记录填充前后 mean/std

三个最容易踩的坑：
  · fillna(method=) 已移除            -> 用 .ffill() / .bfill()
  · fillna(分组聚合Series) 静默失败   -> 用 transform 或 map
  · df.dropna() 默认删掉所有带缺失的行 -> 先 subset 定点删
```

**下一步**：做 `ch04_missing_practice.ipynb`。
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook（答案版内容；练习版自动派生）
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch04 练习：缺失值处理

> 共 9 道基础题 + 1 道综合题（拆 3 小步），合计 **12 个空**
> 建议限时：**30 分钟**

数据：`data/device_defects.csv`（270 行 x 13 列）与 `data/load_curve.csv`（6480 行）

**每题都对应竞赛评分表里的一个计分点**。跑通 `assert` 就说明那一步是对的；
跑不通的报错信息本身就告诉你哪里错了。**不要翻答案版。**
"""
    ),
    code(
        """import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

dev = pd.read_csv("data/device_defects.csv")
print("读入 device_defects.csv ->", dev.shape)"""
    ),
    md(
        """
---

### TODO(1) 缺失普查：逐列

1. 逐列缺失计数存入 `miss`（Series，长度 = 列数）
2. 只保留缺失数 > 0 的列，按缺失数**降序**存入 `miss_sorted`

**考点**：`isna().sum()` 给的是**逐列**结果；别忘了 `sort_values`。
"""
    ),
    code(
        """# @@todo(1) 逐列缺失 -> miss；有缺失的列按降序 -> miss_sorted
# @@hint DataFrame.isna().sum() ；布尔索引筛出 > 0 的，再 .sort_values(ascending=False)
miss = dev.isna().sum()
miss_sorted = miss[miss > 0].sort_values(ascending=False)
# @@end

# ---- 验收 ----
assert len(miss) == 13, f"miss 应该是逐列的，长度等于列数 13，实际 {len(miss)}"
assert miss_sorted.to_dict() == {"备注": 196, "温度": 23, "负荷值": 21}, (
    f"有缺失的列不对：{miss_sorted.to_dict()}"
)
assert miss_sorted.index[0] == "备注", "缺失最多的一列应该是「备注」"
print("TODO(1) 通过 ->", miss_sorted.to_dict(), "| 合计", int(miss.sum()))"""
    ),
    md(
        """
### TODO(2) 缺失率

算出逐列**缺失率（百分数，保留 2 位小数）**，存入 `miss_rate`。

然后取出 `温度` 列的缺失率存入 `temp_miss_rate`（也是百分数，保留 2 位小数）。
"""
    ),
    code(
        """# @@todo(2) 逐列缺失率(%) -> miss_rate；温度列缺失率(%) -> temp_miss_rate
# @@hint DataFrame.isna().mean() 给 0~1 的比例，乘 100 才是百分数；再 .round(2)
miss_rate = (dev.isna().mean() * 100).round(2)
temp_miss_rate = miss_rate["温度"]
# @@end

# ---- 验收 ----
assert miss_rate.shape[0] == 13, "miss_rate 应该逐列都有"
assert abs(miss_rate["负荷值"] - 7.78) < 0.01, f"负荷值缺失率应为 7.78%，实际 {miss_rate['负荷值']}"
assert abs(temp_miss_rate - 8.52) < 0.01, f"温度缺失率应为 8.52%，实际 {temp_miss_rate}"
assert abs(miss_rate["备注"] - 72.59) < 0.01, f"备注缺失率应为 72.59%，实际 {miss_rate['备注']}"
print(f"TODO(2) 通过 -> 负荷值 {miss_rate['负荷值']}% | 温度 {temp_miss_rate}% | 备注 {miss_rate['备注']}%")"""
    ),
    md(
        """
### TODO(3) 行视角：有多少行受影响

1. 含**任意**缺失的行数存入 `n_rows_any_na`
2. **完全无缺失**的行数存入 `n_rows_complete`
3. **每行**缺失列数存入 `n_na_per_row`（Series，长度 = 行数）

**考点**：`axis=1` 表示"横着看"（沿着列方向压缩）→ 得到**每行**的结果。
"""
    ),
    code(
        """# @@todo(3) 含任意缺失的行数 / 完全完整的行数 / 每行缺失列数
# @@hint DataFrame.isna().any(axis=1).sum() ；DataFrame.notna().all(axis=1).sum() ；DataFrame.isna().sum(axis=1)
n_rows_any_na = int(dev.isna().any(axis=1).sum())
n_rows_complete = int(dev.notna().all(axis=1).sum())
n_na_per_row = dev.isna().sum(axis=1)
# @@end

# ---- 验收 ----
assert n_rows_any_na == 211, f"含任意缺失的行应为 211，实际 {n_rows_any_na}"
assert n_rows_complete == 59, f"完全无缺失的行应为 59，实际 {n_rows_complete}"
assert n_rows_any_na + n_rows_complete == len(dev), "两类行数之和应等于总行数"
assert len(n_na_per_row) == 270, "n_na_per_row 是「逐行」的，长度应等于行数"
assert n_na_per_row.max() == 2, f"最多一行缺 2 列，实际最大 {n_na_per_row.max()}"
print(f"TODO(3) 通过 -> 受影响 {n_rows_any_na} 行 / 完整 {n_rows_complete} 行")
print("  每行缺失列数分布:", n_na_per_row.value_counts().sort_index().to_dict())"""
    ),
    md(
        """
### TODO(4) `dropna`：默认口径与定点口径

1. 默认 `dropna()`（任缺一列就删）的结果存入 `d_any`
2. 只按 `负荷值` 一列判断的 `dropna` 结果存入 `d_load_only`

**注意对比**：默认口径会把 13 列里**任何**一列缺失的行都删掉，这个数据里会删掉 78% 的行。
"""
    ),
    code(
        """# @@todo(4) 默认 dropna() -> d_any；按「负荷值」定点删除 -> d_load_only
# @@hint DataFrame.dropna() ；DataFrame.dropna(subset=["列名"])
d_any = dev.dropna()
d_load_only = dev.dropna(subset=["负荷值"])
# @@end

# ---- 验收 ----
assert d_any.shape == (59, 13), f"默认 dropna() 应为 (59, 13)，实际 {d_any.shape}"
assert d_load_only.shape == (249, 13), f"按负荷值删除后应为 (249, 13)，实际 {d_load_only.shape}"
assert d_load_only.shape[0] == len(dev) - miss_sorted["负荷值"], "删除的行数应等于负荷值缺失数"
assert d_load_only["负荷值"].isna().sum() == 0, "定点删除后负荷值不应还有缺失"
print(f"TODO(4) 通过 -> 默认删到 {d_any.shape}，定点删到 {d_load_only.shape}")"""
    ),
    md(
        """
### TODO(5) `dropna(thresh=n)`

`thresh=12` 意思是：**这一行至少有 12 个非空值才保留**（表有 13 列，即最多允许缺 1 列）。

把这个结果存入 `d_thresh`，并算出它比原始数据少了多少行，存入 `n_dropped_thresh`。
"""
    ),
    code(
        """# @@todo(5) dropna(thresh=12) 结果 -> d_thresh；被删掉的行数 -> n_dropped_thresh
# @@hint DataFrame.dropna(thresh=12) ；被删行数 = len(原始) - len(结果)
d_thresh = dev.dropna(thresh=12)
n_dropped_thresh = len(dev) - len(d_thresh)
# @@end

# ---- 验收 ----
assert d_thresh.shape == (241, 13), f"thresh=12 应为 (241, 13)，实际 {d_thresh.shape}"
assert n_dropped_thresh == 29, f"应删掉 29 行，实际 {n_dropped_thresh}"
assert n_dropped_thresh == n_na_per_row.eq(2).sum(), (
    "thresh=12 只删「缺 2 列」的行，所以删掉的行数应等于缺 2 列的行数"
)
print(f"TODO(5) 通过 -> thresh=12 剩 {d_thresh.shape[0]} 行，删掉 {n_dropped_thresh} 行")"""
    ),
    md(
        """
### TODO(6) `ffill` / `bfill`

对一个有连续缺失的 Series 做前向与后向填充：

1. `load_ff`：`负荷值` 前向填充的结果
2. `load_bf`：`负荷值` 后向填充的结果
3. `n_ff_left`：前向填充后**还剩多少个缺失**
4. `n_bf_left`：后向填充后**还剩多少个缺失**

**考点**：pandas 3.0 起 `fillna(method="ffill")` 已移除，要用 `.ffill()` / `.bfill()`。
"""
    ),
    code(
        """# @@todo(6) 负荷值的前向/后向填充，以及各自填充后残余的缺失数
# @@hint Series.ffill() / Series.bfill() ；再用 .isna().sum() 复查
load_ff = dev["负荷值"].ffill()
load_bf = dev["负荷值"].bfill()
n_ff_left = int(load_ff.isna().sum())
n_bf_left = int(load_bf.isna().sum())
# @@end

# ---- 验收 ----
assert n_ff_left == 0, f"前向填完后不应还有缺失，实际 {n_ff_left}"
assert n_bf_left == 0, f"后向填完后不应还有缺失，实际 {n_bf_left}"
assert load_ff.iloc[0] == dev["负荷值"].iloc[0] or pd.isna(dev["负荷值"].iloc[0]), (
    "第一行前面没有值，前向填充不会改动第一行"
)
assert len(load_ff) == len(dev), "填充不改变行数"
print(f"TODO(6) 通过 -> ffill 残余 {n_ff_left} 个，bfill 残余 {n_bf_left} 个")
print("  注意：ffill 会把连续缺失段全部填成同一个值，长缺失段会被压成水平线")"""
    ),
    md(
        """
### TODO(7) 统计量填充：中位数

1. 计算出 `负荷值` 的**中位数**，存入 `load_median`
2. 用这个中位数填充缺失，结果存入 `load_fill_median`
3. 填入的值本身存入 `n_filled_median`（应该等于缺失数）

**注意**：这里必须先清异常值才谈得上"正确的均值"。本数据掺了 9999 哨兵值，
所以均值 379.68 完全不可用，**中位数 87.34 才是可用统计量**。
"""
    ),
    code(
        """# @@todo(7) 负荷值中位数 -> load_median；用它填充 -> load_fill_median；填充个数 -> n_filled_median
# @@hint Series.median() ；Series.fillna(值) ；填充个数 = 原缺失数 - 填后缺失数
load_median = dev["负荷值"].median()
load_fill_median = dev["负荷值"].fillna(load_median)
n_filled_median = int(dev["负荷值"].isna().sum() - load_fill_median.isna().sum())
# @@end

# ---- 验收 ----
assert abs(load_median - 87.34) < 1e-6, f"负荷值中位数应为 87.34，实际 {load_median}"
assert load_fill_median.isna().sum() == 0, "填充后不应还有缺失"
assert n_filled_median == 21, f"应填入 21 个值，实际 {n_filled_median}"
assert abs(load_fill_median.mean() - 356.9467) < 1e-3, (
    f"中位数填充后均值应约 356.95，实际 {load_fill_median.mean():.4f}"
)
print(f"TODO(7) 通过 -> median={load_median}, 填后 mean={load_fill_median.mean():.4f}")"""
    ),
    md(
        """
### TODO(8) 分组填充（MAR 的正确姿势）

按 `设备类型` 分组，用**组内中位数**填充 `负荷值`，结果存入 `load_fill_group`。

**这是本章最重要的一个坑**：不能写 `load.fillna(med_by_type)`——
那个 Series 的索引是"设备类型"，与行索引对不上，会**静默地一个都填不上**。

正确写法二选一：
- `dev.groupby("设备类型")["负荷值"].transform("median")`
- `dev["设备类型"].map(med_by_type)`
"""
    ),
    code(
        """med_by_type = dev.groupby("设备类型")["负荷值"].median()

# @@todo(8) 按设备类型分组，用组内中位数填充负荷值 -> load_fill_group
# @@hint 用 transform("median") 把组内统计量广播回每行，再 .fillna(...)
load_fill_group = dev["负荷值"].fillna(
    dev.groupby("设备类型")["负荷值"].transform("median")
)
# @@end

# ---- 验收 ----
assert load_fill_group.isna().sum() == 0, "分组填充后不应还有缺失"
mask_tr = dev["负荷值"].isna() & (dev["设备类型"] == "变压器")
assert mask_tr.sum() == 6, f"变压器组原本应缺 6 行，实际 {mask_tr.sum()}"
assert np.allclose(load_fill_group[mask_tr].unique(), [521.08]), (
    f"变压器组缺失行应被填成组内中位数 521.08，实际 {load_fill_group[mask_tr].unique()}"
)
print(f"TODO(8) 通过 -> 分组填充后 mean={load_fill_group.mean():.4f}")
print(f"  对比：全局中位数填 {load_fill_median.mean():.4f} <- 抹平了组间差异")"""
    ),
    md(
        """
### TODO(9) 时序插值（`load_curve.csv`）

按 `台区编号` 分组，对每组的 `负荷值` 做**线性插值**，结果存入 `curve_filled`（DataFrame）。

然后再算 A 台区（`STATION_A_01`）插值后的 `负荷值` 均值，存入 `a01_filled_mean`。

**为什么必须分组插值**：曲线是 3 个台区**首尾拼接**的，
不分组的话 A 台区的最后一段会和 C 台区的开头连起来插，插出跨台区的假数据。
"""
    ),
    code(
        """curve = pd.read_csv("data/load_curve.csv", parse_dates=["时间戳"])
print("load_curve.csv ->", curve.shape, "| 负荷值缺失:", int(curve["负荷值"].isna().sum()))
print("按台区缺失:", curve.groupby("台区编号")["负荷值"].apply(lambda s: int(s.isna().sum())).to_dict())


def _interp(s: pd.Series) -> pd.Series:
    \"\"\"对一组做线性插值（交互式定义，避免在挖空块里写 lambda 换行）。\"\"\"
    return s.interpolate(method="linear")


# @@todo(9) 按台区分组线性插值 -> curve_filled；A 台区插值后负荷均值 -> a01_filled_mean
# @@hint DataFrame.groupby(键)[列].transform(_interp) 赋回原表；再筛出台区求 .mean()
curve_filled = curve.copy()
curve_filled["负荷值"] = curve_filled.groupby("台区编号")["负荷值"].transform(_interp)
a01_filled_mean = curve_filled.loc[curve_filled["台区编号"] == "STATION_A_01", "负荷值"].mean()
# @@end

# ---- 验收 ----
assert curve_filled.shape == (6480, 5), f"形状不应改变，实际 {curve_filled.shape}"
assert curve_filled["负荷值"].isna().sum() == 0, "分组插值后不应还有缺失"
assert abs(a01_filled_mean - 660.22) < 0.05, f"A 台区插值后均值应约 660.22，实际 {a01_filled_mean:.2f}"
assert len(curve_filled) == len(curve), "行数不应改变"
print(f"TODO(9) 通过 -> 插值后总缺失 {curve_filled['负荷值'].isna().sum()}，A 台区均值 {a01_filled_mean:.2f}")"""
    ),
    md(
        """
---

## 综合题：从「判定机制」到「清洗报告」

| 步骤 | 产物 | 对应评分点 |
|---|---|---|
| ① 判定机制（MAR 实证） | `rate_cd` / `rate_other` | 数据清洗 · 策略选择依据 |
| ② 缺失指示列 | `dev2` 带 `温度_缺失` 列 | 数据清洗 · 特征构造 |
| ③ 组装清洗报告 | `report` | 数据探索 · 结论输出 |
"""
    ),
    md(
        """
#### 综合 ① 判定 `温度` 缺失的机制

把台区分成两组：`STATION_C_*` / `STATION_D_*` 为 `True`，其余 `False`。

1. 两组的温度缺失率（**百分数，保留 2 位**）分别存入 `rate_cd`、`rate_other`
2. 给出机制判定字符串存入 `temp_mechanism`，取值必须是 `"MCAR"` / `"MAR"` / `"MNAR"` 之一

判定规则：若两组缺失率**差距巨大**（此处 C/D 组 ≈ 23.7%、其他组 = 0%），
说明缺失与「台区」这个已观测变量强烈相关 → **MAR**。
"""
    ),
    code(
        """is_cd = dev["台区编号"].str.startswith(("STATION_C_", "STATION_D_"))
print("C/D 台区行数:", int(is_cd.sum()), "| 其他台区行数:", int((~is_cd).sum()))

# @@todo(10) C/D 组与其他组的温度缺失率(%) -> rate_cd, rate_other；机制 -> temp_mechanism
# @@hint boolean 掩码取值：dev.loc[掩码, "温度"].isna().mean()*100 ；机制填字符串字面量
rate_cd = round(dev.loc[is_cd, "温度"].isna().mean() * 100, 2)
rate_other = round(dev.loc[~is_cd, "温度"].isna().mean() * 100, 2)
temp_mechanism = "MAR"
# @@end

# ---- 验收 ----
assert abs(rate_cd - 23.71) < 0.02, f"C/D 组缺失率应约 23.71%，实际 {rate_cd}"
assert rate_other == 0.0, f"其他台区不应有温度缺失，实际 {rate_other}"
assert temp_mechanism in {"MCAR", "MAR", "MNAR"}, "机制取值必须是 MCAR / MAR / MNAR 之一"
assert temp_mechanism == "MAR", "缺失与已观测变量（台区）强相关 -> MAR"
print(f"综合 ① 通过 -> C/D 组 {rate_cd}% vs 其他组 {rate_other}% -> {temp_mechanism}")"""
    ),
    md(
        """
#### 综合 ② 缺失指示列

在 `dev` 的副本 `dev2` 上新增一列 `温度_缺失`，取值 0 / 1（整数）。

然后算出该列的求和，存入 `n_temp_flag`。
"""
    ),
    code(
        """# @@todo(11) 复制 dev -> dev2；新增整数型「温度_缺失」指示列；求和 -> n_temp_flag
# @@hint DataFrame.copy() ；Series.isna().astype(int) ；Series.sum()
dev2 = dev.copy()
dev2["温度_缺失"] = dev2["温度"].isna().astype(int)
n_temp_flag = int(dev2["温度_缺失"].sum())
# @@end

# ---- 验收 ----
assert "温度_缺失" in dev2.columns, "dev2 应该有「温度_缺失」列"
assert "温度_缺失" not in dev.columns, "不应该直接改原 dev（要用副本）"
assert dev2["温度_缺失"].dtype == np.int64 or str(dev2["温度_缺失"].dtype).startswith("int"), (
    f"指示列应该是整数型，实际 {dev2['温度_缺失'].dtype}"
)
assert n_temp_flag == 23, f"指示列之和应等于温度缺失数 23，实际 {n_temp_flag}"
assert set(dev2["温度_缺失"].unique()) <= {0, 1}, "指示列只能是 0 或 1"
print(f"综合 ② 通过 -> 指示列合计 {n_temp_flag}，取值集合 {sorted(dev2['温度_缺失'].unique())}")"""
    ),
    md(
        """
#### 综合 ③ 组装清洗报告

把这一章算出来的结论组装成 `report` 字典，键必须完全一致：

```
n_cols / n_rows / n_missing_cells / n_rows_any_na / n_rows_complete
worst_col / worst_col_missing / temp_mechanism
load_median / load_mean_after_fill
```

其中：
- `worst_col`：缺失最多的列名；`worst_col_missing`：它的缺失数
- `load_median`：`负荷值` 中位数
- `load_mean_after_fill`：用中位数填充后的整体均值
- 其余引用前面已算好的变量
"""
    ),
    code(
        """# @@todo(12) 组装 report 字典（9 个键）
# @@hint 键名必须完全一致；worst_col 用 miss.idxmax()；worst_col_missing 用 miss.max()
report = {
    "n_cols": dev.shape[1],
    "n_rows": dev.shape[0],
    "n_missing_cells": int(miss.sum()),
    "n_rows_any_na": n_rows_any_na,
    "n_rows_complete": n_rows_complete,
    "worst_col": miss.idxmax(),
    "worst_col_missing": int(miss.max()),
    "temp_mechanism": temp_mechanism,
    "load_median": load_median,
    "load_mean_after_fill": round(float(load_fill_median.mean()), 4),
}
# @@end

# ---- 验收 ----
expect_keys = {
    "n_cols", "n_rows", "n_missing_cells", "n_rows_any_na", "n_rows_complete",
    "worst_col", "worst_col_missing", "temp_mechanism",
    "load_median", "load_mean_after_fill",
}
assert isinstance(report, dict), "report 应该是字典"
assert set(report) == expect_keys, (
    f"键不对。多的是 {set(report) - expect_keys}，缺的是 {expect_keys - set(report)}"
)
assert report["n_cols"] == 13 and report["n_rows"] == 270
assert report["n_missing_cells"] == 240, "全表缺失总量应为 240"
assert report["n_rows_any_na"] == 211 and report["n_rows_complete"] == 59
assert report["worst_col"] == "备注" and report["worst_col_missing"] == 196
assert report["temp_mechanism"] == "MAR"
assert abs(report["load_median"] - 87.34) < 1e-6
assert abs(report["load_mean_after_fill"] - 356.9467) < 1e-3
assert report["n_rows_any_na"] + report["n_rows_complete"] == report["n_rows"], "行数应自洽"
print("综合 ③ 通过 -> 清洗报告：")
for k, v in report.items():
    print(f"  {k:22s} = {v!r}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（三题都能答上来才算过）：

1. `fillna(dev.groupby("设备类型")["负荷值"].median())` 为什么一个都填不上？怎么改？
2. 为什么这个数据的 `负荷值` 要用中位数填而不是均值？均值是多少、中位数是多少？
3. 时序连续缺失段，`ffill` 和 `interpolate` 的结果差在哪？为什么后者更适合负荷曲线？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
