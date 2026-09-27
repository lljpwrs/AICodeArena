"""ch03 —— 数据探索：把一份陌生数据在 5 分钟内摸透

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch03_explore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch03_explore"

HEADER = """import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

DATA = Path("data")
dev = pd.read_csv(DATA / "device_defects.csv")

print("pandas", pd.__version__)
print("读入 device_defects.csv ->", dev.shape)"""

LESSON = [
    md(
        """
# ch03 数据探索：把一份陌生数据在 5 分钟内摸透

> 方向：数据预处理 ｜ 竞赛对应：**数据探索（3 项，每项 2 分）** —— 这是竞赛里
> **明确独立计分**的第一组动作，属于「不做就没分」的送分题。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据探索 · 探索项 | 每项 2 分 | `shape` / `dtypes` / `info` / `describe` / `value_counts` |
| 数据探索 · 可视化项 | 每项 2 分 | 分布直方图、类别计数条形图（见 ch14） |
| 数据清洗的前置判断 | 5 项清洗各 2 分 | 靠本节先定位「缺失在哪一列」「重复有多少」「异常值在哪」 |

**为什么单独成章**：竞赛评分表把「数据探索」拆成独立小项，每项只值 2 分但**必拿**。
很多选手一上来直接 `model.fit()`，最后这些分全丢——不是不会，是没写。

> 标准动作清单（背下来）：
> `shape` → `dtypes` → `info()` → `head()` → `describe()` →
> `isna().sum()` → `duplicated().sum()` → `value_counts()` → `nunique()`
"""
    ),
    md(
        """
## 二、API 速查表

| API | 返回 | 一句话说明 |
|---|---|---|
| `df.shape` | `(行, 列)` | 规模 |
| `df.dtypes` | `Series` | 每列类型，判断有哪些列需要转换 |
| `df.dtypes.astype(str).value_counts()` | `Series` | 类型分布：几列数值、几列文本 |
| `df.info()` | 打印到 stdout | 一次性看：列名 / 非空计数 / dtype / 内存 |
| `df.info(memory_usage="deep")` | 打印 | 把字符串真实占用也算进去 |
| `df.head(n)` / `df.tail(n)` / `df.sample(n)` | `DataFrame` | 看头、看尾、随机抽样 |
| `df.describe()` | `DataFrame` | **只统计数值列**：count/mean/std/min/分位数/max |
| `df.describe(include="all")` | `DataFrame` | 数值列 + 类别列，类别列给 `unique/top/freq` |
| `df.describe(include="object")` | `DataFrame` | 只看类别列（pandas 3 里字符串列 dtype 是 `str`，用 `"object"` 可能取不到，见易错点） |
| `df["列"].value_counts(normalize=, dropna=, bins=)` | `Series` | 类别计数；`normalize` 出占比，`dropna=False` 保留缺失 |
| `df["列"].nunique(dropna=)` | `int` | 去重计数 |
| `df.isna()` / `.notna()` | `DataFrame[bool]` | 缺失掩码 |
| `df.isna().sum()` | `Series` | **逐列**缺失计数（最常用） |
| `df.isna().sum().sum()` | `int` | 全表缺失单元格总数 |
| `df.duplicated(subset=, keep=)` | `Series[bool]` | 重复行掩码 |
| `df.duplicated().sum()` | `int` | 完全重复行数 |
| `df.select_dtypes("number")` | `DataFrame` | 只取数值列 |
| `df.corr(numeric_only=True)` | `DataFrame` | 相关系数矩阵 |
| `df.memory_usage(deep=True)` | `Series` | 逐列内存占用（字节） |
| `df.nunique()` | `Series` | 每列去重计数，一眼看出哪些是分类列 |
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
### 3.1 先看形状和列名（30 秒）

`shape` + `columns` 是最廉价的信息：**行数决定要不要抽样，列数决定要不要减列。**
"""
    ),
    code(
        """print("shape :", dev.shape, "  <- (行, 列)")
print("行数  :", dev.shape[0])
print("列数  :", dev.shape[1], "（= len(dev.columns)）")
print("列名  :")
for i, col in enumerate(dev.columns, start=1):
    print(f"  {i:2d}. {col}")

# 索引默认是 RangeIndex；如果读进来时设过 index_col，这里会不一样
print("\\nindex 类型:", type(dev.index).__name__, "| 起点:", dev.index[0], "| 终点:", dev.index[-1])"""
    ),
    md(
        """
### 3.2 看数据长什么样（`head` / `tail` / `sample`）

三个都要看：

- `head()` 看表头对不对（有没有把标题行当表头）
- `tail()` 看文件有没有**尾部汇总行**（Excel 导出很常见）
- `sample()` 随机抽样，避免只看头部就被「数据是排过序的」骗了
"""
    ),
    code(
        """print("--- head(3) ---")
print(dev.head(3).to_string())

print("\\n--- tail(3)：检查有没有混入「合计」行 ---")
print(dev.tail(3)[["记录ID", "台区编号", "缺陷类型", "负荷值"]].to_string())

print("\\n--- sample(5, random_state=42)：随机抽样 ---")
print(dev.sample(5, random_state=42)[["台区编号", "设备类型", "缺陷类型", "负荷值"]].to_string())"""
    ),
    md(
        """
> **为什么要专门看 `tail()`**：Excel 导出的表经常在最后加一行「合计 / 小计」。
> 那行的 `台区编号` 是「合计」、`负荷值` 是一个大数 —— 一旦进入建模，
> 既污染类别列，又拉偏均值。用 `tail()` 一眼就能发现。
"""
    ),
    md(
        """
### 3.3 `dtypes` 与类型分布（决定后面要转哪些列）

pandas 3.0 起字符串列的 dtype 是 **`str`**，不再是 `object`。
这一点直接影响你写 `select_dtypes` 的方式。
"""
    ),
    code(
        """print("逐列 dtype：")
print(dev.dtypes.to_string())

print("\\n类型分布（每种 dtype 有几列）：")
print(dev.dtypes.astype(str).value_counts().to_string())

# 按类型分组取列名 —— 清洗前必做
num_cols = list(dev.select_dtypes("number").columns)
txt_cols = list(dev.select_dtypes(exclude="number").columns)
print("\\n数值列 (%d) :" % len(num_cols), num_cols)
print("文本列 (%d) :" % len(txt_cols), txt_cols)

print("\\n布尔列:", list(dev.select_dtypes("bool").columns))
print("日期列:", list(dev.select_dtypes("datetime").columns), "  <- 这里为空，说明发现日期还没解析成日期")"""
    ),
    md(
        """
### 3.4 `info()`：一次拿到「列名 + 非空数 + 类型 + 内存」

`info()` 是**性价比最高的单个调用**。它的 `Non-Null Count` 一列，
直接告诉你每列缺多少（`行数 - Non-Null`）。
"""
    ),
    code(
        """dev.info()

print("\\n上面 Non-Null Count 与 270 的差就是该列缺失数。")
print("例如「备注」：270 - %d = %d 个缺失" % (dev["备注"].notna().sum(), dev["备注"].isna().sum()))

# deep=True 会把字符串的真实占用算进去（默认只算指针大小）
print("\\n内存占用（deep=True）：")
mem = dev.memory_usage(deep=True)
print(mem.to_string())
print("合计：%.1f KB" % (mem.sum() / 1024))"""
    ),
    md(
        """
### 3.5 `describe()`：数值列的体检报告

`describe()` **默认只统计数值列**。这是新手最容易搞错的一点：
想看类别列，必须显式 `include="all"`。
"""
    ),
    code(
        """print("--- describe()：只有数值列 ---")
desc_num = dev.describe()
print(desc_num.to_string())
print("\\n形状:", desc_num.shape, " <- 8 行 x 4 个数值列")

print("\\n--- describe(include='all')：数值 + 类别 ---")
desc_all = dev.describe(include="all")
print("形状:", desc_all.shape, " <- 11 行 x 13 列")
print("\\n行名（数值统计量 + 类别统计量）:")
print(list(desc_all.index))

print("\\n类别列部分（unique / top / freq）:")
print(desc_all.loc[["count", "unique", "top", "freq"], txt_cols].to_string())"""
    ),
    md(
        """
> **`describe(include="all")` 的多出来三行是什么意思**：
>
> | 行 | 含义 | 用途 |
> |---|---|---|
> | `unique` | 该列有几个不同取值 | 判断是不是「几乎每行都不同」的 ID 列 |
> | `top` | 出现最多的取值 | 快速发现主导类别 |
> | `freq` | `top` 出现了几次 | `freq` 接近 `count` → 这一列几乎没有信息量 |
"""
    ),
    md(
        """
### 3.6 `value_counts()`：类别分布 + 一个容易被忽略的参数

`dropna=False` 这个参数值得单独记：**缺失值本身就是一个「类别」。**
只看 `dropna=True` 的分布，你会以为这一列很干净。
"""
    ),
    code(
        """print("--- 默认（dropna=True）---")
print(dev["缺陷类型"].value_counts().to_string())

print("\\n--- normalize=True：出占比 ---")
vc_norm = dev["缺陷类型"].value_counts(normalize=True)
print((vc_norm * 100).round(2).head(5).to_string(), "  (单位 %)")

print("\\n--- 处理状态：注意同一含义有多种写法 ---")
print(dev["处理状态"].value_counts(dropna=False).to_string())

print("\\n--- dropna=False：缺失值也是一类 ---")
remark_vc = dev["备注"].value_counts(dropna=False)
print(remark_vc.to_string())

print("\\n--- bins：把数值列当类别分箱统计 ---")
print(dev["负荷值"].value_counts(bins=5, sort=False).to_string())"""
    ),
    md(
        """
> **`处理状态` 这一列就是竞赛里的真实脏数据**：`已处理` / `已 处理` / `处理完成` / `done`
> 是同一个意思，`未处理` / `待处理` 也是同一个意思，末尾还带空格的。
> `value_counts()` 是发现这类问题的**第一现场**——一个业务上只有 3 类的字段，
> 这里冒出 8 个取值，立刻就知道要标准化（ch06 讲）。
"""
    ),
    md(
        """
### 3.7 `nunique()`：一眼分辨「分类列」和「ID 列」

经验法则：`nunique` 接近行数 → 是 ID 类，建模时直接丢；
`nunique` 很小 → 是分类列，可以做分组/编码。
"""
    ),
    code(
        """nu = dev.nunique().sort_values()
print("逐列去重计数（升序）：")
print(nu.to_string())

print("\\n行数 =", len(dev))
print("判定（nunique / 行数）：")
ratio = (nu / len(dev)).round(3)
print(ratio.to_string())

# nunique 的 dropna 参数同样重要
print("\\n备注列 nunique：dropna=True ->", dev["备注"].nunique(),
      "| dropna=False ->", dev["备注"].nunique(dropna=False))"""
    ),
    md(
        """
### 3.8 缺失与重复：清洗的第一份「作战地图」

这两步是所有清洗动作的前置。**先量化，再动手**——
不然你永远不知道自己的清洗有没有效果。
"""
    ),
    code(
        """# ① 逐列缺失
miss = dev.isna().sum()
print("逐列缺失：")
print(miss[miss > 0].sort_values(ascending=False).to_string())
print("\\n全表缺失单元格总数:", int(miss.sum()))
print("只在个别列缺失的列数:", int((miss > 0).sum()))

# ② 缺失率百分比（写报告要用）
miss_rate = (miss / len(dev) * 100).round(2)
print("\\n缺失率 > 0 的列：")
print(miss_rate[miss_rate > 0].to_string())

# ③ 重复行
print("\\n完全重复行数:", int(dev.duplicated().sum()))

# 业务键重复：记录ID 不同但业务内容相同
biz_key = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]
print("业务键重复行数:", int(dev.duplicated(subset=biz_key).sum()))
print("业务键重复行数（keep='last'）:", int(dev.duplicated(subset=biz_key, keep="last").sum()))
print("业务键重复行数（keep=False，全部标出）:", int(dev.duplicated(subset=biz_key, keep=False).sum()), "  <- 重复组的所有成员")"""
    ),
    md(
        """
> `duplicated(keep=...)` 的三种取值，考试和实战都常考：
>
> | `keep` | 含义 | `sum()` 结果代表 |
> |---|---|---|
> | `"first"`（默认） | 保留第一次出现，后面标 True | **应删除的行数** |
> | `"last"` | 保留最后一次出现 | 应删除的行数（另一侧） |
> | `False` | 重复组**所有成员**都标 True | 涉及重复的行总数（≈ 2 倍） |
"""
    ),
    md(
        """
### 3.9 均值 vs 中位数：发现异常值的第一个信号

这是**不画图就能嗅出异常值**的方法：

```
若 mean 远大于 median  ->  右侧有极端大值（尖峰/哨兵码）
若 mean 远小于 median  ->  左侧有极端小值（负数/哨兵码）
```

我们的 `负荷值` 里掺了 `9999` 哨兵值和 `5000` 尖峰，来看它有多明显：
"""
    ),
    code(
        """load = dev["负荷值"]
print("count  :", int(load.count()))
print("mean   :", round(load.mean(), 4))
print("median :", round(load.median(), 4))
print("std    :", round(load.std(), 4))
print("min    :", load.min(), "| max:", load.max(), "  <- max 明显不合理")

print("\\nmean / median =", round(load.mean() / load.median(), 2), " <- 差了 4 倍以上，必然有极端值")

# 换个角度看分位数：50% 和 75% 之间只有几度，99% 却跳了一个量级
print("\\n分位数：")
print(load.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]).round(2).to_string())

# 把哨兵值拿掉再看
normal = load[load < 1000]
print("\\n剔除 >= 1000 的值后：")
print("  count :", int(normal.count()), "（被剔掉 %d 个）" % (int(load.count() - normal.count())))
print("  mean  :", round(normal.mean(), 4), " <- 从 379.68 掉到 79 附近")
print("  median:", round(normal.median(), 4), " <- 中位数几乎没变，这就是中位数抗异常值的原因")"""
    ),
    md(
        """
> **这条规律在竞赛里非常实用**：`describe()` 输出的 `mean` 和 `50%`（中位数）
> 差得离谱 → 立刻去查 `max` 和 `min`，十有八九是哨兵值（`9999` / `-1` / `999`）
> 或者量纲不统一（有的是 kW，有的是 W）。
"""
    ),
    md(
        """
### 3.10 数值列相关性：`corr`

```python
df.corr(numeric_only=True)
```

- pandas 3.0 里**必须**加 `numeric_only=True`（或先 `select_dtypes("number")`），
  否则字符串列会报错
- 相关系数只反映**线性**关系，且**会被异常值严重污染**——先看图或先清异常值再信它
"""
    ),
    code(
        """num = dev.select_dtypes("number")
corr = num.corr(numeric_only=True)
print("相关系数矩阵：")
print(corr.round(3).to_string())

# 只看「负荷值」相关的强度，排个序
print("\\n与「负荷值」的相关性（去掉自身）：")
print(corr["负荷值"].drop("负荷值").sort_values(ascending=False).round(3).to_string())

# 剔除哨兵值后，相关性会变（说明这个矩阵确实被极端值污染了）
clean_num = dev.loc[dev["负荷值"] < 1000].select_dtypes("number")
print("\\n剔除哨兵值后，负荷值 vs 温度的相关系数：")
print("  含哨兵值 :", round(corr.loc["负荷值", "温度"], 3))
print("  剔除后   :", round(clean_num.corr(numeric_only=True).loc["负荷值", "温度"], 3))"""
    ),
    md(
        """
### 3.11 把上面所有动作打包成「数据体检报告」

竞赛上机时间紧，**手写一遍探索清单很浪费时间**。把标准动作封装成一个函数，
每次换数据集只改一行路径——这是最划算的工程投入。
"""
    ),
    code(
        """def health_report(df: pd.DataFrame, name: str = "dataset", top_n: int = 5) -> dict:
    \"\"\"对一份 DataFrame 做标准探索，返回结构化结论。

    对应竞赛「数据探索」的 3 个计分小项：
      ① 规模与结构  ② 类型与缺失  ③ 类别分布
    \"\"\"
    n_rows, n_cols = df.shape
    miss = df.isna().sum()
    num_cols = list(df.select_dtypes("number").columns)
    txt_cols = list(df.select_dtypes(exclude="number").columns)

    report = {
        "名称": name,
        "行数": n_rows,
        "列数": n_cols,
        "数值列数": len(num_cols),
        "文本列数": len(txt_cols),
        "全表缺失单元格": int(miss.sum()),
        "有缺失的列数": int((miss > 0).sum()),
        "缺失最多的列": (miss.idxmax() if miss.max() > 0 else None),
        "缺失最多列的缺失数": int(miss.max()),
        "完全重复行数": int(df.duplicated().sum()),
        "内存_MB": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
    }

    print(f"{'=' * 56}")
    print(f"数据体检报告：{name}")
    print(f"{'=' * 56}")
    print(f"规模        : {n_rows} 行 x {n_cols} 列"
          f"（数值 {len(num_cols)} / 文本 {len(txt_cols)}）")
    print(f"缺失        : 全表 {report['全表缺失单元格']} 个单元格，"
          f"涉及 {report['有缺失的列数']} 列")
    if report["缺失最多的列"]:
        print(f"              最多的是「{report['缺失最多的列']}」"
              f"（{report['缺失最多列的缺失数']} 个）")
    print(f"完全重复行  : {report['完全重复行数']}")
    print(f"内存占用    : {report['内存_MB']} MB")

    print("\\n-- 数值列五数概括 --")
    if num_cols:
        print(df[num_cols].describe().loc[["count", "mean", "50%", "min", "max"]].round(2).to_string())

    print(f"\\n-- 类别列 Top{top_n} --")
    for col in txt_cols:
        nu = df[col].nunique()
        if nu <= 1:
            continue
        top = df[col].value_counts(dropna=False).head(top_n)
        pretty = ", ".join(f"{k!r}:{v}" for k, v in top.items())
        print(f"  {col:8s} ({nu:3d} 类)  {pretty}")

    print(f"{'=' * 56}")
    return report


report = health_report(dev, name="device_defects.csv")
print("\\n返回结构可用作程序化判断，例如：")
print("  缺失列数 =", report["有缺失的列数"], "| 重复行 =", report["完全重复行数"])"""
    ),
    md(
        """
> 这个函数在后面的每一章都会被复用。**把它抄进你的竞赛模板文件**，
> 上机第一件事就是 `health_report(pd.read_csv(path))`，
> 「数据探索」那几项分数直接落袋。
"""
    ),
    md(
        """
---

## 四、易错点清单

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | `describe()` 以为能看到类别列 | 默认只有数值列，要 `include="all"` |
| 2 | 用 `select_dtypes("object")` 找文本列 | pandas 3.0 字符串列 dtype 是 `str`，改用 `exclude="number"` |
| 3 | `value_counts()` 忘看 `dropna=False` | 缺失本身是一大类，漏看会低估问题 |
| 4 | `nunique()` 默认不计缺失 | 想知道「含缺失有几种取值」要 `dropna=False` |
| 5 | `duplicated()` 直接当「要删的行数」 | `keep="first"` 才是；`keep=False` 会翻倍 |
| 6 | `corr()` 不加 `numeric_only=True` | 字符串列会直接报错 |
| 7 | 只看 `head()` 不看 `tail()` | 尾部「合计」行会让统计全错 |
| 8 | 看到 `mean` 异常高就以为数据分布偏态 | 先查 `max`/`min`，多半是哨兵值 |
| 9 | `info()` 默认不显示字符串真实内存 | 大文本列要看真实占用用 `memory_usage(deep=True)` |
| 10 | 探索完不记录结论 | 竞赛按步骤给分，**写了才算**；把关键数字复制到报告里 |

---

## 五、本章小结

```
标准探索顺序（照着敲就有分）：

  shape -> dtypes -> info() -> head()/tail()/sample()
        -> describe(include="all") -> isna().sum() -> duplicated().sum()
        -> value_counts(dropna=False) -> nunique() -> corr(numeric_only=True)

三个「一看就懂」的异常信号：
  ① mean 远大于 median        -> 右侧极端值 / 哨兵码
  ② describe 里 freq ≈ count  -> 这一列没信息量
  ③ 一个 3 类的业务字段 nunique=8 -> 类别写法没统一

封装：把上面全套写成一个 health_report(df) 函数，每次换数据只改路径。
```

**下一步**：做 `ch03_explore_practice.ipynb`。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch03 练习：数据探索

> 共 8 道基础题 + 1 道综合题（拆 3 小步，合计 12 个空）
> 建议限时：**25 分钟**

数据：`data/device_defects.csv`（270 行 x 13 列，含缺失、重复、异常值）

**这章练的是「不漏步」**。每道题都对应竞赛评分表里一个 2 分小项，
写不出来就是那 2 分没了。
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
print("读入 ->", dev.shape)"""
    ),
    md(
        """
---

### TODO(1) 规模：行数与列数

分别取出行数、列数存入 `n_rows`、`n_cols`。
（不要写死 270 / 13，要用属性取。）
"""
    ),
    code(
        """# @@todo(1) 取行数存入 n_rows，列数存入 n_cols
# @@hint DataFrame.shape 返回 (行, 列)
n_rows = dev.shape[0]
n_cols = dev.shape[1]
# @@end

# ---- 验收 ----
assert n_rows == 270, f"行数应为 270，实际 {n_rows}"
assert n_cols == 13, f"列数应为 13，实际 {n_cols}"
assert n_cols == len(dev.columns), "列数应该等于列名个数"
print("TODO(1) 通过 ->", n_rows, "行 x", n_cols, "列")"""
    ),
    md(
        """
### TODO(2) 类型分布 + 按类型分组取列

1. 统计每种 dtype 各有几列，存入 `dtype_counts`（按数量降序的 Series）
2. 取数值列名列表存入 `num_cols`，文本列名列表存入 `txt_cols`

**注意**：pandas 3.0 里字符串列的 dtype 是 `str` 而不是 `object`。
想「取到所有非数值列」，用 `exclude="number"` 最稳。
"""
    ),
    code(
        """# @@todo(2) 统计 dtype 分布存入 dtype_counts；取出数值列名与文本列名
# @@hint Series.value_counts() ；DataFrame.select_dtypes("number") / select_dtypes(exclude="number")
dtype_counts = dev.dtypes.astype(str).value_counts()
num_cols = list(dev.select_dtypes("number").columns)
txt_cols = list(dev.select_dtypes(exclude="number").columns)
# @@end

# ---- 验收 ----
assert dtype_counts.to_dict() == {"str": 9, "float64": 3, "int64": 1}, (
    f"类型分布不对：{dtype_counts.to_dict()}"
)
assert len(num_cols) == 4, f"数值列应为 4 列，实际 {len(num_cols)}：{num_cols}"
assert len(txt_cols) == 9, f"文本列应为 9 列，实际 {len(txt_cols)}：{txt_cols}"
assert set(num_cols) == {"记录ID", "负荷值", "温度", "湿度"}, "数值列清单不对"
print("TODO(2) 通过 -> dtype 分布:", dtype_counts.to_dict())"""
    ),
    md(
        """
### TODO(3) `describe`：数值列体检

1. 对数值列做 `describe()`，结果存入 `desc_num`
2. 从里面取出 `负荷值` 列的**中位数**（`50%` 那一行），存入 `load_median`
"""
    ),
    code(
        """# @@todo(3) 数值列 describe() 存入 desc_num；再取出负荷值的中位数存入 load_median
# @@hint DataFrame.describe() ；用 .loc["50%", "负荷值"] 取值
desc_num = dev.describe()
load_median = desc_num.loc["50%", "负荷值"]
# @@end

# ---- 验收 ----
assert desc_num.shape == (8, 4), f"describe() 形状应为 (8, 4)，实际 {desc_num.shape}"
assert list(desc_num.index) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"], "统计量行名不对"
assert abs(load_median - 87.34) < 1e-6, f"负荷值中位数应为 87.34，实际 {load_median}"
print("TODO(3) 通过 -> 负荷值中位数 =", load_median)"""
    ),
    md(
        """
### TODO(4) `describe(include="all")`：把类别列也统计进来

对全部列做 `describe(include="all")`，存入 `desc_all`，
然后取出 **`缺陷类型`** 这一列的 `unique` 和 `top`，分别存入 `defect_unique`、`defect_top`。
"""
    ),
    code(
        """# @@todo(4) describe(include="all") 存入 desc_all；取缺陷类型的 unique 与 top
# @@hint DataFrame.describe(include="all") ；用 desc_all.loc["unique", "缺陷类型"] 取值
desc_all = dev.describe(include="all")
defect_unique = desc_all.loc["unique", "缺陷类型"]
defect_top = desc_all.loc["top", "缺陷类型"]
# @@end

# ---- 验收 ----
assert desc_all.shape == (11, 13), f"形状应为 (11, 13)，实际 {desc_all.shape}"
assert defect_unique == 10, f"缺陷类型应有 10 种写法，实际 {defect_unique}"
assert defect_top == "发热", f"出现最多的缺陷类型应为「发热」，实际 {defect_top}"
print("TODO(4) 通过 -> 缺陷类型 %d 种，最多的是 %r" % (defect_unique, defect_top))"""
    ),
    md(
        """
### TODO(5) 缺失普查

1. 逐列缺失计数存入 `miss`（Series）
2. 只保留缺失数 > 0 的列，按缺失数**降序**存入 `miss_sorted`
3. 全表缺失单元格总数存入 `n_missing_cells`

**坑位**：`isna().sum()` 给的是逐列结果，别把链式调用写反。
"""
    ),
    code(
        """# @@todo(5) 逐列缺失 -> miss；有缺失的列降序 -> miss_sorted；全表缺失总数 -> n_missing_cells
# @@hint df.isna().sum() ；再 .loc[lambda s: s > 0] 或用布尔索引筛；最后 .sort_values(ascending=False)
miss = dev.isna().sum()
miss_sorted = miss[miss > 0].sort_values(ascending=False)
n_missing_cells = int(miss.sum())
# @@end

# ---- 验收 ----
assert len(miss) == 13, "miss 应该是「逐列」的，长度等于列数"
assert miss_sorted.to_dict() == {"备注": 196, "温度": 23, "负荷值": 21}, (
    f"有缺失的列不对：{miss_sorted.to_dict()}"
)
assert n_missing_cells == 240, f"全表缺失单元格应为 240，实际 {n_missing_cells}"
print("TODO(5) 通过 -> 缺失列:", miss_sorted.to_dict())"""
    ),
    md(
        """
### TODO(6) 重复行普查

1. 完全重复行数存入 `n_dup`
2. 按业务键（`台区编号`+`线路名称`+`设备类型`+`设备编号`+`缺陷类型`+`缺陷等级`+`发现日期`）
   去重后的重复行数存入 `n_dup_biz`

**注意**：业务键重复里既有「完全重复行」，也有「内容几乎相同但 `记录ID`/`备注` 不同」的行。
"""
    ),
    code(
        """biz_key = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]

# @@todo(6) 完全重复行数 -> n_dup；按 biz_key 的重复行数 -> n_dup_biz
# @@hint df.duplicated().sum() ；df.duplicated(subset=列名列表).sum()
n_dup = int(dev.duplicated().sum())
n_dup_biz = int(dev.duplicated(subset=biz_key).sum())
# @@end

# ---- 验收 ----
assert n_dup == 6, f"完全重复行应为 6，实际 {n_dup}"
assert n_dup_biz == 10, f"业务键重复行应为 10，实际 {n_dup_biz}"
assert n_dup_biz > n_dup, "业务重复应该比完全重复多（多出的是记录ID不同的那些）"
print(f"TODO(6) 通过 -> 完全重复 {n_dup} 行，业务重复 {n_dup_biz} 行")"""
    ),
    md(
        """
### TODO(7) `value_counts(dropna=False)`：类别分布

对 `处理状态` 列做 `value_counts(dropna=False)`，存入 `status_vc`。

这题的重点是**看清同一业务含义有几种写法**：
`已处理` / `已 处理` / `处理完成` / `done` 其实是同一类。
"""
    ),
    code(
        """# @@todo(7) 对「处理状态」做 value_counts（保留缺失），存入 status_vc
# @@hint Series.value_counts(dropna=False)
status_vc = dev["处理状态"].value_counts(dropna=False)
# @@end

# ---- 验收 ----
assert len(status_vc) == 8, f"处理状态应出现 8 种写法，实际 {len(status_vc)}"
assert status_vc.index[0] == "已处理", "出现最多的应该还是标准写法「已处理」"
assert set(status_vc.index) >= {"已处理", "已 处理", "处理完成", "done", "未处理", "待处理"}, "少了几种写法"
print("TODO(7) 通过 -> 该列 NunIque 应为 8：")
print(status_vc.to_string())"""
    ),
    md(
        """
### TODO(8) 均值 vs 中位数：嗅出异常值

对 `负荷值` 列分别算 `mean` 和 `median`，存入 `load_mean`、`load_med_raw`。

然后**剔除 >= 1000 的极端值**后再算一次中位数，存入 `load_med_clean`。

（提示：`mean` 会被 `9999` 哨兵值严重拉高，`median` 几乎不受影响。）
"""
    ),
    code(
        """# @@todo(8) 负荷值的均值与中位数；以及剔除极端值后的中位数
# @@hint Series.mean() / Series.median() ；先用布尔条件筛选再算中位数
load_mean = dev["负荷值"].mean()
load_med_raw = dev["负荷值"].median()
load_med_clean = dev.loc[dev["负荷值"] < 1000, "负荷值"].median()
# @@end

# ---- 验收 ----
assert load_mean > 300, f"均值应被哨兵值拉高到 300 以上，实际 {load_mean:.2f}"
assert load_med_raw < 100, f"中位数应远小于均值，实际 {load_med_raw:.2f}"
assert abs(load_med_raw - load_med_clean) < 5, (
    f"中位数几乎不受极端值影响，两者应很接近：{load_med_raw:.2f} vs {load_med_clean:.2f}"
)
assert load_mean / load_med_raw > 3, "均值/中位数应该差 3 倍以上"
print(f"TODO(8) 通过 -> mean={load_mean:.2f}  median={load_med_raw:.2f}  "
      f"clean_median={load_med_clean:.2f}")"""
    ),
    md(
        """
---

## 综合题：把探索动作打包进一份「体检报告」

目标：写出一段能在**任何** DataFrame 上复用、并且**逐项对应竞赛计分点**的探索代码。

| 步骤 | 产物 | 对应评分点 |
|---|---|---|
| ① 缺失总览 | `miss_total` | 数据探索 · 缺失量 |
| ② 主导类别 | `top_defect` | 数据探索 · 类别分布 |
| ③ 组装报告 | `report` | 数据探索 · 结论输出 |
"""
    ),
    md("#### 综合 ① 缺失总览"),
    code(
        """# @@todo(9) 全表缺失单元格总数存入 miss_total
# @@hint df.isna().sum() 给逐列，再 .sum() 得总数
miss_total = int(dev.isna().sum().sum())
# @@end

# ---- 验收 ----
assert miss_total == 240, f"缺失单元格总数应为 240，实际 {miss_total}"
print("综合 ① 通过 -> 全表缺失 =", miss_total)"""
    ),
    md("#### 综合 ② 主导类别"),
    code(
        """# @@todo(10) 缺陷类型出现次数最多的那个取值，存入 top_defect
# @@hint Series.value_counts().index[0]
top_defect = dev["缺陷类型"].value_counts().index[0]
# @@end

# ---- 验收 ----
assert top_defect == "发热", f"出现最多的缺陷类型应为「发热」，实际 {top_defect}"
assert isinstance(top_defect, str), "应该是字符串取值"
print("综合 ② 通过 -> 主导缺陷类型 =", top_defect)"""
    ),
    md(
        """
#### 综合 ③ 组装报告字典

把上面算出来的各项指标组装成一个 `report` 字典，键必须是：

```
n_rows / n_cols / n_num_cols / n_text_cols / n_missing_cells / n_dup_rows / top_defect
```
"""
    ),
    code(
        """# @@todo(11) 组装 report 字典（7 个键，值引用上面已算好的变量或重新计算）
# @@hint dict 字面量；键名必须完全一致
report = {
    "n_rows": n_rows,
    "n_cols": n_cols,
    "n_num_cols": len(num_cols),
    "n_text_cols": len(txt_cols),
    "n_missing_cells": miss_total,
    "n_dup_rows": n_dup,
    "top_defect": top_defect,
}
# @@end

# ---- 验收 ----
expect_keys = {
    "n_rows", "n_cols", "n_num_cols", "n_text_cols",
    "n_missing_cells", "n_dup_rows", "top_defect",
}
assert isinstance(report, dict), "report 应该是字典"
assert set(report) == expect_keys, f"键不对，多的是 {set(report) - expect_keys}，缺的是 {expect_keys - set(report)}"
assert report["n_rows"] == 270 and report["n_cols"] == 13
assert report["n_num_cols"] == 4 and report["n_text_cols"] == 9
assert report["n_num_cols"] + report["n_text_cols"] == report["n_cols"], "两类列数加起来应等于总列数"
assert report["n_missing_cells"] == 240, "缺失总量不对"
assert report["n_dup_rows"] == 6, "重复行数不对"
assert report["top_defect"] == "发热", "主导类别不对"
print("综合 ③ 通过 ->")
for k, v in report.items():
    print(f"  {k:18s} = {v!r}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**：如果现在给你一份全新的数据，你能不能在 3 分钟内
不看笔记把上面 8 个探索动作默写出来？默不出来就说明还没形成肌肉记忆——
回 `_notes/错题本.md` 记一笔，明天再默一次。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
