"""ch02 —— 数据读写：read_csv / read_excel 与导出规范

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch02_read_write.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch02_read_write"

HEADER = """import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 20)

DATA = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

print("pandas", pd.__version__)
print("数据目录内容:", sorted(p.name for p in DATA.iterdir()))"""

LESSON = [
    md(
        """
# ch02 数据读写：`read_csv` / `read_excel` 与导出规范

> 方向：数据预处理 ｜ 竞赛对应：**数据准备及处理 10%**（第一步，读错了后面全错）
> 本节目标：把「读进来」和「导出去」这两件事做到**可控、可复现、不丢精度**。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据准备及处理（10%） | 整条链路的入口 | 正确读取 CSV / Excel，指定编码、类型、列 |
| 数据清洗（5 项，各 2 分） | 第 1 项通常就是「读取并检查数据」 | 用 `dtype` / `na_values` 在读入阶段就统一缺失表示 |
| 结果提交 | 提交格式错 = 白做 | 用 `to_csv(index=False, encoding="utf-8-sig")` 规范导出 |

**竞赛里最常见的 3 种「一上来就翻车」**：

1. 文件是 GBK 编码，默认 utf-8 读 → `UnicodeDecodeError`
2. 数值列里混着 `N/A`、`—`、空串三种缺失写法，只当一种处理 → 统计口径错
3. 做完导出的 CSV 多了一列无名索引列，评审脚本读进来列数不符
"""
    ),
    md(
        """
## 二、API 速查表

### 2.1 `pd.read_csv()` 核心参数

| 参数 | 作用 | 竞赛里的典型用法 |
|---|---|---|
| `sep` | 分隔符 | 默认 `,`；制表符用 `sep="\\t"` |
| `encoding` | 编码 | 中文 CSV 优先试 `utf-8-sig`，报错就换 `gbk` |
| `dtype` | 强制列类型 | `dtype={"设备编号": "str"}` —— 防止 `001` 变成 `1` |
| `na_values` | 追加缺失标记 | `na_values=["N/A", "—", "", "无"]` |
| `keep_default_na` | 是否保留内置缺失词表 | 配合 `na_values` 精调时用 |
| `usecols` | 只读需要的列 | 大文件省内存，也顺手完成「选列」 |
| `nrows` | 只读前 n 行 | 快速摸底，避免全量读 |
| `skiprows` / `header` / `names` | 跳过/指定表头 | 报表型 CSV（前几行是标题）必用 |
| `parse_dates` | 解析日期列 | `parse_dates=["时间戳"]` |
| `index_col` | 指定索引列 | `index_col="记录ID"` |
| `thousands` / `decimal` | 千分位 / 小数点符号 | `thousands=","` |
| `chunksize` | 分块读取 | 返回 `TextFileReader`，可迭代 |
| `on_bad_lines` | 坏行处理 | `"skip"` / `"warn"` / `callable` |

### 2.2 `to_csv()` 关键参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `index` | 是否写索引 | **一律写 `index=False`** |
| `encoding` | 编码 | **`"utf-8-sig"`** —— 带 BOM，Windows Excel 打开不乱码 |
| `na_rep` | 缺失值写什么 | 默认空串；要显式标记写 `na_rep="NULL"` |
| `float_format` | 浮点格式 | `"%.2f"` 控制小数位 |
| `columns` | 只导出部分列 | 提交前裁剪字段 |

### 2.3 `read_excel()` / `to_excel()`

| 参数 | 作用 |
|---|---|
| `sheet_name` | 工作表名或序号；**`sheet_name=None` 读回全部 → 返回 dict** |
| `dtype` / `na_values` / `usecols` / `nrows` | 同 `read_csv` |
| `to_excel(..., index=False)` | 不加会多一列索引 |
| `ExcelWriter` + `to_excel` | 一个文件写多个 sheet |
"""
    ),
    md(
        """
---

## 三、逐节讲解

### 3.0 环境与路径准备

后面每一节的代码都复用这一段（`DATA` 指向数据目录，`OUT_DIR` 放临时产物）：

> `output/` 已写进 `.gitignore`，练习产生的中间文件不会污染仓库。
"""
    ),
    code(HEADER),
    md(
        """
### 3.1 `encoding`：中文 CSV 的第一道坎

同一份数据，编码不同就是两个世界：

- `utf-8` —— 跨平台通用，但**不带 BOM**
- `utf-8-sig` —— 带 BOM，Windows Excel 双击打开不乱码
- `gbk` / `gb18030` —— 国产报表系统导出的事实标准，用 utf-8 读**直接抛异常**

先感受一下这个报错：
"""
    ),
    code(
        """# ① 用默认编码（utf-8）读 GBK 文件
try:
    pd.read_csv(DATA / "device_defects_gbk.csv")
except UnicodeDecodeError as exc:
    print("① 默认编码读取 ->", type(exc).__name__)
    print("   ", str(exc)[:100])

# ② 指定 encoding 后正常
gbk = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gbk")
print("\\n② encoding='gbk' ->", gbk.shape)

# ③ 更稳的写法：gb18030 是 GBK 的超集，兼容面更广
gb18030 = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gb18030")
print("③ encoding='gb18030' ->", gb18030.shape)

# ④ 不指定编码时的「试探」写法（竞赛里省时间）
for enc in ("utf-8-sig", "gbk", "gb18030", "latin-1"):
    try:
        n = len(pd.read_csv(DATA / "device_defects_gbk.csv", encoding=enc))
        print(f"   {enc:10s} 可读，行数 {n}")
    except (UnicodeDecodeError, ValueError):
        print(f"   {enc:10s} 失败")"""
    ),
    md(
        """
> **为什么导出要用 `utf-8-sig`**：`utf-8-sig` 在文件开头写一个不可见的 BOM 标记。
> Windows 版 Excel 靠它判断「这是 UTF-8」，没有 BOM 就会按 GBK 解释 → 中文乱码。
> Linux/macOS 下读 `utf-8-sig` 文件也能正常工作，所以**导出统一用它没有副作用**。
>
> 反过来说：如果读到的第一列列名带 `\\ufeff` 前缀，就是「BOM 没被剥掉」的典型症状。
"""
    ),
    md(
        """
### 3.2 `dtype`：防止「看起来像数字的编号」被吃掉

设备编号、台区编号这类字段，一旦被推断成整数，前导零就没了，
`001` 和 `1` 会合并成同一类 —— 分组统计直接错。
"""
    ),
    code(
        """# 造一个带前导零的小 CSV，看默认推断有多危险
demo = pd.DataFrame({"设备编号": ["001", "002", "010"], "负荷值": [1.0, 2.0, 3.0]})
demo.to_csv(OUT_DIR / "leading_zero.csv", index=False, encoding="utf-8-sig")

auto = pd.read_csv(OUT_DIR / "leading_zero.csv")
forced = pd.read_csv(OUT_DIR / "leading_zero.csv", dtype={"设备编号": "str"})
print("默认推断   ->", auto["设备编号"].tolist(), "| dtype:", auto["设备编号"].dtype)
print("强制 str   ->", forced["设备编号"].tolist(), "| dtype:", forced["设备编号"].dtype)

# 全列统一为 str（摸原始数据时最常用）
all_str = pd.read_csv(OUT_DIR / "leading_zero.csv", dtype=str)
print("\\ndtype=str   ->", all_str["负荷值"].tolist(), "| dtype:", all_str["负荷值"].dtype)"""
    ),
    md(
        """
### 3.3 `na_values`：把多种缺失写法一次性归一

这是**竞赛里最值钱的一个参数**。真实导出的表里，缺失从来不止一种写法：

| 写法 | pandas 默认行为 |
|---|---|
| 空单元格 | ✅ 识别为 `NaN` |
| `N/A`、`NA`、`null`、`NaN` | ✅ 识别为 `NaN` |
| `—`（破折号）、`无`、`/`、`-`、`未填` | ❌ **当成正常字符串** |

我们的 `device_defects_raw.csv` 故意把这三种都用上了，来看差别：
"""
    ),
    code(
        """# 默认读：只有 N/A 与空串被识别，"—" 被当成字符串
default = pd.read_csv(DATA / "device_defects_raw.csv", dtype={"负荷值": "str"})
s = default["负荷值"]
print("默认读取：")
print("  isna() 个数          :", int(s.isna().sum()))
print("  等于 '—' 的个数       :", int((s == "—").sum()), "  <- 漏网的缺失")

# 加上 na_values 一次性归一
normalized = pd.read_csv(
    DATA / "device_defects_raw.csv",
    dtype={"负荷值": "str"},
    na_values=["N/A", "NA", "—", "-", "", "无", "/"],
)
print("\\n加 na_values 后：")
print("  isna() 个数          :", int(normalized["负荷值"].isna().sum()), "  <- 三种写法全部归位")
print("  '—' 残留             :", int((normalized["负荷值"] == "—").sum()))

# 注意：na_values 是「追加」到内置词表，不是替换
# 想完全自己说了算，用 keep_default_na=False
custom = pd.read_csv(
    DATA / "device_defects_raw.csv",
    dtype={"负荷值": "str"},
    na_values=["—"],
    keep_default_na=False,   # 此时 "N/A" 反而不再被当缺失
)
print("\\nkeep_default_na=False 后：")
print("  isna() 个数          :", int(custom["负荷值"].isna().sum()))
print("  'N/A' 残留           :", int((custom["负荷值"] == "N/A").sum()), "  <- 变成普通字符串了")"""
    ),
    md(
        """
> **一句话记住**：`na_values` **追加**缺失词表；`keep_default_na=False` **清空**内置词表。
> 竞赛里安全做法是 `na_values=[...]`（追加）+ 不动 `keep_default_na`。
"""
    ),
    md(
        """
### 3.4 `usecols` / `nrows` / `skiprows`：别把时间花在读文件上

- `usecols` 在**读取阶段**就丢掉无关列 —— 比读进来再 `drop` 省内存得多
- `nrows` 是摸底神器：只读 100 行就能判断 `dtype`、表头、分隔符对不对
- 报表型 CSV 前几行常是标题，用 `skiprows` + `header` 定位真表头
"""
    ),
    code(
        """# ① usecols + nrows：先摸底
peek = pd.read_csv(DATA / "device_defects.csv", usecols=["记录ID", "台区编号", "负荷值"], nrows=5)
print("① usecols + nrows ->", peek.shape)
print(peek)

# ② usecols 可以是「排除法」，用 callable
not_remark = pd.read_csv(
    DATA / "device_defects.csv",
    usecols=lambda c: c not in {"备注", "发现日期"},
    nrows=3,
)
print("\\n② usecols 排除法 ->", list(not_remark.columns))

# ③ skiprows / header：处理「前几行是标题」的报表型 CSV
report_text = (
    "设备缺陷统计报表,,\\n"
    "生成时间 2026-09-01,,\\n"
    "台区编号,缺陷类型,缺陷数\\n"
    "STATION_A_01,渗漏油,3\\n"
    "STATION_B_02,锈蚀,1\\n"
)
(OUT_DIR / "report_style.csv").write_text(report_text, encoding="utf-8-sig")

print("③ 原始文件前 3 行：")
print("\\n".join((OUT_DIR / "report_style.csv").read_text(encoding="utf-8-sig").splitlines()[:3]))

parsed = pd.read_csv(OUT_DIR / "report_style.csv", skiprows=2, encoding="utf-8-sig")
print("\\n③ skiprows=2 后 ->")
print(parsed)
print("   columns:", list(parsed.columns))"""
    ),
    md(
        """
### 3.5 `parse_dates` / `index_col`：读进来就是可分析的时间序列

时间序列题的效率差异，一半来自「读的时候有没有把时间列转好」。
"""
    ),
    code(
        """# ① 不解析：时间列是字符串
naive = pd.read_csv(DATA / "load_curve.csv", nrows=3)
print("① 不解析 -> 时间戳 dtype:", naive["时间戳"].dtype)

# ② parse_dates：一次到位
parsed = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"])
print("② parse_dates -> 时间戳 dtype:", parsed["时间戳"].dtype)

# ③ parse_dates + index_col：直接拿到 DatetimeIndex
ts = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"], index_col="时间戳")
print("③ index_col   -> index 类型:", type(ts.index).__name__, "| dtype:", ts.index.dtype)
print("   shape:", ts.shape, "（原 5 列，时间列变成索引后剩 4 列）")
print("   不同时刻数:", ts.index.nunique(), "= 90 天 x 24 小时")

# 注意：这张表是 3 个台区「首尾相接」堆叠的，索引整体上不单调
print("   索引是否单调递增:", ts.index.is_monotonic_increasing, " <- False，多台区堆叠的正常现象")
grouped = ts.groupby("台区编号")["负荷值"].resample("ME").mean().round(2)
print("\\n   先按台区分组再按月重采样（后面 ch09/ch12 详讲）:")
print(grouped.to_string())"""
    ),
    md(
        """
> **顺序问题**：`parse_dates` 和 `index_col` 可以同时指定同一列，
> pandas 会先解析再设为索引，不用手动 `set_index`。
> 但**不要**先把时间列设成索引再指望 `parse_dates` 救回来——索引一旦建成字符串索引就晚了。
"""
    ),
    md(
        """
### 3.6 `thousands` / `decimal`：千分位与小数点

从 Excel / 财务系统导出的数值常带千分位 `1,234.50`。
只要整列都是这种格式，`thousands=","` 就能直接读成数字。
"""
    ),
    code(
        """# 整列都是千分位格式 —— thousands 管用
plain = pd.DataFrame({"台区编号": ["STATION_A_01", "STATION_B_02"], "负荷值": ["1,234.50", "999.80"]})
plain.to_csv(OUT_DIR / "thousands.csv", index=False, encoding="utf-8-sig")

bad = pd.read_csv(OUT_DIR / "thousands.csv")
good = pd.read_csv(OUT_DIR / "thousands.csv", thousands=",")
print("不加 thousands ->", bad["负荷值"].tolist(), "| dtype:", bad["负荷值"].dtype)
print("加 thousands   ->", good["负荷值"].tolist(), "| dtype:", good["负荷值"].dtype)

# 但如果同一列里还混着单位文字（"12.50 kW"），thousands 也救不了 —— 只能当字符串读出来手工清洗
mixed = pd.read_csv(DATA / "device_defects_raw.csv", dtype={"负荷值": "str"}, thousands=",")
print("\\n混单位的列      -> dtype:", mixed["负荷值"].dtype, "（仍是字符串，留给 ch06 清洗）")
print("  示例值:", mixed["负荷值"].head(6).tolist())"""
    ),
    md(
        """
### 3.7 `chunksize`：大文件分块读

竞赛给的训练集动辄几百 MB。`chunksize` 返回一个可迭代对象，每次吐一个块：

```python
reader = pd.read_csv(path, chunksize=100_000)
for chunk in reader:
    ...  # 这里做聚合，不要 list(reader) 全读进内存
```
"""
    ),
    code(
        """# 分块统计：逐块聚合，最后合并
n_chunks = 0
row_count = 0
load_sum = 0.0
for chunk in pd.read_csv(DATA / "device_defects.csv", chunksize=100):
    n_chunks += 1
    row_count += len(chunk)
    load_sum += chunk["负荷值"].sum()

print(f"分了 {n_chunks} 块，共 {row_count} 行")
print(f"分块求和负荷值 = {load_sum:.4f}")

# 与一次性读取对比，验证一致
whole = pd.read_csv(DATA / "device_defects.csv")
print(f"一次性求和     = {whole['负荷值'].sum():.4f}")
print("两者一致：", abs(load_sum - whole["负荷值"].sum()) < 1e-9)"""
    ),
    md(
        """
### 3.8 导出：`to_csv` 的三条铁律

```python
df.to_csv(path, index=False, encoding="utf-8-sig", na_rep="")
```

| 铁律 | 原因 |
|---|---|
| `index=False` | 否则多一列无名索引，评审脚本按列名取值时会全错位 |
| `encoding="utf-8-sig"` | 让 Windows Excel 打开不乱码 |
| `na_rep` 想清楚 | 空字符串 vs `NULL` vs `NaN`，下游脚本可能只认一种 |
"""
    ),
    code(
        """sample = whole.head(5)[["记录ID", "台区编号", "负荷值", "备注"]]

# 反面：全都用默认值
sample.to_csv(OUT_DIR / "export_bad.csv", encoding="utf-8")
bad_back = pd.read_csv(OUT_DIR / "export_bad.csv")
print("反面导出后读回的列:", list(bad_back.columns), "  <- 多了一列 Unnamed: 0")

# 正面：标准写法
sample.to_csv(OUT_DIR / "export_good.csv", index=False, encoding="utf-8-sig", na_rep="")
good_back = pd.read_csv(OUT_DIR / "export_good.csv")
print("标准导出后读回的列:", list(good_back.columns))
print("缺失值读回后个数:", int(good_back["备注"].isna().sum()))

# 显式标记缺失 + 控制小数位
sample.to_csv(
    OUT_DIR / "export_na.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="NULL",
    float_format="%.2f",
)
print("\\nna_rep='NULL' 导出后读回（注意 NULL 被 pandas 自动识别成 NaN）:")
print(pd.read_csv(OUT_DIR / "export_na.csv")["备注"].tolist())
print("  想看原始文本就加 keep_default_na=False:")
print(pd.read_csv(OUT_DIR / "export_na.csv", keep_default_na=False)["备注"].tolist())"""
    ),
    md(
        """
### 3.9 Excel 读写

```python
pd.read_excel(path, sheet_name=..., dtype=..., usecols=..., nrows=...)
df.to_excel(path, index=False, sheet_name="汇总")
```

- `sheet_name=None` → 返回 `dict[str, DataFrame]`，一次读全部工作表
- 一个文件写多表：用 `pd.ExcelWriter` 作为上下文管理器
"""
    ),
    code(
        """# 单表读取
summary_xlsx = pd.read_excel(DATA / "station_summary.xlsx")
print("单表读取 ->", summary_xlsx.shape, list(summary_xlsx.columns))

# sheet_name=None 读全部工作表
book = pd.read_excel(DATA / "station_summary.xlsx", sheet_name=None)
print("\\nsheet_name=None -> 返回类型:", type(book).__name__, "| 工作表:", list(book))

# 写多表
with pd.ExcelWriter(OUT_DIR / "multi_sheet.xlsx") as writer:
    whole.head(20).to_excel(writer, sheet_name="样本", index=False)
    whole.groupby("设备类型")["负荷值"].mean().round(2).to_excel(writer, sheet_name="设备类型均值")
print("\\n多表写入完成：", (OUT_DIR / "multi_sheet.xlsx").exists())

back = pd.read_excel(OUT_DIR / "multi_sheet.xlsx", sheet_name="设备类型均值")
print("读回第 2 张表:")
print(back.to_string())"""
    ),
    md(
        """
---

## 四、易错点清单

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | 中文文件读不出来只想到 `errors="ignore"` | 先试 `encoding="utf-8-sig"`，再试 `"gbk"` / `"gb18030"` |
| 2 | 导出不加 `index=False` | 永远加上，否则多一列无名索引 |
| 3 | 编号列被推断成整数，前导零丢失 | `dtype={"编号列": "str"}` |
| 4 | 只把空串当缺失，漏掉 `—` / `无` / `NA` | `na_values=[...]` 追加缺失词表 |
| 5 | 用 `keep_default_na=False` 后忘了 `N/A` 也不再生效 | 除非确有需要，不要动这个参数 |
| 6 | 时间列读成字符串，后面 `resample` 报错 | `parse_dates=[...]` + `index_col=...` 一起给 |
| 7 | `chunksize` 结果直接 `list()` 转列表 | 那样等于一次性读入内存，失去分块意义 |
| 8 | 读到的第一个列名带 `\\ufeff` | 文件是 UTF-8 无 BOM 但被当 utf-8-sig 写了 / 反过来，改 `encoding` |
| 9 | `to_excel` 忘记 `index=False` | 同 CSV，会多一列 |
| 10 | 大文件用 `nrows` 看了几行就下结论 | 抽样要覆盖文件尾部：`skiprows` 跳到末尾再看 |

---

## 五、本章小结

```
读进来 ->  encoding 先定  ->  dtype 锁类型  ->  na_values 归缺失  ->  usecols 减列
导出去 ->  index=False  ->  encoding="utf-8-sig"  ->  na_rep 想清楚

不确定文件长什么样时，永远先：
    pd.read_csv(path, nrows=100)
```

**下一步**：做 `ch02_read_write_practice.ipynb`。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch02 练习：数据读写

> 共 7 道基础题 + 1 道综合题（拆 3 小步，合计 10 个空）
> 建议限时：**25 分钟**

**所有题目都基于 `data/` 下的真实文件**，读不通就是真的读不通。
"""
    ),
    code(
        """import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 130)

DATA = Path("data")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

print("数据目录:", sorted(p.name for p in DATA.iterdir()))"""
    ),
    md(
        """
---

### TODO(1) 读 GBK 编码的文件

`device_defects_gbk.csv` 是 GBK 编码。请把它读进来，变量名 `gbk`。

先试一下不加 `encoding` 参数会怎样，再加参数。
"""
    ),
    code(
        """# @@todo(1) 指定正确的编码，把 GBK 文件读成 DataFrame（变量名 gbk）
# @@hint pd.read_csv(路径, encoding=...) ；中文文件优先试 utf-8-sig，报错换 gbk / gb18030
gbk = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gbk")
# @@end

# ---- 验收 ----
assert isinstance(gbk, pd.DataFrame), "gbk 应该是 DataFrame"
assert gbk.shape == (270, 13), f"形状应为 (270, 13)，实际 {gbk.shape}"
assert "缺陷类型" in gbk.columns, "列名乱码了，说明编码不对"
assert gbk["台区编号"].iloc[0].startswith("STATION_"), "内容乱码"
print("TODO(1) 通过 ->", gbk.shape)"""
    ),
    md(
        """
### TODO(2) 用 `dtype` 锁住类型

读 `device_defects_raw.csv`，要求 **所有列都读成字符串**（`dtype=str`），
变量名 `raw_str`。

这样做的意义：原始文件里的 `负荷值` 混着单位、千分位、缺失占位符，
交给 pandas 猜类型只会得到一堆意外。
"""
    ),
    code(
        """# @@todo(2) 读取 raw 文件，所有列强制为字符串，变量名 raw_str
# @@hint pd.read_csv(路径, dtype=str)
raw_str = pd.read_csv(DATA / "device_defects_raw.csv", dtype=str)
# @@end

# ---- 验收 ----
assert raw_str.shape == (270, 13), f"形状应为 (270, 13)，实际 {raw_str.shape}"
assert all(str(dt) == "str" for dt in raw_str.dtypes), "还有列不是字符串类型"
assert raw_str["负荷值"].str.contains("kW", na=False).sum() == 5, "带单位的行数不对，检查是不是被转成数值了"
print("TODO(2) 通过 -> 全部列 dtype 均为 str")"""
    ),
    md(
        """
### TODO(3) `usecols` + `nrows` 快速摸底

只想看 `记录ID`、`台区编号`、`缺陷类型` 三列的**前 8 行**，变量名 `peek`。

要求用 `usecols` 和 `nrows` **在读取阶段**就裁掉多余数据。
"""
    ),
    code(
        """# @@todo(3) 用 usecols + nrows 只读三列前 8 行，变量名 peek
# @@hint pd.read_csv(路径, usecols=[列名列表], nrows=8)
peek = pd.read_csv(DATA / "device_defects.csv", usecols=["记录ID", "台区编号", "缺陷类型"], nrows=8)
# @@end

# ---- 验收 ----
assert peek.shape == (8, 3), f"形状应为 (8, 3)，实际 {peek.shape}"
assert list(peek.columns) == ["记录ID", "台区编号", "缺陷类型"], "列不对"
print("TODO(3) 通过 ->\\n", peek.head(3))"""
    ),
    md(
        """
### TODO(4) `na_values` 统一缺失写法

`device_defects_raw.csv` 的 `负荷值` 列里有 **3 种**缺失写法混用：`N/A`、`—`、空串。
请只用一次 `read_csv`，把它们全部识别为缺失值，变量名 `normalized`。

读完后 `负荷值` 列的缺失个数应该是 **21**。
"""
    ),
    code(
        """# @@todo(4) 用 na_values 把 3 种缺失写法一次归一（配合 dtype=str）
# @@hint pd.read_csv(路径, dtype=str, na_values=["N/A", "—", "", ...])
normalized = pd.read_csv(
    DATA / "device_defects_raw.csv",
    dtype=str,
    na_values=["N/A", "NA", "—", "-", "", "无", "/"],
)
# @@end

# ---- 验收 ----
assert int(normalized["负荷值"].isna().sum()) == 21, (
    f"缺失个数应为 21，实际 {int(normalized['负荷值'].isna().sum())}，有缺失写法漏网了"
)
assert int((normalized["负荷值"] == "—").sum()) == 0, "还有 '—' 残留在值里"
print("TODO(4) 通过 -> 负荷值缺失 =", int(normalized["负荷值"].isna().sum()))"""
    ),
    md(
        """
### TODO(5) `parse_dates` + `index_col`

读 `load_curve.csv`，要求：
- `时间戳` 列**解析为 datetime**
- `时间戳` 同时作为 **索引**
- 结果存入 `ts`

读完后 `ts.index` 应该是 `DatetimeIndex`。
"""
    ),
    code(
        """# @@todo(5) 读 load_curve.csv，把「时间戳」解析为日期并设为索引，存入 ts
# @@hint pd.read_csv(路径, parse_dates=["时间戳"], index_col="时间戳")
ts = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"], index_col="时间戳")
# @@end

# ---- 验收 ----
assert isinstance(ts.index, pd.DatetimeIndex), f"索引类型不对：{type(ts.index).__name__}"
assert ts.shape == (6480, 4), f"形状应为 (6480, 4)，实际 {ts.shape}（原 5 列，时间列已变索引）"
assert ts.index.nunique() == 2160, f"不同时刻数应为 2160（90 天 x 24 小时），实际 {ts.index.nunique()}"
print("TODO(5) 通过 ->", ts.shape, "| 时间范围:", ts.index.min(), "~", ts.index.max())
print("  注意：现在是 3 个台区堆叠，索引不是单调递增的，做 resample 前要先 sort_index")"""
    ),
    md(
        """
### TODO(6) `chunksize` 分块聚合

用 `chunksize=100` 分块读 `device_defects.csv`，
在不把整个文件读进内存的前提下统计出：
- `n_chunks`：一共几块
- `row_count`：总行数

## 坑位提示

`pd.read_csv(..., chunksize=...)` **不直接返回 DataFrame**，而是一个可迭代的读取器。
"""
    ),
    code(
        """n_chunks = 0
row_count = 0

# @@todo(6) 用 chunksize=100 分块读取，得到可迭代的读取器，变量名 reader
# @@hint pd.read_csv(路径, chunksize=100) 返回 TextFileReader
reader = pd.read_csv(DATA / "device_defects.csv", chunksize=100)
# @@end

for chunk in reader:
    n_chunks += 1
    row_count += len(chunk)

# ---- 验收 ----
assert n_chunks == 3, f"块数应为 3，实际 {n_chunks}"
assert row_count == 270, f"总行数应为 270，实际 {row_count}"
print(f"TODO(6) 通过 -> {n_chunks} 块，共 {row_count} 行")"""
    ),
    md(
        """
### TODO(7) 规范导出

把 `df_src` 的前 20 行导出一个 CSV 到 `output/answer_ch02.csv`，要求：

1. **不写索引列**
2. 编码用 **`utf-8-sig`**（Windows Excel 打开不乱码）
3. 缺失值统一写成 `NULL`

导完自己读回来验证一下。
"""
    ),
    code(
        """df_src = pd.read_csv(DATA / "device_defects.csv").head(20)
out_path = OUT_DIR / "answer_ch02.csv"

# @@todo(7) 按要求把 df_src 导出到 out_path
# @@hint DataFrame.to_csv(路径, index=..., encoding=..., na_rep=...)
df_src.to_csv(out_path, index=False, encoding="utf-8-sig", na_rep="NULL")
# @@end

# ---- 验收 ----
assert out_path.exists(), "文件没有生成"
raw_bytes = out_path.read_bytes()
assert raw_bytes[:3] == b"\\xef\\xbb\\xbf", "缺少 UTF-8 BOM，编码不是 utf-8-sig"
assert b"Unnamed" not in raw_bytes, "写出了索引列"
back = pd.read_csv(out_path)
assert back.shape == (20, 13), f"读回来形状应为 (20, 13)，实际 {back.shape}"
text = out_path.read_text(encoding="utf-8-sig")
assert "NULL" in text, "缺失值没有被写成 NULL"
print("TODO(7) 通过 -> 已写出", out_path)"""
    ),
    md(
        """
---

## 综合题：把「读进来 → 精简 → 导出去」走完整

目标是**生成一份干净的、可交付的 CSV**。请完成 3 个小步。

| 步骤 | 产物 | 说明 |
|---|---|---|
| ① 正确读入 | `src` | GBK 编码 + 编号列锁为字符串 |
| ② 读入时精简 | `slim` | 只要 4 列 |
| ③ 规范导出 | `output/final_ch02.csv` | 无索引、utf-8-sig、缺失写 NULL |
"""
    ),
    md("#### 综合 ① 正确读入（编码 + 类型）"),
    code(
        """# @@todo(8) 读 GBK 文件，把「台区编号」「设备编号」强制为字符串，存入 src
# @@hint pd.read_csv(路径, encoding=..., dtype={"列名": "str"})
src = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gbk", dtype={"台区编号": "str", "设备编号": "str"})
# @@end

# ---- 验收 ----
assert src.shape == (270, 13), f"形状应为 (270, 13)，实际 {src.shape}"
assert src["设备编号"].dtype == "str", "设备编号没有锁成字符串"
assert src["台区编号"].iloc[0].startswith("STATION_"), "编码不对导致内容乱码"
print("综合 ① 通过 ->", src.shape)"""
    ),
    md("#### 综合 ② 读取阶段就精简列"),
    code(
        """keep_cols = ["记录ID", "台区编号", "设备类型", "负荷值"]

# @@todo(9) 用 usecols 只读 keep_cols 这 4 列，存入 slim
# @@hint pd.read_csv(路径, usecols=列名列表)
slim = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gbk", usecols=keep_cols)
# @@end

# ---- 验收 ----
assert list(slim.columns) == keep_cols, "列不对"
assert slim.shape == (270, 4), f"形状应为 (270, 4)，实际 {slim.shape}"
print("综合 ② 通过 ->", slim.shape)"""
    ),
    md("#### 综合 ③ 规范导出并回读校验"),
    code(
        """final_path = OUT_DIR / "final_ch02.csv"

# @@todo(10) 把 slim 规范导出到 final_path：不写索引、utf-8-sig、缺失写 NULL
# @@hint DataFrame.to_csv(路径, index=False, encoding="utf-8-sig", na_rep="NULL")
slim.to_csv(final_path, index=False, encoding="utf-8-sig", na_rep="NULL")
# @@end

# ---- 验收 ----
assert final_path.exists(), "文件没有生成"
assert final_path.read_bytes()[:3] == b"\\xef\\xbb\\xbf", "编码不是 utf-8-sig"
round_trip = pd.read_csv(final_path)
assert round_trip.shape == (270, 4), f"回读形状应为 (270, 4)，实际 {round_trip.shape}"
assert list(round_trip.columns) == keep_cols, "回读列名不对，可能写出了索引列"
print("综合 ③ 通过 -> 交付文件:", final_path)"""
    ),
    md(
        """
---

### 复盘提问

**竞赛复盘问自己**：如果这次数据换成 GBK 编码，我的脚本会在第几行崩？
把答案记进 `_notes/错题本.md`。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
