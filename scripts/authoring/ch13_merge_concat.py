"""ch13 —— 连接与合并：`merge` / `join` / `concat` / `merge_asof`

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch13_merge_concat.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch13_merge_concat"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

# 主表：缺陷记录（270 行，18 个台区，每台区多行）
dev = pd.read_csv(DATA / "device_defects.csv")
# 维表：设备台账（16 行，故意少 2 个台区）
info = pd.read_csv(DATA / "device_info.csv")
# 汇总表：台区日均负荷（18 行，与主表台区完全对齐）
summary = pd.read_csv(DATA / "station_summary.csv")

KEY = "台区编号"

# 维表换个中文列名，练「不同列名」的连接
info_cn = info.rename(columns={"容量kVA": "容量"})

# 容量限值调整表：每半月调一次，给 merge_asof 用
limit = pd.DataFrame(
    {
        "生效时间": pd.to_datetime(
            ["2026-01-01", "2026-01-16", "2026-02-01", "2026-02-16", "2026-03-01", "2026-03-16"]
        ),
        "限值": [1000, 1050, 1100, 1120, 1150, 1180],
    }
).sort_values("生效时间").reset_index(drop=True)

# 负荷曲线：取单台区、按时间升序（merge_asof 要求左表键有序）
curve = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"])
hourly = (
    curve[curve[KEY] == "STATION_A_01"]
    .sort_values("时间戳")
    .reset_index(drop=True)
)

print("pandas", pd.__version__)
print("dev", dev.shape, "| info", info.shape, "| summary", summary.shape, "| hourly", hourly.shape)'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def merge_report(result, base_rows):
    """算一条校验公式：结果行数相对左表的膨胀倍数。"""
    return round(len(result) / base_rows, 4)


def match_rate(result, col):
    """右表列的非空率——判断连接命中情况最快的方式。"""
    return round(float(result[col].notna().mean()), 4)


def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    """
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def audit_merge(left, right, key, probe_col):
    """对一次 many_to_one 的 left 连接做质量审计。

    返回形状 / 行数守恒 / 匹配率 / 是否低于 0.95 阈值。
    """
    result = left.merge(right, on=key, how="left", validate="many_to_one")
    rate = round(float(result[probe_col].notna().mean()), 4)
    return {
        "形状": result.shape,
        "行数守恒": len(result) == len(left),
        "匹配率": rate,
        "低命中": rate < 0.95,
    }


print("dev 台区数:", dev[KEY].nunique(), "| info 台区数:", info[KEY].nunique())'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch13 连接与合并：`merge` / `join` / `concat` / `merge_asof`

> 方向：数据预处理 ｜ 竞赛对应：**数据准备及处理**（宽表拼装）+ **数据探索**
> —— 竞赛给的数据永远是"主表 + 若干维表"，
> 能不能把台账、汇总、限值几张表干净地拼到主表上，几乎决定了后面所有步骤。

本章要解决的四个静默错误：

1. **键"看着一样"但匹配 0 行**——多一个空格、大小写不同、类型不同
2. **一对多被当成一对一**——行数悄悄膨胀，均值/计数全错
3. **`concat` 默认保留原索引**——索引重复后 `loc` 一次返回多行
4. **`merge_asof` 的 `direction`** ——`backward` / `forward` 语义相反，选错就是"用了未来数据"
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据准备及处理 | 占技能操作 10% | 主表 + 维表拼装成宽表 |
| 数据清洗 | 每项 2 分 | 连接后的缺失识别与填充决策 |
| 数据探索 | 每项 2 分 | 用非空率反查连接是否命中 |
| 结果交付 | 提交格式错 = 白做 | 索引、列序、列名后缀的整理 |

**为什么单独成章**：连接是**唯一一类"错了也能出结果"的操作**。
筛选错会少行、聚合错会少组，这些还能靠行数对账发现；
但连接错只会让某个字段**整列变成 `NaN`**，或者让行数**多出几十行**——
而"多出几十行"在一张 270 行的表里非常容易被忽略，
直到你发现某组的均值怎么算都对不上。
"""
    ),
    md(
        """
## 二、API 速查表

### 2.1 三个入口的分工

| 入口 | 连接键 | 适用场景 |
|---|---|---|
| `pd.merge(left, right, on=...)` | **列** | 最通用，键是普通列 |
| `left.merge(right, on=...)` | **列** | 同上，链式写法 |
| `left.join(right, on=..., how=...)` | **索引** 或 `on=` 指定的列 | 右侧已 `set_index` |
| `pd.concat([a, b], axis=0)` | **无键，按位置** | 纵向堆叠（追加行） |
| `pd.concat([a, b], axis=1)` | **按索引标签** | 横向拼接（加列） |
| `pd.merge_asof(left, right, on=...)` | **最近键** | 时序最近匹配（生效时间、报价、限值） |

### 2.2 `how` 的四种语义（本数据实测）

| `how` | 保留谁 | 本数据形状 | 语义 |
|---|---|---|---|
| `"inner"` | 两边都有的键 | `(247, 17)` | 只留匹配上的 |
| `"left"` | 左表全部 | `(270, 17)` | **最常用**，右表填 `NaN` |
| `"right"` | 右表全部 | `(247, 17)` | 等价于左右互换的 `left` |
| `"outer"` | 两边并集 | `(270, 17)` | 全保留（此处右表无独有键，所以等于 `left`） |

### 2.3 关键参数

| 参数 | 取值 | 作用 |
|---|---|---|
| `on` | 列名 / 列表 | 连接键（两表同名） |
| `left_on` / `right_on` | 列名 | 两表键名不同时 |
| `left_index` / `right_index` | `bool` | 用索引当键 |
| `suffixes` | `("_左", "_右")` | **非键**同名列的后缀 |
| `validate` | `"one_to_one"` / `"one_to_many"` / `"many_to_one"` / `"many_to_many"` | **连接前强制校验基数**，不合规直接报错 |
| `indicator` | `True` / 字符串 | 加一列标注每行来自哪边 |
| `sort` | `bool` | 按键排序（默认 `False`） |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 三、逐节讲解

### 3.1 心智模型：连接是"按键对齐"，不是"按位置拼接"

```text
dev（主表，270 行，18 个台区，每台区多行）
  └─ 台区编号 ─┐
              ├─ merge on="台区编号" ─▶ 宽表
info（维表，16 行，每台区 1 行）
  └─ 台区编号 ─┘
```

**基数（cardinality）是连接的唯一核心概念**：

| 左表基数 | 右表基数 | 结果行数 | 说明 |
|---|---|---|---|
| 多 | 一 | **= 左表行数** | `many_to_one`，最常见（主表 + 维表） |
| 一 | 多 | **= 右表行数** | `one_to_many` |
| 一 | 一 | **= 键的个数** | `one_to_one` |
| 多 | 多 | **= 笛卡尔积** ⚠️ | `many_to_many`，行数会爆 |

本数据的基数是：`dev` 的 `台区编号` **多**（270 行 / 18 台区）、
`info` 的 `台区编号` **一**（16 行 / 16 台区）→ 所以 `how="left"` 的
结果行数应该**恰好等于 270**。这就是第一条校验公式。
"""
    ),
    code(
        """print(f"{'how':<8}{'结果形状':>14}{'行数是否 == 左表':>18}")
print("-" * 42)
for how in ["inner", "left", "right", "outer"]:
    result = dev.merge(info, on=KEY, how=how)
    print(f"{how:<8}{str(result.shape):>14}{str(len(result) == len(dev)):>18}")

print("\\n左表台区数:", dev[KEY].nunique(), "| 维表台区数:", info[KEY].nunique())
missing = sorted(set(dev[KEY]) - set(info[KEY]))
print("主表有、维表没有的台区:", missing)
print("→ 这", len(missing), "个台区的", len(dev[dev[KEY].isin(missing)]), "行，在 inner 连接里被丢掉了")
print("   inner:", len(dev.merge(info, on=KEY, how="inner")),
      "| left:", len(dev.merge(info, on=KEY, how="left")),
      f"| 差值:", len(dev) - len(dev.merge(info, on=KEY, how="inner")))"""
    ),
    md(
        """
**结论**：`inner` 悄悄丢了 **23** 行（`STATION_B_03` + `STATION_F_02` 的所有记录）。
如果要交一份"每台区统计表"，用 `inner` 会让这两个台区凭空消失——
而题目很可能要求"18 个台区都要有"。

> **判定规则**：**主表 + 维表一律用 `how="left"`**，
> 只在明确知道"业务上就该只保留双方都有的"时才用 `inner`。
> 而且用 `left` 之后要**主动检查 `NaN` 数**（见 §3.2）。
"""
    ),
    md(
        """
### 3.2 难点深挖①：键"看着一样"，却匹配 0 行

**为什么难**：这是连接里最贵的一个坑。`merge` 不报错、不警告、
结果表**列数不变、行数不变**，只有你新加的那一列**整列是 `NaN`**。
而 `NaN` 又很容易被归咎于"数据本身缺失"。

**错误示范**：把维表的键前面多加一个空格（模拟 Excel 里手打的隐形空白）
"""
    ),
    code(
        """dirty_key = info.copy()
dirty_key[KEY] = " " + dirty_key[KEY].astype(str)

clean_result = dev.merge(info, on=KEY, how="left")
dirty_result = dev.merge(dirty_key, on=KEY, how="left")

print(f"{'版本':<16}{'形状':>12}{'匹配行数':>10}{'匹配率':>10}")
print("-" * 48)
for name, result in [("干净键", clean_result), ("多一个空格", dirty_result)]:
    print(f"{name:<16}{str(result.shape):>12}"
          f"{int(result['所属供电所'].notna().sum()):>10}"
          f"{match_rate(result, '所属供电所'):>10.4f}")

print("\\n→ 形状完全一样 (270, 17)，只有匹配率从 0.9148 掉到 0.0")
print("  如果不看匹配率，这个错误会被当成「维表数据缺失」放过")

print("\\n三种典型脏键及修法:")
cases = {
    "首尾空白": lambda s: " " + s + " ",
    "全角空格": lambda s: "\\u3000" + s,
    "大小写": lambda s: s.str.lower(),
}
for label, fn in cases.items():
    broken = info.copy()
    broken[KEY] = fn(broken[KEY])
    r = dev.merge(broken, on=KEY, how="left")
    print(f"  {label:<8} 匹配行数 = {int(r['所属供电所'].notna().sum()):<5}")

print("\\n修法：连接前统一做一遍键规范化 —— "
      "`.astype(str).str.strip().str.replace('\\u3000','').str.upper()`")"""
    ),
    md(
        """
**正误对照**：

| 版本 | 形状 | 匹配行数 | 匹配率 |
|---|---|---|---|
| 干净键 | `(270, 17)` | **247** | **0.9148** |
| 键多一个空格 | `(270, 17)` | **0** | **0.0000** |
| 键有全角空格 | `(270, 17)` | 0 | 0.0 |
| 键被转小写 | `(270, 17)` | 0 | 0.0 |

**4 种情况形状全是 `(270, 17)`**——光看形状一个都发现不了。

> **判定规则（连接的铁律）**：
> **连接后立刻算一遍右表列的非空率**：
> ```python
> result["右表列"].notna().mean()      # 正常应该 ≈ 左表命中率
> ```
> 数字明显偏低 → 先查键的规范性（`strip` / 全角空格 / 大小写 / dtype），
> **不要**先去填缺失值。填完了照样对不上。
>
> 连接前的键规范化一行搞定：
> ```python
> for frame in (dev, info):
>     frame[KEY] = frame[KEY].astype(str).str.strip().str.replace("\\u3000", "").str.upper()
> ```
"""
    ),
    md(
        """
### 3.3 难点深挖②：一对多被当成一对一 → 行数膨胀

**为什么难**：当**右表的键有重复**时，`merge` 会把左表每一行与
右表所有匹配行**两两配对**（笛卡尔积）。行数膨胀，但**不报错**。

一个小例子（3 行 × 3 行 → 5 行）：

```python
a = pd.DataFrame({"k": ["x", "x", "y"], "v": [1, 2, 3]})   # k=x 出现 2 次
b = pd.DataFrame({"k": ["x", "x", "y"], "w": [9, 8, 7]})   # k=x 出现 2 次
a.merge(b, on="k")            # → 5 行！k=x 变成 4 行（2 × 2）
```

**正确做法**：连接时用 `validate` **主动声明基数**，不合规让 pandas 直接报错。
"""
    ),
    code(
        """a = pd.DataFrame({"k": ["x", "x", "y"], "v": [1, 2, 3]})
b = pd.DataFrame({"k": ["x", "x", "y"], "w": [9, 8, 7]})

blow = a.merge(b, on="k")
print("a", a.shape, "× b", b.shape, "→ merge 结果", blow.shape)
print(f"  k == 'x' 的行数: {int((blow['k'] == 'x').sum())}（= 2 × 2 笛卡尔积）")
print(blow.to_string(index=False))

print("\\n用 validate 主动声明基数:")
for v in ["one_to_one", "one_to_many", "many_to_one", "many_to_many"]:
    try:
        result = a.merge(b, on="k", validate=v)
        print(f"  validate={v:<14} OK   shape={result.shape}")
    except pd.errors.MergeError as exc:
        print(f"  validate={v:<14} MergeError: {str(exc).splitlines()[0][:64]}")

print("\\n回到真实数据（dev: 多 / info: 一）:")
for v in ["many_to_one", "one_to_one"]:
    try:
        result = dev.merge(info, on=KEY, how="left", validate=v)
        print(f"  validate={v:<14} OK   shape={result.shape}")
    except pd.errors.MergeError as exc:
        print(f"  validate={v:<14} MergeError: {str(exc).splitlines()[0][:64]}")"""
    ),
    md(
        """
> **判定规则**：**每次 `merge` 都带上 `validate`**。
> 它不改变结果，只是把"你没想清楚基数"这件事变成一个立刻可见的错误。
>
> - 主表 + 维表 → `validate="many_to_one"`（左多右一）
> - 两张明细表按键拼 → `validate="one_to_one"` 或 `"many_to_many"`
>
> 代价是零，收益是"行数膨胀"这类错误**再也不会静默发生**。
> 另一个便宜的护栏是 `indicator=True`，它会加一列 `_merge` 告诉你
> 每行是 `both` / `left_only` / `right_only`。
"""
    ),
    code(
        """flagged = dev.merge(info, on=KEY, how="left", indicator=True)
print("indicator 列的 dtype:", flagged["_merge"].dtype, "← pandas 3.0 是 category")
print("indicator 列的取值分布:")
print(flagged["_merge"].value_counts().to_string())
print("\\n→", int((flagged["_merge"] == "left_only").sum()),
      "行是 left_only，正好等于 270 - 247 =", len(dev) - 247)
print("\\n⚠️ 注意 value_counts 把 right_only = 0 也列出来了 ——")
print("   因为 _merge 是 category，value_counts 默认列出全部类别（含 0 计数）。")
print("   想跳过 0 计数：value_counts().pipe(lambda s: s[s > 0])")

print("\\n用 indicator 直接定位「维表缺哪些台区」:")
print(sorted(flagged.loc[flagged["_merge"] == "left_only", KEY].unique()))"""
    ),
    md(
        """
### 3.4 难点深挖③：`suffixes` 只管**非键**的同名列

**为什么难**：两表有同名列时，pandas 自动加 `_x` / `_y` 后缀
（或你指定的 `suffixes`），**但连接键不加**。所以你会看到
"有的同名列加了后缀、有的没加"——列名规则不一致，后面按名字取列就取错。

```python
a = dev[["记录ID", "台区编号", "负荷值"]].head(3)
b = dev[["记录ID", "台区编号", "负荷值"]].head(3)
a.merge(b, on="台区编号", suffixes=("_左", "_右"))
# 列名 → ['记录ID_左', '台区编号', '负荷值_左', '记录ID_右', '负荷值_右']
#                    ↑ 键没有被加后缀
```

**列序也反直觉**：左表的列（含键，键在它原来的位置）在前，
右表的非键列在后。
"""
    ),
    code(
        """left3 = dev[["记录ID", KEY, "负荷值"]].head(3)
right3 = dev[["记录ID", KEY, "负荷值"]].head(3)

print("左表列:", list(left3.columns))
print("右表列:", list(right3.columns))
print("\\n默认 suffixes:")
print(" ", list(left3.merge(right3, on=KEY).columns))
print("自定义 suffixes=('_左','_右'):")
print(" ", list(left3.merge(right3, on=KEY, suffixes=("_左", "_右")).columns))
print("\\n→ 键 '台区编号' 没有被加后缀；记录ID / 负荷值 都加了")
print("  所以「用列名取列」时绝不能想当然，先 print(columns) 再取")"""
    ),
    md(
        """
> **判定规则**：两表有同名列时，**连接前先改名**，比事后靠后缀猜更安全：
> ```python
> # ✅ 推荐：连接前把右表列名改成带业务含义的名字
> info2 = info.rename(columns={"容量kVA": "容量"})
> # ❌ 事后猜：result["负荷值_x"] 到底是哪张表的？
> ```
> 中文列名做后缀也没问题（`suffixes=("_左", "_右")` 实测可用），
> 但**列名里的 `_x` / `_y` 一定要在交付前清掉**，否则交卷表格很难看。
"""
    ),
    md(
        """
### 3.5 主表 + 多张维表的链式拼装

真实任务几乎不会只拼一张表。本数据要拼成一份"带台账和台区汇总的宽表"：

```text
dev (270, 13)  +  info_cn (16, 5)  +  summary (18, 3)  →  (270, 19)
```

注意 **列数的加法**：`13 + 5 + 3 - 1 - 1 = 19`——
每次连接会**消耗掉一个重复的键列**。

按顺序一次拼完：
"""
    ),
    code(
        """wide = dev.merge(info_cn, on=KEY, how="left").merge(summary, on=KEY, how="left")

print("宽表形状:", wide.shape, "| 预期列数: 13 + 5 + 3 - 1 - 1 = 19")
print("\\n新增的列:", [c for c in wide.columns if c not in dev.columns])
print("\\n各新增列的 NaN 数:")
for col in ["所属供电所", "容量", "投运年份", "供电区域类型", "日均负荷", "户数"]:
    print(f"  {col:<10} NaN = {int(wide[col].isna().sum()):<4}"
          f"非空率 = {match_rate(wide, col):.4f}")

print("\\n→ 台账 4 列都缺 23 行（B_03 / F_02 两个台区在维表里不存在）")
print("  summary 2 列一个都不缺（汇总表有全部 18 个台区）")
print("  这个「23 vs 0」的差异，就是这张宽表质量的全部真相")"""
    ),
    md(
        """
**三条校验公式**（每次连接后都该跑一遍）：

| 公式 | 本数据 | 含义 |
|---|---|---|
| `len(result) == len(dev)` | **270 == 270** ✅ | `many_to_one` 连接行数守恒 |
| `result[KEY].nunique() == dev[KEY].nunique()` | **18 == 18** ✅ | 没有凭空多出台区 |
| `result[右表列].notna().mean()` | 台账 **0.9148** / 汇总 **1.0** | 命中率符合预期 |

如果第三条数字**远低于**你预期的命中率，别急着 `fillna`——先查键。
"""
    ),
    code(
        """checks = {
    "行数守恒": len(wide) == len(dev),
    "台区数守恒": wide[KEY].nunique() == dev[KEY].nunique(),
    "台账命中率符合预期": abs(match_rate(wide, "所属供电所") - 247 / 270) < 1e-9,
    "汇总命中率 == 1": match_rate(wide, "日均负荷") == 1.0,
    "原列一个不少": set(dev.columns).issubset(set(wide.columns)),
}
for name, ok in checks.items():
    print(f"  {'✅' if ok else '❌'} {name}")

print("\\n列数对账: 13 + 5 + 3 - 1(重复键) - 1(重复键) =", 13 + 5 + 3 - 1 - 1,
      "| 实际:", wide.shape[1])"""
    ),
    md(
        """
### 3.6 难点深挖④：`concat` 默认**保留原索引**

**为什么难**：`pd.concat([a, b])` 默认 `ignore_index=False`，
两段的原索引会**原样留着**，于是结果里出现重复索引。
重复索引的后果是 `loc` / `[]` 会**一次返回多行**，
而后续所有"按索引对齐"的操作都开始出错。

```python
c1 = dev.head(3)     # index [0, 1, 2]
c2 = dev.head(2)     # index [0, 1]
pd.concat([c1, c2])                      # index → [0, 1, 2, 0, 1]  ⚠️
pd.concat([c1, c2], ignore_index=True)   # index → [0, 1, 2, 3, 4]  ✅
```
"""
    ),
    code(
        """c1 = dev.head(3)
c2 = dev.head(2)

plain = pd.concat([c1, c2])
fixed = pd.concat([c1, c2], ignore_index=True)

print("c1 索引:", c1.index.tolist(), "| c2 索引:", c2.index.tolist())
print("concat 默认       索引:", plain.index.tolist(), "| 有重复:", plain.index.has_duplicates)
print("concat(ignore_index=True) 索引:", fixed.index.tolist(), "| 有重复:", fixed.index.has_duplicates)
print("长度都是:", len(plain), len(fixed))

print("\\n重复索引的后果 —— 用 label 取值:")
print("  plain.loc[0]     取到", len(plain.loc[0]), "行（本想要 1 行）")
print("  用 iloc 才安全: plain.iloc[0] 取到 1 行")

print("\\n横向 concat（axis=1）按【索引标签】对齐，不按位置:")
c1r = c1.reset_index(drop=True)
c2r = c2.reset_index(drop=True)
print("  c1r", c1r.shape, "| c2r", c2r.shape,
      "| concat(axis=1) →", pd.concat([c1r, c2r], axis=1).shape)
print("  少的那一段补齐 NaN，行数取并集/较长者")"""
    ),
    md(
        """
**`concat` 的另外两个开关**：

| 参数 | 取值 | 作用 |
|---|---|---|
| `join` | `"outer"`（默认）/ `"inner"` | 列不同时取并集（补 `NaN`）还是交集 |
| `keys` | 列表 | 加一级索引区分来源（`MultiIndex`） |

列不一致时的默认行为是**并集 + 补 `NaN`**，这也是一个静默点：
"""
    ),
    code(
        """p1 = dev.head(3)[["记录ID", KEY]]
p2 = dev.head(3)[["记录ID", "负荷值"]]

outer = pd.concat([p1, p2])
inner = pd.concat([p1, p2], join="inner")

print("p1 列:", list(p1.columns), "| p2 列:", list(p2.columns))
print(f"  concat 默认(join='outer') → {outer.shape}  列: {list(outer.columns)}")
print(f"  concat(join='inner')      → {inner.shape}  列: {list(inner.columns)}")
print("\\n默认结果的 NaN 数:")
print(outer.isna().sum().to_string())

keyed = pd.concat([c1, c2], keys=["第一批", "第二批"])
print("\\nconcat(keys=[...]) 索引层级:", keyed.index.names)
print(keyed[["记录ID"]].head(4).to_string())"""
    ),
    md(
        """
> **判定规则**：
> - 纵向堆叠**明细行**（多个文件、多天数据） → 一律 `ignore_index=True`
> - 想保留"数据来自哪一批" → 用 `keys=`，得到一级 `MultiIndex`，**不要**靠原索引猜
> - 横向拼列前，**先 `reset_index(drop=True)`**，否则是按标签对齐而不是按位置
"""
    ),
    md(
        """
---

### 3.7 难点深挖⑤：`merge_asof` —— 用"最近的一个"代替"等于"

**为什么难**：很多业务规则不是"键相等"，而是"**表里查不到就等于前一条生效的**"。
比如容量限值每半月调一次，但负荷是逐小时的：

```text
限值表：2026-01-01 → 1000 ┃ 2026-01-16 → 1050 ┃ 2026-02-01 → 1100 ...
负荷表：逐小时 2160 行，没有任何时间戳能"等于"限值表的生效时间
```

用 `merge` 会一行都匹配不上（时间戳不相等）。
必须用 `pd.merge_asof`——它找的是**不晚于（或不低于）该时刻的最近一条**。

**三个硬约束**：

1. 两边的键**必须已排序**，否则 `ValueError: left keys must be sorted`
2. `direction`：`"backward"`（默认，取 ≤ 的最近一条）/ `"forward"` / `"nearest"`
3. 支持 `by=` 分组（组内各自最近匹配），以及 `tolerance=` 容差
"""
    ),
    code(
        """asof = pd.merge_asof(
    hourly, limit, left_on="时间戳", right_on="生效时间", direction="backward"
)

print("merge_asof(backward) →", asof.shape, "| 限值 NaN:", int(asof["限值"].isna().sum()))
print("\\n限值取值分布（按生效区间，每段应该是整数天 × 24 小时）:")
for value, count in asof["限值"].value_counts().sort_index().items():
    print(f"  限值 {value:<5} → {count:>4} 小时 = {count / 24:.0f} 天")

print("\\n超限条数（负荷值 > 限值）:", int((asof["负荷值"] > asof["限值"]).sum()))

fwd = pd.merge_asof(
    hourly, limit, left_on="时间戳", right_on="生效时间", direction="forward"
)
print("\\nmerge_asof(forward) → 限值 NaN:", int(fwd["限值"].isna().sum()),
      "（最后一个生效时间之后的都匹配不上）")
print("  forward 的分布:", {int(k): int(v) for k, v in fwd["限值"].value_counts().sort_index().items()})

print("\\n未排序时（不报错才怪）:")
try:
    pd.merge_asof(hourly.sample(frac=1, random_state=0), limit,
                  left_on="时间戳", right_on="生效时间")
except ValueError as exc:
    print("  ->", type(exc).__name__, ":", exc)"""
    ),
    md(
        """
**`backward` 与 `forward` 的语义差别**（这决定你有没有用未来数据）：

| `direction` | 取哪一条 | 本质 |
|---|---|---|
| `"backward"`（默认） | 生效时间 **≤** 当前时刻的最近一条 | **用了过去**——做特征安全 |
| `"forward"` | 生效时间 **≥** 当前时刻的最近一条 | **用了未来**——只有做"模拟当时可用信息"才合理 |
| `"nearest"` | 最近的一条（不分前后） | 可能取到未来，**默认别用** |

本数据实测：`backward` 的 `NaN` 是 **0**（全小时都能找到历史限值），
`forward` 的 `NaN` 是 **383**（最后一个生效时间之后的时段查不到"未来限值"）。

> **判定规则**：
> - 做**特征**、算"当时的规则是什么" → **`backward`**
> - 做**回测/复盘**"下一个决策点是什么" → `forward`，但要明确知道自己在用未来
> - `nearest` 在竞赛里几乎用不到，**误用会导致特征泄漏**
"""
    ),
    md(
        """
### 3.8 `join` 与 `merge` 的关系

`join` 是 `merge` 的**索引版快捷方式**：

```python
left.set_index(KEY).join(right.set_index(KEY), how="left")
# 完全等价于
left.merge(right, on=KEY, how="left")
```

什么时候用 `join`：右侧已经 `set_index` 过（比如做过多级索引分析），
不想再 `reset_index` 来回折腾。否则**一律用 `merge`**——
`merge` 的 `on=` / `validate=` / `indicator=` 都比 `join` 清楚。
"""
    ),
    code(
        """via_join = dev.set_index(KEY).join(info.set_index(KEY), how="left", rsuffix="_r")
via_merge = dev.merge(info, on=KEY, how="left")

print("join 结果:", via_join.shape, "| merge 结果:", via_merge.shape)
print("join 把键放到了索引里:", via_join.index.name, "→ 交付前记得 reset_index()")

# 值层面等价性对账
a = via_join["所属供电所"].reset_index(drop=True)
b = via_merge["所属供电所"]
common = a.notna() & b.notna()
print("\\n两者非空位置上的值完全一致:", bool((a[common] == b[common]).all()))
print("非空数:", int(a.notna().sum()), "vs", int(b.notna().sum()))"""
    ),
    md(
        """
---

### 3.9 综合：从"多张表"到"一张可交付宽表"

一份合格的交付宽表要满足：

| 检查项 | 判据 |
|---|---|
| 行数守恒 | `len(wide) == len(主表)` |
| 键不膨胀 | 台区数 == 主表台区数 |
| 命中率符合预期 | 每个新列的非空率都能解释 |
| 列名干净 | 没有 `_x` / `_y` / `_左` / `_右` 残留 |
| 索引干净 | `RangeIndex`，无重复 |
| 缺失有交代 | 每个 `NaN` 都能说出原因 |

下面一次性做完这六件事：
"""
    ),
    code(
        """deliver = (
    dev.merge(info_cn, on=KEY, how="left", validate="many_to_one")
    .merge(summary, on=KEY, how="left", validate="many_to_one")
)

# 列名清理：把可能残留的后缀去掉
deliver.columns = [c.replace("_x", "").replace("_y", "") for c in deliver.columns]
# 索引清理
deliver = deliver.reset_index(drop=True)

audit = {
    "行数": len(deliver),
    "台区数": int(deliver[KEY].nunique()),
    "列数": deliver.shape[1],
    "台账列缺行数": int(deliver["所属供电所"].isna().sum()),
    "汇总列缺行数": int(deliver["日均负荷"].isna().sum()),
    "索引无重复": not deliver.index.has_duplicates,
    "无后缀残留": not any(c.endswith(("_x", "_y", "_左", "_右")) for c in deliver.columns),
}
print("交付前自检:")
for name, value in audit.items():
    print(f"   {name:<14} = {value}")

print("\\n缺失交代: 台账 4 列各缺", audit["台账列缺行数"],
      "行，来自", sorted(set(dev[KEY]) - set(info[KEY])), "两个台区")
print("→ 这两个台区确实不在台账里，属于「业务上就是没有」，保留 NaN 并写明即可")"""
    ),
    md(
        """
**"缺失有交代"是评分的隐藏项**。三种缺失要在报告里区分开：

| 缺失来源 | 识别方法 | 处理 |
|---|---|---|
| 业务上不存在（本台区没台账） | 连接后 `NaN` + `indicator="left_only"` | **保留 `NaN`**，报告里说明 |
| 主表自身缺失（漏采） | 连接前就在 `NaN` | 按 ch04 的规则填充 |
| 键脏导致没匹配上 | 命中率远低于预期 | **先修键**，再重新连接 |

这三者在外表上完全一样（都是 `NaN`），但处理方式完全不同——
把"键脏"当成"业务缺失"填掉，是连接类错误最常见的收尾方式。
"""
    ),
    md(
        """
---

## 四、易错点清单

1. **主表 + 维表用了 `inner`** → 主表独有的键对应的行被静默丢掉（本例 **23** 行）
2. **键有隐形空白 / 全角空格 / 大小写差异** → 形状完全正常，但右表列**整列 `NaN`**
3. **不看右表列的非空率** → 上面这个错误会被当成"数据本身缺失"
4. **右表键有重复却按一对一用** → 行数按笛卡尔积膨胀，不报错
5. **不加 `validate`** → 放弃了"零成本"的基数护栏
6. **以为键也会加后缀** → `suffixes` 只作用于**非键**同名列，键保持原名
7. **`concat` 忘了 `ignore_index=True`** → 索引重复，`loc` 一次返回多行
8. **横向 `concat` 前没 `reset_index`** → 变成按标签对齐，错位却不出错
9. **`concat` 默认 `join="outer"`** → 列不同时静默多出 `NaN` 列
10. **`merge_asof` 前没排序** → `ValueError: left keys must be sorted`
11. **`merge_asof` 的 `direction` 选成 `forward`/`nearest`** → 用了未来数据
12. **连接后不清理 `_x` / `_y` 后缀** → 交付表格列名难看，且容易取错列
13. **用 `fillna(0)` 处理连接产生的 `NaN`** → 把"没台账"变成"台账值为 0"，语义被篡改
14. **`_merge` 是 `category`** → `value_counts()` 会列出 0 计数的类别，别误判成"真的有这类行"

---

## 五、本章小结

| 我想…… | 用哪个 | 关键参数 |
|---|---|---|
| 主表挂维表字段 | `dev.merge(info, on=KEY, how="left")` | `validate="many_to_one"` |
| 两表键名不同 | `merge(left_on=, right_on=)` | — |
| 用索引连接 | `left.join(right)` 或 `left_index=` | — |
| 追加多批明细行 | `pd.concat([...], ignore_index=True)` | `keys=` 标记批次 |
| 横向加列 | `pd.concat([...], axis=1)` | **先 `reset_index(drop=True)`** |
| 查"当时生效的规则" | `pd.merge_asof(..., direction="backward")` | 两边都要排序 |
| 反查连接是否命中 | `result[右表列].notna().mean()` | 或 `indicator=True` |

**一句话总结**：连接的所有错误都指向同一件事——
**你没有验证"这次连接的基数对不对、命中率对不对"**。
所以每次 `merge` 之后固定跑三条：`行数守恒` / `键不膨胀` / `右表列非空率`。
三行代码，能挡掉本章 90% 的坑。
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch13 练习 —— 连接与合并

> 数据：`device_defects.csv`（270 x 13，18 个台区）、
> `device_info.csv`（16 x 5，**少 2 个台区**）、`station_summary.csv`（18 x 3）、
> `load_curve.csv`（单台区 2160 小时）

**做题规则**：从上往下依次填，每个 `____` 处跑通再往下。
验收块 `assert` 不要改，它是你自己对账用的。

| 题号 | 考点 |
|---|---|
| TODO(1) | 四种 `how` 的行数与语义 |
| TODO(2) | 定位"主表有、维表没有"的键 |
| TODO(3) | `indicator=True` 的 `_merge` 列 |
| TODO(4) | 脏键导致静默 0 匹配 |
| TODO(5) | `validate` 拦下基数错误 |
| TODO(6) | 多对多的笛卡尔积膨胀 |
| TODO(7) | `suffixes` 只作用于非键同名列 |
| TODO(8) | 主表 + 两张维表的链式拼装 |
| TODO(9) | `concat` 的索引重复 |
| TODO(10) | `concat` 的列对齐（`join`） |
| TODO(11) | `merge_asof` 的 `backward` |
| TODO(12) | `merge_asof` 的 `forward` 与排序约束 |
| 综合① | 交付宽表 + 六项自检 |
| 综合② | 连接质量审计表 |
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
        """# @@todo(1) 把 dev 与 info 按 台区编号 做四种 how 的连接，行数存入 sizes 字典
#           （键为 "inner" / "left" / "right" / "outer"，值为结果行数）
# @@hint dev.merge(info, on=KEY, how=how) ；再用 len(...)
sizes = {how: len(dev.merge(info, on=KEY, how=how)) for how in ["inner", "left", "right", "outer"]}
# @@end
# ---- 验收 ----
assert set(sizes) == {"inner", "left", "right", "outer"}
assert sizes["inner"] == 247, f"inner 应为 247 行，实际 {sizes['inner']}"
assert sizes["left"] == 270, f"left 应为 270 行（行数守恒），实际 {sizes['left']}"
assert sizes["right"] == 247, f"right 应为 247 行，实际 {sizes['right']}"
assert sizes["outer"] == 270, f"outer 应为 270 行（右表无独有键），实际 {sizes['outer']}"
assert sizes["left"] - sizes["inner"] == 23, "inner 比 left 少的 23 行就是被丢掉的记录"
print("TODO(1) 通过 ->", sizes, "| inner 丢掉", sizes["left"] - sizes["inner"], "行")"""
    ),
    code(
        """# @@todo(2) 找出「主表有、维表没有」的台区编号列表，存入 missing_stations（升序）
#           并统计这些台区在主表里共有多少行，存入 n_missing_rows
# @@hint sorted(set(dev[KEY]) - set(info[KEY])) ；dev[KEY].isin(missing_stations).sum()
missing_stations = sorted(set(dev[KEY]) - set(info[KEY]))
n_missing_rows = int(dev[KEY].isin(missing_stations).sum())
# @@end
# ---- 验收 ----
assert missing_stations == ["STATION_B_03", "STATION_F_02"], f"缺的台区不对：{missing_stations}"
assert n_missing_rows == 23, f"这两个台区共应有 23 行，实际 {n_missing_rows}"
assert dev[KEY].nunique() == 18, "主表应有 18 个台区"
assert info[KEY].nunique() == 16, "维表应有 16 个台区"
assert len(dev.merge(info, on=KEY, how="left")) - n_missing_rows == 247, "对账：left - 缺失行 = inner"
print("TODO(2) 通过 -> 缺", missing_stations, "共", n_missing_rows, "行")"""
    ),
    code(
        """# @@todo(3) 用 indicator=True 做 left 连接存入 flagged，
#           然后统计 _merge 列各取值的行数存入 merge_counts（dict）
# @@hint dev.merge(info, on=KEY, how="left", indicator=True) ；
#        flagged["_merge"].value_counts().to_dict()
flagged = dev.merge(info, on=KEY, how="left", indicator=True)
merge_counts = flagged["_merge"].value_counts().to_dict()
# @@end
# ---- 验收 ----
assert merge_counts.get("both") == 247, f"both 应为 247，实际 {merge_counts.get('both')}"
assert merge_counts.get("left_only") == 23, f"left_only 应为 23，实际 {merge_counts.get('left_only')}"
assert merge_counts.get("right_only") == 0, \\
    "右表无独有键，right_only 应为 0（pandas 3.0 的 _merge 是 category，0 计数也会列出）"
assert merge_counts["both"] + merge_counts["left_only"] == len(dev), "两部分之和应等于左表行数"
assert str(flagged["_merge"].dtype) == "category", "_merge 在 pandas 3.0 是 category 类型"
# indicator 列名可以自定义
named = dev.merge(info, on=KEY, how="left", indicator="来源")
assert "来源" in named.columns, "indicator 可传自定义列名"
print("TODO(3) 通过 ->", merge_counts)
print("   注意：_merge 是 category，value_counts 会把 0 计数的类别也列出来")"""
    ),
    code(
        """# @@todo(4) 演示脏键的静默失败：把 info 的台区编号前置一个空格存入 info_dirty，
#           分别做 left 连接，把「所属供电所」的匹配行数存入 n_clean / n_dirty，
#           并把脏键 strip 后复测的匹配行数存入 n_fixed
# @@hint info_dirty = info.copy() 后改列；用 result["所属供电所"].notna().sum()
info_dirty = info.copy()
info_dirty[KEY] = " " + info_dirty[KEY].astype(str)
n_clean = int(dev.merge(info, on=KEY, how="left")["所属供电所"].notna().sum())
n_dirty = int(dev.merge(info_dirty, on=KEY, how="left")["所属供电所"].notna().sum())
n_fixed = int(
    dev.merge(
        info_dirty.assign(**{KEY: info_dirty[KEY].str.strip()}), on=KEY, how="left"
    )["所属供电所"].notna().sum()
)
# @@end
# ---- 验收 ----
assert n_clean == 247, f"干净键应匹配 247 行，实际 {n_clean}"
assert n_dirty == 0, f"脏键应匹配 0 行（静默失败），实际 {n_dirty}"
assert n_fixed == 247, f"strip 后应恢复 247 行，实际 {n_fixed}"
# 关键：形状完全一样，形状检查发现不了这个错误
assert dev.merge(info_dirty, on=KEY, how="left").shape == dev.merge(info, on=KEY, how="left").shape, \\
    "脏键与干净键的结果形状应完全相同（所以才叫静默失败）"
print("TODO(4) 通过 -> 干净", n_clean, "| 脏键", n_dirty, "| strip 后", n_fixed,
      "| 形状相同:", True)"""
    ),
    code(
        """# @@todo(5) 用 validate 拦下基数错误：
#           对 dev 与 info 做 left 连接，验证 many_to_one 是否通过（布尔量存入 ok_many_to_one）；
#           再用 probe 探测 one_to_one（脚手架里的 probe 会返回 ("err", 异常类名)），存入 probe_one_to_one
# @@hint dev.merge(info, on=KEY, how="left", validate="many_to_one")
#        probe(dev.merge, info, on=KEY, how="left", validate="one_to_one")
ok_many_to_one = bool(
    len(dev.merge(info, on=KEY, how="left", validate="many_to_one")) == 270
)
probe_one_to_one = probe(dev.merge, info, on=KEY, how="left", validate="one_to_one")
# @@end
# ---- 验收 ----
assert ok_many_to_one is True, "many_to_one 校验应通过"
assert probe_one_to_one[0] == "err", "one_to_one 应被拦下（左表键重复）"
assert probe_one_to_one[1] == "MergeError", f"应抛 MergeError，实际 {probe_one_to_one[1]}"
assert int(dev[KEY].duplicated().sum()) == 252, "左表 252 行有重复键，所以不是一对一"
print("TODO(5) 通过 -> many_to_one 通过 | one_to_one 被拦下:", probe_one_to_one[1])"""
    ),
    code(
        """# @@todo(6) 造两把小表演示多对多的笛卡尔积：a = {k:['x','x','y'], v:[1,2,3]}、
#           b = {k:['x','x','y'], w:[9,8,7]}，连接结果存入 blow，
#           形状存入 blow_shape，k=='x' 的行数存入 n_x
# @@hint a.merge(b, on="k")
a = pd.DataFrame({"k": ["x", "x", "y"], "v": [1, 2, 3]})
b = pd.DataFrame({"k": ["x", "x", "y"], "w": [9, 8, 7]})
blow = a.merge(b, on="k")
blow_shape = blow.shape
n_x = int((blow["k"] == "x").sum())
# @@end
# ---- 验收 ----
assert blow_shape == (5, 3), f"笛卡尔积结果应为 (5, 3)，实际 {blow_shape}"
assert n_x == 4, f"k=='x' 应为 2x2=4 行，实际 {n_x}"
assert len(a) + len(b) != len(blow), "多对多连接不能按「相加」估行数"
print("TODO(6) 通过 -> a(3) × b(3) →", blow_shape, "| k=='x' 有", n_x, "行（2x2 笛卡尔积）")"""
    ),
    code(
        """# @@todo(7) 造两把同名列的小表做带 suffixes 的连接：
#           取 dev 前三行的 记录ID / 台区编号 / 负荷值 两份（left3 / right3），
#           用 suffixes=("_左","_右") 连接存入 sfx，列名列表存入 sfx_cols
# @@hint dev[["记录ID", KEY, "负荷值"]].head(3) ；再 .merge(right3, on=KEY, suffixes=(...))
left3 = dev[["记录ID", KEY, "负荷值"]].head(3)
right3 = dev[["记录ID", KEY, "负荷值"]].head(3)
sfx = left3.merge(right3, on=KEY, suffixes=("_左", "_右"))
sfx_cols = list(sfx.columns)
# @@end
# ---- 验收 ----
assert sfx_cols == ["记录ID_左", KEY, "负荷值_左", "记录ID_右", "负荷值_右"], \\
    f"suffixes 列名不对：{sfx_cols}"
assert "台区编号_左" not in sfx_cols and "台区编号_右" not in sfx_cols, \\
    "连接键不应被加后缀"
assert sfx.shape == (3, 5), f"形状应为 (3, 5)，实际 {sfx.shape}"
print("TODO(7) 通过 ->", sfx_cols)
print("   注意：键 '台区编号' 保持原名，其余同名列都加了后缀")"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(8) 主表 + 两张维表链式拼装：dev ← info_cn ← summary（都 left、都 validate="many_to_one"），
#           结果存入 wide；形状存入 wide_shape；新增列列表（按出现顺序）存入 new_cols
# @@hint dev.merge(info_cn, on=KEY, how="left", validate="many_to_one").merge(summary, ...)
#        新增列 = [c for c in wide.columns if c not in dev.columns]
wide = (
    dev.merge(info_cn, on=KEY, how="left", validate="many_to_one")
    .merge(summary, on=KEY, how="left", validate="many_to_one")
)
wide_shape = wide.shape
new_cols = [c for c in wide.columns if c not in dev.columns]
# @@end
# ---- 验收 ----
assert wide_shape == (270, 19), f"宽表应为 (270, 19)，实际 {wide_shape}"
assert new_cols == ["所属供电所", "容量", "投运年份", "供电区域类型", "日均负荷", "户数"], \\
    f"新增列不对：{new_cols}"
assert wide_shape[1] == 13 + 5 + 3 - 1 - 1, "列数应等于 13 + 5 + 3 - 1 - 1"
assert len(wide) == len(dev), "行数必须守恒"
assert int(wide[KEY].nunique()) == 18, "台区数不能膨胀"
assert int(wide["容量"].isna().sum()) == 23, "台账列应缺 23 行"
assert int(wide["户数"].isna().sum()) == 0, "汇总表有全部 18 个台区，不该缺"
print("TODO(8) 通过 ->", wide_shape, "| 新增", len(new_cols), "列 | 台账缺 23 / 汇总缺 0")"""
    ),
    code(
        """# @@todo(9) 把 dev 前三行与后两行纵向 concat：
#           默认结果索引存入 idx_plain，ignore_index 结果索引存入 idx_fixed
#           （都转成 list 方便断言）
# @@hint pd.concat([c1, c2]) ；pd.concat([c1, c2], ignore_index=True)
c1 = dev.head(3)
c2 = dev.head(2)
idx_plain = pd.concat([c1, c2]).index.tolist()
idx_fixed = pd.concat([c1, c2], ignore_index=True).index.tolist()
# @@end
# ---- 验收 ----
assert idx_plain == [0, 1, 2, 0, 1], f"默认应保留原索引（有重复），实际 {idx_plain}"
assert idx_fixed == [0, 1, 2, 3, 4], f"ignore_index=True 应重排索引，实际 {idx_fixed}"
assert len(idx_plain) == 5 and len(idx_fixed) == 5, "两种写法行数应相同"
assert pd.concat([c1, c2]).index.has_duplicates, "默认结果索引必有重复"
assert not pd.concat([c1, c2], ignore_index=True).index.has_duplicates, "重排后不应有重复"
print("TODO(9) 通过 -> 默认索引", idx_plain, "| ignore_index 索引", idx_fixed)"""
    ),
    code(
        """# @@todo(10) 列不一致的 concat：p1 = dev 前三行的 [记录ID, 台区编号]，
#            p2 = dev 前三行的 [记录ID, 负荷值]；
#            默认结果形状存入 shape_outer、列名存入 cols_outer；
#            join="inner" 结果形状存入 shape_inner
# @@hint pd.concat([p1, p2]) ；pd.concat([p1, p2], join="inner")
p1 = dev.head(3)[["记录ID", KEY]]
p2 = dev.head(3)[["记录ID", "负荷值"]]
outer_cat = pd.concat([p1, p2])
shape_outer = outer_cat.shape
cols_outer = list(outer_cat.columns)
shape_inner = pd.concat([p1, p2], join="inner").shape
# @@end
# ---- 验收 ----
assert shape_outer == (6, 3), f"默认并集应为 (6, 3)，实际 {shape_outer}"
assert cols_outer == ["记录ID", KEY, "负荷值"], f"默认应取列并集，实际 {cols_outer}"
assert shape_inner == (6, 1), f"join='inner' 应只留共同列，实际 {shape_inner}"
assert int(outer_cat[KEY].isna().sum()) == 3, "p2 缺 台区编号，应补 3 个 NaN"
assert int(outer_cat["负荷值"].isna().sum()) == 3, "p1 缺 负荷值，应补 3 个 NaN"
print("TODO(10) 通过 -> 默认", shape_outer, cols_outer, "| inner", shape_inner)"""
    ),
    code(
        """# @@todo(11) 用 merge_asof(backward) 给 hourly 挂上"当时生效的限值"：
#            结果存入 asof，形状存入 asof_shape，限值 NaN 数存入 n_asof_nan，
#            限值取值分布（dict）存入 limit_dist，超限条数存入 n_over
# @@hint pd.merge_asof(hourly, limit, left_on="时间戳", right_on="生效时间", direction="backward")
#        分布用 asof["限值"].value_counts().sort_index().to_dict()
#        超限用 (asof["负荷值"] > asof["限值"]).sum()
asof = pd.merge_asof(
    hourly, limit, left_on="时间戳", right_on="生效时间", direction="backward"
)
asof_shape = asof.shape
n_asof_nan = int(asof["限值"].isna().sum())
limit_dist = {int(k): int(v) for k, v in asof["限值"].value_counts().sort_index().items()}
n_over = int((asof["负荷值"] > asof["限值"]).sum())
# @@end
# ---- 验收 ----
assert asof_shape == (2160, 7), f"应为 (2160, 7)，实际 {asof_shape}"
assert n_asof_nan == 0, f"backward 应能匹配到全部小时，实际 {n_asof_nan} 个 NaN"
assert limit_dist == {1000: 360, 1050: 384, 1100: 360, 1120: 312, 1150: 360, 1180: 384}, \\
    f"限值区间分布不对：{limit_dist}"
assert sum(limit_dist.values()) == 2160, "各区间小时数之和应等于总小时数"
assert n_over == 1, f"超限应只有 1 条，实际 {n_over}"
print("TODO(11) 通过 ->", asof_shape, "| NaN", n_asof_nan, "| 超限", n_over, "条")
print("   限值分布:", limit_dist)"""
    ),
    code(
        """# @@todo(12) 对比 forward：用 merge_asof(direction="forward") 存入 fwd，
#            限值 NaN 数存入 n_fwd_nan；并用 probe 探测「未排序」的情况存入 probe_unsorted
# @@hint pd.merge_asof(hourly, limit, left_on="时间戳", right_on="生效时间", direction="forward")
#        probe(pd.merge_asof, hourly.sample(frac=1, random_state=0), limit,
#              left_on="时间戳", right_on="生效时间")
fwd = pd.merge_asof(
    hourly, limit, left_on="时间戳", right_on="生效时间", direction="forward"
)
n_fwd_nan = int(fwd["限值"].isna().sum())
probe_unsorted = probe(
    pd.merge_asof,
    hourly.sample(frac=1, random_state=0),
    limit,
    left_on="时间戳",
    right_on="生效时间",
)
# @@end
# ---- 验收 ----
assert n_fwd_nan == 383, f"forward 应有 383 个 NaN（最后一个生效时间之后），实际 {n_fwd_nan}"
assert fwd.shape == (2160, 7), "形状应与 backward 相同"
assert int(fwd["限值"].isna().sum()) > 0, "forward 会用到未来数据，末尾必然匹配不上"
assert probe_unsorted[0] == "err", "未排序时 merge_asof 应报错"
assert probe_unsorted[1] == "ValueError", f"应抛 ValueError，实际 {probe_unsorted[1]}"
assert n_fwd_nan > n_asof_nan, "forward 的 NaN 数应远多于 backward"
print("TODO(12) 通过 -> backward NaN", n_asof_nan, "| forward NaN", n_fwd_nan,
      "| 未排序 ->", probe_unsorted[1])"""
    ),
    md(
        """
---

## 综合题

### 综合 ① 交付一张干净的宽表并自检

要求同时满足：行数守恒、键不膨胀、**无 `_x`/`_y` 残留**、`RangeIndex`、无重复索引。
"""
    ),
    code(
        """# @@todo 综合①：产出 deliver（dev + info_cn + summary 的宽表，已清列名、已 reset_index），
#          把六项自检结果存入 audit 字典：
#          行数 / 台区数 / 列数 / 台账缺行数 / 汇总缺行数 / 无后缀残留
# @@hint deliver.columns = [c.replace("_x","").replace("_y","") for c in deliver.columns]
#        deliver = deliver.reset_index(drop=True)
deliver = (
    dev.merge(info_cn, on=KEY, how="left", validate="many_to_one")
    .merge(summary, on=KEY, how="left", validate="many_to_one")
)
deliver.columns = [c.replace("_x", "").replace("_y", "") for c in deliver.columns]
deliver = deliver.reset_index(drop=True)
audit = {
    "行数": len(deliver),
    "台区数": int(deliver[KEY].nunique()),
    "列数": deliver.shape[1],
    "台账缺行数": int(deliver["所属供电所"].isna().sum()),
    "汇总缺行数": int(deliver["日均负荷"].isna().sum()),
    "无后缀残留": not any(c.endswith(("_x", "_y", "_左", "_右")) for c in deliver.columns),
}
# @@end
# ---- 验收 ----
assert audit["行数"] == 270, f"行数应守恒（270），实际 {audit['行数']}"
assert audit["台区数"] == 18, f"台区数应守恒（18），实际 {audit['台区数']}"
assert audit["列数"] == 19, f"列数应为 19，实际 {audit['列数']}"
assert audit["台账缺行数"] == 23, f"台账应缺 23 行，实际 {audit['台账缺行数']}"
assert audit["汇总缺行数"] == 0, f"汇总不该缺，实际 {audit['汇总缺行数']}"
assert audit["无后缀残留"] is True, "列名不应残留 _x / _y / _左 / _右"
assert not deliver.index.has_duplicates, "索引不应有重复"
assert isinstance(deliver.index, pd.RangeIndex), "索引应是 RangeIndex"
# 交付前的最后一次对账：原列一个不少
assert set(dev.columns).issubset(set(deliver.columns)), "原表列不得丢失"
print("综合① 通过 -> 交付宽表", deliver.shape)
for name, value in audit.items():
    print(f"   {name:<12} = {value}")"""
    ),
    md(
        """
### 综合 ② 连接质量审计：把"命中率异常"自动报出来

写一段可复用的审计：给定主表、维表和键，自动给出
形状、行数守恒、匹配率、以及**匹配率是否低于阈值**的判断。
用这份数据实测：台账应为 **0.9148**（低于 0.95 阈值），汇总应为 **1.0**。
"""
    ),
    code(
        """# @@todo 综合②：用脚手架里的 audit_merge 对台账与汇总各跑一次审计，
#          结果存入 audit_table（键为 "台账" / "汇总"）
# @@hint audit_table = {"台账": audit_merge(dev, info_cn, KEY, "所属供电所"),
#                       "汇总": audit_merge(dev, summary, KEY, "日均负荷")}
audit_table = {
    "台账": audit_merge(dev, info_cn, KEY, "所属供电所"),
    "汇总": audit_merge(dev, summary, KEY, "日均负荷"),
}
# @@end
# ---- 验收 ----
assert set(audit_table) == {"台账", "汇总"}
assert audit_table["台账"]["形状"] == (270, 17)
assert audit_table["汇总"]["形状"] == (270, 15)
assert audit_table["台账"]["行数守恒"] is True and audit_table["汇总"]["行数守恒"] is True
assert audit_table["台账"]["匹配率"] == 0.9148, f"台账匹配率应为 0.9148，实际 {audit_table['台账']['匹配率']}"
assert audit_table["汇总"]["匹配率"] == 1.0, f"汇总匹配率应为 1.0，实际 {audit_table['汇总']['匹配率']}"
assert audit_table["台账"]["低命中"] is True, "台账命中率 0.9148 < 0.95，应被标为低命中"
assert audit_table["汇总"]["低命中"] is False, "汇总命中率 1.0，不应被标为低命中"
# 关键结论：低命中不等于有错 —— 这里 0.9148 是因为 2 个台区业务上就没有台账
assert round(247 / 270, 4) == 0.9148, "0.9148 正好等于 247/270，说明缺的是那 23 行"
print("综合② 通过 -> 连接质量审计:")
for label, info_row in audit_table.items():
    print(f"   {label}: {info_row['形状']} | 行数守恒={info_row['行数守恒']} "
          f"| 匹配率={info_row['匹配率']} | 低命中={info_row['低命中']}")
print("   结论：台账低命中是「业务上确实没有」，不是键脏 —— 靠 indicator 区分")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（六题都能答上来才算过）：

1. 主表 + 维表为什么一律用 `how="left"`？用 `inner` 会丢什么？
2. 脏键（多一个空格）导致的失败有什么特征？为什么形状检查发现不了？
3. `validate` 参数不改变结果，那要它干什么？
4. `suffixes` 为什么不给连接键加后缀？列序有什么规律？
5. `pd.concat(axis=1)` 是按位置还是按标签对齐？`reset_index` 为什么必须？
6. `merge_asof` 的 `backward` 和 `forward` 分别代表"用过去"还是"用未来"？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
