"""ch06 —— 类型转换与字符串处理：把 8 种脏写法收敛成 3 类

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch06_types_strings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch06_types_strings"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

DATA = Path("data")
dev = pd.read_csv(DATA / "device_defects.csv")
raw = pd.read_csv(DATA / "device_defects_raw.csv", dtype=str)

# 同义映射表（课里会逐条验证它们为什么是必需的）
STATUS_MAP = {"未处理": "待处理", "处理完成": "已处理", "done": "已处理"}
DEFECT_MAP = {"渗油": "渗漏油", "漏油": "渗漏油", "异物搭挂": "异物"}
PREFIX2TYPE = {"CB": "断路器", "CT": "互感器", "IN": "绝缘子", "LA": "避雷器", "TR": "变压器"}
CAT_COLS = ["台区编号", "线路名称", "设备类型", "缺陷类型", "缺陷等级", "处理状态"]

print("pandas", pd.__version__)
print("dev", dev.shape, "| raw", raw.shape)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch06 类型转换与字符串处理：把 8 种脏写法收敛成 3 类

> 方向：数据预处理 ｜ 竞赛对应：**数据准备及处理（技能操作 10%）**
> —— 类型不对，后面所有步骤都是错的：`resample` 要 `datetime`、
> `groupby` 要干净的类别、`fit()` 不接受字符串。

本章要解决的不是「怎么调 `astype`」，而是三件更硬的事：

1. **什么时候转换会静默出错**（不报错但结果错）
2. **同一含义的 8 种写法怎么收敛**（这是真实业务数据的常态）
3. **`category` 到底该在什么阶段用**（用早了会埋雷）
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据准备及处理 | 占技能操作 10% | 类型锁定 → 类别标准化 → 日期解析 |
| 数据探索 | 每项 2 分 | 类型分布普查（`dtypes` / `select_dtypes`） |
| 数据清洗 | 每项 2 分 | 类别脏写收敛、字符串同义合并 |
| 建模前置 | 决定后续全部 | 类别编码、日期索引、数值列可运算 |

**为什么单独成章**：类型与字符串是所有清洗动作的**共同前置**。
缺失值填不了、异常值算不出、`groupby` 分不干净，追到根上八成是类型没锁对。
"""
    ),
    md(
        """
## 二、API 速查表

| API | 返回 | 一句话说明 |
|---|---|---|
| `df.astype({"列": "类型"})` | `DataFrame` | 批量转类型，返回新对象 |
| `pd.to_numeric(s, errors=)` | `Series` | 字符串转数值，`errors` 决定失败怎么办 |
| `.round().astype("Int64")` | `Series` | 带缺失的列转可空整型 |
| `pd.to_datetime(s, format=)` | `Series` | 转日期；`format="mixed"` 容忍多种写法 |
| `s.str.strip()` | `Series` | 去首尾空白（**必做**） |
| `s.str.replace(pat, repl, regex=)` | `Series` | 替换；`regex` 默认 `False`（3.0 起） |
| `s.str.contains(pat, na=)` | `Series[bool]` | 子串匹配，**务必给 `na`** |
| `s.str.extract(regex)` | `DataFrame` | 正则捕获组 → 多列 |
| `s.str.split("_")` | `Series[list]` | 切分，配合 `.str[i]` 取第 i 段 |
| `s.map(dict)` | `Series` | 按字典映射（未命中变 `NaN`，见 ch11） |
| `s.replace(dict)` | `Series` | 按字典替换（未命中**保留原值**） |
| `s.astype("category")` | `Series` | 转分类类型（内存骤降，见 §3.8） |
| `pd.get_dummies(s)` | `DataFrame` | 独热编码 |

> **`map` 和 `replace` 的区别**是本章最容易混的一处：
> `map` 是"查表翻译，查不到就空着"，`replace` 是"改掉这几个值，其他不动"。
> 做**同义合并**用 `replace`，做**等级映射**用 `map`。
"""
    ),
    code(HEADER),
    md(
        """
---

## 三、逐节讲解

### 3.1 类型体检：先知道每列是什么，再决定转什么

清洗第一步永远不是 `fillna`，而是看一眼 `dtypes`。这里有个
**老教程会害你**的地方：很多资料写 `select_dtypes("object")` 选字符串列，
而 pandas 3.0 起字符串列的 dtype 已经是 `str` 了。
"""
    ),
    code(
        """print("逐列 dtype：")
print(dev.dtypes.to_string())

print("\\n类型分布（每种 dtype 有几列）：")
print(dev.dtypes.astype(str).value_counts().to_string())

# 三种"选字符串列"的写法，实测对比
import warnings as _w

with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    cols_legacy = list(dev.select_dtypes("object").columns)

print("\\nselect_dtypes('object') ->", len(cols_legacy), "列")
print("  警告:", [type(x.message).__name__ for x in caught] or "无")
print("  警告内容:", str(caught[0].message)[:70] if caught else "—", "...")

print("\\nselect_dtypes('str')             ->", len(dev.select_dtypes("str").columns), "列  ← 正确写法")
print("select_dtypes(exclude='number')  ->", len(dev.select_dtypes(exclude="number").columns), "列  ← 最稳，不报警告")
print("select_dtypes('number')          ->", list(dev.select_dtypes("number").columns))
print("select_dtypes('datetime')        ->", list(dev.select_dtypes("datetime").columns),
      " ← 空！说明发现日期还是字符串")"""
    ),
    md(
        """
> **实测结论**：`include="object"` 在 3.0 里靠"向后兼容"仍能选到 `str` 列，
> 但每次调用都抛 `Pandas4Warning`，官方声明未来版本移除。
> 所以**能跑不等于该写**——写 `include="str"`（精确）或 `exclude="number"`（最稳）。

---

### 3.2 数值转换：`astype` 有两个会直接抛错的场景

字符串列转数值，"看起来"就是 `astype(float)`。但真实数据里它有两个必然失败的场景：
**列里有非数字字符串**，以及**列里有缺失值**。
"""
    ),
    code(
        """load_text = raw["负荷值"]

print("① 列里有 'kW' / 千分位：")
print("   含 'kW' 的行数:", int(load_text.str.contains("kW", na=False).sum()))
print("   含千分位 ',' 的行数:", int(load_text.str.contains(",", na=False).sum()))

try:
    load_text.astype(float)
    print("   astype(float) 居然成功")
except ValueError as exc:
    print("   astype(float) ->", type(exc).__name__, str(exc)[:80])

print("\\n② errors='coerce'：转不动的变 NaN，不中断")
cleaned = pd.to_numeric(
    load_text.str.replace(" kW", "", regex=False).str.replace(",", "", regex=False),
    errors="coerce",
)
print("   清洗后 NaN 数:", int(cleaned.isna().sum()), " ← 21 个真缺失全部归位")
print("   dtype:", cleaned.dtype, "| 前 3 个非空值:", cleaned.dropna().head(3).tolist())

print("\\n③ errors 三态对比：")
bad = pd.Series(["1.5", "N/A", "2.5"])
for err in ("raise", "coerce", "ignore"):
    try:
        out = pd.to_numeric(bad, errors=err)
        print(f"   errors={err!r:8s} -> dtype={out.dtype}  值={out.tolist()}")
    except Exception as exc:
        print(f"   errors={err!r:8s} -> 抛出 {type(exc).__name__}")"""
    ),
    md(
        """
> **注意 `errors="ignore"` 是最坏的选择**：它不报错、也不转换，
> 直接原样返回字符串 Series。你以为转换成功了，`dtype` 还是 `str`，
> 后面 `s.mean()` 才炸——报错位置离病因十万八千里。
> **要留出清洗余量就用 `coerce`，要立刻暴露问题就用 `raise`（默认），永远不要 `ignore`。**
"""
    ),
    md(
        """
---

### 3.3 难点深挖：带缺失的列怎么转整数（`Int64` 与它的内存陷阱）

这是 pandas 里最反直觉的一处。

**为什么难**：numpy 的 `int64` 没有「缺失」这个概念，所以只要列里有 NaN，
`astype(int)` 就一定失败；而用 `float64` 表示整数，又会把"1"变成"1.0"。
pandas 的解法是**可空整型** `Int64`（注意大写 I）。但这里还埋了第二个坑：
**`Int64` 并不一定更省内存**。
"""
    ),
    code(
        """load = dev["负荷值"]

print("① 直接 astype(int) 失败：")
try:
    load.astype(int)
    print("   居然成功")
except Exception as exc:
    print("   ->", type(exc).__name__, ":", str(exc)[:95])

print("\\n② 先 round() 再转 Int64：")
load_int = load.round().astype("Int64")
print("   dtype:", load_int.dtype)
print("   <NA> 数:", int(load_int.isna().sum()), " ← 21 个原始缺失被保留为 <NA>，不是 0")
print("   前 5 个值:", load_int.head(5).tolist(), " ← 注意是整数，没有 .0")

print("\\n③ 内存对比（反直觉）：")
print(f"   float64 列: {load.memory_usage(deep=True):6d} 字节")
print(f"   Int64  列: {load_int.memory_usage(deep=True):6d} 字节")
print(f"   -> Int64 多占 {load_int.memory_usage(deep=True) - load.memory_usage(deep=True)} 字节")

print("\\n④ 为什么更占内存：")
print("   可空整型底层是 numpy int64 + 一张独立的 bool 掩码表")
print("   = 8 字节/值 + 1 字节/值 = 9 字节，比 float64 的 8 字节/值更贵")
print("   -> 只有『缺失比例低 + 需要精确整数语义』时才值得用 Int64")"""
    ),
    md(
        """
> **判定规则**：
> - 这列**要参与算术**（求和、均值）→ 用 `float64` 就好，别折腾
> - 这列是**编号 / 计数**，且必须精确（对不上台账就是错）→ 用 `Int64`
> - 这列缺失很多 → 更要用 `float64`，`Int64` 的掩码会额外吃掉一份内存

---

### 3.4 难点深挖：`format="mixed"` 只解决一半问题

**为什么难**：`pd.to_datetime` 默认按"整列同一种格式"解析，遇到混格式直接抛
`DateParseError`。pandas 2.0 引入 `format="mixed"` 逐行猜格式，
于是很多人以为它能包打天下。**它猜不出中文日期。**
"""
    ),
    code(
        """dates_raw = raw["发现日期"]

print("① 原始数据的格式分布：")
for pat, name in [("年", "中文 2026年5月3日"), ("/", "斜杠 02/16/2026"), ("-", "连字符 2026-04-15")]:
    hit = dates_raw.str.contains(pat, regex=False, na=False)
    print(f"   含 {pat!r}（{name:22s}）: {int(hit.sum()):3d} 行   样例 {dates_raw[hit].head(1).tolist()}")

print("\\n② 不指定 format：")
try:
    pd.to_datetime(dates_raw)
    print("   成功")
except Exception as exc:
    print("   ->", type(exc).__name__, ":", str(exc)[:70])

print("\\n③ format='mixed' 逐行猜：")
mixed = pd.to_datetime(dates_raw, format="mixed", errors="coerce")
print("   失败（NaT）数:", int(mixed.isna().sum()), " ← 正好等于中文日期的行数")
print("   失败样例:", dates_raw[mixed.isna()].head(2).tolist())

print("\\n④ format='%Y-%m-%d' 写死一种格式：")
fixed_fmt = pd.to_datetime(dates_raw, format="%Y-%m-%d", errors="coerce")
print("   失败数:", int(fixed_fmt.isna().sum()), " ← 只有 77 行能过，其余全废")

print("\\n⑤ 正解：先规整中文，再 mixed")
dates_norm = (
    dates_raw.str.replace("年", "-", regex=False)
    .str.replace("月", "-", regex=False)
    .str.replace("日", "", regex=False)
)
parsed = pd.to_datetime(dates_norm, format="mixed")
print("   规整后失败数:", int(parsed.isna().sum()), " ← 0")
print("   dtype:", parsed.dtype)
print("   范围:", parsed.min().date(), "~", parsed.max().date())

print("\\n⑥ 与清洗版对账（证明解析结果一致，没把月日搞反）：")
print("   与 dev['发现日期'] 完全一致:",
      bool((parsed.astype(str) == pd.to_datetime(dev["发现日期"]).astype(str)).all()))"""
    ),
    md(
        """
> **判定规则**：`format="mixed"` 只解决**分隔符/顺序不一致**（`-` 和 `/` 混用）。
> 遇到**非标准词**（中文年月日、"春"、"Q1"）必须先 `str.replace` 规整成标准形态，
> 再交给 `mixed` 猜。
>
> **还有一个不能忘的**：斜杠格式 `02/16/2026` 到底是「月/日」还是「日/月」是**歧义**的。
> `dayfirst` 默认 `False`（按美式 月/日 解析）。本章数据是 月/日，
> 所以第 ⑥ 步的对账断言必须过——**竞赛里遇到歧义格式，一定要拿已知样本反查一次**，
> 不要凭默认值猜。
"""
    ),
    md(
        """
---

### 3.5 字符串访问器：`str` 全家桶

`.str` 后面挂的是**一整套向量化字符串方法**，它们是"能不用 `apply` 就不用 `apply`"的关键。
"""
    ),
    code(
        """s = dev["处理状态"]

print("① strip：首尾空白（脏数据第一大来源）")
print("   原样类别:", s.nunique(), "->", sorted(s.unique().tolist()))
print("   strip 后 :", s.str.strip().nunique(), "类 ->", sorted(s.str.strip().unique().tolist()))

print("\\n② replace 去内部空白（'已 处理' -> '已处理'）")
stripped = s.str.strip()
print("   仅 strip      :", stripped.nunique(), "类")
print("   strip + 去内部:", stripped.str.replace(r"\\s+", "", regex=True).nunique(), "类")

print("\\n③ contains / startswith / endswith")
print("   含'处理'的:", int(s.str.contains("处理", na=False).sum()))
print("   以'STATION_B' 开头的台区行:", int(dev["台区编号"].str.startswith("STATION_B").sum()))

print("\\n④ split + str[i]：拆复合字段")
sn = dev["设备编号"]
print("   样例:", sn.head(3).tolist())
print("   前缀:", sn.str.split("_").str[0].head(3).tolist())
print("   序号:", sn.str.split("_").str[1].head(3).tolist())

print("\\n⑤ extract：正则捕获组，一次拆成多列")
parsed_sn = sn.str.extract(r"^(?P<prefix>[A-Z]{2})_(?P<num>\\d+)$")
print("   列名:", list(parsed_sn.columns))
print(parsed_sn.head(3).to_string(index=False))

print("\\n⑥ len / zfill / cat / normalize")
print("   长度分布:", dev["缺陷类型"].str.len().value_counts().sort_index().to_dict())
print("   补齐到 6 位:", pd.Series([1, 42]).astype(str).str.zfill(6).tolist())
print("   拼接:", pd.Series(["STATION_A", "STATION_B"]).str.cat(pd.Series(["01", "02"]), sep="_").tolist())
print("   全角转半角:", pd.Series(["ＡＢ１２"]).str.normalize("NFKC").tolist())"""
    ),
    md(
        """
> **两条纪律**：
> 1. `contains` / `match` / `startswith` 一律给 `na=False`，
>    否则遇到 `NaN` 返回 `NaN`，布尔索引直接 `ValueError`
> 2. `str.replace` 从 pandas 3.0 起 `regex` **默认 `False`**（旧版是 `True`）。
>    要按模式替换必须显式写 `regex=True`，否则会把 `.` `(` `[` 当普通字符处理

---

### 3.6 脏类别标准化两步法：去噪 → 同义合并

这是本章的**核心技能**，也是竞赛数据清洗里必考的一项：
业务上只有 3 种状态，台账里却躺着 8 种写法。

标准动作永远是**两步，不能合并**：

| 步骤 | 做什么 | 用什么 | 解决的是 |
|---|---|---|---|
| 第 1 步：去噪 | 去首尾空白、去内部空白、统一大小写 | `.str.strip()` / `.str.replace(r"\\s+","")` / `.str.upper()` | **同一个词的不同书写** |
| 第 2 步：语义合并 | 把同义词映射到标准词 | `.replace(dict)` | **不同的词表示同一含义** |

**为什么必须分两步**：因为第 1 步是**无损的**（机器能判断两个字符串是否等价），
第 2 步是**有损的**（"未处理"和"待处理"是不是一回事，只有业务能判断）。
混在一起写，你就分不清哪一步把数据改坏了。
"""
    ),
    code(
        """status_raw = dev["处理状态"]
print("原始 %d 类：" % status_raw.nunique())
print(status_raw.value_counts().to_string())

# 第 1 步：去噪（无损）
status_step1 = status_raw.str.strip().str.replace(r"\\s+", "", regex=True)
print("\\n第 1 步后 %d 类：" % status_step1.nunique())
print(status_step1.value_counts().to_string())

# 第 2 步：语义合并（有损，依赖业务判断）
status_final = status_step1.replace(STATUS_MAP)
print("\\n第 2 步后 %d 类：" % status_final.nunique())
print(status_final.value_counts().to_string())

print("\\n对账：合并前后总行数必须相等")
print("   去噪前:", len(status_raw), "| 合并后:", len(status_final),
      "| 无丢失:", len(status_raw) == len(status_final))

print("\\n同样的两步法用在缺陷类型上：")
defect_final = dev["缺陷类型"].str.strip().replace(DEFECT_MAP)
print("   原始 %d 类 -> 合并后 %d 类" % (dev["缺陷类型"].nunique(), defect_final.nunique()))
print(defect_final.value_counts().to_string())

print("\\n注意：replace 不会把未列出的值变成 NaN（这是它与 map 的关键区别）")
probe = pd.Series(["渗油", "发热", "锈蚀"])
print("   replace:", probe.replace(DEFECT_MAP).tolist())
print("   map    :", probe.map(DEFECT_MAP).tolist(), " ← 未命中的变 NaN")"""
    ),
    md(
        """
---

### 3.7 结构性字段拆解：编码里藏着的信息

`设备编号` 这种字段看着只是一个 ID，其实**编码本身携带两个字段**：
前缀是设备类型代码，序号是流水号。竞赛里把这类隐性信息显式拆出来，
往往是"特征工程"那几分和"数据预处理"那几分同时到手的操作。
"""
    ),
    code(
        """prefix = dev["设备编号"].str.split("_").str[0]
serial = dev["设备编号"].str.split("_").str[1]

print("① 前缀分布：")
print(prefix.value_counts().to_string())

print("\\n② 前缀与设备类型是否严格一一对应（决定我们敢不敢用前缀反推类型）：")
cross = dev.assign(prefix=prefix).groupby("prefix")["设备类型"].nunique().to_dict()
print("   每个前缀对应的设备类型数量:", cross)
print("   全部 == 1 ?", all(v == 1 for v in cross.values()), " ← 是，可以用前缀反推")

print("\\n③ 前缀 -> 设备类型 映射：")
for key, value in PREFIX2TYPE.items():
    print(f"   {key} -> {value}")

recovered = prefix.map(PREFIX2TYPE)
print("\\n④ 反推结果与原始列逐行一致:", bool((recovered == dev["设备类型"]).all()))

print("\\n⑤ 序号转整数（现在可以安全地当数值特征用了）：")
print("   字符串:", serial.head(3).tolist(), "-> 整数:", serial.astype(int).head(3).tolist())"""
    ),
    md(
        """
---

### 3.8 难点深挖：`category` 的三重身份与两个陷阱

`astype("category")` 一行代码就能把内存打下来，所以特别容易被滥用。
它同时是**存储优化**、**编码容器**、**排序约束**三样东西，
用早了、用错了都会埋雷。

**陷阱一：`category` 只记住出现过的类别。** 一旦整列转成 `category`，
再想 `replace` 一个新值进来，会直接报 `TypeError`。
**陷阱二：`category` 参与算术、`groupby` 时的行为与 `str` 不同**（见 ch10 的 `observed`）。
"""
    ),
    code(
        """n_before = dev.memory_usage(deep=True).sum()
dev_cat = dev.astype({col: "category" for col in CAT_COLS})
n_after = dev_cat.memory_usage(deep=True).sum()

print("① 内存对比（6 个字符串列转 category）：")
print(f"   转换前: {n_before:6d} 字节")
print(f"   转换后: {n_after:6d} 字节")
print(f"   省下   : {n_before - n_after} 字节 ({(1 - n_after / n_before) * 100:.1f}%)")
print("\\n   逐列看（台区编号 18 个不同值）：")
print(f"   str      : {dev['台区编号'].memory_usage(deep=True):6d} 字节")
print(f"   category : {dev['台区编号'].astype('category').memory_usage(deep=True):6d} 字节")

print("\\n② 内部结构：codes + categories")
tc = dev["设备类型"].astype("category")
print("   categories:", tc.cat.categories.tolist())
print("   codes     :", tc.cat.codes.head(5).tolist(), " ← 整数编码，可直接喂模型")
print("   isna 仍然是 isna:", int(tc.isna().sum()))

print("\\n③ 陷阱一：category 列插入新值 -> 报错")
tc_probe = pd.Series(["变压器", "绝缘子"]).astype("category")
try:
    tc_probe.loc[0] = "开关柜"
    print("   居然成功")
except TypeError as exc:
    print("   ->", type(exc).__name__, ":", str(exc)[:80])
print("   正解：先 astype(str) 改完再转回去，或改用 .cat.add_categories()")

print("\\n④ 陷阱二：category 记住的是『全集类别』，子集独热会多出全零列")
sub = dev[dev["设备类型"].isin(["变压器", "绝缘子"])]
print("   子集行数:", len(sub))
print("   子集 str                ->", pd.get_dummies(sub["设备类型"]).shape)
full_dtype = pd.CategoricalDtype(categories=dev["设备类型"].astype("category").cat.categories)
sub_carried = sub["设备类型"].astype(full_dtype)
print("   子集沿用全量 category    ->", pd.get_dummies(sub_carried).shape, " ← 多出 3 列")
zero_cols = [c for c in pd.get_dummies(sub_carried).columns if pd.get_dummies(sub_carried)[c].sum() == 0]
print("   多出来的全零列:", zero_cols)
print("   -> 全零列不会报错，但会白增维度、稀释正则强度")

print("\\n⑤ 正确的临时插值方式：先 .cat.add_categories()")
tc_fix = tc_probe.cat.add_categories(["开关柜"])
tc_fix.loc[0] = "开关柜"
print("   插入后:", tc_fix.tolist())
print("   全局独热维度:", pd.get_dummies(dev["设备类型"]).shape)
print("   维度变化的真正原因：category 记住了『全集类别』")
print("   -> 只取变压器+绝缘子两个子集，str 给出 2 列，沿用全量 category dtype 则给出 5 列（3 列全零）")"""
    ),
    md(
        """
> **判定规则**：`category` 只在**建模前的最后一步**用。
> 清洗中间态一律保持 `str`——因为清洗过程要不断替换新值、合并类别，
> 而 `category` 的类别集合是**冻结的**，中途插入新值直接报错。
>
> **顺序**：清洗用 `str` → 全部清洗完 → 编码前 `astype("category")`。

**顺带一条工程纪律**：训练集与测试集必须**共用同一份 categories**，
否则测试集里没出现过的类别要么变全零列、要么直接把维度搞对不上。写法：

```python
dtype = pd.CategoricalDtype(categories=train["设备类型"].unique())
train["设备类型"] = train["设备类型"].astype(dtype)
test["设备类型"] = test["设备类型"].astype(dtype)   # 沿用同一 dtype
```

---

## 四、易错点清单

1. `s.astype(float)` 遇到非数字字符串抛 `ValueError`，不会自动变 `NaN` → 用 `pd.to_numeric(errors="coerce")`
2. **`errors="ignore"` 是最坏的选项**：不报错、不转换、静默返回原字符串
3. 用 `astype("Int64")` 前**忘记 `.round()`** → 列里有小数时抛 `TypeError: cannot safely cast`
4. `format="mixed"` 遇到中文日期**仍然解析不出来**（本章实测：12 行全失败）
5. 写死 `format="%Y-%m-%d"` → 只对 77/270 行有效，其余变 `NaT` 且不报错（用了 `errors="coerce"` 时）
6. 斜杠日期的**月日歧义**：`02/16/2026` 的 `dayfirst` 默认是 `False`，务必拿已知样本反查
7. `str.contains` 不给 `na=False` → 布尔索引时报 `ValueError`
8. 以为 `str.replace` 默认按正则 → pandas 3.0 起默认 `regex=False`
9. **同义合并该用 `replace` 却用了 `map`** → 未命中的值全变 `NaN`（270 行里可能一多半变空）
10. 只做去空白、不做同义合并 → 8 类只降到 6 类，业务上还是 3 类
11. 清洗中途就把列转成 `category` → 后续 `replace` 新值直接 `TypeError`
12. 以为 `Int64` 一定省内存 → 实测比 `float64` **多占** 270 字节

---

## 五、本章小结

| 场景 | 正确写法 |
|---|---|
| 字符串列转数值 | `pd.to_numeric(s, errors="coerce")`，先 `str.replace` 去单位/千分位 |
| 编号列转整数（有缺失） | `s.round().astype("Int64")` |
| 日期混格式 | 先 `str.replace` 规整中文 → `pd.to_datetime(format="mixed")` |
| 类别脏写 | **两步**：`str.strip()` + `str.replace(r"\\s+","")` → `replace(同义表)` |
| 同义合并 | `replace(dict)`（不是 `map`） |
| 显式拆字段 | `str.split("_").str[i]` 或 `str.extract(regex)` |
| 内存不够 | 清洗全部做完后，`astype("category")` |
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook（EXERCISE 写答案，practice 由构建器自动挖空）
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch06 练习 —— 类型转换与字符串处理

数据：`data/device_defects.csv`（270 行 x 13 列）与 `data/device_defects_raw.csv`（脏格式版）

**每题都有 `assert` 验收块**。跑通即正确，跑不通的报错信息本身就说清了哪里错。
**不要翻答案版。**

| 题号 | 考什么 |
|---|---|
| TODO(1) | 用 `exclude="number"` 挑出字符串列（并验证 `include="object"` 已废弃） |
| TODO(2) | 类别去噪（空白问题） |
| TODO(3) | 类别语义合并（8 类 → 3 类） |
| TODO(4) | 缺陷类型两步标准化（10 类 → 6 类） |
| TODO(5) | 设备编号拆前缀 |
| TODO(6) | 前缀反推设备类型 |
| TODO(7) | 日期规整 + `format="mixed"` 解析 |
| TODO(8) | 正确转可空整型 |
| TODO(9) | `category` 内存压缩 |
| 综合 ① | 完整类型清洗链（类型锁定 → 类别标准化 → 日期解析） |
| 综合 ② | 数据质量报告 |
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
        """# @@todo(1) 挑出全部字符串列（用 exclude="number"），存入 str_cols
# @@hint DataFrame.select_dtypes(exclude="number").columns.tolist()
str_cols = dev.select_dtypes(exclude="number").columns.tolist()
# @@end
# ---- 验收 ----
assert isinstance(str_cols, list), "str_cols 应该是 list"
assert len(str_cols) == 9, f"应有 9 个字符串列，实际 {len(str_cols)}"
assert "发现日期" in str_cols and "处理状态" in str_cols

# 顺带验证：include="object" 在 pandas 3.0 仍能选到，但会报警告（已废弃）
import warnings as _w

with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    legacy_cols = list(dev.select_dtypes("object").columns)
assert legacy_cols == str_cols, "兼容行为下两者应一致"
assert any("Pandas4Warning" in type(x.message).__name__ for x in caught), "应抛出废弃警告"
print("TODO(1) 通过 -> 字符串列", len(str_cols), "个；include='object' 已废弃但仍可用")"""
    ),
    code(
        """# @@todo(2) 给「处理状态」去噪：去首尾空白 + 去内部空白，结果存入 status_step1
# @@hint Series.str.strip() ；Series.str.replace(r"\\s+", "", regex=True)
status_step1 = dev["处理状态"].str.strip().str.replace(r"\\s+", "", regex=True)
# @@end
# ---- 验收 ----
assert status_step1.nunique() == 6, f"去噪后应为 6 类，实际 {status_step1.nunique()}"
counts_step1 = status_step1.value_counts().to_dict()
assert counts_step1["已处理"] == 144, f"已处理应为 144（128+16），实际 {counts_step1.get('已处理')}"
assert counts_step1["处理中"] == 54, f"处理中应为 54（37+17），实际 {counts_step1.get('处理中')}"
assert "处理中 " not in counts_step1 and "已 处理" not in counts_step1, "空白还没去干净"
print("TODO(2) 通过 -> 8 类降到", status_step1.nunique(), "类：", counts_step1)"""
    ),
    code(
        """# @@todo(3) 把去噪结果按 STATUS_MAP 做语义合并，存入 status_final，应只剩 3 类
# @@hint Series.replace(字典)
status_final = status_step1.replace(STATUS_MAP)
# @@end
# ---- 验收 ----
assert status_final.nunique() == 3, f"合并后应为 3 类，实际 {status_final.nunique()}"
counts_final = status_final.value_counts().to_dict()
assert counts_final == {"已处理": 171, "处理中": 54, "待处理": 45}, f"分布不对：{counts_final}"
assert len(status_final) == len(dev), "合并不能改变行数"
assert status_final.isna().sum() == 0, "replace 不应该产生缺失（变出 NaN 说明用错成 map 了）"
print("TODO(3) 通过 -> 最终分布:", counts_final)"""
    ),
    code(
        """# @@todo(4) 缺陷类型两步标准化：先 strip 再去噪，然后按 DEFECT_MAP 合并，存入 defect_final
# @@hint 一步写完：dev["缺陷类型"].str.strip().replace(DEFECT_MAP)
defect_final = dev["缺陷类型"].str.strip().replace(DEFECT_MAP)
# @@end
# ---- 验收 ----
assert dev["缺陷类型"].nunique() == 10, "原始应有 10 种写法"
assert defect_final.nunique() == 6, f"合并后应为 6 类，实际 {defect_final.nunique()}"
counts_defect = defect_final.value_counts().to_dict()
assert counts_defect["渗漏油"] == 61, f"渗漏油应为 61（43+11+7），实际 {counts_defect.get('渗漏油')}"
assert counts_defect["异物"] == 40, f"异物应为 40（29+11），实际 {counts_defect.get('异物')}"
assert counts_defect["锈蚀"] == 36, f"锈蚀应为 36（29+7），实际 {counts_defect.get('锈蚀')}"
assert defect_final.isna().sum() == 0
print("TODO(4) 通过 ->", counts_defect)"""
    ),
    code(
        """# @@todo(5) 从「设备编号」拆出前缀，存入 sn_prefix（形如 'TR' 'CB'）
# @@hint Series.str.split("_").str[0]
sn_prefix = dev["设备编号"].str.split("_").str[0]
# @@end
# ---- 验收 ----
assert set(sn_prefix.unique()) == {"CB", "CT", "IN", "LA", "TR"}, f"前缀集合不对：{set(sn_prefix.unique())}"
counts_prefix = sn_prefix.value_counts().to_dict()
assert counts_prefix["TR"] == 55, f"TR 应为 55，实际 {counts_prefix.get('TR')}"
assert counts_prefix["LA"] == 59, f"LA 应为 59，实际 {counts_prefix.get('LA')}"
print("TODO(5) 通过 -> 前缀分布:", counts_prefix)"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(6) 用 PREFIX2TYPE 把前缀反推回设备类型，存入 recovered_type
# @@hint Series.map(字典)
recovered_type = sn_prefix.map(PREFIX2TYPE)
# @@end
# ---- 验收 ----
assert recovered_type.isna().sum() == 0, "不应有未命中的前缀"
assert recovered_type.nunique() == 5
assert bool((recovered_type == dev["设备类型"]).all()), "反推结果与原始设备类型不一致"
print("TODO(6) 通过 -> 前缀映射与设备类型 270 行全一致")"""
    ),
    code(
        """# @@todo(7) 解析 raw 的「发现日期」：先把中文年月日换成 '-'，再用 format="mixed" 解析，存入 found_date
# @@hint 链式四次 str.replace 把 年/月 换成 '-'、把 日 去掉；再 pd.to_datetime(..., format="mixed")
date_norm = (
    raw["发现日期"].str.replace("年", "-", regex=False)
    .str.replace("月", "-", regex=False)
    .str.replace("日", "", regex=False)
)
found_date = pd.to_datetime(date_norm, format="mixed")
# @@end
# ---- 验收 ----
assert found_date.isna().sum() == 0, f"不应有解析失败，实际失败 {int(found_date.isna().sum())} 行"
assert str(found_date.dtype).startswith("datetime64"), f"dtype 不对：{found_date.dtype}"
assert found_date.min().year == 2026 and found_date.max().month == 6

# 对照：不做规整、直接 mixed 会失败 12 行（中文日期）
naive = pd.to_datetime(raw["发现日期"], format="mixed", errors="coerce")
assert int(naive.isna().sum()) == 12, f"直接 mixed 应失败 12 行，实际 {int(naive.isna().sum())}"
print("TODO(7) 通过 -> 解析 0 失败；直接 mixed 则失败", int(naive.isna().sum()), "行")"""
    ),
    code(
        """# @@todo(8) 把「负荷值」转成可空整型，存入 load_int（先四舍五入）
# @@hint Series.round().astype("Int64")   —— 大写 I 的可空整型
load_int = dev["负荷值"].round().astype("Int64")
# @@end
# ---- 验收 ----
assert str(load_int.dtype) == "Int64", f"dtype 应为 Int64，实际 {load_int.dtype}"
assert int(load_int.isna().sum()) == 21, f"缺失应保留 21 个，实际 {int(load_int.isna().sum())}"
assert int(load_int.notna().sum()) == 249
first_valid = load_int.dropna().iloc[0]
assert isinstance(first_valid, (int, np.integer)) or float(first_valid).is_integer(), \
    f"取到的值 {first_valid!r} 不是整数语义"

# 直接 astype(int) 必须失败——证明 round 这一步不可省
try:
    dev["负荷值"].astype(int)
    raise AssertionError("astype(int) 不该成功（列里有缺失 + 小数）")
except AssertionError:
    raise
except Exception:
    pass
print("TODO(8) 通过 -> Int64 保留", int(load_int.isna().sum()), "个 <NA>，前 3 值:", load_int.head(3).tolist())"""
    ),
    code(
        """# @@todo(9) 把 CAT_COLS 里的列一次性转成 category，存入 dev_cat
# @@hint DataFrame.astype({列: "category" for 列 in CAT_COLS})
dev_cat = dev.astype({col: "category" for col in CAT_COLS})
# @@end
# ---- 验收 ----
assert all(str(dev_cat[c].dtype) == "category" for c in CAT_COLS), "有列没转成 category"
before = dev.memory_usage(deep=True).sum()
after = dev_cat.memory_usage(deep=True).sum()
saved = 1 - after / before
assert saved > 0.6, f"内存应省 60% 以上，实际 {saved:.1%}"
assert abs(saved - 0.670) < 0.01, f"实测应约 67.0%，实际 {saved:.1%}"
assert int(dev_cat["台区编号"].cat.codes.max()) == 17, "台区编号应有 18 个类别（codes 0..17）"

# 陷阱验证：category 列插入新值会报错
probe = pd.Series(["变压器", "绝缘子"]).astype("category")
try:
    probe.loc[0] = "开关柜"
    raise AssertionError("category 列插入新类别本该报 TypeError")
except AssertionError:
    raise
except TypeError:
    pass
print(f"TODO(9) 通过 -> 内存 {before} -> {after} 字节，省 {saved:.1%}")"""
    ),
    md(
        """
---

## 综合题

### 综合 ① 完整类型清洗链

把本章所有动作串成一条流水线，产出一张**可直接进建模**的表 `clean`：

1. 类别列标准化：`处理状态`（8→3 类）、`缺陷类型`（10→6 类）
2. 日期解析：把 `发现日期` 真正转成 `datetime`
3. 数值类型锁定：`负荷值` 保持 `float64`（要参与算术），`记录ID` 转 `Int64`
4. 显式派生：从 `设备编号` 拆出 `设备前缀`
5. 内存优化：字符串列转 `category`（放在最后一步）

要求 `clean` 相对原始表**只增不减行**（270 行）、**不产生新的缺失**。
"""
    ),
    code(
        """# @@todo 综合①：按上面 5 步产出 clean（只挖标准化、类型锁定、派生、内存优化四段）
# @@hint 先 copy；类别列用 .str.strip().replace(...)；日期用 pd.to_datetime(format="mixed")；最后 astype({...:"category"})
clean = dev.copy()
clean["处理状态"] = clean["处理状态"].str.strip().str.replace(r"\\s+", "", regex=True).replace(STATUS_MAP)
clean["缺陷类型"] = clean["缺陷类型"].str.strip().replace(DEFECT_MAP)
clean["发现日期"] = pd.to_datetime(clean["发现日期"], format="mixed")
clean["记录ID"] = clean["记录ID"].astype("Int64")
clean["设备前缀"] = clean["设备编号"].str.split("_").str[0]
clean = clean.astype({col: "category" for col in CAT_COLS})
# @@end
# ---- 验收 ----
assert len(clean) == 270, f"行数不能变，实际 {len(clean)}"
assert str(clean["发现日期"].dtype).startswith("datetime64"), "发现日期没转成日期"
assert str(clean["处理状态"].dtype) == "category", "处理状态没转成 category"
assert clean["处理状态"].nunique() == 3
assert clean["缺陷类型"].nunique() == 6
assert "设备前缀" in clean.columns and clean["设备前缀"].nunique() == 5

# 清洗过程不能把非缺失变成缺失：只允许原有缺失继续存在
for col in ["处理状态", "缺陷类型"]:
    assert clean[col].isna().sum() == 0, f"{col} 出现了新的缺失"
assert int(clean["负荷值"].isna().sum()) == 21, "负荷值缺失数不能被改动"

print("综合① 通过 -> clean", clean.shape)
print("   处理状态:", clean["处理状态"].value_counts().to_dict())
print("   缺陷类型:", {k: int(v) for k, v in clean["缺陷类型"].value_counts().items()})
print("   内存:", clean.memory_usage(deep=True).sum(), "字节（原", dev.memory_usage(deep=True).sum(), "）")"""
    ),
    md(
        """
### 综合 ② 数据质量报告

写一个函数，输入原始 `dev`，输出 5 个关键质量指标。这张报告就是竞赛里
"数据探索"那几分的交出物，也是给评委看的第一张表。
"""
    ),
    code(
        """# @@todo 综合②：产出 5 个质量指标存入 quality 字典
# @@hint 用 dev 现算：字符串列数 / 类别最大脏度 / 日期是否已解析 / 缺失单元格 / 可压缩比例
str_col_count = len(dev.select_dtypes(exclude="number").columns)
max_dirty = int(max(
    dev["处理状态"].nunique(),
    dev["缺陷类型"].nunique(),
))
date_parsed = str(dev["发现日期"].dtype).startswith("datetime64")
missing_cells = int(dev.isna().sum().sum())
compress_ratio = round(
    1 - dev.astype({c: "category" for c in CAT_COLS}).memory_usage(deep=True).sum()
    / dev.memory_usage(deep=True).sum(),
    4,
)
quality = {
    "字符串列数": str_col_count,
    "最大类别脏度": max_dirty,
    "日期已解析": date_parsed,
    "缺失单元格": missing_cells,
    "可压缩比例": compress_ratio,
}
# @@end
# ---- 验收 ----
expect = {"字符串列数", "最大类别脏度", "日期已解析", "缺失单元格", "可压缩比例"}
assert set(quality) == expect, f"键不对。多 {set(quality) - expect}，缺 {expect - set(quality)}"
assert quality["字符串列数"] == 9
assert quality["最大类别脏度"] == 10, "缺陷类型原始 10 种写法，是脏度最高的列"
assert quality["日期已解析"] is False, "原始表里发现日期还是字符串"
assert quality["缺失单元格"] == 240, f"全表缺失单元格应为 240，实际 {quality['缺失单元格']}"
assert abs(quality["可压缩比例"] - 0.6698) < 0.001, \
    f"可压缩比例应约 0.6698，实际 {quality['可压缩比例']}"
print("综合② 通过 -> 数据质量报告：")
for key, value in quality.items():
    print(f"   {key:12s} = {value!r}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（五题都能答上来才算过）：

1. `pd.to_numeric(s, errors="ignore")` 错在哪？为什么它比 `errors="raise"` 更危险？
2. `format="mixed"` 为什么解不了 `2026年5月3日`？正确姿势是什么？
3. `map` 和 `replace` 都能做映射，同义合并该用哪个？用错会看到什么现象？
4. `Int64` 比 `float64` 更省内存吗？实测差多少、为什么？
5. 为什么 `astype("category")` 必须放在清洗的**最后一步**？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
