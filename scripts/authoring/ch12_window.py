"""ch12 —— 窗口计算：`rolling` / `expanding` / `ewm` / `shift` / `diff`

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch12_window.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch12_window"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

KEY_COL = "台区编号"
VALUE_COL = "负荷值"
STATION = "STATION_A_01"

# 负荷曲线：3 个台区 x 2160 小时（2026-01-01 ~ 2026-03-31 逐小时）
curve = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"])
curve = curve.sort_values([KEY_COL, "时间戳"]).reset_index(drop=True)

# 单台区 + DatetimeIndex：时间窗口 rolling("24h") 必须用它
station = curve[curve[KEY_COL] == STATION].set_index("时间戳")
load_ts = station[VALUE_COL]

# 单台区 + RangeIndex：固定窗口 rolling(24) 用这个
load = load_ts.reset_index(drop=True)

# 分组对象（跨台区）
gb = curve.groupby(KEY_COL)

print("pandas", pd.__version__)
print("curve", curve.shape, "| 单台区", load.shape, "| 台区数", curve[KEY_COL].nunique())'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def window_span(w):
    """窗口极差（max - min），给 rolling().apply 用。"""
    return float(np.nanmax(w) - np.nanmin(w))


def peek(series, n=3):
    """看一下前 n 个非缺失值，省得每次手打 dropna().head()。"""
    return series.dropna().head(n).round(4).tolist()


print("load 缺失:", int(load.isna().sum()), "| load_ts 索引:", type(load_ts.index).__name__)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch12 窗口计算：`rolling` / `expanding` / `ewm` / `shift` / `diff`

> 方向：数据预处理 ｜ 竞赛对应：**时序特征工程 + 异常检测**
> —— 只要题目给的是"带时间戳的一列数"（负荷曲线、温度、电流、销量），
> 90% 的特征都从这一族方法里出来。

本章要解决三件事：

1. **窗口是按行序滑的，不是按时间滑的**——不先排序，结果静默错但不报错
2. **默认 `min_periods` 会静默制造 NaN**，而缺失值会把它后面 `window-1` 个窗口
   一起污染掉（指数放大）
3. **`groupby().rolling()` 和 `groupby().shift()` 返回的索引完全不同**，
   前者是两级 `MultiIndex`、后者是原索引——形状不确定是这类 API 的老毛病
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 时序特征工程 | 影响建模分 | `rolling` 统计、滞后特征、环比、滑动分位 |
| 异常检测 | 每项 2 分 | 滑动中位数 / 滑动分位替代全局阈值 |
| 数据清洗 | 每项 2 分 | 用窗口统计做稳健替换，避免全局均值被极值带偏 |
| 结果交付 | 运行时限 | 分组窗口向量化，替代"每组 for 循环" |

**为什么单独成章**：`rolling` 家族有个致命属性——**绝大多数错误都不报错**。
窗口少一行、`min_periods` 用默认值、忘了排序、把未来值算进特征里，
四种错法全部"跑得通、出得来数、不对"。而时序特征一旦把未来信息漏进训练集，
模型离线指标会好看得离谱、上线直接崩——这类错误在赛后复盘里几乎查不出来。
"""
    ),
    md(
        """
## 二、API 速查表

### 2.1 窗口构造器

| 构造器 | 关键参数 | 返回 | 一句话说明 |
|---|---|---|---|
| `s.rolling(w)` | `window` `min_periods` `center` `closed` | `Rolling` | 固定长度滑动窗口 |
| `s.rolling("24h")` | 同上 | `Rolling` | **时间**长度窗口（需 `DatetimeIndex`） |
| `s.expanding(m)` | `min_periods` | `Expanding` | 从头累积到当前行 |
| `s.ewm(...)` | `span` / `halflife` / `alpha` / `adjust` | `ExponentialMovingWindow` | 指数加权，越近权重越大 |

### 2.2 窗口上的聚合（三者通用）

| 方法 | 说明 | 缺失敏感度 |
|---|---|---|
| `.mean()` `.sum()` `.std()` | 常规统计 | 窗口内有效值 < `min_periods` → `NaN` |
| `.max()` `.min()` | 极值 | 同上 |
| `.median()` `.quantile(q)` | 稳健统计（抗极值） | 同上 |
| `.count()` | **有效值个数**（不含 `NaN`） | 不产生 `NaN`，是诊断窗口的钥匙 |
| `.apply(func, raw=True)` | 自定义窗口函数 | `raw=True` 传 ndarray，快一个量级 |

### 2.3 位移与差分

| 方法 | 参数 | 说明 |
|---|---|---|
| `s.shift(n)` | `n` `fill_value` | 向后错 n 行（`n>0` 是**取过去**，用于造滞后特征） |
| `s.diff(n)` | `n` | `s - s.shift(n)`，一阶差分 |
| `s.pct_change(n)` | `n` `fill_method` | `(s - s.shift(n)) / s.shift(n)`，变化率 |
| `gb.shift(n)` | 同 `Series.shift` | **组内**错位，组边界不越界 |

> ⚠️ `s.shift(-1)` 是**取未来值**——造特征时写错符号 = 直接泄漏标签。

### 2.4 索引行为对照（最容易错的一张表）

| 写法 | 返回类型 | 返回索引 |
|---|---|---|
| `gb[col].mean()` | `Series` | 组键（压缩） |
| `gb[col].transform("mean")` | `Series` | **原索引**（广播） |
| `gb[col].shift(1)` | `Series` | **原索引** |
| `gb[col].diff(1)` | `Series` | **原索引** |
| `gb[col].rolling(24).mean()` | `Series` | **两级 `MultiIndex`**（组键 + 原索引） |

**最后一行是本章最大的坑**：`rolling` 挂在 `groupby` 后面时，
结果多出一级索引，`reset_index()` 出来的列名还是 `level_1`。
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
---

## 三、逐节讲解

### 3.1 心智模型：窗口是"按行序"的切片

`rolling(24)` 的意思是：**当前行 + 前面 23 行**（左闭右闭，**含当前行**）。

```text
行号        0   1   2  ...  22   23   24  ...
窗口覆盖         └─────── 24 行 ───────┘
roll[23] = mean(load[0:24])     ← 第 23 行才凑满 24 个
roll[24] = mean(load[1:25])     ← 往后滑一格
```

关键结论：**窗口的边界由"行号"决定，不是由"时间戳"决定。**
数据没排序时，`rolling` 会老老实实按乱序的行号去滑，
算出来的数是"随机 24 行的均值"——**不报错、不警告、纯错**。
"""
    ),
    code(
        """# 验证：roll[23] 就是前 24 行的均值
roll24 = load.rolling(24).mean()
print("roll24.iloc[23]        =", round(roll24.iloc[23], 4))
print("load.iloc[:24].mean()  =", round(load.iloc[:24].mean(), 4))
print("roll24.iloc[24]        =", round(roll24.iloc[24], 4))
print("load.iloc[1:25].mean() =", round(load.iloc[1:25].mean(), 4))

# 换一种证明：把序列倒过来，rolling 的结果完全不同（说明它只认行序）
reversed_load = load.iloc[::-1].reset_index(drop=True)
print("\\n倒序后 roll[23]      =", round(reversed_load.rolling(24).mean().iloc[23], 4))
print("它等于原序列最后 24 行均值 =", round(load.iloc[-24:].mean(), 4), " ← 同一个数，但行号完全变了")"""
    ),
    md(
        """
> **判定规则**：只要用窗口函数，**第一件事就是
> `df.sort_values(["分组键","时间戳"])`**。
> 不排序不会报错，只会让你在最后对账时发现"数怎么都对不上"。
"""
    ),
    md(
        """
### 3.2 五个最常用的窗口统计

日常 80% 的需求就是这几个：

| 想要什么 | 写法 | 典型用途 |
|---|---|---|
| 24h 滑动均值 | `load.rolling(24).mean()` | 趋势特征 |
| 24h 滑动最大 | `load.rolling(24).max()` | 峰值特征 |
| 24h 滑动标准差 | `load.rolling(24).std()` | 波动性特征 |
| 24h 滑动中位数 | `load.rolling(24).median()` | **稳健**趋势（抗尖峰） |
| 24h 有效值个数 | `load.rolling(24).count()` | **诊断窗口**是否完整 |

`.count()` 值得单独说：它返回的是窗口内**非缺失**的个数。
它是唯一"不产生 NaN"的窗口聚合，所以当你想知道
"这一行的窗口到底够不够数"时，看 `count()` 而不是 `mean()` 是不是 `NaN`。
"""
    ),
    code(
        """stats = pd.DataFrame(
    {
        "原值": load,
        "滑动均值": load.rolling(24).mean(),
        "滑动最大": load.rolling(24).max(),
        "滑动标准差": load.rolling(24).std(),
        "滑动中位数": load.rolling(24).median(),
        "窗口有效数": load.rolling(24).count(),
    }
)
print(stats.iloc[[22, 23, 24, 25]].round(3).to_string())
print("\\n窗口有效数取值分布:", stats["窗口有效数"].value_counts().head(4).to_dict())
print("滑动均值 NaN 数:", int(stats["滑动均值"].isna().sum()),
      "| 窗口有效数 == 0 的行数:", int((stats["窗口有效数"] == 0).sum()))"""
    ),
    md(
        """
### 3.3 滞后与差分：造"历史特征"

预测明天负荷时，最有用的特征永远是**昨天同一时刻的负荷**。这就是 `shift`：

```python
load.shift(1)      # 上一小时
load.shift(24)     # 昨天同一小时  ← 日周期特征
load.shift(-1)     # 下一小时      ← ⚠️ 未来值，造特征时禁用
```

`diff` 和 `pct_change` 是 `shift` 的语法糖：

```python
load.diff(1)        ==  load - load.shift(1)          # 一阶差分（绝对值变化）
load.pct_change(1)  ==  (load - load.shift(1)) / load.shift(1)   # 变化率
```

**新版本注意**：`pct_change` 的 `fill_method` 参数在 pandas 3.0 已弃用默认值，
显式传 `fill_method=None` 才是"不做填充"的语义。
"""
    ),
    code(
        """lag = pd.DataFrame(
    {
        "原值": load,
        "滞后1h": load.shift(1),
        "滞后24h": load.shift(24),
        "一阶差分": load.diff(1),
        "小时环比": load.pct_change(1),
        "日环比": load.pct_change(24),
    }
)
print(lag.iloc[[0, 1, 23, 24, 25]].round(4).to_string())
print("\\n各列 NaN 数:")
print(lag.isna().sum().to_string())

# 差分与 shift 的恒等关系
print("\\ndiff(1) == load - load.shift(1) :",
      bool(np.allclose(load.diff(1).dropna(), (load - load.shift(1)).dropna())))"""
    ),
    md(
        """
---

### 3.4 难点深挖①：`min_periods` 的默认值会**静默制造 NaN**

**为什么难**：`rolling(24)` 的 `min_periods` 默认**等于 `window`**，
意思是"窗口里必须有 24 个有效值才给结果"。所以：

- 前 23 行必然 `NaN`（窗口没凑满）
- 窗口里只要有 1 个缺失值，这一行也是 `NaN`

而这两种 `NaN` **混在同一个结果里，无法区分**——你以为只有头部 23 个，
实际有一大堆。

**错误示范**（用默认值，直接拿去填充）：
"""
    ),
    code(
        """r_default = load.rolling(24).mean()
r_mp1 = load.rolling(24, min_periods=1).mean()
r_mp12 = load.rolling(24, min_periods=12).mean()

print(f"{'写法':<34}{'NaN 数':>8}{'第 0 行':>12}{'第 23 行':>12}")
print("-" * 66)
for name, r in [
    ("rolling(24)", r_default),
    ("rolling(24, min_periods=1)", r_mp1),
    ("rolling(24, min_periods=12)", r_mp12),
]:
    print(f"{name:<34}{int(r.isna().sum()):>8}{r.iloc[0]:>12.3f}{r.iloc[23]:>12.3f}")

print("\\n原序列缺失数:", int(load.isna().sum()))
print("说明：默认值下 NaN = 319，远大于「头部 23 + 缺失 23」，因为一个缺失")
print("      会把它后面 23 个窗口一起污染掉（0.5 * 24 * 23 量级）。")"""
    ),
    md(
        """
#### 缺失污染是怎么被"指数放大"的

设第 100 行缺失。那么第 100~123 行这 24 个窗口里**都含**第 100 行：

```text
roll[100] = mean(load[77:101])   ← 含缺失
roll[101] = mean(load[78:102])   ← 含缺失
...
roll[123] = mean(load[100:124])  ← 含缺失
roll[124] = mean(load[101:125])  ← 不含了，恢复正常
```

**1 个缺失值 → 24 个 `NaN` 结果**（而非 1 个）。
这份只有 23 个缺失值的序列，窗口均值里出现了 **319 个 `NaN`**。

**正误对照**：

| 写法 | `NaN` 数 | 第 0 行 | 语义 |
|---|---|---|---|
| `rolling(24)` | **319** | `NaN` | 只信任"窗口满员"的结果 |
| `rolling(24, min_periods=1)` | **0** | `277.300` | 有几行算几行（第 0 行 = 它自己） |
| `rolling(24, min_periods=12)` | 少一些 | `NaN` | 折中：至少半天数据才算 |

**判定规则**：

- **交卷给模型用** → `min_periods=1`（或 `=window//2`），别留一堆 `NaN`
  给下游去猜；但要知道**前几行是"样本不足的均值"，语义很弱**
- **做异常检测/统计对账** → 保持默认，因为"数不够就不下结论"才是对的
- **竞赛里最常见的扣分点** → 用默认值造完特征直接 `dropna()`，
  把头部 23 行 + 319 行连带丢掉，样本量凭空缩水 —— 而裁判只看你的结果表
"""
    ),
    md(
        """
### 3.5 难点深挖②：`rolling` **含当前行** → 把标签泄漏进特征

**为什么难**：算"当前时刻的特征"时，`roll[23]` 用到了 `load[23]`——**它自己**。
如果这一行是训练样本、`load[23]` 又恰好是标签相关量，模型就"看到答案"了。

**错误示范**（预测下一小时，却把当前小时算进特征）：

```python
feat = load.rolling(24).mean()          # ❌ roll[23] 含 load[23]
```

**正误对照**——正确做法是先 `shift(1)` 再 `rolling`：
"""
    ),
    code(
        """leak = load.rolling(24).mean()                      # ❌ 含当前行
safe = load.shift(1).rolling(24).mean()             # ✅ 只用过去 24 行

print("泄漏版  第 23 行:", round(leak.iloc[23], 4), "= mean(load[0:24])  ← 含 load[23]")
print("安全版  第 23 行:", leak.iloc[23] is np.nan, "(窗口未满)")
print("安全版  第 24 行:", round(safe.iloc[24], 4), "= mean(load[0:24])  ← 不含 load[24]")
print("\\n两口径逐行差异（前 5 个非 NaN）:")
diff_now = (leak - safe).dropna()
print(diff_now.head(5).round(4).tolist())
print("\\nNaN 数对比：泄漏版", int(leak.isna().sum()), "| 安全版", int(safe.isna().sum()))
print("→ 安全版多 1 个 NaN（第 23 行也不够了），这是**正确的代价**")"""
    ),
    md(
        """
> **判定规则（背下来）**：
> - 特征要**预测当前/未来** → 一律 `shift(n).rolling(w)`，`n ≥ 1`
> - 只是在**描述过去**（做报表、画趋势图） → `rolling(w)` 直接用，更直观
>
> 一句话：**`rolling` 是"含头含尾"的；做特征必须手动把头掐掉。**
"""
    ),
    md(
        """
### 3.6 难点深挖③：`groupby().rolling()` 的索引和 `groupby().shift()` **完全不同**

**为什么难**：同样挂在 `groupby` 后面，一族返回**两级 `MultiIndex`**，
另一族返回**原索引**。写 `df["新列"] = ...` 回贴时，
一个对得上、一个对不上（而 pandas 会**静默对齐成 `NaN`**，不报错）。

| 写法 | 索引 | 能否直接回贴原表 |
|---|---|---|
| `gb[col].shift(1)` | 原索引 | ✅ 可以 |
| `gb[col].diff(1)` | 原索引 | ✅ 可以 |
| `gb[col].rolling(24).mean()` | `(组键, 原索引)` | ❌ 必须先 `reset_index(drop=True)` |

先看结果长什么样：
"""
    ),
    code(
        """gb_roll = gb[VALUE_COL].rolling(24).mean()
gb_shift = gb[VALUE_COL].shift(1)

print("gb.rolling(24).mean()")
print("  类型:", type(gb_roll).__name__, "| 长度:", len(gb_roll), "| 索引层级:", gb_roll.index.names)
print("  前 3 个值:")
print(gb_roll.head(3).to_string())

print("\\ngb.shift(1)")
print("  类型:", type(gb_shift).__name__, "| 长度:", len(gb_shift), "| 索引层级:", gb_shift.index.names)

print("\\nrolling 结果 reset_index 后的列名:", list(gb_roll.reset_index().columns))"""
    ),
    md(
        """
看到 `索引层级: ['台区编号', None]` 了吗——**它比原表多了一级**。

各组的"第一个有效值"出现在**第 23 行**（组内窗口凑满 24），
所以三组的首个有效值分别是各自前 24 小时的均值：
"""
    ),
    code(
        """print("各组 rolling(24).mean() 的首个有效值（应等于该组前 24 小时均值）")
for name, sub in gb_roll.groupby(level=0):
    first_valid = sub.dropna().iloc[0]
    manual = curve.loc[curve[KEY_COL] == name, VALUE_COL].iloc[:24].mean()
    print(f"  {name}: 窗口值 {first_valid:.4f} | 手算 {manual:.4f} | 一致 {abs(first_valid - manual) < 1e-9}")

print("\\nNaN 数对比：")
print("  gb.rolling(24).mean() →", int(gb_roll.isna().sum()))
print("  gb.shift(1)           →", int(gb_shift.isna().sum()), "（仅每组首行 3 个 + 原有缺失 48 个）")"""
    ),
    md(
        """
**怎么把窗口结果安全回贴原表**——三种写法，推荐第一种：

```python
# ✅ 推荐：直接丢掉 MultiIndex，靠"位置"对齐（前提：顺序与原表一致）
curve["滑动均值"] = gb_roll.reset_index(drop=True).to_numpy()

# ✅ 安全但更绕：用 transform 套一层（保留原索引，pandas 按标签对齐）
curve["滑动均值"] = gb[VALUE_COL].transform(lambda s: s.rolling(24).mean())

# ❌ 直接赋值：pandas 会明确报 TypeError（这是好事，挡住了）
# curve["滑动均值"] = gb_roll
```

下面实测这三种写法，**外加一个真正危险的第四种**：
"行数一样、索引看着也对，但顺序已经错了"——这种 pandas **不报错、不产生 NaN**，
只是把值贴到了错误的行上。
"""
    ),
    code(
        """demo = curve[[KEY_COL, "时间戳", VALUE_COL]].copy()

# 写法 A：丢掉 MultiIndex，转 ndarray 赋回
demo["写法A_去索引"] = gb_roll.reset_index(drop=True).to_numpy()
# 写法 B：套 transform，保留原索引
demo["写法B_transform"] = gb[VALUE_COL].transform(lambda s: s.rolling(24).mean())

print(f"{'写法':<20}{'NaN 数':>8}{'非空数':>10}")
print("-" * 38)
for col in ["写法A_去索引", "写法B_transform"]:
    print(f"{col:<20}{int(demo[col].isna().sum()):>8}{int(demo[col].notna().sum()):>10}")

ok = (demo["写法A_去索引"].dropna().index
      .intersection(demo["写法B_transform"].dropna().index))
print("\\nA 与 B 在共同索引上完全一致:",
      bool(np.allclose(demo.loc[ok, "写法A_去索引"], demo.loc[ok, "写法B_transform"])))

print("\\n【写法 C】直接赋 MultiIndex —— pandas 明确报错（好事）")
try:
    demo["写法C"] = gb_roll
    print("   居然成功了")
except TypeError as exc:
    print("   ->", type(exc).__name__, ":", exc)

print("\\n【写法 D】顺序已错但索引重排过 —— 不报错、不 NaN，静默错位")
bad = (
    gb_roll.reset_index(drop=True)
    .sort_values(ascending=False, na_position="last")   # 顺序被打乱
    .reset_index(drop=True)                             # 索引又变回 0..n-1（骗过了检查）
)
demo["写法D_顺序错"] = bad

both_valid = demo["写法A_去索引"].notna() & demo["写法D_顺序错"].notna()
same = int((demo.loc[both_valid, "写法A_去索引"] == demo.loc[both_valid, "写法D_顺序错"]).sum())
print(f"   D 的 NaN 数: {int(demo['写法D_顺序错'].isna().sum())}（和 A 一样，看不出异常）")
print(f"   与 A 逐行相同的行数: {same} / {int(both_valid.sum())}"
      f"  → 错位率 {1 - same / int(both_valid.sum()):.2%}")
print("\\n   前 3 行对照:")
print(demo.loc[both_valid, ["写法A_去索引", "写法D_顺序错"]].head(3).round(3).to_string())"""
    ),
    md(
        """
**这一步的结论要背**：

- `gb.rolling()` → **`reset_index(drop=True)` 或 `.to_numpy()`** 再回贴
- `gb.shift()` / `gb.diff()` / `gb.transform()` → **可以直接回贴**（索引就是原索引）
- 直接赋 `MultiIndex` → pandas 3.0 **明确报 `TypeError`**，不用担心
- 真正危险的是**写法 D**：顺序错了但索引被"修好"了，
  NaN 数、行数、dtype 全部正常，**只有值放错了行**——这种错误只能靠
  "抽样手算对账"发现

> **判定规则**：回贴窗口结果后，**必须**抽 2~3 行手算验证：
> ```python
> i = 29                                        # STATION_C_01 组内第 29 行
> manual = sub[VALUE_COL].iloc[6:30].mean()     # 该行前 24 小时均值
> assert abs(sub["滑动均值"].iloc[i] - manual) < 1e-9
> ```
> 别只看 `notna().sum()`——它对写法 D 完全无效。
"""
    ),

    md(
        """
---

### 3.7 难点深挖④：`ewm` 的四个参数，只有一个是必须记住的

**为什么难**：`ewm` 同时接受 `span` / `halflife` / `alpha` / `com` 四种写法，
它们**互相等价**但数值完全不同；再加上 `adjust` 的默认值会改结果。
最容易错的是**以为 `span=24` 和 `alpha=0.24` 是一回事**。

三者的换算关系：

```text
alpha = 2 / (span + 1)          span = 24  →  alpha = 2/25 = 0.08
alpha = 1 - 0.5 ** (1/halflife) halflife = 24  →  alpha ≈ 0.0285
```

**正误对照**：
"""
    ),
    code(
        """e_span = load.ewm(span=24).mean()
e_alpha = load.ewm(alpha=2 / 25).mean()
e_alpha_wrong = load.ewm(alpha=0.24).mean()
e_hl = load.ewm(halflife=24).mean()

print("ewm(span=24)       首个值:", round(e_span.iloc[0], 4), "| 末值:", round(e_span.iloc[-1], 4))
print("ewm(alpha=2/25)    首个值:", round(e_alpha.iloc[0], 4), "| 末值:", round(e_alpha.iloc[-1], 4))
print("  → span=24 与 alpha=2/25 等价:", bool(np.allclose(e_span, e_alpha)))
print("\\newm(alpha=0.24)   末值:", round(e_alpha_wrong.iloc[-1], 4), " ← 权重错了一个量级")
print("ewm(halflife=24)   末值:", round(e_hl.iloc[-1], 4), " ← 半衰期口径，衰减更慢")

print("\\n和简单滑动均值比:")
print("  ewm(span=24) 末值:", round(e_span.iloc[-1], 4))
print("  rolling(24)  末值:", round(load.rolling(24).mean().iloc[-1], 4))"""
    ),
    md(
        """
**`adjust` 的区别**（默认 `True`）：

| `adjust` | 首值 | 语义 |
|---|---|---|
| `True`（默认） | `= load[0]` | 归一化加权：早期样本权重被"补齐"，无偏但前期跳变慢 |
| `False` | `= load[0]` | 递推式 `y[t] = (1-α)·y[t-1] + α·x[t]`，**在线计算/流式场景**用的就是它 |

两者首值相同、从第 1 行开始分叉：
"""
    ),
    code(
        """a_true = load.ewm(span=24, adjust=True).mean()
a_false = load.ewm(span=24, adjust=False).mean()

print("adjust=True  前 4 值:", a_true.head(4).round(4).tolist())
print("adjust=False 前 4 值:", a_false.head(4).round(4).tolist())

# 手算 adjust=False 的第 1 个值，验证递推式
alpha = 2 / (24 + 1)
manual_1 = (1 - alpha) * load.iloc[0] + alpha * load.iloc[1]
print(f"\\n手算递推 y[1] = (1-{alpha:.4f})*{load.iloc[0]:.2f} + {alpha:.4f}*{load.iloc[1]:.2f}"
      f" = {manual_1:.4f}")
print("adjust=False 实际 y[1] =", round(a_false.iloc[1], 4), "| 一致:",
      abs(manual_1 - a_false.iloc[1]) < 1e-9)

print("\\n末期两者已收敛，差值:", round(abs(a_true.iloc[-1] - a_false.iloc[-1]), 6))"""
    ),
    md(
        """
> **判定规则（只记这三条）**：
> 1. 想表达"最近 24 期的加权平均" → 用 **`span=window`**，别碰 `alpha`
> 2. 想表达"半衰期 24 期" → 用 **`halflife=24`**（物理意义最清楚）
> 3. 做**流式/在线更新** → 必须 **`adjust=False`**，否则每来一个新点
>    都要重算历史权重，跟递推式对不上
"""
    ),
    md(
        """
### 3.8 时间窗口 `rolling("24h")`：必须先是 `DatetimeIndex`

固定窗口 `rolling(24)` 数的是**行**。如果数据有缺小时（比如某台区跳采），
"24 行"就不是"24 小时"。这时要用时间窗口 `rolling("24h")`。

**两个硬约束**：

1. 索引**必须**是 `DatetimeIndex`，否则直接报
   `ValueError: window must be an integer 0 or greater`
2. 变长窗口的 `min_periods` **默认是 1**（不是窗口长度），
   所以前 23 行**不是 `NaN`**——和固定窗口的行为不一样
"""
    ),
    code(
        """print("① 在 RangeIndex 上用时间窗口 —— 报错")
try:
    load.rolling("24h").mean()
    print("   居然成功了")
except ValueError as exc:
    print("   ->", type(exc).__name__, ":", exc)

print("\\n② 在 DatetimeIndex 上 —— 正常")
r_time = load_ts.rolling("24h").mean()
r_time_mp24 = load_ts.rolling("24h", min_periods=24).mean()
r_fixed = load.rolling(24).mean()

print(f"  {'写法':<34}{'NaN 数':>8}{'第 0 行':>12}{'第 23 行':>12}")
print("-" * 66)
for name, r in [
    ('load_ts.rolling("24h")', r_time),
    ('load_ts.rolling("24h", min_periods=24)', r_time_mp24),
    ("load.rolling(24)", r_fixed),
]:
    print(f"{name:<34}{int(r.isna().sum()):>8}{r.iloc[0]:>12.3f}{r.iloc[23]:>12.3f}")

print("\\n→ 时间窗口默认 min_periods=1：第 0 行就是 load[0] 自己（窗口里只有 1 个点）")"""
    ),
    md(
        """
> **判定规则**：
> - 数据**等间隔且无缺行** → `rolling(24)` 更简单直接
> - 数据**可能跳采/不等间隔** → `rolling("24h")`，并且**显式写 `min_periods`**，
>   否则你会得到一堆"只有 1 个点也算出来的均值"
"""
    ),
    md(
        """
---

### 3.9 难点深挖⑤：全局阈值 vs **滑动阈值**——动态告警的代价

**为什么难**：两种口径都能跑出结果、都不会报错，但**判出来的东西完全不是一回事**。
全局阈值回答的是"这个值在本月算不算高"，滑动阈值回答的是
"这个值在它自己这个时段算不算高"。选错口径 = 告警要么天天响、要么永远不响。

**数据事实**（本台区负荷，逐小时）：

```text
全局 P95 阈值        = 1025.50   ← 一条水平线
滑动 P95 阈值（24h） = 620.36 ~ 1172.79，均值 953.55   ← 一条跟着曲线起伏的线
```

**错误示范**（只看全局阈值，用"是否超过全局 P95"定义高负荷告警）：
"""
    ),
    code(
        """thr_glob = load.quantile(0.95)
thr_dyn = load.rolling(24, min_periods=12).quantile(0.95)

flag_glob = load > thr_glob
flag_dyn = load > thr_dyn

print(f"全局 P95 = {float(thr_glob):.2f}  （一条水平线）")
print(f"滑动 P95 = {float(thr_dyn.min()):.2f} ~ {float(thr_dyn.max()):.2f}"
      f"，均值 {float(thr_dyn.mean()):.2f}（跟着曲线起伏）")
print()
print(f"{'口径':<12}{'判出条数':>10}")
print("-" * 24)
print(f"{'全局':<12}{int(flag_glob.sum()):>10}")
print(f"{'滑动':<12}{int(flag_dyn.sum()):>10}")
print(f"{'两者重叠':<12}{int((flag_glob & flag_dyn).sum()):>10}")
print(f"\\n滑动检出、全局漏掉: {int((flag_dyn & ~flag_glob).sum())} 条")
print(f"全局检出、滑动漏掉: {int((flag_glob & ~flag_dyn).sum())} 条")
missed = load[flag_dyn & ~flag_glob]
print(f"被全局漏掉的值域: {float(missed.min()):.2f} ~ {float(missed.max()):.2f}"
      f"（全是真实用电尖峰，只是没到全月 P95）")"""
    ),
    md(
        """
**结果互相对照**：

| 口径 | 判出条数 | 与另一方重叠 | 特点 |
|---|---|---|---|
| 全局 P95 | **107** | 55 | 阈值恒定，抓"绝对高值" |
| 滑动 P95（24h） | **194** | 55 | 阈值起伏，抓"相对高值" |
| 滑动检出 / 全局漏 | **139** | — | 值域 662.74 ~ 1025.48 的**局部尖峰** |
| 全局检出 / 滑动漏 | **52** | — | 相对本时段其实不算高 |

两个方向的偏差都真实存在：

- **全局漏 139 条** → 夜间或低谷时段的相对尖峰（比如深夜本该 300，
  突然冲到 900）在全局口径下"看着不高"，实际上这是设备异常的信号
- **全局多 52 条** → 晚高峰整体抬高时，一条 900 的负荷并不异常，
  但超过了全局 P95，造成误报

**判定规则**：

| 业务场景 | 该用哪个 | 理由 |
|---|---|---|
| 找"设备该不该检修"的异常 | **滑动阈值** | 关注"相对自身历史"的突变 |
| 找"容量是否够用"的越限 | **全局阈值**（或容量定值） | 关注"绝对物理量"是否超限 |
| 不清楚业务口径时 | 两个都算，**比差集** | 差集里的值域分布会告诉你业务真相 |

> **一句话**：**阈值是不是"动态"取决于业务问的是"相对于谁"**——
> 相对于自己的历史 → 滑动；相对于物理上限 → 全局。
> 竞赛里如果题目只说"识别异常高负荷"，**把两种口径都写出来并说明选择理由**
> 比蒙一个更能拿过程分。
"""
    ),
    md(
        """
#### 附带一个 `center=True` 的实测结论

`center=True` 让窗口**以当前行为中心**（前后各取一半），
所以做一个"不改变趋势的平滑"非常合适。但它的边界行为也反直觉：

| 写法 | `NaN` 数 | 说明 |
|---|---|---|
| `rolling(24, center=True, min_periods=12).mean()` | **0** | 首尾各有半个窗口，靠 `min_periods` 救回来 |
| `rolling(24, center=True, min_periods=24).mean()` | **319** | 与普通 `rolling(24)` 完全一致的 `NaN` 数 |

而平滑的效果非常明显——平滑后的**逐小时差分标准差**从 `65.18` 降到 `3.44`，
降幅 **94.7%**：
"""
    ),
    code(
        """center_mean = load.rolling(24, center=True, min_periods=1).mean()

print("原始序列一阶差分 std :", round(load.diff().std(), 4))
print("平滑后一阶差分 std   :", round(center_mean.diff().std(), 4))
print("波动降幅             :", f"{(1 - center_mean.diff().std() / load.diff().std()) * 100:.2f}%")

print("\\ncenter=True 的 NaN 数:")
for mp in [1, 12, 24]:
    r = load.rolling(24, center=True, min_periods=mp).mean()
    print(f"  min_periods={mp:<3} NaN = {int(r.isna().sum()):<4} "
          f"| 第 0 行 = {'NaN' if pd.isna(r.iloc[0]) else round(r.iloc[0], 3)}")

print("\\n平滑前 3 个差值:", load.diff().dropna().head(3).round(2).tolist())
print("平滑后 3 个差值:", center_mean.diff().dropna().head(3).round(2).tolist())"""
    ),
    md(
        """
> **判用规则**：`center=True` 会用到**当前行之后的值**——
> 做**可视化平滑 / 报表趋势**没问题（反正整段数据都在手上）；
> 做**预测特征**时等于泄漏未来，必须换回 `shift(1).rolling(...)`。
"""
    ),

    md(
        """
---

### 3.10 综合：一张 `24h` 时序特征表

把本章所有手段串起来，产出可以直接喂模型的特征表：

| 特征 | 写法 | 类型 |
|---|---|---|
| 滑动均值 | `rolling(24).mean()` | 趋势 |
| 滑动标准差 | `rolling(24).std()` | 波动 |
| 滞后 1h | `shift(1)` | 短期自相关 |
| 滞后 24h | `shift(24)` | **日周期**（最重要的单特征） |
| 日环比 | `pct_change(24)` | 同比变化 |
| 累计均值 | `expanding().mean()` | 长趋势基准 |

拼装时注意两件事：**特征列全部用 `shift` 版**（防泄漏）、
最后统一看 `dropna()` 后还剩多少样本。
"""
    ),
    code(
        """feat = pd.DataFrame(index=load.index)
feat["负荷值"] = load
feat["滑动均值24"] = load.shift(1).rolling(24).mean()
feat["滑动标准差24"] = load.shift(1).rolling(24).std()
feat["滞后1h"] = load.shift(1)
feat["滞后24h"] = load.shift(24)
feat["日环比"] = load.pct_change(24, fill_method=None)
feat["累计均值"] = load.expanding().mean()

print("特征表形状:", feat.shape)
print("各列 NaN 数:")
print(feat.isna().sum().to_string())
print("\\n完整样本（dropna 后）行数:", len(feat.dropna()), f"/ {len(feat)}")
print("\\n末行特征:")
print(feat.iloc[-1].round(4).to_string())"""
    ),
    md(
        """
---

## 四、易错点清单

1. **没排序就 rolling** → 按行号滑，结果纯错但不报错
2. **`rolling(24)` 默认 `min_periods=24`** → 前 23 行 + 被缺失污染的窗口全是 `NaN`
3. **1 个缺失值污染 24 个窗口结果** → `NaN` 数量指数放大，`dropna()` 会误删大量样本
4. **`rolling` 含当前行** → 造特征时等于把标签泄漏进特征，必须 `shift(1)` 打底
5. **`gb.rolling()` 返回两级 `MultiIndex`** → 直接赋回原表会报 `TypeError`，必须先 `reset_index(drop=True)`
6. **顺序错了但索引"看起来对"** → 不报错、不 NaN，静默错位；只能靠抽样手算对账
7. **`gb.shift()` / `gb.diff()` 返回原索引** → 可直接回贴，但每组首行是 `NaN`
8. **`s.shift(-1)` 是取未来值** → 造特征时的低级但致命错误
9. **`ewm(span=24)` ≠ `ewm(alpha=0.24)`** → 换算关系是 `alpha = 2/(span+1)`
10. **`ewm` 默认 `adjust=True`** → 流式/递推场景必须显式 `adjust=False`
11. **`rolling("24h")` 需要 `DatetimeIndex`** → 在 `RangeIndex` 上直接 `ValueError`
12. **时间窗口的 `min_periods` 默认是 1** → 与固定窗口不同，前几行算出来是"单点均值"
13. **`center=True` 会用到当前行之后的值** → 平滑可以，做特征就是泄漏未来
14. **全局阈值 vs 滑动阈值是两种业务口径** → 本数据上相差 139 条 / 52 条，不是"差不多"

---

## 五、本章小结

| 我想…… | 用哪个 | 关键参数 |
|---|---|---|
| 滑动趋势特征 | `s.rolling(w).mean()` | `min_periods` 显式写 |
| 滑动波动特征 | `s.rolling(w).std()` | — |
| 稳健滑动趋势 | `s.rolling(w, center=True).median()` | `min_periods ≤ w//2` |
| 滞后/周期特征 | `s.shift(n)` | `n=24` 是日周期 |
| 变化率特征 | `s.pct_change(n, fill_method=None)` | 显式关掉填充 |
| 从头累积基准 | `s.expanding().mean()` | `min_periods` 默认为 1 |
| 指数平滑 | `s.ewm(span=w).mean()` | 流式用 `adjust=False` |
| 不等间隔时序 | `s.rolling("24h")` | 需 `DatetimeIndex` |
| 分组滑动 | `gb[col].rolling(w)` | **记得 `reset_index(drop=True)`** |

**一句话总结**：窗口家族的坑不在"会不会写"，而在
**"默认值替你做的决定"**——`min_periods` 替你决定丢多少样本、
`rolling` 替你决定含不含当前行、`gb.rolling` 替你决定返回几级索引。
竞赛里要把这些默认值**全部显式写出来**，因为写出来才对得起"流程完整性"这项分。
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch12 练习 —— 窗口计算

> 数据：`data/load_curve.csv`（3 个台区 × 2160 小时，2026-01-01 ~ 2026-03-31）
> 单台区序列已备好：`load`（RangeIndex）、`load_ts`（DatetimeIndex）

**做题规则**：从上往下依次填，每个 `____` 处跑通再往下。
验收块 `assert` 不要改，它是你自己对账用的。

| 题号 | 考点 |
|---|---|
| TODO(1) | 固定窗口均值 + `min_periods` 默认值 |
| TODO(2) | `min_periods=1` 的代价与收益 |
| TODO(3) | `shift` / `diff` / `pct_change` 造滞后特征 |
| TODO(4) | `expanding` 累计均值 |
| TODO(5) | `ewm` 的 `span` 与 `adjust` |
| TODO(6) | 时间窗口 `rolling("24h")` |
| TODO(7) | 证明窗口按**行序**滑 |
| TODO(8) | 标签泄漏：`shift(1)` 打底 |
| TODO(9) | 分组滚动与两级 `MultiIndex` |
| TODO(10) | 分组 `shift` / `diff` 的索引差异 |
| TODO(11) | `rolling().apply` 自定义窗口统计 |
| TODO(12) | 动态阈值 vs 全局阈值（告警口径之争） |
| 综合① | 24h 时序特征表 + 泄漏自检 |
| 综合② | 分组滑动特征回贴原表 + 对账 |
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
        """# @@todo(1) 用默认参数算 24 小时滑动均值，存入 roll_default；把它的 NaN 个数存入 n_nan
# @@hint load.rolling(24).mean() ；再用 .isna().sum() 统计
roll_default = load.rolling(24).mean()
n_nan = int(roll_default.isna().sum())
# @@end
# ---- 验收 ----
assert n_nan == 319, f"默认参数下应有 319 个 NaN，实际 {n_nan}"
assert abs(roll_default.iloc[23] - load.iloc[:24].mean()) < 1e-9, "roll[23] 应等于前 24 行均值"
assert bool(pd.isna(roll_default.iloc[22])), "第 22 行窗口未满，应为 NaN"
print("TODO(1) 通过 -> NaN", n_nan, "| roll[23] =", round(roll_default.iloc[23], 4))
print("   注意：NaN 数远大于「头部 23 + 序列缺失 23」，缺失会污染后续窗口")"""
    ),
    code(
        """# @@todo(2) 改成 min_periods=1，存入 roll_mp1；统计其 NaN 个数存入 n_nan_mp1，
#         并把第 0 行的值存入 first_mp1
# @@hint load.rolling(24, min_periods=1).mean()
roll_mp1 = load.rolling(24, min_periods=1).mean()
n_nan_mp1 = int(roll_mp1.isna().sum())
first_mp1 = float(roll_mp1.iloc[0])
# @@end
# ---- 验收 ----
assert n_nan_mp1 == 0, f"min_periods=1 时不该有 NaN，实际 {n_nan_mp1}"
assert abs(first_mp1 - load.iloc[0]) < 1e-9, "第 0 行窗口只有 1 个点，应等于原值本身"
assert int(load.isna().sum()) == 23, "单台区缺失应为 23"
assert abs(roll_mp1.iloc[23] - roll_default.iloc[23]) < 1e-9, "第 23 行两版应一致（窗口都满 24）"
print("TODO(2) 通过 -> NaN 0 | first =", first_mp1, "| iloc[23] 两版一致")"""
    ),
    code(
        """# @@todo(3) 造三个滞后特征：滞后1h、滞后24h、日环比，分别存入 lag1 / lag24 / pc24
# @@hint load.shift(1) ；load.shift(24) ；load.pct_change(24, fill_method=None)
lag1 = load.shift(1)
lag24 = load.shift(24)
pc24 = load.pct_change(24, fill_method=None)
# @@end
# ---- 验收 ----
assert int(lag1.isna().sum()) == 24, f"滞后1h 应有 24 个 NaN（首行 1 + 原缺失 23），实际 {int(lag1.isna().sum())}"
assert lag24.iloc[24] == load.iloc[0], "lag24[24] 应等于 load[0]"
assert abs(pc24.iloc[24] - (load.iloc[24] - load.iloc[0]) / load.iloc[0]) < 1e-9, "日环比公式应成立"
assert bool(np.allclose(load.diff(1).dropna(), (load - lag1).dropna())), "diff(1) 应等于 s - s.shift(1)"
print("TODO(3) 通过 -> lag1 NaN", int(lag1.isna().sum()), "| lag24[24] =", round(lag24.iloc[24], 2),
      "| 日环比[24] =", round(pc24.iloc[24], 4))"""
    ),
    code(
        """# @@todo(4) 算累计均值存入 exp_mean；它的 NaN 个数存入 n_nan_exp
# @@hint load.expanding().mean() ；想只在攒够 24 个点后才给值就传 min_periods=24
exp_mean = load.expanding().mean()
n_nan_exp = int(exp_mean.isna().sum())
# @@end
# ---- 验收 ----
assert n_nan_exp == 0, "expanding 默认 min_periods=1，不该有 NaN"
assert abs(exp_mean.iloc[0] - load.iloc[0]) < 1e-9, "第 0 行累计均值 = 原值"
assert abs(exp_mean.iloc[-1] - load.mean()) < 1e-6, "最后一行累计均值 = 全序列均值"
assert int(load.expanding(min_periods=24).mean().isna().sum()) == 23, "min_periods=24 时应有 23 个 NaN"
print("TODO(4) 通过 -> 累计末值", round(exp_mean.iloc[-1], 4), "| 序列均值", round(load.mean(), 4))"""
    ),
    code(
        """# @@todo(5) 算 span=24 的指数加权均值存入 ewm_span，并验证它与 alpha=2/25 等价，
#         布尔结果存入 same；再算 adjust=False 版本存入 ewm_noadj
# @@hint load.ewm(span=24).mean() ；np.allclose(a, b) ；load.ewm(span=24, adjust=False).mean()
ewm_span = load.ewm(span=24).mean()
same = bool(np.allclose(ewm_span, load.ewm(alpha=2 / 25).mean()))
ewm_noadj = load.ewm(span=24, adjust=False).mean()
# @@end
# ---- 验收 ----
assert same is True, "span=24 应等价于 alpha=2/25"
assert abs(ewm_span.iloc[0] - load.iloc[0]) < 1e-9, "ewm 首值应等于原序列首值"
alpha = 2 / 25
manual = (1 - alpha) * load.iloc[0] + alpha * load.iloc[1]
assert abs(ewm_noadj.iloc[1] - manual) < 1e-9, "adjust=False 应符合 y=(1-a)y+ax 递推"
assert abs(ewm_span.iloc[-1] - ewm_noadj.iloc[-1]) < 1.0, "两者末期应已接近收敛"
print("TODO(5) 通过 -> span==alpha:", same, "| adjust=False y[1] =", round(ewm_noadj.iloc[1], 4))"""
    ),
    code(
        """# @@todo(6) 在 DatetimeIndex 上用时间窗口算 24 小时均值，存入 time_roll（默认 min_periods）；
#         再用 min_periods=24 算一份存入 time_roll_mp，两个 NaN 数分别存入 n_time / n_time_mp
# @@hint load_ts.rolling("24h").mean() ；load_ts.rolling("24h", min_periods=24).mean()
time_roll = load_ts.rolling("24h").mean()
time_roll_mp = load_ts.rolling("24h", min_periods=24).mean()
n_time = int(time_roll.isna().sum())
n_time_mp = int(time_roll_mp.isna().sum())
# @@end
# ---- 验收 ----
assert n_time == 0, f"时间窗口默认 min_periods=1，不该有 NaN，实际 {n_time}"
assert n_time_mp == 319, f"min_periods=24 时应与固定窗口一致（319），实际 {n_time_mp}"
assert abs(time_roll.iloc[23] - roll_default.iloc[23]) < 1e-9, "等间隔数据下时间窗口与固定窗口应一致"
assert abs(time_roll.iloc[0] - load_ts.iloc[0]) < 1e-9, "第 0 行窗口只有 1 个点"
# RangeIndex 上应当直接报错
try:
    load.rolling("24h").mean()
    raise AssertionError("RangeIndex 上用时间窗口应该抛 ValueError")
except AssertionError:
    raise
except ValueError:
    pass
print("TODO(6) 通过 -> 默认", n_time, "个 NaN | min_periods=24 时", n_time_mp, "个")"""
    ),
    md(
        """
---

## 进阶题
"""
    ),
    code(
        """# @@todo(7) 证明窗口是「按行序」滑的：把 load 倒序存入 rev，
#         算它的 24 窗口均值存入 rev_roll；验证 rev_roll[23] 等于原序列最后 24 行的均值
# @@hint rev = load.iloc[::-1].reset_index(drop=True) ；load.iloc[-24:].mean()
rev = load.iloc[::-1].reset_index(drop=True)
rev_roll = rev.rolling(24).mean()
# @@end
# ---- 验收 ----
assert abs(rev_roll.iloc[23] - load.iloc[-24:].mean()) < 1e-9, "倒序后第 23 行窗口 = 原序列末 24 行"
assert abs(rev_roll.iloc[23] - roll_default.iloc[23]) > 1, "倒序结果必须与原序不同（否则说明不是按行序）"
assert pd.isna(rev_roll.iloc[22]), "倒序后第 22 行窗口仍未满"
print("TODO(7) 通过 -> 倒序 roll[23] =", round(rev_roll.iloc[23], 4),
      "| 原序 roll[23] =", round(roll_default.iloc[23], 4), "→ 完全不同，证明只认行序")"""
    ),
    code(
        """# @@todo(8) 演示标签泄漏：直接 rolling 存入 leak，先 shift(1) 再 rolling 存入 safe；
#         两者的 NaN 个数分别存入 n_leak / n_safe
# @@hint load.rolling(24).mean() ；load.shift(1).rolling(24).mean()
leak = load.rolling(24).mean()
safe = load.shift(1).rolling(24).mean()
n_leak = int(leak.isna().sum())
n_safe = int(safe.isna().sum())
# @@end
# ---- 验收 ----
assert abs(leak.iloc[23] - load.iloc[:24].mean()) < 1e-9, "泄漏版 roll[23] 含 load[23] 自己"
assert abs(safe.iloc[24] - load.iloc[:24].mean()) < 1e-9, "安全版第 24 行才等于前 24 行均值"
assert pd.isna(safe.iloc[23]), "安全版第 23 行窗口不足，应为 NaN"
assert n_safe == n_leak + 1, f"安全版应恰好多 1 个 NaN（{n_leak+1}），实际 {n_safe}"
assert bool(np.allclose((leak - safe).dropna().abs().gt(0).sum() > 0, True)), "两版数值必须有差异"
print("TODO(8) 通过 -> 泄漏版", n_leak, "个 NaN | 安全版", n_safe, "个（多 1 个是正确代价）")"""
    ),
    code(
        """# @@todo(9) 用分组滚动算 24 小时均值：结果存入 gb_roll；它的 NaN 数存入 n_gb_roll；
#         各组「首个有效值」拿不到就遍历，先用 to_frame 存一份 gb_roll_df（列名改为「窗口均值」）
# @@hint gb[VALUE_COL].rolling(24).mean() ；gb_roll.reset_index() 后列名会是 ['台区编号','level_1','负荷值']
gb_roll = gb[VALUE_COL].rolling(24).mean()
n_gb_roll = int(gb_roll.isna().sum())
gb_roll_df = gb_roll.rename(VALUE_COL).reset_index()
# @@end
# ---- 验收 ----
assert n_gb_roll == 829, f"分组滚动应有 829 个 NaN（3 组 × 含组内头部与缺失污染），实际 {n_gb_roll}"
assert gb_roll.index.names == [KEY_COL, None], f"应返回两级 MultiIndex，实际 {gb_roll.index.names}"
assert list(gb_roll_df.columns) == [KEY_COL, "level_1", VALUE_COL], \\
    f"reset_index 后列名应为 [台区编号, level_1, 负荷值]，实际 {list(gb_roll_df.columns)}"
assert gb_roll_df.shape == (6480, 3), f"形状应为 (6480, 3)，实际 {gb_roll_df.shape}"
assert int(gb_roll_df[VALUE_COL].notna().sum()) == 6480 - 829, "有效值个数应守恒"
print("TODO(9) 通过 -> 索引层级", gb_roll.index.names, "| NaN", n_gb_roll,
      "| reset 后列名", list(gb_roll_df.columns))"""
    ),
    code(
        """# @@todo(10) 对比分组 shift / diff 的索引与 NaN：
#          gb_shift = 组内滞后 1，gb_diff = 组内一阶差分；
#          两者 NaN 数分别存入 n_gb_shift / n_gb_diff
# @@hint gb[VALUE_COL].shift(1) ；gb[VALUE_COL].diff(1)
gb_shift = gb[VALUE_COL].shift(1)
gb_diff = gb[VALUE_COL].diff(1)
n_gb_shift = int(gb_shift.isna().sum())
n_gb_diff = int(gb_diff.isna().sum())
# @@end
# ---- 验收 ----
assert gb_shift.index.names == [None], f"分组 shift 应返回原索引（单层），实际 {gb_shift.index.names}"
assert gb_diff.index.names == [None], "分组 diff 也应返回原索引"
assert len(gb_shift) == 6480, "长度应与原表一致，可直接回贴"
assert n_gb_shift == 51, f"分组 shift NaN 应为 51（每组首行 3 个 + 原有缺失 48），实际 {n_gb_shift}"
assert n_gb_diff == 84, f"分组 diff NaN 应为 84，实际 {n_gb_diff}"
# 组边界：每组第一行必为 NaN
first_rows = gb_shift.groupby(curve[KEY_COL]).head(1)
assert bool(first_rows.isna().all()), "每组第一行必须是 NaN（组内无前一行）"
print("TODO(10) 通过 -> shift NaN", n_gb_shift, "| diff NaN", n_gb_diff,
      "| 索引层级", gb_shift.index.names, "（与 rolling 的两级索引不同）")"""
    ),
    code(
        """# @@todo(11) 用 rolling().apply 算自定义窗口统计：24 小时窗口的极差，存入 roll_span
#           （窗口函数用脚手架里的 window_span，raw=True）
# @@hint load.rolling(24, min_periods=24).apply(window_span, raw=True)
roll_span = load.rolling(24, min_periods=24).apply(window_span, raw=True)
# @@end
# ---- 验收 ----
assert int(roll_span.isna().sum()) == 319, f"与 rolling(24).max() 的 NaN 模式应一致（319），实际 {int(roll_span.isna().sum())}"
manual_first = load.iloc[:24].max() - load.iloc[:24].min()
assert abs(roll_span.iloc[23] - manual_first) < 1e-9, "第 23 行应等于前 24 行的 max-min"
# 与 max()-min() 两种写法对账
pair = load.rolling(24).max() - load.rolling(24).min()
assert bool(np.allclose(roll_span.dropna(), pair.dropna())), "apply 写法应等价于 max()-min()"
print("TODO(11) 通过 -> roll_span[23] =", round(roll_span.iloc[23], 4),
      "| 与 max-min 完全一致:", True)"""
    ),
    code(
        """# @@todo(12) 动态阈值 vs 全局阈值（高负荷告警的口径之争）：
#           全局 P95 存入 thr_glob；滑动 P95（24 窗口、min_periods=12）存入 thr_dyn
#           两个布尔标记存入 flag_glob / flag_dyn（load 是否超过各自阈值）
# @@hint thr_glob = load.quantile(0.95)
#        thr_dyn = load.rolling(24, min_periods=12).quantile(0.95)
#        flag_glob = load > thr_glob ；flag_dyn = load > thr_dyn
thr_glob = load.quantile(0.95)
thr_dyn = load.rolling(24, min_periods=12).quantile(0.95)
flag_glob = load > thr_glob
flag_dyn = load > thr_dyn
# @@end
# ---- 验收 ----
assert abs(float(thr_glob) - 1025.496) < 1e-2, f"全局 P95 应为 1025.50，实际 {float(thr_glob):.3f}"
assert int(thr_dyn.isna().sum()) == 11, f"滑动 P95 应有 11 个 NaN，实际 {int(thr_dyn.isna().sum())}"
assert float(thr_dyn.mean()) < float(thr_glob), "滑动阈值均值应低于全局阈值（局部口径更敏感）"
assert int(flag_glob.sum()) == 107, f"全局口径应判出 107 条，实际 {int(flag_glob.sum())}"
assert int(flag_dyn.sum()) == 194, f"滑动口径应判出 194 条，实际 {int(flag_dyn.sum())}"
assert int((flag_glob & flag_dyn).sum()) == 55, f"重叠应为 55，实际 {int((flag_glob & flag_dyn).sum())}"
assert int((flag_dyn & ~flag_glob).sum()) == 139, f"全局漏掉的应有 139 条，实际 {int((flag_dyn & ~flag_glob).sum())}"
assert int((flag_glob & ~flag_dyn).sum()) == 52, f"滑动漏掉的应有 52 条，实际 {int((flag_glob & ~flag_dyn).sum())}"
# 被全局漏掉的都是「局部尖峰」：值不算全月最高，但相对本时段很高
missed = load[flag_dyn & ~flag_glob]
assert float(missed.min()) > 600, f"被全局漏掉的值应偏小，实际最小值 {float(missed.min()):.2f}"
assert float(missed.max()) <= float(thr_glob), "被全局漏掉的必然都 ≤ 全局阈值"
# 附：center=True 的平滑边界
assert int(load.rolling(24, center=True, min_periods=12).mean().isna().sum()) == 0, \\
    "center=True + min_periods=12 时不应有 NaN"
assert int(load.rolling(24, center=True, min_periods=24).mean().isna().sum()) == 319, \\
    "center=True + min_periods=24 的 NaN 数与普通 rolling(24) 相同"
print("TODO(12) 通过 -> 全局判", int(flag_glob.sum()), "条 | 滑动判", int(flag_dyn.sum()), "条",
      f"| 全局漏 {int((flag_dyn & ~flag_glob).sum())} 条局部尖峰")"""
    ),
    md(
        """
---

## 综合题

### 综合 ① 24h 时序特征表（含泄漏自检）

造一张 6 特征的表，要求**全部特征列都由 `shift` 打底**，最后核对样本量。
"""
    ),
    code(
        """# @@todo 综合①：造 feat 表，含 负荷值 / 滑动均值24 / 滑动标准差24 / 滞后1h / 滞后24h / 日环比 / 累计均值
# @@hint feat = pd.DataFrame(index=load.index)；滑动统计一律先 load.shift(1) 再 rolling(24)
#        日环比用 load.pct_change(24, fill_method=None)；累计均值 load.expanding().mean()
feat = pd.DataFrame(index=load.index)
feat["负荷值"] = load
feat["滑动均值24"] = load.shift(1).rolling(24).mean()
feat["滑动标准差24"] = load.shift(1).rolling(24).std()
feat["滞后1h"] = load.shift(1)
feat["滞后24h"] = load.shift(24)
feat["日环比"] = load.pct_change(24, fill_method=None)
feat["累计均值"] = load.expanding().mean()
# @@end
# ---- 验收 ----
assert feat.shape == (2160, 7), f"形状应为 (2160, 7)，实际 {feat.shape}"
assert list(feat.columns) == [
    "负荷值", "滑动均值24", "滑动标准差24", "滞后1h", "滞后24h", "日环比", "累计均值"
], f"列名不对：{list(feat.columns)}"
# 自检 1：滑动列比「直接用 rolling」多 1 个 NaN（安全版代价）
direct = load.rolling(24).mean()
assert int(feat["滑动均值24"].isna().sum()) == int(direct.isna().sum()) + 1, "shift 打底应恰好多 1 个 NaN"
# 自检 2：泄漏检查 —— 滑动均值必须等于「前 24 小时」，而不是「含当前行的 24 行」
i = len(feat) - 1
window_safe = load.iloc[i - 24:i].mean()     # 过去 24 行（不含第 i 行）
window_leak = load.iloc[i - 23:i + 1].mean()  # 含第 i 行
assert abs(feat["滑动均值24"].iloc[i] - window_safe) < 1e-9, "特征列应等于不含当前行的窗口均值"
assert abs(feat["滑动均值24"].iloc[i] - window_leak) > 1e-6, "特征列不得等于含当前行的窗口均值"
# 自检 3：滞后 24h 的自相关应显著高于滞后 1h（日周期）
corr24 = load.corr(feat["滞后24h"])
corr1 = load.corr(feat["滞后1h"])
assert corr24 > corr1, f"日周期应使 lag24 相关性更高：{corr24:.4f} vs {corr1:.4f}"
# 自检 4：样本量守恒
n_complete = len(feat.dropna())
assert n_complete == 1829, f"dropna 后应剩 1829 行，实际 {n_complete}"
assert n_complete / len(feat) > 0.8, "完整样本占比应 > 80%"
print("综合① 通过 ->", feat.shape, "| dropna 后", n_complete, "行")
print(f"   日周期自相关 {corr24:.4f} > 小时自相关 {corr1:.4f}")
print(feat.iloc[-1].round(4).to_string())"""
    ),
    md(
        """
### 综合 ② 分组滑动特征回贴原表

把三件事一次做完，并证明"回贴对得上"：

1. 组内 24h 滑动均值、组内 24h 滞后值
2. 回贴到 `curve` 原表，**不能出现整列 NaN**
3. 对账：回贴后的值必须与手工按组计算的窗口值一致
"""
    ),
    code(
        """# @@todo 综合②：造 curve2（在 curve 上加 组内滑动均值 / 组内滞后24 / 组内日环比 三列），
#         并把「整表行数 / 三列各自的 NaN 数」存入 report 字典
# @@hint 滑动要用 reset_index(drop=True).to_numpy() 回贴；滞后与环比可直接回贴
gb_roll24 = gb[VALUE_COL].transform(lambda s: s.rolling(24).mean())
curve2 = curve.copy()
curve2["组内滑动均值"] = gb_roll24
curve2["组内滞后24"] = gb[VALUE_COL].shift(24)
curve2["组内日环比"] = gb[VALUE_COL].transform(lambda s: s.pct_change(24, fill_method=None))
report = {
    "行数": len(curve2),
    "滑动均值NaN": int(curve2["组内滑动均值"].isna().sum()),
    "滞后24NaN": int(curve2["组内滞后24"].isna().sum()),
    "日环比NaN": int(curve2["组内日环比"].isna().sum()),
}
# @@end
# ---- 验收 ----
assert set(report) == {"行数", "滑动均值NaN", "滞后24NaN", "日环比NaN"}
assert report["行数"] == 6480, f"行数应为 6480，实际 {report['行数']}"
assert report["滑动均值NaN"] == 829, f"组内滑动均值应有 829 个 NaN，实际 {report['滑动均值NaN']}"
assert report["滞后24NaN"] == 120, \\
    f"组内滞后24 应有 120 个 NaN（每组前 24 行 72 个 + 原缺失 48 个），实际 {report['滞后24NaN']}"
assert report["日环比NaN"] == 168, f"组内日环比应有 168 个 NaN，实际 {report['日环比NaN']}"
# 关键自检：没有任何一列是「整列 NaN」（索引没对齐的信号）
assert all(curve2[c].notna().any() for c in ["组内滑动均值", "组内滞后24", "组内日环比"]), \\
    "出现整列 NaN 说明索引没对齐"
# 对账：抽一个台区的第 30 行，手算组内前 24 小时均值
probe_rows = curve2[curve2[KEY_COL] == "STATION_C_01"].reset_index(drop=True)
manual = probe_rows[VALUE_COL].iloc[6:30].mean()
assert abs(probe_rows["组内滑动均值"].iloc[29] - manual) < 1e-9, \\
    f"STATION_C_01 第 29 行窗口值应与手算一致：{probe_rows['组内滑动均值'].iloc[29]:.4f} vs {manual:.4f}"
# 组边界自检：每组首行必为 NaN
for name, sub in curve2.groupby(KEY_COL):
    assert pd.isna(sub["组内滑动均值"].iloc[0]), f"{name} 组首行应为 NaN"
print("综合② 通过 -> 回贴报告：")
for key, value in report.items():
    print(f"   {key:12s} = {value}")
print(f"   对账：STATION_C_01 第 29 行 {probe_rows['组内滑动均值'].iloc[29]:.4f} == 手算 {manual:.4f}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（六题都能答上来才算过）：

1. `rolling(24).mean()` 在这份数据上为什么会有 319 个 `NaN`？缺失值是怎么"放大"的？
2. 造特征时为什么必须 `shift(1).rolling(24)`？代价是什么？
3. `gb.rolling()` 和 `gb.shift()` 返回的索引差在哪？回贴原表各自要怎么做？
4. `ewm(span=24)`、`ewm(halflife=24)`、`ewm(alpha=0.08)` 是一回事吗？
5. 时间窗口 `rolling("24h")` 的 `min_periods` 默认值是多少？和固定窗口差在哪？
6. 全局阈值和滑动阈值分别回答什么问题？什么业务场景该用哪个？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
