"""ch05 —— 重复值与异常值：统计法会骗你，业务规则不会

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch05_duplicate_outlier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch05_duplicate_outlier"

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

BIZ_KEY = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]
SENTINEL = [9999.0, -1.0, 5000.0]

print("pandas", pd.__version__)
print("读入 device_defects.csv ->", dev.shape)"""


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch05 重复值与异常值：统计法会骗你，业务规则不会

> 方向：数据预处理 ｜ 竞赛对应：**数据清洗第 2、3 项（各 2 分）**
> —— 这是评分表里坑最多的两项：重复去的口径选错会删掉真数据，
> 异常值判据选错会把整列"正常的高值"当成异常全部砍掉。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据清洗 · 重复值 | 2 分 | `duplicated` 两种口径 → 选择保留策略 → `drop_duplicates` → 复查 |
| 数据清洗 · 异常值 | 2 分 | **业务规则法** + 3σ + IQR → 判定 → `clip` / 置 NaN / 删除 → 复查 |
| 数据探索 · 异常信号 | 2 分 | mean vs median、`quantile`、箱线图 |
| 数据可视化 · 箱线图 | 2 分 | 按组画箱线图，一眼看出多峰与离群 |

**本章的核心冲突**：`duplicated` 和 `quantile` 都是**"看起来很简单"**的 API，
但**口径选错的代价极大**——删错了真数据、砍错了有效高值，后面模型全废，
而且**不会报错**。所以本章反复强调一件事：**先算，再判，最后才动手。**
"""
    ),
    md(
        """
## 二、API 速查表

| API | 返回 | 一句话说明 |
|---|---|---|
| `df.duplicated(subset=, keep=)` | `Series[bool]` | 重复行掩码 |
| `df.duplicated().sum()` | `int` | **完全重复**行数（所有列都相同） |
| `df.duplicated(subset=[业务键]).sum()` | `int` | **业务键重复**行数 |
| `duplicated(keep="first")` | 默认 | 保留首次出现 → `sum()` = **应删的行数** |
| `duplicated(keep="last")` | — | 保留最后一次出现 |
| `duplicated(keep=False)` | — | **重复组全部成员**都标 True（≈ 2 倍） |
| `df.drop_duplicates(subset=, keep=)` | `DataFrame` | 去重（默认 `keep="first"`） |
| `df.drop_duplicates(ignore_index=True)` | `DataFrame` | 去重后重置索引 |
| `s.quantile(q)` | `float` / `Series` | 分位数（`q` 可传列表） |
| `q1, q3 = s.quantile([0.25, 0.75])` | 两个 `float` | IQR 的两端 |
| `iqr = q3 - q1` | `float` | 四分位距 |
| `q1 - 1.5*iqr` / `q3 + 1.5*iqr` | `float` | **IQR 法**的上下界（Tukey 围栏） |
| `s.mean()` / `s.std()` | `float` | 3σ 法要的两个量（⚠ 会被极端值污染） |
| `s.clip(lower=, upper=)` | `Series` | **截断**到边界（winsorize） |
| `s.where(cond, other)` | `Series` | `cond` 为 False 处替换 |
| `s.mask(cond, other)` | `Series` | `cond` 为 True 处替换 |
| `s.replace({旧: 新})` | `Series` | 值替换（把哨兵码换成 `NaN`） |
| `s.isin([值])` | `Series[bool]` | 值是否在集合里 |
| `s.between(a, b)` | `Series[bool]` | 是否落在 `[a, b]` 闭区间 |
| `s.abs()` | `Series` | 绝对值（配合 `> 3*std` 用） |
| `plt.boxplot(...)` / `s.plot.box()` | — | 箱线图 |
| `df.plot.box(by=列)` | — | **按组**画箱线图（多峰数据的标配） |
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
### 3.1 重复值的两种口径：**完全重复 vs 业务键重复**

这是本章第一个"选错就出事"的地方。

| 口径 | 判据 | 适用 |
|---|---|---|
| **完全重复** | **所有列**（含 `记录ID`、`备注`）逐值相同 | 数据库导出重复、ETL 重跑导致的物理重复 |
| **业务键重复** | 只有**业务标识列**相同（`记录ID`/`备注` 不同） | 同一条缺陷被两个渠道各录了一次 |

> **为什么不能用完全重复去判业务重复**：
> 由于 `记录ID` 是主键、`备注` 常常不同，"同一条缺陷录了两遍"往往**不是**完全重复。
> 只看 `duplicated()` 会**漏掉**这部分。
>
> **为什么不能用业务键去判物理重复**：
> 反过来，业务键去重会**多删**——两条不同渠道的正常记录，如果业务键碰巧相同，会被误删。

**正确做法：两个口径都算，都报，再决定。**
"""
    ),
    code(
        """print("原始形状:", dev.shape)

# ① 完全重复
dup_full_mask = dev.duplicated()
n_dup_full = int(dup_full_mask.sum())
print("\\n① 完全重复（所有列逐值相同）:", n_dup_full)

# ② 业务键重复（去掉 记录ID / 备注 这些"会不同"的列）
dup_biz_mask = dev.duplicated(subset=BIZ_KEY)
n_dup_biz = int(dup_biz_mask.sum())
print("② 业务键重复:", n_dup_biz, "  <- 比完全重复多", n_dup_biz - n_dup_full, "条")

# 差集长什么样：只在业务键口径下被标为重复的行
only_biz = dev.loc[dup_biz_mask & ~dup_full_mask]
print("\\n只在业务键口径下判为重复的行（%d 条）：" % len(only_biz))
print(only_biz[["记录ID", "台区编号", "设备编号", "缺陷类型", "备注"]].to_string(index=False))
print("  -> 备注列有值（「已复核」「业务重复样本」），说明它们是「同一缺陷被重复录入」")"""
    ),
    md(
        """
### 3.2 `keep` 参数：为什么 `sum()` 会翻倍

`keep` 控制"谁被标成重复"，直接决定 `sum()` 的含义：

| `keep` | 语义 | `sum()` 代表 |
|---|---|---|
| `"first"`（默认） | 保留第一次出现，之后的标 True | ✅ **应删除的行数** |
| `"last"` | 保留最后一次出现 | 应删除的行数（另一侧，数量通常相同） |
| `False` | 重复组**所有成员**都标 True | ❌ 涉及重复的行总数（≈ 2 倍） |

> **竞赛陷阱**：题目问"有多少重复行"，你写 `duplicated(keep=False).sum()` 得到 12，
> 实际应删的行数是 6 —— **答案翻倍，直接判错**。
>
> 判据：**"要删几行" 用默认 `keep="first"`；"有多少行卷进了重复" 才用 `keep=False`。**
"""
    ),
    code(
        """print("完全重复口径下：")
for keep in ("first", "last", False):
    n = int(dev.duplicated(keep=keep).sum())
    print(f"  keep={str(keep):6s} -> duplicated().sum() = {n:3d}   "
          f"drop_duplicates 后 shape = {dev.drop_duplicates(keep=keep).shape}")

print("\\n业务键口径下：")
for keep in ("first", "last", False):
    n = int(dev.duplicated(subset=BIZ_KEY, keep=keep).sum())
    print(f"  keep={str(keep):6s} -> duplicated().sum() = {n:3d}   "
          f"drop_duplicates 后 shape = {dev.drop_duplicates(subset=BIZ_KEY, keep=keep).shape}")

print("\\n注意 keep=False 的数正好是 keep='first' 的 2 倍：")
print("  完全重复  : keep=False = %d, keep='first' = %d"
      % (int(dev.duplicated(keep=False).sum()), int(dev.duplicated().sum())))
print("  业务键重复: keep=False = %d, keep='first' = %d"
      % (int(dev.duplicated(subset=BIZ_KEY, keep=False).sum()),
         int(dev.duplicated(subset=BIZ_KEY).sum())))

# 看看重复组的组长什么样（每条重复出现 2 次）
print("\\n重复组的规模分布（完全重复口径）：")
grp = dev.groupby(list(dev.columns), dropna=False).size()
print(grp[grp > 1].value_counts().sort_index().to_string(), " <- key: 每组出现几次")"""
    ),
    md(
        """
### 3.3 异常值：先分清「哨兵码」和「真极端值」

**这是本章最重要的观念区分**：

| 类型 | 例子 | 本质 | 正确处理 |
|---|---|---|---|
| **哨兵码 / 非法值** | `9999`、`-1`、`999`、`"N/A"` | 采集失败时**人为填的占位**，机制上不可能出现 | **置为 `NaN`**（不是截断！） |
| **真极端值** | 那 3 个 `5000` | 可能是真实的大负荷，也可能是量纲错误 | 先查、再决定置 NaN 或截断 |
| **物理越界值** | 湿度 `150%`、温度 `-300℃` | 违反物理常识，一定错 | 按业务边界修正或置 NaN |

> **致命错误**：把 `9999` 用 `clip` 截断到 408 —— 你只是把"假值"剪成了一个
> 看起来合理的数，**它仍然是个假值**，而且现在**再也认不出它是假的**了。
>
> **正确顺序**：先按业务规则/哨兵码 → 置 `NaN`，再对剩下的真值做统计法检测。
"""
    ),
    code(
        """load = dev["负荷值"]
print("负荷值统计（含哨兵码）：")
print(load.describe().round(4).to_string())

print("\\n哨兵码与物理越界值盘点：")
for v in SENTINEL:
    print(f"  == {v:8.1f} 的个数 : {int((load == v).sum())}")
print(f"  >= 1000 的个数      : {int((load >= 1000).sum())}  (哨兵 9999 x4 + 尖峰 5000 x3)")
print(f"  <  0   的个数       : {int((load < 0).sum())}  (哨兵 -1 x4)")

print("\\n其他列的物理边界检查：")
h = dev["湿度"]
print(f"  湿度 > 100 的个数 : {int((h > 100).sum())}  值 = {sorted(h[h > 100].tolist())}")
print(f"  湿度 < 0   的个数 : {int((h < 0).sum())}  值 = {sorted(h[h < 0].tolist())}")
print(f"  温度越界（<-40 或 >80）: {int(((dev['温度'] < -40) | (dev['温度'] > 80)).sum())}")
print("\\n-> 湿度的越界是「物理不可能」，用业务规则就能 100%% 命中，不需要统计法。")"""
    ),
    md(
        """
### 3.4 三种异常检测口径，以及各自的**失效场景**

| 口径 | 公式 | 前提 | 失效场景 |
|---|---|---|---|
| **业务规则** | 领域边界（湿度 ∈ [0,100]） | 有明确物理边界 | 没有边界概念时不可用 |
| **3σ** | `mean ± 3*std` | 数据**近似正态**且**无极端值污染** | ✅ 本数据直接被哨兵码毁掉（见下） |
| **IQR（Tukey）** | `[q1-1.5*IQR, q3+1.5*IQR]` | 单峰、量纲统一 | ✅ 本数据是**多峰**（5 类设备量级不同）→ 误杀 |

**下面逐一实证这两个"失效"。**
"""
    ),
    code(
        """# ---------- 口径 A：3σ（会被极端值污染）----------
mu, sd = load.mean(), load.std()
sig_lo, sig_hi = mu - 3 * sd, mu + 3 * sd
n_sig = int(((load < sig_lo) | (load > sig_hi)).sum())

print("3σ 法：")
print(f"  mean = {mu:.4f}   std = {sd:.4f}   <- std 被 9999 拉爆了")
print(f"  区间 = [{sig_lo:.2f}, {sig_hi:.2f}]   宽度 = {sig_hi - sig_lo:.1f}")
print(f"  判为异常 : {n_sig} 个  <- 只抓到 7 个 >=1000 的，-1 一个没抓到")
print("  失效原因：均值 379.68（被哨兵拉起）→ 下界低到 -3675，负值根本判不出来")
print("           而正常负荷只有 5 ~ 590，这个区间宽到毫无筛选力")

# ---------- 口径 B：IQR（会被多峰分布毁掉）----------
q1, q3 = load.quantile([0.25, 0.75])
iqr = q3 - q1
iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_iqr = int(((load < iqr_lo) | (load > iqr_hi)).sum())

print("\\nIQR 法（全局）：")
print(f"  q1 = {q1:.4f}   q3 = {q3:.4f}   IQR = {iqr:.4f}")
print(f"  区间 = [{iqr_lo:.4f}, {iqr_hi:.4f}]")
print(f"  判为异常 : {n_iqr} 个  <- 太多了！真哨兵码只有 11 个")

# 为什么会多出这么多？因为负荷值本身是多峰分布
print("\\n  失效原因 —— 按设备类型看负荷值，是 5 个分离的峰：")
print(dev.groupby("设备类型")["负荷值"].agg(["count", "min", "median", "max"]).round(2).to_string())
print("  -> 变压器的中位数 521、绝缘子只有 55，全局分位数把「高压设备集群」整体当成了离群值")"""
    ),
    md(
        """
> **这两段输出的教学价值**：
>
> - 3σ 判出 **7** 个 → **漏掉** 4 个 `-1`
> - IQR 判出 **52** 个 → **多杀** 41 个（真哨兵码只有 11 个）
>
> **所以：在真实数据上，单一统计口径几乎一定出错。**
> 竞赛里的标准打法是 **"业务规则优先 + 统计法补充"**：
>
> ```
> 1. 业务规则能覆盖的（物理边界、哨兵码）-> 直接判，100% 准确
> 2. 剩下的用统计法，但必须【分组】做
> 3. 判出来的"异常"先看样本，确认是假值再动手
> ```
"""
    ),
    md(
        """
### 3.5 分组异常检测：多峰分布的解法

量纲不同的子群体，**必须分组各自定界**。这在电力数据里是常态：

- 按**设备类型**（变压器 / 断路器 / 绝缘子，量级差 10 倍）
- 按**台区**（城区 / 农村，负荷水平不同）
- 按**时段**（峰 / 谷）

```python
def iqr_mask(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k*iqr) | (s > q3 + k*iqr)

dev.groupby("设备类型")["负荷值"].transform(iqr_mask)
```

> `transform` 是这里的关键：它把"每组算一个界"的结果**广播回每一行**，
> 索引天然对齐，可以直接当掩码用。
"""
    ),
    code(
        """def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    \"\"\"对一组做 IQR 围栏判定，返回布尔掩码。\"\"\"
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def sigma_mask(s: pd.Series, k: float = 3.0) -> pd.Series:
    \"\"\"对一组做 3σ 判定，返回布尔掩码。\"\"\"
    mu, sd = s.mean(), s.std()
    return (s < mu - k * sd) | (s > mu + k * sd)


grp_iqr_mask = dev.groupby("设备类型")["负荷值"].transform(iqr_mask).astype(bool)
grp_sig_mask = dev.groupby("设备类型")["负荷值"].transform(sigma_mask).astype(bool)

print("各组的 IQR 围栏与命中的异常数：")
for name, sub in dev.groupby("设备类型")["负荷值"]:
    q1g, q3g = sub.quantile([0.25, 0.75])
    iqrg = q3g - q1g
    lo, hi = q1g - 1.5 * iqrg, q3g + 1.5 * iqrg
    n_hit = int(((sub < lo) | (sub > hi)).sum())
    print(f"  {name:6s} n={len(sub):3d}  围栏=[{lo:8.2f}, {hi:8.2f}]  异常 {n_hit}")

print("\\n三种口径的异常数对比：")
print(f"  全局 IQR(1.5)   : {n_iqr:3d}  <- 误杀大量正常高值")
print(f"  分组 IQR(1.5)   : {int(grp_iqr_mask.sum()):3d}  <- 干净多了")
print(f"  全局 3σ         : {n_sig:3d}")
print(f"  分组 3σ         : {int(grp_sig_mask.sum()):3d}")
print(f"  业务规则（>=1000 或 <0）: {int(((load >= 1000) | (load < 0)).sum()):3d}  <- 真值：11 个哨兵码")

print("\\n分组 IQR 命中的行里，有多少是真的哨兵码：")
print("  命中 >=1000 :", int((grp_iqr_mask & (load >= 1000)).sum()), "/ 7")
print("  命中 <0     :", int((grp_iqr_mask & (load < 0)).sum()), "/ 4")
print("  -> 还剩 %d 个哨兵码没被统计法抓到（-1 落在组内下界之上）"
      % int(((load >= 1000) | (load < 0)).sum() - (grp_iqr_mask & ((load >= 1000) | (load < 0))).sum()))
print("  -> 再次说明：统计法必须配业务规则兜底")"""
    ),
    md(
        """
### 3.6 处理方式：置 NaN、截断、还是删除？

| 方式 | 代码 | 适用 | 副作用 |
|---|---|---|---|
| **置 `NaN`** | `s.replace({哨兵: np.nan})` 或 `s.mask(cond)` | 哨兵码、物理不可能值 | 会制造新的缺失（随后按 ch04 处理） |
| **截断（winsorize）** | `s.clip(lower=, upper=)` | **真极端值**，且你想保留"它很大"这个信息 | 会在边界堆积大量相同值 |
| **删除行** | `df[~mask]` | 异常行占比极低、且该行其他字段也不可信 | 丢数据 |
| **修正** | 按业务公式反算 | 量纲错误（如单位写错，乘以 1000） | 需要业务知识 |

**决策要点**：

- 哨兵码 → **必须置 NaN**（截断会把假值洗成"看起来真"的值）
- 真极端值 → 截断或保留，视建模目标而定
- 物理越界 → 按边界修正（如湿度 `150` 若判定为"传感器满量程"，可置 NaN）

> `clip` 的物理含义是 **"我不怀疑这些点存在，但我不想让它们过度影响统计量"**。
> 一旦你确认某个值是**假**的，它就不该参与任何统计——应该置 `NaN`。
"""
    ),
    code(
        """sentinel_series = dev["负荷值"]
print("=== 单阶段（错误示范）：直接 clip 到 IQR 界 ===")
wrong = sentinel_series.clip(iqr_lo, iqr_hi)
print(f"  clip 后 max = {wrong.max():.4f}  <- 9999 被剪成 408.385「看起来正常」")
print(f"  残余缺失    = {int(wrong.isna().sum())}")
print("  ❌ 问题：假值被「洗白」成了正常值，而且再也认不出它是假的")

print("\\n=== 两阶段（正确做法）===")
stage1 = sentinel_series.replace({v: np.nan for v in SENTINEL})
print(f"  ① 哨兵码置 NaN -> 缺失 {int(stage1.isna().sum())} 个（原缺失 21 + 哨兵 11）")

stage2 = stage1.fillna(stage1.median())
print(f"  ② 中位数填充    -> 缺失 {int(stage2.isna().sum())} 个，"
      f"median={stage2.median():.4f}, mean={stage2.mean():.4f}, max={stage2.max():.4f}")

q1b, q3b = stage2.quantile([0.25, 0.75])
iqrb = q3b - q1b
lo_b, hi_b = q1b - 1.5 * iqrb, q3b + 1.5 * iqrb
print(f"  ③ 重算 IQR 围栏 -> [{lo_b:.4f}, {hi_b:.4f}]"
      f"（比原来的 [{iqr_lo:.2f}, {iqr_hi:.2f}] 收窄了很多）")

n_b = int(((stage2 < lo_b) | (stage2 > hi_b)).sum())
stage3 = stage2.clip(lo_b, hi_b)
print(f"  ④ 判出 {n_b} 个仍偏高的点 -> clip")
print(f"     最终 mean={stage3.mean():.4f}, std={stage3.std():.4f}, "
      f"max={stage3.max():.4f}, min={stage3.min():.4f}")
print(f"     std 从 {wrong.std():.2f} 降到 {stage3.std():.2f}")

print("\\n=== 对比三者的分布特征 ===")
compare = pd.DataFrame({
    "原始(含哨兵)": sentinel_series.describe(),
    "单阶段 clip": wrong.describe(),
    "两阶段清洗": stage3.describe(),
}).round(3)
print(compare.to_string())"""
    ),
    md(
        """
### 3.7 物理边界法：比统计法更可靠的那一类

凡是**有明确物理边界**的字段，永远先用业务规则，不要用统计法：

| 字段 | 物理边界 | 一行代码 |
|---|---|---|
| 湿度 | `[0, 100]` | `h.mask((h < 0) \\| (h > 100))` |
| 温度（户外设备） | `[-40, 80]` | 同上 |
| 功率因数 | `[0, 1]` | 同上 |
| 负荷值 | `[0, +∞)`，且 ≥ 1000 基本是哨兵 | 按台区容量上限定 |
| 有效率 / 占比类 | `[0, 100]` | 同上 |

> 注意 `mask(cond)` 与 `where(cond)` 的**相反语义**：
> - `s.mask(cond, other)`：`cond` 为 **True** 的位置替换 → **"命中条件就换掉"**
> - `s.where(cond, other)`：`cond` 为 **False** 的位置替换 → **"不满足条件就换掉"**
"""
    ),
    code(
        """h = dev["湿度"]
print("湿度原始：min=%.1f, max=%.1f" % (h.min(), h.max()))
bad_h = (h < 0) | (h > 100)
print("越界（物理不可能）个数:", int(bad_h.sum()))

# 两种等价写法：mask / where
m1 = h.mask(bad_h)                      # 命中 bad_h -> NaN
m2 = h.where(~bad_h)                    # 不满足 ~bad_h 即命中 -> NaN
print("mask 与 where 结果是否一致:", bool(m1.isna().equals(m2.isna())))
print("置 NaN 后缺失:", int(m1.isna().sum()))

# 也可以直接按物理边界截断（若判定为量程问题）
c = h.clip(0, 100)
print("\\nclip(0,100) 后：越界数 =", int(((c > 100) | (c < 0)).sum()),
      "| mean = %.4f （原 %.4f）" % (c.mean(), h.mean()))

# 对比：用 IQR 判湿度会怎样
hq1, hq3 = h.quantile([0.25, 0.75])
hiqr = hq3 - hq1
print("\\n用 IQR 判湿度：围栏 = [%.4f, %.4f]" % (hq1 - 1.5 * hiqr, hq3 + 1.5 * hiqr))
print("  IQR 只抓到 %d 个（其中 150 命中，105/120/-5/-3 全部漏网）"
      % int(((h < hq1 - 1.5 * hiqr) | (h > hq3 + 1.5 * hiqr)).sum()))
print("  ✅ 业务规则抓到 5 个 —— 有物理边界时，业务规则完胜统计法")"""
    ),
    md(
        """
### 3.8 箱线图：多峰分布最直观的证据

箱线图是「异常值」这一项的**可视化计分点**，也是说服阅卷人"为什么我要分组处理"的最佳论据。

- **看单列**：`s.plot.box()` —— 一眼看到 `9999` 把整张图压扁
- **按组看**：`df.plot.box(column=值列, by=分组列)` —— 一眼看到 5 个分离的峰

> **箱线图的"须"在哪**：上下须的端点就是 `q1-1.5*IQR` 和 `q3+1.5*IQR`，
> 超出须的点就是 `showfliers` 画出来的离群点。**箱线图 = IQR 法的图形化。**
"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

# ---- 图 1：全局箱线图（被哨兵码压扁）----
ax = axes[0]
ax.boxplot(load.dropna().to_numpy(), showfliers=True, vert=True)
ax.set_title("全局箱线图：哨兵码 9999 把整张图压成一条线")
ax.set_ylabel("负荷值")

# ---- 图 2：分组箱线图（5 个分离的峰）----
ax = axes[1]
groups = [g.dropna().to_numpy() for _, g in dev.groupby("设备类型")["负荷值"]]
labels = [name for name, _ in dev.groupby("设备类型")["负荷值"]]
ax.boxplot(groups, tick_labels=labels, showfliers=True)
ax.set_title("按设备类型分组：5 个量级不同的峰 -> 必须分组定界")
ax.set_ylabel("负荷值")

plt.tight_layout()
plt.show()

print("读图结论（写进报告）：")
print("  1) 左图：q3 只有 192，max 却有 9999 —— 说明存在极端离群点，且量级完全不匹配")
print("  2) 右图：变压器的中位数约 521，绝缘子只有约 55 —— 分布是多峰的")
print("     -> 任何全局统计口径（全局 IQR/3σ）都会把高量级组整体误判为异常")"""
    ),
    md(
        """
### 3.9 完整处理链（顺序很重要）

```
重复值处理链：

  ① duplicated() 完全重复口径        -> 报数
  ② duplicated(subset=业务键)        -> 报数（两个口径都要报）
  ③ 看差集样本，确认哪一类是"真重复"   -> 判据留痕
  ④ drop_duplicates(subset=, keep=)  -> 执行
  ⑤ 复查：duplicated().sum() == 0    -> 归零才算完成


异常值处理链：

  ① 业务规则法（物理边界 / 哨兵码）  -> 先做，100% 准确
  ② 哨兵码 replace 成 NaN            -> 不要 clip 假值！
  ③ 缺失值按 ch04 策略填充            -> 补齐
  ④ 分组统计法（分组 IQR / 3σ）      -> 补充业务规则漏掉的
  ⑤ clip 截断真极端值                -> 保留"它很大"的信息
  ⑥ 复查：max/min 回到合理区间 + 画箱线图
```

> **顺序为什么不能反**：先做统计法会把哨兵码当成"需要截断的真极端值"，
> 于是假值被剪成一个"看起来正常"的数，**污染被永久隐藏**。
"""
    ),
    md(
        """
---

## 四、易错点清单

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | `duplicated(keep=False).sum()` 当"要删的行数" | 结果是 2 倍。要删几行用默认 `keep="first"` |
| 2 | 只用完全重复口径 | 业务重复（`记录ID` 不同）会漏掉。两个口径都算 |
| 3 | 只用业务键口径 | 会误删"业务键碰巧相同"的正常不同记录 |
| 4 | `drop_duplicates()` 不给 `ignore_index=True` | 索引会留窟窿，后续按位置取值会错 |
| 5 | **用 `clip` 处理哨兵码** | 假值会被洗成"看起来真"的值。哨兵码必须 `replace` 成 `NaN` |
| 6 | 用 `mean ± 3σ` 却不清洗极端值 | `std` 会被哨兵码拉爆，区间宽到没有筛选力 |
| 7 | 全局 IQR 处理多峰数据 | 会误杀整个高量级子群。必须 `groupby().transform()` 分组定界 |
| 8 | 有物理边界的字段用统计法 | 湿度 `[0,100]` 这类直接业务规则判，统计法会漏 |
| 9 | `mask` / `where` 语义混用 | `mask`：条件为 True 处替换；`where`：条件为 False 处替换 |
| 10 | `quantile` 传错参数 | `s.quantile(0.25)` 与 `s.quantile([0.25, 0.75])` 返回类型不同 |
| 11 | 删除异常行不看占比 | 占比高时删除会丢太多数据，改截断 |
| 12 | 处理完不复查、不画图 | 必须复查 `duplicated().sum()==0`、`max/min` 合理，并出箱线图 |
| 13 | 忽略了"异常值本身可能是标签信息" | 电力场景里"异常读数"有时正对应"缺陷"，别无脑删 |

---

## 五、本章小结

```
重复值（两个口径都要报）：

  完全重复  : df.duplicated().sum()
  业务键重复: df.duplicated(subset=业务键).sum()
  应删行数  : 用默认 keep="first" 的 sum()
  执行      : df.drop_duplicates(subset=..., ignore_index=True)

异常值（顺序不能反）：

  ① 业务规则 / 哨兵码  ->  100% 准确，优先做
  ② 哨兵码 replace 成 NaN          <- 不要 clip 假值
  ③ 按 ch04 处理缺失
  ④ 分组统计法（groupby().transform(iqr_mask)）补漏
  ⑤ clip 截断真极端值
  ⑥ 复查 max/min + 箱线图

三种口径的失效场景（必须记住）：
  · 3σ      被极端值污染 -> std 爆炸，区间失去筛选力
  · 全局 IQR 多峰分布     -> 误杀整个高量级子群
  · 业务规则 无边界概念   -> 不可用（但有边界时永远优先）
```

**下一步**：做 `ch05_duplicate_outlier_practice.ipynb`。
"""
    ),
]


# =========================================================================== #
# 练习 / 答案 notebook（答案版内容；练习版自动派生）
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch05 练习：重复值与异常值

> 共 8 道基础题 + 1 道综合题（拆 2 步），合计 **10 个空**
> 建议限时：**30 分钟**

数据：`data/device_defects.csv`（270 行 x 13 列）

**这章练的是「口径」**。同一个问题换一个口径，答案可能差一倍甚至十倍。
每题都给了验收断言，跑通即正确。
"""
    ),
    code(
        """import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

dev = pd.read_csv("data/device_defects.csv")
BIZ_KEY = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]
SENTINEL = [9999.0, -1.0, 5000.0]
load = dev["负荷值"]

print("读入 device_defects.csv ->", dev.shape)"""
    ),
    md(
        """
---

### TODO(1) 完全重复：报数 + 去重

1. 完全重复行数存入 `n_dup_full`
2. 去重后的 DataFrame 存入 `d_dedup`（并重置索引）

**考点**：默认 `keep="first"`，此时 `sum()` 就是**应删的行数**。
"""
    ),
    code(
        """# @@todo(1) 完全重复行数 -> n_dup_full；drop_duplicates 并重置索引 -> d_dedup
# @@hint DataFrame.duplicated().sum() ；DataFrame.drop_duplicates(ignore_index=True)
n_dup_full = int(dev.duplicated().sum())
d_dedup = dev.drop_duplicates(ignore_index=True)
# @@end

# ---- 验收 ----
assert n_dup_full == 6, f"完全重复行应为 6，实际 {n_dup_full}"
assert d_dedup.shape == (264, 13), f"去重后应为 (264, 13)，实际 {d_dedup.shape}"
assert d_dedup.duplicated().sum() == 0, "去重后不应还有完全重复行"
assert list(d_dedup.index) == list(range(len(d_dedup))), "别忘了 ignore_index=True 重置索引"
print(f"TODO(1) 通过 -> 完全重复 {n_dup_full} 行，去重后 {d_dedup.shape}")"""
    ),
    md(
        """
### TODO(2) 业务键重复：报数 + 去重

1. 按 `BIZ_KEY`（业务键，已在上方定义）判重复，行数存入 `n_dup_biz`
2. 按业务键去重后的 DataFrame 存入 `d_dedup_biz`（重置索引）

**注意**：这个口径比完全重复**多**，因为 `记录ID` / `备注` 不同但业务内容相同的行也被算进来。
"""
    ),
    code(
        """# @@todo(2) 业务键重复行数 -> n_dup_biz；按业务键去重 -> d_dedup_biz
# @@hint DataFrame.duplicated(subset=列名列表).sum() ；DataFrame.drop_duplicates(subset=..., ignore_index=True)
n_dup_biz = int(dev.duplicated(subset=BIZ_KEY).sum())
d_dedup_biz = dev.drop_duplicates(subset=BIZ_KEY, ignore_index=True)
# @@end

# ---- 验收 ----
assert n_dup_biz == 10, f"业务键重复行应为 10，实际 {n_dup_biz}"
assert d_dedup_biz.shape == (260, 13), f"业务键去重后应为 (260, 13)，实际 {d_dedup_biz.shape}"
assert n_dup_biz > n_dup_full, "业务键口径应该比完全重复口径多"
assert d_dedup_biz.duplicated(subset=BIZ_KEY).sum() == 0, "业务键去重后不应还有业务重复"
print(f"TODO(2) 通过 -> 业务键重复 {n_dup_biz} 行（比完全重复多 {n_dup_biz - n_dup_full} 行）")"""
    ),
    md(
        """
### TODO(3) `keep` 语义：为什么数字会翻倍

1. `keep=False` 口径下完全重复的掩码检出数存入 `n_keep_false`
2. `drop_duplicates(keep=False)` 后的形状行数存入 `n_rows_keep_false`

**考点**：`keep=False` 会把重复组**所有成员**都标 True，
所以 `n_keep_false` 是 `n_dup_full` 的 2 倍，**不能当"要删的行数"**。
"""
    ),
    code(
        """# @@todo(3) keep=False 的检出数 -> n_keep_false；keep=False 去重后的行数 -> n_rows_keep_false
# @@hint DataFrame.duplicated(keep=False).sum() ；DataFrame.drop_duplicates(keep=False)
n_keep_false = int(dev.duplicated(keep=False).sum())
n_rows_keep_false = len(dev.drop_duplicates(keep=False))
# @@end

# ---- 验收 ----
assert n_keep_false == 12, f"keep=False 应检出 12 行，实际 {n_keep_false}"
assert n_keep_false == n_dup_full * 2, "keep=False 的结果正好是 keep='first' 的 2 倍"
assert n_rows_keep_false == 258, f"keep=False 去重后应为 258 行，实际 {n_rows_keep_false}"
assert n_rows_keep_false < 264, "keep=False 会把重复组两端都删掉，比 keep='first' 删得更多"
print(f"TODO(3) 通过 -> keep=False 检出 {n_keep_false} 行（= 2 x {n_dup_full}），去重后剩 {n_rows_keep_false} 行")"""
    ),
    md(
        """
### TODO(4) 物理边界法：湿度的越界值

1. 湿度列中 `> 100` 或 `< 0` 的个数存入 `n_humid_bad`
2. 按物理边界 `[0, 100]` 截断后的湿度存入 `humid_clipped`
3. 截断后仍越界的个数存入 `n_humid_bad_after`

**考点**：有**明确物理边界**的字段永远优先用业务规则，不要用统计法。
"""
    ),
    code(
        """humid = dev["湿度"]

# @@todo(4) 越界个数 -> n_humid_bad；clip(0,100) -> humid_clipped；截断后越界数 -> n_humid_bad_after
# @@hint 布尔条件用 | 连接并各自加括号；Series.clip(lower, upper)
n_humid_bad = int(((humid < 0) | (humid > 100)).sum())
humid_clipped = humid.clip(0, 100)
n_humid_bad_after = int(((humid_clipped < 0) | (humid_clipped > 100)).sum())
# @@end

# ---- 验收 ----
assert n_humid_bad == 5, f"湿度越界应为 5 个，实际 {n_humid_bad}"
assert n_humid_bad_after == 0, f"截断后不应还有越界，实际 {n_humid_bad_after}"
assert humid_clipped.max() == 100 and humid_clipped.min() >= 0, "截断后的范围应是 [0, 100]"
assert humid_clipped.isna().sum() == 0, "湿度列本来没有缺失，截断不应制造缺失"
print(f"TODO(4) 通过 -> 越界 {n_humid_bad} 个，截断后 {n_humid_bad_after} 个")"""
    ),
    md(
        """
### TODO(5) 全局 IQR：算出围栏并数异常

`负荷值` 里掺了 `9999` / `-1` / `5000` 三种哨兵码。先看全局 IQR 会判出多少个：

1. 下四分位存入 `q1`，上四分位存入 `q3`，四分位距存入 `iqr`
2. IQR 围栏下界存入 `iqr_lo`，上界存入 `iqr_hi`
3. 落在围栏外的个数存入 `n_out_iqr`

**提醒**：这个数会**远大于**真实哨兵码个数（11 个）——因为负荷值是多峰的。
"""
    ),
    code(
        """# @@todo(5) q1 / q3 / iqr / 围栏上下界 / 围栏外个数
# @@hint Series.quantile([0.25, 0.75]) 一次拿两个；围栏 = q1-1.5*iqr, q3+1.5*iqr；越界计数用 | 连接两个条件
q1, q3 = load.quantile([0.25, 0.75])
iqr = q3 - q1
iqr_lo = q1 - 1.5 * iqr
iqr_hi = q3 + 1.5 * iqr
n_out_iqr = int(((load < iqr_lo) | (load > iqr_hi)).sum())
# @@end

# ---- 验收 ----
assert abs(q1 - 48.31) < 1e-6, f"q1 应为 48.31，实际 {q1}"
assert abs(q3 - 192.34) < 1e-6, f"q3 应为 192.34，实际 {q3}"
assert abs(iqr - 144.03) < 1e-6, f"IQR 应为 144.03，实际 {iqr}"
assert abs(iqr_lo - -167.735) < 1e-6, f"下界应为 -167.735，实际 {iqr_lo}"
assert abs(iqr_hi - 408.385) < 1e-6, f"上界应为 408.385，实际 {iqr_hi}"
assert n_out_iqr == 52, f"全局 IQR 应判出 52 个，实际 {n_out_iqr}"
assert n_out_iqr > 11, "这个数远大于真哨兵码 11 个 -> 全局 IQR 在多峰数据上误杀严重"
print(f"TODO(5) 通过 -> 围栏 [{iqr_lo:.3f}, {iqr_hi:.3f}]，判出 {n_out_iqr} 个（真哨兵码只有 11 个）")"""
    ),
    md(
        """
### TODO(6) 全局 3σ：被哨兵码污染的典型

1. 用 `mean ± 3*std` 算出上下界，存入 `sig_lo`、`sig_hi`
2. 落在区间外的个数存入 `n_out_3sigma`

**看完数字请回答**：区间宽度是多少？正常负荷的取值范围是多少？
这个区间还有筛选力吗？为什么 `-1` 一个都没被抓到？
"""
    ),
    code(
        """# @@todo(6) 3σ 上下界 -> sig_lo, sig_hi；区间外个数 -> n_out_3sigma
# @@hint Series.mean() / Series.std() ；界 = mean - 3*std, mean + 3*std
mu = load.mean()
sd = load.std()
sig_lo = mu - 3 * sd
sig_hi = mu + 3 * sd
n_out_3sigma = int(((load < sig_lo) | (load > sig_hi)).sum())
# @@end

# ---- 验收 ----
assert abs(mu - 379.68457831325304) < 1e-6, f"均值应为 379.6846，实际 {mu}"
assert abs(sd - 1351.7630979420865) < 1e-6, f"标准差应为 1351.7631，实际 {sd}"
assert abs(sig_lo - -3675.6047) < 0.01, f"下界应约 -3675.60，实际 {sig_lo:.4f}"
assert abs(sig_hi - 4434.9739) < 0.01, f"上界应约 4434.97，实际 {sig_hi:.4f}"
assert n_out_3sigma == 7, f"3σ 应判出 7 个，实际 {n_out_3sigma}"
assert (sig_hi - sig_lo) > 8000, "区间宽度 8000+，而正常负荷只有 5~590 -> 完全没有筛选力"
assert int((load < 0).sum()) == 4, "数据里有 4 个 -1，但 3σ 下界是 -3675 -> 一个都没抓到"
print(f"TODO(6) 通过 -> 3σ 区间 [{sig_lo:.2f}, {sig_hi:.2f}]，判出 {n_out_3sigma} 个，漏掉全部 4 个 -1")"""
    ),
    md(
        """
### TODO(7) 分组 IQR：多峰分布的正解

按 `设备类型` 分组，对每组的 `负荷值` 各自做 IQR 判定：

1. 布尔掩码（与原始行一一对应）存入 `grp_out_mask`
2. 掩码为 True 的个数存入 `n_out_grp`

**考点**：`groupby(...).transform(函数)` 把"每组算一个界"的结果**广播回每一行**，
索引天然对齐，可直接当掩码用。**这是处理多峰数据的标准姿势。**
"""
    ),
    code(
        """def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    \"\"\"对一组做 IQR 围栏判定，返回布尔掩码。\"\"\"
    q1g, q3g = s.quantile([0.25, 0.75])
    iqrg = q3g - q1g
    return (s < q1g - k * iqrg) | (s > q3g + k * iqrg)


# @@todo(7) 按设备类型分组 IQR 掩码 -> grp_out_mask；异常个数 -> n_out_grp
# @@hint DataFrame.groupby(键)[列].transform(函数) ；结果再 .astype(bool)
grp_out_mask = dev.groupby("设备类型")["负荷值"].transform(iqr_mask).astype(bool)
n_out_grp = int(grp_out_mask.sum())
# @@end

# ---- 验收 ----
assert isinstance(grp_out_mask, pd.Series), "grp_out_mask 应该是 Series"
assert len(grp_out_mask) == len(dev), "掩码长度应等于行数（transform 会广播回每行）"
assert set(grp_out_mask.unique()) <= {True, False}, "掩码只能是布尔值"
assert n_out_grp == 10, f"分组 IQR 应判出 10 个，实际 {n_out_grp}"
assert n_out_grp < n_out_iqr, "分组 IQR 远比全局 IQR 精确"
assert int((grp_out_mask & (load >= 1000)).sum()) == 7, "7 个 >=1000 的哨兵码应全部命中"
assert int((grp_out_mask & (load < 0)).sum()) == 2, "只能命中 4 个 -1 中的 2 个"
print(f"TODO(7) 通过 -> 分组 IQR 判出 {n_out_grp} 个（全局 IQR 是 {n_out_iqr} 个，误杀 {n_out_iqr - n_out_grp} 个）")
print("  仍漏掉 2 个 -1 -> 说明统计法必须配业务规则兜底")"""
    ),
    md(
        """
### TODO(8) 两阶段清洗：哨兵 → NaN → 填充 → clip

这是本章的核心流程，**顺序不能反**：

1. `s1`：把 `SENTINEL`（`9999` / `-1` / `5000`）替换成 `np.nan`
2. `s2`：用 `s1` 的中位数填充（按 ch04 的策略）
3. 用 `s2` 重算 IQR 围栏（`lo2` / `hi2`），判出仍偏高的个数存入 `n_out_2`
4. `clean_load`：把 `s2` 截断到 `[lo2, hi2]`

**如果先 clip 再判哨兵，假值会被剪成"看起来正常"的数**——这是最常见的错误。
"""
    ),
    code(
        """# @@todo(8) 哨兵置 NaN -> s1；中位数填充 -> s2；重算围栏 -> lo2/hi2；越界数 -> n_out_2；截断 -> clean_load
# @@hint Series.replace({旧值: np.nan, ...}) ；Series.fillna(Series.median()) ；再用 quantile 重算围栏
s1 = load.replace({v: np.nan for v in SENTINEL})
s2 = s1.fillna(s1.median())
q1b, q3b = s2.quantile([0.25, 0.75])
iqrb = q3b - q1b
lo2 = q1b - 1.5 * iqrb
hi2 = q3b + 1.5 * iqrb
n_out_2 = int(((s2 < lo2) | (s2 > hi2)).sum())
clean_load = s2.clip(lo2, hi2)
# @@end

# ---- 验收 ----
assert int(s1.isna().sum()) == 32, f"哨兵置 NaN 后应有 32 个缺失（原 21 + 哨兵 11），实际 {int(s1.isna().sum())}"
assert int(s2.isna().sum()) == 0, "填充后不应还有缺失"
assert abs(s2.median() - 86.05) < 1e-6, f"填充后中位数应为 86.05，实际 {s2.median()}"
assert abs(s2.mean() - 156.678) < 1e-3, f"填充后均值应约 156.678，实际 {s2.mean():.4f}"
assert abs(lo2 - -119.9212) < 1e-3, f"新下界应约 -119.9212，实际 {lo2:.4f}"
assert abs(hi2 - 348.5687) < 1e-3, f"新上界应约 348.5687，实际 {hi2:.4f}"
assert n_out_2 == 45, f"新围栏下应判出 45 个，实际 {n_out_2}"
assert abs(clean_load.mean() - 128.246) < 1e-3, f"截断后均值应约 128.246，实际 {clean_load.mean():.4f}"
assert int(clean_load.isna().sum()) == 0, "清洗后不应有缺失"
assert clean_load.max() <= hi2 + 1e-9, "截断后最大值不应超过上界"
print(f"TODO(8) 通过 -> 哨兵 11 个置 NaN，填充后 mean {s2.mean():.3f} -> 截断后 {clean_load.mean():.3f}")
print(f"  std: {load.std():.2f}（原始，被污染）-> {clean_load.std():.2f}（清洗后）")"""
    ),
    md(
        """
---

## 综合题：从清洗到可视化

| 步骤 | 产物 | 对应评分点 |
|---|---|---|
| ① 分组箱线图 | `boxes` | 数据可视化 · 箱线图（2 分） |
| ② 组装报告 | `report` | 数据探索 · 结论输出 |
"""
    ),
    md(
        """
#### 综合 ① 按设备类型的箱线图

画出按 `设备类型` 分组的 `负荷值` 箱线图，共 **5 个箱体**。

1. `fig, ax` 用 `plt.subplots`
2. `boxes` 接住 `ax.boxplot(...)` 的返回值（后面要用它数箱体个数）
3. 给图加标题（中文别忘设置字体，前面已经设过）

**提示**：`boxplot` 需要"每组一个数组"，用
`[g.dropna().to_numpy() for _, g in dev.groupby("设备类型")["负荷值"]]`。
"""
    ),
    code(
        """# @@todo(9) 画按设备类型分组的负荷值箱线图：fig/ax 用 plt.subplots，返回值存入 boxes
# @@hint ax.boxplot(数组列表, tick_labels=标签列表) ；数组列表用列表推导式从 groupby 取每组数组
fig, ax = plt.subplots(figsize=(9, 4.6))
boxes = ax.boxplot(
    [g.dropna().to_numpy() for _, g in dev.groupby("设备类型")["负荷值"]],
    tick_labels=sorted(dev["设备类型"].unique()),
)
# @@end

ax.set_title("按设备类型的负荷值箱线图（多峰 + 哨兵码）")
ax.set_ylabel("负荷值")
plt.tight_layout()
plt.show()

# ---- 验收 ----
assert len(boxes["boxes"]) == 5, f"应有 5 个箱体，实际 {len(boxes['boxes'])}"
assert len(boxes["fliers"]) == 5, "每组都应画出离群点（个数可以为 0）"
flier_max = max(
    (float(f.get_ydata().max()) if len(f.get_ydata()) else 0.0)
    for f in boxes["fliers"]
)
assert flier_max > 1000, f"应能看到 9999 这个离群点，实际最大离群值 {flier_max}"
print("综合 ① 通过 -> 已画出 5 组箱线图，各组离群点数：",
      [len(f.get_ydata()) for f in boxes["fliers"]])
print("  读图结论：箱体高度差异极大（变压器 ~521 vs 绝缘子 ~55）-> 分布是多峰的")
print("            -> 全局统计口径必然误判，必须分组定界")"""
    ),
    md(
        """
#### 综合 ② 组装清洗报告

把这一章的结论组装成 `report` 字典，键必须完全一致：

```
n_rows / n_dup_full / n_dup_biz / n_dup_after_dedup
n_sentinel / n_out_iqr / n_out_grp / n_humid_bad
clean_mean / clean_max
```

其中：
- `n_dup_after_dedup`：按业务键去重后**仍有的完全重复行数**（应为 0）
- `n_sentinel`：哨兵码总数（`>= 1000` 的与 `< 0` 的之和）
- `clean_mean` / `clean_max`：两阶段清洗后 `clean_load` 的均值与最大值（保留 4 位小数取整）
"""
    ),
    code(
        """# @@todo(10) 组装 report 字典（10 个键）
# @@hint 键名必须完全一致；n_sentinel = (load>=1000).sum() + (load<0).sum()；clean_mean 用 round(float(...), 4)
report = {
    "n_rows": dev.shape[0],
    "n_dup_full": n_dup_full,
    "n_dup_biz": n_dup_biz,
    "n_dup_after_dedup": int(d_dedup_biz.duplicated().sum()),
    "n_sentinel": int((load >= 1000).sum() + (load < 0).sum()),
    "n_out_iqr": n_out_iqr,
    "n_out_grp": n_out_grp,
    "n_humid_bad": n_humid_bad,
    "clean_mean": round(float(clean_load.mean()), 4),
    "clean_max": round(float(clean_load.max()), 4),
}
# @@end

# ---- 验收 ----
expect_keys = {
    "n_rows", "n_dup_full", "n_dup_biz", "n_dup_after_dedup", "n_sentinel",
    "n_out_iqr", "n_out_grp", "n_humid_bad", "clean_mean", "clean_max",
}
assert isinstance(report, dict), "report 应该是字典"
assert set(report) == expect_keys, (
    f"键不对。多的是 {set(report) - expect_keys}，缺的是 {expect_keys - set(report)}"
)
assert report["n_rows"] == 270
assert report["n_dup_full"] == 6 and report["n_dup_biz"] == 10
assert report["n_dup_after_dedup"] == 0, "业务键去重后不应还有完全重复行"
assert report["n_sentinel"] == 11, "哨兵码应为 11 个（7 个 >=1000 + 4 个负值）"
assert report["n_out_iqr"] == 52 and report["n_out_grp"] == 10
assert report["n_humid_bad"] == 5
assert abs(report["clean_mean"] - 128.246) < 1e-3
assert abs(report["clean_max"] - 348.5687) < 1e-3
assert report["n_out_grp"] < report["n_out_iqr"], "分组口径应比全局口径精确"
print("综合 ② 通过 -> 清洗报告：")
for k, v in report.items():
    print(f"  {k:22s} = {v!r}")"""
    ),
    md(
        """
---

### 复盘提问

**复盘问自己**（四题都能答上来才算过）：

1. `duplicated(keep=False).sum()` 得到 12，题目问"有多少重复行"，能填 12 吗？为什么？
2. 为什么 `9999` 不能 `clip` 到 408？正确做法是什么？
3. 全局 IQR 判出 52 个、分组 IQR 判出 10 个。哪个对？错的那个错在哪？
4. 湿度的 5 个越界值，为什么用业务规则比用 IQR 更靠谱？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
