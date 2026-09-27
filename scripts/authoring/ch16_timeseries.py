"""ch16 —— 时间序列处理：`resample` / 日频特征 / 滞后特征

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch16_timeseries.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch16_timeseries"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

DATA = Path("data")

load = pd.read_csv(DATA / "load_curve.csv")
load["时间戳"] = pd.to_datetime(load["时间戳"])

KEY = "台区编号"
VAL = "负荷值"
TS = "时间戳"

# 单台区小时级序列（DatetimeIndex）
st = load[load[KEY] == "STATION_A_01"].sort_values(TS).set_index(TS)

print("pandas", pd.__version__)
print("load", load.shape, "| 台区", load[KEY].nunique(),
      "| 时间范围", load[TS].min().date(), "~", load[TS].max().date())
print("st", st.shape, "| 负荷值 NaN", int(st[VAL].isna().sum()))'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


print("st 索引类型:", type(st.index).__name__, "| 频率:", st.index.freq)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch16 时间序列处理：`resample` / 日频特征 / 滞后特征

> 方向：数据预处理 ｜ 竞赛对应：**数据准备及处理**（时序特征是负荷/功率预测题的
> 第一步，与 `05_timeseries` 直接衔接）+ **数据探索**（按周期看分布）。

时间序列的坑围绕两件事：**时间轴补齐**与**滞后错位**：

1. **`resample` 会把时间轴补齐**——没有数据的时间段自动出现（值是 NaN 或 0）
2. **`resample("W")` 的标签是右端（周日）**——不看 `label` 参数，日期全对不上
3. **滞后特征必须组内 `shift`**——不分组直接 shift，上一台区的尾巴塞给下一台区
4. **`shift` 之后的 NaN 有三种来源**——头部 / 组内原缺 / 传播，数字对不上先分家
5. **DatetimeIndex 支持字符串切片**——`st.loc["2026-01"]` 一行取一个月

> 本章的 `assert` 全部可以在方向 README 的「ch16 专项真值」对上。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据准备及处理 | 占技能操作 10% | 重采样、日频特征、滞后/差分特征 |
| 数据探索 | 每项 2 分 | 按天/周聚合、节假日对比、字符串切片 |

**学习目标**：把小时级负荷曲线变成"日均表 + 6 列时序特征"，全程不错位、
不泄漏未来。
"""
    ),
    md(
        """
## 二、API 速查表

| 方法 / 类 | 关键参数 | 返回 | 一句话说明 |
|---|---|---|---|
| `pd.to_datetime` | — | Series/DatetimeIndex | 字符串 → 时间戳 |
| `set_index(时间列)` | — | DataFrame | 时序操作的前提 |
| `resample("D")` | `on` `label` `closed` | Resampler | 重采样聚合（**补齐时间轴**） |
| `asfreq("h")` | — | DataFrame | 只改频率不聚合（补行填 NaN） |
| `pd.Grouper(key, freq)` | — | Grouper | groupby 里的重采样键 |
| `.dt` 访问器 | `year` `month` `dayofweek` `hour` | Series | 日历特征 |
| `shift(n)` | — | Series | 滞后 n 步（**必须组内**） |
| `diff(n)` / `pct_change(n)` | — | Series | 差分 / 变化率 |
| `df.loc["2026-01"]` | — | DataFrame | 字符串切片 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 两种写法等价：`on=` 与 DatetimeIndex

`resample` 要么在 DatetimeIndex 上做，要么用 `on=` 指定时间列。
结果完全一致——但**竞赛里推荐 set_index**，因为后面的切片、特征都靠它。
"""
    ),
    code(
        """first_station = load[KEY].iloc[0]
sub = load[load[KEY] == first_station].sort_values(TS)

r_on = sub.resample("D", on=TS)[VAL].mean()
r_idx = sub.set_index(TS).resample("D")[VAL].mean()

print("on= 版:", r_on.shape, "| DatetimeIndex 版:", r_idx.shape,
      "| 一致:", bool(np.allclose(r_on, r_idx)))
print("索引类型:", type(r_idx.index).__name__, "| 索引名:", r_idx.index.name)

assert r_on.shape == (90,) and r_idx.shape == (90,)
assert bool(np.allclose(r_on, r_idx))
assert type(r_idx.index).__name__ == "DatetimeIndex\""""
    ),
    md(
        """
## 3.2 `resample` 的频率与标签

数据是 2026-01-01 ~ 2026-03-31 的小时级曲线。`"D"` 出 90 桶、`"W"` 出 14 桶；
**周桶默认标签是右端（周日）**，`label="left"` 才变成周一开头——
报表日期对不上，十有八九是这个。
"""
    ),
    code(
        """for f in ["D", "W", "12h", "3D"]:
    r = st.resample(f)[VAL].mean()
    print(f"{f}: {len(r)} 桶 | 首桶 {r.index[0].date()} | NaN {int(r.isna().sum())}")

w = st.resample("W")[VAL].mean()
w_left = st.resample("W", label="left")[VAL].mean()
print("W 默认标签(右端):", [str(x.date()) for x in w.index[:3]])
print("label=left 标签 :", [str(x.date()) for x in w_left.index[:3]])

assert len(w) == 14
assert str(w.index[0].date()) == "2026-01-04", "默认标签是周日结尾"
assert str(w_left.index[0].date()) == "2025-12-28", "label=left 是周一开头\""""
    ),
    md(
        """
### 3.3 难点深挖：`resample` 会补齐时间轴——NaN 从哪来要看 `count`

**为什么难**：`resample` 的语义是"**时间轴上有这个桶就给这一行**"，没数据
的桶也会出现。本数据小时级连续、无缺口，但**负荷值本身有 NaN**——均值带
NaN 的桶不是"缺桶"，是"桶里有原缺"。两种 NaN 用 `count` 一分就清。

**错误示范**：

```python
st.resample("h")[VAL].mean().isna().sum()   # 23 ← 以为是缺了 23 个小时
```

**正误对照**：

| 检查 | 结果 | 含义 |
|---|---|---|
| `resample("h").size()` | 2160 桶、0 空桶 | 时间轴完整，没有缺口 |
| `resample("h")[VAL].mean().isna().sum()` | **23** | 全是**原缺传播**（A_01 有 23 个 NaN） |
| `resample("D")[VAL].agg(["mean","count"])` | 最小 count **18** | 有天缺到只剩 18 个点 |
| `asfreq("h")` | `(2160, 4)`，NaN 行 23 | asfreq 只改频率不聚合 |

**判定规则**：**resample 出 NaN 先问两件事——桶真的缺吗（看 size / count），
还是桶内原缺在传播（mean 对 NaN 不免疫）？** 需要无 NaN 的日均表，聚合前
先决定填充策略（`interpolate` / 分组填充，见 ch04）。
"""
    ),
    code(
        """hourly = st.resample("h")
print("小时桶数:", len(hourly.size()), "| 空桶:", int((hourly.size() == 0).sum()))
print("小时均值 NaN 桶:", int(hourly[VAL].mean().isna().sum()),
      "← 全部来自 A_01 的", int(st[VAL].isna().sum()), "个原缺")

daily = st.resample("D")[VAL].agg(["mean", "count", "max", "min"])
print("日均表:", daily.shape)
print("count=24 的完整天数:", int((daily["count"] == 24).sum()),
      "| 最小 count:", int(daily["count"].min()))
print("日均 mean min/max:", round(float(daily['mean'].min()), 2),
      "~", round(float(daily['mean'].max()), 2), "| NaN 天数:", int(daily['mean'].isna().sum()))

af = st.asfreq("h")
print("asfreq:", af.shape, "| NaN 行:", int(af[VAL].isna().sum()))

assert int((hourly.size() == 0).sum()) == 0, "时间轴无缺口"
assert int(hourly[VAL].mean().isna().sum()) == 23
assert daily.shape == (90, 4)
assert int((daily['count'] == 24).sum()) == 77 and int(daily['count'].min()) == 18
assert int(daily['mean'].isna().sum()) == 0, "每天至少 18 个点，均值不 NaN\""""
    ),
    md(
        """
## 3.4 `pd.Grouper`：分组重采样

三个台区都要按天聚合？`groupby([台区, Grouper(freq)])` 一步到位，
产物是"台区 × 时间"的多级索引 Series——`unstack` 即得宽表（衔接 ch15）。
"""
    ),
    code(
        """gd = load.groupby([KEY, pd.Grouper(key=TS, freq="D")])[VAL].mean()
wd = gd.unstack(0)

print("Grouper 产物:", gd.shape, "| 索引名:", gd.index.names)
print("unstack 后:", wd.shape, "| 列:", wd.columns.tolist())
print("A_01 首日均值:", round(float(gd.iloc[0]), 4))

assert gd.shape == (270,) and gd.index.names == [KEY, TS]
assert wd.shape == (90, 3)
assert round(float(gd.iloc[0]), 4) == 484.035"""
    ),
    md(
        """
## 3.5 日频特征：`.dt` 访问器

月份 / 小时 / 星期 / 周末，四列特征是时序题的起手式。
`DatetimeIndex` 直接 `.month`；普通 Series 走 `.dt.month`——别混。
"""
    ),
    code(
        """feat = st.copy()
feat["月份"] = feat.index.month
feat["小时"] = feat.index.hour
feat["星期"] = feat.index.dayofweek
feat["是否周末"] = (feat.index.dayofweek >= 5).astype(int)

print("月份取值:", sorted(feat["月份"].unique().tolist()))
print("星期分布:", feat["星期"].value_counts().sort_index().to_dict())
print("周末行数:", int(feat["是否周末"].sum()))
print("isocalendar 返回:", type(feat.index.isocalendar()).__name__)

assert sorted(feat["月份"].unique().tolist()) == [1, 2, 3]
assert int(feat["是否周末"].sum()) == 624
assert int(feat["星期"].value_counts()[2]) == 288, "周三只有 288 行（2 月少 4 天）\""""
    ),
    md(
        """
### 3.6 难点深挖：滞后特征必须组内 `shift`——错位是静默的

**为什么难**：`shift(24)` 在多台区表上不分组调用，**不报错**，但每个台区的
前 24 行拿到的是**上一个台区的尾巴**。数据按台区排序后三段拼接，只有两处
"接缝"，肉眼根本看不出来——直到模型训练时发现特征泄漏了别的台区。

**错误示范**：

```python
g["不分组滞后"] = g[VAL].shift(24)      # ← A_01 的前 24 行 = 表尾台区的最后 24 行
```

**正误对照**（按台区排序后）：

| 写法 | NaN 数 | 成分 |
|---|---|---|
| 不分组 `shift(24)` | 24 | 只有表头 24 行，**接缝处错位 48 行** |
| `groupby(KEY)[VAL].shift(24)` | **120** | 头部 3×24=72 + 组内原缺传播 48 |

**判定规则**：**多实体时序表，shift / diff / pct_change 永远包在
`groupby(实体)` 里；写完先验一句：每组前 n 行必须全 NaN。**
"""
    ),
    code(
        """g = load.sort_values([KEY, TS]).copy()
g["滞后1天"] = g.groupby(KEY)[VAL].shift(24)
g["日环比"] = g.groupby(KEY)[VAL].diff(24)
g["不分组滞后"] = g[VAL].shift(24)

n_lead = 3 * 24
print("滞后1天 NaN:", int(g["滞后1天"].isna().sum()),
      "= 头部", n_lead, "+ 内部", int(g["滞后1天"].isna().sum()) - n_lead)
print("日环比 NaN:", int(g["日环比"].isna().sum()),
      "= 头部", n_lead, "+ 内部", int(g["日环比"].isna().sum()) - n_lead)
print("不分组滞后 NaN:", int(g["不分组滞后"].isna().sum()),
      "| 与分组版 NaN 模式不同的行:", int((g["滞后1天"].isna() != g["不分组滞后"].isna()).sum()))
print("各组前 24 行滞后全 NaN:", bool(g.groupby(KEY)["滞后1天"].head(24).isna().all()))
print("滞后1天 与原值相关:", round(float(g[[VAL, '滞后1天']].corr().iloc[0, 1]), 4))

assert int(g["滞后1天"].isna().sum()) == 120
assert int(g["日环比"].isna().sum()) == 168
assert int((g["滞后1天"].isna() != g["不分组滞后"].isna()).sum()) == 48, "接缝错位 48 行"
assert bool(g.groupby(KEY)["滞后1天"].head(24).isna().all())
assert round(float(g[[VAL, '滞后1天']].corr().iloc[0, 1]), 4) == 0.9593"""
    ),
    md(
        """
## 3.7 时序切片：字符串直接切

DatetimeIndex 支持字符串切片：`"2026-01"` 取一个月、`"2026-01-01"` 取一天、
起止字符串取区间。不用 `pd.Timestamp` 包来包去。
"""
    ),
    code(
        """jan = st.loc["2026-01"]
day1 = st.loc["2026-01-01"]
week1 = st.loc["2026-01-01":"2026-01-07"]

print("1 月切片:", jan.shape, "| 单日:", day1.shape,
      "| 单日均值:", round(float(day1[VAL].mean()), 3), "| 首周:", week1.shape)

assert jan.shape == (744, 4)
assert day1.shape == (24, 4)
assert round(float(day1[VAL].mean()), 3) == 484.035
assert week1.shape == (168, 4)"""
    ),
    md(
        """
## 3.8 综合题：节假日 vs 工作日

数据自带 `是否节假日` 列。按它分组看负荷差异，是探索题的送分动作——
但注意均值差异要**配数字**，只说"有差异"拿不到分。
"""
    ),
    code(
        """hol_mean = load.loc[load["是否节假日"], VAL].mean()
work_mean = load.loc[~load["是否节假日"], VAL].mean()

print("节假日均值:", round(float(hol_mean), 3),
      "| 工作日均值:", round(float(work_mean), 3),
      "| 差:", round(float(hol_mean - work_mean), 3))
print("节假日行数:", int(load["是否节假日"].sum()))

assert round(float(hol_mean), 3) == 513.004
assert round(float(work_mean), 3) == 678.191
assert int(load["是否节假日"].sum()) == 720"""
    ),
    md(
        """
---

## 四、易错点清单

1. `resample` 只对 DatetimeIndex（或 `on=` 指定列）生效。
2. `resample("W")` 标签默认右端（周日），报表对不上先查 `label`。
3. resample 补齐时间轴：空桶也会出现；NaN 先分清"缺桶"还是"原缺传播"（看 count）。
4. `asfreq` 只改频率不聚合，非对齐频率直接插 NaN。
5. 日历特征：DatetimeIndex 直接 `.month`，Series 要 `.dt.month`。
6. 多实体表 shift / diff / pct_change 必须组内，否则接缝静默错位。
7. shift 后的 NaN 三种来源（头部 / 原缺传播 / diff 多减一段），数字要对账。
8. 滞后特征是"用过去预测现在"，切训练集前先想清楚泄漏方向。

## 五、本章小结

- `resample` 三步：定频率 → 定标签（`label`/`closed`）→ 定聚合；`count` 永远陪跑。
- 分组重采样用 `pd.Grouper`；宽表用 `unstack`。
- 日历特征四件套 + `isocalendar()`（返回 DataFrame）。
- 滞后/差分包进 `groupby(实体)`，验收标准：每组前 n 行全 NaN。

### 复盘提问

1. resample 出现 NaN，怎么判断是"缺桶"还是"原缺传播"？
2. `"W"` 标签默认在哪端？`label` 与 `closed` 分别控制什么？
3. 不分组 `shift(24)` 在本数据上错位几行？为什么是 48 不是 72？
4. `.month` 和 `.dt.month` 分别用在什么对象上？
5. 滞后特征的验收标准是什么？

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
# ch16 练习：时间序列处理

与讲解版逐 Cell 对应，`assert` 验收保留。卡住回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 数据准备及处理 | 重采样 + 日频特征 + 滞后特征 |
| 数据探索 | 字符串切片 + 节假日对比 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：resample 频率与标签（对应讲解 §3.2）"),
    code(
        """# @@todo(1) 对 st 按周重采样取负荷均值，存入 w；再存 label=\"left\" 版 w_left
# @@hint st.resample("W")[VAL].mean()
w = st.resample("W")[VAL].mean()
w_left = st.resample("W", label="left")[VAL].mean()
# @@end

print("W 桶数:", len(w), "| 默认首桶:", w.index[0].date(),
      "| left 首桶:", w_left.index[0].date())
assert len(w) == 14
assert str(w.index[0].date()) == "2026-01-04"
assert str(w_left.index[0].date()) == "2025-12-28\""""
    ),
    md("## 题 2：NaN 三分家（对应讲解 §3.3）"),
    code(
        """# @@todo(2) 对 st 按小时重采样：桶数存入 n_buckets、空桶数存入 n_empty、
#            负荷均值 NaN 桶数存入 n_nan
hr = st.resample("h")
n_buckets = len(hr.size())
n_empty = int((hr.size() == 0).sum())
n_nan = int(hr[VAL].mean().isna().sum())
# @@end

# @@todo(3) 对 st 按天聚合 mean/count/max/min 存入 daily；最小 count 存入 min_cnt
daily = st.resample("D")[VAL].agg(["mean", "count", "max", "min"])
min_cnt = int(daily["count"].min())
# @@end

print("小时桶:", n_buckets, "| 空桶:", n_empty, "| NaN 桶:", n_nan)
print("日均表:", daily.shape, "| 最小 count:", min_cnt)
assert n_buckets == 2160 and n_empty == 0 and n_nan == 23
assert daily.shape == (90, 4) and min_cnt == 18"""
    ),
    md("## 题 3：pd.Grouper 分组重采样（对应讲解 §3.4）"),
    code(
        """# @@todo(4) 全表按 [台区, Grouper(天)] 求负荷均值存入 gd；unstack 成宽表存入 wd
# @@hint pd.Grouper(key=TS, freq="D")
gd = load.groupby([KEY, pd.Grouper(key=TS, freq="D")])[VAL].mean()
wd = gd.unstack(0)
# @@end

print("gd:", gd.shape, gd.index.names, "| wd:", wd.shape)
assert gd.shape == (270,) and gd.index.names == [KEY, TS]
assert wd.shape == (90, 3)
assert round(float(gd.iloc[0]), 4) == 484.035"""
    ),
    md("## 题 4：日频特征四件套（对应讲解 §3.5）"),
    code(
        """# @@todo(5) 给 feat 加四列：月份 / 小时 / 星期 / 是否周末（周末=1）
# @@hint feat.index.month ；(feat.index.dayofweek >= 5).astype(int)
feat = st.copy()
feat["月份"] = feat.index.month
feat["小时"] = feat.index.hour
feat["星期"] = feat.index.dayofweek
feat["是否周末"] = (feat.index.dayofweek >= 5).astype(int)
# @@end

print("月份:", sorted(feat["月份"].unique().tolist()),
      "| 周末行数:", int(feat["是否周末"].sum()))
assert sorted(feat["月份"].unique().tolist()) == [1, 2, 3]
assert int(feat["是否周末"].sum()) == 624
assert int(feat["星期"].value_counts()[2]) == 288"""
    ),
    md("## 题 5：组内滞后（对应讲解 §3.6）"),
    code(
        """# @@todo(6) 按 [台区, 时间戳] 排序后，组内 shift(24) 生成「滞后1天」、
#            diff(24) 生成「日环比」；再生成不分组版「不分组滞后」
g = load.sort_values([KEY, TS]).copy()
g["滞后1天"] = g.groupby(KEY)[VAL].shift(24)
g["日环比"] = g.groupby(KEY)[VAL].diff(24)
g["不分组滞后"] = g[VAL].shift(24)
# @@end

print("滞后1天 NaN:", int(g["滞后1天"].isna().sum()),
      "| 日环比 NaN:", int(g["日环比"].isna().sum()))
print("错位行数:", int((g["滞后1天"].isna() != g["不分组滞后"].isna()).sum()))
print("验收 - 各组前 24 行全 NaN:",
      bool(g.groupby(KEY)["滞后1天"].head(24).isna().all()))
assert int(g["滞后1天"].isna().sum()) == 120
assert int(g["日环比"].isna().sum()) == 168
assert int((g["滞后1天"].isna() != g["不分组滞后"].isna()).sum()) == 48"""
    ),
    md("## 题 6（综合）：字符串切片 + 节假日对比（对应讲解 §3.7-3.8）"),
    code(
        """# @@todo(7) 用字符串切片取 A_01 的 1 月整月存入 jan、单日 2026-01-01 存入 day1
jan = st.loc["2026-01"]
day1 = st.loc["2026-01-01"]
# @@end

# @@todo(8) 全表按是否节假日分两组算负荷均值：hol_mean / work_mean
# @@hint load.loc[load["是否节假日"], VAL].mean()
hol_mean = load.loc[load["是否节假日"], VAL].mean()
work_mean = load.loc[~load["是否节假日"], VAL].mean()
# @@end

print("1 月:", jan.shape, "| 单日:", day1.shape,
      "| 单日均值:", round(float(day1[VAL].mean()), 3))
print("节假日:", round(float(hol_mean), 3), "vs 工作日:", round(float(work_mean), 3))
assert jan.shape == (744, 4) and day1.shape == (24, 4)
assert round(float(day1[VAL].mean()), 3) == 484.035
assert round(float(hol_mean), 3) == 513.004
assert round(float(work_mean), 3) == 678.191"""
    ),
    md(
        """
---

## 综合自查

1. 题 1 两种标签各适合什么报表场景？
2. 题 2 的 23 个 NaN 桶是缺口吗？用什么列证明？
3. 题 5 的 120 和 48 分别怎么算出来的？
4. 滞后特征的验收标准是什么？为什么重要？

全部答得上来，进入 ch17（性能与内存）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
