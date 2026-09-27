#!/usr/bin/env python3
"""ch01 —— 时序数据与特征工程：规则时间轴 / 时间特征 / 滞后滑窗 / 防泄漏 / 周期分解

本章主线（五条）：

1. **脏时间戳 → 规则时间轴**：乱序、重复、3 种混排格式，以及 `asfreq` 为什么必须排在最后
2. **时间特征**：`.dt` 派生 + 节假日标记 + **周期整数的 sin/cos 编码**（23h 与 0h 才是相邻的）
3. **滞后 / 滑窗特征**：组内 `shift`、`rolling` 的 NaN 传播（70 个缺失会污染 1009 行）
4. **监督矩阵与三种泄漏**：未来窗口统计量进特征 / 随机切分 / 全量标准化（最后一个是负结果）
5. **周期分解**：手写「移动平均 + 按周期分组均值 + 残差」，并用「高/低负荷段残差比」判定
   该用加法还是对数（1.171 vs 1.011）；再用残差 ACF 诊断出「日周期幅度随季节变」

数据：`data/load_curve.csv`（2 台区 × 8760 h）、`data/load_curve_messy.csv`（336 行脏快照）
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python coding/05_timeseries/data/make_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch01_features"

# =========================================================================== #
# 1. 公共导入 / 数据加载 / 脚手架（两版都给，不挖空）                          #
# =========================================================================== #

IMPORTS = '''from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

DATA = Path("data")          # notebook 的 cwd = coding/05_timeseries
CSV = DATA / "load_curve.csv"
MESSY_CSV = DATA / "load_curve_messy.csv"

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
print("原始行数", len(raw), "| 列", list(raw.columns))
print("台区", sorted(raw["台区编号"].unique()))
print("时间跨度", raw["时间戳"].min(), "~", raw["时间戳"].max())
print("原始时间戳单调递增？", raw["时间戳"].is_monotonic_increasing)
'''

SETUP = '''# 规则化的第一步：去重 + 按时间排序。
# **顺序不能反** —— 先用 asfreq 再排序，规则时间轴会被重复时刻搞乱（见 1.1.1）
df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
print("去重排序后", len(df), "行 | 单调递增", df["时间戳"].is_monotonic_increasing)
print("负荷值缺失", int(df["负荷值"].isna().sum()),
      "| 温度缺失", int(df["温度"].isna().sum()))
print(df.head(3))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
HOLIDAY_RANGES = [
    ("2025-01-01", "2025-01-01"),   # 元旦
    ("2025-01-28", "2025-02-04"),   # 春节
    ("2025-04-04", "2025-04-06"),   # 清明
    ("2025-05-01", "2025-05-05"),   # 劳动节
    ("2025-05-31", "2025-06-02"),   # 端午
    ("2025-10-01", "2025-10-08"),   # 中秋 + 国庆
]
HOLIDAYS = set()
for _lo, _hi in HOLIDAY_RANGES:
    HOLIDAYS |= set(pd.date_range(_lo, _hi, freq="D").strftime("%Y-%m-%d"))


def clean_series(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值。

    为什么不能直接用「全局中位数 × 3」：工业台区是「夜间低 + 日间高」的双峰
    分布，全局阈值会把整段日间峰误判成异常（实测 B 台区会误杀 382 个点）。
    """
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


def cyc(values, period):
    """把「周期整数」编码成 (sin, cos) 两列，原理见 1.2.1。"""
    ang = 2 * np.pi * np.asarray(values, dtype=float) / period
    return np.sin(ang), np.cos(ang)


def acf(s, lag):
    """自相关（转发 pandas 内置实现，便于阅读）。"""
    return float(s.autocorr(lag))
'''

# =========================================================================== #
# 2. 任务代码块                                                              #
# =========================================================================== #

T1_CODE = '''a_frame = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳").sort_index()
b_frame = df[df["台区编号"] == "STATION_B_02"].set_index("时间戳").sort_index()

# @@todo 把 A 台区变成**小时级规则序列**——缺了的时刻要显式补成 NaN
# @@hint Series.asfreq("h")：对齐到小时频率；它只对齐、不聚合（与 resample 的区别在这）
s_a = a_frame["负荷值"].asfreq("h")
# @@end

# @@todo B 台区同样处理
s_b = b_frame["负荷值"].asfreq("h")
# @@end

print("A 台区 %d 个整点 | 缺失 %d | 完整率 %.6f"
      % (len(s_a), int(s_a.isna().sum()), 1 - s_a.isna().mean()))
print("B 台区 %d 个整点 | 缺失 %d | 完整率 %.6f"
      % (len(s_b), int(s_b.isna().sum()), 1 - s_b.isna().mean()))

# ---- 验收 ----
assert len(s_a) == 8760 and len(s_b) == 8760, "两个台区都应是 8760 个整点"
assert int(s_a.isna().sum()) == 70, f"A 台区应缺 70 个，实际 {int(s_a.isna().sum())}"
assert int(s_b.isna().sum()) == 79, f"B 台区应缺 79 个，实际 {int(s_b.isna().sum())}"
assert isinstance(s_a.index, pd.DatetimeIndex) and s_a.index.freq is not None
'''

T2_CODE = '''messy = pd.read_csv(MESSY_CSV, encoding="utf-8-sig")
print("列名（注意两侧带空格）:", list(messy.columns))
print("时间戳前 4 行:", messy[" 时间戳 "].head(4).tolist())

# @@todo 把 3 种混排格式统一解析成 datetime
# @@hint 直接 pd.to_datetime(列) 会 ValueError（"2025/03/01 01:00" 不匹配默认格式）
# @@hint 只加 format="mixed" 仍会 DateParseError（中文的 "2025年3月1日 2时" 解析不了）
# @@hint 先把中文「年月日时」换成 ASCII 分隔符，再 format="mixed"
ts_text = (messy[" 时间戳 "]
           .str.replace("年", "-", regex=False).str.replace("月", "-", regex=False)
           .str.replace("日", " ", regex=False).str.replace("时", ":00", regex=False))
ts_dt = pd.to_datetime(ts_text, format="mixed")
# @@end

# @@todo 把负荷列转成数值——千分位逗号要先去掉，脏文本交给 errors="coerce"
# @@hint Series.str.replace(",", "", regex=False) 然后 pd.to_numeric(..., errors="coerce")
load_num = pd.to_numeric(messy["负荷值(kW)"].str.replace(",", "", regex=False),
                         errors="coerce")
# @@end

print("解析后 NaT %d | 负荷列 NaN %d | 哨兵 9999 %d"
      % (int(ts_dt.isna().sum()), int(load_num.isna().sum()), int((load_num == 9999).sum())))

# ---- 验收 ----
assert len(messy) == 336
assert int(ts_dt.isna().sum()) == 0, "336 行时间戳应全部解析成功"
assert int(load_num.isna().sum()) == 8, "去掉千分位后仍有 8 个脏文本（--/N/A/空串）"
assert int((load_num == 9999).sum()) == 3
'''

T3_CODE = '''feat = pd.DataFrame(index=s_a.index)

# @@todo 派生 4 个基础时间特征
# @@hint 全部走 DatetimeIndex 属性：hour / dayofweek / month / dayofyear
# @@hint dayofweek 是 0=周一 … 6=周日
feat["hour"] = s_a.index.hour
feat["dow"] = s_a.index.dayofweek
feat["month"] = s_a.index.month
feat["doy"] = s_a.index.dayofyear
# @@end

# @@todo 用脚手架里的 HOLIDAYS 集合标出节假日（0/1）
# @@hint index.strftime("%Y-%m-%d") 得到日期字符串 → .isin(HOLIDAYS) → .astype(int)
feat["is_holiday"] = s_a.index.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)
# @@end

feat["is_weekend"] = (s_a.index.dayofweek >= 5).astype(int)

print(feat.head(3))
print("节假日小时 %d | 周末小时 %d"
      % (int(feat["is_holiday"].sum()), int(feat["is_weekend"].sum())))

# ---- 验收 ----
assert int(feat["is_holiday"].sum()) == 672, "2025 年法定节假日 28 天 × 24 = 672 小时"
assert int(feat["is_weekend"].sum()) == 2496
assert int(feat["doy"].max()) == 365 and int(feat["hour"].max()) == 23
'''

T4_CODE = '''# @@todo 把 hour 编码成 (sin, cos) 两列（周期 24）
# @@hint 脚手架已有 cyc(values, period)，返回 (sin, cos) 两个数组
feat["hour_sin"], feat["hour_cos"] = cyc(s_a.index.hour, 24)
# @@end

# @@todo 星期几同样处理（周期 7）
feat["dow_sin"], feat["dow_cos"] = cyc(s_a.index.dayofweek, 7)
# @@end

d_raw = abs(23 - 0)
d_cyc = float(np.hypot(feat["hour_sin"].iloc[23] - feat["hour_sin"].iloc[0],
                       feat["hour_cos"].iloc[23] - feat["hour_cos"].iloc[0]))
d_12 = float(np.hypot(feat["hour_sin"].iloc[12] - feat["hour_sin"].iloc[0],
                      feat["hour_cos"].iloc[12] - feat["hour_cos"].iloc[0]))
print("23 点 vs 0 点：原始距离 %d → 编码后弦长 %.6f" % (d_raw, d_cyc))
print("对照 0 点 vs 12 点：原始距离 12 → 编码后弦长 %.6f" % d_12)
print("理论值 2*sin(pi/24) = %.6f | 2*sin(pi/2) = %.6f"
      % (2 * np.sin(np.pi / 24), 2 * np.sin(np.pi / 2)))

# ---- 验收 ----
assert d_raw == 23
assert abs(d_cyc - 2 * np.sin(np.pi / 24)) < 1e-12, "23h 与 0h 在编码空间只差 0.261"
assert abs(d_12 - 2.0) < 1e-12, "0h 与 12h 恰好相差一个直径"
'''

T5_CODE = '''d2 = df.sort_values(["台区编号", "时间戳"]).reset_index(drop=True)

# @@todo 先做**错误示范**——全局 shift（不分组）
# @@hint Series.shift(1)
d2["lag_bad"] = d2["负荷值"].shift(1)
# @@end

# @@todo 再做正确写法——按台区**组内** shift
# @@hint DataFrame.groupby("台区编号")["负荷值"].shift(1)
d2["lag_ok"] = d2.groupby("台区编号")["负荷值"].shift(1)
# @@end

a_last = float(d2[d2["台区编号"] == "STATION_A_01"]["负荷值"].iloc[-1])
print("A 台区最后一行负荷            : %.2f" % a_last)
print("B 台区首行 全局 shift 取到     : %.2f" % float(d2["lag_bad"].iloc[8760]))
print("B 台区首行 组内 shift 取到     : %s" % d2["lag_ok"].iloc[8760])
print("组内 shift 的 NaN 总数         : %d" % int(d2["lag_ok"].isna().sum()))

# ---- 验收 ----
assert abs(float(d2["lag_bad"].iloc[8760]) - a_last) < 1e-9, "全局 shift 把 A 的末行喂给了 B"
assert pd.isna(d2["lag_ok"].iloc[8760]), "组内 shift 的首行必须是 NaN"
assert int(d2["lag_ok"].isna().sum()) == 151, "149 个原始缺失移位 + 2 个组首行"
'''

T6_CODE = '''# @@todo 清洗 A 台区序列（脚手架已有 clean_series）
s_a_clean = clean_series(s_a)
# @@end

print("1 步自相关：未清洗 %.6f → 清洗后 %.6f" % (acf(s_a, 1), acf(s_a_clean, 1)))

# @@todo 分别在未清洗 / 已清洗序列上做 rolling(24).mean()，数出 NaN 个数
# @@hint Series.shift(1) 先错开一格（避免把当前时刻算进窗口）
# @@hint rolling(24) 默认 min_periods=24：窗口里只要有一个 NaN，整行就是 NaN
nan_raw = int(s_a.shift(1).rolling(24).mean().isna().sum())
nan_clean = int(s_a_clean.shift(1).rolling(24).mean().isna().sum())
# @@end

print("rolling(24).mean() 的 NaN：未清洗 %d | 清洗后 %d" % (nan_raw, nan_clean))

# ---- 验收 ----
assert nan_raw == 1009, "只有 70 个缺失，却污染了 1009 行——NaN 沿窗口向后传播"
assert nan_clean == 24, "清洗后只剩开头的窗口不完整导致的 24 行"
'''

T7_CODE = '''w = pd.DataFrame(index=s_a_clean.index)
w["y"] = s_a_clean.shift(-1)         # 目标：下一小时的负荷
prev = s_a_clean.shift(1)            # 上一时刻（用来构造滑窗，天然不含未来）

# @@todo 构造 3 个滞后特征 lag1 / lag2 / lag24
# @@hint Series.shift(k)：正数 = 往回看 k 步
w["lag1"] = s_a_clean.shift(1)
w["lag2"] = s_a_clean.shift(2)
w["lag24"] = s_a_clean.shift(24)
# @@end

# @@todo 用「上一时刻起的 24 小时窗口」算均值和标准差
# @@hint prev 已定义为 shift(1)，直接 prev.rolling(24).mean() / prev.rolling(24).std()
w["ma24"] = prev.rolling(24).mean()
w["std24"] = prev.rolling(24).std()
# @@end

w["hour_sin"], w["hour_cos"] = cyc(w.index.hour, 24)
w["dow_sin"], w["dow_cos"] = cyc(w.index.dayofweek, 7)

FEAT = ["lag1", "lag2", "lag24", "ma24", "std24",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
model = w[FEAT + ["y"]].dropna()
print("特征数 %d | dropna 前 %d 行 → 后 %d 行（损失 %d）"
      % (len(FEAT), len(w), len(model), len(w) - len(model)))
print("lag1  与 y 相关 %.6f" % float(model["lag1"].corr(model["y"])))
print("lag24 与 y 相关 %.6f  ← 比 lag1 还高" % float(model["lag24"].corr(model["y"])))
print("ma24  与 y 相关 %.6f" % float(model["ma24"].corr(model["y"])))

# ---- 验收 ----
assert len(w) - len(model) == 25, "lag24 吃掉 24 行 + y 的尾部 1 行"
assert abs(float(model["lag1"].corr(model["y"])) - 0.653541) < 1e-6
assert abs(float(model["lag24"].corr(model["y"])) - 0.826982) < 1e-6
'''

T8_CODE = '''cut = int(len(w) * 0.8)
tr = w.iloc[:cut][FEAT + ["y"]].dropna()
te = w.iloc[cut:][FEAT + ["y"]].dropna()
print("切分点 %d | 时间 %s | 训练 %d / 测试 %d"
      % (cut, w.index[cut], len(tr), len(te)))
print("训练段 y 均值 %.4f | 测试段 y 均值 %.4f" % (tr["y"].mean(), te["y"].mean()))

scaler = StandardScaler().fit(tr[FEAT])
base = Ridge(alpha=1.0).fit(scaler.transform(tr[FEAT]), tr["y"])
mae_base = mean_absolute_error(te["y"], base.predict(scaler.transform(te[FEAT])))
print("① 干净基线 测试 MAE %.6f" % mae_base)

# @@todo 构造「未来 24 小时平均负荷」这个**看起来完全合理**的泄漏特征
# @@hint 想让窗口落进未来：先 shift(-1) 把序列整体前移，再 rolling(24).mean()
# @@hint 想清楚：y 就是 s_a_clean 在 t+1 的值，那这个窗口里含不含 y？
w["fut_ma24"] = s_a_clean.shift(-1).rolling(24).mean()
# @@end

FEAT_LEAK = FEAT + ["fut_ma24"]
tr2 = w.iloc[:cut][FEAT_LEAK + ["y"]].dropna()
te2 = w.iloc[cut:][FEAT_LEAK + ["y"]].dropna()
sc2 = StandardScaler().fit(tr2[FEAT_LEAK])
leaked = Ridge(alpha=1.0).fit(sc2.transform(tr2[FEAT_LEAK]), tr2["y"])
mae_leak = mean_absolute_error(te2["y"], leaked.predict(sc2.transform(te2[FEAT_LEAK])))
print("② 加 fut_ma24 测试 MAE %.6f（%+.6f）← 指标变好，是假的"
      % (mae_leak, mae_leak - mae_base))

# @@todo 用随机抽样做一次「错误切分」
# @@hint DataFrame.sample(frac=0.8, random_state=42) 拿随机训练集，drop 掉它们就是测试集
rand_tr = model.sample(frac=0.8, random_state=42)
rand_te = model.drop(rand_tr.index)
# @@end

# @@todo 在随机切分上训练 + 评估
# @@hint 与基线三步完全一样：fit scaler → fit Ridge → predict 后算 MAE
sc_r = StandardScaler().fit(rand_tr[FEAT])
m_r = Ridge(alpha=1.0).fit(sc_r.transform(rand_tr[FEAT]), rand_tr["y"])
mae_rand = mean_absolute_error(rand_te["y"], m_r.predict(sc_r.transform(rand_te[FEAT])))
# @@end

mdl_x = model.iloc[: int(len(model) * 0.8)][FEAT]
sc_t = StandardScaler().fit(mdl_x)
m_t = Ridge(alpha=1.0).fit(sc_t.transform(mdl_x), model.iloc[: int(len(model) * 0.8)]["y"])
mae_time = mean_absolute_error(
    model.iloc[int(len(model) * 0.8):]["y"],
    m_t.predict(sc_t.transform(model.iloc[int(len(model) * 0.8):][FEAT])))
print("③ 随机切分 MAE %.6f vs 时间切分 MAE %.6f（差 %+.6f）"
      % (mae_rand, mae_time, mae_rand - mae_time))

# @@todo 标准化参数改用**全量**数据拟合（第三种泄漏，看它还灵不灵）
# @@hint 只把 fit(tr[FEAT]) 换成 fit(w[FEAT].dropna())，其余完全不动
sc_all = StandardScaler().fit(w[FEAT].dropna())
m_all = Ridge(alpha=1.0).fit(sc_all.transform(tr[FEAT]), tr["y"])
mae_scaler = mean_absolute_error(te["y"], m_all.predict(sc_all.transform(te[FEAT])))
# @@end

print("④ 全量标准化 MAE %.6f（%+.6f）← 几乎没动"
      % (mae_scaler, mae_scaler - mae_base))
print("   lag1 全量均值 %.4f vs 训练段均值 %.4f（差 %.4f）"
      % (w["lag1"].mean(), tr["lag1"].mean(), abs(w["lag1"].mean() - tr["lag1"].mean())))

# ---- 验收 ----
assert abs(mae_base - 26.797903) < 1e-5
assert abs(mae_leak - 19.958735) < 1e-5 and mae_leak < mae_base
assert abs(mae_rand - 25.643746) < 1e-5 and mae_rand < mae_time
assert abs(mae_scaler - 26.797910) < 1e-5, "标准化泄漏在本数据上几乎无影响——这是负结果"
'''

T9_CODE = '''s = s_a_clean

# @@todo 低频项 = 居中 24 小时移动平均
# @@hint rolling(24, center=True, min_periods=24).mean()；首尾各 12 行会是 NaN，
# @@hint 再用 .bfill().ffill() 把边缘补齐
trend = s.rolling(24, center=True, min_periods=24).mean().bfill().ffill()
# @@end

detr = s - trend

# @@todo 日周期项 = 去趋势后「按小时」分组的均值（广播回原长度）
# @@hint groupby(detr.index.hour).transform("mean")
season = detr.groupby(detr.index.hour).transform("mean")
# @@end

resid = detr - season
recon_err = float((trend + season + resid - s).abs().max())
tr_by_m = trend.groupby(trend.index.month).mean()
print("重建最大绝对误差 %.3e（三层相加必然还原原序列）" % recon_err)
print("低频项 月均 peak/valley %.4f / %.4f（月 %d / %d）"
      % (tr_by_m.max(), tr_by_m.min(), int(tr_by_m.idxmax()), int(tr_by_m.idxmin())))
print("日周期项 peak/valley %.4f / %.4f（hour %d / %d）"
      % (season.max(), season.min(),
         int(season.groupby(season.index.hour).mean().idxmax()),
         int(season.groupby(season.index.hour).mean().idxmin())))
print("残差 std %.6f | 残差 std / 序列均值 %.6f" % (resid.std(), resid.std() / s.mean()))

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
n_show = 24 * 21
axes[0].plot(s.index[:n_show], s.iloc[:n_show], lw=0.8, label="原序列")
axes[0].plot(trend.index[:n_show], trend.iloc[:n_show], lw=1.6, label="低频项（24h 居中均值）")
axes[0].set_ylabel("负荷 kW")
axes[0].legend(loc="upper right")
axes[1].plot(detr.index[:n_show], detr.iloc[:n_show], lw=0.6, color="gray", label="去趋势")
axes[1].plot(season.index[:n_show], season.iloc[:n_show], lw=1.6, color="tab:orange",
             label="日周期项")
axes[1].set_ylabel("去趋势后")
axes[1].legend(loc="upper right")
axes[2].plot(resid.index[:n_show], resid.iloc[:n_show], lw=0.6, color="crimson")
axes[2].set_ylabel("残差")
axes[2].set_xlabel("时间")
plt.tight_layout()

# ---- 验收 ----
assert recon_err < 1e-9
assert abs(float(season.max()) - 121.8327) < 1e-3
assert abs(float(season.min()) - (-60.1567)) < 1e-3
assert abs(float(resid.std()) - 20.673831) < 1e-5
'''

T10_CODE = '''ls = np.log(s)
print("取对数前 min/max %.2f / %.2f（必须全 > 0 才能取对数）" % (s.min(), s.max()))

# @@todo 在**对数域**做一模一样的分解
# @@hint 低频项：rolling(24, center=True, min_periods=24).mean().bfill().ffill()
l_trend = ls.rolling(24, center=True, min_periods=24).mean().bfill().ffill()
# @@end

# @@todo 对数域的残差 = 去低频后再减掉「按小时分组均值」
# @@hint 把上一行的结果放在变量 l_detr 里，避免重复写三遍相同的表达式
l_detr = ls - l_trend
l_resid = l_detr - l_detr.groupby(l_detr.index.hour).transform("mean")
# @@end

# @@todo 算「高负荷段 / 低负荷段」的残差标准差之比（加法与对数各一个）
# @@hint 用中位数切成两段：resid[x >= x.quantile(0.5)] 与 resid[x < x.quantile(0.5)]
add_ratio = float(resid[s >= s.quantile(0.5)].std() / resid[s < s.quantile(0.5)].std())
log_ratio = float(l_resid[ls >= ls.quantile(0.5)].std() / l_resid[ls < ls.quantile(0.5)].std())
# @@end

print("加法：残差 std %.6f | 高/低段比 %.6f  ← 高负荷段波动明显更大" % (resid.std(), add_ratio))
print("对数：残差 std %.6f | 高/低段比 %.6f  ← 两段几乎相等" % (l_resid.std(), log_ratio))
print("换算成相对残差 expm1(%.6f) = %.6f" % (l_resid.std(), float(np.expm1(l_resid.std()))))

# ---- 验收 ----
assert abs(add_ratio - 1.171187) < 1e-5, "加法分解的残差方差随负荷水平增长 17%"
assert abs(log_ratio - 1.011203) < 1e-5, "对数分解后残差方差不再随水平变化"
'''

T11_CODE = '''LAGS = (1, 24, 168)

# @@todo 看两个残差的自相关——是快速衰减（白噪声）还是平坦（还有慢变结构）？
# @@hint 脚手架 acf(series, lag)
acf_add = [round(acf(resid, k), 4) for k in LAGS]
acf_log = [round(acf(l_resid, k), 4) for k in LAGS]
# @@end

print("lag            ", "  ".join("lag%-6d" % k for k in LAGS))
print("加法残差 ACF   ", "  ".join("%+.4f " % v for v in acf_add))
print("对数残差 ACF   ", "  ".join("%+.4f " % v for v in acf_log))

# @@todo 诊断——日周期的**幅度**是不是随季节在变？
# @@hint 先用 (month, hour) 分组把 l_detr 压成 288 格，再算每个月内 24 小时的最大-最小
amp = (l_detr.groupby([l_detr.index.month, l_detr.index.hour]).mean()
       .groupby(level=0).apply(lambda x: x.max() - x.min()))
# @@end

print("日周期幅度按月（对数域）:", {int(k): round(float(v), 4) for k, v in amp.items()})

# @@todo 把季节项从「按 hour」换成「按 (月, 小时)」，看残差是否变白
# @@hint groupby([index.month, index.hour]).transform("mean")
season2 = l_detr.groupby([l_detr.index.month, l_detr.index.hour]).transform("mean")
resid2 = l_detr - season2
# @@end

print("季节项改 (月,小时) 后：残差 std %.6f → %.6f | ACF(1) %+.4f → %+.4f"
      % (l_resid.std(), resid2.std(), acf(l_resid, 1), acf(resid2, 1)))

# ---- 验收 ----
assert abs(float(amp.loc[7]) - 0.4567) < 1e-3, "7 月的日周期幅度最大"
assert abs(float(amp.loc[1]) - 0.3496) < 1e-3, "1 月的日周期幅度最小"
assert abs(resid2.std() - 0.034551) < 1e-6
assert acf(resid2, 24) < 0.05, "换成 (月,小时) 后 lag24 的自相关基本消失"
'''

T12_CODE = '''# @@todo 把长表 pivot 成「时间 × 台区」宽表
# @@hint df.pivot(index="时间戳", columns="台区编号", values="负荷值")
wide = df.pivot(index="时间戳", columns="台区编号", values="负荷值")
# @@end

s_b_clean = clean_series(s_b)
pair = pd.DataFrame({"A": s_a_clean, "B": s_b_clean})
print("宽表 shape %s | 两列全 NaN 的行 %d" % (wide.shape, int(wide.isna().all(axis=1).sum())))

# @@todo 比较两个台区的「日周期形状」——各自按小时取均值（24×2），再求相关系数
# @@hint pair.groupby(pair.index.hour).mean() → .corr() → 取 .iloc[0, 1]
pair_h = pair.groupby(pair.index.hour).mean()
shape_corr = float(pair_h.corr().iloc[0, 1])
# @@end

print("A-B 同时刻相关 %.6f（几乎不相关）" % float(s_a_clean.corr(s_b_clean)))
print("A-B 日周期形状相关 %.6f（负相关：晚高峰 vs 日间平台）" % shape_corr)

temp = (df[df["台区编号"] == "STATION_A_01"]
        .set_index("时间戳").sort_index()["温度"].asfreq("h").interpolate())

# @@todo 验证「温度—负荷」是 V 形而不是线性
# @@hint 线性看 s_a_clean.corr(temp)；V 形看 (temp - 22).abs().corr(s_a_clean)
lin = float(s_a_clean.corr(temp))
vshape = float((temp - 22).abs().corr(s_a_clean))
# @@end

bins = pd.cut(temp, [-10, 0, 10, 20, 26, 32, 40])
print("线性相关 %.6f | |T-22| 相关 %.6f" % (lin, vshape))
print("温度分箱后的负荷均值:",
      {str(k): round(float(v), 1) for k, v in s_a_clean.groupby(bins).mean().items()})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(pair_h.index, pair_h["A"], marker="o", label="A 居民型")
axes[0].plot(pair_h.index, pair_h["B"], marker="s", label="B 工业型")
axes[0].set_xlabel("小时")
axes[0].set_ylabel("平均负荷 kW")
axes[0].legend()
axes[1].plot(temp, s_a_clean, ".", ms=1, alpha=0.2)
axes[1].set_xlabel("温度 ℃")
axes[1].set_ylabel("负荷 kW")
axes[1].set_title("温度—负荷：V 形")
plt.tight_layout()

# ---- 验收 ----
assert wide.shape == (8760, 2)
assert abs(shape_corr - (-0.040653)) < 1e-6
assert abs(lin - (-0.188929)) < 1e-6
assert abs(vshape - 0.406437) < 1e-6, "换成 V 形度量后相关度从 0.19 涨到 0.41"
'''

LESSON = [
    md(
        """
# ch01 时序数据与特征工程（讲解版）

> **本节考点**：把「一份脏 CSV」变成「能喂给模型的监督学习矩阵」的全部中间步骤。
> 时序题失分很少失在模型上 —— **大多失在特征和切分上**。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 数据准备及处理 | 技能操作 10% | 规则时间轴、时间特征、滞后 / 滑窗特征 |
| 模型训练 | **技能操作 40%** | 特征矩阵是训练的直接输入，差一步模型就吃不进去 |
| 新能源功率 / 负荷预测 | 行业赛题 | 赛题第一步就是本章；`02_sklearn/final` 的综合题也复用本章套路 |

**与 `01_pandas/ch16` 的分工**：ch16 讲的是 pandas 的时序 API（`resample` / `.dt` / `shift`）；
本章讲的是**预测视角** —— 怎么造特征、怎么防泄漏、怎么拆周期。两者刻意不重复。

## 学习目标

1. 把「乱序 + 重复 + 缺失 + 3 种时间戳格式」的 CSV 变成规则时间轴
2. 说清**周期整数为什么必须 sin/cos 编码**，并会算编码前后的距离
3. 会组装「滞后 + 滑窗 + 日历」特征，并知道每一行 NaN 是怎么来的
4. **能当场认出三种时序泄漏**，并知道哪一种在本数据上其实"失灵"
5. 能手写三层周期分解，并用「高/低负荷段残差比」判断该不该取对数

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `pd.to_datetime(x, format=...)` | `format="mixed"` | `DatetimeIndex` | 混合格式解析；中文日期要先换成 ASCII |
| `Series.asfreq("h")` | 频率串 | 同类型 | **只对齐、不聚合**，把缺的时刻补成 NaN |
| `Series.resample("D").mean()` | 频率串 | 聚合后 | **会聚合**，桶内多个值压成一个 |
| `DatetimeIndex.hour / .dayofweek / .dayofyear` | — | `Index` | 派生时间特征；`dayofweek` 0 = 周一 |
| `Series.shift(k)` | `k > 0` 往回看 | 同类型 | 滞后特征；**必须组内 shift** |
| `Series.rolling(n)` | `min_periods` | `Rolling` | 滑窗；默认遇 NaN 整行皆 NaN |
| `Series.autocorr(lag)` | lag | `float` | 自相关，用来诊断残差是不是白噪声 |
| `DataFrame.pivot(...)` | index / columns / values | `DataFrame` | 长表 → 宽表（多台区对齐） |
| `pd.cut(x, bins)` | 分箱边界 | `Categorical` | 温度等连续量的分段统计 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md(
        """
## 1.1 从 CSV 到「规则时间轴」

时序数据进模型前必须先过三关：**去重 → 排序 → 对齐到规则频率**。

先看数据有多脏。`load_curve.csv` 是我们自己造的（`data/make_data.py`，`seed=42`），
故意在文件**末尾**追加了 6 行重复记录 —— 所以原始文件的时间戳是**倒着跳的**：

```
原始行数 17526 | 原始时间戳单调递增？ False
去重排序后 17520 行 | 单调递增 True
```

那 6 行重复的代价是什么？如果直接 `set_index("时间戳")`，索引里就会出现重复时刻，
后面的 `asfreq` 会直接报错，或者静默给出错误的时间轴。

**三关的顺序不能换**：

| 步骤 | 做什么 | 为什么必须在这个位置 |
|---|---|---|
| ① `drop_duplicates()` | 去掉完全重复的记录 | 放在 `set_index` 前，重复时刻才会一起消失 |
| ② `sort_values("时间戳")` | 让时间单调递增 | `asfreq` 要求索引单调，否则结果错乱 |
| ③ `asfreq("h")` | 对齐到整点，缺的补 NaN | 必须在①②之后；它是唯一能让你「数出缺了几小时」的操作 |

**`asfreq` 与 `resample` 的区别**（这是最容易混的一对）：

```python
s.asfreq("D")             # 只对齐日频：原来没有的那天 → NaN
s.resample("D").mean()    # 真聚合：把那天 24 小时压成一个均值
```

同一句话「按天重采样」，前者是**对齐**、后者是**聚合**。时序特征里要"补出缺失时刻"，
用的是 `asfreq`；要"把小时数据压成日数据"（比如做日频预测），才用 `resample`。
"""
    ),
    code(T1_CODE),
    md(
        """
### 1.1.1 难点深挖：三种时间戳格式，两种标准写法都失败

`load_curve_messy.csv` 是同一份台账的「脏快照」（336 行 = 14 天 × 24 h），
时间戳按行循环 3 种格式混排：

```
'2025-03-01 00:00:00'
'2025/03/01 01:00'
'2025年3月1日 2时'
```

**错误示范一 —— 直接 `pd.to_datetime`**：

```
ValueError: time data "2025/03/01 01:00" doesn't match format "%Y-%m-%d %H:%M:%S".
             You might want to try:
                 - passing format if your strings have a consistent format
```

第一个斜杠格式就把它顶回来了。

**错误示范二 —— 加 `format="mixed"`**（"混合格式"这个参数看起来正是为它准备的）：

```
DateParseError: Unknown datetime string format, unable to parse: 2025年3月1日 2时
```

**正确写法 —— 先把中文单位换成 ASCII 分隔符，再 `mixed`**：

```python
ts_text = (messy[" 时间戳 "]
           .str.replace("年", "-", regex=False).str.replace("月", "-", regex=False)
           .str.replace("日", " ", regex=False).str.replace("时", ":00", regex=False))
ts_dt = pd.to_datetime(ts_text, format="mixed")      # NaT = 0
```

**判定规则**：

| 情况 | 用什么 |
|---|---|
| 全列格式一致 | `pd.to_datetime(x, format="精确格式")`（最快，也最安全） |
| 多种 ASCII 格式混排 | `pd.to_datetime(x, format="mixed")` |
| 含中文单位 | **先字符串归一化**，再 `mixed`；`format` 参数救不了它 |

同一个 CSV 里还埋了另外两个字典型坑，一并处理：

- **列名两侧带空格**：`' 时间戳 '` —— 直接 `df["时间戳"]` 会 `KeyError`
- **数值列混文本**：`"--"` / `"N/A"` / `""` / `"1,234.5"`（千分位）/ `" 632.1 "`（前后空格）。
  `pd.to_numeric(..., errors="coerce")` 只解决前四个里的后三个字符串，
  **千分位逗号必须先用 `.str.replace(",", "")` 去掉** —— 否则 `"1,234.5"` 会被静默变成 NaN，
  你根本不知道丢了一个真值。实测：不去逗号 → NaN 10 个；去了逗号 → NaN 8 个。
"""
    ),
    code(T2_CODE),
    md(
        """
## 1.2 时间特征工程

`.dt` / `DatetimeIndex` 能派生出一整族特征，竞赛里最常用的就这几个：

| 特征 | 写法 | 说明 |
|---|---|---|
| 小时 | `index.hour` | 0~23，日周期的基本单位 |
| 星期几 | `index.dayofweek` | 0 = 周一，6 = 周日 |
| 月份 | `index.month` | 1~12，年季节 |
| 年积日 | `index.dayofyear` | 1~365，连续的年周期坐标 |
| 是否周末 | `(index.dayofweek >= 5).astype(int)` | 工业负荷周末腰斩 |
| 是否节假日 | `index.strftime("%Y-%m-%d").isin(HOLIDAYS)` | 需要一份节假日表 |

节假日表用**硬编码集合**：2025 年法定节假日 28 天 = **672 小时**（`HOLIDAY_RANGES`
已在脚手架里给好）。注意别把调休上班日算进去 —— 那是"周六但上班"，对工业负荷影响相反。

**注意**：原始 CSV 里 `是否节假日` / `是否周末` 这两列我们**故意没提供**，
让你从时间戳自己派生。理由是：任何能从时间戳算出来的东西都不该占数据列 ——
一旦写死进 CSV，换个年份就全错。
"""
    ),
    code(T3_CODE),
    md(
        """
### 1.2.1 难点深挖：周期整数直接当特征，23 点和 0 点就"离得最远"

**为什么难**：`hour` 天生是**环状**的（0 点的前一小时是 23 点），
但把它当一个普通整数交给模型，模型看到的是**一条直线上的 0~23**。
于是「23 点」和「0 点」在数值上差了 23 —— 全年最远的距离，而它们实际只差 1 小时。

**错误示范**：把 `hour` 直接当特征。

```
hour:      0   1   2  ...  22  23
距离(23,0) = |23 - 0| = 23          ← 全年最远
距离(0,12) = |0 - 12| = 12
```

对树模型，`<=23` 和 `<=0` 是两个完全不同的分裂方向，模型得靠大量样本才学得回来；
对线性模型，更糟 —— 它只能拟合出一条**直线**，而日周期是**双峰曲线**。

**正误对照**：把周期整数编码成单位圆上的 (sin, cos) 两列：

```python
feat["hour_sin"], feat["hour_cos"] = cyc(s_a.index.hour, 24)
```

| 对比 | 原始空间（按 hour） | 编码空间（弦长） |
|---|---|---|
| 23 点 vs 0 点 | **23** | **0.261052** |
| 0 点 vs 12 点 | 12 | 2.000000 |
| 理论值 | — | 23↔0：`2·sin(π/24) = 0.261052`；0↔12：`2·sin(π/2) = 2` |

编码后，「23 点与 0 点最近」这个事实**直接写进了数值里**，模型不用再学。

**判定规则**：

> **凡是"绕一圈会回到起点"的整数特征，一律 sin/cos 双列编码。**
> 小时（24）、星期（7）、月份（12）、风向（360°）、方位角都是这一类。
> 反例：`年积日`（0~365）本身**不环状**（12/31 与 1/1 之间确实隔着一年），
> 但如果你关心"冬天"这件事，可以另加 `doy` 的 sin/cos 两列 —— 两个特征不冲突，各管一段。
"""
    ),
    code(T4_CODE),
    md(
        """
## 1.3 滞后特征与滑动窗口

时序预测的核心思路：**把"过去"变成"特征列"**。

```python
w["y"]     = s.shift(-1)      # 目标：下一小时
w["lag1"]  = s.shift(1)       # 特征：上一小时
w["lag24"] = s.shift(24)      # 特征：昨天同一时刻
```

`shift(k)` 的符号规则：**正数往回看（过去的 k 步），负数往未来看（只有做目标时才用）**。

滑动窗口统计同样重要 —— 它能一次性概括"最近一段时间的水平和波动"：

```python
prev     = s.shift(1)                 # 关键：先把当前时刻错开
w["ma24"]  = prev.rolling(24).mean()  # 过去 24 小时均值
w["std24"] = prev.rolling(24).std()   # 过去 24 小时波动
```

**`prev` 那一行不能省**。如果写成 `s.rolling(24).mean()`，那么窗口 `[t-23, t]`
**包含 t 本身**；而你的目标是 `y = s[t+1]` —— 虽然 t 不是 y，但 `s[t]` 与 `s[t+1]`
高度相关（1 步自相关 0.87），等于给模型递了半个答案。这就是**"看起来没问题"的泄漏**。

### 1.3.1 难点深挖：`shift` 不分组，A 台区的负荷会喂给 B 台区

**为什么难**：`shift` 是**纯位置操作**，它根本不知道"台区"这回事。
一旦长表里多个台区首尾相接，每个台区的**第一行**都会取到**上一个台区的最后一行**。
它不报错、不警告，你甚至会觉得"不是有值吗，挺好"。

**错误示范 / 正误对照**（A 台区排在 B 前面）：

```python
d2["lag_bad"] = d2["负荷值"].shift(1)                              # 全局
d2["lag_ok"]  = d2.groupby("台区编号")["负荷值"].shift(1)          # 组内
```

实测：

| 位置 | 全局 shift | 组内 shift |
|---|---|---|
| B 台区首行 | **497.96** ← A 台区最后一行的负荷 | `NaN` ✓ |
| 总 NaN 数 | 150 | **151**（149 个原始缺失移位 + 2 个组首行） |

`497.96` 这个数**看起来完全正常**：B 台区的负荷也在几百的量级，模型不会觉得奇怪。
但它是一台**别的台区**在**另一天**的负荷。

**判定规则**：

> **只要表里存在"多个实体"（台区 / 线路 / 设备 / 站点 / 用户），
> 所有 `shift` / `rolling` / `diff` / `pct_change` 都必须先 `groupby` 再调用。**
> 判断方法很土但很有效：**数一数分组前后的 NaN 个数**。
> 组内做，NaN 数应该恰好等于「各组的首行数之和」（本例 = 2）。
"""
    ),
    code(T5_CODE),
    md(
        """
### 1.3.2 难点深挖：70 个缺失值，污染了 1009 行滑窗

**为什么难**：`rolling(n)` 的默认 `min_periods = n` ——
**窗口里只要有一个 NaN，整个结果就是 NaN**。所以缺失值不是"坏一个点"，
而是"坏它后面的一整段窗口"。

**错误示范 / 正误对照**：

```python
s_a.shift(1).rolling(24).mean().isna().sum()          # 未清洗 → 1009
s_a_clean.shift(1).rolling(24).mean().isna().sum()    # 清洗后 → 24
```

A 台区只有 **70** 个缺失点（6 段连续掉线 + 25 个散点 + 13 个低温失效 + 26 个原始空值），
但 `rolling(24)` 把它放大成了 **1009** 行不可用 —— 是缺失数目的 **14 倍**。

而且这只是"能不能算"的问题。更要命的是**统计量被污染**：

| 指标 | 未清洗 | 清洗后 |
|---|---|---|
| 1 步自相关 | **0.277454** | **0.868876** |
| lag1 ↔ y 相关 | 0.213682 | 0.653325 |

自相关从 0.87 掉到 0.28 —— 你完全看不出这是一条日周期极强的负荷曲线，
会误以为"这数据没规律"，从而去上更复杂的模型。

**判定规则**：

> **构造任何时序特征之前，先清洗，且清洗的阈值要"分组"**。
> 本例 `clean_series` 用的是**按小时分组的中位数 × 3**，而不是全局中位数：
> B 台区是「夜间 250 + 日间 850」的双峰分布，按小时分组后的中位数在 **276.87 ~ 934.21**
> 之间，相差 3.4 倍；用全局中位数做阈值会**误杀 382 个正常日间峰**。
"""
    ),
    code(T6_CODE),
    md(
        """
## 1.4 监督学习矩阵与切分

把前面的特征拼成矩阵，就得到可以训练的 `(X, y)`：

```python
FEAT = ["lag1", "lag2", "lag24", "ma24", "std24",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
model = w[FEAT + ["y"]].dropna()
```

**每一行 NaN 是怎么来的，要能一口说清**：

| 来源 | 吃掉几行 |
|---|---|
| `lag24` | 前 24 行（窗口还没攒够） |
| `y = shift(-1)` | 最后 1 行 |
| 合计 | **25 行** |

一个反直觉的发现：**`lag24` 与 `y` 的相关性（0.826982）比 `lag1`（0.653541）还高**。

道理不难：`y` 是 `t+1` 时刻，`lag1` 是 `t-1` —— 两者隔着 2 小时，正好跨在日周期的
陡坡上；而 `lag24` 是 `t-23`，与 `t+1` 恰好**隔了整 24 小时，日内相位完全对齐**，
只差一天的水平差异。所以：

> **"最近的过去"不等于"最有用的特征"。周期对齐（lag = 周期的整数倍）往往更有价值。**

### 1.4.1 难点深挖：三种泄漏，两种是真的

**泄漏** = 训练时用到了预测时拿不到的信息。它不会报错，只会让指标**虚高**，
然后在真实场景里原形毕露。下面实测三种，结论并不一样。

**① 未来窗口统计量进特征（真泄漏，杀伤力最大）**

```python
w["fut_ma24"] = s_a_clean.shift(-1).rolling(24).mean()   # "未来 24 小时平均负荷"
```

这个名字听起来平淡无奇，但它窗口覆盖 `[t+1, t+24]` —— **`y` 就是 `s[t+1]`，正在里面**。

| 方案 | 测试 MAE |
|---|---|
| 干净基线 | 26.797903 |
| 加 `fut_ma24` | **19.958735**（−6.839168） |

指标一下好了 25%。如果你在比赛里看到这个，应该**立刻回过味来**：
一个特征让 MAE 掉了四分之一，先怀疑泄漏，再高兴。

**② 随机切分（真泄漏）**

```python
rand_tr = model.sample(frac=0.8, random_state=42)     # 随机抽 80% 当训练集
rand_te = model.drop(rand_tr.index)
```

时序数据相邻样本高度相关（1 步自相关 0.87），随机切分等于**让训练集里躺着测试集的"邻居"**。

| 切分方式 | 测试 MAE |
|---|---|
| 时间切分（前 80% 训练） | 26.762002 |
| 随机切分 | **25.643746**（−1.118257） |

**③ 标准化参数用全量数据（在本数据上"失灵"）**

```python
sc_all = StandardScaler().fit(w[FEAT].dropna())       # 用到了测试段的均值/方差
```

教科书说这也是泄漏。但实测：

| 方案 | 测试 MAE |
|---|---|
| 训练段标准化 | 26.797903 |
| 全量标准化 | 26.797910（**+0.000007**） |

**几乎完全没动**。为什么？两个原因叠加：

1. **Ridge 对特征的线性缩放不敏感** —— 等比例放缩后，正则项的相对强度变化极小
2. **本数据分布平稳** —— 训练段 `lag1` 均值 454.65、全量 461.72，只差 1.5%

所以这个"泄漏"在这里**测不出效果**。这不是说它可以不管，而是说：

> **泄漏的严重程度取决于「泄漏信息有多大」和「模型有多敏感」。**
> 换成对尺度敏感的模型（KNN、SVM、神经网络）、或分布漂移明显的场景
> （新能源功率的装机扩容、负荷的年增长），第 ③ 种立刻会显形。
> **判断法则**：不管测出来影响多小，**标准化参数只能来自训练段** —— 因为它零成本。

| 泄漏类型 | 特征 | 判别口诀 |
|---|---|---|
| 未来窗口进特征 | 特征窗口的时间范围**跨越切分点、或覆盖 y** | 逐个特征问：**"我在预测时刻 t+1，这个数我算得出来吗？"** |
| 随机切分 | 用了 `train_test_split(shuffle=True)` / `sample()` | 时序**只有一种合法切法：按时间** |
| 全量统计量 | `fit(全量)` 而不是 `fit(训练段)` | 任何 `fit` 之前先问：这个对象见过测试集吗？ |
"""
    ),
    code(T7_CODE),
    code(T8_CODE),
    md(
        """
## 1.5 周期分解：手写一个简化版 STL

分解的经典三层模型：

```
原序列 = 低频项（趋势 + 年季节） + 日周期项 + 残差
```

手写步骤（这就是 STL 的核心思想，只是我们用固定窗口代替了它的迭代平滑）：

```python
trend  = s.rolling(24, center=True, min_periods=24).mean().bfill().ffill()   # ① 低频
detr   = s - trend                                                           # ② 去低频
season = detr.groupby(detr.index.hour).transform("mean")                     # ③ 日周期
resid  = detr - season                                                       # ④ 残差
```

三个细节：

- **`center=True`**：窗口跨在中心，两边各 12 小时。不加 `center` 的话，移动平均会引入
  **12 小时的相位滞后** —— 你会看到"趋势项总是晚半天"。
- **`min_periods=24` + `bfill().ffill()`**：首尾各 12 行凑不满窗口，先用最近的有效值补上，
  避免边缘出现 NaN 污染后续计算。
- **`transform("mean")` 而不是 `mean()`**：`transform` 会把 24 个分组的均值**广播回原长度**，
  这样 `detr - season` 才能逐元素相减。

实测（A 台区）：

```
重建最大绝对误差 1.137e-13            三层相加必然还原原序列
低频项 月均 peak/valley 524.8098 / 405.9264（月 1 / 5）
日周期项 peak/valley 121.8327 / -60.1567（hour 20 / 0）
残差 std 20.673831 | 残差 std / 序列均值 0.044923
```

注意"低频项"里其实**包含年季节**（1 月最高、5 月最低）—— 24 小时移动平均消得掉日周期，
消不掉慢变的季节。这是符合预期的：我们只把"快"的日周期单独拆出来。

### 1.5.1 难点深挖：加法还是对数？用"残差方差稳不稳"判断

**为什么难**：绝大多数教科书直接默认加法分解（`y = T + S + R`）。
但真实负荷是**乘性**的 —— 冬季高负荷时，1 小时内的波动幅度也更大。
这时加法分解会把"波动随水平增长"这个结构**塞进残差**，残差就不再是同方差噪声。

**判别方法（不需要知道数据怎么生成的）**：把残差按**负荷水平**切成两段，比较两段的标准差。

```python
add_ratio = resid[s >= s.quantile(0.5)].std() / resid[s < s.quantile(0.5)].std()
log_ratio = l_resid[ls >= ls.quantile(0.5)].std() / l_resid[ls < ls.quantile(0.5)].std()
```

**正误对照**：

| 分解方式 | 残差 std | 高负荷段 / 低负荷段 残差 std 比 |
|---|---|---|
| 加法（原尺度） | 20.673831 | **1.171187** ← 高段波动大了 17% |
| 对数（`log` 后分解） | 0.043815 | **1.011203** ← 两段基本持平 ✓ |

对数分解后，比值从 1.171 掉到 1.011 —— **残差方差不再随负荷水平变化**，
这才符合同方差假设，后续的 MAE / 置信区间才有意义。

**判定规则**：

> **先做加法分解，再算「高/低负荷段残差 std 比」。**
> 比值明显 > 1.1 → 数据是乘性的 → **取对数再分解**（并把预测结果 `exp` 回去）。
> 比值在 1.0 附近 → 加法就够。
>
> 顺带一个工程收益：对数域上误差是**相对误差**，天然贴合 MAPE 这类竞赛常用指标。

### 1.5.2 难点深挖：残差 ACF 是"平坦"的，说明还有东西没拆出来

分解完就完事了吗？看残差的自相关。

| lag | 1 | 24 | 168 |
|---|---|---|---|
| 加法残差 ACF | +0.4416 | +0.4044 | +0.4069 |
| 对数残差 ACF | +0.4431 | +0.3919 | +0.4032 |

**白噪声的 ACF 应该在 lag≥2 后迅速掉到 0 附近**。这里却是一整条 **0.4 的平线** ——
不随 lag 衰减，说明残差里还有**慢变结构**没被拆掉。

诊断方向：固定不动的日周期项，能不能描述"冬天峰更尖、夏天峰更平"这种事？
把去低频后的序列按 `(月, 小时)` 压成 288 格，再算每个月内 24 小时的**幅度**：

```
日周期幅度按月（对数域）:
{1: 0.3496, 2: 0.3492, 3: 0.3660, 4: 0.3728, 5: 0.3798, 6: 0.4302,
 7: 0.4567, 8: 0.4448, 9: 0.3928, 10: 0.3501, 11: 0.3563, 12: 0.3602}
```

**7 月 0.4567 vs 1 月 0.3496，差了 31%** —— 日周期项不是"一个固定形状"，
它的形状本身随季节在变。用一个全局固定的 24 小时形状去减，必然减不干净。

修法很直接：把季节项从「按 hour」改成「按 (月, 小时)」分组：

| 季节项分组 | 残差 std | ACF(1) | ACF(24) |
|---|---|---|---|
| 按 `hour` | 0.043815 | +0.4431 | +0.3919 |
| 按 `(月, hour)` | **0.034551** | **+0.1513** | **+0.0159** |

lag24 的自相关从 0.39 掉到 **0.016** —— 残差基本变白了，残差 std 也降了 21%。

**判定规则**：

> **分解完必须看残差 ACF。**
> - ACF 快速衰减到 0 → 残差是白噪声，分解到位
> - ACF 在 lag = 周期的整数倍处出峰 → 还有周期没拆干净（加一层周周期）
> - ACF **平坦在高位不衰减** → 有慢变结构，优先怀疑「周期幅度受季节调制」，
>   把季节项从"单变量分组"升级为"(季节, 周期) 双变量分组"
>
> 代价提示：`(月, hour)` 是 288 个格子，每格约 30 个样本，还撑得住；
> 如果换成 `(日, hour)` 就是 8760 格、每格 1 个样本 —— **过拟合的边界要自己看住**。
"""
    ),
    code(T9_CODE),
    code(T10_CODE),
    code(T11_CODE),
    md(
        """
## 1.6 多序列对齐与外生变量

真实赛题常常给多个台区 / 多个电站。用 `pivot` 拉成宽表，让它们在**同一时间轴上对齐**：

```python
wide = df.pivot(index="时间戳", columns="台区编号", values="负荷值")   # (8760, 2)
```

宽表的好处是：**两列的时间戳天然一一对应**，可以直接做相关、做比值、做差分。
注意 `pivot` 要求 `(index, columns)` 组合唯一 —— 所以之前的去重是前提。

### 两个台区，完全不相关

| 指标 | 值 | 说明 |
|---|---|---|
| A-B 同时刻相关 | 0.056371 | 几乎无关 |
| A-B **日周期形状**相关 | **−0.040653** | **负相关**：居民晚高峰 vs 工业日间平台 |
| A-B 一阶差分相关 | −0.309388 | 波动方向也相反 |

**这是很实用的一条**：把"日周期形状"单独拎出来比
（两列都先清洗，再 `pd.DataFrame({"A": ..., "B": ...}).groupby(index.hour).mean()` 压成 24×2），
比直接看原始序列的相关性更能说明问题。两个台区负荷水平差不多（均值 460 vs 475），
但**形状相反** —— 如果只报"相关系数 0.06"，你既不知道它们无关，也不知道它们为什么无关。

### 外生变量：温度与负荷是 V 形，不是线性

`温度` 是电力负荷最经典的外生变量。A 台区（居民型）：

| 度量 | 相关系数 |
|---|---|
| 温度 ↔ 负荷（线性） | −0.188929 |
| **&#124;T − 22&#124; ↔ 负荷** | **0.406437** |

分箱看更清楚（`pd.cut`）：

| 温度区间 | (−10, 0] | (0, 10] | (10, 20] | (20, 26] | (26, 32] | (32, 40] |
|---|---|---|---|---|---|---|
| 平均负荷 | 489.3 | 499.0 | 442.7 | **422.2** | 481.5 | 507.6 |

**两头高、中间低** —— 这就是 V 形（准确说是 U 形）：天冷要取暖、天热要制冷，
20~26 ℃ 最省电。

这正是**"线性相关系数会骗人"的典型案例**：

> **物理上先想清楚关系形状，再选特征形式。**
> 温度—负荷用 `|T − 22|`（或直接给出 `T` 和 `T²` 两列）比只给 `T` 强得多。
> 相关度从 0.19 涨到 0.41，只需要把原始温度**换成距离舒适区的距离**。
"""
    ),
    code(T12_CODE),
    md(
        """
## 易错点清单

1. **`drop_duplicates` 一定要在 `set_index("时间戳")` 之前**。之后再做，重复时刻已经污染了索引。
2. **`asfreq` ≠ `resample`**。前者对齐补 NaN，后者真聚合。想"数出缺失了几小时"只能用前者。
3. **`pd.to_datetime` 遇到中文日期会直接失败**，`format="mixed"` 也救不了 —— 先做字符串归一化。
4. **`to_numeric(errors="coerce")` 不认千分位逗号**，`"1,234.5"` 会被静默变成 NaN。先去逗号。
5. **周期整数（小时 / 星期 / 月份）必须 sin/cos 编码**，否则 23 点与 0 点成了一根线上的两端。
6. **`shift` / `rolling` / `diff` 不分组 = 跨实体污染**。判断方法：数分组前后的 NaN 个数。
7. **`rolling(n)` 默认 `min_periods=n`**，一个 NaN 毁掉整个窗口。70 个缺失 → 1009 行不可用。
8. **构造滑窗特征前先 `shift(1)`**，否则窗口含当前时刻，而当前时刻与目标高度相关。
9. **异常值检测的阈值要分组**。双峰分布（工业负荷）用全局中位数会误杀整段日间峰（实测 382 个）。
10. **先清洗再算自相关**。不清洗时 1 步自相关 0.277，清洗后 0.869 —— 差了一个量级。
11. **`lag24` 可能比 `lag1` 更有用**，因为周期对齐比"离得近"更重要。
12. **时序只能按时间切分**。随机切分让 MAE 虚降 1.12，且完全不报错。
13. **特征窗口绝不能覆盖 `y` 所在区间**。`shift(-1).rolling(24)` 这类写法必须一眼识破。
14. **标准化参数只能来自训练段**。哪怕本数据上测不出差别（+0.000007），也要照做。
15. **`center=True` 的移动平均没有相位滞后**；不加的话趋势项会晚半天。
16. **分解完要看残差 ACF**。平坦不衰减 = 有慢变结构；在周期整数倍处出峰 = 周期没拆净。
17. **乘性数据要取对数**。判据：高/低负荷段的残差 std 比 > 1.1。
18. **温度对负荷是 V 形**。线性相关系数 −0.19，换成 `|T−22|` 就是 +0.41。

## 本章小结

**一句话**：时序建模的质量，八成取决于"进模型之前"这段流水线。

**四步流水线**：

```
① 规整时间轴   drop_duplicates → sort_values → asfreq("h")
② 清洗+分组阈值 按小时分组的中位数×3 判异常 → interpolate
③ 造特征       日历(sin/cos) + 滞后(组内 shift) + 滑窗(先 shift 再 rolling)
④ 切分+分解    按时间 8:2 切分 → 三层分解 → 残差 ACF 验收
```

**三个"必须背下来"的判定规则**：

| 场景 | 规则 |
|---|---|
| 特征是不是泄漏了 | 问："我在预测 `t+1`，这个数**算得出来**吗？" |
| 该加法还是对数分解 | 高/低负荷段残差 std 比 > 1.1 → 取对数 |
| 分解到位没有 | 残差 ACF 快速衰减到 0；平坦 = 还没拆完 |

**下一步**：本章拿到了干净的 `(X, y)` 矩阵。`ch02` 会把视角转向**统计方法** ——
ADF 平稳性检验、差分、ACF/PACF 定阶、ARIMA；`ch03` 开始上手写 RNN。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch01 时序数据与特征工程（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`data/load_curve.csv`（2 台区 × 8760 h）+ `data/load_curve_messy.csv`（336 行脏快照）。
>
> **每题都跟了 `# 提示：`**，照着提示能独立做完。
> 如果 assert 的数字对不上，**先回头看上一步的输出**，别急着改 assert —— 那些数字都是实跑出来的。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 规则时间轴（去重 → 排序 → 对齐） | 2 | §1.1 |
| 2 | 混合时间戳 + 脏数值列 | 2 | §1.1.1 |
| 3 | 时间特征 + 节假日标记 | 2 | §1.2 |
| 4 | sin/cos 周期编码 | 2 | §1.2.1 |
| 5 | 组内 / 全局 `shift` | 2 | §1.3.1 |
| 6 | 清洗与 `rolling` 的 NaN 传播 | 2 | §1.3.2 |
| 7 | 监督矩阵组装 | 2 | §1.4 |
| 8 | 三种泄漏实测 | 4 | §1.4.1 |
| 9 | 三层周期分解 | 2 | §1.5 |
| 10 | 加法 vs 对数 | 3 | §1.5.1 |
| 11 | 残差 ACF 诊断 | 3 | §1.5.2 |
| 12 | 多序列对齐 + 温度 V 形 | 3 | §1.6 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md("## 任务 1：把 CSV 变成规则时间轴（§1.1）"),
    code(T1_CODE),
    md("## 任务 2：解析混合格式时间戳 + 脏数值列（§1.1.1）"),
    code(T2_CODE),
    md("## 任务 3：派生时间特征与节假日标记（§1.2）"),
    code(T3_CODE),
    md("## 任务 4：周期整数的 sin/cos 编码（§1.2.1）"),
    code(T4_CODE),
    md("## 任务 5：组内 shift 与全局 shift 的差别（§1.3.1）"),
    code(T5_CODE),
    md("## 任务 6：清洗，以及 rolling 的 NaN 传播（§1.3.2）"),
    code(T6_CODE),
    md("## 任务 7：组装监督学习矩阵（§1.4）"),
    code(T7_CODE),
    md("## 任务 8：三种泄漏实测（§1.4.1）"),
    code(T8_CODE),
    md("## 任务 9：手写三层周期分解（§1.5）"),
    code(T9_CODE),
    md("## 任务 10：加法分解还是对数分解（§1.5.1）"),
    code(T10_CODE),
    md("## 任务 11：用残差 ACF 诊断分解是否到位（§1.5.2）"),
    code(T11_CODE),
    md("## 任务 12：多序列对齐 + 温度的非线性（§1.6）"),
    code(T12_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 12 个代码块的 assert 全部通过
- [ ] `drop_duplicates` 必须在哪一步之前？不这样做会怎样？
- [ ] `asfreq` 与 `resample` 的区别是什么？想"数出缺了几小时"该用哪个？
- [ ] 为什么 `hour` 要做 sin/cos 编码？23 点与 0 点编码后的距离是多少？
- [ ] B 台区首行用全局 `shift` 会取到哪个值？它为什么"看起来很合理"？
- [ ] 70 个缺失为什么会让 **1009** 行 `rolling(24)` 结果变成 NaN？
- [ ] 清洗异常值为什么不能直接用全局中位数？B 台区会被误杀多少个点？
- [ ] 三种泄漏分别是什么？哪一种在本数据上**测不出效果**，为什么？
- [ ] 怎么判断该用加法还是对数分解？给出那条可背的判据。
- [ ] 残差 ACF 平坦意味着什么？怎么修？
- [ ] 为什么温度要用 `|T − 22|` 而不是 `T`？
- [ ] `lag24` 为什么比 `lag1` 与 `y` 更相关？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 原始行数 / 去重后行数 | 17526 / 17520 |
| 完全重复行 | 6 |
| 原始时间戳单调递增？ | False → 排序后 True |
| A / B 台区缺失数 | 70 / 79 |
| 节假日小时 / 周末小时 | 672 / 2496 |
| 23↔0 编码弦长 | 0.261052 |
| 组内 shift 的 NaN 数 | 151 |
| B 台区首行被污染的值 | 497.96 |
| 1 步自相关：未清洗 / 清洗后 | 0.277454 / 0.868876 |
| `rolling(24)` NaN：未清洗 / 清洗后 | 1009 / 24 |
| 监督矩阵 dropna 损失 | 25 |
| `lag1` / `lag24` 与 `y` 相关 | 0.653541 / 0.826982 |
| 基线测试 MAE | 26.797903 |
| 加 `fut_ma24` 后 MAE | 19.958735 |
| 随机切分 / 时间切分 MAE | 25.643746 / 26.762002 |
| 全量标准化 MAE | 26.797910 |
| 日周期项 peak / valley | 121.8327 / −60.1567 |
| 加法高/低段残差 std 比 | 1.171187 |
| 对数高/低段残差 std 比 | 1.011203 |
| 7 月 / 1 月 日周期幅度 | 0.4567 / 0.3496 |
| `(月, 小时)` 分组后残差 std | 0.034551 |
| A-B 日周期形状相关 | −0.040653 |
| 温度线性 / V 形相关 | −0.188929 / 0.406437 |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
