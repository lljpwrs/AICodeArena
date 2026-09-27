#!/usr/bin/env python3
"""生成 05_timeseries 用的合成时序数据。固定 seed=42，断网可跑、幂等。

产出（`coding/05_timeseries/data/`）：

| 文件 | 内容 | 行数 |
|---|---|---|
| `load_curve.csv` | 台区小时负荷 + 温度（2 台区 × 全年 8760 h） | 17526（含 6 行重复） |
| `pv_power.csv` | 光伏电站小时功率 + 辐照度 + 天气类型 | 8763（含 3 行重复） |
| `load_curve_messy.csv` | 「脏」负荷快照（时间戳混 3 种格式 / 数值列混文本） | 336 |

设计原则（见 `docs/合成数据规范.md`）：

- **编号纯虚构**：`STATION_A_01` / `STATION_B_02` / `PV_STATION_P_01`，
  不得出现任何看起来像真实台区 / 线路 / 站点的命名
- 负荷用**乘法模型**：`基线 × 日周期 × 年季节 × 周周期 × 节假日 × 温度效应 × 趋势 × (1+噪声)`
  —— 乘性结构是真实的，也正好给 ch01「加法分解 vs 对数分解」留出对照
- 两条曲线**故意做成不同形态**：`STATION_A_01` 居民型（晚高峰 + 早次峰），
  `STATION_B_02` 工业型（日间平台 + 周末腰斩），跨台区对比才有意义
- **故意埋脏**：连续缺失段、MCAR 散点缺失、MAR 低温缺失、异常尖峰、
  负值、哨兵 `9999`、完全重复行（追加在文件**末尾**，破坏时间有序性）
- 每个 CSV < 1 MB，直接提交仓库

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python coding/05_timeseries/data/make_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

YEAR = 2025
N_HOURS = 8760  # 2025 年非闰年

# 2025 年法定节假日（不含调休上班日）。ch01 里会再硬编码一份，保证 notebook 自包含。
HOLIDAY_RANGES = [
    ("2025-01-01", "2025-01-01"),  # 元旦
    ("2025-01-28", "2025-02-04"),  # 春节
    ("2025-04-04", "2025-04-06"),  # 清明
    ("2025-05-01", "2025-05-05"),  # 劳动节
    ("2025-05-31", "2025-06-02"),  # 端午
    ("2025-10-01", "2025-10-08"),  # 中秋 + 国庆
]

# base=基线负荷(kW)；weekend/holiday=乘性系数；heat/cool=温升/温降的负荷弹性
STATIONS = {
    "STATION_A_01": dict(
        label="居民型", kind="residential", base=520.0,
        weekend=1.06, holiday=0.82, heat=0.011, cool=0.017,
    ),
    "STATION_B_02": dict(
        label="工业型", kind="industrial", base=880.0,
        weekend=0.62, holiday=0.45, heat=0.005, cool=0.000,
    ),
}

PV_NAME = "PV_STATION_P_01"
PV_CAP = 1200.0  # 装机容量 kW

WEATHER_STATES = ["晴", "多云", "阴", "雨"]
WEATHER_TRANS = {
    "晴": [0.62, 0.25, 0.09, 0.04],
    "多云": [0.32, 0.38, 0.20, 0.10],
    "阴": [0.18, 0.32, 0.34, 0.16],
    "雨": [0.22, 0.30, 0.28, 0.20],
}
CLOUD_FACTOR = {"晴": 1.00, "多云": 0.66, "阴": 0.30, "雨": 0.14}


# --------------------------------------------------------------------------- #
# 物理 / 统计成分
# --------------------------------------------------------------------------- #


def _circ_dist(x: np.ndarray, center: float, period: float) -> np.ndarray:
    """环形距离（处理跨年：12-31 与 01-01 只差 1 天）。"""
    raw = np.abs(np.asarray(x, dtype=float) - center)
    return np.minimum(raw, period - raw)


def daily_shape(hour: np.ndarray, kind: str) -> np.ndarray:
    """日内 24 h 形状。resident=晚高峰+早次峰；industrial=日间超高斯平台。"""
    h = np.asarray(hour, dtype=float)
    if kind == "residential":
        return (
            0.66
            + 0.18 * np.exp(-(((h - 8.0) / 2.2) ** 2))
            + 0.10 * np.exp(-(((h - 13.0) / 3.0) ** 2))
            + 0.34 * np.exp(-(((h - 19.5) / 2.4) ** 2))
        )
    return 0.30 + 0.72 * np.exp(-(((h - 13.0) / 5.0) ** 4))


def seasonal_factor(doy: np.ndarray) -> np.ndarray:
    """年季节：冬峰（采暖）+ 夏峰（制冷）双峰，春秋回落到 1.0。"""
    d = np.asarray(doy, dtype=float)
    dd = _circ_dist(d, 20.0, 365.0)
    ds = np.abs(d - 205.0)
    return 1.0 + 0.155 * np.exp(-((dd / 40.0) ** 2)) + 0.115 * np.exp(-((ds / 34.0) ** 2))


def temperature_series(doy: np.ndarray, hour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """气温：年周期（7 月最高）+ 日周期（15 时最高）+ 噪声。"""
    d = np.asarray(doy, dtype=float)
    h = np.asarray(hour, dtype=float)
    return (
        16.5
        + 11.0 * np.cos(2 * np.pi * (d - 200.0) / 365.0)
        + 4.8 * np.sin(2 * np.pi * (h - 9.0) / 24.0)
        + rng.normal(0.0, 1.8, len(d))
    )


def holiday_mask(stamps: pd.DatetimeIndex) -> np.ndarray:
    days = stamps.normalize()
    flag = np.zeros(len(stamps), dtype=bool)
    for lo, hi in HOLIDAY_RANGES:
        flag |= (days >= pd.Timestamp(lo)) & (days <= pd.Timestamp(hi))
    return flag


def _pick_positions(rng, k, lo, hi, blocked):
    out = []
    while len(out) < k:
        p = int(rng.integers(lo, hi))
        if p not in blocked:
            out.append(p)
            blocked.add(p)
    return out


# --------------------------------------------------------------------------- #
# 主曲线
# --------------------------------------------------------------------------- #


def build_load_curve(rng: np.random.Generator) -> pd.DataFrame:
    """台区小时负荷：乘法模型合成 + 4 类脏数据注入。"""
    stamps = pd.date_range(f"{YEAR}-01-01 00:00", periods=N_HOURS, freq="h")
    doy = stamps.dayofyear.to_numpy()
    hour = stamps.hour.to_numpy()
    dow = stamps.dayofweek.to_numpy()
    is_holiday = holiday_mask(stamps)
    temp = temperature_series(doy, hour, rng)
    trend = 1.0 + 0.035 * np.arange(N_HOURS) / (N_HOURS - 1)

    frames = []
    for name, cfg in STATIONS.items():
        shape = daily_shape(hour, cfg["kind"])
        if cfg["kind"] == "residential":
            temp_eff = (
                1.0
                + cfg["heat"] * np.maximum(0.0, 20.0 - temp)
                + cfg["cool"] * np.maximum(0.0, temp - 26.0)
            )
        else:
            temp_eff = 1.0 + cfg["heat"] * np.maximum(0.0, 22.0 - temp)
        weekly = np.where(dow >= 5, cfg["weekend"], 1.0)
        hol = np.where(is_holiday, cfg["holiday"], 1.0)
        noise = rng.normal(0.0, 0.026, N_HOURS)
        load = cfg["base"] * shape * seasonal_factor(doy) * weekly * hol * temp_eff * trend
        load = load * (1.0 + noise)
        frames.append(
            pd.DataFrame(
                {
                    "时间戳": stamps,
                    "台区编号": name,
                    "负荷值": np.round(load, 2),
                    "温度": np.round(temp, 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def inject_load_dirty(curve: pd.DataFrame, rng: np.random.Generator) -> dict:
    """注入缺失 / 异常 / 哨兵 / 重复。返回真值统计（供 README 与 assert 使用）。"""
    n_before = len(curve)
    stat = {"segments": {}, "mcar": {}, "mar": {}, "spike": {}, "neg": {}, "sentinel": 0}

    for k, name in enumerate(STATIONS):
        lo, hi = k * N_HOURS, (k + 1) * N_HOURS
        blocked: set[int] = set()

        # ① 连续缺失段 6 段，长度 2~9 h（采集终端掉线）
        segs = []
        for _ in range(6):
            start = int(rng.integers(lo, hi - 10))
            length = int(rng.integers(2, 10))
            block = set(range(start, start + length))
            if block & blocked:
                continue
            blocked |= block
            segs.append((start, length))
        stat["segments"][name] = segs

        # ② MCAR 散点缺失 25 个
        mcar = _pick_positions(rng, 25, lo, hi, blocked)
        stat["mcar"][name] = len(mcar)

        # ③ MAR：低温（< 1.0 ℃）时段额外 10% 缺失（传感器低温失效）
        cold = [p for p in range(lo, hi) if curve.at[p, "温度"] < 1.0 and p not in blocked]
        mar = [p for p in cold if rng.random() < 0.10]
        blocked |= set(mar)
        stat["mar"][name] = len(mar)

        null_idx = sorted(blocked)
        curve.loc[null_idx, "负荷值"] = np.nan

        # ④ 异常尖峰 4 个（3.2~4.8 倍）
        spike = _pick_positions(rng, 4, lo, hi, blocked)
        for p in spike:
            curve.at[p, "负荷值"] = np.round(float(curve.at[p, "负荷值"]) * rng.uniform(3.2, 4.8), 2)
        stat["spike"][name] = len(spike)

        # ⑤ 负值 2 个（传感器反向）
        neg = _pick_positions(rng, 2, lo, hi, blocked)
        for p in neg:
            curve.at[p, "负荷值"] = np.round(-0.15 * float(curve.at[p, "负荷值"]), 2)
        stat["neg"][name] = len(neg)

        # ⑥ 哨兵 9999 一个
        sent = _pick_positions(rng, 1, lo, hi, blocked)[0]
        curve.at[sent, "负荷值"] = 9999.0
        stat["sentinel"] += 1

        # ⑦ 温度缺失 15 个
        tmiss = _pick_positions(rng, 15, lo, hi, set())
        curve.loc[tmiss, "温度"] = np.nan

    # ⑧ 完全重复行 6 行，追加到**文件末尾**（故意破坏时间有序性）
    valid = curve.index[curve["负荷值"].notna()].to_numpy()
    dup_idx = rng.choice(valid, size=6, replace=False)
    curve = pd.concat([curve, curve.loc[dup_idx]], ignore_index=True)

    stat["rows_before_dup"] = n_before
    stat["dup_rows"] = 6
    stat["rows_after_dup"] = len(curve)
    return curve, stat


def _markov_weather(rng: np.random.Generator, days: int) -> list[str]:
    out, cur = [], "多云"
    for _ in range(days):
        cur = str(rng.choice(WEATHER_STATES, p=WEATHER_TRANS[cur]))
        out.append(cur)
    return out


def build_pv_power(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """光伏电站小时功率：晴空辐照 → 理论功率；云衰减 → 实际功率。"""
    stamps = pd.date_range(f"{YEAR}-01-01 00:00", periods=N_HOURS, freq="h")
    doy = stamps.dayofyear.to_numpy()
    hour = stamps.hour.to_numpy()
    temp = temperature_series(doy, hour, rng)

    weather_days = _markov_weather(rng, 365)
    weather = np.array([weather_days[int(d) - 1] for d in doy])
    base_cloud = np.array([CLOUD_FACTOR[w] for w in weather], dtype=float)
    cloud = np.clip(base_cloud * rng.normal(1.0, 0.05, N_HOURS), 0.03, 1.0)

    solar_season = 1.0 + 0.18 * np.cos(2 * np.pi * (doy - 172.0) / 365.0)
    sun_angle = np.sin(np.pi * (hour - 6.6) / 11.6)
    # 845 × 1.18（夏至季节因子）= 997 W/m²，压在辐照度常数 1000 以下，
    # 同时让「理论功率」峰值 1196 kW 略低于装机 1200 kW，避免出现被 clip 的平台
    clear_irr = np.maximum(0.0, 845.0 * sun_angle) * solar_season
    irr = clear_irr * cloud

    tcell = temp + 0.028 * irr
    p_theo = np.minimum(PV_CAP * clear_irr / 1000.0, PV_CAP)
    p_real = PV_CAP * irr / 1000.0 * (1.0 - 0.0038 * (tcell - 25.0))
    p_real = p_real + rng.normal(0.0, 7.0, N_HOURS)
    p_real = np.where(clear_irr <= 0.0, 0.0, np.maximum(0.0, p_real))

    df = pd.DataFrame(
        {
            "时间戳": stamps,
            "电站编号": PV_NAME,
            "实际功率": np.round(p_real, 2),
            "理论功率": np.round(p_theo, 2),
            "辐照度": np.round(irr, 1),
            "天气类型": weather,
        }
    )

    stat = {}
    night = np.flatnonzero(clear_irr <= 0.0)
    # ① 夜间辐照度非零（传感器零点漂移）8 个
    drift = rng.choice(night, size=8, replace=False)
    df.loc[drift, "辐照度"] = np.round(rng.uniform(5.0, 40.0, 8), 1)

    # ② 实际功率负值 5 个（逆变器异常）
    day = np.flatnonzero((clear_irr > 0.0) & (p_real > 50.0))
    negp = rng.choice(day, size=5, replace=False)
    df.loc[negp, "实际功率"] = np.round(-rng.uniform(5.0, 25.0, 5), 2)

    # ③ 缺失：连续 3 段（4~12 h）+ 散点 20 个
    miss: set[int] = set()
    segs = []
    while len(segs) < 3:
        start = int(rng.integers(0, N_HOURS - 12))
        length = int(rng.integers(4, 13))
        block = set(range(start, start + length))
        if block & miss:
            continue
        miss |= block
        segs.append((start, length))
    spot = _pick_positions(rng, 20, 0, N_HOURS, miss)
    miss |= set(spot)
    df.loc[sorted(miss), "实际功率"] = np.nan

    # ④ 完全重复 3 行，追加末尾
    valid = df.index[df["实际功率"].notna()].to_numpy()
    dup = rng.choice(valid, size=3, replace=False)
    df = pd.concat([df, df.loc[dup]], ignore_index=True)

    stat["night_drift"] = 8
    stat["neg_power"] = 5
    stat["segments"] = segs
    stat["missing"] = len(miss)
    stat["dup_rows"] = 3
    stat["rows"] = len(df)
    return df, stat


def build_messy(rng: np.random.Generator) -> pd.DataFrame:
    """14 天 × 24 h「脏」快照：时间戳混 3 种格式 + 负荷列混文本。"""
    days = 14
    stamps = pd.date_range("2025-03-01 00:00", periods=days * 24, freq="h")
    hour = stamps.hour.to_numpy()
    doy = stamps.dayofyear.to_numpy()
    load = np.round(
        520.0 * daily_shape(hour, "residential") * seasonal_factor(doy), 2
    )
    temp = np.round(temperature_series(doy, hour, rng), 1)

    def fmt(ts: pd.Timestamp, k: int) -> str:
        if k % 3 == 0:
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        if k % 3 == 1:
            return ts.strftime("%Y/%m/%d %H:%M")
        return "%d年%d月%d日 %d时" % (ts.year, ts.month, ts.day, ts.hour)

    ts_txt = [fmt(ts, k) for k, ts in enumerate(stamps)]
    load_txt = ["%.2f" % v for v in load]

    # 15 个互不重叠的位置：前 12 个放「文本脏」，后 3 个放哨兵 9999
    dirty = _pick_positions(rng, 15, 0, len(stamps), set())
    bad = ["--", "N/A", "", "1,234.5", " 632.1 "]
    for i, p in enumerate(dirty[:12]):
        load_txt[int(p)] = bad[i % len(bad)]
    for p in dirty[12:]:
        load_txt[int(p)] = "9999"

    return pd.DataFrame(
        {
            " 时间戳 ": ts_txt,
            "台区编号": "STATION_A_01",
            "负荷值(kW)": load_txt,
            "温度(℃)": ["%.1f" % v for v in temp],
        }
    )


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def main() -> None:
    rng = np.random.default_rng(SEED)

    curve, lstat = inject_load_dirty(build_load_curve(rng), rng)
    pv, pstat = build_pv_power(rng)
    messy = build_messy(rng)

    curve.to_csv(OUT_DIR / "load_curve.csv", index=False, encoding="utf-8-sig")
    pv.to_csv(OUT_DIR / "pv_power.csv", index=False, encoding="utf-8-sig")
    messy.to_csv(OUT_DIR / "load_curve_messy.csv", index=False, encoding="utf-8-sig")

    clean = curve.drop_duplicates().reset_index(drop=True)
    a = clean[clean["台区编号"] == "STATION_A_01"].reset_index(drop=True)
    b = clean[clean["台区编号"] == "STATION_B_02"].reset_index(drop=True)
    raw = clean.dropna(subset=["负荷值"])
    med = float(raw["负荷值"].median())
    thr = 3.0 * med
    ok = raw[raw["负荷值"].between(0.0, thr)]
    a_ok = ok[ok["台区编号"] == "STATION_A_01"]
    b_ok = ok[ok["台区编号"] == "STATION_B_02"]
    neg_n = int((raw["负荷值"] < 0).sum())
    sent_n = int((raw["负荷值"] == 9999.0).sum())
    spike_n = len(raw) - len(ok) - neg_n - sent_n

    print("=" * 72)
    print("load_curve.csv")
    print("-" * 72)
    print(f"  行数（含重复）      : {lstat['rows_after_dup']}")
    print(f"  完全重复行          : {int(curve.duplicated().sum())}")
    print(f"  去重后行数          : {len(clean)}")
    print(f"  台区数              : {clean['台区编号'].nunique()}  {sorted(clean['台区编号'].unique())}")
    print(f"  时间范围            : {clean['时间戳'].min()} ~ {clean['时间戳'].max()}")
    print(f"  每台区小时数        : {len(a)} / {len(b)}")
    print(f"  负荷值缺失 A/B      : {int(a['负荷值'].isna().sum())} / {int(b['负荷值'].isna().sum())}")
    print(f"  温度缺失 A/B        : {int(a['温度'].isna().sum())} / {int(b['温度'].isna().sum())}")
    for name in STATIONS:
        segs = lstat["segments"][name]
        lens = [s[1] for s in segs]
        print(f"  {name} 连续缺失段 : {len(segs)} 段，长度 {lens}，最长 {max(lens)} h，共 {sum(lens)} h")
        print(f"  {name} MCAR/MAR/尖峰/负值/哨兵 : {lstat['mcar'][name]} / "
              f"{lstat['mar'][name]} / {lstat['spike'][name]} / {lstat['neg'][name]} / 1")
    print(f"  哨兵 9999 合计      : {int((clean['负荷值'] == 9999.0).sum())}")
    print(f"  负值合计            : {int((clean['负荷值'] < 0).sum())}")
    print(f"  异常阈值（3×中位数）: {thr:.4f}")
    print(f"  阈值外异常数        : {len(raw) - len(ok)}  "
          f"= 尖峰 {spike_n} + 负值 {neg_n} + 哨兵 {sent_n}")
    print("     ↑ 注入的尖峰是 8 个：全局阈值只看「绝对高度」，")
    print("       低负荷时段被放大的尖峰（如凌晨 400×3.2=1280 < 1306）会漏检")
    print(f"  剔除异常后 min/max  : {ok['负荷值'].min():.2f} / {ok['负荷值'].max():.2f}")
    print(f"  全量 中位数/均值    : {med:.4f} / {raw['负荷值'].mean():.4f}")
    print(f"  A 台区 均值/std     : {a_ok['负荷值'].mean():.4f} / {a_ok['负荷值'].std():.4f}")
    print(f"  B 台区 均值/std     : {b_ok['负荷值'].mean():.4f} / {b_ok['负荷值'].std():.4f}")
    wk = a_ok.assign(wd=a_ok["时间戳"].dt.dayofweek)
    print(f"  A 工作日/周末均值   : {wk.loc[wk['wd'] < 5, '负荷值'].mean():.4f} / "
          f"{wk.loc[wk['wd'] >= 5, '负荷值'].mean():.4f}")
    wk = b_ok.assign(wd=b_ok["时间戳"].dt.dayofweek)
    print(f"  B 工作日/周末均值   : {wk.loc[wk['wd'] < 5, '负荷值'].mean():.4f} / "
          f"{wk.loc[wk['wd'] >= 5, '负荷值'].mean():.4f}")
    by_hour = a_ok.groupby(a_ok["时间戳"].dt.hour)["负荷值"].mean()
    print(f"  A 按小时均值 peak/valley hour : {int(by_hour.idxmax())} / {int(by_hour.idxmin())}")
    print(f"  A 按小时均值 peak/valley 值   : {by_hour.max():.4f} / {by_hour.min():.4f}")
    print(f"  A hour 均值 24 值   : {[round(float(v), 2) for v in by_hour]}")
    by_hour = b_ok.groupby(b_ok["时间戳"].dt.hour)["负荷值"].mean()
    print(f"  B 按小时均值 peak/valley hour : {int(by_hour.idxmax())} / {int(by_hour.idxmin())}")
    print(f"  B 按小时均值 peak/valley 值   : {by_hour.max():.4f} / {by_hour.min():.4f}")
    print(f"  B hour 均值 24 值   : {[round(float(v), 2) for v in by_hour]}")
    by_m = a_ok.groupby(a_ok["时间戳"].dt.month)["负荷值"].mean()
    print(f"  A 月均 peak/valley 月         : {int(by_m.idxmax())} / {int(by_m.idxmin())}")
    print(f"  A 月均 peak/valley 值         : {by_m.max():.4f} / {by_m.min():.4f}")
    print(f"  A 月均 12 值        : {[round(float(v), 2) for v in by_m]}")
    print()

    pv_c = pv.drop_duplicates().reset_index(drop=True)
    gen = float(pv_c.loc[pv_c["实际功率"].between(0.0, PV_CAP), "实际功率"].sum())
    print("=" * 72)
    print("pv_power.csv")
    print("-" * 72)
    print(f"  行数（含重复）      : {pstat['rows']}")
    print(f"  完全重复行          : {int(pv.duplicated().sum())}")
    print(f"  去重后行数          : {len(pv_c)}")
    print(f"  电站数              : {pv_c['电站编号'].nunique()}")
    print(f"  实际功率缺失        : {int(pv_c['实际功率'].isna().sum())}（连续段 "
          f"{[s[1] for s in pstat['segments']]} + 散点 20）")
    print(f"  夜间辐照度非零      : {pstat['night_drift']}")
    print(f"  实际功率负值        : {int((pv_c['实际功率'] < 0).sum())}")
    print(f"  辐照度 min/max      : {pv_c['辐照度'].min():.1f} / {pv_c['辐照度'].max():.1f}")
    print(f"  理论功率 max        : {pv_c['理论功率'].max():.2f}")
    print(f"  实际功率 max        : {pv_c['实际功率'].max():.2f}")
    print(f"  年发电量 (kWh)      : {gen:.2f}")
    print(f"  等效利用小时 (h)    : {gen / PV_CAP:.4f}")
    print(f"  容量因子            : {gen / (PV_CAP * N_HOURS):.6f}")
    print(f"  天气类型分布        : {pv_c['天气类型'].value_counts().to_dict()}")
    print()

    print("=" * 72)
    print("load_curve_messy.csv")
    print("-" * 72)
    print(f"  行数                : {len(messy)}")
    print(f"  列名                : {list(messy.columns)}")
    print(f"  时间戳示例          : {messy[' 时间戳 '].iloc[[0, 1, 2, 3]]}")
    print(f"  负荷列文本脏        : {sum(1 for v in messy['负荷值(kW)'] if not v.replace('.', '', 1).lstrip('-').isdigit())}")
    print(f"  哨兵 9999 个数      : {int((messy['负荷值(kW)'] == '9999').sum())}")
    print("=" * 72)


if __name__ == "__main__":
    main()
