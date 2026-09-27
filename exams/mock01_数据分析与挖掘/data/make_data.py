"""生成 mock01（数据分析与挖掘）模拟卷的合成数据。

固定 seed=42，断网可跑、结果完全可复现。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python data/make_data.py

产出（写在本脚本同目录，均为虚构数据）：

    defect_records.csv    缺陷记录主表，**GBK 编码**，1313 行
                          · 「负荷值_kW」是【字符串】：「12.5 kW」「1,234.50」「N/A」「—」「」
                          · 「发现日期」混 4 种格式（含中文日期「2025年3月15日」）
                          · 「缺陷等级」「处理状态」同一含义多种写法（含首尾空格）
                          · 含完全重复行、业务重复行
                          · 「处理时长_小时」含 9999 哨兵值、-1、3σ 外尖峰
    device_master.csv     设备台账（故意缺 12 个设备编号，用于练 left join 出 NaN）
    load_curve.csv        台区小时负荷曲线（8760 行，含温度、含缺失段）

字段命名全部为虚构编号（STATION_* / LINE_* / DEV_*），不对应任何真实电网实体。

目标标签：`缺陷等级` 为「严重 / 危急」时视为高风险（severity=1），
可由「设备类型 + 投运年份 + 负荷水平」学习得到，理论准确率约 0.85。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_RECORD = 1300
OUT_DIR = Path(__file__).resolve().parent

STATIONS = [f"STATION_{c}_{i:02d}" for c in "ABC" for i in (1, 2, 3, 4)]
LINES = ["LINE_ALPHA_1", "LINE_BETA_2", "LINE_GAMMA_3", "LINE_DELTA_4"]
DEVICE_TYPES = ["变压器", "断路器", "绝缘子", "避雷器", "互感器"]
DEFECT_TYPES = ["渗漏油", "锈蚀", "破损", "发热", "异物", "放电痕迹"]
SEVERITIES = ["一般", "严重", "危急"]
STATUS = ["已处理", "处理中", "未处理"]

LOAD_BASE = {"变压器": 520.0, "断路器": 180.0, "绝缘子": 60.0, "避雷器": 45.0, "互感器": 90.0}

N_DUP_FULL = 8          # 完全重复行
N_DUP_BIZ = 5           # 业务重复行（仅记录 ID 不同）
N_LOAD_AS_TEXT = 10     # 负荷值带单位
N_LOAD_THOUSAND = 8     # 负荷值带千分位
N_LOAD_NA = 6           # 负荷值 "N/A"
N_LOAD_DASH = 5         # 负荷值 "—"
N_LOAD_EMPTY = 4        # 负荷值空串
N_SENTINEL = 3          # 处理时长 9999 哨兵
N_NEGATIVE = 3          # 处理时长 -1
N_SPIKE = 4             # 处理时长 3σ 外尖峰
N_CN_DATE = 130         # 中文日期行数
MISS_DURATION = 0.05    # 处理时长缺失比例
MISS_TEMP = 0.06        # 温度缺失比例

CURVE_STATIONS = ["STATION_A_01", "STATION_B_01"]
CURVE_START = "2025-01-01"


def build_devices(rng: np.random.Generator) -> pd.DataFrame:
    n_dev = 300
    return pd.DataFrame({
        "设备编号": [f"DEV_{i:04d}" for i in range(1, n_dev + 1)],
        "台区编号": rng.choice(STATIONS, size=n_dev),
        "线路名称": rng.choice(LINES, size=n_dev),
        "设备类型": rng.choice(DEVICE_TYPES, size=n_dev),
        "投运年份": rng.integers(2005, 2024, size=n_dev),
    })


def build_records(rng: np.random.Generator, devices: pd.DataFrame) -> pd.DataFrame:
    pick = rng.integers(0, len(devices), size=N_RECORD)
    dev = devices.iloc[pick].reset_index(drop=True)

    base = np.array([LOAD_BASE[t] for t in dev["设备类型"]])
    load_true = base * rng.lognormal(mean=0.0, sigma=0.28, size=N_RECORD)
    load_true = np.round(load_true, 1)

    days = rng.integers(0, 365, size=N_RECORD)
    dates = pd.Timestamp(CURVE_START) + pd.to_timedelta(days, unit="D")
    duration = np.round(rng.gamma(shape=2.4, scale=9.0, size=N_RECORD), 1)

    risk = (
        1.15 * (dev["设备类型"].to_numpy() == "变压器")
        + 0.85 * (dev["设备类型"].to_numpy() == "互感器")
        + 0.020 * (2024 - dev["投运年份"].to_numpy())
        + 0.0045 * (load_true - load_true.mean())
        + rng.normal(0, 1.0, size=N_RECORD)
    )
    high = risk > np.quantile(risk, 0.65)

    severity = np.where(high, rng.choice(["严重", "危急"], size=N_RECORD, p=[0.72, 0.28]), "一般")

    df = pd.DataFrame({
        "记录ID": [f"REC-{i:04d}" for i in range(1, N_RECORD + 1)],
        "设备编号": dev["设备编号"].to_numpy(),
        "缺陷类型": rng.choice(DEFECT_TYPES, size=N_RECORD),
        "缺陷等级": severity,
        "发现日期": dates,
        "处理状态": rng.choice(STATUS, size=N_RECORD, p=[0.62, 0.23, 0.15]),
        "处理时长_小时": duration,
        "负荷值_kW": load_true,
    })
    return df


def dirty_dates(rng: np.random.Generator, df: pd.DataFrame) -> pd.Series:
    """把日期列揉成 4 种格式。"""
    out = []
    for i, ts in enumerate(df["发现日期"]):
        if i < N_CN_DATE:
            out.append(f"{ts.year}年{ts.month}月{ts.day}日")
        else:
            style = rng.integers(0, 3)
            if style == 0:
                out.append(ts.strftime("%Y-%m-%d"))
            elif style == 1:
                out.append(ts.strftime("%Y/%m/%d"))
            else:
                out.append(ts.strftime("%m/%d/%Y"))
    return rng.permutation(np.array(out, dtype=object))


def dirty_load_text(rng: np.random.Generator, df: pd.DataFrame) -> pd.Series:
    """把数值列揉成带单位 / 千分位 / 占位符的字符串。"""
    out = df["负荷值_kW"].map(lambda v: f"{v:.1f}")
    idx = rng.choice(len(df), size=N_LOAD_AS_TEXT + N_LOAD_THOUSAND, replace=False)
    for j, i in enumerate(idx):
        if j < N_LOAD_AS_TEXT:
            out.iloc[i] = f"{df['负荷值_kW'].iloc[i]:.2f} kW"
        else:
            out.iloc[i] = f"{df['负荷值_kW'].iloc[i] * 12:,.2f}"
    idx2 = rng.choice(len(df), size=N_LOAD_NA + N_LOAD_DASH + N_LOAD_EMPTY, replace=False)
    for j, i in enumerate(idx2):
        if j < N_LOAD_NA:
            out.iloc[i] = "N/A"
        elif j < N_LOAD_NA + N_LOAD_DASH:
            out.iloc[i] = "—"
        else:
            out.iloc[i] = ""
    return out.to_numpy()


def dirty_severity(rng: np.random.Generator, df: pd.DataFrame) -> pd.Series:
    """同一含义多种写法。"""
    alias = {"一般": ["一般", "一般 ", "轻微"], "严重": ["严重", "较重"], "危急": ["危急", "紧急"]}
    return pd.Series([rng.choice(alias[v]) for v in df["缺陷等级"]])


def dirty_status(rng: np.random.Generator, df: pd.DataFrame) -> pd.Series:
    alias = {"已处理": ["已处理", "已处理 ", "处理完成"], "处理中": ["处理中", "在办"], "未处理": ["未处理", "未处理 "]}
    return pd.Series([rng.choice(alias[v]) for v in df["处理状态"]])


def inject_outliers(rng: np.random.Generator, series: pd.Series) -> pd.Series:
    out = series.to_numpy(dtype=float).copy()
    idx = rng.choice(len(out), size=N_SENTINEL + N_NEGATIVE + N_SPIKE, replace=False)
    for j, i in enumerate(idx):
        if j < N_SENTINEL:
            out[i] = 9999.0
        elif j < N_SENTINEL + N_NEGATIVE:
            out[i] = -1.0
        else:
            out[i] = out[i] * 9 + 400
    mask = rng.random(len(out)) < MISS_DURATION
    out[mask] = np.nan
    return pd.Series(out)


def build_load_curve(rng: np.random.Generator) -> pd.DataFrame:
    stamps = pd.date_range(CURVE_START, periods=365 * 24, freq="h")
    frames = []
    for k, station in enumerate(CURVE_STATIONS):
        hour = stamps.hour.to_numpy()
        doy = stamps.dayofyear.to_numpy()
        profile = 300 + 120 * np.sin((hour - 7) / 24 * 2 * np.pi) + 60 * np.cos((hour - 19) / 24 * 2 * np.pi)
        seasonal = 70 * np.sin((doy - 90) / 365 * 2 * np.pi)
        noise = rng.normal(0, 18, size=len(stamps))
        load = np.round(profile + seasonal + noise + k * 90, 2)
        temp = np.round(18 + 12 * np.sin((doy - 110) / 365 * 2 * np.pi) + 6 * np.sin((hour - 14) / 24 * 2 * np.pi), 2)
        temp = temp + rng.normal(0, 1.2, size=len(stamps))
        frames.append(pd.DataFrame({"时间戳": stamps, "台区编号": station, "负荷值_kW": load, "温度": temp}))
    curve = pd.concat(frames, ignore_index=True)
    mask = rng.random(len(curve)) < MISS_TEMP
    curve.loc[mask, "温度"] = np.nan
    return curve


def main() -> None:
    rng = np.random.default_rng(SEED)

    devices = build_devices(rng)
    records = build_records(rng, devices)

    records["发现日期"] = dirty_dates(rng, records)
    records["负荷值_kW"] = dirty_load_text(rng, records)
    records["缺陷等级"] = dirty_severity(rng, records)
    records["处理状态"] = dirty_status(rng, records)
    records["处理时长_小时"] = inject_outliers(rng, records["处理时长_小时"])

    dup_full = records.sample(N_DUP_FULL, random_state=SEED)
    dup_biz = records.sample(N_DUP_BIZ, random_state=SEED + 1).copy()
    dup_biz["记录ID"] = [f"REC-9{i:03d}" for i in range(N_DUP_BIZ)]
    records = pd.concat([records, dup_full, dup_biz], ignore_index=True)
    records = records.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    master = devices.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    master = master.iloc[:-12]          # 故意缺 12 个设备编号

    curve = build_load_curve(rng)

    records.to_csv(OUT_DIR / "defect_records.csv", index=False, encoding="gbk")
    master.to_csv(OUT_DIR / "device_master.csv", index=False, encoding="utf-8")
    curve.to_csv(OUT_DIR / "load_curve.csv", index=False, encoding="utf-8")

    n_dup = int(records.duplicated().sum())
    print(f"defect_records.csv   {records.shape[0]} 行 × {records.shape[1]} 列（GBK）")
    print(f"  完全重复行         {n_dup}")
    print(f"  发现日期格式       {records['发现日期'].map(lambda s: '中文' if '年' in s else ('斜杠' if '/' in s else '横杠')).value_counts().to_dict()}")
    print(f"  处理时长缺失       {int(records['处理时长_小时'].isna().sum())}")
    print(f"  处理时长 9999      {int((records['处理时长_小时'] == 9999).sum())}")
    print(f"  处理时长 -1        {int((records['处理时长_小时'] == -1).sum())}")
    print(f"  负荷值非数值写法   {int(records['负荷值_kW'].str.contains('kW|,|N/A|—').sum() + (records['负荷值_kW'] == '').sum())}")
    print(f"device_master.csv    {master.shape[0]} 行（缺 12 个设备）")
    print(f"load_curve.csv       {curve.shape[0]} 行，温度缺失 {int(curve['温度'].isna().sum())}")
    print(f"\n高风险占比（严重+危急）  {records['缺陷等级'].isin(['严重', '较重', '危急', '紧急']).mean():.4f}")


if __name__ == "__main__":
    main()
