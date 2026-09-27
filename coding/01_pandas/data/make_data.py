"""生成 01_pandas 各章共用的合成数据。

固定 seed=42，重复运行结果完全一致；不依赖网络，断网可跑。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python data/make_data.py

产出（写在本脚本同目录，均为虚构数据）：

    device_defects_raw.csv   完全原始数据
                             · 负荷值列是【字符串】：「12.5 kW」「1,234.50」「N/A」「—」「」
                             · 发现日期列混 4 种格式（含中文日期）
                             · 缺陷类型 / 处理状态 同一含义多种写法
                             · 含缺失、重复、异常值
    device_defects.csv       轻度加工：仅把「负荷值」转为数值（其余脏数据原样保留）
    device_defects_gbk.csv   device_defects.csv 的 GBK 编码版本
                             · 专供 ch02 练「UnicodeDecodeError / encoding 参数」
    device_info.csv          设备台账（故意少 2 个台区，用于练 left join 出 NaN）
    station_summary.csv      台区汇总（户数、日均负荷）
    station_summary.xlsx     同上内容的 Excel 版本，供 ch02 练 read_excel
    load_curve.csv           台区小时负荷曲线（含温度、节假日标记、含缺失段）

字段命名全部为虚构编号（STATION_* / LINE_* / BRANCH_*），不对应任何真实电网实体。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BASE = 260
OUT_DIR = Path(__file__).resolve().parent

STATIONS = [f"STATION_{c}_{i:02d}" for c in "ABCDEF" for i in (1, 2, 3)]
LINES = [
    "LINE_ALPHA_1",
    "LINE_BETA_2",
    "LINE_GAMMA_3",
    "LINE_DELTA_4",
    "LINE_EPSILON_5",
    "LINE_ZETA_6",
]
BRANCHES = [f"BRANCH_{i:02d}" for i in range(1, 6)]
DEVICE_TYPES = ["变压器", "断路器", "绝缘子", "避雷器", "互感器"]
DEVICE_PREFIX = {"变压器": "TR", "断路器": "CB", "绝缘子": "IN", "避雷器": "LA", "互感器": "CT"}
DEFECT_TYPES = ["渗漏油", "锈蚀", "破损", "发热", "异物", "放电痕迹"]
LEVELS = ["一般", "严重", "危急"]
STATUS = ["已处理", "处理中", "未处理"]

LOAD_BASE = {"变压器": 520.0, "断路器": 180.0, "绝缘子": 60.0, "避雷器": 45.0, "互感器": 90.0}

DATE_STYLES = ["%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"]
N_CN_DATE = 12          # 中文日期行数
N_LOAD_AS_TEXT = 10     # 带单位 / 千分位的行数
N_DUP_FULL = 6          # 完全重复行数
N_DUP_BIZ = 4           # 业务重复行数（仅记录ID不同）
N_SENTINEL = 4          # 9999 哨兵值
N_NEGATIVE = 4          # -1
N_SPIKE = 3             # 3σ 外尖峰
MISSING_RATE = 0.08     # 负荷值 MCAR 缺失比例
MAR_RATE = 0.25         # 温度在指定台区上的缺失比例
REMARK_KEEP_RATE = 0.25 # 备注列保留比例（即约 75% 缺失）

CURVE_STATIONS = ["STATION_A_01", "STATION_C_01", "STATION_E_01"]
CURVE_DAYS = 90
CURVE_START = "2026-01-01"
HOLIDAYS = {
    *(f"2026-01-0{d}" for d in (1, 2, 3)),                    # 元旦
    *(f"2026-02-{d:02d}" for d in range(16, 23)),             # 春节假期
}


def pick(rng: np.random.Generator, n: int, pool: np.ndarray, used: set[int]) -> np.ndarray:
    """从 pool 中不放回抽 n 个，且不与 used 冲突。"""
    candidates = np.array([i for i in pool if i not in used])
    chosen = rng.choice(candidates, n, replace=False)
    used.update(int(i) for i in chosen)
    return chosen


def build_device_defects(rng: np.random.Generator) -> pd.DataFrame:
    """生成设备缺陷记录。"""
    station = rng.choice(STATIONS, N_BASE)
    line = rng.choice(LINES, N_BASE)
    dev_type = rng.choice(DEVICE_TYPES, N_BASE)
    defect = rng.choice(DEFECT_TYPES, N_BASE)
    level = rng.choice(LEVELS, N_BASE, p=[0.60, 0.28, 0.12])
    status = rng.choice(STATUS, N_BASE, p=[0.62, 0.24, 0.14])

    counters: dict[str, int] = {}
    dev_no = []
    for t in dev_type:
        counters[t] = counters.get(t, 0) + 1
        dev_no.append(f"{DEVICE_PREFIX[t]}_{counters[t]:03d}")

    load = np.array([LOAD_BASE[t] for t in dev_type]) + rng.normal(0, 30, N_BASE)
    load = np.round(np.clip(load, 5.0, None), 2)
    temp = np.round(rng.normal(22, 8, N_BASE), 1)
    humid = np.round(rng.uniform(30, 90, N_BASE), 1)
    remark = [("已复核" if rng.random() < REMARK_KEEP_RATE else np.nan) for _ in range(N_BASE)]
    found_dt = pd.Timestamp(CURVE_START) + pd.to_timedelta(rng.integers(0, 181, N_BASE), unit="D")

    df = pd.DataFrame(
        {
            "记录ID": np.arange(1, N_BASE + 1),
            "台区编号": station,
            "线路名称": line,
            "设备类型": dev_type,
            "设备编号": dev_no,
            "缺陷类型": defect,
            "缺陷等级": level,
            "发现日期": found_dt,
            "负荷值": load,
            "温度": temp,
            "湿度": humid,
            "处理状态": status,
            "备注": remark,
        }
    )

    # ---- 1. 缺失值（MCAR + MAR）----
    df.loc[rng.choice(N_BASE, int(round(N_BASE * MISSING_RATE)), replace=False), "负荷值"] = np.nan
    mar_pool = df.index[df["台区编号"].str.startswith(("STATION_C_", "STATION_D_"))]
    n_mar = int(round(len(mar_pool) * MAR_RATE))
    df.loc[rng.choice(mar_pool, n_mar, replace=False), "温度"] = np.nan

    # ---- 2. 异常值 ----
    used: set[int] = set()
    df.loc[pick(rng, N_SENTINEL, df.index, used), "负荷值"] = 9999.0
    df.loc[pick(rng, N_NEGATIVE, df.index, used), "负荷值"] = -1.0
    df.loc[pick(rng, N_SPIKE, df.index, used), "负荷值"] = 5000.0

    used_h: set[int] = set()
    for value in (150.0, 120.0, 105.0, -5.0, -3.0):
        df.loc[pick(rng, 1, df.index, used_h), "湿度"] = value

    # ---- 3. 类别取值不一致 ----
    defect_variants = {"渗漏油": ["渗油", "漏油"], "异物": ["异物搭挂"], "锈蚀": ["锈蚀 ", "锈蚀"]}
    for value, variants in defect_variants.items():
        idx = df.index[df["缺陷类型"] == value]
        if len(idx) == 0:
            continue
        hit = rng.choice(idx, max(1, int(len(idx) * 0.3)), replace=False)
        df.loc[hit, "缺陷类型"] = rng.choice(variants, len(hit))

    status_variants = {"已处理": ["已 处理", "处理完成", "done"], "未处理": ["待处理"]}
    for value, variants in status_variants.items():
        idx = df.index[df["处理状态"] == value]
        if len(idx) == 0:
            continue
        hit = rng.choice(idx, max(1, int(len(idx) * 0.25)), replace=False)
        df.loc[hit, "处理状态"] = rng.choice(variants, len(hit))
    hit = df.index[df["处理状态"] == "处理中"]
    if len(hit):
        df.loc[rng.choice(hit, max(1, len(hit) // 3), replace=False), "处理状态"] = "处理中 "

    # ---- 4. 重复行 ----
    dup_full = df.loc[[9, 19, 29, 39, 49, 59]].copy()
    dup_biz = df.loc[[100, 110, 120, 130]].copy()
    dup_biz["记录ID"] = np.arange(901, 901 + len(dup_biz))
    dup_biz["备注"] = "业务重复样本"
    df = pd.concat([df, dup_full, dup_biz], ignore_index=True)

    # ---- 5. 打散（让重复行不相邻，更接近真实导出）----
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def to_raw_text(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """把数值列和日期列转成真实导出里常见的混乱字符串形态。"""
    raw = df.copy()

    # 负荷值 -> 字符串（含单位、千分位、缺失占位符）
    # 缺失占位符刻意用满 3 种，且等量分布：
    #   "N/A" 与 "" 会被 pandas 读成 NaN，"—" 不会 —— 这是 ch06 的核心考点
    placeholders = ["N/A", "—", ""]
    load_text: list[str] = []
    missing_seen = 0
    for value in df["负荷值"]:
        if pd.isna(value):
            load_text.append(placeholders[missing_seen % len(placeholders)])
            missing_seen += 1
        else:
            load_text.append(f"{float(value):.2f}")

    wrappable = [i for i, s in enumerate(load_text) if s.replace(".", "", 1).isdigit()]
    for i in rng.choice(wrappable, min(N_LOAD_AS_TEXT, len(wrappable)), replace=False):
        value = float(load_text[i])
        load_text[i] = f"{value:,.2f}" if i % 2 == 0 else f"{value:.2f} kW"
    # 大数值（哨兵 9999 / 尖峰 5000）一律加千分位，制造「1,234.50」这类脏格式
    for i, s in enumerate(load_text):
        if s.replace(".", "", 1).isdigit() and float(s) >= 1000:
            load_text[i] = f"{float(s):,.2f}"
    raw["负荷值"] = load_text

    # 发现日期 -> 混 4 种格式
    styles = rng.integers(0, len(DATE_STYLES), len(df))
    date_text = [
        d.strftime(DATE_STYLES[s]) for d, s in zip(df["发现日期"], styles)
    ]
    for i in range(N_CN_DATE):
        d = df["发现日期"].iloc[i]
        date_text[i] = f"{d.year}年{d.month}月{d.day}日"
    raw["发现日期"] = date_text

    return raw


def build_device_info(rng: np.random.Generator) -> pd.DataFrame:
    """设备台账。故意漏掉 2 个台区，用于练 left join 产生 NaN。"""
    stations = [s for s in STATIONS if s not in {"STATION_B_03", "STATION_F_02"}]
    return pd.DataFrame(
        {
            "台区编号": stations,
            "所属供电所": rng.choice(BRANCHES, len(stations)),
            "容量kVA": rng.choice([315, 400, 500, 630, 800], len(stations)),
            "投运年份": rng.choice([2005, 2010, 2014, 2018, 2021, 2024], len(stations)),
            "供电区域类型": rng.choice(["城区", "郊区", "农村"], len(stations), p=[0.5, 0.3, 0.2]),
        }
    )


def build_station_summary(rng: np.random.Generator, defects: pd.DataFrame) -> pd.DataFrame:
    """台区汇总：户数 + 日均负荷（由缺陷表聚合而来，保证两表能对上）。"""
    grouped = defects.groupby("台区编号")["负荷值"].mean().rename("日均负荷").reset_index()
    grouped["户数"] = rng.integers(80, 900, len(grouped))
    return grouped


def build_load_curve(rng: np.random.Generator) -> pd.DataFrame:
    """台区小时负荷曲线：日周期 + 周周期 + 节假日效应 + 缺失段。"""
    hours = pd.date_range(CURVE_START, periods=CURVE_DAYS * 24, freq="h")
    frames = []
    for station in CURVE_STATIONS:
        t = np.arange(len(hours))
        hour_of_day = hours.hour.to_numpy()
        weekday = hours.dayofweek.to_numpy()
        is_holiday = np.array([ts.strftime("%Y-%m-%d") in HOLIDAYS for ts in hours])

        daily = 1.0 + 0.45 * np.sin((hour_of_day - 7) / 24 * 2 * np.pi)
        weekly = np.where(weekday >= 5, 0.88, 1.0)
        holiday_effect = np.where(is_holiday, 0.78, 1.0)
        trend = 1.0 + 0.00012 * t
        noise = rng.normal(0, 0.035, len(hours))

        load = 620.0 * daily * weekly * holiday_effect * trend * (1 + noise)
        temp = (
            14.0
            + 9.0 * np.sin((t / (24 * 365)) * 2 * np.pi - 1.2)
            + 5.5 * np.sin((hour_of_day - 9) / 24 * 2 * np.pi)
            + rng.normal(0, 1.2, len(hours))
        )

        frames.append(
            pd.DataFrame(
                {
                    "时间戳": hours,
                    "台区编号": station,
                    "负荷值": np.round(load, 2),
                    "温度": np.round(temp, 1),
                    "是否节假日": is_holiday,
                }
            )
        )

    curve = pd.concat(frames, ignore_index=True)

    # 注入缺失：单点缺失 + 连续 6 小时整段缺失
    used: set[int] = set()
    curve.loc[pick(rng, 30, curve.index, used), "负荷值"] = np.nan
    for start in (500, 1500, 3000):
        curve.loc[curve.index[start : start + 6], "负荷值"] = np.nan
    return curve


def main() -> None:
    rng = np.random.default_rng(SEED)

    defects = build_device_defects(rng)
    raw = to_raw_text(defects, rng)
    info = build_device_info(rng)
    summary = build_station_summary(rng, defects)
    curve = build_load_curve(rng)

    raw.to_csv(OUT_DIR / "device_defects_raw.csv", index=False, encoding="utf-8-sig")
    defects.to_csv(OUT_DIR / "device_defects.csv", index=False, encoding="utf-8-sig")
    defects.to_csv(OUT_DIR / "device_defects_gbk.csv", index=False, encoding="gbk")
    info.to_csv(OUT_DIR / "device_info.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "station_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_excel(OUT_DIR / "station_summary.xlsx", index=False)
    curve.to_csv(OUT_DIR / "load_curve.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"device_defects.csv  行数 {len(defects)}  列数 {defects.shape[1]}")
    print("-" * 60)
    print("【真值备忘】")
    print(f"  完全重复行（所有列相同）      : {int(defects.duplicated().sum())}")
    biz_key = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]
    print(f"  业务重复行（按业务键去重）    : {int(defects.duplicated(subset=biz_key).sum())}")
    print(f"  负荷值缺失（MCAR，不含哨兵）  : {int(defects['负荷值'].isna().sum())}")
    print(f"  温度缺失（MAR，C/D 台区）     : {int(defects['温度'].isna().sum())}")
    print(f"  湿度列缺失                    : {int(defects['湿度'].isna().sum())}")
    print(f"  备注列缺失                    : {int(defects['备注'].isna().sum())} / {len(defects)}")
    print(f"  负荷值 == 9999                : {int((defects['负荷值'] == 9999.0).sum())}")
    print(f"  负荷值 == -1                  : {int((defects['负荷值'] == -1.0).sum())}")
    print(f"  负荷值 == 5000                : {int((defects['负荷值'] == 5000.0).sum())}")
    print(f"  湿度 > 100 或 < 0             : {int(((defects['湿度'] > 100) | (defects['湿度'] < 0)).sum())}")
    print(f"  缺陷类型取值种类              : {sorted(defects['缺陷类型'].unique())}")
    print(f"  处理状态取值种类              : {sorted(defects['处理状态'].unique())}")
    print(f"  台区数（缺陷表）              : {defects['台区编号'].nunique()}")
    print(f"  台区数（台账表）              : {info['台区编号'].nunique()}  ← 故意少 2 个")
    print(f"  发现日期格式种类（raw）       : 4 种")
    print(f"  负荷值列 dtype（raw）         : 字符串")
    print("=" * 60)
    print(f"device_info.csv     行数 {len(info)}")
    print(f"station_summary.csv 行数 {len(summary)}")
    print(f"load_curve.csv      行数 {len(curve)}  台区 {curve['台区编号'].nunique()} 个")
    print(f"  曲线负荷缺失                  : {int(curve['负荷值'].isna().sum())}")
    print("-" * 60)
    print("【编码练习文件】")
    print("  device_defects_gbk.csv  -> GBK 编码，用 utf-8 读会 UnicodeDecodeError")
    print("  station_summary.xlsx    -> 供 read_excel 使用")
    print("=" * 60)
    print(f"输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
