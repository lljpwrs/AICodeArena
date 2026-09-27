"""02_sklearn 合成数据生成 —— 机器学习方向专用

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python coding/02_sklearn/data/make_data.py

产出三份 CSV（与 01_pandas 的数据独立，面向建模场景）：
    defect_ml.csv    分类数据：预测缺陷是否"危急"（二分类，不平衡）
    load_reg.csv     回归数据：预测台区"日均负荷"
    station_load.csv 聚类数据：台区用电形态（3 类自然簇 + 量纲陷阱）

原则（docs/合成数据规范.md）：
    - seed=42 固定，可复现
    - 埋坑：缺失（MCAR + MAR）、类别不平衡、量纲差异悬殊、
      类别列取值集合在训练/测试段不重合（练 handle_unknown）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# 1) defect_ml.csv —— 二分类（目标：是否危急）
# --------------------------------------------------------------------------- #

def make_defect_ml() -> pd.DataFrame:
    n = 600

    dev_types = ["变压器", "断路器", "互感器", "避雷器", "绝缘子"]
    dev_p = [0.25, 0.20, 0.18, 0.17, 0.20]
    defect_types = ["渗漏油", "锈蚀", "破损", "发热", "异物", "放电痕迹"]
    defect_p = [0.20, 0.18, 0.15, 0.20, 0.12, 0.15]
    stations = [f"STATION_{g}_{i:02d}" for g in "ABCDEF" for i in range(1, 7)]
    lines = [f"LINE_{w}_{k}" for w in ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
             for k in range(1, 4)]

    df = pd.DataFrame({
        "记录ID": np.arange(1, n + 1),
        "台区编号": RNG.choice(stations, n),
        "线路名称": RNG.choice(lines, n),
        "设备类型": RNG.choice(dev_types, n, p=dev_p),
        "缺陷类型": RNG.choice(defect_types, n, p=defect_p),
        "投运年限": RNG.integers(1, 31, n).astype(float),
        "负荷值": np.round(RNG.normal(420, 150, n), 2).clip(40, None),
        "温度": np.round(RNG.normal(35, 8, n), 1),
        "湿度": np.round(RNG.normal(65, 12, n), 1).clip(10, 100),
        "巡检耗时分钟": np.round(RNG.normal(45, 12, n), 0).clip(10, None),
    })

    # --- 目标变量：危急概率由 负荷值 / 温度 / 缺陷类型 驱动 ---
    z = (
        -2.4
        + 0.005 * (df["负荷值"] - 420)
        + 0.10 * (df["温度"] - 35)
        + 1.4 * df["缺陷类型"].isin(["放电痕迹", "发热"])
        + 0.012 * df["投运年限"]
    )
    prob = 1 / (1 + np.exp(-z))
    df["是否危急"] = (RNG.random(n) < prob).astype(int)

    # --- 埋坑 1：负荷值 MCAR 缺失 8% ---
    miss_load = RNG.choice(n, size=48, replace=False)
    df.loc[miss_load, "负荷值"] = np.nan

    # --- 埋坑 2：温度 MAR——C/D 区台区缺 25% ---
    cd_mask = df["台区编号"].str.startswith(("STATION_C_", "STATION_D_"))
    cd_idx = df.index[cd_mask].to_numpy()
    miss_temp = RNG.choice(cd_idx, size=int(len(cd_idx) * 0.25), replace=False)
    df.loc[miss_temp, "温度"] = np.nan

    return df


# --------------------------------------------------------------------------- #
# 2) load_reg.csv —— 回归（目标：日均负荷）
# --------------------------------------------------------------------------- #

def make_load_reg() -> pd.DataFrame:
    n = 800

    region_types = ["城区", "郊区", "工业区", "商业区"]
    region_p = [0.35, 0.30, 0.20, 0.15]
    capacities = np.array([200, 315, 400, 630, 800, 1000, 1250])
    cap_p = [0.15, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05]

    df = pd.DataFrame({
        "供电区域类型": RNG.choice(region_types, n, p=region_p),
        "容量kVA": RNG.choice(capacities, n, p=cap_p).astype(float),
        "投运年限": RNG.integers(1, 31, n).astype(float),
        "月平均温度": np.round(RNG.normal(18, 9, n), 1),
        "月平均湿度": np.round(RNG.normal(68, 10, n), 1),
    })

    # 户数与容量强相关（真实世界就是这样，也给后面做特征工程留话题）
    df["户数"] = (df["容量kVA"] / 6 + RNG.normal(0, 8, n)).round(0).astype(int)
    df["台区编号"] = [f"STATION_{RNG.choice(list('ABCDEF'))}_{RNG.integers(1, 7):02d}"
                      for _ in range(n)]

    y = (
        0.32 * df["容量kVA"]
        + 1.9 * df["户数"]
        + 6.0 * (df["月平均温度"] - 18).abs()
        + 40 * (df["供电区域类型"] == "工业区")
        + RNG.normal(0, 30, n)
    )
    df["日均负荷"] = np.round(y.clip(50, None), 2)
    return df


# --------------------------------------------------------------------------- #
# 3) station_load.csv —— 聚类（用电形态：居民 / 商业 / 工业）
# --------------------------------------------------------------------------- #

def make_station_load() -> pd.DataFrame:
    """独立 RNG(seed=7)，不影响前两份数据的随机流。"""
    rng = np.random.default_rng(7)
    n_per = {"居民型": 135, "商业型": 90, "工业型": 75}

    specs = {
        #                 日均负荷   夜间负荷率  峰谷差率   周末负荷率   报装容量
        "居民型": ((260, 55), (0.55, 0.06), (0.58, 0.07), (0.98, 0.05), (315, 30)),
        "商业型": ((520, 85), (0.20, 0.05), (0.75, 0.06), (0.65, 0.06), (800, 80)),
        "工业型": ((900, 110), (0.75, 0.05), (0.25, 0.05), (0.85, 0.04), (1250, 100)),
    }

    rows = []
    for kind, n in n_per.items():
        (m1, s1), (m2, s2), (m3, s3), (m4, s4), (m5, s5) = specs[kind]
        rows.append(pd.DataFrame({
            "台区编号": [f"TR_{kind[0]}{i:03d}" for i in range(n)],
            "用电类型": kind,
            "日均负荷": np.round(rng.normal(m1, s1, n), 1).clip(30, None),
            "夜间负荷率": np.round(rng.normal(m2, s2, n), 3).clip(0.02, 0.98),
            "峰谷差率": np.round(rng.normal(m3, s3, n), 3).clip(0.02, 0.98),
            "周末负荷率": np.round(rng.normal(m4, s4, n), 3).clip(0.02, 1.0),
            "报装容量kVA": rng.choice([315, 400, 630, 800, 1000, 1250, 1600], n).astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    defect = make_defect_ml()
    reg = make_load_reg()
    station = make_station_load()

    defect.to_csv(HERE / "defect_ml.csv", index=False, encoding="utf-8-sig")
    reg.to_csv(HERE / "load_reg.csv", index=False, encoding="utf-8-sig")
    station.to_csv(HERE / "station_load.csv", index=False, encoding="utf-8-sig")

    # ---- 真值速报（README 真值备忘的锚点）----
    print("=" * 62)
    print("defect_ml.csv", defect.shape)
    print("  是否危急 =1:", int(defect["是否危急"].sum()),
          "| 占比:", round(defect["是否危急"].mean(), 4))
    print("  负荷值缺失:", int(defect["负荷值"].isna().sum()))
    print("  温度缺失:", int(defect["温度"].isna().sum()),
          "| 湿度缺失:", int(defect["湿度"].isna().sum()))
    print("  台区数:", defect["台区编号"].nunique(),
          "| 线路数:", defect["线路名称"].nunique(),
          "| 设备类型数:", defect["设备类型"].nunique(),
          "| 缺陷类型数:", defect["缺陷类型"].nunique())
    cd = defect["台区编号"].str.startswith(("STATION_C_", "STATION_D_"))
    print("  C/D 区行数:", int(cd.sum()), "| C/D 区温度缺失:",
          int(defect.loc[cd, "温度"].isna().sum()),
          "| 非C/D区温度缺失:", int(defect.loc[~cd, "温度"].isna().sum()))
    print("  各设备类型计数:", defect["设备类型"].value_counts().to_dict())
    print("  危急中 放电痕迹/发热 占比:",
          round(defect.loc[defect["是否危急"] == 1, "缺陷类型"]
                .isin(["放电痕迹", "发热"]).mean(), 4))
    print("  负荷值 min/max:", round(defect["负荷值"].min(), 2),
          "/", round(defect["负荷值"].max(), 2))
    print("  温度 mean:", round(defect["温度"].mean(), 4))
    print("  湿度 mean:", round(defect["湿度"].mean(), 4))
    print("  投运年限 min/max:", int(defect["投运年限"].min()),
          "/", int(defect["投运年限"].max()))
    print("  巡检耗时分钟 mean:", round(defect["巡检耗时分钟"].mean(), 4))

    print("=" * 62)
    print("load_reg.csv", reg.shape)
    print("  日均负荷 mean/std:", round(reg["日均负荷"].mean(), 4),
          "/", round(reg["日均负荷"].std(), 4))
    print("  日均负荷 min/max:", round(reg["日均负荷"].min(), 2),
          "/", round(reg["日均负荷"].max(), 2))
    print("  容量kVA 取值:", sorted(reg["容量kVA"].unique().tolist()))
    print("  供电区域类型计数:", reg["供电区域类型"].value_counts().to_dict())
    print("  户数 min/max:", int(reg["户数"].min()), "/", int(reg["户数"].max()))
    print("  日均负荷与容量相关系数:",
          round(float(reg["日均负荷"].corr(reg["容量kVA"])), 4))
    print("  日均负荷与户数相关系数:",
          round(float(reg["日均负荷"].corr(reg["户数"])), 4))
    print("  台区数:", reg["台区编号"].nunique())
    print("=" * 62)
    print("station_load.csv", station.shape)
    print("  用电类型计数:", station["用电类型"].value_counts().to_dict())
    print("  按类型的日均负荷均值:",
          {k: round(v, 2) for k, v in station.groupby("用电类型")["日均负荷"].mean().items()})
    print("  按类型的夜间负荷率均值:",
          {k: round(v, 4) for k, v in station.groupby("用电类型")["夜间负荷率"].mean().items()})
    print("=" * 62)
    print("文件大小(KB):",
          {p.name: round(p.stat().st_size / 1024, 1)
           for p in sorted(HERE.glob("*.csv"))})


if __name__ == "__main__":
    main()
