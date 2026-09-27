"""final_01 —— 综合题：全流程清洗 + 探索 + 可视化 + 分析报告

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/final_01_full_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas" / "final"
NAME = "final_01_full_pipeline"

HEADER = '''import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

# 中文显示：字体列表按平台命中率从高到低排；负号必须显式修复
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("../data")

print("pandas", pd.__version__)'''

SCAFFOLD = r'''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def close_all():
    """每次画完关掉图窗，避免内存里堆几十张图。"""
    plt.close("all")


def png_bytes(fig):
    """把图渲染成 PNG 字节流（不写文件即可验证 savefig 真的产出了图）。"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf, dpi=110, bbox_inches="tight")
    return buf.getvalue()


def is_png(data):
    """PNG 魔数校验：b"\\x89PNG"。"""
    return data[:4] == b"\x89PNG"


print("脚手架就绪：probe / png_bytes / is_png / close_all")'''

# --------------------------------------------------------------------------- #
# 各阶段代码（lesson 用完整版；exercise 版加 @@todo 挖空标记）
# --------------------------------------------------------------------------- #

S0_CODE = '''# 反例：GBK 文件用 utf-8 读，直接炸
status, msg = probe(pd.read_csv, DATA / "device_defects_gbk.csv")
print("utf-8 读 GBK 文件 →", status, "|", msg)

gbk = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gbk")
print("gbk 编码读取成功:", gbk.shape)

# 主角：最脏的原始导出（utf-8-sig 编码）
raw = pd.read_csv(DATA / "device_defects_raw.csv", encoding="utf-8-sig")
print("raw:", raw.shape, "| 负荷值 dtype:", raw["负荷值"].dtype)

# 负荷值列缺失对账：自动识别 14 个 vs 漏网的 "—" 7 个 = 真缺失 21 个
print("自动识别缺失:", int(raw["负荷值"].isna().sum()),
      "| 漏网的 -- :", int((raw["负荷值"] == "—").sum()),
      "| 含 kW:", int(raw["负荷值"].str.contains("kW").sum()),
      "| 含千分位:", int(raw["负荷值"].str.contains(",").sum()))

assert status == "err" and msg == "UnicodeDecodeError", "utf-8 读 GBK 必须报错"
assert gbk.shape == (270, 13)
assert raw.shape == (270, 13)
assert pd.api.types.is_string_dtype(raw["负荷值"]), "raw 的负荷值还是字符串"
assert int(raw["负荷值"].isna().sum()) == 14, "N/A 与空串被自动识别"
assert int((raw["负荷值"] == "—").sum()) == 7, "-- 不会被自动识别"
assert int(raw["负荷值"].str.contains("kW").sum()) == 5
assert int(raw["负荷值"].str.contains(",").sum()) == 7
'''

S1_CODE = '''# 三步走：去单位 -> 去千分位 -> to_numeric(errors="coerce")
load = (raw["负荷值"].astype(str)
        .str.replace(" kW", "", regex=False)
        .str.replace(",", "", regex=False))
df = raw.assign(负荷值=pd.to_numeric(load, errors="coerce"))

print("清洗后 dtype:", df["负荷值"].dtype, "| 缺失:", int(df["负荷值"].isna().sum()))
print(">=1000:", int((df["负荷值"] >= 1000).sum()), "(9999 x4 + 5000 x3)",
      "| <0:", int((df["负荷值"] < 0).sum()), "(-1 x4)")

assert df["负荷值"].dtype == np.dtype("float64")
assert int(df["负荷值"].isna().sum()) == 21, "14 自动识别 + 7 个 -- 全部落水"
assert int((df["负荷值"] >= 1000).sum()) == 7
assert int((df["负荷值"] == 9999).sum()) == 4
assert int((df["负荷值"] == 5000).sum()) == 3
assert int((df["负荷值"] == -1).sum()) == 4
'''

S2_CODE = '''# 中文日期先替换成年月日，再交给 format="mixed" 一次解析 4 种格式
d = (df["发现日期"].astype(str)
     .str.replace("年", "-", regex=False)
     .str.replace("月", "-", regex=False)
     .str.replace("日", "", regex=False))
df["发现日期"] = pd.to_datetime(d, format="mixed")

print("NaT:", int(df["发现日期"].isna().sum()),
      "| 范围:", df["发现日期"].min(), "~", df["发现日期"].max())

assert int(df["发现日期"].isna().sum()) == 0, "4 种格式全部解析成功"
assert df["发现日期"].min() == pd.Timestamp("2026-01-02")
assert df["发现日期"].max() == pd.Timestamp("2026-06-30")
'''

S3_CODE = '''# 处理状态：去空白 + 同义合并，8 种写法 -> 3 类
df["处理状态"] = (df["处理状态"].astype(str).str.strip()
                 .replace({"已 处理": "已处理", "处理完成": "已处理", "done": "已处理",
                           "处理中 ": "处理中", "待处理": "未处理"}))
print("状态类别:", df["处理状态"].nunique(),
      df["处理状态"].value_counts().to_dict())

# 缺陷类型：去空白 + 同义合并，10 种写法 -> 6 类
df["缺陷类型"] = df["缺陷类型"].astype(str).str.strip()
df["缺陷类型"] = df["缺陷类型"].replace({"渗油": "渗漏油", "漏油": "渗漏油", "异物搭挂": "异物"})
print("缺陷类别:", df["缺陷类型"].nunique(),
      df["缺陷类型"].value_counts().to_dict())

assert df["处理状态"].nunique() == 3
assert int((df["处理状态"] == "已处理").sum()) == 171
assert int((df["处理状态"] == "处理中").sum()) == 54
assert int((df["处理状态"] == "未处理").sum()) == 45
assert df["缺陷类型"].nunique() == 6
assert int((df["缺陷类型"] == "渗漏油").sum()) == 61
assert int((df["缺陷类型"] == "异物").sum()) == 40
'''

S4_CODE = '''# 第一击：完全重复行（所有列都相同）
d1 = df.drop_duplicates()
print("全列去重:", d1.shape)

# 第二击：业务键重复（记录ID不同但业务上是同一条）
BIZ_KEY = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]
d2 = d1.drop_duplicates(subset=BIZ_KEY).reset_index(drop=True)
print("业务键去重:", d2.shape)

assert d1.shape == (264, 13), "完全重复 6 行"
assert d2.shape == (260, 13), "业务键重复再删 4 行"
assert d2.index.tolist() == list(range(260)), "reset_index 让后续索引连续"
'''

S5_CODE = '''# 哨兵与缺失：负荷值 >=1000 或 <0 一律视为哨兵，先记账再置 NaN
n_sent = int(((d2["负荷值"] >= 1000) | (d2["负荷值"] < 0)).sum())
d2["负荷值"] = d2["负荷值"].mask((d2["负荷值"] >= 1000) | (d2["负荷值"] < 0))
n_load_na = int(d2["负荷值"].isna().sum())
print("哨兵值:", n_sent, "| 置 NaN 后缺失:", n_load_na)

# 湿度：物理范围 [0, 100]，越界截断
n_hum = int(((d2["湿度"] > 100) | (d2["湿度"] < 0)).sum())
d2["湿度"] = d2["湿度"].clip(0, 100)
print("湿度越界:", n_hum, "| clip 后越界:", int(((d2["湿度"] > 100) | (d2["湿度"] < 0)).sum()))

# 温度：MAR 缺失（仅 C/D 台区），全局中位数填充
temp_med = float(d2["温度"].median())
d2["温度"] = d2["温度"].fillna(temp_med)
print("温度缺失 23 -> 中位数填充:", temp_med)

# 负荷值：按设备类型的中位数填充（transform 保证与原表逐行对齐）
grp_med = d2.groupby("设备类型")["负荷值"].transform("median")
d2["负荷值"] = d2["负荷值"].fillna(grp_med)
print("填充后缺失:", int(d2["负荷值"].isna().sum()))
print("填充后 mean:", round(float(d2["负荷值"].mean()), 4),
      "| median:", round(float(d2["负荷值"].median()), 4),
      "| max:", round(float(d2["负荷值"].max()), 4))

assert n_sent == 11 and n_load_na == 32, "21 原缺失 + 11 哨兵"
assert n_hum == 5 and int(((d2["湿度"] > 100) | (d2["湿度"] < 0)).sum()) == 0
assert temp_med == 21.9
assert int(d2["温度"].isna().sum()) == 0
assert int(d2["负荷值"].isna().sum()) == 0
assert round(float(d2["负荷值"].mean()), 4) == 175.3413
assert round(float(d2["负荷值"].median()), 4) == 88.695
assert round(float(d2["负荷值"].max()), 4) == 588.45, "截断后最大值是尖峰填充的结果"
'''

S6_CODE = '''# 台账只有 16 个台区，缺陷表有 18 个：left join 必然出 NaN
info = pd.read_csv(DATA / "device_info.csv")
dfm = d2.merge(info, on="台区编号", how="left")

# 先记账再填充：缺键不报错不警告，只是安静地 NaN
n_info_na = int(dfm["所属供电所"].isna().sum())
na_stations = sorted(dfm.loc[dfm["所属供电所"].isna(), "台区编号"].unique().tolist())
print("info:", info.shape, "| merge 后:", dfm.shape,
      "| 所属供电所缺失:", n_info_na)
print("缺失台区:", na_stations)

# 字符列的缺失用明确标记，不用数字硬填
dfm["所属供电所"] = dfm["所属供电所"].fillna("未知")
print("填充后各供电所记录数:", dfm["所属供电所"].value_counts().to_dict())

assert info.shape == (16, 5)
assert dfm.shape == (260, 17), "13 + 5 台账列 - 1 连接键 = 17"
assert n_info_na == 23, "不报错不警告，只是安静地 NaN"
assert na_stations == ["STATION_B_03", "STATION_F_02"]
assert int((dfm["所属供电所"] == "未知").sum()) == 23
'''

S7_CODE = '''# 透视表：设备类型 x 缺陷等级 的平均负荷
pt = dfm.pivot_table(index="设备类型", columns="缺陷等级", values="负荷值", aggfunc="mean")
print("pivot:", pt.shape)
print(pt.round(4))

# 命名聚合：一张「各设备类型业务画像」表
rep = (dfm.groupby("设备类型")
       .agg(记录数=("记录ID", "count"),
            危急数=("缺陷等级", lambda s: int((s == "危急").sum())),
            平均负荷=("负荷值", "mean")))
rep["危急占比"] = rep["危急数"] / rep["记录数"]
print(rep.round(4))

done_rate = float((dfm["处理状态"] == "已处理").mean())
print("全局已处理率:", round(done_rate, 4))

assert pt.shape == (5, 3), "5 种设备 x 3 个等级"
assert round(float(pt.loc["变压器", "危急"]), 4) == 528.394
assert rep.shape == (5, 4)
assert int(rep["危急数"].sum()) == 30, "全表危急 30 条"
assert int(rep.loc["避雷器", "危急数"]) == 12, "避雷器危急最多"
assert round(float(rep.loc["避雷器", "危急占比"]), 4) == 0.2143
assert round(done_rate, 4) == 0.6308
'''

S8_CODE = '''# 分箱：负荷值四档（清洗后无缺失，NaN 必须是 0）
bins = pd.cut(dfm["负荷值"], [0, 50, 150, 300, 1000], labels=["极低", "低", "中", "高"])
print("cut 计数:", bins.value_counts().reindex(["极低", "低", "中", "高"]).to_dict(),
      "| NaN:", int(bins.isna().sum()))

# 风险分级：np.select 按条件从上到下命中
risk = np.select([dfm["负荷值"] > 300, dfm["负荷值"] > 150],
                 ["高风险", "中风险"], default="低风险")
dfm["风险等级"] = risk
print("风险分级:", pd.Series(risk).value_counts().to_dict())

ct = pd.crosstab(dfm["设备类型"], dfm["风险等级"])
print("crosstab:", ct.shape)
print(ct)

assert int(bins.isna().sum()) == 0, "负荷值已填充，分箱不应再有 NaN"
assert bins.value_counts().reindex(["极低", "低", "中", "高"]).tolist() == [62, 109, 36, 53]
assert pd.Series(risk).value_counts().to_dict() == {"低风险": 171, "高风险": 53, "中风险": 36}
assert int(((dfm["风险等级"] == "高风险") & (dfm["缺陷等级"] == "危急")).sum()) == 5
assert ct.shape == (5, 3)
'''

S9A_CODE = '''# 图 1：各设备类型危急缺陷数（柱状）
fig1, ax1 = plt.subplots(figsize=(6, 3.5))
rep["危急数"].plot(kind="bar", ax=ax1, title="各设备类型危急缺陷数")
fig1.tight_layout()
data1 = png_bytes(fig1)
print("bar 图 PNG 字节数:", len(data1), "| 魔数合法:", is_png(data1))
close_all()

assert is_png(data1), "savefig 真的产出了 PNG"
'''

S9B_CODE = '''# 图 2：负荷值分布（直方图，range 压住长尾）
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
dfm["负荷值"].plot(kind="hist", bins=20, range=(0, 1000), ax=ax2, title="清洗后负荷值分布")
fig2.tight_layout()
counts, _ = np.histogram(dfm["负荷值"], bins=20, range=(0, 1000))
print("hist 20 柱总频数:", int(counts.sum()), "← 必须等于行数，一根都不能丢")
close_all()

assert int(counts.sum()) == len(dfm) == 260, "range 内数据一行不丢"
assert is_png(png_bytes(fig2))
close_all()
'''

S9C_CODE = '''# 图 3：台区 A_01 的日均负荷走势（时序线图）
load_curve = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"])
st = (load_curve[load_curve["台区编号"] == "STATION_A_01"]
      .sort_values("时间戳").set_index("时间戳"))

daily = st["负荷值"].resample("D").mean()
print("daily:", daily.shape, "| NaN:", int(daily.isna().sum()),
      "| min:", round(float(daily.min()), 2), "| max:", round(float(daily.max()), 2))

hol_mean = float(st.loc[st["是否节假日"], "负荷值"].mean())
work_mean = float(st.loc[~st["是否节假日"], "负荷值"].mean())
print("节假日日均:", round(hol_mean, 3), "| 工作日日均:", round(work_mean, 3))

fig3, ax3 = plt.subplots(figsize=(8, 3))
daily.plot(ax=ax3, title="STATION_A_01 日均负荷走势")
fig3.tight_layout()
assert is_png(png_bytes(fig3))
close_all()

assert daily.shape == (90,) and int(daily.isna().sum()) == 0
assert round(float(daily.min()), 2) == 428.33
assert round(float(daily.max()), 2) == 784.83
assert round(hol_mean, 3) == 513.516 and round(work_mean, 3) == 678.735
'''

S10_CODE = '''# 汇总全流程关键数字，交付一份可入库的分析报告
report = pd.DataFrame({
    "指标": ["原始行数", "清洗后行数", "负荷值缺失(原始)", "哨兵值", "日期解析成功",
             "状态类别(前/后)", "缺陷类别(前/后)", "危急记录", "高风险记录", "台账缺失台区"],
    "数值": [270, len(dfm), 21, n_sent, int(dfm["发现日期"].notna().sum()),
             "8 -> 3", "10 -> 6", int((dfm["缺陷等级"] == "危急").sum()),
             int((dfm["风险等级"] == "高风险").sum()), 2],
})
print(report.to_string(index=False))

assert report["数值"].tolist() \\
    == [270, 260, 21, 11, 260, "8 -> 3", "10 -> 6", 30, 53, 2]
print("全流程交付完成：raw(270,13) -> dfm(260,18) + 三张图 + 一份报告")
'''


# --------------------------------------------------------------------------- #
# 练习版各阶段代码（@@todo 挖空标记 + 真实解法，练习版由 nb_builder 自动挖空）
# --------------------------------------------------------------------------- #

E0_CODE = '''# @@todo(1) 用默认编码读 GBK 文件必炸：用 probe 把这次失败调用接住
# @@hint probe(函数, 参数) 返回 (状态, 异常类型名)
status, msg = probe(pd.read_csv, DATA / "device_defects_gbk.csv")
# @@end

# @@todo(2) 分别用 encoding="gbk" 与 encoding="utf-8-sig" 读入 gbk 与 raw
# @@hint pd.read_csv(路径, encoding=...)
gbk = pd.read_csv(DATA / "device_defects_gbk.csv", encoding="gbk")
raw = pd.read_csv(DATA / "device_defects_raw.csv", encoding="utf-8-sig")
# @@end

print("utf-8 读 GBK 文件 →", status, "|", msg)
print("gbk 编码读取成功:", gbk.shape)
print("raw:", raw.shape, "| 负荷值 dtype:", raw["负荷值"].dtype)

# 负荷值列缺失对账：自动识别 14 个 vs 漏网的 "—" 7 个 = 真缺失 21 个
print("自动识别缺失:", int(raw["负荷值"].isna().sum()),
      "| 漏网的 -- :", int((raw["负荷值"] == "—").sum()),
      "| 含 kW:", int(raw["负荷值"].str.contains("kW").sum()),
      "| 含千分位:", int(raw["负荷值"].str.contains(",").sum()))

assert status == "err" and msg == "UnicodeDecodeError", "utf-8 读 GBK 必须报错"
assert gbk.shape == (270, 13)
assert raw.shape == (270, 13)
assert pd.api.types.is_string_dtype(raw["负荷值"]), "raw 的负荷值还是字符串"
assert int(raw["负荷值"].isna().sum()) == 14, "N/A 与空串被自动识别"
assert int((raw["负荷值"] == "—").sum()) == 7, "-- 不会被自动识别"
assert int(raw["负荷值"].str.contains("kW").sum()) == 5
assert int(raw["负荷值"].str.contains(",").sum()) == 7
'''

E1_CODE = '''# @@todo(3) 把负荷值从字符串清洗成数值：去 kW 单位、去千分位，再转数值让脏值落水
# @@hint str.replace(..., regex=False) 链式 + pd.to_numeric(errors="coerce")
load = (raw["负荷值"].astype(str)
        .str.replace(" kW", "", regex=False)
        .str.replace(",", "", regex=False))
df = raw.assign(负荷值=pd.to_numeric(load, errors="coerce"))
# @@end

print("清洗后 dtype:", df["负荷值"].dtype, "| 缺失:", int(df["负荷值"].isna().sum()))
print(">=1000:", int((df["负荷值"] >= 1000).sum()), "(9999 x4 + 5000 x3)",
      "| <0:", int((df["负荷值"] < 0).sum()), "(-1 x4)")

assert df["负荷值"].dtype == np.dtype("float64")
assert int(df["负荷值"].isna().sum()) == 21, "14 自动识别 + 7 个 -- 全部落水"
assert int((df["负荷值"] >= 1000).sum()) == 7
assert int((df["负荷值"] == 9999).sum()) == 4
assert int((df["负荷值"] == 5000).sum()) == 3
assert int((df["负荷值"] == -1).sum()) == 4
'''

E2_CODE = '''# @@todo(4) 统一 4 种日期格式：先把中文「年月日」替换掉，再用 format="mixed" 一次解析
# @@hint str.replace 三连 + pd.to_datetime(format="mixed")
d = (df["发现日期"].astype(str)
     .str.replace("年", "-", regex=False)
     .str.replace("月", "-", regex=False)
     .str.replace("日", "", regex=False))
df["发现日期"] = pd.to_datetime(d, format="mixed")
# @@end

print("NaT:", int(df["发现日期"].isna().sum()),
      "| 范围:", df["发现日期"].min(), "~", df["发现日期"].max())

assert int(df["发现日期"].isna().sum()) == 0, "4 种格式全部解析成功"
assert df["发现日期"].min() == pd.Timestamp("2026-01-02")
assert df["发现日期"].max() == pd.Timestamp("2026-06-30")
'''

E3_CODE = '''# @@todo(5) 标准化处理状态：去空白 + 同义合并，把 8 种写法收编成 3 类
# @@hint str.strip() 后用 replace(映射表)；映射表见讲解版阶段 3
df["处理状态"] = (df["处理状态"].astype(str).str.strip()
                 .replace({"已 处理": "已处理", "处理完成": "已处理", "done": "已处理",
                           "处理中 ": "处理中", "待处理": "未处理"}))
# @@end

print("状态类别:", df["处理状态"].nunique(),
      df["处理状态"].value_counts().to_dict())

# @@todo(6) 标准化缺陷类型：先去空白，再合并「渗油/漏油/异物搭挂」等同义写法
# @@hint 先 str.strip()，再 replace({"渗油": "渗漏油", ...})
df["缺陷类型"] = df["缺陷类型"].astype(str).str.strip()
df["缺陷类型"] = df["缺陷类型"].replace({"渗油": "渗漏油", "漏油": "渗漏油", "异物搭挂": "异物"})
# @@end

print("缺陷类别:", df["缺陷类型"].nunique(),
      df["缺陷类型"].value_counts().to_dict())

assert df["处理状态"].nunique() == 3
assert int((df["处理状态"] == "已处理").sum()) == 171
assert int((df["处理状态"] == "处理中").sum()) == 54
assert int((df["处理状态"] == "未处理").sum()) == 45
assert df["缺陷类型"].nunique() == 6
assert int((df["缺陷类型"] == "渗漏油").sum()) == 61
assert int((df["缺陷类型"] == "异物").sum()) == 40
'''

E4_CODE = '''# @@todo(7) 第一击：删除完全重复行，结果存入 d1
# @@hint DataFrame.drop_duplicates() 默认按所有列
d1 = df.drop_duplicates()
# @@end
print("全列去重:", d1.shape)

# 业务键：记录ID可以不同，其余 7 列相同就是同一条业务记录
BIZ_KEY = ["台区编号", "线路名称", "设备类型", "设备编号", "缺陷类型", "缺陷等级", "发现日期"]

# @@todo(8) 第二击：按业务键去重并 reset_index，结果存入 d2
# @@hint drop_duplicates(subset=BIZ_KEY) + reset_index(drop=True)
d2 = d1.drop_duplicates(subset=BIZ_KEY).reset_index(drop=True)
# @@end
print("业务键去重:", d2.shape)

assert d1.shape == (264, 13), "完全重复 6 行"
assert d2.shape == (260, 13), "业务键重复再删 4 行"
assert d2.index.tolist() == list(range(260)), "reset_index 让后续索引连续"
'''

E5_CODE = '''# 哨兵先记账：负荷值 >=1000 或 <0 的行数
n_sent = int(((d2["负荷值"] >= 1000) | (d2["负荷值"] < 0)).sum())

# @@todo(9) 把哨兵值置为 NaN：用 mask 让 >=1000 或 <0 的位置落水
# @@hint Series.mask(条件) 条件为 True 的位置变 NaN
d2["负荷值"] = d2["负荷值"].mask((d2["负荷值"] >= 1000) | (d2["负荷值"] < 0))
# @@end
n_load_na = int(d2["负荷值"].isna().sum())
print("哨兵值:", n_sent, "| 置 NaN 后缺失:", n_load_na)

# 湿度越界先记账：物理范围 [0, 100] 之外共 5 个
n_hum = int(((d2["湿度"] > 100) | (d2["湿度"] < 0)).sum())

# @@todo(10) 湿度越界截断：clip 到物理范围 [0, 100]
# @@hint Series.clip(lower, upper)
d2["湿度"] = d2["湿度"].clip(0, 100)
# @@end
print("湿度越界:", n_hum, "| clip 后越界:", int(((d2["湿度"] > 100) | (d2["湿度"] < 0)).sum()))

# 温度：MAR 缺失（仅 C/D 台区），用全局中位数填充
temp_med = float(d2["温度"].median())

# @@todo(11) 温度缺失用中位数 temp_med 填充
# @@hint Series.fillna(temp_med)
d2["温度"] = d2["温度"].fillna(temp_med)
# @@end
print("温度缺失 23 -> 中位数填充:", temp_med)

# @@todo(12) 负荷值缺失用「设备类型分组中位数」回填：先 transform 再 fillna
# @@hint groupby("设备类型")["负荷值"].transform("median") 返回与原表逐行对齐的序列
grp_med = d2.groupby("设备类型")["负荷值"].transform("median")
d2["负荷值"] = d2["负荷值"].fillna(grp_med)
# @@end
print("填充后缺失:", int(d2["负荷值"].isna().sum()))
print("填充后 mean:", round(float(d2["负荷值"].mean()), 4),
      "| median:", round(float(d2["负荷值"].median()), 4),
      "| max:", round(float(d2["负荷值"].max()), 4))

assert n_sent == 11 and n_load_na == 32, "21 原缺失 + 11 哨兵"
assert n_hum == 5 and int(((d2["湿度"] > 100) | (d2["湿度"] < 0)).sum()) == 0
assert temp_med == 21.9
assert int(d2["温度"].isna().sum()) == 0
assert int(d2["负荷值"].isna().sum()) == 0
assert round(float(d2["负荷值"].mean()), 4) == 175.3413
assert round(float(d2["负荷值"].median()), 4) == 88.695
assert round(float(d2["负荷值"].max()), 4) == 588.45, "截断后最大值是尖峰填充的结果"
'''

E6_CODE = '''# @@todo(13) 读入台账 device_info.csv，并与 d2 按台区编号左连接，结果存入 dfm
# @@hint pd.read_csv(...) + d2.merge(info, on="台区编号", how="left")
info = pd.read_csv(DATA / "device_info.csv")
dfm = d2.merge(info, on="台区编号", how="left")
# @@end

# 先记账再填充：缺键不报错不警告，只是安静地 NaN
n_info_na = int(dfm["所属供电所"].isna().sum())
na_stations = sorted(dfm.loc[dfm["所属供电所"].isna(), "台区编号"].unique().tolist())
print("info:", info.shape, "| merge 后:", dfm.shape,
      "| 所属供电所缺失:", n_info_na)
print("缺失台区:", na_stations)

# @@todo(14) 字符列缺失用明确标记填充：所属供电所 -> "未知"
# @@hint Series.fillna("未知")
dfm["所属供电所"] = dfm["所属供电所"].fillna("未知")
# @@end
print("填充后各供电所记录数:", dfm["所属供电所"].value_counts().to_dict())

assert info.shape == (16, 5)
assert dfm.shape == (260, 17), "13 + 5 台账列 - 1 连接键 = 17"
assert n_info_na == 23, "不报错不警告，只是安静地 NaN"
assert na_stations == ["STATION_B_03", "STATION_F_02"]
assert int((dfm["所属供电所"] == "未知").sum()) == 23
'''

E7_CODE = '''# @@todo(15) 透视表：设备类型 x 缺陷等级 的平均负荷，结果存入 pt
# @@hint pivot_table(index=..., columns=..., values="负荷值", aggfunc="mean")
pt = dfm.pivot_table(index="设备类型", columns="缺陷等级", values="负荷值", aggfunc="mean")
# @@end
print("pivot:", pt.shape)
print(pt.round(4))

# @@todo(16) 命名聚合：按设备类型统计 记录数/危急数/平均负荷，再算危急占比，结果存入 rep
# @@hint groupby("设备类型").agg(记录数=("记录ID", "count"), ...) + lambda 计危急数
rep = (dfm.groupby("设备类型")
       .agg(记录数=("记录ID", "count"),
            危急数=("缺陷等级", lambda s: int((s == "危急").sum())),
            平均负荷=("负荷值", "mean")))
rep["危急占比"] = rep["危急数"] / rep["记录数"]
# @@end
print(rep.round(4))

done_rate = float((dfm["处理状态"] == "已处理").mean())
print("全局已处理率:", round(done_rate, 4))

assert pt.shape == (5, 3), "5 种设备 x 3 个等级"
assert round(float(pt.loc["变压器", "危急"]), 4) == 528.394
assert rep.shape == (5, 4)
assert int(rep["危急数"].sum()) == 30, "全表危急 30 条"
assert int(rep.loc["避雷器", "危急数"]) == 12, "避雷器危急最多"
assert round(float(rep.loc["避雷器", "危急占比"]), 4) == 0.2143
assert round(done_rate, 4) == 0.6308
'''

E8_CODE = '''# @@todo(17) 分箱：负荷值按 [0, 50, 150, 300, 1000] 分四档，标签 极低/低/中/高
# @@hint pd.cut(dfm["负荷值"], [0, 50, 150, 300, 1000], labels=[...])
bins = pd.cut(dfm["负荷值"], [0, 50, 150, 300, 1000], labels=["极低", "低", "中", "高"])
# @@end
print("cut 计数:", bins.value_counts().reindex(["极低", "低", "中", "高"]).to_dict(),
      "| NaN:", int(bins.isna().sum()))

# @@todo(18) 风险分级：np.select 两级条件（>300 高风险、>150 中风险），写入 风险等级 列
# @@hint np.select([条件1, 条件2], ["高风险", "中风险"], default="低风险")
risk = np.select([dfm["负荷值"] > 300, dfm["负荷值"] > 150],
                 ["高风险", "中风险"], default="低风险")
dfm["风险等级"] = risk
# @@end
print("风险分级:", pd.Series(risk).value_counts().to_dict())

ct = pd.crosstab(dfm["设备类型"], dfm["风险等级"])
print("crosstab:", ct.shape)
print(ct)

assert int(bins.isna().sum()) == 0, "负荷值已填充，分箱不应再有 NaN"
assert bins.value_counts().reindex(["极低", "低", "中", "高"]).tolist() == [62, 109, 36, 53]
assert pd.Series(risk).value_counts().to_dict() == {"低风险": 171, "高风险": 53, "中风险": 36}
assert int(((dfm["风险等级"] == "高风险") & (dfm["缺陷等级"] == "危急")).sum()) == 5
assert ct.shape == (5, 3)
'''

E9A_CODE = '''# 图 1：各设备类型危急缺陷数（柱状）
fig1, ax1 = plt.subplots(figsize=(6, 3.5))

# @@todo(19) 用 rep 画各设备类型危急缺陷数的柱状图到 ax1 上
# @@hint rep["危急数"].plot(kind="bar", ax=ax1, title=...)
rep["危急数"].plot(kind="bar", ax=ax1, title="各设备类型危急缺陷数")
# @@end

fig1.tight_layout()
data1 = png_bytes(fig1)
print("bar 图 PNG 字节数:", len(data1), "| 魔数合法:", is_png(data1))
close_all()

assert is_png(data1), "savefig 真的产出了 PNG"
'''

E9B_CODE = '''# 图 2：负荷值分布（直方图，range 压住长尾）
fig2, ax2 = plt.subplots(figsize=(6, 3.5))

# @@todo(20) 画清洗后负荷值的直方图：bins=20，range=(0, 1000)
# @@hint dfm["负荷值"].plot(kind="hist", bins=..., range=..., ax=ax2)
dfm["负荷值"].plot(kind="hist", bins=20, range=(0, 1000), ax=ax2, title="清洗后负荷值分布")
# @@end

fig2.tight_layout()
counts, _ = np.histogram(dfm["负荷值"], bins=20, range=(0, 1000))
print("hist 20 柱总频数:", int(counts.sum()), "← 必须等于行数，一根都不能丢")
close_all()

assert int(counts.sum()) == len(dfm) == 260, "range 内数据一行不丢"
assert is_png(png_bytes(fig2))
close_all()
'''

E9C_CODE = '''# 图 3：台区 A_01 的日均负荷走势（时序线图）
load_curve = pd.read_csv(DATA / "load_curve.csv", parse_dates=["时间戳"])
st = (load_curve[load_curve["台区编号"] == "STATION_A_01"]
      .sort_values("时间戳").set_index("时间戳"))

# @@todo(21) 对 A_01 的负荷值按日重采样取均值，结果存入 daily
# @@hint st["负荷值"].resample("D").mean()
daily = st["负荷值"].resample("D").mean()
# @@end
print("daily:", daily.shape, "| NaN:", int(daily.isna().sum()),
      "| min:", round(float(daily.min()), 2), "| max:", round(float(daily.max()), 2))

# 节假日 / 工作日的时段均值，交叉验证时序规律
hol_mean = float(st.loc[st["是否节假日"], "负荷值"].mean())
work_mean = float(st.loc[~st["是否节假日"], "负荷值"].mean())
print("节假日日均:", round(hol_mean, 3), "| 工作日日均:", round(work_mean, 3))

fig3, ax3 = plt.subplots(figsize=(8, 3))

# @@todo(22) 把 daily 画成折线图到 ax3 上
# @@hint daily.plot(ax=ax3, title=...)
daily.plot(ax=ax3, title="STATION_A_01 日均负荷走势")
# @@end

fig3.tight_layout()
assert is_png(png_bytes(fig3))
close_all()

assert daily.shape == (90,) and int(daily.isna().sum()) == 0
assert round(float(daily.min()), 2) == 428.33
assert round(float(daily.max()), 2) == 784.83
assert round(hol_mean, 3) == 513.516 and round(work_mean, 3) == 678.735
'''

E10_CODE = '''# @@todo(23) 汇总全流程关键数字，生成 report DataFrame（指标列 + 数值列，见讲解版阶段 10）
# @@hint pd.DataFrame({"指标": [...], "数值": [...]})，数值从前面各步骤的变量取
report = pd.DataFrame({
    "指标": ["原始行数", "清洗后行数", "负荷值缺失(原始)", "哨兵值", "日期解析成功",
             "状态类别(前/后)", "缺陷类别(前/后)", "危急记录", "高风险记录", "台账缺失台区"],
    "数值": [270, len(dfm), 21, n_sent, int(dfm["发现日期"].notna().sum()),
             "8 -> 3", "10 -> 6", int((dfm["缺陷等级"] == "危急").sum()),
             int((dfm["风险等级"] == "高风险").sum()), 2],
})
# @@end
print(report.to_string(index=False))

assert report["数值"].tolist() \\
    == [270, 260, 21, 11, 260, "8 -> 3", "10 -> 6", 30, 53, 2]
print("全流程交付完成：raw(270,13) -> dfm(260,18) + 三张图 + 一份报告")
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# final_01 综合题：全流程清洗 + 探索 + 可视化 + 分析报告

> 方向：数据预处理 ｜ 竞赛对应：**数据探索 / 数据清洗 / 数据可视化是独立计分项**
> + **「数据准备及处理」占技能操作权重 10%**。综合题就是把这些分一次拿满的载体。

前面 18 章每章只打一个点，综合题考的是**串联**：给一份真实导出的脏数据
（`device_defects_raw.csv`，270 行 x 13 列），从读取到交付一条龙跑完。
全程贯穿着前 18 章埋的坑：

| 阶段 | 坑 | 出处 |
|---|---|---|
| 0 读取 | GBK 编码炸 utf-8；`—` 不会被自动识别成 NaN | ch02 / ch06 |
| 1 数值清洗 | `kW` 单位、千分位、3 种缺失占位符混在一列 | ch06 |
| 2 日期 | 4 种格式混排（含中文日期），`format="mixed"` 一次解析 | ch06 |
| 3 类别 | 同义写法 8 -> 3、10 -> 6 | ch06 |
| 4 去重 | 全列去重与业务键去重是两件事 | ch05 |
| 5 异常 | 哨兵先记账再置 NaN；分组中位数填充用 `transform` | ch04 / ch05 / ch11 |
| 6 合并 | left join 缺台账**不报错**，安静地 NaN | ch13 |
| 7 聚合 | pivot_table 与命名聚合对账 | ch10 |
| 8 分箱 | cut 越界静默 NaN；np.select 条件从上到下 | ch09 / ch07 |
| 9 可视化 | hist 要对账；中文字体与负号 | ch18 |

> 本章所有 `assert` 的真值都沉淀在方向 README 的「final 专项真值」表里。
"""
    ),
    md(
        """
## 一、学习目标与流水线全景

**学习目标**：拿到任意一份脏 CSV，能在 30 分钟内交付
「一份干净表 + 三张图 + 一份关键指标报告」，且每一步的数字都对得上账。

```
raw(270,13)──读取──> 负荷值转数值 ──> 日期统一 ──> 类别标准化
   │                                                    │
   └──── 全列去重(264) ──> 业务键去重(260) <─────────────┘
                              │
              哨兵置NaN -> 分组中位数填充 <- 湿度clip / 温度填充
                              │
                  merge台账(260,17) -> 聚合画像 + 风险分箱
                              │
                  三张图(bar/hist/line) + 最终报告
```

## 二、API 速查表

| 方法 / 参数 | 关键点 | 一句话说明 |
|---|---|---|
| `read_csv(encoding="gbk")` | utf-8 读 GBK 必炸 | 编码错是最先撞上的墙 |
| `read_csv(na_values=...)` | 默认只认部分占位符 | `—` 不会自动变 NaN |
| `str.replace(..., regex=False)` | 字面量替换 | 去单位 / 千分位 |
| `pd.to_numeric(errors="coerce")` | 非法值 -> NaN | 字符串转数值的兜底 |
| `pd.to_datetime(format="mixed")` | 多格式混排 | 中文日期要先预处理 |
| `drop_duplicates(subset=...)` | 全列 vs 业务键 | 两连击各删各的 |
| `Series.mask(cond)` | 条件为 True -> NaN | 哨兵值先记账再下水 |
| `groupby(...).transform("median")` | 逐行对齐 | 填充必须 transform 不能 agg |
| `merge(how="left")` | 缺键安静 NaN | 合并后必查 isna |
| `pivot_table` / 命名聚合 | 两种聚合口径 | 结果必须互相咬合 |
| `pd.cut` / `np.select` | 越界 NaN / 条件顺序 | 分箱与分级互为对照 |
| `df.plot` + `savefig` | BytesIO 可验证 | 图不落盘也能验真伪 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 阶段 0：读取与编码——最先撞上的墙

GBB 文件用 utf-8 读直接 `UnicodeDecodeError`；而真正的脏点藏在负荷值列：
**`N/A` 与空串会被 pandas 自动识别成 NaN，`—` 不会**——默认 na_values
不认识长破折号，它就混在字符串里漏网。
"""
    ),
    code(S0_CODE),
    md(
        """
### 难点串联：缺失数对账是清洗的第一动作

**为什么难**：`read_csv` 不报错 ≠ 读对了。占位符漏识别、编码错位、
首列粘 BOM，全都**安静地**进入内存。

**判定规则**：读进任何脏表，先做三件事——`shape`、`dtypes`、
**每列 `isna().sum()` 与业务预期对账**。本例对账结果：14 自动 + 7 漏网 = 21，
和 ch03/ch04 章的真值完全咬合。

## 阶段 1：负荷值——从字符串到数值

三种脏形态：`123.45 kW`（单位）、`1,234.50`（千分位）、`N/A / — / 空串`
（占位符）。处理顺序：**去单位 -> 去千分位 -> `to_numeric(errors="coerce")`**，
`coerce` 把残留的 `—` 一并落水成 NaN，缺失数从 14 涨到 21。
"""
    ),
    code(S1_CODE),
    md(
        """
**为什么这样做**：`errors="coerce"` 是字符串转数值的兜底动作——与其让
`to_numeric` 在脏值上炸掉，不如让它把脏值统一变成 NaN，交给下游统一处理。
缺失从 14 -> 21 的增量本身就是一层对账。

## 阶段 2：日期——4 种格式混排

`2026-05-03` / `2026/05/03` / `05/03/2026` / `2026年5月3日`。
中文日期任何 format 都不认，先替换 `年月日`，再 `format="mixed"` 一次收编。
"""
    ),
    code(S2_CODE),
    md(
        """
**为什么这样做**：直接 `to_datetime(raw)` 会 `DateParseError`；
`format="mixed"` 能吃多种格式，但中文日期仍要预处理。解析成功的验收
标准是 `isna().sum() == 0`——任何一行 NaT 都意味着格式没猜全。

## 阶段 3：类别标准化——8 -> 3、10 -> 6

处理状态的 8 种写法（`处理中␣` / `已␣处理` / `处理完成` / `done` / `待处理`…）
与缺陷类型的 10 种写法，靠 **strip + 同义映射表** 收编。
"""
    ),
    code(S3_CODE),
    md(
        """
**为什么这样做**：不做这步，`groupby` 会把 `已处理` 和 `done` 当成两类，
后面的聚合报表直接失真。映射表要显式写出来——竞赛评分时它就是你的
「数据字典」，说明你知道每种脏写法对应哪个正类。

## 阶段 4：去重两连击

完全重复 6 行（`drop_duplicates` 全列），业务键重复再删 4 行
（`记录ID` 不同但业务上是同一条）。
"""
    ),
    code(S4_CODE),
    code(S5_CODE),
    md(
        """
### 难点串联：填充为什么必须 `transform` 而不是 `agg`

**为什么难**：`agg("median")` 返回每组一个数的**窄表**，拿它去 `fillna`
会对不齐行；`transform("median")` 返回**与原表逐行对齐**的序列，这才是
「组内中位数回填」的正确形状。

**判定规则**：**凡是「组统计量回填到组内每一行」，一律 `transform`**；
`agg` 是做报表用的。

**为什么先 mask 后填充**：哨兵（9999 / -1 / 5000）如果不先置 NaN，
中位数会被它拉偏；先记账（`n_sent`）再下水，缺失数 21 + 11 = 32 也是一层对账。

## 阶段 6：合并台账——left join 的安静 NaN

台账只有 16 个台区，缺陷表有 18 个：`STATION_B_03`、`STATION_F_02`
在台账里不存在。`how="left"` **不报错、不警告**，23 行记录的台账列安静地 NaN。
"""
    ),
    code(S6_CODE),
    md(
        """
**为什么这样做**：合并完必查 `isna().sum()`——这是 ch13 的核心考点。
字符列的缺失用 `"未知"` 这种明确标记，不要用 0 / -1 硬填，否则下游
统计口径又被污染。

## 阶段 7：聚合画像——两种口径必须咬合

`pivot_table` 给「设备类型 x 缺陷等级」的平均负荷矩阵；命名聚合给
「记录数 / 危急数 / 平均负荷 / 危急占比」的业务画像。变压器-危急 = 528.394
与 ch10 在 270 行原始数据上的真值**不同**（本表是清洗后的 260 行），
这正是「对账要看口径」的活教材。
"""
    ),
    code(S7_CODE),
    md(
        """
## 阶段 8：分箱与风险分级

`pd.cut` 四档分箱——清洗后负荷值无缺失、无越界（min 5 以上 / max 588.45），
所以分箱 **NaN 必须是 0**；这是 ch09「越界静默 NaN」考点的反向验收。
`np.select` 从上到下命中条件，产出高 / 中 / 低三级，`crosstab` 交叉验证。
"""
    ),
    code(S8_CODE),
    md(
        """
### 难点串联：cut 的 NaN 是验收信号不是噪音

**为什么难**：ch09 里 `cut` 越界值静默 NaN 曾坑过对账；这里反着用——
**前置清洗做得干净，分箱 NaN 就必须是 0**。它从坑变成了验收断言。

**判定规则**：`cut` / `qcut` 之后第一件事永远是 `isna().sum()`：
不为 0 时，要么补边界（`-np.inf, np.inf`），要么回头查清洗。

## 阶段 9：可视化三件套——柱 / 直方 / 时序线

每张图都做 PNG 魔数验证（`\\x89PNG`），不落盘也能证明图真的画出来了。
直方图做频数对账：`range=(0, 1000)` 内 20 柱总频数必须等于行数 260。
"""
    ),
    code(S9A_CODE),
    code(S9B_CODE),
    code(S9C_CODE),
    md(
        """
**为什么这样做**：竞赛的可视化是独立计分项——图出得来是底线，
**标题 / 轴标签 / 中文不方框 / 图存得下来**才是满分动作。BytesIO +
PNG 魔数是「存得下来」的可验证版本。

## 阶段 10：最终报告——把关键数字钉死

把全流程的对账数字汇总成一张报告表。这张表既是交付物，
也是你答辩时「知道自己每一步做了什么」的证据链。
"""
    ),
    code(S10_CODE),
    md(
        """
---

## 三、易错点清单

1. utf-8 读 GBK 文件直接 `UnicodeDecodeError`——编码是读取的第一道对账。
2. `read_csv` 默认 na_values 不认 `—`，缺失数对账不做，7 个脏值混进字符串列。
3. 字符串转数值用 `to_numeric(errors="coerce")`，让脏值统一落水而不是炸掉。
4. 中文日期必须先替换 `年月日` 再 `format="mixed"`，且验收 `isna()==0`。
5. 类别不标准化，聚合报表直接失真（`已处理` 和 `done` 被当两类）。
6. 全列去重与业务键去重是两件事，先后各删 6 行、4 行。
7. 哨兵不先置 NaN 就算中位数，填充基准全被拉偏。
8. 组内填充必须 `transform`，`agg` 形状对不上。
9. left join 缺键**不报错**，合并后必查 `isna().sum()`。
10. `cut` 之后必查 NaN——清洗干净时分箱 NaN 必须为 0。

## 四、本章小结

- 综合题的得分逻辑是**流程完整性**：每一步有小验收，全流程有总对账。
- 清洗五连：读 -> 转 -> 析 -> 标 -> 重，缺一步后面全歪。
- 对账是主旋律：缺失对账、去重对账、填充对账、分箱对账、频数对账。
- 交付物三件：干净表 + 三张图 + 关键指标报告。

### 复盘提问

1. `—` 为什么不会被自动识别成 NaN？缺失数从 14 到 21 增加在哪一步？
2. 为什么填充用 `transform("median")` 而不是 `agg`？
3. merge 后怎么发现台账缺了 2 个台区？缺失为什么是 23 行而不是 2 行？
4. 清洗后 `cut` 的 NaN 为什么必须是 0？什么时候它不为 0？
5. hist 的 `range=(0, 1000)` 里 20 柱总频数必须等于多少？为什么？

答不上来的，回对应章节重看难点深挖，再到 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# final_01 练习：全流程综合题

与讲解版逐 Cell 对应。整条流水线被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即交付通过。卡住就回讲解版看对应阶段。
"""
    ),
    md(
        """
## 一、考点回顾

| 阶段 | 覆盖章节 | 核心动作 |
|---|---|---|
| 读取 | ch02 / ch06 | 编码、na_values、dtype 对账 |
| 清洗 | ch06 / ch05 | 字符串转数值、日期、类别标准化、去重 |
| 补全 | ch04 / ch05 / ch11 | 哨兵置 NaN、clip、分组中位数 transform 填充 |
| 合并 | ch13 | left join + 缺键排查 |
| 分析 | ch07 / ch09 / ch10 | 聚合画像、分箱、风险分级 |
| 交付 | ch18 | 三张图 + 报告 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：读取与编码对账（对应讲解 阶段 0）"),
    code(E0_CODE),
    md(
        """
**为什么这样做**：读脏数据先做缺失对账——14 自动识别 + 7 个 `—` = 21，
这组数字后面每一步都要用到。
"""
    ),
    md("## 题 2：负荷值从字符串到数值（对应讲解 阶段 1）"),
    code(E1_CODE),
    md("## 题 3：日期 4 格式统一（对应讲解 阶段 2）"),
    code(E2_CODE),
    md("## 题 4：类别标准化 8->3、10->6（对应讲解 阶段 3）"),
    code(E3_CODE),
    md("## 题 5：去重两连击（对应讲解 阶段 4）"),
    code(E4_CODE),
    md(
        """
**为什么这样做**：先全列去重再业务键去重，顺序反了会把「记录ID不同
但内容相同」的行留下来——业务键才是判断重复的业务口径。
"""
    ),
    md("## 题 6：哨兵、越界与分组填充（对应讲解 阶段 5）"),
    code(E5_CODE),
    md("## 题 7：合并台账与缺键排查（对应讲解 阶段 6）"),
    code(E6_CODE),
    md("## 题 8：聚合画像（对应讲解 阶段 7）"),
    code(E7_CODE),
    md("## 题 9：分箱与风险分级（对应讲解 阶段 8）"),
    code(E8_CODE),
    md("## 题 10：可视化三件套（对应讲解 阶段 9）"),
    code(E9A_CODE),
    code(E9B_CODE),
    code(E9C_CODE),
    md(
        """
**为什么这样做**：每张图都过 `png_bytes` + PNG 魔数验证，hist 做频数
对账（260 行一根不丢），line 用节假日 / 工作日均值交叉验证时序规律。
"""
    ),
    md("## 题 11：最终报告（对应讲解 阶段 10）"),
    code(E10_CODE),
    md(
        """
---

## 综合自查

1. 题 1 的缺失对账数字（14 + 7）如果对不上，后面哪几个 assert 会连锁崩？
2. 题 6 为什么必须先 mask 哨兵再填充？直接填充会怎样？
3. 题 7 的 23 行缺失是怎么算出来的？（提示：不是 2）
4. 题 9 的 `cut` NaN 必须为 0——它在替你验收前面哪一步？
5. 题 11 报告里的「高风险记录 53」和「危急记录 30」为什么不相等？

全部对得上账，pandas 方向 19/19 交付完成。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
