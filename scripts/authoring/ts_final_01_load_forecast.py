#!/usr/bin/env python3
"""final_01 —— 台区负荷日前预测全流程（清洗 → 特征 → 基线 → 模型 → 多步 → 评估 → 报告）

这是 `05_timeseries` 的收尾综合题：把 ch01~ch08 的零件串成一条能交付的流水线。
不做新算法，全部考点都在「顺序」「口径」「归因」上。

主线七段：

1. **清洗**：按小时分组的中位数 ×3 判异常 —— 全局阈值会把工业台区的日间峰整段误杀
2. **特征**：日历 sin/cos + 温度 V 形（`|T−22|`）+ 滞后 + 滑窗（一律先 `shift(1)`）
3. **切分与泄漏自检**：按时间 8:2；再用「手工切片」核对 `ma24` / `sd24` / `lag1` 的口径
4. **四条朴素基线**：持续性 30.2482 / 日周期 21.5898 / 周周期 19.1805 / 周周期×水平比 21.6910
   —— **周周期赢日周期**（测试段落在 10~12 月，星期对齐比「离得近」更重要）
5. **学习模型**：Ridge 14.2251、LightGBM 12.6689（R² 0.9386），重要性 top1 = `lag24`
6. **日前 24 步**（每天 0 点发布，70 天）：直接法 16.1107 / 递归法 13.1118 / 日周期基线 21.6100
   —— **反直觉结论**：递归法的误差累积只有 **+0.0138 kW**（占 0.46%），
   而「特征是否对齐到目标时刻」值 **3.01 kW**。教材里「递归必然崩」在这个 horizon 上不成立
7. **归因 + 报告**：用 teacher forcing 把「特征集差」与「误差累积」拆开，再自动生成结题报告

另含一个负结果：**LSTM 归一化前后差 23.5 倍**（409.7550 → 17.4668），
但调完参仍然输给 LightGBM —— 小样本 + 强周期结构上，手工特征 + GBDT 更划算。

数据：`../data/load_curve.csv`（2 台区 × 8760 h，故意埋脏）
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python coding/05_timeseries/data/make_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries" / "final"
NAME = "final_01_load_forecast"

# =========================================================================== #
# 1. 公共导入 / 脚手架（两版都给，不挖空）                                     #
# =========================================================================== #

IMPORTS = '''import os

# ⚠️ 本机 lightgbm 用 libomp、torch 用 libiomp，两套 OpenMP 同时开多线程会互相踩内存，
# 直接 SIGSEGV（try/except 抓不住、报错也看不到）。必须在任何 import 之前把线程数钉死。
os.environ["OMP_NUM_THREADS"] = "1"

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA = Path("../data")          # notebook 的 cwd = coding/05_timeseries/final
CSV = DATA / "load_curve.csv"

H = 24                          # 预测视野：未来 24 小时（日前预测）
SEED = 42
STATION = "STATION_A_01"        # 居民型：晚高峰 + 早次峰
OTHER = "STATION_B_02"          # 工业型：日间平台 + 周末腰斩
N_HEAD = 168                    # 滑窗最长 168 小时，特征表头部会有 168 行 NaN

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 40)
print("numpy", np.__version__, "| pandas", pd.__version__, "| lightgbm", lgb.__version__)
'''

SCAFFOLD = '''# --------------------------------------------------------------------------- #
# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
# --------------------------------------------------------------------------- #

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

# 多步预测用的「精简特征集」：只保留「目标时刻前一小时就能算出来」的量
RED_COLS = ["h_sin", "h_cos", "d_sin", "d_cos", "m_sin", "m_cos",
            "is_weekend", "is_holiday", "T", "T_dev", "T2",
            "lag24", "lag48", "lag168", "ma24", "ma168", "sd24"]


def cal_block(idx, prefix=""):
    """把一组时间戳变成日历特征。

    `prefix` 用来给「目标时刻」的日历列加前缀：原点 t 的日历列叫 `h_sin`，
    目标 t+h 的日历列叫 `tgt_h_sin`。不加前缀直接拼表会**列名重名**，
    LightGBM 会直接抛 `Feature (h_sin) appears more than one time`。
    """
    d = {
        "h_sin": np.sin(2 * np.pi * idx.hour.to_numpy(float) / 24),
        "h_cos": np.cos(2 * np.pi * idx.hour.to_numpy(float) / 24),
        "d_sin": np.sin(2 * np.pi * idx.dayofweek.to_numpy(float) / 7),
        "d_cos": np.cos(2 * np.pi * idx.dayofweek.to_numpy(float) / 7),
        "m_sin": np.sin(2 * np.pi * idx.month.to_numpy(float) / 12),
        "m_cos": np.cos(2 * np.pi * idx.month.to_numpy(float) / 12),
        "is_weekend": (idx.dayofweek >= 5).astype(int),
        "is_holiday": idx.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int),
    }
    return pd.DataFrame({prefix + k: v for k, v in d.items()}, index=idx)


def clean_by_hour(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值。

    为什么不能用「全局中位数 × 3」：工业台区是「夜间低 + 日间高」的双峰分布，
    全局阈值会把整段日间峰误判成异常。居民台区单峰，用全局阈值看不出问题 ——
    换到 B 台区就露馅。
    """
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


def make_features(y, t):
    """日历 + 温度 + 滞后 + 滑窗。滑窗一律「先 shift(1) 再 rolling」，防泄漏。"""
    idx = y.index
    f = cal_block(idx)
    f["T"] = t.to_numpy(float)
    f["T_dev"] = np.abs(t.to_numpy(float) - 22.0)
    f["T2"] = t.to_numpy(float) ** 2
    f["y"] = y.to_numpy(float)
    for k in (1, 2, 3, 24, 25, 48, 168):
        f[f"lag{k}"] = y.shift(k).to_numpy(float)
    base = y.shift(1)
    for w in (3, 24, 168):
        f[f"ma{w}"] = base.rolling(w).mean().to_numpy(float)
        f[f"sd{w}"] = base.rolling(w).std().to_numpy(float)
    return f


def fit_lgb(X, y, n_estimators=300, learning_rate=0.06, num_leaves=31):
    """全流程统一超参的 LightGBM 封装：固定 seed、单线程，保证可复现。"""
    m = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate,
                          num_leaves=num_leaves, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          n_jobs=1, random_state=SEED, verbose=-1)
    m.fit(X, y)
    return m


def metrics(y_true, y_pred):
    """MAE / RMSE / MAPE(%) / R2 —— 四个一起返回，竞赛报告要的就是这一组。"""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    return (float(mean_absolute_error(a, b)),
            float(np.sqrt(mean_squared_error(a, b))),
            float(np.mean(np.abs((a - b) / a)) * 100),
            float(r2_score(a, b)))


def metric_row(name, m):
    """拼成排行榜的一行。"""
    return dict(模型=name, MAE=round(m[0], 4), RMSE=round(m[1], 4),
                MAPE=round(m[2], 4), R2=round(m[3], 6))


def seg_mean_row(seg, name, lo, hi):
    """把「按小时索引的误差表」在 [lo, hi] 这一段上取均值，拼成一行。"""
    return dict(时段=name,
                gbm=round(float(seg.loc[lo:hi, "gbm"].mean()), 4),
                lag24=round(float(seg.loc[lo:hi, "lag24"].mean()), 4),
                lag168=round(float(seg.loc[lo:hi, "lag168"].mean()), 4))


def red_row(tgt_pos, hist, cal_tgt, temp):
    """多步预测的一行精简特征。

    tgt_pos : 目标时刻在全序列里的整数位置
    hist    : 「截止到目标时刻**前 1 小时**」的完整负荷序列（允许含预测值）
    cal_tgt : 目标时刻的日历表（列名带 tgt_ 前缀）
    temp    : 温度全序列

    注意 `lag24 = hist[-24]`：因为 hist 的**最后一个元素**就是「目标前一小时」。
    写成 `hist[-25]` 就整体晚了一格 —— 这个 off-by-one 会让 MAE 从 13 涨到 21，
    且不报任何错。
    """
    c = cal_tgt.iloc[tgt_pos]
    return [c["tgt_h_sin"], c["tgt_h_cos"], c["tgt_d_sin"], c["tgt_d_cos"],
            c["tgt_m_sin"], c["tgt_m_cos"], c["tgt_is_weekend"], c["tgt_is_holiday"],
            temp[tgt_pos], abs(temp[tgt_pos] - 22.0), temp[tgt_pos] ** 2,
            hist[-24], hist[-48], hist[-168],
            float(np.mean(hist[-24:])), float(np.mean(hist[-168:])),
            float(np.std(hist[-24:]))]


def reduced_matrix(starts, yv, cal_tgt, temp):
    """精简特征矩阵：每个起点 p 一行，历史是 `yv[:p]`（截止到 p-1，目标为 y[p]）。"""
    return pd.DataFrame([red_row(p, yv[:p], cal_tgt, temp) for p in starts],
                        columns=RED_COLS)


def teacher_forcing_rows(opos, yv, cal_tgt, temp):
    """teacher forcing 的特征矩阵：每个发布点 × 每个 h，历史一律用**真值**。

    这不是可上线的做法（用到了未来真值），它的唯一用途是当**上界诊断**：
    「假如每一步喂进去的历史都是准的，这套精简特征最好能到多少」。
    """
    return pd.DataFrame(
        [red_row(p + h, yv[:p + h], cal_tgt, temp) for p in opos for h in range(1, H + 1)],
        columns=RED_COLS)


def train_lstm(yv, tv, h_sin, h_cos, cut_full, opos, mu, sd, tmu, tsd,
               seq_len=168, hidden=48, epochs=20, lr=3e-3):
    """四通道序列 [负荷, 温度, sin(h), cos(h)] → 单层 LSTM → 一次输出未来 24 步。

    mu/sd/tmu/tsd 是归一化参数。传 `0/1` 就是「不归一化」，传训练段的均值/标准差
    就是「归一化」—— 这是本章唯一要调的四个数，其余代码一个字都不用改。

    返回 (预测矩阵, 首个 epoch 的 loss, 末个 epoch 的 loss, 参数量)。
    """
    import torch
    import torch.nn as nn

    mat = np.stack([(yv - mu) / sd, (tv - tmu) / tsd, h_sin, h_cos], axis=1).astype(np.float32)
    starts = np.array([p for p in range(seq_len, cut_full) if p + H <= cut_full])
    x_tr = np.stack([mat[p - seq_len:p] for p in starts])
    y_tr = ((np.stack([yv[p + 1:p + 1 + H] for p in starts]) - mu) / sd).astype(np.float32)
    x_te = np.stack([mat[p - seq_len:p] for p in opos])

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(4, hidden, batch_first=True)
            self.head = nn.Linear(hidden, H)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1])

    torch.manual_seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.SmoothL1Loss()
    xt = torch.tensor(x_tr, device=dev)
    yt = torch.tensor(y_tr, device=dev)
    hist = []
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        total = 0.0
        for b in range(0, len(xt), 128):
            j = perm[b:b + 128]
            opt.zero_grad()
            loss = lossf(net(xt[j]), yt[j])
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        hist.append(total / len(xt))
    with torch.no_grad():
        pred = net(torch.tensor(x_te, device=dev)).cpu().numpy() * sd + mu
    return pred, hist[0], hist[-1], sum(q.numel() for q in net.parameters())
'''

SETUP = '''raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
print("原始行数", len(raw), "| 时间戳单调递增？", raw["时间戳"].is_monotonic_increasing)
print("台区", sorted(raw["台区编号"].unique()))

# 去重 → 排序。**顺序不能反**：先 asfreq 再排序，规则时间轴会被重复时刻搞乱
df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
print("去重后", len(df), "行 | 重复行", len(raw) - len(df))

one = df[df["台区编号"] == STATION].set_index("时间戳").sort_index()
s_raw = one["负荷值"].asfreq("h")          # 未清洗的规则序列（这一步用不到温度）
t_raw = one["温度"].asfreq("h")
print("规则化后", len(s_raw), "小时 | 负荷缺失", int(s_raw.isna().sum()),
      "| 温度缺失", int(t_raw.isna().sum()))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

# =========================================================================== #
# 2. 任务代码块                                                               #
# =========================================================================== #

T1_CODE = '''# @@todo 统计「越界点」：落在 [0, 3×同小时中位数] 之外的时刻，缺失本身不算越界
# @@hint 用 s_raw.groupby(s_raw.index.hour).transform("median") 拿到「同小时的中位数」
# @@hint 两头都要卡：负值（−68.78）与哨兵 9999 都是越界，所以是 between(0, ...) 取反
med = s_raw.groupby(s_raw.index.hour).transform("median")
bad = ~s_raw.between(0, 3 * med) & s_raw.notna()
# @@end
print("越界点数", int(bad.sum()), "| 其中负值", int((s_raw < 0).sum()))
print("越界原值", sorted(np.round(s_raw[bad].to_numpy(), 2).tolist()))

# @@todo 生成清洗后的负荷 y 与补齐后的温度 t
# @@hint clean_by_hour 已经把「三倍阈值判异常 + 线性插值」两步封好了
# @@hint 温度用 interpolate(limit_direction="both")：首尾的缺失只有双向插值才补得掉
y = clean_by_hour(s_raw)
t = t_raw.interpolate(limit_direction="both")
# @@end
print("清洗后缺失", int(y.isna().sum()), "| 被改动的点数", int((y - s_raw).abs().gt(1e-9).sum()))
print("清洗前 mean/std %.4f/%.4f → 清洗后 %.4f/%.4f"
      % (s_raw.mean(), s_raw.std(), y.mean(), y.std()))
print("最大绝对修正量 %.2f kW" % float((y - s_raw).abs().max()))

assert int(bad.sum()) == 7
assert int((s_raw < 0).sum()) == 2
assert int(y.isna().sum()) == 0 and int(t.isna().sum()) == 0
assert int((y - s_raw).abs().gt(1e-9).sum()) == 7
assert abs(float(y.std()) - 72.1933) < 1e-3
assert abs(float((y - s_raw).abs().max()) - 9480.56) < 0.01
'''

T2_CODE = '''# @@todo 生成特征表 FEAT
# @@hint make_features(y, t) 返回的表里**同时含目标列 y**，后面切分时要把它摘出去
FEAT = make_features(y, t)
# @@end
print("形状", FEAT.shape, "| 列名", list(FEAT.columns))
print("含 NaN 的行", int(FEAT.isna().any(axis=1).sum()))
print("节假日小时", int(FEAT["is_holiday"].sum()), "| 周末小时", int(FEAT["is_weekend"].sum()))
print("温度线性相关 %.6f | |T-22| 相关 %.6f"
      % (np.corrcoef(t, y)[0, 1], np.corrcoef(np.abs(t - 22.0), y)[0, 1]))

# 周期的「整数距离」是骗人的：23 点与 0 点相差 23，实际只差 1 小时。
# 把它们编码成单位圆上的 (sin, cos) 后，两点间的**弦长**才反映真实距离。
p23 = pd.Timestamp("2025-03-05 23:00")
p00 = pd.Timestamp("2025-03-06 00:00")

# @@todo 算出 23 点与 0 点在 (sin, cos) 平面上的欧氏距离
# @@hint np.hypot(dx, dy) 就是二范数；两个分量都从 FEAT 里按时间戳取即可
chord_2300 = float(np.hypot(FEAT.loc[p23, "h_sin"] - FEAT.loc[p00, "h_sin"],
                            FEAT.loc[p23, "h_cos"] - FEAT.loc[p00, "h_cos"]))
# @@end
print("23↔0 编码弦长 %.6f（不编码时是 23）" % chord_2300)

assert FEAT.shape == (8760, 25)
assert int(FEAT.isna().any(axis=1).sum()) == 168
assert int(FEAT["is_holiday"].sum()) == 672
assert int(FEAT["is_weekend"].sum()) == 2496
assert abs(float(np.corrcoef(np.abs(t - 22.0), y)[0, 1]) - 0.406437) < 1e-5
assert abs(chord_2300 - 0.261052) < 1e-6
# FEAT 还没 dropna，所以第 500 行就是全序列第 500 小时 —— 它的 lag168 应等于第 332 小时的负荷
assert np.isclose(FEAT["lag168"].iloc[500], y.iloc[500 - N_HEAD])
'''

T3_CODE = '''# @@todo 丢掉头部 168 行（滑窗冷启动），再按时间 8:2 切分
# @@hint 时序**只能按时间切**，绝对不能 shuffle —— 随机切分会让「未来」漏进训练集
# @@hint dropna 之后行序仍是时间序，直接 iloc 切片即可
OK = FEAT.dropna()
# @@end

# @@todo 摘出特征列（把目标列 y 排除掉）、算出样本数与切点，然后切成 tr / te
# @@hint COLS 用列表推导式：把 OK.columns 里除 "y" 之外的全部留下
# @@hint cut = int(n * 0.8)；切片用 OK.iloc[:cut] 与 OK.iloc[cut:]
COLS = [c for c in OK.columns if c != "y"]
n = len(OK)
cut = int(n * 0.8)
tr, te = OK.iloc[:cut], OK.iloc[cut:]
# @@end
print("样本 %d | 训练 %d | 测试 %d | 特征 %d 列" % (n, len(tr), len(te), len(COLS)))
print("训练", tr.index.min(), "~", tr.index.max())
print("测试", te.index.min(), "~", te.index.max())
print("训练 y mean/std %.4f/%.4f | 测试 %.4f/%.4f"
      % (tr["y"].mean(), tr["y"].std(), te["y"].mean(), te["y"].std()))

# ---- 泄漏自检：滑窗统计量必须只用到「目标前一小时及以前」----
# OK 第 i_check 行对应全序列位置 i_check + 168。rolling 默认右对齐：位置 j 的
# ma24 窗口是 j-23..j，而 base 已经 shift(1)，所以它实际覆盖 y[j-24:j]。
i_check = 300
j_check = i_check + N_HEAD

# @@todo 用手工切片核对 ma24 / sd24 的口径（lag1 已给，照着写）
# @@hint 手工值是 y.iloc[j_check-24 : j_check].mean() 与 .std(ddof=1)
# @@hint sd 用样本标准差，对应 pandas rolling().std() 的默认行为（ddof=1）
ma24_manual = float(y.iloc[j_check - 24:j_check].mean())
sd24_manual = float(y.iloc[j_check - 24:j_check].std(ddof=1))
# @@end
lag1_manual = float(y.iloc[j_check - 1])
print("ma24  表内 %.4f | 手工 %.4f" % (OK["ma24"].iloc[i_check], ma24_manual))
print("sd24  表内 %.4f | 手工 %.4f" % (OK["sd24"].iloc[i_check], sd24_manual))
print("lag1  表内 %.4f | 手工 %.4f" % (OK["lag1"].iloc[i_check], lag1_manual))

assert (n, len(tr), len(te), len(COLS)) == (8592, 6873, 1719, 24)
assert tr.index.max() < te.index.min()          # 训练段与测试段完全不重叠
assert np.isclose(OK["ma24"].iloc[i_check], ma24_manual)
assert np.isclose(OK["sd24"].iloc[i_check], sd24_manual)
assert np.isclose(OK["lag1"].iloc[i_check], lag1_manual)
assert abs(float(te["y"].mean()) - 483.9447) < 1e-3
'''

T4_CODE = '''# 水平比：最近 24 小时均值 ÷ 它之前的 24 小时均值 —— 把周周期基线的整体水位拉回来
lvl = (te["ma24"] / te["ma24"].shift(24).bfill()).fillna(1.0)

base_rows = []
# @@todo 前两条基线：持续性（用 lag1）与 日周期（用 lag24）
# @@hint 真值是 te["y"]，拼行用 metric_row(名字, metrics(真值, 预测))
# @@hint 四条基线里，持续性最弱、日周期是「不动脑筋」的默认基线
base_rows.append(metric_row("① 持续性 lag1", metrics(te["y"], te["lag1"])))
base_rows.append(metric_row("② 日周期 lag24", metrics(te["y"], te["lag24"])))
# @@end

# @@todo 后两条基线：周周期（用 lag168）与「周周期 × 水平比」
# @@hint 水平比 lvl 已经算好，直接乘到 lag168 上
base_rows.append(metric_row("③ 周周期 lag168", metrics(te["y"], te["lag168"])))
base_rows.append(metric_row("④ 周周期×水平比", metrics(te["y"], te["lag168"] * lvl)))
# @@end

base_tbl = pd.DataFrame(base_rows)
print(base_tbl.to_string(index=False))
print("日周期 / 周周期 MAE 比 %.4f" % (base_tbl["MAE"].iloc[1] / base_tbl["MAE"].iloc[2]))

assert abs(float(base_tbl["MAE"].iloc[0]) - 30.2482) < 1e-3
assert abs(float(base_tbl["MAE"].iloc[1]) - 21.5898) < 1e-3
assert abs(float(base_tbl["MAE"].iloc[2]) - 19.1805) < 1e-3
assert base_tbl["MAE"].iloc[2] < base_tbl["MAE"].iloc[1]        # 周周期赢日周期
assert base_tbl["MAE"].iloc[2] < base_tbl["MAE"].iloc[3]        # 水平比反而帮了倒忙
'''

T5_CODE = '''# @@todo 标准化特征（只用训练段的均值/标准差），再训 Ridge(alpha=1.0)
# @@hint StandardScaler().fit(训练特征)：**只能拟合训练段**！测试段统计量不能进 scaler
# @@hint 目标 y 不标准化，这样 MAE 直接就是 kW，不用再反变换回来
# @@hint fit 用 scaler.transform(tr[COLS])，预测用 scaler.transform(te[COLS])
scaler = StandardScaler().fit(tr[COLS])
ridge = Ridge(alpha=1.0).fit(scaler.transform(tr[COLS]), tr["y"].to_numpy(float))
pred_ridge = ridge.predict(scaler.transform(te[COLS]))
# @@end

# @@todo 训 LightGBM 并预测
# @@hint 树模型对特征单调变换不敏感，**不需要**标准化，直接把原始特征喂进去
# @@hint fit_lgb(X, y) 就是脚手架里那个统一超参的封装
gbm = fit_lgb(tr[COLS], tr["y"].to_numpy(float))
pred_gbm = gbm.predict(te[COLS])
# @@end

lgb_tbl = pd.DataFrame([
    metric_row("⑤ Ridge(α=1)", metrics(te["y"], pred_ridge)),
    metric_row("⑥ LightGBM", metrics(te["y"], pred_gbm)),
])
leader = pd.concat([base_tbl, lgb_tbl], ignore_index=True)
print(leader.to_string(index=False))

imp = pd.Series(gbm.feature_importances_, index=COLS).sort_values(ascending=False)
print("特征重要性 top8", {k: int(v) for k, v in imp.head(8).items()})
print("重要性最低 5 个", {k: int(v) for k, v in imp.tail(5).items()})

assert abs(float(leader["MAE"].iloc[4]) - 14.2251) < 1e-2
assert abs(float(leader["MAE"].iloc[5]) - 12.6689) < 1e-2
assert imp.index[0] == "lag24"                     # 最重要的居然是「昨天同一时刻」
assert float(leader["R2"].iloc[5]) > 0.93
assert float(leader["MAE"].iloc[5]) < float(leader["MAE"].iloc[4])
'''

T6_CODE = '''res_gbm = te["y"].to_numpy(float) - pred_gbm

# @@todo 按小时分组，算出 LightGBM 与两条基线的平均绝对误差
# @@hint 用 pd.DataFrame({...}).groupby(te.index.hour).mean()；绝对值用 np.abs
# @@hint 三条列分别叫 gbm / lag24 / lag168（后面的分时段汇总依赖这三个列名）
seg = pd.DataFrame({
    "gbm": np.abs(res_gbm),
    "lag24": np.abs(te["y"] - te["lag24"]),
    "lag168": np.abs(te["y"] - te["lag168"]),
}).groupby(te.index.hour).mean()
# @@end
print(seg.round(4).to_string())

# @@todo 把 24 小时并成 5 个时段，看看误差是怎么分布的
# @@hint seg_mean_row(seg, 时段名, lo, hi) 已经在脚手架里给好了
# @@hint 时段划分：夜间 0-5 / 早峰 6-9 / 日间 10-16 / 晚峰 17-21 / 深夜 22-23
SEGMENTS = {"夜间": (0, 5), "早峰": (6, 9), "日间": (10, 16), "晚峰": (17, 21), "深夜": (22, 23)}
seg_tbl = pd.DataFrame([seg_mean_row(seg, nm, lo, hi) for nm, (lo, hi) in SEGMENTS.items()])
# @@end
print(seg_tbl.to_string(index=False))

mon_tbl = pd.DataFrame({"m": te.index.month, "gbm": np.abs(res_gbm)}).groupby("m").mean()
print("分月 MAE", {int(k): round(float(v), 4) for k, v in mon_tbl["gbm"].items()})

assert abs(float(seg.loc[17:21, "gbm"].mean()) - 16.7882) < 1e-3
assert abs(float(seg.loc[0:5, "gbm"].mean()) - 11.3963) < 1e-3
assert seg.loc[17:21, "gbm"].mean() > seg.loc[10:16, "gbm"].mean()      # 晚峰最难
assert seg.loc[17:21, "gbm"].mean() < seg.loc[17:21, "lag24"].mean()    # 但仍显著优于基线
assert len(seg_tbl) == 5
'''

T7_CODE = '''tcal = cal_block(y.index, "tgt_")        # 目标时刻的日历（列名带 tgt_ 前缀）
pos = {q: i for i, q in enumerate(y.index)}
tr_pos = np.array([pos[q] for q in tr.index])
te_pos = np.array([pos[q] for q in te.index])
yv = y.to_numpy(float)
tv = t.to_numpy(float)

# @@todo 选出「发布时刻」：测试段里每天 0 点，且 24 小时后还没出测试段
# @@hint 0 点的位置满足 y.index[i].hour == 0；下界是 te_pos[0]，上界是 te_pos[-1] - H
d0 = np.array([i for i in range(len(yv)) if y.index[i].hour == 0])
opos = np.array([i for i in d0 if i >= te_pos[0] and i + H <= te_pos[-1]])
# @@end
truth = np.stack([yv[opos + h] for h in range(1, H + 1)], axis=1)
print("发布日数", len(opos), "|", y.index[opos[0]], "~", y.index[opos[-1]])
print("真值矩阵 shape", truth.shape, "（= 发布日数 × 视野）")

# ---- 直接法：为 h=1..24 各训一个模型 ----
direct_models = {}
for h in range(1, H + 1):
    # @@todo 训练第 h 步的模型：特征 = 「原点 t 的特征」 + 「目标 t+h 的日历」
    # @@hint 用 pd.concat([...], axis=1) 横着拼；目标日历列名带 tgt_ 前缀，不会重名
    # @@hint 训练样本要满足 tr_pos + h <= tr_pos[-1]，否则目标会落进测试段（泄答）
    # @@hint fit_lgb(X, y) 就是脚手架里那个统一超参的封装
    ok = tr_pos + h <= tr_pos[-1]
    X = pd.concat([tr[ok].reset_index(drop=True)[COLS],
                   tcal.iloc[tr_pos[ok] + h].reset_index(drop=True)], axis=1)
    direct_models[h] = fit_lgb(X, yv[tr_pos[ok] + h])
    # @@end
print("已训练", len(direct_models), "个模型")

# @@todo 先把「属于发布点」的测试行用布尔掩码挑出来
# @@hint mask = np.isin(te_pos, opos)；te[mask] 就是那 70 行原点特征
# @@hint 预测结果矩阵用 np.empty((len(opos), H)) 先占好位置
mask = np.isin(te_pos, opos)
pred_direct = np.empty((len(opos), H))
# @@end
for h in range(1, H + 1):
    # @@todo 拼出第 h 步的特征并预测
    # @@hint 特征 = 「原点 t 的特征」 + 「目标 t+h 的日历」，用 pd.concat([...], axis=1) 横着拼
    # @@hint 预测结果写进 pred_direct[:, h - 1]
    XX = pd.concat([te[mask].reset_index(drop=True)[COLS],
                    tcal.iloc[opos + h].reset_index(drop=True)], axis=1)
    pred_direct[:, h - 1] = direct_models[h].predict(XX)
    # @@end

truth_naive = np.stack([yv[opos + h - 24] for h in range(1, H + 1)], axis=1)
mae_direct = np.abs(truth - pred_direct).mean(axis=0)
mae_naive = np.abs(truth - truth_naive).mean(axis=0)
print("直接法 日均 MAE %.4f | 日周期基线 %.4f" % (mae_direct.mean(), mae_naive.mean()))
print("h=1 %.4f → h=24 %.4f（×%.3f）"
      % (mae_direct[0], mae_direct[-1], mae_direct[-1] / mae_direct[0]))

assert len(opos) == 70
assert truth.shape == (70, 24)
assert abs(float(mae_direct.mean()) - 16.1107) < 1e-2
assert abs(float(mae_direct[0]) - 12.0749) < 1e-2
assert abs(float(mae_direct[-1]) - 13.6702) < 1e-2
assert float(mae_direct.mean()) < float(mae_naive.mean())
'''

T8_CODE = '''# 只训一个「下一小时」模型：足够 168 小时历史起点才算样本
ok_rec = tr_pos - 1 >= N_HEAD
rec = fit_lgb(reduced_matrix(tr_pos[ok_rec], yv, tcal, tv), yv[tr_pos[ok_rec]])
print("递归法训练样本", int(ok_rec.sum()))

# @@todo 用老师强迫（teacher forcing）划出上界：同一个模型，但历史一律喂真值
# @@hint teacher_forcing_rows(opos, yv, tcal, tv) 已经把 70×24 行的特征拼好了
# @@hint 预测结果 reshape(len(opos), H) 才是 (发布日 × 视野) 的矩阵
p_tf = rec.predict(teacher_forcing_rows(opos, yv, tcal, tv)).reshape(len(opos), H)
# @@end

# ---- 递归法：每预测一格就把结果接回历史 ----
pred_rec = np.empty((len(opos), H))
for i, p in enumerate(opos):
    # @@todo 把「截止到原点 t 的观测」拷成一个可增长的列表
    # @@hint **off-by-one 陷阱**：要预测 t+1，历史必须含 t 本身
    # @@hint 所以是 yv[:p + 1]（末尾是 y[t]），写成 yv[:p] 会整体晚一格
    ext = list(yv[:p + 1])
    # @@end
    for k in range(1, H + 1):
        # @@todo 造出目标 t+k 的特征、预测它，并把预测值接回 ext
        # @@hint red_row(目标位置, 历史数组, 目标日历表, 温度序列) 已在脚手架里给好
        # @@hint 预测出来是 numpy 标量，float(...) 转一下再 append
        pred_rec[i, k - 1] = float(rec.predict(pd.DataFrame(
            [red_row(p + k, np.asarray(ext), tcal, tv)], columns=RED_COLS))[0])
        ext.append(pred_rec[i, k - 1])
        # @@end

mae_tf = np.abs(truth - p_tf).mean(axis=0)
mae_rec = np.abs(truth - pred_rec).mean(axis=0)
print("teacher forcing 日均 MAE %.4f" % mae_tf.mean())
print("递归法           日均 MAE %.4f" % mae_rec.mean())
print("递归法 h=1 %.4f → h=24 %.4f（×%.3f）"
      % (mae_rec[0], mae_rec[-1], mae_rec[-1] / mae_rec[0]))

assert abs(float(mae_tf.mean()) - 13.0980) < 1e-2
assert abs(float(mae_rec.mean()) - 13.1118) < 1e-2
assert float(mae_rec.mean()) < float(mae_direct.mean())     # 反直觉：递归反而更好
'''

T9_CODE = '''# @@todo 把「递归 vs 直接」的差距拆成两块
# @@hint 特征集差 = teacher forcing（同样精简特征，但历史是真值）− 直接法
# @@hint 误差累积 = 递归法 − teacher forcing
gap_feature = float(mae_tf.mean() - mae_direct.mean())
gap_drift = float(mae_rec.mean() - mae_tf.mean())
# @@end
print("特征对齐带来的收益 %+.4f kW | 递归的误差累积 %+.4f kW" % (gap_feature, gap_drift))
print("误差累积占两者之和的 %.1f%%" % (100 * gap_drift / (gap_feature + gap_drift)))

print("三类多步特征的口径对照：")
print("  ❶ 日周期基线（把昨天的 24h 直接平移一天）")
print("  ❷ 直接法（24 个模型，特征全部锁死在原点 t）")
print("  ❸ 精简特征 + 真值历史（上界诊断，不可上线）")
print("  ❹ 精简特征 + 递归历史（可上线）")

step_tbl = pd.DataFrame({"h": np.arange(1, H + 1),
                         "❶日周期基线": np.round(mae_naive, 4),
                         "❷直接法": np.round(mae_direct, 4),
                         "❸精简+真值": np.round(mae_tf, 4),
                         "❹精简+递归": np.round(mae_rec, 4)})
print(step_tbl.to_string(index=False))

assert abs(gap_feature + 3.0127) < 1e-2          # 负值：精简特征反而更好
assert abs(gap_drift - 0.0138) < 1e-2            # 误差累积几乎为零
assert abs(gap_drift / (gap_feature + gap_drift)) < 0.05
assert float(mae_naive[-1] / mae_naive[0]) > 1.2         # 基线越往后越差
assert float(mae_rec[-1] / mae_rec[0]) < 1.2             # 递归法几乎不随 h 变差
'''

T10_CODE = '''# @@todo 第一遍：**故意不做任何归一化**（四个参数全给 0 / 1）
# @@hint 负荷量级约 460、温度约 20、sin/cos 在 [-1,1] —— 三条通道差了整整两个数量级
# @@hint 只换这四个数，train_lstm 里其余代码一个字都不改
# @@hint 第 5 个参数是 cut_full = cut + 168（特征矩阵切点在**全序列**里的位置）
pred_raw, l0_raw, l1_raw, nparam = train_lstm(
    yv, tv, FEAT["h_sin"].to_numpy(float), FEAT["h_cos"].to_numpy(float),
    cut + N_HEAD, opos, 0.0, 1.0, 0.0, 1.0)
# @@end

# @@todo 第二遍：只改归一化参数 —— 换成**训练段**的均值 / 标准差
# @@hint 负荷与温度各用各的 mu/sd；两者都只能来自 cut 之前（训练段）
# @@hint 写成 yv[:cut + N_HEAD].mean() / .std() 与 tv[:cut + N_HEAD].mean() / .std()
pred_norm, l0_norm, l1_norm, _ = train_lstm(
    yv, tv, FEAT["h_sin"].to_numpy(float), FEAT["h_cos"].to_numpy(float),
    cut + N_HEAD, opos,
    float(yv[:cut + N_HEAD].mean()), float(yv[:cut + N_HEAD].std()),
    float(tv[:cut + N_HEAD].mean()), float(tv[:cut + N_HEAD].std()))
# @@end

mae_raw = float(np.abs(truth - pred_raw).mean())
mae_norm = float(np.abs(truth - pred_norm).mean())
print("参数量", nparam, "| 训练样本", cut + N_HEAD - N_HEAD - H + 1)
print("不归一化：loss %.5f → %.5f | 日均 MAE %.4f" % (l0_raw, l1_raw, mae_raw))
print("归一化　：loss %.5f → %.5f | 日均 MAE %.4f" % (l0_norm, l1_norm, mae_norm))
print("归一化把误差压低了 %.1f 倍" % (mae_raw / mae_norm))
print("但仍打不过 LightGBM（%.4f）与递归法（%.4f）" % (float(leader["MAE"].iloc[5]), mae_rec.mean()))

assert abs(mae_raw - 409.7550) < 0.5
assert abs(mae_norm - 17.4668) < 0.5
assert mae_raw / mae_norm > 20                 # 归一化与否差了一个数量级
assert mae_norm > float(mae_rec.mean())        # 归一化之后仍然输
'''

T11_CODE = '''# @@todo 对 LightGBM 的测试段残差做 Ljung-Box 检验与 ACF 计算
# @@hint acorr_ljungbox(残差, lags=[24, 48, 72], return_df=True) 返回 lb_stat / lb_pvalue
# @@hint 原假设是「残差是白噪声」—— p < 0.05 就是**拒绝**白噪声，说明还有结构没抓住
# @@hint ACF 用 pd.Series(res_gbm).autocorr(k)，k 取 1 / 24 / 168
lb = acorr_ljungbox(res_gbm, lags=[24, 48, 72], return_df=True)
res_acf = {k: round(float(pd.Series(res_gbm).autocorr(k)), 6) for k in (1, 24, 168)}
# @@end
print("Ljung-Box p 值", dict(zip([24, 48, 72], np.round(lb["lb_pvalue"].to_numpy(), 8))))
print("残差 ACF", res_acf)
print("残差 std / y std = %.4f" % (res_gbm.std() / te["y"].std()))

assert bool((lb["lb_pvalue"].to_numpy() < 0.01).all())     # 强烈拒绝白噪声
assert abs(res_acf[1] - (-0.055219)) < 1e-5
assert abs(float(res_gbm.std() / te["y"].std()) - 0.2456) < 1e-3
assert abs(res_acf[168]) < abs(res_acf[24])               # 周周期残差比日周期小
'''

T12_CODE = '''fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

# 最后 7 个发布日拼成一个连续 168 小时窗口：起点 = 第 7 个发布日的下一小时
i0 = int(opos[-7] + 1)
xidx = y.index[i0:i0 + 168]

# @@todo 图一：真值 / 多步预测（7 天 × 24）/ 日周期基线，三条曲线叠在同一时间轴上
# @@hint 预测矩阵 pred_direct[-7:] 是 (7, 24)，ravel() 之后正好按时间顺序排成 168 个点
# @@hint 日周期基线是 yv[i0-24 : i0-24+168]
# @@hint 标签依次为 真值 / 多步预测(7天×24) / 日周期基线；记得 ax.legend(fontsize=8)
ax = axes[0]
ax.plot(xidx, yv[i0:i0 + 168], label="真值", lw=1.8, color="#1f77b4")
ax.plot(xidx, pred_direct[-7:].ravel(), label="多步预测(7天×24)", lw=1.3, color="#d62728")
ax.plot(xidx, yv[i0 - 24:i0 - 24 + 168], label="日周期基线", lw=1.0, ls="--", color="#7f7f7f")
ax.set_title("最后 7 天：日前 24 步预测")
ax.set_ylabel("负荷 / kW")
ax.legend(fontsize=8)
ax.tick_params(axis="x", rotation=30)
# @@end

# @@todo 图二：按小时的平均绝对误差，画 gbm / lag24 / lag168 三条线
# @@hint seg 的行索引就是小时 0..23，直接当 x 轴
ax = axes[1]
ax.plot(seg.index, seg["gbm"], marker="o", label="LightGBM", color="#2ca02c")
ax.plot(seg.index, seg["lag24"], marker="s", ls="--", label="日周期基线", color="#7f7f7f")
ax.plot(seg.index, seg["lag168"], marker="^", ls=":", label="周周期基线", color="#ff7f0e")
ax.set_title("误差的时段结构")
ax.set_xlabel("小时")
ax.set_ylabel("MAE / kW")
ax.legend(fontsize=8)
# @@end

# @@todo 图三：特征重要性前 15 名，横向条形图
# @@hint imp 已经按重要性降序排好；barh 的 y 轴从下往上，所以取 top[::-1] 让最大值在顶部
ax = axes[2]
top = imp.head(15)[::-1]
ax.barh(top.index, top.to_numpy(), color="#9467bd")
ax.set_title("特征重要性 top15")
ax.set_xlabel("split 次数")
# @@end

plt.tight_layout()
print("三张图已生成")

assert len(axes) == 3
assert imp.head(15).index[0] == "lag24"
assert seg["gbm"].idxmax() in (19, 20)
'''

T13_CODE = '''best_name = leader.loc[leader["MAE"].idxmin(), "模型"]
best_mae = float(leader["MAE"].min())
worst_name = leader.loc[leader["MAE"].idxmax(), "模型"]
gain_vs_naive = float(mae_naive.mean()) / float(mae_direct.mean())

# @@todo 生成结题报告（一个 markdown 字符串），把关键数字串进去
# @@hint 至少覆盖：数据规模 / 最佳单步模型与 MAE / 多步三档 MAE / 误差累积 / 环境版本
# @@hint 用 f-string 拼接；字符串之间要有 "\\n" 换行，否则整段挤成一行
# @@hint 百分比格式用 {x:.4f}%，不要写 %.4f（会和 f-string 的 {} 打架）
REPORT = (
    "# 台区负荷日前预测 · 结题报告\\n\\n"
    f"- 数据：{STATION}，{len(s_raw)} 小时规则序列，清洗后可用样本 {n} 个\\n"
    f"- 切分：训练 {len(tr)} / 测试 {len(te)}（按时间 8:2，"
    f"测试段 {te.index.min():%Y-%m-%d} ~ {te.index.max():%Y-%m-%d}）\\n"
    f"- 最佳单步模型：{best_name}，MAE = {best_mae:.4f} kW，"
    f"MAPE = {float(leader['MAPE'].min()):.4f}%，R2 = {float(leader['R2'].max()):.6f}\\n"
    f"- 最弱模型：{worst_name}，MAE = {float(leader['MAE'].max()):.4f} kW\\n"
    f"- 日前 24 步（每天 0 点发布，共 {len(opos)} 天）："
    f"直接法 {float(mae_direct.mean()):.4f} kW / 递归法 {float(mae_rec.mean()):.4f} kW / "
    f"日周期基线 {float(mae_naive.mean()):.4f} kW，提升 {gain_vs_naive:.2f} 倍\\n"
    f"- 归因：递归的误差累积仅 {gap_drift:+.4f} kW，"
    f"真正拉开差距的是特征对齐（{gap_feature:+.4f} kW）\\n"
    f"- 环境：lightgbm {lgb.__version__}，OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}\\n"
)
# @@end
print(REPORT)

assert "最佳单步模型" in REPORT
assert best_name == "⑥ LightGBM"
assert "12.6689" in REPORT
assert "16.1107" in REPORT
assert "13.1118" in REPORT
assert REPORT.count("\\n") >= 6
'''

T14_CODE = '''# @@todo 换数据源：取 B 台区（工业型），复用同一套清洗
# @@hint 一律走 clean_by_hour，不要复制粘贴逻辑 —— 台区之间只有数据不同
# @@hint 温度同样 interpolate(limit_direction="both")
one_b = df[df["台区编号"] == OTHER].set_index("时间戳").sort_index()
y_b = clean_by_hour(one_b["负荷值"].asfreq("h"))
t_b = one_b["温度"].asfreq("h").interpolate(limit_direction="both")
# @@end

# @@todo 造特征表并做同样的 8:2 切分
# @@hint make_features(y_b, t_b) → dropna → 特征列 = 除 "y" 外的全部
# @@hint cut_b = int(len(FB) * 0.8)
FB = make_features(y_b, t_b).dropna()
COLS_B = [c for c in FB.columns if c != "y"]
cut_b = int(len(FB) * 0.8)
tr_b, te_b = FB.iloc[:cut_b], FB.iloc[cut_b:]
# @@end

# @@todo 训同样的 LightGBM 并预测
# @@hint fit_lgb 的默认超参与 A 台区完全一致 —— 不调参才是「同一套流程」的证据
gbm_b = fit_lgb(tr_b[COLS_B], tr_b["y"].to_numpy(float))
pred_b = gbm_b.predict(te_b[COLS_B])
# @@end

b_tbl = pd.DataFrame([
    metric_row("① 持续性 lag1", metrics(te_b["y"], te_b["lag1"])),
    metric_row("② 日周期 lag24", metrics(te_b["y"], te_b["lag24"])),
    metric_row("③ 周周期 lag168", metrics(te_b["y"], te_b["lag168"])),
    metric_row("⑥ LightGBM", metrics(te_b["y"], pred_b)),
])
print("B 台区（工业型，日间平台 + 周末腰斩）")
print(b_tbl.to_string(index=False))

seg_b = pd.DataFrame({"gbm": np.abs(te_b["y"].to_numpy(float) - pred_b),
                      "lag24": np.abs(te_b["y"] - te_b["lag24"])}).groupby(te_b.index.hour).mean()
print("B 台区 晚峰(17-21) MAE %.4f | 日间(10-16) MAE %.4f"
      % (seg_b.loc[17:21, "gbm"].mean(), seg_b.loc[10:16, "gbm"].mean()))

assert abs(float(b_tbl["MAE"].iloc[1]) - 71.3448) < 1e-2    # 日周期基线在 B 上彻底崩了
assert abs(float(b_tbl["MAE"].iloc[2]) - 17.1623) < 1e-2    # 周周期基线反而很强
assert abs(float(b_tbl["MAE"].iloc[3]) - 13.1559) < 1e-2
assert float(b_tbl["MAE"].iloc[3]) < float(b_tbl["MAE"].iloc[1])
assert seg_b.loc[10:16, "gbm"].mean() > seg_b.loc[17:21, "gbm"].mean()   # A 与 B 的难点时段正好相反
'''

# =========================================================================== #
# 3. 讲解版                                                                   #
# =========================================================================== #

LESSON = [
    md(
        """
# final_01 台区负荷日前预测全流程（讲解版）

> **本节考点**：不是新算法，而是把 ch01~ch08 的零件**按正确顺序**串成一条能交付的流水线。
> 综合题失分几乎从不在模型上 —— 失在**清洗口径**、**切分时机**、**多步策略**、**归因**这四件事上。

| 竞赛评分点 | 分值含义 | 本节覆盖 |
|---|---|---|
| 数据准备及处理 | 技能操作 10% | 规则时间轴、分组阈值清洗、滞后 / 滑窗 / 日历特征 |
| 模型训练 | **技能操作 40%** | 四条基线 + Ridge + LightGBM + LSTM，同一套评估口径 |
| 新能源功率 / 负荷预测 | 行业赛题 | **本节就是这条赛题的完整骨架** |
| 结果分析与报告 | 技能操作 10% | 分时段 / 分月误差、残差检验、归因拆解、自动报告 |

## 学习目标

1. 按「去重 → 排序 → 对齐 → 分组阈值清洗」的顺序把脏 CSV 变成干净序列，并说清每一步为什么不能换位
2. 说清**为什么周周期基线会赢日周期基线**，以及这个结论在什么数据上会反转
3. 会写**直接法**与**递归法**两套多步预测，并知道各自的代价
4. **能把「递归更差」拆成「特征集差」与「误差累积」两块**，并用数据说明哪一块才是主导
5. 会说清 LSTM 在这类小样本强周期数据上**为什么输给 LightGBM**（并亲眼见到输入不归一化的下场）

## 流水线全景

```
CSV(脏) ──去重/排序──▶ 规则时间轴 ──分组阈值+插值──▶ 干净序列
   │                                                      │
   │                                          ┌───────────┴───────────┐
   │                                    日历 sin/cos            温度 |T−22|
   │                                          └───────────┬───────────┘
   │                                                      ▼
   └──────────────────────────────────────▶ 特征表 ──按时间 8:2──▶ (X_tr, y_tr) / (X_te, y_te)
                                                                     │
                      ┌──────────────────────────────────────────────┤
                      ▼                        ▼                     ▼
                四条朴素基线            Ridge / LightGBM         LSTM（负结果）
                      └──────────────────────┬──────────────────────┘
                                             ▼
                              单步评估 → 误差的时段结构 → 残差白噪声检验
                                             │
                                             ▼
                              日前 24 步：直接法 / 递归法 / 上界诊断
                                             │
                                             ▼
                                    归因拆解 → 结题报告 → 跨台区泛化
```

## API 速查表

| 方法 / 属性 | 关键参数 | 一句话 |
|---|---|---|
| `s.groupby(s.index.hour).transform("median")` | — | 每个时刻的「同小时中位数」，分组阈值的分母 |
| `s.where(cond).interpolate()` | `limit_direction` | 条件为假置 NaN 再插值；`"both"` 才补首尾缺失 |
| `Series.shift(k)` / `rolling(n)` | `min_periods` | 滞后 / 滑窗；**滑窗必须先 `shift(1)`** |
| `pd.concat([...], axis=1)` | — | 横着拼表；列重名会让 LightGBM 直接报错 |
| `StandardScaler().fit(tr[X])` | — | **只能拟合训练段**；测试段统计量进 scaler 即泄漏 |
| `np.isin(a, b)` | — | 布尔掩码，用来从测试段里挑出「发布时刻」 |
| `acorr_ljungbox(x, lags=...)` | `return_df=True` | 残差白噪声检验；p < 0.05 即拒绝 |
| `pd.Series(x).autocorr(k)` | — | 滞后 k 自相关，用来定位没抓住的周期 |
| `lightgbm.LGBMRegressor` | `n_jobs`, `random_state` | 单线程 + 固定 seed 才可复现 |
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(SETUP),
    code(PLOT_SETUP),
    md(
        """
## 第一节 · 清洗：阈值必须分组

先看数据有多脏。`load_curve.csv` 是 `data/make_data.py`（`seed=42`）造的，故意在文件**末尾**追加了 6 行重复记录 —— 所以原始文件的时间戳是**倒着跳的**：

```
原始行数 17526 | 时间戳单调递增？ False
去重后 17520 行 | 重复行 6
规则化后 8760 小时 | 负荷缺失 70 | 温度缺失 15
```

A 台区（居民型）的越界点只有 **7 个**，但每一个都很极端：

| 越界原值 | 类型 |
|---|---|
| −68.78 / −59.34 | 负值（物理上不可能） |
| 1678.19 / 1776.24 / 1860.10 / 2240.30 | 尖峰（正常峰值约 740） |
| **9999.00** | 哨兵值（传感器掉线的经典占位） |

清洗后 `std` 从 **129.2674** 掉到 **72.1933** —— 只改 7 个点，方差就砍掉了四成。
最大单点修正量 **9480.56 kW**，全部来自那个 9999。

**为什么阈值要「按小时分组」**：

```python
med = s.groupby(s.index.hour).transform("median")     # 同小时的中位数
bad = ~s.between(0, 3 * med) & s.notna()
```

A 台区是居民型（晚高峰 + 早次峰），用全局中位数 ×3 也能过；但 B 台区是工业型
（夜间低、日间平台），**全局阈值会把整段日间平台误判成异常**。
换数据源时才暴露的坑，就是靠「一开始就分组」躲掉的。
"""
    ),
    md("## 任务 1：分组阈值清洗（§一）"),
    code(T1_CODE),
    md(
        """
## 第二节 · 特征：日历、温度的 V 形、滞后与滑窗

`make_features` 一次性给出 **24 个特征列**（表里还有目标列 `y`，所以列数是 25）：

| 组 | 列 | 备注 |
|---|---|---|
| 日历 | `h_sin/h_cos`、`d_sin/d_cos`、`m_sin/m_cos` | 周期整数必须编码，见下 |
| 标志位 | `is_weekend`、`is_holiday` | 2025 法定节假日共 672 小时 |
| 温度 | `T`、`T_dev = \\|T−22\\|`、`T2` | 温度—负荷是 **V 形**，不是线性 |
| 滞后 | `lag1/2/3/24/25/48/168` | 周期对齐比「离得近」更重要 |
| 滑窗 | `ma3/24/168`、`sd3/24/168` | **先 `shift(1)` 再 `rolling`** |

### 周期整数为什么必须 sin/cos 编码

23 点与 0 点在**实际时间**上只差 1 小时，但作为**整数**相差 23。
把它们放到单位圆上再算**弦长**：

$$\\text{chord} = \\sqrt{(\\sin\\theta_{23}-\\sin\\theta_{0})^2 + (\\cos\\theta_{23}-\\cos\\theta_{0})^2} = 0.261052$$

不编码时这个「距离」是 23 —— 模型会以为 23 点与 0 点是全天最远的两个时刻。
真实弦长 0.261052 ≈ 2·sin(π/24)，正好对应 1 小时的真实间隔。

### 温度是 V 形，不是线性

| 度量 | 相关系数 |
|---|---|
| `T` ↔ 负荷（线性） | **−0.188929** |
| `\\|T − 22\\|` ↔ 负荷 | **0.406437** |

线性相关看起来「几乎没关系」，换成「离舒适区的距离」之后立刻涨到 0.41。
**先想清楚物理关系的形状，再决定特征形式** —— 这是本章最省事的一次涨点。

### 滑窗特征的两个硬约束

1. **先 `shift(1)` 再 `rolling(n)`**：否则窗口含当前时刻，而当前时刻与目标高度相关。
2. **`rolling` 默认 `min_periods=n`**：本数据前 **168 行**（最长窗口 168）全是 NaN，
   `dropna()` 会如实丢掉它们 —— 8592 = 8760 − 168。
"""
    ),
    md("## 任务 2：特征工程与周期编码（§二）"),
    code(T2_CODE),
    md(
        """
## 第三节 · 切分与泄漏自检

```
样本 8592 | 训练 6873 | 测试 1719 | 特征 24 列
训练 2025-01-08 00:00:00 ~ 2025-10-21 08:00:00
测试 2025-10-21 09:00:00 ~ 2025-12-31 23:00:00
训练 y mean/std 452.7457/71.7995 | 测试 483.9447/65.5642
```

三条**必须**做的切分纪律：

1. **按时间切，绝不 shuffle**。随机切分会让未来样本混进训练集，MAE 会"变好"且完全不报错。
2. **标准化参数只能来自训练段**。哪怕本数据上测不出差别，也要照做 —— 换个数据就会翻车。
3. **切分必须在特征构造之后、模型之前**，且验证阶段不许再动。

### 泄漏自检：用「手工切片」对账

光说"我没泄漏"没有说服力。可执行的办法是**挑一行，把滑窗统计量的口径手工重算一遍**：

`OK` 第 300 行对应全序列位置 `300 + 168 = 468`。`rolling` 默认右对齐，位置 `j` 的
`ma24` 窗口是 `j-23 .. j`，而输入已经 `shift(1)`，所以它实际覆盖 `y[j-24 : j]`：

| 列 | 表内值 | 手工值 |
|---|---|---|
| `ma24` | 549.2075 | `y.iloc[444:468].mean()` = 549.2075 |
| `sd24` | 70.7176 | `y.iloc[444:468].std(ddof=1)` = 70.7176 |
| `lag1` | 543.8600 | `y.iloc[467]` = 543.8600 |

两边一致 ⇒ 滑窗确实只用到了「目标前一小时及以前」。注意 `sd` 用的是**样本标准差**
（`ddof=1`），对应 `pandas.rolling().std()` 的默认行为 —— 用 `np.std` 的默认 `ddof=0`
会差一个 `√(n/(n−1))` 的系数。
"""
    ),
    md("## 任务 3：按时间切分 + 泄漏自检（§三）"),
    code(T3_CODE),
    md(
        """
## 第四节 · 四条朴素基线：周周期赢了日周期

| 模型 | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| ① 持续性 `lag1` | 30.2482 | 39.2032 | 6.1115 | 0.642265 |
| ② 日周期 `lag24` | 21.5898 | 27.9801 | 4.4492 | 0.817771 |
| **③ 周周期 `lag168`** | **19.1805** | **24.6527** | **3.9295** | **0.858536** |
| ④ 周周期×水平比 | 21.6910 | 27.7810 | 4.4740 | 0.820355 |

**③ 比 ② 好 11.2%**（21.5898 → 19.1805）。原因有二：

1. 测试段落在 **10~12 月**（秋冬季），负荷整体在爬升。`lag24` 每天都在追当前水位，
   但 `lag168` 用「上周同一时刻」，天然对上**星期几**；本数据有强周周期
   （A 台区周末略有抬升，B 台区周末腰斩）。
2. 只有 70 个发布日时，**工作日/周末错位的代价**远大于「水位偏差」的代价。

那 ④ 为什么反而变差（19.1805 → 21.6910）？因为「水平比」是用**最近 24 小时 / 再前 24 小时**
估的，短期比值噪声很大，把本来干净的周周期基线**搅浑**了。
**加减一个修正项之前，先量它带来的方差**。

> **可背结论**：日周期是"默认基线"而不是"正确基线"。序列有周周期时，
> **先比 `lag168`，再决定要不要用 `lag24`**。
"""
    ),
    md("## 任务 4：四条朴素基线（§四）"),
    code(T4_CODE),
    md(
        """
## 第五节 · 从基线到学习模型

| 模型 | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| ⑤ Ridge(α=1) | 14.2251 | 18.2559 | 2.9471 | 0.9224 |
| **⑥ LightGBM** | **12.6689** | **16.2461** | **2.6106** | **0.9386** |

两个模型都比最好的基线（19.1805）低一大截，说明**特征里确实有基线用不到的信息**。

特征重要性 top8：

| 特征 | `lag24` | `lag1` | `lag168` | `T_dev` | `T` | `lag25` | `h_cos` | `sd24` |
|---|---|---|---|---|---|---|---|---|
| split 次数 | **692** | 674 | 632 | 616 | 550 | 480 | 479 | 461 |

三条「滞后」包揽前三 —— 这符合直觉。真正值得注意的是第 4、5 名：
**`T_dev`（离 22℃ 的距离）排在原始温度 `T` 前面**，与前面那个 0.19 → 0.41 的相关系数
完全对上：模型也在用"V 形"这个形状。

重要性最低的 5 个里出现了 `is_weekend`（35）。这不代表周末不重要 ——
**周周期已经被 `lag168` 吸收了**，标志位就成了冗余。特征重要性永远要**成组**读，
不要单看某一个。

> **树模型不需要标准化**，但 **Ridge 必须**。所以两条路线各自预处理；
> 共用一份「标准化后的表」反而会让 GBDT 丢掉可解释的分裂点。
"""
    ),
    md("## 任务 5：Ridge 与 LightGBM（§五）"),
    code(T5_CODE),
    md(
        """
## 第六节 · 误差的时段结构

整体 MAE 12.6689 只是个平均数。**它藏在哪个时段**才是能落地的信息。

| 时段 | 小时 | LightGBM | 日周期基线 | 周周期基线 |
|---|---|---|---|---|
| 夜间 | 0-5 | 11.3963 | 18.8260 | 16.8589 |
| 早峰 | 6-9 | 13.0667 | 22.4204 | 19.9580 |
| 日间 | 10-16 | 11.2119 | 20.3810 | 18.2461 |
| **晚峰** | **17-21** | **16.7882** | **26.7640** | **23.7982** |
| 深夜 | 22-23 | 10.4362 | 19.4134 | 16.2289 |

**晚峰（17-21 点）的 MAE 是日间的 1.5 倍**。这不是模型不行 —— 三条基线在晚峰的
误差同样是全天最高的（26.76 / 23.80）。晚峰**本身就更难预测**：爬坡陡、
受行为随机性影响大。分时段的正确读法是**看相对值**：

| 时段 | LightGBM / 日周期基线 |
|---|---|
| 夜间 | 0.605 |
| 日间 | 0.550 |
| **晚峰** | **0.627** |

即使除以基线，晚峰也没被拉平 —— 说明这里还剩没抓住的结构。
**下一步该往这儿投时间，而不是全天均匀地加特征。**

分月看更平（10 月 12.1156 / 11 月 12.5286 / 12 月 12.9943），
12 月略高，与冬季采暖负荷爬升一致。
"""
    ),
    md("## 任务 6：误差的时段结构（§六）"),
    code(T6_CODE),
    md(
        """
## 第七节 · 日前 24 步：直接法

真实业务里不是「预测下一小时」，而是**每天 0 点一次性给出未来 24 小时**。
修正一下评估口径：**发布时刻 = 测试段里每天 0 点**，共 **70 天**
（2025-10-22 ~ 2025-12-30）；真值矩阵是 `(70, 24)`。

### 直接法（direct）：为每个 h 训一个模型

```python
for h in range(1, H + 1):
    X = pd.concat([原点特征, 目标时刻 t+h 的日历], axis=1)
    models[h] = fit_lgb(X, yv[tr_pos + h])
```

两个关键点：

1. **特征分两半**：一半锁死在原点 `t`（`lag1`、`ma3`…），一半来自**目标时刻** `t+h`
   （日历、温度）。日历是确定可算的，温度在真实场景里用预测值 —— 本节用实况替代，
   属于已知简化。
2. **训练样本必须满足 `tr_pos + h <= tr_pos[-1]`**。否则 `t+h` 会落进测试段，
   模型在训练时就看到了测试段的答案。这个错误**不会报错**，只会让你的 MAE 好看得可疑。

代价也是明摆着的：**24 个模型**，训练时间和内存都是单步的 24 倍。
"""
    ),
    md("## 任务 7：直接法多步预测（§七）"),
    code(T7_CODE),
    md(
        """
## 第八节 · 递归法，以及那个 off-by-one

递归法只训 **1 个**「下一小时」模型，然后反复代入：

```python
ext = list(yv[:p + 1])           # ⚠️ 要预测 t+1，历史必须含 t 本身
for k in range(1, H + 1):
    yhat = model(red_row(p + k, ext, ...))
    ext.append(yhat)
```

### ⚠️ off-by-one：本章最容易踩、且最不出声的坑

`yv[:p + 1]` 的**最后一个元素**是 `y[t]` —— 这正是预测 `y[t+1]` 所需要的。
写成 `yv[:p]`（末尾是 `y[t-1]`）整体晚一格，特征全部"错位一小时"。

实测代价：

| 写法 | 日均 MAE |
|---|---|
| `yv[:p]`（晚一格） | **21.1492** |
| `yv[:p + 1]`（正确） | **13.1118** |

**差 8.04 kW（×1.61），而且不报任何错**。因为 `red_row` 里的 `hist[-24]` 依然是「某个真实存在的
24 小时前的值」，数值完全合法，只是对应错了时刻。这类错误只能靠**和上界对照**发现 ——
所以下一步要做 teacher forcing。

### 上界诊断（teacher forcing）

用**同一个模型、同一套特征**，但历史一律喂**真值**（含未来真值）。这不是能上线的做法，
它的唯一作用是划出**这套特征的天花板**：

| 做法 | 日均 MAE | 能不能上线 |
|---|---|---|
| ❸ 精简特征 + 真值历史 | 13.0980 | ❌ 作弊，只作诊断 |
| ❹ 精简特征 + 递归历史 | 13.1118 | ✅ |
"""
    ),
    md("## 任务 8：递归法与 teacher forcing 上界（§八）"),
    code(T8_CODE),
    md(
        """
## 第九节 · 归因：拆开「特征差」与「误差累积」

这一步是本章的核心。三档结果摆在一起：

| 做法 | 日均 MAE | 提升/损失 |
|---|---|---|
| ❶ 日周期基线 | 21.6100 | — |
| ❷ 直接法（全特征，锁在原点） | 16.1107 | 比基线好 25.5% |
| ❸ 精简特征 + 真值历史（上界） | 13.0980 | 比直接法好 3.0127 |
| ❹ 精简特征 + 递归历史（可上线） | 13.1118 | 只比 ❸ 差 **0.0138** |

拆开看：

| 项 | 值 | 解释 |
|---|---|---|
| 特征对齐收益 = ❸ − ❷ | **−3.0127** | **精简特征（对齐到目标时刻）比原点特征好 3.01 kW** |
| 误差累积 = ❹ − ❸ | **+0.0138** | **递归只付出 0.0138 kW 的代价** |
| 误差累积占比 | **0.46%** | 主导因素是特征，不是累积 |

### 为什么这里「递归误差累积」几乎为零？

因为 `red_row` 用的 `lag24 / lag48 / lag168`，在 **h ≤ 24** 时**全部是真值**：
预测 `y[t+h]` 需要 `y[t+h-24]`，而 `t+h-24 ≤ t` 永远落在观测区间内。
真正会混入预测值的只有 `ma24 / ma168 / sd24` 这类滚动量 —— 它们的权重远低于 `lag*`。

> **可背结论**：教材里「递归法误差会指数累积」在 **horizon ≤ 周期长度** 时不成立。
> 判断标准不是「教材怎么说」，而是 **`h` 是否越过你依赖的最长滞后**。
> 本数据最长滞后 168 小时，horizon 只有 24 —— 所以递归安全。
> 把 horizon 提到 200 小时以上，结论就会翻过来。

顺带一个反直觉：**直接法反而更差**。因为它 24 个模型都只用「原点 `t` 的特征」，
随着 h 增大，`lag1`、`ma3` 这些量与 `y[t+h]` 的相关性迅速衰减；而递归法用的是
「目标时刻前 1 小时」的特征，h 再大也不衰减。看逐 h 曲线：

| | h=1 | h=24 | 倍数 |
|---|---|---|---|
| ❶ 日周期基线 | 16.3739 | 20.7994 | **×1.270**（越往后越差） |
| ❷ 直接法（原点特征） | 12.0749 | 13.6702 | ×1.132 |
| ❹ 递归法 | 10.6980 | 12.0191 | ×1.123 |

直接法那条 ×1.132 看着"稳"，其实是**特征与目标的错配在各 h 上均匀地烂**；
递归法起点更低（h=1 只有 10.6980），代价是随 h 缓慢变差。
"""
    ),
    md("## 任务 9：归因拆解（§九）"),
    code(T9_CODE),
    md(
        """
## 第十节 · LSTM 对照：一个负结果，和一个 23.5 倍的教训

前面训练与评估留下的问题是：**换神经网络能不能更好？**
本节用最朴素的形态试一次：把 `[负荷, 温度, sin(h), cos(h)]` 四个通道的过去 168 小时
喂进单层 LSTM（hidden 48），一次输出未来 24 步。

### 第一遍：不做任何归一化

| 量 | 值 |
|---|---|
| 训练 loss（首个 epoch → 末个） | **450.94540 → 380.24695** |
| 日均 MAE | **409.7550** |

loss 几乎没降，MAE 424 意味着**模型什么都没学到**（负荷均值约 460，
"永远输出均值"的 MAE 也在 400 量级）。

原因在看数据时就已经摆在眼前：三条通道的**量级差了整整两个数量级** ——
负荷约 460、温度约 20、sin/cos 在 `[-1, 1]`。梯度被负荷通道彻底主导，
另外三条通道对网络来说约等于常数。

### 第二遍：只改四个数

```python
mu, sd   = yv[:cut_full].mean(), yv[:cut_full].std()      # 负荷
tmu, tsd = tv[:cut_full].mean(), tv[:cut_full].std()      # 温度
```

其余代码**一个字都没改**：

| 量 | 不归一化 | 归一化 |
|---|---|---|
| loss（首 → 末） | 450.95 → 380.25 | **0.27880 → 0.03995** |
| 日均 MAE | 409.7550 | **17.4668** |
| 差距 | | **23.5 倍** |

### 但归一化之后还是输了

| 模型 | 日均 MAE |
|---|---|
| ❹ 递归 + LightGBM | **13.1118** |
| LightGBM 单步 | **12.6689** |
| LSTM（归一化，hidden 48，20 epoch） | 17.4668 |
| LSTM（hidden 64，40 epoch，调过学习率） | 19.2211 |

**调参完全没用**（17.4668 → 19.2211，反而更差）。这不是调得不够，是**信息量不够**：
训练集只有 6873 个点，参数量虽然只有 1.15 万，但 LSTM 需要从原始序列里**自己学出**
`lag24`、`lag168`、`|T−22|` 这些量；而 GBDT 是我们**手工喂给它的**。
在「1 年 × 单台区 × 小时级」这个规模上，**特征工程的收益压过了表示学习的收益**。

> **可背结论**：
> ① 神经网络输入**必须归一化** —— 这不是"调优技巧"，是能不能训起来的分界线；
> ② 小样本强周期场景下，先把手写特征 + GBDT 做到位，再考虑上网络；
> ③ 归一化参数**只能来自训练段**（本节用的是 `cut_full` 之前的数据）。
"""
    ),
    md("## 任务 10：LSTM 对照实验（§十）"),
    code(T10_CODE),
    md(
        """
## 第十一节 · 残差诊断：还漏了什么

MAE 12.6689 只告诉你"平均差多少"，不告诉你"还差在哪"。把残差当序列再查一遍：

| 检验 | 结果 | 读法 |
|---|---|---|
| Ljung-Box p（lag 24 / 48 / 72） | **3.17e-06 / 6e-08 / 6.93e-06** | 全部 < 0.01，**强烈拒绝白噪声** |
| 残差 ACF(1) | −0.055219 | 几乎没有 1 步结构 |
| 残差 ACF(24) | −0.111229 | 日周期还有残留（且**为负**） |
| 残差 ACF(168) | +0.032333 | 周周期基本被 `lag168` 吃干净了 |
| 残差 std / y std | 0.2456 | 残差标准差只有目标标准差的 1/4 |

**残差不是白噪声 ⇒ 还有可抓的结构。** 但注意 `ACF(24) = −0.111` 是**负的** ——
这说明模型在日周期上**过冲**了（白天预测偏高、夜里偏低，残差符号随小时翻转），
而不是"漏掉了日周期"。这类负自相关的修法通常是**降低模型的日周期敏感度**
（比如把 `lag24` 换成 `lag24` 的滑动平均），而不是再加更强的日历特征。

区分这两种情形很重要：
- **正 ACF(24)** → 日周期没抓够 → 加特征
- **负 ACF(24)** → 日周期抓过头 → 加平滑 / 降复杂度

**残差检验的价值就在这里：它把"还能不能涨点"和"往哪个方向涨"分开回答了。**
"""
    ),
    md("## 任务 11：残差白噪声检验（§十一）"),
    code(T11_CODE),
    md(
        """
## 第十二节 · 三张图：把结论画出来

报告里放三张图就够：

1. **最后 7 天的预测曲线** —— 让人一眼看出「跟得上形状吗、峰值高低对不对」
2. **按小时的平均绝对误差** —— 三类模型三条线，直接暴露"哪个时段最难"
3. **特征重要性 top15** —— 交代模型到底在用什么

画整年、画全部 70 天的预测没有意义（分辨率不够、也看不出差异）。
**图要服务于你要下的那个结论。**
"""
    ),
    md("## 任务 12：画三张图（§十二）"),
    code(T12_CODE),
    md(
        """
## 第十三节 · 自动生成结题报告

竞赛交付物里，一个**能自动重算的报告**比手写结论值钱得多 —— 数据换了、模型换了，
报告里的数字跟着变，不会出现"结论写 12.6，代码跑出来 15.2"这种事。

写法上只有一个要点：**所有数字都用 f-string 从变量里取，不要手打**。

```python
f"- 最佳单步模型：{best_name}，MAE = {best_mae:.4f} kW"
```

本章用 `assert "12.6689" in REPORT` 这类断言把报告**钉在真值上** ——
报告一旦和计算结果脱节，测试立刻红。
"""
    ),
    md("## 任务 13：自动生成结题报告（§十三）"),
    code(T13_CODE),
    md(
        """
## 第十四节 · 跨台区泛化：同一套代码，换一份数据

B 台区是**工业型**：日间平台、周末腰斩。全程只换数据源，代码一个字不改。

| 模型 | B 台区 MAE | A 台区 MAE |
|---|---|---|
| ① 持续性 `lag1` | 59.8748 | 30.2482 |
| ② 日周期 `lag24` | **71.3448** | 21.5898 |
| ③ 周周期 `lag168` | **17.1623** | 19.1805 |
| ⑥ LightGBM | 13.1559 | 12.6689 |

**日周期基线在 B 上彻底崩了**（71.3448，比持续性还差），而周周期基线仍很强。
原因就是那个「周末腰斩」：`lag24` 在周末会用**工作日**的数据去预测，
偏差直接就是半个台区。

误差的时段分布也**正好反过来**：

| 时段 | A 台区（居民型） | B 台区（工业型） |
|---|---|---|
| 日间 10-16 | 11.2119 | **21.7747** |
| 晚峰 17-21 | **16.7882** | 12.2618 |

A 台区难在晚峰（爬坡陡、行为随机），B 台区难在日间（负荷高、周末/工作日切换）。

> **可背结论**：**"哪个时段最难"没有普适答案，它由负荷形态决定。**
> 上线前必须按目标台区自己的误差分布来决定"往哪儿投优化"。
> 同理，**"日周期基线够不够用"也是台区相关的** —— A 上够用，B 上灾难。
"""
    ),
    md("## 任务 14：跨台区泛化（§十四）"),
    code(T14_CODE),
    md(
        """
## 易错点清单

1. **`drop_duplicates` 必须在 `set_index` 之前**。之后再做，重复时刻已经进了索引。
2. **`asfreq` ≠ `resample`**。前者只对齐补 NaN，后者真聚合。想"数出缺了几小时"只能用前者。
3. **异常阈值必须按小时分组**。全局中位数 ×3 会把工业台区的整段日间平台误杀。
4. **哨兵值 `9999` 不会自己暴露**。它落在 `between(0, 3×med)` 之外，但如果你偷懒只判负值，它就留下了。
5. **插值要指定 `limit_direction="both"`**。默认只向后插，序列**开头**的缺失补不掉。
6. **滑窗必须先 `shift(1)`**。`rolling` 默认右对齐且含当前点。
7. **`rolling(n)` 默认 `min_periods=n`**。168 小时的最长窗口会让表头 168 行全 NaN，`dropna` 必须做且要解释清楚。
8. **`sd` 的 `ddof` 要对齐**。`pandas.rolling().std()` 是 `ddof=1`，`np.std` 默认 `ddof=0`。
9. **时序只能按时间切**。随机切分让 MAE 虚降且完全不报错。
10. **标准化参数只能来自训练段**。哪怕本数据测不出差别也要照做。
11. **周期整数必须 sin/cos 编码**。23 点与 0 点的整数距离是 23，真实距离是 1。
12. **温度要转成「离舒适区的距离」**。线性相关 −0.19，`|T−22|` 相关 +0.41。
13. **日周期不一定是正确基线**。有强周周期时先比 `lag168`；B 台区上两者差 4 倍。
14. **别乱加修正项**。"周周期 × 水平比"看着聪明，实测反而把基线搅浑（19.18 → 21.69）。
15. **`pd.concat` 拼表前先查列重名**。重名会让 LightGBM 直接抛 `Feature ... appears more than one time`。
16. **直接法的训练样本要卡 `tr_pos + h <= tr_pos[-1]`**。越界即泄答，且不报错。
17. **递归法的历史必须含原点本身**（`yv[:p+1]`）。写成 `yv[:p]` 整体晚一格，MAE 从 13.11 涨到 21.15。
18. **多步评估要先统一「发布口径」**。按每天 0 点发布（70 天）和逐小时发布（1695 点）算出来的 MAE 不可比。
19. **递归法的误差累积在 `h ≤ 最长滞后` 时可以忽略**。本数据 horizon 24、最长滞后 168，累积只有 0.0138 kW。
20. **比"哪个模型好"更该问"哪个环节好"**。这里 3.01 kW 来自特征对齐，只有 0.0138 kW 来自累积。
21. **不只是看 MAE，要看它藏在哪个时段**。A 台区晚峰最难，B 台区日间最难。
22. **残差 ACF(24) 为负是"过冲"，为正才是"漏抓"**。两者修法完全相反。
23. **神经网络输入必须归一化**。不归一化时 loss 基本不降，MAE 409.76 vs 17.47，差 23.5 倍。
24. **归一化救不了小样本**。LSTM 从 17.47 调到 19.22（越调越差），仍然输给 LightGBM 的 12.67。
25. **报告里的数字必须来自变量**。手打的结论迟早和代码脱节；用 assert 把它钉住。

## 本章小结

### 一条可复用的流水线（顺序不可换）

```
① 规整    drop_duplicates → sort_values → asfreq("h")
② 清洗    按小时分组中位数×3 判异常 → 正负双向 interpolate
③ 特征    日历 sin/cos + |T−22| + 滞后(shift) + 滑窗(先 shift 再 rolling)
④ 切分    按时间 8:2；标准化参数只 fit 训练段；手工切片对账
⑤ 基线    持续性 / 日周期 / 周周期 / 周周期×水平比 —— 先量基线，再上模型
⑥ 模型    Ridge（要标准化）与 LightGBM（不要标准化）
⑦ 评估    MAE/RMSE/MAPE/R² + 分时段 + 分月 + 残差 Ljung-Box/ACF
⑧ 多步    直接法 / 递归法，并用 teacher forcing 划上界做归因
⑨ 交付    自动生成报告（数字全部来自变量），跨台区复测
```

### 三个「必须背下来」的判定规则

| 场景 | 规则 |
|---|---|
| 基线选哪个 | 有强周周期就**先试 `lag168`**；`lag24` 只是默认值，不是正确答案 |
| 多步用直接还是递归 | 看 **horizon 是否越过最长滞后**：没越过 → 递归（1 个模型、误差累积≈0）；越过了 → 直接法 |
| 加不加特征 | 先看**残差 ACF 的符号**：正 → 漏抓（加特征）；负 → 过冲（加平滑） |

### 三组关键数字

| 指标 | 值 | 含义 |
|---|---|---|
| 最好基线 / 最好单步模型 | 19.1805 / **12.6689** | 特征工程带来 **34%** 提升 |
| 日前 24 步（直接 / 递归 / 基线） | 16.1107 / **13.1118** / 21.6100 | 递归比基线好 **39.3%** |
| 特征对齐 / 误差累积 | **3.0127** / 0.0138 | 决定多步效果的是前者，不是后者 |

### 与前面章节的对应

| 本节环节 | 详细展开在 |
|---|---|
| 规则时间轴、sin/cos、滞后滑窗、泄漏三问 | `ch01` |
| ADF / 差分 / ACF·PACF / ARIMA | `ch02` |
| RNN 单步前向与 BPTT | `ch03` |
| LSTM / GRU 门控与梯度流 | `ch04` |
| 多步策略（直接 / 递归） | `ch05` |
| Dataset 与按时间切分 | `ch06` |
| 训练循环、早停、评估口径 | `ch07` |
| 基线体系、LightGBM、残差诊断 | `ch08` |

**下一步**：`05_timeseries` 到此结束。把这套流程套到真实赛题数据上时，
优先复查的是 **③ 特征口径** 与 **⑧ 多步口径** —— 这两处出错都不会报错，
只会让你的分数虚高或虚低。
"""
    ),
]

# =========================================================================== #
# 4. 练习版                                                                   #
# =========================================================================== #

EXERCISE = [
    md(
        """
# final_01 台区负荷日前预测全流程（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`../data/load_curve.csv`（2 台区 × 8760 h，含重复行 / 缺失 / 尖峰 / 负值 / 哨兵 9999）。
>
> **每题都跟了 `# 提示：`**，照着提示能独立做完。
> 如果 assert 的数字对不上，**先回头看上一步的输出**，别急着改 assert —— 那些数字都是实跑出来的。

## 环境提示（本章必读）

`lightgbm` 与 `torch` 在本机各带一套 OpenMP（`libomp` / `libiomp`），
同时开多线程会互相踩内存直接崩掉进程，`try/except` 都抓不住。
`IMPORTS` 已经**在任何 import 之前**把 `OMP_NUM_THREADS` 钉成 `1`，
**不要删掉那两行**，否则任务 10（LSTM）会莫名段错误。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 分组阈值清洗 | 4 | §一 |
| 2 | 特征工程与周期编码 | 2 | §二 |
| 3 | 按时间切分 + 泄漏自检 | 7 | §三 |
| 4 | 四条朴素基线 | 2 | §四 |
| 5 | Ridge 与 LightGBM | 2 | §五 |
| 6 | 误差的时段结构 | 2 | §六 |
| 7 | 直接法多步预测 | 4 | §七 |
| 8 | 递归法与 teacher forcing | 3 | §八 |
| 9 | 归因拆解 | 1 | §九 |
| 10 | LSTM 对照实验 | 2 | §十 |
| 11 | 残差白噪声检验 | 1 | §十一 |
| 12 | 画三张图 | 3 | §十二 |
| 13 | 自动生成结题报告 | 1 | §十三 |
| 14 | 跨台区泛化 | 3 | §十四 |

> **建议顺序**：1 → 2 → 3 必须先做完，后面全部依赖它们产出的 `y / t / FEAT / tr / te`。
> 第 7、8 两题依赖第 3 题的 `cut`；第 10 题依赖第 7 题的 `opos`。
"""
    ),
    code(IMPORTS),
    code(SCAFFOLD),
    code(SETUP),
    code(PLOT_SETUP),
    md("## 任务 1：分组阈值清洗（§一）"),
    code(T1_CODE),
    md("## 任务 2：特征工程与周期编码（§二）"),
    code(T2_CODE),
    md("## 任务 3：按时间切分 + 泄漏自检（§三）"),
    code(T3_CODE),
    md("## 任务 4：四条朴素基线（§四）"),
    code(T4_CODE),
    md("## 任务 5：Ridge 与 LightGBM（§五）"),
    code(T5_CODE),
    md("## 任务 6：误差的时段结构（§六）"),
    code(T6_CODE),
    md("## 任务 7：直接法多步预测（§七）"),
    code(T7_CODE),
    md("## 任务 8：递归法与 teacher forcing 上界（§八）"),
    code(T8_CODE),
    md("## 任务 9：归因拆解（§九）"),
    code(T9_CODE),
    md("## 任务 10：LSTM 对照实验（§十）"),
    code(T10_CODE),
    md("## 任务 11：残差白噪声检验（§十一）"),
    code(T11_CODE),
    md("## 任务 12：画三张图（§十二）"),
    code(T12_CODE),
    md("## 任务 13：自动生成结题报告（§十三）"),
    code(T13_CODE),
    md("## 任务 14：跨台区泛化（§十四）"),
    code(T14_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 14 个任务的所有 assert 全部通过
- [ ] `drop_duplicates` 必须在哪一步之前？不这样做会怎样？
- [ ] 清洗阈值为什么必须「按小时分组」？换成 B 台区会发生什么？
- [ ] 本数据里 `9999` 这个值代表什么？它在 7 个越界点里占了几个？
- [ ] 温度用 `|T−22|` 而不是 `T`，相关系数从多少涨到多少？
- [ ] `rolling(168)` 为什么会让表头 168 行全变 NaN？8592 这个数字怎么来的？
- [ ] 泄漏自检里 `ma24` 的窗口实际覆盖 `y` 的哪一段？为什么不是 `y[j-23:j+1]`？
- [ ] 为什么周周期基线（19.1805）比日周期基线（21.5898）好？
- [ ] 「周周期 × 水平比」为什么反而更差？这说明加修正项之前该先量什么？
- [ ] 直接法的训练样本为什么要卡 `tr_pos + h <= tr_pos[-1]`？
- [ ] 递归法里 `yv[:p+1]` 写成 `yv[:p]` 会让 MAE 从多少涨到多少？
- [ ] teacher forcing 为什么不能上线？它的作用是什么？
- [ ] 「特征对齐收益」与「误差累积」各是多少？谁在主导？
- [ ] 为什么这里的递归误差累积几乎为零？把 horizon 提到 200 小时会怎样？
- [ ] LSTM 不归一化时 loss 从 451 降到多少？归一化后是 0.287 降到多少？
- [ ] 为什么 LSTM 调参（17.4668 → 19.2211）反而更差？
- [ ] 残差 `ACF(24) = −0.111` 是「漏抓」还是「过冲」？该怎么修？
- [ ] A 台区和 B 台区「最难的时段」分别是什么？为什么相反？
- [ ] B 台区的日周期基线 MAE 是多少？为什么这么差？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 原始行数 / 去重后 / 重复行 | 17526 / 17520 / 6 |
| A 台区缺失（负荷 / 温度） | 70 / 15 |
| 越界点数（其中负值） | 7 / 2 |
| 清洗前 std / 清洗后 std | 129.2674 / 72.1933 |
| 最大单点修正量 | 9480.56 kW |
| 特征表形状 / 含 NaN 行 | (8760, 25) / 168 |
| 节假日小时 / 周末小时 | 672 / 2496 |
| 23↔0 编码弦长 | 0.261052 |
| 温度线性 / V 形相关 | −0.188929 / 0.406437 |
| 样本 / 训练 / 测试 | 8592 / 6873 / 1719 |
| 测试段 y 均值 | 483.9447 |
| ① 持续性 `lag1` MAE | 30.2482 |
| ② 日周期 `lag24` MAE | 21.5898 |
| ③ 周周期 `lag168` MAE | 19.1805 |
| ④ 周周期×水平比 MAE | 21.6910 |
| ⑤ Ridge(α=1) MAE | 14.2251 |
| ⑥ LightGBM MAE / R² | 12.6689 / 0.9386 |
| 特征重要性 top1 / top2 | `lag24` 692 / `lag1` 674 |
| 分时段 MAE：夜间 / 日间 / 晚峰 | 11.3963 / 11.2119 / 16.7882 |
| 分月 MAE：10 / 11 / 12 月 | 12.1156 / 12.5286 / 12.9943 |
| 发布日数 / 真值矩阵 | 70 / (70, 24) |
| ❷ 直接法 日均 MAE | 16.1107 |
| ❹ 递归法 日均 MAE | 13.1118 |
| ❶ 日周期基线 日均 MAE | 21.6100 |
| ❸ teacher forcing 日均 MAE | 13.0980 |
| 特征对齐收益 / 误差累积 | −3.0127 / +0.0138 |
| 递归 off-by-one（`yv[:p]`）MAE | 21.1492 |
| LSTM 不归一化 MAE | 409.7550 |
| LSTM 归一化 MAE | 17.4668 |
| LSTM 归一化前后倍数 | 23.5 |
| 残差 Ljung-Box p（24/48/72） | 3.17e-06 / 6e-08 / 6.93e-06 |
| 残差 ACF(1) / ACF(24) / ACF(168) | −0.055219 / −0.111229 / +0.032333 |
| 残差 std / y std | 0.2456 |
| B 台区 日周期 / 周周期 MAE | 71.3448 / 17.1623 |
| B 台区 LightGBM MAE | 13.1559 |
| B 台区 日间 / 晚峰 MAE | 21.7747 / 12.2618 |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
