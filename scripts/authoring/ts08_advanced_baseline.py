#!/usr/bin/env python3
"""ch08 —— 进阶与基线：朴素基线家族 / 特征工程 + LightGBM / HistGB 对照 / TCN / ProbSparse Transformer

本章主线（七条）：

1. **朴素基线家族**：`last value` / `seasonal naive(t-24, t-168)` / `moving average`
   —— 24 步 MAE 就是「及格线」，没打过基线就没资格说模型好
2. **特征工程 + LightGBM 真基线**：滞后(1/2/3/24/25/168) + 滑窗(ma24/ma168/std24/min/max)
   + 日历 sin·cos（不给整数）+ 节假日 + 温度（含 `|T-22|`），24 个步长各训一个模型
3. **与 sklearn `HistGradientBoostingRegressor` 同族对照**：同一份特征、同一切分、比 MAE
4. **TCN**：手写 `CausalConv1d`（左侧 padding）+ 膨胀卷积 + 残差块；
   给出感受野公式 `RF = 1 + (k-1)·Σd` 与**实测值**
5. **Transformer 时序（Informer 的 ProbSparse 简化版）**：只对 `top-u` 个 query 做注意力
   （`u = c·ln L`），并给出「打分次数」的节省比例随 L 的变化
6. **蒸馏**：把序列长度砍半的机制与代价
7. **全基线对比表** + 「什么时候该上深度模型」

数据：`data/load_curve.csv` → A 台区（居民型）8760 h，`clean_series` 后取 `.values`
口径：`window = 72`、`horizon = 24`、起点从 `START = 168` 开始、按时间 8:2 切分（与 ch07 完全一致）
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ts08_advanced_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "05_timeseries"
NAME = "ch08_advanced_baseline"

# =========================================================================== #
# 1. 公共导入 / 数据加载 / 脚手架（两版都给，不挖空）                          #
# =========================================================================== #

IMPORTS = '''import math
import os
import time
from pathlib import Path

# 本题要同时用 lightgbm 与 torch，而这两个库各自静态链接了一份 OpenMP 运行时
# （lightgbm 走 libomp，torch 走 libiomp）。在 macOS/arm64 上让两份运行时各自
# 开满线程，会撞出 `OMP: Error #179: Function pthread_mutex_init failed`，
# 内核直接段错误。**在 import 之前**把线程数限成 1 就能绕开这个初始化冲突。
# 这是一条真实的工程踩坑记录，不是可有可无的仪式。
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path("data")          # notebook 的 cwd = coding/05_timeseries
CSV = DATA / "load_curve.csv"

pd.set_option("display.width", 190)
pd.set_option("display.max_columns", 60)

raw = pd.read_csv(CSV, encoding="utf-8-sig", parse_dates=["时间戳"])
print("原始行数", len(raw), "| 列", list(raw.columns))
print("lightgbm", lgb.__version__, "| torch", torch.__version__, "| CUDA 可用？", torch.cuda.is_available())
'''

SETUP = '''df = raw.drop_duplicates().sort_values("时间戳").reset_index(drop=True)
frame = df[df["台区编号"] == "STATION_A_01"].set_index("时间戳").sort_index()


def clean_series(s):
    """按**小时分组**的中位数做 3 倍阈值判异常，再线性插值（ch01 §1.3.2）。"""
    med = s.groupby(s.index.hour).transform("median")
    return s.where(s.between(0, 3 * med)).interpolate()


s = clean_series(frame["负荷值"].asfreq("h"))
ser = s.values
temp = frame["温度"].asfreq("h").interpolate()

print("A 台区 %d 个整点 | 清洗后 mean %.6f" % (len(ser), ser.mean()))
print("温度 mean %.4f | 缺失 %d" % (temp.mean(), int(temp.isna().sum())))
'''

PLOT_SETUP = '''import matplotlib
import matplotlib.pyplot as plt

# macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
print("matplotlib", matplotlib.__version__, "| 中文字体 PingFang SC")
'''

SCAFFOLD = '''# 脚手架：下面的常量与工具函数两版都有，你只填 @@todo 块里的内容
WINDOW, HORIZON = 72, 24
START = 168          # 起点从 168 开始：lag168 需要 168 小时历史（ch07 用同一口径）

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


def mae_all(y_true, y_pred):
    """24 步合起来的单个 MAE —— 本章所有对比表的排序依据。"""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mae_by_step(y_true, y_pred):
    """返回长度 HORIZON 的逐步 MAE。"""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=0)
'''

# =========================================================================== #
# 2. 任务代码块                                                              #
# =========================================================================== #

T1_CODE = '''# @@todo 构造预测起点：从 START 起，到「后面还留得下 24 步」为止
# @@hint np.arange(START, len(ser) - HORIZON + 1)
origins = np.arange(START, len(ser) - HORIZON + 1)
# @@end

# @@todo 目标矩阵 Y[t, h] = ser[t + h]（h 从 0 数，对应「预测第 h+1 步」）
# @@hint np.stack([ser[t:t + HORIZON] for t in origins])
Y = np.stack([ser[t:t + HORIZON] for t in origins])
# @@end

# @@todo 原始输入窗口 X[t] = ser[t - WINDOW:t]**不含 t**（深度模型吃这个）
# @@hint np.stack([ser[t - WINDOW:t] for t in origins])
X = np.stack([ser[t - WINDOW:t] for t in origins])
# @@end

# @@todo 按时间 8:2 切分（与 ch07 完全一致，绝不 shuffle）
# @@hint cut = int(len(origins) * 0.8)
cut = int(len(origins) * 0.8)
# @@end

X_test, Y_test = X[cut:], Y[cut:]
oc = origins[cut:]
t_test = frame.index[oc]
print("样本 %d | 切分点 %d | 训练 %d / 测试 %d" % (len(origins), cut, cut, len(oc)))
print("测试段 %s ~ %s" % (t_test[0], t_test[-1]))

# @@todo 基线①：last value —— 把**起点上一小时**的值复制 24 遍
# @@hint np.tile(ser[oc - 1][:, None], (1, HORIZON))；[:, None] 是把它变成列向量
P_last = np.tile(ser[oc - 1][:, None], (1, HORIZON))
# @@end

# @@todo 基线②：seasonal naive t-24 —— 用「昨天同一小时」
# @@hint ser[t + np.arange(1, HORIZON + 1) - 24]，h=1 取 t-23、h=24 取 t
P_sn24 = np.stack([ser[t + np.arange(1, HORIZON + 1) - 24] for t in oc])
# @@end

# @@todo 基线③：seasonal naive t-168 —— 用「上周同一小时」
# @@hint 与 t-24 只差一个数字：168 = 7 × 24
P_sn168 = np.stack([ser[t + np.arange(1, HORIZON + 1) - 168] for t in oc])
# @@end

# @@todo 基线④：moving average —— 过去 24 小时均值，同样复制 24 遍
# @@hint np.array([ser[t - 24:t].mean() for t in oc])[:, None] 再 np.tile
P_ma24 = np.tile(np.array([ser[t - 24:t].mean() for t in oc])[:, None], (1, HORIZON))
# @@end

BASELINES = [("last value", P_last), ("seasonal naive t-24", P_sn24),
             ("seasonal naive t-168", P_sn168), ("moving average 24", P_ma24)]
print()
print("%-22s %10s %10s %10s" % ("基线", "24步 MAE", "h=1 MAE", "h=24 MAE"))
for nm, P in BASELINES:
    print("%-22s %10.4f %10.4f %10.4f" % (nm, mae_all(Y_test, P), mae_by_step(Y_test, P)[0],
                                          mae_by_step(Y_test, P)[23]))
print()
print("及格线 = 最好的朴素基线 %.6f（%s）"
      % (min(mae_all(Y_test, P) for _, P in BASELINES),
         min(BASELINES, key=lambda kv: mae_all(Y_test, kv[1]))[0]))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
axes[0].bar([nm for nm, _ in BASELINES], [mae_all(Y_test, P) for _, P in BASELINES],
            color=["tab:gray", "tab:blue", "tab:green", "tab:orange"])
axes[0].set_ylabel("24 步 MAE (kW)")
axes[0].set_title("朴素基线家族：及格线之争")
axes[0].tick_params(axis="x", rotation=18)
for nm, P in BASELINES:
    axes[1].plot(np.arange(1, HORIZON + 1), mae_by_step(Y_test, P), marker="o", ms=3, label=nm)
axes[1].set_xlabel("预测步长 h（小时）")
axes[1].set_ylabel("MAE (kW)")
axes[1].set_title("朴素基线的误差随步长几乎平")
axes[1].legend(fontsize=8)
plt.tight_layout()

# ---- 验收 ----
assert len(origins) == 8569 and cut == 6855
assert X.shape == (8569, WINDOW) and X_test.shape == (1714, WINDOW)
assert X[0, -1] == ser[START - 1] and Y[0, 0] == ser[START], "窗口不含 t，目标是 t 起"
assert abs(mae_all(Y_test, P_last) - 60.524392) < 1e-5
assert abs(mae_all(Y_test, P_sn24) - 32.998794) < 1e-5
assert abs(mae_all(Y_test, P_sn168) - 31.560227) < 1e-5
assert abs(mae_all(Y_test, P_ma24) - 45.527850) < 1e-5
assert abs(mae_by_step(Y_test, P_last)[0] - 30.0939) < 1e-4
assert abs(mae_by_step(Y_test, P_last)[23] - 21.6066) < 1e-4
assert abs(mae_by_step(Y_test, P_sn168)[0] - 31.5845) < 1e-4
assert mae_all(Y_test, P_sn168) < mae_all(Y_test, P_sn24) < mae_all(Y_test, P_ma24) < mae_all(Y_test, P_last)
'''

T2_CODE = '''idx = pd.DatetimeIndex(frame.index[origins])      # 注意是**全部**起点，不是测试段
Ft = pd.DataFrame(index=np.arange(len(origins)))

for lag in (1, 2, 3, 24, 25, 168):
    # @@todo 滞后特征：把「往回看 lag 小时」的值写成 lag<lag> 列
    # @@hint ser[origins - lag]，列名用 "lag%d" % lag
    Ft["lag%d" % lag] = ser[origins - lag]
    # @@end

# @@todo 过去 24 小时的滑窗统计：均值 / 标准差 / 最小值 / 最大值
# @@hint 窗口是 ser[t - 24:t]（**不含 t**）；先用 np.stack 拼成 (n, 24) 再聚合
W24 = np.stack([ser[t - 24:t] for t in origins])
Ft["ma24"], Ft["std24"] = W24.mean(1), W24.std(1)
Ft["min24"], Ft["max24"] = W24.min(1), W24.max(1)
# @@end

# @@todo 过去 168 小时的滑动均值
# @@hint np.stack([ser[t - 168:t] for t in origins]).mean(1)
Ft["ma168"] = np.stack([ser[t - 168:t] for t in origins]).mean(1)
# @@end

for nm, period, val in (("hour", 24, idx.hour), ("dow", 7, idx.dayofweek),
                        ("month", 12, idx.month - 1)):
    # @@todo 把周期整数编码成 (sin, cos) 两列 —— **不给整数**，理由见 §8.2.1
    # @@hint 角度 = 2π·val/period；列名 nm + "_sin" / nm + "_cos"
    Ft[nm + "_sin"] = np.sin(2 * np.pi * val / period)
    Ft[nm + "_cos"] = np.cos(2 * np.pi * val / period)
    # @@end

# @@todo 节假日 0/1
# @@hint idx.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)
Ft["is_holiday"] = idx.strftime("%Y-%m-%d").isin(HOLIDAYS).astype(int)
# @@end

# @@todo 温度：原始值 + 「离舒适区 22℃ 的距离」（ch01 §1.6 的结论：V 形不是线性）
# @@hint T = temp.values[origins - 1]（起点上一小时，是**已知**的）；dist 用 np.abs(T - 22)
T_prev = temp.values[origins - 1]
Ft["temp"] = T_prev
Ft["temp_dev"] = np.abs(T_prev - 22)
# @@end

FEATS = list(Ft.columns)
Fv = np.ascontiguousarray(Ft.values, dtype="float32")
Yv = np.ascontiguousarray(Y, dtype="float32")
vc = int(cut * 0.85)                    # 训练段内部再切 15% 当验证段（按时间，不 shuffle）
F_fit, F_val, F_te = Fv[:vc].copy(), Fv[vc:cut].copy(), Fv[cut:].copy()
Y_fit, Y_val = Yv[:vc].copy(), Yv[vc:cut].copy()
print("特征 %d 个：%s" % (len(FEATS), FEATS))
print("NaN 个数 %d | 拟合段 %d / 验证段 %d / 测试段 %d"
      % (int(np.isnan(Fv).sum()), len(F_fit), len(F_val), len(F_te)))

t0 = time.time()
P_lgb = np.zeros_like(Y_test, dtype=float)
importance = np.zeros(len(FEATS))
best_iters = []
for h in range(HORIZON):
    # @@todo 第 h 步：建一个 LightGBM，用验证段早停，再对测试集预测
    # @@hint n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.8,
    # @@hint colsample_bytree=0.8, random_state=42, verbose=-1；
    # @@hint fit 时传 eval_X=F_val, eval_y=Y_val[:, h] 与 callbacks=[lgb.early_stopping(20, verbose=False)]
    model_h = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbose=-1, n_jobs=1)
    model_h.fit(F_fit, Y_fit[:, h], eval_X=F_val, eval_y=Y_val[:, h],
                callbacks=[lgb.early_stopping(20, verbose=False)])
    P_lgb[:, h] = model_h.predict(F_te)
    # @@end
    # @@todo 把这一步模型的特征重要性累加进 importance（注意别用 +=，练习版会挖掉赋值目标）
    # @@hint model_h.feature_importances_
    importance = importance + model_h.feature_importances_
    # @@end
    best_iters.append(model_h.best_iteration_ or model_h.n_estimators)

print()
print("LightGBM 24 个模型共耗时 %.1f s | 平均 best_iteration %.1f（上限 150）"
      % (time.time() - t0, float(np.mean(best_iters))))
print("LightGBM 24 步 MAE %.6f（h=1 %.4f → h=24 %.4f）"
      % (mae_all(Y_test, P_lgb), mae_by_step(Y_test, P_lgb)[0], mae_by_step(Y_test, P_lgb)[23]))
print("比最好的朴素基线 %.6f 好了 %.6f（降 %.1f%%）"
      % (mae_all(Y_test, P_sn168), mae_all(Y_test, P_sn168) - mae_all(Y_test, P_lgb),
         100 * (1 - mae_all(Y_test, P_lgb) / mae_all(Y_test, P_sn168))))

# ---- 验收 ----
assert len(FEATS) == 20 and Fv.shape == (8569, 20)
assert int(np.isnan(Fv).sum()) == 0, "所有特征都必须能算出来（lag168 靠 START=168 保证）"
assert 10 < mae_all(Y_test, P_lgb) < 25, "训练结果依赖随机种子，用区间断言；但必须远好于基线"
assert mae_all(Y_test, P_lgb) < mae_all(Y_test, P_sn168) * 0.6, "至少要比基线好 40%"
assert all(1 <= it <= 150 for it in best_iters)
assert float(importance.sum()) > 0
'''

T3_CODE = '''# @@todo 按累计重要性排序，取前 10 个特征
# @@hint np.argsort(-importance)[:10]
order = np.argsort(-importance)[:10]
# @@end

top_df = pd.DataFrame({"特征": [FEATS[i] for i in order],
                       "累计重要性": [int(importance[i]) for i in order]})
print(top_df.to_string(index=False))
print()
print("重要性前 3：%s" % " / ".join(FEATS[i] for i in order[:3]))
print("lag24 的排名 %d | lag168 的排名 %d | temp_dev 的排名 %d"
      % (list(order).index(FEATS.index("lag24")) + 1 if FEATS.index("lag24") in order else -1,
         list(order).index(FEATS.index("lag168")) + 1 if FEATS.index("lag168") in order else -1,
         list(order).index(FEATS.index("temp_dev")) + 1 if FEATS.index("temp_dev") in order else -1))

fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.barh([FEATS[i] for i in order][::-1], [importance[i] for i in order][::-1], color="tab:blue")
ax.set_xlabel("累计重要性（24 个模型相加，`feature_importances_`）")
ax.set_title("LightGBM 特征重要性 top-10")
plt.tight_layout()

# ---- 验收 ----
assert len(order) == 10 and len(set(order.tolist())) == 10
assert set(order.tolist()) <= set(range(len(FEATS)))
assert importance[order[0]] >= importance[order[9]] >= 0, "排序必须真的降序"
'''

T4_CODE = '''H_SEL = (0, 11, 23)               # h=1 / h=12 / h=24 三个代表步长
LGB_MAE, HGB_MAE = {}, {}
for h in H_SEL:
    # @@todo 同一份特征、同一切分，换成 sklearn 的 HistGradientBoostingRegressor
    # @@hint max_iter=150, learning_rate=0.05, max_leaf_nodes=31, random_state=42, early_stopping=False
    # @@hint 只用拟合段训练（F_fit / Y_fit），预测 F_te 后算 MAE
    hgb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_leaf_nodes=31,
                                        random_state=42, early_stopping=False)
    hgb.fit(F_fit, Y_fit[:, h])
    HGB_MAE[h] = float(np.mean(np.abs(hgb.predict(F_te) - Y_test[:, h])))
    # @@end
    # @@todo LightGBM 在同一个 h 上的 MAE —— 它已经训好了，直接从 P_lgb 里取第 h 列
    # @@hint mae_by_step(Y_test, P_lgb)[h]
    LGB_MAE[h] = float(mae_by_step(Y_test, P_lgb)[h])
    # @@end

print("%-8s %12s %12s %10s" % ("步长", "LightGBM", "HistGB", "差值"))
for h in H_SEL:
    print("h=%-5d %12.4f %12.4f %+10.4f" % (h + 1, LGB_MAE[h], HGB_MAE[h], HGB_MAE[h] - LGB_MAE[h]))
print()
print("三点平均：LightGBM %.4f | HistGB %.4f | 差 %+.4f（相对 %.2f%%）"
      % (np.mean(list(LGB_MAE.values())), np.mean(list(HGB_MAE.values())),
         np.mean(list(HGB_MAE.values())) - np.mean(list(LGB_MAE.values())),
         100 * (np.mean(list(HGB_MAE.values())) / np.mean(list(LGB_MAE.values())) - 1)))

# ---- 验收 ----
assert set(H_SEL) == {0, 11, 23} and len(LGB_MAE) == len(HGB_MAE) == 3
assert all(10 < v < 25 for v in HGB_MAE.values()), "同族对照的两个库应在同一个量级"
for h in H_SEL:
    assert abs(HGB_MAE[h] - LGB_MAE[h]) < 4, "同族模型的差距不应该超过 4 kW"
'''

T5_CODE = '''DS = (1, 2, 4, 8)               # 每个残差块的膨胀率
K = 3                          # 卷积核宽度
CH = 16                        # 通道数


class CausalConv1d(nn.Module):
    """因果卷积：只在**左边** padding，于是输出第 i 个位置只依赖输入 ≤ i 的位置。"""

    def __init__(self, cin, cout, k, d=1):
        super().__init__()
        # @@todo 左侧要补多少格，才能既「因果」又保持输出长度不变
        # @@hint 膨胀卷积的等效核宽是 (k-1)*d + 1，所以在左边补 (k-1)*d 格
        self.pad = (k - 1) * d
        # @@end
        self.conv = nn.Conv1d(cin, cout, k, dilation=d)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))      # (左, 右)


class TCNBlock(nn.Module):
    """残差块：两个同膨胀率的因果卷积 + 一条恒等跳连。"""

    def __init__(self, ch, k, d):
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, k, d)
        # @@todo 第二个因果卷积（通道数与膨胀率都和第一个一样）
        # @@hint CausalConv1d(ch, ch, k, d)
        self.conv2 = CausalConv1d(ch, ch, k, d)
        # @@end

    def forward(self, x):
        # @@todo 残差块的前向：两层因果卷积 + ReLU，最后把输入原样加回去
        # @@hint torch.relu(x + torch.relu(self.conv2(torch.relu(self.conv1(x)))))
        return torch.relu(x + torch.relu(self.conv2(torch.relu(self.conv1(x)))))
        # @@end


class TCN(nn.Module):
    def __init__(self, ch=CH, k=K, ds=DS, out=HORIZON):
        super().__init__()
        self.inp = CausalConv1d(1, ch, k, 1)
        self.blocks = nn.ModuleList([TCNBlock(ch, k, d) for d in ds])
        # @@todo 输出头：把「最后一个时刻」的 ch 维特征映射成 24 步
        # @@hint nn.Linear(ch, out)
        self.head = nn.Linear(ch, out)
        # @@end

    def forward(self, x):
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        # @@todo 只取最后一个时刻的特征（它已经聚合了左边一大段历史）
        # @@hint h 的形状是 (B, ch, L)，取 h[:, :, -1]
        return self.head(h[:, :, -1])
        # @@end


# @@todo 感受野公式：RF = 1 + (k-1) · Σ(每个卷积的膨胀率)
# @@hint 每个残差块里有**两个**卷积，所以 ds 里每个 d 都要数两遍
RF = 1 + (K - 1) * sum(2 * d for d in DS)
# @@end

with torch.random.fork_rng():
    torch.manual_seed(0)
    # 用**纯线性**的因果卷积堆量感受野：加了 ReLU 扰动会被吃掉，量不准
    stack = nn.ModuleList()
    cin = 1
    for d in DS:
        for _ in range(2):
            stack.append(CausalConv1d(cin, 4, K, d))
            cin = 4
    clean_in = torch.zeros(1, 1, WINDOW)
    # @@todo 把输入第 0 个位置扰动 +1
    # @@hint bumped = clean_in.clone()，然后把 bumped[0, 0, 0] 设成 1.0
    bumped = clean_in.clone()
    bumped[0, 0, 0] = 1.0
    # @@end
    with torch.no_grad():
        out_clean, out_bumped = clean_in, bumped
        for layer in stack:
            out_clean, out_bumped = layer(out_clean), layer(out_bumped)
    # @@todo 数出「输出里被扰动的时刻个数」—— 这就是**实测**感受野
    # @@hint diff = (out_clean - out_bumped).abs().max(dim=1)[0][0].numpy()
    # @@hint rf_measured = int((diff > 1e-8).sum())
    diff = (out_clean - out_bumped).abs().max(dim=1)[0][0].numpy()
    rf_measured = int((diff > 1e-8).sum())
    # @@end

print("感受野公式 RF = 1 + (k-1)·Σ2d = 1 + %d×%d = %d" % (K - 1, sum(2 * d for d in DS), RF))
print("实测感受野 = %d（扰动输入第 0 格，输出里有 %d 格跟着变）" % (rf_measured, rf_measured))
print("窗口长度 %d > RF %d：窗口最前面 %d 格对最后一个输出**没有影响**" % (WINDOW, RF, WINDOW - RF))
print("参数与层数：%d 个因果卷积层（含输入层），通道 %d" % (1 + 2 * len(DS), CH))

# ---- 验收 ----
assert RF == 61, "1 + (3-1)·2·(1+2+4+8) = 61"
assert rf_measured == RF, "实测感受野必须与公式一致"
assert rf_measured < WINDOW, "RF < 窗口：最前面 11 格是白给的"
assert (1 + 2 * len(DS)) == 9
'''

T6_CODE = '''STRIDE, EPOCHS, BS, LR = 4, 12, 128, 1e-2
MX, SX = float(X[:cut].mean()), float(X[:cut].std())
MY, SY = float(Y[:cut].mean()), float(Y[:cut].std())
Xt = torch.tensor((X[:cut] - MX) / SX, dtype=torch.float32)[:, None, :]
Xe = torch.tensor((X_test - MX) / SX, dtype=torch.float32)[:, None, :]
Yt = torch.tensor((Y[:cut] - MY) / SY, dtype=torch.float32)
Xs, Ys = Xt[::STRIDE], Yt[::STRIDE]
print("深度模型训练样本 %d（stride=%d，只用 1/%d 的训练样本以守住算力预算）"
      % (len(Xs), STRIDE, STRIDE))
print("标准化只来自训练段：MX %.4f / SX %.4f | MY %.4f / SY %.4f" % (MX, SX, MY, SY))

t0 = time.time()
with torch.random.fork_rng():
    torch.manual_seed(1)
    tcn = TCN()
    opt = torch.optim.Adam(tcn.parameters(), lr=LR)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(Xs))
        for i in range(0, len(Xs), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            # @@todo 一个 mini-batch 前向 + 反向：MSE 损失
            # @@hint ((tcn(Xs[idx]) - Ys[idx]) ** 2).mean().backward()
            ((tcn(Xs[idx]) - Ys[idx]) ** 2).mean().backward()
            # @@end
            opt.step()
    tcn.eval()
    with torch.no_grad():
        # @@todo 推理：一次前向拿 24 步，再反标准化回 kW
        # @@hint tcn(Xe).numpy() * SY + MY
        P_tcn = tcn(Xe).numpy() * SY + MY
        # @@end

tcn_secs = time.time() - t0

print("TCN 训练 %.1f s（%d 个样本 × %d 轮 = %.1f 万次「样本-轮」）"
      % (tcn_secs, len(Xs), EPOCHS, len(Xs) * EPOCHS / 1e4))
print("算力效率：%.0f µs / 样本-轮（卷积在 CPU 上就是这个价）"
      % (1e6 * tcn_secs / (len(Xs) * EPOCHS)))
print("TCN 24 步 MAE %.6f（h=1 %.4f → h=24 %.4f）"
      % (mae_all(Y_test, P_tcn), mae_by_step(Y_test, P_tcn)[0], mae_by_step(Y_test, P_tcn)[23]))
print("对照：LightGBM %.6f | 最好朴素基线 %.6f"
      % (mae_all(Y_test, P_lgb), mae_all(Y_test, P_sn168)))
print("TCN 参数个数 %d" % sum(p.numel() for p in tcn.parameters()))

# ---- 验收 ----
assert P_tcn.shape == Y_test.shape == (1714, HORIZON)
assert 10 < mae_all(Y_test, P_tcn) < 60, "随机训练结果 → 区间断言"
assert mae_all(Y_test, P_tcn) > mae_all(Y_test, P_lgb), "本章结论：这个数据规模上深度模型打不过 LightGBM"
assert sum(p.numel() for p in tcn.parameters()) < 10000, "TCN 很小（省算力）"
'''

T7_CODE = '''L = WINDOW
C_SPARSE = 5.0


class ProbSparseAttention(nn.Module):
    """Informer 的 ProbSparse 简化版：只对 top-u 个 query 做注意力。

    其余 query 的注意力输出直接取 V 的均值（等价于「均匀看所有时刻」）。
    """

    def __init__(self, d_model, c=C_SPARSE):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.d, self.c = d_model, c

    def forward(self, x):
        B, length, D = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        # @@todo 稀疏度上限 u = c · ln L（向上取整，且至少 1）
        # @@hint max(1, int(math.ceil(self.c * math.log(length))))
        U = max(1, int(math.ceil(self.c * math.log(length))))
        # @@end
        # @@todo 用 U 个**随机采样**的 key 估计每个 query 的稀疏度：打分后取 max − mean
        # @@hint ks = torch.randperm(length)[:U]；s = Q @ K[:, ks].transpose(1, 2) / math.sqrt(D)
        # @@hint score = s.max(-1).values - s.mean(-1)
        ks = torch.randperm(length)[:U]
        s = Q @ K[:, ks].transpose(1, 2) / math.sqrt(D)
        score = s.max(-1).values - s.mean(-1)
        # @@end
        # @@todo 取出稀疏度最高的 U 个 query（top-u），其余 query 不参与注意力
        # @@hint torch.topk(score, U, dim=1).indices
        u_idx = torch.topk(score, U, dim=1).indices
        # @@end
        full = Q @ K.transpose(1, 2) / math.sqrt(D)
        row = torch.zeros_like(full)
        row.scatter_(1, u_idx.unsqueeze(-1).expand(-1, -1, length), 1.0)
        att = torch.softmax(full, -1) * row
        att = att / att.sum(-1, keepdim=True).clamp_min(1e-9)
        out = att @ V
        mean_v = V.mean(1, keepdim=True).expand(-1, length, -1)
        return torch.where(row.sum(-1, keepdim=True) > 0, out, mean_v)


U72 = max(1, int(math.ceil(C_SPARSE * math.log(L))))
# @@todo 注意力「打分次数」：全注意力 = L²；ProbSparse = 采样打分 L·U + 选中后 U·L
# @@hint L * L 与 2 * L * U72
score_full = L * L
score_sparse = 2 * L * U72
# @@end
print("L = %d | u = c·ln L = %.4f → 向上取整 %d（占 L 的 %.1f%%）"
      % (L, C_SPARSE * math.log(L), U72, 100 * U72 / L))
print("打分次数：全注意力 %d | ProbSparse %d | 省下 %.1f%%"
      % (score_full, score_sparse, 100 * (1 - score_sparse / score_full)))
print()
print("省下的比例随 L 变化（c = %.1f，u = ⌈c·ln L⌉）：" % C_SPARSE)
print("%8s %6s %12s %12s %9s" % ("L", "u", "全注意力", "ProbSparse", "省下"))
RATIOS = []
for l in (72, 168, 336, 720, 1440):
    u = max(1, int(math.ceil(C_SPARSE * math.log(l))))
    RATIOS.append((l, u, 1 - 2 * l * u / (l * l)))
    print("%8d %6d %12d %12d %8.1f%%" % (l, u, l * l, 2 * l * u, 100 * RATIOS[-1][2]))

# @@todo 蒸馏：用 stride=2 的卷积把序列长度砍半，两层之后就只剩 1/4
# @@hint LENS = [WINDOW, WINDOW // 2, WINDOW // 4]
LENS = [WINDOW, WINDOW // 2, WINDOW // 4]
# @@end
print()
print("蒸馏（stride=2 的卷积）后序列长度：%s（总压缩 %d 倍）" % (LENS, LENS[0] // LENS[-1]))

fig, ax = plt.subplots(figsize=(7.2, 4.0))
ax.plot([r[0] for r in RATIOS], [100 * r[2] for r in RATIOS], marker="o", color="tab:red", label="ProbSparse 省下的比例")
ax.plot([r[0] for r in RATIOS], [100 * r[1] / r[0] for r in RATIOS], marker="s", color="tab:blue",
        label="u/L（被选中的 query 占比）")
ax.set_xscale("log")
ax.set_xlabel("序列长度 L（对数刻度）")
ax.set_ylabel("百分比 %")
ax.set_title("ProbSparse 的收益随序列变长而变大")
ax.legend(fontsize=9)
plt.tight_layout()

# ---- 验收 ----
assert U72 == 22, "⌈5·ln72⌉ = ⌈21.38⌉ = 22"
assert score_full == 5184 and score_sparse == 3168
assert abs(score_sparse / score_full - 0.6111) < 1e-3, "L=72 时只省 38.9%"
assert RATIOS[-1][2] > 0.9, "L=1440 时省下 90% 以上"
assert LENS == [72, 36, 18]
assert max(1, int(math.ceil(C_SPARSE * math.log(720)))) == 33
'''

T8_CODE = '''D_MODEL, TF_EPOCHS, TF_BS, TF_LR = 32, 20, 256, 5e-3


class FullAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        self.d = d_model

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        return torch.softmax(Q @ K.transpose(1, 2) / math.sqrt(self.d), -1) @ V


class TSFormer(nn.Module):
    """最小版时序 Transformer：正弦位置编码 + 1 层注意力 + 前馈，最后取最后一个时刻。"""

    def __init__(self, d_model=D_MODEL, attn="sparse", out=HORIZON):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        pe = torch.zeros(WINDOW, d_model)
        pos = torch.arange(WINDOW).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.attn = FullAttention(d_model) if attn == "full" else ProbSparseAttention(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.ReLU(), nn.Linear(2 * d_model, d_model))
        self.head = nn.Linear(d_model, out)
        self.mode = attn

    def forward(self, x):
        h = self.embed(x.transpose(1, 2)) + self.pe.unsqueeze(0)
        h = self.norm(h + self.attn(h))
        h = self.norm(h + self.ff(h))
        return self.head(h[:, -1])


TF_PRED, TF_TIME = {}, {}
print("两个变体都用**全部** %d 个训练样本（Transformer 比 TCN 便宜得多，见 §8.4.4）" % len(Xt))
for mode in ("full", "sparse"):
    t0 = time.time()
    with torch.random.fork_rng():
        torch.manual_seed(3)                 # 同一个种子 → 两个变体**初始权重完全相同**
        model = TSFormer(attn=mode)
        opt = torch.optim.Adam(model.parameters(), lr=TF_LR)
        for ep in range(TF_EPOCHS):
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), TF_BS):
                idx = perm[i:i + TF_BS]
                opt.zero_grad()
                # @@todo 一个 mini-batch 的 MSE 反向传播
                # @@hint ((model(Xt[idx]) - Yt[idx]) ** 2).mean().backward()
                ((model(Xt[idx]) - Yt[idx]) ** 2).mean().backward()
                # @@end
                opt.step()
        model.eval()
        with torch.no_grad():
            # @@todo 推理并反标准化
            # @@hint model(Xe).numpy() * SY + MY
            TF_PRED[mode] = model(Xe).numpy() * SY + MY
            # @@end
    TF_TIME[mode] = time.time() - t0
    print("Transformer[%-6s] %5.1f s | 24 步 MAE %.6f（h=1 %.4f → h=24 %.4f）"
          % (mode, TF_TIME[mode], mae_all(Y_test, TF_PRED[mode]),
             mae_by_step(Y_test, TF_PRED[mode])[0], mae_by_step(Y_test, TF_PRED[mode])[23]))
print()
print("算力账：TCN 花了 %.1f s 只买到 %.1f 万次「样本-轮」；"
      % (tcn_secs, len(Xs) * EPOCHS / 1e4))
print("        Transformer 花 %.1f s 买到 %.1f 万次 —— 每个样本-轮便宜 %.1f 倍"
      % (TF_TIME["sparse"], len(Xt) * TF_EPOCHS / 1e4,
         (tcn_secs / (len(Xs) * EPOCHS)) / (TF_TIME["sparse"] / (len(Xt) * TF_EPOCHS))))
print("ProbSparse 与全注意力的 MAE 差 %+.4f | 训练耗时差 %+.1f s"
      % (mae_all(Y_test, TF_PRED["sparse"]) - mae_all(Y_test, TF_PRED["full"]),
         TF_TIME["sparse"] - TF_TIME["full"]))
print("注意：L=72 时 ProbSparse 只省 38.9%% 的打分次数，却多付了随机采样与 top-k 的开销，")
print("      所以在这个长度上「省时间」体现不出来 —— 收益要等 L 上百甚至上千（§8.5.1）")

# ---- 验收 ----
for mode in ("full", "sparse"):
    assert TF_PRED[mode].shape == (1714, HORIZON)
    assert 10 < mae_all(Y_test, TF_PRED[mode]) < 45, "随机训练结果 → 区间断言"
assert abs(mae_all(Y_test, TF_PRED["sparse"]) - mae_all(Y_test, TF_PRED["full"])) < 8, "两个变体应在同一量级"
assert mae_all(Y_test, TF_PRED["full"]) < mae_all(Y_test, P_tcn), "Transformer 比 TCN 准（TCN 受算力限制只训了 1/4 样本）"
assert U72 == 22
'''

T9_CODE = '''# @@todo 把本章所有模型的「24 步 MAE」汇总成一张表，并按 MAE 升序排
# @@hint 先写 {名字: (预测矩阵)} 再算 MAE；排序用 sorted(rows.items(), key=lambda kv: kv[1])
preds = {
    "last value": P_last,
    "seasonal naive t-24": P_sn24,
    "seasonal naive t-168": P_sn168,
    "moving average 24": P_ma24,
    "LightGBM（20 特征）": P_lgb,
    "TCN（因果卷积 + 膨胀）": P_tcn,
    "Transformer（全注意力）": TF_PRED["full"],
    "Transformer（ProbSparse）": TF_PRED["sparse"],
}
ranked = sorted(((nm, mae_all(Y_test, P)) for nm, P in preds.items()), key=lambda kv: kv[1])
# @@end

base = dict(ranked)["seasonal naive t-168"]
print("%-26s %10s %11s %10s" % ("模型 / 基线", "24步 MAE", "相对基线", "h=1 MAE"))
for nm, v in ranked:
    print("%-26s %10.4f %10.1f%% %10.4f"
          % (nm, v, 100 * v / base, mae_by_step(Y_test, preds[nm])[0]))
print()
print("及格线（最好的朴素基线）= %.6f" % base)
print("第一名 %s = %.6f，相对基线提升 %.1f%%"
      % (ranked[0][0], ranked[0][1], 100 * (1 - ranked[0][1] / base)))

names = [nm for nm, _ in ranked]
vals = [v for _, v in ranked]
colors = ["tab:red" if nm.startswith("LightGBM") else
          ("tab:gray" if any(k in nm for k in ("last", "seasonal", "moving")) else "tab:blue")
          for nm in names]
fig, ax = plt.subplots(figsize=(9.4, 4.4))
ax.barh(names[::-1], vals[::-1], color=colors[::-1])
ax.axvline(base, color="black", ls="--", lw=1.2, label="及格线（朴素基线 %.1f）" % base)
ax.set_xlabel("24 步 MAE (kW)")
ax.set_title("全基线对比：特征工程 + LightGBM 第一，深度模型没打过它")
ax.legend(fontsize=9)
plt.tight_layout()

# ---- 验收 ----
assert len(ranked) == 8 and len(preds) == 8
assert [v for _, v in ranked] == sorted(vals), "必须真的按 MAE 升序"
assert ranked[0][0] == "LightGBM（20 特征）", "本章结论：LightGBM 排第一"
assert ranked[-1][0] == "last value" and ranked[-1][1] == max(vals)
assert dict(ranked)["seasonal naive t-168"] < dict(ranked)["seasonal naive t-24"], "周朴素强于日朴素"
assert ranked[0][1] / base < 0.6
assert dict(ranked)["TCN（因果卷积 + 膨胀）"] > ranked[0][1], "深度模型输给了 LightGBM"
'''

LESSON = [
    md(
        """
# ch08 进阶与基线（讲解版）

> **本节考点**：模型选择（**10%**）+ 模型调参（**10%**）+ 新能源功率 / 负荷预测（行业赛题）。
> 本章的立场很直白：**先有基线，才有资格说模型好。**
> 很多参赛者的「深度模型」跑不过一行 `ser[t-168]`，只是没人告诉他们。

| 竞赛评分点 | 分值含义 | 本章覆盖 |
|---|---|---|
| 模型选择 | **10%** | 朴素基线家族 → 树模型 → 深度模型，用同一张表排序 |
| 模型调参 | **10%** | LightGBM 关键超参 + `early_stopping_rounds` + 特征重要性 |
| 新能源功率 / 负荷预测 | 行业赛题 | 台区负荷的 24 步预测全流程 |
| 模型训练 | 技能操作 40% | TCN 的因果卷积 / 感受野、Transformer 的 ProbSparse |

**与 ch07 的分工**：ch07 解决「**怎么量**一个好模型」（指标 / 早停 / 残差诊断），
本章解决「**谁是好模型**」（基线家族 / 特征工程 / 深度模型对比）。
两章的 `window = 72`、`horizon = 24`、起点 `START = 168`、8:2 时间切分**完全一致**，
所以两章的 MAE **可以直接比**。

## 学习目标

1. 能写出 `last value` / `seasonal naive(t-24, t-168)` / `moving average` 四种朴素基线，
   并说清**为什么 `t-168` 常常比 `t-24` 还好**
2. 会用「滞后 + 滑窗 + 日历 sin·cos + 节假日 + 温度」搭出一份 LightGBM 能吃 20 维特征矩阵
3. 会做**同族对照**：同一份特征、同一切分，比 LightGBM 与 sklearn `HistGradientBoosting` 的 MAE
4. 能手写 `CausalConv1d`，推导并**实测** TCN 的感受野
5. 能实现 Informer 的 ProbSparse 简化版（`u = c·ln L`），并算清它**省在哪、什么长度才划算**
6. 能对着全基线对比表回答：**什么时候该上深度模型**

## API 速查表

| 方法 / 属性 | 关键参数 | 返回 | 一句话 |
|---|---|---|---|
| `LGBMRegressor` | `n_estimators` / `num_leaves` / `learning_rate` | 模型 | 叶子优先生长，小数据上比 XGBoost 稳 |
| `model.fit(..., eval_X=, eval_y=)` | 验证集 | 模型 | 配合 `lgb.early_stopping(n)` 自动定轮数 |
| `model.feature_importances_` | — | `array` | **分裂次数**（不是增益），可跨模型相加 |
| `HistGradientBoostingRegressor` | `max_iter` / `max_bins` | 模型 | sklearn 的直方图提升，与 LightGBM 同族 |
| `F.pad(x, (left, right))` | 元组 | `Tensor` | 因果卷积的关键：**只补左边** |
| `nn.Conv1d(..., dilation=d)` | `dilation` | `Layer` | 膨胀卷积：不增参数，指数级扩大感受野 |
| `torch.einsum` / `Q @ K.transpose(1,2)` | — | `Tensor` | 注意力打分，形状 `(B, L, L)` |
| `torch.topk(score, u, dim=1)` | `u` | `(values, indices)` | ProbSparse 选 top-u 个 query |
| `torch.randperm(n)[:u]` | — | `Tensor` | 随机采样 key，用来估稀疏度 |

## 本章统一口径（与 ch07 一致，可直接对比）

```
数据   data/load_curve.csv → A 台区（居民型）→ clean_series → 8760 个整点
窗口   window = 72   步长 horizon = 24   起点 START = 168
切分   前 80% 训练（6855） / 后 20% 测试（1714），绝不 shuffle
打分   24 步 MAE = 所有 (起点, 步长) 上绝对误差的均值
```

### 开工前的一个环境坑（本章必读）

本章要**同时**用 lightgbm 与 torch，而这两个库各自静态链接了一份 OpenMP 运行时
（lightgbm 走 `libomp`，torch 走 `libiomp`）。在 macOS/arm64 上让两份运行时各自开满线程，
会撞出 `OMP: Error #179: Function pthread_mutex_init failed` —— 这不是 Python 异常，
而是**内核级段错误**（`try/except` 抓不到，进程直接消失），排查起来极其痛苦。

修法只有一行，但**必须在 `import` 之前**：

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"      # 先限线程，再 import lightgbm / torch
```

下一格（导入单元格）已经这么做了，并写了注释。**这是本章踩得最深的一个坑，值得单独记一笔。**
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md(
        """
## 8.1 朴素基线家族：先给「及格线」定价

**朴素基线**（naive baseline）是「不需要学任何参数」的预测器。它们不是玩具 ——
它们定义了**及格线**：任何模型如果打不过它们，就说明这个模型**连数据的周期结构都没学到**。

本章的四个基线：

| 基线 | 公式 | 直觉 |
|---|---|---|
| `last value` | $\\hat y_{t+h}=y_{t-1}$ | 用「上一小时」外推 24 小时 |
| `seasonal naive t-24` | $\\hat y_{t+h}=y_{t+h-24}$ | 用「昨天同一小时」 |
| `seasonal naive t-168` | $\\hat y_{t+h}=y_{t+h-168}$ | 用「上周同一小时」 |
| `moving average 24` | $\\hat y_{t+h}=\\overline{y_{t-24:t}}$ | 用「过去 24 小时均值」 |

**先看四个基线的实测（24 步 MAE）**：

| 基线 | 24 步 MAE | h=1 MAE | h=24 MAE |
|---|---|---|---|
| last value | 60.524392 | 30.0939 | 21.6066 |
| seasonal naive t-24 | 32.998794 | 33.0215 | 33.1928 |
| **seasonal naive t-168** | **31.560227** | 31.5845 | 31.6804 |
| moving average 24 | 45.527850 | 44.9067 | 46.2157 |

**三件必须读出来的事**：

1. **`last value` 的整体 MAE 最差（60.52），但它的 h=1 只有 30.09、h=24 只要 21.61** ——
   因为它是唯一一个「随步长振荡」的基线：h=24 时正好对上了同一个小时（周期对齐），
   误差突然变小。**这提醒你：看 MAE 一定要分开看逐步曲线，不要只看一个总数。**
2. **`moving average 24` 连 `t-24` 都打不过**（45.53 vs 33.00）：24 小时均值把「日周期」**抹平了** ——
   它预测的是一条平线，而真实曲线每天都在起伏。**平滑 ≠ 准确。**
3. **`t-168` 比 `t-24` 好**（31.56 vs 33.00）：见 §8.1.1。
"""
    ),
    code(T1_CODE),
    md(
        """
### 8.1.1 难点深挖：为什么「上周同一小时」比「昨天同一小时」还准？

**为什么难**：直觉上「昨天」比「上周」近得多，信息应该更新鲜。但实测 `t-168` 赢了
`t-24`（31.560227 vs 32.998794，差 **1.44 kW，4.4%**）。

原因是本数据的**乘法结构**：

```
负荷 = 基线 × 日周期(hour) × 年季节(doy) × 周周期(dow, 工作日/周末) × 节假日 × 温度 × 趋势 × (1+噪声)
```

`t-24` 只对齐了**小时**这一个因子，但 t-24 小时前是**昨天**；而 t-168 小时前是
**上周的同一天同一小时** —— 它同时对齐了 `hour` 和 `dow` 两个因子。

| 基线 | 对齐了哪些因子 | 还差什么 |
|---|---|---|
| `t-24` | hour | **dow 错位**（周日 ↔ 周六不同） |
| `t-168` | hour + dow | 年季节（7 天漂移很小）、节假日、温度 |
| `last value` | 无 | hour 全错位 |

A 台区是**居民型**：周末比工作日更忙（周末系数 1.06）。所以 `t-24` 在周末附近
会拿「工作日」去预测「周末」，误差自然变大。

**判定规则（非常实用的一条）**：

> **先把「周期长度」列出来，然后试 `t−P`（P = 周期）的朴素基线。**
> 如果你的数据有**周周期**，那 `t-168` 是一个必须跑的基线；
> 有**月周期**就试 `t-720`。**最强的那个朴素基线，就是你的真实及格线。**
>
> 反过来说：如果 `t-168` 打不过 `t-24`，说明数据里**没有显著的周周期**，
> 后面做特征时 `dow` 相关的特征就可以省了 —— 这是一个几乎零成本的特征筛选实验。
"""
    ),
    md(
        """
## 8.2 特征工程 + LightGBM：本章的「真基线」

树模型在**表格化时序特征**上是性价比之王：不需要标准化、能吃缺失、训练秒级、
特征重要性可解释。本章的 20 维特征分四组：

| 组 | 特征 | 为什么 |
|---|---|---|
| **滞后** | `lag1` `lag2` `lag3` `lag24` `lag25` `lag168` | 短期惯性与两个周期（24 / 168） |
| **滑窗统计** | `ma24` `std24` `min24` `max24` `ma168` | 最近的水平与波动 |
| **日历** | `hour_sin/cos` `dow_sin/cos` `month_sin/cos` `is_holiday` | 周期结构，**必须 sin·cos 编码** |
| **外生** | `temp` `temp_dev` | 温度的**线性**分量与 **V 形**分量 |

**多步预测怎么建？** 树模型一次只吐一个数，想做 24 步有两条路：

| 方式 | 做法 | 代价 |
|---|---|---|
| **每步一个模型**（本章用） | 训 24 个 LightGBM，第 h 个学 `Y[:, h]` | 24 倍训练时间，但每个步长都能独立早停、独立调参 |
| 单模型 + horizon 特征 | 把 `h` 也当一个特征塞进去 | 便宜，但树要在「h」这个维度上做分裂，精度通常更差 |
| 递归 | 只训 h=1，然后自己喂自己 | 便宜，但误差累积（ch07 §7.3.1） |

**训练配置**（每个步长一份，全部带早停）：

```python
LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31,
              subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
model.fit(F_fit, Y_fit[:, h], eval_X=F_val, eval_y=Y_val[:, h],
          callbacks=[lgb.early_stopping(20, verbose=False)])
```

注意验证段是**训练段内部的最后 15%**（`vc = int(cut * 0.85)`）—— 与测试段严格隔开。
"""
    ),
    code(T2_CODE),
    md(
        """
### 8.2.1 难点深挖：日历特征为什么必须 sin·cos —— 树模型的版本

ch01 §1.2.1 已经讲过「周期整数直接当特征，23 点和 0 点离得最远」，那是从**距离**的角度。
对**树模型**还有一个更具体的理由：

树模型的一次分裂是 `hour <= 阈值`。要把「23 点与 0 点是相邻的」这件事表达出来，
树必须学会**两条互不相连的规则**（`hour >= 22` 或 `hour <= 1`）。
而单棵树的判定路径是**从上到下的轴对齐切分**，要做到「首尾相接」需要
**两次分裂 + 一次 OR 组合** —— 树得为它多花好几层深度。

换成 `hour_sin` / `hour_cos` 之后，23 点和 0 点在**特征空间里的欧氏距离**只有 0.261052
（ch01 实测），一次 `hour_cos >= 0.966` 这样的分裂就能把「深夜」整段切出来。

| 编码 | 23 点 vs 0 点的距离 | 树要几次分裂才能表示「相邻」 |
|---|---|---|
| 原始 `hour` | 23（数值上最远） | 2 次 + 逻辑组合 |
| `hour_sin/cos` | 0.261052（几乎重合） | 1 次 |

**同样的道理适用于 `dow`（周期 7，周末与周一相邻吗？不是，但周五与周六相邻）×
`month`（周期 12，12 月与 1 月相邻）**。

> **判定规则**：任何**周期性**的整数特征，交给任何模型之前都先 sin·cos 一次。
> 代价是特征数翻倍，收益是「周期性」这件事**不用模型自己学**。
>
> 反例提醒：`dayofyear`（1~365）**不要**盲目做 sin·cos 再丢掉原值 ——
> 年周期的「两端相接」是对的，但 `doy` 的**单调性**（3 月在第 1 季度、9 月在第 3 季度）
> 也是有用信息。两个都留，让模型选。
"""
    ),
    code(T3_CODE),
    md(
        """
## 8.3 同族对照：LightGBM vs sklearn `HistGradientBoosting`

「同族」意思是两者都是**直方图 + 梯度提升决策树**：

| 维度 | LightGBM | sklearn HistGB |
|---|---|---|
| 分裂方式 | **叶子优先**（leaf-wise，按最大增益挑叶子） | **层优先**（level-wise，逐层长满） |
| 直方图分箱 | `max_bin=255` 自适应 | `max_bins=255`，**特征级分箱** |
| 正则 | `num_leaves` + `min_child_samples` + L1/L2 | `max_leaf_nodes` + `l2_regularization` |
| 早停 | `callbacks=[lgb.early_stopping(n)]` | `early_stopping=True`（内部切验证集） |
| 分类特征 | 原生支持 `categorical_feature` | 原生支持 `categorical_features` |
| 速度 | 快（尤其多线程 / 大数据） | 慢一些，但**零依赖、API 更稳定** |

**对照实验的设计要点**（这是「模型选择」这个考点的核心动作）：

> **同一个任务、同一份特征、同一个切分，只换库。**
> 任何一处不同，比出来的差异就分不清是「库的差异」还是「设置的差异」。

为了把对照做得更干净，这里只比 `h = 1 / 12 / 24` 三个代表步长：取**首 / 中 / 末**三步，
就能看出「LightGBM 与 HistGB 的差随步长怎么变」（实测每步一次 `HistGB` 拟合只要约 **0.3 s**，
24 步全跑也只多 7 s；之所以不全跑，是因为本章的算力预算要留给 §8.4/§8.5 的两个深度模型）。
LightGBM 那边直接复用 §8.2 已经训好的 24 个模型。
"""
    ),
    code(T4_CODE),
    md(
        """
### 8.3.1 难点深挖：同族 ≠ 同一个模型 —— 分裂策略的取舍

两个库的差距看起来很小，但**失败模式完全不同**：

| 现象 | 主要原因 |
|---|---|
| LightGBM 在小数据上更容易**过拟合** | 叶子优先生长：同样是 31 个叶子，它会**先长宽的那一枝**，深度更深、单叶子样本更少 |
| HistGB 在特征很多时**更慢** | 层优先会**把整层长满**，哪怕某些分裂增益很低 |
| HistGB 不需要 `min_child_samples` 也很稳 | 层优先天然限制深度，加上内部 `l2_regularization=0` 的默认正则 |
| LightGBM 的 `feature_importances_` 更大 | 它的默认是**分裂次数**；HistGB 没有直接等价接口（要用 `permutation_importance`） |

**判定规则**：

> **「哪个库更好」这个问题本身没有意义，要问「在什么数据规模上、用什么参数」。**
>
> - 样本 < 1 万、特征 < 50：两个库的差距通常在**噪声量级**，选 API 顺手的那一个
> - 样本 > 100 万：LightGBM 的速度优势会变成决定性的（差几倍训练时间）
> - 需要**可复现的默认行为**、不想装额外依赖：sklearn `HistGB` 更省心
>
> 真正的行动建议是：**两个都跑一遍，取 MAE 低的那个，并把对照实验写进报告** ——
> 这本身就是「模型选择 10%」的得分点。
"""
    ),
    md(
        """
## 8.4 TCN：因果卷积 + 膨胀 + 残差

### 8.4.1 因果卷积（causal convolution）

时序预测**绝对不能看未来**。普通 `nn.Conv1d` 的核是居中的，会把 `t+1` 的信息混进 `t` 的输出
—— 这就是一种泄漏。**因果卷积**的修法极其简单：

```python
# 只在左边补 (k-1)*d 格，右边一格不补
F.pad(x, (self.pad, 0))
```

这样输出第 `i` 个位置只依赖输入 `≤ i` 的位置，且**长度不变**。

### 8.4.2 膨胀卷积（dilated convolution）与感受野

一层普通卷积（k=3）只能看 3 个小时。要看 24 小时（日周期）要么堆 12 层，要么用**膨胀**：

```
d=1:  ○ ○ ●          只看相邻 3 格
d=2:  ○ . ○ . ●      中间隔一格
d=4:  ○ . . . ○ . . . ●
```

**感受野（receptive field）公式**：把每一层卷积的膨胀率加起来

$$RF = 1 + (k-1)\\sum_i d_i$$

本章的 TCN 有 4 个残差块，每块 2 个卷积，所以：

$$RF = 1 + (3-1)\\times 2\\times(1+2+4+8) = 1 + 2 \\times 30 = 61$$

**实测验证**（这一步很关键，别信公式信实测）：把一个**纯线性**的因果卷积堆
（不加 ReLU，否则扰动会被激活函数吃掉）的第 0 个输入位置扰动 `+1`，
数输出里有几个位置跟着变 —— 结果就是感受野。

### 8.4.3 残差块与输出头

```
x ──► CausalConv1d(d) ──► ReLU ──► CausalConv1d(d) ──► ReLU ──► (+) ──► ReLU ──►
│                                                                ▲
└────────────────────────────── 恒等跳连 ─────────────────────────┘
```

最后只取**最后一个时刻**的特征（它已经聚合了左边 61 小时），过一个线性层输出 24 步。
"""
    ),
    code(T5_CODE),
    md(
        """
### 8.4.4 难点深挖：感受野 61 < 窗口 72，最前面 11 格是白给的

**为什么难**：大家搭网络时常直接写「输入窗口 72」，然后默认**整个窗口都被用上了**。
但因果卷积的输出 `y[T-1]` 只依赖 `x[T-61 : T]` —— **窗口的第 0~10 格对预测没有任何影响**。

| 量 | 值 |
|---|---|
| 窗口长度 | 72 |
| 感受野（公式 = 实测） | **61** |
| 被浪费的输入 | 72 − 61 = **11 格** |

**这不是 bug，是设计取舍**：

| 选择 | 后果 |
|---|---|
| 把窗口从 72 缩到 61 | 训练快一点，信息不丢 —— 但你得知道为什么是 61 |
| 加一层 `d=16` 的块 | RF 变成 1 + 2·2·(1+2+4+8+16) = 125 > 72，整个窗口都能看到，但**参数与算力都涨** |
| 缩到 `d=(1,2,4)` | RF = 29，连一个日周期（24）都刚够，周周期完全看不到 |

**判定规则**：

> **搭 TCN 的第一步是算感受野，不是调学习率。**
> 判断标准是：`RF` 应该 **≥ 你希望模型看到的最长周期**（本例：至少 24，最好 ≥ 168）。
> 本章 `RF = 61` 能覆盖日周期（24）但覆盖不到周周期（168）——
> 这解释了为什么 TCN 在 §8.6 里表现平平：**它的结构决定了它看不到周周期，
> 而 LightGBM 通过 `lag168` / `ma168` 两个特征直接把周周期喂了进去。**
"""
    ),
    code(T6_CODE),
    md(
        """
## 8.5 Transformer 时序：Informer 的 ProbSparse 简化版

标准自注意力的复杂度是 $O(L^2)$ —— $L$ 是序列长度。Informer（AAAI 2021 最佳论文）
的核心观察是：**注意力打分矩阵是稀疏的**，绝大多数 query 的注意力分布接近**均匀分布**
（即「看谁都一样」），只有少数 query 有**尖锐的峰**（真正在挑信息）。

于是它只对**最有价值的 top-u 个 query** 做注意力：

$$u = c \\cdot \\ln L, \\qquad c \\approx 5$$

**怎么判断「有价值」？** 用「最大打分 − 平均打分」当**稀疏度**：

$$M(q_i, K) = \\max_j \\frac{q_i k_j^\\top}{\\sqrt d} \\;-\\; \\frac{1}{L}\\sum_j \\frac{q_i k_j^\\top}{\\sqrt d}$$

- 分布均匀 → `max ≈ mean` → `M ≈ 0` → **跳过**
- 有尖峰 → `max ≫ mean` → `M` 大 → **选中它**

**关键工程技巧**：为了算 `M` 而先算完整的 $L \\times L$ 打分矩阵，等于白省。
所以 Informer 用 **U 个随机采样的 key** 来估 `M`（`U = u`），把打分次数从 $L^2$ 降到 $Lu$。

**蒸馏（distilling）**：编码器里每过一层，用 `stride=2` 的卷积把序列长度**砍半**：

```
72 → 36 → 18
```

直觉是「相邻时刻的特征高度冗余，隔一个取一个损失很小」，
同时把下一层注意力的 $L^2$ 直接降到 $1/4$。

**本章的实现**（最小可运行版）：单层注意力、单头、`d_model=32`，
未选中的 query 用 `V` 的均值兜底（等价于「均匀看所有时刻」）。
"""
    ),
    code(T7_CODE),
    md(
        """
### 8.5.1 难点深挖：ProbSparse 省的是「打分次数」，L 小的时候根本不划算

**为什么难**：看到 $O(L \\ln L)$ 就以为「一定更快」是典型的复杂度误读。
$c\\ln L$ 里的常数 **c = 5** 意味着**交叉点在 L ≈ 148** 附近 —— 更小的 L 上，
$u = 5\\ln L$ 占 $L$ 的比例高得吓人。

**实测（打分次数 = 一个 query 与一个 key 做一次点积）**：

| L | u = ⌈c·ln L⌉ | 全注意力 L² | ProbSparse 2Lu | 省下 |
|---|---|---|---|---|
| **72**（本章） | **22** | 5184 | 3168 | **38.9%** |
| 168 | 26 | 28224 | 8736 | 69.0% |
| 336 | 30 | 112896 | 20160 | 82.1% |
| 720 | 33 | 518400 | 47520 | 90.8% |
| 1440 | 37 | 2073600 | 106560 | 94.9% |

**两个必须写进报告的结论**：

1. **L = 72 时只省 38.9%，而它还要额外付「随机采样 key + top-k 排序」的工程开销** ——
   实测训练耗时**没有变快**（见下面的输出）。在这个长度上，ProbSparse 的价值是
   **理论上的**，不是工程上的。
2. **收益随 L 单调上升**：L = 1440 时省 94.9%。所以 Informer 用在实际的长序列场景
   （电力/气象的 720~1440 步长序列）才划算。

**判定规则**：

> 上任何「稀疏注意力」之前，**先算一遍 $u/L$**。
> $u/L < 5\\%$ 才值得（即 $L > 100$ 起步，实务上 $L \\ge 500$）；
> $L$ 只有几十的时候，**直接上全注意力**，然后再考虑「蒸馏砍长度」这条更实际的路。
>
> 顺带一个反直觉点：`u` 随 L **只按对数增长** —— L 从 72 涨到 1440（20 倍），
> u 只从 22 涨到 37（1.7 倍）。**这正是 ProbSparse 能省下 95% 的原因。**
"""
    ),
    code(T8_CODE),
    md(
        """
### 8.5.2 难点深挖：蒸馏砍半，砍掉的到底是什么？

`stride=2` 的卷积把 `72 → 36 → 18`。**为什么敢砍？**

因为时序的相邻时刻**高度冗余**：本数据 1 步自相关 **0.868876**（ch01 实测），
相邻两小时的信息几乎重复。隔一个取一个，理论上损失很小。

**但它有真实代价，必须说清**：

| 砍掉的 | 为什么会出问题 |
|---|---|
| **高频细节** | 负荷的陡坡（早 7 点爬升、晚 20 点回落）恰好是「相邻两点差异最大」的地方，也正是最需要精度的时刻 |
| **精确的相位** | 砍半之后每格代表 2 小时，预测再上采样回 24 步时会有「阶梯感」 |
| **记忆的长度** | 长度砍半的同时，注意力覆盖的**时间跨度**其实变长了（这是好事），但**分辨率**下降了 |

**判定规则**：

> 蒸馏是「用**分辨率**换**长度**」。判断该不该用，看你的预测目标：
> - 目标是**日峰谷时刻**（要准到小时）→ 谨慎蒸馏，或只在深层用
> - 目标是**日总量 / 趋势**（小时级误差可以平均掉）→ 大胆蒸馏
>
> 本章的 TCN 与 Transformer 都只用 stride=1，**没有蒸馏** ——
> 因为在 `L = 72` 这个量级上，蒸馏省不下多少算力，却会丢掉峰值处的分辨率。
> 这也是「不要照搬论文配置」的一个具体例子：**Informer 的蒸馏是为 L = 1440 设计的。**
"""
    ),
    code(T9_CODE),
    md(
        """
## 8.6 全基线对比表怎么读

一张合格的对比表要满足三条：

1. **同一个测试集、同一个打分函数**（本章：1714 个起点 × 24 步，MAE）
2. **有基线**（朴素基线是「0 分线」，不是「陪跑」）
3. **有「相对基线」这一列** —— 只看 15.72 这个绝对数，读者不知道它好不好

**本章实测排序（24 步 MAE，详见上面单元格的输出）**：

| 名次 | 模型 | 24 步 MAE | 相对及格线 | 说明 |
|---|---|---|---|---|
| 1 | **LightGBM（20 特征）** | **15.720011** | **49.8%** | 特征工程 + 24 个步长各一个模型 + 早停 |
| 2 | Transformer（全注意力） | 20.033704 | 63.5% | 全部 6855 个训练样本、20 轮 |
| 3 | Transformer（ProbSparse） | 20.200566 | 64.0% | 与全注意力同种子、同轮数 |
| 4 | TCN（因果卷积 + 膨胀） | 29.980845 | 95.0% | RF=61，只用了 1/4 训练样本 |
| 5 | seasonal naive t-168 | **31.560227** | **100.0%（及格线）** | 上周同一小时 |
| 6 | seasonal naive t-24 | 32.998794 | 104.6% | 昨天同一小时 |
| 7 | moving average 24 | 45.527850 | 144.3% | 最近 24 小时均值 |
| 8 | last value | 60.524392 | 191.8% | 拿最后一个观测值硬撑 24 步 |

> **HistGB 不进这张表**：它是**同族对照**，只在 h=1 / 12 / 24 三个步长上各训一个模型
> （见 §8.3），三点平均 LightGBM **15.4988** vs HistGB **15.7334**，差 **+0.2346（1.51%）**，
> 属于同一量级。要放进 8 行表就得再训 24 个模型，为了对照不值当。

**三条必须读出来的事**：

1. **及格线不是「陪跑」**：`seasonal naive t-168` 用一行代码拿到 31.560227，
   而 `last value` 要 60.524392 —— **同样叫「基线」，差了一倍**。选错基线会高估自己的进步。
2. **深度模型确实赢了基线，但没赢 LightGBM**：Transformer 20.03（好 36.5%）、
   TCN 29.98（只好了 5.0%）—— 都越过了及格线，但离 15.72 还差得远。
3. **朴素的 `t-24` 输给 `t-168`**（32.998794 vs 31.560227）说明**周周期比日周期更值钱** ——
   这正好解释了为什么 LightGBM 的 `lag168` / `ma168` 是它的杀手锏。

**跨章对照（同样测试集）**：ch07 的直接多输出 MLP 是 **17.035627** ——
比朴素基线好得多（−46%），但**仍然打不过 LightGBM**。
差别很直观：LightGBM 有 `lag168` / `ma168` / `temp` 这些**显式特征**，
而 MLP 只有 72 个原始数值，周周期得靠它自己从窗口里「悟」出来。

> **报告写法建议**：把「及格线」那一行**加粗**放在表里，
> 并给一列「相对基线提升 %」。评委一眼就能看出你的工作有没有价值。
"""
    ),
    md(
        """
## 8.7 本章小结：什么时候该上深度模型

这是「模型选择 10%」的核心答案。先把结论摆出来：

| 判据 | 该上深度模型 | 该留在 LightGBM / 统计模型 |
|---|---|---|
| **样本量** | 单序列 > 10 万点，或多序列合并训练 | 单序列几千~几万点（本章 8760 点） |
| **外生变量** | 变量多、交互复杂、要端到端学表征 | 变量可枚举、能手工造出强特征 |
| **预测形态** | 需要**概率预测** / 多变量联合 / 长序列（L ≥ 500） | 单点预测、24 步以内 |
| **周期结构** | 周期不规整（多周期叠加、周期漂移） | 周期规整，`lag24/lag168` 就够 |
| **算力预算** | 有 GPU、能训几十轮 | CPU、两分钟要出结果 |
| **可解释性** | 不需要（或可用 attention 权重凑） | 需要特征重要性写报告 |

**回看本章的三条实测证据**：

1. **TCN 打不过 LightGBM**：RF=61 覆盖不到周周期（168），而 LightGBM 用 `lag168`
   显式把周周期喂了进去 —— **深度模型的「自动特征提取」在这里输给了手工特征**
2. **ProbSparse 在 L=72 上省不下时间**（只省 38.9% 的打分次数，还多了采样与 top-k 开销）——
   Informer 的价值在长序列，本章的序列太短
3. **Transformer 把全部 6855 个样本训满 20 轮，仍差 LightGBM 4.31 kW**（20.03 vs 15.72）——
   而 TCN 只训 1/4 样本时差得更远（29.98）。两个深度模型都**越过了及格线**，
   但要追平 LightGBM 需要的是**数量级更多**的算力与数据，
   而这一份数据（8760 点、单序列、周期规整）**根本不支持**这种投入

**那深度模型的价值在哪？** 三个真实场景：

| 场景 | 为什么树模型不行 |
|---|---|
| **多序列联合训练**（几百个台区一起训） | 树模型只能一列一列喂，学不到「跨台区的相似模式」 |
| **概率预测 / 分位数预测** | 树模型要给分位数就得训很多个模型；深度模型一个多头就够 |
| **极端长序列**（L ≥ 1000，如气象） | 树模型的特征数量会爆炸；注意力/卷积天然处理长依赖 |

> **一句话小结（可以背）**：
> **「先把朴素基线和 LightGBM 打满，再决定要不要上深度模型。
> 深度模型不是更高级，只是更贵 —— 花得起，且任务确实需要，才用。」**

**下一步**：`final/` 综合题会把 ch01~ch08 串成一条完整流水线
（清洗 → 特征 → 多模型对比 → 多步预测 → 评估可视化 → 报告），
并且**强制要求**你在报告里给出「及格线」和「相对基线提升」两列。
"""
    ),
]

EXERCISE = [
    md(
        """
# ch08 进阶与基线（练习版）

> 补全所有 `____`，让每个代码块末尾的 `assert` 全绿。
> 数据：`data/load_curve.csv` → A 台区 8760 h；窗口 72、步长 24、起点从 168 开始、按时间 8:2 切分（与 ch07 完全一致）。
>
> **每题都跟了 `# 提示：`**，照着提示能独立做完。
> 涉及 torch / LightGBM 训练的断言都是**区间或相对大小**（不是死值）；
> **朴素基线与感受野公式是确定性的**，那几个数才卡死。

## 任务清单

| 任务 | 主题 | 挖空 | 对应讲解小节 |
|---|---|---|---|
| 1 | 朴素基线家族（last / sn24 / sn168 / ma24） | 7 | §8.1 |
| 2 | 20 维特征 + LightGBM 24 个模型 | 11 | §8.2 |
| 3 | 特征重要性 top-10 | 1 | §8.2 |
| 4 | 与 sklearn HistGB 同族对照 | 2 | §8.3 |
| 5 | 因果卷积 + 残差块 + 感受野公式与实测 | 6 | §8.4 |
| 6 | TCN 训练 | 2 | §8.4.4 |
| 7 | ProbSparse：u、稀疏度、top-u、打分次数、蒸馏 | 4 | §8.5 |
| 8 | Transformer 训练（全注意力 vs ProbSparse） | 2 | §8.5.1 |
| 9 | 全基线对比表 | 1 | §8.6 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(PLOT_SETUP),
    code(SCAFFOLD),
    md("## 任务 1：朴素基线家族 —— 先给及格线定价（§8.1）"),
    code(T1_CODE),
    md("## 任务 2：特征工程 + LightGBM 真基线（§8.2）"),
    code(T2_CODE),
    md("## 任务 3：特征重要性 top-10（§8.2.1）"),
    code(T3_CODE),
    md("## 任务 4：同族对照 —— sklearn HistGradientBoosting（§8.3）"),
    code(T4_CODE),
    md("## 任务 5：TCN 的因果卷积、残差块与感受野（§8.4）"),
    code(T5_CODE),
    md("## 任务 6：训练 TCN 并评估（§8.4.4）"),
    code(T6_CODE),
    md("## 任务 7：ProbSparse 注意力与蒸馏（§8.5）"),
    code(T7_CODE),
    md("## 任务 8：Transformer 训练，全注意力 vs ProbSparse（§8.5.1）"),
    code(T8_CODE),
    md("## 任务 9：全基线对比表（§8.6）"),
    code(T9_CODE),
    md(
        """
## 自查清单（能不看笔记答出来才算过）

- [ ] 9 个代码块的 assert 全部通过
- [ ] 四个朴素基线的 24 步 MAE 分别是多少？及格线是谁？
- [ ] 为什么 `t-168` 比 `t-24` 还好？它对齐了哪几个周期因子？
- [ ] `last value` 的整体 MAE 最差，为什么它的 h=24 反而最小？
- [ ] 树模型为什么也要求日历特征做 sin·cos 编码？23 点与 0 点的编码距离是多少？
- [ ] LightGBM 与 sklearn HistGB 的三个主要差异是什么（分裂方式 / 正则 / 早停）？
- [ ] 因果卷积为什么只补左边？补多少格？
- [ ] 感受野公式写出来，并代入本章参数算出结果；实测值是多少？
- [ ] RF=61 < 窗口 72 意味着什么？TCN 为什么打不过 LightGBM？
- [ ] ProbSparse 的 `u` 怎么算？L=72 时 u 是多少、省下多少打分次数？
- [ ] 为什么 L=72 时 ProbSparse 并不更快？
- [ ] 蒸馏把 72 变成多少？它砍掉的是什么？

## 关键真值对照（做完再对）

| 量 | 值 |
|---|---|
| 样本数 / 切分点 / 训练 / 测试 | 8569 / 6855 / 6855 / 1714 |
| 测试段时段 | 2025-10-20 15:00 ~ 2025-12-31 00:00 |
| last value：24 步 / h1 / h24 MAE | 60.524392 / 30.0939 / 21.6066 |
| seasonal naive t-24 | 32.998794（h1 33.0215、h24 33.1928） |
| **seasonal naive t-168（及格线）** | **31.560227**（h1 31.5845、h24 31.6804） |
| moving average 24 | 45.527850 |
| 特征个数 / 拟合段 / 验证段 / 测试段 | 20 / 5826 / 1029 / 1714 |
| 感受野公式值 = 实测值 | 61 |
| 被浪费的输入格数（72 − 61） | 11 |
| 因果卷积层数（含输入层） | 9 |
| ProbSparse：u（L=72, c=5） | 22 |
| 打分次数：全注意力 / ProbSparse | 5184 / 3168（省 38.9%） |
| L=720 时省下的打分比例 | 90.8%（u=33） |
| L=1440 时省下的打分比例 | 94.9%（u=37） |
| 蒸馏后长度链 | 72 → 36 → 18 |
| TCN 训练样本数（stride=4）/ Transformer 训练样本数 | 1714 / 6855 |

> 上面全是**确定性**的量（不同机器上重跑一模一样），所以代码里可以卡死值。
> 下面是**训练出来的**量，只做区间断言，这张表里给的是本章实跑的一次参考值：

| 量（参考值，随种子/线程数浮动） | 值 |
|---|---|
| LightGBM 24 步 / h1 / h24 MAE | 15.720011 / 15.0203 / 16.0834 |
| HistGB 三点平均 vs LightGBM 三点平均 | 15.7334 vs 15.4988（差 +0.2346） |
| TCN 24 步 / h1 / h24 MAE（stride=4，12 轮） | 29.980845 / 27.5360 / 22.2883 |
| Transformer 全注意力 / ProbSparse 24 步 MAE | 20.033704 / 20.200566 |
| Transformer 两个变体的训练耗时 | 12.6 s / 19.6 s |
"""
    ),
]

if __name__ == "__main__":
    report([build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)])
