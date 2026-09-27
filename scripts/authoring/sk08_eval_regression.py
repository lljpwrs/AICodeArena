#!/usr/bin/env python3
"""ch08 回归评估：MAE/MSE/RMSE/MAPE/R²/残差分析（load_reg.csv）。

构建：/Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk08_eval_regression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"

IMPORTS = '''from pathlib import Path

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (max_error, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error,
                             r2_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data")
'''

SETUP = '''# 数据与口径沿用 ch04：NUM 5 列 + OneHot(供电区域类型)，台区编号是 ID 列不进特征
reg = pd.read_csv(DATA / "load_reg.csv")
NUM = ["容量kVA", "投运年限", "月平均温度", "月平均湿度", "户数"]
CAT = ["供电区域类型"]
TARGET = "日均负荷"

tr_idx, te_idx = train_test_split(np.arange(len(reg)), test_size=0.2, random_state=42)
reg_tr, reg_te = reg.iloc[tr_idx], reg.iloc[te_idx]
y_tr = reg_tr[TARGET].to_numpy()
y_te = reg_te[TARGET].to_numpy()

oh = OneHotEncoder(sparse_output=False).fit(reg_tr[CAT])
Xtr = np.hstack([reg_tr[NUM].to_numpy(), oh.transform(reg_tr[CAT])])
Xte = np.hstack([reg_te[NUM].to_numpy(), oh.transform(reg_te[CAT])])
ss = StandardScaler().fit(Xtr)

# 两个待评估模型：线性（scaled）与 RF——本章重点是「怎么评」，不是「怎么训」
lin = LinearRegression().fit(ss.transform(Xtr), y_tr)
p_lin = lin.predict(ss.transform(Xte))
rf = RandomForestRegressor(n_estimators=200, random_state=42).fit(Xtr, y_tr)
p_rf = rf.predict(Xte)

print("sklearn", sklearn.__version__, "| train/test:", len(tr_idx), "/", len(te_idx),
      "| y_te mean/std:", round(float(y_te.mean()), 4), round(float(y_te.std()), 4))
assert p_lin.shape == (160,) and p_rf.shape == (160,)
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def reg_report(tag, y_true, y_pred):
    """一行输出 R2 / RMSE / MAE，返回 (r2, rmse, mae)。"""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{tag}: R2={r2:.4f} RMSE={rmse:.4f} MAE={mae:.4f}")
    return r2, rmse, mae


print("脚手架就绪：reg_report")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) 造「均值基线」：用训练段均值当预测，算 MAE / RMSE / R²
# @@hint np.full(len(y_te), y_tr.mean())；三个指标各调一次 metrics 函数
base = np.full(len(y_te), y_tr.mean())
mae_b = mean_absolute_error(y_te, base)
rmse_b = float(np.sqrt(mean_squared_error(y_te, base)))
r2_b = r2_score(y_te, base)
# @@end
print(f"均值基线: MAE={mae_b:.4f} RMSE={rmse_b:.4f} R2={r2_b:.4f}")
print("解读: R2 为负——比『用测试集均值当预测』还差一点，因为它用的是训练段均值")

assert round(mae_b, 4) == 139.8473
assert round(rmse_b, 4) == 175.6490
assert round(float(r2_b), 4) == -0.0123, "负 R²：模型上限就是『不作为』也它不如"
'''

E2_CODE = '''# @@todo(2) 手工复算 R²：1 - SSE/SST，与 sklearn 对照
# @@hint sse = ((y_te - p_lin) ** 2).sum()；sst = ((y_te - y_te.mean()) ** 2).sum()
sse = ((y_te - p_lin) ** 2).sum()
sst = ((y_te - y_te.mean()) ** 2).sum()
r2_manual = 1 - sse / sst
# @@end
print("手工 R2:", round(float(r2_manual), 6), "| sklearn:",
      round(float(r2_score(y_te, p_lin)), 6))

assert round(float(r2_manual), 6) == round(float(r2_score(y_te, p_lin)), 6) == 0.923381
assert round(float(r2_manual), 4) == 0.9234
'''

E3_CODE = '''# @@todo(3) R² 的三种极端：预测全取负 / 常数=测试集均值 / 常数=训练段均值
# @@hint r2_score(y_te, -p_lin)；r2_score(y_te, np.full(len(y_te), y_te.mean()))
r2_neg = r2_score(y_te, -p_lin)
r2_te_mean = r2_score(y_te, np.full(len(y_te), y_te.mean()))
r2_tr_mean = r2_score(y_te, np.full(len(y_te), y_tr.mean()))
# @@end
print(f"预测取负: R2={r2_neg:.4f} | 常数=测试集均值: {r2_te_mean:.4f} | 常数=训练段均值: {r2_tr_mean:.4f}")

assert round(float(r2_neg), 4) == -21.0964, "方向反了能有多惨，R² 没有下限"
assert round(float(r2_te_mean), 6) == 0.0, "R² 的 0 点定义就是测试集均值"
assert round(float(r2_tr_mean), 4) == -0.0123
'''

E4_CODE = '''# @@todo(4) MAPE 的单位不变性：y 与预测同乘 0.1，MAE 变 MAPE 不变
# @@hint mean_absolute_percentage_error(y_te * 0.1, p_lin * 0.1)
mape_full = mean_absolute_percentage_error(y_te, p_lin)
mape_scaled = mean_absolute_percentage_error(y_te * 0.1, p_lin * 0.1)
# @@end
print(f"MAPE 原始={mape_full:.4f} | 同乘 0.1 后={mape_scaled:.6f}（不变）| MAE 变为 {mean_absolute_error(y_te * 0.1, p_lin * 0.1):.4f}")

# MAPE 的真陷阱：y 接近 0 时分母爆炸（合成演示，本数据 y 最小 103.64 安全）
y_fake = np.array([100.0, 100.0, 1.0])
p_fake = np.array([90.0, 110.0, 2.0])
print("合成小分母 MAPE:", round(float(mean_absolute_percentage_error(y_fake, p_fake)), 4),
      "← 第三行 100% 误差独占大头")
print("对照 MAE:", round(float(mean_absolute_error(y_fake, p_fake)), 4))

assert round(float(mape_full), 4) == 0.1203
assert round(float(mape_scaled), 6) == 0.120294
assert round(float(mean_absolute_percentage_error(y_fake, p_fake)), 4) == 0.4, "0.1+0.1+1.0 的均值"
'''

E5_CODE = '''# @@todo(5) 残差基本量：e = y - y_hat，算线性与 RF 的残差均值
# @@hint res_lin = y_te - p_lin
res_lin = y_te - p_lin
res_rf = y_te - p_rf
mean_lin = float(res_lin.mean())
mean_rf = float(res_rf.mean())
# @@end
print(f"线性残差 mean={mean_lin:.4f} std={float(res_lin.std()):.4f} | RF 残差 mean={mean_rf:.4f}")
print("解读: 残差均值≈0 是无偏的必要条件，但≠0 也不报错——要自己查")

assert round(mean_lin, 4) == 3.0419
assert round(mean_rf, 4) == 1.9413
assert round(float(res_lin.std()), 4) == 48.2273
'''

E6_CODE = '''# @@todo(6) 大误差定位：|残差| 的 p90，以及超过 p90 的样本真实负荷均值
# @@hint np.quantile(np.abs(res_lin), 0.9)
q90 = float(np.quantile(np.abs(res_lin), 0.9))
big = y_te[np.abs(res_lin) > q90]
big_mean = float(big.mean())
# @@end
print(f"|残差| p90={q90:.4f} | 大误差样本均值={big_mean:.4f} vs 全体 {float(y_te.mean()):.4f}")
print(f"RF 的 p90={float(np.quantile(np.abs(res_rf), 0.9)):.4f}（把大误差也压小了）")

assert round(q90, 4) == 79.0801
assert round(big_mean, 4) == 367.6875
assert round(float(np.quantile(np.abs(res_rf), 0.9)), 4) == 60.9175
'''

E7_CODE = '''# @@todo(7) 离群敏感对照：median_AE vs MAE，以及 max_error
# @@hint median_absolute_error(y_te, p_lin)；max_error(y_te, p_lin)
med_lin = median_absolute_error(y_te, p_lin)
med_rf = median_absolute_error(y_te, p_rf)
mx_lin = max_error(y_te, p_lin)
mx_rf = max_error(y_te, p_rf)
# @@end
print(f"median_AE: 线性={med_lin:.4f} RF={med_rf:.4f} | MAE: 线性={mean_absolute_error(y_te, p_lin):.4f}")
print(f"max_error: 线性={mx_lin:.4f} RF={mx_rf:.4f}")
print("解读: median < mean 说明误差右偏（少数大离群）；报告里带一个 max_error 更诚实")

assert round(float(med_lin), 4) == 28.3693
assert round(float(med_rf), 4) == 23.3062
assert round(float(mx_lin), 4) == 175.2451
assert round(float(mx_rf), 4) == 115.9893
'''

E8_CODE = '''# @@todo(8) 交叉验证里的回归指标：scoring 用 neg_ 前缀，越接近 0 越好
# @@hint cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
X_all = np.hstack([reg[NUM].to_numpy(), oh.transform(reg[CAT])])
cv_r2 = cross_val_score(LinearRegression(), ss.transform(X_all), reg[TARGET], cv=5,
                        scoring="r2")
cv_nrmse = cross_val_score(LinearRegression(), ss.transform(X_all), reg[TARGET], cv=5,
                           scoring="neg_root_mean_squared_error")
# @@end
print("cv R2:", np.round(cv_r2, 4).tolist())
print("cv negRMSE:", np.round(cv_nrmse, 4).tolist(), "← 全是负的：sklearn 惯例『越大越好』")
print("真正的 RMSE =", np.round(-cv_nrmse, 4).tolist())

assert np.round(cv_r2, 4).tolist() == [0.9476, 0.9387, 0.9463, 0.9447, 0.9335]
assert np.round(cv_nrmse, 4).tolist() == [-41.9493, -46.0615, -44.3359, -45.0196, -47.8884]
assert (cv_nrmse < 0).all(), "neg_ 前缀指标恒为负"
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 一切评估从基线开始：均值基线给出「不作为」的成绩单
base = np.full(len(y_te), y_tr.mean())
mae_b = mean_absolute_error(y_te, base)
rmse_b = float(np.sqrt(mean_squared_error(y_te, base)))
r2_b = r2_score(y_te, base)
print(f"均值基线: MAE={mae_b:.4f} RMSE={rmse_b:.4f} R2={r2_b:.4f}")
print(f"线性    : MAE={mean_absolute_error(y_te, p_lin):.4f} RMSE={float(np.sqrt(mean_squared_error(y_te, p_lin))):.4f} R2={r2_score(y_te, p_lin):.4f}")
print(f"RF      : MAE={mean_absolute_error(y_te, p_rf):.4f} RMSE={float(np.sqrt(mean_squared_error(y_te, p_rf))):.4f} R2={r2_score(y_te, p_rf):.4f}")

assert round(mae_b, 4) == 139.8473
assert round(rmse_b, 4) == 175.6490
assert round(float(r2_b), 4) == -0.0123, "负 R²：用训练段均值当预测，比测试集均值还差一点"
assert round(float(r2_score(y_te, p_lin)), 4) == 0.9234
assert round(float(r2_score(y_te, p_rf)), 4) == 0.9531
assert round(float(mean_absolute_error(y_te, p_rf)), 4) == 30.0258
assert round(float(np.sqrt(mean_squared_error(y_te, p_rf))), 4) == 37.8045
'''

S2_CODE = '''# 手工复算 R²：它不是魔法，就是 1 - SSE/SST
sse = ((y_te - p_lin) ** 2).sum()
sst = ((y_te - y_te.mean()) ** 2).sum()
r2_manual = 1 - sse / sst
print("手工 R2:", round(float(r2_manual), 6), "| sklearn:",
      round(float(r2_score(y_te, p_lin)), 6))
print("SSE(误差平方和):", round(float(sse), 2), "| SST(总平方和):", round(float(sst), 2))

assert round(float(r2_manual), 6) == round(float(r2_score(y_te, p_lin)), 6) == 0.923381
'''

S3_CODE = '''# R² 的三种极端：没有下限、0 点定义、以及「方向反了」有多惨
r2_neg = r2_score(y_te, -p_lin)
r2_te_mean = r2_score(y_te, np.full(len(y_te), y_te.mean()))
r2_tr_mean = r2_score(y_te, np.full(len(y_te), y_tr.mean()))
print(f"预测取负: R2={r2_neg:.4f} | 常数=测试集均值: {r2_te_mean:.4f} | 常数=训练段均值: {r2_tr_mean:.4f}")
print("解读: R² 的 0 点是『预测测试集均值』；训练/测试分布一漂移，常数预测就变负")

assert round(float(r2_neg), 4) == -21.0964, "R² 没有下限，-21 说明方向性错误被重罚"
assert round(float(r2_te_mean), 6) == 0.0
assert round(float(r2_tr_mean), 4) == -0.0123
'''

S4_CODE = '''# MAPE：单位不变是优点，小分母爆炸是原罪
mape_full = mean_absolute_percentage_error(y_te, p_lin)
mape_scaled = mean_absolute_percentage_error(y_te * 0.1, p_lin * 0.1)
print(f"MAPE 原始={mape_full:.4f} | 同乘 0.1 后={mape_scaled:.6f}（不变）| MAE 同乘后={mean_absolute_error(y_te * 0.1, p_lin * 0.1):.4f}")

y_fake = np.array([100.0, 100.0, 1.0])
p_fake = np.array([90.0, 110.0, 2.0])
print("合成小分母 MAPE:", round(float(mean_absolute_percentage_error(y_fake, p_fake)), 4),
      "| 对照 MAE:", round(float(mean_absolute_error(y_fake, p_fake)), 4))
print("解读: MAE 一样是 30，但第三行分母只有 1，MAPE 被拽到 36.67%")
print("本数据 y 最小", round(float(y_te.min()), 2), "，暂时安全；负荷预测遇低谷时段就要当心")

assert round(float(mape_full), 4) == 0.1203
assert round(float(mape_scaled), 6) == 0.120294
assert round(float(mean_absolute_percentage_error(y_fake, p_fake)), 4) == 0.4, "0.1+0.1+1.0 的均值"
assert round(float(mean_absolute_error(y_fake, p_fake)), 4) == 7.0
assert round(float(y_te.min()), 2) == 103.64
'''

S5_CODE = '''# 残差 = y - y_hat：回归评估的「原始凭证」
res_lin = y_te - p_lin
res_rf = y_te - p_rf
print(f"线性残差 mean={float(res_lin.mean()):.4f} std={float(res_lin.std()):.4f} | RF 残差 mean={float(res_rf.mean()):.4f}")
print("解读: 残差均值 3.04 vs 预测量级 373 —— 全局近似无偏；std 48.2 就是 RMSE 的量级")

assert round(float(res_lin.mean()), 4) == 3.0419
assert round(float(res_rf.mean()), 4) == 1.9413
assert round(float(res_lin.std()), 4) == 48.2273
'''

S6_CODE = '''# 大误差定位：p90 分位把最差的 10% 样本拎出来
q90 = float(np.quantile(np.abs(res_lin), 0.9))
big = y_te[np.abs(res_lin) > q90]
print(f"|残差| p90={q90:.4f} | 大误差样本均值={float(big.mean()):.4f} vs 全体 {float(y_te.mean()):.4f}")
print(f"RF 的 p90={float(np.quantile(np.abs(res_rf), 0.9)):.4f}")
print("解读: 大误差样本的真实负荷并不极端（367.7 vs 373.7）——误差不在极值端，而在结构上")

assert round(q90, 4) == 79.0801
assert round(float(big.mean()), 4) == 367.6875
assert round(float(np.quantile(np.abs(res_rf), 0.9)), 4) == 60.9175
'''

S7_CODE = '''# 离群敏感对照：mean vs median，配一个 max_error 更诚实
med_lin = median_absolute_error(y_te, p_lin)
med_rf = median_absolute_error(y_te, p_rf)
print(f"median_AE: 线性={med_lin:.4f} RF={med_rf:.4f} | MAE: 线性={mean_absolute_error(y_te, p_lin):.4f}")
print(f"max_error: 线性={max_error(y_te, p_lin):.4f} RF={max_error(y_te, p_rf):.4f}")
print("解读: median 28.37 < MAE 37.63 → 误差右偏，少数大离群抬高了平均值")

assert round(float(med_lin), 4) == 28.3693
assert round(float(med_rf), 4) == 23.3062
assert round(float(max_error(y_te, p_lin)), 4) == 175.2451
assert round(float(max_error(y_te, p_rf)), 4) == 115.9893
assert med_lin < mean_absolute_error(y_te, p_lin), "median < mean：右偏证据"
'''

S8_CODE = '''# 交叉验证中的回归指标：scoring 带 neg_ 前缀，越接近 0 越好
X_all = np.hstack([reg[NUM].to_numpy(), oh.transform(reg[CAT])])
cv_r2 = cross_val_score(LinearRegression(), ss.transform(X_all), reg[TARGET], cv=5,
                        scoring="r2")
cv_nrmse = cross_val_score(LinearRegression(), ss.transform(X_all), reg[TARGET], cv=5,
                           scoring="neg_root_mean_squared_error")
print("cv R2:", np.round(cv_r2, 4).tolist())
print("cv negRMSE:", np.round(cv_nrmse, 4).tolist(), "← 全是负的：sklearn 统一『越大越好』")
print("真正的 RMSE =", np.round(-cv_nrmse, 4).tolist())
print("解读: 直接拿 negRMSE 当 RMSE 报出去，等于把误差全报成负数——经典口误")

assert np.round(cv_r2, 4).tolist() == [0.9476, 0.9387, 0.9463, 0.9447, 0.9335]
assert np.round(cv_nrmse, 4).tolist() == [-41.9493, -46.0615, -44.3359, -45.0196, -47.8884]
assert (cv_nrmse < 0).all(), "neg_ 前缀指标恒为负"
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    code(IMPORTS),
    md(
        """
# ch08 回归评估：MAE / RMSE / R² / 残差分析

> 数据：`load_reg.csv`（800 行，目标「日均负荷」）。口径沿用 ch04：NUM 5 列 +
> OneHot(供电区域类型)，`test_size=0.2, random_state=42`，台区编号是 ID 列不进特征。

**本章考点**（对应竞赛「模型性能评估 10%」）

1. 指标全家桶怎么选：MAE / RMSE / MAPE / R² 各自回答什么问题
2. R² 的三种极端与 0 点定义
3. 残差分析：误差藏在哪、右偏还是无偏
4. 交叉验证里的 `neg_` 前缀陷阱

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| R² 没有下限 | 预测取负 R²=-21.1，方向错误重罚 | §3.3 难点深挖 |
| MAPE 小分母爆炸 | y→0 时 MAPE 失控，MAE 无恙 | §3.4 难点深挖 |
| median < mean 的信号 | 误差右偏，报告只给 MAE 会漏 | §3.7 |
| negRMSE 当 RMSE 报 | 符号口误，评委会抓 | §3.8 |
"""
    ),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 3.1 基线先行：先知道「不作为」值多少分

任何回归指标都要和基线对照：**用训练段均值当预测**。它给出 MAE=139.85、
RMSE=175.65、R²=-0.0123（注意是负的——为什么训练均值比测试均值还差一点，
见 §3.3）。模型的所有光鲜数字，都要减掉这张底牌才有意义。
"""
    ),
    code(S1_CODE),
    md(
        """
## 3.2 R² 的解剖：1 - SSE/SST

`r2_score` 不是黑盒：SSE 是模型误差平方和，SST 是「都用测试集均值」的误差平方和。
分子比分母小 → 正；打平 → 0；超过 → 负。sklearn 与手工复算到小数点后 6 位一致。
"""
    ),
    code(S2_CODE),
    md(
        """
### 3.3 难点深挖：R² 的三种极端

- **预测取负**（方向全反）：R²=-21.0964。R² **没有下限**，方向性错误被平方重罚。
- **常数=测试集均值**：R²=0.000000，精确的 0——这就是 0 点的定义。
- **常数=训练段均值**：R²=-0.0123。训练/测试分布有漂移时，「不作为」也是负分。

**答辩要点**：R² 只说明「比均值预测好多少」，它不等于模型好坏的绝对量度——
两份数据的 R² 不可直接横比。
"""
    ),
    code(S3_CODE),
    md(
        """
### 3.4 难点深挖：MAPE 的单位不变性与小分母爆炸

MAPE 同乘 0.1 完全不变（0.120294 → 0.120294），跨单位可比是它的卖点；
但分母是真实值：合成数据里第三行 y=1 时，一行就把 MAPE 拽到 40%，
而 MAE 依然镇定（7.0）。**低谷负荷/接近零的量测场景慎用 MAPE**。
本数据 y 最小 103.64，暂时安全。
"""
    ),
    code(S4_CODE),
    md(
        """
## 3.5 残差：评估的原始凭证

e = y - y_hat。线性模型残差均值 3.04（量级 373，近似无偏），std 48.23 正是
RMSE 的量级（RMSE=sqrt(mean(e²))）。RF 残差均值 1.94。残差均值≈0 是必要
条件而非充分条件——结构性偏差藏在分位与相关性里，往下看。
"""
    ),
    code(S5_CODE),
    md(
        """
## 3.6 大误差定位：分位数比直方图好操作

|残差| 的 p90=79.08，把最差 10% 样本拎出来：真实负荷均值 367.69 vs 全体
373.72——**大误差不在极值端，而是结构性的**（某些区域/形态系统偏差）。
RF 的 p90=60.92，把大误差也压小了，这就是集成模型在「误差分布」上的改善。
"""
    ),
    code(S6_CODE),
    md(
        """
### 3.7 难点深挖：median < mean 的信号

median_AE 28.37 < MAE 37.63：**误差右偏**，少数大离群抬高平均值。max_error
线性 175.25 / RF 115.99——最差单点依然不小，报告里给出它比只报 RMSE 诚实。
"""
    ),
    code(S7_CODE),
    md(
        """
### 3.8 难点深挖：交叉验证里的 neg_ 前缀

sklearn 的 scoring 统一「越大越好」，于是 RMSE 变成 `neg_root_mean_squared_error`，
五折全是负数（-41.95 ~ -47.89）。**报成绩前先取负**——把 -45 当 RMSE 报出去
是经典口误。R² 本身越大越好，无前缀。
"""
    ),
    code(S8_CODE),
    md(
        """
## 小结

1. **基线先行**：均值基线 MAE=139.85 / R²=-0.0123，模型数字减底牌才有意义。
2. R² = 1 - SSE/SST，无下限、0 点=测试集均值；跨数据集不可横比。
3. MAPE 单位不变但小分母爆炸；MAE 对离群稳健、RMSE 平方放大。
4. 残差分析三件套：均值（无偏性）、p90 定位、median vs mean（右偏信号）。
5. CV 里 `neg_` 前缀指标恒为负，报成绩先取负。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    code(IMPORTS),
    md(
        """
# ch08 回归评估（练习版）

> 按提示补全 `____`，跑通所有 assert。数据与两个待评估模型已在脚手架就绪：
> `p_lin`（线性）与 `p_rf`（RF），`y_te` 是测试段真实值。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外。
"""
    ),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：均值基线

用训练段均值当预测，算出 MAE / RMSE / R²，看看「不作为」值多少分。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：手工复算 R²

R² 就是 1 - SSE/SST，手工算一遍与 sklearn 对照。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：R² 的三种极端

预测全取负、常数=测试集均值、常数=训练段均值，各算一次 R²。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：MAPE 的单位不变性

y 与预测同乘 0.1，验证 MAE 缩水、MAPE 不变；再用合成数据看小分母爆炸。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：残差基本量

e = y - y_hat，算线性与 RF 的残差均值与线性残差的 std。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：大误差定位

|残差| 的 p90，以及超过 p90 的样本真实负荷均值——误差藏在哪？
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：离群敏感对照

median_AE vs MAE，加 max_error——验证误差右偏。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：交叉验证里的 neg_ 前缀

5 折交叉验证，scoring 分别用 r2 与 neg_root_mean_squared_error。
"""
    ),
    code(E8_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 任务完成。自查：R² 的 0 点定义、negRMSE 的符号、
median 与 MAE 的大小关系，三个概念都能脱稿说出。
"""
    ),
]

NAME = "ch08_eval_regression"

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
