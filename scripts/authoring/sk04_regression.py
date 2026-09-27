"""sk04 —— 回归模型：线性 / Ridge / Lasso / 决策树 / 随机森林 / GBDT

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk04_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch04_regression"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

DATA = Path("data")

reg = pd.read_csv(DATA / "load_reg.csv")     # 回归：预测「日均负荷」

NUM = ["容量kVA", "投运年限", "月平均温度", "月平均湿度", "户数"]
CAT = ["供电区域类型"]
TARGET = "日均负荷"

# 台区编号是 ID 列，直接当特征会出事（§3.7 演示），本章特征矩阵不含它
tr_idx, te_idx = train_test_split(np.arange(len(reg)), test_size=0.2, random_state=42)
reg_tr, reg_te = reg.iloc[tr_idx], reg.iloc[te_idx]
y_tr, y_te = reg_tr[TARGET], reg_te[TARGET]

# 独热 + 标准化（预处理细节 ch02 讲透，这里直接组装）
oh = OneHotEncoder(sparse_output=False).fit(reg_tr[CAT])
feat = NUM + [f"区域_{c}" for c in oh.categories_[0]]
Xtr = np.hstack([reg_tr[NUM].to_numpy(), oh.transform(reg_tr[CAT])])
Xte = np.hstack([reg_te[NUM].to_numpy(), oh.transform(reg_te[CAT])])
ss = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

print("sklearn", sklearn.__version__)
print("train", Xtr_s.shape, "test", Xte_s.shape,
      "| y 均值 train", round(float(y_tr.mean()), 2), "test", round(float(y_te.mean()), 2))'''

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
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 线性回归：scaled 版与 raw 版各 fit 一次
lin = LinearRegression().fit(Xtr_s, y_tr)
lin_raw = LinearRegression().fit(Xtr, y_tr)

print("scaled coef_:", np.round(lin.coef_, 4).tolist())
print("scaled intercept:", round(float(lin.intercept_), 4))
print("raw coef_   :", np.round(lin_raw.coef_, 4).tolist())

reg_report("LIN  train", y_tr, lin.predict(Xtr_s))
r2_te, rmse_te, mae_te = reg_report("LIN  test ", y_te, lin.predict(Xte_s))
reg_report("LIN-raw test", y_te, lin_raw.predict(Xte))

# 解读：容量kVA 的标准化系数最大（每 +1 标准差 → 负荷 +113.57）
# 注意视角差：线性系数里容量第一，RF 特征重要性里户数第一(0.7587) —— 两把尺子
top = feat[int(np.argmax(np.abs(lin.coef_[: len(NUM)])))]
print("数值特征中 |系数| 最大:", top)

assert round(float(lin.intercept_), 4) == 393.0875
assert round(r2_te, 4) == 0.9234 and round(rmse_te, 4) == 48.3232
assert round(float(mae_te), 4) == 37.6304
assert round(r2_score(y_tr, lin.predict(Xtr_s)), 4) == 0.9480
assert round(r2_score(y_te, lin_raw.predict(Xte)), 4) == 0.9234, "R2 对线性缩放不变"
assert not np.allclose(lin.coef_, lin_raw.coef_), "但系数完全不同"
assert top == "容量kVA", "线性系数视角: 容量最强(113.57)"
'''

S2_CODE = '''# Ridge 收缩 vs Lasso 归零：正则化是一把有刻度的刀
for alpha in (0.1, 1.0, 10.0):
    rd = Ridge(alpha=alpha).fit(Xtr_s, y_tr)
    r2 = r2_score(y_te, rd.predict(Xte_s))
    print(f"Ridge a={alpha:<6} test R2={r2:.4f} | 容量系数={rd.coef_[0]:.3f}")

print()
for alpha in (0.1, 1.0, 10.0, 100.0):
    la = Lasso(alpha=alpha).fit(Xtr_s, y_tr)
    r2 = r2_score(y_te, la.predict(Xte_s))
    nz = int((la.coef_ != 0).sum())
    kept = [feat[i] for i in range(len(feat)) if la.coef_[i] != 0]
    print(f"Lasso a={alpha:<6} test R2={r2:.4f} | 非零 {nz}/9 | 保留: {kept}")

assert int((Lasso(alpha=1.0).fit(Xtr_s, y_tr).coef_ != 0).sum()) == 6
assert int((Lasso(alpha=10.0).fit(Xtr_s, y_tr).coef_ != 0).sum()) == 3
la10 = Lasso(alpha=10.0).fit(Xtr_s, y_tr)
kept10 = {feat[i] for i in range(len(feat)) if la10.coef_[i] != 0}
assert kept10 == {"容量kVA", "户数", "区域_工业区"}, "a=10 只留最强三路信号"
assert round(r2_score(y_te, Ridge(alpha=10.0).fit(Xtr_s, y_tr).predict(Xte_s)), 4) == 0.9245
'''

S3_CODE = '''# Lasso 的尺度敏感性：同一 alpha，scaled 与 raw 判完全不同的死刑
la_s = Lasso(alpha=1.0).fit(Xtr_s, y_tr)
la_r = Lasso(alpha=1.0).fit(Xtr, y_tr)
print("scaled 非零:", int((la_s.coef_ != 0).sum()), "| raw 非零:", int((la_r.coef_ != 0).sum()))
print("scaled coef:", np.round(la_s.coef_, 3).tolist())
print("raw   coef:", np.round(la_r.coef_, 4).tolist(),
      "← 容量kVA 量纲大(百位)，原始尺度下惩罚相对变小")

assert int((la_r.coef_ != 0).sum()) == 7
assert round(float(la_r.coef_[0]), 4) == 0.3892, "raw 版容量系数 0.39 vs scaled 版 113"
assert round(float(abs(la_s.coef_[0])), 3) == 113.061, "同一特征，两种尺度差 290 倍"
'''

S4_CODE = '''# 决策树：不限制深度 = 把训练段背下来
dt = DecisionTreeRegressor(random_state=42).fit(Xtr_s, y_tr)
print("全深树: depth =", dt.get_depth(), "| leaves =", dt.get_n_leaves())
reg_report("DT-all train", y_tr, dt.predict(Xtr_s))
reg_report("DT-all test ", y_te, dt.predict(Xte_s))

# 剪枝：max_depth 是最常用的泛化阀门
dt5 = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr_s, y_tr)
dt9 = DecisionTreeRegressor(max_depth=9, random_state=42).fit(Xtr_s, y_tr)
reg_report("DT-d5 train", y_tr, dt5.predict(Xtr_s))
reg_report("DT-d9 train", y_tr, dt9.predict(Xtr_s))
print("DT-d5 test R2:", round(r2_score(y_te, dt5.predict(Xte_s)), 4))
print("DT-d9 test R2:", round(r2_score(y_te, dt9.predict(Xte_s)), 4))

# 树不吃 scale：逐列线性变换不改变「谁和谁比较」，预测逐位一致
dt_ns = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr, y_tr)
print("树 no-scale 与 scale 版预测逐位一致:",
      bool((dt_ns.predict(Xtr) == dt5.predict(Xtr_s)).all()))

assert dt.get_depth() == 21 and dt.get_n_leaves() == 640, "640 行被切到 640 片叶子"
assert round(r2_score(y_tr, dt.predict(Xtr_s)), 4) == 1.0, "全深树背下训练段"
assert round(r2_score(y_te, dt.predict(Xte_s)), 4) == 0.9228
assert round(r2_score(y_te, dt9.predict(Xte_s)), 4) == 0.9339, "适度剪枝反而更好"
assert (dt_ns.predict(Xtr) == dt5.predict(Xtr_s)).all()
'''

S5_CODE = '''# 集成：多个树投票，test R2 一路涨
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(Xtr_s, y_tr)
reg_report("RF  train", y_tr, rf.predict(Xtr_s))
rf_r2 = r2_score(y_te, rf.predict(Xte_s))
print("RF test R2:", round(rf_r2, 4))

gbr = GradientBoostingRegressor(random_state=42).fit(Xtr_s, y_tr)
reg_report("GBR train", y_tr, gbr.predict(Xtr_s))
gbr_r2 = r2_score(y_te, gbr.predict(Xte_s))
print("GBR test R2:", round(gbr_r2, 4))

imp = dict(zip(feat, np.round(rf.feature_importances_, 4).tolist()))
print("RF 特征重要性:", imp)
print("重要性前 2:", sorted(imp, key=imp.get, reverse=True)[:2])

assert round(rf_r2, 4) == 0.9529 and round(gbr_r2, 4) == 0.9627
assert imp["户数"] == 0.7587 and imp["容量kVA"] == 0.1903
assert sorted(imp, key=imp.get, reverse=True)[:2] == ["户数", "容量kVA"]
'''

S6_CODE = '''# R2 可以为负：比「无脑猜均值」还差就是负
dummy = np.full(len(y_te), y_tr.mean())
r2_dummy = r2_score(y_te, dummy)
print("预测常数=训练均值:", round(r2_dummy, 4), "← 测试段均值略高，R2 变负")

# RMSE vs MAE：平方让离群残差说了算
res = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
print("残差", res.tolist(),
      "→ RMSE", round(float(np.sqrt(np.mean(res ** 2))), 4),
      "vs MAE", round(float(np.mean(res)), 4))
res2 = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
print("离群残差 100:", round(float(np.sqrt(np.mean(res2 ** 2))), 2),
      "vs", round(float(np.mean(res2)), 2), "← RMSE 被拉爆")

assert round(r2_dummy, 4) == -0.0123, "R2 是可为负的"
assert round(float(np.sqrt(np.mean(res ** 2))), 4) == 4.5607
assert round(float(np.mean(res)), 4) == 2.8
'''

S7_CODE = '''# ID 列陷阱：台区编号当特征，新台区一来就露馅
oh_id = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(reg_tr[["台区编号"]])
print("训练段学到", oh_id.categories_[0].shape[0], "个台区；测试段独有台区:",
      len(set(reg_te["台区编号"]) - set(reg_tr["台区编号"])), "（这次是运气好）")

new_row = pd.DataFrame({"台区编号": ["STATION_NEW_99"]})
out_new = oh_id.transform(new_row)
print("新台区 STATION_NEW_99 独热后:", out_new.shape,
      "| 该行独热和:", float(out_new.sum()), "← 全 0，模型只能瞎猜")

assert out_new.shape == (1, 36)
assert float(out_new.sum()) == 0.0
'''


# =========================================================================== #
# 练习版各节代码（@@todo 挖空标记 + 真实解法）
# =========================================================================== #

E1_CODE = '''# @@todo(1) 分别在标准化矩阵与原始矩阵上 fit LinearRegression
# @@hint LinearRegression().fit(Xtr_s, y_tr) 与 LinearRegression().fit(Xtr, y_tr)
lin = LinearRegression().fit(Xtr_s, y_tr)
lin_raw = LinearRegression().fit(Xtr, y_tr)
# @@end

print("scaled coef_:", np.round(lin.coef_, 4).tolist())
print("scaled intercept:", round(float(lin.intercept_), 4))
print("raw coef_   :", np.round(lin_raw.coef_, 4).tolist())

# @@todo(2) 用脚手架的 reg_report 评估 scaled 版的 train 与 test 表现
# @@hint reg_report("LIN  test ", y_te, lin.predict(Xte_s))
reg_report("LIN  train", y_tr, lin.predict(Xtr_s))
r2_te, rmse_te, mae_te = reg_report("LIN  test ", y_te, lin.predict(Xte_s))
# @@end
reg_report("LIN-raw test", y_te, lin_raw.predict(Xte))

top = feat[int(np.argmax(np.abs(lin.coef_[: len(NUM)])))]
print("数值特征中 |系数| 最大:", top)

assert round(float(lin.intercept_), 4) == 393.0875
assert round(r2_te, 4) == 0.9234 and round(rmse_te, 4) == 48.3232
assert round(float(mae_te), 4) == 37.6304
assert round(r2_score(y_tr, lin.predict(Xtr_s)), 4) == 0.9480
assert round(r2_score(y_te, lin_raw.predict(Xte)), 4) == 0.9234, "R2 对线性缩放不变"
assert not np.allclose(lin.coef_, lin_raw.coef_), "但系数完全不同"
assert top == "容量kVA", "线性系数视角: 容量最强(113.57)"
'''

E2_CODE = '''# @@todo(3) 用 Ridge 拟合 alpha=10 的版本，并评估 test R2
# @@hint Ridge(alpha=...).fit(Xtr_s, y_tr)；r2_score 用 reg_report 也行
rd10 = Ridge(alpha=10.0).fit(Xtr_s, y_tr)
r2_ridge = r2_score(y_te, rd10.predict(Xte_s))
# @@end
print("Ridge a=10 test R2:", round(r2_ridge, 4),
      "| 容量系数:", round(float(rd10.coef_[0]), 3))

# @@todo(4) 依次 fit Lasso alpha=1 与 alpha=10，数一数各保留几个非零系数
# @@hint (la.coef_ != 0).sum()；保留的特征用 feat[i] 对照
la1 = Lasso(alpha=1.0).fit(Xtr_s, y_tr)
la10 = Lasso(alpha=10.0).fit(Xtr_s, y_tr)
nz1, nz10 = int((la1.coef_ != 0).sum()), int((la10.coef_ != 0).sum())
kept10 = {feat[i] for i in range(len(feat)) if la10.coef_[i] != 0}
# @@end
print("Lasso a=1 非零:", nz1, "/9 | a=10 非零:", nz10, "/9 | a=10 保留:", kept10)

assert round(r2_ridge, 4) == 0.9245
assert round(float(rd10.coef_[0]), 3) == 102.09, "Ridge 只收缩不归零"
assert nz1 == 6 and nz10 == 3
assert kept10 == {"容量kVA", "户数", "区域_工业区"}, "a=10 只留最强三路信号"
'''

E3_CODE = '''# @@todo(5) 在「未标准化」的 Xtr 上 fit Lasso(alpha=1.0)，与 scaled 版对比非零系数个数
# @@hint Lasso(alpha=1.0).fit(Xtr, y_tr)；对照 la_s 的非零数
la_s = Lasso(alpha=1.0).fit(Xtr_s, y_tr)
la_r = Lasso(alpha=1.0).fit(Xtr, y_tr)
nz_s, nz_r = int((la_s.coef_ != 0).sum()), int((la_r.coef_ != 0).sum())
# @@end
print("scaled 非零:", nz_s, "| raw 非零:", nz_r)
print("scaled coef:", np.round(la_s.coef_, 3).tolist())
print("raw   coef:", np.round(la_r.coef_, 4).tolist(),
      "← 容量kVA 量纲大(百位)，原始尺度下惩罚相对变小")

assert nz_r == 7
assert round(float(la_r.coef_[0]), 4) == 0.3892, "raw 版容量系数 0.39 vs scaled 版 113"
assert round(float(abs(la_s.coef_[0])), 3) == 113.061, "同一特征，两种尺度差 290 倍"
'''

E4_CODE = '''# @@todo(6) fit 不限深度的决策树，打印深度与叶子数，评估 train/test
# @@hint DecisionTreeRegressor(random_state=42)；dt.get_depth() / dt.get_n_leaves()
dt = DecisionTreeRegressor(random_state=42).fit(Xtr_s, y_tr)
reg_report("DT-all train", y_tr, dt.predict(Xtr_s))
r2_dt = r2_score(y_te, dt.predict(Xte_s))
# @@end
print("全深树: depth =", dt.get_depth(), "| leaves =", dt.get_n_leaves())
print("DT-all test R2:", round(r2_dt, 4))

# @@todo(7) fit max_depth=9 的决策树并评估 test R2，与全深树对照
# @@hint DecisionTreeRegressor(max_depth=..., random_state=42)
dt9 = DecisionTreeRegressor(max_depth=9, random_state=42).fit(Xtr_s, y_tr)
r2_d9 = r2_score(y_te, dt9.predict(Xte_s))
# @@end
print("DT-d9 test R2:", round(r2_d9, 4), "← 适度剪枝反而更好")

# 树不吃 scale：逐列线性变换不改变「谁和谁比较」
dt_ns = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr, y_tr)
dt5 = DecisionTreeRegressor(max_depth=5, random_state=42).fit(Xtr_s, y_tr)
print("树 no-scale 与 scale 版预测逐位一致:",
      bool((dt_ns.predict(Xtr) == dt5.predict(Xtr_s)).all()))

assert dt.get_depth() == 21 and dt.get_n_leaves() == 640, "640 行被切到 640 片叶子"
assert round(r2_score(y_tr, dt.predict(Xtr_s)), 4) == 1.0, "全深树背下训练段"
assert round(r2_dt, 4) == 0.9228
assert round(r2_d9, 4) == 0.9339, "适度剪枝反而更好"
assert (dt_ns.predict(Xtr) == dt5.predict(Xtr_s)).all()
'''

E5_CODE = '''# @@todo(8) fit RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1) 并评估 test
# @@hint rf.predict(Xte_s)；特征重要性在 rf.feature_importances_
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1).fit(Xtr_s, y_tr)
rf_r2 = r2_score(y_te, rf.predict(Xte_s))
# @@end

# @@todo(9) fit GradientBoostingRegressor(random_state=42) 并评估 test
# @@hint 与 RF 同款写法
gbr = GradientBoostingRegressor(random_state=42).fit(Xtr_s, y_tr)
gbr_r2 = r2_score(y_te, gbr.predict(Xte_s))
# @@end

imp = dict(zip(feat, np.round(rf.feature_importances_, 4).tolist()))
print("RF test R2:", round(rf_r2, 4), "| GBR test R2:", round(gbr_r2, 4))
print("RF 特征重要性:", imp)
print("重要性前 2:", sorted(imp, key=imp.get, reverse=True)[:2])

assert round(rf_r2, 4) == 0.9529 and round(gbr_r2, 4) == 0.9627
assert imp["户数"] == 0.7587 and imp["容量kVA"] == 0.1903
assert sorted(imp, key=imp.get, reverse=True)[:2] == ["户数", "容量kVA"]
'''

E6_CODE = '''# @@todo(10) 构造「预测常数=训练均值」的基线，计算它的 test R2
# @@hint np.full(len(y_te), y_tr.mean())；R2 会是负的
dummy = np.full(len(y_te), y_tr.mean())
r2_dummy = r2_score(y_te, dummy)
# @@end
print("预测常数=训练均值:", round(r2_dummy, 4), "← 测试段均值略高，R2 变负")

# @@todo(11) 对残差数组 [1,1,1,1,10] 手算 RMSE 与 MAE，体会平方的放大效应
# @@hint RMSE = sqrt(mean(res**2))；MAE = mean(res)
res = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
rmse_r = float(np.sqrt(np.mean(res ** 2)))
mae_r = float(np.mean(res))
# @@end
print("残差", res.tolist(), "→ RMSE", round(rmse_r, 4), "vs MAE", round(mae_r, 4))

assert round(r2_dummy, 4) == -0.0123, "R2 是可为负的"
assert round(rmse_r, 4) == 4.5607
assert round(mae_r, 4) == 2.8
'''

E7_CODE = '''# @@todo(12) 用 handle_unknown="ignore" 的 OneHotEncoder 编码一个训练段没见过的新台区
# @@hint oh_id.transform(单行 DataFrame)；看该行独热和
oh_id = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(reg_tr[["台区编号"]])
new_row = pd.DataFrame({"台区编号": ["STATION_NEW_99"]})
out_new = oh_id.transform(new_row)
# @@end
print("训练段学到", oh_id.categories_[0].shape[0], "个台区；测试段独有台区:",
      len(set(reg_te["台区编号"]) - set(reg_tr["台区编号"])), "（这次是运气好）")
print("新台区 STATION_NEW_99 独热后:", out_new.shape,
      "| 该行独热和:", float(out_new.sum()), "← 全 0，模型只能瞎猜")

assert out_new.shape == (1, 36)
assert float(out_new.sum()) == 0.0
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch04 回归模型：线性 / Ridge / Lasso / 决策树 / 随机森林 / GBDT

> 方向：机器学习 ｜ 竞赛对应：**模型训练 40%**（回归是训练题的主体）
> + **模型性能评估 10%**（R² / RMSE / MAE 三件套贯穿全章）。

数据换成 `load_reg.csv`（800 行 x 8 列，预测台区「日均负荷」）。
本章六族模型同台：**线性家族**（Linear / Ridge / Lasso）与**树家族**
（决策树 / 随机森林 / GBDT），外加三个必考的评估陷阱。

本章实测出四个反直觉行为（sklearn 1.9.1 / 本数据）：

- **线性回归的 R² 对缩放完全不变**（0.9480 / 0.9234），系数却面目全非
- **Lasso 未 scale 时判 7/9 非零，scale 后判 6/9**——alpha 的「尺度」不同
- **全深决策树 640 行切出 640 片叶子**，train R² = 1.0，但适度剪枝 test 更好
- **预测「训练段均值」的 test R² = −0.0123**——R² 真的可以是负数

> 本章 `assert` 真值全部沉淀在方向 README 的「ch04 专项真值」表里。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 模型训练 40% | 六族模型 fit/predict 按步给分 | 线性 / Ridge / Lasso / DT / RF / GBDT |
| 模型性能评估 10% | 指标选型与解读 | R² / RMSE / MAE、负 R²、系数与重要性解读 |

**学习目标**：同一个回归任务上跑通六族模型并报出 test R²，
能解释「为什么线性模型的系数不可直接比较」「Lasso 归零了谁」「树为什么不吃 scale」。

## 二、API 速查表

| 类 / 函数 | 关键参数 | 一句话说明 |
|---|---|---|
| `LinearRegression` | — | 普通最小二乘；系数受尺度影响 |
| `Ridge` | `alpha` | L2 收缩，只压不删 |
| `Lasso` | `alpha` | L1 归零，自带特征选择；**尺度敏感** |
| `DecisionTreeRegressor` | `max_depth` `random_state` | 单树；不限深=背表 |
| `RandomForestRegressor` | `n_estimators` `random_state` `n_jobs` | Bagging；`feature_importances_` |
| `GradientBoostingRegressor` | `random_state` | Boosting；本数据 test R² 最强 |
| `r2_score` / `mean_squared_error` / `mean_absolute_error` | — | R² / RMSE(sqrt MSE) / MAE |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 `LinearRegression`：R² 不变，系数万变

标准化与否对线性回归的**预测**毫无影响（R² 逐位一致），
但**系数**完全是两套——这直接决定你能不能拿系数讲故事。
"""
    ),
    code(S1_CODE),
    md(
        """
### 3.2 难点深挖：为什么系数必须「标准化后再比」

**为什么难**：未标准化时，容量kVA（量纲百位）系数 0.39、月平均温度系数
-0.71，看似「容量不重要」；标准化后容量系数 113.06——**每 +1 标准差的
负荷变化**才是可比口径。竞赛答辩里拿原始系数排重要性，是经典翻车点。

**正误对照**（同一模型，同一数据）：

| 口径 | 容量kVA 系数 | 户数系数 | test R² |
|---|---|---|---|
| 未标准化 | 0.3892 | 1.4498 | 0.9234 |
| 标准化 | **113.061** | **71.9111** | 0.9234（不变） |

**判定规则**：**「要预测」用什么尺度都行，「要解读」必须标准化后看系数**；
R² 对特征的线性缩放不变，所以它验证不了你有没有踩这个坑。
"""
    ),
    code(S2_CODE),
    md(
        """
## 3.3 `Ridge` 收缩 vs `Lasso` 归零

两个正则化邻居，性格完全不同：Ridge 把系数**等比压缩**（永不归零），
Lasso 直接**判零**（顺带做了特征选择）。alpha 越大下手越重。
"""
    ),
    code(S3_CODE),
    md(
        """
### 3.4 难点深挖：Lasso 的尺度敏感性

**为什么难**：Lasso 的惩罚项对**系数绝对值**求和。同一特征换尺度，
系数跟着换尺度，惩罚的「相对重量」就变了——同一个 `alpha=1.0`，
scaled 版判 6/9 非零，raw 版判 7/9 非零，归零的是**不同的特征**。

**正误对照**（`Lasso(alpha=1.0)`）：

| 输入 | 非零系数 | 容量kVA 系数 |
|---|---|---|
| 标准化矩阵 | 6/9 | 113.061 |
| 原始矩阵 | 7/9 | 0.3892 |

**判定规则**：**Lasso（以及所有带 L1 的弹性网）之前必标准化**；
alpha 的含义只在你固定的尺度里有意义，换尺度要重调 alpha。
Ridge 同理受影响但症状温和（只收缩不归零，选择效应弱）。
"""
    ),
    code(S4_CODE),
    md(
        """
## 3.5 决策树：背表机器与剪枝阀门

不限深度 = 每个训练样本一片叶子（640 行 → 640 叶，train R² = 1.0）。
`max_depth` 是最常用的泛化阀门：本数据 depth=9 比「不限深」test R² 高
（0.9339 vs 0.9228）。另有一个反直觉事实：**树不吃 scale**——
逐列线性变换不改变「谁和谁比较」，预测逐位一致。
"""
    ),
    code(S5_CODE),
    md(
        """
## 3.6 随机森林与 GBDT：集成的上限

RF（Bagging）与 GBR（Boosting）把 test R² 推到 0.9529 / 0.9627，
显著超过单树与线性模型。`feature_importances_` 给出模型视角的重要性：
户数 0.7587 ≫ 容量kVA 0.1903，其余特征都在陪跑。
"""
    ),
    code(S6_CODE),
    md(
        """
### 3.7 难点深挖：R² 可以为负 & RMSE 被离群残差拉爆

**为什么难**：两个评估直觉都靠不住——「R² 最低是 0」（错，可以是负）；
「RMSE 和 MAE 差不多」（错，平方让离群残差主导）。

**错误示范**（实测输出）：

```python
r2_score(y_te, np.full(len(y_te), y_tr.mean()))   # -0.0123 ← 比无脑猜均值还差
残差 [1, 1, 1, 1, 100]: RMSE=44.73 vs MAE=20.80   # 一个离群残差把 RMSE 拉爆
```

**正误对照**：

| 指标 | 残差 [1,1,1,1,10] | 残差 [1,1,1,1,100] |
|---|---|---|
| RMSE | 4.5607 | **44.73** |
| MAE | 2.8000 | 20.80 |

**判定规则**：**R² 为负 = 模型比「预测训练段均值」还差**，先查有没有
真 bug 再谈调参；误差里存在大离群点时，RMSE 会放大它的存在——
想「如实报告平均误差」用 MAE，想「重罚大错」用 RMSE，报告时两个都给。
"""
    ),
    code(S7_CODE),
    md(
        """
---

## 四、易错点清单

1. 线性回归的 R² 对缩放不变——它验证不了系数是否可解读。
2. 原始系数排重要性是经典翻车点：容量 0.39 vs 标准化后 113.06。
3. Ridge 只收缩不归零；想用正则化做特征选择用 Lasso。
4. Lasso 的 alpha 只在固定尺度里有意义：换尺度必须重调。
5. Lasso 之前必标准化，否则归零决策（特征选择）跟着尺度走。
6. 全深决策树把训练段背下来（640 叶、train R²=1.0），max_depth 要显式给。
7. 树不吃 scale：对树做标准化不改变预测，纯属白干（但无害）。
8. RF 的 `feature_importances_` 是模型视角，与相关性视角可能不一致。
9. R² 可以为负（本数据 −0.0123）：比猜均值还差 = 有 bug 先查 bug。
10. RMSE 对离群残差平方放大（44.73 vs MAE 20.80），报告两个都给。
11. ID 列别进特征矩阵：本数据测试段无新台区是运气，新台区一来独热全 0。

## 五、本章小结

- 线性家族：`LinearRegression` 看基线，`Ridge` 抗共线，`Lasso` 顺带选特征——三者都要标准化。
- 树家族：单树必须剪枝，RF/GBR 提上限（0.9529 / 0.9627）；树不吃 scale。
- 评估三件套 R² / RMSE / MAE：R² 可负、RMSE 放大离群、两个误差指标都报。
- 系数与重要性都是「模型视角」，解读前先问自己：尺度统一了吗？

### 复盘提问

1. 为什么同一份数据两种尺度的线性回归 R² 一模一样、系数差 290 倍？
2. Ridge 和 Lasso 的本质区别是什么？想做特征选择选哪个？
3. Lasso 未 scale 时归零模式为什么不可信？
4. 全深树的 train R²=1.0 说明什么？本数据 max_depth=9 为什么反而更好？
5. R² 什么时候是负的？RMSE 和 MAE 差距拉大说明什么？

答不上来的，回讲解版看对应难点深挖，再到 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch04 练习：回归模型

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 模型训练 40% | 线性 / Ridge / Lasso / DT / RF / GBDT 的 fit-predict |
| 模型性能评估 10% | R² / RMSE / MAE 与负 R² 陷阱 |

## 二、API 速查

| 类 | 关键参数 | 一句话 |
|---|---|---|
| `LinearRegression` | — | 最小二乘基线 |
| `Ridge(alpha)` | L2 收缩 | 只压不删 |
| `Lasso(alpha)` | L1 归零 | 尺度敏感，先标准化 |
| `DecisionTreeRegressor(max_depth)` | 剪枝阀门 | 不限深=背表 |
| `RandomForestRegressor(n_estimators)` | Bagging | importance 可读 |
| `GradientBoostingRegressor` | Boosting | 本数据最强 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：线性回归与两种尺度（对应讲解 §3.1-3.2）"),
    code(E1_CODE),
    md(
        """
**为什么这样做**：scaled 与 raw 各 fit 一次不是浪费——R² 的一致性证明
「缩放不影响预测」，系数的差异证明「缩放影响解读」。两者合起来才是完整结论。
"""
    ),
    md("## 题 2：Ridge 收缩与 Lasso 归零（对应讲解 §3.3）"),
    code(E2_CODE),
    md("## 题 3：Lasso 的尺度敏感性（对应讲解 §3.4）"),
    code(E3_CODE),
    md("## 题 4：决策树的背表与剪枝（对应讲解 §3.5）"),
    code(E4_CODE),
    md("## 题 5：随机森林与 GBDT（对应讲解 §3.6）"),
    code(E5_CODE),
    md("## 题 6：负 R² 与 RMSE 的放大效应（对应讲解 §3.7）"),
    code(E6_CODE),
    md("## 题 7：ID 列陷阱（对应讲解 §3.7）"),
    code(E7_CODE),
    md(
        """
---

## 综合自查

1. 题 1 里 R² 一致但系数差 290 倍，各自说明什么？
2. 题 2 的 Lasso a=10 只留三路信号——这三路为什么是它们？
3. 题 3 的 raw 版 Lasso 归零决策为什么不可信？
4. 题 4 全深树 train R²=1.0 而 test 0.9228；d9 反而 0.9339——剪枝在 trade 什么？
5. 题 6 的 −0.0123 是怎么来的？评委问「R2 最低多少」你怎么答？

全部答得上来，进入 ch05（分类模型）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
