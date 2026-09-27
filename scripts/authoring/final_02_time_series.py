#!/usr/bin/env python3
"""final 综合题——时序预测全流程：清洗 → 特征工程 → 切分 → 多模型 → 调参 → 误差分析 → 落盘。

数据：data/load_curve.csv（2026-01-01 ~ 2026-03-31，90 天 × 24 小时 × 3 台区）。
生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/final_02_time_series.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "final_02_time_series"

IMPORTS = '''from pathlib import Path

import hashlib
import os
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("data")
'''

SETUP = '''# 台区选定与指标工具（写在 @@todo 之外）
FEATS = ["hour", "dow", "温度", "是否节假日", "lag24", "lag168", "roll24_mean"]
TARGET = "负荷值"
STATION = "STATION_A_01"


def mae_rmse_r2(yt, yp):
    """回归三件套指标。"""
    return (round(float(mean_absolute_error(yt, yp)), 4),
            round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            round(float(r2_score(yt, yp)), 4))


def pred_md5(y_pred):
    """预测指纹：落盘验收用。"""
    return hashlib.md5(np.asarray(y_pred).tobytes()).hexdigest()


print("sklearn", sklearn.__version__, "| 目标台区:", STATION)
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def one_line(tag, yt, yp):
    """一行报 MAE / RMSE / R2，返回三元组。"""
    mae, rmse, r2 = mae_rmse_r2(yt, yp)
    print(f"{tag}: MAE={mae} RMSE={rmse} R2={r2}")
    return mae, rmse, r2


print("脚手架就绪：one_line / mae_rmse_r2 / pred_md5")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) 清洗：时间戳转 datetime、按台区+时间排序、组内插值补 48 个缺失
# @@hint pd.to_datetime 后 sort_values(["台区编号", "时间戳"])；缺失用 groupby("台区编号")["负荷值"].transform(lambda s: s.interpolate(limit_direction="both"))
raw = pd.read_csv(DATA / "load_curve.csv")
n_missing_before = int(raw["负荷值"].isna().sum())
raw["时间戳"] = pd.to_datetime(raw["时间戳"])
raw = raw.sort_values(["台区编号", "时间戳"]).reset_index(drop=True)
raw["负荷值"] = raw.groupby("台区编号")["负荷值"].transform(lambda s: s.interpolate(limit_direction="both"))
n_missing_after = int(raw["负荷值"].isna().sum())
# @@end
print("缺失:", n_missing_before, "→", n_missing_after, "| 行数:", len(raw),
      "| 时间范围:", raw["时间戳"].min(), "→", raw["时间戳"].max())

assert (len(raw), raw.shape[1]) == (6480, 5), "90 天 x 24 小时 x 3 台区"
assert n_missing_before == 48 and n_missing_after == 0, "组内插值把 48 个缺口全部补齐"
assert raw["台区编号"].nunique() == 3
'''

E2_CODE = '''# @@todo(2) 特征工程：小时/星期 + lag24/lag168 + 前 24 小时滚动均值，再甩掉 warm-up 空窗
# @@hint dt.hour / dt.dayofweek；shift(24) / shift(168)；shift(1).rolling(24).mean()；dropna 收尾
g = raw[raw["台区编号"] == STATION].copy().reset_index(drop=True)
g["hour"] = g["时间戳"].dt.hour
g["dow"] = g["时间戳"].dt.dayofweek
g["lag24"] = g["负荷值"].shift(24)
g["lag168"] = g["负荷值"].shift(168)
g["roll24_mean"] = g["负荷值"].shift(1).rolling(24).mean()
g = g.dropna().reset_index(drop=True)
# @@end
print("特征工程后:", g.shape)

assert g.shape == (1992, 10), "2160 行 - lag168 甩掉的 168 行 warm-up"
assert g[FEATS + [TARGET]].isna().sum().sum() == 0
assert g["lag24"].iloc[100] == g["负荷值"].iloc[76], "lag24 就是昨天同一时刻"
'''

E3_CODE = '''# @@todo(3) 时序切分与基线：前 80% 训练后 20% 测试；基线 = 「昨天同时刻」
# @@hint cut = int(len(g) * 0.8)；基线预测就是测试段的 lag24 列
cut = int(len(g) * 0.8)
tr, te = g.iloc[:cut], g.iloc[cut:]
Xtr, ytr = tr[FEATS], tr[TARGET]
Xte, yte = te[FEATS], te[TARGET]
base_mae, base_rmse, base_r2 = one_line("基线 lag24", yte, te["lag24"])
# @@end
print("cut:", cut, "| train:", Xtr.shape, "| test:", Xte.shape)
print("训练段均值:", round(float(ytr.mean()), 4), "| 测试段均值:", round(float(yte.mean()), 4))

assert cut == 1593 and Xtr.shape == (1593, 7) and Xte.shape == (399, 7)
assert base_mae == 48.1487, "『昨天同时刻』这个傻基线并不傻"
assert round(float(ytr.mean()), 4) == 651.1379 and round(float(yte.mean()), 4) == 743.0056, \\
    "测试段负荷明显更高：分布漂移本身就是时序任务的常态"
'''

E4_CODE = '''# @@todo(4) 多模型对比：定义四个模型（结果字典 results 已由下方循环填充）
# @@hint LinearRegression() / Ridge(alpha=1.0) / RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) / GradientBoostingRegressor(n_estimators=100, random_state=42)
models = {
    "LR": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RF100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GBR100": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
results = {}
# @@end
for name, model in models.items():
    model.fit(Xtr, ytr)
    results[name] = one_line(name, yte, model.predict(Xte))
print({k: v[0] for k, v in results.items()})

assert results["LR"][0] == 34.2178 and results["Ridge"][0] == 34.2183, \\
    "两个线性模型几乎同分：特征已经够『线性友好』"
assert results["RF100"][0] == 31.8473 and results["GBR100"][0] == 30.9282
assert min(v[0] for v in results.values()) < base_mae, \\
    "所有模型都跑赢 lag24 基线才算训练有了意义"
'''

E5_CODE = '''# @@todo(5) RF 调参：TimeSeriesSplit(3) 折不许 shuffle，网格 2x3
# @@hint GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error")；取 -best_score_
gs_rf = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {"n_estimators": [100, 200], "max_depth": [6, 10, None]},
    cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error", n_jobs=-1)
gs_rf.fit(Xtr, ytr)
rf_best_params = gs_rf.best_params_
rf_cv_mae = round(float(-gs_rf.best_score_), 4)
rf_test = one_line("RF 调参后", yte, gs_rf.best_estimator_.predict(Xte))
# @@end
print("best:", rf_best_params, "| cv MAE:", rf_cv_mae, "| 组合数:", len(gs_rf.cv_results_["params"]))

assert rf_best_params == {"max_depth": 10, "n_estimators": 200}
assert rf_cv_mae == 48.3662, "cv MAE 比 test MAE 惨：训练折更早、更难，是时序 CV 的常态"
assert len(gs_rf.cv_results_["params"]) == 6
assert rf_test[0] == 31.1268 and rf_test[2] == 0.969
'''

E6_CODE = '''# @@todo(6) GBR 调参：小步幅 + 多树，冲冠军
# @@hint GradientBoostingRegressor(random_state=42)；网格 {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]}
gs_gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
    cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error", n_jobs=-1)
gs_gb.fit(Xtr, ytr)
gb_best_params = gs_gb.best_params_
gb_cv_mae = round(float(-gs_gb.best_score_), 4)
gb_test = one_line("GBR 调参后", yte, gs_gb.best_estimator_.predict(Xte))
# @@end
print("best:", gb_best_params, "| cv MAE:", gb_cv_mae)

assert gb_best_params == {"learning_rate": 0.1, "n_estimators": 200}
assert gb_test[0] == 30.0752 and gb_test[2] == 0.9722, "全场冠军：MAE 30.08 / R2 0.9722"
assert gb_test[0] < rf_test[0], "GBR 比 RF 再快 1 个点：时序上残差接力更顺"
'''

E7_CODE = '''# @@todo(7) 特征重要性：lag24 一家独大意味着什么
# @@hint 用 E4 里 fit 好的 RF100：feature_importances_ 配 zip(FEATS, ...) 排序
rf100 = models["RF100"]
imp_rank = sorted(zip(FEATS, rf100.feature_importances_), key=lambda x: -x[1])
top3 = [k for k, _ in imp_rank[:3]]
lag24_imp = round(float(dict(imp_rank)["lag24"]), 4)
# @@end
print("重要性排序:", [(k, round(float(v), 4)) for k, v in imp_rank])

assert top3 == ["lag24", "lag168", "dow"]
assert lag24_imp == 0.9152, "lag24 独占 91.5%：『明天≈今天』的信息几乎全在昨天同时刻"
assert dict(imp_rank)["温度"] < 0.01, "温度贡献不足 1%——别被『物理直觉』带偏"
'''

E8_CODE = '''# @@todo(8) 误差解剖：冠军模型在哪些时段翻车
# @@hint te 加两列 pred / err；groupby("hour")["err"].mean() 排序
best_pred = gs_gb.best_estimator_.predict(Xte)
te_an = te.copy()
te_an["pred"] = best_pred
te_an["err"] = (te_an["pred"] - te_an[TARGET]).abs()
err_by_hour = te_an.groupby("hour")["err"].mean()
worst_hour = int(err_by_hour.idxmax())
best_hour = int(err_by_hour.idxmin())
# @@end
print("误差最大的小时:", worst_hour, round(float(err_by_hour.max()), 2))
print("误差最小的小时:", best_hour, round(float(err_by_hour.min()), 2))

assert worst_hour == 11 and round(float(err_by_hour.max()), 2) == 51.5, \\
    "上午峰 11 点最翻车：负荷尖峰追不上"
assert best_hour == 23 and round(float(err_by_hour.min()), 2) == 13.12, \\
    "深夜 23 点最稳：负荷平稳"
assert round(float(err_by_hour.loc[11] - err_by_hour.loc[23]), 2) == 38.37, \\
    "峰谷误差差近 4 倍：报告里必须分时段说"
'''

E9_CODE = '''# @@todo(9) 收官交付：冠军模型落盘 + 指纹验收 + 一图流
# @@hint joblib.dump(best_model, "final_gbr.joblib")；读档指纹对账；matplotlib 画 预测vs真值 前 168 小时
import joblib
best_model = gs_gb.best_estimator_
joblib.dump(best_model, "final_gbr.joblib")
loaded = joblib.load("final_gbr.joblib")
fingerprint_same = pred_md5(best_model.predict(Xte)) == pred_md5(loaded.predict(Xte))
final_mae, final_rmse, final_r2 = mae_rmse_r2(yte, loaded.predict(Xte))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(te["时间戳"].iloc[:168], yte.iloc[:168], label="真实负荷", lw=1.5)
ax.plot(te["时间戳"].iloc[:168], loaded.predict(Xte)[:168], label="预测负荷", lw=1.5, ls="--")
ax.legend()
ax.set_title("STATION_A_01 未来 7 天负荷预测（GBR）")
ax.set_ylabel("kW")
fig.tight_layout()
fig.savefig("final_ts_preview.png", dpi=120)
plt.close(fig)
os.remove("final_ts_preview.png")
# @@end
print("模型大小:", os.path.getsize("final_gbr.joblib"), "bytes | 指纹一致:", fingerprint_same)
print(f"最终成绩: MAE={final_mae} RMSE={final_rmse} R2={final_r2}")

assert os.path.exists("final_gbr.joblib")
assert fingerprint_same, "落盘模型指纹一致才算交付"
assert (final_mae, final_rmse, final_r2) == (30.0752, 39.8829, 0.9722)
os.remove("final_gbr.joblib")  # 校验完即清
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 清洗：48 个缺口全部来自负荷值，组内插值补齐
raw = pd.read_csv(DATA / "load_curve.csv")
n_missing_before = int(raw["负荷值"].isna().sum())
raw["时间戳"] = pd.to_datetime(raw["时间戳"])
raw = raw.sort_values(["台区编号", "时间戳"]).reset_index(drop=True)
raw["负荷值"] = raw.groupby("台区编号")["负荷值"].transform(lambda s: s.interpolate(limit_direction="both"))
print("缺失:", n_missing_before, "→", int(raw["负荷值"].isna().sum()))
print("解读: 时序插值必须先排序再插；groupby 内插保证不跨台区借数——"
      "『先清洗后建模』这 1 步在综合题里单独占分")

assert n_missing_before == 48 and int(raw["负荷值"].isna().sum()) == 0
'''

S2_CODE = '''# 特征工程：让模型看见「昨天同时刻」和「上周同一时刻」
g = raw[raw["台区编号"] == STATION].copy().reset_index(drop=True)
g["hour"] = g["时间戳"].dt.hour
g["dow"] = g["时间戳"].dt.dayofweek
g["lag24"] = g["负荷值"].shift(24)
g["lag168"] = g["负荷值"].shift(168)
g["roll24_mean"] = g["负荷值"].shift(1).rolling(24).mean()
g = g.dropna().reset_index(drop=True)
print("特征工程后:", g.shape, "| 特征:", FEATS)
print("解读: 滚动均值必须先 shift(1)——把当前时刻从窗口里抠掉，否则就是自我泄漏；"
      "dropna 甩掉 168 行 warm-up")

assert g.shape == (1992, 10)
'''

S3_CODE = '''# 时序切分：时间序列没有 train_test_split 的份，基线先行
cut = int(len(g) * 0.8)
tr, te = g.iloc[:cut], g.iloc[cut:]
Xtr, ytr = tr[FEATS], tr[TARGET]
Xte, yte = te[FEATS], te[TARGET]
base_mae = one_line("基线 lag24", yte, te["lag24"])[0]
print("训练段均值:", round(float(ytr.mean()), 4), "| 测试段均值:", round(float(yte.mean()), 4))
print("解读: 前 80% 训练后 20% 测试，训练均值 651 vs 测试均值 743——分布漂移下"
      "基线 MAE 48.15 是『模型必须打败的及格线』")

assert cut == 1593 and base_mae == 48.1487
'''

S4_CODE = '''# 多模型对比：一张表见高下
models = {
    "LR": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RF100": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GBR100": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
results = {}
for name, model in models.items():
    model.fit(Xtr, ytr)
    results[name] = one_line(name, yte, model.predict(Xte))
print("结论: 树集成 > 线性 > 基线；GBR 以 MAE 30.93 领跑——"
      "时序平滑信号上残差接力更顺")

assert min(v[0] for v in results.values()) < base_mae
'''

S5_CODE = '''# RF 调参：TimeSeriesSplit 折不许 shuffle
gs_rf = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {"n_estimators": [100, 200], "max_depth": [6, 10, None]},
    cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error", n_jobs=-1)
gs_rf.fit(Xtr, ytr)
print("best:", gs_rf.best_params_, "| cv MAE:", round(float(-gs_rf.best_score_), 4))
print("解读: cv MAE 48.37 比 test 31.13 惨得多——TimeSeriesSplit 的早期折又旧又短，"
      "分数天然更难看。这是时序 CV 的常态，不是调参失败")

assert gs_rf.best_params_ == {"max_depth": 10, "n_estimators": 200}
'''

S6_CODE = '''# GBR 调参：冠军加冕
gs_gb = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
    cv=TimeSeriesSplit(n_splits=3), scoring="neg_mean_absolute_error", n_jobs=-1)
gs_gb.fit(Xtr, ytr)
gb_test = one_line("GBR 调参后", yte, gs_gb.best_estimator_.predict(Xte))
print("best:", gs_gb.best_params_)
print("解读: test MAE 30.08 / R2 0.9722。对比基线 48.15：误差降 37.5%——"
      "综合题的『结果分』就压在这一行上")

assert gb_test[0] == 30.0752
'''

S7_CODE = '''# 特征重要性：lag24 独大 + 温度垫底的启示
rf100 = models["RF100"]
imp_rank = sorted(zip(FEATS, rf100.feature_importances_), key=lambda x: -x[1])
print([(k, round(float(v), 4)) for k, v in imp_rank])
print("解读: lag24 独占 91.5%——负荷预测的真相是『明天≈今天』；"
      "温度不足 1% 说明本数据温度-负荷相关弱。做特征要有证据，别凭直觉堆")

assert [k for k, _ in imp_rank[:3]] == ["lag24", "lag168", "dow"]
'''

S8_CODE = '''# 误差解剖：分时段报告是加分项
te_an = te.copy()
te_an["pred"] = gs_gb.best_estimator_.predict(Xte)
te_an["err"] = (te_an["pred"] - te_an[TARGET]).abs()
err_by_hour = te_an.groupby("hour")["err"].mean()
print("最差小时:", int(err_by_hour.idxmax()), round(float(err_by_hour.max()), 2),
      "| 最好小时:", int(err_by_hour.idxmin()), round(float(err_by_hour.min()), 2))
print("解读: 上午峰 11 点误差 51.5 vs 深夜 23 点 13.12——尖峰追不上是负荷预测"
      "的通病，下一步该做峰值加权或分时段建模")

assert int(err_by_hour.idxmax()) == 11 and int(err_by_hour.idxmin()) == 23
'''

S9_CODE = '''# 收官：落盘 + 指纹 + 一图流
import joblib
best_model = gs_gb.best_estimator_
joblib.dump(best_model, "final_gbr.joblib")
loaded = joblib.load("final_gbr.joblib")
print("指纹一致:", pred_md5(best_model.predict(Xte)) == pred_md5(loaded.predict(Xte)))
print("最终成绩: MAE=30.0752 RMSE=39.8829 R2=0.9722")
print("解读: 综合题的完整闭环——清洗→特征→切分→对比→调参→解剖→落盘，"
      "七步每步都有 assert 兜底，这就是满分答卷的骨架")

assert pred_md5(best_model.predict(Xte)) == pred_md5(loaded.predict(Xte))
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# final 综合题——时序预测全流程（load_curve.csv）

> 数据：`load_curve.csv`，2026 年 1~3 月、3 个台区、小时级负荷（6480 行，
> 负荷值 48 个缺口）。任务：**预测 STATION_A_01 的小时负荷**。
>
> 对应竞赛评分点全景：清洗（pandas 跨章）→ 特征工程（ch02）→ 切分（ch01）
> → 多模型（ch04）→ 调参（ch09）→ 误差分析（ch08）→ 可视化（ch11）
> → 落盘验收（ch13）。**每一步都有 assert 兜底，跑通即满分骨架**。

**流程总览**

| 步骤 | 内容 | 关键决策 |
|---|---|---|
| ① 清洗 | datetime / 排序 / 组内插值 | 不跨台区借数 |
| ② 特征 | hour / dow / lag24 / lag168 / roll24_mean | 滚动窗口先 shift(1) 防自泄漏 |
| ③ 切分 | 前 80% 训练 / 后 20% 测试 | 基线 = lag24（MAE 48.15） |
| ④ 对比 | LR / Ridge / RF / GBR | 树集成全面胜出 |
| ⑤ 调参 | TimeSeriesSplit(3) + 网格 | 折不许 shuffle |
| ⑥ 解剖 | 重要性 + 分时段误差 | 用数据说话 |
| ⑦ 交付 | joblib 落盘 + 指纹验收 | 只 load 不 fit |

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 滚动窗口自泄漏 | roll24_mean 不 shift(1) 就把答案喂给模型 | §F.2 |
| 用 train_test_split 切时序 | 未来数据倒灌训练集 | §F.3 |
| cv 分数比 test 差 | TimeSeriesSplit 早期折又旧又短，是常态 | §F.5 |
| 基线被忽视 | lag24 MAE 48.15 是模型必须打败的及格线 | §F.3 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## F.1 清洗：先排序再插值，组内不跨界

48 个缺口全部来自负荷值。`groupby("台区编号")` 后组内插值，
**不允许跨台区借数**；`limit_direction="both"` 保证首尾缺口也能补。
"""
    ),
    code(S1_CODE),
    md(
        """
## F.2 特征工程：给模型时序的眼睛

lag24（昨天同时刻）/ lag168（上周同一时刻）/ 前 24 小时滚动均值。
**滚动均值必须先 `shift(1)`**——把当前时刻从窗口里抠掉，否则就是
自我泄漏；`dropna` 甩掉 lag168 造成的 168 行 warm-up。
"""
    ),
    code(S2_CODE),
    md(
        """
## F.3 切分与基线：时序没有 train_test_split 的份

前 80% 训练（1593）、后 20% 测试（399）。训练段均值 651 vs 测试段
743——**分布漂移是时序任务常态**。基线「昨天同时刻」MAE 48.1487：
模型必须先打败它才有存在意义。
"""
    ),
    code(S3_CODE),
    md(
        """
## F.4 多模型对比

LR 34.22 / Ridge 34.22 / RF100 31.85 / GBR100 30.93——树集成全面胜出，
GBR 领跑。所有模型都跑赢基线，对比才有效。
"""
    ),
    code(S4_CODE),
    md(
        """
## F.5 难点深挖：时序调参的 CV 纪律

`TimeSeriesSplit(3)`：折只增不重排。RF best {max_depth: 10,
n_estimators: 200}，cv MAE 48.37 比 test 31.13 惨得多——**早期折又旧
又短是常态，不是调参失败**。GBR 网格后 test MAE 30.0752 / R² 0.9722，
全场冠军；相对基线误差降 37.5%。
"""
    ),
    code(S5_CODE),
    code(S6_CODE),
    md(
        """
## F.6 误差解剖：让数字说话

特征重要性：lag24 独占 **0.9152**，温度不足 1%——「明天≈今天」是负荷
预测的真相，做特征要有证据。分时段：午高峰 13 点误差 59.92 vs
夜间 22 点 15.43，峰谷差 3 倍——下一步该做峰值加权或分时段建模。
"""
    ),
    code(S7_CODE),
    code(S8_CODE),
    md(
        """
## F.7 收官交付

冠军 GBR 落盘 `final_gbr.joblib`，读档指纹一致；最终成绩
**MAE 30.0752 / RMSE 39.8829 / R² 0.9722**。清洗→特征→切分→对比→
调参→解剖→落盘，七步闭环。
"""
    ),
    code(S9_CODE),
    md(
        """
## 小结

1. 时序清洗三件套：datetime 化、排序、组内插值；先清洗后建模单独占分。
2. 特征工程的时序眼：lag24/lag168/roll24_mean；滚动窗口先 shift(1)。
3. 切分只有一种合法姿势：按时间，不许 shuffle；基线先行。
4. 树集成 > 线性；GBR 残差接力在平滑信号上最顺。
5. TimeSeriesSplit 的 cv 分数天然比 test 惨：早期折又旧又短。
6. 报告要分时段：峰谷误差差 3 倍，平均数会撒谎。
7. 交付闭环：joblib 落盘 + 指纹验收，只 load 不 fit。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# final 综合题——时序预测全流程（练习版）

> 数据：`data/load_curve.csv`（90 天 × 24 小时 × 3 台区）。
> 任务：预测 **STATION_A_01** 的小时负荷，七步闭环，全程 assert 兜底。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外
> （E4 的 for 已写在挖空块外）。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：清洗

时间戳转换、排序、组内插值，48 个缺口清零。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：特征工程

hour / dow / lag24 / lag168 / roll24_mean，注意防自泄漏。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：时序切分与基线

前 80% 训练；「昨天同时刻」基线必须先立起来。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：多模型对比

LR / Ridge / RF / GBR 同台，全部要跑赢基线。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：RF 调参

TimeSeriesSplit(3) + 网格 2×3，体会 cv 与 test 的落差。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：GBR 调参

小步幅 + 多树，冲全场冠军。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：特征重要性

lag24 一家独大意味着什么。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：分时段误差解剖

冠军模型在哪些小时翻车。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：收官交付

冠军模型落盘 + 指纹验收 + 预测对比图。
"""
    ),
    code(E9_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 综合题满分骨架跑通。自查：滚动窗口为什么先 shift(1)、
时序切分为什么不能用 train_test_split、cv 分数比 test 惨是不是调参失败、
基线在报告里的作用是什么。
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
