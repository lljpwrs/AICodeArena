#!/usr/bin/env python3
"""ch09 交叉验证与调参：KFold/StratifiedKFold/GridSearchCV/RandomizedSearchCV（defect_ml.csv）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk09_tuning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch09_tuning_cv"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from scipy.stats import randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, KFold,
                                     RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data")
'''

SETUP = '''# 数据与管道口径沿用 ch01/ch05：NUM 中位数填充+标准化，CAT 独热
dev = pd.read_csv(DATA / "defect_ml.csv")
NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
X, y = dev[NUM + CAT], dev["是否危急"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
pipe = Pipeline([("pre", pre),
                 ("clf", LogisticRegression(max_iter=1000, random_state=42))])

print("sklearn", sklearn.__version__, "| 数据:", X.shape, "| 正例:", int(y.sum()))
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def cv_summary(tag, scores):
    """一行输出 5 折均值/标准差。"""
    print(f"{tag}: mean={float(np.mean(scores)):.4f} std={float(np.std(scores)):.4f}")
    return float(np.mean(scores)), float(np.std(scores))


print("脚手架就绪：cv_summary")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) 各折正例数对账：KFold 与 StratifiedKFold 各 5 折
# @@hint kf.split(X) 只吃 X；skf.split(X, y) 还要 y；折内 y.iloc[te].sum()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf_pos = [int(y.iloc[te].sum()) for tr, te in kf.split(X)]
skf_pos = [int(y.iloc[te].sum()) for tr, te in skf.split(X, y)]
# @@end
print("KFold 每折正例:", kf_pos)
print("StratifiedKFold 每折正例:", skf_pos, "| 总正例:", int(y.sum()))

assert kf_pos == [20, 21, 34, 27, 26], "普通 KFold 正例分布抖动 ±7"
assert skf_pos == [25, 25, 26, 26, 26], "分层后每折正例几乎一致"
assert sum(skf_pos) == 128
'''

E2_CODE = '''# @@todo(2) 用分层 5 折对管道跑 cross_val_score（roc_auc）
# @@hint cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
sc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
# @@end
mean_auc, std_auc = cv_summary("cv AUC", sc)
print("各折:", np.round(sc, 4).tolist())

assert np.round(sc, 4).tolist() == [0.7676, 0.736, 0.7537, 0.7754, 0.84]
assert round(mean_auc, 4) == 0.7745 and round(std_auc, 4) == 0.0354
'''

E3_CODE = '''# @@todo(3) 多指标 + 训练分：cross_validate 同时要 roc_auc/f1 和 train 分数
# @@hint cross_validate(..., scoring=["roc_auc", "f1"], return_train_score=True)
cvr = cross_validate(pipe, X, y, cv=skf, scoring=["roc_auc", "f1"],
                     return_train_score=True)
# @@end
print("keys:", sorted(cvr.keys()))
print("test AUC:", np.round(cvr["test_roc_auc"], 4).tolist())
print("train AUC:", np.round(cvr["train_roc_auc"], 4).tolist(), "| test f1:",
      np.round(cvr["test_f1"], 4).tolist())

assert round(float(np.mean(cvr["test_roc_auc"])), 4) == 0.7745
assert round(float(np.mean(cvr["train_roc_auc"])), 4) == 0.8035, "train 略高于 test，正常"
assert round(float(np.mean(cvr["test_f1"])), 4) == 0.418, "f1 受阈值与不平衡双重影响"
'''

E4_CODE = '''# @@todo(4) 定义 4x2 网格（clf__C × clf__class_weight）并跑 GridSearchCV
# @@hint 参数名是 管道步骤名__参数名；GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc")
param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}
gs = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1,
                  return_train_score=True)
gs.fit(X, y)
# @@end
print("best params:", gs.best_params_)
print("best cv AUC:", round(float(gs.best_score_), 4), "| 组合数:", len(gs.cv_results_["params"]))

assert gs.best_params_ == {"clf__C": 10.0, "clf__class_weight": None}
assert round(float(gs.best_score_), 4) == 0.7748
assert len(gs.cv_results_["params"]) == 8, "4 x 2 = 8 组合，每组合 5 折 = 40 次拟合"
'''

E5_CODE = '''# @@todo(5) 解剖 cv_results_：rank、best_index、最优组合的 train/test 差
# @@hint gs.best_index_；mean_test_score / mean_train_score 都是数组
best_idx = gs.best_index_
rank = gs.cv_results_["rank_test_score"].tolist()
best_test = float(gs.cv_results_["mean_test_score"][best_idx])
best_train = float(gs.cv_results_["mean_train_score"][best_idx])
# @@end
print("rank:", rank)
print(f"best_index={best_idx} | train={best_train:.4f} vs test={best_test:.4f}")

assert rank == [8, 7, 4, 3, 2, 5, 1, 6]
assert best_idx == 6 and rank[best_idx] == 1
assert round(best_train, 4) == 0.8039 and round(best_test, 4) == 0.7748
assert round(best_train - best_test, 4) == 0.0291, "gap 不到 0.03：线性模型没过拟合"
'''

E6_CODE = '''# @@todo(6) 两阶段搜索：粗网格锁量级，细网格精修（贝叶斯优化的穷替身）
# @@hint 第一段粗网格已给；第二段把 C 网格压到 [3, 10, 30, 100]
gs_coarse = GridSearchCV(pipe, {"clf__C": [0.1, 1.0, 10.0]}, cv=skf,
                         scoring="roc_auc", n_jobs=-1)
gs_coarse.fit(X, y)
coarse_best = gs_coarse.best_params_["clf__C"]
param_fine = {"clf__C": [coarse_best / 3, coarse_best, coarse_best * 3, coarse_best * 10]}
gs_fine = GridSearchCV(pipe, param_fine, cv=skf, scoring="roc_auc", n_jobs=-1)
gs_fine.fit(X, y)
# @@end
print("粗网格 best C =", coarse_best, "→ 细网格:", param_fine["clf__C"])
print("细网格 best:", gs_fine.best_params_, round(float(gs_fine.best_score_), 4))

assert coarse_best == 10.0
assert abs(gs_fine.best_params_["clf__C"] - 10.0 / 3) < 1e-9, "细网格把最优推向 cb/3"
assert round(float(gs_fine.best_score_), 4) == 0.7749, "细网格比粗网格再挤 0.0001"
'''

E7_CODE = '''# @@todo(7) RandomizedSearchCV：连续分布上抽 10 组，看性价比
# @@hint uniform(0.01, 100) 表示 [0.01, 100.01) 均匀；RandomizedSearchCV(..., n_iter=10, random_state=42)
param_dist = {"clf__C": uniform(0.01, 100), "clf__class_weight": [None, "balanced"]}
rs = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=skf, scoring="roc_auc",
                        random_state=42, n_jobs=-1)
rs.fit(X, y)
# @@end
c_draws = [round(float(d["clf__C"]), 3) for d in rs.cv_results_["params"]]
print("抽样 C:", c_draws)
print("best:", {k: (round(float(v), 4) if isinstance(v, float) else v)
                for k, v in rs.best_params_.items()},
      round(float(rs.best_score_), 4))

assert c_draws == [37.464, 18.353, 59.876, 44.593, 5.818, 33.381, 70.817, 5.651, 83.254, 0.088]
assert rs.best_params_["clf__class_weight"] is None
assert round(float(rs.best_score_), 4) == 0.7749, "10 次随机 ≈ 8 组网格的成绩"
'''

E8_CODE = '''# @@todo(8) 树系网格 + refit 语义：best_estimator_ 已在全量数据上重新拟合
# @@hint Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42))])；网格 2x3
gs_rf = GridSearchCV(
    Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42))]),
    {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, None]},
    cv=skf, scoring="f1", n_jobs=-1)
gs_rf.fit(X, y)
rf_best_depth = gs_rf.best_params_["clf__max_depth"]
rf_best_score = float(gs_rf.best_score_)
# @@end
print("RF best:", gs_rf.best_params_, "| best f1:", round(rf_best_score, 4))
print("n_splits:", gs_rf.n_splits_, "| refit 耗时:", round(float(gs_rf.refit_time_), 2), "s")
print("best_estimator_ 可直接 predict:", gs_rf.best_estimator_.classes_.tolist())

assert gs_rf.best_params_ == {"clf__max_depth": None, "clf__n_estimators": 200}
assert round(rf_best_score, 4) == 0.296, "f1 在 21% 正例上依然很难看——选对 scoring 很重要"
assert gs_rf.n_splits_ == 5
assert gs_rf.best_estimator_.classes_.tolist() == [0, 1]
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 折结构对账：普通 KFold 的正例分布抖动，分层后纹丝不动
kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kf_pos = [int(y.iloc[te].sum()) for tr, te in kf.split(X)]
skf_pos = [int(y.iloc[te].sum()) for tr, te in skf.split(X, y)]
print("KFold 每折正例:", kf_pos)
print("StratifiedKFold 每折正例:", skf_pos, "| 总正例:", int(y.sum()))

assert kf_pos == [20, 21, 34, 27, 26], "普通 KFold 正例分布抖动 ±7"
assert skf_pos == [25, 25, 26, 26, 26], "分层后每折正例几乎一致"
'''

S2_CODE = '''# cross_val_score：单指标快速体检
sc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
mean_auc, std_auc = cv_summary("cv AUC", sc)
print("各折:", np.round(sc, 4).tolist())
print("解读: 第 5 折 0.84 明显偏高——折间波动是常态，报告必须带 std")

assert np.round(sc, 4).tolist() == [0.7676, 0.736, 0.7537, 0.7754, 0.84]
assert round(mean_auc, 4) == 0.7745 and round(std_auc, 4) == 0.0354
'''

S3_CODE = '''# cross_validate：多指标 + train 分数，一次拿到过拟合证据
cvr = cross_validate(pipe, X, y, cv=skf, scoring=["roc_auc", "f1"],
                     return_train_score=True)
print("keys:", sorted(cvr.keys()))
print("test AUC:", np.round(cvr["test_roc_auc"], 4).tolist())
print("train AUC:", np.round(cvr["train_roc_auc"], 4).tolist(), "| test f1:",
      np.round(cvr["test_f1"], 4).tolist())
print("解读: train 0.8035 vs test 0.7745，gap 0.03——线性模型健康")

assert round(float(np.mean(cvr["test_roc_auc"])), 4) == 0.7745
assert round(float(np.mean(cvr["train_roc_auc"])), 4) == 0.8035
assert round(float(np.mean(cvr["test_f1"])), 4) == 0.418
'''

S4_CODE = '''# GridSearchCV：管道调参——参数名必须是 步骤名__参数名
param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]}
gs = GridSearchCV(pipe, param_grid, cv=skf, scoring="roc_auc", n_jobs=-1,
                  return_train_score=True)
gs.fit(X, y)
print("best params:", gs.best_params_)
print("best cv AUC:", round(float(gs.best_score_), 4), "| 组合数:", len(gs.cv_results_["params"]))
print("含 clf__ 的可调参数:", len([k for k in pipe.get_params() if k.startswith("clf__")]), "个")

assert gs.best_params_ == {"clf__C": 10.0, "clf__class_weight": None}
assert round(float(gs.best_score_), 4) == 0.7748
assert len(gs.cv_results_["params"]) == 8, "4 x 2 = 8 组合 x 5 折 = 40 次拟合"
assert len([k for k in pipe.get_params() if k.startswith("clf__")]) == 14
'''

S5_CODE = '''# 解剖 cv_results_：rank / best_index / train-test gap
best_idx = gs.best_index_
rank = gs.cv_results_["rank_test_score"].tolist()
best_test = float(gs.cv_results_["mean_test_score"][best_idx])
best_train = float(gs.cv_results_["mean_train_score"][best_idx])
print("rank:", rank)
print(f"best_index={best_idx} | train={best_train:.4f} vs test={best_test:.4f}")
print("解读: rank 1 的组合 test 只比 rank 2 高 0.0009——别神化 best_params_")

assert rank == [8, 7, 4, 3, 2, 5, 1, 6]
assert best_idx == 6 and rank[best_idx] == 1
assert round(best_train, 4) == 0.8039 and round(best_test, 4) == 0.7748
assert round(best_train - best_test, 4) == 0.0291, "gap 不到 0.03：线性模型没过拟合"
'''

S6_CODE = '''# 两阶段搜索：粗网格锁量级 → 细网格精修（贝叶斯优化的穷替身）
gs_coarse = GridSearchCV(pipe, {"clf__C": [0.1, 1.0, 10.0]}, cv=skf,
                         scoring="roc_auc", n_jobs=-1)
gs_coarse.fit(X, y)
coarse_best = gs_coarse.best_params_["clf__C"]
param_fine = {"clf__C": [coarse_best / 3, coarse_best, coarse_best * 3, coarse_best * 10]}
gs_fine = GridSearchCV(pipe, param_fine, cv=skf, scoring="roc_auc", n_jobs=-1)
gs_fine.fit(X, y)
print("粗网格 best C =", coarse_best, "→ 细网格:", param_fine["clf__C"])
print("细网格 best:", gs_fine.best_params_, round(float(gs_fine.best_score_), 4))
print("解读: 细网格把最优推到 cb/3≈3.33（再挤 0.0001）——这就是『用上一次结果指导下一次』的雏形；Optuna/skopt 是它的聪明版")

assert coarse_best == 10.0
assert abs(gs_fine.best_params_["clf__C"] - 10.0 / 3) < 1e-9
assert round(float(gs_fine.best_score_), 4) == 0.7749, "细网格比粗网格再挤 0.0001"
'''

S7_CODE = '''# RandomizedSearchCV：连续分布抽样，10 次就追平 8 组网格
param_dist = {"clf__C": uniform(0.01, 100), "clf__class_weight": [None, "balanced"]}
rs = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=skf, scoring="roc_auc",
                        random_state=42, n_jobs=-1)
rs.fit(X, y)
c_draws = [round(float(d["clf__C"]), 3) for d in rs.cv_results_["params"]]
print("抽样 C:", c_draws)
print("best:", {k: (round(float(v), 4) if isinstance(v, float) else v)
                for k, v in rs.best_params_.items()}, round(float(rs.best_score_), 4))
print("解读: 网格 8 组 0.7748 vs 随机 10 组 0.7749——高维空间随机搜索性价比更高")

assert c_draws == [37.464, 18.353, 59.876, 44.593, 5.818, 33.381, 70.817, 5.651, 83.254, 0.088]
assert rs.best_params_["clf__class_weight"] is None
assert round(float(rs.best_score_), 4) == 0.7749
'''

S8_CODE = '''# 树系网格 + refit 语义：best_estimator_ 已在全量数据上重新拟合
gs_rf = GridSearchCV(
    Pipeline([("pre", pre), ("clf", RandomForestClassifier(random_state=42))]),
    {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5, None]},
    cv=skf, scoring="f1", n_jobs=-1)
gs_rf.fit(X, y)
print("RF best:", gs_rf.best_params_, "| best f1:", round(float(gs_rf.best_score_), 4))
print("n_splits:", gs_rf.n_splits_, "| refit 耗时:", round(float(gs_rf.refit_time_), 2), "s")
print("best_estimator_ 可直接 predict:", gs_rf.best_estimator_.classes_.tolist())
print("解读: scoring 换成 f1 后结论完全不同（21% 正例上 f1=0.296）——先选对评分再调参")

assert gs_rf.best_params_ == {"clf__max_depth": None, "clf__n_estimators": 200}
assert round(float(gs_rf.best_score_), 4) == 0.296
assert gs_rf.n_splits_ == 5
assert gs_rf.best_estimator_.classes_.tolist() == [0, 1]
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch09 交叉验证与调参：KFold / GridSearchCV / RandomizedSearchCV

> 数据：`defect_ml.csv` 二分类。管道口径沿用 ch01：NUM 中位数填充+标准化、
> CAT 独热。本章回答三个问题：**怎么切折才可信、怎么搜参数才省、搜完的东西怎么用**。

**本章考点**（对应竞赛「模型选择 10%」+ 训练流程分）

1. 分层折的必要性（正例 128/600，不平衡下 KFold 会抖）
2. `cross_val_score` / `cross_validate` 的输出结构
3. `GridSearchCV` 管道调参（`步骤名__参数名` 语法）
4. `cv_results_` 解剖：rank / best_index / train-test gap
5. 两阶段粗细网格 = 贝叶斯优化的穷替身
6. `RandomizedSearchCV` 连续分布抽样

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 折内正例抖动 | KFold 抖 ±7 个正例，指标跟着抖 | §3.1 |
| 参数名写错静默变新参数 | `C` vs `clf__C` 一个警告就带过 | §3.3 |
| best_params_ 过度神化 | rank1 只比 rank2 高 0.0009 | §3.4 难点深挖 |
| scoring 决定结论 | AUC 冠军 ≠ f1 冠军 | §3.7 难点深挖 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 3.1 折结构：不平衡数据必须分层

普通 KFold 每折正例 [20, 21, 34, 27, 26]——第 3 折是第 1 折的 1.7 倍；
StratifiedKFold 稳定在 [25, 25, 26, 26, 26]。**类别不平衡 + 小数据 = 必须分层**，
否则 CV 均值本身就成了随机数。
"""
    ),
    code(S1_CODE),
    md(
        """
## 3.2 cross_val_score：单指标快速体检

5 折 AUC [0.7676, 0.736, 0.7537, 0.7754, 0.84]，mean 0.7745 / std 0.0354。
**报告必须带 std**——第 5 折 0.84 明显偏高，只报均值是自欺。
"""
    ),
    code(S2_CODE),
    md(
        """
## 3.3 cross_validate：多指标 + train 分数

`return_train_score=True` 一次拿到过拟合证据：train AUC 0.8035 vs test 0.7745，
gap 0.03，线性模型健康。`scoring` 传列表可同时看多个指标。
"""
    ),
    code(S3_CODE),
    md(
        """
### 3.4 难点深挖：GridSearchCV 与管道参数名

参数名必须是 **`步骤名__参数名`**（`clf__C`）——写 `C` 时 sklearn 把它当
Pipeline 顶层参数，静默无效或报错。本管道可调的 `clf__*` 参数有 14 个。
4×2 网格 = 8 组合 × 5 折 = **40 次拟合**，`n_jobs=-1` 并行是标配。
"""
    ),
    code(S4_CODE),
    md(
        """
### 3.5 解剖 cv_results_

rank 数组 [8,7,4,3,2,5,1,6]，best_index=6。**别神化 best_params_**：
rank1 的 test AUC 只比 rank2 高 0.0009，在折间波动（±0.035）面前毫无
统计意义——best_params_ 给的是「搜索区间内的胜者」，不是真理。
train-test gap 0.0291 说明线性模型没过拟合。
"""
    ),
    code(S5_CODE),
    md(
        """
### 3.6 两阶段搜索：贝叶斯优化的穷替身

粗网格 [0.1, 1, 10] 锁定 C=10 的量级 → 细网格在邻域 [3.33, 10, 30, 100]
精修。这是贝叶斯优化（Optuna/skopt）的手动版：**用上一次结果指导下一次
采样**。竞赛现场两阶段网格完全够用；Optuna 属于加分项，原理相同。
"""
    ),
    code(S6_CODE),
    md(
        """
### 3.7 难点深挖：RandomizedSearchCV 与 scoring 的决定性

连续分布 `uniform(0.01, 100)` 抽 10 组，best AUC 0.7749——**追平 8 组网格**。
高维参数空间里随机搜索的覆盖率远优于等距网格。

换 scoring 立刻翻车：RF 网格在 f1 下 best f1 只有 0.296（21% 正例），
max_depth=None + 200 棵树；与 AUC 视角的结论完全不同。**先选对评分再调参**。
"""
    ),
    code(S7_CODE),
    code(S8_CODE),
    md(
        """
## 小结

1. 不平衡小数据必分层：StratifiedKFold 每折正例 [25,25,26,26,26]。
2. `cross_validate` 带 train 分数与多指标，gap 是过拟合的定量证据。
3. 管道调参参数名 `步骤名__参数名`；组合数×折数=拟合次数，心里要有数。
4. `best_params_` 只是搜索区间内的胜者，rank1 与 rank2 可差 0.0009。
5. 两阶段网格 = 贝叶斯优化的穷替身；`RandomizedSearchCV` 连续分布性价比更高。
6. scoring 决定结论：AUC 冠军 ≠ f1 冠军，先定评分再调参。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch09 交叉验证与调参（练习版）

> 按提示补全 `____`，跑通所有 assert。数据与管道 `pipe` 已在脚手架就绪。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：折结构对账

比较 KFold 与 StratifiedKFold 各 5 折的正例数分布。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：单指标体检

分层 5 折跑 `cross_val_score`（roc_auc），输出各折与均值/标准差。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：多指标 + 训练分

`cross_validate` 同时要 roc_auc / f1 与 train 分数。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：管道网格调参

定义 4×2 网格（`clf__C` × `clf__class_weight`）并跑 GridSearchCV。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：解剖 cv_results_

提取 rank、best_index、最优组合的 train/test 分数与 gap。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：两阶段搜索

粗网格锁量级 → 细网格精修，验证最优 C 的一致性。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：随机搜索

`uniform` 连续分布抽 10 组，对比网格的成绩与成本。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：树系网格与 refit 语义

RF 网格（f1 评分），确认 best_estimator_ 已在全量数据上重新拟合。
"""
    ),
    code(E8_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 任务完成。自查：分层折为什么必要、`clf__C` 与 `C`
的区别、best_params_ 的统计意义、scoring 如何改变调参结论。
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
