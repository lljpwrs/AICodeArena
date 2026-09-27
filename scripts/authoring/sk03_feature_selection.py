"""sk03 —— 特征选择与降维：VarianceThreshold / SelectKBest / RFE / RFECV / PCA / t-SNE

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk03_feature_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch03_feature_selection"

HEADER = '''import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_selection import (RFE, RFECV, SelectKBest,
                                       VarianceThreshold, chi2, f_classif,
                                       mutual_info_classif)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path("data")

dev = pd.read_csv(DATA / "defect_ml.csv")

NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
TARGET = "是否危急"

X = dev[NUM + CAT]
y = dev[TARGET]

Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 本章统一口径：训练段 fit 的 median 填充（特征选择不接受 NaN）
imp = SimpleImputer(strategy="median").fit(Xs_tr[NUM])
Xdf = pd.DataFrame(imp.transform(Xs_tr[NUM]), columns=NUM)   # DataFrame：特征名可读
Xte_df = pd.DataFrame(imp.transform(Xs_te[NUM]), columns=NUM)
Xi = Xdf.to_numpy()
ss = StandardScaler().fit(Xi)
Xs_out = ss.transform(Xi)                                    # 标准化矩阵（含负值）

print("sklearn", sklearn.__version__)
print("Xdf", Xdf.shape, "| Xs_out", Xs_out.shape, "| 正例率",
      round(float(ys_tr.mean()), 4))'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


print("脚手架就绪：probe")'''


# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 方差筛选：删掉「几乎不变」的列 —— 但它是唯一看不懂量纲的方法
vt = VarianceThreshold(threshold=100.0).fit(Xdf)
print("threshold=100 的 variances_ (真方差):", np.round(vt.variances_, 2).tolist())
print("threshold=100 保留:", vt.get_feature_names_out().tolist(),
      "| 变换后形状:", vt.transform(Xdf).shape)

# 版本坑（sklearn 1.9）：threshold=0 时 variances_ 被改写成 min(方差, ptp)
vt0 = VarianceThreshold(threshold=0.0).fit(Xdf)
ptp = np.ptp(Xdf.to_numpy(), axis=0)
print("threshold=0 的 variances_:", np.round(vt0.variances_, 2).tolist(),
      "← 不是方差！是 min(方差, ptp)")
print("各列 ptp:", np.round(ptp, 2).tolist())
print("min(方差, ptp) 验证:",
      bool(np.allclose(vt0.variances_, np.minimum(vt.variances_, ptp))))

# 陷阱演示：标准化后方差全为 1，方差阈值法彻底失效
vt_scaled = VarianceThreshold(threshold=0.5).fit(Xs_out)
print("scale 后每列方差全为 1 → threshold=0.5 全保留:",
      vt_scaled.get_feature_names_out().tolist())

assert np.allclose(np.round(vt.variances_, 2), [22063.89, 57.05, 129.85, 74.68, 138.52])
assert vt.get_feature_names_out().tolist() == ["负荷值", "湿度", "巡检耗时分钟"]
assert vt.transform(Xdf).shape == (480, 3)
assert np.allclose(vt0.variances_, [795.12, 44.4, 66.8, 29.0, 72.0])
assert vt0.get_support(indices=True).tolist() == [0, 1, 2, 3, 4]
assert len(vt_scaled.get_feature_names_out()) == 5, "scale 后方差全 1，阈值法失效"
'''

S2_CODE = '''# 过滤法主力：SelectKBest + f_classif（方差分析 F 检验）
skb = SelectKBest(f_classif, k=3).fit(Xdf, ys_tr)
print("F 分数:", np.round(skb.scores_, 4).tolist())
print("p 值:", [f"{v:.3e}" for v in skb.pvalues_])
print("k=3 选中:", skb.get_feature_names_out().tolist())
print("support:", skb.get_support().tolist())
Xk = skb.transform(Xdf)
print("变换后形状:", Xk.shape, "← transform 只保留选中的 3 列")

assert np.allclose(np.round(skb.scores_, 4), [17.938, 37.2561, 1.0475, 0.1498, 0.092])
assert skb.get_feature_names_out().tolist() == ["负荷值", "温度", "湿度"]
assert skb.get_support().tolist() == [True, True, True, False, False]
assert Xk.shape == (480, 3)
assert float(skb.pvalues_[0]) < 1e-4 and float(skb.pvalues_[1]) < 1e-6
'''

S3_CODE = '''# 三种评分的「口味」不同：选出来的特征集可以不一样
# 1) chi2 要求非负：标准化后的数据含负值，直接炸
skb_neg = SelectKBest(chi2, k=3)
status, msg = probe(skb_neg.fit, Xs_out, ys_tr)
print("chi2 吃到标准化数据(含负值) →", status, "|", msg)

# 2) 原始非负数据上 chi2 正常
skb_chi = SelectKBest(chi2, k=3).fit(Xdf, ys_tr)
print("chi2 选中:", skb_chi.get_feature_names_out().tolist())
print("chi2 分数:", np.round(skb_chi.scores_, 2).tolist())

# 3) mutual_info_classif 能捕捉非线性关系（固定 random_state 才可复现）
mi_func = partial(mutual_info_classif, random_state=42)
skb_mi = SelectKBest(mi_func, k=3).fit(Xdf, ys_tr)
print("mutual_info 互信息:", np.round(skb_mi.scores_, 4).tolist())
print("mutual_info 选中:", skb_mi.get_feature_names_out().tolist())

print("f_classif 选中  : 负荷值 / 温度 / 湿度")
print("mutual_info 选中: 负荷值 / 温度 / 投运年限 ← 只有它看到了投运年限的非线性信号")

assert status == "err" and msg == "ValueError", "chi2 要求非负"
assert skb_chi.get_feature_names_out().tolist() == ["负荷值", "温度", "湿度"]
assert round(float(skb_chi.scores_[0]), 2) == 905.5
assert skb_mi.get_feature_names_out().tolist() == ["负荷值", "温度", "投运年限"]
assert round(float(skb_mi.scores_[1]), 4) == 0.0555
'''

S4_CODE = '''# 包裹法：围着模型转圈选特征 —— 视角和过滤法完全不同
lr = LogisticRegression(max_iter=1000)
rfe = RFE(lr, n_features_to_select=3).fit(Xdf, ys_tr)
print("RFE 选中:", rfe.get_feature_names_out().tolist())
print("RFE ranking:", rfe.ranking_.tolist(), "← 1 是保留，2/3 是第几轮被淘汰")

# RFECV：连「留几个」都让交叉验证说了算
rfecv = RFECV(
    LogisticRegression(max_iter=1000), step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", min_features_to_select=1, n_jobs=1).fit(Xdf, ys_tr)
print("RFECV 最优特征数:", int(rfecv.n_features_))
print("RFECV 选中:", rfecv.get_feature_names_out().tolist())
print("各特征数下的 CV 均值:", np.round(rfecv.cv_results_["mean_test_score"], 4).tolist())

assert rfe.get_feature_names_out().tolist() == ["温度", "湿度", "巡检耗时分钟"]
assert rfe.ranking_.tolist() == [3, 1, 1, 2, 1]
assert int(rfecv.n_features_) == 5, "CV 说 5 个全要：删谁降谁"
assert rfecv.get_feature_names_out().tolist() == NUM
assert np.allclose(np.round(rfecv.cv_results_["mean_test_score"], 4),
                   [0.7833, 0.7875, 0.7875, 0.7979, 0.8083])
'''

S5_CODE = '''# PCA 的第一大坑：不缩放，第一主成分被大量纲特征绑架
pca_raw = PCA(random_state=42).fit(Xi)
print("未 scale 的方差贡献率:", np.round(pca_raw.explained_variance_ratio_, 4).tolist(),
      "← 负荷值一列吃掉 98%")

# 标准化后方差才「公平」参与主成分
pca_s = PCA(random_state=42).fit(Xs_out)
ratio = pca_s.explained_variance_ratio_
print("scale 后的方差贡献率:", np.round(ratio, 4).tolist())
print("累积:", np.round(np.cumsum(ratio), 4).tolist())

# 降到 2 维：transform / 逆变换误差 / 载荷解读
p2 = PCA(n_components=2, random_state=42).fit(Xs_out)
Z = p2.transform(Xs_out)
print("降维后形状:", Z.shape, "| 保留方差:", round(float(p2.explained_variance_ratio_.sum()), 4))
print("PC1 载荷:", np.round(p2.components_[0], 4).tolist())
top = NUM[int(np.argmax(np.abs(p2.components_[0])))]
print("PC1 载荷绝对值最大的特征:", top)
rec = p2.inverse_transform(Z)
err = float(np.mean((rec - Xs_out) ** 2))
print("逆变换重构 MSE:", round(err, 4), "← 57% 的方差信息被丢掉了")

assert round(float(pca_raw.explained_variance_ratio_[0]), 4) == 0.9822
assert np.allclose(np.round(ratio, 4), [0.214, 0.2127, 0.2003, 0.1977, 0.1753])
assert Z.shape == (480, 2) and round(float(p2.explained_variance_ratio_.sum()), 4) == 0.4267
assert top == "负荷值"
assert round(err, 4) == 0.5733
'''

S6_CODE = '''# t-SNE：只做可视化，不是特征转换器
ts = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
Zt = ts.fit_transform(Xs_out)
print("t-SNE 输出形状:", Zt.shape)
print("KL 散度:", round(float(ts.kl_divergence_), 4), "(越小局部结构保持越好)")
print("迭代次数:", int(ts.n_iter_))

# 三条纪律：不可 transform 新数据 / 不进 Pipeline / 结果不可复现到小数点
# 注意：TSNE 没有 transform 属性，ts.transform 在求值时就会 AttributeError，
# 所以用 getattr 把「属性访问」本身交给 probe 接住
st = probe(getattr, ts, "transform")
print("访问 ts.transform →", st[0], "|", st[1])

assert Zt.shape == (480, 2)
assert float(ts.kl_divergence_) > 0
assert st[0] == "err" and st[1] == "AttributeError"
'''


# =========================================================================== #
# 练习版各节代码（@@todo 挖空标记 + 真实解法）
# =========================================================================== #

E1_CODE = '''# @@todo(1) 定义 threshold=100 的 VarianceThreshold，fit 训练段并变换
# @@hint VarianceThreshold(threshold=...).fit(Xdf)；vt.transform(Xdf)
vt = VarianceThreshold(threshold=100.0).fit(Xdf)
Xvt = vt.transform(Xdf)
# @@end
print("threshold=100 的 variances_ (真方差):", np.round(vt.variances_, 2).tolist())
print("threshold=100 保留:", vt.get_feature_names_out().tolist(),
      "| 变换后形状:", Xvt.shape)

# 版本坑（sklearn 1.9）：threshold=0 时 variances_ 被改写成 min(方差, ptp)
vt0 = VarianceThreshold(threshold=0.0).fit(Xdf)
ptp = np.ptp(Xdf.to_numpy(), axis=0)
print("threshold=0 的 variances_:", np.round(vt0.variances_, 2).tolist(),
      "← 不是方差！是 min(方差, ptp)")
print("各列 ptp:", np.round(ptp, 2).tolist())
print("min(方差, ptp) 验证:",
      bool(np.allclose(vt0.variances_, np.minimum(vt.variances_, ptp))))

# 陷阱演示：标准化后方差全为 1，方差阈值法彻底失效
vt_scaled = VarianceThreshold(threshold=0.5).fit(Xs_out)
print("scale 后每列方差全为 1 → threshold=0.5 全保留:",
      vt_scaled.get_feature_names_out().tolist())

assert np.allclose(np.round(vt.variances_, 2), [22063.89, 57.05, 129.85, 74.68, 138.52])
assert vt.get_feature_names_out().tolist() == ["负荷值", "湿度", "巡检耗时分钟"]
assert Xvt.shape == (480, 3)
assert np.allclose(vt0.variances_, [795.12, 44.4, 66.8, 29.0, 72.0])
assert vt0.get_support(indices=True).tolist() == [0, 1, 2, 3, 4]
assert len(vt_scaled.get_feature_names_out()) == 5, "scale 后方差全 1，阈值法失效"
'''

E2_CODE = '''# @@todo(2) 定义 SelectKBest(f_classif, k=3)，fit 训练段并变换
# @@hint SelectKBest(f_classif, k=...)；fit 需要同时给 Xdf 与 ys_tr
skb = SelectKBest(f_classif, k=3).fit(Xdf, ys_tr)
Xk = skb.transform(Xdf)
# @@end
print("F 分数:", np.round(skb.scores_, 4).tolist())
print("p 值:", [f"{v:.3e}" for v in skb.pvalues_])
print("k=3 选中:", skb.get_feature_names_out().tolist())
print("support:", skb.get_support().tolist())
print("变换后形状:", Xk.shape, "← transform 只保留选中的 3 列")

assert np.allclose(np.round(skb.scores_, 4), [17.938, 37.2561, 1.0475, 0.1498, 0.092])
assert skb.get_feature_names_out().tolist() == ["负荷值", "温度", "湿度"]
assert skb.get_support().tolist() == [True, True, True, False, False]
assert Xk.shape == (480, 3)
assert float(skb.pvalues_[0]) < 1e-4 and float(skb.pvalues_[1]) < 1e-6
'''

E3_CODE = '''# @@todo(3) 用 probe 验证：chi2 吃到标准化数据（含负值）会怎样
# @@hint 先定义 SelectKBest(chi2, k=3)，再 probe(skb_neg.fit, Xs_out, ys_tr)
skb_neg = SelectKBest(chi2, k=3)
status, msg = probe(skb_neg.fit, Xs_out, ys_tr)
# @@end
print("chi2 吃到标准化数据(含负值) →", status, "|", msg)

# @@todo(4) 在原始非负的 Xdf 上用 chi2 做 k=3 的特征选择
# @@hint SelectKBest(chi2, k=3).fit(Xdf, ys_tr)
skb_chi = SelectKBest(chi2, k=3).fit(Xdf, ys_tr)
# @@end
print("chi2 选中:", skb_chi.get_feature_names_out().tolist())
print("chi2 分数:", np.round(skb_chi.scores_, 2).tolist())

# @@todo(5) 用 mutual_info_classif 做 k=3 的特征选择（用 partial 固定 random_state=42）
# @@hint mi_func = partial(mutual_info_classif, random_state=42) 已在 HEADER 导入
mi_func = partial(mutual_info_classif, random_state=42)
skb_mi = SelectKBest(mi_func, k=3).fit(Xdf, ys_tr)
# @@end
print("mutual_info 互信息:", np.round(skb_mi.scores_, 4).tolist())
print("mutual_info 选中:", skb_mi.get_feature_names_out().tolist())
print("f_classif 选中  : 负荷值 / 温度 / 湿度")
print("mutual_info 选中: 负荷值 / 温度 / 投运年限 ← 只有它看到了投运年限的非线性信号")

assert status == "err" and msg == "ValueError", "chi2 要求非负"
assert skb_chi.get_feature_names_out().tolist() == ["负荷值", "温度", "湿度"]
assert round(float(skb_chi.scores_[0]), 2) == 905.5
assert skb_mi.get_feature_names_out().tolist() == ["负荷值", "温度", "投运年限"]
assert round(float(skb_mi.scores_[1]), 4) == 0.0555
'''

E4_CODE = '''# @@todo(6) 定义 RFE(LogisticRegression(max_iter=1000), n_features_to_select=3) 并 fit
# @@hint RFE(模型, n_features_to_select=...)；fit 要给 Xdf 与 ys_tr
lr = LogisticRegression(max_iter=1000)
rfe = RFE(lr, n_features_to_select=3).fit(Xdf, ys_tr)
# @@end
print("RFE 选中:", rfe.get_feature_names_out().tolist())
print("RFE ranking:", rfe.ranking_.tolist(), "← 1 是保留，2/3 是第几轮被淘汰")

# @@todo(7) 定义 RFECV：cv 用分层 5 折（shuffle=True, 种子 42），scoring 用 accuracy
# @@hint RFECV(模型, step=1, cv=StratifiedKFold(...), scoring=..., min_features_to_select=1, n_jobs=1)
rfecv = RFECV(
    LogisticRegression(max_iter=1000), step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="accuracy", min_features_to_select=1, n_jobs=1).fit(Xdf, ys_tr)
# @@end
print("RFECV 最优特征数:", int(rfecv.n_features_))
print("RFECV 选中:", rfecv.get_feature_names_out().tolist())
print("各特征数下的 CV 均值:", np.round(rfecv.cv_results_["mean_test_score"], 4).tolist())

assert rfe.get_feature_names_out().tolist() == ["温度", "湿度", "巡检耗时分钟"]
assert rfe.ranking_.tolist() == [3, 1, 1, 2, 1]
assert int(rfecv.n_features_) == 5, "CV 说 5 个全要：删谁降谁"
assert rfecv.get_feature_names_out().tolist() == NUM
assert np.allclose(np.round(rfecv.cv_results_["mean_test_score"], 4),
                   [0.7833, 0.7875, 0.7875, 0.7979, 0.8083])
'''

E5_CODE = '''# @@todo(8) 对未标准化的 Xi fit 全量 PCA，看第一主成分吃掉多少方差
# @@hint PCA(random_state=42).fit(Xi)
pca_raw = PCA(random_state=42).fit(Xi)
ratio_raw = pca_raw.explained_variance_ratio_
# @@end
print("未 scale 的方差贡献率:", np.round(ratio_raw, 4).tolist(),
      "← 负荷值一列吃掉 98%")

# @@todo(9) 对标准化后的 Xs_out 定义 PCA(n_components=2)，transform 得到 Z
# @@hint 先 PCA(random_state=42).fit(Xs_out) 看 5 个主成分的方差贡献率，再降 2 维
pca_s = PCA(random_state=42).fit(Xs_out)
ratio = pca_s.explained_variance_ratio_
p2 = PCA(n_components=2, random_state=42).fit(Xs_out)
Z = p2.transform(Xs_out)
# @@end
print("scale 后的方差贡献率:", np.round(ratio, 4).tolist())
print("累积:", np.round(np.cumsum(ratio), 4).tolist())
print("降维后形状:", Z.shape, "| 保留方差:", round(float(p2.explained_variance_ratio_.sum()), 4))
print("PC1 载荷:", np.round(p2.components_[0], 4).tolist())
top = NUM[int(np.argmax(np.abs(p2.components_[0])))]
print("PC1 载荷绝对值最大的特征:", top)

# @@todo(10) 用 inverse_transform 重构并计算重构 MSE
# @@hint rec = p2.inverse_transform(Z)；err = float(np.mean((rec - Xs_out) ** 2))
rec = p2.inverse_transform(Z)
err = float(np.mean((rec - Xs_out) ** 2))
# @@end
print("逆变换重构 MSE:", round(err, 4), "← 57% 的方差信息被丢掉了")

assert round(float(ratio_raw[0]), 4) == 0.9822
assert np.allclose(np.round(ratio, 4), [0.214, 0.2127, 0.2003, 0.1977, 0.1753])
assert Z.shape == (480, 2) and round(float(p2.explained_variance_ratio_.sum()), 4) == 0.4267
assert top == "负荷值"
assert round(err, 4) == 0.5733
'''

E6_CODE = '''# @@todo(11) 定义 TSNE(n_components=2, random_state=42, init="pca") 并 fit_transform
# @@hint ts.fit_transform(Xs_out)；结果存入 Zt
ts = TSNE(n_components=2, random_state=42, init="pca", perplexity=30)
Zt = ts.fit_transform(Xs_out)
# @@end
print("t-SNE 输出形状:", Zt.shape)
print("KL 散度:", round(float(ts.kl_divergence_), 4), "(越小局部结构保持越好)")
print("迭代次数:", int(ts.n_iter_))

# 三条纪律：不可 transform 新数据 / 不进 Pipeline / 结果不可复现到小数点
# 注意：TSNE 没有 transform 属性，ts.transform 在求值时就会 AttributeError，
# 所以用 getattr 把「属性访问」本身交给 probe 接住
st = probe(getattr, ts, "transform")
print("访问 ts.transform →", st[0], "|", st[1])

assert Zt.shape == (480, 2)
assert float(ts.kl_divergence_) > 0
assert st[0] == "err" and st[1] == "AttributeError"
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch03 特征选择与降维：VarianceThreshold / SelectKBest / RFE / RFECV / PCA / t-SNE

> 方向：机器学习 ｜ 竞赛对应：**模型训练 40%**（特征处理是训练流程的一部分）
> + **模型选择 10%**（过滤/包裹/嵌入三条路线的选型本身就是考点）。

特征工程的下一站：**列太多怎么办**。三条路线一个都不能混：

| 路线 | 代表 API | 视角 |
|---|---|---|
| 过滤（filter） | `VarianceThreshold` / `SelectKBest` | 先按统计量打分，与模型无关 |
| 包裹（wrapper） | `RFE` / `RFECV` | 围着一个模型反复试 |
| 变换（transform） | `PCA` / `t-SNE` | 不删列，把列「压」进新坐标系 |

本章实测出四个反直觉行为（sklearn 1.9.1 / 本数据）：

- **chi2 要求非负**：吃到标准化数据直接 `ValueError`
- **三种过滤法选中集互不相同**：f_classif 与 mutual_info 只差一列
- **RFECV 说 5 个特征全要**：CV 曲线单调上升，删谁降谁
- **不 scale 的 PCA 被负荷值绑架**：第一主成分吃掉 98.22% 方差

> 本章 `assert` 真值全部沉淀在方向 README 的「ch03 专项真值」表里。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 模型训练 40% | 特征处理按步给分 | 方差筛选、SelectKBest 三种评分、RFE/RFECV、PCA |
| 模型选择 10% | 路线选型 | 过滤 vs 包裹 vs 变换的适用场景 |

**学习目标**：拿到 5 列数值特征，能跑通过滤→包裹→变换三条路线，
**并说清每条路线选出来的特征为什么不一样**——不一样不是 bug，是视角不同。

## 二、API 速查表

| 类 / 方法 | 关键参数 | 关键属性 | 一句话说明 |
|---|---|---|---|
| `VarianceThreshold` | `threshold` | `variances_` | 删低方差列；看不懂量纲 |
| `SelectKBest` | `score_func` `k` | `scores_` `pvalues_` `get_support()` | 统计量打分取前 k |
| `f_classif` | — | — | 方差分析 F 检验（线性） |
| `chi2` | — | — | 卡方检验，**要求非负** |
| `mutual_info_classif` | `random_state` | — | 互信息（非线性），必须固定种子 |
| `RFE` | `n_features_to_select` `step` | `ranking_` `support_` | 递归剔除 |
| `RFECV` | `cv` `scoring` `min_features_to_select` | `n_features_` `cv_results_` | 交叉验证定特征数 |
| `PCA` | `n_components` `random_state` | `explained_variance_ratio_` `components_` | 方差最大化降维 |
| `t-SNE` | `n_components` `perplexity` `init` | `kl_divergence_` | 可视化专用，不可 transform |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 `VarianceThreshold`：最简单的过滤，也最容易用错

方差筛选删「几乎不变」的列——但方差是**有量纲的**：负荷值方差 795、
投运年限方差 29，同一个阈值对两者意义完全不同。
"""
    ),
    code(S1_CODE),
    md(
        """
### 3.2 难点深挖：方差阈值的量纲陷阱

**为什么难**：`threshold=100` 看似客观，实际是「负荷值过、其余全死」——
因为负荷值量纲大。更隐蔽的是：**标准化后方差全为 1**，方差阈值法彻底失效
（`threshold=0.5` 时 5 列全保留，什么也筛不出来）。

**错误示范**（实测输出）：

```python
VarianceThreshold(threshold=100).fit(Xdf)      # 保留 负荷值/湿度/巡检 —— 量纲说了算
VarianceThreshold(threshold=0.5).fit(Xs_out)   # 5 列全保留 —— scale 后全是 1
```

**正误对照**：

| 数据 | threshold=100 的结果 |
|---|---|
| 原始量纲 | (480, 3)，负荷值/湿度/巡检耗时 |
| 标准化后 | 全保留（方差全 1），方法失效 |

**判定规则**：**方差筛选只用在原始量纲、且各列量纲相近的数据上**；
量纲差异大就交给 SelectKBest / 模型法，不要硬设阈值。

**版本坑（sklearn 1.9 实测）**：`threshold=0` 时 `variances_` 被改写成
`min(方差, ptp)`（本数据 [795.12, 44.4, 66.8, 29.0, 72.0]，逐列等于真方差
与极差的较小者）——旧版教程里「variances_ 就是方差」在 1.9 的
**threshold=0 场景下不再成立**；threshold>0 时 `variances_` 仍是真方差
[22063.89, 57.05, 129.85, 74.68, 138.52]。拿 variances_ 对账前，先看
自己用的 threshold 是不是 0。
"""
    ),
    md(
        """
## 3.3 `SelectKBest` + `f_classif`：过滤法主力

F 检验给每列打分：负荷值 F=17.94、温度 F=37.26 显著（p < 1e-4），
湿度/投运年限/巡检耗时 p > 0.3 不显著。k=3 取前三个。
**传入 DataFrame 而不是 ndarray**，`get_feature_names_out()` 才有可读列名。
"""
    ),
    code(S2_CODE),
    md(
        """
### 3.4 难点深挖：三种评分三种口味，选中集互不相同

**为什么难**：都是 SelectKBest，换个 `score_func` 选中的特征就变了——
这不是 bug，是三种统计量回答的问题不同：

| 评分 | 原理 | 本数据 k=3 选中 | 限制 |
|---|---|---|---|
| `f_classif` | 组间/组内方差比（线性） | 负荷值 / 温度 / 湿度 | 无 |
| `chi2` | 卡方统计量 | 负荷值 / 温度 / 湿度 | **要求非负** |
| `mutual_info_classif` | 互信息（任意非线性） | 负荷值 / 温度 / **投运年限** | 必须固定 `random_state` |

**错误示范**（实测输出）：

```python
SelectKBest(chi2, k=3).fit(Xs_out, ys_tr)
# ValueError: Input X must be non-negative.
```

**判定规则**：**有负值先想 f_classif / mutual_info；要用 chi2 就别做带
符号的变换**（标准化、中心化都会引入负值）。关心非线性信号就上
mutual_info，并写死 `random_state`。
"""
    ),
    code(S3_CODE),
    md(
        """
## 3.5 `RFE` / `RFECV`：包裹法，围着模型转

RFE 从全部特征出发，反复训练 → 删最弱 → 再训练，`ranking_` 记录淘汰轮次。
RFECV 连「留几个」都交给交叉验证——本数据的答案出人意料：**5 个全要**，
CV 均值从 0.7833 单调升到 0.8083，删谁降谁。
"""
    ),
    code(S4_CODE),
    md(
        """
### 3.6 难点深挖：PCA 不 scale = 被大量纲绑架

**为什么难**：PCA 按方差找方向。不缩放时负荷值（std≈154）支配一切，
第一主成分「吃掉」98.22% 方差——它只是负荷值轴的别名，
降维降了个寂寞。

**错误示范**（实测输出）：

```python
PCA().fit(Xi).explained_variance_ratio_   # [0.9822, 0.0063, ...] ← 假象
```

**正误对照**：

| 数据 | 方差贡献率 |
|---|---|
| 未 scale | **0.9822** / 0.0063 / 0.0056 / ... |
| scale 后 | 0.214 / 0.2127 / 0.2003 / 0.1977 / 0.1753（五个方向几乎同等重要） |

**判定规则**：**PCA 之前必须标准化**（除非各列本来同量纲）。
降维保留了多少方差，`explained_variance_ratio_.sum()` 一个数看清；
本数据 5 个方向方差几乎均匀——**它们本来就不相关，PCA 压不动**，
这也是「PCA 不是万能压缩」的反例。
"""
    ),
    code(S5_CODE),
    md(
        """
### 3.7 难点深挖：t-SNE 的三条纪律

**为什么难**：t-SNE 输出的「簇」很好看，新手直接把它当特征转换器接进
Pipeline——然后发现它**没有 `transform` 方法**（`AttributeError`），
而且同参数重跑数值都对不上（KL 散度、坐标）。

**判定规则**：t-SNE 只做**可视化**（探索标签在低维怎么分布），
不做特征、不进 Pipeline、不写进交付指标。要在 Pipeline 里降维用 `PCA`。
"""
    ),
    code(S6_CODE),
    md(
        """
---

## 四、易错点清单

1. 方差阈值看不懂量纲：threshold=100 只是「负荷值说了算」。
2. 标准化后方差全 1，`VarianceThreshold` 彻底失效。
3. sklearn 1.9 的 `threshold=0` 把 `variances_` 改写成 min(方差, ptp)，不是真方差。
4. `chi2` 要求非负：标准化/中心化后的数据直接 `ValueError`。
5. `mutual_info_classif` 不固定 `random_state` 结果不可复现。
6. 三种过滤法选中集互不相同——选型本身就是答案的一部分。
7. `SelectKBest` 传 DataFrame 才有可读的 `get_feature_names_out()`。
8. `RFE.ranking_` 里 1 是保留，其余数字是淘汰轮次。
9. RFECV 的结论可以反直觉：本数据 5 个特征全要，删谁降谁。
10. PCA 前必须标准化，否则第一主成分被大量纲绑架（98.22%）。
11. t-SNE 没有 `transform`、不可复现、只做可视化。

## 五、本章小结

- 过滤法快但瞎（不看模型）；包裹法准但慢（围着模型转）；变换法压维不删列。
- `SelectKBest` 的 `score_func` 决定视角：线性 F 检验 / 非负卡方 / 非线性互信息。
- 特征少（<10 列）时 RFECV 可能告诉你「全都要」——别为了降维而降维。
- PCA 三件套：先 scale、看 `explained_variance_ratio_`、用 `components_` 解读。

### 复盘提问

1. 为什么标准化之后 `VarianceThreshold` 失效？
2. 三种 `score_func` 各回答什么问题？选中的特征集为什么不同？
3. `chi2` 对输入的硬约束是什么？怎么绕过去？
4. RFECV 说 5 个全要，说明这份数据的什么性质？
5. PCA 不 scale 会怎样？t-SNE 为什么不能进 Pipeline？

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
# ch03 练习：特征选择与降维

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 模型训练 40% | VarianceThreshold / SelectKBest / RFE / RFECV / PCA / t-SNE |
| 模型选择 10% | 过滤 vs 包裹 vs 变换的选型 |

## 二、API 速查

| 类 | 关键参数 | 一句话 |
|---|---|---|
| `VarianceThreshold` | `threshold` | 删低方差列（怕量纲） |
| `SelectKBest` | `score_func` `k` | 统计量打分 |
| `RFE` / `RFECV` | `n_features_to_select` / `cv` `scoring` | 包裹法 |
| `PCA` | `n_components` | 方差降维（先 scale！） |
| `TSNE` | `init` `perplexity` | 只做可视化 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：方差筛选与量纲陷阱（对应讲解 §3.1-3.2）"),
    code(E1_CODE),
    md(
        """
**为什么这样做**：方差筛选的验收要同时看 `variances_` 与
`get_feature_names_out()`——留下的列是不是「量纲大」而不是「信号强」，
一眼就能对出来。
"""
    ),
    md("## 题 2：SelectKBest + f_classif（对应讲解 §3.3）"),
    code(E2_CODE),
    md("## 题 3：三种评分三种口味（对应讲解 §3.4）"),
    code(E3_CODE),
    md(
        """
**为什么这样做**：chi2 的非负约束用 probe 验证而不是背结论；
mutual_info 用 `partial` 固定种子。两个选中集只差一列——
投运年限的非线性信号只有互信息看得见。
"""
    ),
    md("## 题 4：RFE 与 RFECV（对应讲解 §3.5）"),
    code(E4_CODE),
    md("## 题 5：PCA 的绑架与降维（对应讲解 §3.6）"),
    code(E5_CODE),
    md("## 题 6：t-SNE 与三条纪律（对应讲解 §3.7）"),
    code(E6_CODE),
    md(
        """
---

## 综合自查

1. 题 1 里 threshold=100 留下的三列，是「信号强」还是「量纲大」？
2. 题 3 两个选中集只差一列，差的那列说明什么？
3. 题 4 的 RFECV 给出「5 个全要」，如果评委问你为什么还要做特征选择，怎么答？
4. 题 5 的 0.9822 和 0.4267 分别是什么？哪个才是 PCA 该有的样子？
5. t-SNE 的输出能塞进 Pipeline 当特征吗？为什么？

全部答得上来，进入 ch04（回归模型）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
