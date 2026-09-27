#!/usr/bin/env python3
"""ch11 评估可视化：ROC/PR/混淆矩阵热力图/学习曲线/验证曲线/特征重要性（defect_ml.csv）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk11_eval_viz.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch11_eval_viz"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import matplotlib

matplotlib.use("Agg")  # 无头环境：只画不弹窗
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, auc, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, learning_curve,
                                     train_test_split, validation_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 中文标签与负号正常显示（mac 优先 PingFang；竞赛机可换 SimHei）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

DATA = Path("data")
'''

SETUP = '''# 数据与划分口径沿用 ch05/ch10：分层 8:2、median 填充、独热、标准化
dev = pd.read_csv(DATA / "defect_ml.csv")
NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
TARGET = "是否危急"
y_all = dev[TARGET]

tr_idx, te_idx = train_test_split(np.arange(len(dev)), test_size=0.2,
                                  random_state=42, stratify=y_all)
dev_tr, dev_te = dev.iloc[tr_idx], dev.iloc[te_idx]
ys_tr, ys_te = dev_tr[TARGET], dev_te[TARGET]

imp = SimpleImputer(strategy="median").fit(dev_tr[NUM])
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(dev_tr[CAT])
feat = NUM + [f"{c}_{v}" for c, vals in zip(CAT, oh.categories_) for v in vals]
Xtr = np.hstack([imp.transform(dev_tr[NUM]), oh.transform(dev_tr[CAT])])
Xte = np.hstack([imp.transform(dev_te[NUM]), oh.transform(dev_te[CAT])])
ss = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr)
dtree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr_s, ys_tr)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("sklearn", sklearn.__version__, "| train", Xtr_s.shape,
      "| test", Xte_s.shape, "| 已就绪: lr / dtree / skf")
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def finish(fig, ax, title, xlabel, ylabel):
    """统一的收尾三件套：标题、轴标、close 释放内存。"""
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.canvas.draw()
    plt.close(fig)
    return fig


print("脚手架就绪：finish")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) ROC 曲线：取点、画线、对账 AUC
# @@hint roc_curve(ys_te, p_te) 返回 (fpr, tpr, thresholds)；p_te = lr.predict_proba(Xte_s)[:, 1]
p_te = lr.predict_proba(Xte_s)[:, 1]
fpr, tpr, thr = roc_curve(ys_te, p_te)
auc_curve = round(float(auc(fpr, tpr)), 4)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="tab:red", lw=2, label=f"LR (AUC={auc_curve:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="瞎猜线")
ax.plot(fpr, tpr, "o", ms=2, color="tab:red")
finish(fig, ax, "ROC 曲线", "假正率 FPR", "真正率 TPR")
# @@end
print("ROC 点数:", len(fpr), "| 首阈值:", thr[0], "| AUC:", auc_curve)
print("与 roc_auc_score 对账:", round(float(roc_auc_score(ys_te, p_te)), 4))

assert len(fpr) == 39, "120 个测试样本 + 1 个无穷阈值 = 39 个不重复 TPR 档位"
assert thr[0] == float("inf"), "首阈值是 inf：把所有人都判负的起点"
assert auc_curve == round(float(roc_auc_score(ys_te, p_te)), 4) == 0.7946, \\
    "auc(fpr, tpr) 与 roc_auc_score 一个数"
'''

E2_CODE = '''# @@todo(2) PR 曲线：正例稀少时比 ROC 更敏感
# @@hint precision_recall_curve(ys_te, p_te) 返回 (precision, recall, thresholds)，注意 prec 比 thr 长 1
prec, rec, thr_p = precision_recall_curve(ys_te, p_te)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rec, prec, color="tab:blue", lw=2)
ax.axhline(float(ys_te.mean()), color="gray", ls="--", lw=1,
           label=f"瞎猜线=正例先验 {ys_te.mean():.4f}")
finish(fig, ax, "PR 曲线", "召回率 Recall", "精确率 Precision")
# @@end
print("prec/rec/thr 长度:", len(prec), len(rec), len(thr_p))
print("首点 (rec, prec):", round(float(rec[0]), 4), round(float(prec[0]), 4))
print("末点 (rec, prec):", round(float(rec[-1]), 4), round(float(prec[-1]), 4))

assert len(prec) == len(rec) == 121 and len(thr_p) == 120, \\
    "prec/rec 比 thr 多一个端点（+1 结构）"
assert round(float(prec[0]), 4) == 0.2167, "左端点=全判正：precision=正例先验 26/120"
assert round(float(rec[0]), 4) == 1.0 and round(float(rec[-1]), 4) == 0.0
assert round(float(prec[-1]), 4) == 1.0, "右端点=只挑最有把握的一个且命中"
'''

E3_CODE = '''# @@todo(3) 混淆矩阵热力图：dtree 的 120 个判决落到 2x2
# @@hint confusion_matrix(ys_te, y_pred)；ConfusionMatrixDisplay(cm, display_labels=["一般", "危急"]).plot()
y_pred = dtree.predict(Xte_s)
cm = confusion_matrix(ys_te, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["一般", "危急"])
disp.plot(cmap="Blues")
plt.title("混淆矩阵：深 5 决策树")
plt.close()
# @@end
print(cm)

assert cm.shape == (2, 2)
assert cm.tolist() == [[89, 5], [20, 6]], "TN 89 / FP 5 / FN 20 / TP 6"
assert cm[1, 1] == 6 and cm[1, 0] == 20, \\
    "21 个真危急只抓住 6 个：准确率好看（0.7917）掩盖了漏报大头"
assert round(float((cm[0, 0] + cm[1, 1]) / cm.sum()), 4) == 0.7917
'''

E4_CODE = '''# @@todo(4) 学习曲线（LR）：样本量加到 4 份，分数怎么走
# @@hint learning_curve(model, Xtr_s, ys_tr, cv=skf, scoring="accuracy", train_sizes=np.array([0.2,0.4,0.6,0.8,1.0]), n_jobs=-1)
sizes, tr_sc, va_sc = learning_curve(LogisticRegression(max_iter=1000, random_state=42),
                                     Xtr_s, ys_tr, cv=skf, scoring="accuracy",
                                     train_sizes=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                                     n_jobs=-1)
tr_mean = np.round(tr_sc.mean(axis=1), 4).tolist()
va_mean = np.round(va_sc.mean(axis=1), 4).tolist()
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sizes, tr_mean, "o-", label="训练分")
ax.plot(sizes, va_mean, "s-", label="验证分")
ax.fill_between(sizes, va_mean, tr_mean, alpha=0.1)
finish(fig, ax, "学习曲线：LR", "训练样本数", "accuracy")
# @@end
print("sizes:", sizes.tolist())
print("train:", tr_mean)
print("valid:", va_mean)

assert sizes.tolist() == [76, 153, 230, 307, 384], \\
    "最大只有 384 = 480 x 4/5：留 1/5 当验证，样本上限被 CV 折卡死"
assert va_mean == [0.7771, 0.7938, 0.8, 0.8208, 0.8188], "验证分随样本量整体上行"
assert va_mean[-1] > va_mean[0] and tr_mean[-1] < tr_mean[0], \\
    "两条线在靠拢：LR 欠拟合端，加数据还有肉吃"
'''

E5_CODE = '''# @@todo(5) 学习曲线（全深树）：过拟合的标准照
# @@hint 换 DecisionTreeClassifier(random_state=42)，其余同上
sizes2, tr2, va2 = learning_curve(DecisionTreeClassifier(random_state=42),
                                  Xtr_s, ys_tr, cv=skf, scoring="accuracy",
                                  train_sizes=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                                  n_jobs=-1)
tr2_mean = np.round(tr2.mean(axis=1), 4).tolist()
va2_mean = np.round(va2.mean(axis=1), 4).tolist()
gap = round(tr2_mean[-1] - va2_mean[-1], 4)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sizes2, tr2_mean, "o-", label="训练分")
ax.plot(sizes2, va2_mean, "s-", label="验证分")
finish(fig, ax, "学习曲线：全深决策树", "训练样本数", "accuracy")
# @@end
print("train:", tr2_mean)
print("valid:", va2_mean, "| 终点 gap:", gap)

assert tr2_mean == [1.0] * 5, "全深树任何样本量都把训练集背到满分"
assert va2_mean == [0.6979, 0.7167, 0.7229, 0.7208, 0.7167], "验证分原地踏步 ~0.72"
assert gap == 0.2833, "gap 0.2833：与 LR 的收敛图并排一放，过拟合一目了然"
'''

E6_CODE = '''# @@todo(6) 验证曲线（KNN）：k 从 1 扫到 15，找复杂度甜点
# @@hint validation_curve(KNeighborsClassifier(), Xtr_s, ys_tr, param_name="n_neighbors", param_range=ks, cv=skf, scoring="accuracy", n_jobs=-1)
ks = [1, 3, 5, 7, 9, 11, 13, 15]
k_tr, k_va = validation_curve(KNeighborsClassifier(), Xtr_s, ys_tr,
                              param_name="n_neighbors", param_range=ks,
                              cv=skf, scoring="accuracy", n_jobs=-1)
k_tr_mean = np.round(k_tr.mean(axis=1), 4).tolist()
k_va_mean = np.round(k_va.mean(axis=1), 4).tolist()
best_k = ks[int(np.argmax(k_va_mean))]
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(ks, k_tr_mean, "o-", label="训练分")
ax.plot(ks, k_va_mean, "s-", label="验证分")
ax.axvline(best_k, color="gray", ls="--", lw=1, label=f"best k={best_k}")
finish(fig, ax, "验证曲线：KNN", "n_neighbors", "accuracy")
# @@end
print("k:", ks)
print("train:", k_tr_mean)
print("valid:", k_va_mean, "| best k:", best_k)

assert k_tr_mean[0] == 1.0, "k=1 每个训练点都是自己的最近邻——训练分必然满分"
assert best_k == 5 and k_va_mean[ks.index(best_k)] == 0.7958, \\
    "验证分峰值 k=5：左端复杂度过拟合，右端模型太钝欠拟合"
assert min(k_va_mean) == 0.7, "k=1 的验证分垫底：训练分 1.0 是假象"
'''

E7_CODE = '''# @@todo(7) 验证曲线（RF）：树的数量维度——平坦曲线说明「买稳定」
# @@hint param_name="n_estimators", param_range=[10, 50, 100, 200]，模型 RandomForestClassifier(random_state=42)
ns = [10, 50, 100, 200]
n_tr, n_va = validation_curve(RandomForestClassifier(random_state=42), Xtr_s, ys_tr,
                              param_name="n_estimators", param_range=ns,
                              cv=skf, scoring="accuracy", n_jobs=-1)
n_tr_mean = np.round(n_tr.mean(axis=1), 4).tolist()
n_va_mean = np.round(n_va.mean(axis=1), 4).tolist()
best_n = ns[int(np.argmax(n_va_mean))]
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(ns, n_tr_mean, "o-", label="训练分")
ax.plot(ns, n_va_mean, "s-", label="验证分")
finish(fig, ax, "验证曲线：随机森林", "n_estimators", "accuracy")
# @@end
print("n:", ns)
print("train:", n_tr_mean)
print("valid:", n_va_mean, "| best n:", best_n)

assert n_tr_mean == [0.9714, 0.9984, 1.0, 1.0], "RF 训练分基本满分：自助采样留了 oob 空隙"
assert n_va_mean == [0.8021, 0.8146, 0.8042, 0.8125], "验证分 ±1.2 个点内抖动"
assert best_n == 50, "峰值在 50：与 ch10 数量曲线同款结论，树多买稳定不买精度"
'''

E8_CODE = '''# @@todo(8) 特征重要性条形图：RF100 的 16 个特征排排坐
# @@hint rf = RandomForestClassifier(n_estimators=100, random_state=42) fit 后看 feature_importances_；np.argsort 倒序
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
imp_sum = round(float(rf.feature_importances_.sum()), 4)
order = np.argsort(rf.feature_importances_)[::-1]
top5 = [feat[i] for i in order[:5]]
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh([feat[i] for i in order][::-1], rf.feature_importances_[order][::-1],
        color="tab:orange")
finish(fig, ax, "随机森林特征重要性", "重要性", "特征")
# @@end
print("重要性总和:", imp_sum)
print("top5:", top5)

assert imp_sum == 1.0, "tree 系重要性是归一化的，总和恒为 1"
assert top5 == ["温度", "负荷值", "湿度", "投运年限", "巡检耗时分钟"], \\
    "数值特征压倒独热特征：标准化后独热列信号被稀释，看图才直观"
'''

E9_CODE = '''# @@todo(9) 四宫格拼图：一张画布讲完一个模型
# @@hint plt.subplots(2, 2, figsize=(11, 8)) 返回 (fig, 2x2 的 ax 数组)；axes.ravel() 拉平；ax1, ax2, ax3, ax4 = axes.ravel()
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
ax1, ax2, ax3, ax4 = axes.ravel()
ax1.plot(fpr, tpr, color="tab:red", lw=2)
ax1.plot([0, 1], [0, 1], "k--", lw=1)
ax2.plot(rec, prec, color="tab:blue", lw=2)
ax3.plot(sizes, tr_mean, "o-", label="训练")
ax3.plot(sizes, va_mean, "s-", label="验证")
ax3.legend(fontsize=8)
ax4.plot(ks, k_tr_mean, "o-", label="训练")
ax4.plot(ks, k_va_mean, "s-", label="验证")
ax4.legend(fontsize=8)
# @@end
for ax, (t, x, y) in zip(axes.ravel(), [
        ("ROC", "FPR", "TPR"), ("PR", "Recall", "Precision"),
        ("学习曲线 LR", "样本数", "acc"), ("验证曲线 KNN", "k", "acc")]):
    ax.set_title(t, fontsize=11)
    ax.set_xlabel(x, fontsize=9)
    ax.set_ylabel(y, fontsize=9)
fig.tight_layout()
fig.savefig("ch11_dashboard.png", dpi=120)
plt.close(fig)
print("四宫格已保存: ch11_dashboard.png")

import os
assert axes.shape == (2, 2), "2x2 四宫格"
assert os.path.exists("ch11_dashboard.png")
assert os.path.getsize("ch11_dashboard.png") > 10_000, "四张子图的 PNG 不可能小于 10KB"
os.remove("ch11_dashboard.png")  # 校验完即清，不留垃圾
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# ROC：阈值滑动下的 (FPR, TPR) 轨迹；39 个档位、首阈值 inf
p_te = lr.predict_proba(Xte_s)[:, 1]
fpr, tpr, thr = roc_curve(ys_te, p_te)
auc_curve = round(float(auc(fpr, tpr)), 4)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, color="tab:red", lw=2, label=f"LR (AUC={auc_curve:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="瞎猜线")
ax.plot(fpr, tpr, "o", ms=2, color="tab:red")
finish(fig, ax, "ROC 曲线", "假正率 FPR", "真正率 TPR")
print("ROC 点数:", len(fpr), "| 首阈值:", thr[0], "| AUC:", auc_curve)
print("解读: AUC 只看排序不看阈值；sklearn 1.9 起首阈值是 inf（全员判负起点）")

assert len(fpr) == 39 and thr[0] == float("inf") and auc_curve == 0.7946
'''

S2_CODE = '''# PR：正例只占 21.3% 时，PR 比 ROC 更能暴露问题
prec, rec, thr_p = precision_recall_curve(ys_te, p_te)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(rec, prec, color="tab:blue", lw=2)
ax.axhline(float(ys_te.mean()), color="gray", ls="--", lw=1,
           label=f"瞎猜线=正例先验 {ys_te.mean():.4f}")
finish(fig, ax, "PR 曲线", "召回率 Recall", "精确率 Precision")
print("长度结构 prec/rec/thr:", len(prec), len(rec), len(thr_p))
print("解读: 左端点=全判正 (rec=1, prec=0.2167=先验)；右端点=只挑最稳的一发且命中；"
      "曲线永远在瞎猜线上方才有用")

assert (len(prec), len(rec), len(thr_p)) == (121, 121, 120)
assert round(float(prec[0]), 4) == 0.2167 and round(float(prec[-1]), 4) == 1.0
'''

S3_CODE = '''# 混淆矩阵热力图：2x2 里藏着漏报大头
y_pred = dtree.predict(Xte_s)
cm = confusion_matrix(ys_te, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["一般", "危急"])
disp.plot(cmap="Blues")
plt.title("混淆矩阵：深 5 决策树")
plt.close()
print(cm)
print("解读: acc 0.7917 看着体面，FN=20/21——四分之三的真危急漏掉了。"
      "不平衡数据先看对角线外的格子，再看准确率")

assert cm.tolist() == [[89, 5], [20, 6]]
'''

S4_CODE = '''# 学习曲线 LR：两条线靠拢 = 加数据还有肉吃
sizes, tr_sc, va_sc = learning_curve(LogisticRegression(max_iter=1000, random_state=42),
                                     Xtr_s, ys_tr, cv=skf, scoring="accuracy",
                                     train_sizes=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                                     n_jobs=-1)
tr_mean = np.round(tr_sc.mean(axis=1), 4).tolist()
va_mean = np.round(va_sc.mean(axis=1), 4).tolist()
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sizes, tr_mean, "o-", label="训练分")
ax.plot(sizes, va_mean, "s-", label="验证分")
ax.fill_between(sizes, va_mean, tr_mean, alpha=0.1)
finish(fig, ax, "学习曲线：LR", "训练样本数", "accuracy")
print("sizes:", sizes.tolist())
print("train:", tr_mean)
print("valid:", va_mean)
print("解读: 最大 384=480x4/5，样本上限被 CV 折卡死；valid 0.7771→0.8208 上行、"
      "train 缓降，两线在靠拢——欠拟合端，堆数据有效")

assert sizes.tolist() == [76, 153, 230, 307, 384]
assert va_mean == [0.7771, 0.7938, 0.8, 0.8208, 0.8188]
'''

S5_CODE = '''# 学习曲线全深树：train 满分横线 vs valid 原地踏步
sizes2, tr2, va2 = learning_curve(DecisionTreeClassifier(random_state=42),
                                  Xtr_s, ys_tr, cv=skf, scoring="accuracy",
                                  train_sizes=np.array([0.2, 0.4, 0.6, 0.8, 1.0]),
                                  n_jobs=-1)
tr2_mean = np.round(tr2.mean(axis=1), 4).tolist()
va2_mean = np.round(va2.mean(axis=1), 4).tolist()
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(sizes2, tr2_mean, "o-", label="训练分")
ax.plot(sizes2, va2_mean, "s-", label="验证分")
finish(fig, ax, "学习曲线：全深决策树", "训练样本数", "accuracy")
print("train:", tr2_mean)
print("valid:", va2_mean, "| 终点 gap:", round(tr2_mean[-1] - va2_mean[-1], 4))
print("解读: gap 0.2833。与上一张并排看：一条在收敛、一条在背书——"
      "过拟合诊断图，比任何文字都有说服力")

assert tr2_mean == [1.0] * 5
assert round(tr2_mean[-1] - va2_mean[-1], 4) == 0.2833
'''

S6_CODE = '''# 验证曲线 KNN：k=1 训练满分是假象，峰值 k=5
ks = [1, 3, 5, 7, 9, 11, 13, 15]
k_tr, k_va = validation_curve(KNeighborsClassifier(), Xtr_s, ys_tr,
                              param_name="n_neighbors", param_range=ks,
                              cv=skf, scoring="accuracy", n_jobs=-1)
k_tr_mean = np.round(k_tr.mean(axis=1), 4).tolist()
k_va_mean = np.round(k_va.mean(axis=1), 4).tolist()
best_k = ks[int(np.argmax(k_va_mean))]
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(ks, k_tr_mean, "o-", label="训练分")
ax.plot(ks, k_va_mean, "s-", label="验证分")
ax.axvline(best_k, color="gray", ls="--", lw=1, label=f"best k={best_k}")
finish(fig, ax, "验证曲线：KNN", "n_neighbors", "accuracy")
print("train:", k_tr_mean)
print("valid:", k_va_mean, "| best k:", best_k)
print("解读: 左端 k=1 train=1.0 / valid=0.70（背书）；右端 k=15 两线趋同（变钝）；"
      "甜点 k=5——验证曲线就是『单参数版的调参可视化』")

assert k_tr_mean[0] == 1.0 and best_k == 5 and k_va_mean[2] == 0.7958
'''

S7_CODE = '''# 验证曲线 RF：平坦 = 树的数量只买稳定
ns = [10, 50, 100, 200]
n_tr, n_va = validation_curve(RandomForestClassifier(random_state=42), Xtr_s, ys_tr,
                              param_name="n_estimators", param_range=ns,
                              cv=skf, scoring="accuracy", n_jobs=-1)
n_tr_mean = np.round(n_tr.mean(axis=1), 4).tolist()
n_va_mean = np.round(n_va.mean(axis=1), 4).tolist()
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(ns, n_tr_mean, "o-", label="训练分")
ax.plot(ns, n_va_mean, "s-", label="验证分")
finish(fig, ax, "验证曲线：随机森林", "n_estimators", "accuracy")
print("train:", n_tr_mean)
print("valid:", n_va_mean, "| best n:", ns[int(np.argmax(n_va_mean))])
print("解读: 验证分 0.8021→0.8146→0.8042→0.8125，±1.2 个点抖动——平坦曲线说明 "
      "n_estimators 不是关键超参，竞赛里给 100/200 即可，时间花在 max_depth 上")

assert n_va_mean == [0.8021, 0.8146, 0.8042, 0.8125]
'''

S8_CODE = '''# 特征重要性条形图：一图定调「哪些特征在干活」
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
order = np.argsort(rf.feature_importances_)[::-1]
fig, ax = plt.subplots(figsize=(6, 5))
ax.barh([feat[i] for i in order][::-1], rf.feature_importances_[order][::-1],
        color="tab:orange")
finish(fig, ax, "随机森林特征重要性", "重要性", "特征")
print("top5:", [feat[i] for i in order[:5]])
print("解读: 总和恒为 1（归一化）；数值特征温度/负荷/湿度领跑——"
      "独热列被稀释成十几个小柱，重要性只看『集合体』别逐列纠结")

assert [feat[i] for i in order[:5]] == ["温度", "负荷值", "湿度", "投运年限", "巡检耗时分钟"]
assert round(float(rf.feature_importances_.sum()), 4) == 1.0
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch11 评估可视化：ROC / PR / 混淆矩阵 / 学习曲线 / 验证曲线 / 特征重要性

> 数据：`defect_ml.csv` 二分类，口径沿用 ch05/ch10。matplotlib 无头渲染
> （Agg 后端），中文标签用 `PingFang SC` 系列，竞赛机可换 `SimHei`。
>
> 本章目标：**把 ch07/ch08 的指标画成图，把「模型好不好」变成「一眼看懂」**。
> 竞赛评分点「模型性能评估 10%」里，图表质量是可见的加分项。

**本章考点**

1. `roc_curve` 的点结构（+1 端点、首阈值 inf）与 AUC 对账
2. `precision_recall_curve` 的 +1 结构与瞎猜线（正例先验）
3. `ConfusionMatrixDisplay`：准确率掩盖漏报
4. `learning_curve`：欠拟合/过拟合的两张标准照
5. `validation_curve`：单参数扫描的甜点定位
6. `feature_importances_` 条形图与归一化

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| 曲线 API 长度 +1 | prec/rec 比 thr 多一个端点 | §11.1/11.2 |
| 样本上限被 CV 卡死 | learning_curve 最大 480×4/5=384 | §11.3 |
| 学习曲线两张照 | LR 收敛 vs 全深树背书 | §11.3/11.4 |
| 验证曲线两端 | k=1 背书 / k=15 变钝 | §11.5 |
| importance 逐列纠结 | 独热列被稀释，要看集合体 | §11.7 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 11.1 ROC 曲线：排序能力的一张图

39 个不重复 TPR 档位（120 样本 + 1 个 inf 起点），首阈值 inf 是 sklearn 1.9
的行为变更点。`auc(fpr, tpr)` 与 `roc_auc_score` 是同一个数（0.7946）——
画完图务必和标量指标对账，防「图是图、数是数」两张皮。
"""
    ),
    code(S1_CODE),
    md(
        """
## 11.2 PR 曲线：不平衡数据的放大镜

正例只占 21.3%，ROC 看着都体面，PR 曲线才暴露真面目。瞎猜线是**正例先验
0.2167 的水平线**——曲线在这条线上方才有价值。长度结构：prec/rec 121、
thr 120（+1 端点），两端点分别对应「全判正」与「只挑最稳一发」。
"""
    ),
    code(S2_CODE),
    md(
        """
## 11.3 难点深挖：混淆矩阵与准确率的遮羞布

深 5 决策树 acc 0.7917 看着体面，但混淆矩阵 FN=20——21 个真危急漏掉
19 个。**不平衡数据先看对角线外的格子**：FP=5 是虚惊（代价小），
FN=20 是漏报（代价大），两张脸完全不同。
"""
    ),
    code(S3_CODE),
    md(
        """
## 11.4 学习曲线：LR 的收敛照与全深树的背书照

`learning_curve` 最大样本 384 = 480×4/5——**留 1/5 做验证，样本上限被
CV 折卡死**，想要更大样本只能减折数。LR 的两条线在靠拢（欠拟合端，
加数据有效）；全深树 train 恒 1.0、valid 原地 0.72，gap 0.2833。
**两张图并排贴进报告，过拟合不用解释**。
"""
    ),
    code(S4_CODE),
    code(S5_CODE),
    md(
        """
## 11.5 难点深挖：验证曲线的左右两端

KNN：k=1 训练分 1.0（每个点是自己最近邻）、验证 0.70 垫底；k=15 两线
趋同（模型变钝）；甜点 k=5（0.7958）。RF 的 n_estimators 曲线平坦
（±1.2 个点）——**平坦 = 该参数不是关键，时间花在 max_depth 上**。
"""
    ),
    code(S6_CODE),
    code(S7_CODE),
    md(
        """
## 11.6 特征重要性：一图定调

RF100 重要性总和恒为 1（归一化）。top5：温度 0.2244 / 负荷值 0.1804 /
湿度 0.1363 / 投运年限 0.1334 / 巡检耗时 0.1219——数值特征压倒独热特征，
因为独热列被稀释成十几个小柱。**报告里看集合体，别逐列纠结**。
"""
    ),
    code(S8_CODE),
    md(
        """
## 小结

1. ROC 看排序（AUC 0.7946），PR 看不平衡（瞎猜线=先验 0.2167）。
2. 曲线 API 的 +1 端点结构：prec/rec 比 thr 多一个。
3. 混淆矩阵：FN=20 的漏报被 acc 0.7917 完全遮住。
4. learning_curve 两张标准照：LR 收敛（gap 缩小）vs 全深树背书（gap 0.2833）。
5. validation_curve 甜点定位：KNN best k=5；RF 的 n_estimators 平坦不值得调。
6. 特征重要性总和恒 1，独热列要看集合体。
7. 竞赛图表三要素：中文标签、瞎猜线参照、轴标签——三样都全才算交付。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch11 评估可视化（练习版）

> 按提示补全 `____`，跑通所有 assert。数据、模型（lr / dtree）、
> 分层折 `skf` 已在脚手架就绪；`finish()` 负责图表收尾。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：ROC 曲线

取点、画线、与 `roc_auc_score` 对账。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：PR 曲线

长度结构、两个端点、瞎猜线。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：混淆矩阵热力图

dtree 的 2×2 判决表，找到被准确率遮住的漏报。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：学习曲线（LR）

样本量阶梯与两条线的走向。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：学习曲线（全深树）

过拟合的标准照：train 全 1.0 vs valid 原地踏步。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：验证曲线（KNN）

k 扫描找甜点，两端各有各的病。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：验证曲线（RF）

n_estimators 的平坦曲线说明什么。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：特征重要性条形图

top5 排名与归一化总和。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：四宫格拼图

2×2 画布把 ROC / PR / 学习曲线 / 验证曲线拼成一张交付图并存盘。
"""
    ),
    code(E9_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 任务完成。自查：曲线 API 的 +1 结构、学习曲线两张照
怎么读、验证曲线两端各是什么病、特征重要性为什么看集合体。
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
