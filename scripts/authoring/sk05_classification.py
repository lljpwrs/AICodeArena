"""sk05 —— 分类模型：逻辑回归 / KNN / 朴素贝叶斯 / SVM / 决策树 / 集成

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk05_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch05_classification"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

DATA = Path("data")

dev = pd.read_csv(DATA / "defect_ml.csv")    # 分类：预测「是否危急」

NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
TARGET = "是否危急"
y = dev[TARGET]

# 分层切分：21.3% 的正例比例在训练/测试段保持一致
tr_idx, te_idx = train_test_split(np.arange(len(dev)), test_size=0.2,
                                  random_state=42, stratify=y)
dev_tr, dev_te = dev.iloc[tr_idx], dev.iloc[te_idx]
ys_tr, ys_te = dev_tr[TARGET], dev_te[TARGET]

# 数值列 median 填充 + 独热 + 标准化（预处理细节 ch02/ch03 已讲透）
imp = SimpleImputer(strategy="median").fit(dev_tr[NUM])
oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(dev_tr[CAT])
feat = NUM + [f"{c}_{v}" for c, vals in zip(CAT, oh.categories_) for v in vals]
Xtr = np.hstack([imp.transform(dev_tr[NUM]), oh.transform(dev_tr[CAT])])
Xte = np.hstack([imp.transform(dev_te[NUM]), oh.transform(dev_te[CAT])])
ss = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = ss.transform(Xtr), ss.transform(Xte)

print("sklearn", sklearn.__version__)
print("train", Xtr_s.shape, "正例", int(ys_tr.sum()), "/", len(ys_tr),
      "| test", Xte_s.shape, "正例", int(ys_te.sum()), "/", len(ys_te))'''

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

S1_CODE = '''# 逻辑回归：先看「敢报几个正例」，再看概率输出
lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr)
yp = lr.predict(Xte_s)
cm = confusion_matrix(ys_te, yp)
print("test 预测正例:", int(yp.sum()), "| 实际正例:", int(ys_te.sum()))
print("混淆矩阵 [[TN FP] / [FN TP]]:")
print(cm)

proba = lr.predict_proba(Xte_s)
dec = lr.decision_function(Xte_s)
print("proba 行和:", np.round(proba.sum(axis=1)[:3], 6).tolist(),
      "| classes_:", lr.classes_.tolist())
print("decision range:", round(float(dec.min()), 4), "→", round(float(dec.max()), 4))
print("dec == logit(p) 验证:", bool(np.allclose(dec, np.log(proba[:, 1] / proba[:, 0]))))
top = feat[int(np.argmax(np.abs(lr.coef_[0])))]
print("|coef| 最大:", top, round(float(lr.coef_[0].max()), 4),
      "| intercept:", round(float(lr.intercept_[0]), 4))

assert int(yp.sum()) == 10, "模型只敢报 10 个危急，实际有 26 个"
assert cm.tolist() == [[89, 5], [21, 5]], "FN=21 远大于 FP=5：默认阈值偏向多数类"
assert proba.shape == (120, 2) and lr.classes_.tolist() == [0, 1]
assert float(np.abs(proba.sum(axis=1) - 1).max()) < 1e-9
assert np.allclose(dec, np.log(proba[:, 1] / proba[:, 0])), "decision_function 就是 log-odds"
assert top == "温度" and round(float(lr.coef_[0].max()), 4) == 0.8106
assert round(float(lr.intercept_[0]), 4) == -1.7024
'''

S2_CODE = '''# 准确率陷阱：全预测「负类」的哑模型和逻辑回归打平了
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr_s, ys_tr)
yp_d = dummy.predict(Xte_s)
acc_lr = accuracy_score(ys_te, yp)
acc_dm = accuracy_score(ys_te, yp_d)
print(f"逻辑回归 acc={acc_lr:.4f} | 哑模型 acc={acc_dm:.4f}")
print("哑模型 precision:", round(precision_score(ys_te, yp_d, zero_division=0), 4),
      "| recall:", round(recall_score(ys_te, yp_d), 4))
print(classification_report(ys_te, yp, target_names=["一般", "危急"], zero_division=0))

assert round(acc_dm, 4) == 0.7833, "全预测负类 = 94/120"
assert round(acc_lr, 4) == 0.7833, "两者准确率打平——准确率在 21% 正例上失效"
assert precision_score(ys_te, yp_d, zero_division=0) == 0.0
assert recall_score(ys_te, yp_d) == 0.0
assert round(f1_score(ys_te, yp), 4) == 0.2778, "F1 才看得出逻辑回归真的学到了东西"
'''

S3_CODE = '''# 阈值移动：0.5 不是法律，Youden J 找更划算的工作点
p1 = proba[:, 1]
for t in (0.3, 0.5, 0.7):
    yp_t = (p1 >= t).astype(int)
    print(f"t={t}: 预测正例 {int(yp_t.sum()):>2} acc={accuracy_score(ys_te, yp_t):.4f} "
          f"P={precision_score(ys_te, yp_t, zero_division=0):.4f} "
          f"R={recall_score(ys_te, yp_t):.4f}")

fpr, tpr, thr = roc_curve(ys_te, p1)
j = int(np.argmax(tpr - fpr))
print("阈值数:", len(thr), "| 首阈值:", thr[0])
print(f"Youden 最优 t={thr[j]:.4f}: TPR={tpr[j]:.4f} FPR={fpr[j]:.4f}")

assert len(thr) == 39
assert bool(np.isinf(thr[0])), "sklearn 1.9 首阈值是 inf（无样本被预测为正）"
assert round(float(thr[j]), 4) == 0.2455
assert round(float(tpr[j]), 4) == 0.8077 and round(float(fpr[j]), 4) == 0.2660
assert round(recall_score(ys_te, (p1 >= 0.3).astype(int)), 4) == 0.6538
'''

S4_CODE = '''# class_weight="balanced"：训练时给少数类的损失加倍
lr_b = LogisticRegression(max_iter=1000, class_weight="balanced",
                          random_state=42).fit(Xtr_s, ys_tr)
yp_b = lr_b.predict(Xte_s)
cm_b = confusion_matrix(ys_te, yp_b)
print("balanced 混淆矩阵:")
print(cm_b)
print("balanced 预测正例:", int(yp_b.sum()),
      "| recall:", round(recall_score(ys_te, yp_b), 4),
      "| acc:", round(accuracy_score(ys_te, yp_b), 4))

assert cm_b.tolist() == [[63, 31], [5, 21]], "FN 21→5，代价是 FP 5→31"
assert int(yp_b.sum()) == 52, "正例预测数 10 → 52"
assert round(recall_score(ys_te, yp_b), 4) == 0.8077
assert round(accuracy_score(ys_te, yp_b), 4) == 0.7000, "准确率换 recall，看业务要哪个"
'''

S5_CODE = '''# KNN：距离被量纲绑架——和树相反，scale 是刚需不是可选项
knn5 = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ys_tr)
knn5_raw = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ys_tr)
acc_s = accuracy_score(ys_te, knn5.predict(Xte_s))
acc_raw = accuracy_score(ys_te, knn5_raw.predict(Xte))
p_s = precision_score(ys_te, knn5.predict(Xte_s), zero_division=0)
p_raw = precision_score(ys_te, knn5_raw.predict(Xte), zero_division=0)
print(f"KNN k=5 scaled: acc={acc_s:.4f} P={p_s:.4f}")
print(f"KNN k=5 raw   : acc={acc_raw:.4f} P={p_raw:.4f}")

for k in (3, 11):
    knn_k = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ys_tr)
    print(f"KNN k={k} scaled: acc={accuracy_score(ys_te, knn_k.predict(Xte_s)):.4f} "
          f"R={recall_score(ys_te, knn_k.predict(Xte_s)):.4f}")

assert round(acc_s, 4) == 0.7500 and round(acc_raw, 4) == 0.8000
assert round(p_raw, 4) == 0.6250 and round(p_s, 4) == 0.3000, "raw 版误报反而更少"
assert round(recall_score(ys_te, knn5.predict(Xte_s)), 4) == 0.1154
'''

S6_CODE = '''# 高斯朴素贝叶斯：对逐列仿射缩放完全免疫
gnb = GaussianNB().fit(Xtr_s, ys_tr)
gnb_raw = GaussianNB().fit(Xtr, ys_tr)
acc_s = accuracy_score(ys_te, gnb.predict(Xte_s))
acc_raw = accuracy_score(ys_te, gnb_raw.predict(Xte))
same = bool((gnb.predict(Xte_s) == gnb_raw.predict(Xte)).all())
print(f"GNB scaled acc={acc_s:.4f} | raw acc={acc_raw:.4f} | 预测逐位一致: {same}")
print("GNB theta_[0] 前 3(类 0 均值):", np.round(gnb.theta_[0][:3], 4).tolist())

assert round(acc_s, 4) == 0.6750 and round(acc_raw, 4) == 0.6750
assert same, "逐列减均值除方差不改变高斯似然的 argmax"
assert round(float(gnb.theta_[0][0]), 4) == -0.0988
assert round(recall_score(ys_te, gnb.predict(Xte_s)), 4) == 0.4231, "NB 敢报正例"
'''

S7_CODE = '''# SVM：LinearSVC 无概率输出；RBF 核 raw 版直接躺平
lsvc = LinearSVC(max_iter=10000, random_state=42).fit(Xtr_s, ys_tr)
svc = SVC(random_state=42).fit(Xtr_s, ys_tr)
svc_raw = SVC(random_state=42).fit(Xtr, ys_tr)
acc_l = accuracy_score(ys_te, lsvc.predict(Xte_s))
acc_v = accuracy_score(ys_te, svc.predict(Xte_s))
acc_r = accuracy_score(ys_te, svc_raw.predict(Xte))
print(f"LinearSVC scaled: acc={acc_l:.4f} P={precision_score(ys_te, lsvc.predict(Xte_s), zero_division=0):.4f}")
print(f"SVC rbf scaled  : acc={acc_v:.4f}")
print(f"SVC rbf raw     : acc={acc_r:.4f} P={precision_score(ys_te, svc_raw.predict(Xte), zero_division=0):.4f} "
      f"R={recall_score(ys_te, svc_raw.predict(Xte)):.4f} ← 全预测负类躺平")

st_proba = probe(getattr, lsvc, "predict_proba")
print("LinearSVC.predict_proba →", st_proba)
svc_p = SVC(probability=True, random_state=42).fit(Xtr_s, ys_tr)
print("SVC(probability=True) proba shape:", svc_p.predict_proba(Xte_s).shape)

assert round(acc_l, 4) == 0.8000 and round(precision_score(ys_te, lsvc.predict(Xte_s), zero_division=0), 4) == 0.6250
assert round(acc_v, 4) == 0.7583
assert round(acc_r, 4) == 0.7833 and recall_score(ys_te, svc_raw.predict(Xte)) == 0.0
assert st_proba == ("err", "AttributeError"), "LinearSVC 没有 predict_proba"
assert svc_p.predict_proba(Xte_s).shape == (120, 2), "probability=True 走内部 CV 估算"
'''

S8_CODE = '''# 树与集成：过拟合现场 + 本数据准确率冠军
dt_full = DecisionTreeClassifier(random_state=42).fit(Xtr_s, ys_tr)
dt5 = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr_s, ys_tr)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr_s, ys_tr)
gb = GradientBoostingClassifier(random_state=42).fit(Xtr_s, ys_tr)

leaves_full = dt_full.get_n_leaves()
leaves5 = dt5.get_n_leaves()
train_acc = accuracy_score(ys_tr, dt_full.predict(Xtr_s))
acc5 = accuracy_score(ys_te, dt5.predict(Xte_s))
acc_rf = accuracy_score(ys_te, rf.predict(Xte_s))
acc_gb = accuracy_score(ys_te, gb.predict(Xte_s))
imp_rf = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
print(f"DT 不限深: {leaves_full} 叶 train acc={train_acc:.4f} test={accuracy_score(ys_te, dt_full.predict(Xte_s)):.4f}")
print(f"DT d=5  : {leaves5} 叶 test acc={acc5:.4f} ← 全场准确率冠军")
print(f"RF test acc={acc_rf:.4f} | GB test acc={acc_gb:.4f}")
print("RF importance top4:", {k: round(v, 4) for k, v in imp_rf.head(4).items()})

assert leaves_full == 95 and round(train_acc, 4) == 1.0, "背表现场"
assert leaves5 == 18 and round(acc5, 4) == 0.7917
assert round(acc_rf, 4) == 0.7667 and round(acc_gb, 4) == 0.7750
assert round(float(imp_rf.iloc[0]), 4) == 0.2220 and imp_rf.index[0] == "温度"
'''

S9_CODE = '''# AUC 复盘：准确率平平的逻辑回归，排序能力全场第一
auc_lr = roc_auc_score(ys_te, proba[:, 1])
auc_rf = roc_auc_score(ys_te, rf.predict_proba(Xte_s)[:, 1])
auc_gb = roc_auc_score(ys_te, gb.predict_proba(Xte_s)[:, 1])
print(f"AUC: LR={auc_lr:.4f} > RF={auc_rf:.4f} > GB={auc_gb:.4f}")
print("复盘：准确率上 LR 与哑模型打平、树系反而更高，但 AUC 说 LR 的分数最会排序")

assert round(auc_lr, 4) == 0.7946
assert round(auc_rf, 4) == 0.6991 and round(auc_gb, 4) == 0.6809
assert auc_lr > auc_rf, "不平衡数据上 AUC 比准确率更会讲故事"
'''


E1_CODE = '''# @@todo(1) 训练逻辑回归，预测测试段并算混淆矩阵
# @@hint LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr) + predict + confusion_matrix
lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr)
yp = lr.predict(Xte_s)
cm = confusion_matrix(ys_te, yp)
# @@end
print("test 预测正例:", int(yp.sum()), "| 实际正例:", int(ys_te.sum()))
print("混淆矩阵 [[TN FP] / [FN TP]]:")
print(cm)

# @@todo(2) 取 predict_proba 与 decision_function，验证 decision == logit(p)
# @@hint lr.predict_proba(Xte_s) / lr.decision_function(Xte_s)；np.allclose(dec, np.log(proba[:, 1] / proba[:, 0]))
proba = lr.predict_proba(Xte_s)
dec = lr.decision_function(Xte_s)
# @@end
print("proba 行和:", np.round(proba.sum(axis=1)[:3], 6).tolist(),
      "| classes_:", lr.classes_.tolist())
print("decision range:", round(float(dec.min()), 4), "→", round(float(dec.max()), 4))
print("dec == logit(p) 验证:", bool(np.allclose(dec, np.log(proba[:, 1] / proba[:, 0]))))
top = feat[int(np.argmax(np.abs(lr.coef_[0])))]
print("|coef| 最大:", top, round(float(lr.coef_[0].max()), 4),
      "| intercept:", round(float(lr.intercept_[0]), 4))

assert int(yp.sum()) == 10, "模型只敢报 10 个危急，实际有 26 个"
assert cm.tolist() == [[89, 5], [21, 5]], "FN=21 远大于 FP=5：默认阈值偏向多数类"
assert proba.shape == (120, 2) and lr.classes_.tolist() == [0, 1]
assert float(np.abs(proba.sum(axis=1) - 1).max()) < 1e-9
assert np.allclose(dec, np.log(proba[:, 1] / proba[:, 0])), "decision_function 就是 log-odds"
assert top == "温度" and round(float(lr.coef_[0].max()), 4) == 0.8106
assert round(float(lr.intercept_[0]), 4) == -1.7024
'''

E2_CODE = '''# @@todo(3) 训练全预测负类的哑模型，与逻辑回归的准确率对账
# @@hint DummyClassifier(strategy="most_frequent").fit(Xtr_s, ys_tr)
dummy = DummyClassifier(strategy="most_frequent").fit(Xtr_s, ys_tr)
yp_d = dummy.predict(Xte_s)
acc_lr = accuracy_score(ys_te, yp)
acc_dm = accuracy_score(ys_te, yp_d)
# @@end
print(f"逻辑回归 acc={acc_lr:.4f} | 哑模型 acc={acc_dm:.4f}")
print("哑模型 precision:", round(precision_score(ys_te, yp_d, zero_division=0), 4),
      "| recall:", round(recall_score(ys_te, yp_d), 4))
print(classification_report(ys_te, yp, target_names=["一般", "危急"], zero_division=0))

assert round(acc_dm, 4) == 0.7833, "全预测负类 = 94/120"
assert round(acc_lr, 4) == 0.7833, "两者准确率打平——准确率在 21% 正例上失效"
assert precision_score(ys_te, yp_d, zero_division=0) == 0.0
assert recall_score(ys_te, yp_d) == 0.0
assert round(f1_score(ys_te, yp), 4) == 0.2778, "F1 才看得出逻辑回归真的学到了东西"
'''

E3_CODE = '''# 阈值扫描先给出分数列（脚手架，不用挖空）
p1 = proba[:, 1]
for t in (0.3, 0.5, 0.7):
    yp_t = (p1 >= t).astype(int)
    print(f"t={t}: 预测正例 {int(yp_t.sum()):>2} acc={accuracy_score(ys_te, yp_t):.4f} "
          f"P={precision_score(ys_te, yp_t, zero_division=0):.4f} "
          f"R={recall_score(ys_te, yp_t):.4f}")

# @@todo(4) 用 roc_curve 取阈值序列，找 Youden J（TPR-FPR）最大的阈值
# @@hint fpr, tpr, thr = roc_curve(ys_te, p1)；j = int(np.argmax(tpr - fpr))
fpr, tpr, thr = roc_curve(ys_te, p1)
j = int(np.argmax(tpr - fpr))
# @@end
print("阈值数:", len(thr), "| 首阈值:", thr[0])
print(f"Youden 最优 t={thr[j]:.4f}: TPR={tpr[j]:.4f} FPR={fpr[j]:.4f}")

assert len(thr) == 39
assert bool(np.isinf(thr[0])), "sklearn 1.9 首阈值是 inf（无样本被预测为正）"
assert round(float(thr[j]), 4) == 0.2455
assert round(float(tpr[j]), 4) == 0.8077 and round(float(fpr[j]), 4) == 0.2660
assert round(recall_score(ys_te, (p1 >= 0.3).astype(int)), 4) == 0.6538
'''

E4_CODE = '''# @@todo(5) 训练 class_weight="balanced" 的逻辑回归，对比混淆矩阵
# @@hint LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lr_b = LogisticRegression(max_iter=1000, class_weight="balanced",
                          random_state=42).fit(Xtr_s, ys_tr)
yp_b = lr_b.predict(Xte_s)
cm_b = confusion_matrix(ys_te, yp_b)
# @@end
print("balanced 混淆矩阵:")
print(cm_b)
print("balanced 预测正例:", int(yp_b.sum()),
      "| recall:", round(recall_score(ys_te, yp_b), 4),
      "| acc:", round(accuracy_score(ys_te, yp_b), 4))

assert cm_b.tolist() == [[63, 31], [5, 21]], "FN 21→5，代价是 FP 5→31"
assert int(yp_b.sum()) == 52, "正例预测数 10 → 52"
assert round(recall_score(ys_te, yp_b), 4) == 0.8077
assert round(accuracy_score(ys_te, yp_b), 4) == 0.7000, "准确率换 recall，看业务要哪个"
'''

E5_CODE = '''# @@todo(6) 训练 KNN k=5 的 scaled 版与 raw 版，对账准确率与 precision
# @@hint KNeighborsClassifier(n_neighbors=5)；scaled 用 Xtr_s/Xte_s，raw 用 Xtr/Xte
knn5 = KNeighborsClassifier(n_neighbors=5).fit(Xtr_s, ys_tr)
knn5_raw = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ys_tr)
acc_s = accuracy_score(ys_te, knn5.predict(Xte_s))
acc_raw = accuracy_score(ys_te, knn5_raw.predict(Xte))
p_s = precision_score(ys_te, knn5.predict(Xte_s), zero_division=0)
p_raw = precision_score(ys_te, knn5_raw.predict(Xte), zero_division=0)
# @@end
print(f"KNN k=5 scaled: acc={acc_s:.4f} P={p_s:.4f}")
print(f"KNN k=5 raw   : acc={acc_raw:.4f} P={p_raw:.4f}")

for k in (3, 11):
    knn_k = KNeighborsClassifier(n_neighbors=k).fit(Xtr_s, ys_tr)
    print(f"KNN k={k} scaled: acc={accuracy_score(ys_te, knn_k.predict(Xte_s)):.4f} "
          f"R={recall_score(ys_te, knn_k.predict(Xte_s)):.4f}")

assert round(acc_s, 4) == 0.7500 and round(acc_raw, 4) == 0.8000
assert round(p_raw, 4) == 0.6250 and round(p_s, 4) == 0.3000, "raw 版误报反而更少"
assert round(recall_score(ys_te, knn5.predict(Xte_s)), 4) == 0.1154
'''

E6_CODE = '''# @@todo(7) 训练 scaled 与 raw 两版高斯朴素贝叶斯，验证预测逐位一致
# @@hint GaussianNB().fit(...) 两版；same = bool((gnb.predict(Xte_s) == gnb_raw.predict(Xte)).all())
gnb = GaussianNB().fit(Xtr_s, ys_tr)
gnb_raw = GaussianNB().fit(Xtr, ys_tr)
acc_s = accuracy_score(ys_te, gnb.predict(Xte_s))
acc_raw = accuracy_score(ys_te, gnb_raw.predict(Xte))
same = bool((gnb.predict(Xte_s) == gnb_raw.predict(Xte)).all())
# @@end
print(f"GNB scaled acc={acc_s:.4f} | raw acc={acc_raw:.4f} | 预测逐位一致: {same}")
print("GNB theta_[0] 前 3(类 0 均值):", np.round(gnb.theta_[0][:3], 4).tolist())

assert round(acc_s, 4) == 0.6750 and round(acc_raw, 4) == 0.6750
assert same, "逐列减均值除方差不改变高斯似然的 argmax"
assert round(float(gnb.theta_[0][0]), 4) == -0.0988
assert round(recall_score(ys_te, gnb.predict(Xte_s)), 4) == 0.4231, "NB 敢报正例"
'''

E7_CODE = '''# @@todo(8) 训练 LinearSVC 与 SVC（scaled/raw 各一），对账三份准确率
# @@hint LinearSVC(max_iter=10000, random_state=42)；SVC(random_state=42)
lsvc = LinearSVC(max_iter=10000, random_state=42).fit(Xtr_s, ys_tr)
svc = SVC(random_state=42).fit(Xtr_s, ys_tr)
svc_raw = SVC(random_state=42).fit(Xtr, ys_tr)
acc_l = accuracy_score(ys_te, lsvc.predict(Xte_s))
acc_v = accuracy_score(ys_te, svc.predict(Xte_s))
acc_r = accuracy_score(ys_te, svc_raw.predict(Xte))
# @@end
print(f"LinearSVC scaled: acc={acc_l:.4f} P={precision_score(ys_te, lsvc.predict(Xte_s), zero_division=0):.4f}")
print(f"SVC rbf scaled  : acc={acc_v:.4f}")
print(f"SVC rbf raw     : acc={acc_r:.4f} P={precision_score(ys_te, svc_raw.predict(Xte), zero_division=0):.4f} "
      f"R={recall_score(ys_te, svc_raw.predict(Xte)):.4f} ← 全预测负类躺平")

# @@todo(9) 验证 LinearSVC 没有 predict_proba；再用 SVC(probability=True) 拿概率
# @@hint probe(getattr, lsvc, "predict_proba")；SVC(probability=True, random_state=42)
st_proba = probe(getattr, lsvc, "predict_proba")
svc_p = SVC(probability=True, random_state=42).fit(Xtr_s, ys_tr)
# @@end
print("LinearSVC.predict_proba →", st_proba)
print("SVC(probability=True) proba shape:", svc_p.predict_proba(Xte_s).shape)

assert round(acc_l, 4) == 0.8000 and round(precision_score(ys_te, lsvc.predict(Xte_s), zero_division=0), 4) == 0.6250
assert round(acc_v, 4) == 0.7583
assert round(acc_r, 4) == 0.7833 and recall_score(ys_te, svc_raw.predict(Xte)) == 0.0
assert st_proba == ("err", "AttributeError"), "LinearSVC 没有 predict_proba"
assert svc_p.predict_proba(Xte_s).shape == (120, 2), "probability=True 走内部 CV 估算"
'''

E8_CODE = '''# @@todo(10) 训练全深决策树与 max_depth=5 的剪枝树，对比叶子数与 train/test acc
# @@hint DecisionTreeClassifier(random_state=42)；get_n_leaves()
dt_full = DecisionTreeClassifier(random_state=42).fit(Xtr_s, ys_tr)
dt5 = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xtr_s, ys_tr)
leaves_full = dt_full.get_n_leaves()
leaves5 = dt5.get_n_leaves()
train_acc = accuracy_score(ys_tr, dt_full.predict(Xtr_s))
acc5 = accuracy_score(ys_te, dt5.predict(Xte_s))
# @@end
print(f"DT 不限深: {leaves_full} 叶 train acc={train_acc:.4f} test={accuracy_score(ys_te, dt_full.predict(Xte_s)):.4f}")
print(f"DT d=5  : {leaves5} 叶 test acc={acc5:.4f} ← 全场准确率冠军")

# @@todo(11) 训练 RF(200 棵) 与 GBDT，读 RF 的特征重要性第一名
# @@hint RandomForestClassifier(n_estimators=200, random_state=42)；pd.Series(rf.feature_importances_, index=feat)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr_s, ys_tr)
gb = GradientBoostingClassifier(random_state=42).fit(Xtr_s, ys_tr)
acc_rf = accuracy_score(ys_te, rf.predict(Xte_s))
acc_gb = accuracy_score(ys_te, gb.predict(Xte_s))
imp_rf = pd.Series(rf.feature_importances_, index=feat).sort_values(ascending=False)
# @@end
print(f"RF test acc={acc_rf:.4f} | GB test acc={acc_gb:.4f}")
print("RF importance top4:", {k: round(v, 4) for k, v in imp_rf.head(4).items()})

assert leaves_full == 95 and round(train_acc, 4) == 1.0, "背表现场"
assert leaves5 == 18 and round(acc5, 4) == 0.7917
assert round(acc_rf, 4) == 0.7667 and round(acc_gb, 4) == 0.7750
assert round(float(imp_rf.iloc[0]), 4) == 0.2220 and imp_rf.index[0] == "温度"
'''

E9_CODE = '''# @@todo(12) 用三份 predict_proba 算 AUC，验证逻辑回归反超集成
# @@hint roc_auc_score(ys_te, proba[:, 1])；rf/gb 用 predict_proba(Xte_s)[:, 1]
auc_lr = roc_auc_score(ys_te, proba[:, 1])
auc_rf = roc_auc_score(ys_te, rf.predict_proba(Xte_s)[:, 1])
auc_gb = roc_auc_score(ys_te, gb.predict_proba(Xte_s)[:, 1])
# @@end
print(f"AUC: LR={auc_lr:.4f} > RF={auc_rf:.4f} > GB={auc_gb:.4f}")
print("复盘：准确率上 LR 与哑模型打平、树系反而更高，但 AUC 说 LR 的分数最会排序")

assert round(auc_lr, 4) == 0.7946
assert round(auc_rf, 4) == 0.6991 and round(auc_gb, 4) == 0.6809
assert auc_lr > auc_rf, "不平衡数据上 AUC 比准确率更会讲故事"
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch05 讲解：分类模型

任务：在 `defect_ml.csv`（600×11，正例 21.3%）上预测设备缺陷**是否危急**。
六族分类器 + 一套不平衡评估工具（混淆矩阵 / P / R / F1 / AUC / 阈值）。

本章把三个最常见的翻车现场摆在一起：

- **准确率 0.7833 的逻辑回归，和「全预测负类」的哑模型打平**
- **SVC 不 scale 就躺平**（R=0，一个正例都不敢报）
- **全深决策树 train acc=1.0，test 只有 0.65**

> 本章 `assert` 真值全部沉淀在方向 README 的「ch05 专项真值」表里。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 模型训练 40% | 六族分类器 fit/predict 按步给分 | 逻辑回归 / KNN / 朴素贝叶斯 / SVM / DT / RF+GB |
| 模型性能评估 10% | 不平衡指标的选型与解读 | 混淆矩阵 / P / R / F1 / AUC / 阈值移动 / class_weight |

**学习目标**：跑通六族分类器并报出 test 指标；能解释「为什么准确率会骗人」
「decision_function 和 predict_proba 什么关系」「哪些模型必须 scale、哪些免疫」。

## 二、API 速查表

| 类 / 函数 | 关键参数 | 一句话说明 |
|---|---|---|
| `LogisticRegression` | `max_iter` `class_weight` | 线性分类基线；`predict_proba` / `decision_function` |
| `KNeighborsClassifier` | `n_neighbors` | 惰性学习；**量纲敏感** |
| `GaussianNB` | — | 高斯朴素贝叶斯；**仿射缩放免疫**、训练最快 |
| `LinearSVC` | `max_iter` | 线性 SVM；**无 `predict_proba`** |
| `SVC` | `kernel` `gamma` `probability` | RBF 核；raw 输入会躺平；`probability=True` 才有概率 |
| `DecisionTreeClassifier` | `max_depth` | 不限深=背表 |
| `RandomForestClassifier` / `GradientBoostingClassifier` | `n_estimators` | 集成两兄弟；`feature_importances_` |
| `DummyClassifier` | `strategy` | 哑基线：先用它戳穿准确率幻觉 |
| `confusion_matrix` / `precision_score` / `recall_score` / `f1_score` | — | 不平衡四件套 |
| `roc_curve` / `roc_auc_score` | — | 阈值扫描与排序能力 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 `LogisticRegression`：只敢报 10 个危急

逻辑回归学的是 **log-odds**：`decision_function` 就是 `log(p/(1-p))`，
`predict_proba` 是它的 sigmoid。默认阈值 0.5 在 21% 正例上非常保守——
120 个测试样本里只敢报 10 个正例，漏掉 21 个。
"""
    ),
    code(S1_CODE),
    md(
        """
### 3.2 难点深挖：准确率陷阱

**为什么难**：94 个负例 / 120 个测试样本，「全预测负类」就有 78.33% 准确率。
逻辑回归同样 0.7833——**单看准确率，你无法分辨「学到了」和「什么都没学」**。

**正误对照**：

| 模型 | acc | precision | recall | 结论 |
|---|---|---|---|---|
| `DummyClassifier(most_frequent)` | 0.7833 | 0.0 | 0.0 | 什么都没学 |
| `LogisticRegression` | 0.7833 | 0.5 | 0.1923 | 真的学到了（F1=0.2778） |

**判定规则**：**不平衡任务先看混淆矩阵和 F1 / AUC，准确率只配做参考**。
竞赛里评委问「你的模型比全预测负类好在哪」，答不出 recall 就是没答上。
"""
    ),
    code(S2_CODE),
    md(
        """
## 3.3 阈值与 ROC：分数比标签更有用

`predict` 把 0.5 一刀切。降低阈值 recall 立刻上来（0.3 → 0.6538）；
`roc_curve` 扫出全部 39 个阈值，Youden J（TPR−FPR 最大）给出最优工作点 0.2455。
"""
    ),
    code(S3_CODE),
    md(
        """
## 3.4 `class_weight="balanced"`：在训练端改代价

阈值移动是在**预测端**调，`class_weight` 是在**训练端**给少数类损失加倍。
效果类似但原理不同：FN 21→5，代价是 FP 5→31，准确率 0.7833→0.7000。
"""
    ),
    code(S4_CODE),
    md(
        """
### 3.5 KNN：距离被量纲绑架

KNN 靠距离投票，大量纲特征独裁距离——和树**相反**（树对单调变换免疫）。
本数据 raw 版 acc 0.8000 反而比 scaled 0.7500 高、误报更少（P=0.625），
说明 scale 不是万能改善，但 **KNN 不做 scale 的结果不可解释、不可控**。
"""
    ),
    code(S5_CODE),
    md(
        """
### 3.6 朴素贝叶斯：仿射缩放免疫

高斯朴素贝叶斯的似然是逐列高斯密度。对某列做「减均值除方差」，
密度函数跟着平移缩放，**argmax 不变**——预测逐位一致。
所以 GNB 不需要标准化；但它的「特征条件独立」假设在本数据上并不成立，
acc 0.6750 是六族里最低的一档，换来的是训练几乎零成本。
"""
    ),
    code(S6_CODE),
    md(
        """
## 3.7 SVM：LinearSVC 没有概率，RBF 核 raw 版躺平

三个独立事实：`LinearSVC` **没有 `predict_proba`**（要用概率换
`SVC(probability=True)`，内部走 CV 估算）；RBF 核对量纲极其敏感，
raw 输入直接躺平（R=0）；线性场景下 `LinearSVC`（0.8000）可以比
逻辑回归（0.7833）更高——两者损失函数不同，不是同一把尺子。
"""
    ),
    code(S7_CODE),
    md(
        """
## 3.8 树与集成：背表现场与准确率冠军

不限深的决策树 95 片叶子把训练段背熟（train acc=1.0），test 只有 0.6500；
`max_depth=5` 剪到 18 叶，test 0.7917——**全场准确率冠军**。
RF/GB 在 0.7667 / 0.7750，特征重要性第一名是温度（0.2220）。
"""
    ),
    code(S8_CODE),
    md(
        """
## 3.9 AUC 复盘：谁的分数最会排序

准确率打平哑模型的逻辑回归，AUC 0.7946 反超 RF（0.6991）和 GB（0.6809）——
AUC 衡量「随机抽一个正例和一个负例，正例分数更高」的概率，
与阈值无关、与类别先验无关。**不平衡数据上，选模型看 AUC，报成绩看 F1**。
"""
    ),
    code(S9_CODE),
    md(
        """
## 四、本章小结

| 事实 | 数字 | 出处 |
|---|---|---|
| 默认阈值多保守 | 预测正例 10 vs 实际 26，FN=21 | §3.1 |
| 准确率幻觉 | 哑模型 = 逻辑回归 = 0.7833 | §3.2 |
| 阈值是免费旋钮 | t=0.3 → recall 0.6538；Youden 最优 0.2455 | §3.3 |
| 训练端调权重 | balanced 后 FN 21→5、FP 5→31 | §3.4 |
| KNN 量纲敏感 | raw 0.8000 vs scaled 0.7500，且更准 | §3.5 |
| GNB 缩放免疫 | 预测逐位一致（0.6750） | §3.6 |
| LinearSVC 无概率 | `predict_proba` → AttributeError | §3.7 |
| 剪枝才有钱赚 | 全深 0.6500 → d5 0.7917 冠军 | §3.8 |
| AUC 反超 | LR 0.7946 > RF 0.6991 > GB 0.6809 | §3.9 |

### 自测五问

1. 题 1 里模型只敢报 10 个正例，三种补救手段（阈值 / class_weight / 换模型）分别动的是什么？
2. 题 2 的哑模型和逻辑回归 acc 相同，你如何在 30 秒内向评委证明后者更有用？
3. 题 5 KNN raw 版反而更准，scale 到底该不该做？
4. 题 7 为什么 `SVC(probability=True)` 有概率而 `LinearSVC` 死活没有？
5. 题 8 全场准确率冠军是 max_depth=5 的单树——这说明 RF/GB「更强」错在哪？

全部答得上来，进入 ch06（模型评估与交叉验证）。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch05 练习：分类模型

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 模型训练 40% | 逻辑回归 / KNN / 朴素贝叶斯 / SVM / DT / RF+GB 的 fit-predict |
| 模型性能评估 10% | 混淆矩阵 / P / R / F1 / AUC / 阈值 / class_weight |

## 二、API 速查

| 类 | 关键参数 | 一句话 |
|---|---|---|
| `LogisticRegression(max_iter)` | `class_weight` | 线性基线，有概率输出 |
| `DummyClassifier(strategy)` | — | 哑基线，戳穿准确率幻觉 |
| `KNeighborsClassifier(n_neighbors)` | — | 量纲敏感 |
| `GaussianNB` | — | 缩放免疫 |
| `LinearSVC` / `SVC(probability)` | `gamma` | 前者无概率 |
| `DecisionTreeClassifier(max_depth)` | — | 不限深=背表 |
| `roc_curve` / `roc_auc_score` | — | 阈值扫描 / 排序能力 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：逻辑回归与概率输出（对应讲解 §3.1）"),
    code(E1_CODE),
    md(
        """
**为什么这样做**：`decision_function == logit(p)` 是逻辑回归的「身份证」——
验证它，你就确认了 proba 不是黑盒输出，阈值移动才有意义。
"""
    ),
    md("## 题 2：准确率陷阱（对应讲解 §3.2）"),
    code(E2_CODE),
    md("## 题 3：阈值移动与 Youden J（对应讲解 §3.3）"),
    code(E3_CODE),
    md("## 题 4：class_weight（对应讲解 §3.4）"),
    code(E4_CODE),
    md("## 题 5：KNN 与量纲（对应讲解 §3.5）"),
    code(E5_CODE),
    md("## 题 6：朴素贝叶斯缩放免疫（对应讲解 §3.6）"),
    code(E6_CODE),
    md("## 题 7：SVM 三兄弟（对应讲解 §3.7）"),
    code(E7_CODE),
    md("## 题 8：树与集成（对应讲解 §3.8）"),
    code(E8_CODE),
    md("## 题 9：AUC 复盘（对应讲解 §3.9）"),
    code(E9_CODE),
    md(
        """
## 收官自查

1. 题 1 的 FN=21 说明什么？如果你是运检班组长，敢直接上线这个模型吗？
2. 题 3 的 Youden 阈值 0.2455 换来 TPR 0.8077、FPR 0.2660，这批 FP 的业务代价谁承担？
3. 题 5 与题 6 结论相反（KNN 敏感、GNB 免疫），一句话说清分界线在哪。
4. 题 7 raw 版 SVC 的 P=R=0 与题 2 哑模型的 P=R=0 是同一种躺平吗？
5. 题 9 为什么 LR 的 AUC 能反超树系集成？AUC 到底在量什么？

全部答得上来，进入 ch06（模型评估与交叉验证）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
