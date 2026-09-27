"""sk07 —— 分类评估：classification_report / average 语义 / PR 曲线 / 概率质量

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk07_eval_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch07_eval_classification"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, classification_report,
                             cohen_kappa_score, f1_score, fbeta_score, log_loss,
                             matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA = Path("data")

dev = pd.read_csv(DATA / "defect_ml.csv")
NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]

# ---- 任务 A（多分类）：用 5 个数值特征预测 设备类型 ----
Xs_tr, Xs_te, ym_tr, ym_te = train_test_split(dev[NUM], dev["设备类型"],
                                              test_size=0.2, random_state=42,
                                              stratify=dev["设备类型"])

# ---- 任务 B（二分类）：预测 是否危急（与 ch05 同款，但只用数值特征）----
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(dev[NUM], dev["是否危急"],
                                              test_size=0.2, random_state=42,
                                              stratify=dev["是否危急"])

# 预处理（median 填充 + 标准化），两任务共用同一套统计量
imp = SimpleImputer(strategy="median").fit(Xs_tr)
ss = StandardScaler().fit(imp.transform(Xs_tr))
Xm_tr, Xm_te = ss.transform(imp.transform(Xs_tr)), ss.transform(imp.transform(Xs_te))
Xb_tr2, Xb_te2 = ss.transform(imp.transform(Xb_tr)), ss.transform(imp.transform(Xb_te))

print("sklearn", sklearn.__version__)
print("任务A 多分类:", Xm_tr.shape, "5 类 | 任务B 二分类:", Xb_tr2.shape,
      "正例占比", round(float(yb_tr.mean()), 4))'''

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

S1_CODE = '''# 任务 A：一个接近瞎猜的 5 分类器，恰好是练评估工具的好材料
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xm_tr, ym_tr)
ym_p = rf.predict(Xm_te)
acc_m = rf.score(Xm_te, ym_te)
proba_m = rf.predict_proba(Xm_te)
print("classes_:", rf.classes_.tolist(), "| n_classes:", rf.n_classes_)
print("test acc:", round(float(acc_m), 4), "（瞎猜基线 ≈ 0.25）")
print("proba shape:", proba_m.shape,
      "| 每行和为 1:", bool(np.allclose(proba_m.sum(axis=1), 1)))

assert rf.classes_.tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert rf.n_classes_ == 5 and proba_m.shape == (120, 5)
assert round(float(acc_m), 4) == 0.2667, "只比瞎猜好一点，但评估工具照常工作"
assert bool(np.allclose(proba_m.sum(axis=1), 1))
'''

S2_CODE = '''# average 三兄弟：同一个预测，三份成绩单
for avg in ("micro", "macro", "weighted"):
    print(f"{avg:>9}: P={precision_score(ym_te, ym_p, average=avg):.4f} "
          f"R={recall_score(ym_te, ym_p, average=avg):.4f} "
          f"F1={f1_score(ym_te, ym_p, average=avg):.4f}")

per_p = precision_score(ym_te, ym_p, average=None)
per_r = recall_score(ym_te, ym_p, average=None)
print("逐类 P:", np.round(per_p, 4).tolist())
print("逐类 R:", np.round(per_r, 4).tolist(), "← 变压器 0.55 vs 互感器 0.11")
print("micro P == accuracy:", precision_score(ym_te, ym_p, average="micro") == (ym_p == ym_te).mean())

assert np.round(per_p, 4).tolist() == [0.2, 0.3953, 0.16, 0.3333, 0.125]
assert np.round(per_r, 4).tolist() == [0.1053, 0.5484, 0.1667, 0.24, 0.1429]
assert round(f1_score(ym_te, ym_p, average="micro"), 4) == 0.2667
assert round(f1_score(ym_te, ym_p, average="macro"), 4) == 0.2346
assert round(f1_score(ym_te, ym_p, average="weighted"), 4) == 0.2547
assert round(precision_score(ym_te, ym_p, average="macro"), 4) == 0.2427, "macro < weighted：少数类拖后腿"
'''

S3_CODE = '''# classification_report：一张表读全，output_dict 才能进程序
rd = classification_report(ym_te, ym_p, output_dict=True)
print(classification_report(ym_te, ym_p, digits=4))
print("macro avg:", {k: round(v, 4) for k, v in rd["macro avg"].items() if k != "support"})
print("weighted avg:", {k: round(v, 4) for k, v in rd["weighted avg"].items() if k != "support"})
print("accuracy key:", round(rd["accuracy"], 4))
print("变压器行:", {k: round(v, 4) for k, v in rd["变压器"].items() if k != "support"})

assert round(rd["accuracy"], 4) == 0.2667
assert round(rd["macro avg"]["f1-score"], 4) == 0.2346
assert round(rd["weighted avg"]["f1-score"], 4) == 0.2547
assert round(rd["变压器"]["recall"], 4) == 0.5484
assert round(rd["互感器"]["precision"], 4) == 0.2
assert rd["互感器"]["support"] == 19 and rd["变压器"]["support"] == 31
'''

S4_CODE = '''# 多分类混淆矩阵：对角线是答对的，行内看「谁被认成了谁」
cm = pd.crosstab(ym_te, ym_p)
diag = pd.Series(np.diag(cm.to_numpy()), index=cm.index)
support = cm.sum(axis=1)
row_err = support - diag
worst = row_err.idxmax()
print(support.to_dict(), "← support 与 classification_report 一致")
print("误分最多的真实类:", worst, "误分", int(row_err[worst]), "/",
      int(support[worst]), "| 对角线和:", int(diag.sum()), "/", len(ym_te))

assert support.to_dict() == {"互感器": 19, "变压器": 31, "断路器": 24, "绝缘子": 25, "避雷器": 21}
assert worst == "断路器" and int(row_err[worst]) == 20, "24 个断路器错了 20 个"
assert int(diag.sum()) == 32, "对角线和 = 0.2667 * 120"
'''

S5_CODE = '''# 任务 B：概率质量——log_loss 与 Brier 分数惩罚「自信的错误」
lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xb_tr2, yb_tr)
rf2 = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xb_tr2, yb_tr)
p_lr = lr.predict_proba(Xb_te2)[:, 1]
p_rf = rf2.predict_proba(Xb_te2)[:, 1]
print(f"log_loss: LR={log_loss(yb_te, p_lr):.4f} | RF={log_loss(yb_te, p_rf):.4f} "
      f"| 全 0.5 基线={log_loss(yb_te, np.full(len(yb_te), 0.5)):.4f}")
print(f"brier   : LR={brier_score_loss(yb_te, p_lr):.4f} | RF={brier_score_loss(yb_te, p_rf):.4f}")
print("解读: RF 的 AUC 不差，但概率质量更差——树的投票概率天生过冲")

ll_lr = round(float(log_loss(yb_te, p_lr)), 4)
ll_rf = round(float(log_loss(yb_te, p_rf)), 4)
assert ll_lr == 0.4554 and ll_rf == 0.5255
assert round(float(log_loss(yb_te, np.full(len(yb_te), 0.5))), 4) == 0.6931, "ln(2)：瞎猜均匀分布的代价"
assert round(float(brier_score_loss(yb_te, p_lr)), 4) == 0.1458
assert round(float(brier_score_loss(yb_te, p_rf)), 4) == 0.1679
assert ll_rf > ll_lr, "RF 概率校准更差，log_loss 会说真话"
'''

S6_CODE = '''# PR-AUC（average_precision）：不平衡任务比 ROC-AUC 更挑剔的尺
auc_lr = roc_auc_score(yb_te, p_lr)
auc_rf = roc_auc_score(yb_te, p_rf)
ap_lr = average_precision_score(yb_te, p_lr)
ap_rf = average_precision_score(yb_te, p_rf)
print(f"ROC-AUC: LR={auc_lr:.4f} | RF={auc_rf:.4f}")
print(f"PR-AUC : LR={ap_lr:.4f} | RF={ap_rf:.4f} | 正例先验={float(yb_te.mean()):.4f}")
print("解读: 先验 0.2167 是 PR-AUC 的『瞎猜线』，ROC-AUC 的瞎猜线是 0.5")

assert round(float(auc_lr), 4) == 0.7406 and round(float(auc_rf), 4) == 0.6641
assert round(float(ap_lr), 4) == 0.4931 and round(float(ap_rf), 4) == 0.372
assert round(float(yb_te.mean()), 4) == 0.2167
assert ap_lr > 2 * float(yb_te.mean()), "PR-AUC 2 倍于先验才算真有排序能力"
'''

S7_CODE = '''# precision_recall_curve：长度陷阱与曲线上的 F1 最优点
prec, rec, thr = precision_recall_curve(yb_te, p_lr)
print("长度: prec", len(prec), "| rec", len(rec), "| thr", len(thr),
      "→ len(prec) = len(thr) + 1")
print("曲线末点 (prec, rec):", round(float(prec[-1]), 4), round(float(rec[-1]), 4),
      "← 阈值最低处：全部判正")

f1s = 2 * prec * rec / (prec + rec)
j = int(np.nanargmax(f1s[:-1]))
print(f"曲线 F1 最优: t={thr[j]:.4f} P={prec[j]:.4f} R={rec[j]:.4f} F1={f1s[j]:.4f}")

assert len(prec) == 121 and len(rec) == 121 and len(thr) == 120
assert round(float(prec[-1]), 4) == 1.0 and round(float(rec[-1]), 4) == 0.0
assert round(float(thr[j]), 4) == 0.2186
assert round(float(prec[j]), 4) == 0.4 and round(float(rec[j]), 4) == 0.6923
assert round(float(f1s[j]), 4) == 0.507
'''

S8_CODE = '''# fbeta 家族与「单一数值」三兄弟：beta 拉锯、MCC 看全部四个格子
yb_p = lr.predict(Xb_te2)
f2 = fbeta_score(yb_te, yb_p, beta=2)
f05 = fbeta_score(yb_te, yb_p, beta=0.5)
f1 = f1_score(yb_te, yb_p)
print(f"beta 拉锯: f0.5={f05:.4f} > f1={f1:.4f} > f2={f2:.4f}（漏检贵选 f2，误报贵选 f0.5）")
print(f"balanced_accuracy={balanced_accuracy_score(yb_te, yb_p):.4f} "
      f"| MCC={matthews_corrcoef(yb_te, yb_p):.4f} | kappa={cohen_kappa_score(yb_te, yb_p):.4f}")

assert round(float(f2), 4) == 0.3846 and round(float(f05), 4) == 0.5769
assert round(float(f1), 4) == 0.4615
assert round(float(balanced_accuracy_score(yb_te, yb_p)), 4) == 0.6518
assert round(float(matthews_corrcoef(yb_te, yb_p)), 4) == 0.4024
assert round(float(cohen_kappa_score(yb_te, yb_p)), 4) == 0.3706
assert f05 > f1 > f2, "beta 越大越看重 recall，而本模型 recall 短板"
'''


E1_CODE = '''# @@todo(1) 训练 5 分类随机森林，报 acc、类别顺序与概率矩阵形状
# @@hint RandomForestClassifier(n_estimators=200, random_state=42).fit(Xm_tr, ym_tr)；rf.score(Xm_te, ym_te)
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xm_tr, ym_tr)
ym_p = rf.predict(Xm_te)
acc_m = rf.score(Xm_te, ym_te)
proba_m = rf.predict_proba(Xm_te)
# @@end
print("classes_:", rf.classes_.tolist(), "| n_classes:", rf.n_classes_)
print("test acc:", round(float(acc_m), 4), "（瞎猜基线 ≈ 0.25）")
print("proba shape:", proba_m.shape,
      "| 每行和为 1:", bool(np.allclose(proba_m.sum(axis=1), 1)))

assert rf.classes_.tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert rf.n_classes_ == 5 and proba_m.shape == (120, 5)
assert round(float(acc_m), 4) == 0.2667, "只比瞎猜好一点，但评估工具照常工作"
assert bool(np.allclose(proba_m.sum(axis=1), 1))
'''

E2_CODE = '''# 循环骨架不用挖空，挖空的是循环体
for avg in ("micro", "macro", "weighted"):
    # @@todo(2) 用当前 average 档位打印 P / R / F1（一行搞定）
    # @@hint precision_score(ym_te, ym_p, average=avg) / recall_score(...) / f1_score(...)
    print(f"{avg:>9}: P={precision_score(ym_te, ym_p, average=avg):.4f} R={recall_score(ym_te, ym_p, average=avg):.4f} F1={f1_score(ym_te, ym_p, average=avg):.4f}")
    # @@end

# @@todo(3) 拿逐类分数：average=None 返回逐类数组
# @@hint per_p = precision_score(ym_te, ym_p, average=None)
per_p = precision_score(ym_te, ym_p, average=None)
per_r = recall_score(ym_te, ym_p, average=None)
# @@end
print("逐类 P:", np.round(per_p, 4).tolist())
print("逐类 R:", np.round(per_r, 4).tolist(), "← 变压器 0.55 vs 互感器 0.11")
print("micro P == accuracy:", precision_score(ym_te, ym_p, average="micro") == (ym_p == ym_te).mean())

assert np.round(per_p, 4).tolist() == [0.2, 0.3953, 0.16, 0.3333, 0.125]
assert np.round(per_r, 4).tolist() == [0.1053, 0.5484, 0.1667, 0.24, 0.1429]
assert round(f1_score(ym_te, ym_p, average="micro"), 4) == 0.2667
assert round(f1_score(ym_te, ym_p, average="macro"), 4) == 0.2346
assert round(f1_score(ym_te, ym_p, average="weighted"), 4) == 0.2547
assert round(precision_score(ym_te, ym_p, average="macro"), 4) == 0.2427, "macro < weighted：少数类拖后腿"
'''

E3_CODE = '''# @@todo(4) classification_report 的 output_dict 形态：取 macro/weighted avg 与 accuracy
# @@hint rd = classification_report(ym_te, ym_p, output_dict=True)；键 "macro avg" / "weighted avg" / "accuracy"
rd = classification_report(ym_te, ym_p, output_dict=True)
# @@end
print(classification_report(ym_te, ym_p, digits=4))
print("macro avg:", {k: round(v, 4) for k, v in rd["macro avg"].items() if k != "support"})
print("weighted avg:", {k: round(v, 4) for k, v in rd["weighted avg"].items() if k != "support"})
print("accuracy key:", round(rd["accuracy"], 4))
print("变压器行:", {k: round(v, 4) for k, v in rd["变压器"].items() if k != "support"})

assert round(rd["accuracy"], 4) == 0.2667
assert round(rd["macro avg"]["f1-score"], 4) == 0.2346
assert round(rd["weighted avg"]["f1-score"], 4) == 0.2547
assert round(rd["变压器"]["recall"], 4) == 0.5484
assert round(rd["互感器"]["precision"], 4) == 0.2
assert rd["互感器"]["support"] == 19 and rd["变压器"]["support"] == 31
'''

E4_CODE = '''# @@todo(5) 用 crosstab 建混淆矩阵，找 support、误分最多的真实类与对角线和
# @@hint cm = pd.crosstab(ym_te, ym_p)；diag = np.diag(cm.to_numpy())；row_err = support - diag
cm = pd.crosstab(ym_te, ym_p)
diag = pd.Series(np.diag(cm.to_numpy()), index=cm.index)
support = cm.sum(axis=1)
row_err = support - diag
worst = row_err.idxmax()
# @@end
print(support.to_dict(), "← support 与 classification_report 一致")
print("误分最多的真实类:", worst, "误分", int(row_err[worst]), "/",
      int(support[worst]), "| 对角线和:", int(diag.sum()), "/", len(ym_te))

assert support.to_dict() == {"互感器": 19, "变压器": 31, "断路器": 24, "绝缘子": 25, "避雷器": 21}
assert worst == "断路器" and int(row_err[worst]) == 20, "24 个断路器错了 20 个"
assert int(diag.sum()) == 32, "对角线和 = 0.2667 * 120"
'''

E5_CODE = '''# @@todo(6) 训练二分类 LR 与 RF（只用数值特征），对比 log_loss 与 Brier
# @@hint lr.fit(Xb_tr2, yb_tr)；p = model.predict_proba(Xb_te2)[:, 1]；log_loss(yb_te, p)
lr = LogisticRegression(max_iter=1000, random_state=42).fit(Xb_tr2, yb_tr)
rf2 = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xb_tr2, yb_tr)
p_lr = lr.predict_proba(Xb_te2)[:, 1]
p_rf = rf2.predict_proba(Xb_te2)[:, 1]
# @@end
print(f"log_loss: LR={log_loss(yb_te, p_lr):.4f} | RF={log_loss(yb_te, p_rf):.4f} "
      f"| 全 0.5 基线={log_loss(yb_te, np.full(len(yb_te), 0.5)):.4f}")
print(f"brier   : LR={brier_score_loss(yb_te, p_lr):.4f} | RF={brier_score_loss(yb_te, p_rf):.4f}")
print("解读: RF 的 AUC 不差，但概率质量更差——树的投票概率天生过冲")

ll_lr = round(float(log_loss(yb_te, p_lr)), 4)
ll_rf = round(float(log_loss(yb_te, p_rf)), 4)
assert ll_lr == 0.4554 and ll_rf == 0.5255
assert round(float(log_loss(yb_te, np.full(len(yb_te), 0.5))), 4) == 0.6931, "ln(2)：瞎猜均匀分布的代价"
assert round(float(brier_score_loss(yb_te, p_lr)), 4) == 0.1458
assert round(float(brier_score_loss(yb_te, p_rf)), 4) == 0.1679
assert ll_rf > ll_lr, "RF 概率校准更差，log_loss 会说真话"
'''

E6_CODE = '''# @@todo(7) 算两模型的 ROC-AUC 与 PR-AUC，并和正例先验对账
# @@hint roc_auc_score(yb_te, p_lr)；average_precision_score(yb_te, p_lr)
auc_lr = roc_auc_score(yb_te, p_lr)
auc_rf = roc_auc_score(yb_te, p_rf)
ap_lr = average_precision_score(yb_te, p_lr)
ap_rf = average_precision_score(yb_te, p_rf)
# @@end
print(f"ROC-AUC: LR={auc_lr:.4f} | RF={auc_rf:.4f}")
print(f"PR-AUC : LR={ap_lr:.4f} | RF={ap_rf:.4f} | 正例先验={float(yb_te.mean()):.4f}")
print("解读: 先验 0.2167 是 PR-AUC 的『瞎猜线』，ROC-AUC 的瞎猜线是 0.5")

assert round(float(auc_lr), 4) == 0.7406 and round(float(auc_rf), 4) == 0.6641
assert round(float(ap_lr), 4) == 0.4931 and round(float(ap_rf), 4) == 0.372
assert round(float(yb_te.mean()), 4) == 0.2167
assert ap_lr > 2 * float(yb_te.mean()), "PR-AUC 2 倍于先验才算真有排序能力"
'''

E7_CODE = '''# @@todo(8) 取 precision_recall_curve 三件套，验证长度关系并找曲线上的 F1 最优点
# @@hint prec, rec, thr = precision_recall_curve(yb_te, p_lr)；f1s = 2*prec*rec/(prec+rec)；np.nanargmax(f1s[:-1])
prec, rec, thr = precision_recall_curve(yb_te, p_lr)
f1s = 2 * prec * rec / (prec + rec)
j = int(np.nanargmax(f1s[:-1]))
# @@end
print("长度: prec", len(prec), "| rec", len(rec), "| thr", len(thr),
      "→ len(prec) = len(thr) + 1")
print("曲线末点 (prec, rec):", round(float(prec[-1]), 4), round(float(rec[-1]), 4),
      "← 阈值最低处：全部判正")
print(f"曲线 F1 最优: t={thr[j]:.4f} P={prec[j]:.4f} R={rec[j]:.4f} F1={f1s[j]:.4f}")

assert len(prec) == 121 and len(rec) == 121 and len(thr) == 120
assert round(float(prec[-1]), 4) == 1.0 and round(float(rec[-1]), 4) == 0.0
assert round(float(thr[j]), 4) == 0.2186
assert round(float(prec[j]), 4) == 0.4 and round(float(rec[j]), 4) == 0.6923
assert round(float(f1s[j]), 4) == 0.507
'''

E8_CODE = '''# @@todo(9) 算 fbeta 三档（0.5/1/2）、balanced_accuracy、MCC、kappa
# @@hint fbeta_score(yb_te, yb_p, beta=2)；matthews_corrcoef / cohen_kappa_score / balanced_accuracy_score
yb_p = lr.predict(Xb_te2)
f2 = fbeta_score(yb_te, yb_p, beta=2)
f05 = fbeta_score(yb_te, yb_p, beta=0.5)
f1 = f1_score(yb_te, yb_p)
# @@end
print(f"beta 拉锯: f0.5={f05:.4f} > f1={f1:.4f} > f2={f2:.4f}（漏检贵选 f2，误报贵选 f0.5）")
print(f"balanced_accuracy={balanced_accuracy_score(yb_te, yb_p):.4f} "
      f"| MCC={matthews_corrcoef(yb_te, yb_p):.4f} | kappa={cohen_kappa_score(yb_te, yb_p):.4f}")

assert round(float(f2), 4) == 0.3846 and round(float(f05), 4) == 0.5769
assert round(float(f1), 4) == 0.4615
assert round(float(balanced_accuracy_score(yb_te, yb_p)), 4) == 0.6518
assert round(float(matthews_corrcoef(yb_te, yb_p)), 4) == 0.4024
assert round(float(cohen_kappa_score(yb_te, yb_p)), 4) == 0.3706
assert f05 > f1 > f2, "beta 越大越看重 recall，而本模型 recall 短板"
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch07 讲解：分类评估

ch05 讲「模型敢不敢报正例」，本章讲「**指标本身怎么算、怎么选、怎么读**」。
两个任务：任务 A 用 5 个数值特征做 5 分类（预测设备类型，模型接近瞎猜——
**正好当评估工具的标定板**）；任务 B 复用是否危急的二分类，聚焦概率质量。

本章四个必考现场：

- **micro == accuracy，macro < weighted**——average 参数的三种人格
- **RF 的 AUC 不差但 log_loss 更差**——概率质量与排序能力是两回事
- **`precision_recall_curve` 的 prec 比 thr 长 1**——末点 (1.0, 0.0) 是坑
- **f0.5 > f1 > f2**——beta 拉锯暴露 recall 短板

> 本章 `assert` 真值全部沉淀在方向 README 的「ch07 专项真值」表里。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 模型性能评估 10% | 指标选型、计算与解读 | average 语义 / classification_report / 混淆矩阵 / log_loss / Brier / PR 曲线 / F-beta / MCC |

**学习目标**：给任意分类结果配一套完整成绩单；能解释「为什么 micro 就是
accuracy」「PR-AUC 的瞎猜线为什么不是 0.5」「beta 怎么选」。

## 二、API 速查表

| 类 / 函数 | 关键参数 | 一句话说明 |
|---|---|---|
| `precision_score` / `recall_score` / `f1_score` | `average` `zero_division` | micro=accuracy / macro=类平均 / weighted=按 support 加权 |
| `classification_report` | `digits` `output_dict` | 一张表读全；`output_dict=True` 才能进程序 |
| `log_loss` | — | 惩罚「自信的错误」，全 0.5 = ln2 |
| `brier_score_loss` | — | 概率均方误差，越低校准越好 |
| `average_precision_score` | — | PR-AUC；瞎猜线 = 正例先验 |
| `precision_recall_curve` | — | 返回 (prec, rec, thr)，**prec 长 1** |
| `fbeta_score` | `beta` | beta>1 偏向 recall，<1 偏向 precision |
| `matthews_corrcoef` / `cohen_kappa_score` | — | 不平衡任务的单一数值指标 |
| `balanced_accuracy_score` | — | 每类 recall 的平均 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 任务 A：接近瞎猜的 5 分类器

设备类型由特征几乎不可预测（acc 0.2667 vs 瞎猜 0.25）——这不是坏材料，
**评估工具不关心模型好坏**，正好用它标定每个指标的语义。
"""
    ),
    code(S1_CODE),
    md(
        """
### 3.2 难点深挖：average 三兄弟

**为什么难**：多分类下 `precision_score` 必须给 `average`，三种取值
语义完全不同，竞赛填错一个全盘皆输。

| average | 语义 | 本数据 F1 |
|---|---|---|
| `micro` | 全局统计 TP/FP/FN（等价 accuracy） | 0.2667 |
| `macro` | 每类算完再平均（**各类平权**） | 0.2346 |
| `weighted` | 按 support 加权（大类说了算） | 0.2547 |

**判定规则**：**不平衡任务报 macro（少数类不缺席），类别均衡报 micro/weighted
都行**；macro < weighted 是少数类拖后腿的信号，两者差距越大越不平衡。
"""
    ),
    code(S2_CODE),
    md(
        """
## 3.3 `classification_report`：给人看的表，给程序看的 dict

打印版用 `digits=4` 对齐小数；要进程序必须 `output_dict=True`——
返回 dict 里每类一行 + macro avg / weighted avg / accuracy 三个汇总键。
"""
    ),
    code(S3_CODE),
    md(
        """
## 3.4 多分类混淆矩阵：对角线是答对的

5×5 矩阵行是真实类、列是预测类。support 与 classification_report 完全一致；
行内最大误分格告诉你「谁被认成了谁」。本数据 24 个断路器错了 20 个。
"""
    ),
    code(S4_CODE),
    md(
        """
### 3.5 难点深挖：概率质量——AUC 骗不了 log_loss

**为什么难**：RF 的投票概率（每棵树投票占比）天生**过冲**——0/1 附近
堆积，边界样本不模糊。排序能力（AUC）尚可，但「自信的错误」被
log_loss / Brier 狠狠惩罚。

| 指标 | LogisticRegression | RandomForest | 全 0.5 基线 |
|---|---|---|---|
| ROC-AUC | **0.7406** | 0.6641 | 0.5 |
| log_loss | **0.4554** | 0.5255 | 0.6931 = ln2 |
| Brier | **0.1458** | 0.1679 | 0.25 |

**判定规则**：**模型分数只用来排序（阈值/Top-K）→ 看 AUC；分数本身要当
概率用（成本核算/风险定价）→ 必须看 log_loss 或 Brier**。两者背离时，
先查校准（`CalibratedClassifierCV`，ch09 再展开）。
"""
    ),
    code(S5_CODE),
    md(
        """
## 3.6 PR-AUC：不平衡任务上更挑剔的尺

ROC-AUC 的瞎猜线是 0.5；**PR-AUC 的瞎猜线是正例先验 0.2167**。
LR 的 AP=0.4931 是先验的 2.3 倍——不平衡任务上这比「AUC 0.74」更有说服力。
"""
    ),
    code(S6_CODE),
    md(
        """
### 3.7 难点深挖：`precision_recall_curve` 的长度陷阱

返回值 `prec`/`rec` 比 `thr` **长 1**：末点 (1.0, 0.0) 是「阈值无穷低、
全部判正」的锚点，没有对应阈值。拿 `thr` 直接索引 `prec` 不裁剪就越界。
裁掉末点后 F1 在 **t=0.2186** 最大（P=0.4，R=0.6923）——比 ROC 的
Youden 阈值更偏向正类召回。
"""
    ),
    code(S7_CODE),
    md(
        """
## 3.8 F-beta 拉锯与单一数值三兄弟

beta>1 偏向 recall（漏检贵），beta<1 偏向 precision（误报贵）。
本模型 recall 短板，所以 f0.5 > f1 > f2 单调排开——**beta 的选择本身就是
业务问题的答案**。MCC 同时看四个格子，不平衡场景比 F1 更稳。
"""
    ),
    code(S8_CODE),
    md(
        """
## 四、本章小结

| 事实 | 数字 | 出处 |
|---|---|---|
| micro == accuracy | 5 分类 acc = micro F1 = 0.2667 | §3.2 难点深挖 |
| macro < weighted | 0.2346 < 0.2547，少数类拖后腿 | §3.2 |
| 最差类 | 断路器 24 个错 20 个；互感器 R=0.1053 | §3.4 |
| 概率质量背离 | AUC LR 更高，log_loss LR 也更好；RF 过冲 | §3.5 难点深挖 |
| PR 瞎猜线 | 正例先验 0.2167，LR AP=0.4931 | §3.6 |
| PR 长度陷阱 | prec 121 / thr 120，末点 (1.0, 0.0) | §3.7 难点深挖 |
| beta 拉锯 | f0.5 0.5769 > f1 0.4615 > f2 0.3846 | §3.8 |

### 自测五问

1. 题 2 里 micro、macro、weighted 三份成绩单，评审该信哪份？为什么？
2. 题 5 的 RF「AUC 不差但 log_loss 差」——什么时候这个 RF 仍然可用？
3. 题 6 的 PR-AUC 0.4931 和 ROC-AUC 0.7406，哪个更能说明模型有用？
4. 题 7 的末点 (1.0, 0.0) 为什么没有阈值？直接 `prec[len(thr)]` 会怎样？
5. 运检业务「漏检贵」，题 8 该选哪个 beta？分数高低说明业务代价落在哪？

全部答得上来，进入 ch08（回归评估）。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch07 练习：分类评估

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 模型性能评估 10% | average 语义 / classification_report / 混淆矩阵 / log_loss / PR / F-beta / MCC |

## 二、API 速查

| 函数 | 关键参数 | 一句话 |
|---|---|---|
| `f1_score(average=)` | micro / macro / weighted | 三种人格 |
| `classification_report(output_dict=True)` | `digits` | 进程序用 dict |
| `log_loss` / `brier_score_loss` | — | 概率质量双尺 |
| `average_precision_score` | — | PR-AUC，瞎猜线=先验 |
| `precision_recall_curve` | — | prec 比 thr 长 1 |
| `fbeta_score(beta=)` | — | beta 拉锯 |
| `matthews_corrcoef` / `cohen_kappa_score` | — | 四格全看的单一数值 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：接近瞎猜的 5 分类器（对应讲解 §3.1）"),
    code(E1_CODE),
    md("## 题 2：average 三兄弟（对应讲解 §3.2）"),
    code(E2_CODE),
    md("## 题 3：classification_report 双形态（对应讲解 §3.3）"),
    code(E3_CODE),
    md("## 题 4：多分类混淆矩阵（对应讲解 §3.4）"),
    code(E4_CODE),
    md("## 题 5：概率质量——log_loss 与 Brier（对应讲解 §3.5）"),
    code(E5_CODE),
    md("## 题 6：PR-AUC 与先验基线（对应讲解 §3.6）"),
    code(E6_CODE),
    md("## 题 7：PR 曲线结构与 F1 最优阈值（对应讲解 §3.7）"),
    code(E7_CODE),
    md("## 题 8：F-beta 拉锯与 MCC/kappa（对应讲解 §3.8）"),
    code(E8_CODE),
    md(
        """
## 收官自查

1. 题 2 的 macro 0.2346 和 weighted 0.2547，差值在告诉你什么？
2. 题 4 的混淆矩阵行和 == classification_report 的 support——这个一致性说明两把工具共享什么输入？
3. 题 5 换成「只做 Top-K 预警」场景，log_loss 还重要吗？
4. 题 7 的 F1 最优阈值 0.2186 和 ch05 的 Youden 0.2455 为什么不同？两个目标在 trade 什么？
5. 题 8 的 MCC 0.4024 说明模型「有点用」——kappa 和 MCC 都修正了什么？

全部答得上来，进入 ch08（回归评估）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
