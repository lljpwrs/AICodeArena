#!/usr/bin/env python3
"""ch12 类别不平衡：class_weight / 阈值移动 / SMOTE / imblearn Pipeline（defect_ml.csv）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk12_imbalance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch12_imbalance"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data")
'''

SETUP = '''# 数据与划分口径沿用 ch05：分层 8:2、median 填充、独热、标准化
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("sklearn", sklearn.__version__, "| imblearn",
      __import__("imblearn").__version__)
print("训练正例:", int(ys_tr.sum()), "/", len(ys_tr),
      "| 测试正例:", int(ys_te.sum()), "/", len(ys_te))
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def report_cm(tag, yt, yp):
    """一行报 acc/P/R/f1 与混淆矩阵，返回 (acc, recall, f1)。"""
    acc = round(float(accuracy_score(yt, yp)), 4)
    rec = round(float(recall_score(yt, yp)), 4)
    f1v = round(float(f1_score(yt, yp, zero_division=0)), 4)
    print(f"{tag}: acc={acc} P={round(float(precision_score(yt, yp, zero_division=0)), 4)} "
          f"R={rec} f1={f1v} cm={confusion_matrix(yt, yp).tolist()}")
    return acc, rec, f1v


print("脚手架就绪：report_cm")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) 基线体检：默认阈值下 0.7833 的准确率是怎么来的
# @@hint base = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr)；report_cm("基线", ys_te, base.predict(Xte_s))
base = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr)
base_acc, base_rec, base_f1 = report_cm("基线 thr=0.5", ys_te, base.predict(Xte_s))
# @@end
print(f"全判负也能拿 {round(float(1 - ys_te.mean()), 4)}——基线只比『躺平』多抓住 {int(confusion_matrix(ys_te, base.predict(Xte_s))[1, 1])} 个正例")

assert base_acc == 0.7833
assert base_rec == 0.1923, "召回只有 0.19：26 个危急漏掉 21 个"
assert base_f1 == 0.2778, "f1 被 21% 正例拉到难看——这才是真实成绩单"
'''

E2_CODE = '''# @@todo(2) class_weight=balanced：按类频率反加权，FN 直接 21 → 5
# @@hint LogisticRegression(..., class_weight="balanced")
cw = LogisticRegression(max_iter=1000, random_state=42,
                        class_weight="balanced").fit(Xtr_s, ys_tr)
cw_acc, cw_rec, cw_f1 = report_cm("balanced", ys_te, cw.predict(Xte_s))
# @@end
print(f"acc {base_acc} → {cw_acc}，recall {base_rec} → {cw_rec}：用 8 个点准确率换 6 倍召回")

assert cw_acc == 0.7 and cw_rec == 0.8077, "FN 21 → 5"
assert cw_f1 == 0.5385, "f1 几乎翻倍（0.2778 → 0.5385）"
assert cw_acc < base_acc and cw_rec > base_rec, "训练时调的是损失，测试时看的指标必须换"
'''

E3_CODE = '''# @@todo(3) 自定义权重：{0:1, 1:4} 比全自动更进一步
# @@hint class_weight 传字典 {0: 1, 1: 4}
cw4 = LogisticRegression(max_iter=1000, random_state=42,
                         class_weight={0: 1, 1: 4}).fit(Xtr_s, ys_tr)
cw4_acc, cw4_rec, cw4_f1 = report_cm("w={0:1,1:4}", ys_te, cw4.predict(Xte_s))
# @@end
print("balanced 的隐式权重:", {0: round(float(len(ys_tr) / (2 * (len(ys_tr) - ys_tr.sum()))), 4),
                            1: round(float(len(ys_tr) / (2 * int(ys_tr.sum()))), 4)})

assert cw4_acc == 0.7 and cw4_rec == 0.8462 and cw4_f1 == 0.55, \\
    "手动 4 倍比 balanced 的自动 ~2.35 倍更激进，f1 再挤一点"
assert cw4_rec > cw_rec, "权重越大 → 正例损失越贵 → 模型越敢报正例"
'''

E4_CODE = '''# @@todo(4) 阈值移动：不重训模型，只动判决线
# @@hint p_te = base.predict_proba(Xte_s)[:, 1]；yp = (p_te >= t).astype(int)
p_te = base.predict_proba(Xte_s)[:, 1]
thr_acc, thr_rec, thr_f1 = report_cm("阈值 0.3", ys_te, (p_te >= 0.3).astype(int))
t02_cm = confusion_matrix(ys_te, (p_te >= 0.2).astype(int))
# @@end
print("阈值 0.2 的 cm:", t02_cm.tolist())
print("对照 balanced 的 cm:", confusion_matrix(ys_te, cw.predict(Xte_s)).tolist())

assert thr_acc == 0.7583 and thr_rec == 0.6538, "阈值 0.5 → 0.3：召回 0.19 → 0.65"
assert thr_f1 == 0.5397, "f1 0.5397 甚至超过 balanced 的 0.5385"
assert t02_cm.tolist() == confusion_matrix(ys_te, cw.predict(Xte_s)).tolist(), \\
    "阈值 0.2 的判决和 balanced 完全一致——class_weight 就是隐式的阈值移动"
'''

E5_CODE = '''# @@todo(5) SMOTE 正确用法：只对训练集插值造样本
# @@hint sm = SMOTE(random_state=42)；Xtr_sm, ys_tr_sm = sm.fit_resample(Xtr_s, ys_tr)
sm = SMOTE(random_state=42)
Xtr_sm, ys_tr_sm = sm.fit_resample(Xtr_s, ys_tr)
lr_sm = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_sm, ys_tr_sm)
sm_acc, sm_rec, sm_f1 = report_cm("SMOTE+LR", ys_te, lr_sm.predict(Xte_s))
# @@end
print("SMOTE 前:", Xtr_s.shape, "正例", int(ys_tr.sum()))
print("SMOTE 后:", Xtr_sm.shape, "正例", int(ys_tr_sm.sum()))

assert Xtr_sm.shape == (756, 16) and int(ys_tr_sm.sum()) == 378, \\
    "auto 策略补到 1:1：102 → 378 正例（新增 276 个合成点）"
assert sm_rec == 0.7692 and sm_acc == 0.6917, "SMOTE 效果与 balanced 一个档次，但多了造样本的随机性"
'''

E6_CODE = '''# @@todo(6) 经典作弊：在测试集上 SMOTE（错误示范，看数字怎么失真）
# @@hint 对 Xte_s 也 fit_resample 就污染了；结果只在 print 里围观，不许写进报告
Xte_sm, ys_te_sm = SMOTE(random_state=42).fit_resample(Xte_s, ys_te)
bad_acc = round(float(accuracy_score(ys_te_sm, lr_sm.predict(Xte_sm))), 4)
# @@end
print("作弊后的『acc』:", bad_acc, "（测试样本被人工补成 1:1，此数字作废）")
print("正确姿势: SMOTE 只碰训练集，测试集永远保持原始分布")

assert len(ys_te_sm) != len(ys_te), "测试集 120 → 188，已经不是同一道题了"
assert bad_acc != round(float(accuracy_score(ys_te, lr_sm.predict(Xte_s))), 4), \\
    "污染前后数字对不上——一旦混入提交结果就是事故"
'''

E7_CODE = '''# @@todo(7) imblearn Pipeline：CV 里防 SMOTE 泄漏的正确姿势
# @@hint ImbPipeline([("smote", SMOTE(random_state=42)), ("clf", LogisticRegression(max_iter=1000, random_state=42))])；cross_val_score(..., scoring="recall", cv=skf)
imb_pipe = ImbPipeline([("smote", SMOTE(random_state=42)),
                        ("clf", LogisticRegression(max_iter=1000, random_state=42))])
sm_sc = cross_val_score(imb_pipe, Xtr_s, ys_tr, cv=skf, scoring="recall")
sm_cv = round(float(sm_sc.mean()), 4)
plain_sc = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                           Xtr_s, ys_tr, cv=skf, scoring="recall")
plain_cv = round(float(plain_sc.mean()), 4)
# @@end
print("SMOTE 管道 CV recall 各折:", np.round(sm_sc, 4).tolist(), "| mean", sm_cv)
print("普通 LR  CV recall 各折:", np.round(plain_sc, 4).tolist(), "| mean", plain_cv)

assert sm_cv == 0.6067 and plain_cv == 0.3533, "SMOTE 在 CV 内平均把召回拉高 25 个点"
assert sm_cv > plain_cv, "imblearn 的 Pipeline 会在每折训练段内部重采样：验证折保持原味"
'''

E8_CODE = '''# @@todo(8) 三方案对表：不平衡数据选型一眼定
# @@hint 三个已拟合的模型 + 一条手工阈值，全部过 report_cm
acc_b, rec_b, f1_b = report_cm("balanced", ys_te, cw.predict(Xte_s))
acc_s, rec_s, f1_s = report_cm("SMOTE", ys_te, lr_sm.predict(Xte_s))
acc_t, rec_t, f1_t = report_cm("阈值 0.3", ys_te, (p_te >= 0.3).astype(int))
# @@end
print("选型口径: 漏报代价大 → 保 recall（balanced）；只要排序 → 原模型+移动阈值；"
      "要造样本 → SMOTE（且只在训练集）")

assert max(f1_b, f1_s, f1_t) == f1_t == 0.5397, "本数据上『不动模型只动阈值』f1 最高"
assert acc_t > acc_b > acc_s, "阈值移动连 acc 都保得住：0.7583 > 0.70 > 0.6917"
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 基线体检：0.7833 的准确率是什么成色
base = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_s, ys_tr)
base_acc, base_rec, base_f1 = report_cm("基线 thr=0.5", ys_te, base.predict(Xte_s))
print("全判负的躺平分:", round(float(1 - ys_te.mean()), 4))
print("解读: 模型只比躺平多抓 5 个正例，代价是漏 21 个——不平衡数据"
      "先看 recall/f1 再看 acc，否则 21.3% 的正例率就是你白捡的分")

assert (base_acc, base_rec, base_f1) == (0.7833, 0.1923, 0.2778)
'''

S2_CODE = '''# class_weight=balanced：损失函数里反加权（不改数据）
cw = LogisticRegression(max_iter=1000, random_state=42,
                        class_weight="balanced").fit(Xtr_s, ys_tr)
report_cm("balanced", ys_te, cw.predict(Xte_s))
print("解读: FN 21 → 5。balanced 的隐式权重 = n/(2*类计数)：负类 ~0.55、正类 ~2.35，"
      "正例错一次罚 4 倍多。零成本方案，竞赛首选")

assert round(float(accuracy_score(ys_te, cw.predict(Xte_s))), 4) == 0.7
assert round(float(recall_score(ys_te, cw.predict(Xte_s))), 4) == 0.8077
'''

S3_CODE = '''# 自定义权重：业务代价直接写进字典
cw4 = LogisticRegression(max_iter=1000, random_state=42,
                         class_weight={0: 1, 1: 4}).fit(Xtr_s, ys_tr)
report_cm("w={0:1,1:4}", ys_te, cw4.predict(Xte_s))
print("解读: 权重比 = 漏报代价比。变压器危急缺陷漏一次的代价远高于虚惊，"
      "给 1:4 甚至 1:10 都合理——把业务对话翻译成字典就交差了")

assert round(float(recall_score(ys_te, cw4.predict(Xte_s))), 4) == 0.8462
'''

S4_CODE = '''# 阈值移动：模型不动，判决线动
p_te = base.predict_proba(Xte_s)[:, 1]
for t in (0.5, 0.4, 0.3, 0.2, 0.15):
    report_cm(f"阈值 {t}", ys_te, (p_te >= t).astype(int))
print("阈值 0.2 的 cm == balanced 的 cm:",
      (confusion_matrix(ys_te, (p_te >= 0.2).astype(int)).tolist()
       == confusion_matrix(ys_te, cw.predict(Xte_s)).tolist()))
print("解读: 降阈值 = 越敢报正例。class_weight 本质就是『训练时的阈值移动』——"
      "两者二选一即可；阈值移动还有个好处：不用重训模型，交卷时按指标现调")

assert (confusion_matrix(ys_te, (p_te >= 0.2).astype(int)).tolist()
        == confusion_matrix(ys_te, cw.predict(Xte_s)).tolist())
'''

S5_CODE = '''# SMOTE：在少数类样本之间线性插值造新样本（只动训练集！）
sm = SMOTE(random_state=42)
Xtr_sm, ys_tr_sm = sm.fit_resample(Xtr_s, ys_tr)
lr_sm = LogisticRegression(max_iter=1000, random_state=42).fit(Xtr_sm, ys_tr_sm)
report_cm("SMOTE+LR", ys_te, lr_sm.predict(Xte_s))
print("SMOTE 前:", Xtr_s.shape, "正例", int(ys_tr.sum()), "→ 后:", Xtr_sm.shape,
      "正例", int(ys_tr_sm.sum()))
print("解读: auto 策略补到 1:1（102→378，新增 276 个合成点）；效果与 balanced 同档，"
      "但随机性更大、多一个要调的超参（k_neighbors）")

assert Xtr_sm.shape == (756, 16) and int(ys_tr_sm.sum()) == 378
assert round(float(recall_score(ys_te, lr_sm.predict(Xte_s))), 4) == 0.7692
'''

S6_CODE = '''# 错误示范：测试集 SMOTE = 人工改题
Xte_sm, ys_te_sm = SMOTE(random_state=42).fit_resample(Xte_s, ys_te)
bad_acc = round(float(accuracy_score(ys_te_sm, lr_sm.predict(Xte_sm))), 4)
print("作弊后的『acc』:", bad_acc, "| 真实 acc:",
      round(float(accuracy_score(ys_te, lr_sm.predict(Xte_s))), 4))
print("解读: 测试集 120 → 188（94:94），其中 68 个是合成的——指标再好看也是自娱自乐。"
      "铁律：**重采样永远只碰训练集**")

assert len(ys_te_sm) == 188 and len(ys_te) == 120, "测试集 120 → 188（94:94），已经不是同一道题了"
'''

S7_CODE = '''# imblearn Pipeline：CV 每折训练段内部重采样，验证折保持原味
imb_pipe = ImbPipeline([("smote", SMOTE(random_state=42)),
                        ("clf", LogisticRegression(max_iter=1000, random_state=42))])
sm_sc = cross_val_score(imb_pipe, Xtr_s, ys_tr, cv=skf, scoring="recall")
plain_sc = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                           Xtr_s, ys_tr, cv=skf, scoring="recall")
print("SMOTE 管道 CV recall:", np.round(sm_sc, 4).tolist(), "| mean", round(float(sm_sc.mean()), 4))
print("普通 LR  CV recall:", np.round(plain_sc, 4).tolist(), "| mean", round(float(plain_sc.mean()), 4))
print("解读: 如果先在全体训练数据上 SMOTE 再切折，验证折里就混进了训练折的合成样本——"
      "CV 分数虚高。imblearn 的 Pipeline 把 SMOTE 塞进每折内部，天然防泄漏")

assert round(float(sm_sc.mean()), 4) == 0.6067
assert round(float(plain_sc.mean()), 4) == 0.3533
'''

S8_CODE = '''# 三方案对表：同一个测试集，三种打法
report_cm("基线 thr=0.5", ys_te, base.predict(Xte_s))
report_cm("balanced", ys_te, cw.predict(Xte_s))
report_cm("SMOTE", ys_te, lr_sm.predict(Xte_s))
report_cm("阈值 0.3", ys_te, (p_te >= 0.3).astype(int))
print("解读: 本数据（样本少、正例 21%）上：阈值移动 f1 最高且 acc 保得住；"
      "balanced 最省事；SMOTE 效果同档但更重。**竞赛推荐顺序：class_weight → "
      "阈值移动 → SMOTE**")

assert round(float(f1_score(ys_te, (p_te >= 0.3).astype(int), zero_division=0)), 4) == 0.5397
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch12 类别不平衡：class_weight / 阈值移动 / SMOTE

> 数据：`defect_ml.csv`，正例占 21.3%（危急缺陷）。不平衡数据的三个处方：
> **改损失（class_weight）、改判决（阈值移动）、改数据（SMOTE）**。
>
> 竞赛评分点「模型性能评估」里，不平衡数据用错指标 = 白干。

**本章考点**

1. 准确率的遮羞布：基线 0.7833 vs 召回 0.1923
2. `class_weight="balanced"` 与自定义权重的隐式权重公式
3. 阈值移动与 class_weight 的等价性
4. SMOTE 的正确姿势：只碰训练集
5. 测试集重采样 = 作弊
6. `imblearn.Pipeline` 防 CV 泄漏

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| acc 遮羞布 | 21.3% 正例率白送的分 | §12.1 |
| balanced 真身 | 隐式权重 n/(2·类计数)，≈ 阈值移动 | §12.2/12.4 |
| SMOTE 泄漏 | 先重采样后切折 → 验证折混入合成样本 | §12.6 |
| 测试集重采样 | 指标全部作废 | §12.5 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 12.1 准确率的遮羞布

基线 LR acc 0.7833，看着比「全判负躺平」的 0.7867 差不太多——因为
21.3% 的正例率本来就白送。拆开看：26 个真危急只抓住 5 个（recall 0.1923），
f1 只有 0.2778。**不平衡数据先报 recall/f1，acc 是最不诚实的指标**。
"""
    ),
    code(S1_CODE),
    md(
        """
## 12.2 难点深挖：class_weight 的真身

`class_weight="balanced"` 的隐式权重 = n/(2·类计数)：本数据约
负类 0.55 / 正类 2.35——正例错一次罚 4 倍多。效果：FN 21 → 5，
recall 0.1923 → 0.8077，代价是 acc 掉 8 个点。自定义字典
`{0: 1, 1: 4}` 更激进（recall 0.8462），适合把业务代价直接写进去。
"""
    ),
    code(S2_CODE),
    code(S3_CODE),
    md(
        """
## 12.3 阈值移动：不重训，只动判决线

阈值 0.5 → 0.3：recall 0.1923 → 0.6538，f1 0.5397 反超 balanced。
**阈值 0.2 的混淆矩阵与 balanced 完全一致**——class_weight 本质就是
训练时的阈值移动，两者二选一。阈值移动的额外好处：不重训模型，
交卷时按指标现调。
"""
    ),
    code(S4_CODE),
    md(
        """
## 12.4 SMOTE：改数据

SMOTE 在少数类样本之间线性插值：训练集 480 → 756（正例 102 → 378，
补到 1:1）。测试集成绩与 balanced 同档（recall 0.7692），但多了一层
随机性和 k_neighbors 超参。**铁律：只碰训练集**——测试集重采样
（120 → 188）后所有指标作废，那是人工改题。
"""
    ),
    code(S5_CODE),
    code(S6_CODE),
    md(
        """
## 12.5 难点深挖：imblearn Pipeline 防 CV 泄漏

先在全体训练数据上 SMOTE 再切折 = 验证折里混入训练折的合成样本，
CV 分数虚高。`imblearn.pipeline.Pipeline` 把 SMOTE 塞进每折内部：
CV recall 0.6067 vs 普通 LR 0.3533，且这个差距是干净的。
"""
    ),
    code(S7_CODE),
    md(
        """
## 12.6 三方案对表

| 方案 | acc | recall | f1 | 成本 |
|---|---|---|---|---|
| 基线 thr=0.5 | 0.7833 | 0.1923 | 0.2778 | 0 |
| balanced | 0.70 | 0.8077 | 0.5385 | 一个参数 |
| SMOTE | 0.6917 | 0.7692 | 0.5195 | 造样本+超参 |
| 阈值 0.3 | **0.7583** | 0.6538 | **0.5397** | 不重训 |

竞赛推荐顺序：**class_weight → 阈值移动 → SMOTE**。漏报代价大的业务
（变压器危急缺陷）保 recall；只要排序能力（AUC）则原模型 + 移动阈值。
"""
    ),
    code(S8_CODE),
    md(
        """
## 小结

1. 不平衡数据先报 recall/f1：基线 acc 0.7833 里藏着 FN=21。
2. `class_weight="balanced"` 隐式权重 n/(2·类计数)，零成本首选。
3. 自定义字典 = 业务代价直译；权重越大模型越敢报正例。
4. 阈值移动与 class_weight 等价（阈值 0.2 == balanced 的 cm）。
5. SMOTE 只碰训练集；测试集重采样 = 人工改题，指标作废。
6. CV 里必须用 `imblearn.Pipeline`，先重采样后切折是泄漏。
7. 选型：class_weight → 阈值移动 → SMOTE；本数据阈值移动 f1 最高。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch12 类别不平衡（练习版）

> 按提示补全 `____`，跑通所有 assert。数据、划分、预处理与 `skf` 已就绪。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：基线体检

默认阈值下 acc 与 recall 的巨大反差。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：class_weight=balanced

FN 21 → 5 的零成本方案。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：自定义权重

`{0: 1, 1: 4}` 比全自动更激进。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：阈值移动

不重训只动判决线，并发现与 balanced 的等价性。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：SMOTE 正确用法

只对训练集插值，看样本量与成绩变化。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：作弊示范

在测试集上 SMOTE 会发生什么（只围观，不模仿）。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：imblearn Pipeline

CV 里防泄漏的正确姿势。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：三方案对表

balanced / SMOTE / 阈值移动，同一测试集见真章。
"""
    ),
    code(E8_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 任务完成。自查：balanced 的隐式权重公式、
阈值 0.2 为什么和 balanced 同判决、SMOTE 为什么只碰训练集、
CV 防泄漏的正确姿势。
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
