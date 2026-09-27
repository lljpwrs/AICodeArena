#!/usr/bin/env python3
"""ch10 集成与融合：Bagging/RandomForest/AdaBoost/GB/Voting/Stacking（defect_ml.csv）。

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk10_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch10_ensemble"

IMPORTS = '''from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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

# 三个异构成员先练好，后面 Voting/Stacking 直接复用
members = [("lr", LogisticRegression(max_iter=1000, random_state=42)),
           ("svc", SVC(probability=True, random_state=42)),
           ("dtree", DecisionTreeClassifier(max_depth=5, random_state=42))]
for _, est in members:
    est.fit(Xtr_s, ys_tr)

print("sklearn", sklearn.__version__, "| train", Xtr_s.shape,
      "| test", Xte_s.shape, "| 成员:", [name for name, _ in members])
'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def acc_of(model, Xt, yt):
    """一行算测试集 accuracy。"""
    return round(float(accuracy_score(yt, model.predict(Xt))), 4)


def staged_acc(model, Xt, yt):
    """staged_predict 的逐轮测试 accuracy 曲线。"""
    return [round(float(accuracy_score(yt, p)), 4)
            for p in model.staged_predict(Xt)]


print("脚手架就绪：acc_of / staged_acc")'''

# =========================================================================== #
# 练习 notebook 挖空版
# =========================================================================== #

E1_CODE = '''# @@todo(1) 单树 vs Bagging：一棵全深树装进 50 棵树的袋子里
# @@hint BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50, oob_score=True, random_state=42)
tree = DecisionTreeClassifier(random_state=42).fit(Xtr_s, ys_tr)
tree_acc = acc_of(tree, Xte_s, ys_te)
bag = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50,
                        oob_score=True, random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
bag_acc = acc_of(bag, Xte_s, ys_te)
bag_oob = round(float(bag.oob_score_), 4)
# @@end
print("单树 acc:", tree_acc, "| 叶子数:", tree.get_n_leaves())
print("Bagging50 acc:", bag_acc, "| oob:", bag_oob)

assert tree.get_n_leaves() == 95, "全深树 95 叶，把训练集背得滚瓜烂熟"
assert tree_acc == 0.65, "单树泛化差：全深决策树是典型的高方差弱学习器"
assert bag_acc == 0.7833, "50 棵树投票后 +13.3 个点"
assert bag_oob == 0.7958, "oob 是免费的验证集：不用切数据就能估泛化"
'''

E2_CODE = '''# @@todo(2) Bagging 数量曲线的骨架：建一个空字典存 {树数: acc}
# @@hint 就一个空字典；下面的扫描循环已写好，不用挖
bag_scores = {}
# @@end
for n in [10, 25, 100]:
    b = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=n,
                          random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
    bag_scores[n] = acc_of(b, Xte_s, ys_te)
print("数量→acc:", bag_scores)

assert bag_scores == {10: 0.7667, 25: 0.7833, 100: 0.775}
assert max(bag_scores.values()) - min(bag_scores.values()) <= 0.02, \\
    "加树主要买的是稳定，不是精度：25 棵之后曲线躺平甚至回落"
'''

E3_CODE = '''# @@todo(3) 随机森林：Bagging + 每次分裂只看 sqrt(d) 个特征
# @@hint RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)；重要性在 feature_importances_
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42,
                            n_jobs=-1).fit(Xtr_s, ys_tr)
rf_acc = acc_of(rf, Xte_s, ys_te)
rf_auc = round(float(roc_auc_score(ys_te, rf.predict_proba(Xte_s)[:, 1])), 4)
top3 = [feat[i] for i in np.argsort(rf.feature_importances_)[::-1][:3]]
# @@end
print("RF100 acc:", rf_acc, "| auc:", rf_auc, "| oob:", round(float(rf.oob_score_), 4))
print("内部单棵 max_features:", rf.estimators_[0].max_features, "| top3:", top3)

assert rf.estimators_[0].max_features == "sqrt", "16 特征的 sqrt≈4：RF 靠特征随机化去相关"
assert rf_acc == 0.7667 and rf_auc == 0.6872
assert round(float(rf.oob_score_), 4) == 0.8083, "oob 0.8083 比 test 0.7667 乐观——样本量不同，别拿 oob 交卷"
assert top3 == ["温度", "负荷值", "湿度"], "重要性与物理直觉一致：温度/负荷/湿度"
'''

E4_CODE = '''# @@todo(4) AdaBoost：staged 曲线看『练过头』长什么样
# @@hint AdaBoostClassifier(n_estimators=100, random_state=42)；staged_acc(model, Xte_s, ys_te)
ada = AdaBoostClassifier(n_estimators=100, random_state=42).fit(Xtr_s, ys_tr)
ada_curve = staged_acc(ada, Xte_s, ys_te)
ada_best_n = int(np.argmax(ada_curve)) + 1
ada_final = ada_curve[-1]
w_head = [round(float(w), 4) for w in ada.estimator_weights_[:5]]
# @@end
print("Ada 曲线峰值:", max(ada_curve), "@ 第", ada_best_n, "轮 | 最终:", ada_final)
print("前 5 轮基学习器权重:", w_head)

assert len(ada_curve) == 100
assert ada_best_n == 20 and max(ada_curve) == 0.8, "第 20 轮就到 0.80"
assert ada_final == 0.7667, "练到 100 轮反而回落——boosting 对噪声/过拟合不免疫"
assert w_head[0] == 1.3477 and w_head[2] == 0.2237, "第一棵树最重 1.3477；有的树几乎白学 0.2237"
'''

E5_CODE = '''# @@todo(5) GradientBoosting：staged 峰值 + learning_rate 敏感性
# @@hint GradientBoostingClassifier(n_estimators=100, random_state=42) 先 fit；staged_acc(model, Xte_s, ys_te)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xtr_s, ys_tr)
gb_curve = staged_acc(gb, Xte_s, ys_te)
gb_best_n = int(np.argmax(gb_curve)) + 1
# @@end
lr_scores = {}
for lr in [0.05, 0.2, 0.5]:
    g = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                   random_state=42).fit(Xtr_s, ys_tr)
    lr_scores[lr] = acc_of(g, Xte_s, ys_te)
print("GB staged 峰值:", max(gb_curve), "@ 第", gb_best_n, "轮")
print("learning_rate → acc:", lr_scores)

assert len(gb_curve) == 100
assert gb_best_n == 15 and max(gb_curve) == 0.8, "GB 15 轮就封顶，比 Ada 的 20 轮还早"
assert lr_scores == {0.05: 0.775, 0.2: 0.775, 0.5: 0.7917}, \\
    "本数据上 lr=0.5 反而最好——步子大不一定是坏事，lr 和 n_estimators 要一起调"
'''

E6_CODE = '''# @@todo(6) 手写 soft 投票 = VotingClassifier(voting="soft")：拆开看黑盒
# @@hint 三个成员 predict_proba(Xte_s)[:, 1] 求平均，>=0.5 记 1
proba_avg = np.mean([est.predict_proba(Xte_s)[:, 1] for _, est in members], axis=0)
hand_soft = (proba_avg >= 0.5).astype(int)
soft = VotingClassifier(estimators=members, voting="soft", n_jobs=-1).fit(Xtr_s, ys_tr)
soft_acc = acc_of(soft, Xte_s, ys_te)
agree = float(np.mean(hand_soft == soft.predict(Xte_s)))
# @@end
print("手写平均概率 acc:", round(float(accuracy_score(ys_te, hand_soft)), 4),
      "| VotingClassifier soft acc:", soft_acc, "| 预测一致率:", agree)

assert soft_acc == 0.7917
assert agree == 1.0, "VotingClassifier soft 就是平均概率套阈值，没有任何魔法"
'''

E7_CODE = '''# @@todo(7) 加权投票：给最强成员 3 倍话语权
# @@hint VotingClassifier(estimators=members, voting="soft", weights=[1, 1, 3])
weighted = VotingClassifier(estimators=members, voting="soft", weights=[1, 1, 3],
                            n_jobs=-1).fit(Xtr_s, ys_tr)
w_acc = acc_of(weighted, Xte_s, ys_te)
hard = VotingClassifier(estimators=members, voting="hard", n_jobs=-1).fit(Xtr_s, ys_tr)
hard_acc = acc_of(hard, Xte_s, ys_te)
# @@end
print("hard:", hard_acc, "| soft 等权:", 0.7917, "| soft w=[1,1,3]:", w_acc)

assert hard_acc == 0.7833 and w_acc == 0.8, "硬票 0.7833 → 等权软票 0.7917 → 加权 0.8"
assert w_acc > hard_acc, "给深 5 树的 dtree 加权后反超硬投票：成员质量不同就该区别对待"
'''

E8_CODE = '''# @@todo(8) Stacking：元特征形状与元学习器系数
# @@hint StackingClassifier(estimators=members, final_estimator=LogisticRegression(...), cv=5)；stack.transform(Xte_s) 看元特征
stack = StackingClassifier(estimators=members,
                           final_estimator=LogisticRegression(max_iter=1000,
                                                              random_state=42),
                           cv=5, n_jobs=-1).fit(Xtr_s, ys_tr)
stack_acc = acc_of(stack, Xte_s, ys_te)
stack_auc = round(float(roc_auc_score(ys_te, stack.predict_proba(Xte_s)[:, 1])), 4)
meta_shape = stack.transform(Xte_s).shape
coef = [round(float(c), 4) for c in stack.final_estimator_.coef_[0]]
# @@end
print("元特征 shape:", meta_shape, "| final coef:", coef)
print("Stacking acc:", stack_acc, "| auc:", stack_auc)

assert meta_shape == (120, 3), "二分类 Stacking 元特征是各成员的正类概率，不是 6 维！"
assert coef == [2.8091, 1.4407, 0.4144], "lr 话语权最大 2.81，dtree 最小 0.41"
assert stack_acc == 0.8 and stack_auc == 0.7721, "本章冠军：0.80 / AUC 0.7721"
'''

E9_CODE = '''# @@todo(9) 多样性体检：成员预测越不一致，融合越有肉吃
# @@hint 两两一致率 np.mean(p1 == p2)；再手写多数票对账 hard voting
p_lr = members[0][1].predict(Xte_s)
p_svc = members[1][1].predict(Xte_s)
p_tree = members[2][1].predict(Xte_s)
agree_ls = round(float(np.mean(p_lr == p_svc)), 4)
agree_lt = round(float(np.mean(p_lr == p_tree)), 4)
agree_st = round(float(np.mean(p_svc == p_tree)), 4)
P = np.vstack([p_lr, p_svc, p_tree])
hand_majority = (P.sum(axis=0) >= 2).astype(int)
maj_acc = round(float(accuracy_score(ys_te, hand_majority)), 4)
# @@end
print("一致率 lr~svc:", agree_ls, "| lr~tree:", agree_lt, "| svc~tree:", agree_st)
print("手写多数票 acc:", maj_acc, "（= hard voting 0.7833）")

assert agree_ls == 0.875 and agree_lt == 0.875 and agree_st == 0.8833, \\
    "成员间仍有 11~12% 分歧——这正是融合能涨点的空间"
assert maj_acc == 0.7833, "手写多数票 == VotingClassifier(hard)，黑盒再次拆穿"
'''

# =========================================================================== #
# 讲解 notebook 各节代码
# =========================================================================== #

S1_CODE = '''# 单树 vs Bagging：一棵 95 叶的全深树 acc 0.65，装进 50 棵树的袋子 0.7833
tree = DecisionTreeClassifier(random_state=42).fit(Xtr_s, ys_tr)
tree_acc = acc_of(tree, Xte_s, ys_te)
bag = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=50,
                        oob_score=True, random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
bag_acc = acc_of(bag, Xte_s, ys_te)
bag_oob = round(float(bag.oob_score_), 4)
print("单树 acc:", tree_acc, "| 叶子数:", tree.get_n_leaves())
print("Bagging50 acc:", bag_acc, "| oob:", bag_oob)
print("解读: 全深树高方差（换一批数据换一副面孔）；Bagging 用自助采样把方差摊薄，"
      "oob 0.7958 是每棵树用没见过的 37% 样本白送的验证分")

assert tree.get_n_leaves() == 95
assert tree_acc == 0.65 and bag_acc == 0.7833 and bag_oob == 0.7958
'''

S2_CODE = '''# Bagging 数量曲线：加树买的是稳定不是精度
bag_scores = {}
for n in [10, 25, 100]:
    b = BaggingClassifier(DecisionTreeClassifier(random_state=42), n_estimators=n,
                          random_state=42, n_jobs=-1).fit(Xtr_s, ys_tr)
    bag_scores[n] = acc_of(b, Xte_s, ys_te)
print("数量→acc:", bag_scores)
print("解读: 10→25 涨 1.7 个点，25→100 反而 -0.8 个点——bagging 收益对数封顶，"
      "竞赛里 50~100 棵足够，别为 0.5 个点烧时间")

assert bag_scores == {10: 0.7667, 25: 0.7833, 100: 0.775}
'''

S3_CODE = '''# 随机森林 = Bagging + 每次分裂只看 sqrt(d)≈4 个特征：让树们「错得不一样」
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42,
                            n_jobs=-1).fit(Xtr_s, ys_tr)
rf_acc = acc_of(rf, Xte_s, ys_te)
rf_auc = round(float(roc_auc_score(ys_te, rf.predict_proba(Xte_s)[:, 1])), 4)
top3 = [feat[i] for i in np.argsort(rf.feature_importances_)[::-1][:3]]
print("RF100 acc:", rf_acc, "| auc:", rf_auc, "| oob:", round(float(rf.oob_score_), 4))
print("单棵 max_features:", rf.estimators_[0].max_features, "| top3:", top3)
print("解读: oob 0.8083 > test 0.7667——oob 估计的是 480 训练样本、test 是 120 个更难的抽样，"
      "报告交 test，oob 只当免费体检")

assert rf.estimators_[0].max_features == "sqrt"
assert rf_acc == 0.7667 and rf_auc == 0.6872 and round(float(rf.oob_score_), 4) == 0.8083
assert top3 == ["温度", "负荷值", "湿度"]
'''

S4_CODE = '''# AdaBoost：逐轮加权、staged 曲线暴露『练过头』
ada = AdaBoostClassifier(n_estimators=100, random_state=42).fit(Xtr_s, ys_tr)
ada_curve = staged_acc(ada, Xte_s, ys_te)
ada_best_n = int(np.argmax(ada_curve)) + 1
w_head = [round(float(w), 4) for w in ada.estimator_weights_[:5]]
print("Ada 曲线峰值:", max(ada_curve), "@ 第", ada_best_n, "轮 | 最终:", ada_curve[-1])
print("前 5 轮基学习器权重:", w_head)
print("解读: 第 20 轮 0.80 → 第 100 轮 0.7667——boosting 一路追着难分样本跑，"
      "噪声样本权重越滚越大，练久了反而背噪声；staged 曲线是免费的早停仪表盘")

assert len(ada_curve) == 100 and ada_best_n == 20
assert max(ada_curve) == 0.8 and ada_curve[-1] == 0.7667
assert w_head[0] == 1.3477 and w_head[2] == 0.2237
'''

S5_CODE = '''# GradientBoosting：残差接力 + learning_rate 敏感性
gb = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(Xtr_s, ys_tr)
gb_curve = staged_acc(gb, Xte_s, ys_te)
lr_scores = {lr: acc_of(GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                                   random_state=42).fit(Xtr_s, ys_tr),
                        Xte_s, ys_te) for lr in [0.05, 0.2, 0.5]}
print("GB staged 峰值:", max(gb_curve), "@ 第", int(np.argmax(gb_curve)) + 1, "轮")
print("learning_rate → acc:", lr_scores)
print("解读: lr 大步子与 n_estimators 小树数是杠杆两端——教科书说『lr 小 + 树多更稳』，"
      "但本数据 600 行、lr=0.5 反而 0.7917 最优：小数据上别迷信教条，网格说了算")

assert len(gb_curve) == 100 and max(gb_curve) == 0.8
assert lr_scores == {0.05: 0.775, 0.2: 0.775, 0.5: 0.7917}
'''

S6_CODE = '''# 拆穿 VotingClassifier 的黑盒：soft = 平均概率 + 0.5 阈值
proba_avg = np.mean([est.predict_proba(Xte_s)[:, 1] for _, est in members], axis=0)
hand_soft = (proba_avg >= 0.5).astype(int)
soft = VotingClassifier(estimators=members, voting="soft", n_jobs=-1).fit(Xtr_s, ys_tr)
soft_acc = acc_of(soft, Xte_s, ys_te)
agree = float(np.mean(hand_soft == soft.predict(Xte_s)))
print("手写平均概率 acc:", round(float(accuracy_score(ys_te, hand_soft)), 4),
      "| VotingClassifier soft acc:", soft_acc, "| 预测一致率:", agree)
print("解读: 一致率 1.0——面试官问『soft voting 原理』，就答平均概率过阈值")

assert soft_acc == 0.7917 and agree == 1.0
'''

S7_CODE = '''# 等权 vs 加权：成员质量不同就该区别对待
weighted = VotingClassifier(estimators=members, voting="soft", weights=[1, 1, 3],
                            n_jobs=-1).fit(Xtr_s, ys_tr)
w_acc = acc_of(weighted, Xte_s, ys_te)
hard = VotingClassifier(estimators=members, voting="hard", n_jobs=-1).fit(Xtr_s, ys_tr)
hard_acc = acc_of(hard, Xte_s, ys_te)
print("hard:", hard_acc, "| soft 等权: 0.7917 | soft w=[1,1,3]:", w_acc)
print("解读: 单成员成绩 svc 0.7583 / lr 0.7833 / dtree 0.7917——给最强者 3 倍票力，"
      "0.7917 → 0.80；硬投票丢掉置信度信息，天然吃亏一档")

assert hard_acc == 0.7833 and w_acc == 0.8
'''

S8_CODE = '''# Stacking：成员概率当新特征，元学习器学『信谁多一点』
stack = StackingClassifier(estimators=members,
                           final_estimator=LogisticRegression(max_iter=1000,
                                                              random_state=42),
                           cv=5, n_jobs=-1).fit(Xtr_s, ys_tr)
stack_acc = acc_of(stack, Xte_s, ys_te)
stack_auc = round(float(roc_auc_score(ys_te, stack.predict_proba(Xte_s)[:, 1])), 4)
coef = [round(float(c), 4) for c in stack.final_estimator_.coef_[0]]
print("元特征 shape:", stack.transform(Xte_s).shape, "| final coef:", coef)
print("Stacking acc:", stack_acc, "| auc:", stack_auc)
print("解读: 二分类元特征是 3 个成员的正类概率（不是 6 维！多分类才是 类数x成员数）；"
      "元学习器 coef [2.81, 1.44, 0.41]——LR 学会了『最信 lr，其次 svc，dtree 只做参考』。"
      "cv=5 保证元特征没有『自己评自己』的泄漏")

assert stack_acc == 0.8 and stack_auc == 0.7721
assert coef == [2.8091, 1.4407, 0.4144]
'''

S9_CODE = '''# 多样性体检：成员一致率 vs 融合收益
p_lr = members[0][1].predict(Xte_s)
p_svc = members[1][1].predict(Xte_s)
p_tree = members[2][1].predict(Xte_s)
P = np.vstack([p_lr, p_svc, p_tree])
maj_acc = round(float(accuracy_score(ys_te, (P.sum(axis=0) >= 2).astype(int))), 4)
print("一致率 lr~svc: 0.875 | lr~tree: 0.875 | svc~tree: 0.8833")
print("手写多数票 acc:", maj_acc)
print("解读: 成员间仍有 ~12% 分歧是融合涨点的前提——如果三个模型错得一样，"
      "怎么投都是同样的错。 diversity > 单个模型再强")

assert maj_acc == 0.7833
'''

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch10 集成与融合：Bagging / Boosting / Voting / Stacking

> 数据：`defect_ml.csv` 二分类，划分与预处理口径沿用 ch05。三个异构成员
> （LR / SVC / 深度 5 决策树）已预先拟合好，供 Voting 与 Stacking 复用。
>
> 本章回答三个问题：**集成凭什么涨点、四种集成各怎么工作、怎么证明它没作弊**。

**本章考点**（对应竞赛「模型训练 40%」中的模型选择与组合）

1. Bagging 与 OOB 分数的含义及边界
2. RandomForest 与 Bagging 的唯一差别：特征随机化
3. AdaBoost / GradientBoosting 的 staged 曲线与过拟合回落
4. VotingClassifier hard / soft / weights 三档
5. StackingClassifier 元特征的形状（二分类陷阱）
6. 成员多样性 = 融合涨点的前提

**难点索引**

| 难点 | 为什么是坑 | 位置 |
|---|---|---|
| oob 与 test 分数打架 | oob 0.8083 > test 0.7667，别拿 oob 交卷 | §10.2 |
| boosting 练过头 | Ada 第 20 轮 0.80 → 第 100 轮 0.7667 | §10.3 |
| lr 越小越稳是教条 | 本数据 lr=0.5 反而最优 | §10.4 |
| Stacking 元特征维度 | 二分类是 (n, 成员数)，不是 (n, 2x成员数) | §10.5 |
| 融合必然涨点的迷信 | 成员错得一样时怎么投都白搭 | §10.6 |
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 10.1 Bagging：把高方差摊薄

一棵 95 叶的全深决策树 acc 只有 0.65——它把 480 条训练数据背得滚瓜烂熟，
换一批数据就换一副面孔。Bagging（Bootstrap AGGregatING）用自助采样训练
50 棵各见 63% 数据的树，投票后 0.7833；`oob_score=True` 顺手用每棵树没见过
的约 37% 样本做免费验证，oob 0.7958。
"""
    ),
    code(S1_CODE),
    md(
        """
## 10.2 难点深挖：数量曲线与 oob 的边界

树的数量 10 → 25 → 100：0.7667 → 0.7833 → 0.775。**加树买的是稳定，
不是精度**，收益对数封顶。RF 的 oob 0.8083 比 test 0.7667 高出 4 个点——
两者评估的样本集不同，**报告交 test 分数，oob 只当免费体检**。
"""
    ),
    code(S2_CODE),
    code(S3_CODE),
    md(
        """
## 10.3 难点深挖：boosting 的 staged 曲线会『回头』

AdaBoost 第 20 轮就到 0.80，练到 100 轮回落到 0.7667——boosting 一路给
难分样本加权，噪声样本的权重越滚越大。GradientBoosting 15 轮封顶 0.80。
**`staged_predict` 是免费的早停仪表盘**，竞赛里值得画一眼。
"""
    ),
    code(S4_CODE),
    md(
        """
## 10.4 learning_rate：教条与网格的对决

教科书说「lr 小 + 树多更稳」，但本数据 600 行、lr=0.5 反而 0.7917 最优
（0.05 和 0.2 都是 0.775）。**小数据上别迷信教条，让网格说话**。
"""
    ),
    code(S5_CODE),
    md(
        """
## 10.5 难点深挖：Voting 与 Stacking 的黑盒拆解

`VotingClassifier(voting="soft")` 手写等价实现与官方结果一致率 1.0——
就是平均概率过 0.5 阈值。加权投票给最强成员 3 倍票力后 0.7917 → 0.80。

Stacking 的元特征 shape 是 **(120, 3)**：二分类下每个成员只贡献正类
概率一列（不是 2x3=6 维，多分类才是 类数×成员数）。元学习器系数
[2.81, 1.44, 0.41] 说明它学会了「最信 LR」——cv=5 生成元特征，
避免「自己评自己」的泄漏。
"""
    ),
    code(S6_CODE),
    code(S7_CODE),
    code(S8_CODE),
    md(
        """
## 10.6 多样性：融合涨点的前提

三个成员两两一致率 0.875 / 0.875 / 0.8833——仍有约 12% 的分歧，
这正是融合能从 0.7833 涨到 0.80 的空间。**如果成员错得一样，
怎么投都是同样的错**：选成员看多样性，不只看单个模型多强。

本章成绩单：单树 0.65 → Bagging 0.7833 → RF 0.7667 → Ada 0.7667
→ GB 0.775 → 加权 Voting 0.80 → **Stacking 0.80（AUC 0.7721）**。
"""
    ),
    code(S9_CODE),
    md(
        """
## 小结

1. Bagging 把方差摊薄：单树 0.65 → 50 棵 0.7833；oob 是免费体检不是考卷。
2. 加树买稳定不买精度：25 棵之后曲线躺平。
3. RF = Bagging + 特征随机化（sqrt(d)）；重要性 top3 温度/负荷/湿度。
4. boosting 会练过头：staged 曲线免费给早停信号（Ada 峰值@20、GB@15）。
5. soft voting = 平均概率过阈值；加权投票按成员质量分配话语权。
6. Stacking 二分类元特征是 (n, 成员数)；cv 生成元特征防泄漏。
7. 多样性是融合涨点的前提，成员一致率 0.875 时融合仍有肉吃。
"""
    ),
]

# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch10 集成与融合（练习版）

> 按提示补全 `____`，跑通所有 assert。数据、划分与三个成员
> `members`（lr / svc / dtree，已拟合）已在脚手架就绪。
>
> 注意：挖空块内每条语句保持**单行**；循环/分支写在挖空块之外。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 任务 1：单树 vs Bagging

一棵全深树装进 50 棵树的袋子，看 acc 与 oob 的变化。
"""
    ),
    code(E1_CODE),
    md(
        """
## 任务 2：Bagging 数量曲线

10 / 25 / 100 棵树，验证「加树买稳定不买精度」。
"""
    ),
    code(E2_CODE),
    md(
        """
## 任务 3：随机森林与特征重要性

RF100 训练，确认 max_features、acc / auc / oob 与 top3 特征。
"""
    ),
    code(E3_CODE),
    md(
        """
## 任务 4：AdaBoost staged 曲线

找出峰值轮数与最终轮分数的差——boosting 练过头的证据。
"""
    ),
    code(E4_CODE),
    md(
        """
## 任务 5：GradientBoosting 与 learning_rate

staged 峰值 + 三个学习率的敏感性实测。
"""
    ),
    code(E5_CODE),
    md(
        """
## 任务 6：手写 soft 投票

平均概率过阈值，与 `VotingClassifier(voting="soft")` 对账。
"""
    ),
    code(E6_CODE),
    md(
        """
## 任务 7：加权投票

给最强成员 3 倍话语权，对比 hard 与等权 soft。
"""
    ),
    code(E7_CODE),
    md(
        """
## 任务 8：Stacking 元特征

确认元特征形状与元学习器系数，别掉进 6 维陷阱。
"""
    ),
    code(E8_CODE),
    md(
        """
## 任务 9：成员多样性体检

两两一致率 + 手写多数票对账 hard voting。
"""
    ),
    code(E9_CODE),
    md(
        """
## 完成标准

所有 assert 通过 = 任务完成。自查：oob 和 test 哪个能写进报告、
boosting 什么时候该早停、Stacking 二分类元特征为什么是 3 列、
融合涨点的前提是什么。
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
