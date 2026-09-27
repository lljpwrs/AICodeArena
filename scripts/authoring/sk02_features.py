"""sk02 —— 特征工程：scaler 三兄弟 / 编码器 / 填充 / 多项式特征

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk02_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch02_feature_engineering"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)

DATA = Path("data")

dev = pd.read_csv(DATA / "defect_ml.csv")   # 二分类：预测「是否危急」

NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
TARGET = "是否危急"

X = dev[NUM + CAT]
y = dev[TARGET]

# 分层切分（ch01 的结论：分类任务 stratify=y 无条件写上）
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("sklearn", sklearn.__version__)
print("train", Xs_tr.shape, "test", Xs_te.shape)
print("训练段数值列缺失:", Xs_tr[NUM].isna().sum().tolist())'''

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

S1_CODE = '''# 三兄弟同台：都在训练段 fit，都只对数值列
ss = StandardScaler().fit(Xs_tr[NUM])
mm = MinMaxScaler().fit(Xs_tr[NUM])
rs = RobustScaler().fit(Xs_tr[NUM])

print("StandardScaler mean_:", np.round(ss.mean_, 4).tolist())
print("StandardScaler scale_:", np.round(ss.scale_, 4).tolist())
print("MinMaxScaler  data_min_:", mm.data_min_.tolist())
print("MinMaxScaler  data_max_:", mm.data_max_.tolist())
print("RobustScaler  center_ :", np.round(rs.center_, 4).tolist())
print("RobustScaler  scale_ :", np.round(rs.scale_, 4).tolist())

# RobustScaler 的 center_ 就是中位数，scale_ 就是 IQR —— 和手算对账
q1 = Xs_tr["负荷值"].quantile(0.25)
q3 = Xs_tr["负荷值"].quantile(0.75)
print("负荷值 IQR 手算:", round(float(q3 - q1), 4), "vs RobustScaler scale_[0]:",
      round(float(rs.scale_[0]), 4))
print("RobustScaler center_ == SimpleImputer(median) 的统计量:",
      bool(np.allclose(rs.center_, [427.33, 35.1, 64.9, 16.0, 44.0])))

assert round(float(ss.mean_[0]), 4) == 422.6926, "训练段负荷值均值"
assert round(float(ss.scale_[0]), 4) == 154.4385, "训练段负荷值标准差"
assert float(mm.data_min_[0]) == 40.0 and float(mm.data_max_[0]) == 835.12
assert round(float(rs.center_[0]), 4) == 427.33 and round(float(rs.scale_[0]), 4) == 215.18
assert round(float(q3 - q1), 4) == 215.18
'''

S2_CODE = '''# 关键认知：scaler 对 NaN 是「视而不见」——fit 不报错、transform 原样保留
tr_t = ss.transform(Xs_tr[NUM])
print("训练段 transform 后 nanmean:", np.round(np.nanmean(tr_t, axis=0), 10).tolist())
print("训练段 transform 后 nanstd :", np.round(np.nanstd(tr_t, axis=0), 4).tolist())
print("transform 输出里的 NaN 个数:", int(np.isnan(tr_t).sum()),
      "← 负荷值 36 + 温度 36，原样穿透")

# 测试段：只能 transform 不能重新 fit，均值自然不在 0
te_t = ss.transform(Xs_te[NUM])
print("测试段 transform 后 nanmean:", np.round(np.nanmean(te_t, axis=0), 4).tolist())
print("测试段 transform 后 nanstd :", np.round(np.nanstd(te_t, axis=0), 4).tolist())

assert np.allclose(np.nanmean(tr_t, axis=0), 0, atol=1e-10), "训练段均值恰为 0"
assert np.allclose(np.nanstd(tr_t, axis=0), 1, atol=1e-10), "训练段标准差恰为 1"
assert int(np.isnan(tr_t).sum()) == 72, "NaN 原样穿透：36 + 36"
assert round(float(np.nanmean(te_t, axis=0)[0]), 4) == 0.0832, "测试段均值不在 0"
assert round(float(np.nanstd(te_t, axis=0)[0]), 4) == 0.9115, "测试段标准差不在 1"
'''

S3_CODE = '''# MinMaxScaler 的暗坑：测试段数值可以越出 [0, 1]，不报错不警告
te_mt = mm.transform(Xs_te[NUM])
print("测试段 >1 的元素个数:", int((te_mt > 1).sum()),
      "| <0 的元素个数:", int((te_mt < 0).sum()))
for i, col in enumerate(NUM):
    n_over = int((te_mt[:, i] > 1).sum()) + int((te_mt[:, i] < 0).sum())
    if n_over:
        print(f"  {col}: 越界 {n_over} 个, max={np.nanmax(te_mt[:, i]):.4f}")

assert int((te_mt > 1).sum()) == 3, "温度 1 个 + 湿度 2 个"
assert int((te_mt < 0).sum()) == 0
assert round(float(np.nanmax(te_mt[:, 1])), 4) == 1.0045, "温度越界幅度不大但确实越了"
assert round(float(np.nanmax(te_mt[:, 2])), 4) == 1.0180
assert round(float(np.nanmax(te_mt[:, 0])), 4) == 0.9556, "负荷值 max 恰好落在训练段内"
'''

S4_CODE = '''# SimpleImputer：median / most_frequent / constant 三种策略
imp = SimpleImputer(strategy="median").fit(Xs_tr[NUM])
print("median statistics_:", np.round(imp.statistics_, 4).tolist())
Xi = imp.transform(Xs_tr[NUM])
print("填充后缺失:", int(np.isnan(Xi).sum()), "| 形状:", Xi.shape)

# add_indicator：把「这里曾经缺失」本身变成一列特征
imp_ind = SimpleImputer(strategy="median", add_indicator=True).fit(Xs_tr[NUM])
Xi_ind = imp_ind.transform(Xs_tr[NUM])
print("add_indicator 形状:", Xi_ind.shape, "= 5 原列 + 2 缺失指示列")
print("指示列合计（缺失次数）:", int(Xi_ind[:, 5:].sum()), "= 36 + 36")

# 类别列用 most_frequent；constant 可自定义占位值
imp_cat = SimpleImputer(strategy="most_frequent").fit(Xs_tr[CAT])
print("most_frequent statistics_:", imp_cat.statistics_.tolist())
imp_const = SimpleImputer(strategy="constant", fill_value=-1).fit(Xs_tr[NUM])
print("constant(-1) 变换后最小值:", float(imp_const.transform(Xs_tr[NUM]).min()))

assert np.allclose(np.round(imp.statistics_, 4), [427.33, 35.1, 64.9, 16.0, 44.0])
assert int(np.isnan(Xi).sum()) == 0
assert Xi_ind.shape == (480, 7) and int(Xi_ind[:, 5:].sum()) == 72
assert imp_cat.statistics_.tolist() == ["变压器", "渗漏油"]
assert float(imp_const.transform(Xs_tr[NUM]).min()) == -1.0
'''

S5_CODE = '''# OneHotEncoder：码位序、drop、min_frequency、unknown 四个考点
oh = OneHotEncoder(sparse_output=False).fit(Xs_tr[CAT])
print("设备类型类别(码位序):", oh.categories_[0].tolist())
print("缺陷类型类别(码位序):", oh.categories_[1].tolist())
X_oh = oh.transform(Xs_tr[CAT])
print("独热维度:", X_oh.shape, "= (5 设备 + 6 缺陷) 类")

# drop="first"：每列扔掉第一个类别当基准，防共线（线性模型常用）
oh_d = OneHotEncoder(sparse_output=False, drop="first").fit(Xs_tr[CAT])
print("drop=first 维度:", oh_d.transform(Xs_tr[CAT]).shape, "= 11 - 2")

# 全 0 行 == 基准类（互感器/发热）——drop 之后这是语义而不是丢失
oh_d_out = oh_d.transform(Xs_tr[CAT][Xs_tr["设备类型"] == "互感器"].head(1))
print("互感器行 drop 后独热和:", float(oh_d_out[0].sum()), "← 设备类型段全 0 = 基准类")

assert oh.categories_[0].tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert X_oh.shape == (480, 11)
assert oh_d.transform(Xs_tr[CAT]).shape == (480, 9)
assert float(oh_d_out[0][:5].sum()) == 0.0, "设备类型前 5 列全 0"
'''

S6_CODE = '''# min_frequency：低频类别合并进 infrequent 桶
# 训练段缺陷类型计数: 渗漏油 94 / 放电痕迹 85 / 锈蚀 84 / 发热 80 / 破损 77 / 异物 60
oh_m1 = OneHotEncoder(sparse_output=False, min_frequency=0.15).fit(Xs_tr[CAT])
print("min_frequency=0.15(72): 维度", oh_m1.transform(Xs_tr[CAT]).shape,
      "| 只有异物 60 被合并")
print("  特征名含 infrequent:", [n for n in oh_m1.get_feature_names_out() if "infrequent" in n])

# 阈值抬到 0.2(96)：缺陷类型最大的渗漏油 94 也 < 96 → 整列 6 类全被吞掉！
oh_m2 = OneHotEncoder(sparse_output=False, min_frequency=0.2).fit(Xs_tr[CAT])
print("min_frequency=0.2(96): 维度", oh_m2.transform(Xs_tr[CAT]).shape,
      "| 缺陷类型整列失效")
print("  infrequent 桶:", [c.tolist() for c in oh_m2.infrequent_categories_])

# unknown：ignore 遇未知类别输出全 0 行
oh_ign = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(Xs_tr[CAT])
unk = pd.DataFrame({"设备类型": ["杆塔"], "缺陷类型": ["发热"]})
out_unk = oh_ign.transform(unk)
print("ignore 遇未知:", out_unk.shape, "| 该行独热和:", float(out_unk.sum()),
      "← 只有未知段归零，已知段保持编码")

# 1.9 实测：drop="first" + handle_unknown="ignore" 也能跑（旧教程说必报错）
oh_bad = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
st_fit, _ = probe(oh_bad.fit, Xs_tr[CAT])
out_bad = oh_bad.transform(unk)
print("drop+ignore: fit →", st_fit, "| transform 未知 →", out_bad.shape,
      "| 独热和:", float(out_bad.sum()))

assert oh_m1.transform(Xs_tr[CAT]).shape == (480, 11), "只合并 1 个类别时维度不变"
assert "缺陷类型_infrequent_sklearn" in oh_m1.get_feature_names_out().tolist()
assert oh_m2.transform(Xs_tr[CAT]).shape == (480, 5), "整列被吞: (2 设备保留 + 1) + (0 + 1)"
assert out_unk.shape == (1, 11)
assert float(out_unk[0][:5].sum()) == 0.0 and float(out_unk.sum()) == 1.0, "未知段归零，已知段保留"
assert st_fit == "ok" and out_bad.shape == (1, 9) and float(out_bad.sum()) == 0.0, "未知段归零"
'''

S7_CODE = '''# OrdinalEncoder：一个整数一个类别，树模型的省内存选项
oe = OrdinalEncoder().fit(Xs_tr[CAT])
print("设备类型码位序:", oe.categories_[0].tolist())
print("前 3 行编码:", oe.transform(Xs_tr[CAT].head(3)).tolist())

# 默认遇到未知类别直接 ValueError —— ch01 的 unknown_value=-1 是解法
status, msg = probe(
    oe.transform,
    pd.DataFrame({"设备类型": ["杆塔"], "缺陷类型": ["放电痕迹"]}))
print("OrdinalEncoder 遇未知 →", status, "|", msg)

assert oe.categories_[0].tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert oe.transform(Xs_tr[CAT].head(1)).tolist()[0][0] == 3.0, "互感器 = 码位 3"
assert status == "err" and msg == "ValueError"
'''

S8_CODE = '''# PolynomialFeatures：拒绝 NaN、维度公式、数值爆炸三连
status, msg = probe(PolynomialFeatures(degree=2).fit, Xs_tr[NUM])
print("直接对含 NaN 的数值列 fit →", status, "|", msg)

# 必须先填充再升维
Xi = imp.transform(Xs_tr[NUM])                     # 复用 §3.4 的 median 填充
pf2b = PolynomialFeatures(degree=2, include_bias=False).fit(Xi)
Xi2 = pf2b.transform(Xi)
print("degree=2 无 bias:", Xi2.shape, "= 5 + 5*6/2")
pf2i = PolynomialFeatures(degree=2, include_bias=False,
                          interaction_only=True).fit(Xi)
print("interaction_only:", pf2i.transform(Xi).shape, "= 5 + 5*4/2")
pf3b = PolynomialFeatures(degree=3, include_bias=False).fit(Xi)
print("degree=3 无 bias:", pf3b.transform(Xi).shape)
print("特征名前 7:", pf2b.get_feature_names_out()[:7].tolist())
print("平方项:", [n for n in pf2b.get_feature_names_out() if "^2" in n])

# 数值爆炸：负荷值 ~835 平方后 69 万，交互项量纲彻底失控
mx = int(np.unravel_index(np.argmax(Xi2), Xi2.shape)[1])
print("未缩放直接 poly 的最大值:", round(float(Xi2.max()), 1),
      "| 来自特征:", pf2b.get_feature_names_out()[mx])

# 正确顺序：poly 之后再缩放一次（或先 scale 再 poly 再 scale）
ss2 = StandardScaler().fit(Xi2)
print("poly 后再 scale：x0 std", round(float(ss2.scale_[0]), 2),
      "| x0*x1 std", round(float(ss2.scale_[5]), 2), "← 交互项仍然巨大")

# 测试段：复用训练段的 imputer / poly，形状必然一致
print("测试段 poly 形状:", pf2b.transform(imp.transform(Xs_te[NUM])).shape)

assert status == "err" and msg == "ValueError", "PolynomialFeatures 拒绝 NaN"
assert Xi2.shape == (480, 20) and pf2i.transform(Xi).shape == (480, 15)
assert pf3b.transform(Xi).shape == (480, 55)
assert round(float(Xi2.max()), 1) == 697425.4
assert pf2b.get_feature_names_out()[mx] == "x0^2"
assert pf2b.transform(imp.transform(Xs_te[NUM])).shape == (120, 20)
'''


# =========================================================================== #
# 练习版各节代码（@@todo 挖空标记 + 真实解法）
# =========================================================================== #

E1_CODE = '''# @@todo(1) 在训练段数值列上 fit StandardScaler / MinMaxScaler / RobustScaler
# @@hint StandardScaler().fit(Xs_tr[NUM])，三兄弟同一份数据
ss = StandardScaler().fit(Xs_tr[NUM])
mm = MinMaxScaler().fit(Xs_tr[NUM])
rs = RobustScaler().fit(Xs_tr[NUM])
# @@end

print("StandardScaler mean_:", np.round(ss.mean_, 4).tolist())
print("StandardScaler scale_:", np.round(ss.scale_, 4).tolist())
print("MinMaxScaler  data_min_:", mm.data_min_.tolist())
print("MinMaxScaler  data_max_:", mm.data_max_.tolist())
print("RobustScaler  center_ :", np.round(rs.center_, 4).tolist())
print("RobustScaler  scale_ :", np.round(rs.scale_, 4).tolist())

# RobustScaler 的 center_ 就是中位数，scale_ 就是 IQR —— 和手算对账
q1 = Xs_tr["负荷值"].quantile(0.25)
q3 = Xs_tr["负荷值"].quantile(0.75)
print("负荷值 IQR 手算:", round(float(q3 - q1), 4), "vs RobustScaler scale_[0]:",
      round(float(rs.scale_[0]), 4))
print("RobustScaler center_ == SimpleImputer(median) 的统计量:",
      bool(np.allclose(rs.center_, [427.33, 35.1, 64.9, 16.0, 44.0])))

assert round(float(ss.mean_[0]), 4) == 422.6926, "训练段负荷值均值"
assert round(float(ss.scale_[0]), 4) == 154.4385, "训练段负荷值标准差"
assert float(mm.data_min_[0]) == 40.0 and float(mm.data_max_[0]) == 835.12
assert round(float(rs.center_[0]), 4) == 427.33 and round(float(rs.scale_[0]), 4) == 215.18
assert round(float(q3 - q1), 4) == 215.18
'''

E2_CODE = '''# @@todo(2) 用训练段 fit 好的 ss / mm 分别 transform 训练段与测试段数值列
# @@hint ss.transform(Xs_tr[NUM])；测试段只许 transform 不许重新 fit
tr_t = ss.transform(Xs_tr[NUM])
te_t = ss.transform(Xs_te[NUM])
# @@end

# @@todo(3) 再对测试段做 MinMaxScaler 变换，数一数越出 [0, 1] 的元素个数
# @@hint mm.transform(Xs_te[NUM])；(te_mt > 1).sum() 与 (te_mt < 0).sum()
te_mt = mm.transform(Xs_te[NUM])
n_over_hi = int((te_mt > 1).sum())
n_over_lo = int((te_mt < 0).sum())
# @@end

print("训练段 transform 后 nanmean:", np.round(np.nanmean(tr_t, axis=0), 10).tolist()[:3], "...")
print("训练段 transform 后 nanstd :", np.round(np.nanstd(tr_t, axis=0), 4).tolist())
print("transform 输出里的 NaN 个数:", int(np.isnan(tr_t).sum()),
      "← 负荷值 36 + 温度 36，原样穿透")
print("测试段 transform 后 nanmean:", np.round(np.nanmean(te_t, axis=0), 4).tolist())
print("测试段 transform 后 nanstd :", np.round(np.nanstd(te_t, axis=0), 4).tolist())
print("MinMax 测试段越界: >1 有", n_over_hi, "个 | <0 有", n_over_lo, "个")

assert np.allclose(np.nanmean(tr_t, axis=0), 0, atol=1e-10), "训练段均值恰为 0"
assert np.allclose(np.nanstd(tr_t, axis=0), 1, atol=1e-10), "训练段标准差恰为 1"
assert int(np.isnan(tr_t).sum()) == 72, "NaN 原样穿透：36 + 36"
assert round(float(np.nanmean(te_t, axis=0)[0]), 4) == 0.0832, "测试段均值不在 0"
assert round(float(np.nanstd(te_t, axis=0)[0]), 4) == 0.9115, "测试段标准差不在 1"
assert n_over_hi == 3, "温度 1 个 + 湿度 2 个"
assert n_over_lo == 0
assert round(float(np.nanmax(te_mt[:, 1])), 4) == 1.0045, "温度越界幅度不大但确实越了"
assert round(float(np.nanmax(te_mt[:, 2])), 4) == 1.0180
assert round(float(np.nanmax(te_mt[:, 0])), 4) == 0.9556, "负荷值 max 恰好落在训练段内"
'''

E3_CODE = '''# @@todo(4) 定义 median 策略 + add_indicator 的 SimpleImputer，fit 后填充训练段数值列
# @@hint SimpleImputer(strategy=..., add_indicator=True)
imp = SimpleImputer(strategy="median").fit(Xs_tr[NUM])
Xi = imp.transform(Xs_tr[NUM])
# @@end
print("median statistics_:", np.round(imp.statistics_, 4).tolist())
print("填充后缺失:", int(np.isnan(Xi).sum()), "| 形状:", Xi.shape)

# @@todo(5) 给上面这个需求加上缺失指示列：add_indicator=True，存入 Xi_ind
# @@hint 指示列在输出矩阵的第 5 列之后，Xi_ind[:, 5:].sum() 就是缺失次数
imp_ind = SimpleImputer(strategy="median", add_indicator=True).fit(Xs_tr[NUM])
Xi_ind = imp_ind.transform(Xs_tr[NUM])
# @@end
print("add_indicator 形状:", Xi_ind.shape, "= 5 原列 + 2 缺失指示列")
print("指示列合计（缺失次数）:", int(Xi_ind[:, 5:].sum()), "= 36 + 36")

# @@todo(6) 类别列用 most_frequent 策略填充，存入 imp_cat
# @@hint SimpleImputer(strategy="most_frequent").fit(Xs_tr[CAT])
imp_cat = SimpleImputer(strategy="most_frequent").fit(Xs_tr[CAT])
# @@end
print("most_frequent statistics_:", imp_cat.statistics_.tolist())
imp_const = SimpleImputer(strategy="constant", fill_value=-1).fit(Xs_tr[NUM])
print("constant(-1) 变换后最小值:", float(imp_const.transform(Xs_tr[NUM]).min()))

assert np.allclose(np.round(imp.statistics_, 4), [427.33, 35.1, 64.9, 16.0, 44.0])
assert int(np.isnan(Xi).sum()) == 0
assert Xi_ind.shape == (480, 7) and int(Xi_ind[:, 5:].sum()) == 72
assert imp_cat.statistics_.tolist() == ["变压器", "渗漏油"]
assert float(imp_const.transform(Xs_tr[NUM]).min()) == -1.0
'''

E4_CODE = '''# @@todo(7) 定义非稀疏输出的 OneHotEncoder，fit 训练段类别列并 transform
# @@hint OneHotEncoder(sparse_output=False)
oh = OneHotEncoder(sparse_output=False).fit(Xs_tr[CAT])
X_oh = oh.transform(Xs_tr[CAT])
# @@end
print("设备类型类别(码位序):", oh.categories_[0].tolist())
print("缺陷类型类别(码位序):", oh.categories_[1].tolist())
print("独热维度:", X_oh.shape, "= (5 设备 + 6 缺陷) 类")

# @@todo(8) 定义 drop="first" 的 OneHotEncoder 并 transform，看维度少了几列
# @@hint OneHotEncoder(sparse_output=False, drop="first")
oh_d = OneHotEncoder(sparse_output=False, drop="first").fit(Xs_tr[CAT])
oh_d_out = oh_d.transform(Xs_tr[CAT][Xs_tr["设备类型"] == "互感器"].head(1))
# @@end
print("drop=first 维度:", oh_d.transform(Xs_tr[CAT]).shape, "= 11 - 2")
print("互感器行 drop 后独热和:", float(oh_d_out[0].sum()), "← 设备类型段全 0 = 基准类")

assert oh.categories_[0].tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert X_oh.shape == (480, 11)
assert oh_d.transform(Xs_tr[CAT]).shape == (480, 9)
assert float(oh_d_out[0][:5].sum()) == 0.0, "设备类型前 5 列全 0"
'''

E5_CODE = '''# 训练段缺陷类型计数: 渗漏油 94 / 放电痕迹 85 / 锈蚀 84 / 发热 80 / 破损 77 / 异物 60

# @@todo(9) 定义 min_frequency=0.15 的 OneHotEncoder 并 transform，观察异物 60 是否被合并
# @@hint OneHotEncoder(sparse_output=False, min_frequency=...)
oh_m1 = OneHotEncoder(sparse_output=False, min_frequency=0.15).fit(Xs_tr[CAT])
out_m1 = oh_m1.transform(Xs_tr[CAT])
# @@end
print("min_frequency=0.15(72): 维度", out_m1.shape, "| 只有异物 60 被合并")
print("  特征名含 infrequent:", [n for n in oh_m1.get_feature_names_out() if "infrequent" in n])

# @@todo(10) 阈值抬到 0.2 再试一次：注意缺陷类型整列的遭遇
# @@hint 0.2 * 480 = 96，比训练段最大的类别计数 94 还大
oh_m2 = OneHotEncoder(sparse_output=False, min_frequency=0.2).fit(Xs_tr[CAT])
out_m2 = oh_m2.transform(Xs_tr[CAT])
# @@end
print("min_frequency=0.2(96): 维度", out_m2.shape, "| 缺陷类型整列失效")
print("  infrequent 桶:", [c.tolist() for c in oh_m2.infrequent_categories_])

assert out_m1.shape == (480, 11), "只合并 1 个类别时维度不变"
assert "缺陷类型_infrequent_sklearn" in oh_m1.get_feature_names_out().tolist()
assert out_m2.shape == (480, 5), "整列被吞: (2 设备保留 + 1) + (0 + 1)"
'''

E6_CODE = '''# @@todo(11) 定义 handle_unknown="ignore" 的 OneHotEncoder，对含未知类别「杆塔」的行 transform
# @@hint unk 是一行 DataFrame；看输出的形状与该行独热和
oh_ign = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(Xs_tr[CAT])
unk = pd.DataFrame({"设备类型": ["杆塔"], "缺陷类型": ["发热"]})
out_unk = oh_ign.transform(unk)
# @@end
print("ignore 遇未知:", out_unk.shape, "| 该行独热和:", float(out_unk.sum()))

# @@todo(12) 用 probe 验证：drop="first" 与 handle_unknown="ignore" 组合，fit 与 transform 是否报错
# @@hint probe(oh_bad.fit, Xs_tr[CAT]) 先接住 fit，再正常调用 transform
oh_bad = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
st_fit, _ = probe(oh_bad.fit, Xs_tr[CAT])
out_bad = oh_bad.transform(unk)
# @@end
print("drop+ignore: fit →", st_fit, "| transform 未知 →", out_bad.shape,
      "| 独热和:", float(out_bad.sum()))

assert out_unk.shape == (1, 11)
assert float(out_unk[0][:5].sum()) == 0.0 and float(out_unk.sum()) == 1.0, "未知段归零，已知段保留"
assert st_fit == "ok" and out_bad.shape == (1, 9) and float(out_bad.sum()) == 0.0, "未知段归零"
'''

E7_CODE = '''# @@todo(13) 定义 OrdinalEncoder，fit 训练段类别列，打印码位序与前 3 行编码
# @@hint OrdinalEncoder().fit(Xs_tr[CAT])；.transform(Xs_tr[CAT].head(3))
oe = OrdinalEncoder().fit(Xs_tr[CAT])
oe_head = oe.transform(Xs_tr[CAT].head(3))
# @@end
print("设备类型码位序:", oe.categories_[0].tolist())
print("前 3 行编码:", oe_head.tolist())

# @@todo(14) 用 probe 验证：默认 OrdinalEncoder 遇到未知类别「杆塔」会怎样
# @@hint probe(oe.transform, 单行 DataFrame)
status, msg = probe(
    oe.transform,
    pd.DataFrame({"设备类型": ["杆塔"], "缺陷类型": ["放电痕迹"]}))
# @@end
print("OrdinalEncoder 遇未知 →", status, "|", msg)

assert oe.categories_[0].tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert oe.transform(Xs_tr[CAT].head(1)).tolist()[0][0] == 3.0, "互感器 = 码位 3"
assert status == "err" and msg == "ValueError"
'''

E8_CODE = '''# @@todo(15) 用 probe 验证：PolynomialFeatures 直接 fit 含 NaN 的数值列会怎样
# @@hint probe(PolynomialFeatures(degree=2).fit, Xs_tr[NUM])
status, msg = probe(PolynomialFeatures(degree=2).fit, Xs_tr[NUM])
# @@end
print("直接对含 NaN 的数值列 fit →", status, "|", msg)

# @@todo(16) 先用题 3 的 imp 填充，再定义 degree=2 / include_bias=False 的 PolynomialFeatures
# @@hint imp.transform(Xs_tr[NUM]) 得到 Xi；PolynomialFeatures(degree=2, include_bias=False).fit(Xi)
Xi = imp.transform(Xs_tr[NUM])
pf2b = PolynomialFeatures(degree=2, include_bias=False).fit(Xi)
Xi2 = pf2b.transform(Xi)
# @@end
print("degree=2 无 bias:", Xi2.shape, "= 5 + 5*6/2")
pf2i = PolynomialFeatures(degree=2, include_bias=False,
                          interaction_only=True).fit(Xi)
print("interaction_only:", pf2i.transform(Xi).shape, "= 5 + 5*4/2")
pf3b = PolynomialFeatures(degree=3, include_bias=False).fit(Xi)
print("degree=3 无 bias:", pf3b.transform(Xi).shape)
print("特征名前 7:", pf2b.get_feature_names_out()[:7].tolist())
print("平方项:", [n for n in pf2b.get_feature_names_out() if "^2" in n])

# @@todo(17) 找出未缩放直接 poly 的最大值来自哪个特征，并对 poly 输出再做一次 StandardScaler
# @@hint np.argmax 用 np.unravel_index 解包；StandardScaler().fit(Xi2)
mx = int(np.unravel_index(np.argmax(Xi2), Xi2.shape)[1])
ss2 = StandardScaler().fit(Xi2)
# @@end
print("未缩放直接 poly 的最大值:", round(float(Xi2.max()), 1),
      "| 来自特征:", pf2b.get_feature_names_out()[mx])
print("poly 后再 scale：x0 std", round(float(ss2.scale_[0]), 2),
      "| x0*x1 std", round(float(ss2.scale_[5]), 2), "← 交互项仍然巨大")

print("测试段 poly 形状:", pf2b.transform(imp.transform(Xs_te[NUM])).shape)

assert status == "err" and msg == "ValueError", "PolynomialFeatures 拒绝 NaN"
assert Xi2.shape == (480, 20) and pf2i.transform(Xi).shape == (480, 15)
assert pf3b.transform(Xi).shape == (480, 55)
assert round(float(Xi2.max()), 1) == 697425.4
assert pf2b.get_feature_names_out()[mx] == "x0^2"
assert pf2b.transform(imp.transform(Xs_te[NUM])).shape == (120, 20)
'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch02 特征工程：scaler 三兄弟 / 编码器 / 填充 / 多项式特征

> 方向：机器学习 ｜ 竞赛对应：**模型训练 40%**（预处理是流水线的第一段，按步给分）
> + **模型选择 10%**（选错缩放/编码方式，后面全歪）。

ch01 把流水线搭起来了，这一章填流水线里的**每一节管道**。四个主题，
每个都藏着「不报错但结果错」的坑：

1. **scaler 三兄弟**（Standard / MinMax / Robust）——量纲统一的三种口径
2. **SimpleImputer**——缺失填充的策略选择与「缺失本身就是特征」
3. **OneHot / OrdinalEncoder**——类别编码的维度、基准类与未知类别
4. **PolynomialFeatures**——特征升维的维度公式与数值爆炸

本章实测出三个反直觉行为（sklearn 1.9.1）：

- **scaler 对 NaN 视而不见**：fit 不报错、transform 原样穿透
- **MinMaxScaler 测试段越出 [0, 1]**：不报错不警告
- **`drop="first"` + `handle_unknown="ignore"` 能跑**：旧教程说必报错，实测不报

> 本章 `assert` 真值全部沉淀在方向 README 的「ch02 专项真值」表里。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 模型训练 40% | 预处理按步给分 | 三种 scaler、SimpleImputer、两种 encoder、PolynomialFeatures |
| 模型选择 10% | 预处理方式与模型要匹配 | 线性模型 drop=first、树模型 Ordinal、min_frequency 降维 |

**学习目标**：拿到一列数值 / 类别特征，能立刻说出该用什么预处理、
它对缺失 / 未知类别 / 离群值分别是什么行为——特别是**哪些行为是静默的**。

## 二、API 速查表

| 类 / 方法 | 关键参数 | 属性 | 一句话说明 |
|---|---|---|---|
| `StandardScaler` | — | `mean_` `scale_` | 减均值除标准差；NaN 原样穿透 |
| `MinMaxScaler` | — | `data_min_` `data_max_` | 压到 [0,1]；测试段可越界 |
| `RobustScaler` | — | `center_` `scale_` | 中位数 + IQR，抗离群值 |
| `SimpleImputer` | `strategy` `fill_value` `add_indicator` | `statistics_` | 缺失填充；indicator 把缺失变特征 |
| `OneHotEncoder` | `drop` `min_frequency` `handle_unknown` | `categories_` | 类别 → 独热；码位序 |
| `OrdinalEncoder` | `unknown_value` `handle_unknown` | `categories_` | 类别 → 整数；默认遇未知就炸 |
| `PolynomialFeatures` | `degree` `include_bias` `interaction_only` | — | 特征升维；**拒绝 NaN** |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        """
## 3.1 scaler 三兄弟：同数据、三种口径

三个缩放器都在**训练段** fit（泄漏问题 ch01 讲透，本章不重复），
看的是它们各自记住了什么统计量。
"""
    ),
    code(S1_CODE),
    md(
        """
### 3.2 难点深挖：scaler 对 NaN 视而不见

**为什么难**：直觉上「含缺失的数据进 scaler 会报错」，实际不会——
sklearn 内部用 nanmean / nanvar 计算统计量，fit 安静通过；
transform 输出里 NaN **原样保留**，把问题推给下游模型。

**错误示范**（实测输出）：

```python
ss.fit(Xs_tr[NUM])          # 不报错！
ss.transform(Xs_tr[NUM])    # 输出 72 个 NaN（36 + 36）
```

**正误对照**：

| 环节 | 行为 |
|---|---|
| `StandardScaler.fit` 含 NaN | 不报错，`mean_` 按忽略 NaN 算 |
| `transform` 输出 | NaN 原样穿透（72 个） |
| `PolynomialFeatures.fit` 含 NaN | **`ValueError`**（§3.8） |

**判定规则**：**scaler 不负责缺失**。填缺失是 `SimpleImputer` 的活，
两者在 `Pipeline` 里的顺序必须是 imputer 在前、scaler 在后；
别指望缩放器替你挡住缺失，也别在缩放前忘了 `PolynomialFeatures` 会炸。
"""
    ),
    code(S2_CODE),
    md(
        """
### 3.3 难点深挖：MinMaxScaler 的测试段越界

**为什么难**：MinMax 的语义是「压到 [0, 1]」，但它只保证**训练段**如此。
测试段出现比训练段 max 更大的值时，输出直接 >1——**不报错、不警告**，
下游如果假设输入在 [0,1]（比如某些神经网络激活、图像模型）就静默出错。

**错误示范**（实测输出）：

```python
mm.transform(Xs_te[NUM])    # 温度 1 个 >1（max 1.0045），湿度 2 个 >1（max 1.0180）
```

**正误对照**：

| 写法 | 测试段范围 |
|---|---|
| `MinMaxScaler`（默认） | 可越界：1.0045 / 1.0180 |
| `RobustScaler` / `StandardScaler` | 语义本来就不限定范围，无此坑 |

**判定规则**：**MinMax 只在「取值范围封闭」的场景用**（图像像素、
物理量纲明确的传感器）。负荷、温度这类有长尾的电力数据用
`StandardScaler` 或 `RobustScaler`；本例负荷值 max 835.12 恰好落在
训练段内纯属运气，不能赌。
"""
    ),
    code(S3_CODE),
    md(
        """
## 3.4 `SimpleImputer`：填充策略与「缺失即特征」

数值列 median、类别列 most_frequent、占位填充 constant。
`add_indicator=True` 把「这个位置曾经缺失」变成一列 0/1 特征——
缺失模式本身带信息（本数据温度缺失集中在 C/D 台区，就是 MAR 结构）。

一个漂亮的互证：**`RobustScaler.center_` 与 `SimpleImputer(strategy="median").statistics_`
逐列相等**（都是各列中位数 427.33 / 35.1 / 64.9 / 16.0 / 44.0）。
"""
    ),
    code(S4_CODE),
    md(
        """
## 3.5 `OneHotEncoder`：维度、基准类与 infrequent 桶

四个考点一次讲完：

1. **码位序**：类别顺序按字典序（互感器 < 变压器 < ...），不是业务序
2. **`drop="first"`**：每列扔掉一个类别当基准，防共线；全 0 行 == 基准类
3. **`min_frequency`**：低频类别合并进 `infrequent_sklearn` 桶
4. **`handle_unknown`**：未知类别只让所在段归零，行内已知类别保持编码
"""
    ),
    code(S5_CODE),
    md(
        """
### 3.6 难点深挖：`min_frequency` 的维度陷阱

**为什么难**：以为「合并低频类别 = 降维」，但**只合并 1 个类别时维度
不变**（它从自己的独热列变成 infrequent 列，一换一）。更险的是阈值
抬得太高，会把**整列所有类别**都吞进桶里，独热编码直接失效。

**错误示范**（实测输出）：

```python
# 训练段缺陷类型最大计数 94（渗漏油）
OneHotEncoder(min_frequency=0.15).fit(...)   # (480, 11) ← 异物 60 被合并，维度没变！
OneHotEncoder(min_frequency=0.2).fit(...)    # (480, 5)  ← 0.2*480=96 > 94，整列 6 类全进桶！
```

**正误对照**：

| 配置 | 维度 | 缺陷类型段 |
|---|---|---|
| `min_frequency=0.15`（72） | (480, 11) | 异物 → infrequent，5 具名 + 1 桶 |
| `min_frequency=0.2`（96） | (480, 5) | **全部 6 类 → infrequent，独热失效** |

**判定规则**：**每次调 `min_frequency` 必查两件事**——
`transform 后的 shape` 和 `get_feature_names_out()` 里 infrequent 列的个数。
合理阈值是「低频尾部有 ≥2 个类别、其余都稳稳高于阈值」。
"""
    ),
    code(S6_CODE),
    md(
        """
### 3.7 难点深挖：`drop="first"` + `handle_unknown="ignore"` 到底报不报错

**为什么难**：大量教程（包括 sklearn 旧文档）说这两个参数互斥、
遇到未知类别会 `ValueError`。sklearn 1.9.1 实测：**fit 不报错、
transform 未知类别也不报错**：未知段全 0、已知段保留（行 (1, 9)，设备类型段独热和 0）。

**判定规则**：**以实测为准，背旧结论会翻车**。工程上仍要谨慎：
drop + ignore 的全 0 行和「基准类」无法区分，线性模型会把未知类别
当成基准类处理——接受不了这个语义就别组合这两个参数。
"""
    ),
    code(S7_CODE),
    md(
        """
### 3.8 难点深挖：`PolynomialFeatures` 的拒绝、爆炸与顺序

**为什么难**：三个坑叠在一个类上——

1. **拒绝 NaN**：与 scaler 相反，`fit` 直接 `ValueError`，必须先填充
2. **维度公式**：5 个特征 degree=2 无 bias = 5 + 5×6/2 = **20**，degree=3 = **55**；
   `interaction_only` 去掉平方项只剩交互项 = 5 + 5×4/2 = **15**
3. **数值爆炸**：负荷值 ~835 平方后 **697425.4**（x0² 项），交互项
   x0·x1 的标准差高达 127980——不缩放直接喂线性模型必炸

**正误对照**：

| 顺序 | 结果 |
|---|---|
| poly → 直接建模 | x0² 量纲 69 万，梯度被大特征绑架 |
| impute → poly → **scale** | 交互项也被拉回单位方差，安全 |

**判定规则**：**升维三步曲 `impute → poly → scale` 写进 Pipeline，
一步都不能少**；测试段全程复用训练段 fit 的转换器，形状必然一致。
"""
    ),
    code(S8_CODE),
    md(
        """
---

## 四、易错点清单

1. scaler fit 含 NaN 不报错，transform 原样穿透——填缺失是 imputer 的活。
2. MinMaxScaler 只保证训练段在 [0,1]，测试段可越界（实测 1.0045 / 1.0180）。
3. `RobustScaler.center_` == 各列中位数 == `SimpleImputer(median).statistics_`。
4. `add_indicator=True` 让形状 5 → 7：缺失模式本身就是 MAR 结构证据。
5. OneHot 类别顺序是码位序；`drop="first"` 后全 0 行 == 基准类，不是丢失。
6. `min_frequency` 只合并 1 个类别时维度不变；阈值过高整列失效（0.2 → (480,5)）。
7. `drop="first"` + `handle_unknown="ignore"` 在 1.9.1 实测能跑，全 0 行与基准类不可区分。
8. `OrdinalEncoder` 默认遇未知类别 `ValueError`；树模型用它要配 `unknown_value`。
9. `PolynomialFeatures` 拒绝 NaN（与 scaler 相反），必须先填充。
10. 未缩放直接 poly：x0² 最大值 697425.4——升维三步曲 impute → poly → scale。

## 五、本章小结

- 三兄弟怎么选：默认 `StandardScaler`；重尾用 `RobustScaler`；
  封闭量纲才用 `MinMaxScaler`（并接受测试段越界风险）。
- 编码怎么选：线性模型 OneHot（+drop），树模型 Ordinal（配 unknown_value）。
- 低频类别治理用 `min_frequency`，改完必查 shape 与特征名。
- 升维三步曲：impute → poly → scale，一步不能少。

### 复盘提问

1. 为什么 scaler 对 NaN 不报错而 PolynomialFeatures 会炸？
2. MinMaxScaler 测试段越界的根源是什么？什么数据敢用它？
3. `min_frequency=0.15` 和 `0.2` 在本数据上差在哪？为什么 0.2 是灾难？
4. `drop="first"` 之后全 0 行是什么语义？和 `ignore` 组合有什么风险？
5. poly 后的最大值来自哪个特征？为什么 poly 之后还要再 scale？

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
# ch02 练习：特征工程

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        """
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 模型训练 40% | 三种 scaler / SimpleImputer / 两种 encoder / PolynomialFeatures |
| 模型选择 10% | 预处理方式与模型类型匹配 |

## 二、API 速查

| 类 | 关键参数 | 一句话 |
|---|---|---|
| `StandardScaler` | — | 减均值除标准差 |
| `MinMaxScaler` | — | 压 [0,1]，测试段可越界 |
| `RobustScaler` | — | 中位数 + IQR |
| `SimpleImputer` | `strategy` `add_indicator` | 缺失填充 |
| `OneHotEncoder` | `drop` `min_frequency` `handle_unknown` | 类别 → 独热 |
| `OrdinalEncoder` | `unknown_value` | 类别 → 整数 |
| `PolynomialFeatures` | `degree` `include_bias` `interaction_only` | 升维（拒绝 NaN） |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：scaler 三兄弟同台（对应讲解 §3.1）"),
    code(E1_CODE),
    md(
        """
**为什么这样做**：三个缩放器记的统计量完全不同——均值/标准差、
min/max、中位数/IQR。RobustScaler 的 center_ 和 scale_ 必须能与
pandas 手算的中位数、IQR 对上账，这一致性就是你的验收标准。
"""
    ),
    md("## 题 2：NaN 穿透与测试段行为（对应讲解 §3.2-3.3）"),
    code(E2_CODE),
    md("## 题 3：SimpleImputer 三种策略（对应讲解 §3.4）"),
    code(E3_CODE),
    md("## 题 4：OneHotEncoder 维度与基准类（对应讲解 §3.5）"),
    code(E4_CODE),
    md("## 题 5：min_frequency 的维度陷阱（对应讲解 §3.6）"),
    code(E5_CODE),
    md(
        """
**为什么这样做**：0.15 和 0.2 只差 0.05，结果从「维度不变」跳到
「整列失效」。低频治理没有免费的默认值，改一次查一次 shape。
"""
    ),
    md("## 题 6：未知类别的两种结局（对应讲解 §3.7）"),
    code(E6_CODE),
    md("## 题 7：OrdinalEncoder 与未知类别（对应讲解 §3.7）"),
    code(E7_CODE),
    md("## 题 8：PolynomialFeatures 三连坑（对应讲解 §3.8）"),
    code(E8_CODE),
    md(
        """
**为什么这样做**：先 probe 确认「poly 拒绝 NaN」再补缺失，这个顺序
本身就是考点；爆炸值 697425.4 来自 x0²，它解释了为什么升维三步曲
impute → poly → scale 少一步都不行。
"""
    ),
    md(
        """
---

## 综合自查

1. 三兄弟各记什么统计量？哪个和 `SimpleImputer(median)` 互证？
2. 题 2 的 72 个 NaN 是怎么来的？下游模型会遇到什么问题？
3. 题 5 两档阈值的结果差异说明什么？怎么避免「整列失效」？
4. 题 6 的组合参数在旧教程里说必报错，实测为什么能跑？风险在哪？
5. 升维三步曲是哪三步？测试段为什么要复用训练段的转换器？

全部答得上来，进入 ch03（特征选择与降维）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
