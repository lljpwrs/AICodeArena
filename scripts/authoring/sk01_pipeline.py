"""ch01 —— 数据划分与流水线：train_test_split / KFold / TimeSeriesSplit / Pipeline

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/sk01_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "02_sklearn"
NAME = "ch01_data_split_pipeline"

HEADER = '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

warnings.simplefilter("ignore")
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (KFold, StratifiedKFold, TimeSeriesSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

DATA = Path("data")

dev = pd.read_csv(DATA / "defect_ml.csv")   # 分类：预测「是否危急」
reg = pd.read_csv(DATA / "load_reg.csv")    # 回归：预测「日均负荷」（后面章节用）

NUM = ["负荷值", "温度", "湿度", "投运年限", "巡检耗时分钟"]
CAT = ["设备类型", "缺陷类型"]
TARGET = "是否危急"

X = dev[NUM + CAT]
y = dev[TARGET]

print("sklearn", sklearn.__version__)
print("dev", dev.shape, "| 危急", int(y.sum()), f"({y.mean():.1%})")
print("reg", reg.shape)'''

SCAFFOLD = '''# 脚手架：写在 @@todo 之外，练习版原样保留
def probe(fn, *args, **kwargs):
    """把「调用可能抛异常」写成返回值，避免在挖空块里写 try/except。

    返回 ("ok", 结果) 或 ("err", 异常类型名)。
    
"""
    try:
        return "ok", fn(*args, **kwargs)
    except Exception as exc:
        return "err", type(exc).__name__


def fold_report(tag, splits, y_series):
    """打印每一折的 (train 行数, test 行数, test 正例数, test 正例率)。
"""
    for i, (tr, te) in enumerate(splits):
        pos = int(y_series.iloc[te].sum())
        print(f"  {tag} fold{i}: train={len(tr)} test={len(te)} "
              f"test正例={pos} 正例率={pos / len(te):.4f}")


def make_ct(remainder="drop"):
    """统一的 ColumnTransformer：数值 = 中位数填充 + 标准化；类别 = 众数填充 + OneHot。
"""
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)],
                             remainder=remainder)


# 演示 TimeSeriesSplit 用的时序数据：365 天日负荷（干净，只为看折结构）
TS = pd.DataFrame({"t": np.arange(365),
                   "负荷": np.random.default_rng(42).normal(400, 50, 365)})

print("TS", TS.shape)'''


# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        
"""
# ch01 数据划分与流水线：train_test_split / 三种 CV / Pipeline

> 方向：机器学习 ｜ 竞赛对应：**模型选择 10%**（划分方式选错 = 从源头泄漏）
> + **模型训练 40%**（Pipeline 组装是训练流程的第一步，按步给分）。

这一章本身不训练任何"像样"的模型——它解决的是**模型开跑之前的所有事**。
竞赛里最常见的隐形丢分就藏在这里，而且全部**不报错**：

1. `train_test_split` 不给 `random_state` → 结果不可复现
2. 不平衡数据不分层 → 训练/测试正例占比漂移（实测 0.225 vs 0.1667）
3. **先全量 fit 缩放器再划分 → 数据泄漏**，指标虚高还自我感觉良好
4. 时序数据用 `KFold(shuffle=True)` → 用未来预测过去
5. `ColumnTransformer` 默认 `remainder="drop"` → 忘写的列**静默消失**
6. 测试集冒出新类别 → `OneHotEncoder` 默认直接 `ValueError`

> 本章的 `assert` 全部可以在方向 README 的「ch01 专项真值」对上。
"""
    ),
    md(
        
"""
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 模型选择 10% | 划分/验证方式是否正确 | `train_test_split` 三参数、三种 CV 选型 |
| 模型训练 40% | 流程完整性按步给分 | `Pipeline` + `ColumnTransformer` 组装、`fit` / `score` |

**学习目标**：拿到一张新表，能在 5 分钟内搭出一条「划分 → 预处理 → 模型」的
无泄漏流水线，并能说清每一步为什么这么切。
"""
    ),
    md(
        
"""
## 二、API 速查表

| 方法 / 类 | 关键参数 | 返回 | 一句话说明 |
|---|---|---|---|
| `train_test_split` | `test_size` `random_state` `stratify` | 4 个数组 | 一次性切 train/test |
| `KFold` | `n_splits` `shuffle` `random_state` | 迭代器 | 普通折外验证 |
| `StratifiedKFold` | 同上 | 迭代器 | **分类必用**：每折保持正例占比 |
| `TimeSeriesSplit` | `n_splits` `test_size` `gap` | 迭代器 | **时序必用**：train 只增不重排 |
| `Pipeline` | `[(名字, 估计器), ...]` | 估计器 | 顺序执行 fit/transform/predict |
| `ColumnTransformer` | `transformers` `remainder` | 转换器 | 按列选不同的预处理 |
| `SimpleImputer` | `strategy` | 转换器 | 缺失填充 |
| `StandardScaler` | — | 转换器 | 减均值除标准差 |
| `OneHotEncoder` | `handle_unknown` `sparse_output` | 转换器 | 类别 → 独热 |
| `OrdinalEncoder` | `handle_unknown` `unknown_value` | 转换器 | 类别 → 整数 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md(
        
"""
## 3.1 `train_test_split`：保留法划分

三个参数一个都不能省：`test_size` 定比例、`random_state` 定可复现、
`stratify` 定是否按标签分层。**竞赛评分表里"结果可复现"是硬要求。**
"""
    ),
    code(
        """X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

print("形状:", X_tr.shape, X_te.shape, y_tr.shape, y_te.shape)
print("训练段正例:", int(y_tr.sum()), f"({y_tr.mean():.4f})",
      "| 测试段正例:", int(y_te.sum()), f"({y_te.mean():.4f})")
print("训练段索引前 6:", X_tr.index[:6].tolist())

# 可复现性：同样的参数，切两次必须完全一致
again = train_test_split(X, y, test_size=0.2, random_state=42)
print("同参数再切一次，索引一致:", bool(again[0].index.equals(X_tr.index)))

# 反例：不给 random_state
a = train_test_split(X, y, test_size=0.2)
b = train_test_split(X, y, test_size=0.2)
print("不给 random_state，两次一致:", bool(a[0].index.equals(b[0].index)), "← 不可复现")

assert X_tr.shape == (480, 7) and X_te.shape == (120, 7), "600 行按 8:2 切"
assert int(y_tr.sum()) == 108 and int(y_te.sum()) == 20, "seed=42 的划分真值"
assert X_tr.index[:6].tolist() == [145, 9, 375, 523, 188, 131], "索引真值可锚定"
assert not a[0].index.equals(b[0].index), "不给 seed 两次结果应不同"
"""
    ),
    md(
        
"""
## 3.2 `stratify`：不平衡数据的救命参数

本数据危急占比 21.3%。不分层时，测试段占比随机漂到 16.7%；
分层后锁在 21.7%。**二分类 / 多分类任务，`stratify=y` 无条件写上。**
"""
    ),
    code(
        """Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("分层后：训练正例", int(ys_tr.sum()), f"({ys_tr.mean():.4f})",
      "| 测试正例", int(ys_te.sum()), f"({ys_te.mean():.4f})")

# 对照：换个 test_size，索引完全不同 —— 参数一变，划分就变
X2_tr, _ = train_test_split(X, test_size=0.25, random_state=42)
print("test_size=0.25 的训练段前 5 索引:", X2_tr.index[:5].tolist())

assert int(ys_tr.sum()) == 102 and int(ys_te.sum()) == 26, "分层真值"
assert round(ys_tr.mean(), 4) == 0.2125 and round(ys_te.mean(), 4) == 0.2167, \\
    "分层后两段占比都应贴近全局 0.2133"
assert X2_tr.index[:5].tolist() == [593, 531, 353, 332, 534], "test_size 一变划分全变"
"""
    ),
    md(
        
"""
### 3.3 难点深挖：数据泄漏——为什么"先缩放再划分"是错的

**为什么难**：这是 sklearn 题里最经典的隐形丢分。泄漏**不报错**，
指标还往往**更好看**， Reviewer 不盯着预处理顺序根本看不出来。

**错误示范**（贴实际输出）：

```python
sc_full = StandardScaler().fit(X[NUM])          # ← 在全量 600 行上 fit
mu_full = sc_full.mean_
# 之后才 train_test_split —— 测试集的均值/方差已经混进了训练变换里
```

全量 fit 与"只用训练段 fit"的均值差（本数据实测）：
`[0.2692, 0.2457, 0.1444, 0.0083, 0.2092]`——`温度` 全量均值 35.264014 vs
训练段 35.509685。缩放器见过测试集，测试集就不再"未知"。

**正误对照**：

| 写法 | 后果 |
|---|---|
| `fit(全量)` 再切分 | 泄漏：测试信息进入训练变换，指标虚高 |
| 把缩放器放进 `Pipeline`，只 `fit` 训练段 | 每次用训练段统计量，测试段只 `transform` |

**判定规则**：**一切 fit（填充、缩放、编码、特征选择）只允许接触训练段；
`Pipeline` 把预处理和模型绑成一个估计器，就是让 sklearn 帮你在交叉验证里
自动做到这一点。** 背下来：`Pipeline` 不是让代码好看，是防泄漏的机制。
"""
    ),
    code(
        """sc_full = StandardScaler().fit(X[NUM])
mu_full = sc_full.mean_

sc_tr = StandardScaler().fit(X_tr[NUM])          # 只用上面 §3.1 的训练段
print("全量均值    :", np.round(mu_full, 6).tolist())
print("训练段均值  :", np.round(sc_tr.mean_, 6).tolist())
print("差值        :", np.round(np.abs(mu_full - sc_tr.mean_), 4).tolist())
print("温度均值: 全量", round(float(mu_full[1]), 6),
      "vs 训练段", round(float(sc_tr.mean_[1]), 6), "← 不相等 = 全量 fit 混入了测试信息")

assert round(float(mu_full[1]), 6) == 35.264014, "全量温度均值真值"
assert round(float(sc_tr.mean_[1]), 6) == 35.509685, "训练段温度均值真值"
"""
    ),
    md(
        
"""
### 3.4 难点深挖：`KFold` vs `StratifiedKFold` vs `TimeSeriesSplit`

**为什么难**：三种分割器对同一份数据给出**都合法但语义完全不同**的折；
选错了不报错，评估结果先是难看（分层缺位），再是作弊（时序混洗）。

**错误示范**：不平衡数据用普通 `KFold`——每折正例数 20 ~ 34 跳变，
折间指标不稳定还找不到原因。

**正误对照**（本数据 600 行 / 128 正例，5 折）：

| 分割器 | 每折 test 正例 | 正例率范围 |
|---|---|---|
| `StratifiedKFold` | [25, 25, 26, 26, 26] | 0.208 ~ 0.217 ✅ |
| `KFold(shuffle=True)` | [20, 21, 34, 27, 26] | 0.167 ~ 0.283 ❌ |

**判定规则**（背下来）：

1. **有标签的分类任务** → `StratifiedKFold`（`shuffle=True` + `random_state`）
2. **无标签（聚类）/ 回归** → `KFold`（没有标签可分层）
3. **数据有序、时间相关** → `TimeSeriesSplit`，**永远不许 shuffle**
"""
    ),
    code(
        """skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_report("strat", skf.split(X, y), y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_report("kfold", kf.split(X), y)

skf_pos = [int(y.iloc[te].sum()) for _, te in skf.split(X, y)]
kf_pos = [int(y.iloc[te].sum()) for _, te in kf.split(X, y)]
print("strat 各折正例:", skf_pos, "| kfold:", kf_pos)
print("折间正例率波动: strat", round(max(skf_pos) - min(skf_pos), 2),
      "个 vs kfold", round(max(kf_pos) - min(kf_pos), 2), "个")

assert skf_pos == [25, 25, 26, 26, 26], "StratifiedKFold 真值"
assert kf_pos == [20, 21, 34, 27, 26], "KFold 真值（折间跳变是特性不是 bug）"
"""
    ),
    md(
        
"""
## 3.5 `TimeSeriesSplit`：时序数据的唯一合法切法

train 从最早期开始、只增不减；test 永远在 train 的**未来**。
`gap` 用来在两者之间留空窗（比如标签要 T+1 才知道）。
"""
    ),
    code(
        """tss = TimeSeriesSplit(n_splits=5)
for i, (tr, te) in enumerate(tss.split(TS)):
    print(f"fold{i}: train {tr[0]}..{tr[-1]} ({len(tr)}) | "
          f"test {te[0]}..{te[-1]} ({len(te)})")

print("\\n加 gap=7、固定 test_size=60：")
tss2 = TimeSeriesSplit(n_splits=3, test_size=60, gap=7)
for i, (tr, te) in enumerate(tss2.split(TS)):
    print(f"fold{i}: train 0..{tr[-1]} ({len(tr)}) | test {te[0]}..{te[-1]} "
          f"({len(te)}) | 中间隔 {te[0] - tr[-1] - 1} 天")

tss_splits = list(TimeSeriesSplit(n_splits=5).split(TS))
train_sizes = [len(tr) for tr, _ in tss_splits]
test_sizes = [len(te) for _, te in tss_splits]
print("\\ntrain 尺寸:", train_sizes, "| test 尺寸:", test_sizes)

assert train_sizes == [65, 125, 185, 245, 305], "train 递增 +65 每折"
assert test_sizes == [60] * 5, "test 恒 60"
assert tss_splits[0][1][0] == 65, "fold0 的 test 从 65 开始"
g3 = list(TimeSeriesSplit(n_splits=3, test_size=60, gap=7).split(TS))
assert [len(tr) for tr, _ in g3] == [178, 238, 298], "gap=7 折结构真值"
assert all(te[0] - tr[-1] - 1 == 7 for tr, te in g3), "train 末与 test 始之间空 7 天"
"""
    ),
    md(
        
"""
## 3.6 `Pipeline` + `ColumnTransformer`：把预处理装进流水线

数值列：中位数填充 + 标准化；类别列：众数填充 + OneHot。
两路变换在 `ColumnTransformer` 里拼起来，再和模型一起装进 `Pipeline`。
"""
    ),
    code(
        """ct = make_ct()
Xt = ct.fit_transform(X)

print("变换后形状:", Xt.shape, "= 5 数值 + 11 独热")
print("特征名:", ct.get_feature_names_out()[:8].tolist(), "...")

num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
ct_manual = ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])
print("与 make_ct 等价:", bool((ct_manual.fit_transform(X) == Xt).all()))

assert Xt.shape == (600, 16), "5 数值 + (5+6) 独热 = 16 列"
assert len(ct.get_feature_names_out()) == 16
assert ct.get_feature_names_out()[0] == "num__负荷值"
assert "cat__设备类型_互感器" in ct.get_feature_names_out().tolist()
"""
    ),
    md(
        
"""
### 3.7 难点深挖：`ColumnTransformer` 默认 `remainder="drop"` 会静默丢列

**为什么难**：给 DataFrame 加了一列（比如刚算出的特征），忘了把它写进
`transformers`——**不报错、不警告**，那一列直接消失，模型少用一路信息。

**错误示范**：

```python
X3 = X.assign(备注=["x"] * len(X))
ct.fit_transform(X3).shape      # (600, 16) ← 和不加"备注"一模一样！
```

**正误对照**：

| 写法 | 输出形状 | 备注 |
|---|---|---|
| 默认 `remainder="drop"` | (600, 16) | `备注` 被静默丢弃 |
| `remainder="passthrough"` | (600, 17) | 末列名 `remainder__备注` |

**判定规则**：**每次改完特征列，必查 `ct.get_feature_names_out()` 的长度**；
明确想保留剩余列时才用 `remainder="passthrough"`，其余场景保持 drop 但要心里有数。
"""
    ),
    code(
        """X3 = X.assign(备注=["x"] * len(X))
Xt2 = make_ct().fit_transform(X3)
Xt3 = make_ct(remainder="passthrough").fit_transform(X3)

print("不加备注:", make_ct().fit_transform(X).shape,
      "| 加了备注(默认):", Xt2.shape, "→ 一模一样，备注被静默丢弃")
print("remainder=passthrough:", Xt3.shape,
      "| 末列名:", ct3_name := make_ct(remainder="passthrough")
      .fit(X3).get_feature_names_out()[-1])

assert Xt2.shape == (600, 16), "默认 drop：新列静默消失"
assert Xt3.shape == (600, 17), "passthrough：多出一列"
assert ct3_name == "remainder__备注", "被保留列的特征名前缀"
"""
    ),
    md(
        
"""
### 3.9 难点深挖：测试集冒出新类别——`handle_unknown` 的两难

**为什么难**：训练段没有的类别在测试段出现，是真实数据的常态
（竞赛切分数据后必现）。`OneHotEncoder` 默认 `handle_unknown="error"`，
一旦出现直接 `ValueError`；改 `"ignore"` 能跑，但该行独热**全 0**。

**正误对照**（训练段学过 5 种设备类型，测试出现 `杆塔`）：

| 写法 | 结果 |
|---|---|
| 默认 `handle_unknown="error"` | **`ValueError: Found unknown categories ['杆塔'] ...`** |
| `handle_unknown="ignore"` | 全 0 行，`shape (1, 5)`，不报错 |
| `OrdinalEncoder(unknown_value=-1, handle_unknown="use_encoded_value")` | 输出 `-1`，不报错 |

**判定规则**：**线性和线性核模型 → OneHot + `ignore`（全 0 行 = "未见过"）；
树模型 → OneHot 或 `OrdinalEncoder` 皆可，但必须显式给 `unknown_value`，
不能留默认让它在预测时炸。** OneHot 类别顺序是**码位序**（互感器 < 变压器 <
断路器 < 绝缘子 < 避雷器），不是业务序，与 pandas 的 Categorical 同一个坑。
"""
    ),
    code(
        """oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
oh.fit(X[["设备类型"]])
print("学到的类别(码位序):", oh.categories_[0].tolist())
out = oh.transform(pd.DataFrame({"设备类型": ["杆塔", "变压器"]}))
print("ignore 遇未知类别: shape", out.shape,
      "| 未知行独热和", float(out[0].sum()), "| 已知行独热和", float(out[1].sum()))

oh_strict = OneHotEncoder(sparse_output=False)
oh_strict.fit(X[["设备类型"]])
status, msg = probe(oh_strict.transform, pd.DataFrame({"设备类型": ["杆塔"]}))
print("默认 error →", status, "|", str(msg)[:60], "...")

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
oe.fit(X[["设备类型"]])
print("OrdinalEncoder 遇未知:", oe.transform(pd.DataFrame({"设备类型": ["杆塔"]})).ravel().tolist())

assert oh.categories_[0].tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert out.shape == (2, 5) and float(out[0].sum()) == 0.0, "未知类别 → 全 0 行"
assert status == "err" and msg == "ValueError", "默认配置直接报错"
"""
    ),
    md(
        
"""
## 3.10 全流程真值：从原始 CSV 到可交付流水线

一条 `Pipeline` = `ColumnTransformer`（防泄漏预处理）+ 逻辑回归。
注意最后的数字：模型预测正例 9 个、实际 26 个——**这不是错，
是 0.5 阈值在 21% 不平衡数据上的必然表现**，ch07（评估）和 ch12（不平衡）展开。
"""
    ),
    code(
        """pipe = Pipeline([("ct", make_ct()), ("clf", LogisticRegression(max_iter=1000))])
pipe.fit(Xs_tr, ys_tr)                      # 只接触训练段！

acc = pipe.score(Xs_te, ys_te)              # score 内部只 transform 测试段
pred = pipe.predict(Xs_te)
print("Pipeline test accuracy:", round(float(acc), 4))
print("预测正例", int(pred.sum()), "个 vs 实际正例", int(ys_te.sum()), "个 → 阈值问题预告")
print("原始特征数:", pipe.named_steps["ct"].n_features_in_,
      "| 变换后维度:", pipe.named_steps["clf"].n_features_in_)

assert round(float(acc), 4) == 0.7917, "分层切分 + 该流水线的真值"
assert int(pred.sum()) == 9 and int(ys_te.sum()) == 26, "不平衡数据的阈值现象"
assert pipe.named_steps["ct"].n_features_in_ == 7, "ct 看到的是 7 个原始特征"
"""
    ),
    md(
        
"""
---

## 四、易错点清单

1. `train_test_split` 不写 `random_state` → 不可复现，评分表"结果一致"直接丢分。
2. 分类任务不写 `stratify=y` → 训练/测试正例占比漂移（0.225 vs 0.1667）。
3. **先全量 `fit` 缩放/填充/编码再划分 = 数据泄漏**，指标虚高不报错。
4. 时序数据用 `KFold(shuffle=True)` = 用未来预测过去，必须 `TimeSeriesSplit`。
5. `ColumnTransformer` 默认 `remainder="drop"` → 忘写的列静默消失。
6. `OneHotEncoder` 默认 `handle_unknown="error"` → 测试段新类别直接 `ValueError`。
7. `KFold` 与 `StratifiedKFold` 每折正例可差 14 个（20 vs 34）——指标不稳先查分层。
8. `Pipeline` 的最后一步才是模型；取预处理器用 `pipe.named_steps["ct"]`。
9. `n_features_in_` 是**原始特征数**（7），变换后维度看 `clf.n_features_in_`（16）。
10. OneHot 的类别顺序是码位序，不是数据出现序、更不是业务序。

## 五、本章小结

- 划分三件套：`test_size` + `random_state` + `stratify`，一个不能省。
- 三种 CV 的判定规则：分类分层、回归普通折、时序不混洗。
- `Pipeline` + `ColumnTransformer` 是防泄漏机制，不是代码美化。
- 每次改特征后必查 `get_feature_names_out()`。
- 未知类别提前声明 `handle_unknown` / `unknown_value`。

### 复盘提问

1. 为什么"先在全量上 fit 标准化、再划分"是错的？数值上差多少？
2. 三种分割器分别什么时候用？时序数据的 `gap` 参数是干什么的？
3. `ColumnTransformer` 默认会怎么处理没写进 `transformers` 的列？
4. `handle_unknown="ignore"` 遇到新类别时，独热编码输出是什么？
5. `pipe.score()` 内部对测试段做了 fit 还是 transform？

答不上来的，回 `_notes/错题本.md` 记一笔。
"""
    ),
]


# =========================================================================== #
# 练习 notebook
# =========================================================================== #

EXERCISE = [
    md(
        
"""
# ch01 练习：数据划分与流水线

与讲解版逐 Cell 对应。核心代码被挖空，`assert` 验收**保留**——
跑通所有 `assert` 即自查通过。卡住就回讲解版看对应小节。
"""
    ),
    md(
        
"""
## 一、考点回顾

| 竞赛评分点 | 本节覆盖 |
|---|---|
| 模型选择 10% | 划分 + 三种 CV 选型 |
| 模型训练 40% | Pipeline + ColumnTransformer 组装 |
"""
    ),
    md(
        
"""
## 二、API 速查

| 方法 | 关键参数 | 一句话 |
|---|---|---|
| `train_test_split` | `test_size` `random_state` `stratify` | 保留法划分 |
| `StratifiedKFold` | `n_splits` `shuffle` `random_state` | 分类折外验证 |
| `TimeSeriesSplit` | `n_splits` `test_size` `gap` | 时序折外验证 |
| `Pipeline` | `[(名, 步骤)]` | 顺序执行防泄漏 |
| `ColumnTransformer` | `transformers` `remainder` | 按列预处理 |
"""
    ),
    code(HEADER),
    code(SCAFFOLD),
    md("## 题 1：保留法划分（对应讲解 §3.1）"),
    code(
        """# @@todo(1) 按 8:2 切分 X, y，固定随机种子 42，得到 X_tr, X_te, y_tr, y_te
# @@hint train_test_split(X, y, test_size=..., random_state=...)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
# @@end

# 切两次必须一致（可复现是评分硬要求）
again = train_test_split(X, y, test_size=0.2, random_state=42)
print("形状:", X_tr.shape, X_te.shape, "| 索引前 6:", X_tr.index[:6].tolist())
print("可复现:", bool(again[0].index.equals(X_tr.index)))

assert X_tr.shape == (480, 7) and X_te.shape == (120, 7)
assert int(y_tr.sum()) == 108 and int(y_te.sum()) == 20
assert X_tr.index[:6].tolist() == [145, 9, 375, 523, 188, 131]
"""
    ),
    md(
        
"""
**为什么这样做**：`random_state=42` 让评分官重跑你的代码得到一模一样的划分；
不写这个参数，两次运行连你自己都对不上。
"""
    ),
    md("## 题 2：不平衡数据的分层切分（对应讲解 §3.2）"),
    code(
        """# @@todo(2) 在题 1 基础上加分层参数，得到 Xs_tr, Xs_te, ys_tr, ys_te
# @@hint stratify 接的是标签 y
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# @@end

print("分层后：训练正例", int(ys_tr.sum()), f"({ys_tr.mean():.4f})",
      "| 测试正例", int(ys_te.sum()), f"({ys_te.mean():.4f})")

assert int(ys_tr.sum()) == 102 and int(ys_te.sum()) == 26
assert round(ys_tr.mean(), 4) == 0.2125 and round(ys_te.mean(), 4) == 0.2167
"""
    ),
    md("## 题 3：StratifiedKFold vs KFold（对应讲解 §3.4）"),
    code(
        """# @@todo(3) 定义 5 折的 StratifiedKFold（shuffle=True，种子 42），并生成折列表
# @@hint StratifiedKFold(n_splits=..., shuffle=..., random_state=...) 的 .split(X, y)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
strat_splits = list(skf.split(X, y))
# @@end

# @@todo(4) 再定义一个普通 KFold（同样 5 折、shuffle、种子 42），生成折列表
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kfold_splits = list(kf.split(X))
# @@end

skf_pos = [int(y.iloc[te].sum()) for _, te in strat_splits]
kf_pos = [int(y.iloc[te].sum()) for _, te in kfold_splits]
print("strat 各折正例:", skf_pos)
print("kfold 各折正例:", kf_pos, "← 折间跳变，分层缺位")

assert skf_pos == [25, 25, 26, 26, 26]
assert kf_pos == [20, 21, 34, 27, 26]
"""
    ),
    md("## 题 4：TimeSeriesSplit 折结构（对应讲解 §3.5）"),
    code(
        """# @@todo(5) 对 TS 定义 5 折的 TimeSeriesSplit，并生成折列表
# @@hint TimeSeriesSplit(n_splits=...) 的 .split(TS)
tss = TimeSeriesSplit(n_splits=5)
ts_splits = list(tss.split(TS))
# @@end

train_sizes = [len(tr) for tr, _ in ts_splits]
test_sizes = [len(te) for _, te in ts_splits]
print("train:", train_sizes, "| test:", test_sizes)

assert train_sizes == [65, 125, 185, 245, 305]
assert test_sizes == [60] * 5
assert ts_splits[0][1][0] == 65
"""
    ),
    md(
        
"""
**为什么这样做**：时序折的 train 只能来自 test 的过去。索引 0..64 训练、
65..124 验证，才符合"上线时模型只见过历史"的真实场景。
"""
    ),
    md("## 题 5：ColumnTransformer 组装与 remainder 陷阱（对应讲解 §3.6-3.7）"),
    code(
        """# @@todo(6) 复用脚手架的 make_ct()：对 X 做变换，结果存入 Xt
ct = make_ct()
Xt = ct.fit_transform(X)
# @@end

print("变换后形状:", Xt.shape)
assert Xt.shape == (600, 16)
assert len(ct.get_feature_names_out()) == 16
assert ct.get_feature_names_out()[0] == "num__负荷值"

# @@todo(7) 给 X 加一列「备注」后用默认 make_ct() 变换，观察是否被静默丢弃
# @@hint X.assign(备注=[...] * len(X))
X3 = X.assign(备注=["x"] * len(X))
Xt_drop = make_ct().fit_transform(X3)
Xt_pass = make_ct(remainder="passthrough").fit_transform(X3)
# @@end

print("默认:", Xt_drop.shape, "| passthrough:", Xt_pass.shape)
assert Xt_drop.shape == (600, 16), "默认 drop：备注列静默消失"
assert Xt_pass.shape == (600, 17), "passthrough：多出一列"
"""
    ),
    md("## 题 6：未知类别与全流程流水线（对应讲解 §3.9-3.10）"),
    code(
        """# @@todo(8) 定义 handle_unknown="ignore" 的 OneHotEncoder（非稀疏输出），
#            在 X[['设备类型']] 上 fit，并对 ['杆塔', '变压器'] 做变换存入 out
# @@hint OneHotEncoder(handle_unknown=..., sparse_output=False)
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
oh.fit(X[["设备类型"]])
out = oh.transform(pd.DataFrame({"设备类型": ["杆塔", "变压器"]}))
# @@end

print("学到的类别:", oh.categories_[0].tolist())
print("未知类别行的独热和:", float(out[0].sum()))
assert oh.categories_[0].tolist() == ["互感器", "变压器", "断路器", "绝缘子", "避雷器"]
assert out.shape == (2, 5) and float(out[0].sum()) == 0.0
"""
    ),
    code(
        """# @@todo(9) 把 make_ct() 与 LogisticRegression(max_iter=1000) 装进一条 Pipeline
# @@hint Pipeline([("ct", ...), ("clf", ...)])
pipe = Pipeline([("ct", make_ct()), ("clf", LogisticRegression(max_iter=1000))])
# @@end

# @@todo(10) 用分层训练段 fit，在分层测试段上 score，结果存入 acc
pipe.fit(Xs_tr, ys_tr)
acc = pipe.score(Xs_te, ys_te)
# @@end

pred = pipe.predict(Xs_te)
print("test accuracy:", round(float(acc), 4))
print("预测正例", int(pred.sum()), "vs 实际", int(ys_te.sum()), "← 阈值现象，ch07 见")

assert round(float(acc), 4) == 0.7917
assert int(pred.sum()) == 9 and int(ys_te.sum()) == 26
"""
    ),
    md(
        
"""
---

## 综合自查

1. 题 1、2 的差别只有 `stratify=y` 一个参数——为什么占比从漂移变成锁定？
2. 题 3 两种分割器差在哪？什么数据必须用第三种？
3. 题 7 里被丢掉的列，怎么改参数能保住？
4. 题 9 的 Pipeline 为什么天然防泄漏？

全部答得上来，进入 ch02（特征工程）。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
