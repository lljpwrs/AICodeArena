"""dbg02 —— sklearn 数据泄漏（改错题）。

四道题的错误代码均已实跑确认：
  · 题 1 划分前标准化     → 全量统计量混入训练过程
  · 题 2 全量 fit 编码器  → 测试集独有类别泄漏进特征空间
  · 题 3 测试集 fit_transform → 覆盖 scaler，测试集统计量泄漏
  · 题 4 时序随机切分     → 测试点散落在训练区间内部

运行：
    python scripts/authoring/dbg02_sklearn_leak.py
"""

from __future__ import annotations

from pathlib import Path

from _debug_core import build_debug

OUT = Path(__file__).resolve().parents[2] / "coding" / "99_debug"
NAME = "dbg02_sklearn_leak"

INTRO = [
    """# 改错题 · sklearn 数据泄漏

**数据泄漏（data leakage）** 是竞赛里最隐蔽的失分点：代码一行不错，指标还很好看，
但模型一上真实数据就崩 —— 因为它「偷看」了测试集。

这类错误有个共同特征：**它不会报错**。只能靠你对流程顺序的敏感度发现。
""",
    """## 排查手法

问自己一句：**「这一步用到的信息，在真实预测时能拿到吗？」**

- 标准化用的均值/方差 —— 来自训练集还是全量？
- 编码器的类别表 —— 见过测试集的类别吗？
- 切分之后，测试集有没有再被「碰」过？
- 时序数据，测试集在训练集的**之后**吗？

> 改完对 `_solution.ipynb`，每题都写了「为什么」和「怎么自查」。
""",
]

BUG1 = '''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = np.column_stack([rng.normal(0, 1, 200), rng.normal(100, 15, 200)])
y = (X[:, 1] > 100).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.35, random_state=42
)
print("划分完成", X_train.shape, X_test.shape)
'''

FIX1 = '''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = np.column_stack([rng.normal(0, 1, 200), rng.normal(100, 15, 200)])
y = (X[:, 1] > 100).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("训练集均值", np.round(X_train.mean(axis=0), 6))
print("测试集均值", np.round(X_test.mean(axis=0), 4))

# 为什么：StandardScaler 的「均值 / 标准差」是**从数据里算出来的统计量**。
# 划分前 fit_transform，等于把测试集的分布信息提前喂进了预处理流程 —— 典型泄漏。
# 顺序铁律：先划分 → 只对训练集 fit → 测试集只许 transform。
assert np.allclose(X_train.mean(axis=0), 0, atol=1e-9)
assert not np.allclose(X_test.mean(axis=0), 0, atol=1e-9)
print("✅ 通过")
'''

BUG2 = '''
import numpy as np
from sklearn.preprocessing import OneHotEncoder

cats = np.array([["A"], ["B"], ["A"], ["C"], ["D"], ["B"], ["E"], ["A"]])
train_idx = [0, 1, 2, 3, 4]
test_idx = [5, 6, 7]

encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(cats)

print("编码器学到的类别：", encoder.categories_[0].tolist())
'''

FIX2 = '''
import numpy as np
from sklearn.preprocessing import OneHotEncoder

cats = np.array([["A"], ["B"], ["A"], ["C"], ["D"], ["B"], ["E"], ["A"]])
train_idx = [0, 1, 2, 3, 4]
test_idx = [5, 6, 7]

encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(cats[train_idx])

print("编码器学到的类别：", encoder.categories_[0].tolist())
print("测试集编码：\\n", encoder.transform(cats[test_idx]).toarray())

# 为什么：编码器把「见过哪些类别」当成知识存下来，并为每个类别分配一个 one-hot 维度。
# 用全量 fit，测试集独有的类别（E）也会挤进特征空间 —— 等于提前知道测试集有哪些值。
# 这正是 `handle_unknown="ignore"` 存在的意义：让 transform 阶段没见过的类别（E）全部置 0。
assert encoder.categories_[0].tolist() == ["A", "B", "C", "D"]
assert encoder.transform(cats[test_idx]).toarray().shape[1] == 4
print("✅ 通过")
'''

BUG3 = '''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = np.column_stack([rng.normal(0, 1, 200), rng.normal(100, 15, 200)])

X_train, X_test = train_test_split(X, test_size=0.35, random_state=42)

scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print("测试集缩放后均值：", np.round(X_test_scaled.mean(axis=0), 6))
print("scaler 现在的均值：", np.round(scaler.mean_, 4))
'''

FIX3 = '''
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
X = np.column_stack([rng.normal(0, 1, 200), rng.normal(100, 15, 200)])

X_train, X_test = train_test_split(X, test_size=0.35, random_state=42)

scaler = StandardScaler().fit(X_train)
X_test_scaled = scaler.transform(X_test)

print("测试集缩放后均值：", np.round(X_test_scaled.mean(axis=0), 4))
print("scaler 的均值仍是训练集的：", np.round(scaler.mean_, 4))

# 为什么：`fit_transform` = `fit` + `transform`，它会**重新计算并覆盖** scaler 的均值/方差。
# 在测试集上调用有两个后果：
#   ① 测试集的统计量泄漏进预处理；
#   ② 已经 fit 好的 scaler 被**污染** —— 之后再拿它 transform 别的东西，
#      用的就是测试集的均值了（比第一个后果更隐蔽）。
# 自查手法：打印 `scaler.mean_`，看它是否还等于训练集均值。
assert not np.allclose(X_test_scaled.mean(axis=0), 0, atol=1e-9)
assert np.allclose(scaler.mean_, X_train.mean(axis=0))
print("✅ 通过")
'''

BUG4 = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ts = pd.DataFrame({
    "时间": pd.date_range("2025-01-01", periods=100),
    "负荷": np.random.default_rng(42).normal(100, 10, 100),
})

train, test = train_test_split(ts, test_size=0.25, random_state=42)

print("训练集：", train["时间"].min().date(), "~", train["时间"].max().date())
print("测试集：", test["时间"].min().date(), "~", test["时间"].max().date())
'''

FIX4 = '''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # noqa: F401

ts = pd.DataFrame({
    "时间": pd.date_range("2025-01-01", periods=100),
    "负荷": np.random.default_rng(42).normal(100, 10, 100),
})

ts = ts.sort_values("时间").reset_index(drop=True)
cut = int(len(ts) * 0.75)
train, test = ts.iloc[:cut], ts.iloc[cut:]

print("训练集：", train["时间"].min().date(), "~", train["时间"].max().date())
print("测试集：", test["时间"].min().date(), "~", test["时间"].max().date())

# 为什么：`train_test_split` 默认 `shuffle=True`，会把时间点打乱。
# 时序任务里这意味着**用未来的数据预测过去** —— 测试集不再是「未来」，
# 指标会虚高，上线就崩。
# 正确做法：按时间顺序切，前 75% 训练、后 25% 测试（或用 TimeSeriesSplit）。
assert train["时间"].max() < test["时间"].min()
print("✅ 通过")
'''

TASKS = [
    ("划分之前就标准化了",
     "先 `fit_transform` 全量数据，再划分训练/测试集。\n\n"
     "跑一遍看看：不报错，一切正常 —— 但这种「正常」正是问题所在。\n\n"
     "**要求**：把顺序改成「先划分，再 fit 训练集」。",
     BUG1, FIX1),
    ("用全量数据 fit 编码器",
     "训练集只有 A~D，测试集里有 E。\n\n"
     "跑一遍看看：编码器学到的类别是 `['A','B','C','D','E']` —— **测试集的 E 泄漏了**。",
     BUG2, FIX2),
    ("在测试集上误用 fit_transform",
     "先用训练集 fit 了 scaler，然后对测试集调用了 `fit_transform`。\n\n"
     "跑一遍看看：测试集缩放后均值被压成 `0`，而 `scaler.mean_` 也变成了测试集的均值。",
     BUG3, FIX3),
    ("时序数据被随机打乱切分",
     "对一条按日期排好的负荷曲线调用 `train_test_split`。\n\n"
     "跑一遍看看：测试集的时间跨度 `01-01 ~ 04-01` **落在训练集 `01-02 ~ 04-10` 内部**。",
     BUG4, FIX4),
]

if __name__ == "__main__":
    stats = build_debug(OUT, NAME, INTRO, TASKS)
    print(f"{stats['name']}: {stats['tasks']} 题 ｜ 题目卷 {stats['quiz_cells']} cells ｜ 答案卷 {stats['sol_cells']} cells")
