"""dbg03 —— 时序预测陷阱（改错题）。

四道题的错误代码均已实跑确认：
  · 题 1 rolling 忘 shift → 窗口含当日，未来信息泄漏
  · 题 2 递归 off-by-one  → 历史漏掉原点，整体错位一格
  · 题 3 直接法列名重名   → concat 不报错，下游建模才炸
  · 题 4 bfill 借未来值   → 用后面的值往回填

运行：
    python scripts/authoring/dbg03_timeseries_pitfalls.py
"""

from __future__ import annotations

from pathlib import Path

from _debug_core import build_debug

OUT = Path(__file__).resolve().parents[2] / "coding" / "99_debug"
NAME = "dbg03_timeseries_pitfalls"

INTRO = [
    """# 改错题 · 时序预测陷阱

时序方向的错误有个统一的名字：**未来信息泄漏**（look-ahead bias）。

它比普通数据泄漏更阴 —— 因为**很多写法看起来天经地义**，
效果还特别好（因为在偷看答案）。等到真实预测时，一切都崩了。
""",
    """## 一句话判据

**「站在预测时刻 t，这个数字我现在拿得到吗？」**

- 移动平均的窗口里有没有 t 当天？
- 缺失值是用**之前**的值填的，还是之后的？
- 多步预测时，历史序列的最后一个点是不是原点本身？
- 目标时刻的日历特征，有没有和原点的列重名？

> 改完对 `_solution.ipynb`。这套坑与 `05_timeseries/final` 的实测结论一致。
""",
]

BUG1 = '''
import pandas as pd

s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="负荷")

s_ma = s.rolling(3).mean()

print(s_ma.tolist())
'''

FIX1 = '''
import pandas as pd

s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="负荷")

s_ma = s.shift(1).rolling(3).mean()

print(s_ma.tolist())

# 为什么：`rolling(3)` 的窗口是「当前行 + 前 2 行」—— **含当日**。
# 把这个值当预测特征，等于把要预测的目标当成了输入。
# 观察：原写法第 3 个值是 2.0（= mean(1,2,3)），改后是 NaN，第 4 个才是 2.0。
# 口诀：**做特征先 shift(1)，算标签才不用 shift。**
assert pd.isna(s_ma.iloc[2])
assert abs(s_ma.iloc[3] - 2.0) < 1e-9
print("✅ 通过")
'''

BUG2 = '''
yv = [float(i) for i in range(10)]
origin = 5

history = yv[:origin]

print("历史序列 =", history)
print("长度 =", len(history))
'''

FIX2 = '''
yv = [float(i) for i in range(10)]
origin = 5

history = list(yv[:origin + 1])

print("历史序列 =", history)
print("长度 =", len(history))

# 为什么：要预测 t+1，历史里必须**含 t 本身**。
# `yv[:origin]` 是左闭右开切片，拿到的是 t 之前的点，把原点漏掉了。
# 这个 off-by-one 不会报错，只会让整个多步预测错位一格 —— 误差不大不小，
# 很难从指标上看出来，属于最难查的一类 bug。
assert len(history) == origin + 1
assert history[-1] == 5.0
print("✅ 通过")
'''

BUG3 = '''
import pandas as pd

origin_feat = pd.DataFrame({"h_sin": [0.1, 0.2], "lag1": [1.0, 2.0]})
target_cal = pd.DataFrame({"h_sin": [0.9, 0.8]})

X = pd.concat([origin_feat, target_cal], axis=1)

print(X.columns.tolist())
'''

FIX3 = '''
import pandas as pd

origin_feat = pd.DataFrame({"h_sin": [0.1, 0.2], "lag1": [1.0, 2.0]})
target_cal = pd.DataFrame({"h_sin": [0.9, 0.8]})

target_cal = target_cal.add_prefix("tgt_")

X = pd.concat([origin_feat, target_cal], axis=1)

print(X.columns.tolist())

# 为什么：直接法（每步一个模型）要把「原点的滞后特征」和「目标时刻的日历特征」
# 拼在一起，两者都可能叫 h_sin（小时的正弦）。
# `pd.concat(axis=1)` **不会**因列名重复报错，但下游会：
#   · LightGBM 抛 `Feature (h_sin) appears more than one time.`
#   · sklearn 按列名选择会**静默取错列** —— 后者更危险。
# 约定：目标时刻的日历特征统一加 `tgt_` 前缀。
assert X.columns.tolist() == ["h_sin", "lag1", "tgt_h_sin"]
assert len(X.columns) == len(set(X.columns))
print("✅ 通过")
'''

BUG4 = '''
import numpy as np
import pandas as pd

s = pd.Series([1.0, np.nan, np.nan, 4.0])

s_filled = s.bfill()

print(s_filled.tolist())
'''

FIX4 = '''
import numpy as np
import pandas as pd

s = pd.Series([1.0, np.nan, np.nan, 4.0])

s_filled = s.ffill()

print(s_filled.tolist())

# 为什么：`bfill` = backward fill，用**后面**的值往回填 —— 在时序里「后面」就是未来。
# 第 2、3 位被填成 4.0，可那个 4.0 是明天/后天才有的数。模型上线时根本拿不到。
# 时序缺失一律用 `ffill`（向前填充），或按业务规则填（如同时刻的历史中位数）。
assert s_filled.tolist() == [1.0, 1.0, 1.0, 4.0]
print("✅ 通过")
'''

TASKS = [
    ("rolling 忘了 shift",
     "想造一个「用过去 3 天算今天」的移动平均特征。\n\n"
     "跑一遍看看：结果 `[nan, nan, 2.0, 3.0, 4.0]` —— 第 3 个值（2.0）已经含当天了。",
     BUG1, FIX1),
    ("递归预测的历史漏了原点",
     "要从原点 t=5 往后滚动预测。\n\n"
     "跑一遍看看：历史长度是 `5`，最后一个值是 `4.0` —— **原点 5.0 丢了**。",
     BUG2, FIX2),
    ("直接法的特征列名撞车",
     "把原点特征与目标时刻日历特征横向拼接。\n\n"
     "跑一遍看看：列名变成 `['h_sin', 'lag1', 'h_sin']`，两个同名。",
     BUG3, FIX3),
    ("缺失值用 bfill（借了未来）",
     "用 `bfill` 填缺失值。\n\n"
     "跑一遍看看：第 2、3 位都被填成 `4.0` —— 那是未来的值。",
     BUG4, FIX4),
]

if __name__ == "__main__":
    stats = build_debug(OUT, NAME, INTRO, TASKS)
    print(f"{stats['name']}: {stats['tasks']} 题 ｜ 题目卷 {stats['quiz_cells']} cells ｜ 答案卷 {stats['sol_cells']} cells")
