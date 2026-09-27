"""dbg01 —— pandas 陷阱（改错题）。

四道题的「错误代码」均已实跑确认：
  · 题 1 链式赋值      → 抛 ChainedAssignmentError 警告，且列根本没建出来
  · 题 2 sort_values   → 静默无效，原表顺序不变
  · 题 3 groupby 赋值  → 索引对齐失败，整列变 NaN
  · 题 4 astype        → ValueError: could not convert string to float

运行：
    python scripts/authoring/dbg01_pandas_traps.py
"""

from __future__ import annotations

from pathlib import Path

from _debug_core import build_debug

OUT = Path(__file__).resolve().parents[2] / "coding" / "99_debug"
NAME = "dbg01_pandas_traps"

INTRO = [
    """# 改错题 · pandas 陷阱

每题给一段**有问题**的代码。它要么报错，要么更糟 —— **不报错，但结果是错的**。

你的任务：找出问题、修复它，让底部的 `assert` 通过。

> 这类题对应竞赛里的真实场景：**代码跑得通，但结论是错的**。
> 比赛时不会有人告诉你「这里写错了」，只有指标不对劲。
""",
    """## 怎么用

1. 先**跑一遍**错误代码，观察它到底给出了什么结果 —— 不要凭直觉猜。
2. 说出「为什么错」，再动手改。
3. 改完对 `_solution.ipynb`（每题都写了「为什么」）。

> 判据：能一句话说清「错在哪、为什么」，比改对更重要。
""",
]

BUG1 = '''
import pandas as pd

df = pd.DataFrame({
    "台区": ["A", "B", "A", "C", "B"],
    "负荷": [100.0, 220.0, 130.0, 90.0, 260.0],
})

df[df["负荷"] > 200]["重载"] = 1

print(df)
print("有「重载」列吗？", "重载" in df.columns)
'''

FIX1 = '''
import pandas as pd

df = pd.DataFrame({
    "台区": ["A", "B", "A", "C", "B"],
    "负荷": [100.0, 220.0, 130.0, 90.0, 260.0],
})

df.loc[df["负荷"] > 200, "重载"] = 1

print(df)

# 为什么：`df[cond]` 拿到的是一份**副本**，在副本上赋值永远不会写回原表。
# pandas 3.0 起默认开启 Copy-on-Write，链式赋值必然失败，并抛出
# ChainedAssignmentError（当前是警告，未来会升级为异常）。
# 正确姿势：`.loc[行条件, 列名] = 值`，一步完成。
assert "重载" in df.columns
assert df["重载"].sum() == 2
print("✅ 通过")
'''

BUG2 = '''
import pandas as pd

df = pd.DataFrame({"台区": ["A", "B", "C"], "负荷": [100.0, 260.0, 130.0]})

df.sort_values("负荷", ascending=False)

print(df)
'''

FIX2 = '''
import pandas as pd

df = pd.DataFrame({"台区": ["A", "B", "C"], "负荷": [100.0, 260.0, 130.0]})

df = df.sort_values("负荷", ascending=False)

print(df)

# 为什么：pandas 的「整理型」方法默认 `inplace=False`，返回新对象而不动原表。
# `df.sort_values(...)` 单独一行 = 结果被丢掉 = 什么也没发生，且**不报错**。
# 同类方法：dropna / drop_duplicates / fillna / reset_index / astype / rename ...
assert df.iloc[0]["负荷"] == 260.0
print("✅ 通过")
'''

BUG3 = '''
import pandas as pd

d = pd.DataFrame({"组": ["x", "x", "y", "y"], "值": [1.0, 3.0, 10.0, 20.0]})

d["按组中位数"] = d.groupby("组")["值"].median()

print(d)
'''

FIX3 = '''
import pandas as pd

d = pd.DataFrame({"组": ["x", "x", "y", "y"], "值": [1.0, 3.0, 10.0, 20.0]})

d["按组中位数"] = d.groupby("组")["值"].transform("median")

print(d)

# 为什么：`groupby().median()` 返回的是「每组一行」的聚合结果，索引是组名（x / y）。
# 把它赋回原表时按**索引对齐** —— 原表索引是 0,1,2,3，对不上 → 整列变 NaN。
# 想让「同一组每一行都填该组的统计值」，要用 `.transform(...)`：
# 它返回与原表**等长**的序列，索引一一对应。
assert d["按组中位数"].tolist() == [2.0, 2.0, 15.0, 15.0]
print("✅ 通过")
'''

BUG4 = '''
import pandas as pd

d = pd.DataFrame({"负荷": ["1.5", "N/A", "3.0", ""]})

d["负荷"] = d["负荷"].astype(float)

print(d)
'''

FIX4 = '''
import pandas as pd

d = pd.DataFrame({"负荷": ["1.5", "N/A", "3.0", ""]})

d["负荷"] = pd.to_numeric(d["负荷"], errors="coerce")

print(d)

# 为什么：`astype(float)` 遇到第一个转不了的值就**整列失败**，抛 ValueError。
# 而真实数据里永远有 "N/A" / "—" / 空串 / 带单位的字符串。
# `pd.to_numeric(errors="coerce")` 把转不了的值变成 NaN，保住其余数据。
# 更完整的清洗还要先去掉 "kW" 单位与千分位逗号（见 mock01 的 C3）。
assert d["负荷"].iloc[0] == 1.5
assert pd.isna(d["负荷"].iloc[1])
assert pd.isna(d["负荷"].iloc[3])
print("✅ 通过")
'''

TASKS = [
    ("链式赋值静默失效",
     "想给「负荷 > 200」的行打上「重载」标记。\n\n"
     "跑一遍看看：会打印一条 `ChainedAssignmentError` 警告，然后 `有「重载」列吗？ False`。",
     BUG1, FIX1),
    ("sort_values 忘了接收返回值",
     "想按负荷从大到小排序。\n\n"
     "跑一遍看看：`print(df)` 出来的顺序**完全没变**，而且一句报错都没有。",
     BUG2, FIX2),
    ("分组统计写成了聚合",
     "想给每一行都填上「它所在组的中位数」。\n\n"
     "跑一遍看看：`按组中位数` 整列是 `NaN`。",
     BUG3, FIX3),
    ("astype 遇到脏字符串",
     "想把字符串列转成数值。\n\n"
     "跑一遍看看：`ValueError: could not convert string to float: 'N/A'` —— 整列失败。",
     BUG4, FIX4),
]

if __name__ == "__main__":
    stats = build_debug(OUT, NAME, INTRO, TASKS)
    print(f"{stats['name']}: {stats['tasks']} 题 ｜ 题目卷 {stats['quiz_cells']} cells ｜ 答案卷 {stats['sol_cells']} cells")
