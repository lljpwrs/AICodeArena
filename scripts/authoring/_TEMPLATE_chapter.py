#!/usr/bin/env python3
"""章节脚本模板 —— **这是模板，本身不产出 notebook**。

用途：新增一章时 `cp` 一份改名再填内容，比从零写快得多，也不会漏掉分区。

    cp scripts/authoring/_TEMPLATE_chapter.py scripts/authoring/nlp08_xxx.py

然后只改 **4 个地方**（下面都标了 `[改这里]`）：
    1. 文件头 docstring（写清本章主线与生成命令）
    2. `OUT`   —— 章节所在目录
    3. `NAME`  —— 章节基名（三件套的文件名前缀）
    4. 正文     —— LESSON / EXERCISE

设计要点（都是踩过坑才加进来的，别删）：

- **`lesson` 与 `exercise` 是两个独立列表**：讲解版可以讲得更细，
  练习版只保留「脚手架 + 挖空 + assert」。
- **`SCAFFOLD` 只放工具函数与常量**，不放待补函数本体。
  否则容易和 `@@todo` 块里的同名函数重复定义（历史坑：final_02 踩过）。
- **`SETUP` 里写死的常量**（如 `CELL` / `BINS`）是给 `assert` 当锚点用的，
  不是可有可无——它让「参数」与「结果」分离，改数据时不至于处处联动。
- **assert 一律写在 `@@end` 之后**，且真值必须实跑得到（写 `/tmp/` 探针脚本验证）。
- 挖空块内**不许出现控制流**；多行推导的续行**不许以 `for` / `if` 开头**
  （`nb_builder` 会抛 `BuildError`）。

运行：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/_TEMPLATE_chapter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

# =========================================================================== #
# 1. 基础路径与命名                                                          #
# =========================================================================== #

OUT = Path(__file__).resolve().parents[2] / "coding" / "04_nlp"   # [改这里]
NAME = "_TEMPLATE_demo"                                            # [改这里]

# =========================================================================== #
# 2. 公共导入 / 数据加载（两版都给，不挖空）                                  #
# =========================================================================== #

IMPORTS = '''from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")          # notebook cwd = coding/04_nlp
CSV = DATA / "synth_text.csv"
'''

SETUP = '''df = pd.read_csv(CSV)
print("样本数", len(df), "| 列", list(df.columns))
print(df["缺陷类型"].value_counts().to_dict())
'''

# =========================================================================== #
# 3. 脚手架：工具函数与常量（两版都给；**不要放待补函数本体**）              #
# =========================================================================== #

SCAFFOLD = '''# 脚手架：下面的工具函数两版都有，你只填 @@todo 块里的内容
def near(a, b, eps=1e-6):
    """浮点比较，避免 assert 里的 == 抖动。"""
    return abs(float(a) - float(b)) < eps
'''

# =========================================================================== #
# 4. 任务代码块（每个块一个独立常量，便于讲解版复用）                        #
# =========================================================================== #

T1_CODE = '''# @@todo 取缺陷描述的长度统计（转成字符数）
# @@hint Series.str.len() 与 .describe()
desc_len = df["缺陷描述"].str.len()
# @@end

# ---- 验收 ----
assert len(desc_len) == 600, f"应有 600 条，实际 {len(desc_len)}"
'''

T2_CODE = '''label_n = df["缺陷类型"].value_counts()

# @@todo 取出出现次数最多的标签与它的计数
# @@hint Series.idxmax() / Series.max()
top_label, top_n = label_n.idxmax(), label_n.max()
# @@end

# ---- 验收 ----
assert top_n == 120, f"每类应各 120 条，实际最多 {top_n}"
'''

# =========================================================================== #
# 5. 讲解版（LESSON）                                                        #
# =========================================================================== #

LESSON = [
    md(
        """
# <章节标题>（讲解版）

> 本节考点 / 学习目标 / 数据来源，一句话说清。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md(
        """
## 1.1 小节标题

正文说明……（讲解版可以比练习版多讲，这是允许的差异之一）
"""
    ),
    code(T1_CODE),
    code(T2_CODE),
    md(
        """
## 易错点清单

1. ……
"""
    ),
    md(
        """
## 本章小结

……
"""
    ),
]

# =========================================================================== #
# 6. 练习 / 答案版（EXERCISE）—— 写的是**答案版**内容，练习版自动派生        #
# =========================================================================== #

EXERCISE = [
    md(
        """
# <章节标题>（练习版）

> 补全 `____`，跑通所有 assert。真值来自实跑。
"""
    ),
    code(IMPORTS),
    code(SETUP),
    code(SCAFFOLD),
    md("## 任务 1：……"),
    code(T1_CODE),
    md("## 任务 2：……"),
    code(T2_CODE),
    md(
        """
## 自查清单

- [ ] ……
"""
    ),
]

if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
