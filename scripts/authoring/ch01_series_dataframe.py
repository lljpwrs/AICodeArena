"""ch01 —— Series 与 DataFrame 基础

生成命令：
    /Users/luolinjie/miniconda3/envs/self/bin/python scripts/authoring/ch01_series_dataframe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nb_builder import build, code, md, report  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "coding" / "01_pandas"
NAME = "ch01_series_dataframe"

HEADER = """import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

# 中文字体（macOS 可用字体，按优先级回退）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

print("pandas", pd.__version__)
print("numpy ", np.__version__)"""

# =========================================================================== #
# 讲解 notebook
# =========================================================================== #

LESSON = [
    md(
        """
# ch01 Series 与 DataFrame 基础

> 方向：数据预处理 ｜ 竞赛对应：**数据探索**（独立计分项，每项 2 分）
> 定位：本章是后面 14 章的地基。地基不牢，后面所有章节都会在同一个地方反复摔。
"""
    ),
    md(
        """
## 一、本节考点

| 竞赛评分点 | 分值含义 | 本节覆盖的操作 |
|---|---|---|
| 数据探索 | 3 项探索，每项 2 分 | 用 `shape` / `dtypes` / `columns` / `index` 在 10 秒内摸清一份陌生数据 |
| 数据清洗 | 5 项清洗，每项 2 分 | 精确定位目标列、按条件取子集、**安全地**批量赋值 |
| 全部后续章节 | 技能操作 100% 都建立在它上面 | `loc` / `iloc` 是所有 pandas 操作入口 |

这一章的 API 本身不难，难在两件事：

1. **索引器语义不同**：`loc` 是标签、`iloc` 是位置，且**切片的开闭区间不一样**。
2. **pandas 3.0 起 Copy-on-Write 强制开启**，老的链式赋值写法会静默失效。

竞赛上机时写错的典型表现不是报错，而是「结果悄悄不对」——这是最致命的失分方式。
"""
    ),
    md(
        """
## 二、学习目标

学完本章你应该能：

- [ ] 用 dict / list / ndarray 三种方式构造 `Series` 和 `DataFrame`，并说清 `index` 的作用
- [ ] 不查文档就能说出 `[]` / `loc` / `iloc` / `at` / `iat` 各自接受什么、返回什么
- [ ] 背下这条铁律：**`loc` 切片左闭右闭，`iloc` 切片左闭右开**
- [ ] 一眼看出哪些写法会触发链式赋值，并改成正确写法
- [ ] 解释「索引对齐」为什么会让两个非空 Series 相加出现 `NaN`

---

## 三、API 速查表

### 3.1 构造

| API | 参数要点 | 返回 | 一句话说明 |
|---|---|---|---|
| `pd.Series(data, index=..., name=..., dtype=...)` | `data` 可为 list / ndarray / dict | Series | dict 构造时，`index` 会**筛选并重排**键 |
| `pd.DataFrame(data, index=..., columns=...)` | `data` 可为 dict / 二维 ndarray / 记录列表 | DataFrame | dict-of-list 是最贴近业务表的写法 |
| `pd.DataFrame(np_array, columns=[...])` | 二维数组必须同时给 `columns` | DataFrame | 默认 `RangeIndex` |

### 3.2 结构属性

| 属性 | 返回 | 说明 |
|---|---|---|
| `.shape` | `(行数, 列数)` | DataFrame 是二元组；Series 是 `(n,)` |
| `.columns` | `Index` | 列名 |
| `.index` | `Index` | 行标签 |
| `.dtypes` | `Series` | **逐列** dtype；DataFrame 用这个，Series 用 `.dtype` |
| `.size` / `.ndim` | `int` | 元素总数 / 维度数 |

### 3.3 索引器

| 索引器 | 接受 | 切片区间 | 说明 |
|---|---|---|---|
| `df["列名"]` | 列名 / 列名列表 | — | 选**列**；`df[切片]` 选行但**已不推荐** |
| `df.loc[行标签, 列名]` | 标签 | **左闭右闭** | 最通用、最推荐 |
| `df.iloc[行位置, 列位置]` | 整数位置 | **左闭右开** | 不知道标签、只知道第几行时用 |
| `df.at[行标签, 列名]` | 标签 | — | 取**单个标量**，比 `loc` 快 |
| `df.iat[行位置, 列位置]` | 位置 | — | 取单个标量，比 `iloc` 快 |
"""
    ),
    md(
        """
---

## 四、逐节讲解

### 4.1 先把环境和中文字体准备好

macOS 上 matplotlib 默认字体不含中文，不设置的话图上全是方块。
"""
    ),
    code(HEADER),
    md(
        """
### 4.2 构造 Series

`Series` = 带标签的一维数组。**标签（index）就是它的灵魂**，后面所有的
「索引对齐」行为都是从这来的。
"""
    ),
    code(
        """# ① 从 list 构造 —— 默认 RangeIndex
s1 = pd.Series([512.4, 96.2, 880.7])
print("从 list 构造：")
print(s1, "\\ndtype:", s1.dtype, "| index:", list(s1.index))

# ② 从 ndarray 构造 —— 顺手指定 index 和 name
s2 = pd.Series(
    np.array([512.4, 96.2, 880.7, 45.1]),
    index=["STATION_A_01", "STATION_B_02", "STATION_C_01", "STATION_D_03"],
    name="负荷值",
)
print("\\n从 ndarray 构造（带自定义 index）：")
print(s2)

# ③ 从 dict 构造 —— 注意 index 参数会「筛选并重排」，而不是补充
s3 = pd.Series({"STATION_A_01": 512.4, "STATION_B_02": 96.2, "STATION_C_01": 880.7})
print("\\n从 dict 构造，再按 index 重排：")
print(s3.reindex(["STATION_C_01", "STATION_A_01", "STATION_Z_99"]))"""
    ),
    md(
        """
> **易错点**：`reindex` 里出现原数据没有的标签 `STATION_Z_99` 时，pandas **不报错**，
> 而是补一个 `NaN`。竞赛里这是「结果多出几行空值」的常见原因。
> 想要严格校验，得显式检查 `s3.reindex(...).isna().sum()`。
"""
    ),
    code(
        """# dtype 推断规则：全整数 -> int64；含缺失 -> float64；全字符串 -> str（pandas 3.0 新增）
print("全整数      :", pd.Series([1, 2, 3]).dtype)
print("含 None     :", pd.Series([1, None, 3]).dtype, "  <- 整数被迫升为 float")
print("全字符串    :", pd.Series(["渗漏油", "锈蚀"]).dtype, "  <- pandas 3.0 起默认 str，不再是 object")
print("显式指定    :", pd.Series(["1", "2"], dtype="int64").dtype)
print("category    :", pd.Series(["一般", "严重", "一般"], dtype="category").dtype)"""
    ),
    md(
        """
### 4.3 构造 DataFrame

业务上最常遇到的三种来源：**dict-of-list**（手写测试数据）、
**二维 ndarray**（模型输出）、**记录列表**（接口返回的 JSON）。
"""
    ),
    code(
        """# ① dict-of-list —— 最贴近「一张表」的写法
df = pd.DataFrame(
    {
        "线路名称": ["LINE_ALPHA_1", "LINE_BETA_2", "LINE_GAMMA_3", "LINE_DELTA_4"],
        "设备类型": ["变压器", "断路器", "绝缘子", "避雷器"],
        "负荷值": [520.5, 180.2, 62.0, 45.8],
        "缺陷数": [2, 0, 5, 1],
    }
)
print("① dict-of-list：")
print(df)
print("\\nshape:", df.shape, "| columns:", list(df.columns), "| index:", list(df.index))

# ② 二维 ndarray —— 一定要显式给 columns，否则列名是 0/1/2...
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("\\n② ndarray（不给 columns）：")
print(pd.DataFrame(arr))
print("\\n② ndarray（给 columns）：")
print(pd.DataFrame(arr, columns=["温度", "湿度", "负荷值"]))

# ③ 记录列表 —— 接口返回的 JSON 常见形态
records = [
    {"台区编号": "STATION_A_01", "负荷值": 512.4},
    {"台区编号": "STATION_B_02", "负荷值": 96.2},
]
print("\\n③ 记录列表：")
print(pd.DataFrame(records))

# ④ 标量广播：所有行填同一个值
print("\\n④ 标量广播：")
print(pd.DataFrame({"台区编号": "STATION_A_01", "采样点": 0}, index=[0, 1, 2]))"""
    ),
    md(
        """
### 4.4 结构属性：10 秒摸清一份陌生数据

竞赛里「数据探索」是独立计分项。第一步永远是这 6 个属性，
不用看内容就能判断这份数据能不能直接建模。
"""
    ),
    code(
        """print("shape     :", df.shape, "       <- (行, 列)")
print("columns   :", list(df.columns))
print("index     :", list(df.index), "  <- 默认 RangeIndex")
print("dtypes    :")
print(df.dtypes.to_string())
print("size      :", df.size, "            <- 元素总数 = 行 x 列")
print("ndim      :", df.ndim)

# Series 与 DataFrame 的属性差异：dtype vs dtypes
print("\\nSeries.dtype :", df["负荷值"].dtype)
print("Series.shape :", df["负荷值"].shape, "  <- 一维，只有一个数")

# 只看数值列 / 只看类别列（清洗前的固定动作）
print("\\n数值列:", list(df.select_dtypes("number").columns))
print("文本列:", list(df.select_dtypes(exclude="number").columns))"""
    ),
    md(
        """
### 4.5 五个索引器，一次讲清

记住这张「手术刀分级表」：

| 索引器 | 你手里有什么 | 取什么 |
|---|---|---|
| `df["c"]` | 列名 | 一列（Series） |
| `df[["c1","c2"]]` | 列名列表 | 多列（DataFrame） |
| `df.loc[行标签, 列名]` | **标签** | 任意子集 |
| `df.iloc[行号, 列号]` | **位置** | 任意子集 |
| `df.at / df.iat` | 标签 / 位置 | **单个值**（最快） |
"""
    ),
    code(
        """t = pd.DataFrame(
    {"负荷值": [520.5, 180.2, 62.0, 45.8], "缺陷数": [2, 0, 5, 1]},
    index=["STATION_A_01", "STATION_B_02", "STATION_C_01", "STATION_D_03"],
)
print(t)

print("\\n--- [] 选列 ---")
print("df['负荷值']     ->", type(t["负荷值"]).__name__)
print("df[['负荷值']]   ->", type(t[["负荷值"]]).__name__, "（多一层方括号 = DataFrame）")

print("\\n--- loc：标签 ---")
print("loc['STATION_C_01', '负荷值']            ->", t.loc["STATION_C_01", "负荷值"])
print("loc[['STATION_A_01','STATION_D_03']]     ->")
print(t.loc[["STATION_A_01", "STATION_D_03"]])

print("\\n--- iloc：位置 ---")
print("iloc[2, 0]        ->", t.iloc[2, 0], "（第 3 行第 1 列，同样指向 STATION_C_01）")
print("iloc[[0, 3], 0]   ->", t.iloc[[0, 3], 0].to_dict())

print("\\n--- at / iat：取单个标量 ---")
print("at['STATION_C_01','负荷值'] ->", t.at["STATION_C_01", "负荷值"])
print("iat[2, 0]                   ->", t.iat[2, 0])"""
    ),
    md(
        """
> **`loc` 与 `iloc` 的等价关系**：当 index 是默认 `RangeIndex` 且你按顺序取时，两者结果相同。
> 但只要做过 `sort_values` / `dropna` / `reset_index`，行号就和标签脱钩了——
> 这时用错索引器**不会报错**，只会取到错误的数据行。竞赛里这是隐形失分大户。
"""
    ),
    md(
        """
### 4.6 切片的开闭区间（本章最该背下来的一条）

```
loc  切片：左闭右闭   df.loc["b":"d"]  ->  b, c, d
iloc 切片：左闭右开   df.iloc[1:4]     ->  第 1,2,3 行（不含第 4 行）
```
"""
    ),
    code(
        """seg = pd.DataFrame({"负荷值": [10, 20, 30, 40, 50, 60]}, index=list("abcdef"))
print(seg)

print("\\nloc['b':'d']   ->", seg.loc["b":"d", "负荷值"].tolist(), "  <- 含 d，左闭右闭")
print("iloc[1:4]      ->", seg.iloc[1:4, 0].tolist(), "  <- 不含 4，左闭右开")

# 行切片 + 列名列表，是最常用的一种「取子表」写法
sub = seg.loc["b":"d", ["负荷值"]]
print("\\nloc['b':'d', ['负荷值']] 的行标签:", list(sub.index), "| shape:", sub.shape)

# 想「按位置取、但要含末尾」，就用 iloc 时把右界 +1
print("\\niloc[1:3+1, 0] ->", seg.iloc[1:3 + 1, 0].tolist())

# 只给行切片时，列默认全取
print("\\niloc[0:2] 默认取全部列，shape =", seg.iloc[0:2].shape)"""
    ),
    md(
        """
### 4.7 布尔索引：条件筛选

三条硬规则：

1. 每个条件**必须**用圆括号包起来
2. 多条件用 `&` / `|` / `~`，**不能用 `and` / `or` / `not`**
3. 条件是 Series，长度必须和被筛对象一致
"""
    ),
    code(
        """d = pd.DataFrame(
    {
        "台区编号": ["STATION_A_01", "STATION_B_02", "STATION_C_01", "STATION_A_01", "STATION_D_03"],
        "负荷值": [520.5, 180.2, 62.0, 45.8, 9999.0],
        "缺陷数": [2, 0, 5, 1, 3],
    }
)
print(d)

mask = d["负荷值"] > 200
print("\\n布尔 Series：", mask.tolist())
print("\\nd[mask]：")
print(d[mask])

# 多条件
print("\\n负荷 > 100 且 缺陷数 >= 2：")
print(d[(d["负荷值"] > 100) & (d["缺陷数"] >= 2)])

# 取反、isin、between
print("\\n负荷 <= 100（取反）：", d[~(d["负荷值"] > 100)]["台区编号"].tolist())
print("特定台区（isin）    :", d[d["台区编号"].isin(["STATION_A_01", "STATION_D_03"])].shape[0], "行")
print("负荷在 100~600（between）:", d[d["负荷值"].between(100, 600)]["台区编号"].tolist())"""
    ),
    md(
        """
### 4.8 pandas 3.0 的 Copy-on-Write：链式赋值为什么失效了

这是**本章最重要的一节**，也是你从老教程迁移过来最容易踩的坑。

pandas 3.0 起 `mode.copy_on_write` **强制为 True 且无法关闭**。后果：

- 任何切片、`df[col]`、`df.loc[...]` 返回的都不是可以「穿透改回原表」的视图
- `df[mask]["col"] = 值` 这种链式赋值 → **赋值作用在临时副本上，静默丢弃**
- 同时抛出一个 `ChainedAssignmentError` 警告提醒你

先看错误示范：
"""
    ),
    code(
        """bad = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bad[bad["a"] > 2]["b"] = 999          # <- 链式赋值

print("警告类型 :", [type(w.message).__name__ for w in caught])
print("b 的实际值:", bad["b"].tolist(), "  <- 完全没变，赋值被丢弃了")
print("a 的筛选结果本身就是临时副本:", bad[bad["a"] > 2]["b"].tolist())"""
    ),
    md(
        """
正确写法有三种，**都用一次方括号 / `loc` 定位到底**：
"""
    ),
    code(
        """# 写法 A：单层 []，直接对目标列整体赋值（推荐）
good_a = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
good_a["b"] = np.where(good_a["a"] > 2, 999, good_a["b"])
print("A 整体赋值   :", good_a["b"].tolist())

# 写法 B：loc[行条件, 列名] —— 最标准
good_b = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
good_b.loc[good_b["a"] > 2, "b"] = 999
print("B loc 条件赋值:", good_b["b"].tolist())

# 写法 C：先 copy 再改（当你要保留原表时）
src = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
good_c = src.copy()
good_c.loc[good_c["a"] > 2, "b"] = 999
print("C 副本上改   :", good_c["b"].tolist(), "| 原表 b:", src["b"].tolist())

# 新增列同理：先整列给默认值，再条件覆盖
good_d = pd.DataFrame({"a": [1, 2, 3, 4]})
good_d["风险等级"] = "低"
good_d.loc[good_d["a"] >= 3, "风险等级"] = "高"
print("D 新增列     :", good_d["风险等级"].tolist())

# 取列做运算时，如果想「脱离原表」，显式 copy
detached = src["b"].copy()
detached.iloc[0] = -1
print("E 显式 copy  :", detached.tolist(), "| 原表 b:", src["b"].tolist())"""
    ),
    md(
        """
> **一句话记住**：**赋值语句左边只能出现一次方括号的定位**。
> 出现两次（`df[x][y] = ...`）就一定有问题。
>
> 另外注意 `inplace=True`：pandas 3.0 里它仍然可用、也不报警告，
> 但在 CoW 语义下它和「重新赋值」已经等价，反而让代码更难读、更容易在链式调用里失效。
> **本仓库统一不用 `inplace`**，一律用 `df = df.xxx()`。
"""
    ),
    md(
        """
### 4.9 索引对齐：两个非空 Series 相加为什么出 NaN

pandas 的算术运算**按标签对齐，不按位置对齐**。标签对不上的位置填 `NaN`。
这在按台区/线路做多表核对时极其常见。
"""
    ),
    code(
        """a = pd.Series([10.0, 20.0, 30.0], index=["STATION_A_01", "STATION_B_02", "STATION_C_01"])
b = pd.Series([1.0, 2.0, 3.0], index=["STATION_B_02", "STATION_C_01", "STATION_D_03"])
print("a =", a.to_dict())
print("b =", b.to_dict())

print("\\na + b        =", (a + b).to_dict(), "  <- 标签取并集，缺的位置是 NaN")
print("a.add(b, fill_value=0) =", a.add(b, fill_value=0).to_dict(), "  <- 缺失当 0")
print("a.align(b)   =")
left, right = a.align(b)
print("  left :", left.to_dict())
print("  right:", right.to_dict())

# reindex 是「对齐」的手动版
print("\\nreindex 补齐 =", a.reindex(["STATION_A_01", "STATION_C_01", "STATION_Z_99"]).to_dict())
print("reindex 缺失数 =", int(a.reindex(["STATION_A_01", "STATION_Z_99"]).isna().sum()))"""
    ),
    md(
        """
### 4.10 属性访问 `df.col` 的坑

`df.负荷值` 能跑通，但不要用：

- 列名含空格 / 中文 / 与内置方法重名时，行为会变
- `df.count` 拿到的是**方法对象**，不是叫 `count` 的那一列
"""
    ),
    code(
        """n = pd.DataFrame({"count": [1, 2], "mean": [3, 4], "index": [5, 6]})
print("正常列 df['count'] :", n["count"].tolist())
print("属性访问 df.count  :", type(n.count).__name__, " <- 拿到的是方法，不是列！")
print("df.mean 同理       :", type(n.mean).__name__)
print("df.index 同理      :", type(n.index).__name__, " <- 直接就是索引对象")

# 列名含空格时必须用 []，属性访问会语法错误
n2 = pd.DataFrame({"巡检 时长": [1.0, 2.0]})
print("\\n列名含空格时：", n2["巡检 时长"].tolist())

print("\\n结论：一律用 df['列名']，不要用 df.列名")
print("列名清单:", list(n.columns))"""
    ),
    md(
        """
---

## 五、易错点清单

| # | 易错点 | 正确做法 |
|---|---|---|
| 1 | `loc` 切片当成左闭右开 | 记住 **loc 左闭右闭**；要「含末尾」就用 loc，要「不含末尾」就用 iloc |
| 2 | `df[mask]["col"] = 值` | 改 `df.loc[mask, "col"] = 值`（CoW 下链式赋值静默失效） |
| 3 | 布尔条件不写括号 / 用 `and` | `(cond1) & (cond2)`，不能用 `and` / `or` |
| 4 | 认为 `reindex` 会报错 | 它**不报错**，缺标签补 `NaN`；要校验就查 `isna().sum()` |
| 5 | 以为 `df["c"]` 返回的是视图 | pandas 3.0 起不是，改了不影响原表；要改就用 `loc` |
| 6 | `df.列名` 属性访问 | 一律 `df["列名"]`；`df.count` 会拿到方法 |
| 7 | 忘了 pandas 3.0 字符串 dtype 是 `str` 不是 `object` | 判字符串列用 `select_dtypes("object")` 会漏掉，改用 `select_dtypes(exclude="number")` |
| 8 | `df[0:2]` 这种「纯切片选行」 | 语法还能用但语义模糊，统一改 `df.iloc[0:2]` |
| 9 | 以为 `inplace=True` 更省内存 | 3.0 下与重新赋值等价，且易在链式调用里失效，**统一不用** |

---

## 六、本章小结

```
一列拿 Series  ->  df["列名"]
多列拿表      ->  df[["列1", "列2"]]
按标签取      ->  df.loc[行标签, 列名]      切片：左闭右闭
按位置取      ->  df.iloc[行号, 列号]       切片：左闭右开
取单个值      ->  df.at / df.iat

赋值铁律      ->  等号左边只能有一次方括号定位
```

**下一步**：去做同目录的 `ch01_series_dataframe_practice.ipynb`。
练习版和答案版逐 Cell 对应，题目里保留了完整的 `assert` 验收块——
你填完直接 `Run All`，跑通就说明对了。
"""
    ),
]

# =========================================================================== #
# 练习 / 答案（共用一份源，练习版自动挖空）
# =========================================================================== #

EXERCISE = [
    md(
        """
# ch01 练习：Series 与 DataFrame 基础

> **怎么用**：打开本文件 → 找到所有 `# TODO(n)` → 把 `____` 替换成你认为正确的代码 →
> 直接 `Run All`。**跑通即为正确**，不需要翻答案。
> 全部卡住再看同目录 `ch01_series_dataframe_solution.ipynb`。

- 共 8 题，其中最后 1 题是跨小节综合题
- 难度：TODO(1)~(3) 单方法直用 → TODO(4)~(7) 易错场景 → TODO(8) 综合
- 建议限时：**25 分钟**（模拟竞赛上机节奏）
"""
    ),
    code(
        """import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

print("pandas", pd.__version__)"""
    ),
    md(
        """
---

### TODO(1) 用字典构造 DataFrame

下面已经给好了 `data` 字典。请把它构造成 DataFrame，变量名 `df`，
**列顺序必须与字典的键顺序一致**。
"""
    ),
    code(
        """data = {
    "线路名称": ["LINE_ALPHA_1", "LINE_BETA_2", "LINE_GAMMA_3", "LINE_DELTA_4", "LINE_EPSILON_5"],
    "巡检时长": [1.5, 2.0, 0.75, 3.25, 1.0],
    "缺陷数": [2, 0, 5, 1, 3],
    "是否带电作业": [True, False, True, False, False],
}

# @@todo(1) 把上面的字典构造成 DataFrame，变量名 df
# @@hint pandas.DataFrame(...)
df = pd.DataFrame(data)
# @@end

# ---- 验收 ----
assert isinstance(df, pd.DataFrame), "df 必须是一个 DataFrame"
assert df.shape == (5, 4), "形状不对，检查行数/列数"
assert list(df.columns) == ["线路名称", "巡检时长", "缺陷数", "是否带电作业"], "列顺序不齐"
assert df["缺陷数"].dtype.kind == "i", "缺陷数应该是整数类型"
print("TODO(1) 通过 ->\\n", df)"""
    ),
    md(
        """
### TODO(2) 带自定义索引的 Series + 标签取值

用给定的 `values` 和 `index` 构造一个 Series（变量名 `load`），
然后取出标签为 `STATION_C_01` 的那个值存入 `target`。
"""
    ),
    code(
        """values = [512.4, 96.2, 880.7, 45.1]
labels = ["STATION_A_01", "STATION_B_02", "STATION_C_01", "STATION_D_03"]

# @@todo(2) 构造 Series（变量名 load），再按标签取出 STATION_C_01 的值存入 target
# @@hint pd.Series(data, index=...) ；取值用 Series.loc[标签] 或 at
load = pd.Series(values, index=labels, name="负荷值")
target = load.loc["STATION_C_01"]
# @@end

# ---- 验收 ----
assert isinstance(load, pd.Series), "load 必须是 Series"
assert list(load.index) == labels, "索引不对"
assert load.name == "负荷值", "别忘了给 Series 起个 name"
assert abs(target - 880.7) < 1e-9, "取到的值不对，检查是不是用了 iloc 的位置语义"
print("TODO(2) 通过 -> target =", target)"""
    ),
    md(
        """
### TODO(3) `loc` 取子表

用 `df` 取出：
- 行标签 **1 到 3（含两端）**
- 列 **「巡检时长」和「缺陷数」**

结果存入 `sub`。
"""
    ),
    code(
        """# @@todo(3) 用 loc 取出 行标签 1~3（含）+ 两列，存入 sub
# @@hint DataFrame.loc[行切片, 列名列表] ；loc 的切片是左闭右闭
sub = df.loc[1:3, ["巡检时长", "缺陷数"]]
# @@end

# ---- 验收 ----
assert isinstance(sub, pd.DataFrame), "sub 应该是 DataFrame"
assert sub.shape == (3, 2), f"形状应为 (3, 2)，实际 {sub.shape}"
assert list(sub.index) == [1, 2, 3], "行标签不对，注意 loc 切片含右端"
assert list(sub.columns) == ["巡检时长", "缺陷数"], "列不对"
print("TODO(3) 通过 ->\\n", sub)"""
    ),
    md(
        """
### TODO(4) `iloc` 取子表（左闭右开）

用 `iloc` 取出**前 4 行**的**第 1、2 列（位置 0 和 1）**，存入 `part`。

注意这题故意和上一题形成对比：同样看起来是「取一段」，`iloc` 的右端**不含**。
"""
    ),
    code(
        """# @@todo(4) 用 iloc 取前 4 行、位置 0/1 两列，存入 part
# @@hint DataFrame.iloc[行位置切片, 列位置列表]
part = df.iloc[0:4, [0, 1]]
# @@end

# ---- 验收 ----
assert part.shape == (4, 2), f"形状应为 (4, 2)，实际 {part.shape}"
assert list(part.columns) == ["线路名称", "巡检时长"], "列不对，位置 0/1 就是前两列"
assert part.index.tolist() == [0, 1, 2, 3], "行不对"
print("TODO(4) 通过 ->\\n", part)"""
    ),
    md(
        """
### TODO(5) 布尔索引：多条件筛选

筛选出**「缺陷数 >= 2」且「巡检时长 < 3」**的记录，存入 `risk`。

提示：多条件要各自加括号，用 `&` 连接。
"""
    ),
    code(
        """# @@todo(5) 多条件筛选，存入 risk
# @@hint (条件1) & (条件2) ；不能用 and
risk = df[(df["缺陷数"] >= 2) & (df["巡检时长"] < 3)]
# @@end

# ---- 验收 ----
assert isinstance(risk, pd.DataFrame), "risk 应该是 DataFrame"
assert set(risk["线路名称"]) == {"LINE_ALPHA_1", "LINE_GAMMA_3", "LINE_EPSILON_5"}, "筛出来的行不对"
assert risk.shape[0] == 3, "行数不对，检查是不是漏了某个条件"
print("TODO(5) 通过 ->\\n", risk[["线路名称", "巡检时长", "缺陷数"]])"""
    ),
    md(
        """
### TODO(6) 安全赋值（本章最大的坑）

在 `work` 这个副本上：
1. 新增一列 `风险等级`，默认全都是 `"低"`
2. 把**「缺陷数 >= 4」**的行的 `风险等级` 改成 `"高"`

**必须用 `loc` 的条件赋值写法**——用 `work[work["缺陷数"] >= 4]["风险等级"] = "高"`
在 pandas 3.0 下会静默失效。
"""
    ),
    code(
        """work = df.copy()
high_mask = work["缺陷数"] >= 4

# @@todo(6) 先新增「风险等级」列、默认整列都是「低」；再把 high_mask 命中的行改成「高」
# @@hint 新增列直接给列名赋值；条件赋值用 DataFrame.loc[布尔条件, 列名] = 新值
work["风险等级"] = "低"
work.loc[high_mask, "风险等级"] = "高"
# @@end

# ---- 验收 ----
assert "风险等级" in work.columns, "没有新增列"
assert work["风险等级"].tolist() == ["低", "低", "高", "低", "低"], "赋值结果不对"
assert df.shape == (5, 4), "原表 df 被改动了，说明对副本的操作穿透回了原表"
print("TODO(6) 通过 ->\\n", work[["线路名称", "缺陷数", "风险等级"]])"""
    ),
    md(
        """
### TODO(7) 索引对齐

给定两个索引**不完全相同**的 Series，计算 `s1 + s2` 存入 `total`，
并统计 `total` 中有多少个缺失值存入 `n_missing`。
"""
    ),
    code(
        """s1 = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])
s2 = pd.Series([1.0, 2.0, 3.0], index=["b", "c", "d"])

# @@todo(7) 计算 s1 + s2 存入 total；统计 total 的缺失个数存入 n_missing
# @@hint 直接相加，pandas 会按标签对齐；缺失计数用 Series.isna().sum()
total = s1 + s2
n_missing = int(total.isna().sum())
# @@end

# ---- 验收 ----
assert isinstance(total, pd.Series), "total 应该是 Series"
assert set(total.index) == {"a", "b", "c", "d"}, "索引应该是两者的并集"
assert abs(total.loc["b"] - 21.0) < 1e-9, "b 位置的值不对"
assert n_missing == 2, f"缺失个数应为 2，实际 {n_missing}"
print("TODO(7) 通过 ->", total.to_dict())"""
    ),
    md(
        """
---

## 综合题：真实数据上的数据探索五步

下面用本章合成的设备缺陷数据 `data/device_defects.csv`（270 行 x 13 列）
走一遍竞赛里「数据探索」的标准动作。请按 4 个 `TODO` 依次完成。

| 步骤 | 产物 | 对应竞赛评分点 |
|---|---|---|
| ① 看结构 | `size` | 数据探索（形状/规模） |
| ② 选列 | `sub` | 数据探索（字段识别） |
| ③ 按标签取段 | `seg` | 数据清洗（定位目标记录） |
| ④ 派生新列 | `flagged` | 数据清洗（条件标记） |
"""
    ),
    code(
        """dev = pd.read_csv("data/device_defects.csv")
print("读入完成：", dev.shape)
print(dev.dtypes.to_string())"""
    ),
    md("#### 综合题 ① 看结构"),
    code(
        """# @@todo(8) 把「行数, 列数」组成的元组存入 size
# @@hint DataFrame.shape 返回的就是 (行, 列)
size = dev.shape
# @@end

# ---- 验收 ----
assert size == (270, 13), f"应为 (270, 13)，实际 {size}"
assert len(dev.columns) == 13, "列数不对"
print("综合 ① 通过 -> size =", size)"""
    ),
    md("#### 综合题 ② 选列（`loc` 的列选择形式）"),
    code(
        """subset_cols = ["台区编号", "设备类型", "负荷值"]

# @@todo(9) 取出 subset_cols 这三列存入 sub（保持原行数、原列顺序）
# @@hint DataFrame.loc[:, 列名列表] ；或直接用 df[列名列表]
sub = dev.loc[:, subset_cols]
# @@end

# ---- 验收 ----
assert list(sub.columns) == subset_cols, "列的选取或顺序不对"
assert sub.shape == (dev.shape[0], 3), "行数应该保持和原表一致"
print("综合 ② 通过 -> sub.shape =", sub.shape)"""
    ),
    md(
        """
#### 综合题 ③ 按标签取一段（`loc` 左闭右闭）

取**行标签 10 到 19（含两端）**、上面那三列，存入 `seg`。
"""
    ),
    code(
        """# @@todo(10) 用 loc 取行标签 10~19（含）、subset_cols 三列，存入 seg
# @@hint dev 读进来是默认 RangeIndex，所以标签就是行号
seg = dev.loc[10:19, subset_cols]
# @@end

# ---- 验收 ----
assert seg.shape == (10, 3), f"形状应为 (10, 3)，实际 {seg.shape}"
assert seg.index.tolist() == list(range(10, 20)), "行标签不对，别忘了 loc 切片含右端"
print("综合 ③ 通过 ->\\n", seg.head(3))"""
    ),
    md(
        """
#### 综合题 ④ 派生新列（条件标记）

在 `seg` 的**副本**上新增一列 `是否高负荷`：
- `负荷值 > 600` → `"是"`
- 其余 → `"否"`

结果存入 `flagged`（不要改动 `seg`）。
"""
    ),
    code(
        """high_load = seg["负荷值"] > 600

# @@todo(11) 在 seg 副本上派生「是否高负荷」列：默认整列「否」，high_load 命中的行改「是」
# @@hint flagged = seg.copy()，再整列赋「否」，再 DataFrame.loc[条件, 列名] = 「是」
flagged = seg.copy()
flagged["是否高负荷"] = "否"
flagged.loc[high_load, "是否高负荷"] = "是"
# @@end

# ---- 验收 ----
assert isinstance(flagged, pd.DataFrame), "flagged 应该是 DataFrame"
assert "是否高负荷" in flagged.columns, "没有新增列"
assert set(flagged["是否高负荷"]) <= {"是", "否"}, "取值只能是 是和 否"
expected_high = int((seg["负荷值"] > 600).sum())
assert (flagged["是否高负荷"] == "是").sum() == expected_high, "标记「是」的行数不对"
assert "是否高负荷" not in seg.columns, "seg 被改动了，应该在副本上操作"
print("综合 ④ 通过；高负荷记录数 =", expected_high)
print(flagged.head(5))"""
    ),
    md(
        """
---

### 复盘提问

把没跑通的那题记进 `_notes/错题本.md`，写清「错在哪 + 正确写法」。
"""
    ),
]


if __name__ == "__main__":
    stats = build(OUT, NAME, lesson=LESSON, exercise=EXERCISE)
    report([stats])
    print(f"输出目录：{OUT}")
